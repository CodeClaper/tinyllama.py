#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "cache.h"

namespace {

constexpr int kThreads = 256;
// The kernels use a grid-stride loop over the block pairs, so the grid can be
// capped to a value every GPU supports.
constexpr int kMaxGridBlocks = 65535;

// Copy `block_size` elements of every block pair. Each block pair is stored as
// two consecutive entries in `block_mapping`: {src_block, dst_block}.
template <typename scalar_t>
__global__ void copy_block_pairs_kernel(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    const int64_t* __restrict__ block_mapping,
    int64_t block_size,
    int64_t num_pairs)
{
    for (int64_t i = blockIdx.x; i < num_pairs; i += gridDim.x) {
        const int64_t src_block = block_mapping[2 * i];
        const int64_t dst_block = block_mapping[2 * i + 1];
        const scalar_t* src_ptr = src + src_block * block_size;
        scalar_t* dst_ptr = dst + dst_block * block_size;
        for (int64_t j = threadIdx.x; j < block_size; j += blockDim.x) {
            dst_ptr[j] = src_ptr[j];
        }
    }
}

// Build a [num_pairs, 2] int64 tensor holding the (src, dst) block pairs.
torch::Tensor make_block_mapping_tensor(
    const std::unordered_map<int64_t, int64_t>& block_mapping,
    const torch::Device& device)
{
    std::vector<int64_t> pairs;
    pairs.reserve(block_mapping.size() * 2);
    for (const auto& [src_block, dst_block] : block_mapping) {
        pairs.push_back(src_block);
        pairs.push_back(dst_block);
    }

    const auto options = torch::TensorOptions()
                             .dtype(torch::kInt64)
                             .device(torch::kCPU);
    const auto cpu_mapping = torch::from_blob(
        pairs.data(),
        {static_cast<int64_t>(block_mapping.size()), 2},
        options);
    return cpu_mapping.to(device);
}

// Launch the block-pair copy kernel for `num_pairs` pairs stored in
// `block_mapping` (a [num_pairs, 2] int64 device tensor).
void launch_block_pairs(
    torch::Tensor& src,
    torch::Tensor& dst,
    const torch::Tensor& block_mapping,
    int64_t num_pairs,
    c10::cuda::CUDAStream stream)
{
    const int64_t block_size = src.numel() / src.size(0);
    const int grid =
        static_cast<int>(std::min<int64_t>(num_pairs, kMaxGridBlocks));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        src.scalar_type(), "copy_block_pairs_kernel", [&] {
            copy_block_pairs_kernel<scalar_t>
                <<<grid, kThreads, 0, stream>>>(
                    src.data_ptr<scalar_t>(),
                    dst.data_ptr<scalar_t>(),
                    block_mapping.data_ptr<int64_t>(),
                    block_size,
                    num_pairs);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Expand a src -> [dst...] mapping into a [num_pairs, 2] int64 device tensor.
torch::Tensor make_copy_block_mapping_tensor(
    const std::unordered_map<int64_t, std::vector<int64_t>>& block_mapping,
    const torch::Device& device,
    int64_t* num_pairs)
{
    int64_t total_pairs = 0;
    for (const auto& [src_block, dst_blocks] : block_mapping) {
        total_pairs += static_cast<int64_t>(dst_blocks.size());
    }

    std::vector<int64_t> pairs;
    pairs.reserve(total_pairs * 2);
    for (const auto& [src_block, dst_blocks] : block_mapping) {
        for (const int64_t dst_block : dst_blocks) {
            pairs.push_back(src_block);
            pairs.push_back(dst_block);
        }
    }

    *num_pairs = static_cast<int64_t>(pairs.size() / 2);
    const auto options = torch::TensorOptions()
                             .dtype(torch::kInt64)
                             .device(torch::kCPU);
    const auto cpu_mapping = torch::from_blob(
        pairs.data(), {*num_pairs, 2}, options);
    // Keep the copy synchronous: `pairs` is destroyed when this function
    // returns.
    return cpu_mapping.to(device);
}

}  // namespace

void swap_blocks_launch(
    torch::Tensor& src,
    torch::Tensor& dst,
    const std::unordered_map<int64_t, int64_t>& block_mapping,
    c10::cuda::CUDAStream stream)
{
    const int64_t num_pairs = static_cast<int64_t>(block_mapping.size());
    if (num_pairs == 0) {
        return;
    }

    const int64_t block_bytes =
        src.element_size() * (src.numel() / src.size(0));
    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    const bool src_cuda = src.is_cuda();
    const bool dst_cuda = dst.is_cuda();

    if (!src_cuda && !dst_cuda) {
        // Both caches live in host memory, so a plain memcpy is enough.
        for (const auto& [src_block, dst_block] : block_mapping) {
            std::memcpy(dst_ptr + dst_block * block_bytes,
                        src_ptr + src_block * block_bytes,
                        block_bytes);
        }
        return;
    }

    // Make sure the copies are issued on the device that owns the cache.
    const at::cuda::OptionalCUDAGuard device_guard(
        src_cuda ? src.device() : dst.device());

    if (src_cuda && dst_cuda) {
        // One kernel launch for all pairs is much cheaper than one
        // cudaMemcpyAsync per block when swapping many small blocks.
        const torch::Tensor block_mapping_tensor =
            make_block_mapping_tensor(block_mapping, src.device());
        launch_block_pairs(src, dst, block_mapping_tensor, num_pairs, stream);
        return;
    }

    // GPU <-> CPU: the copy has to go through the memory copy engine, which
    // also works for pageable (non-pinned) host memory.
    const cudaMemcpyKind kind =
        src_cuda ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    for (const auto& [src_block, dst_block] : block_mapping) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            dst_ptr + dst_block * block_bytes,
            src_ptr + src_block * block_bytes,
            block_bytes,
            kind,
            stream));
    }
}

void copy_blocks_launch(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::unordered_map<int64_t, std::vector<int64_t>>& block_mapping,
    c10::cuda::CUDAStream stream)
{
    if (key_caches.empty() && value_caches.empty()) {
        return;
    }

    const torch::Device device =
        key_caches.empty() ? value_caches[0].device() : key_caches[0].device();
    const at::cuda::OptionalCUDAGuard device_guard(device);

    // The mapping is shared by every layer, so build it only once.
    int64_t num_pairs = 0;
    const torch::Tensor block_mapping_tensor =
        make_copy_block_mapping_tensor(block_mapping, device, &num_pairs);
    if (num_pairs == 0) {
        return;
    }

    // `copy_blocks` copies blocks in place, so src and dst are the same
    // tensor.
    for (auto& key_cache : key_caches) {
        launch_block_pairs(
            key_cache, key_cache, block_mapping_tensor, num_pairs, stream);
    }
    for (auto& value_cache : value_caches) {
        launch_block_pairs(
            value_cache, value_cache, block_mapping_tensor, num_pairs, stream);
    }
}
