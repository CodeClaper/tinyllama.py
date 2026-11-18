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
__global__ void swap_blocks_kernel(
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
        const int64_t block_size = block_bytes / src.element_size();
        const int grid =
            static_cast<int>(std::min<int64_t>(num_pairs, kMaxGridBlocks));

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            src.scalar_type(), "swap_blocks_kernel", [&] {
                swap_blocks_kernel<scalar_t>
                    <<<grid, kThreads, 0, stream>>>(
                        src.data_ptr<scalar_t>(),
                        dst.data_ptr<scalar_t>(),
                        block_mapping_tensor.data_ptr<int64_t>(),
                        block_size,
                        num_pairs);
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
