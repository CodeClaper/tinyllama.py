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

// Scatter the key/value states of `num_tokens` tokens into the paged cache.
// One CUDA block handles one token; the threads of the block walk over the
// `num_heads * head_size` elements of the token.
//
// The key cache stores `x` contiguous head elements per (block, head) pair,
// so an element at head offset `d` lives at
//   (d / x, block_offset, d % x)
// inside the [head_size / x, block_size, x] slice of its head.
template <typename scalar_t, typename cache_t, typename slot_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache,
    cache_t* __restrict__ value_cache,
    const slot_t* __restrict__ slot_mapping,
    int64_t key_stride,
    int64_t value_stride,
    int num_heads,
    int head_size,
    int block_size,
    int x)
{
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = static_cast<int64_t>(slot_mapping[token_idx]);
    if (slot_idx < 0) {
        // Padding token that should not be written to the cache.
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const int64_t x_blocks = head_size / x;
    const int num_elements = num_heads * head_size;

    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        const int head_idx = i / head_size;
        const int head_offset = i % head_size;
        const int x_idx = head_offset / x;
        const int x_offset = head_offset % x;

        const int64_t tgt_key_idx =
            ((block_idx * num_heads + head_idx) * x_blocks + x_idx) *
                block_size * x +
            block_offset * x + x_offset;
        const int64_t tgt_value_idx =
            ((block_idx * num_heads + head_idx) * head_size + head_offset) *
                block_size +
            block_offset;

        key_cache[tgt_key_idx] = static_cast<cache_t>(
            key[token_idx * key_stride + i]);
        value_cache[tgt_value_idx] = static_cast<cache_t>(
            value[token_idx * value_stride + i]);
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

void reshape_and_cache_launch(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    c10::cuda::CUDAStream stream)
{
    const int num_tokens = static_cast<int>(slot_mapping.size(0));
    if (num_tokens == 0) {
        return;
    }

    const int num_heads = static_cast<int>(key.size(1));
    const int head_size = static_cast<int>(key.size(2));
    const int block_size = static_cast<int>(key_cache.size(3));
    const int x = static_cast<int>(key_cache.size(4));

    const int64_t key_stride = key.stride(0);
    const int64_t value_stride = value.stride(0);

    const dim3 grid(num_tokens);
    const dim3 threads(std::min(num_heads * head_size, 512));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(),
        "reshape_and_cache_kernel", [&] {
            using key_scalar_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                key_cache.scalar_type(), "reshape_and_cache_kernel", [&] {
                    using cache_scalar_t = scalar_t;
                    AT_DISPATCH_INDEX_TYPES(
                        slot_mapping.scalar_type(),
                        "reshape_and_cache_kernel", [&] {
                            reshape_and_cache_kernel<
                                key_scalar_t, cache_scalar_t, index_t>
                                <<<grid, threads, 0, stream>>>(
                                    key.data_ptr<key_scalar_t>(),
                                    value.data_ptr<key_scalar_t>(),
                                    key_cache.data_ptr<cache_scalar_t>(),
                                    value_cache.data_ptr<cache_scalar_t>(),
                                    slot_mapping.data_ptr<index_t>(),
                                    key_stride,
                                    value_stride,
                                    num_heads,
                                    head_size,
                                    block_size,
                                    x);
                        });
                });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
