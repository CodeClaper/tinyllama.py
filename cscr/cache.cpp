#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "cache.h"

namespace {

// Number of bytes occupied by a single block of a contiguous cache tensor.
int64_t block_size_in_bytes(const torch::Tensor& cache) {
    TORCH_CHECK(cache.dim() >= 1, "cache tensors must have at least one dimension");
    TORCH_CHECK(cache.size(0) > 0, "cache tensors must contain at least one block");
    return cache.element_size() * (cache.numel() / cache.size(0));
}

// Validate a single GPU cache tensor used by `copy_blocks` and check that the
// block indices of the mapping are in range.
void check_copy_cache(
    const torch::Tensor& cache,
    const torch::Device& device,
    const std::unordered_map<int64_t, std::vector<int64_t>>& block_mapping) {
    TORCH_CHECK(cache.is_cuda(), "cache tensors must be CUDA tensors");
    TORCH_CHECK(cache.device() == device,
                "all cache tensors must be on the same GPU");
    TORCH_CHECK(cache.dim() >= 1, "cache tensors must have at least one dimension");
    TORCH_CHECK(cache.size(0) > 0, "cache tensors must contain at least one block");
    TORCH_CHECK(cache.is_contiguous(), "cache tensors must be contiguous");

    const int64_t num_blocks = cache.size(0);
    for (const auto& [src_block, dst_blocks] : block_mapping) {
        TORCH_CHECK(src_block >= 0 && src_block < num_blocks,
                    "src block index ", src_block, " is out of range [0, ",
                    num_blocks, ")");
        for (const int64_t dst_block : dst_blocks) {
            TORCH_CHECK(dst_block >= 0 && dst_block < num_blocks,
                        "dst block index ", dst_block, " is out of range [0, ",
                        num_blocks, ")");
        }
    }
}

// Resolve the dtype of the KV cache from the `kv_cache_dtype` string.
// "auto" (and an empty string) keeps the dtype of the key/value tensors.
at::ScalarType resolve_kv_cache_dtype(
    const std::string& kv_cache_dtype,
    at::ScalarType key_dtype) {
    std::string dtype = kv_cache_dtype;
    std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (dtype.empty() || dtype == "auto") {
        return key_dtype;
    }
    if (dtype == "half" || dtype == "float16") {
        return at::kHalf;
    }
    if (dtype == "bfloat16" || dtype == "bf16") {
        return at::kBFloat16;
    }
    if (dtype == "float" || dtype == "float32") {
        return at::kFloat;
    }
    if (dtype == "double" || dtype == "float64") {
        return at::kDouble;
    }
    TORCH_CHECK(false, "unsupported kv_cache_dtype: ", kv_cache_dtype);
}

}  // namespace

void swap_blocks(torch::Tensor src, torch::Tensor dst,
                 const std::unordered_map<int64_t, int64_t>& block_mapping) {
    const bool src_cuda = src.is_cuda();
    const bool dst_cuda = dst.is_cuda();

    TORCH_CHECK(src_cuda || src.device().is_cpu(), "src must be a CPU or CUDA tensor");
    TORCH_CHECK(dst_cuda || dst.device().is_cpu(), "dst must be a CPU or CUDA tensor");
    TORCH_CHECK(!(src_cuda && dst_cuda) || src.device().index() == dst.device().index(),
                "src and dst must be on the same GPU");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.element_size() == dst.element_size(),
                "src and dst must have the same element size");

    if (block_mapping.empty()) {
        return;
    }

    TORCH_CHECK(block_size_in_bytes(src) == block_size_in_bytes(dst),
                "src and dst blocks must have the same size");

    const int64_t num_src_blocks = src.size(0);
    const int64_t num_dst_blocks = dst.size(0);
    for (const auto& [src_block, dst_block] : block_mapping) {
        TORCH_CHECK(src_block >= 0 && src_block < num_src_blocks,
                    "src block index ", src_block, " is out of range [0, ",
                    num_src_blocks, ")");
        TORCH_CHECK(dst_block >= 0 && dst_block < num_dst_blocks,
                    "dst block index ", dst_block, " is out of range [0, ",
                    num_dst_blocks, ")");
    }

    swap_blocks_launch(src, dst, block_mapping,
                       at::cuda::getCurrentCUDAStream());
}

void copy_blocks(
    std::vector<torch::Tensor> key_caches,
    std::vector<torch::Tensor> value_caches,
    const std::unordered_map<int64_t, std::vector<int64_t>>& block_mapping) {
    TORCH_CHECK(key_caches.size() == value_caches.size(),
                "key_caches and value_caches must have the same number of layers");

    if (key_caches.empty() || block_mapping.empty()) {
        return;
    }

    const torch::Device device = key_caches[0].device();
    for (const auto& key_cache : key_caches) {
        check_copy_cache(key_cache, device, block_mapping);
    }
    for (const auto& value_cache : value_caches) {
        check_copy_cache(value_cache, device, block_mapping);
    }

    copy_blocks_launch(key_caches, value_caches, block_mapping,
                       at::cuda::getCurrentCUDAStream());
}

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype) {
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");

    TORCH_CHECK(key.device() == value.device() &&
                    key.device() == key_cache.device() &&
                    key.device() == value_cache.device() &&
                    key.device() == slot_mapping.device(),
                "key, value, caches and slot_mapping must be on the same GPU");

    TORCH_CHECK(key.dim() == 3,
                "key must be [num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(value.sizes() == key.sizes(),
                "value must have the same shape as key");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must share the same dtype");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

    TORCH_CHECK(key_cache.dim() == 5,
                "key_cache must be [num_blocks, num_kv_heads, head_size / x, "
                "block_size, x]");
    TORCH_CHECK(value_cache.dim() == 4,
                "value_cache must be [num_blocks, num_kv_heads, head_size, "
                "block_size]");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
                "key_cache and value_cache must share the same dtype");

    TORCH_CHECK(slot_mapping.dim() == 1,
                "slot_mapping must be [num_tokens]");
    TORCH_CHECK(slot_mapping.is_contiguous(),
                "slot_mapping must be contiguous");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt32 ||
                    slot_mapping.scalar_type() == torch::kInt64,
                "slot_mapping must be int32 or int64");

    const int64_t num_tokens = key.size(0);
    const int64_t num_heads = key.size(1);
    const int64_t head_size = key.size(2);
    const int64_t block_size = key_cache.size(3);
    const int64_t x = key_cache.size(4);

    TORCH_CHECK(slot_mapping.size(0) == num_tokens,
                "slot_mapping must have one entry per token");
    TORCH_CHECK(x > 0, "the last dimension of key_cache must be positive");
    TORCH_CHECK(head_size % x == 0,
                "head_size must be divisible by the last dimension of "
                "key_cache");
    TORCH_CHECK(block_size > 0 && num_heads > 0 && head_size > 0,
                "cache dimensions must be positive");

    TORCH_CHECK(key_cache.size(0) == value_cache.size(0),
                "key_cache and value_cache must have the same number of blocks");
    TORCH_CHECK(key_cache.size(1) == num_heads,
                "key_cache num_heads must match key");
    TORCH_CHECK(value_cache.size(1) == num_heads,
                "value_cache num_heads must match key");
    TORCH_CHECK(key_cache.size(2) == head_size / x,
                "key_cache head_size / x must match key");
    TORCH_CHECK(value_cache.size(2) == head_size,
                "value_cache head_size must match key");
    TORCH_CHECK(value_cache.size(3) == block_size,
                "value_cache block_size must match key_cache");

    const at::ScalarType cache_dtype =
        resolve_kv_cache_dtype(kv_cache_dtype, key.scalar_type());
    TORCH_CHECK(key_cache.scalar_type() == cache_dtype,
                "key_cache dtype does not match kv_cache_dtype");
    TORCH_CHECK(value_cache.scalar_type() == cache_dtype,
                "value_cache dtype does not match kv_cache_dtype");

    if (num_tokens == 0) {
        return;
    }

    const at::cuda::OptionalCUDAGuard device_guard(key.device());
    reshape_and_cache_launch(key, value, key_cache, value_cache, slot_mapping,
                             at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swap_blocks", &swap_blocks,
          "Copy KV cache blocks from src to dst (CUDA)",
          pybind11::arg("src"), pybind11::arg("dst"),
          pybind11::arg("block_mapping"));
    m.def("copy_blocks", &copy_blocks,
          "Copy KV cache blocks to multiple destinations (CUDA)",
          pybind11::arg("key_caches"), pybind11::arg("value_caches"),
          pybind11::arg("block_mapping"));
    m.def("reshape_and_cache", &reshape_and_cache,
          "Write key/value states into the paged KV cache (CUDA)",
          pybind11::arg("key"), pybind11::arg("value"),
          pybind11::arg("key_cache"), pybind11::arg("value_cache"),
          pybind11::arg("slot_mapping"),
          pybind11::arg("kv_cache_dtype"));
}
