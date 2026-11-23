#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "attention.h"

namespace {

// Resolve the dtype of the KV cache from the `kv_cache_dtype` string.
// "auto" (and an empty string) keeps the dtype of the cache tensors.
at::ScalarType resolve_kv_cache_dtype(
    const std::string& kv_cache_dtype,
    at::ScalarType cache_dtype) {
    std::string dtype = kv_cache_dtype;
    std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (dtype.empty() || dtype == "auto") {
        return cache_dtype;
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

bool is_supported_float_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16 ||
           dtype == at::kFloat || dtype == at::kDouble;
}

// Return the ALiBi slopes as a contiguous float32 tensor on `device`.
// An undefined tensor is returned when no slopes were supplied.
torch::Tensor prepare_alibi_slopes(
    const std::optional<torch::Tensor>& alibi_slopes,
    const torch::Device& device) {
    if (!alibi_slopes.has_value() || !alibi_slopes->defined()) {
        return torch::Tensor();
    }

    torch::Tensor slopes = *alibi_slopes;
    TORCH_CHECK(slopes.scalar_type() == torch::kFloat32,
                "alibi_slopes must be float32");
    TORCH_CHECK(slopes.dim() == 1, "alibi_slopes must be [num_heads]");

    if (slopes.is_cuda()) {
        TORCH_CHECK(slopes.device() == device,
                    "alibi_slopes must be on the same GPU as query");
    } else {
        slopes = slopes.to(device);
    }
    return slopes.contiguous();
}

}  // namespace

torch::Tensor paged_attention(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
    TORCH_CHECK(block_tables.is_cuda(), "block_tables must be a CUDA tensor");
    TORCH_CHECK(context_lens.is_cuda(), "context_lens must be a CUDA tensor");

    const torch::Device device = query.device();
    TORCH_CHECK(key_cache.device() == device &&
                    value_cache.device() == device &&
                    block_tables.device() == device &&
                    context_lens.device() == device,
                "query, caches, block_tables and context_lens must be on the "
                "same GPU");

    TORCH_CHECK(query.dim() == 3,
                "query must be [num_seqs, num_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 5,
                "key_cache must be [num_blocks, num_kv_heads, head_dim / x, "
                "block_size, x]");
    TORCH_CHECK(value_cache.dim() == 4,
                "value_cache must be [num_blocks, num_kv_heads, head_dim, "
                "block_size]");
    TORCH_CHECK(block_tables.dim() == 2,
                "block_tables must be [num_seqs, max_blocks_per_seq]");
    TORCH_CHECK(context_lens.dim() == 1,
                "context_lens must be [num_seqs]");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(),
                "block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(),
                "context_lens must be contiguous");

    TORCH_CHECK(block_tables.scalar_type() == torch::kInt32,
                "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt32,
                "context_lens must be int32");

    TORCH_CHECK(is_supported_float_dtype(query.scalar_type()),
                "query dtype must be float16, bfloat16, float32 or float64");
    TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
                "key_cache and value_cache must share the same dtype");
    TORCH_CHECK(is_supported_float_dtype(key_cache.scalar_type()),
                "cache dtype must be float16, bfloat16, float32 or float64");

    const at::ScalarType cache_dtype =
        resolve_kv_cache_dtype(kv_cache_dtype, key_cache.scalar_type());
    TORCH_CHECK(key_cache.scalar_type() == cache_dtype,
                "key_cache dtype does not match kv_cache_dtype");
    TORCH_CHECK(value_cache.scalar_type() == cache_dtype,
                "value_cache dtype does not match kv_cache_dtype");

    const int64_t num_seqs = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_dim = query.size(2);
    const int64_t num_blocks = key_cache.size(0);
    const int64_t cache_num_kv_heads = key_cache.size(1);
    const int64_t x = key_cache.size(4);

    TORCH_CHECK(num_seqs > 0, "query must contain at least one sequence");
    TORCH_CHECK(num_heads > 0 && head_dim > 0,
                "query must have a positive number of heads and head_dim");
    TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
    TORCH_CHECK(num_heads % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(num_kv_heads == cache_num_kv_heads,
                "num_kv_heads must match key_cache");
    TORCH_CHECK(num_blocks > 0, "cache must contain at least one block");
    TORCH_CHECK(block_size > 0 && block_size == key_cache.size(3),
                "block_size must match key_cache");
    TORCH_CHECK(value_cache.size(3) == block_size,
                "block_size must match value_cache");
    TORCH_CHECK(x > 0 && head_dim % x == 0,
                "head_dim must be divisible by the last dimension of "
                "key_cache");
    TORCH_CHECK(key_cache.size(2) == head_dim / x,
                "key_cache head_dim / x must match query");
    TORCH_CHECK(value_cache.size(0) == num_blocks,
                "key_cache and value_cache must have the same number of blocks");
    TORCH_CHECK(value_cache.size(1) == num_kv_heads,
                "value_cache num_kv_heads must match key_cache");
    TORCH_CHECK(value_cache.size(2) == head_dim,
                "value_cache head_dim must match query");
    TORCH_CHECK(head_dim <= 1024, "head_dim too large for the reference kernel");

    TORCH_CHECK(max_context_len >= 0, "max_context_len must be non-negative");
    TORCH_CHECK(block_tables.size(0) == num_seqs,
                "block_tables must have one row per sequence");
    TORCH_CHECK(context_lens.size(0) == num_seqs,
                "context_lens must have one entry per sequence");
    TORCH_CHECK(
        block_tables.size(1) >=
            (max_context_len + block_size - 1) / block_size,
        "block_tables does not have enough columns for max_context_len");

    const at::cuda::OptionalCUDAGuard device_guard(device);
    const torch::Tensor alibi =
        prepare_alibi_slopes(alibi_slopes, device);
    if (alibi.defined()) {
        TORCH_CHECK(alibi.numel() == num_heads,
                    "alibi_slopes must have one value per query head");
    }

    auto out = torch::empty_like(query);
    paged_attention_launch(
        out, query, key_cache, value_cache, block_tables, context_lens, scale,
        max_context_len, alibi.defined() ? alibi.data_ptr<float>() : nullptr,
        at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention", &paged_attention,
          "PagedAttention decode forward (CUDA)",
          pybind11::arg("query"), pybind11::arg("key_cache"),
          pybind11::arg("value_cache"), pybind11::arg("num_kv_heads"),
          pybind11::arg("scale"), pybind11::arg("block_tables"),
          pybind11::arg("context_lens"), pybind11::arg("block_size"),
          pybind11::arg("max_context_len"), pybind11::arg("alibi_slopes"),
          pybind11::arg("kv_cache_dtype"));
}
