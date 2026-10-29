#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention.h"

torch::Tensor paged_attention(torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache, 
                              torch::Tensor block_tables, torch::Tensor seq_lens, double scale) 
{
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
    TORCH_CHECK(block_tables.is_cuda(), "block_tables must be a CUDA tensor");
    TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
    TORCH_CHECK(query.dim() == 3, "query must be [num_seqs, num_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [num_blocks, num_kv_heads, block_size, head_dim]");
    TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache must have the same shape as key_cache");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must be [num_seqs, max_blocks_per_seq]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_seqs]");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(block_tables.scalar_type() == torch::kInt32, "block_tables must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(query.scalar_type() == key_cache.scalar_type() && query.scalar_type() == value_cache.scalar_type(), "query, key_cache and value_cache must share the same dtype");
    TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16 || query.scalar_type() == torch::kFloat32, "dtype must be float16, bfloat16 or float32");

    const int64_t num_seqs = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_dim = query.size(2);
    const int64_t num_kv_heads = key_cache.size(1);
    const int64_t cache_head_dim = key_cache.size(3);

    TORCH_CHECK(num_seqs > 0, "query must contain at least one sequence");
    TORCH_CHECK(head_dim == cache_head_dim, "query head_dim must match the cache head_dim");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(block_tables.size(0) == num_seqs, "block_tables must have one row per sequence");
    TORCH_CHECK(seq_lens.size(0) == num_seqs, "seq_lens must have one entry per sequence");
    TORCH_CHECK(head_dim <= 1024, "head_dim too large for the reference kernel");

    auto out = torch::empty_like(query);
    const at::cuda::OptionalCUDAGuard device_guard(query.device());

    paged_attention_launch(out, query, key_cache, value_cache, block_tables,
                           seq_lens, scale, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention", &paged_attention,
          "PagedAttention decode forward (CUDA)",
          pybind11::arg("query"), pybind11::arg("key_cache"),
          pybind11::arg("value_cache"), pybind11::arg("block_tables"),
          pybind11::arg("seq_lens"), pybind11::arg("scale"));
}
