#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Launch the paged-attention decode kernel.
//
// Layouts (contiguous):
//   query        : [num_seqs, num_heads, head_dim]
//   key_cache    : [num_blocks, num_kv_heads, head_dim / x, block_size, x]
//   value_cache  : [num_blocks, num_kv_heads, head_dim, block_size]
//   block_tables : [num_seqs, max_blocks_per_seq]  (int32)
//   context_lens : [num_seqs]                      (int32)
//   alibi_slopes : optional [num_heads]            (float32)
//   out          : [num_seqs, num_heads, head_dim]
//
// `x` is the last dimension of `key_cache` (typically `16 / element_size`).
// `query[i]` is the newest token of sequence `i`, so the number of query
// rows must equal the number of sequences (decode phase only).
//
// The cache tensors may use a different floating point dtype than `query`;
// cached keys/values are converted to float while the kernel runs.
void paged_attention_launch(
    torch::Tensor& out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    double scale,
    int64_t max_context_len,
    const float* alibi_slopes,
    c10::cuda::CUDAStream stream);
