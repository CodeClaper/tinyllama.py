#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Launch the paged-attention decode kernel.
//
// Layouts (contiguous):
//   query        : [num_seqs, num_heads, head_dim]
//   key_cache    : [num_blocks, num_kv_heads, block_size, head_dim]
//   value_cache  : [num_blocks, num_kv_heads, block_size, head_dim]
//   block_tables : [num_seqs, max_blocks_per_seq]  (int32)
//   seq_lens     : [num_seqs]                      (int32)
//   out          : [num_seqs, num_heads, head_dim]
//
// `query[i]` is the newest token of sequence `i`, so the number of query
// rows must equal the number of sequences (decode phase only).
void paged_attention_launch(
    torch::Tensor& out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    double scale,
    c10::cuda::CUDAStream stream);
