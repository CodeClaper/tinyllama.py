#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <unordered_map>

// Copy KV cache blocks from `src` to `dst`.
//
// Layouts (contiguous, first dimension is the block index):
//   key_cache   : [num_blocks, num_kv_heads, head_size / x, block_size, x]
//   value_cache : [num_blocks, num_kv_heads, head_size, block_size]
//
// `src` and `dst` may hold a different number of blocks (e.g. the GPU and
// CPU caches), but a single block must have the same size in both tensors.
//
// `block_mapping` maps `src` block index -> `dst` block index, so the copy
// always reads from `src` and writes to `dst`.
//
// `src` and `dst` may live on different devices (GPU -> CPU or CPU -> GPU).
// The copies are issued asynchronously on `stream`; the caller is
// responsible for synchronizing the stream before using the result.
void swap_blocks_launch(
    torch::Tensor& src,
    torch::Tensor& dst,
    const std::unordered_map<int64_t, int64_t>& block_mapping,
    c10::cuda::CUDAStream stream);
