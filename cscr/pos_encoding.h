#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void rotary_embedding_neox_lanch(
    torch::Tensor& out,
    const torch::Tensor& postions,
    const torch::Tensor& query,
    const torch::Tensor& key,
    int head_size,
    const torch::Tensor& cos_sin_cache,
    c10::cuda::CUDAStream stream);
