#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Root mean square normalization over the last dimension.
//
// Layouts (contiguous):
//   input  : [..., hidden_size]
//   weight : [hidden_size]
//   out    : [..., hidden_size]
//
//   out = input * weight * rsqrt(mean(input^2) + eps)
//
// `out`, `input` and `weight` must share the same floating point dtype. The
// kernel is launched asynchronously on `stream`.
void rms_norm_launch(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps,
    c10::cuda::CUDAStream stream);

// Fused residual-add + RMSNorm, applied in place.
//
// Layouts (contiguous):
//   input    : [..., hidden_size]  (in/out)
//   residual : [..., hidden_size]  (in/out)
//   weight   : [hidden_size]
//
//   residual += input
//   input     = residual * weight * rsqrt(mean(residual^2) + eps)
//
// This is the residual stream + normalization pair used between transformer
// layers: `input` ends up holding the normalized activations and `residual`
// holds the un-normalized sum for the next layer.
//
// `input`, `residual` and `weight` must share the same floating point dtype.
// The kernel is launched asynchronously on `stream`.
void fused_add_rms_norm_launch(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps,
    c10::cuda::CUDAStream stream);
