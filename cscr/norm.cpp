#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "norm.h"

namespace {

// Validate the tensors shared by both normalization entry points.
void check_norm_inputs(const torch::Tensor& input, const torch::Tensor& weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must be [hidden_size]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int64_t hidden_size = input.size(-1);
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    TORCH_CHECK(weight.size(0) == hidden_size,
                "weight size must match the last input dimension");
    TORCH_CHECK(weight.device() == input.device(),
                "input and weight must be on the same GPU");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "input and weight must share the same dtype");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 ||
                    input.scalar_type() == torch::kBFloat16 ||
                    input.scalar_type() == torch::kFloat32,
                "dtype must be float16, bfloat16 or float32");
}

void check_norm_residual(const torch::Tensor& input,
                         const torch::Tensor& residual) {
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual.sizes() == input.sizes(),
                "residual must have the same shape as input");
    TORCH_CHECK(residual.device() == input.device(),
                "residual must be on the same GPU as input");
    TORCH_CHECK(residual.scalar_type() == input.scalar_type(),
                "residual must share the input dtype");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
}

}  // namespace

torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, double eps) {
    check_norm_inputs(input, weight);

    auto out = torch::empty_like(input);
    const at::cuda::OptionalCUDAGuard device_guard(input.device());

    rms_norm_launch(out, input, weight, eps,
                    at::cuda::getCurrentCUDAStream());
    return out;
}

void fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                        torch::Tensor weight, double eps) {
    check_norm_inputs(input, weight);
    check_norm_residual(input, residual);

    const at::cuda::OptionalCUDAGuard device_guard(input.device());

    fused_add_rms_norm_launch(input, residual, weight, eps,
                              at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm,
          "Root mean square normalization (CUDA)",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps"));
    m.def("fused_add_rms_norm", &fused_add_rms_norm,
          "Fused residual add + RMSNorm, applied in place (CUDA)",
          pybind11::arg("input"), pybind11::arg("residual"),
          pybind11::arg("weight"), pybind11::arg("eps"));
}
