#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "pos_encoding.h"

torch::Tensor rotary_embedding_neox(
    const torch::Tensor& postions,
    const torch::Tensor& query,
    const torch::Tensor& key,
    int head_size,
    const torch::Tensor& cos_sin_cache,
    c10::cuda::CUDAStream stream) 
{
    auto out = torch::empty_like(query);
    const at::cuda::OptionalCUDAGuard device_guard(query.device());
    rotary_embedding_neox_lanch(out, postions, query, key, head_size, cos_sin_cache, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_embedding_neox", &rotary_embedding_neox,
          "Apply GPT-Neox style rotary enbedding to query and key (CUDA)",
          pybind11::arg("postions"), pybind11::arg("query"),
          pybind11::arg("key"), pybind11::arg("head_size"),
          pybind11::arg("cos_sin_cache"));
}
