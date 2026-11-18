#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <pybind11/stl.h>

#include <cstdint>
#include <unordered_map>

#include "cache.h"

namespace {

// Number of bytes occupied by a single block of a contiguous cache tensor.
int64_t block_size_in_bytes(const torch::Tensor& cache) {
    TORCH_CHECK(cache.dim() >= 1, "cache tensors must have at least one dimension");
    TORCH_CHECK(cache.size(0) > 0, "cache tensors must contain at least one block");
    return cache.element_size() * (cache.numel() / cache.size(0));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swap_blocks", &swap_blocks,
          "Copy KV cache blocks from src to dst (CUDA)",
          pybind11::arg("src"), pybind11::arg("dst"),
          pybind11::arg("block_mapping"));
}
