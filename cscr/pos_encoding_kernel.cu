#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "pos_encoding.h"

namespace {

template <typename scalar_t, typename index_t>
__global__ void rotary_embedding_neox_kernel(
    scalar_t* __restrict__ out,
    const index_t* __restrict__ positions,
    const scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache,
    int rot_dim,
    int64_t out_stride,
    int64_t query_stride,
    int64_t key_stride,
    int num_heads,
    int num_kv_heads,
    int head_size) 
{
    const int64_t token_idx = blockIdx.x;
    const int64_t pos = static_cast<int64_t>(positions[token_idx]);
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
    const int embed_dim = rot_dim / 2;

    const int query_n = num_heads * embed_dim;
    for (int i = threadIdx.x; i < query_n; i += blockDim.x) {
        const int head_idx = i / embed_dim;
        const int rot_offset = i - head_idx * embed_dim;
        const int64_t token_head =
            token_idx * query_stride + static_cast<int64_t>(head_idx) * head_size;
        const int x_index = rot_offset;
        const int y_index = rot_offset + embed_dim;

        const float cos = static_cast<float>(cache_ptr[rot_offset]);
        const float sin = static_cast<float>(cache_ptr[rot_offset + embed_dim]);
        const float x = static_cast<float>(query[token_head + x_index]);
        const float y = static_cast<float>(query[token_head + y_index]);

        scalar_t* out_head = out + token_idx * out_stride +
                             static_cast<int64_t>(head_idx) * head_size;
        out_head[x_index] = static_cast<scalar_t>(x * cos - y * sin);
        out_head[y_index] = static_cast<scalar_t>(y * cos + x * sin);
    }

    const int key_n = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < key_n; i += blockDim.x) {
        const int head_idx = i / embed_dim;
        const int rot_offset = i - head_idx * embed_dim;
        const int64_t token_head =
            token_idx * key_stride + static_cast<int64_t>(head_idx) * head_size;
        const int x_index = rot_offset;
        const int y_index = rot_offset + embed_dim;

        const float cos = static_cast<float>(cache_ptr[rot_offset]);
        const float sin = static_cast<float>(cache_ptr[rot_offset + embed_dim]);
        const float x = static_cast<float>(key[token_head + x_index]);
        const float y = static_cast<float>(key[token_head + y_index]);

        key[token_head + x_index] = static_cast<scalar_t>(x * cos - y * sin);
        key[token_head + y_index] = static_cast<scalar_t>(y * cos + x * sin);
    }
}

}  // namespace

void rotary_embedding_neox_lanch(
    torch::Tensor& out,
    const torch::Tensor& postions,
    const torch::Tensor& query,
    const torch::Tensor& key,
    int head_size,
    const torch::Tensor& cos_sin_cache,
    c10::cuda::CUDAStream stream) 
{
    TORCH_CHECK(postions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be a CUDA tensor");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(cos_sin_cache.is_contiguous(), "cos_sin_cache must be contiguous");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key must share dtype");
    TORCH_CHECK(query.scalar_type() == cos_sin_cache.scalar_type(), "query and cos_sin_cache must share dtype");
    TORCH_CHECK(query.scalar_type() == out.scalar_type(), "query and out must share dtype");
    TORCH_CHECK(query.dim() == 2, "query must be [num_tokens, num_heads * head_size]");
    TORCH_CHECK(key.dim() == 2, "key must be [num_tokens, num_kv_heads * head_size]");
    TORCH_CHECK(postions.dim() == 1, "positions must be [num_tokens]");

    const int64_t num_tokens = postions.numel();
    TORCH_CHECK(num_tokens > 0, "positions must contain at least one token");
    TORCH_CHECK(query.size(0) == num_tokens, "query must have one row per token");
    TORCH_CHECK(key.size(0) == num_tokens, "key must have one row per token");
    TORCH_CHECK(head_size > 0, "head_size must be positive");
    TORCH_CHECK(query.size(1) % head_size == 0, "query size must be divisible by head_size");
    TORCH_CHECK(key.size(1) % head_size == 0, "key size must be divisible by head_size");

    const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
    TORCH_CHECK(rot_dim > 0 && rot_dim % 2 == 0, "rotary dimension must be even and positive");
    TORCH_CHECK(rot_dim <= head_size, "rotary dimension must not exceed head_size");

    const int num_heads = static_cast<int>(query.size(1) / head_size);
    const int num_kv_heads = static_cast<int>(key.size(1) / head_size);

    const int threads = rot_dim < 1024 ? rot_dim : 1024;
    const dim3 grid(static_cast<unsigned int>(num_tokens));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
        "rotary_embedding_neox_kernel", [&] {
            AT_DISPATCH_INDEX_TYPES(postions.scalar_type(), "rotary_embedding_neox_positions", [&] {
                rotary_embedding_neox_kernel<scalar_t, index_t>
                    <<<grid, threads, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        postions.data_ptr<index_t>(),
                        query.data_ptr<scalar_t>(),
                        key.data_ptr<scalar_t>(),
                        cos_sin_cache.data_ptr<scalar_t>(),
                        rot_dim,
                        out.stride(0),
                        query.stride(0),
                        key.stride(0),
                        num_heads,
                        num_kv_heads,
                        head_size);
            });
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
