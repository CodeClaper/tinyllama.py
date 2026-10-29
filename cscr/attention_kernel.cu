#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "attention.h"

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ float block_reduce_sum(float* shared, int tid, int n, float val) 
{
    shared[tid] = val;
    __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

template <typename scalar_t>
__global__ void paged_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale) 
{
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int group_size = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / group_size;

    extern __shared__ float smem[];
    float* q_s = smem;
    float* acc_s = smem + head_dim;
    float* red_s = smem + 2 * head_dim;

    const int64_t q_offset =
        (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_dim;
    const scalar_t* q_ptr = query + q_offset;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_s[d] = static_cast<float>(q_ptr[d]);
        acc_s[d] = 0.0f;
    }
    __syncthreads();

    const int seq_len = seq_lens[seq_idx];
    const int* block_table = block_tables + static_cast<int64_t>(seq_idx) * max_blocks_per_seq;

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int pos = 0; pos < seq_len; ++pos) {
        const int logical_block = pos / block_size;
        const int slot = pos % block_size;
        const int physical_block = block_table[logical_block];
        if (physical_block < 0) {
            continue;
        }

        const int64_t cache_offset =
            ((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
                 block_size +
             slot) *
            head_dim;

        const scalar_t* k_ptr = key_cache + cache_offset;

        float qk = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            qk += q_s[d] * static_cast<float>(k_ptr[d]);
        }
        qk = block_reduce_sum(red_s, tid, blockDim.x, qk);

        const float score = qk * scale;
        const float new_max = fmaxf(running_max, score);
        const float alpha = __expf(running_max - new_max);
        const float p = __expf(score - new_max);

        running_sum = running_sum * alpha + p;
        running_max = new_max;

        const scalar_t* v_ptr = value_cache + cache_offset;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            acc_s[d] = acc_s[d] * alpha + p * static_cast<float>(v_ptr[d]);
        }
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    scalar_t* out_ptr = out + q_offset;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = static_cast<scalar_t>(acc_s[d] * inv_sum);
    }
}

}  // namespace

void paged_attention_launch(
    torch::Tensor& out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    double scale,
    c10::cuda::CUDAStream stream) 
{
    const int num_seqs = static_cast<int>(query.size(0));
    const int num_heads = static_cast<int>(query.size(1));
    const int head_dim = static_cast<int>(query.size(2));
    const int num_kv_heads = static_cast<int>(key_cache.size(1));
    const int block_size = static_cast<int>(key_cache.size(2));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    const dim3 grid(num_seqs, num_heads);
    const int threads = kThreads;
    const size_t shared_bytes = (2 * head_dim + threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
        "paged_attention_kernel", [&] {
            paged_attention_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
                out.data_ptr<scalar_t>(),
                query.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                block_tables.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
                static_cast<float>(scale));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
