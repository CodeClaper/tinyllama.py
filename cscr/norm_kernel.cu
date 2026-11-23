#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include "norm.h"

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float block_reduce_sum(float* shared, int tid, int n,
                                                  float val) {
    shared[tid] = val;
    __syncthreads();
    for (int stride = n >> 1; stride > 0; stride >>= 1) {
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
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int64_t hidden_size,
    float eps) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t offset = row * hidden_size;

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        const float x = static_cast<float>(input[offset + i]);
        sum_sq += x * x;
    }
    sum_sq = block_reduce_sum(shared, tid, blockDim.x, sum_sq);

    const float mean_sq = sum_sq / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_sq + eps);

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        const float x = static_cast<float>(input[offset + i]);
        const float w = static_cast<float>(weight[i]);
        out[offset + i] = static_cast<scalar_t>(x * inv_rms * w);
    }
}

template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    int64_t hidden_size,
    float eps) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t offset = row * hidden_size;

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        const float x = static_cast<float>(input[offset + i]) +
                        static_cast<float>(residual[offset + i]);
        sum_sq += x * x;
        residual[offset + i] = static_cast<scalar_t>(x);
    }
    sum_sq = block_reduce_sum(shared, tid, blockDim.x, sum_sq);

    const float mean_sq = sum_sq / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_sq + eps);

    for (int64_t i = tid; i < hidden_size; i += blockDim.x) {
        const float x = static_cast<float>(residual[offset + i]);
        const float w = static_cast<float>(weight[i]);
        input[offset + i] = static_cast<scalar_t>(x * inv_rms * w);
    }
}

}  // namespace

void rms_norm_launch(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps,
    c10::cuda::CUDAStream stream) {
    const int64_t hidden_size = input.size(-1);
    const int64_t num_tokens = input.numel() / hidden_size;
    if (num_tokens == 0) {
        return;
    }

    const dim3 grid(static_cast<unsigned int>(num_tokens));
    const int threads = kThreads;
    const size_t shared_bytes = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "rms_norm_kernel", [&] {
            rms_norm_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                hidden_size,
                static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_add_rms_norm_launch(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps,
    c10::cuda::CUDAStream stream) {
    const int64_t hidden_size = input.size(-1);
    const int64_t num_tokens = input.numel() / hidden_size;
    if (num_tokens == 0) {
        return;
    }

    const dim3 grid(static_cast<unsigned int>(num_tokens));
    const int threads = kThreads;
    const size_t shared_bytes = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "fused_add_rms_norm_kernel", [&] {
            fused_add_rms_norm_kernel<scalar_t>
                <<<grid, threads, shared_bytes, stream>>>(
                    input.data_ptr<scalar_t>(),
                    residual.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    hidden_size,
                    static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
