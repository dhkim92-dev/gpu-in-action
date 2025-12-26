#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include "helper.hpp"
#include "cuda_helper.hpp"

__global__ void dot_product_kernel(
    const float* a, 
    const float* b, 
    float* partial_sum, 
    size_t n
) {
    extern __shared__ float cache[];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t cache_index = threadIdx.x;
    float temp = 0.0f;

    while (tid < n)
    {
        temp += a[tid] * b[tid]; // 
        tid += blockDim.x * gridDim.x;
    }

    cache[cache_index] = temp; // store elementwise product in shared memory
    __syncthreads();

    // Reduction in shared memory
    size_t i = blockDim.x / 2;
    while (i != 0)
    {
        if (cache_index < i)
        {
            cache[cache_index] += cache[cache_index + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cache_index == 0)
    {
        partial_sum[blockIdx.x] = cache[0];
    }
}

__global__ void reduce_sum_kernel(
    const float* inputs,
    float* outputs,
    size_t n
) {
    extern __shared__ float cache[];
    const size_t lid = threadIdx.x;
    const size_t lsz = blockDim.x;
    const size_t gid = blockIdx.x * lsz * 2 + lid;

    float sum = 0.0f;
    if (gid < n) {
        sum = inputs[gid];
    }
    if (gid + lsz < n) {
        sum += inputs[gid + lsz];
    }

    cache[lid] = sum;
    __syncthreads();

    for (size_t stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            cache[lid] += cache[lid + stride];
        }
        __syncthreads();
    }

    if (lid == 0) {
        outputs[blockIdx.x] = cache[0];
    }
}

void host_dot_product(const float* a, const float* b, float* result, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    *result = sum;
}

int main(void)
{
    const size_t n = 16384;
    const size_t sz_mem_vec = sizeof(float) * n;
    const size_t threads_per_block = 256;

    // grid-stride loop이므로 grid 크기를 적당히 제한해도 전체 n을 커버할 수 있습니다.
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const size_t blocks_for_input = (n + threads_per_block - 1) / threads_per_block;
    const size_t max_blocks = static_cast<size_t>(prop.multiProcessorCount) * 32;
    const size_t blocks_per_grid = std::min(blocks_for_input, max_blocks);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float h_dot_gpu = 0.0f;
    float h_dot_cpu = 0.0f;
    float *d_a, *d_b;
    float *d_ping, *d_pong;
    size_t sz_mem_partial_sum = sizeof(float) * blocks_per_grid;

    init_random_values_f32(h_a, n);
    init_random_values_f32(h_b, n);
    
    cuda_check(cudaMalloc((void**)&d_a, sz_mem_vec), "cudaMalloc d_a");
    cuda_check(cudaMalloc((void**)&d_b, sz_mem_vec), "cudaMalloc d_b");
    cuda_check(cudaMalloc((void**)&d_ping, sz_mem_partial_sum), "cudaMalloc d_ping");
    cuda_check(cudaMalloc((void**)&d_pong, sz_mem_partial_sum), "cudaMalloc d_pong");

    cuda_check(cudaMemcpy(d_a, h_a, sz_mem_vec, cudaMemcpyHostToDevice), "cudaMemcpy to d_a");
    cuda_check(cudaMemcpy(d_b, h_b, sz_mem_vec, cudaMemcpyHostToDevice), "cudaMemcpy to d_b");

    BENCHMARK_START(gpu_dot_product)
    dot_product_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(
        d_a,
        d_b,
        d_ping,
        n
    );
    cuda_check(cudaGetLastError(), "dot_product_kernel launch");

    // partial sums를 GPU에서 계속 줄여서 최종 1개 값까지 만듭니다.
    size_t current_n = blocks_per_grid;
    while (current_n > 1) {
        const size_t groups = (current_n + (threads_per_block * 2) - 1) / (threads_per_block * 2);
        reduce_sum_kernel<<<groups, threads_per_block, threads_per_block * sizeof(float)>>>(
            d_ping,
            d_pong,
            current_n
        );
        cuda_check(cudaGetLastError(), "reduce_sum_kernel launch");
        std::swap(d_ping, d_pong);
        current_n = groups;
    }

    cuda_check(cudaDeviceSynchronize(), "Kernel execution");
    BENCHMARK_END(gpu_dot_product)

    cuda_check(cudaMemcpy(&h_dot_gpu, d_ping, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy to h_dot_gpu");

    BENCHMARK_START(cpu_dot_product)
    host_dot_product(h_a, h_b, &h_dot_cpu, n);
    BENCHMARK_END(cpu_dot_product)

    LOG_INFO("Dot Product CPU Result: %f", h_dot_cpu);
    LOG_INFO("Dot Product CUDA Result: %f", h_dot_gpu);

    cuda_check(cudaFree(d_a), "cudaFree d_a");
    cuda_check(cudaFree(d_b), "cudaFree d_b");
    cuda_check(cudaFree(d_ping), "cudaFree d_ping");
    cuda_check(cudaFree(d_pong), "cudaFree d_pong");

    delete[] h_a;
    delete[] h_b;

    return 0;
}