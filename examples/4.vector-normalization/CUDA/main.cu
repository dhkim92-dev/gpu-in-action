#include <cuda_runtime.h>
#include <algorithm>
#include "cuda_helper.hpp"
#include "helper.hpp"

__global__ void square(
    const float *d_input,
    float *d_output,
    size_t n
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        d_output[tid] = d_input[tid] * d_input[tid];
    }
}

__global__ void reduce(
    const float *d_input,
    float *d_output,
    size_t n
) {
    extern __shared__ float cache[];
    const size_t lid = threadIdx.x;
    const size_t lsz = blockDim.x;
    const size_t gid = blockIdx.x * lsz * 2 + lid;

    float sum = 0.0f;
    if (gid < n) {
        sum = d_input[gid];
    }
    if (gid + lsz < n) {
        sum += d_input[gid + lsz];
    }
    cache[lid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = lsz / 2 ; offset > 0 ; offset>>=1 ) {
        if (lid < offset) {
            cache[lid] += cache[lid + offset];
        }
        __syncthreads();
    }

    if (lid == 0) {
        d_output[blockIdx.x] = cache[0];
    }
}

__global__ void normalize(
    const float *d_input,
    float *d_output,
    const float *d_square_sum,
    size_t n
) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float norm = sqrtf(d_square_sum[0]);
        d_output[tid] = d_input[tid] / norm;
    }
}

void h_vector_normalization(
    const float* input,
    float* output,
    size_t size
) {
    float norm = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        norm += input[i] * input[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] / norm;
    }
}

void gpu_square(
    const float* d_input,
    float* d_output,
    size_t n
) {
    size_t sz_blk = 256;
    size_t nr_blk = (n + sz_blk - 1) / sz_blk;

    square<<<nr_blk, sz_blk>>>(d_input, d_output, n);
    cuda_check(cudaGetLastError(), "Kernel launch square");
}

void gpu_reduce(
    float* &d_input,
    float* &d_output,
    size_t n
) {
    size_t sz_blk = 256;
    size_t nr_blk = (n + sz_blk * 2 - 1) / (sz_blk * 2);
    size_t sz_shm = sizeof(float) * sz_blk;
    while ( n > 1 ) {
        nr_blk = (n + sz_blk*2 - 1) / (sz_blk*2);
        reduce<<<nr_blk, sz_blk, sz_shm>>>(d_input, d_output, n);
        cuda_check(cudaGetLastError(), "Kernel launch reduce");
        std::swap(d_input, d_output);
        n = nr_blk;
    }
}

void gpu_normalize(
    const float* d_input,
    float* d_output,
    const float* d_square_sum,
    size_t n
) {
    size_t sz_blk = 256;
    size_t nr_blk = (n + sz_blk - 1) / sz_blk;
    normalize<<<nr_blk, sz_blk>>>(d_input, d_output, d_square_sum, n);
    cuda_check(cudaGetLastError(), "Kernel launch normalize");
}

int main(void)
{
    size_t n = 1 << 20; // 1M elements
    size_t sz_mem = sizeof(float) * n;

    float* h_input = new float[n];
    float* h_output_gpu = new float[n];
    float* h_output_cpu = new float[n];
    gpgpu_detail::rand_seeded = false;
    init_random_values_f32(h_input, static_cast<int>(n));

    float *d_input, *d_output, *d_ping, *d_pong, *d_square_sum;
    cuda_check(cudaMalloc((void**)&d_input, sz_mem), "cudaMalloc d_input");
    cuda_check(cudaMalloc((void**)&d_output, sz_mem), "cudaMalloc d_output");
    cuda_check(cudaMalloc((void**)&d_ping, sz_mem), "cudaMalloc d_ping");
    cuda_check(cudaMalloc((void**)&d_pong, sz_mem), "cudaMalloc d_pong");
    cuda_check(cudaMemcpy(d_input, h_input, sz_mem, cudaMemcpyHostToDevice), "cudaMemcpy to d_input");
    cuda_check(cudaMalloc((void**)&d_square_sum, sizeof(float)), "cudaMalloc d_square_sum");

    // GPU Vector Normalization
    BENCHMARK_START(gpu_vector_normalization)
    // Step 1: Square
    gpu_square(d_input, d_ping, n);
    // Step 2: Reduce to get sum of squares
    gpu_reduce(d_ping, d_pong, n);
    // Step 3. Copy the reduced sum to d_square_sum
    cudaMemcpy(d_square_sum, d_ping, sizeof(float), cudaMemcpyDeviceToDevice);
    // Step 4: Normalize
    gpu_normalize(d_input, d_output, d_square_sum, n);
    BENCHMARK_END(gpu_vector_normalization)
    cuda_check(cudaDeviceSynchronize(), "GPU Vector Normalization");
    cuda_check(cudaMemcpy(h_output_gpu, d_output, sz_mem, cudaMemcpyDeviceToHost), "cudaMemcpy to h_output_gpu");
    // CPU Vector Normalization
    BENCHMARK_START(cpu_vector_normalization)
    h_vector_normalization(h_input, h_output_cpu, n);
    BENCHMARK_END(cpu_vector_normalization)
    // Verify results
    for (size_t i = 0; i < n; ++i) {
        //  print first 10 results
        if (i < 10) {
            LOG_INFO("Result[%zu]: GPU=%f, CPU=%f", i, h_output_gpu[i], h_output_cpu[i]);
        }
        if (std::fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5f) {
            LOG_ERROR("Mismatch at index %zu: GPU=%f, CPU=%f", i, h_output_gpu[i], h_output_cpu[i]);
            break;
        }
    }

    return 0;
}
