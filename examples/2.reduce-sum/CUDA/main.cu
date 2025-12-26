#include <cuda_runtime.h>
#include <iostream>
#include <utility>
#include <cstddef>

#include "helper.hpp"
#include "cuda_helper.hpp"

__global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
	extern __shared__ float sdata[];

	const int tid = (int)threadIdx.x;
	const int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	float sum = 0.0f;

	if (gid < n) {
		sum = input[gid];
	}
	if (gid + blockDim.x < n) {
		sum += input[gid + blockDim.x];
	}
	sdata[tid] = sum;
	__syncthreads();

	// Reduction in shared memory
	for (int stride = ((int)blockDim.x) / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sdata[tid] += sdata[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[(int)blockIdx.x] = sdata[0];
	}
}


static float host_reduce_sum(const float* data, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

int main()
{
    const size_t input_size = 1024;
    const size_t sz_mem_input = sizeof(float) * input_size;
    const size_t sz_blk = 256;
    const size_t sz_shmem = sizeof(float) * sz_blk;

    float* h_input = new float[input_size];
    init_random_values_f32(h_input, input_size);

    float *d_in = nullptr;
    float *d_out = nullptr;

    cuda_check(cudaMalloc((void**)&d_in, sz_mem_input), "cudaMalloc(&d_in, sz_mem_input)");
    cuda_check(cudaMalloc((void**)&d_out, sz_mem_input), "cudaMalloc(&d_out, sz_memb_input)");
    cuda_check(cudaMemcpy(d_in, h_input, sz_mem_input, cudaMemcpyHostToDevice), "cudaMemcpy(d_in, d_out)");

    size_t current_n = input_size;
    BENCHMARK_START(cuda_reduce_sum)
    while (current_n > 1) {
        size_t nr_blk = (current_n + (sz_blk * 2 - 1)) / (sz_blk * 2);
        size_t nr_works= nr_blk * sz_blk;
        reduce_sum<<<nr_works, sz_blk, sz_shmem>>>(d_in, d_out, static_cast<int>(current_n));
        cuda_check(cudaGetLastError(), "kernel launch");
        std::swap(d_in, d_out);
        current_n = nr_blk;
    }
    BENCHMARK_END(cuda_reduce_sum)
    float h_output = 0.0f;
    cuda_check(cudaMemcpy(&h_output, d_in, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(h_output, d_in)");
    float host_sum_result = 0.0f;
    BENCHMARK_START(cpu_reduce_sum) 
    host_sum_result = host_reduce_sum(h_input, static_cast<int>(input_size));
    BENCHMARK_END(cpu_reduce_sum)

    LOG_INFO("Reduction Sum CPU Result: %f", host_sum_result);
    LOG_INFO("Reduction Sum CUDA Result: %f", h_output);

    delete[] h_input;
    return 0;
}
