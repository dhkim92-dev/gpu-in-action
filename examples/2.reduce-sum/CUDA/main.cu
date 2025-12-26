#include <cuda_runtime.h>
#include <iostream>
#include <utility>

#include "helper.hpp"
#include "cuda_helper.hpp"

// Kernel is defined in kernel.cu and linked in.
extern "C" __global__ void reduce_sum(const float* input, float* output, int n);

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
    init_random_inputs_f32(h_input, input_size);

    float *d_in = nullptr;
    float *d_out = nullptr;

    cuda_check(cudaMalloc((void**)&d_in, sz_mem_input));
    cuda_check(cudaMalloc((void**)&d_out, sz_mem_input));
    cuda_check(cudaMemcpy(d_in, h_input, sz_mem_input, cudaMemcpyHostToDevice));

    size_t nr_blk = (input_size + (sz_blk * 2 - 1)) / (sz_blk * 2);

    BENCHMARK_START(cuda_reduce_sum)
    while (nr_blk > 1) {
        reduce_sum<<<nr_blk, sz_blk, sz_shmem>>>(d_in, d_out, static_cast<int>(input_size));
        cuda_check(cudaGetLastError());
        std::swap(d_in, d_out);
        input_size = nr_blk;
        nr_blk = (input_size + (sz_blk * 2 - 1)) / (sz_blk * 2);
    }
    BENCHMARK_END(cuda_reduce_sum);
    float h_output = 0.0f;
    cuda_check(cudaMemcpy(&h_output, d_in, sizeof(float), cudaMemcpyDeviceToHost));
    float host_sum_result = 0.0f;
    BENCHMARK_START(cpu_reduce_sum) 
    host_sum_result = host_reduce_sum(h_input, static_cast<int>(input_size));
    BENCHMARK_END(cpu_reduce_sum)

    LOG_INFO("Reduction Sum CPU Result: %f", host_sum_result);
    LOG_INFO("Reduction Sum CUDA Result: %f", h_output);

    delete[] h_input;
    return 0;
}
