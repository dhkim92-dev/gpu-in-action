#include <cuda_runtime.h>
#include <iostream>
#include <utility>

#include "helper.hpp"

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
    const int local_size = 32; // "local_size = 32float" -> 32 threads, 32 floats in shared
    const size_t shared_bytes = (size_t)local_size * sizeof(float);
    const size_t bytes = input_size * sizeof(float);

    float* h_input = new float[input_size];
    init_random_values_f32(h_input, (int)input_size);
    // for (size_t i = 0; i < input_size; ++i) h_input[i] = (float)(i + 1);

    float* d_a = nullptr;
    float* d_b = nullptr;
    cuda_check(cudaMalloc((void**)&d_a, bytes), "cudaMalloc(d_a)");
    cuda_check(cudaMalloc((void**)&d_b, bytes), "cudaMalloc(d_b)");
    cuda_check(cudaMemcpy(d_a, h_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    float* d_in = d_a;
    float* d_out = d_b;
    size_t n = input_size;

    while (n > 1) {
        const size_t blocks = (n + local_size - 1) / local_size;
        reduce_sum<<<(unsigned)blocks, (unsigned)local_size, shared_bytes>>>(d_in, d_out, (int)n);
        cuda_check(cudaGetLastError(), "reduce_sum launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        std::swap(d_in, d_out);
        n = blocks;
    }

    float h_output = 0.0f;
    cuda_check(cudaMemcpy(&h_output, d_in, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    const float expected = host_reduce_sum(h_input, input_size);
    std::cout << "GPU result: " << h_output << ", CPU result: " << expected << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_input;
    return 0;
}