#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(err) << std::endl;
        std::exit((int)err);
    }
}

#define CUDA_BENCHMARK(benchmark_name, ...) { \
    cudaEvent_t start_##benchmark_name, stop_##benchmark_name; \
    cuda_check(cudaEventCreate(&start_##benchmark_name), "cudaEventCreate start"); \
    cuda_check(cudaEventCreate(&stop_##benchmark_name), "cudaEventCreate stop"); \
    cuda_check(cudaEventRecord(start_##benchmark_name), "cudaEventRecord start"); \
    __VA_ARGS__; \
    cuda_check(cudaEventRecord(stop_##benchmark_name), "cudaEventRecord stop"); \
    cuda_check(cudaEventSynchronize(stop_##benchmark_name), "cudaEventSynchronize stop"); \
    float elapsed_time_##benchmark_name = 0.0f; \
    cuda_check(cudaEventElapsedTime(&elapsed_time_##benchmark_name, start_##benchmark_name, stop_##benchmark_name), "cudaEventElapsedTime"); \
    std::printf("[BENCH][%s:%d] %s : %.3f ms\n", __FILE__, __LINE__, #benchmark_name, elapsed_time_##benchmark_name); \
    cuda_check(cudaEventDestroy(start_##benchmark_name), "cudaEventDestroy start"); \
    cuda_check(cudaEventDestroy(stop_##benchmark_name), "cudaEventDestroy stop"); \
}
