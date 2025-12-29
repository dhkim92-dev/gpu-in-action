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

#define CUDA_BENCHMARK(func_call, benchmark_name) { \
    cudaEvent_t start_##benchmark_name, stop_##benchmark_name; \
    cudaCheckError(cudaEventCreate(&start_##benchmark_name)); \
    cudaCheckError(cudaEventCreate(&stop_##benchmark_name)); \
    cudaCheckError(cudaEventRecord(start_##benchmark_name)); \
    func_call; \
    cudaCheckError(cudaEventRecord(stop_##benchmark_name)); \
    cudaCheckError(cudaEventSynchronize(stop_##benchmark_name)); \
    float elapsed_time_##benchmark_name = 0.0f; \
    cudaCheckError(cudaEventElapsedTime(&elapsed_time_##benchmark_name, start_##benchmark_name, stop_##benchmark_name)); \
    std::printf("[BENCH][%s:%d] %s : %.3f ms\n", __FILE__, __LINE__, #benchmark_name, elapsed_time_##benchmark_name); \
    cudaCheckError(cudaEventDestroy(start_##benchmark_name)); \
    cudaCheckError(cudaEventDestroy(stop_##benchmark_name)); \
}
