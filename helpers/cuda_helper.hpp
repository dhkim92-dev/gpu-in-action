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
