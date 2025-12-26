#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
//#include "helper.hpp"

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

#ifndef CHECK_CUDA
#define CHECK_CUDA(err) \
{ \
    if (err != cudaSuccess)  \
    { \
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
    } \
}
#endif

void init_random_values(float* data, int size) 
{
    for (int i = 0; i < size; ++i) 
    {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(void) 
{
    const int n = 1024;
    std::vector<float> h_A(n), h_B(n), h_C(n);
    init_random_values(h_A.data(), n);
    init_random_values(h_B.data(), n);
    float *d_A, *d_B, *d_C;

    cudaError_t err;
    CHECK_CUDA(cudaMalloc((void**)&d_A, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), n * sizeof(float), cudaMemcpyHostToDevice ));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), n * sizeof(float), cudaMemcpyHostToDevice ));
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, n * sizeof(float), cudaMemcpyDeviceToHost ));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    std::cout << "Vector A : ";
    for (int i = 0; i < 10; ++i) 
    {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\nVector B : ";
    for (int i = 0; i < 10; ++i) 
    {
        std::cout << h_B[i] << " ";
    }
    std::cout << "\nVector C (A + B) : ";
    for (int i = 0; i < 10; ++i)
    {
    // .2f 표기>>
        std::cout << std::fixed << std::setprecision(2) << h_C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
