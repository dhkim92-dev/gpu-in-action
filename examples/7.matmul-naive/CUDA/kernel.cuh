#include <cuda_runtime.h>

__global__ void matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    uint32_t M,
    uint32_t K,
    uint32_t N
);
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    uint32_t M,
    uint32_t K,
    uint32_t N
);