#include "kernel.cuh"

/**
 * Naive Matrix Multiplication Kernel
 * C[M][N] = A[M][K] * B[K][N]
 * Each thread computes one element of the output matrix C
 * by accumulating results into a local variable.
 * Parameters:
 *   A: Input matrix A of size M x K
 *   B: Input matrix B of size K x N
 *   C: Output matrix C of size M x N
 *   M: Number of rows in matrix A and C
 *   K: Number of columns in matrix A and rows in matrix B
 *   N: Number of columns in matrix B and C
 */
__global__ void matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    __shared__ float sub_A[16][16];
    __shared__ float sub_B[16][16];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int l_row = threadIdx.y;
    unsigned int l_col = threadIdx.x;
    float value = 0.0f;

    sub_A[l_row][l_col] = 0.0f;
    sub_B[l_row][l_col] = 0.0f;
    
    // load tiles 
    for (unsigned int t = 0; t < (K + 16 - 1) / 16; ++t) {
        if (row < M && t * 16 + l_col < K) {
            sub_A[l_row][l_col] = A[row * K + t * 16 + l_col];
        } else {
            sub_A[l_row][l_col] = 0.0f;
        }
        if (t * 16 + l_row < K &&  col < N) {
            sub_B[l_row][l_col] = B[(t * 16 + l_row) * N + col];
        } else {
            sub_B[l_row][l_col] = 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < 16; ++k) {
            value += sub_A[l_row][k] * sub_B[k][l_col];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}