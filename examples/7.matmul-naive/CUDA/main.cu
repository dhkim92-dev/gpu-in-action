#include "cuda_helper.hpp"
#include "helper.hpp"
#include "kernel.cuh"

void host_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            C[row * N + col] = 0.0f;
            for (int k = 0; k < K; ++k) {
                C[row * N + col] += A[row * K + k] * B[k * N + col];
            }
        }
    }
}

void gpu_matmul_naive(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    matmul_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
}

void gpu_matmul_tiled(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    matmul_tiled_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
}

int main(void)
{
    const uint32_t M = 1024;
    const uint32_t K = 768;
    const uint32_t N = 512;
    const size_t sz_A = M * K;
    const size_t sz_B = K * N;
    const size_t sz_C = M * N;

    float* h_A = new float[sz_A];
    float* h_B = new float[sz_B];
    float* h_C = new float[sz_C];

    init_random_values_f32(h_A, sz_A);
    init_random_values_f32(h_B, sz_B);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaMalloc((void**)&d_A, sz_A * sizeof(float));
    cudaMalloc((void**)&d_B, sz_B * sizeof(float));
    cudaMalloc((void**)&d_C, sz_C * sizeof(float));
    cudaMemcpy(d_A, h_A, sz_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz_B * sizeof(float), cudaMemcpyHostToDevice);

    HOST_BENCHMARK(host_matmul(h_A, h_B, h_C, M, K, N), host_matmul);
}