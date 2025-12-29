#include "cuda_helper.hpp"
#include "helper.hpp"
#include "kernel.cuh"

bool compare_matrices(
    const float* mat1,
    const float* mat2,
    int M,
    int N,
    float epsilon = 1e-4f
) {
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(mat1[i] - mat2[i]) > epsilon) {
            LOG_ERROR("Mismatch at index %d: mat1=%f, mat2=%f", i, mat1[i], mat2[i]);
            return false;
        }
    }
    return true;
}

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
    CUDA_BENCHMARK(cuda_matmul_naive, matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N));
    cudaDeviceSynchronize();
}

void gpu_matmul_tiled(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    CUDA_BENCHMARK(cuda_matmul_tiled, matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N));
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

    auto h_A = std::make_unique<float[]>(sz_A);
    auto h_B = std::make_unique<float[]>(sz_B);
    auto h_C = std::make_unique<float[]>(sz_C);

    init_random_values_f32(h_A.get(), sz_A);
    init_random_values_f32(h_B.get(), sz_B);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaMalloc((void**)&d_A, sz_A * sizeof(float));
    cudaMalloc((void**)&d_B, sz_B * sizeof(float));
    cudaMalloc((void**)&d_C, sz_C * sizeof(float));
    cudaMemcpy(d_A, h_A.get(), sz_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.get(), sz_B * sizeof(float), cudaMemcpyHostToDevice);

    HOST_BENCHMARK(host_matmul(h_A.get(), h_B.get(), h_C.get(), M, K, N), host_matmul);
    gpu_matmul_naive(d_A, d_B, d_C, M, K, N);
    auto h_C_gpu_naive = std::make_unique<float[]>(sz_C);
    cudaMemcpy(h_C_gpu_naive.get(), d_C, sz_C * sizeof(float), cudaMemcpyDeviceToHost);
    auto result = compare_matrices(h_C.get(), h_C_gpu_naive.get(), M, N);
    LOG_INFO("Verification of naive GPU matmul: %s", result ? "PASSED" : "FAILED");
    gpu_matmul_tiled(d_A, d_B, d_C, M, K, N);
    auto h_C_gpu_tiled = std::make_unique<float[]>(sz_C);
    cudaMemcpy(h_C_gpu_tiled.get(), d_C, sz_C * sizeof(float), cudaMemcpyDeviceToHost);
    result = compare_matrices(h_C.get(), h_C_gpu_tiled.get(), M, N);
    LOG_INFO("Verification of tiled GPU matmul: %s", result ? "PASSED" : "FAILED");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}