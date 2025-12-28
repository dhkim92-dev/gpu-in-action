#include "cuda_helper.hpp"
#include "helper.hpp"
#include "kernel.cuh"

void h_matrix_tranpose(
    const float* input, 
    float *output, 
    const int width, 
    const int height)
{
    for ( int y = 0; y < height; y++ ) {
        for ( int x = 0; x < width; x++ ) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

void gpu_transpose_naive(
    const float* d_input,
    float* d_output,
    float* h_output,
    const int width,
    const int height
)
{
    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3( (width + blockSize.x - 1) / blockSize.x,
                          (height + blockSize.y - 1) / blockSize.y );
    transpose_naive<<< gridSize, blockSize >>>(d_input, d_output, width, height);
    cuda_check(cudaMemcpy(h_output, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost), "memcpy failed, d_output to h_output");
}

void gpu_transpose_tiled_bank_conflict(
    const float* d_input,
    float* d_output,
    float* h_output,
    const int width,
    const int height
)
{
    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3( (width + blockSize.x - 1) / blockSize.x,
                          (height + blockSize.y - 1) / blockSize.y );
    transpose_tiled_bank_conflict<<< gridSize, blockSize >>>(d_input, d_output, width, height);
    cuda_check(cudaMemcpy(h_output, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost), "memcpy failed, d_output to h_output");
}

void gpu_transpose_tiled_optimized(
    const float* d_input,
    float* d_output,
    float* h_output,
    const int width,
    const int height
)
{
    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3( (width + blockSize.x - 1) / blockSize.x,
                          (height + blockSize.y - 1) / blockSize.y );
    transpose_tiled_optimized<<< gridSize, blockSize >>>(d_input, d_output, width, height);
    cuda_check(cudaMemcpy(h_output, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost), "memcpy failed, d_output to h_output");
}

bool verify_result(
    const float* h_output_gpu,
    const float* h_output_cpu,
    const int n
)
{
    for ( int i = 0 ; i < n ; ++i ) {
        if ( h_output_gpu[i] != h_output_cpu[i] ) {
            LOG_DEBUG("Mismatch at index %d: GPU %d != CPU %d", i, h_output_gpu[i], h_output_cpu[i]);
            return false;
        }
    }
    return true;
}

int main(void)
{
    const int width  = 1920;
    const int height = 1080;
    const size_t sz_mem = sizeof(float) * width * height;
    float* h_input = new float[width * height];
    float* h_output = new float[width * height];
    float* h_output_gpu = new float[width * height];

    gpgpu_detail::rand_seeded = true;
    init_random_values_f32(h_input, width * height);

    float* d_input = nullptr;
    float* d_output = nullptr;
    cuda_check(cudaMalloc((void**)&d_input, sz_mem), "malloc failed, d_input");
    cuda_check(cudaMalloc((void**)&d_output, sz_mem), "malloc failed, d_output");
    cuda_check(cudaMemcpy(d_input, h_input, sz_mem, cudaMemcpyHostToDevice), "memcpy failed, h_input to d_input");

    // Host transpose
    BENCHMARK_START(host_transpose)
    h_matrix_tranpose(h_input, h_output, width, height);
    BENCHMARK_END(host_transpose)

    // GPU naive transpose
    BENCHMARK_START(gpu_transpose_naive) 
    gpu_transpose_naive(d_input, d_output, h_output_gpu, width, height);
    BENCHMARK_END(gpu_transpose_naive)
    bool result = verify_result(h_output_gpu, h_output, width * height) ;
    LOG_INFO("GPU naive transpose %s", result ? "PASSED" : "FAILED");

    // d_output fill zero
    cuda_check(cudaMemset(d_output, 0, sz_mem), "cudaMemset failed, d_output");
    // GPU tiled transpose with bank conflict
    BENCHMARK_START(gpu_transpose_tiled_bank_conflict)
    gpu_transpose_tiled_bank_conflict(d_input, d_output, h_output, width, height);
    BENCHMARK_END(gpu_transpose_tiled_bank_conflict)
    result = verify_result(h_output_gpu, h_output, width * height) ;
    LOG_INFO("GPU tiled transpose with bank conflict %s", result ? "PASSED" : "FAILED");

    // d_output fill zero
    cuda_check(cudaMemset(d_output, 0, sz_mem), "cudaMemset failed, d_output");
    // GPU tiled optimized transpose
    BENCHMARK_START(gpu_transpose_tiled_optimized)
    gpu_transpose_tiled_optimized(d_input, d_output, h_output, width, height);
    BENCHMARK_END(gpu_transpose_tiled_optimized)    
    result = verify_result(h_output_gpu, h_output, width * height) ;
    LOG_INFO("GPU tiled optimized transpose %s", result ? "PASSED" : "FAILED");

    cuda_check(cudaFree(d_input), "cudaFree failed");
    cuda_check(cudaFree(d_output), "cudaFree failed");
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_gpu;

    return 0;
}