#include "kernel.cuh"
#include "cuda_helper.hpp"
#include "helper.hpp"

bool compare_result(
    const float* ref,
    const float* res,
    const int size,
    const float epsilon = 1e-4f
) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(ref[i] - res[i]) > epsilon) {
            LOG_ERROR("Mismatch at index %d: ref=%f, res=%f", i, ref[i], res[i]);
            return false;
        }
    }
    return true;
}

// output size same with input size (no padding, no stride)
void h_conv2d(
    const float *input,
    float *output,
    const float *filter,
    const int W,
    const int H,
    const int FW,
    const int FH,
) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < FH; ++ky) {
                for (int kx = 0; kx < FW; ++kx) {
                    int in_x = x + kx - FW / 2;
                    int in_y = y + ky - FH / 2;
                    if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                        sum += input[in_y * W + in_x] * filter[ky * FW + kx];
                    }
                }
            }
            output[y * W + x] = sum;
        }
    }
}

void gpu_conv2d_naive(
    const float* d_input,
    float* d_output,
    float* d_filter,
    const int W,
    const int H,
    const int FW,
    const int FH,
    float* h_output
) {
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    CUDA_BENCHMARK(
        gpu_conv2d_naive,
        conv2d_naive<<<grid, block>>>(
            d_input,
            d_output,
            d_filter,
            W,
            H,
            FW,
            FH
        )
    );
    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_output,
        W * H * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    cudaDeviceSynchronize();
}

void gpu_conv2d_tiled(
    const float* d_input,
    float* d_output,
    float* d_filter,
    const int W,
    const int H,
    const int FW,
    const int FH,
    float* h_output
) {
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    size_t sz_tile_shm = (block.x + FW - 1) * (block.y + FH - 1) * sizeof(float);
    size_t sz_filter_shm = FW * FH * sizeof(float);

    CUDA_BENCHMARK(
        gpu_conv2d_tiled,
        conv2d_tiled<<<grid, block, sz_tile_shm + sz_filter_shm>>>(
            d_input,
            d_output,
            d_filter,
            W,
            H,
            FW,
            FH
        )
    );
    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_output,
        W * H * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    cudaDeviceSynchronize();
}

int main(void)
{
    const int W = 1024;
    const int H = 1024;
    const int FW = 7;
    const int FH = 7;
    const size_t sz_input = W * H * sizeof(float);
    const size_t sz_filter = FW * FH * sizeof(float);
    const size_t sz_output = W * H * sizeof(float);

    auto h_input = std::make_unique<float[]>(W * H);
    auto h_filter = std::make_unique<float[]>(FW * FH);
    auto h_output = std::make_unique<float[]>(W * H);
    auto h_output_gpu = std::make_unique<float[]>(W * H);

    init_random_values_f32(h_input.get(), W * H);
    init_random_values_f32(h_filter.get(), FW * FH);

    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_input, sz_input));
    CUDA_CHECK(cudaMalloc((void**)&d_filter, sz_filter));
    CUDA_CHECK(cudaMalloc((void**)&d_output, sz_output));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.get(), sz_input, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.get(), sz_filter, cudaMemcpyHostToDevice));

    HOST_BENCHMARK(
        h_conv2d(
            h_input.get(),
            h_output.get(),
            h_filter.get(),  
            W,
            H,
            FW,
            FH
        ),  cpu_conv2d
    );

    gpu_conv2d_naive(
        d_input,
        d_output,
        d_filter,
        W,
        H,
        FW,
        FH,
        h_output_gpu.get()
    );
    bool correct = compare_result(
        h_output.get(),
        h_output_gpu.get(),
        W * H
    );
    LOG_INFO("GPU Naive Conv2D correctness: %s", correct ? "PASSED" : "FAILED");

    gpu_conv2d_tiled(
        d_input,
        d_output,
        d_filter,
        W,
        H,
        FW,
        FH,
        h_output_gpu.get()
    );
    correct = compare_result(
        h_output.get(),
        h_output_gpu.get(),
        W * H
    );
    LOG_INFO("GPU Tiled Conv2D correctness: %s", correct ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
