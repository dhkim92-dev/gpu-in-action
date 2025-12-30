#include "kernel.cuh"

__global__ void conv2d_naive(
    const float* input,
    float* output,
    const float* filter, 
    int W,
    int H,
    int KW,
    int KH
) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (  int fr = 0; fr < KH; fr++) {
        for (int fc = 0; fc < KW; fc++) {
            int in_r = r + fr - KH / 2;
            int in_c = c + fc - KW / 2;
            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                sum += input[in_r * W + in_c] * filter[fr * KW + fc];
            }
        }
    }
    if (r < H && c < W) {
        output[r * W + c] = sum;
    }
}

__global__ void conv2d_tiled(
    const float* input,
    float* output,
    const float* filter, 
    int W,
    int H,
    int KW,
    int KH
) {
    extern __shared__ float s_shm[];
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int r = blockIdx.y * blockDim.y + tr;
    int c = blockIdx.x * blockDim.x + tc;
    int tile_rows = blockDim.y + KH - 1;
    int tile_cols = blockDim.x + KW - 1;
    float* tile = s_shm;
    float* shm_filter = s_shm + tile_rows * tile_cols;

    // Load input to shared memory tile
    for (int i = tr; i < blockDim.y + KH - 1; i += blockDim.y) {
        for (int j = tc; j < blockDim.x + KW - 1; j += blockDim.x) {
            int in_r = blockIdx.y * blockDim.y + i - KH / 2;
            int in_c = blockIdx.x * blockDim.x + j - KW / 2;
            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                tile[i * (blockDim.x + KW - 1) + j] = input[in_r * W + in_c];
            } else {
                tile[i * (blockDim.x + KW - 1) + j] = 0.0f;
            }
        }
    }
    // Load Filter to shared memory
    for (int i = tr; i < KH; i += blockDim.y) {
        for (int j = tc; j < KW; j += blockDim.x) {
            shm_filter[i * KW + j] = filter[i * KW + j];
        }
    }
    __syncthreads();
    // Compute convolution
    float sum = 0.0f;
    for (int fr = 0; fr < KH; fr++) {
        for (int fc = 0; fc < KW; fc++) {
            sum += tile[(tr + fr) * (blockDim.x + KW - 1) + (tc + fc)] * shm_filter[fr * KW + fc];
        }
    }

    if (r < H && c < W) {
        output[r * W + c] = sum;
    }
}

