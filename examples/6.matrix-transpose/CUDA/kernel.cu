#include "kernel.cuh"

#define TILE_SIZE 16

__global__ void transpose_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int height
)
{
    // x = column index
    // y = row index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < width && y < height ) {
        output[x * height + y] = input[y * width + x];
    }
}


__global__ void transpose_tiled_bank_conflict(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int height
)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data to shared memory
    if ( x < width && y < height ) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads(); // Ensure all data is loaded

    // Write transposed data to output matrix
    int tx = blockIdx.y * TILE_SIZE + threadIdx.x;
    int ty = blockIdx.x * TILE_SIZE + threadIdx.y;

    if ( tx < height && ty < width ) {
        // match OpenCL linearization: output[row=ty][col=tx] with row-major width==height
        output[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_tiled_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int height
)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data to shared memory
    if ( x < width && y < height ) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads(); // Ensure all data is loaded

    // Write transposed data to output matrix
    int tx = blockIdx.y * TILE_SIZE + threadIdx.x;
    int ty = blockIdx.x * TILE_SIZE + threadIdx.y;

    if ( tx < height && ty < width ) {
        // match OpenCL linearization: output[row=ty][col=tx] with row-major width==height
        output[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
    }
}