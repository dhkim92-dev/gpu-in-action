__global__ void transpose_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int height
);


__global__ void transpose_tiled_bank_conflict(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int height
);

__global__ void transpose_tiled_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int height
);