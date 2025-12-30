#include <cuda_runtime.h>

__global__ void conv2d_naive(
    const float* input,
    const float* output,
    float* filter, 
    int W,
    int H,
    int KW,
    int KH
);

__global__ void conv2d_tiled(
    const float* input,
    float* output,
    float* filter, 
    int W,
    int H,
    int KW,
    int KH
);