#pragma once
#include <cuda_runtime.h>

__device__ int scan1(
    int val,
    __restrict__ int* cache
); 

__global__ void scan4(
    int4* __restrict__ src,
    int4* __restrict__ dst,
    int* __restrict__ gsum,
    const int n
); 

__global__ void uniform_update(
    int4* __restrict__ output,
    const int* __restrict__ group_sums
);

__global__ void scan_ed( 
    int* __restrict__ src,
    int* __restrict__ dst
); 