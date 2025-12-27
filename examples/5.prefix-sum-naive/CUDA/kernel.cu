#include "kernel.cuh"

__device__ int scan1(
    int val,
    __restrict__ int* cache
) {
    int tid = threadIdx.x;
    cache[tid] = 0;
    tid += blockDim.x;
    cache[tid] = val;

    for ( int offset = 1 ; offset < blockDim.x ; offset <<= 1) {
        __syncthreads();
        int t = cache[tid] + cache[tid - offset];
        __syncthreads();
        cache[tid] = t;
    }
}

__global__ void scan4(
    int4* __restrict__ src,
    int4* __restrict__ dst,
    int* __restrict__ gsum,
    const int n
) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    __shared__ int cache[128];
    int4 data = src[id];
    data.y += data.x;
    data.z += data.y;
    data.w += data.z;

    int val = scan1(data.w, cache);
    dst[id] = data + (int4)(val - data.w);

    if (threadIdx.x == blockDim.x - 1) {
        gsum[blockIdx.x + 1] = val;
    }
}

__global__ void uniform_update(
    int4* __restrict__ output,
    const int* __restrict__ group_sums
) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int gid = blockIdx.x;
    if (gid != 0) {
        int4 val = output[id];
        val += group_sums[gid];
        output[id] = val;
    }
}

__global__ void scan_ed(
    int* __restrict__ src,
    int* __restrict__ dst
) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int cache[];
    int val = src[id];
    dst[id] = scan1(val, cache);
}


