#include "kernel.cuh"

__device__ int scan1(
    int val,
    int* __restrict__ cache
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
    return cache[tid];
}

__global__ void scan4(
    int4* __restrict__ src,
    int4* __restrict__ dst,
    int* __restrict__ gsum,
    const int n
) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int cache[128];
    
    int4 data = make_int4(0, 0, 0, 0);
    if (id < n) {
        data = src[id];
        data.y += data.x;
        data.z += data.y;
        data.w += data.z;
    }

    int val = scan1(data.w, cache);
    
    if (id < n) {
        int offset = val - data.w;
        int4 result;
        result.x = data.x + offset;
        result.y = data.y + offset;
        result.z = data.z + offset;
        result.w = data.w + offset;
        dst[id] = result;
    }
    
    if (id == 0) gsum[0] = 0;
    
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
        int sum = group_sums[gid];
        int4 val = output[id];
        val.x += sum;
        val.y += sum;
        val.z += sum;
        val.w += sum;
        output[id] = val;
    }
}

__global__ void scan_ed(
    int* __restrict__ src,
    int* __restrict__ dst
) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int cache[];
    int val = src[id];
    dst[id] = scan1(val, cache);
}


