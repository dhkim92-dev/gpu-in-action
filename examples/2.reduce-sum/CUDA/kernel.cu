// NOTE:
// - This kernel produces one partial sum per block: output[blockIdx.x].
// - To get a single value, launch this kernel repeatedly (ping-pong buffers)
//   until the number of blocks becomes 1, or do a final CPU sum.
// - Declared extern "C" so cuModuleGetFunction("reduce_sum") works.

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
	extern __shared__ float sdata[];

	const int tid = (int)threadIdx.x;
	const int gid = (int)blockIdx.x * (int)blockDim.x + tid;

	float v = 0.0f;
	if (gid < n) {
		v = input[gid];
	}
	sdata[tid] = v;
	__syncthreads();

	// Reduction in shared memory
	for (int stride = ((int)blockDim.x) / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sdata[tid] += sdata[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[(int)blockIdx.x] = sdata[0];
	}
}
#endif

