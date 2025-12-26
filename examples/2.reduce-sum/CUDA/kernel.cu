
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
	extern __shared__ float sdata[];

	const int tid = (int)threadIdx.x;
	const int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	float sum = 0.0f;

	if (gid < n) {
		sum = input[gid];
	}
	if (gid + blockDim.x < n) {
		sum += input[gid + blockDim.x];
	}
	sdata[tid] = sum;
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

