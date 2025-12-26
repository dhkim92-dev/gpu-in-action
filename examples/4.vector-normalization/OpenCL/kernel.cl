__kernel void square(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int gid = get_global_id(0);
    if ( gid < n ) {
        output[gid] = input[gid] * input[gid];
    }
}

/**
* Each workgroup handles two elements per thread
* and performs a local reduction using local memory.
 */
__kernel void reduce(
    __global const float* input,
    __global float* output,
    __local float* local_mem,
    const int n
) {
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int gid = get_group_id(0) * get_local_size(0) * 2 + lid;

    float sum = 0.0f;

    if (gid < n) {
        sum += input[gid];
    }
    if (gid + lsz < n) {
        sum += input[gid + lsz];
    }

    local_mem[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = lsz / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            local_mem[lid] += local_mem[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        output[get_group_id(0)] = local_mem[0];
    }
}

__kernel void normalize(
    __global const float* input,
    __global float* output,
    __global const float* square_sum,
    const int n
) {
    const int gid = get_global_id(0);
    const float denom = sqrt(square_sum[0]);
    if ( gid < n ) {
        output[gid] = input[gid] / denom;
    }
}
