/**
 * Dot product in two phases:
 * 1) dot_product_reduce_first: compute partial sums of a[i]*b[i] per work-group.
 * 2) reduce_sum: repeatedly reduce partial sums until a single value remains.
 */

__kernel void dot_product_reduce_first(
    __global const float* a,
    __global const float* b,
    __global float* output,
    __local float* local_sums,
    const int n
) {
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int gid = get_group_id(0) * lsz * 2 + lid;

    float sum = 0.0f;

    if (gid < n) {
        sum = a[gid] * b[gid];
    }
    if (gid + lsz < n) {
        sum += a[gid + lsz] * b[gid + lsz];
    }

    local_sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = local_sums[0];
    }
}

/**
 * Same reduction kernel as example 2 (reduce-sum).
 * Each work-group has N work-items and reduces 2*N inputs into one output.
 */
__kernel void reduce_sum(
    __global const float* inputs,
    __global float* output,
    __local float* local_sums,
    const int sz_workitems
) {
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int gid = get_group_id(0) * lsz * 2 + lid;

    float sum = 0.0f;
    if (gid < sz_workitems)
        sum = inputs[gid];
    if (gid + lsz < sz_workitems)
        sum += inputs[gid + lsz];

    local_sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = local_sums[0];
    }
}
