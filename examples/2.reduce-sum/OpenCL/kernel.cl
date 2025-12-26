/**
* Each of work group consist N threads,
* handle 2*N global items 
* local_sums[local_id] = global_items[global_id] + global_items[global_id + N];
* For example, if N = 32
* local_sums is 32 slots array
* workgroup 0 handle global_items[0...63]
* workgroup 1 handle global_itmes[64...127] 
* and so on
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

    // local reduction
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
