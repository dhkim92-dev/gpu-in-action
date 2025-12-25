/**
* Kernel to perform reduction sum on an array of floats.
* Each work-group computes a partial sum which can be further reduced on the host.
* ex workgroup size = 4
* stride = 2
* Input:  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
* step 1. Calculate local sums within each work-group
* WG0 = local_sums[0] = 1, local_sums[1] = 2, local_sums[2] = 3, local_sums[3] = 4
*        local_sums[0] += local_sums[2] -> 4 local_sum[1] += local_sums[3] -> 6
*        local_sums = 4 6 x x
*        local_sums[0] = 10
* WG1 = local_sums[0] = 5, local_sums[1] = 6, local_sums[2] = 7, local_sums[3] = 8
*        local_sums[0] += local_sums[2] -> 12 local_sum[1] += local_sums[3] -> 14
*        local_sums = 12 14 x x
*        local_sums[0] = 26
* WG2 = local_sums[0] = 9, local_sums[1] = 10, local_sums[2] = 11, local_sums[3] = 12
*        local_sums[0] += local_sums[2] -> 20 local_sum[1] += local_sums[3] -> 22
*       local_sums = 20 22 x x
*        local_sums[0] = 42
* WG3 = local_sums[0] = 13, local_sums[1] = 14, local_sums[2] = 15, local_sums[3] = 16
*        local_sums[0] += local_sums[2] -> 28 local_sum[1] += local_sums[3] -> 30  
*        local_sums = 28 30 x x
*        local_sums[0] = 58
* step 2. Output from each work-group: 10 26 42 58
* Final sum on host: 10 + 26 + 42 + 58 = 136
* outputs += local_sums[0]
*/
__kernel void reduce_sum(
    __global const  float* inputs, 
    __global float *output,
    __local float* local_sums,
    const int sz_workitems
) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Load data into local memory
    local_sums[local_id] = (global_id < sz_workitems) ? inputs[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in local memory
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_sums[local_id] += local_sums[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write the result of this work-group to global memory
    if (local_id == 0) {
        output[get_group_id(0)] = local_sums[0];
    }
}