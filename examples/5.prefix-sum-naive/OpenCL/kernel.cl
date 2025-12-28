int scan1(int val,__local int* cache) 
{
    int lid = get_local_id(0);
    int lsz = get_local_size(0);

    cache[lid] = 0;
    lid += lsz;
    cache[lid] = val;

    for (int offset = 1; offset < lsz ; offset <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int t = cache[lid] + cache[lid - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        cache[lid] = t;
    }
    return cache[lid];
}

__kernel void scan4(
    __global int4* src,
    __global int4* dst,
    __global int* gsum,
    int n
) 
{
    int id = get_global_id(0);
    __local int cache[128];
    
    int4 data = (int4)(0);
    if (id < n) {
        data = src[id];
        data.y += data.x;
        data.z += data.y;
        data.w += data.z;
    }

    int val = scan1(data.w, cache);
    
    if (id < n) {
        dst[id] = data + (int4)(val - data.w);
    }
    if (id == 0) gsum[0] = 0;
    if (get_local_id(0) == get_local_size(0) - 1) {
        gsum[get_group_id(0) + 1] = val;
    }
}

__kernel void uniform_update(
    __global int4* output,
    __global int* group_sums
) {
    int id = get_global_id(0);
    int gid = get_group_id(0);
    if (gid != 0) {
        int4 val = output[id];
        val += group_sums[gid];
        output[id] = val;
    }
}

__kernel void scan_ed(
    __global int* src,
    __global int* dst,
    __local int* cache
) {
    int id = get_global_id(0);
    int data = src[id];
    dst[id] = scan1(data, cache);
}