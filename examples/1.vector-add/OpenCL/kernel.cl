__kernel void vector_add(__global const float *a, __global const float *b, __global float *c, const int n) {
    int id = get_global_id(0); // Get the unique ID of the current work item

    if (id < n) {
        c[id] = a[id] + b[id]; // Perform element-wise addition
    }
}
