#include "helper.hpp"
#include "opencl_helper.hpp"

static float host_dot_product(const float* a, const float* b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main(void)
{
    CL_CONTEXT_INIT

    const size_t n = 16384;
    const size_t sz_mem_vec = sizeof(float) * n;
    const size_t local_size = 256; // must be <= device max work-group size
    const size_t sz_mem_local = sizeof(float) * local_size;

    float* h_a = new float[n];
    float* h_b = new float[n];

    gpgpu_detail::rand_seeded = true;
    init_random_values_f32(h_a, static_cast<int>(n));
    init_random_values_f32(h_b, static_cast<int>(n));

    cl_mem d_a = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sz_mem_vec,
        h_a,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for vector A");

    cl_mem d_b = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sz_mem_vec,
        h_b,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for vector B");

    cl_mem d_ping = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sz_mem_vec,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create ping buffer");

    cl_mem d_pong = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sz_mem_vec,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create pong buffer");

    std::string kernel_code = read_kernel_file("kernel.cl");
    const char* code_cstr = kernel_code.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &code_cstr, nullptr, &err);
    CHECK_CL_ERROR(err, "Failed to create program with source");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CHECK_CL_ERROR(err, "Failed to build program");

    cl_kernel k_dot_first = clCreateKernel(program, "dot_product_reduce_first", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel dot_product_reduce_first");

    cl_kernel k_reduce = clCreateKernel(program, "reduce_sum", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel reduce_sum");

    LOG_INFO("Dot Product (OpenCL) launching... n=%zu local_size=%zu", n, local_size);

    // First pass: a[i]*b[i] + reduction within each work-group.
    size_t current_n = n;
    size_t groups = (current_n + (local_size * 2) - 1) / (local_size * 2);
    size_t global_work_size = groups * local_size;
    int n_arg = static_cast<int>(current_n);

    BENCHMARK_START(gpu_dot_product)

    err  = clSetKernelArg(k_dot_first, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(k_dot_first, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(k_dot_first, 2, sizeof(cl_mem), &d_ping);
    err |= clSetKernelArg(k_dot_first, 3, sz_mem_local, nullptr);
    err |= clSetKernelArg(k_dot_first, 4, sizeof(int), &n_arg);
    CHECK_CL_ERROR(err, "Failed to set kernel args for dot_product_reduce_first");

    err = clEnqueueNDRangeKernel(
        queue,
        k_dot_first,
        1,
        nullptr,
        &global_work_size,
        &local_size,
        0,
        nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "Failed to enqueue dot_product_reduce_first");

    current_n = groups;

    // Next passes: reduce partial sums until one value remains.
    while (current_n > 1) {
        groups = (current_n + (local_size * 2) - 1) / (local_size * 2);
        global_work_size = groups * local_size;
        n_arg = static_cast<int>(current_n);

        err  = clSetKernelArg(k_reduce, 0, sizeof(cl_mem), &d_ping);
        err |= clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &d_pong);
        err |= clSetKernelArg(k_reduce, 2, sz_mem_local, nullptr);
        err |= clSetKernelArg(k_reduce, 3, sizeof(int), &n_arg);
        CHECK_CL_ERROR(err, "Failed to set kernel args for reduce_sum");

        err = clEnqueueNDRangeKernel(
            queue,
            k_reduce,
            1,
            nullptr,
            &global_work_size,
            &local_size,
            0,
            nullptr,
            nullptr
        );
        CHECK_CL_ERROR(err, "Failed to enqueue reduce_sum");

        std::swap(d_ping, d_pong);
        current_n = groups;
    }

    BENCHMARK_END(gpu_dot_product)

    clFinish(queue);

    float h_dot_gpu = 0.0f;
    clEnqueueReadBuffer(queue, d_ping, CL_TRUE, 0, sizeof(float), &h_dot_gpu, 0, nullptr, nullptr);

    float h_dot_cpu = 0.0f;
    BENCHMARK_START(cpu_dot_product)
    h_dot_cpu = host_dot_product(h_a, h_b, static_cast<int>(n));
    BENCHMARK_END(cpu_dot_product)

    LOG_INFO("Dot Product CPU Result: %f", h_dot_cpu);
    LOG_INFO("Dot Product CL  Result: %f", h_dot_gpu);

    // Cleanup
    clReleaseKernel(k_dot_first);
    clReleaseKernel(k_reduce);
    clReleaseProgram(program);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_ping);
    clReleaseMemObject(d_pong);

    delete[] h_a;
    delete[] h_b;

    CL_CONTEXT_CLEANUP
    return 0;
}