#include "opencl_helper.hpp"
#include "helper.hpp"

#include <algorithm>
#include <cmath>


void h_vector_normalization(const float* input, float* output, int size)
{
    float norm = 0.0f;
    for (int i = 0; i < size; ++i) {
        norm += input[i] * input[i];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] / norm;
    }
}

/**
 * GPU Square
 * Squares each element of the input buffer and writes to the output buffer.
 * @param queue The OpenCL command queue.
 * @param kernel The OpenCL kernel for squaring.
 * @param d_input The input buffer on the device.
 * @param d_output The output buffer on the device.
 * @param n The number of elements to process.
 */
void gpu_square(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem& d_input,
    cl_mem& d_output,
    const int n
) {
    size_t sz_local = 256;
    size_t sz_global = ((static_cast<size_t>(n) + sz_local - 1) / sz_local) * sz_local;
    cl_int err = CL_SUCCESS;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);  // input vector
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output); // output vector
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    CHECK_CL_ERROR(err, "Failed to set kernel args for square");
    CHECK_CL_ERROR(clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        nullptr,
        &sz_global,
        &sz_local,
        0,
        nullptr,
        nullptr
    ),"Failed to enqueue square kernel");
}
/**
 * GPU Reduce Sum
 * Performs reduction sum on the input buffer and writes the result to the output buffer.
 * @param queue The OpenCL command queue.
 * @param kernel The OpenCL kernel for reduction sum.
 * @param d_input The input buffer on the device.
 * @param d_output The output buffer on the device.
 * @param n The number of elements to reduce.
 * @param local_size The local work-group size.
 * @param sz_mem_local The size of local memory to allocate for the kernel.
 */
void gpu_reduce_sum(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem& d_input,
    cl_mem& d_output,
    int n,
    size_t local_size,
    size_t sz_mem_local
) {
    // each of kernel thread processes two elements
    cl_int err = CL_SUCCESS;

    while ( n > 1 ) {
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
        err |= clSetKernelArg(kernel, 2, sz_mem_local, nullptr);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &n);
        CHECK_CL_ERROR(err, "Failed to set kernel args for reduce_sum");
        const size_t groups = (n + (local_size * 2) - 1) / (local_size * 2);
        const size_t global_work_size = groups * local_size;
        err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            nullptr,
            &global_work_size,
            &local_size,
            0,
            nullptr,
            nullptr
        );
        CHECK_CL_ERROR(err, "Failed to enqueue reduce_sum");
        std::swap(d_input, d_output);
        n = static_cast<int>(groups);
    }
}

void gpu_vec_norm(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_input,
    cl_mem d_sum,
    const int n
) {
    // when we start this function, first element of d_output contains the result of reduction sum 
    // so, use it to devide each element of d_input
    cl_int err = CL_SUCCESS;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);  // input vector
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_input); // normalized output vector
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_sum); // normalized output vector
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    CHECK_CL_ERROR(err, "Failed to set kernel args for vector_normalization");

    const size_t global_work_size = ((n + 255) / 256) * 256;
    const size_t local_work_size = 256;
    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        nullptr,
        &global_work_size,
        &local_work_size,
        0,
        nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "Failed to enqueue vector_normalization");
}


int main(void)
{
    CL_CONTEXT_INIT

    const size_t n = 1 << 20; // 1M elements, memory usage ~4MB
    const size_t sz_mem_vec = sizeof(float) * n;
    const size_t lsz = 256; // must be <= device max work-group size
    const size_t sz_mem_local = sizeof(float) * lsz;
    float* h_input = new float[n];
    gpgpu_detail::rand_seeded = false;
    init_random_values_f32(h_input, static_cast<int>(n));

    // d_input : input vector
    cl_mem d_input = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sz_mem_vec,
        h_input,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for input vector");

    // d_ping : output buffer for partial sums (and final sumsq at index 0)
    cl_mem d_ping = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sz_mem_vec,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for d_ping vector");
    
    // d_pong : swap buffer for reduction sum
    cl_mem d_pong = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sz_mem_vec,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for d_pong vector");
    // d_sum : buffer to hold final sumsq result
    cl_mem d_sum = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float),
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for d_sum");

    std::string kernel_code = read_kernel_file("kernel.cl");
    const char* code_cstr = kernel_code.c_str();
    cl_program program = clCreateProgramWithSource(
        context,
        1,
        &code_cstr,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create program with source");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::fprintf(stderr, "OpenCL Build Error:\n%s\n", log.data());
        CHECK_CL_ERROR(err, "Failed to build program");
    }

    cl_kernel k_square = clCreateKernel(
        program,
        "square",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for square");

    cl_kernel k_reduce = clCreateKernel(
        program,
        "reduce",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for reduce");

    cl_kernel k_normalize = clCreateKernel(
        program,
        "normalize",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for vector_normalization");

    BENCHMARK_START(gpu_vector_normalization)

    gpu_square(
        queue,
        k_square,
        d_input,
        d_ping,
        static_cast<int>(n)
    );

    gpu_reduce_sum(
        queue,
        k_reduce,
        d_ping,
        d_pong,
        static_cast<int>(n),
        lsz,
        sz_mem_local
    );

    CHECK_CL_ERROR(clEnqueueCopyBuffer(
        queue,
        d_ping,
        d_sum,
        0,
        0,
        sizeof(float),
        0,
        nullptr,
        nullptr
    ), "Failed to copy final sum to d_sum buffer");

    gpu_vec_norm(
        queue,
        k_normalize,
        d_input,
        d_sum,
        static_cast<int>(n)
    );
    BENCHMARK_END(gpu_vector_normalization)
    float* h_output_gpu = new float[n];
    clEnqueueReadBuffer(
        queue,
        d_input,
        CL_TRUE,
        0,
        sz_mem_vec,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    );
    clFinish(queue);

    float* h_output_cpu = new float[n];
    BENCHMARK_START(cpu_vector_normalization)
    h_vector_normalization(h_input, h_output_cpu, static_cast<int>(n));
    BENCHMARK_END(cpu_vector_normalization)

    // Verify results
    // print first 10 elements
    for (size_t i = 0; i < 10; ++i) {
        LOG_INFO("Element %zu: GPU=%f, CPU=%f", i, h_output_gpu[i], h_output_cpu[i]);
    }
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5f) {
            LOG_ERROR("Mismatch at index %zu: GPU=%f, CPU=%f", i, h_output_gpu[i], h_output_cpu[i]);
        }
    }

    clReleaseKernel(k_reduce);
    clReleaseKernel(k_normalize);
    clReleaseProgram(program);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_ping);
    clReleaseMemObject(d_pong);

    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;

    CL_CONTEXT_CLEANUP
    return 0;
}