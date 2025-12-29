#include "opencl_helper.hpp"
#include "helper.hpp"
#include <cstring>

void host_matrix_transpose(
    const float* input,
    float* output,
    const int width,
    const int height
) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            output[c * height + r] = input[r * width + c];
        }
    }
}

/**
 * GPU Matrix Transpose
 * Transposes a matrix on the GPU.
 * input size and output size be should be padded to multiple of 16 for tiled version.
 * @param queue The OpenCL command queue.
 * @param kernel The OpenCL kernel for matrix transpose.
 * @param d_input The input matrix buffer on the device. 
 * @param d_output The output matrix buffer on the device.
 * @param h_output_gpu The output matrix buffer on the host to read results into.
 * @param width The width of the input matrix.
 * @param height The height of the input matrix.
 */
void gpu_matrix_transpose_naive(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_input,
    cl_mem d_output,
    float* h_output_gpu,
    const int width,
    const int height
) {
    cl_int err = CL_SUCCESS;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_CL_ERROR(err, "Failed to set kernel args for matrix transpose");
    cl_event benchmark_event;
    const size_t global_work_size[2] = {
        static_cast<size_t>(width),
        static_cast<size_t>(height)
    };
    LOG_DEBUG("Enqueue naive matrix transpose: \n\tgws=(%zu, %zu) \n\td_output: %p \n\td_input : %p", 
        global_work_size[0], global_work_size[1],
        d_output, d_input);
    
    CL_BENCHMARK(
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            global_work_size,
            nullptr,
            0,
            nullptr,
            &benchmark_event
        ), 
        gpu_matrix_transpose_naive, 
        benchmark_event);
    CHECK_CL_ERROR(clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * width * height,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    ), "Failed to read back output matrix");

    clFinish(queue);
}

void gpu_matrix_transpose_tiled_bank_conflict(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_input,
    cl_mem d_output,
    float* h_output_gpu,
    const int width,
    const int height
) {
    cl_int err = CL_SUCCESS;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_CL_ERROR(err, "Failed to set kernel args for matrix transpose");
    cl_event benchmark_event;
    const size_t local_work_size[2] = {16, 16};
    const size_t global_work_size[2] = {
        static_cast<size_t>((width + 15) / 16) * 16,
        static_cast<size_t>((height + 15) / 16) * 16
    };
    CL_BENCHMARK(clEnqueueNDRangeKernel(
        queue,
        kernel,
        2,
        nullptr,
        global_work_size,
        local_work_size,
        0,
        nullptr,
        &benchmark_event
    ), gpu_matrix_transpose_tiled_bank_conflict, benchmark_event);
    CHECK_CL_ERROR(clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * width * height,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    ), "Failed to read back output matrix");

    clFinish(queue);
}

void gpu_matrix_transpose_tiled_optimized(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_input,
    cl_mem d_output,
    float* h_output_gpu,
    const int width,
    const int height
) {
    cl_int err = CL_SUCCESS;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_CL_ERROR(err, "Failed to set kernel args for matrix transpose");
    cl_event benchmark_event;
    const size_t local_work_size[2] = {16, 16};
    const size_t global_work_size[2] = {
        static_cast<size_t>((width + 15) / 16) * 16,
        static_cast<size_t>((height + 15) / 16) * 16
    };
    CL_BENCHMARK(clEnqueueNDRangeKernel(
        queue,
        kernel,
        2,
        nullptr,
        global_work_size,
        local_work_size,
        0,
        nullptr,
        &benchmark_event
    ), gpu_matrix_transpose_optimized, benchmark_event);
    CHECK_CL_ERROR(clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * width * height,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    ), "Failed to read back output matrix");

    clFinish(queue);
}


bool compare_result(const float* ref, const float* res, int size) 
{
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(ref[i] - res[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] << ", res=" << res[i] << std::endl;
            return false;
        }
    }
    return true;
}

void print_matrix(const float* matrix, int width, int height, const char *msg) 
{
    printf("%s (%dx%d):\n", msg, height, width);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            printf("%.2f ", matrix[r * width + c]);
        }
        printf("\n");
    }
}

int main(void)
{
    CL_CONTEXT_INIT;

    const int width = 1920;
    const int height = 1080;
    const size_t sz_mem = sizeof(float) * width * height;
    float* h_input = new float[width * height];
    float* h_output_gpu = new float[width * height];
    float* h_output_cpu = new float[width * height];

    gpgpu_detail::rand_seeded = true;
    init_random_values_f32(h_input, width * height);;
    // print_matrix(h_input, width, height, "host_input_matrix");

    cl_mem d_input = clCreateBuffer(context,
        CL_MEM_READ_ONLY,
        sz_mem,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for input matrix");
    cl_mem d_output = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY,
        sz_mem,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for output matrix");

    CHECK_CL_ERROR(clEnqueueWriteBuffer(
        queue,
        d_input,
        CL_TRUE,
        0,
        sz_mem,
        h_input,
        0,
        nullptr,
        nullptr
    ), "Failed to write to input buffer");

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
    CHECK_CL_ERROR(err, "Failed to build program");
    cl_kernel k_transpose = clCreateKernel(
        program,
        "transpose_naive",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for matrix transpose");
    cl_kernel k_transpose_tiled_bank_conflict = clCreateKernel(
        program,
        "transpose_tiled_bank_conflict",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for matrix transpose tiled bank conflict");
    cl_kernel k_transpose_tiled_optimized = clCreateKernel(
        program,
        "transpose_tiled_optimized",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for matrix transpose tiled optimized");

    BENCHMARK_START(cpu_matrix_transpose)
    host_matrix_transpose(
        h_input,
        h_output_cpu,
        width,
        height
    );
    BENCHMARK_END(cpu_matrix_transpose);
    // print_matrix(h_output_cpu, height, width, "host_transposed_matrix");

    gpu_matrix_transpose_naive(
        queue,
        k_transpose,
        d_input,
        d_output,
        h_output_gpu,
        width,
        height
    );
    // print_matrix(h_output_gpu, height, width, "gpu_transposed_matrix_naive");
    compare_result(h_output_cpu, h_output_gpu, width * height);
    memset(h_output_gpu, 0, sz_mem);
    fill_buffer(queue, d_output, 0, sz_mem);
    clFinish(queue);

    gpu_matrix_transpose_tiled_bank_conflict(
        queue,
        k_transpose_tiled_bank_conflict,
        d_input,
        d_output,
        h_output_gpu,
        width,
        height
    );
    // print_matrix(h_output_gpu, height, width, "gpu_transposed_matrix_tiled_bank_conflict");
    compare_result(h_output_cpu, h_output_gpu, width * height);
    memset(h_output_gpu, 0, sz_mem);
    fill_buffer(queue, d_output, 0, sz_mem);
    clFinish(queue);

    gpu_matrix_transpose_tiled_optimized(
        queue,
        k_transpose_tiled_optimized,
        d_input,
        d_output,
        h_output_gpu,
        width,
        height
    );
    // print_matrix(h_output_gpu, height, width, "gpu_transposed_matrix_tiled_optimized");
    compare_result(h_output_cpu, h_output_gpu, width * height);
    fill_buffer(queue, d_output, 0, sz_mem);
    clFinish(queue);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(k_transpose);
    clReleaseKernel(k_transpose_tiled_bank_conflict);
    clReleaseKernel(k_transpose_tiled_optimized);
    clReleaseProgram(program);
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    CL_CONTEXT_CLEANUP;
    return 0;
}