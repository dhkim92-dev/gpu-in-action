#include "opencl_helper.hpp"
#include "helper.hpp"

bool compare_result(
    const float* ref,
    const float* res,
    const int size,
    const float epsilon = 1e-4f
) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(ref[i] - res[i]) > epsilon) {
            LOG_ERROR("Mismatch at index %d: ref=%f, res=%f", i, ref[i], res[i]);
            return false;
        }
    }
    return true;
}

// output size same with input size (no padding, no stride)
void h_conv2d(
    const float *input,
    float *output,
    const float *filter,
    const int W,
    const int H,
    const int FW,
    const int FH
) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < FH; ++ky) {
                for (int kx = 0; kx < FW; ++kx) {
                    int in_x = x + kx - FW / 2;
                    int in_y = y + ky - FH / 2;
                    if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                        sum += input[in_y * W + in_x] * filter[ky * FW + kx];
                    }
                }
            }
            output[y * W + x] = sum;
        }
    }
}

void gpu_conv2d_naive(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_input,
    cl_mem d_output,
    cl_mem d_filter,
    const int W,
    const int H,
    const int FW,
    const int FH,
    float * h_output_gpu
) {
    // kernel signature: (input, output, filter, W, H, FW, FH)
    const size_t sz_global[2] = {
        static_cast<size_t>((W + 15) / 16) * 16,
        static_cast<size_t>((H + 15) / 16) * 16
    };
    const size_t sz_local[2] = {16, 16};
    cl_event benchmark_event;
    cl_set_kernel_args(kernel, d_input, d_output, d_filter, W, H, FW, FH);
    CL_BENCHMARK(
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            sz_global,
            sz_local,
            0,
            nullptr,
            &benchmark_event
        ),
        device_conv2d_naive,
        benchmark_event
    );
    CHECK_CL_ERROR(clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * W * H,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    ), "Failed to read back output image");
    clFinish(queue);
}

void gpu_conv2d_tiled(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_input,
    cl_mem d_output,
    cl_mem d_filter,
    const int W,
    const int H,
    const int FW,
    const int FH,
    float * h_output_gpu
) {
    const size_t sz_global[2] = {
        static_cast<size_t>((W + 15) / 16) * 16,
        static_cast<size_t>((H + 15) / 16) * 16
    };
    const size_t sz_local[2] = {16, 16};
    const size_t sz_shm = (16 + FW - 1) * (16 + FH - 1) * sizeof(float); // local mem size
    cl_set_kernel_args(kernel, d_input, d_output, d_filter, SZ_LMEM(sz_shm), SZ_LMEM(FW * FH * sizeof(float)), W, H, FW, FH);
    cl_event benchmark_event;
    CL_BENCHMARK(
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            sz_global,
            sz_local,
            0,
            nullptr,
            &benchmark_event
        ),
        device_conv2d_tiled,
        benchmark_event
    );
    CHECK_CL_ERROR(clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * W * H,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    ), "Failed to read back output image");
    clFinish(queue);
}

int main(void)
{
    CL_CONTEXT_INIT

    // Debug: print OpenCL platform/device info
    {
        size_t sz = 0;
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &sz);
        std::vector<char> pname(sz);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, pname.data(), nullptr);
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &sz);
        std::vector<char> dname(sz);
        clGetDeviceInfo(device, CL_DEVICE_NAME, sz, dname.data(), nullptr);
        cl_device_type dtype = 0;
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
        std::printf("[DEBUG] OpenCL platform=%s device=%s type=0x%llx\n", pname.data(), dname.data(), (unsigned long long)dtype);
    }

    const int W = 1024;
    const int H = 1024;
    const int FW = 7;
    const int FH = 7;
    const size_t sz_input = sizeof(float) * W * H;
    const size_t sz_filter= sizeof(float) * FW * FH;
    const size_t sz_output = sizeof(float) * W * H;
    auto h_input = std::make_unique<float[]>(W * H);
    auto h_output_cpu = std::make_unique<float[]>(W * H);
    auto h_output_gpu = std::make_unique<float[]>(W * H);
    auto h_filter = std::make_unique<float[]>(FW * FH);

    gpgpu_detail::rand_seeded = true;
    init_random_values_f32(h_input.get(), W * H);
    init_random_values_f32(h_filter.get(), FW * FH);
    cl_mem d_input = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sz_input,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for input image");
    cl_mem d_filter = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sz_filter,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for filter kernel");
    cl_mem d_output = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sz_output,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for output image");

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

    cl_kernel k_conv2d_naive = clCreateKernel(
        program,
        "conv2d_naive",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for conv2d_naive");
    cl_kernel k_conv2d_tiled = clCreateKernel(
        program,
        "conv2d_tiled",
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create kernel for conv2d_tiled");

    // Debug: print kernel function names and kernel handles to ensure distinct kernels
    {
        char namebuf[256];
        size_t ret = 0;
        clGetKernelInfo(k_conv2d_naive, CL_KERNEL_FUNCTION_NAME, sizeof(namebuf), namebuf, &ret);
        std::printf("[DEBUG] k_conv2d_naive handle=%p name=%s\n", (void*)k_conv2d_naive, namebuf);
        clGetKernelInfo(k_conv2d_tiled, CL_KERNEL_FUNCTION_NAME, sizeof(namebuf), namebuf, &ret);
        std::printf("[DEBUG] k_conv2d_tiled handle=%p name=%s\n", (void*)k_conv2d_tiled, namebuf);
    }

    clEnqueueWriteBuffer(
        queue,
        d_input,
        CL_TRUE,
        0,
        sz_input,
        h_input.get(),
        0,
        nullptr,
        nullptr
    );

    clEnqueueWriteBuffer(
        queue,
        d_filter,
        CL_TRUE,
        0,
        sz_filter,
        h_filter.get(),
        0,
        nullptr,
        nullptr
    );

    HOST_BENCHMARK(
        h_conv2d(
            h_input.get(),
            h_output_cpu.get(),
            h_filter.get(),  
            W,
            H,
            FW,
            FH
        ),  cpu_conv2d
    );

    gpu_conv2d_naive(
        queue,
        k_conv2d_naive,
        d_input,
        d_output,
        d_filter,
        W,
        H,
        FW,
        FH,
        h_output_gpu.get()
    );
    bool res = compare_result(h_output_cpu.get(), h_output_gpu.get(), W * H);
    LOG_INFO("Conv2D Naive Result: %s", res ? "PASS" : "FAIL");

    gpu_conv2d_tiled(
        queue,
        k_conv2d_tiled,
        d_input,
        d_output,
        d_filter,
        W,
        H,
        FW,
        FH,
        h_output_gpu.get()
    );
    res = compare_result(h_output_cpu.get(), h_output_gpu.get(), W * H);
    LOG_INFO("Conv2D Tiled Result: %s", res ? "PASS" : "FAIL");

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_output);
    clReleaseKernel(k_conv2d_naive);
    clReleaseKernel(k_conv2d_tiled);
    clReleaseProgram(program);

    CL_CONTEXT_CLEANUP
    return 0;
}