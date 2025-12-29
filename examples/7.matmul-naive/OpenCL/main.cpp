#include "helper.hpp"
#include "opencl_helper.hpp"

void host_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            C[row * N + col] = 0.0f;
            for (int k = 0; k < K; ++k) {
                C[row * N + col] += A[row * K + k] * B[k * N + col];
            }
        }
    }
}

bool compare_matrices(
    const float* mat1,
    const float* mat2,
    int M,
    int N,
    float epsilon = 1e-5f
) {
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(mat1[i] - mat2[i]) > epsilon) {
            LOG_ERROR("Mismatch at index %d: mat1=%f, mat2=%f", i, mat1[i], mat2[i]);
            return false;
        }
    }
    return true;
}

void device_matmul_naive(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_A,
    cl_mem d_B,
    cl_mem d_C,
    uint32_t M,
    uint32_t K,
    uint32_t N
) {
    size_t global_work_size[2] = {
        static_cast<size_t>(N),
        static_cast<size_t>(M)
    };
    cl_event benchmark_event;
    cl_set_kernel_args(kernel, d_A, d_B, d_C, M, K, N);
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
        matmul_naive,
        benchmark_event
    );
}

void device_matmul_tiled_basic(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem d_A,
    cl_mem d_B,
    cl_mem d_C,
    int M,
    int K,
    int N,
    int TILE_SIZE) {
    size_t global_work_size[2] = {
        static_cast<size_t>((N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE),
        static_cast<size_t>((M + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE)
    };
    size_t local_work_size[2] = {
        static_cast<size_t>(TILE_SIZE),
        static_cast<size_t>(TILE_SIZE)
    };
    cl_event benchmark_event;
    cl_set_kernel_args(kernel, d_A, d_B, d_C, M, K, N);
    CL_BENCHMARK(
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            global_work_size,
            local_work_size,
            0,
            nullptr,
            &benchmark_event
        ),
        matmul_tiled_basic,
        benchmark_event
    );
}

int main(void)
{
    CL_CONTEXT_INIT
    uint32_t M = 1024;
    uint32_t K = 768;
    uint32_t N = 512;

    auto h_A = std::make_unique<float[]>(M * K);
    auto h_B = std::make_unique<float[]>(K * N);
    auto h_output = std::make_unique<float[]>(M * N);
    auto h_output_gpu = std::make_unique<float[]>(M * N);

    init_random_values_f32(h_A.get(), M * K);
    init_random_values_f32(h_B.get(), K * N);
    std::memset(h_output.get(), 0, sizeof(float) * M * N);
    std::memset(h_output_gpu.get(), 0, sizeof(float) * M * N);

    cl_mem d_A = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * M * K,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "clCreateBuffer d_A");
    cl_mem d_B = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * K * N,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "clCreateBuffer d_B");
    cl_mem d_output = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * M * N,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "clCreateBuffer d_output");
    // Copy data to device
    clEnqueueWriteBuffer(
        queue,
        d_A,
        CL_TRUE,
        0,
        sizeof(float) * M * K,
        h_A.get(),
        0,
        nullptr,
        nullptr
    );
    // Copy B to device
    clEnqueueWriteBuffer(
        queue,
        d_B,
        CL_TRUE,
        0,
        sizeof(float) * K * N,
        h_B.get(),
        0,
        nullptr,
        nullptr
    );

    cl_program program = nullptr;
    cl_kernel k_matmul_naive = nullptr;
    cl_kernel k_matmul_tiled_basic = nullptr;

    std::string kernel_source = read_kernel_file("kernel.cl");
    const char* source_str = kernel_source.c_str();
    size_t source_size = kernel_source.length();
    program = clCreateProgramWithSource(
        context,
        1,
        &source_str,
        &source_size,
        &err
    );
    CHECK_CL_ERROR(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    check_program_build_result(err, program, device);
    k_matmul_naive = clCreateKernel(program, "matmul_naive", &err);
    CHECK_CL_ERROR(err, "clCreateKernel matmul_naive");
    k_matmul_tiled_basic = clCreateKernel(program, "matmul_tiled_basic", &err);
    CHECK_CL_ERROR(err, "clCreateKernel matmul_tiled_basic");


    HOST_BENCHMARK(host_matmul(h_A.get(), h_B.get(), h_output.get(), M, K, N), host_matmul);

    // Naive MatMul
    device_matmul_naive( 
        queue,
        k_matmul_naive,
        d_A,
        d_B,
        d_output,
        M,
        K,
        N
    );
    // Copy result back to host
    clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * M * N,
        h_output_gpu.get(),
        0,
        nullptr,
        nullptr
    );
    // Verify correctness
    auto res = compare_matrices(
        h_output.get(),
        h_output_gpu.get(),
        M,
        N
    );

    LOG_INFO("MatMul Naive result: %s", res ? "PASS" : "FAIL");

    fill_buffer(queue, d_output, 0, sizeof(float) * M * N);

    device_matmul_tiled_basic( 
        queue,
        k_matmul_tiled_basic,
        d_A,
        d_B,
        d_output,
        M,
        K,
        N,
        16  
    );
    // Copy result back to host
    clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(float) * M * N,
        h_output_gpu.get(),
        0,
        nullptr,
        nullptr
    );
    // Verify correctness
    res = compare_matrices(
        h_output.get(),
        h_output_gpu.get(),
        M,
        N
    );
    LOG_INFO("MatMul Tiled Basic result: %s", res ? "PASS" : "FAIL");


    CL_CONTEXT_CLEANUP
    return 0;
}