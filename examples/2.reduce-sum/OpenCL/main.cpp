#include "helper.hpp"
#include "opencl_helper.hpp"

static size_t round_up(size_t value, size_t multiple)
{
    if (multiple == 0) return value;
    const size_t remainder = value % multiple;
    return remainder == 0 ? value : (value + multiple - remainder);
}

float host_reduce_sum(const float* data, int size) 
{
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) 
    {
        sum += data[i];
    }
    return sum;
}

int main(void) 
{
    CL_CONTEXT_INIT

    float * h_input;
    float h_output;

    const size_t input_size = 1024;
    const size_t sz_mem_input = sizeof(float) * input_size;
    const size_t sz_wg = 32; // local work-group size (must be <= device limit)
    const size_t sz_mem_local = sizeof(float) * sz_wg;

    h_input = new float[input_size];
    for (size_t i = 0; i < input_size; i++) {
        h_input[i] = static_cast <float> (i + 1); // 1.0, 2.0, ..., N
    }
    init_random_values_f32(h_input, input_size);

    cl_mem d_input = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        sz_mem_input, 
        h_input, 
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for input data");
    cl_mem d_output = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE,
        sz_mem_input,
        nullptr, 
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for output data");
    std::string kernel_code = read_kernel_file("kernel.cl");
    const char* code_cstr = kernel_code.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &code_cstr, nullptr, &err);
    CHECK_CL_ERROR(err, "Failed to create program with source");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CHECK_CL_ERROR(err, "Failed to build program");
    cl_kernel kernel = clCreateKernel(program, "reduce_sum", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel");

    LOG_INFO("Reduction Sum CL Kernel Launching... input_size=%zu local_size=%zu", input_size, sz_wg);


    size_t call_count = 0;
    size_t current_n = input_size;
    while (current_n > 1) {
        const size_t groups = (current_n + sz_wg - 1) / sz_wg; // output elements this pass
        const size_t global_work_size = groups * sz_wg;        // must be multiple of local size
        const int n_arg = static_cast<int>(current_n);

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
        err |= clSetKernelArg(kernel, 2, sz_mem_local, nullptr);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &n_arg);
        CHECK_CL_ERROR(err, "Failed to set kernel arguments");

        std::cout << "Reduction step " << call_count++
                  << ": N=" << current_n
                  << " global_work_size=" << global_work_size
                  << " local_work_size=" << sz_wg
                  << " groups=" << groups
                  << std::endl;

        err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            nullptr,
            &global_work_size,
            &sz_wg,
            0,
            nullptr,
            nullptr
        );
        CHECK_CL_ERROR(err, "Failed to enqueue kernel");

        // Next pass reduces the partial sums we just wrote
        std::swap(d_input, d_output);
        current_n = groups;
    }

    clFinish(queue);

    clEnqueueReadBuffer(queue, d_input, CL_TRUE, 0, sizeof(float), &h_output, 0, nullptr, nullptr);
    LOG_INFO("Reduction Sum CPU Result: %f", host_reduce_sum(h_input, static_cast<int>(input_size)));
    LOG_INFO("Reduction Sum CL Result: %f", h_output);

    CL_CONTEXT_CLEANUP
    return 0;
}