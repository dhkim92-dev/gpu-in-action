#include "helper.hpp"
#include "opencl_helper.hpp"

void init_sizes(
    cl_context ctx,
    std::vector<cl_mem>& d_grp_sums,
    std::vector<int32_t>& gszs,
    std::vector<int32_t>& lszs,
    std::vector<int32_t>& limits,
    size_t lsz,
    int size
) {

    while ( size > 4 * lsz ) {
        int32_t gsz = (size + 3) / 4;
        int32_t nr_groups = (gsz + lsz - 1) / lsz;
        limits.push_back( gsz );
        gszs.push_back( nr_groups * lsz * 4 );
        lszs.push_back( lsz ); 
        size = nr_groups;
        d_grp_sums.push_back( clCreateBuffer(
            ctx,
            CL_MEM_READ_WRITE,
            sizeof(int) * (size + 1),
            nullptr,
            nullptr
        ));
    }

    if ( size ) {
        d_grp_sums.push_back(nullptr);
        limits.push_back( size );
        gszs.push_back( size );
        lszs.push_back( size );
    }

    // for ( int i = 0 ; i < d_grp_sums.size() ; ++i ) {
    //     LOG_DEBUG("d_grp_sums[%d] = %p\n", i, d_grp_sums[i]);
    // }
}

void gpu_prefix_sum(
    cl_command_queue queue,
    cl_kernel k_scan4,
    cl_kernel k_scan_ed,
    cl_kernel k_uniform_update,
    cl_mem d_input,
    cl_mem d_output,
    std::vector<cl_mem>& d_grp_sums,
    const std::vector<int32_t>& gszs,
    const std::vector<int32_t>& lszs,
    const std::vector<int32_t>& limits,
    const int n
) 
{
    std::vector<cl_mem> d_srcs = {d_input};
    std::vector<cl_mem> d_dsts = {d_output};

    for(auto d_grp : d_grp_sums) 
    {
        d_srcs.push_back(d_grp);
        d_dsts.push_back(d_grp);
    }

    for(uint32_t i = 0 ; i < d_grp_sums.size() ; ++i ) 
    {
        cl_mem d_src = d_srcs[i];
        cl_mem d_dst = d_dsts[i];
        cl_mem d_grp_sum = d_grp_sums[i];
        int32_t gsz = gszs[i];
        int32_t lsz = lszs[i];
        int32_t limit = limits[i];
        cl_int err = CL_SUCCESS;

        LOG_DEBUG("Processing level %u : \n\td_src: %p \n\td_dst: %p \n\td_grp_sum: %p \n\tgws=%d, \n\tlws=%d \n\tlimit=%d", i, d_src, d_dst, d_grp_sum, gsz, lsz, limit);

        if ( d_grp_sum != nullptr ) 
        {
            // scan4 
            const size_t global_work_size = gsz;
            const size_t local_work_size = lsz;
            err  = clSetKernelArg(k_scan4, 0, sizeof(cl_mem), &d_src);           err |= clSetKernelArg(k_scan4, 1, sizeof(cl_mem), &d_dst);
            err |= clSetKernelArg(k_scan4, 2, sizeof(cl_mem), &d_grp_sum);
            err |= clSetKernelArg(k_scan4, 3, sizeof(int), &limit);
            LOG_DEBUG("Enqueue scan4: \n\tgws=%zu, \n\tlws=%zu \n\tlimits : %d \n\td_dst: %p \n\td_src : %p", global_work_size, local_work_size, limit, d_dst, d_src);
            CHECK_CL_ERROR(err, "Failed to set kernel args for scan4");
            err = clEnqueueNDRangeKernel(
                queue,
                k_scan4,
                1,
                nullptr,
                &global_work_size,
                &local_work_size,
                0,
                nullptr,
                nullptr
            );
            CHECK_CL_ERROR(err, "Failed to enqueue scan4");
        }
        else 
        {
            // call scan_ed
            uint32_t sz_lmem  = sizeof(cl_int) * lsz * 2;
            const size_t global_work_size = gsz;
            const size_t local_work_size = lsz;
            err  = clSetKernelArg(k_scan_ed, 0, sizeof(cl_mem), &d_src);
            err |= clSetKernelArg(k_scan_ed, 1, sizeof(cl_mem), &d_dst);
            err |= clSetKernelArg(k_scan_ed, 2,  sz_lmem, nullptr);
            LOG_DEBUG("Enqueue scan_ed: \n\tgws=%zu, \n\tlws=%zu \n\td_dst: %p \n\td_src : %p", global_work_size, local_work_size, d_dst, d_src);
            CHECK_CL_ERROR(err, "Failed to set kernel args for scan_ed");
            err = clEnqueueNDRangeKernel(
                queue,
                k_scan_ed,
                1,
                nullptr,
                &global_work_size,
                &local_work_size,
                0,
                nullptr,
                nullptr
            );
            CHECK_CL_ERROR(err, "Failed to enqueue scan_ed");
        }
    }

    for ( uint32_t i = d_grp_sums.size() - 1 ; i > 0 ; --i ) 
    {
        cl_mem d_dst = d_dsts[i - 1];
        cl_mem d_src_sum = d_dsts[i];
        int32_t gsz = gszs[i - 1];
        int32_t lsz = lszs[i - 1];
        cl_int err = CL_SUCCESS;

        // uniform_update
        const size_t global_work_size = gsz / 4;  // int4 단위이므로 4로 나눔
        const size_t local_work_size = lsz;
        err  = clSetKernelArg(k_uniform_update, 0, sizeof(cl_mem), &d_dst);
        err |= clSetKernelArg(k_uniform_update, 1, sizeof(cl_mem), &d_src_sum);
        LOG_DEBUG("Enqueue uniform_update: \n\tgws=%zu, \n\tlws=%zu \n\td_dst: %p \n\td_src_sum : %p", global_work_size, local_work_size, d_dst, d_src_sum);
        CHECK_CL_ERROR(err, "Failed to set kernel args for uniform_update");
        err = clEnqueueNDRangeKernel(
            queue,
            k_uniform_update,
            1,
            nullptr,
            &global_work_size,
            &local_work_size,
            0,
            nullptr,
            nullptr
        );
        CHECK_CL_ERROR(err, "Failed to enqueue uniform_update");
    }
}

// inclusive prefix sum
void h_prefixsum(int *src, int *dst, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += src[i];
        dst[i] = sum;
    }
}

int main(void)
{
    CL_CONTEXT_INIT

    const int n = 1 << 26;  
    const size_t sz_mem = sizeof(int) * n;
    const size_t lsz = 64;
    int* h_input = new int[n];
    int* h_output_gpu = new int[n];
    int* h_output_cpu = new int[n];

    init_random_values_i32(h_input, n, 8);
    // for (int i = 0; i < n; ++i) {
        // h_input[i] = 1; // simple case
    // }
    cl_mem d_input = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sz_mem,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for input array");
    LOG_DEBUG("[GPU] Input Addr: %p", d_input);
    cl_mem d_output = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * n,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create buffer for output array");
    LOG_DEBUG("[GPU] Output Addr: %p", d_output);

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
    clFinish(queue);

    // Keep the kernel source string alive for the duration of program creation.
    // Calling .c_str() on a temporary std::string yields a dangling pointer.
    std::string kernel_code = read_kernel_file("kernel.cl");
    const char* kernel_source = kernel_code.c_str();
    cl_program program = clCreateProgramWithSource(
        context,
        1,
        &kernel_source,
        nullptr,
        &err
    );
    CHECK_CL_ERROR(err, "Failed to create program from source");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    check_program_build_result(err, program, device);

    cl_kernel k_scan4 = clCreateKernel(program,"scan4", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel for scan4");
    cl_kernel k_scan_ed = clCreateKernel(program, "scan_ed", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel for scan_ed");
    cl_kernel k_uniform_update = clCreateKernel(program, "uniform_update", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel for uniform_update");

    std::vector<cl_mem> d_grp_sums={};
    std::vector<int32_t> gszs = {};
    std::vector<int32_t> lszs = {};
    std::vector<int32_t> limits = {};
    LOG_DEBUG("Initializing sizes...");
    init_sizes(context, d_grp_sums, gszs, lszs, limits, lsz, n);

    BENCHMARK_START(host_prefixsum)
    h_prefixsum(h_input, h_output_cpu, n);
    BENCHMARK_END(host_prefixsum)

    BENCHMARK_START(gpu_prefix_sum)
    gpu_prefix_sum(
        queue,
        k_scan4,
        k_scan_ed,
        k_uniform_update,
        d_input,
        d_output,
        d_grp_sums,
        gszs,
        lszs,
        limits,
        n
    );

    CHECK_CL_ERROR(clEnqueueReadBuffer(
        queue,
        d_output,
        CL_TRUE,
        0,
        sizeof(int) * n,
        h_output_gpu,
        0,
        nullptr,
        nullptr
    ), "Failed to read output buffer");
    clFinish(queue);
    // Verify results
    BENCHMARK_END(gpu_prefix_sum)

    bool match = true;
    for (int i = 0; i < n; ++i) {
        if (h_output_gpu[i] != h_output_cpu[i]) {
            match = false;
            std::cerr << "Mismatch at index " << i << ": GPU result "<< h_output_gpu[i]
                      << ", CPU result " << h_output_cpu[i] << std::endl;
            break;
        }
    }

    LOG_INFO("Prefix sum %s", match ? "PASSED" : "FAILED");

    clReleaseKernel(k_scan4);
    clReleaseKernel(k_scan_ed);
    clReleaseKernel(k_uniform_update);
    clReleaseProgram(program);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    for (auto d_buf : d_grp_sums) {
        if (d_buf != nullptr) {
            clReleaseMemObject(d_buf);
        }
    }
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;

    CL_CONTEXT_CLEANUP
    return 0;
}