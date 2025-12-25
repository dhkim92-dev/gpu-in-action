#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "helper.hpp"

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::printf("code : %d, message : %s", err, msg); \
        exit(EXIT_FAILURE); \
    }

// Function to read kernel file as string
std::string read_kernel_file(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open kernel file: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void init_random_values(float* data, int size) 
{
    for (int i = 0; i < size; ++i) 
    {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() 
{
    // Initialize data
    const int n = 1024;
    std::vector<float> a(n, 0.0f), b(n, 0.0f), c(n);

    // init random values for a and b 
    init_random_values(a.data(), n);
    init_random_values(b.data(), n);

    cl_int err;

    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERROR(err, "Failed to get platform ID");

    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_ERROR(err, "Failed to get device ID");

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERROR(err, "Failed to create context");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "Failed to create command queue");

    // Create buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data(), &err);
    CHECK_ERROR(err, "Failed to create buffer for A");

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data(), &err);
    CHECK_ERROR(err, "Failed to create buffer for B");

    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &err);
    CHECK_ERROR(err, "Failed to create buffer for C");

    // Read and build kernel
    std::string kernelSource = read_kernel_file("kernel.cl");
    const char *kernelCode = kernelSource.c_str();
    size_t kernelSize = kernelSource.size();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelCode, &kernelSize, &err);
    CHECK_ERROR(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build Error:\n" << log.data() << std::endl;
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CHECK_ERROR(err, "Failed to create kernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    CHECK_ERROR(err, "Failed to set kernel argument 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    CHECK_ERROR(err, "Failed to set kernel argument 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    CHECK_ERROR(err, "Failed to set kernel argument 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    CHECK_ERROR(err, "Failed to set kernel argument 3");

    // Execute kernel
    size_t globalWorkSize = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to enqueue kernel");

    // Read back results
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * n, c.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read buffer C");

    // Verify results
    std::cout << "Result of vector addition (first 10 elements):" << std::endl;
    std::cout << "vector A\n";
    for (int i = 0; i < 10; ++i) 
    {
        std::printf("%.2f ", a[i]);
    }
    std::cout << "\nvector B\n";
    for (int i = 0; i < 10; ++i) 
    {
        std::printf("%.2f ", b[i]);
    }

    std::cout << "\nvector C (A + B)\n";
    for (int i = 0; i < 10; ++i) 
    {
        std::printf("%.2f ", c[i]);
    }
    std::cout << std::endl;

    for (int i = 0; i < n; ++i) 
    {
        if (c[i] != a[i] + b[i]) 
        {
            std::cerr << "Verification failed at index " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }


    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
