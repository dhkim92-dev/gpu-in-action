#pragma once
#ifdef USE_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>


const char* get_cl_error_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        // Add more error codes as needed
        default: return "Unknown OpenCL error";
    }
}

#ifndef CHECK_CL(err) \ 
#define CHECK_CL(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << get_cl_error_string(err) << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }
#endif

std::string read_kernel_file(const char* file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string kernel_code = ss.str();
    char* code_cstr = new char[kernel_code.size() + 1];
    std::strcpy(code_cstr, kernel_code.c_str());
    return code_cstr;
}

#define CL_CONTEXT_INIT \
    cl_int err; \
    cl_platform_id platform; \
    err = clGetPlatformIDs(1, &platform, nullptr); \
    CHECK_CL(err); \
    cl_device_id device; \
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr); \
    CHECK_CL(err); \
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); \
    CHECK_CL(err); \
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err); \
    CHECK_CL(err);

#define CL_CONTEXT_CLEANUP \
    clReleaseCommandQueue(queue); \
    clReleaseContext(context);


#endif
