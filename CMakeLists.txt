# compile_commands.json 생성 옵션 추가
cmake_minimum_required(VERSION 3.27)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(GPU_IN_ACTION LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# Backend options
# =========================
option(USE_OPENCL "Use OpenCL backend" OFF)
option(USE_CUDA   "Use CUDA backend" OFF)
option(USE_CUDA_STATIC "Link static cudart instead of shared library" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 
#option(USE_VULKAN "Use Vulkan backend" OFF)

if(NOT DEFINED LOG_LEVEL)
    set(LOG_LEVEL 1)  # INFO
endif()

# =========================
# Global backend interface
# =========================
add_library(GPGPUBackend INTERFACE)

# =========================
# OpenCL
# =========================
if(USE_OPENCL)
    message(STATUS "OpenCL backend enabled")

    find_package(OpenCL REQUIRED)
    target_compile_definitions(GPGPUBackend INTERFACE USE_OPENCL)

    if(APPLE)
        target_link_libraries(GPGPUBackend INTERFACE "-framework OpenCL")
        target_include_directories(GPGPUBackend INTERFACE /System/Library/Frameworks/OpenCL.framework/Headers)
    else()
        target_link_libraries(GPGPUBackend INTERFACE OpenCL::OpenCL)
    endif()
endif()

# =========================
# CUDA
# =========================
if(USE_CUDA)
    message(STATUS "CUDA backend enabled")

    enable_language(CUDA)
    target_compile_definitions(GPGPUBackend INTERFACE USE_CUDA)

    if (USE_CUDA_STATIC) 
        target_link_libraries(GPGPUBackend INTERFACE /usr/local/cuda/lib64/libcudart_static.a)
        target_compile_definitions(GPGPUBackend INTERFACE USE_CUDA_STATIC)
    else()
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(GPGPUBackend INTERFACE CUDA::cudart)
        target_compile_definitions(GPGPUBackend INTERFACE USE_CUDA_SHARED)
    endif()
endif()

# =========================
# Vulkan (주석 유지)
# =========================
#[[ f(USE_VULKAN)
    message(STATUS "Vulkan backend enabled")
    target_compile_definitions(GPGPUBackend INTERFACE USE_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(GPGPUBackend INTERFACE Vulkan::Vulkan)
# endif()]]

# =========================
# Helpers
# =========================
add_library(GPGPUHelpers INTERFACE)
target_include_directories(GPGPUHelpers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/helpers)
target_compile_definitions(GPGPUHelpers INTERFACE LOG_LEVEL=${LOG_LEVEL})

# =========================
# Examples
# =========================
add_subdirectory(examples)

