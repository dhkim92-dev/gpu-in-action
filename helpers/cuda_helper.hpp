#include <cuda_runtime.h>
#include <string>
#include <iosream>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(err) << std::endl;
        std::exit((int)err);
    }
}

std::string read_kernel_file(const std::string& filename) 
{
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) 
    {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file_stream)), std::istream
        .buf_iterator<char>());
}