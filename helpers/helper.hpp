#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <memory>

namespace gpgpu_detail {
    bool rand_seeded = false;
}

#ifndef LOG_LEVEL
#define LOG_LEVEL 1  // default = INFO
#endif

enum class LogLevel : int {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3
};

constexpr LogLevel CURRENT_LOG_LEVEL =
    static_cast<LogLevel>(LOG_LEVEL);

#define _LOG_IMPL(level, tag, fmt, ...)                          \
    do {                                                         \
        if constexpr (static_cast<int>(level) >=                \
                      static_cast<int>(CURRENT_LOG_LEVEL)) {    \
            std::fprintf(stderr,                                 \
                "[%s] %s:%d: " fmt "\n",                         \
                tag, __FILE__, __LINE__, ##__VA_ARGS__);         \
        }                                                        \
    } while (0)

#define LOG_DEBUG(fmt, ...) _LOG_IMPL(LogLevel::DEBUG, "DEBUG", fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  _LOG_IMPL(LogLevel::INFO,  "INFO",  fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  _LOG_IMPL(LogLevel::WARN,  "WARN",  fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) _LOG_IMPL(LogLevel::ERROR, "ERROR", fmt, ##__VA_ARGS__)

// ============================
// Benchmark macro
// ============================
// usage: BENCHMARK(foo());
// prints: [BENCH][file:line] foo() : XX.XXX ms
#define HOST_BENCHMARK(func, benchmark_name) {                       \
    auto __host_bench_start_##benchmark_name =                        \
        std::chrono::high_resolution_clock::now();                    \
    func;                                                             \
    auto __host_bench_end_##benchmark_name =                          \
        std::chrono::high_resolution_clock::now();                    \
    double __host_bench_ms_##benchmark_name =                         \
        std::chrono::duration<double, std::milli>(                    \
            __host_bench_end_##benchmark_name -                       \
            __host_bench_start_##benchmark_name).count();             \
    std::printf("[BENCH][%s:%d] %s : %.3f ms\n",                     \
        __FILE__, __LINE__, #benchmark_name,                         \
        __host_bench_ms_##benchmark_name);                           \
}


void init_random_values_f32(float* data, int size) 
{
    if (gpgpu_detail::rand_seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }
    for (int i = 0; i < size; ++i) 
    {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void init_random_values_i32(int* data, int size, int limit = 10) 
{
    if (gpgpu_detail::rand_seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }
    for (int i = 0; i < size; ++i) 
    {
        data[i] = rand() % limit;  
   }
}

inline void seed_rand_u32(unsigned int seed)
{
    std::srand(seed);
    gpgpu_detail::rand_seeded = true;
}

inline void seed_rand_time()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    gpgpu_detail::rand_seeded = true;
}

#define BENCHMARK_START(func_name) {                                               \
        auto __bench_start_##func_name =                                           \
        std::chrono::high_resolution_clock::now();

#define BENCHMARK_END(func_name) \
    auto __bench_end_##func_name =                                             \
        std::chrono::high_resolution_clock::now();                           \
    double __bench_ms_##func_name =                                          \
        std::chrono::duration<double, std::milli>(                           \
            __bench_end_##func_name - __bench_start_##func_name).count();    \
    std::printf("[BENCH][%s:%d] %s : %.3f ms\n",                           \
        __FILE__, __LINE__, #func_name, __bench_ms_##func_name);            \
    }

