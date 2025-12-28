#include "helper.hpp"
#include "cuda_helper.hpp"
#include "kernel.cuh"

void h_prefix_sum(
    int* __restrict__ src,
    int* __restrict__ dst,
    const int n
) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += src[i];
        dst[i] = sum;
    }
}

void init_sizes(
    std::vector<int>& gszs,
    std::vector<int>& lszs,
    std::vector<int>& limits,
    std::vector<int*>& d_grps,
    int sm,
    int n
) {
    LOG_DEBUG("init_sizes start");
    while ( n > sm * 4) {
        int gsz = (n + 3) / 4;
        int nr_blks = (gsz + sm - 1) / sm;
        limits.push_back(gsz);
        gszs.push_back(nr_blks * sm * 4);
        lszs.push_back(sm);
        n = nr_blks;
        int* d_grp = nullptr;
        cuda_check(cudaMalloc((void**)&d_grp, sizeof(int) * (n + 1)), "Fail to malloc d_grp");
        d_grps.push_back(d_grp);
    }

    if ( n ) {
        d_grps.push_back(nullptr);
        limits.push_back(n);
        gszs.push_back(n);
        lszs.push_back(n);
    }
    LOG_DEBUG("init_sizes end");
}

void gpu_prefix_sum(
    int* d_input,
    int* d_output,
    std::vector<int *>& d_grps,
    std::vector<int>& gszs,
    std::vector<int>& lszs,
    std::vector<int>& limits,
    int sm,
    int n
) {
    std::vector<int*> d_srcs = { d_input };
    std::vector<int*> d_dsts = { d_output };

    for ( auto d_grp : d_grps ) {
        d_srcs.push_back(d_grp);
        d_dsts.push_back(d_grp);
    }

    for ( int i = 0 ; i < d_grps.size() ; ++i ) {
        int* d_grp = d_grps[i];
        int gsz = gszs[i];
        int lsz = lszs[i];
        int limit = limits[i];
        int* d_src = d_srcs[i];
        int* d_dst = d_dsts[i];

        if ( d_grp != nullptr ) {
            int nr_blk = (gsz + lsz - 1) / lsz;
            int4* d_src4 = reinterpret_cast<int4*>(d_src);
            int4* d_dst4 = reinterpret_cast<int4*>(d_dst);
            scan4<<<nr_blk, lsz>>>(
                d_src4,
                d_dst4,
                d_grp,
                limit
            );
            //cuda_check(cudaGetLastError(), "scan4 launch failed.");
        } else {
            int nr_blk = (limit + lsz - 1) / lsz;
            int sz_sm = sizeof(int) * lsz * 2;
            scan_ed<<<nr_blk, lsz, sz_sm>>>(
                d_src,
                d_dst
            );
            //cuda_check(cudaGetLastError(), "scan_ed launch failed.");
        }
    }

    for ( int i = d_grps.size() - 1 ; i > 0 ; --i ) {
        int * d_dst = d_dsts[i-1];
        int* d_src_sum = d_dsts[i];  // use scanned result from previous level
        int gsz = gszs[i-1];
        int lsz = lszs[i-1];
        int limit = limits[i-1];

        int nr_blk = (gsz/4 + lsz - 1) / lsz;
        uniform_update<<<nr_blk, lsz>>>(
            reinterpret_cast<int4*>(d_dst),
            d_src_sum
        );  
        // cuda_check(cudaGetLastError(), "uniform_update launch failed");
    }
}

int main(void)
{
    const int n = 1 << 20;
    const size_t sz_mem = sizeof(int) * n;
    const int sm = 64;
    int* h_input = new int[n];
    int* h_output_gpu = new int[n];
    int* h_output_cpu = new int[n];
    int* d_input = nullptr;
    int* d_output = nullptr;
    
    std::vector<int> gszs;
    std::vector<int> lszs;
    std::vector<int> limits;
    std::vector<int*> d_grps;

    init_random_values_i32(h_input, n, 100);
    cuda_check(cudaMalloc((void**)&d_input, sz_mem), "malloc failed, d_input");
    cuda_check(cudaMalloc((void**)&d_output, sz_mem), "malloc failed, d_output");
    cuda_check(cudaMemcpy(d_input, h_input, sz_mem, cudaMemcpyHostToDevice), "memcpy failed, h_input to d_input");

    BENCHMARK_START(host_prefix_sum)
    h_prefix_sum(h_input, h_output_cpu, n);
    BENCHMARK_END(host_prefix_sum)

    init_sizes(gszs, lszs, limits, d_grps, sm, n);
    BENCHMARK_START(gpu_prefix_sum)
    gpu_prefix_sum(d_input, d_output, d_grps, gszs, lszs, limits, sm, n);
    cuda_check(cudaMemcpy(h_output_gpu, d_output, sz_mem, cudaMemcpyDeviceToHost), "fail to memcpy to h_output_gpu from d_output");
    BENCHMARK_END(gpu_prefix_sum)
    // Verify results
    bool match = true;
    for (int i = 0; i < n; ++i) {
        if (h_output_gpu[i] != h_output_cpu[i]) {
            match = false;
            printf("Mismatch at index %d: GPU %d != CPU %d\n", i, h_output_gpu[i], h_output_cpu[i]);
            break; 
        }
    }
    LOG_INFO("Prefix sum %s", match ? "PASSED" : "FAILED");

    cuda_check(cudaFree(d_input), "cudaFree failed");
    cuda_check(cudaFree(d_output), "cudaFree failed");
    for (auto ptr : d_grps) {
        if (ptr != nullptr) {
            cuda_check(cudaFree(ptr), "cudaFree failed");
        }
    }

    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    return 0;
}