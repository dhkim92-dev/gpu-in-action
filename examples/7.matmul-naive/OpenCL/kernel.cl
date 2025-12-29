
/**
* Naive matrix multiplication kernel
* C = A * B
* A is of size M x K
* B is of size K X N
* C is of size M x N
* Work items size is (M x N)
* Each work-item computes one element of C
* M = [number of rows of A]
* K = [number of columns of A / number of rows of B]
* N = [number of columns of B]
 */
__kernel void matmul_naive(
    __global const float* A,
    __global const float* B,
    __global float* C,
    uint M,
    uint K,
    uint N
) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if ( row < M && col < N ) {
        C[row * N + col] = 0.0f;
        for ( int k = 0 ; k < K; ++k ) {
            C[row * N + col] += A[row * K + k] * B[k * N + col];
        }
    }
}

#define TILE_SIZE 16

/**
* Tiled matrix multiplication kernel (basic)
* C = A * B
* A is of size M x K
* B is of size K X N
* C is of size M x N
* Work items size is (M x N)
 */
__kernel void matmul_tiled_basic(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint M,
    const uint K,
    const uint N
) {
    __local float sub_A[TILE_SIZE][TILE_SIZE];
    __local float sub_B[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(1);
    int col = get_global_id(0);
    int l_row = get_local_id(1);
    int l_col = get_local_id(0);

    float sum = 0.0f;

    // tid = tile index
    for ( int tid = 0 ; tid < ( (K + TILE_SIZE - 1) / TILE_SIZE ) ; ++tid ){
        int A_row = row;
        int A_col = tid * TILE_SIZE + l_col;
        int B_row = tid * TILE_SIZE + l_row;;
        int B_col = col;

        // Load tile from A and B into local memory
        if ( A_row < M && A_col < K ) {
            sub_A[l_row][l_col] = A[A_row * K + A_col];
        } else {
            sub_A[l_row][l_col] = 0.0f;
        }

        if ( B_row < K && B_col < N ) {
            sub_B[l_row][l_col] = B[B_row * N + B_col];
        } else {
            sub_B[l_row][l_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for ( int k = 0 ; k < TILE_SIZE ; ++k ) {
            sum += sub_A[l_row][k] * sub_B[k][l_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ( row < M && col < N ) {
        C[row * N + col] = sum;
    }
}