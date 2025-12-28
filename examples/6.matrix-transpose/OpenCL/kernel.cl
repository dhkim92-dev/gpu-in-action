#define TILE_SIZE 16

/**
* Naive matrix transpose kernel
* Input matrix is of size W x H
* Output matrix is of size H x W
* Each work-item transposes one element
* W = [width of input matrix]
* H = [height of input matrix]
 */
__kernel void transpose_naive(
    __global const float* input,
    __global float* output,
    const int W,
    const int H
) {
    // input size row : H, col : W
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Transpose element
    // But we access output in a column major order, 
    // which is not coalesced memory access
    // Output size row : W, col : H
    // 2D Expression
    // output[y][x] = input[x][y];
    if ( x < W && y < H) {
        output[x * H + y] = input[y * W + x];
    }
}

/**
* Tiled matrix transpose kernel with bank conflict
* Input matrix is of size W x H [row = H, col = W]
* Output matrix is of size H x W
* W = [width of input matrix]
* H = [height of input matrix]
 */
__kernel void transpose_tiled_bank_conflict(
    __global const float* input,
    __global float* output,
    const int W,
    const int H
) {
    __local float tile[TILE_SIZE][TILE_SIZE];

    int x = get_global_id(0);  // col index in input
    int y = get_global_id(1);  // row index in input
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_group_id(0);
    int gy= get_group_id(1);

    // Load input[y][x] into tile[ly][lx] (coalesced read)
    if (x < W && y < H) {
        tile[ly][lx] = input[y * W + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write transposed: output[x][y] = input[y][x]
    // 타일 좌표를 전치하여 출력
    int tx = gy * TILE_SIZE + lx;  // 출력의 col (원래 입력의 row 방향)
    int ty = gx * TILE_SIZE + ly;  // 출력의 row (원래 입력의 col 방향)

    // tile[lx][ly]를 읽을 때 bank conflict 발생 (같은 bank에서 읽음)
    if (tx < H && ty < W) {
        output[ty * H + tx] = tile[lx][ly];
    }
}


__kernel void transpose_tiled_optimized(
    __global const float* input,
    __global float* output,
    const int W,
    const int H
) {
    __local float tile[TILE_SIZE][TILE_SIZE+1]; // +1 to avoid bank conflict

    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_group_id(0);
    int gy= get_group_id(1);

    if (x < W && y < H) {
        tile[ly][lx] = input[y * W + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int tx = gy * TILE_SIZE + lx;  
    int ty = gx * TILE_SIZE + ly;

    if (tx < H && ty < W) {
        output[ty * H + tx] = tile[lx][ly];
    }

}