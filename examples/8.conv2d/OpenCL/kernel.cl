
/**
* Basic convolution 2D kernel (naive implementation)
* Input image is of size W x H
* stride always 1
* filter is of size FW x FH
* output size should be W x H 
* @param input: input image in row-major order
* @param output: output image in row-major order
* @param filter: filter in row-major order
* @param W: width of input image
* @param H: height of input image
* @param FW: width of filter
* @param FH: height of filter
 */
__kernel void conv2d_naive(
    __global const float* input,
    __global float* output,
    __global const float* filter,
    const int W,
    const int H,
    const int FW,
    const int FH
) {
    int out_x = get_global_id(0); // output column index
    int out_y = get_global_id(1); // output row index
    float sum = 0.0f;

    // Convolution operation
    for (int fy = 0; fy < FH; fy++) {
        for (int fx = 0; fx < FW; fx++) {
            int in_x = out_x + fx - FW / 2;
            int in_y = out_y + fy - FH / 2;
            // Boundary check
            if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                float in_val = input[in_y * W + in_x];
                float filter_val = filter[fy * FW + fx];
                sum += in_val * filter_val;
            }
        }
    }
    // Write output
    if (out_x < W && out_y < H) {
        output[out_y * W + out_x] = sum;
    } 
}

/**
* Tiled convolution 2D kernel.
* @param input: input image in row-major order
* @param output: output image in row-major order
* @param filter: filter in row-major order
* @param tile: local memory tile for input patch. size should be (TILE_W + FW -1) x (TILE_H + FH -1)
* @param W: width of input image
* @param H: height of input image
* @param FW: width of filter
* @param FH: height of filter
 */
__kernel void conv2d_tiled(
    __global const float* input,
    __global float* output,
    __global const float* filter,
    __local float* tile,
    __local float* l_filter,
    const int W,
    const int H,
    const int FW,
    const int FH
) {
    int or = get_global_id(1); // output row index
    int oc = get_global_id(0); // output column index
    int lr = get_local_id(1); // local row index
    int lc = get_local_id(0); // local column index
    int gr = get_group_id(1); // group row index
    int gc = get_group_id(0); // group column index

    const int TILE_W = get_local_size(0);
    const int TILE_H = get_local_size(1);

    // Load input patch into local memory tile
    for (int ty = lr; ty < TILE_H + FH - 1; ty += get_local_size(1)) {
        for (int tx = lc; tx < TILE_W + FW - 1; tx += get_local_size(0)) {
            int in_x = gc * TILE_W + tx - FW / 2;
            int in_y = gr * TILE_H + ty - FH / 2;
            if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                tile[ty * (TILE_W + FW - 1) + tx] = input[in_y * W + in_x];
            } else {
                tile[ty * (TILE_W + FW - 1) + tx] = 0.0f;
            }
        }
    }

    // Load filter into local memory
    for (int fy = lr; fy < FH; fy += get_local_size(1)) {
        for (int fx = lc; fx < FW; fx += get_local_size(0)) {
            l_filter[fy * FW + fx] = filter[fy * FW + fx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // do convolution
    float sum = 0.0f;
    for (int fy = 0; fy < FH; fy++) {
        for (int fx = 0; fx < FW; fx++) {
            float in_val = tile[(lr + fy) * (TILE_W + FW - 1) + (lc + fx)];
            sum += in_val * l_filter[fy * FW + fx];
        }
    }   
    // Write output
    if (oc < W && or < H) {
        output[or * W + oc] = sum;
    }
}