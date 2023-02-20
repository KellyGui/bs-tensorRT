#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct AffineMatrix{
    float value[6];
};

// __global__ void bilinear_kernel(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, AffineMatrix d2s, int edge);

void preprocess_kernel_img(uint8_t* src, int src_width,int src_height,
    float* dst, int dst_width, int dst_height, cudaStream_t stream);