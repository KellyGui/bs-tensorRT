#include "preprocess.h"
#include "string.h"
#include <opencv2/opencv.hpp>



__global__ void bilinear_kernel(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge)
{
    int position = blockDim.x*blockIdx.x + threadIdx.x;
    if(position>edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c_r, c_g, c_b;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c_r = const_value_st;
        c_g = const_value_st;
        c_b = const_value_st;
    } else {
        int x1 = floorf(src_x);
        int y1 = floorf(src_y);
        int x2 = x1+1;
        int y2 = y1+1;

        uint8_t* Q11_b = src+y1*3*src_width+3*x1;
        uint8_t* Q11_g = src+y1*3*src_width+x1*3+1;
        uint8_t* Q11_r = src+y1*3*src_width+3*x1+2;

        uint8_t* Q12_b = src+y2*3*src_width+x1*3;
        uint8_t* Q12_g = Q12_b+1;
        uint8_t* Q12_r = Q12_g+1;
        
        uint8_t* Q22_b = src+y2*3*src_width+x2*3;
        uint8_t* Q22_g = Q22_b+1;
        uint8_t* Q22_r = Q22_g+1;

        uint8_t* Q21_b = src+y1*3*src_width+x2*3;
        uint8_t* Q21_g = Q21_b+1;
        uint8_t* Q21_r = Q21_g+1;
    
        float R1_r = (float)(*Q11_r)*(x2-src_x)/(x2-x1)+ (float)(*Q21_r)*(src_x-x1)/(x2-x1);
        float R1_g = (float)(*Q11_g)*(x2-src_x)/(x2-x1)+ (float)(*Q21_g)*(src_x-x1)/(x2-x1);
        float R1_b = (float)(*Q11_b)*(x2-src_x)/(x2-x1)+ (float)(*Q21_b)*(src_x-x1)/(x2-x1);

        float R2_r = (float)(*Q12_r)*(x2-src_x)/(x2-x1)+ (float)(*Q22_r)*(src_x-x1)/(x2-x1);
        float R2_g = (float)(*Q12_g)*(x2-src_x)/(x2-x1)+ (float)(*Q22_g)*(src_x-x1)/(x2-x1);
        float R2_b = (float)(*Q12_b)*(x2-src_x)/(x2-x1)+ (float)(*Q22_b)*(src_x-x1)/(x2-x1);

        c_r = R1_r*(y2-src_y)/(y2-y1)+ R2_r*(src_y-y1)/(y2-y1);
        c_g = R1_g*(y2-src_y)/(y2-y1)+ R2_g*(src_y-y1)/(y2-y1);
        c_b = R1_b*(y2-src_y)/(y2-y1)+ R2_b*(src_y-y1)/(y2-y1);
        
        //normalize
        // mean 103.53, 116.28, 123.675
        // std  57.375, 57.12, 58.395
        c_r = (c_r-103.53)/57.375;
        c_g = (c_g-116.28)/57.12;
        c_b = (c_g- 123.675)/58.395;

        int area = dst_width*dst_height;
        float* pdst_cb = dst + dy * dst_width + dx;
        float* pdst_cg = pdst_cb + area;
        float* pdst_cr = pdst_cg + area;
        *pdst_cr = c_r;
        *pdst_cg = c_g;
        *pdst_cb = c_b;

    }

}

__global__ void warpaffine_kernel( 
    uint8_t* src, int src_line_size, int src_width, 
    int src_height, float* dst, int dst_width, 
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}


void preprocess_kernel_img(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream) {
    AffineMatrix s2d,d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));  // copy m2x3_d2s to d2s

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    // bilinear_kernel<<<blocks, threads, 0, stream>>>(
    //     src, src_width,
    //     src_height, dst, dst_width,
    //     dst_height, 128, d2s, jobs);

    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        src, src_width*3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);
}


// int main()
// {
//     cudaSetDevice(0);
//     std::String name;
//     cin>>name;
//     cv::Mat img = cv::imread(name);
//     uint8_t* img_host = nullptr;
//     memcpy(img_host,img.data,3*img.cols*img.rows);
//     float* buffer;
//     cudaStream_t stream;
//     cudaMalloc((void**)&buffer,3*img.cols*img.rows, cudaMemcpyHostToDevice);
//     preprocess_kernel_img(img_host, img.cols, img.rows, buffer, 1024, 1024, stream, stream );                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )
//     //
// }