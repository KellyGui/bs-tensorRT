#include "argmax.h"
#include <iostream>
#include "cuda_utils.h"
#include <float.h>
#include <stdio.h>

//Token for git
//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

__global__ void kernel_argmax(float* data, int* output, int height, int width, int classNum)
{
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    int idx = iy*width+ix;
    int size = height*width;
    if(idx<size)
    {
        output[idx]=0;
        for(int stride=1;stride<classNum;++stride){
            if(data[output[idx]*size+idx] < data[stride*size+idx])
                output[idx]=stride;
        }
    }

}

void argmax(float* input, int* output,  int height, int width, int classNum)
{
    // cudaEvent_t start1, stop1;
    // float time1;

    // cudaEventCreate(&start1);

    // cudaEventCreate(&stop1); 

    // dim3 grid_size(1,1,1);
    // dim3 block_size(width, height, 1);
    dim3 block_size(32,32,1);
    int blockpreGridx = (width-1)/block_size.x+1;
    int blockperGridy = (height-1)/block_size.y+1;
    dim3 grid_size(blockpreGridx,blockperGridy,1);

    kernel_argmax <<< grid_size ,block_size >>>(input, output, height, width, classNum);
    // cudaEventRecord(stop1, 0);

    // cudaEventSynchronize(stop1);
    // cudaEventElapsedTime(&time1, start1, stop1);

    // std::cout<<"cuda calculating time is "<<time1<<"ms\n";

    cudaDeviceSynchronize();
}


