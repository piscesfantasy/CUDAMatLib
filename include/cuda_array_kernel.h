#ifndef CUDA_ARRAY_KERNEL_H
#define CUDA_ARRAY_KERNEL_H

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define SEGSIZE 256
#define BLKSIZE 16
#define BLKSIZE_LARGE 512

template <typename Type>
__global__ void vecAdd(Type *in1, Type *in2, Type *out, int len)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < len)
        out[idx] = in1[idx]+in2[idx];
}

template <typename Type>
__global__ void vecProd(Type *in1, Type *in2, Type *out, int len)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < len)
        out[idx] = in1[idx]*in2[idx];
}

template <typename Type>
__global__ void total(Type *input, Type *output, int len)
{
    __shared__ Type tmp[BLKSIZE*2];

    //@@ Load a segment of the input vector into shared memory
    int inputIdx = 2*(blockIdx.x*blockDim.x+threadIdx.x);
    if (inputIdx < len)
        tmp[2*threadIdx.x] = input[inputIdx];
    else
        tmp[2*threadIdx.x] = 0;
    ++inputIdx;
    if (inputIdx < len)
        tmp[2*threadIdx.x+1] = input[inputIdx];
    else
        tmp[2*threadIdx.x+1] = 0;

    //@@ Traverse the reduction tree
    for (int stride=BLKSIZE; stride>=1; stride/=2)
    {
        __syncthreads();
        if (threadIdx.x<stride)
            tmp[threadIdx.x]+=tmp[threadIdx.x+stride];
    }	

    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    output[blockIdx.x] = tmp[0];
}

template <typename Type>
__global__ void scan(Type * input, Type * output, int len)
{
    __shared__ Type tmp[2*BLOCK_SIZE];
    int loadingIdx = 2*(blockIdx.x*blockDim.x+threadIdx.x);

    // Read memory
    if (loadingIdx<len)
        tmp[2*threadIdx.x] = input[loadingIdx];
    else
        tmp[2*threadIdx.x] = 0.0;
    if (loadingIdx+1<len)
        tmp[2*threadIdx.x+1] = input[loadingIdx+1];
    else
        tmp[2*threadIdx.x+1] = 0.0;

    // Reduction phase
    for (int stride=1; stride<=BLOCK_SIZE; stride*=2) {
        __syncthreads();
        int reductionIdx = (threadIdx.x+1)*2*stride-1;
        if (reductionIdx < 2*BLOCK_SIZE)
            tmp[reductionIdx]+=tmp[reductionIdx-stride];
    }

    // Post reduction reverse phase
    for (int stride=BLOCK_SIZE/2; stride>0; stride/=2)
    {
        __syncthreads();
        int reductionIdx = (threadIdx.x+1)*2*stride-1;
        if (reductionIdx+stride < 2*BLOCK_SIZE)
            tmp[reductionIdx+stride]+=tmp[reductionIdx];
    }

    // Write memory
    __syncthreads();
    if (loadingIdx<len)
        output[loadingIdx] = tmp[2*threadIdx.x];
    if (loadingIdx+1<len)
        output[loadingIdx+1] = tmp[2*threadIdx.x+1];
}

template <typename Type>
__global__ void offset(Type * input, Type * output, int len)
{
    __shared__ Type tmp[2*BLOCK_SIZE];
    int loadingIdx = 2*(blockIdx.x*blockDim.x+threadIdx.x);

    // Read memory
    if (loadingIdx<len)
        tmp[2*threadIdx.x] = input[loadingIdx];
    else
        tmp[2*threadIdx.x] = 0.0;
    if (loadingIdx+1<len)
        tmp[2*threadIdx.x+1] = input[loadingIdx+1];
    else
        tmp[2*threadIdx.x+1] = 0.0;

    // Add offset
    for (int i=blockIdx.x; i>0; --i)
    {
        tmp[2*threadIdx.x] += input[2*BLOCK_SIZE*i-1];
        tmp[2*threadIdx.x+1] += input[2*BLOCK_SIZE*i-1];
    }

    // Write memory
    __syncthreads();
    if (loadingIdx<len)
        output[loadingIdx] = tmp[2*threadIdx.x];
    if (loadingIdx+1<len)
        output[loadingIdx+1] = tmp[2*threadIdx.x+1];
}

#endif
