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

#endif
