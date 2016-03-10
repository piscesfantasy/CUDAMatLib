#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename Type>
__global__ void vecAdd(Type *in1, Type *in2, Type *out, int len)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < len)
        out[idx] = in1[idx]+in2[idx];
}

#endif
