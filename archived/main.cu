#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

using namespace std;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void vecAdd(int *in1, int *in2, int *out, int len)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < len)
        out[idx] = in1[idx]+in2[idx];
}

int main()
{
    int *i1 = new int[8];
    int *i2 = new int[8];
    int *i3 = new int[8];
    int *d_in1, *d_in2, *d_out;
    int len = 8;

    for (int i=0; i<len; ++i){
        i1[i] = i*2;
        i2[i] = i*3;
    }

    cudaMalloc((void**) &d_in1, len*sizeof(int));
    cudaMalloc((void**) &d_in2, len*sizeof(int));
    cudaMalloc((void**) &d_out, len*sizeof(int));
    cudaCheckErrors("1");

    cudaMemcpy(d_in1, i1, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, i2, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("2");

    dim3 grid((len-1)/8+1, 1, 1);
    dim3 block(8, 1, 1);

    vecAdd<<<grid, block>>>(d_in1, d_in2, d_out, len);
    cudaDeviceSynchronize();

    cudaMemcpy(i3, d_out, len*sizeof(int), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("3");

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    cout<<i3[2]<<endl;
}
