#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
/*#include <cuda_runtime_api.h>*/

using namespace std;

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
    //@@ Insert code to implement vector addition here
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < len){out[idx]=in1[idx]+in2[idx];}
}

int main(int argc, char **argv) {
    //wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    //args = wbArg_read(argc, argv);

    //hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
    //hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = ( float * )malloc(inputLength * sizeof(float));

    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceInput1, inputLength*sizeof(float));
    cudaMalloc((void**) &deviceInput2, inputLength*sizeof(float));
    cudaMalloc((void**) &deviceOutput, inputLength*sizeof(float));

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 grid((inputLength-1)/8+1, 1, 1);
    dim3 block(8, 1, 1);

    //@@ Launch the GPU Kernel here
    vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost);	

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
