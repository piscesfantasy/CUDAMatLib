#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define SegSize 256

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < len)
		out[idx] = in1[idx] + in2[idx];
}

int main(int argc, char ** argv) {
    int inputLength;
    float * hostInputA;
    float * hostInputB;
    float * hostOutput;

    //args = wbArg_read(argc, argv);

    //hostInputA = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    //hostInputB = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
	
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	
	float *d_A0, *d_B0, *d_C0;// device memory for stream 0
	float *d_A1, *d_B1, *d_C1;// device memory for stream 1
	float *d_A2, *d_B2, *d_C2;// device memory for stream 0
	float *d_A3, *d_B3, *d_C3;// device memory for stream 1

	cudaMalloc((void**) &d_A0, inputLength*sizeof(float));
	cudaMalloc((void**) &d_B0, inputLength*sizeof(float));
	cudaMalloc((void**) &d_C0, inputLength*sizeof(float));
	cudaMalloc((void**) &d_A1, inputLength*sizeof(float));
	cudaMalloc((void**) &d_B1, inputLength*sizeof(float));
	cudaMalloc((void**) &d_C1, inputLength*sizeof(float));
	cudaMalloc((void**) &d_A2, inputLength*sizeof(float));
	cudaMalloc((void**) &d_B2, inputLength*sizeof(float));
	cudaMalloc((void**) &d_C2, inputLength*sizeof(float));
	cudaMalloc((void**) &d_A3, inputLength*sizeof(float));
	cudaMalloc((void**) &d_B3, inputLength*sizeof(float));
	cudaMalloc((void**) &d_C3, inputLength*sizeof(float));
	
	for (int bias=0; bias<inputLength; bias+=SegSize*4) {
		cudaMemcpyAsync(d_A0, hostInputA+bias, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, hostInputB+bias, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(d_A1, hostInputA+bias+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B1, hostInputB+bias+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
		vecAdd<<<SegSize/16, 16, 0, stream0>>>(d_A0, d_B0, d_C0, inputLength);
		
		cudaMemcpyAsync(d_A2, hostInputA+bias+SegSize*2, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B2, hostInputB+bias+SegSize*2, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream2);
		vecAdd<<<SegSize/16, 16, 0, stream1>>>(d_A1, d_B1, d_C1, inputLength);
		cudaMemcpyAsync(hostOutput+bias, d_C0, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
		
		cudaMemcpyAsync(d_A3, hostInputA+bias+SegSize*3, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B3, hostInputB+bias+SegSize*3, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream3);
		vecAdd<<<SegSize/16, 16, 0, stream2>>>(d_A2, d_B2, d_C2, inputLength);
		cudaMemcpyAsync(hostOutput+bias+SegSize, d_C1, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
		
		vecAdd<<<SegSize/16, 16, 0, stream3>>>(d_A3, d_B3, d_C3, inputLength);
		cudaMemcpyAsync(hostOutput+bias+SegSize*2, d_C2, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream2);
		
		cudaMemcpyAsync(hostOutput+bias+SegSize*3, d_C3, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream3);
	}
	cudaDeviceSynchronize();
	
	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
	cudaFree(d_A2);
	cudaFree(d_B2);
	cudaFree(d_C2);
	cudaFree(d_A3);
	cudaFree(d_B3);
	cudaFree(d_C3);
	free(hostInputA);
    free(hostInputB);
    free(hostOutput);

    return 0;
}
