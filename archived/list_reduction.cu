// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#define BLOCK_SIZE 512 //@@ You can change this

__global__ void total(float * input, float * output, int len) {
	__shared__ float tmp[BLOCK_SIZE*2];
	
	//@@ Load a segment of the input vector into shared memory
	int inputIdx = 2*(blockIdx.x*blockDim.x+threadIdx.x);
	if (inputIdx < len)
		tmp[2*threadIdx.x] = input[inputIdx];
	else
		tmp[2*threadIdx.x] = 0.0;
	++inputIdx;
	if (inputIdx < len)
		tmp[2*threadIdx.x+1] = input[inputIdx];
	else
		tmp[2*threadIdx.x+1] = 0.0;
	
	//@@ Traverse the reduction tree
	for (int stride=BLOCK_SIZE; stride>=1; stride/=2) {
		__syncthreads();
		if (threadIdx.x<stride)
			tmp[threadIdx.x]+=tmp[threadIdx.x+stride];
	}	
	
	//@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
	output[blockIdx.x] = tmp[0];
}

int main(int argc, char ** argv) {
    int ii;
    //wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    //args = wbArg_read(argc, argv);

    //hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    //@@ Allocate GPU memory here
	cudaMalloc((void **) &deviceInput, numInputElements*sizeof(float));
	cudaMalloc((void **) &deviceOutput, numOutputElements*sizeof(float));
	
    //@@ Copy memory to the GPU here
	cudaMemcpy(deviceInput, hostInput, numInputElements*sizeof(float),
               cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
	dim3 block(BLOCK_SIZE, 1, 1);
	dim3 grid(numOutputElements, 1, 1);

    //@@ Launch the GPU Kernel here
	total<<<grid, block>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
	
    //@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostOutput, deviceOutput, numOutputElements*sizeof(float),
               cudaMemcpyDeviceToHost);
	
    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    //@@ Free the GPU memory here
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

    free(hostInput);
    free(hostOutput);

    return 0;
}
