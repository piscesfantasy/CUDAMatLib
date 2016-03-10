// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

	__shared__ float tmp[2*BLOCK_SIZE];
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
	for (int stride=BLOCK_SIZE/2; stride>0; stride/=2) {
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

__global__ void offset(float * input, float * output, int len) {
	__shared__ float tmp[2*BLOCK_SIZE];
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
	for (int i=blockIdx.x; i>0; --i) {
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

int main(int argc, char ** argv) {
    //wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    //args = wbArg_read(argc, argv);

    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));

    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));

    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));

    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));

    //@@ Initialize the grid and block dimensions here
	dim3 block(BLOCK_SIZE, 1, 1);
	dim3 grid((numElements-1)/(BLOCK_SIZE*2)+1, 1, 1);

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	scan<<<grid, block>>>(deviceInput, deviceInput, numElements);
	offset<<<grid, block>>>(deviceInput, deviceOutput, numElements);

    cudaDeviceSynchronize();
	
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));

//	for (int i=0; i<numElements; ++i)
//		wbLog(TRACE, i, ": ", hostInput[i], " -> ", hostOutput[i]);
	
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(hostInput);
    free(hostOutput);

    return 0;
}