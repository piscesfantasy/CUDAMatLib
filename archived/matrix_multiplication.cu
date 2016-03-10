
// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	int C_x = blockIdx.x*blockDim.x + threadIdx.x;
	int C_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (C_x < numCRows && C_y < numCColumns){
		float tmp_c_element = 0;
		
	  	int A_x = C_x;
		int B_y = C_y;
		for (int i=0; i<numAColumns; ++i) {
			int A_idx = A_x*numAColumns+i;
            int B_idx = i*numBColumns+B_y;
            tmp_c_element += A[A_idx]*B[B_idx];
        }

        C[C_x*numCColumns+C_y] = tmp_c_element;
    }
}

int main(int argc, char **argv) {
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    //args = wbArg_read(argc, argv);

    //hostA = ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    //hostB = ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    //@@ Allocate the hostC matrix
    hostC = ( float * )malloc(numCRows*numCColumns*sizeof(float)); 

    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceA, numARows*numAColumns*sizeof(float));
    cudaMalloc((void**) &deviceB, numBRows*numBColumns*sizeof(float));
    cudaMalloc((void**) &deviceC, numCRows*numCColumns*sizeof(float));

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 grid((numCRows-1)/16+1, (numCColumns-1)/16+1, 1);
    dim3 block(16, 16, 1);

    //@@ Launch the GPU Kernel here
    matrixMultiply<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);	

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
