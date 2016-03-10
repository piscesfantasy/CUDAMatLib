#include <wb.h>

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  	//@@ Insert code to implement matrix multiplication here
  	//@@ You have to use shared memory for this MP
	__shared__ float tmpA[16][16];
  	__shared__ float tmpB[16][16];

  	// The element in C a certain thread is in charge of	
  	int C_x = blockIdx.x*blockDim.x + threadIdx.x;
  	int C_y = blockIdx.y*blockDim.y + threadIdx.y;
	float tmp_c_element = 0;

 	// Load tile_idx-th tile from A and B needed to compute current block in C
	for (int tile_idx=0; tile_idx<(numBRows-1)/16+1; ++tile_idx) {
		// Load the element in A and B a certain thread is in charge of
		int A_x = C_x;
		int A_y = tile_idx*blockDim.y + threadIdx.y;
		int B_x = tile_idx*blockDim.x + threadIdx.x;
		int B_y = C_y;
		if (A_x<numARows && A_y<numAColumns)
			tmpA[threadIdx.x][threadIdx.y] = A[A_x*numAColumns+A_y];
		else
			tmpA[threadIdx.x][threadIdx.y] = 0;
		if (B_x<numBRows && B_y<numBColumns)
			tmpB[threadIdx.x][threadIdx.y] = B[B_x*numBColumns+B_y];
		else
			tmpB[threadIdx.x][threadIdx.y] = 0;
		__syncthreads();
		
		// Calculate the element in C a certain thread is in charge of
		if (C_x < numCRows && C_y < numCColumns) {
			for (int i=0; i<16; ++i)
				tmp_c_element += tmpA[threadIdx.x][i]*tmpB[i][threadIdx.y];
		}
		__syncthreads();
	}
	
	if (C_x < numCRows && C_y < numCColumns)
		C[C_x*numCColumns+C_y] = tmp_c_element;
}

int main(int argc, char **argv) {
    //wbArg_t args;
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

    hostA = ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

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
    matrixMultiplyShared<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
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
