#ifndef CUDA_MATRIX_KERNEL_H
#define CUDA_MATRIX_KERNEL_H

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLKSIZE_2D 16

template <typename Type>
__global__ void matrixMultiply(Type *A, Type *B, Type *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns)
{
	int C_x = blockIdx.x*blockDim.x + threadIdx.x;
	int C_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (C_x < numCRows && C_y < numCColumns){
		Type tmp_c_element = 0;
		
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

template <typename Type>
__global__ void matrixMultiplyShared(Type *A, Type *B, Type *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns)
{
	__shared__ Type tmpA[BLKSIZE_2D][BLKSIZE_2D];
  	__shared__ Type tmpB[BLKSIZE_2D][BLKSIZE_2D];

  	// The element in C a certain thread is in charge of	
  	int C_x = blockIdx.x*blockDim.x + threadIdx.x;
  	int C_y = blockIdx.y*blockDim.y + threadIdx.y;
	Type tmp_c_element = 0;

 	// Load tile_idx-th tile from A and B needed to compute current block in C
	for (int tile_idx=0; tile_idx<(numBRows-1)/BLKSIZE_2D+1; ++tile_idx) {
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
			for (int i=0; i<BLKSIZE_2D; ++i)
				tmp_c_element += tmpA[threadIdx.x][i]*tmpB[i][threadIdx.y];
		}
		__syncthreads();
	}
	
	if (C_x < numCRows && C_y < numCColumns)
		C[C_x*numCColumns+C_y] = tmp_c_element;
}

#endif
