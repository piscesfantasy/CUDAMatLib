#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include "cuda_matrix_kernel.h"
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

template <typename Type>
class CUDA_matrix
{
    public:
        CUDA_matrix(){}
        CUDA_matrix(const vector< vector<Type> >& input)
        {
            num_rows = input.size();
            num_cols = input[0].size();
            _val = new Type[num_rows*num_cols];
            for (int r=0; r<num_rows; ++r)
                for (int c=0; c<num_cols; ++c)
                    _val[r*num_cols+c] = input[r][c];
        }
        CUDA_matrix(Type** input, const int& _r, const int& _c)
        {
            num_rows = _r;
            num_cols = _c;
            _val = new Type[num_rows*num_cols];
            for (int r=0; r<num_rows; ++r)
                for (int c=0; c<num_cols; ++c)
                    _val[r*num_cols+c] = input[r][c];
        }
        virtual ~CUDA_matrix(){ if (_val!=NULL) delete [] _val; }

        int getNumRows(){ return num_rows;}
        int getNumCols(){ return num_cols;}
        Type* getValue() const{ return _val; }

    private:
        int num_rows;
        int num_cols;
        Type *_val;
};

template <typename Type>
void CUDA_matrix<Type>::multiply(CUDA_matrix<Type> const &input, CUDA_matrix<Type> *output)
{
    Type *hostA; // The A matrix
    Type *hostB; // The B matrix
    Type *hostC; // The output C matrix
    Type *d_in1;
    Type *d_in2;
    Type *d_out;
    int numBRows = input.getNumRows();
    int numBColumns = input.getNumCols();
    int numCRows;
    int numCColumns;

    numCRows = num_rows;
    numCColumns = numBColumns;

    hostC = ( Type * )malloc(numCRows*numCColumns*sizeof(Type)); 

    cudaMalloc((void**) &d_in1, num_rows*num_cols*sizeof(Type));
    cudaMalloc((void**) &d_in2, numBRows*numBColumns*sizeof(Type));
    cudaMalloc((void**) &d_out, numCRows*numCColumns*sizeof(Type));

    cudaMemcpy(d_in1, hostA, num_rows*num_cols*sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, hostB, numBRows*numBColumns*sizeof(Type), cudaMemcpyHostToDevice);

    dim3 grid((numCRows-1)/16+1, (numCColumns-1)/16+1, 1);
    dim3 block(16, 16, 1);

    matrixMultiply<<<grid, block>>>(d_in1, d_in2, d_out, num_rows, num_cols, numBRows, numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();

    cudaMemcpy(hostC, d_out, numCRows*numCColumns*sizeof(Type), cudaMemcpyDeviceToHost);	

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    return 0;
}

#endif
