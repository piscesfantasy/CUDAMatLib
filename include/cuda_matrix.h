#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include "cuda_matrix_kernel.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#define SEGSIZE 256
#define BLKSIZE 16

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
        CUDA_matrix() {_val = NULL;}
        CUDA_matrix(const vector< vector<Type> >& input);
        CUDA_matrix(Type** input, const int& _r, const int& _c);
        CUDA_matrix(const CUDA_matrix<Type>& other);
        virtual ~CUDA_matrix();

        void init(const int& r, const int& c);

        int getNumRows() const {return num_rows;}
        int getNumCols() const {return num_cols;}
        int size() const {return num_rows*num_cols;}
        Type* getValue() const {return _val;}
        Type* getCUDAValue() const { return cuda_val; }
        void cudaMemcpyToDevice() const { 
            cudaMemcpy(cuda_val, _val, size()*sizeof(Type), cudaMemcpyHostToDevice);
        }

        // cumulative summation
        //virtual void cumulate();

        // convolution
        //virtual void convolve(const Type** mask, const int& m_length, const int& m_width);

    private:
        int num_rows;
        int num_cols;
        Type *_val;
        Type *cuda_val; // cuda memory
};


template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(const vector< vector<Type> >& input)
{
    _val = NULL;
    init(input.size(), input[0].size());
    for (int r=0; r<num_rows; ++r)
        for (int c=0; c<num_cols; ++c)
            _val[r*num_cols+c] = input[r][c];
}

template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(Type** input, const int& _r, const int& _c)
{
    _val = NULL;
    init(_r, _c);
    for (int r=0; r<num_rows; ++r)
        for (int c=0; c<num_cols; ++c)
            _val[r*num_cols+c] = input[r][c];
}

template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(const CUDA_matrix<Type>& other)
{
    _val = NULL;
    init(other.getNumRows(), other.getNumCols());
    for (int r=0; r<num_rows; ++r)
        for (int c=0; c<num_cols; ++c)
            _val[r*num_cols+c] = other.getValue()[r*num_cols+c];
}

template <typename Type>
CUDA_matrix<Type>::~CUDA_matrix()
{
    if (_val!=NULL)
    {
        delete [] _val;
        cudaFree(cuda_val);
    }
}
        
template <typename Type>
void CUDA_matrix<Type>::init(const int& r, const int& c)
{
    num_rows = r;
    num_cols = c;
    if (_val!=NULL)
    {
        delete [] _val;
        cudaFree(cuda_val);
    }
    _val = new Type[num_rows*num_cols];
    cudaMalloc((void**) &cuda_val, size()*sizeof(Type));
}

template <typename Type>
void CUDA_matrix_multiply(CUDA_matrix<Type> &in1, CUDA_matrix<Type> &in2, CUDA_matrix<Type> &out)
{
    if (in1.getNumCols()!=in2.getNumRows())
    {
        cerr<<"ERROR: can't multiply matrices, number of row and column doesn't match"<<endl;
        return;
    }

    out.init(in1.getNumRows(), in2.getNumCols());
    cudaCheckErrors("cudaMalloc @ CUDA_matrix_multiply");

    in1.cudaMemcpyToDevice();
    in2.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_matrix_multiply");

    dim3 grid((out.getNumRows()-1)/BLKSIZE+1, (out.getNumCols()-1)/BLKSIZE+1, 1);
    dim3 block(BLKSIZE, BLKSIZE, 1);

    matrixMultiply<<<grid, block>>>(in1.getCUDAValue(), in2.getCUDAValue(), out.getCUDAValue(), in1.getNumRows(), in1.getNumCols(), in2.getNumRows(), in2.getNumCols(), out.getNumRows(), out.getNumCols());
    cudaDeviceSynchronize();

    cudaMemcpy(out.getValue(), out.getCUDAValue(), out.size()*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_matrix_multiply");

    return;
}

#endif
