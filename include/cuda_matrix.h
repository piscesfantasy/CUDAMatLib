#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include "cuda_object.h"
#include "cuda_matrix_kernel.h"
#include <cassert>

using namespace std;

template <typename Type>
class CUDA_matrix : public CUDA_object<Type>
{
    public:
        // Constructor/Destructor
        CUDA_matrix(){}
        CUDA_matrix(const int& _r, const int& _c);
        CUDA_matrix(Type** input, const int& _r, const int& _c);
        CUDA_matrix(const vector< vector<Type> >& input);
        CUDA_matrix(const CUDA_matrix<Type>& other);
        virtual ~CUDA_matrix();

        // proxy class
        template <typename Type2>
        class CUDA_matrix_row {
            friend class CUDA_matrix<Type2>;
            public:
                Type2& operator[](const size_t c){return parent._val[offset+c];}
            private:
                CUDA_matrix_row(CUDA_matrix &p, int o) : parent(p), offset(o) {}
                CUDA_matrix& parent;
                int offset;
        };

        // Set/Get
        virtual int size() const {return num_rows*num_cols;}
        virtual int getNumRows() const {return num_rows;}
        virtual int getNumCols() const {return num_cols;}
        virtual void resize(const int& r, const int& c);
        virtual void setValue(Type** input);
        virtual void setValue(const vector< vector<Type> >& input);
        virtual CUDA_matrix_row<Type> operator[](const size_t r) { return CUDA_matrix_row<Type>(*this, r*num_cols); }

        // Operations
        template <typename Type2>
        friend void CUDA_matrix_multiply(CUDA_matrix<Type2> &in1, CUDA_matrix<Type2> &in2, CUDA_matrix<Type2> &out);
        //virtual Type sum();
        //virtual Type min();
        //virtual Type max();
        //virtual void cumulate();
        virtual void convolve(const CUDA_matrix<Type>& mask);

    private:
        int num_rows;
        int num_cols;
};

template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(const int& _r, const int& _c) : num_rows(_r), num_cols(_c)
{
    this->reset();
}

template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(Type** input, const int& _r, const int& _c) : num_rows(_r), num_cols(_c)
{
    this->reset();
    setValue(input);
}

template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(const vector< vector<Type> >& input): num_rows(input.size()), num_cols(input[0].size())
{
    this->reset();
    setValue(input);
}

template <typename Type>
CUDA_matrix<Type>::CUDA_matrix(const CUDA_matrix<Type>& other): num_rows(other.getNumRows()), num_cols(other.getNumCols())
{
    this->reset();
    for (int r=0; r<num_rows; ++r)
        for (int c=0; c<num_cols; ++c)
            this->_val[r*num_cols+c] = other.getValue()[r*num_cols+c];
}

template <typename Type>
CUDA_matrix<Type>::~CUDA_matrix()
{
    if (this->_val!=NULL)
    {
        delete [] this->_val;
        cudaFree(this->cuda_val);
    }
}
        
template <typename Type>
void CUDA_matrix<Type>::resize(const int& r, const int& c)
{
    num_rows = r;
    num_cols = c;
    this->reset();
}
        
template <typename Type>
void CUDA_matrix<Type>::setValue(Type** input)
{
    for (int r=0; r<num_rows; ++r)
        for (int c=0; c<num_cols; ++c)
            this->_val[r*num_cols+c] = input[r][c];
}

template <typename Type>
void CUDA_matrix<Type>::setValue(const vector< vector<Type> >& input)
{
    for (int r=0; r<num_rows; ++r)
        for (int c=0; c<num_cols; ++c)
            this->_val[r*num_cols+c] = input[r][c];
}

template <typename Type>
void CUDA_matrix<Type>::convolve(const CUDA_matrix<Type>& mask)
{
    assert(mask.getNumRows()<=num_rows);
    assert(mask.getNumCols()<=num_cols);

    Type *d_out;
    cudaMalloc((void **) &d_out, size()*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_matrix::convolve");

    this->cudaMemcpyToDevice();
    mask.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpy @ CUDA_matrix::convolve");

	int tile_width = BLKSIZE_2D-mask.getNumCols()+1;
	int tile_length = BLKSIZE_2D-mask.getNumRows()+1;
	dim3 block(BLKSIZE_2D, BLKSIZE_2D, 1);
	dim3 grid((num_rows-1)/tile_width+1, (num_cols-1)/tile_length+1, 1);

	Convolution<<<grid, block>>>(this->cuda_val, num_rows, num_cols, mask.getCUDAValue(), mask.getNumRows(), mask.getNumCols(), tile_length, tile_width, d_out);
    cudaDeviceSynchronize();
	
    cudaMemcpy(this->_val, d_out, size()*sizeof(Type), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_matrix::convolve");

    cudaFree(d_out);
}

template <typename Type>
void CUDA_matrix_multiply(CUDA_matrix<Type> &in1, CUDA_matrix<Type> &in2, CUDA_matrix<Type> &out)
{
    if (in1.num_cols!=in2.num_rows)
    {
        cerr<<"ERROR: can't multiply matrices, number of row and column doesn't match"<<endl;
        return;
    }

    out.resize(in1.num_rows, in2.num_cols);
    cudaCheckErrors("cudaMalloc @ CUDA_matrix_multiply");

    in1.cudaMemcpyToDevice();
    in2.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_matrix_multiply");

    dim3 grid((out.num_rows-1)/BLKSIZE_2D+1, (out.num_cols-1)/BLKSIZE_2D+1, 1);
    dim3 block(BLKSIZE_2D, BLKSIZE_2D, 1);

    matrixMultiplyShared<<<grid, block>>>(in1.cuda_val, in2.cuda_val, out.cuda_val, in1.num_rows, in1.num_cols, in2.num_rows, in2.num_cols, out.num_rows, out.num_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(out._val, out.cuda_val, out.size()*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_matrix_multiply");
}

#endif
