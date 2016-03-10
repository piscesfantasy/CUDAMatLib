#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include "cuda_device.h"
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

template <typename Type> class CUDA_vector
{
    public:
        CUDA_vector(){}
        CUDA_vector(Type* data, const int& l)
        {
            len = l;
            val = new Type[len];
            for (int i=0; i<len; ++i)
                val[i] = data[i];
        }
        virtual ~CUDA_vector() {if (val!=NULL) delete [] val;}

        Type &operator[](const size_t idx){ return val[idx]; }
        int getLen() const{ return len; }
        Type* getValue() const{ return val; }

        void add( CUDA_vector<Type> const&);
        //Type getSum();
        //void getCdf(Type*);

    private:
        Type* val;
        int len;
};

template <typename Type>
void CUDA_vector<Type>::add(CUDA_vector<Type> const &input)
{
    if (input.getLen()!=len)
        return;

    Type *d_in1, *d_in2, *d_out;

    cudaMalloc((void**) &d_in1, len*sizeof(Type));
    cudaMalloc((void**) &d_in2, len*sizeof(Type));
    cudaMalloc((void**) &d_out, len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_vector<Type>::add");

    cudaMemcpy(d_in1, val, len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, input.getValue(), len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_vector<Type>::add");

    dim3 grid((len-1)/8+1, 1, 1);
    dim3 block(8, 1, 1);

    vecAdd<Type><<<grid, block>>>(d_in1, d_in2, d_out, len);
    cudaDeviceSynchronize();

    cudaMemcpy(val, d_out, len*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_vector<Type>::add");

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
}

#endif
