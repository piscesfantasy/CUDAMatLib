#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "cuda_array_kernel.h"
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
class CUDA_array
{
    public:
        CUDA_array(const int& l) : _len(l)
        {
            _val = new Type[_len];
        }

        CUDA_array(Type* data, const int& l) : _len(l)
        {
            _val = new Type[_len];
            setValue(data);
        }

        CUDA_array(vector<Type>& data) : _len(data.size())
        {
            _val = new Type[_len];
            setValue(&data[0]);
        }

        virtual ~CUDA_array()
        {
            if (_val!=NULL)
                delete [] _val;
        }

        Type &operator[](const size_t idx){ return _val[idx]; }

        int len() const{ return _len; }

        Type* getValue() const{ return _val; }

        void setValue(Type* data)
        {
            for (int i=0; i<_len; ++i)
                _val[i] = data[i];
        }

        int add( CUDA_array<Type> const&);
        int add_stream( CUDA_array<Type> const&);

    private:
        Type* _val;
        int _len;
};

template <typename Type>
int CUDA_array<Type>::add(CUDA_array<Type> const &input)
{
    if (input.len()!=_len)
    {
        cerr<<"ERROR: can't add arrays with different length"<<endl;
        return 1;
    }

    Type *d_in1, *d_in2, *d_out;

    cudaMalloc((void**) &d_in1, _len*sizeof(Type));
    cudaMalloc((void**) &d_in2, _len*sizeof(Type));
    cudaMalloc((void**) &d_out, _len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::add");

    cudaMemcpy(d_in1, _val, _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, input.getValue(), _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::add");

    dim3 grid((_len-1)/BLKSIZE+1, 1, 1);
    dim3 block(BLKSIZE, 1, 1);

    vecAdd<Type><<<grid, block>>>(d_in1, d_in2, d_out, _len);
    cudaDeviceSynchronize();

    cudaMemcpy(_val, d_out, _len*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::add");

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    return 0;
}

template <typename Type>
int CUDA_array<Type>::add_stream(CUDA_array<Type> const &input)
{
    if (input.len()!=_len)
    {
        cerr<<"ERROR: can't add arrays with different length"<<endl;
        return 1;
    }

    Type *addend = input.getValue();
    Type *ans = (Type *) malloc(_len * sizeof(Type));

    cudaStream_t stream0, stream1, stream2, stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    Type *d_A0, *d_B0, *d_C0;// device memory for stream 0
    Type *d_A1, *d_B1, *d_C1;// device memory for stream 1
    Type *d_A2, *d_B2, *d_C2;// device memory for stream 0
    Type *d_A3, *d_B3, *d_C3;// device memory for stream 1

    cudaMalloc((void**) &d_A0, _len*sizeof(Type));
    cudaMalloc((void**) &d_B0, _len*sizeof(Type));
    cudaMalloc((void**) &d_C0, _len*sizeof(Type));
    cudaMalloc((void**) &d_A1, _len*sizeof(Type));
    cudaMalloc((void**) &d_B1, _len*sizeof(Type));
    cudaMalloc((void**) &d_C1, _len*sizeof(Type));
    cudaMalloc((void**) &d_A2, _len*sizeof(Type));
    cudaMalloc((void**) &d_B2, _len*sizeof(Type));
    cudaMalloc((void**) &d_C2, _len*sizeof(Type));
    cudaMalloc((void**) &d_A3, _len*sizeof(Type));
    cudaMalloc((void**) &d_B3, _len*sizeof(Type));
    cudaMalloc((void**) &d_C3, _len*sizeof(Type));

    for (int bias=0; bias<_len; bias+=SEGSIZE*4)
    {
        cudaMemcpyAsync(d_A0, _val+bias, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, addend+bias, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream0);

        cudaMemcpyAsync(d_A1, _val+bias+SEGSIZE, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, addend+bias+SEGSIZE, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream1);
        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream0>>>(d_A0, d_B0, d_C0, _len);

        cudaMemcpyAsync(d_A2, _val+bias+SEGSIZE*2, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_B2, addend+bias+SEGSIZE*2, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream2);
        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream1>>>(d_A1, d_B1, d_C1, _len);
        cudaMemcpyAsync(ans+bias, d_C0, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(d_A3, _val+bias+SEGSIZE*3, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(d_B3, addend+bias+SEGSIZE*3, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream3);
        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream2>>>(d_A2, d_B2, d_C2, _len);
        cudaMemcpyAsync(ans+bias+SEGSIZE, d_C1, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream1);

        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream3>>>(d_A3, d_B3, d_C3, _len);
        cudaMemcpyAsync(ans+bias+SEGSIZE*2, d_C2, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream2);

        cudaMemcpyAsync(ans+bias+SEGSIZE*3, d_C3, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream3);
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

    for (int i=0; i<_len; ++i)
        _val[i] = ans[i];

    delete [] addend;
    delete [] ans;
    return 0;
}

#endif
