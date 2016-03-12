#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "cuda_array_kernel.h"
#include <stdio.h>

#define SegSize 256

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
        CUDA_array(){}
        CUDA_array(Type* data, const int& l)
        {
            setValue(data, l);
        }
        CUDA_array(const vector<Type>& data)
        {
            setValue(&data[0], data.size());
        }
        virtual ~CUDA_array() {if (_val!=NULL) delete [] _val;}

        Type &operator[](const size_t idx){ return _val[idx]; }
        size_type len() const{ return _len; }
        Type* getValue() const{ return __val; }
        void setValue(Type* data, const size_type& l)
        {
            _len = l;
            _val = new Type[_len];
            for (int i=0; i<_len; ++i)
                _val[i] = data[i];
        }

        void add( CUDA_array<Type> const&);
        void add_stream( CUDA_array<Type> const&);

    private:
        Type* _val;
        size_type _len;
};

template <typename Type>
void CUDA_array<Type>::add(CUDA_array<Type> const &input)
{
    if (input.len()!=_len)
        return;

    Type *d_in1, *d_in2, *d_out;

    cudaMalloc((void**) &d_in1, len*sizeof(Type));
    cudaMalloc((void**) &d_in2, len*sizeof(Type));
    cudaMalloc((void**) &d_out, len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::add");

    cudaMemcpy(d_in1, _val, _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, input.getValue(), _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::add");

    dim3 grid((_len-1)/8+1, 1, 1);
    dim3 block(8, 1, 1);

    vecAdd<Type><<<grid, block>>>(d_in1, d_in2, d_out, _len);
    cudaDeviceSynchronize();

    cudaMemcpy(_val, d_out, _len*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::add");

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
}

template <typename Type>
void CUDA_array<Type>::add_stream(CUDA_array<Type> const &input)
{
    if (input.len()!=_len)
        return;

    Type *addend = input.getValue();
    hostOutput = (Type *) malloc(_len * sizeof(Type));
	
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
    
    for (int bias=0; bias<_len; bias+=SegSize*4)
    {
        cudaMemcpyAsync(d_A0, _val+bias, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, addend+bias, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream0);

        cudaMemcpyAsync(d_A1, _val+bias+SegSize, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, addend+bias+SegSize, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream1);
        vecAdd<<<SegSize/16, 16, 0, stream0>>>(d_A0, d_B0, d_C0, _len);

        cudaMemcpyAsync(d_A2, _val+bias+SegSize*2, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_B2, addend+bias+SegSize*2, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream2);
        vecAdd<<<SegSize/16, 16, 0, stream1>>>(d_A1, d_B1, d_C1, _len);
        cudaMemcpyAsync(hostOutput+bias, d_C0, SegSize*sizeof(Type), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(d_A3, _val+bias+SegSize*3, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(d_B3, addend+bias+SegSize*3, SegSize*sizeof(Type), cudaMemcpyHostToDevice, stream3);
        vecAdd<<<SegSize/16, 16, 0, stream2>>>(d_A2, d_B2, d_C2, _len);
        cudaMemcpyAsync(hostOutput+bias+SegSize, d_C1, SegSize*sizeof(Type), cudaMemcpyDeviceToHost, stream1);

        vecAdd<<<SegSize/16, 16, 0, stream3>>>(d_A3, d_B3, d_C3, _len);
        cudaMemcpyAsync(hostOutput+bias+SegSize*2, d_C2, SegSize*sizeof(Type), cudaMemcpyDeviceToHost, stream2);

        cudaMemcpyAsync(hostOutput+bias+SegSize*3, d_C3, SegSize*sizeof(Type), cudaMemcpyDeviceToHost, stream3);
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

    return 0;
}

#endif
