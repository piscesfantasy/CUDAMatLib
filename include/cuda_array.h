#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "cuda_array_kernel.h"
#include <stdio.h>
#include <iostream>
#include <vector>

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

/**
 * CUDA_array allocates both host and device memories upon construction. But
 * cudaMemcpy is performed right before calculation.
 * Pros:
 *  - In operation intense cases, save time allocating new device memories.
 *  - Can still overwrite values between operations.
 * Cons:
 *  - Don't keep too many CUDA_array objects alive, otherwise would cause
 *    device memory starvation.
*/
template <typename Type>
class CUDA_array
{
    public:
        CUDA_array(const int& l);
        CUDA_array(Type* data, const int& l);
        CUDA_array(vector<Type>& data);
        CUDA_array(const CUDA_array<Type>& other);
        virtual ~CUDA_array();

        Type &operator[](const size_t idx){ return _val[idx]; }
        int len() const { return _len; }
        Type* getValue() const { return _val; }
        Type* getCUDAValue() const { return cuda_val; }
        void setValue(Type* data);
        void cudaMemcpyToDevice() const { 
            cudaMemcpy(cuda_val, _val, _len*sizeof(Type), cudaMemcpyHostToDevice);
        }

        // vector addition
        virtual void add(CUDA_array<Type> const&);

        // Note: current editon of add_stream would cause segmentation fault
        // upon cudaFree, might be due to failing in malloc
        //void add_stream(CUDA_array<Type> const&);

        // vector dot product
        virtual Type inner_prod(CUDA_array<Type> const&);

        // calculate total
        virtual Type sum();

        // calculate max
        //virtual Type max();

        // cumulative summation
        virtual void cumulate();

        // convolution
        //void convolve(const Type* mask, const int& m_len);

    private:
        int _len;
        Type *_val; // local memory
        Type *cuda_val; // cuda memory
};

template <typename Type>
CUDA_array<Type>::CUDA_array(const int& l) : _len(l)
{
    _val = new Type[_len];
    cudaMalloc((void**) &cuda_val, _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::CUDA_array(Type* data, const int& l) : _len(l)
{
    _val = new Type[_len];
    setValue(data);
    cudaMalloc((void**) &cuda_val, _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::CUDA_array(vector<Type>& data) : _len(data.size())
{
    _val = new Type[_len];
    setValue(&data[0]);
    cudaMalloc((void**) &cuda_val, _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::CUDA_array(const CUDA_array<Type>& other)
{
    _len = other.len();
    _val = new Type[_len];
    setValue(other.getValue());
    cudaMalloc((void**) &cuda_val, _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::~CUDA_array()
{
    if (_val!=NULL)
    {
        delete [] _val;
        cudaFree(cuda_val);
    }
}

template <typename Type>
void CUDA_array<Type>::setValue(Type* data)
{
    for (int i=0; i<_len; ++i)
        _val[i] = data[i];
}

template <typename Type>
void CUDA_array<Type>::add(CUDA_array<Type> const &input)
{
    if (input.len()!=_len)
    {
        cerr<<"ERROR: can't add arrays with different length"<<endl;
        return;
    }

    Type *d_out;

    cudaMalloc((void**) &d_out, _len*sizeof(Type));
    cudaMemset(d_out, 0, _len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::add");

    cudaMemcpyToDevice();
    input.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::add");

    dim3 grid((_len-1)/BLKSIZE+1, 1, 1);
    dim3 block(BLKSIZE, 1, 1);

    vecAdd<Type><<<grid, block>>>(cuda_val, input.getCUDAValue(), d_out, _len);
    cudaDeviceSynchronize();

    cudaMemcpy(_val, d_out, _len*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::add");

    cudaFree(d_out);
}

template <typename Type>
Type CUDA_array<Type>::inner_prod(CUDA_array<Type> const &input)
{
    if (input.len()!=_len)
    {
        cerr<<"ERROR: can't inner product arrays with different length"<<endl;
        return Type();
    }

    Type *d_out;
    Type *h_out = new Type[_len];

    cudaMalloc((void**) &d_out, _len*sizeof(Type));
    cudaMemset(d_out, 0, _len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::inner_prod");

    cudaMemcpyToDevice();
    input.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::inner_prod");

    dim3 grid((_len-1)/BLKSIZE+1, 1, 1);
    dim3 block(BLKSIZE, 1, 1);

    vecProd<Type><<<grid, block>>>(cuda_val, input.getCUDAValue(), d_out, _len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, _len*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::inner_prod");

    Type ans = CUDA_array<Type>(h_out, _len).sum();

    cudaFree(d_out);
    delete [] h_out;

    return ans;
}

template <typename Type>
Type CUDA_array<Type>::sum()
{
    Type *d_tmp;
    int reduced_len = _len / (BLKSIZE<<1);
    if (_len % (BLKSIZE<<1))
        reduced_len+=1;
    Type *reduced_val = new Type[reduced_len];

    cudaMalloc((void **) &d_tmp, reduced_len*sizeof(Type));
    cudaMemset(d_tmp, 0, reduced_len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::sum");

    cudaMemcpy(cuda_val, _val, _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::sum");

    dim3 block(BLKSIZE, 1, 1);
    dim3 grid(reduced_len, 1, 1);
    total<<<grid, block>>>(cuda_val, d_tmp, _len);
    cudaDeviceSynchronize();

    cudaMemcpy(reduced_val, d_tmp, reduced_len*sizeof(Type), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::sum");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input.
     ********************************************************************/
    for (int i=1; i<reduced_len; ++i)
        reduced_val[0]+=reduced_val[i];
    Type ans = reduced_val[0];

    cudaFree(d_tmp);
    delete [] reduced_val;

    return ans;
}

template <typename Type>
void CUDA_array<Type>::cumulate()
{
    Type *d_tmp;

    cudaMalloc((void**)&d_tmp, _len*sizeof(Type));
    cudaMemcpy(cuda_val, _val, _len*sizeof(Type), cudaMemcpyHostToDevice);

    dim3 block(BLKSIZE, 1, 1);
    dim3 grid((_len-1)/(BLKSIZE*2)+1, 1, 1);

    cudaMemset(d_tmp, 0, _len*sizeof(Type));
    scan<<<grid, block>>>(cuda_val, d_tmp, _len);
    cudaDeviceSynchronize();

    cudaMemset(cuda_val, 0, _len*sizeof(Type));
    offset<<<grid, block>>>(d_tmp, cuda_val, _len);
    cudaDeviceSynchronize();
	
    cudaMemcpy(_val, cuda_val, _len*sizeof(Type), cudaMemcpyDeviceToHost);
	
    cudaFree(d_tmp);
}

/*
template <typename Type>
void CUDA_array<Type>::add_stream(CUDA_array<Type> const &input)
{
   if (input.len()!=_len)
   {
   cerr<<"ERROR: can't add arrays with different length"<<endl;
   return;
   }

   Type *addend = input.getValue();
   Type *ans = new Type[_len];

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
{
    cout<<i<<endl;
    _val[i] = ans[i];
}

delete [] addend;
delete [] ans;
}
*/

#endif
