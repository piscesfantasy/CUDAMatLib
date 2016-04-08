#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "cuda_object.h"
#include "cuda_array_kernel.h"

using namespace std;

template <typename Type>
class CUDA_array : public CUDA_object<Type>
{
    public:
        CUDA_array(){}
        CUDA_array(const int& l);
        CUDA_array(Type* input, const int& l);
        CUDA_array(vector<Type>& input);
        CUDA_array(const CUDA_array<Type>& other);
        virtual ~CUDA_array();

        int size() const { return _len; }

        void resize(const int& l);
        void setValue(Type* input);
        void setValue(const vector<Type>& input);

        Type &operator[](const size_t idx){ return this->_val[idx]; }

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
};

template <typename Type>
CUDA_array<Type>::CUDA_array(const int& l) : _len(l)
{
    this->_val = new Type[_len];
    cudaMalloc((void**) &(this->cuda_val), _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::CUDA_array(Type* input, const int& l) : _len(l)
{
    this->_val = new Type[_len];
    setValue(input);
    cudaMalloc((void**) &(this->cuda_val), _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::CUDA_array(vector<Type>& input) : _len(input.size())
{
    this->_val = new Type[_len];
    setValue(input);
    cudaMalloc((void**) &(this->cuda_val), _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::CUDA_array(const CUDA_array<Type>& other)
{
    _len = other.size();
    this->_val = new Type[_len];
    setValue(other.getValue());
    cudaMalloc((void**) &(this->cuda_val), _len*sizeof(Type));
}

template <typename Type>
CUDA_array<Type>::~CUDA_array()
{
    if (this->_val!=NULL)
    {
        delete [] this->_val;
        cudaFree(this->cuda_val);
    }
}

template <typename Type>
void CUDA_array<Type>::resize(const int& l)
{
    _len = l;
    this->reset();
}

template <typename Type>
void CUDA_array<Type>::setValue(Type* input)
{
    for (int i=0; i<_len; ++i)
        this->_val[i] = input[i];
}

template <typename Type>
void CUDA_array<Type>::setValue(const vector<Type>& input)
{
    for (int i=0; i<_len; ++i)
        this->_val[i] = input[i];
}

template <typename Type>
void CUDA_array<Type>::add(CUDA_array<Type> const &input)
{
    if (input.size()!=_len)
    {
        cerr<<"ERROR: can't add arrays with different length"<<endl;
        return;
    }

    Type *d_out;

    cudaMalloc((void**) &d_out, _len*sizeof(Type));
    cudaMemset(d_out, 0, _len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::add");

    this->cudaMemcpyToDevice();
    input.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::add");

    dim3 grid((_len-1)/BLKSIZE+1, 1, 1);
    dim3 block(BLKSIZE, 1, 1);

    vecAdd<Type><<<grid, block>>>(this->cuda_val, input.getCUDAValue(), d_out, _len);
    cudaDeviceSynchronize();

    cudaMemcpy(this->_val, d_out, _len*sizeof(Type), cudaMemcpyDeviceToHost);	
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::add");

    cudaFree(d_out);
}

template <typename Type>
Type CUDA_array<Type>::inner_prod(CUDA_array<Type> const &input)
{
    if (input.size()!=_len)
    {
        cerr<<"ERROR: can't inner product arrays with different length"<<endl;
        return Type();
    }

    Type *d_out;
    Type *h_out = new Type[_len];

    cudaMalloc((void**) &d_out, _len*sizeof(Type));
    cudaMemset(d_out, 0, _len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::inner_prod");

    this->cudaMemcpyToDevice();
    input.cudaMemcpyToDevice();
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::inner_prod");

    dim3 grid((_len-1)/BLKSIZE+1, 1, 1);
    dim3 block(BLKSIZE, 1, 1);

    vecProd<Type><<<grid, block>>>(this->cuda_val, input.getCUDAValue(), d_out, _len);
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

    cudaMemcpy(this->cuda_val, this->_val, _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::sum");

    dim3 block(BLKSIZE, 1, 1);
    dim3 grid(reduced_len, 1, 1);
    total<<<grid, block>>>(this->cuda_val, d_tmp, _len);
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
    cudaMemcpy(this->cuda_val, this->_val, _len*sizeof(Type), cudaMemcpyHostToDevice);

    dim3 block(BLKSIZE, 1, 1);
    dim3 grid((_len-1)/(BLKSIZE*2)+1, 1, 1);

    cudaMemset(d_tmp, 0, _len*sizeof(Type));
    scan<<<grid, block>>>(this->cuda_val, d_tmp, _len);
    cudaDeviceSynchronize();

    cudaMemset(this->cuda_val, 0, _len*sizeof(Type));
    offset<<<grid, block>>>(d_tmp, this->cuda_val, _len);
    cudaDeviceSynchronize();
	
    cudaMemcpy(this->_val, this->cuda_val, _len*sizeof(Type), cudaMemcpyDeviceToHost);
	
    cudaFree(d_tmp);
}

/*
template <typename Type>
void CUDA_array<Type>::add_stream(CUDA_array<Type> const &input)
{
   if (input.size()!=_len)
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
