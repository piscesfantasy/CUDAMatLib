#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "cuda_object.h"
#include "cuda_array_kernel.h"

using namespace std;

template <typename Type>
class CUDA_array : public CUDA_object<Type>
{
    public:
        // Constructor/Destructor
        CUDA_array(){}
        CUDA_array(const int& l);
        CUDA_array(Type* input, const int& l);
        CUDA_array(vector<Type>& input);
        CUDA_array(const CUDA_array<Type>& other);
        virtual ~CUDA_array();

        // Set/Get
        virtual int size() const { return _len; }
        virtual void resize(const int& l);
        virtual void setValue(Type* input);
        virtual void setValue(const vector<Type>& input);
        virtual Type &operator[](const size_t idx){ return this->_val[idx]; }

        // Operations
        virtual void add(CUDA_array<Type> const&);
        virtual Type inner_prod(CUDA_array<Type> const&);
        virtual Type sum();
        virtual Type min();
        virtual Type max();
        virtual void cumulate();
        //virtual void convolve(const Type* mask, const int& m_len);

    protected:
        // Template method for all reduce operations (sum, min, max et. al.)
        virtual Type reduce(void (*f)(Type*, Type*, int));

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

    dim3 grid((_len-1)/BLKSIZE_1D+1, 1, 1);
    dim3 block(BLKSIZE_1D, 1, 1);

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

    dim3 grid((_len-1)/BLKSIZE_1D+1, 1, 1);
    dim3 block(BLKSIZE_1D, 1, 1);

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
    CUDA_array<Type>::reduce(total);
}

template <typename Type>
Type CUDA_array<Type>::min()
{
    CUDA_array<Type>::reduce(getMin);
}

template <typename Type>
Type CUDA_array<Type>::max()
{
    CUDA_array<Type>::reduce(getMax);
}

template <typename Type>
Type CUDA_array<Type>::reduce(void (*reducer)(Type*, Type*, int))
{
    Type *d_tmp1, *d_tmp2;
    int reduced_len = _len / (BLKSIZE_1D<<1);
    if (_len % (BLKSIZE_1D<<1))
        reduced_len+=1;

    cudaMalloc((void **) &d_tmp1, _len*sizeof(Type));
    cudaMemset(d_tmp1, 0, _len*sizeof(Type));
    cudaMalloc((void **) &d_tmp2, _len*sizeof(Type));
    cudaMemset(d_tmp2, 0, _len*sizeof(Type));
    cudaCheckErrors("cudaMalloc @ CUDA_array<Type>::sum");

    dim3 block(BLKSIZE_1D, 1, 1);
    dim3 grid(reduced_len, 1, 1);

    // 1st reduce
    cudaMemcpy(this->cuda_val, this->_val, _len*sizeof(Type), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyHostToDevice @ CUDA_array<Type>::sum");
    reducer<<<grid, block>>>(this->cuda_val, d_tmp1, _len);
    cudaDeviceSynchronize();

    // iteratively reduce until length=1
    while (reduced_len>1)
    {
        reducer<<<grid, block>>>(d_tmp1, d_tmp2, reduced_len);
        cudaDeviceSynchronize();
        if (reduced_len==1)
        {
            d_tmp1 = d_tmp2;
            break;
        }
        reduced_len /= (BLKSIZE_1D<<1);
        if (_len % (BLKSIZE_1D<<1))
            reduced_len+=1;
        reducer<<<grid, block>>>(d_tmp2, d_tmp1, reduced_len);
        cudaDeviceSynchronize();
        reduced_len /= (BLKSIZE_1D<<1);
        if (_len % (BLKSIZE_1D<<1))
            reduced_len+=1;
    }

    Type *_ans = new Type[1];
    cudaMemcpy(_ans, d_tmp1, sizeof(Type), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyDeviceToHost @ CUDA_array<Type>::sum");
    Type ans = _ans[0];

    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    delete [] _ans;

    return ans;
}

template <typename Type>
void CUDA_array<Type>::cumulate()
{
    Type *d_tmp;

    cudaMalloc((void**)&d_tmp, _len*sizeof(Type));
    cudaMemcpy(this->cuda_val, this->_val, _len*sizeof(Type), cudaMemcpyHostToDevice);

    dim3 block(BLKSIZE_1D, 1, 1);
    dim3 grid((_len-1)/(BLKSIZE_1D*2)+1, 1, 1);

    cudaMemset(d_tmp, 0, _len*sizeof(Type));
    scan<<<grid, block>>>(this->cuda_val, d_tmp, _len);
    cudaDeviceSynchronize();

    cudaMemset(this->cuda_val, 0, _len*sizeof(Type));
    offset<<<grid, block>>>(d_tmp, this->cuda_val, _len);
    cudaDeviceSynchronize();
	
    cudaMemcpy(this->_val, this->cuda_val, _len*sizeof(Type), cudaMemcpyDeviceToHost);
	
    cudaFree(d_tmp);
}

#endif
