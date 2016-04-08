#ifndef CUDA_OBJECT_H
#define CUDA_OBJECT_H

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
 * CUDA_object allocates both host and device memories upon construction.
 * ( SUGGESTION: call cudaMemcpyToDevice() right before running any CUDA device
 *               calculation in derived classes!!! )
 * Pros:
 *  - In operation intense cases, save time allocating new device memories.
 *  - Can still overwrite values between operations.
 * Cons:
 *  - Don't keep too many CUDA_array objects alive, otherwise would cause
 *    device memory starvation.
*/
template <typename Type>
class CUDA_object
{
    public:
        CUDA_object() {
            _val = NULL;
            cuda_val = NULL;
        }
        virtual ~CUDA_object(){};

        virtual void reset() {
            if (_val!=NULL)
            {
                delete [] _val;
                cudaFree(cuda_val);
            }
            _val = new Type[size()];
            cudaMalloc((void**) &cuda_val, size()*sizeof(Type));
        }

        virtual int size() const = 0;

        virtual Type* getValue() const { return _val; }

        virtual void cudaMemcpyToDevice() const { 
            cudaMemcpy(cuda_val, _val, size()*sizeof(Type), cudaMemcpyHostToDevice);
        }
        virtual Type* getCUDAValue() const { return cuda_val; }

    protected:
        Type *_val; // local memory
        Type *cuda_val; // cuda memory
};

#endif
