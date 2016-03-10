#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

class CUDA_vector
{
    public:
        CUDA_vector(){}
        virtual ~CUDA_vector(){}

        void add(const CUDA_vector&);
        double getSum();
        void getCdf(double*);

    private:
        double* v;
        int len;
};

#endif
