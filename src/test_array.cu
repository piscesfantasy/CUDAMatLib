#include "cuda_array.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

template <typename Type>
void simple_add(Type *x, Type *y, int len)
{
    for (int i=0; i<len; ++i)
        x[i]+=y[i];
}

template <typename Type>
Type simple_inner_prod(Type *x, Type *y, int len)
{
    Type ans = 0;
    for (int i=0; i<len; ++i)
        ans+=x[i]*y[i];
    return ans;
}

template <typename Type>
Type simple_sum(Type *x, int len)
{
    Type sum = 0;
    for (int i=0; i<len; ++i)
        sum+=x[i];
    return sum;
}

template <typename Type>
void simple_cumulate(Type *x, int len)
{
    for (int i=1; i<len; ++i)
        x[i]+=x[i-1];
}

inline int get_rand_int() { return rand()%100; }

inline double get_rand_double() { return (double)rand()/1.0e7; }

template <typename Type>
void compare(const Type& val_cuda, const Type& val_simple, const string& _type, const string& func_name)
{
    if (abs((double)(val_cuda-val_simple))/abs(val_cuda+val_simple)>=1.0e-6)
    {
        cerr<<"["<<_type<<"] "<<func_name<<" result in different "<<abs((double)(val_cuda-val_simple))<<": \n"
            <<"\tCUDA:      "<<val_cuda<<"\n"
            <<"\treference: "<<val_simple<<endl;
    }
}

template <typename Type>
int test(const string& _type, const int& len, Type (*get_rand)())
{
    // Test constructors
    vector<Type> array_1;
    Type *array_2 = new Type[len];
    Type *array_3 = new Type[len];
    for (int i=0; i<len; ++i){
        array_1.push_back((*get_rand)());
        array_2[i] = get_rand();
        array_3[i] = array_1[i];
    }
    CUDA_array<Type> cuda_1(array_1);
    CUDA_array<Type> cuda_2(len);
    cuda_2.setValue(array_2);
    CUDA_array<Type> cuda_3(array_3, len);
    CUDA_array<Type> cuda_4(cuda_2); // not used, just for test

    // Test add
    cuda_1.add(cuda_2);
    simple_add(&array_1[0], array_2, len);
    for (int i=0; i<len; ++i)
        compare(cuda_1[i], array_1[i], _type, "CUDA_array::add");

    // Test sum
    Type sum1 = cuda_2.sum();
    Type sum2 = simple_sum(array_2, len);
    compare(sum1, sum2, _type, "CUDA_array::sum");

    // Test inner product
    Type inner_prod1 = cuda_2.inner_prod(cuda_3);
    Type inner_prod2 = simple_inner_prod(array_2, array_3, len);
    compare(inner_prod1, inner_prod2, _type, "CUDA_array::inner_prod");

    // Test cumulate (prefix sum)
    cuda_3.cumulate();
    simple_cumulate(array_3, len);
    for (int i=0; i<len; ++i)
        compare(cuda_3[i], array_3[i], _type, "CUDA_array::cumulate");

    delete [] array_2;
    delete [] array_3;

    cout<<_type<<" test completed"<<endl;
    return 0;
}

int main()
{
    cout<<"Begin testing......"<<endl;
    int len = 10000;
    srand(time(0));
    test<int>("int", len, get_rand_int);
    test<double>("double", len, get_rand_double);
    return 0;
}
