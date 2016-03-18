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
Type simple_sum(Type *x, int len)
{
    Type sum = 0;
    for (int i=0; i<len; ++i)
        sum+=x[i];
    return sum;
}

inline int get_rand_int() { return rand()%100; }

inline double get_rand_double() { return (double)rand()/10000; }

template <typename Type>
int test(const string& _type, const int& len, Type (*get_rand)())
{
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
    CUDA_array<Type> cuda_3(array_3, len); // not used, just for test
    cuda_2.setValue(array_2);
    CUDA_array<Type> cuda_4(cuda_2); // not used, just for test

    cuda_1.add(cuda_2);
    simple_add(array_3, array_2, len);
    for (int i=0; i<len; ++i)
    {
        if (cuda_1[i]!=array_3[i])
        {
            cerr<<"["<<_type<<"] CUDA add() result in different values: \n"
                <<"\tCUDA:      "<<cuda_1[i]<<"\n"
                <<"\treference: "<<array_3[i]<<endl;
        }
    }
    /*cuda_4.add_stream(cuda_3);
    for (int i=0; i<len; ++i)
    {
        if (cuda_4[i]!=array_3[i])
        {
            cerr<<"[Type] CUDA add_stream() result in different values: \n"
                <<"\tCUDA:      "<<cuda_4[i]<<"\n"
                <<"\treference: "<<array_3[i]<<endl;
        }
    }*/

    Type sum1 = cuda_3.sum();
    Type sum2 = simple_sum(array_3, len);
    if (sum1 != sum2)
    {
        cerr<<"["<<_type<<"] CUDA sum() result in different values: \n"
            <<"\tCUDA:      "<<sum1<<"\n"
            <<"\treference: "<<sum2<<endl;
    }

    cout<<_type<<" test completed"<<endl;
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
