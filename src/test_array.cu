#include "cuda_array.h"
#include <iostream>
#include <cstdlib>

using namespace std;

void simple_add(int *x, int *y, int len){
    for (int i=0; i<len; ++i)
        x[i]+=y[i];
}

void simple_add(double *x, double *y, double len){
    for (int i=0; i<len; ++i)
        x[i]+=y[i];
}

int main()
{
    cout<<"Begin testing......"<<endl;
    int len = 10000;

    // int array
    vector<int> array_int_1;
    int *array_int_2 = new int[len];
    int *array_int_3 = new int[len];
    for (int i=0; i<len; ++i){
        array_int_1.push_back(rand()%100);
        array_int_2[i] = rand()%100;
        array_int_3[i] = array_int_1[i];
    }
    CUDA_array<int> cuda_int_1(array_int_1);
    CUDA_array<int> cuda_int_2(len);
    CUDA_array<int> cuda_int_3(array_int_3, len); // not used, just for test
    cuda_int_2.setValue(array_int_2);
    cuda_int_1.add(cuda_int_2);
    simple_add(array_int_3, array_int_2, len);
    for (int i=0; i<len; ++i)
    {
        if (cuda_int_1[i]!=array_int_3[i])
        {
            cout<<"[int] Different value: \n"
                <<"\tCUDA:      "<<cuda_int_1[i]<<"\n"
                <<"\treference: "<<array_int_3[i]<<endl;
        }
    }
    cout<<"Int test completed"<<endl;

    // double array
    vector<double> array_double_1;
    double *array_double_2 = new double[len];
    double *array_double_3 = new double[len];
    for (int i=0; i<len; ++i){
        array_double_1.push_back((double)rand()/10000);
        array_double_2[i] = (double)rand()/10000;
        array_double_3[i] = array_double_1[i];
    }
    CUDA_array<double> cuda_double_1(array_double_1);
    CUDA_array<double> cuda_double_2(len);
    CUDA_array<double> cuda_double_3(array_double_3, len); // not used, just for test
    cuda_double_2.setValue(array_double_2);
    cuda_double_1.add(cuda_double_2);
    simple_add(array_double_3, array_double_2, len);
    for (int i=0; i<len; ++i)
    {
        if (cuda_double_1[i]!=array_double_3[i])
        {
            cout<<"[double] Different value: \n"
                <<"\tCUDA:      "<<cuda_double_1[i]<<"\n"
                <<"\treference: "<<array_double_3[i]<<endl;
        }
    }
    cout<<"Double test completed"<<endl;

    return 0;
}
