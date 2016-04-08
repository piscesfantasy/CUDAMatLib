#include "cuda_matrix.h"
#include <iostream>
#include <cstdlib>

using namespace std;

template <typename Type>
void simple_matrix_mul(const vector< vector<Type> > &x, Type **y, const int& dim1, const int& dim2, vector< vector<Type> > &z)
{
    for (int i=0; i<dim1; ++i){
        z.push_back(vector<Type>());
        for (int j=0; j<dim1; ++j)
            z[i].push_back(0);
    }

    for (int i=0; i<dim1; ++i)
        for (int j=0; j<dim1; ++j)
            for (int p=0; p<dim2; ++p)
                z[i][j]+=x[i][p]*y[p][j];
}

inline int get_rand_int() { return rand()%100; }

inline double get_rand_double() { return (double)rand()/10000; }

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
int test(const string& _type, const int& dim1, const int& dim2, Type (*get_rand)())
{
    vector< vector<Type> > mat_1;
    for (int i=0; i<dim1; ++i){
        mat_1.push_back(vector<Type>());
        for (int j=0; j<dim2; ++j)
            mat_1[i].push_back((*get_rand)());
    }

    Type **mat_2 = new Type*[dim2];
    for (int i=0; i<dim2; ++i){
        mat_2[i] = new Type[dim1];
        for (int j=0; j<dim1; ++j)
            mat_2[i][j] = (*get_rand)();
    }

    CUDA_matrix<Type> cuda_1(mat_1);
    CUDA_matrix<Type> cuda_2(mat_2, dim2, dim1);
    CUDA_matrix<Type> cuda_3;
    CUDA_matrix_multiply<Type>(cuda_1, cuda_2, cuda_3);

    vector< vector<Type> > mat_3;
    simple_matrix_mul(mat_1, mat_2, dim1, dim2, mat_3);
    for (int i=0; i<dim1; ++i)
        for (int j=0; j<dim1; ++j)
            compare(mat_3[i][j], cuda_3[i][j], _type, "CUDA_matrix_multiply");

    cout<<_type<<" test completed"<<endl;
}

int main()
{
    cout<<"Begin testing......"<<endl;
    int dim1 = 100;
    int dim2 = 1000;
    test<int>("int", dim1, dim2, get_rand_int);
    test<double>("double", dim1, dim2, get_rand_double);
    return 0;
}
