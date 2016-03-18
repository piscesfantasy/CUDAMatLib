#include "cuda_matrix.h"
#include <iostream>
#include <cstdlib>

using namespace std;

inline int get_rand_int() { return rand()%100; }

inline double get_rand_double() { return (double)rand()/10000; }

template <typename Type>
int test(const string& _type, const int& size, const int& width, Type (*get_rand)())
{
    vector< vector<Type> > mat_1;
    Type **mat_2 = new Type*[width];
    for (int i=0; i<width; ++i){
        mat_1.push_back(vector<Type>());
        mat_2[i] = new Type[width];
        for (int j=0; j<width; ++j){
            mat_1[i].push_back((*get_rand)());
            mat_2[i][j] = (*get_rand)();
        }
    }
    CUDA_matrix<Type> cuda_1(mat_1);
    CUDA_matrix<Type> cuda_2(mat_2, width, width);
    CUDA_matrix<Type> cuda_3;
    CUDA_matrix_multiply<Type>(cuda_1, cuda_2, cuda_3);
    cout<<_type<<" test completed"<<endl;
}

int main()
{
    cout<<"Begin testing......"<<endl;
    int size = 10000;
    int width = 100;
    test<int>("int", size, width, get_rand_int);
    test<double>("double", size, width, get_rand_double);
    return 0;
}
