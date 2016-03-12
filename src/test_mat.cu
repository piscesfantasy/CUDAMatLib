#include "cuda_matrix.h"
#include <iostream>

using namespace std;

int main()
{
    vector< vector<int> > mat1;
    int **mat2;
    CUDA_array<int> tmpi1(mat1);
    CUDA_array<int> tmpi2(mat2, 3, 3);

    vector< vector<double> > mat1;
    double **mat2;
    CUDA_array<double> tmpi1(mat1);
    CUDA_array<double> tmpi2(mat2, 3, 3);

    tmpi1.add(tmpi2);
    tmpd1.add(tmpd2);
    cout<<tmpi1[0]<<" "<<tmpi1[1]<<" "<<tmpi1[2]<<endl;
    cout<<tmpd1[0]<<" "<<tmpd1[1]<<" "<<tmpd1[2]<<endl;
    return 0;
}
