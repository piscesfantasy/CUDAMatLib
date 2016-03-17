#include "cuda_matrix.h"
#include <iostream>
#include <cstdlib>

using namespace std;

int main()
{
    cout<<"Begin testing......"<<endl;
    int size = 10000;
    int width = 100;

    vector< vector<int> > mat_int_1;
    int **mat_int_2 = new int*[width];
    for (int i=0; i<width; ++i){
        mat_int_1.push_back(vector<int>());
        mat_int_2[i] = new int[width];
        for (int j=0; j<width; ++j){
            mat_int_1[i].push_back(rand()%100);
            mat_int_2[i][j] = rand()%100;
        }
    }
    CUDA_matrix<int> cuda_int_1(mat_int_1);
    CUDA_matrix<int> cuda_int_2(mat_int_2, width, width);
    CUDA_matrix<int> cuda_int_3;
    CUDA_matrix_multiply<int>(cuda_int_1, cuda_int_2, cuda_int_3);
    cout<<"Int test completed"<<endl;

    // double array
    vector< vector<double> > mat_double_1;
    double **mat_double_2 = new double*[width];
    for (int i=0; i<width; ++i){
        mat_double_1.push_back(vector<double>());
        mat_double_2[i] = new double[width];
        for (int j=0; j<width; ++j){
            mat_double_1[i].push_back((double)rand()/10000);
            mat_double_2[i][j] = (double)rand()/10000;
        }
    }
    CUDA_matrix<double> cuda_double_1(mat_double_1);
    CUDA_matrix<double> cuda_double_2(mat_double_2, width, width);
    CUDA_matrix<double> cuda_double_3;
    CUDA_matrix_multiply<double>(cuda_double_1, cuda_double_2, cuda_double_3);
    cout<<"Double test completed"<<endl;

    return 0;
}
