#include "cuda_vector.h"
#include <iostream>

using namespace std;

int main()
{
    int i1[3] = {1,2,3};
    int i2[3] = {4,40,400};
    double d1[3] = {1.1,2.2,3.3};
    double d2[3] = {11,22,33};
    CUDA_vector<int> tmpi1(i1, 3);
    CUDA_vector<double> tmpd1(d1, 3);
    CUDA_vector<int> tmpi2(i2, 3);
    CUDA_vector<double> tmpd2(d2, 3);
    tmpi1.add(tmpi2);
    tmpd1.add(tmpd2);
    cout<<tmpi1[0]<<" "<<tmpi1[1]<<" "<<tmpi1[2]<<endl;
    cout<<tmpd1[0]<<" "<<tmpd1[1]<<" "<<tmpd1[2]<<endl;
    return 0;
}
