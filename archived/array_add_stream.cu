// Note: current editon of add_stream would cause segmentation fault
// upon cudaFree, might be due to failing in malloc
//void add_stream(CUDA_array<Type> const&);

template <typename Type>
void CUDA_array<Type>::add_stream(CUDA_array<Type> const &input)
{
    if (input.size()!=_len)
    {
        cerr<<"ERROR: can't add arrays with different length"<<endl;
        return;
    }

    Type *addend = input.getValue();
    Type *ans = new Type[_len];

    cudaStream_t stream0, stream1, stream2, stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    Type *d_A0, *d_B0, *d_C0;// device memory for stream 0
    Type *d_A1, *d_B1, *d_C1;// device memory for stream 1
    Type *d_A2, *d_B2, *d_C2;// device memory for stream 0
    Type *d_A3, *d_B3, *d_C3;// device memory for stream 1

    cudaMalloc((void**) &d_A0, _len*sizeof(Type));
    cudaMalloc((void**) &d_B0, _len*sizeof(Type));
    cudaMalloc((void**) &d_C0, _len*sizeof(Type));
    cudaMalloc((void**) &d_A1, _len*sizeof(Type));
    cudaMalloc((void**) &d_B1, _len*sizeof(Type));
    cudaMalloc((void**) &d_C1, _len*sizeof(Type));
    cudaMalloc((void**) &d_A2, _len*sizeof(Type));
    cudaMalloc((void**) &d_B2, _len*sizeof(Type));
    cudaMalloc((void**) &d_C2, _len*sizeof(Type));
    cudaMalloc((void**) &d_A3, _len*sizeof(Type));
    cudaMalloc((void**) &d_B3, _len*sizeof(Type));
    cudaMalloc((void**) &d_C3, _len*sizeof(Type));

    for (int bias=0; bias<_len; bias+=SEGSIZE*4)
    {
        cudaMemcpyAsync(d_A0, _val+bias, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, addend+bias, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream0);

        cudaMemcpyAsync(d_A1, _val+bias+SEGSIZE, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, addend+bias+SEGSIZE, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream1);
        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream0>>>(d_A0, d_B0, d_C0, _len);

        cudaMemcpyAsync(d_A2, _val+bias+SEGSIZE*2, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_B2, addend+bias+SEGSIZE*2, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream2);
        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream1>>>(d_A1, d_B1, d_C1, _len);
        cudaMemcpyAsync(ans+bias, d_C0, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(d_A3, _val+bias+SEGSIZE*3, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(d_B3, addend+bias+SEGSIZE*3, SEGSIZE*sizeof(Type), cudaMemcpyHostToDevice, stream3);
        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream2>>>(d_A2, d_B2, d_C2, _len);
        cudaMemcpyAsync(ans+bias+SEGSIZE, d_C1, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream1);

        vecAdd<<<SEGSIZE/BLKSIZE, BLKSIZE, 0, stream3>>>(d_A3, d_B3, d_C3, _len);
        cudaMemcpyAsync(ans+bias+SEGSIZE*2, d_C2, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream2);

        cudaMemcpyAsync(ans+bias+SEGSIZE*3, d_C3, SEGSIZE*sizeof(Type), cudaMemcpyDeviceToHost, stream3);
    }
    cudaDeviceSynchronize();

    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);
    cudaFree(d_A3);
    cudaFree(d_B3);
    cudaFree(d_C3);

    for (int i=0; i<_len; ++i)
    {
        cout<<i<<endl;
        _val[i] = ans[i];
    }

    delete [] addend;
    delete [] ans;
}
