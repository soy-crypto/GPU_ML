#include <iostream>
#include <cuda_runtime.h>

__global__
void copy_kernel(float* A,float* B,int N)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<N)
        B[i]=A[i];
}

int main()
{
    int N = 1<<26;

    float *A,*B;

    cudaMallocManaged(&A,N*sizeof(float));
    cudaMallocManaged(&B,N*sizeof(float));

    for(int i=0;i<N;i++)
        A[i]=1;

    cudaEvent_t start,stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    copy_kernel<<<(N + 255) / 256,256>>>(A, B, N);

    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float ms;

    cudaEventElapsedTime(&ms,start,stop);

    float gb = (float)N*sizeof(float)*2/1e9;

    std::cout<<"Bandwidth " <<gb/(ms/1000) <<" GB/s\n";

    cudaFree(A);
    cudaFree(B);
}