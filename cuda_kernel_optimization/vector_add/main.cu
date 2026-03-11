#include <iostream>

__global__
void vecAdd(float* A,float* B,float* C,int N)
{
    int i= blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        C[i]=A[i]+B[i];
    }

}

int main()
{
    int N = 1 << 20;
    float *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));
    for(int i = 0; i < N; i++)
    {
        A[i] = 1;
        B[i] = 2;
    }

    vecAdd<<<(N + 255) / 256, 256>>>(A, B, C, N);
    cudaDeviceSynchronize();
    std::cout<<C[0]<<"\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    //Return
    return;
}