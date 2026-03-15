#include <iostream>
#include <vector>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// COPY KERNEL (COALESCED MEMORY ACCESS)
////////////////////////////////////////////////////////////

__global__ void copy_kernel(float* A, float* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
        B[i] = A[i];
}

////////////////////////////////////////////////////////////
// STRIDED MEMORY ACCESS (BAD COALESCING)
////////////////////////////////////////////////////////////

__global__ void strided_kernel(float* A, float* B, int N, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = i * stride;

    if(idx < N)
        B[idx] = A[idx];
}

////////////////////////////////////////////////////////////
// COMPUTE TEST
////////////////////////////////////////////////////////////

__global__ void compute_kernel(float* A, float* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        float val = A[i];

        for(int k = 0; k < 100; k++)
            val = val * 1.1f + 0.5f;

        B[i] = val;
    }
}

////////////////////////////////////////////////////////////
// BANDWIDTH TEST
////////////////////////////////////////////////////////////

float run_bandwidth(float *A, float *B, int N)
{
    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    copy_kernel<<<grid, block>>>(A, B, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float bytes = (float)N * sizeof(float) * 2;
    float gb = bytes / 1e9;

    return gb / (ms / 1000.0f);
}

////////////////////////////////////////////////////////////
// STRIDED TEST
////////////////////////////////////////////////////////////

float run_strided(float *A, float *B, int N)
{
    int stride = 4;

    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    strided_kernel<<<grid, block>>>(A, B, N, stride);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float bytes = (float)N * sizeof(float) * 2;
    float gb = bytes / 1e9;

    return gb / (ms / 1000.0f);
}

////////////////////////////////////////////////////////////
// COMPUTE TEST
////////////////////////////////////////////////////////////

float run_compute(float *A, float *B, int N)
{
    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<grid, block>>>(A, B, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{
    int N = 1 << 26;

    // CPU memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    for(int i = 0; i < N; i++)
        h_A[i] = 1.0f;

    // GPU memory
    float *d_A, *d_B;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));

    // copy CPU -> GPU
    cudaMemcpy(d_A, h_A.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Running GPU Microbenchmarks\n\n";

    float bw1 = run_bandwidth(d_A, d_B, N);
    std::cout << "Coalesced bandwidth: " << bw1 << " GB/s\n";

    float bw2 = run_strided(d_A, d_B, N);
    std::cout << "Strided bandwidth: " << bw2 << " GB/s\n";

    float compute_time = run_compute(d_A, d_B, N);
    std::cout << "Compute kernel time: " << compute_time << " ms\n";

    // copy GPU -> CPU
    cudaMemcpy(h_B.data(), d_B, N*sizeof(float), cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}