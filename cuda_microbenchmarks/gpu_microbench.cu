#include <iostream>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// BANDWIDTH TEST (COPY KERNEL)
////////////////////////////////////////////////////////////

__global__
void copy_kernel(float* A, float* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        B[i] = A[i];
}

////////////////////////////////////////////////////////////
// STRIDED MEMORY ACCESS (BAD COALESCING)
////////////////////////////////////////////////////////////

__global__
void strided_kernel(float* A, float* B, int N, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = i * stride;

    if (idx < N)
        B[idx] = A[idx];
}

////////////////////////////////////////////////////////////
// COMPUTE / OCCUPANCY TEST
////////////////////////////////////////////////////////////

__global__
void compute_kernel(float* A, float* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        float val = A[i];

        for(int k = 0; k < 100; k++)
        {
            val = val * 1.1f + 0.5f;
        }

        B[i] = val;
    }
}

////////////////////////////////////////////////////////////
// BANDWIDTH CALCULATION
////////////////////////////////////////////////////////////

float run_bandwidth(float *A, float *B, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    copy_kernel<<<(N + 255) / 256, 256>>>(A, B, N);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float gb = (float)N * sizeof(float) * 2 / 1e9;

    return gb / (ms / 1000);
}

////////////////////////////////////////////////////////////
// STRIDED TEST
////////////////////////////////////////////////////////////

float run_strided(float *A, float *B, int N)
{
    int stride = 4;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    strided_kernel<<<(N + 255) / 256, 256>>>(A, B, N, stride);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float gb = (float)N * sizeof(float) * 2 / 1e9;

    return gb / (ms / 1000);
}

////////////////////////////////////////////////////////////
// COMPUTE TEST
////////////////////////////////////////////////////////////

float run_compute(float *A, float *B, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<(N + 255) / 256, 256>>>(A, B, N);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms;
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{
    int N = 1 << 26;

    float *A, *B;

    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));

    for(int i = 0; i < N; i++)
        A[i] = 1.0f;

    std::cout << "Running GPU Microbenchmarks\n\n";

    float bw1 = run_bandwidth(A, B, N);
    std::cout << "Coalesced bandwidth: " << bw1 << " GB/s\n";

    float bw2 = run_strided(A, B, N);
    std::cout << "Strided bandwidth: " << bw2 << " GB/s\n";

    float compute_time = run_compute(A, B, N);
    std::cout << "Compute kernel time: " << compute_time << " ms\n";

    cudaFree(A);
    cudaFree(B);

    return 0;
}