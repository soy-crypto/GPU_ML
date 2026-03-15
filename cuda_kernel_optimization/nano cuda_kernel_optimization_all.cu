#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define TILE 16

////////////////////////////////////////////////////////////
// CPU GEMM
////////////////////////////////////////////////////////////

void cpu_gemm(const float* A, const float* B, float* C, int N)
{
    for(int i = 0;i < N; ++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for(int k = 0;k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;

        }

    }

}

////////////////////////////////////////////////////////////
// CUDA VECTOR ADD
////////////////////////////////////////////////////////////

__global__ void vector_add_kernel(float* A, float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        C[i] = A[i] + B[i];
    }

}

////////////////////////////////////////////////////////////
// CUDA NAIVE GEMM
////////////////////////////////////////////////////////////
__global__ void gemm_naive_kernel(float* A,float* B,float* C,int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float sum = 0.0f;

        for(int k = 0;k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

////////////////////////////////////////////////////////////
// CUDA TILED GEMM
/*
for each tile pair (t):
    load A_tile
    load B_tile
    synchronize threads
    multiply row of A_tile × column of B_tile
    add to sum
*/

////////////////////////////////////////////////////////////

__global__ void gemm_tiled_kernel(float* A, float* B, float* C,int N)
{
    // Shared memory tiles used by all threads in the block
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    // Thread position inside the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global matrix position this thread computes
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // Loop over tile pairs needed for the dot product
    for(int t = 0; t < N / TILE; t++)
    {
        // Each thread loads one element of A and B into shared memory
        A_tile[ty][tx] = A[row * N + t * TILE + tx];
        B_tile[ty][tx] = B[(t * TILE + ty) * N + col];

        // Wait until all threads finish loading the tiles
        __syncthreads();

        // Compute partial dot product using the loaded tiles
        for(int k = 0; k < TILE; k++)
        {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        // Ensure all threads finished using the tiles
        // before they are overwritten in the next iteration
        __syncthreads();
    }

    // Write the final result for this thread's matrix element
    C[row * N + col] = sum;
}


////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{

    ////////////////////////////////////////////////////////////
    // Vector Add Test
    ////////////////////////////////////////////////////////////

    std::cout << "==== Vector Add ====" << std::endl;
    int N = 1 << 20;
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int grid = (N + block - 1) / block;
    int block = 256;
    vector_add_kernel<<<grid,block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result example: " << C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    ////////////////////////////////////////////////////////////
    // GEMM TEST
    ////////////////////////////////////////////////////////////
    std::cout << "\n==== GEMM ====" << std::endl;
    int M = 256;
    size_t bytes = M*M*sizeof(float);

    std::vector<float> h_A(M*M);
    std::vector<float> h_B(M*M);
    std::vector<float> h_C(M*M);

    for(int i=0;i<M*M;i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *g_A, *g_B, *g_C;

    cudaMalloc(&g_A, bytes);
    cudaMalloc(&g_B, bytes);
    cudaMalloc(&g_C, bytes);

    cudaMemcpy(g_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_B, h_B.data(), bytes,cudaMemcpyHostToDevice);

    dim3 grid2(M/TILE, M/TILE);
    dim3 block2(TILE, TILE);
    gemm_naive_kernel<<<grid2,block2>>>(g_A, g_B, g_C, M);
    cudaMemcpy(h_C.data(), g_C,bytes, cudaMemcpyDeviceToHost);

    std::cout << "Naive GEMM result example: " << h_C[0] << std::endl;

    gemm_tiled_kernel<<<grid2,block2>>>(g_A, g_B, g_C, M);

    cudaMemcpy(h_C.data(), g_C,bytes, cudaMemcpyDeviceToHost);

    std::cout << "Tiled GEMM result example: " << h_C[0] << std::endl;

    cudaFree(g_A);
    cudaFree(g_B);
    cudaFree(g_C);

    return 0;
}
