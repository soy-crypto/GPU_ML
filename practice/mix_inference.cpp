#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <cuda_runtime.h>
#include <cfloat>


__global__ void relu_kernel(const float* input, float* output, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N)
    {
        output[index] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }

}


class Operator
{
    public:
        virtual Tensor forward(const Tensor& input) = 0;
        virtual ~Operator() = default;
};


class GPUReLU: public operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            // Host init
            Tensor output(input.getRows(), input.getCols());

            // Kernel init
            float* d_input, d_output;
            size_t bytes = input.getSize() * sizeof(float);
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);
        
            // Call kernel
            int block = 256;
            int grid = (size + block - 1) / block;
            relu_kernel<<<grid, block>>>(d_input, d_output, size);
            cudaGetLastError();
            cudaDeviceSynchronize();
            cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyHostToDevice);

            // Free space
            cudaFree(d_input);
            cudaFree(d_output);

            // Return
            return output;
        }

};


class GPUSoftMax: public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            // Host init
            Tensor output(input.getRows(), input.getCols());

            // Kernel init
            float* d_input = nullptr, d_output = nullptr;
            size_t bytes = input.getSize() * sizeof(float);
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);

            // Call kernel
            int block = 256, grid = input.getRows();
            softmax_kernel<<<grid, block>>>(d_input, d_output, input.getRows(), input.getCols());
            cudaGetLastError();
            cudaDeviceSynchronize();
            cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyDeviceToHost);

            // Free Gpu
            cudaFree(d_input);
            cudaFree(d_output);

            //Return
            return output;
        }

};


__global__ void softmax_kernel(const float* input, float* output, int rows, int cols)
{
    // Check
    int row = blockIdx.x, threads_per_block = blockDim.x, tid = threadIdx.x;
    if(row >= rows)
    {
        return;
    }

    // Init
    __shared__ float max[256], sum[256];
    const float* in_array = input + row * cols;
    float* out_array = output + row * cols;

    // Compute
    int index = tid;
    /* find local max */
    float local_max = -FLT_MAX;
    for(int i = index; i < cols; i += threads_per_block)
    {
        local_max = fmaxf(local_max, in_array[i]);
    }

    max[index] = local_max;
    __syncthreads();

    /* find global max */
    float global_max = -FLT_MAX;
    for(int offset = threads_per_block / 2; offset > 0; offset /= 2)
    {
        if(index < offset)
        {
            max[index] = fmaxf(max[index], max[index + offset]);
        }
        
        __syncthreads();
    }

    global_max = max[0];

    /* compute local sum */
    float local_sum = 0.0f;
    for(int i = index; i < cols; i += threads_per_block)
    {
        local_sum += expf(in_array[i] - global_max);
    } 

    sum[index] = local_sum;
    __syncthreads();

    /** find global sum */
    float global_sum = 0.0f;
    for(int offset = threads_per_block / 2; offset > 0; offset /= 2)
    {
        if(index < offset)
        {
            sum[index] = sum[index] + sum[index + offset];
        }

        __syncthreads();
    }

    global_sum = sum[0];

    /* normalize */
    for(int i = index; i < cols; i += threads_per_block)
    {
        out_array[i] = expf(in_array[i] - global_max) / global_sum;
    }   

    // Return
    return;

}