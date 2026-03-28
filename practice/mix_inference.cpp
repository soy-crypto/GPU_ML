#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <cuda_runtime.h>


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
            float* d_input, d_output;
            size_t bytes = input.getSize() * sizeof(float);
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);

            // Call kernel
            int block = 256, grid = input.getRows();
            softmax_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
            cudaGetLastError();
            cudaDeviceSynchronize();
            cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyDeviceToHost);

            // Free Gpu
            cudaFree(d_input);
            cudaFree(d_output);

            //Return
            return output;
            
        }


}