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
            // Init
            Tensor output(input.getRows(), input.getCols());
            int size = input.getSize();
            size_t bytes = static_cast<size_t>(size) * sizeof(float);
            
            float* d_input, d_output;
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);

            int block = 256;
            int grid = (size + block - 1) / block;

            // Host to Device
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);
            
            // Call kernel
            relu_kernel<<<grid, block>>>(d_input, d_output, size);
            cudaGetLastError();
            cudaDeviceSynchronize();

            // data to host
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
            Tensor output(input.getRows(), input.getCols());
            int size = input.getSize();
            size_t bytes = size * sizeof(float);

            float* d_input, d_output;
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);

            int block = 256, grid = input.getRows();

            //GPU
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);
            

            


            //Return

        }


}