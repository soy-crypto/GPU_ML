#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

class DeviceTensor
{
    private:
        float* data;
        int rows, cols;

    public:
        DeviceTensor(int r, int c): rows(r), cols(c) { cudaMalloc(&data, r * c * sizeof(float)); }
        ~DevicTenssor() { cudaMalloc(data); }

        float* getData() { return data; }
        int getRows() { return rows; }
        int getCols() { return cols; }
        int getSize() { return rows * cols; }

        void copyToDevice(float* hData) { cudaMemcpy(data, hData, getSize() * sizeof(float), cudaMemcpyHostToDevice); }
        void copyToHost(float* hData) { cudaMemcpy(hData, data, getSize() * sizeof(flolat), cudaMemcpyDeviceToHost); }

};


class GPUOperator
{
    public:
        virtual ~GPUOperator() = default;
        virtual DeviceTensor forward(const DeviceTensor& input) = 0;
        
};


__global__ void relu_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }

    return;
}


__global__ void softmax_kernel(const float* input, float* output, int rows, int cols)
{
    // Check
    int row = blockIdx.x, tid = threadIdx.x, threads = blcokDim.x
    if(row >= rows)
    {
        return;
    }

    // Init
    

}