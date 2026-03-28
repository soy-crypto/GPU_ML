#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// Device Tensor
////////////////////////////////////////////////////////////

class DeviceTensor
{
private:
    float* data;
    int rows, cols;

public:
    DeviceTensor(int r, int c) : rows(r), cols(c)
    {
        cudaMalloc(&data, r * c * sizeof(float));
    }

    ~DeviceTensor()
    {
        cudaFree(data);
    }

    float* getData() { return data; }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getSize() const { return rows * cols; }

    void copyFromHost(float* h)
    {
        cudaMemcpy(data, h, getSize() * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copyToHost(float* h)
    {
        cudaMemcpy(h, data, getSize() * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

////////////////////////////////////////////////////////////
// Operator Interface
////////////////////////////////////////////////////////////

class GPUOperator
{
public:
    virtual DeviceTensor forward(const DeviceTensor& input) = 0;
    virtual ~GPUOperator() = default;
};

////////////////////////////////////////////////////////////
// ReLU Kernel
////////////////////////////////////////////////////////////

__global__ void relu_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = fmaxf(input[idx], 0.0f);
}

////////////////////////////////////////////////////////////
// YOUR Row-wise Softmax Kernel
////////////////////////////////////////////////////////////

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (row >= rows) return;

    const float* in  = input  + row * cols;
    float*       out = output + row * cols;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    // 1. max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += threads)
        local_max = fmaxf(local_max, in[i]);

    smax[tid] = local_max;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        __syncthreads();
    }

    float max_val = smax[0];

    // 2. sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads)
        local_sum += expf(in[i] - max_val);

    ssum[tid] = local_sum;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            ssum[tid] += ssum[tid + offset];
        __syncthreads();
    }

    float sum_val = ssum[0];

    // 3. normalize
    for (int i = tid; i < cols; i += threads)
        out[i] = expf(in[i] - max_val) / sum_val;
}

////////////////////////////////////////////////////////////
// Operators
////////////////////////////////////////////////////////////

class GPUReLU : public GPUOperator
{
public:
    DeviceTensor forward(const DeviceTensor& input) override
    {
        DeviceTensor output(input.getRows(), input.getCols());

        int N = input.getSize();
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        relu_kernel<<<blocks, threads>>>(
            input.getData(),
            output.getData(),
            N
        );

        cudaDeviceSynchronize();
        return output;
    }
};

class GPUSoftmax : public GPUOperator
{
public:
    DeviceTensor forward(const DeviceTensor& input) override
    {
        DeviceTensor output(input.getRows(), input.getCols());

        int rows = input.getRows();
        int cols = input.getCols();

        int threads = 256;
        int blocks = rows;

        softmax_kernel<<<blocks, threads>>>(
            input.getData(),
            output.getData(),
            rows,
            cols
        );

        cudaDeviceSynchronize();
        return output;
    }
};

////////////////////////////////////////////////////////////
// Graph
////////////////////////////////////////////////////////////

class GPUGraph
{
private:
    std::vector<std::unique_ptr<GPUOperator>> ops;

public:
    void add_op(std::unique_ptr<GPUOperator> op)
    {
        ops.push_back(std::move(op));
    }

    DeviceTensor run(DeviceTensor input)
    {
        DeviceTensor x = std::move(input);

        for (auto& op : ops)
        {
            x = op->forward(x);
        }

        return x;
    }
};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    int rows = 1;
    int cols = 6;

    float input[6] = {-2, -1, 0, 1, 2, 3};

    DeviceTensor d_input(rows, cols);
    d_input.copyFromHost(input);

    GPUGraph graph;
    graph.add_op(std::make_unique<GPUReLU>());
    graph.add_op(std::make_unique<GPUSoftmax>());

    DeviceTensor d_out = graph.run(std::move(d_input));

    float output[6];
    d_out.copyToHost(output);

    for (int i = 0; i < rows * cols; i++)
        std::cout << output[i] << " ";

    std::cout << std::endl;
}