#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (row >= rows)
    {
        return;
    }

    const float* in_array = input + row * cols;
    float* out_array = output + row * cols;

    __shared__ float localMax[256];
    __shared__ float localSum[256];

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += threads)
    {
        local_max = fmaxf(local_max, in_array[i]);
    }

    localMax[tid] = local_max;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            localMax[tid] = fmaxf(localMax[tid], localMax[tid + offset]);
        }
        __syncthreads();
    }

    float global_max = localMax[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads)
    {
        local_sum += expf(in_array[i] - global_max);
    }

    localSum[tid] = local_sum;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            localSum[tid] += localSum[tid + offset];
        }
        __syncthreads();
    }

    float global_sum = localSum[0];

    for (int i = tid; i < cols; i += threads)
    {
        out_array[i] = expf(in_array[i] - global_max) / global_sum;
    }
}

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

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& other) noexcept
        : data(other.data), rows(other.rows), cols(other.cols)
    {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    DeviceTensor& operator=(DeviceTensor&& other) noexcept
    {
        if (this != &other)
        {
            cudaFree(data);
            data = other.data;
            rows = other.rows;
            cols = other.cols;

            other.data = nullptr;
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    float* getData() const { return data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getSize() const { return rows * cols; }

    void copyToDevice(const float* hData)
    {
        cudaMemcpy(data, hData, getSize() * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copyToHost(float* hData) const
    {
        cudaMemcpy(hData, data, getSize() * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

class GPUOperator
{
public:
    virtual ~GPUOperator() = default;
    virtual DeviceTensor forward(const DeviceTensor& input) = 0;
};

class GPUReLU : public GPUOperator
{
public:
    DeviceTensor forward(const DeviceTensor& input) override
    {
        DeviceTensor output(input.getRows(), input.getCols());

        int N = input.getSize();
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        relu_kernel<<<blocks, threads>>>(input.getData(), output.getData(), N);
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

        int threads = 256;
        int blocks = input.getRows();

        softmax_kernel<<<blocks, threads>>>(
            input.getData(),
            output.getData(),
            input.getRows(),
            input.getCols()
        );

        cudaDeviceSynchronize();
        return output;
    }
};

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

int main()
{
    float input[6] = {-2, -1, 0, 1, 2, 3};
    int rows = 2, cols = 3;

    DeviceTensor d_input(rows, cols);
    d_input.copyToDevice(input);

    GPUGraph graph;
    graph.add_op(std::make_unique<GPUReLU>());
    graph.add_op(std::make_unique<GPUSoftmax>());

    DeviceTensor d_output = graph.run(std::move(d_input));

    float output[6];
    d_output.copyToHost(output);

    for (int i = 0; i < rows * cols; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}