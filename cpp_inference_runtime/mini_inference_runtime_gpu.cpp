#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// Simple CUDA error check macro
////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if(err != cudaSuccess)                                               \
        {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;               \
            std::exit(1);                                                    \
        }                                                                    \
    } while(0)

////////////////////////////////////////////////////////////
// Tensor (CPU-side tensor)
////////////////////////////////////////////////////////////

class Tensor
{
private:
    std::vector<float> data;
    int rows;
    int cols;

public:
    Tensor(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}

    float& operator()(int r, int c)
    {
        return data[r * cols + c];
    }

    float operator()(int r, int c) const
    {
        return data[r * cols + c];
    }

    float* getData()
    {
        return data.data();
    }

    const float* getData() const
    {
        return data.data();
    }

    int getRows() const
    {
        return rows;
    }

    int getCols() const
    {
        return cols;
    }

    int getSize() const
    {
        return rows * cols;
    }
};

////////////////////////////////////////////////////////////
// Operator interface
////////////////////////////////////////////////////////////

class Operator
{
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual ~Operator() = default;
};

////////////////////////////////////////////////////////////
// CPU ReLU
////////////////////////////////////////////////////////////

class ReLU : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        Tensor output(input.getRows(), input.getCols());

        const float* inputData = input.getData();
        float* outputData = output.getData();

        for(int i = 0; i < input.getSize(); i++)
        {
            outputData[i] = std::max(0.0f, inputData[i]);
        }

        return output;
    }
};

////////////////////////////////////////////////////////////
// CPU Softmax
// This implementation treats the whole tensor as one vector.
// For learning purposes, that is fine.
////////////////////////////////////////////////////////////

class Softmax : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        Tensor output(input.getRows(), input.getCols());

        const float* inputData = input.getData();
        float* outputData = output.getData();

        float maxVal = inputData[0];
        for(int i = 1; i < input.getSize(); i++)
        {
            maxVal = std::max(maxVal, inputData[i]);
        }

        float sum = 0.0f;
        for(int i = 0; i < input.getSize(); i++)
        {
            outputData[i] = std::exp(inputData[i] - maxVal);
            sum += outputData[i];
        }

        for(int i = 0; i < input.getSize(); i++)
        {
            outputData[i] /= sum;
        }

        return output;
    }
};

////////////////////////////////////////////////////////////
// CUDA ReLU kernel
////////////////////////////////////////////////////////////

__global__ void relu_kernel(const float* input, float* output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

////////////////////////////////////////////////////////////
// GPU ReLU operator
//
// Flow:
// 1. copy CPU tensor -> GPU
// 2. launch CUDA kernel
// 3. copy GPU result -> CPU tensor
////////////////////////////////////////////////////////////

class GPUReLU : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        Tensor output(input.getRows(), input.getCols());

        const int N = input.getSize();
        const size_t bytes = N * sizeof(float);

        float* d_input = nullptr;
        float* d_output = nullptr;

        CHECK_CUDA(cudaMalloc(&d_input, bytes));
        CHECK_CUDA(cudaMalloc(&d_output, bytes));

        CHECK_CUDA(cudaMemcpy(
            d_input,
            input.getData(),
            bytes,
            cudaMemcpyHostToDevice));

        int block = 256;
        int grid = (N + block - 1) / block;

        relu_kernel<<<grid, block>>>(d_input, d_output, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(
            output.getData(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));

        return output;
    }
};

////////////////////////////////////////////////////////////
// Graph
////////////////////////////////////////////////////////////

class Graph
{
private:
    std::vector<Operator*> ops;

public:
    void add_op(Operator* op)
    {
        ops.push_back(op);
    }

    Tensor run(const Tensor& input)
    {
        Tensor x = input;

        for(Operator* op : ops)
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
    // Create input tensor
    Tensor input(1, 6);
    float* data = input.getData();

    // Example input: contains negative and positive values
    data[0] = -2.0f;
    data[1] = -1.0f;
    data[2] =  0.0f;
    data[3] =  1.0f;
    data[4] =  2.0f;
    data[5] =  3.0f;

    // Build graph:
    // GPU ReLU -> CPU Softmax
    Graph graph;
    GPUReLU gpu_relu;
    Softmax softmax;

    graph.add_op(&gpu_relu);
    graph.add_op(&softmax);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    double latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Input:  ";
    for(int i = 0; i < input.getSize(); i++)
    {
        std::cout << input.getData()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Output: ";
    for(int i = 0; i < output.getSize(); i++)
    {
        std::cout << output.getData()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Latency: " << latency_ms << " ms\n";
    std::cout << "Inference done\n";

    return 0;
}