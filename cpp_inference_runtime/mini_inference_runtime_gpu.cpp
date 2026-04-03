#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

// ---------------- ReLU ----------------
__global__ void relu_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = fmaxf(0.0f, input[idx]);
}

// ---------------- Softmax ----------------
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* in = input + row * cols;
    float* out = output + row * cols;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, in[i]);

    smax[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        __syncthreads();
    }

    float max_val = smax[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        local_sum += expf(in[i] - max_val);

    ssum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            ssum[tid] += ssum[tid + offset];
        __syncthreads();
    }

    float sum_val = ssum[0];

    for (int i = tid; i < cols; i += blockDim.x)
        out[i] = expf(in[i] - max_val) / sum_val;
}

// ---------------- Minimal Op ----------------
class Op
{
public:
    virtual ~Op() = default;

    // in-place transform: input → output
    virtual void run(float* input, float* output, int rows, int cols) = 0;
};

// ---------------- ReLU Op ----------------
class ReLUOp : public Op
{
public:
    void run(float* input, float* output, int rows, int cols) override
    {
        int N = rows * cols;
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        relu_kernel<<<blocks, threads>>>(input, output, N);
    }
};

// ---------------- Softmax Op ----------------
class SoftmaxOp : public Op
{
public:
    void run(float* input, float* output, int rows, int cols) override
    {
        int threads = 256;
        int blocks = rows;

        softmax_kernel<<<blocks, threads>>>(input, output, rows, cols);
    }
};

// ---------------- Minimal Graph ----------------
class Graph
{
private:
    std::vector<std::unique_ptr<Op>> ops;

public:
    void add(std::unique_ptr<Op> op)
    {
        ops.push_back(std::move(op));
    }

    void run(float* d_input, float* d_output, int rows, int cols)
    {
        float* current = d_input;
        float* buffer;

        cudaMalloc(&buffer, rows * cols * sizeof(float));

        for (size_t i = 0; i < ops.size(); i++)
        {
            ops[i]->run(current, buffer, rows, cols);

            // swap input/output
            std::swap(current, buffer);
        }

        // ensure result is in d_output
        cudaMemcpy(d_output, current, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(buffer);
    }
};

// ---------------- Main ----------------
int main()
{
    float h_input[6] = {-2, -1, 0, 1, 2, 3};
    float h_output[6];

    int rows = 2, cols = 3;

    float *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    Graph graph;
    graph.add(std::make_unique<ReLUOp>());
    graph.add(std::make_unique<SoftmaxOp>());

    graph.run(d_input, d_output, rows, cols);

    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows * cols; i++)
        std::cout << h_output[i] << " ";

    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
}