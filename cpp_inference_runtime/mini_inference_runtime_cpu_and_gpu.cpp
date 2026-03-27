#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// CUDA error check
////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;               \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

////////////////////////////////////////////////////////////
// Tensor (CPU)
////////////////////////////////////////////////////////////

class Tensor
{
    private:
        std::vector<float> data;
        int rows;
        int cols;

    public:
        Tensor(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}

        float* getData() { return data.data(); }
        const float* getData() const { return data.data(); }

        int getRows() const { return rows; }
        int getCols() const { return cols; }
        int getSize() const { return rows * cols; }
        
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
// CUDA ReLU kernel
////////////////////////////////////////////////////////////

__global__ void relu_kernel(const float* input, float* output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

////////////////////////////////////////////////////////////
// CUDA Softmax kernel (single block)
////////////////////////////////////////////////////////////

__global__ void softmax_kernel(const float* input, float* output, int N)
{
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    int tid = threadIdx.x;

    // 1. find max
    float local_max = -1e20f;
    for (int i = tid; i < N; i += blockDim.x)
    {
        local_max = fmaxf(local_max, input[i]);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];

    // 2. sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
    {
        local_sum += expf(input[i] - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = s_sum[0];

    // 3. normalize
    for (int i = tid; i < N; i += blockDim.x)
    {
        output[i] = expf(input[i] - max_val) / sum_val;
    }
}

////////////////////////////////////////////////////////////
// GPU ReLU
////////////////////////////////////////////////////////////

class GPUReLU : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            int N = input.getSize();
            size_t bytes = N * sizeof(float);

            float* d_input = nullptr;
            float* d_output = nullptr;

            CHECK_CUDA(cudaMalloc(&d_input, bytes));
            CHECK_CUDA(cudaMalloc(&d_output, bytes));

            CHECK_CUDA(cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice));

            int block = 256;
            int grid = (N + block - 1) / block;

            relu_kernel<<<grid, block>>>(d_input, d_output, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyDeviceToHost));

            CHECK_CUDA(cudaFree(d_input));
            CHECK_CUDA(cudaFree(d_output));

            return output;
        }
};

////////////////////////////////////////////////////////////
// GPU Softmax
////////////////////////////////////////////////////////////

class GPUSoftmax : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            int N = input.getSize();
            size_t bytes = N * sizeof(float);

            float* d_input = nullptr;
            float* d_output = nullptr;

            CHECK_CUDA(cudaMalloc(&d_input, bytes));
            CHECK_CUDA(cudaMalloc(&d_output, bytes));

            CHECK_CUDA(cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice));

            int block = 256;

            softmax_kernel<<<1, block>>>(d_input, d_output, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyDeviceToHost));

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
        std::vector<std::unique_ptr<Operator>> ops;

    public:
        void add_op(std::unique_ptr<Operator> op)
        {
            ops.push_back(std::move(op));
        }

        Tensor run(const Tensor& input)
        {
            Tensor x = input;

            for (const auto& op : ops)
            {
                Tensor out = op->forward(x);
                x = std::move(out);
            }

            return x;
        }

};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    Tensor input(1, 6);
    float* data = input.getData();

    data[0] = -2.0f;
    data[1] = -1.0f;
    data[2] =  0.0f;
    data[3] =  1.0f;
    data[4] =  2.0f;
    data[5] =  3.0f;

    Graph graph;

    graph.add_op(std::make_unique<GPUReLU>());
    graph.add_op(std::make_unique<GPUSoftmax>());

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    double latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Input:  ";
    for (int i = 0; i < input.getSize(); i++)
    {
        std::cout << input.getData()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Output: ";
    for (int i = 0; i < output.getSize(); i++)
    {
        std::cout << output.getData()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Latency: " << latency_ms << " ms\n";
    std::cout << "Inference done\n";

    return 0;

}