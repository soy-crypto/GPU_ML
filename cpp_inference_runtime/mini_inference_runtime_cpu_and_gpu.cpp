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
        cudaError_t err = (call);                                            \
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
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
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
            size_t bytes = static_cast<size_t>(N) * sizeof(float);

            float* d_input = nullptr;
            float* d_output = nullptr;

            CHECK_CUDA(cudaMalloc(&d_input, bytes));
            CHECK_CUDA(cudaMalloc(&d_output, bytes));

            CHECK_CUDA(cudaMemcpy(
                d_input,
                input.getData(),
                bytes,
                cudaMemcpyHostToDevice
            ));

            int block = 256;
            int grid = (N + block - 1) / block;

            relu_kernel<<<grid, block>>>(d_input, d_output, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(
                output.getData(),
                d_output,
                bytes,
                cudaMemcpyDeviceToHost
            ));

            CHECK_CUDA(cudaFree(d_input));
            CHECK_CUDA(cudaFree(d_output));

            return output;
        }
};

////////////////////////////////////////////////////////////
// Multi-block Softmax kernels
////////////////////////////////////////////////////////////

// Stage 1: each block computes a partial max
__global__ void block_max_kernel(const float* input, float* block_max, int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < N) ? input[idx] : -1e20f;
    sdata[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        block_max[blockIdx.x] = sdata[0];
    }
}

// Stage 2: each block computes a partial sum using the global max
__global__ void block_sum_kernel(const float* input,
                                 float* block_sum,
                                 float max_val,
                                 int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (idx < N)
    {
        val = expf(input[idx] - max_val);
    }

    sdata[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        block_sum[blockIdx.x] = sdata[0];
    }
}

// Stage 3: normalize
__global__ void normalize_kernel(const float* input,
                                 float* output,
                                 float max_val,
                                 float sum_val,
                                 int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        output[idx] = expf(input[idx] - max_val) / sum_val;
    }
}

////////////////////////////////////////////////////////////
// GPU Softmax (multi-block)
////////////////////////////////////////////////////////////

class GPUSoftmax : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            int N = input.getSize();
            size_t bytes = static_cast<size_t>(N) * sizeof(float);

            int block = 256;
            int grid = (N + block - 1) / block;

            float* d_input = nullptr;
            float* d_output = nullptr;
            float* d_block_max = nullptr;
            float* d_block_sum = nullptr;

            CHECK_CUDA(cudaMalloc(&d_input, bytes));
            CHECK_CUDA(cudaMalloc(&d_output, bytes));
            CHECK_CUDA(cudaMalloc(&d_block_max, grid * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_block_sum, grid * sizeof(float)));

            CHECK_CUDA(cudaMemcpy(
                d_input,
                input.getData(),
                bytes,
                cudaMemcpyHostToDevice
            ));

            // Stage 1: block-wise max
            block_max_kernel<<<grid, block>>>(d_input, d_block_max, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // Copy partial max values back to CPU and reduce
            std::vector<float> h_block_max(grid);
            CHECK_CUDA(cudaMemcpy(
                h_block_max.data(),
                d_block_max,
                grid * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            float global_max = h_block_max[0];
            for (int i = 1; i < grid; i++)
            {
                global_max = std::max(global_max, h_block_max[i]);
            }

            // Stage 2: block-wise sum(exp(x - global_max))
            block_sum_kernel<<<grid, block>>>(d_input, d_block_sum, global_max, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // Copy partial sum values back to CPU and reduce
            std::vector<float> h_block_sum(grid);
            CHECK_CUDA(cudaMemcpy(
                h_block_sum.data(),
                d_block_sum,
                grid * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            float global_sum = 0.0f;
            for (int i = 0; i < grid; i++)
            {
                global_sum += h_block_sum[i];
            }

            // Stage 3: normalize
            normalize_kernel<<<grid, block>>>(
                d_input,
                d_output,
                global_max,
                global_sum,
                N
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(
                output.getData(),
                d_output,
                bytes,
                cudaMemcpyDeviceToHost
            ));

            CHECK_CUDA(cudaFree(d_input));
            CHECK_CUDA(cudaFree(d_output));
            CHECK_CUDA(cudaFree(d_block_max));
            CHECK_CUDA(cudaFree(d_block_sum));

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
    Tensor input(1, 12);
    float* data = input.getData();

    data[0]  = -2.0f;
    data[1]  = -1.0f;
    data[2]  =  0.0f;
    data[3]  =  1.0f;
    data[4]  =  2.0f;
    data[5]  =  3.0f;
    data[6]  =  0.5f;
    data[7]  = -0.5f;
    data[8]  =  4.0f;
    data[9]  =  1.5f;
    data[10] = -3.0f;
    data[11] =  2.5f;

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