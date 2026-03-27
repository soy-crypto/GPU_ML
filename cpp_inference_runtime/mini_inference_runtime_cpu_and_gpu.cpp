#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <cuda_runtime.h>

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
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

            float *d_input, *d_output;

            // 1. Allocate
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);

            // 2. H2D
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);

            // 3. Kernel
            int block = 256;
            int grid  = (N + block - 1) / block;

            relu_kernel<<<grid, block>>>(d_input, d_output, N);

            cudaGetLastError();
            cudaDeviceSynchronize();

            // 4. D2H
            cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyDeviceToHost);

            // 5. Free
            cudaFree(d_input);
            cudaFree(d_output);

            return output;
        }
};

////////////////////////////////////////////////////////////
// Softmax kernels (multi-block)
////////////////////////////////////////////////////////////

// Stage 1: block max
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
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);

        __syncthreads();
    }

    if (tid == 0)
        block_max[blockIdx.x] = sdata[0];
}

// Stage 2: block sum
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
        val = expf(input[idx] - max_val);

    sdata[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            sdata[tid] += sdata[tid + offset];

        __syncthreads();
    }

    if (tid == 0)
        block_sum[blockIdx.x] = sdata[0];
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
        output[idx] = expf(input[idx] - max_val) / sum_val;
}

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

            int block = 256;
            int grid  = (N + block - 1) / block;

            float *d_input, *d_output;
            float *d_block_max, *d_block_sum;

            // 1. Allocate
            cudaMalloc(&d_input, bytes);
            cudaMalloc(&d_output, bytes);
            cudaMalloc(&d_block_max, grid * sizeof(float));
            cudaMalloc(&d_block_sum, grid * sizeof(float));

            // 2. H2D
            cudaMemcpy(d_input, input.getData(), bytes, cudaMemcpyHostToDevice);

            // ===== Stage 1: max =====
            block_max_kernel<<<grid, block>>>(d_input, d_block_max, N);
            cudaDeviceSynchronize();

            std::vector<float> h_block_max(grid);
            cudaMemcpy(h_block_max.data(), d_block_max,
                    grid * sizeof(float), cudaMemcpyDeviceToHost);

            float global_max = h_block_max[0];
            for (int i = 1; i < grid; i++)
                global_max = std::max(global_max, h_block_max[i]);

            // ===== Stage 2: sum =====
            block_sum_kernel<<<grid, block>>>(d_input, d_block_sum, global_max, N);
            cudaDeviceSynchronize();

            std::vector<float> h_block_sum(grid);
            cudaMemcpy(h_block_sum.data(), d_block_sum,
                    grid * sizeof(float), cudaMemcpyDeviceToHost);

            float global_sum = 0.0f;
            for (float v : h_block_sum)
                global_sum += v;

            // ===== Stage 3: normalize =====
            normalize_kernel<<<grid, block>>>(
                d_input, d_output, global_max, global_sum, N
            );

            cudaGetLastError();
            cudaDeviceSynchronize();

            // 3. D2H
            cudaMemcpy(output.getData(), d_output, bytes, cudaMemcpyDeviceToHost);

            // 4. Free
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_block_max);
            cudaFree(d_block_sum);

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

    double latency =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Output: ";
    for (int i = 0; i < output.getSize(); i++)
        std::cout << output.getData()[i] << " ";

    std::cout << "\nLatency: " << latency << " ms\n";

    return 0;
}