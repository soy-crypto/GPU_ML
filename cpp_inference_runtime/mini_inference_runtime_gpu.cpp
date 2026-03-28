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
        int size;

    public:
        DeviceTensor(int n) : size(n)
        {
            cudaMalloc(&data, n * sizeof(float));
        }

        ~DeviceTensor()
        {
            cudaFree(data);
        }

        float* getData() { return data; }
        int getSize() const { return size; }

        void copyFromHost(float* h)
        {
            cudaMemcpy(data, h, size * sizeof(float), cudaMemcpyHostToDevice);
        }

        void copyToHost(float* h)
        {
            cudaMemcpy(h, data, size * sizeof(float), cudaMemcpyDeviceToHost);
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
// Softmax Kernel (1-block version)
////////////////////////////////////////////////////////////

__global__ void softmax_kernel(const float* input, float* output, int N)
{
    __shared__ float smax[256];
    __shared__ float ssum[256];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += threads)
        local_max = fmaxf(local_max, input[i]);

    smax[tid] = local_max;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        __syncthreads();
    }

    float max_val = smax[0];

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads)
        local_sum += expf(input[i] - max_val);

    ssum[tid] = local_sum;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            ssum[tid] += ssum[tid + offset];
        __syncthreads();
    }

    float sum_val = ssum[0];

    for (int i = tid; i < N; i += threads)
        output[i] = expf(input[i] - max_val) / sum_val;

}

////////////////////////////////////////////////////////////
// Operators
////////////////////////////////////////////////////////////

class GPUReLU : public GPUOperator
{
    public:
        DeviceTensor forward(const DeviceTensor& input) override
        {
            DeviceTensor output(input.getSize());

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
            DeviceTensor output(input.getSize());

            int threads = 256;

            softmax_kernel<<<1, threads>>>(
                input.getData(),
                output.getData(),
                input.getSize()
            );

            cudaDeviceSynchronize();
            return output;
        }

};

////////////////////////////////////////////////////////////
// Graph (Correct Design)
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
    int N = 6;

    float input[6] = {-2, -1, 0, 1, 2, 3};

    DeviceTensor d_input(N);
    d_input.copyFromHost(input);

    GPUGraph graph;
    graph.add_op(std::make_unique<GPUReLU>());
    graph.add_op(std::make_unique<GPUSoftmax>());

    DeviceTensor d_out = graph.run(std::move(d_input));

    float output[6];
    d_out.copyToHost(output);

    for (int i = 0; i < N; i++)
        std::cout << output[i] << " ";

    std::cout << std::endl;
}