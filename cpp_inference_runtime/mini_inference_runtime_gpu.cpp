#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
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
// Host Tensor
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
// Device Tensor (RAII)
////////////////////////////////////////////////////////////

class DeviceTensor
{
    private:
        float* data;
        int rows;
        int cols;

    public:
        DeviceTensor(int r, int c) : data(nullptr), rows(r), cols(c)
        {
            CHECK_CUDA(cudaMalloc(&data, getSize() * sizeof(float)));
        }

        ~DeviceTensor()
        {
            if (data != nullptr)
            {
                cudaFree(data);
            }
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
                if (data != nullptr)
                {
                    cudaFree(data);
                }

                data = other.data;
                rows = other.rows;
                cols = other.cols;

                other.data = nullptr;
                other.rows = 0;
                other.cols = 0;
            }
            return *this;
        }

        float* getData()
        {
            return data;
        }

        const float* getData() const
        {
            return data;
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

        void copyFromHost(const Tensor& hostTensor)
        {
            if (hostTensor.getRows() != rows || hostTensor.getCols() != cols)
            {
                std::cerr << "Shape mismatch in copyFromHost\n";
                std::exit(1);
            }

            CHECK_CUDA(cudaMemcpy(
                data,
                hostTensor.getData(),
                getSize() * sizeof(float),
                cudaMemcpyHostToDevice
            ));
        }

        void copyToHost(Tensor& hostTensor) const
        {
            if (hostTensor.getRows() != rows || hostTensor.getCols() != cols)
            {
                std::cerr << "Shape mismatch in copyToHost\n";
                std::exit(1);
            }

            CHECK_CUDA(cudaMemcpy(
                hostTensor.getData(),
                data,
                getSize() * sizeof(float),
                cudaMemcpyDeviceToHost
            ));
        }
};

////////////////////////////////////////////////////////////
// GPU Operator interface
////////////////////////////////////////////////////////////

class GPUOperator
{
    public:
        virtual DeviceTensor forward(const DeviceTensor& input) = 0;
        virtual ~GPUOperator() = default;
};

////////////////////////////////////////////////////////////
// ReLU kernel
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

class GPUReLU : public GPUOperator
{
    public:
        DeviceTensor forward(const DeviceTensor& input) override
        {
            DeviceTensor output(input.getRows(), input.getCols());

            int N = input.getSize();
            int block_size = 256;
            int grid_size = (N + block_size - 1) / block_size;

            relu_kernel<<<grid_size, block_size>>>(
                input.getData(),
                output.getData(),
                N
            );

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            return output;
        }
};

////////////////////////////////////////////////////////////
// Simple GPU Softmax
// One block version for demo / learning purpose
////////////////////////////////////////////////////////////

__global__ void softmax_kernel(const float* input, float* output, int N)
{
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_max = -1e20f;
    for (int i = tid; i < N; i += stride)
    {
        local_max = fmaxf(local_max, input[i]);
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += stride)
    {
        local_sum += expf(input[i] - max_val);
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = shared_sum[0];

    for (int i = tid; i < N; i += stride)
    {
        output[i] = expf(input[i] - max_val) / sum_val;
    }
}

////////////////////////////////////////////////////////////
// GPU Softmax
////////////////////////////////////////////////////////////

class GPUSoftmax : public GPUOperator
{
    public:
        DeviceTensor forward(const DeviceTensor& input) override
        {
            DeviceTensor output(input.getRows(), input.getCols());

            int N = input.getSize();
            int block_size = 256;

            softmax_kernel<<<1, block_size>>>(
                input.getData(),
                output.getData(),
                N
            );

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            return output;
        }
};

////////////////////////////////////////////////////////////
// GPU Graph
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

        DeviceTensor run(const DeviceTensor& input)
        {
            DeviceTensor x(input.getRows(), input.getCols());

            CHECK_CUDA(cudaMemcpy(
                x.getData(),
                input.getData(),
                input.getSize() * sizeof(float),
                cudaMemcpyDeviceToDevice
            ));

            for (const auto& op : ops)
            {
                DeviceTensor out = op->forward(x);
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
    float* h_data = input.getData();

    h_data[0] = -2.0f;
    h_data[1] = -1.0f;
    h_data[2] =  0.0f;
    h_data[3] =  1.0f;
    h_data[4] =  2.0f;
    h_data[5] =  3.0f;

    DeviceTensor d_input(input.getRows(), input.getCols());
    d_input.copyFromHost(input);

    GPUGraph graph;
    graph.add_op(std::make_unique<GPUReLU>());
    graph.add_op(std::make_unique<GPUSoftmax>());

    auto start = std::chrono::high_resolution_clock::now();
    DeviceTensor d_output = graph.run(d_input);
    auto end = std::chrono::high_resolution_clock::now();

    Tensor output(input.getRows(), input.getCols());
    d_output.copyToHost(output);

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