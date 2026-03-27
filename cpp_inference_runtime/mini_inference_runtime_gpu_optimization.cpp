#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

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
        int rows, cols;

    public:
        Tensor(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}

        float* getData() { return data.data(); }
        const float* getData() const { return data.data(); }

        int getRows() const { return rows; }
        int getCols() const { return cols; }
        int getSize() const { return rows * cols; }
};

////////////////////////////////////////////////////////////
// Device Buffer (Memory Reuse)
////////////////////////////////////////////////////////////

class DeviceBuffer
{
    public:
        float* data;
        int size;

        DeviceBuffer(int n) : size(n)
        {
            CHECK_CUDA(cudaMalloc(&data, n * sizeof(float)));
        }

        ~DeviceBuffer()
        {
            if (data) cudaFree(data);
        }
};

////////////////////////////////////////////////////////////
// Device Tensor (uses external buffer)
////////////////////////////////////////////////////////////

class DeviceTensor
{
    private:
        float* data;
        int size;

    public:
        DeviceTensor(float* ptr, int n) : data(ptr), size(n) {}

        float* getData() { return data; }
        const float* getData() const { return data; }
        int getSize() const { return size; }
};

////////////////////////////////////////////////////////////
// Fused Kernel (ReLU + Softmax)
////////////////////////////////////////////////////////////

__global__ void fused_relu_softmax_kernel(const float* input, float* output, int N)
{
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    int tid = threadIdx.x;

    float local_max = -1e20f;

    for (int i = tid; i < N; i += blockDim.x)
    {
        float val = fmaxf(input[i], 0.0f); // ReLU fused
        local_max = fmaxf(local_max, val);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // reduction max
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x)
    {
        float val = fmaxf(input[i], 0.0f);
        local_sum += expf(val - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // reduction sum
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = s_sum[0];

    for (int i = tid; i < N; i += blockDim.x)
    {
        float val = fmaxf(input[i], 0.0f);
        output[i] = expf(val - max_val) / sum_val;
    }
}

////////////////////////////////////////////////////////////
// Fused Operator
////////////////////////////////////////////////////////////

class FusedReluSoftmax
{
    public:
        void forward(const DeviceTensor& input, DeviceTensor& output)
        {
            int N = input.getSize();
            int block = 256;

            fused_relu_softmax_kernel<<<1, block>>>(
                input.getData(),
                output.getData(),
                N
            );

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
};

////////////////////////////////////////////////////////////
// GPU Graph (Memory Reuse)
////////////////////////////////////////////////////////////

class GPUGraph
{
    private:
        std::unique_ptr<DeviceBuffer> buffer1;
        std::unique_ptr<DeviceBuffer> buffer2;
        int size;

        FusedReluSoftmax op;

    public:
        GPUGraph(int n) : size(n)
        {
            buffer1 = std::make_unique<DeviceBuffer>(n);
            buffer2 = std::make_unique<DeviceBuffer>(n);
        }

        DeviceTensor run(const float* h_input)
        {
            CHECK_CUDA(cudaMemcpy(
                buffer1->data,
                h_input,
                size * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            DeviceTensor input(buffer1->data, size);
            DeviceTensor output(buffer2->data, size);

            op.forward(input, output);

            return output;
        }
};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    int N = 6;

    Tensor input(1, N);
    float* data = input.getData();

    data[0] = -2;
    data[1] = -1;
    data[2] = 0;
    data[3] = 1;
    data[4] = 2;
    data[5] = 3;

    GPUGraph graph(N);

    auto start = std::chrono::high_resolution_clock::now();
    DeviceTensor d_out = graph.run(input.getData());
    auto end = std::chrono::high_resolution_clock::now();

    Tensor output(1, N);

    CHECK_CUDA(cudaMemcpy(
        output.getData(),
        d_out.getData(),
        N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    double latency =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Output: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << output.getData()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Latency: " << latency << " ms\n";
}