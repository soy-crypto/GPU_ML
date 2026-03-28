#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// Device Buffer (reuse memory)
////////////////////////////////////////////////////////////

class DeviceBuffer
{
public:
    float* data;
    int size;

    DeviceBuffer(int n) : size(n)
    {
        cudaMalloc(&data, n * sizeof(float));
    }

    ~DeviceBuffer()
    {
        cudaFree(data);
    }
};

////////////////////////////////////////////////////////////
// Device Tensor (light wrapper)
////////////////////////////////////////////////////////////

class DeviceTensor
{
private:
    float* data;
    int size;

public:
    DeviceTensor(float* ptr, int n) : data(ptr), size(n) {}

    float* getData() { return data; }
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
    int threads = blockDim.x;

    // 1. max (after ReLU)
    float local_max = -FLT_MAX;

    for (int i = tid; i < N; i += threads)
    {
        float val = fmaxf(input[i], 0.0f);
        local_max = fmaxf(local_max, val);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        __syncthreads();
    }

    float max_val = s_max[0];

    // 2. sum
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += threads)
    {
        float val = fmaxf(input[i], 0.0f);
        local_sum += expf(val - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            s_sum[tid] += s_sum[tid + offset];
        __syncthreads();
    }

    float sum_val = s_sum[0];

    // 3. normalize
    for (int i = tid; i < N; i += threads)
    {
        float val = fmaxf(input[i], 0.0f);
        output[i] = expf(val - max_val) / sum_val;
    }
}

////////////////////////////////////////////////////////////
// Operator (fused)
////////////////////////////////////////////////////////////

class FusedReluSoftmax
{
public:
    void forward(const DeviceTensor& input, DeviceTensor& output)
    {
        int N = input.getSize();
        int threads = 256;

        fused_relu_softmax_kernel<<<1, threads>>>(
            input.getData(),
            output.getData(),
            N
        );

        cudaDeviceSynchronize();
    }
};

////////////////////////////////////////////////////////////
// Inference Graph
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

    float* run(float* h_input)
    {
        // H → D
        cudaMemcpy(buffer1->data, h_input, size * sizeof(float),
                   cudaMemcpyHostToDevice);

        DeviceTensor input(buffer1->data, size);
        DeviceTensor output(buffer2->data, size);

        // Compute
        op.forward(input, output);

        return output.getData();
    }
};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    int N = 6;

    std::vector<float> input = {-2, -1, 0, 1, 2, 3};

    GPUGraph graph(N);

    float* d_out = graph.run(input.data());

    std::vector<float> output(N);

    cudaMemcpy(output.data(), d_out,
               N * sizeof(float), cudaMemcpyDeviceToHost);

    for (float v : output)
        std::cout << v << " ";

    std::cout << std::endl;
}