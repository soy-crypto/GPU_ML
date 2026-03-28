#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////
// Device Buffer
////////////////////////////////////////////////////////////

class DeviceBuffer
{
private:
    float* data;
    int rows, cols;

public:
    DeviceBuffer(int r, int c) : rows(r), cols(c)
    {
        cudaMalloc(&data, rows * cols * sizeof(float));
    }

    ~DeviceBuffer()
    {
        cudaFree(data);
    }

    float* getData() { return data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getSize() const { return rows * cols; }
};

////////////////////////////////////////////////////////////
// Device Tensor (light wrapper)
////////////////////////////////////////////////////////////

class DeviceTensor
{
private:
    float* data;
    int rows, cols;

public:
    DeviceTensor(float* ptr, int r, int c) : data(ptr), rows(r), cols(c) {}

    float* getData() { return data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getSize() const { return rows * cols; }
};

////////////////////////////////////////////////////////////
// Fused ReLU + Row-wise Softmax Kernel
////////////////////////////////////////////////////////////

__global__ void fused_relu_softmax_kernel(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (row >= rows) return;

    const float* in  = input  + row * cols;
    float*       out = output + row * cols;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    // 1. max after ReLU
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += threads)
    {
        float val = fmaxf(in[i], 0.0f);
        local_max = fmaxf(local_max, val);
    }

    smax[tid] = local_max;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        __syncthreads();
    }

    float max_val = smax[0];

    // 2. sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads)
    {
        float val = fmaxf(in[i], 0.0f);
        local_sum += expf(val - max_val);
    }

    ssum[tid] = local_sum;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
            ssum[tid] += ssum[tid + offset];
        __syncthreads();
    }

    float sum_val = ssum[0];

    // 3. normalize
    for (int i = tid; i < cols; i += threads)
    {
        float val = fmaxf(in[i], 0.0f);
        out[i] = expf(val - max_val) / sum_val;
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
        int rows = input.getRows();
        int cols = input.getCols();

        int threads = 256;
        int blocks = rows;

        fused_relu_softmax_kernel<<<blocks, threads>>>(
            input.getData(),
            output.getData(),
            rows,
            cols
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
    std::unique_ptr<DeviceBuffer> input_buffer;
    std::unique_ptr<DeviceBuffer> output_buffer;

    int rows, cols;
    FusedReluSoftmax op;

public:
    GPUGraph(int r, int c) : rows(r), cols(c)
    {
        input_buffer  = std::make_unique<DeviceBuffer>(rows, cols);
        output_buffer = std::make_unique<DeviceBuffer>(rows, cols);
    }

    float* run(float* h_input)
    {
        cudaMemcpy(
            input_buffer->getData(),
            h_input,
            rows * cols * sizeof(float),
            cudaMemcpyHostToDevice
        );

        DeviceTensor input(input_buffer->getData(), rows, cols);
        DeviceTensor output(output_buffer->getData(), rows, cols);

        op.forward(input, output);

        return output.getData();
    }
};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    int rows = 1;
    int cols = 6;

    float input[6] = {-2, -1, 0, 1, 2, 3};

    GPUGraph graph(rows, cols);

    float* d_out = graph.run(input);

    float output[6];
    cudaMemcpy(
        output,
        d_out,
        rows * cols * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < rows * cols; i++)
        std::cout << output[i] << " ";

    std::cout << std::endl;
    return 0;
}