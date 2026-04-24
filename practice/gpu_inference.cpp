#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

// relu
__global__ void relu_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }

};


// softmax
__global__ void softmax_kernel(const float* input, float* output, int row, int cols)
{
    // init
    int row = blockIdx.x, tid = threadIdx.x;
    const float* in = input + row * cols;
    float* out = ouput + row * cols;
    __shared__ float smax[256], ssum[256];

    // compute
    /** get global maximum and sum */
    float local_max = -FLX_MAX;
    for(int i = tid; i < cols; i += blockDim.x)
    {
        local_max = fmax(local_max, in[i]);
    }

    smax[tid] = local_max;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if(tid < offset)
        {
            smax[tid] = fmax(smax[tid], smax[tid + offset]);
            __syncthreads();
        }
    }

    float global_max = smax[0];

    /** compute softmax */
    float local_sum = 0.0f;
    for(int i = tid; i < cols; i += blockDIm.x)
    {
        if(i < cols)
        {
            local_sum += expf(in[s] - global_max);
        }
    }

    ssum[tid] = local_sum;
    __synsthreads();

    for(int offset = blodkDim.x / 2; offset > 0; offset /= 2)
    {
        if(tid < offset)
        {
            ssum[tid] += ssum[tid + offset];
            __synsthreads();
        }

    }

    float global_sum = ssum[0];

    for(int i = tid; i < cols; i += blockDim.x)
    {
        out[i] = expf(in[i] - global_max) / global_sum;
    }
    
};


class Op
{
    public:
        virtual ~Op() = default;
        virtual void run(float* input, float* output, int row, int cols) = 0;
};


class ReLUOp: public Op
{
    public:
        void run(float* input, float* output, int rows, int cols) override
        {
            int thread = 256, blocks = (rows * cols + thread - 1) / thread;
            relu_kernel<<blocks, thread>>(input, output, N)  
        }

};


class SoftmaxOp: public Op
{
    public:
        void run(float* input, float* output, int rows, int cols) override
        {
            int thread = 256, blocks = rows;
            softmax_kernel<<blocks, thread>>(input, output, rows, cols);
        }

};


class Graph
{
    private:
        std::vector<std::unique_ptr<Op>> ops;

    public:
        void add(std::unique_ptr<Op> op)
        {
            ops.push_back(std::move(op))
        }

        void run(float* input, float* output, int rows, int cols)
        {
            // compute ops
            float *in = input, *out = null;
            cudaMalloc(&out, rows * cols * sizeof(float));
            for(size_t i = 0; i < ops.size(); i++)
            {
                ops[i].run(in, out, rows, cols);
                std::swap(in, out);
            }

            // update output
            cudaMemcpy(output, in, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);

            // free buffer
            cudaFree(out);
            
        }

};


int main()
{
    // init
    /* host input and output */
    int rows = 2, cols = 3;
    float host_input[6] = {-3, -2, -1, 1, 2, 3};
    float host_output;
    
    /* device input and output */
    float *device_input, *device_output;
    cudaMalloc(&device_input, rows * cols * sizeof(float));
    cudaMalloc(&device_output, rows * cols * sizeof(float));
    cudaMemcpy(device_input, host_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    /* graph */
    Graph graph;
    graph.add(std::make_unique<ReLUOp>());
    graph.add(std::make_unique<SoftmaxOp>());

    // run
    graph.run(device_input, device_output, rows, cols);
    cudaMemcpy(host_output, device_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // free buffer
    cudaFree(device_input);
    cudaFree(device_output);

    // Return
    return 0;
}