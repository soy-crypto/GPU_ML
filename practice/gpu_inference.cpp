#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }

    return;
}


__global__ void softmax_kernel(const float* input, float* output, int rows, int cols)
{
    // Check
    int row = blockIdx.x, tid = threadIdx.x, threads = blcokDim.x
    if(row >= rows)
    {
        return;
    }

    // Init
    const float* in_array = input + row * cols;
    float* out_array = out + row * cols;
    __shared__ float localMax[256];
    __shared__ float localSum[256];
    
    //Compute
    /** find local max and lcoal sum */
    float local_max = -FLT_MAX, local_sum = 0.0f;
    for(int i = tid; i < cols; i += cols)
    {
        local_max = fmaxf(0.0f, in_array[i]);
        local_sum += in_array[i];
    }

    localMax[tid] = local_max;
    localSum[tid] = local_sum;
    __syncthreads();

    /* find global max and global sum */
    float global_max, global_sum;
    for(int offset = threads / 2; offset > 0; offset /= 2)
    {
        if(tid < offset)
        {
            localSum[tid] = localSum[tid] + localSum[tid + offset];
            localMax[tid] = fmaxf(localMax[tid], localMax[tid + offset];);
        }
        
        __syncthreads();

    }

    global_max = localMax[0];
    global_sum = localSum[0];

    /* normalize the input */
    for(int i = tid; i < cols; i += cols)
    {
        out_array[i] = expf(in_array[i] - global_max) / global_sum;
    }


    // Return
    return;
}


class DeviceTensor
{
    private:
        float* data;
        int rows, cols;

    public:
        DeviceTensor(int r, int c): rows(r), cols(c) { cudaMalloc(&data, r * c * sizeof(float)); }
        ~DevicTenssor() { cudaMalloc(data); }

        float* getData() { return data; }
        int getRows() { return rows; }
        int getCols() { return cols; }
        int getSize() { return rows * cols; }

        void copyToDevice(float* hData) { cudaMemcpy(data, hData, getSize() * sizeof(float), cudaMemcpyHostToDevice); }
        void copyToHost(float* hData) { cudaMemcpy(hData, data, getSize() * sizeof(flolat), cudaMemcpyDeviceToHost); }

};


class GPUOperator
{
    public:
        virtual ~GPUOperator() = default;
        virtual DeviceTensor forward(const DeviceTensor& input) = 0;
        
};


class GPUReLU: public GPUOperator
{
    public:
        DeviceTensor forward(const DeviceTensor& input) override
        {
            DeviceTensor output(input.getRows(), input.getCols());
            
            int N = input.getSize();
            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            
            relu_kernel<<<blocks, threads>>>(input.getData(), output.getData(), N);

            cudaDeviceSynchronize();
            return output;
        }
    
};


class GPUSoftmax: public GPUOperator
{
    public:
        DeviceTensor forward(const DeviceTensor& input) override
        {
            DeviceTensor output(input.getData(), output.getData());

            int threads = 256, blocks = input.getRows();
            softmax_kernel<<<blocks, threads>>>(input.getData(), output.getData(), intput.getRows(), input.getCols());
            
            cudaDeviceSynchronize();
            
            return output;
        }
};


class GPUGraph
{
    private:
        std::vector<std::unique_ptr<GPUOperator>> ops;
    
    public:
        void add_op(std::unique_ptr<GPUOperator> op) { ops.push_bakc(std::move(op)); }
        DeviceTensor run(DeviceTensor input)
        {
            DeviceTensor x = input;
            for(auto& op : ops)
            {
                x = x->forward(x);
            }

            return x;
        }

};


int main()
{
    // Init
    float intput[6] = {-2, -1, 0, 1, 2, 3};
    int rows = 2, cols = 3;
    DeviceTensor d_input(rows, cols);
    d_input.copyToDevice(input);

    // Graph
    GPUgraph graph;
    graph.add_Op(std::make_unique<GPUReLU>());
    graph.add_op(std::make_unique<GPUSoftmax>());

    // compute
    DeviceTensor d_output = graph.run(d_input);
    float output[6];
    d_output.copyToHost(output);

    // Print
    for(int i = 0; i < rows * cols; i++)
    {
        std::cout << output[i] << " ";
    }

    std::cout << std::endl;
}