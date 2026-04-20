#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------- CUDA Checks ----------------
#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr "CUDA error\n"; std::exit(1); \
    }

// ---------------- Tensor ----------------
struct Tensor 
{
    float* data;
    int rows;
    int cols;

    int numel() const { return rows * cols; }
    size_t bytes() const { return numel() * sizeof(float); }
};

// ---------------- DeviceTensor ----------------
class DeviceTensor 
{
    Tensor t_;
public:
    DeviceTensor() { t_.data = nullptr; }
    ~DeviceTensor() { if (t_.data) cudaFree(t_.data); }

    void allocate(int r, int c) 
    {
        t_.rows = r;
        t_.cols = c;
        CHECK_CUDA(cudaMalloc(&t_.data, t_.bytes()));
    }

    Tensor view() { return t_; }
};

// ---------------- CUDA Kernels ----------------

// Linear: Y = XW + b
__global__ void linear_kernel(const float* X, const float* W, const float* B, float* Y, int rows, int in_c, int out_c)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < out_c) 
    {
        float sum = B[c];
        for (int k = 0; k < in_c; k++) 
        {
            sum += X[r * in_c + k] * W[k * out_c + c];
        }

        Y[r * out_c + c] = sum;

    }

}

__global__ void relu_kernel(const float* in, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = fmaxf(0.0f, in[i]);
}

__global__ void softmax_kernel(const float* in, float* out, int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    const float* x = in + row * cols;
    float* y = out + row * cols;

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, x[i]);

    smax[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) 
    {
        if (tid < s)
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);

        __syncthreads();
    }

    float max_val = smax[0];

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum += expf(x[i] - max_val);

    ssum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) 
    {
        if (tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }

    float denom = ssum[0];

    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - max_val) / denom;

}

// ---------------- Op Base ----------------
class Op 
{
public:
    virtual ~Op() = default;
    virtual Tensor infer(const Tensor& in) = 0;
    virtual void run(const Tensor& in, const Tensor& out) = 0;

};

// ---------------- LinearOp ----------------
class LinearOp : public Op 
{
    int in_c_, out_c_;
    float* d_W_;
    float* d_B_;

public:
    LinearOp(int in_c, int out_c, const std::vector<float>& W, const std::vector<float>& B): in_c_(in_c), out_c_(out_c)
    {
        CHECK_CUDA(cudaMalloc(&d_W_, W.size()*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_, B.size()*sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_W_, W.data(), W.size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));
    }

    Tensor infer(const Tensor& in) override 
    {
        return {nullptr, in.rows, out_c_};
    }

    void run(const Tensor& in, const Tensor& out) override 
    {
        dim3 threads(16,16);
        dim3 blocks((out.cols+15)/16, (in.rows+15)/16);

        linear_kernel<<<blocks, threads>>>(in.data, d_W_, d_B_, out.data, in.rows, in_c_, out_c_);
    }

};

// ---------------- ReLUOp ----------------
class ReLUOp : public Op 
{
public:
    Tensor infer(const Tensor& in) override { return in; }

    void run(const Tensor& in, const Tensor& out) override 
    {
        int N = in.numel();
        relu_kernel<<<(N+255)/256, 256>>>(in.data, out.data, N);
    }

};

// ---------------- SoftmaxOp ----------------
class SoftmaxOp : public Op 
{
public:
    Tensor infer(const Tensor& in) override { return in; }

    void run(const Tensor& in, const Tensor& out) override 
    {
        softmax_kernel<<<in.rows, 256>>>(in.data, out.data, in.rows, in.cols);
    }

};

// ---------------- Graph ----------------
class Graph 
{
    std::vector<std::unique_ptr<Op>> ops_;
    DeviceTensor bufA_, bufB_;
    Tensor out_shape_;

public:
    void add(std::unique_ptr<Op> op) 
    {
        ops_.push_back(std::move(op));
    }

    void plan(int rows, int cols) 
    {
        Tensor cur{nullptr, rows, cols};

        int max_cols = cols;

        for (auto& op : ops_) 
        {
            cur = op->infer(cur);
            max_cols = std::max(max_cols, cur.cols);
        }

        bufA_.allocate(rows, max_cols);
        bufB_.allocate(rows, max_cols);

        out_shape_ = cur;

    }

    Tensor output_shape() { return out_shape_; }

    void run(const Tensor& input, const Tensor& output) 
    {
        Tensor cur = bufA_.view();
        Tensor nxt = bufB_.view();

        CHECK_CUDA(cudaMemcpy(cur.data, input.data, input.bytes(), cudaMemcpyDeviceToDevice));

        cur.rows = input.rows;
        cur.cols = input.cols;

        for (auto& op : ops_) {
            Tensor out_shape = op->infer(cur);
            nxt.rows = out_shape.rows;
            nxt.cols = out_shape.cols;

            op->run(cur, nxt);
            std::swap(cur, nxt);
        }

        CHECK_CUDA(cudaMemcpy(output.data, cur.data, output.bytes(), cudaMemcpyDeviceToDevice));
    }
    
};

// ---------------- Main ----------------
int main()
{
    int rows = 2, cols = 3;

    float h_input[6] = {-2,-1, 0, 1, 2, 3};

    std::vector<float> W = 
    {
        0.2, -0.5, 0.1, 0.4,
        0.7, 0.3, -0.2, 0.8,
        -0.6, 0.9, 0.5, -0.1
    };

    std::vector<float> B = {0.1, 0.2, -0.1, 0.05};

    DeviceTensor d_in, d_out;
    d_in.allocate(rows, cols);

    Graph g;
    g.add(std::make_unique<LinearOp>(3, 4, W, B));
    g.add(std::make_unique<ReLUOp>());
    g.add(std::make_unique<SoftmaxOp>());

    g.plan(rows, cols);

    Tensor out_shape = g.output_shape();
    d_out.allocate(out_shape.rows, out_shape.cols);

    CHECK_CUDA(cudaMemcpy(d_in.view().data, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    g.run(d_in.view(), d_out.view());

    std::vector<float> h_out(out_shape.numel());
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out.view().data, out_shape.bytes(), cudaMemcpyDeviceToHost));

    for (float v : h_out) 
    {   
        std::cout << v << " ";
    }

    std::cout << std::endl;

}