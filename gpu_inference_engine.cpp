#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ============================================================
// CUDA kernels
// ============================================================

// C[M, N] = A[M, K] * B[K, N]
__global__ void matmul_kernel(const float* A,
                              const float* B,
                              float* C,
                              int M,
                              int N,
                              int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// out[j] = sum_i a[i] * B[i, j]
// a shape: [K], B shape: [K, N], out shape: [N]
__global__ void weighted_sum_kernel(const float* a,
                                    const float* B,
                                    float* out,
                                    int K,
                                    int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += a[i] * B[i * N + j];
        }
        out[j] = sum;
    }
}

// scores[i] = dot(q, K[i])
// q shape [D], K_cache shape [L, D], scores shape [L]
__global__ void attention_scores_kernel(const float* q,
                                        const float* K_cache,
                                        float* scores,
                                        int L,
                                        int D,
                                        float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < L) {
        float sum = 0.0f;
        const float* k = K_cache + idx * D;
        for (int i = 0; i < D; ++i) {
            sum += q[i] * k[i];
        }
        scores[idx] = sum * scale;
    }
}

// One-block softmax for a vector of length N.
// N should be <= a few thousand for this simple implementation.
__global__ void softmax_1d_kernel(const float* input, float* output, int N) {
    __shared__ float smax[256];
    __shared__ float ssum[256];

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = smax[0];

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            ssum[tid] += ssum[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = ssum[0];

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_val;
    }
}

// ============================================================
// Small utility classes
// ============================================================

struct Timer {
    std::chrono::high_resolution_clock::time_point start_time;

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end_time - start_time;
        return diff.count();
    }
};

class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}

    explicit DeviceBuffer(size_t num_floats) : ptr_(nullptr), size_(0) {
        allocate(num_floats);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t num_floats) {
        release();
        size_ = num_floats;
        CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(float)));
    }

    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    size_t size() const { return size_; }

private:
    float* ptr_;
    size_t size_;
};

static void fill_random(std::vector<float>& v, float scale = 0.1f) {
    for (float& x : v) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        x = (r - 0.5f) * 2.0f * scale;
    }
}

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

// ============================================================
// Linear layer: y = xW
// x shape [1, in_dim], W shape [in_dim, out_dim], y shape [1, out_dim]
// ============================================================

class Linear {
public:
    Linear(int in_dim, int out_dim)
        : in_dim_(in_dim), out_dim_(out_dim), d_weight_(in_dim * out_dim) {
        std::vector<float> h_weight(in_dim * out_dim);
        fill_random(h_weight, 0.2f);
        CUDA_CHECK(cudaMemcpy(d_weight_.data(),
                              h_weight.data(),
                              h_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void forward(const float* d_input, float* d_output) const {
        dim3 threads(16, 16);
        dim3 blocks((out_dim_ + 15) / 16, 1);
        matmul_kernel<<<blocks, threads>>>(d_input, d_weight_.data(), d_output, 1, out_dim_, in_dim_);
    }

private:
    int in_dim_;
    int out_dim_;
    DeviceBuffer d_weight_;
};

// ============================================================
// Embedding
// ============================================================

class Embedding {
public:
    Embedding(int vocab_size, int dim) : vocab_size_(vocab_size), dim_(dim), table_(vocab_size * dim) {
        h_table_.resize(vocab_size * dim);
        fill_random(h_table_, 0.5f);
        CUDA_CHECK(cudaMemcpy(table_.data(),
                              h_table_.data(),
                              h_table_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void lookup_host(int token_id, std::vector<float>& out) const {
        if (token_id < 0 || token_id >= vocab_size_) {
            throw std::out_of_range("token_id out of range");
        }
        out.resize(dim_);
        const float* src = h_table_.data() + token_id * dim_;
        std::memcpy(out.data(), src, dim_ * sizeof(float));
    }

private:
    int vocab_size_;
    int dim_;
    std::vector<float> h_table_;
    DeviceBuffer table_;
};

// ============================================================
// KV Cache
// ============================================================

class KVCache {
public:
    KVCache(int max_seq_len, int dim)
        : max_seq_len_(max_seq_len),
          dim_(dim),
          cur_len_(0),
          d_keys_(max_seq_len * dim),
          d_values_(max_seq_len * dim) {}

    void append(const float* d_k, const float* d_v) {
        if (cur_len_ >= max_seq_len_) {
            throw std::runtime_error("KV cache full");
        }

        CUDA_CHECK(cudaMemcpy(d_keys_.data() + cur_len_ * dim_,
                              d_k,
                              dim_ * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(d_values_.data() + cur_len_ * dim_,
                              d_v,
                              dim_ * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        ++cur_len_;
    }

    int length() const { return cur_len_; }
    int dim() const { return dim_; }

    const float* keys() const { return d_keys_.data(); }
    const float* values() const { return d_values_.data(); }

    void reset() { cur_len_ = 0; }

private:
    int max_seq_len_;
    int dim_;
    int cur_len_;
    DeviceBuffer d_keys_;
    DeviceBuffer d_values_;
};

// ============================================================
// Minimal decoder engine
// ============================================================

class MiniDecoderEngine {
public:
    MiniDecoderEngine(int vocab_size, int hidden_dim, int max_seq_len)
        : vocab_size_(vocab_size),
          hidden_dim_(hidden_dim),
          max_seq_len_(max_seq_len),
          embedding_(vocab_size, hidden_dim),
          q_proj_(hidden_dim, hidden_dim),
          k_proj_(hidden_dim, hidden_dim),
          v_proj_(hidden_dim, hidden_dim),
          o_proj_(hidden_dim, vocab_size),
          cache_(max_seq_len, hidden_dim),
          d_input_(hidden_dim),
          d_q_(hidden_dim),
          d_k_(hidden_dim),
          d_v_(hidden_dim),
          d_scores_(max_seq_len),
          d_weights_(max_seq_len),
          d_context_(hidden_dim) {}

    void reset_cache() {
        cache_.reset();
    }

    int decode_one_token(int token_id, double& step_ms) {
        Timer timer;
        timer.start();

        // 1) embedding lookup on host, copy to device
        std::vector<float> h_embed;
        embedding_.lookup_host(token_id, h_embed);
        CUDA_CHECK(cudaMemcpy(d_input_.data(),
                              h_embed.data(),
                              hidden_dim_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // 2) Q, K, V projections
        q_proj_.forward(d_input_.data(), d_q_.data());
        k_proj_.forward(d_input_.data(), d_k_.data());
        v_proj_.forward(d_input_.data(), d_v_.data());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3) append K and V to cache
        cache_.append(d_k_.data(), d_v_.data());

        // 4) attention scores: q against all cached K
        int L = cache_.length();
        float scale = 1.0f / std::sqrt(static_cast<float>(hidden_dim_));
        int score_threads = 256;
        int score_blocks = (L + score_threads - 1) / score_threads;
        attention_scores_kernel<<<score_blocks, score_threads>>>(
            d_q_.data(), cache_.keys(), d_scores_.data(), L, hidden_dim_, scale);

        // 5) softmax over scores
        softmax_1d_kernel<<<1, 256>>>(d_scores_.data(), d_weights_.data(), L);

        // 6) weighted sum over cached V -> context
        int context_threads = 256;
        int context_blocks = (hidden_dim_ + context_threads - 1) / context_threads;
        weighted_sum_kernel<<<context_blocks, context_threads>>>(
            d_weights_.data(), cache_.values(), d_context_.data(), L, hidden_dim_);

        // 7) output projection to logits
        DeviceBuffer d_logits(vocab_size_);
        o_proj_.forward(d_context_.data(), d_logits.data());

        CUDA_CHECK(cudaDeviceSynchronize());
        step_ms = timer.stop_ms();

        // 8) copy logits back and greedy argmax
        std::vector<float> h_logits(vocab_size_);
        CUDA_CHECK(cudaMemcpy(h_logits.data(),
                              d_logits.data(),
                              vocab_size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        return argmax(h_logits);
    }

    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                              int max_new_tokens,
                              std::vector<double>& step_latencies_ms) {
        if (prompt_tokens.empty()) {
            throw std::invalid_argument("prompt_tokens must not be empty");
        }

        reset_cache();

        std::vector<int> output = prompt_tokens;
        step_latencies_ms.clear();

        // Prefill prompt into cache.
        // For this minimal engine, we simply step through tokens one by one.
        int next_token = -1;
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            double ms = 0.0;
            next_token = decode_one_token(prompt_tokens[i], ms);
        }

        // Generate new tokens autoregressively.
        int current_token = next_token;
        for (int step = 0; step < max_new_tokens; ++step) {
            double ms = 0.0;
            int predicted = decode_one_token(current_token, ms);
            step_latencies_ms.push_back(ms);
            output.push_back(predicted);
            current_token = predicted;
        }

        return output;
    }

private:
    int vocab_size_;
    int hidden_dim_;
    int max_seq_len_;

    Embedding embedding_;
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;

    KVCache cache_;

    DeviceBuffer d_input_;
    DeviceBuffer d_q_;
    DeviceBuffer d_k_;
    DeviceBuffer d_v_;
    DeviceBuffer d_scores_;
    DeviceBuffer d_weights_;
    DeviceBuffer d_context_;
};

// ============================================================
// Simple benchmark helper
// ============================================================

static void print_tokens(const std::vector<int>& tokens, const std::string& name) {
    std::cout << name << ": ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
}

static void print_benchmark(const std::vector<double>& latencies_ms) {
    if (latencies_ms.empty()) return;

    double total = 0.0;
    for (double x : latencies_ms) total += x;

    double avg = total / static_cast<double>(latencies_ms.size());
    double ttft = latencies_ms.front();
    double tokens_per_sec = 1000.0 / avg;

    std::cout << "\nBenchmark\n";
    std::cout << "TTFT / first generated token: " << ttft << " ms\n";
    std::cout << "Average per-token latency:   " << avg << " ms\n";
    std::cout << "Throughput:                  " << tokens_per_sec << " tokens/s\n";
}

// ============================================================
// Main
// ============================================================

int main() {
    try {
        std::srand(42);

        // Tiny toy dimensions so the code stays easy to read.
        const int vocab_size = 64;
        const int hidden_dim = 32;
        const int max_seq_len = 64;
        const int max_new_tokens = 8;

        MiniDecoderEngine engine(vocab_size, hidden_dim, max_seq_len);

        // Fake prompt token ids
        std::vector<int> prompt = {1, 5, 9, 7};

        std::vector<double> latencies_ms;
        std::vector<int> generated = engine.generate(prompt, max_new_tokens, latencies_ms);

        print_tokens(prompt, "Prompt");
        print_tokens(generated, "Generated sequence");
        print_benchmark(latencies_ms);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Failure: " << e.what() << std::endl;
        return 1;
    }
}
