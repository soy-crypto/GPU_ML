#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

////////////////////////////////////////////////////////////
// Tensor
////////////////////////////////////////////////////////////
class Tensor
{
private:
    std::vector<float> data;
    int rows;
    int cols;

public:
    Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    float& operator()(int r, int c) { return data[r * cols + c]; }
    float operator()(int r, int c) const { return data[r * cols + c]; }

    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getSize() const { return rows * cols; }
};

////////////////////////////////////////////////////////////
// Operator Interface
////////////////////////////////////////////////////////////
class Operator
{
public:
    virtual ~Operator() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};

////////////////////////////////////////////////////////////
// ReLU
////////////////////////////////////////////////////////////
class ReLU : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        Tensor output(input.getRows(), input.getCols());

        const float* in = input.getData();
        float* out = output.getData();

        for (int i = 0; i < input.getSize(); i++)
        {
            out[i] = std::max(0.0f, in[i]);
        }

        return output; // RVO / move
    }
};

////////////////////////////////////////////////////////////
// Softmax (simple version: whole tensor)
////////////////////////////////////////////////////////////
class Softmax : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        Tensor output(input.getRows(), input.getCols());

        const float* in = input.getData();
        float* out = output.getData();

        // Find max (for numerical stability)
        float maxVal = in[0];
        for (int i = 1; i < input.getSize(); i++)
        {
            maxVal = std::max(maxVal, in[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < input.getSize(); i++)
        {
            out[i] = std::exp(in[i] - maxVal);
            sum += out[i];
        }

        // Normalize
        for (int i = 0; i < input.getSize(); i++)
        {
            out[i] /= sum;
        }

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

    Tensor run(const Tensor& input) const
    {
        Tensor x = input;

        for (const auto& op : ops)
        {
            x = op->forward(x); // relies on move / RVO
        }

        return x;
    }
};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////
int main()
{
    // Input
    Tensor input(1, 3);
    float* data = input.getData();

    for (int i = 0; i < input.getSize(); i++)
    {
        data[i] = static_cast<float>(i);
    }

    // Graph
    Graph graph;
    graph.add_op(std::make_unique<ReLU>());
    graph.add_op(std::make_unique<Softmax>());

    // Run
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    double latency =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Output
    std::cout << "Output:\n";
    const float* out = output.getData();

    for (int i = 0; i < output.getSize(); i++)
    {
        std::cout << out[i] << " ";
    }

    std::cout << "\nLatency: " << latency << " ms\n";

    return 0;
}