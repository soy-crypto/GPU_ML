#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

// Tensor
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

// Operator
class Operator
{
public:
    virtual ~Operator() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};

// ReLU (1D loop is fine)
class ReLU : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        int size = input.getSize();

        Tensor output(input.getRows(), input.getCols());

        const float* in = input.getData();
        float* out = output.getData();

        for (int i = 0; i < size; i++)
        {
            out[i] = std::max(0.0f, in[i]);
        }

        return output;
    }
};

// Softmax (row-wise)
class Softmax : public Operator
{
public:
    Tensor forward(const Tensor& input) override
    {
        int rows = input.getRows();
        int cols = input.getCols();

        Tensor output(rows, cols);

        const float* in = input.getData();
        float* out = output.getData();

        for (int r = 0; r < rows; r++)
        {
            const float* row_in = in + r * cols;
            float* row_out = out + r * cols;

            float maxVal = *std::max_element(row_in, row_in + cols);

            float sum = 0.0f;
            for (int c = 0; c < cols; c++)
            {
                row_out[c] = std::exp(row_in[c] - maxVal);
                sum += row_out[c];
            }

            for (int c = 0; c < cols; c++)
            {
                row_out[c] /= sum;
            }
        }

        return output;
    }
};

// Graph
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
            x = op->forward(x);
        }

        return x;
    }
};

// main
int main()
{
    // Input
    Tensor input(1, 3);
    float* in = input.getData();

    for (int i = 0; i < input.getSize(); i++)
    {
        in[i] = static_cast<float>(i);
    }

    // Graph
    Graph graph;
    graph.add_op(std::make_unique<ReLU>());
    graph.add_op(std::make_unique<Softmax>());

    // Run
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    // Output
    const float* out = output.getData();
    for (int i = 0; i < output.getSize(); i++)
    {
        std::cout << out[i] << " ";
    }

    double latency = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\nlatency: " << latency << " ms\n";

    return 0;
}
