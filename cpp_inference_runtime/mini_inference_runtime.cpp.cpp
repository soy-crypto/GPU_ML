#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

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
        Tensor(int r, int c): rows(r), cols(c), data(r * c, 0.0f) {}

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
// Operator Interface
////////////////////////////////////////////////////////////

class Operator
{
    public:
        virtual Tensor forward(const Tensor& input) = 0;
};

class ReLU : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            const float* inputData = input.getData();
            float* outData = output.getData();

            for (int i = 0; i < input.getSize(); i++)
            {
                outData[i] = std::max(0.0f, inputData[i]);
            }

            return output;
        }
};

class Softmax : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            const float* inputData = input.getData();
            float* outData = output.getData();

            float maxVal = inputData[0];

            for (int i = 1; i < input.getSize(); i++)
            {
                maxVal = std::max(maxVal, inputData[i]);
            }

            float sum = 0.0f;

            for (int i = 0; i < input.getSize(); i++)
            {
                outData[i] = std::exp(inputData[i] - maxVal);
                sum += outData[i];
            }

            for (int i = 0; i < input.getSize(); i++)
            {
                outData[i] /= sum;
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
        std::vector<Operator*> ops;

    public:
        void add_op(Operator* op)
        {
            ops.push_back(op);
        }

        Tensor run(const Tensor& input)
        {
            Tensor x = input;

            for (auto op : ops)
            {
                x = op->forward(x);
            }

            return x;
        }
};

////////////////////////////////////////////////////////////
// Main Runtime
////////////////////////////////////////////////////////////

int main()
{
    // Initialize tensor
    Tensor input(1,3);
    float* data = input.getData();

    for(int i=0;i<input.getSize();i++)
    {
        data[i] = static_cast<float>(i);
    }

    // Build computation graph
    Graph graph;
    ReLU relu;
    Softmax softmax;

    graph.add_op(&relu);
    graph.add_op(&softmax);

    // Run inference
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    // Print output
    std::cout << "Output: ";
    double latency = std::chrono::duration<double,std::milli>(end - start).count();
    float* out = output.getData();
    for(int i=0;i<output.getSize();i++)
    {
        std::cout << out[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "Latency: " << latency << " ms\n";
    std::cout << "Inference done\n";

    //Return
    return 0;
}