#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

//Tensor
class Tensor
{
    private:
        std::vector<float> data;
        int rows;
        int cols;
    
    public:
        Tensor(int r, int c): rows(r), cols(c), data(r * c, 0.0f) {}
        float& operator()(int r, int c) { return data[r * cols + c]; }
        float operator()(int r, int c) const { return data[r * cols + c]; }
        float* getData() { return data.data(); }
        const float* getData() const { return data.data(); }
        int getRows() const { return rows; }
        int getCols() const { return cols; }
        int getSize() const { return rows * cols; }

};


class Operator
{
    public: 
        virtual ~Operator() = default;
        virtual Tensor forward(const Tensor& input) = 0;
    
};


class ReLU: public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());
            float* out = output.getData();
            
            const float* in = input.getData();
            int size = input.getSize();
            for(int i = 0; i < size; i++)
            {
                out[i] = std::max(0.0f, in[i]);
            }

            return output;
        }

};


class Softmax: public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());
            float* out = output.getData();

            const float* in = input.getData();
            int rows = input.getRows(), cols = input.getCols();
            for(int i = 0; i < rows; i++)
            {
                const float* row_in = in + i * cols;
                float* row_out = out + i * cols;
                
                float maxVal = *(std::max_element(row_in, row_in + cols));
                float sum = 0.0f;
                for(int k = 0; k < cols; k++)
                {
                    row_out[k] = std::exp(row_in[k] - maxVal);
                    sum += row_out[k];
                }

                for(int k = 0; k < cols; k++)
                {
                    row_out[k] /= sum;
                }
                
            }//for

            // Return
            return output;
            
        }//forward
        

};


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
            for(const auto& op : ops)
            {
                x = op->forward(x);
            }

            return x;
        }

};


int main()
{
    // Input
    Tensor input(1, 3);
    float* data = input.getData();
    int size = input.getSize();
    for(int i = 0; i < size; i++)
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

    // Print
    const float* out = output.getData();
    for(int i = 0; i < output.getSize(); i++)
    {
        std::cout << out[i] << " ";
    }

    double latency = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\ latency: " << latency << " ms" << std::endl;

    // Return
    return 0;
}