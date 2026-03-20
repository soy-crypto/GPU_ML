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
        int getCols() const { return cols;}
        int getSize() const { return rows * cols;}
    
};


//Operator
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
            const float* in = input.getData();
            float* out = output.getData();
            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::max(0.0f, in[i]);
            }

            //return
            return output;
        }

};


class Softmax: public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            //Output
            Tensor output(input.getRows(), input.getCols());
            float* out = output.getData();

            //normalization
            const float* in = input.getData();
            float maxVal = *(std::max_element(in, in + input.getSize())), sum = 0.0f;
            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::exp(in[i] - maxVal);
                sum += out[i];
            }

            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] /= sum;
            }

            //return
            return output;
        }

};


class Graph
{
    private:
        std::vector<std::unique_ptr<Operator>> ops;

    public:
        void add_op(std::unique_ptr<Operator> op) { ops.push_back(std::move(op)); }
        Tensor run(const Tensor& input) const
        {
            Tensor x = input;
            for(const auto& op : ops)
            {
                x = op->forward(x);
            }

            return x;
        }
        
}