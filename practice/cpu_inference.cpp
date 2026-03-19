#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

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
        virtual ~Operator() {}
        virtual Tensor forward(const Tensor& input) = 0;
};


class ReLU : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            //Get in and out
            Tensor output(input.getRows(), input.getCols());
            const float* in = input.getData();
            float* out = output.getData();
            
            //ReLU
            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::max(0.0f, in[i]);
            }

            //Return
            return output;
        }

};


class Softmax : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            
        }
}