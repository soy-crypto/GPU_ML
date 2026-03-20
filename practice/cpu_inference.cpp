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

            //Input
            const float* in = input.getData();

            //init
            
            //compute output
            float maxVal = 0.0f;
            for(int i = 0; i < input.getSize(); i++)
            {
                maxVal = std::max(maxVal, in[i]);
            }

            
        }

};