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