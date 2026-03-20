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
        float operator()(int r, int c) const { return data[r * cols + c] }
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
        virtal Tensor forward(const Tensor& input) = 0;
}

class ReLU: public Operator
{
    
}