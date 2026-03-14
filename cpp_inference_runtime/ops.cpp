#include "ops.h"
#include <algorithm>
#include <cmath>

Tensor ReLU::forward(const Tensor& input) 
{
    Tensor output = Tensor(input.getRows(), input.getCols());
    float* outData = output.getData();
    const float* inputData = input.getData();
    for (size_t i = 0; i < input.getSize(); i++) 
    {
        outData[i] = std::max(0.0f, inputData[i]);
    }

    return output;
}


Tensor Softmax::forward(const Tensor& input) 
{

    Tensor output = Tensor(input.getRows(), input.getCols());
    float* outData = output.getData();
    const float* inputData = input.getData();
    float sum = 0.0f;
    for (size_t i = 0; i < input.getSize(); i++) 
    {
        outData[i] = std::exp(inputData[i]);
        sum += outData[i];
    }

    for (size_t i = 0; i < input.getSize(); i++) 
    {
        outData[i] /= sum;
    }

    return output;
}