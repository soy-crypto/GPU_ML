#include "ops.h"
#include <algorithm>
#include <cmath>

Tensor ReLU::forward(const Tensor& input) {

    Tensor output = input;

    for (size_t i = 0; i < input.data.size(); i++) {
        output.data[i] = std::max(0.0f, input.data[i]);
    }

    return output;
}

Tensor Softmax::forward(const Tensor& input) {

    Tensor output = input;

    float sum = 0.0f;

    for (size_t i = 0; i < input.data.size(); i++) {
        output.data[i] = std::exp(input.data[i]);
        sum += output.data[i];
    }

    for (size_t i = 0; i < input.data.size(); i++) {
        output.data[i] /= sum;
    }

    return output;
}