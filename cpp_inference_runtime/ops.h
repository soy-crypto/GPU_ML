#pragma once
#include "tensor.h"

class Operator {
public:
    virtual Tensor forward(const Tensor& input) = 0;
};

class ReLU : public Operator {
public:
    Tensor forward(const Tensor& input) override;
};

class Softmax : public Operator {
public:
    Tensor forward(const Tensor& input) override;
};