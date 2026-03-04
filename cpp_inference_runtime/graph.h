#pragma once
#include <vector>
#include "ops.h"

class Graph {
public:
    std::vector<Operator*> ops;

    void add_op(Operator* op);

    Tensor run(const Tensor& input);
};