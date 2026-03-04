#include "graph.h"

void Graph::add_op(Operator* op) {
    ops.push_back(op);
}

Tensor Graph::run(const Tensor& input) {

    Tensor x = input;

    for (auto op : ops) {
        x = op->forward(x);
    }

    return x;
}