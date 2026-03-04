#pragma once
#include <vector>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() {}

    Tensor(std::vector<int> s) : shape(s) {
        int total = 1;
        for (int v : s) total *= v;
        data.resize(total);
    }

    int size() const {
        return data.size();
    }
};