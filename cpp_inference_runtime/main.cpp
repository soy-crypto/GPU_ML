#include <iostream>
#include <chrono>

#include "tensor.h"
#include "ops.h"
#include "graph.h"

int main() {

    Tensor input({1,3});

    input.data = {1.0f, 2.0f, 3.0f};

    Graph graph;

    ReLU relu;
    Softmax softmax;

    graph.add_op(&relu);
    graph.add_op(&softmax);

    auto start = std::chrono::high_resolution_clock::now();

    Tensor output = graph.run(input);

    auto end = std::chrono::high_resolution_clock::now();

    double latency =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Output: ";

    for (float v : output.data) {
        std::cout << v << " ";
    }

    std::cout << std::endl;

    std::cout << "Latency: " << latency << " ms\n";

    std::cout << "Inference done\n";

    return 0;
}