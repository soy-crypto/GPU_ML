#include <iostream>
#include <chrono>

#include "tensor.h"
#include "ops.h"
#include "graph.h"

int main() 
{
    //Init
    /** Init data */
    Tensor input(1, 3);
    float* data = input.getData();
    for(size_t i = 0; i < input.getSize(); i++)
    {
        data[i] = 1.0f * i;
    }
    
    /** Init grap */
    Graph graph;
    ReLU relu;
    Softmax softmax;

    graph.add_op(&relu);
    graph.add_op(&softmax);

    //Computation
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    //Update latency
    double latency = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Output: ";
    float* out = output.getData();
    for (size_t i = 0; i < output.getSize(); i++) 
    {
        std::cout << out[i] << " ";
    }
    
    //Output the result of the graph
    std::cout << std::endl;
    std::cout << "Latency: " << latency << " ms\n";
    std::cout << "Inference done\n";
    
    //Return
    return 0;
}