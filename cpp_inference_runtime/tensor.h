#pragma once
#include <vector>

class Tensor 
{

    public:
        std::vector<float> data;
        int rows;
        int cols;

        Tensor(int r,int c);
        float& operator()(int r,int c);

};