#pragma once
#include <vector>

class Tensor 
{
    private:
        std::vector<float> data;
        int rows;
        int cols;

    public:
        //constructor
        Tensor(int r, int c);

        //Accessors
        float& operator()(int r, int c);
        float operator()(int r, int c) const;

        //Data pointers
        const float* getData() const;
        int getRows() const;
        int getCols() const;
        int getSize() const;
};