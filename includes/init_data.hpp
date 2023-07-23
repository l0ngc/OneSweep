#pragma once
#include <vector>
#include <random>

void initKeys(uint* data, uint count, const int & lower, const int & upper)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<uint> distribution(lower, upper);

    for (uint i = 0; i < count; i++)
    {
        data[i] = distribution(generator);
    }
}