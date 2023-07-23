#pragma once

#include <vector>
#include <iostream>
#include <string>

void printVec(const std::string& text, const std::vector<uint>& vec) {
    std::cout << text << ": ";
    for (const auto& value : vec) {
        std::cout << value << "\t";
    }
    std::cout << std::endl;
}
