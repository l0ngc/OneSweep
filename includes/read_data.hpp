#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<float> read_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return std::vector<float>();
    }

    std::vector<float> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float value;
        if (iss >> value) {
            data.push_back(value);
        }
    }

    file.close();
    return data;
}

// void process_data(const std::string& basePath, const std::vector<std::string>& sortedFilenames,
//                   const std::vector<std::string>& unorderedFilenames,
//                   std::vector<std::vector<float>>& sortedData, std::vector<std::vector<float>>& unorderedData) {
//     // 读取并存储已排序数据
//     for (const std::string& filename : sortedFilenames) {
//         std::string filePath = basePath + filename;
//         std::vector<float> data = read_data(filePath);
//         sortedData.push_back(data);
//     }

//     // 读取并存储未排序数据
//     for (const std::string& filename : unorderedFilenames) {
//         std::string filePath = basePath + filename;
//         std::vector<float> data = read_data(filePath);
//         unorderedData.push_back(data);
//     }
// }
void process_data(std::vector<std::vector<float>>& sortedData, std::vector<std::vector<float>>& unorderedData) {
    // 读取并存储已排序数据
    std::string basePath = "../data/";
    std::vector<std::string> sortedFilenames = {
        "float_sorted_2^10.csv", "float_sorted_2^11.csv", "float_sorted_2^12.csv",
        "float_sorted_2^13.csv", "float_sorted_2^14.csv", "float_sorted_2^15.csv",
        "float_sorted_2^16.csv", "float_sorted_2^17.csv", "float_sorted_2^18.csv",
        "float_sorted_2^19.csv", "float_sorted_2^20.csv",
        //  "float_sorted_2^21.csv",
        // "float_sorted_2^22.csv", "float_sorted_2^23.csv", "float_sorted_2^24.csv",
        // "float_sorted_2^25.csv"
    };
    std::vector<std::string> unorderedFilenames = {
        "float_sorted_2^10.csv", "float_sorted_2^11.csv", "float_sorted_2^12.csv",
        "float_sorted_2^13.csv", "float_sorted_2^14.csv", "float_sorted_2^15.csv",
        "float_sorted_2^16.csv", "float_sorted_2^17.csv", "float_sorted_2^18.csv",
        "float_sorted_2^19.csv", "float_sorted_2^20.csv",
        //  "float_sorted_2^21.csv",
        // "float_sorted_2^22.csv", "float_sorted_2^23.csv", "float_sorted_2^24.csv",
        // "float_sorted_2^25.csv"
    };
    for (const std::string& filename : sortedFilenames) {
        std::string filePath = basePath + filename;
        std::vector<float> data = read_data(filePath);
        sortedData.push_back(data);
    }

    // 读取并存储未排序数据
    for (const std::string& filename : unorderedFilenames) {
        std::string filePath = basePath + filename;
        std::vector<float> data = read_data(filePath);
        unorderedData.push_back(data);
    }
}


    // std::string basePath = "../data/";
    // std::vector<std::string> sortedFilenames = {
    //     "float_sorted_2^10.csv", "float_sorted_2^11.csv", "float_sorted_2^12.csv",
    //     "float_sorted_2^13.csv", "float_sorted_2^14.csv", "float_sorted_2^15.csv",
    //     "float_sorted_2^16.csv", "float_sorted_2^17.csv", "float_sorted_2^18.csv",
    //     "float_sorted_2^19.csv", "float_sorted_2^20.csv"
    // };
    // std::vector<std::string> unorderedFilenames = {
    //     "float_unorder_2^10.csv", "float_unorder_2^11.csv", "float_unorder_2^12.csv",
    //     "float_unorder_2^13.csv", "float_unorder_2^14.csv", "float_unorder_2^15.csv",
    //     "float_unorder_2^16.csv", "float_unorder_2^17.csv", "float_unorder_2^18.csv",
    //     "float_unorder_2^19.csv", "float_unorder_2^20.csv"
    // };