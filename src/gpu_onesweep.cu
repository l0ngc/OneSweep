#include <iostream>
#include <cub/cub.cuh>
#include "cuda_runtime.h"
#include "init_data.hpp"
#include <vector>
#include <algorithm>
#include "onesweep.hpp"
#include "utils.hpp"
#include "rs_scan.h"
#include "rs_sort.h"
// #define DEBUG

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: program_name count" << std::endl;
        return 1;
    }

    const uint count = 1 << std::stoi(argv[1]);
    std::cout << "Problem size: " << count << std::endl;
    std::vector<uint> h_input(count);
    std::vector<uint> h_output(count);
    initKeys(h_input.data(), count, 0, count);
    std::clock_t start;

// record cpu kernel
    start = std::clock();
    std::sort(h_input.begin(), h_input.end());
    double cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "CPU time: " << cpu_duration << " s" << std::endl;

// record gpu reduce and scan kernel
    uint *d_in, *d_out;
    initKeys(h_input.data(), count, 0, count);
    start = std::clock();
    cudaMalloc(&d_in, sizeof(uint) * count);
    cudaMalloc(&d_out, sizeof(uint) * count);
    cudaMemcpy(d_in, h_input.data(), count * sizeof(uint), cudaMemcpyHostToDevice);
    radix_sort(d_out, d_in, count);
    double rs_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "Reduce and Scan time: " << rs_duration << " s" << std::endl;
    cudaMemcpy(h_output.data(), d_out, sizeof(uint) * count, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);

// record gpu oners_durationsweep kernel
    initKeys(h_input.data(), count, 0, count);
    start = std::clock();
    onesweepRadixSort(h_input.data(), h_output.data(), count);
    double onesweep_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "Onesweep: time: " << onesweep_time << " s" << std::endl;

// record cub sort kernel
    // uint *d_in, *d_out;
    initKeys(h_input.data(), count, 0, count);
    start = std::clock();
    cudaMalloc((void**)&d_in, count * sizeof(uint));
    cudaMalloc((void**)&d_out, count * sizeof(uint));
    cudaMemcpy(d_in, h_input.data(), count * sizeof(uint), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, count);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, count);
    double cub_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "CUB time: " << cub_time << " s" << std::endl;
    cudaFree(d_out);
    cudaFree(d_in);


#ifdef DEBUG
    printVec("Input", h_input);
    printVec("Sorted Output", h_output);
#endif

    return 0;
}
