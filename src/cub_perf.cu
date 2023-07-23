#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <read_data.hpp>
#include <cub/cub.cuh>
#include <iomanip>

int main() {
    std::vector<std::vector<float>> sortedData;
    std::vector<std::vector<float>> unorderedData;

    process_data(sortedData, unorderedData);

    // 使用 CUB 的 Radix Sort 进行测速
    for (int i = 0; i < sortedData.size(); ++i) {
        const std::vector<float>& data = sortedData[i];
        int num_items = data.size();
        float keys_out[num_items];

        float *d_keys_in, *d_keys_out;
        cudaMalloc((void**)&d_keys_in, num_items * sizeof(float));
        cudaMalloc((void**)&d_keys_out, num_items * sizeof(float));

        cudaMemcpy(d_keys_in, data.data(), num_items * sizeof(float), cudaMemcpyHostToDevice);

        // Determine temporary device storage requirements
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run sorting operation and measure time
        auto start = std::chrono::steady_clock::now();
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        // Copy sorted keys back to host
        cudaMemcpy(keys_out, d_keys_out, num_items * sizeof(float), cudaMemcpyDeviceToHost);

        // Output sorted keys
        // for (int i = 0; i < 10; ++i) {
        //     std::cout << keys_out[i] << " ";
        // }
        // std::cout << std::endl;

        // Output data size and computation time
        std::cout << "Data size: " << num_items << std::endl;
        std::cout << "Computation time: " << std::fixed << std::setprecision(6) << elapsed_seconds.count() << " seconds" << std::endl;

        // Free device memory
        cudaFree(d_keys_in);
        cudaFree(d_keys_out);
        cudaFree(d_temp_storage);
    }
}