// #include <iostream>
// #include <string>
// #include <vector>
// #include <chrono>
// #include <read_data.hpp>
// #include <iomanip>


// int main() {
//     std::string basePath = "../data/";
//     std::vector<std::string> sortedFilenames = {
//         "float_sorted_2^10.csv", "float_sorted_2^11.csv", "float_sorted_2^12.csv",
//         "float_sorted_2^13.csv", "float_sorted_2^14.csv", "float_sorted_2^15.csv",
//         "float_sorted_2^16.csv", "float_sorted_2^17.csv", "float_sorted_2^18.csv",
//         "float_sorted_2^19.csv", "float_sorted_2^20.csv"
//     };
//     std::vector<std::string> unorderedFilenames = {
//         "float_unorder_2^10.csv", "float_unorder_2^11.csv", "float_unorder_2^12.csv",
//         "float_unorder_2^13.csv", "float_unorder_2^14.csv", "float_unorder_2^15.csv",
//         "float_unorder_2^16.csv", "float_unorder_2^17.csv", "float_unorder_2^18.csv",
//         "float_unorder_2^19.csv", "float_unorder_2^20.csv"
//     };

//     std::vector<std::vector<float>> sortedData;
//     std::vector<std::vector<float>> unorderedData;

//     process_data(basePath, sortedFilenames, unorderedFilenames, sortedData, unorderedData);

// }

#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
#include <iostream>

int main()
{
    int num_items = 7;
    int keys_in[] = {8, 6, 7, 5, 3, 0, 9};
    int keys_out[num_items];

    // Declare and allocate device-accessible pointers for sorting data
    int *d_keys_in, *d_keys_out;
    cudaMalloc((void**)&d_keys_in, num_items * sizeof(int));
    cudaMalloc((void**)&d_keys_out, num_items * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_keys_in, keys_in, num_items * sizeof(int), cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

    // Copy sorted keys back to host
    cudaMemcpy(keys_out, d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);

    // Output sorted keys
    for (int i = 0; i < num_items; ++i) {
        std::cout << keys_out[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_temp_storage);

    return 0;
}
