#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Define your constants here
#define DIGIT_SIZE 3

// Define your CUDA kernel functions here
__global__ void exclusiveSum(uint* histogram);

__global__ void exclusiveSum(uint* histogram)
{
    uint local_id = threadIdx.x;
    uint num_digit_binnings = 1 << DIGIT_SIZE;
    uint sum = 0;

    for (uint i = 0; i < num_digit_binnings; i++) {
        uint tmp = histogram[local_id * num_digit_binnings + i];
        histogram[local_id * num_digit_binnings + i] = sum;
        sum += tmp;
    }
}


int main()
{
    // Initialize CUDA device and context

    // Define your input data size and other variables
    const unsigned int num_digit_binnings = 1 << DIGIT_SIZE; // Example: number of digit binnings
    const unsigned int size = num_digit_binnings * sizeof(uint); // Example: size of input data in bytes

    // Allocate and initialize input data on the host
    std::vector<uint> h_histogram(num_digit_binnings);
    // Initialize input data with random values
    for (unsigned int i = 0; i < num_digit_binnings; i++) {
        h_histogram[i] = rand() % 10; // Example: random values between 0 and 9
    }

    // Print the Input
    std::cout << "Input Series:" << std::endl;
    for (unsigned int i = 0; i < num_digit_binnings; i++) {
        std::cout << h_histogram[i] << " ";
    }
    std::cout << std::endl;
    
    // Create CUDA device pointers for input data
    uint* d_histogram;

    // Allocate memory on the GPU for input data
    cudaMalloc((void**)&d_histogram, size);

    // Copy input data from host to device
    cudaMemcpy(d_histogram, h_histogram.data(), size, cudaMemcpyHostToDevice);

    // Launch the exclusiveSum kernel
    exclusiveSum<<<1, num_digit_binnings>>>(d_histogram);

    // Copy the result back from device to host
    cudaMemcpy(h_histogram.data(), d_histogram, size, cudaMemcpyDeviceToHost);




    // Print the result
    std::cout << "Exclusive Prefix Sum:" << std::endl;
    for (unsigned int i = 0; i < num_digit_binnings; i++) {
        std::cout << h_histogram[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory on the GPU
    cudaFree(d_histogram);

    // Cleanup and exit
    return 0;
}
