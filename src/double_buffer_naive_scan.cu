#include <cub/cub.cuh>
#include <iostream>
#include <cmath>
#include <iostream>

#include <cub/cub.cuh>
#include <iostream>
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 256
// 这里是通过定义了一个double buffer的共享内存，来实现数据之间的共享
__global__ void scan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    // printf("My thid:%d", thid);
    int pout = 0, pin = 1;

    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    // temp[pout * BLOCK_SIZE + thid] = (thid > 0) ? g_idata[thid] : 0;
    temp[pout * BLOCK_SIZE + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        printf("My offset:%d", offset);
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (thid >= offset)
            temp[pout * BLOCK_SIZE + thid] += temp[pin * BLOCK_SIZE + thid - offset];
        else
            temp[pout * BLOCK_SIZE + thid] = temp[pin * BLOCK_SIZE + thid];
        __syncthreads();
    }

    g_odata[thid] = temp[pout * BLOCK_SIZE + thid]; // write output
}


int main() {
    int n = 8;
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[n];

    float *device_input, *device_output;
    cudaMalloc((void**)&device_input, sizeof(float) * n);
    cudaMalloc((void**)&device_output, sizeof(float) * n);

    cudaMemcpy(device_input, input, sizeof(float) * n, cudaMemcpyHostToDevice);

    scan<<<1, n, sizeof(float) * n>>>(device_output, device_input, n);

    cudaMemcpy(output, device_output, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    std::cout << "Input: ";
    for (int i = 0; i < n; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < n; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
