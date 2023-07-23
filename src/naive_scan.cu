#include <cub/cub.cuh>
#include <iostream>
#include <cmath>
// #define DEBUG
// First attempt, to do simple sum_scan
__global__ void sumScan(int* x, int n, int d) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k < n){
        if (k >= (1 << d-1)){
            #ifdef DEBUG
            printf("%d ", k);
            #endif
            x[k] = x[k - (1 << (d-1))] + x[k];
            }
    }
}

void hostSumScan(int* x, int n) {
    int* device_x;
    cudaMalloc((void**)&device_x, sizeof(int) * n);
    cudaMemcpy(device_x, x, sizeof(int) * n, cudaMemcpyHostToDevice);

    for (int d = 1; d <= ceil(log2(n)); d++) {
        #ifdef DEBUG
        std::cout << "Tier" << d << ":  ";
        for (int i = 0; i < n; i++) {
            std::cout << " " << x[i] << " ";
        }
        #endif
        std::cout << std::endl;

        int numBlocks = (n + 255) / 256;
        sumScan<<<numBlocks, 256>>>(device_x, n, d);
        cudaMemcpy(x, device_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    cudaFree(device_x);
}

int main() {
    int n = 9;
    int x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    hostSumScan(x, n);

    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
