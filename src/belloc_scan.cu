#include <iostream>

__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = g_idata[2 * thid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();

        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    if (thid == 0) {
        temp[n - 1] = 0; // clear the last element
    }

    for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();

        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    // write back data, from shared memory to device memory
    g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
    g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

int main() {
    int n = 8;
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[n];

    float *device_input, *device_output;
    cudaMalloc((void**)&device_input, sizeof(float) * n);
    cudaMalloc((void**)&device_output, sizeof(float) * n);

    cudaMemcpy(device_input, input, sizeof(float) * n, cudaMemcpyHostToDevice);

    prescan<<<1, n/2, sizeof(float) * n>>>(device_output, device_input, n);

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
