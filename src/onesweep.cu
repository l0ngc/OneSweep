#include "onesweep.hpp"

// #define DEBUG
__global__ void upfrontHistogram(const uint* data,
                                 uint* histogram)
{
    uint global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint local_id = threadIdx.x;

    uint local_data = data[global_id];

    __shared__ uint local_hist[NUM_DIGIT_BINNINGS * (KEY_SIZE / DIGIT_SIZE)];

    if (local_id < NUM_DIGIT_PLACES) {
        for (uint i = 0; i < NUM_DIGIT_BINNINGS; i++) {
            local_hist[local_id * NUM_DIGIT_BINNINGS + i] = 0;
        }
    }
    
    __syncthreads();

    for (uint i = 0; i < NUM_DIGIT_PLACES; i++)
    {
        uint idx = (local_data >> (i * DIGIT_SIZE)) & 0xff;
        atomicAdd(&local_hist[idx + i * NUM_DIGIT_BINNINGS], 1);
    }

    __syncthreads();

    if (local_id < NUM_DIGIT_PLACES) {
        for (int i = 0; i < NUM_DIGIT_BINNINGS; i++) {
            atomicAdd(&histogram[local_id * NUM_DIGIT_BINNINGS + i], local_hist[local_id * NUM_DIGIT_BINNINGS + i]);
        }
    }
}

__global__ void histSum(uint *histogram)
{
    uint local_id = threadIdx.x;
    uint sum = 0;
    if(local_id < NUM_DIGIT_PLACES) {
        for (uint i = 0; i < NUM_DIGIT_BINNINGS; i++)
        {
            uint tmp = histogram[local_id * NUM_DIGIT_BINNINGS + i];
            histogram[local_id * NUM_DIGIT_BINNINGS + i] = sum;
            sum += tmp;
        }
    }
}

__global__ void scanFlags(uint *data,
                                        const uint *histogram,
                                        uint *flag_IPS,
                                        uint digit_place,
                                        uint *output)                        
{
    uint local_id = threadIdx.x;
    uint pid = blockIdx.x;

    __shared__ uint EPS[NUM_DIGIT_BINNINGS];
    __shared__ uint aggregate[NUM_DIGIT_BINNINGS];

    __syncthreads();
    
    uint local_data = data[pid * blockDim.x + local_id];

    if (local_id < NUM_DIGIT_BINNINGS)
    {
        EPS[local_id] = 0;
        aggregate[local_id] = 0;
    }

    __syncthreads();

    uint idx = (local_data >> (digit_place * DIGIT_SIZE)) & 0xff;
    atomicAdd(&aggregate[idx], 1);
    __syncthreads();

    if (local_id < NUM_DIGIT_BINNINGS)
    {
        if (pid > 0)
        {
            flag_IPS[pid * NUM_DIGIT_BINNINGS + local_id] = (aggregate[local_id] | 0x40000000);// 010 ..., after >> 30, you got 1
        }
        else if (pid == 0)
        {
            EPS[local_id] = histogram[digit_place * NUM_DIGIT_BINNINGS + local_id];
            flag_IPS[local_id] = ((EPS[local_id] + aggregate[local_id]) | 0x80000000); // 100 ..., after >> 30, you got 2
        }
    }
}

__global__ void scanShuffle(uint *data,
                                        const uint *histogram,
                                        uint *flag_IPS,
                                        uint digit_place,
                                        uint *output)                                 
{
    uint pid = blockIdx.x;
    uint local_id = threadIdx.x;

    __shared__ uint EPS[NUM_DIGIT_BINNINGS];
    __syncthreads();
    
    uint local_data = data[pid * blockDim.x + local_id];

    if (local_id < NUM_DIGIT_BINNINGS)
    {
        EPS[local_id] = 0;
        // aggregate[local_id] = 0;
    }

    __syncthreads();

    uint idx = (local_data >> (digit_place * DIGIT_SIZE)) & 0xff;
    // atomicAdd(&aggregate[idx], 1);
    __syncthreads();

    if (local_id < NUM_DIGIT_BINNINGS)
    {
        if (pid == 0)
        {
            EPS[local_id] = histogram[digit_place * NUM_DIGIT_BINNINGS + local_id];
        }
    }
    if (pid > 0 && local_id < NUM_DIGIT_BINNINGS)
    {
        int ppid = pid - 1;
        while (ppid >= 0)
        {
            uint p_IPS = flag_IPS[ppid * NUM_DIGIT_BINNINGS + local_id];
            if ((p_IPS >> (KEY_SIZE - 2)) == 2)
            {
                EPS[local_id] += (p_IPS & 0x3fffffff);
                break;
            }
            else if ((p_IPS >> (KEY_SIZE - 2)) == 1)
            {
                EPS[local_id] += (p_IPS & 0x3fffffff);
                ppid--;                
            }
        }
        flag_IPS[pid*NUM_DIGIT_BINNINGS + local_id] += (EPS[local_id] | 0x40000000);
    }
    __syncthreads();

    for (uint i = 0; i < blockDim.x; i++)
    {
        if (local_id == i)
        {
            output[EPS[idx]] = local_data;
            EPS[idx] = EPS[idx] + 1;
        }
        __syncthreads();
    }   
}

void onesweepRadixSort(uint* h_input, uint* h_output, uint count){
    const uint size = count * sizeof(uint);
    uint* d_input;
    uint* d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    const uint block_size = 256;
    const uint num_blocks = (count + block_size - 1) / block_size;

    uint* d_histogram;
    cudaMalloc((void**)&d_histogram, NUM_DIGIT_BINNINGS * NUM_DIGIT_PLACES * sizeof(uint));
    // std::cout << "Launch upfrontHistogram kernel" << std::endl;
    upfrontHistogram<<<num_blocks, block_size>>>(d_input, d_histogram);
    cudaDeviceSynchronize();

    // Launch histSum kernel
    // std::cout << "Launch histSum kernel" << std::endl;
    histSum<<<1, NUM_DIGIT_PLACES>>>(d_histogram);
    cudaDeviceSynchronize();

    // // Launch chainedScanDigitBinning kernel
    // std::cout << "Launch chainedScanDigitBinning kernel" << std::endl;
    uint* d_flag_IPS;
    cudaMalloc((void**)&d_flag_IPS, NUM_DIGIT_BINNINGS * num_blocks * sizeof(uint));
    cudaMemset(d_flag_IPS, 0, NUM_DIGIT_BINNINGS * num_blocks * sizeof(uint));

    for (uint i = 0; i < NUM_DIGIT_PLACES; i++)
    {
        cudaMemset(d_flag_IPS, 0, NUM_DIGIT_BINNINGS * num_blocks * sizeof(uint));
        scanFlags<<<num_blocks, block_size>>>(d_input, d_histogram, d_flag_IPS, i, d_input);
        cudaDeviceSynchronize();
        scanShuffle<<<num_blocks, block_size>>>(d_input, d_histogram, d_flag_IPS, i, d_input);
    }

    cudaMemcpy(h_output, d_input, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram);
    cudaFree(d_flag_IPS);
}