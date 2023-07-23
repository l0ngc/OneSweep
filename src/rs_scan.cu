#include "rs_scan.h"

#define MAX_BLOCK_SZ 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__global__
void gpu_add_block_sums(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const size_t numElems)
{
    unsigned int d_block_sum_val = d_block_sums[blockIdx.x];
    unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
    }
}

__global__
void gpu_prescan(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const unsigned int len,
    const unsigned int shmem_sz,
    const unsigned int max_elems_per_block)
{
    extern __shared__ unsigned int s_out[];
    int thid = threadIdx.x;
    int ai = thid;
    int bi = thid + blockDim.x;
    s_out[thid] = 0;
    s_out[thid + blockDim.x] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory size
    //  must be a 2 * blockDim.x + blockDim.x/num_banks
    s_out[thid + blockDim.x + (blockDim.x >> LOG_NUM_BANKS)] = 0;
    
    __syncthreads();
    
    // Copy input data to shared memory
    // to avoid bankconflict, because memory store in gpu are in banks
    // so it is necessary
    // bank conflict, because one bank can only suffer one thread access as one time
    // however, so, if bank conflict happens, the trainning will be seriel, then the performance will be one step worse
    // bacause we are using the tree way to access memories, so the worst case, the bank conflict problem will be very serios
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
    if (cpy_idx < len)
    {
        s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < len)
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
    }

    // For both upsweep and downsweep:
    // Sequential indices with conflict free padding
    //  Amount of padding = target index / num banks
    //  This "shifts" the target indices by one every multiple
    //   of the num banks
    // offset controls the stride and starting index of 
    //  target elems at every iteration
    // d just controls which threads are active
    // Sweeps are pivoted on the last element of shared memory

    // Upsweep/Reduce step
    int offset = 1;
    for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0) 
    { 
        d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
        s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (int d = 1; d < max_elems_per_block; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int temp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += temp;
        }
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (cpy_idx < len)
    {
        d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
        if (cpy_idx + blockDim.x < len)
            d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
    }
}
 
void sum_scan_blelloch(unsigned int* const d_out,
    const unsigned int* const d_in,
    const size_t numElems)
{
    // Zero out d_out
    cudaMemset(d_out, 0, numElems * sizeof(unsigned int));

    // Set up number of threads and blocks
    
    unsigned int block_sz = MAX_BLOCK_SZ / 2;
    unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
    //unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
    // UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
    //  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
    unsigned int grid_sz = numElems / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (numElems % max_elems_per_block != 0) 
        grid_sz += 1;

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    unsigned int* d_block_sums;
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz);

    // Sum scan data allocated to each block
    //gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
    gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_out, 
                                                                    d_in, 
                                                                    d_block_sums, 
                                                                    numElems, 
                                                                    shmem_sz,
                                                                    max_elems_per_block);

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if (grid_sz <= max_elems_per_block)
    {
        unsigned int* d_dummy_blocks_sums;
        cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int));
        cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int));
        //gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
        gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_block_sums, 
                                                                    d_block_sums, 
                                                                    d_dummy_blocks_sums, 
                                                                    grid_sz, 
                                                                    shmem_sz,
                                                                    max_elems_per_block);
        cudaFree(d_dummy_blocks_sums);
    }
    // Else, recurse on this same function as you'll need the full-blown scan
    //  for the block sums
    else
    {
        unsigned int* d_in_block_sums;
        cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz);
        cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice);
        sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
        cudaFree(d_in_block_sums);
    }

    gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

    cudaFree(d_block_sums);
}
