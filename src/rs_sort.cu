#include "rs_sort.h"

#define MAX_BLOCK_SZ 128

__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    // need shared memory array for:
    // - block's share of the input data (local sort will be put here too)
    // - mask outputs
    // - scanned mask outputs
    // - merged scaned mask outputs ("local prefix sum")
    // - local sums of scanned mask outputs
    // - scanned local sums of scanned mask outputs

    // for all radix combinations:
    //  build mask output for current radix combination
    //  scan mask ouput
    //  store needed value from current prefix sum array to merged prefix sum array
    //  store total sum of mask output (obtained from scan) to global block sum array
    // calculate local sorted address from local prefix sum and scanned mask output's total sums
    // shuffle input block according to calculated local sorted addresses
    // shuffle local prefix sums according to calculated local sorted addresses
    // copy locally sorted array back to global memory
    // copy local prefix sum array back to global memory
    // 所以说这一大部分代码，就对应着整个方格的小方块
    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;
    // s_mask_out[] will be scanned in place
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = &s_data[max_elems_per_block];
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

    unsigned int thid = threadIdx.x;
    // 将数据从全局输入数据，拷贝到shared_dat里面，来增加访问速度
    // Copy block's portion of global input data to shared memory
    // 拷贝数据
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;

    __syncthreads();

    // To extract the correct 2 bits, we first shift the number
    //  to the right until the correct 2 bits are in the 2 LSBs,
    //  then mask on the number with 11 (3) to remove the bits
    //  on the left
    unsigned int t_data = s_data[thid];
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
    // 这里处理的事2bits，也就是4位
    // 每次处理一个数值，也就是一个value
    for (unsigned int i = 0; i < 4; ++i)
    {
        // Zero out s_mask_out
        // 每次申城一个特殊的mask来进行处理，一个是2bits，也就是4种可能的数值
        // 先每一个都置0
        s_mask_out[thid] = 0;
        if (thid == 0)
            s_mask_out[s_mask_out_len - 1] = 0;

        __syncthreads();
        // 进行bit mask，对数据进行处理
        // build bit mask output
        bool val_equals_i = false;
        // 所以i就是通道树目，每一次通过i进行mask
        // 这就得到了mask的结果，存放在s_mask_out里面，一堆0和1，对应着2ii
        // 这一部分的mask有两个目的，一个是计数，一个是本地的shuffle
        // 计数的话，是每一个pass进行prefix sum之后，将得到的值，放到外边block sum数组对应的位置
        // for the masking ~
        if (cpy_idx < d_in_len)
        {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        // Scan mask outputs (Hillis-Steele)
        // 计算下来prefix sum，也就是scan结果
        // 这一部分完成一次Fb3，也就是做一次scan，根据前一步mask出来的结果，这里的scan是inclusive的
        int partner = 0;
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            partner = thid - (1 << d);
            if (partner >= 0) {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }
        // 这个时候最后一位，对应的就是一共有多少个这个数值
        // Shift elements to produce the same effect as exclusive scan
        // 对数据进行向右的shift，也就是每个thread都把数据拿出来，然后往右放一格
        // 对scan的结果，向右挪一格，调整成exlusive的scan
        unsigned int cpy_val = 0;
        cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();
        // 最后一位，应该对应的是这个pass的数量
        // 这里相当于把局部的，单个mask的scan结果，输出到总体上，输出到外层
        // 所以 d_clock_sums就对应着？？
        if (thid == 0)
        {
            // Zero out first element to produce the same effect as exclusive scan
            // 这里的total sum就对应着，每一个pass一共总数有多少个，并且把这个total sum，放到block sum的数组里面
            // blocksum的数组，应该是这么定义的，通过i的树木，也就是0～4，分成4块，每一块内部应该是blockIdx的长度，也就是gridDim
            // 这里的total sum就是这个pass，比如0，一共有多少个
            // so, the s_mask_out is already prefixed, then the last number is the final value of our data
            s_mask_out[0] = 0;
            
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();
        //这一部分我还不知道干啥用的，我再看看，我估计是用来局部shuffle的
        // 代码的目的是将数据写出来到s_merged_scan_mask_out
        // 如果想局部进行排序，其实本身的prescan的结果，就能用来排序了吧
        // 现在就是把局部的prescan结果写出来
        if (val_equals_i && (cpy_idx < d_in_len))
        {
            // 这里一个thid只会实现一次
            // 所以这里，就是记录自己前边有多少个同种元素！！！
            s_merged_scan_mask_out[thid] = s_mask_out[thid];
        }

        __syncthreads();
    }

    // Scan mask output sums
    // Just do a naive scan since the array is really small
    // 只有一个线程这样操作，好
    // s_mask_out_sums里面记录了，每一个pass的总数
    // 这里就是记录了，每一个block局部，从0～3的数字个数
    // 所以s_scan_mask_out_sums里面就记录了，每一个格子里面有多少数据
    if (thid == 0)
    {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();
    // 将计算好的prefix sum，还有排序好的结果，写回去
    // 计算好thid
    // 所以
    // 这一部分是用来进行局部排序和shuffle的
    // 上边那一步就是分桶，这一步就是从桶里面取出数据，找到新的坐标
    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        // 这一步的话就是先讲局部的数据根据scan来确定好位置对么
        // 所以到这里，每一个都找到了自己所需要的下一个position，然后把数据放过去了
        s_data[new_pos] = t_data;
        // 相当于更换一下位置，更换一下自己t_prefix_sum的位置
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;
        
        __syncthreads();
        // 最后将所需要的数据再贴回来，用来接下来的全局的radix sort
        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global 
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
        d_out_sorted[cpy_idx] = s_data[thid];
    }
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    // 把数据放到right positions
    // 这一部分的原理就是将数据根据Pd[n] + m放到正确的地方
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out
    // block.x用来表示自己在第n个bar，也就是chunk n
    // 需要这个值，也就是d
    // 需要上一步获得的d_scan_block_sums
    // 需要找到他在prefix sums里面自己对应的数值，也就是m
    // Pd[n] + m
    // d_scan_block_sums[d * gridDim.x + blockIdx.x] + prefix_sums[idx]

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        // extract out first 2 digits
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];

        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
            + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len)
{
    unsigned int block_sz = MAX_BLOCK_SZ;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = d_in_len / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % max_elems_per_block != 0)
        grid_sz += 1;

    unsigned int* d_prefix_sums;
    unsigned int d_prefix_sums_len = d_in_len;
    cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len);
    cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len);

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = 4 * grid_sz; // 4-way split
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int* d_scan_block_sums;
    cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_data_len = max_elems_per_block;
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = 4; // 4-way split
    unsigned int s_scan_mask_out_sums_len = 4;
    unsigned int shmem_sz = (s_data_len 
                            + s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);


    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2)
    {
        // 先进性局部的基数排序
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out, 
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                d_in_len, 
                                                                max_elems_per_block);

        //unsigned int* h_test = new unsigned int[d_in_len];
        //checkCudaErrors(cudaMemcpy(h_test, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost));
        //for (unsigned int i = 0; i < d_in_len; ++i)
        //    std::cout << h_test[i] << " ";
        //std::cout << std::endl;
        //delete[] h_test;
        // 对块和数组进行
        // scan global block sum array
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        // 将排序好的块进行重新排序
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in, 
                                                    d_out, 
                                                    d_scan_block_sums, 
                                                    d_prefix_sums, 
                                                    shift_width, 
                                                    d_in_len, 
                                                    max_elems_per_block);
    }
    cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);

    cudaFree(d_scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(d_prefix_sums);
}
