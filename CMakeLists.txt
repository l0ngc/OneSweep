
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(radix_sort)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(include)

# add_executable(radix_sort src/main.cu)
# add_executable(cub_sort src/cub_perf.cu)
# #naive scan
# add_executable(naive src/naive_scan.cu)
# # doubel buffer naive
# add_executable(double_buffer_naive src/double_buffer_naive_scan.cu)
# add_executable(belloc src/belloc_scan.cu)

# add_executable(cpu_onesweep src/cpu_onesweep.cpp)
add_executable(gpu_onesweep src/gpu_onesweep.cu src/onesweep.cu src/rs_scan.cu src/rs_sort.cu)
# add_executable(exclusive src/exclusive_scan.cu)