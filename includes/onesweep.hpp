#pragma once
#include <cub/cub.cuh>
#include "cuda_runtime.h"
#include <iostream>
#define DIGIT_SIZE 8
#define KEY_SIZE 32

const uint NUM_DIGIT_BINNINGS = 1 << DIGIT_SIZE;
const uint NUM_DIGIT_PLACES = KEY_SIZE / DIGIT_SIZE;

void onesweepRadixSort(uint* h_input, uint* h_output, uint count);