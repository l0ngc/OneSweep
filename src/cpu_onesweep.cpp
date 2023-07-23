#include <iostream>

const int DIGIT_SIZE = 8; // Number of bits per digit
const int NUM_DIGIT_BINNINGS = 1 << DIGIT_SIZE; // Number of bins per digit
const int DIGIT_PLACE = sizeof(unsigned int) * 8 / DIGIT_SIZE; // Number of digit places

void radixSort(unsigned int *keys, int sizeOfKeys) {
    const unsigned int mask = (1 << DIGIT_SIZE) - 1;
    int *hist = (int*)malloc(NUM_DIGIT_BINNINGS * sizeof(int));
    unsigned int *copy = (unsigned int*)malloc(sizeOfKeys * sizeof(unsigned int));

    for (int i = 0; i < DIGIT_PLACE; i++) {
        for (int j = 0; j < NUM_DIGIT_BINNINGS; j++)
            hist[j] = 0;

        for (int g = 0; g < sizeOfKeys; g++) {
            int histroIdx = (keys[g] >> (i * DIGIT_SIZE)) & mask;
            copy[g] = keys[g];
            hist[histroIdx]++;
        }

        int sum = 0;
        for (int s = 0; s < NUM_DIGIT_BINNINGS; s++) {
            int tmp = hist[s];
            hist[s] = sum;
            sum += tmp;
        }

        for (int g = 0; g < sizeOfKeys; g++) {
            unsigned int tmp = (copy[g] >> (i * DIGIT_SIZE)) & mask;
            int newIdx = hist[tmp];
            keys[newIdx] = copy[g];
            hist[tmp]++;
        }
    }
    free(copy);
    free(hist);
}

int main() {
    unsigned int keys[] = { 4293699190, 1680956, 123456789, 987654321 };
    int sizeOfKeys = sizeof(keys) / sizeof(keys[0]);

    std::cout << "Before sorting:" << std::endl;
    for (int i = 0; i < sizeOfKeys; i++) {
        std::cout << keys[i] << " ";
    }
    std::cout << std::endl;

    radixSort(keys, sizeOfKeys);

    std::cout << "After sorting:" << std::endl;
    for (int i = 0; i < sizeOfKeys; i++) {
        std::cout << keys[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
