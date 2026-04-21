#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define NUM_BINS 256

// Serial histogram — ground truth
void cpu_histogram(const uint8_t* data, int n, unsigned int* hist) {
    memset(hist, 0, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < n; i++) {
        hist[data[i]]++;
    }
}

// Serial prefix sum (inclusive scan) to build CDF
void cpu_prefix_sum(const unsigned int* hist, unsigned int* cdf) {
    cdf[0] = hist[0];
    for (int i = 1; i < NUM_BINS; i++) {
        cdf[i] = cdf[i-1] + hist[i];
    }
}

// Validation — compares two arrays of length `len`
// Prints PASS or the first failing index with both values
int validate(const unsigned int* ref, const unsigned int* gpu, int len, const char* label) {
    for (int i = 0; i < len; i++) {
        if (ref[i] != gpu[i]) {
            printf("[FAIL] %s mismatch at index %d: CPU=%u GPU=%u\n",
                   label, i, ref[i], gpu[i]);
            return 0;
        }
    }
    printf("[PASS] %s\n", label);
    return 1;
}