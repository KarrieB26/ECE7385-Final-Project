#ifndef CPU_BASELINE_CUH
#define CPU_BASELINE_CUH

#include <stdint.h>

void cpu_histogram(const uint8_t* data, int n, unsigned int* hist);
void cpu_prefix_sum(const unsigned int* hist, unsigned int* cdf);
int validate(const unsigned int* ref, const unsigned int* gpu, int len, const char* label);

#endif