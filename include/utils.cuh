#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro — wrap every CUDA call with this
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s — %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define NUM_BINS 256

#endif