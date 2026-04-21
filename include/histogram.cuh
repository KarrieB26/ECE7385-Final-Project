#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include <stdint.h>
#include <cuda_runtime.h>

// Naive kernel — global memory atomics only
__global__ void histogram_naive(const uint8_t* data, unsigned int* hist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        atomicAdd(&hist[data[i]], 1);
    }
}

#endif