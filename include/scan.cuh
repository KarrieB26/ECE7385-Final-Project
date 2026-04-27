#ifndef SCAN_CUH
#define SCAN_CUH

#include <stdint.h>
#include <cuda_runtime.h>
#include "utils.cuh"

// Kogge-Stone inclusive scan for NUM_BINS-sized histogram
__global__ void kogge_stone_scan(unsigned int* d_data, const int n) {
    __shared__ unsigned int temp[NUM_BINS];

    int tid = threadIdx.x;

    if (tid < NUM_BINS) {
        temp[tid] = (tid < n) ? d_data[tid] : 0;
    }

    __syncthreads();

    for (int stride = 1; stride < n; stride *= 2) {
        unsigned int val = 0;

        if (tid >= stride && tid < n) {
            val = temp[tid - stride];
        }

        __syncthreads();

        if (tid < n) {
            temp[tid] += val;
        }

        __syncthreads();
    }

    if (tid < n) {
        d_data[tid] = temp[tid];
    }
}

// Brent-Kung inclusive scan for NUM_BINS-sized histogram
__global__ void brent_kung_scan(unsigned int* d_data, const int n) {
    __shared__ unsigned int temp[NUM_BINS];

    int tid = threadIdx.x;

    if (tid < NUM_BINS) {
        temp[tid] = (tid < n) ? d_data[tid] : 0;
    }

    __syncthreads();

    // Reduction / upsweep phase
    for (int stride = 1; stride <= n / 2; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;

        if (idx < n) {
            temp[idx] += temp[idx - stride];
        }

        __syncthreads();
    }

    // Downsweep-style accumulation phase for inclusive scan
    for (int stride = n / 4; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;

        if (idx + stride < n) {
            temp[idx + stride] += temp[idx];
        }

        __syncthreads();
    }

    if (tid < n) {
        d_data[tid] = temp[tid];
    }
}

// Histogram equalization mapping kernel
__global__ void equalize_mapping(const uint8_t* __restrict__ d_in,
                                 uint8_t* d_out,
                                 const unsigned int* __restrict__ d_cdf,
                                 const int n,
                                 const int total_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        unsigned int cdf_val = d_cdf[d_in[i]];
        float normalized = ((float)cdf_val / (float)total_pixels) * 255.0f;

        if (normalized > 255.0f) normalized = 255.0f;
        if (normalized < 0.0f) normalized = 0.0f;

        d_out[i] = (uint8_t)normalized;
    }
}

#endif