#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include "../include/histogram.cuh"
#include "../include/cpu_baseline.cuh"

int main(int argc, char** argv) {
    int n = 1 << 20; // 1M elements
    size_t bytes = n * sizeof(uint8_t);

    // Host input — controlled uniform distribution
    uint8_t* h_data = (uint8_t*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_data[i] = (uint8_t)(i % 256);
    }

    // CPU and GPU histogram outputs
    unsigned int h_hist_cpu[NUM_BINS] = {0};
    unsigned int h_hist_gpu[NUM_BINS] = {0};

    // Run CPU baseline
    cpu_histogram(h_data, n, h_hist_cpu);

    // Device allocations
    uint8_t* d_data;
    unsigned int* d_hist;
    cudaCheckError(cudaMalloc(&d_data, bytes));
    cudaCheckError(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));

    // H2D transfer
    cudaCheckError(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    printf("[Config] n=%d bins=%d threads=%d blocks=%d\n",
           n, NUM_BINS, threads, blocks);

    // Launch kernel
    cudaCheckError(cudaEventRecord(start));
    histogram_naive<<<blocks, threads>>>(d_data, d_hist, n);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaEventSynchronize(stop));

    float ms = 0.0f;
    cudaCheckError(cudaEventElapsedTime(&ms, start, stop));
    printf("[Naive GPU] Kernel time: %.4f ms\n", ms);

    // D2H result
    cudaCheckError(cudaMemcpy(h_hist_gpu, d_hist,
                  NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Exact CPU vs GPU validation
    validate(h_hist_cpu, h_hist_gpu, NUM_BINS, "Histogram matches CPU");

    // Optional sanity sum
    unsigned int total = 0;
    for (int i = 0; i < NUM_BINS; i++) total += h_hist_gpu[i];
    printf("[Naive GPU] Sum of histogram bins (must equal %d): %u\n", n, total);

    // Cleanup
    cudaCheckError(cudaFree(d_data));
    cudaCheckError(cudaFree(d_hist));
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));
    free(h_data);

    return 0;
}