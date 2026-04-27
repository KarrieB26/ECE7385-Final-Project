#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../include/utils.cuh"
#include "../include/scan.cuh"

void cpu_prefix_sum_test(const unsigned int* hist, unsigned int* cdf, int n) {
    cdf[0] = hist[0];
    for (int i = 1; i < n; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
}

int validate_array(const unsigned int* ref, const unsigned int* test, int n, const char* label) {
    for (int i = 0; i < n; i++) {
        if (ref[i] != test[i]) {
            printf("[FAIL] %s mismatch at index %d: CPU=%u GPU=%u\n",
                   label, i, ref[i], test[i]);
            return 0;
        }
    }

    printf("[PASS] %s\n", label);
    return 1;
}

int main() {
    const int n = NUM_BINS;

    unsigned int h_hist[NUM_BINS];
    unsigned int h_cpu_cdf[NUM_BINS];
    unsigned int h_ks_cdf[NUM_BINS];
    unsigned int h_bk_cdf[NUM_BINS];

    // Simple controlled histogram input
    // hist = [1, 2, 3, 4, ...]
    for (int i = 0; i < NUM_BINS; i++) {
        h_hist[i] = i + 1;
    }

    cpu_prefix_sum_test(h_hist, h_cpu_cdf, n);

    unsigned int* d_scan;
    cudaCheckError(cudaMalloc(&d_scan, NUM_BINS * sizeof(unsigned int)));

    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    float ks_ms = 0.0f;
    float bk_ms = 0.0f;

    // ------------------------------
    // Test Kogge-Stone
    // ------------------------------
    cudaCheckError(cudaMemcpy(d_scan, h_hist,
                              NUM_BINS * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

    cudaCheckError(cudaEventRecord(start));
    kogge_stone_scan<<<1, NUM_BINS>>>(d_scan, n);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaEventSynchronize(stop));
    cudaCheckError(cudaEventElapsedTime(&ks_ms, start, stop));

    cudaCheckError(cudaMemcpy(h_ks_cdf, d_scan,
                              NUM_BINS * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

    printf("[Kogge-Stone] Time: %.6f ms\n", ks_ms);
    validate_array(h_cpu_cdf, h_ks_cdf, n, "Kogge-Stone CDF correct");

    // ------------------------------
    // Test Brent-Kung
    // ------------------------------
    cudaCheckError(cudaMemcpy(d_scan, h_hist,
                              NUM_BINS * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

    cudaCheckError(cudaEventRecord(start));
    brent_kung_scan<<<1, NUM_BINS>>>(d_scan, n);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaEventSynchronize(stop));
    cudaCheckError(cudaEventElapsedTime(&bk_ms, start, stop));

    cudaCheckError(cudaMemcpy(h_bk_cdf, d_scan,
                              NUM_BINS * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

    printf("[Brent-Kung] Time: %.6f ms\n", bk_ms);
    validate_array(h_cpu_cdf, h_bk_cdf, n, "Brent-Kung CDF correct");

    // ------------------------------
    // Test equalization mapping
    // ------------------------------
    const int num_pixels = 16;
    uint8_t h_in[num_pixels] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
    };

    uint8_t h_out[num_pixels];

    uint8_t* d_in;
    uint8_t* d_out;
    unsigned int* d_cdf;

    cudaCheckError(cudaMalloc(&d_in, num_pixels * sizeof(uint8_t)));
    cudaCheckError(cudaMalloc(&d_out, num_pixels * sizeof(uint8_t)));
    cudaCheckError(cudaMalloc(&d_cdf, NUM_BINS * sizeof(unsigned int)));

    cudaCheckError(cudaMemcpy(d_in, h_in,
                              num_pixels * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));

    // Use Kogge-Stone result as CDF for mapping test
    cudaCheckError(cudaMemcpy(d_cdf, h_ks_cdf,
                              NUM_BINS * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;

    equalize_mapping<<<blocks, threads>>>(d_in, d_out, d_cdf, num_pixels, h_cpu_cdf[NUM_BINS - 1]);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(h_out, d_out,
                              num_pixels * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost));

    printf("[Mapping] Input -> Output sample:\n");
    for (int i = 0; i < num_pixels; i++) {
        printf("  %u -> %u\n", h_in[i], h_out[i]);
    }

    printf("[PASS] Mapping kernel executed\n");

    cudaCheckError(cudaFree(d_scan));
    cudaCheckError(cudaFree(d_in));
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_cdf));
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    return 0;
}