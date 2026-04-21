#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include "../include/histogram.cuh"
#include "../include/cpu_baseline.cuh"

// data loader
extern uint8_t* load_image_flat(const char* path, int* out_n);

int main(int argc, char** argv) {
    // Define a list of images to benchmark
    const char* image_paths[] = {
        "data/chest_xray/train/NORMAL/IM-0115-0001.jpeg",
        "data/chest_xray/train/NORMAL/IM-0117-0001.jpeg",
        "data/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"
    };

    int num_images = sizeof(image_paths) / sizeof(image_paths[0]);

    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        int n;
        const char* img_path = image_paths[img_idx];
        
        printf("\n>>> Benchmarking Image [%d/%d]: %s\n", img_idx + 1, num_images, img_path);
        
        // 1. Load Data (Sneha)
        uint8_t* h_data = load_image_flat(img_path, &n);

        // 2. CPU Baseline (Karrie)
        unsigned int h_hist_cpu[NUM_BINS] = {0};
        cpu_histogram(h_data, n, h_hist_cpu);

        // 3. Device Setup & H2D (Tri)
        unsigned char* d_data;
        unsigned int* d_hist;
        cudaCheckError(cudaMalloc(&d_data, n * sizeof(unsigned char)));
        cudaCheckError(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
        cudaCheckError(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
        cudaCheckError(cudaMemcpy(d_data, h_data, n * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // 4. GPU Execution & Timing
        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        cudaCheckError(cudaEventRecord(start));
        histogram_naive<<<blocks, threads>>>(d_data, d_hist, n);
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaDeviceSynchronize());

        float ms = 0.0f;
        cudaCheckError(cudaEventElapsedTime(&ms, start, stop));

        // 5. Results & Validation
        unsigned int h_hist_gpu[NUM_BINS] = {0};
        cudaCheckError(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        printf("[Result] Size: %d px | Time: %.4f ms\n", n, ms);
        validate(h_hist_cpu, h_hist_gpu, NUM_BINS, "Histogram Match Check");

        // 6. Cleanup per image
        cudaCheckError(cudaFree(d_data));
        cudaCheckError(cudaFree(d_hist));
        cudaCheckError(cudaEventDestroy(start));
        cudaCheckError(cudaEventDestroy(stop));
        free(h_data); 
    }

    printf("\nAll images processed successfully!\n");
    return 0;
}