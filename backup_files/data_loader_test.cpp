#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t* load_image_flat(const char* path, int* out_n) {
    int w, h, channels;

    uint8_t* img = stbi_load(path, &w, &h, &channels, 1);

    if (!img) {
        fprintf(stderr, "ERROR: Could not load %s\n", path);
        return NULL;
    }

    *out_n = w * h;

    printf("[SUCCESS] Loaded: %s\n", path);
    printf("Dimensions: %d x %d\n", w, h);
    printf("Pixel Count: %d\n", *out_n);
    printf("Original Channels: %d\n", channels);

    printf("Sample Pixels: ");
    for (int i = 0; i < 5; i++) {
        printf("%u ", img[i]);
    }
    printf("\n");

    return img;
}

// int main() {
//     int n;
//     const char* path = "../data/chest_xray/train/NORMAL/IM-0115-0001.jpeg";

//     uint8_t* pixels = load_image_flat(path, &n);

//     if (pixels) {
//         printf("Audit Passed: Image loaded correctly.\n");
//         stbi_image_free(pixels);
//     }

//     return 0;
// }
