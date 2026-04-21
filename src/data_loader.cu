#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t* load_image_flat(const char* path, int* out_n) {
    int w, h, channels;

    uint8_t* img = stbi_load(path, &w, &h, &channels, 1);

    if (!img) {
        fprintf(stderr, "ERROR: stbi_load failed for %s\n", path);
        exit(1);
    }

    *out_n = w * h;

    printf("[DataLoader] Loaded %s | %dx%d | channels_raw=%d | pixels=%d\n",
           path, w, h, channels, *out_n);

    printf("[DataLoader] Sample pixels [0,1,2,3,4]: %u %u %u %u %u\n",
           img[0], img[1], img[2], img[3], img[4]);

    return img;
}
