#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <string.h>
#include <math.h>
#include "render.h"

void composite(float *target, float* image, float red, float green, float blue, int len) {
    int x;
    for (x=0; x<len; x++) {
        target[x*3] += (image[x] * red); 
        target[x*3+1] += (image[x] * green); 
        target[x*3+2] += (image[x] * blue);
    }
}

void composite16(uint32_t *target, uint16_t* image, float red, float green, float blue, int len) {
    int x;
    for (x=0; x<len; x++) {
        target[x*3] += image[x] * red;
        target[x*3+1] += image[x] * green;
        target[x*3+2] += image[x] * blue;
    }
}

float* image_as_float(uint16_t* target, int len) {
    float *out = (float *)malloc(len * sizeof(float));
    for (int x=0; x<len; x++) {
        out[x] = target[x] / 65535.0f;
    }
    return out;
}

void rescale_intensity(float* target, float imin, float imax, int len) {
    clip(target, imin, imax, 1, len);

    float factor = 1.0f / (imax - imin);

    for (int x=0; x<len; x++) {
        target[x] -= imin;
        target[x] *= factor;
    }
}

void rescale_intensity16(uint16_t* target, int imin, int imax, int len) {
    clip16(target, imin, imax, 1, len);

    float factor = 65535.0f / (imax - imin);

    for (int x=0; x<len; x++) {
        target[x] -= imin;
        target[x] = (uint16_t)(factor*target[x]);
    }
}

void clip(float* target, float min, float max, int channels, int len) {
    for (int x=0; x<len*channels; x++) {
        const float t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip16(uint16_t* target, uint16_t min, uint16_t max, int channels, int len) {
    for (int x=0; x<len*channels; x++) {
        const uint16_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip32(uint32_t* target, uint16_t min, uint16_t max, int channels, int len) {
    for (int x=0; x<len*channels; x++) {
        const uint32_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip32_conv8(uint32_t* target, uint8_t* output, uint16_t min, uint16_t max, int channels, int len) {
    for (int x=0; x<len*channels; x++) {
        const uint32_t t = target[x] < min ? min : target[x];
        output[x] = (uint8_t)((t > max ? max : t) / 256);
    }
}