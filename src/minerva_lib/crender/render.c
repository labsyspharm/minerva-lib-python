#ifdef __cplusplus
extern "C" {
#endif

#include "render.h"
#include <stdio.h>
/**
 * C code which optimizes rendering vs doing the calculations in numpy.
 * The methods can be called from Python with ctypes.
*/

void composite16(uint32_t *target, uint16_t* image, const float red, const float green, const float blue, const int len) {
    int x;
    const uint32_t r = red * 65535.0f;
    const uint32_t g = green * 65535.0f;
    const uint32_t b = blue * 65535.0f;
    for (x=0; x<len; x++) {
        target[x*3] += image[x] * r / 65535;
        target[x*3+1] += image[x] * g / 65535;
        target[x*3+2] += image[x] * b / 65535;
    }
}

void composite32(uint64_t *target, uint32_t* image, const float red, const float green, const float blue, const int len) {
    int x;
    const uint64_t r = red * 4294967295.0;
    const uint64_t g = green * 4294967295.0;
    const uint64_t b = blue * 4294967295.0;
    for (x=0; x<len; x++) {
        target[x*3] += image[x] * r / 4294967295.0;
        target[x*3+1] += image[x] * g / 4294967295.0;
        target[x*3+2] += image[x] * b / 4294967295.0;
    }
}

void composite8(uint16_t *target, uint8_t* image, const float red, const float green, const float blue, const int len) {
    int x;
    const uint16_t r = red * 255.0f;
    const uint16_t g = green * 255.0f;
    const uint16_t b = blue * 255.0f;
    for (x=0; x<len; x++) {
        target[x*3] += image[x] * r / 255;
        target[x*3+1] += image[x] * g / 255;
        target[x*3+2] += image[x] * b / 255;
    }
}

void rescale_intensity16(uint16_t* target, const uint16_t imin, const uint16_t imax, const int len) {
    clip16(target, imin, imax, len);

    const float factor = 65535.0f / (imax - imin);
    int x;
    for (x=0; x<len; x++) {
        target[x] -= imin;
        target[x] = (uint16_t)(factor*target[x]);
    }
}

void rescale_intensity32(uint32_t* target, const uint32_t imin, const uint32_t imax, const int len) {
    clip32(target, imin, imax, len);

    const double factor = 4294967295.0 / (imax - imin);
    int x;
    for (x=0; x<len; x++) {
        target[x] -= imin;
        target[x] = (uint32_t)(factor*target[x]);
    }
}

void rescale_intensity8(uint8_t* target, const uint8_t imin, const uint8_t imax, const int len) {
    clip8(target, imin, imax, len);

    const float factor = 255.0f / (imax - imin);
    int x;
    for (x=0; x<len; x++) {
        target[x] -= imin;
        target[x] = (uint8_t)(factor*target[x]);
    }
}

void clip8(uint8_t* target, const uint8_t min, const uint8_t max, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint8_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip16(uint16_t* target, const uint16_t min, const uint16_t max, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint16_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip32(uint32_t* target, const uint32_t min, const uint32_t max, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint32_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip32_conv8(uint32_t* target, uint8_t* output, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint32_t t = target[x] > 65535 ? 65535 : target[x];
        output[x] = (uint8_t)(t / 256);
    }
}

void clip64_conv8(uint64_t* target, uint8_t* output, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint64_t t = target[x] > 4294967295 ? 4294967295 : target[x];
        output[x] = (uint8_t)(t / 16777216);
    }
}

void clip16_conv8(uint16_t* target, uint8_t* output, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint64_t t = target[x] > 255 ? 255 : target[x];
        output[x] = (uint8_t)(t);
    }
}

#ifdef __cplusplus
}
#endif

