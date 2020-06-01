#ifdef __cplusplus
extern "C" {
#endif

#include "render.h"

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

void rescale_intensity16(uint16_t* target, const uint16_t imin, const uint16_t imax, const int len) {
    clip16(target, imin, imax, len);

    const float factor = 65535.0f / (imax - imin);
    int x;
    for (x=0; x<len; x++) {
        target[x] -= imin;
        target[x] = (uint16_t)(factor*target[x]);
    }
}

void clip16(uint16_t* target, const uint16_t min, const uint16_t max, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint16_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

void clip32_conv8(uint32_t* target, uint8_t* output, const uint16_t min, const uint16_t max, const int len) {
    int x;
    for (x=0; x<len; x++) {
        const uint32_t t = target[x] > max ? max : target[x];
        output[x] = (uint8_t)(t / 256);
    }
}

#ifdef __cplusplus
}
#endif

