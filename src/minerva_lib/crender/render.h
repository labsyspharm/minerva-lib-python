#ifndef RENDER_H
#define RENDER_H

#include <stdint.h>

void clip(float* target, float min, float max, int channels, int len);
void clip16(uint16_t* target, uint16_t min, uint16_t max, int channels, int len);
void clip32_conv8(uint32_t* target, uint8_t* output, uint16_t min, uint16_t max, int channels, int len);
void rescale_intensity(float* target, float imin, float imax, int len);
void rescale_intensity16(uint16_t* target, int imin, int imax, int len);
float* image_as_float(uint16_t* target, int len);
void composite(float *target, float* image, float red, float green, float blue, int len);
void composite16(uint32_t *target, uint16_t* image, float red, float green, float blue, int len);
void clip32(uint32_t* target, uint16_t min, uint16_t max, int channels, int len);

#endif