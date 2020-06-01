#ifndef RENDER_H
#define RENDER_H

#ifdef _WIN32
    #define DllExport   __declspec( dllexport )
#else
    #define DllExport
#endif

#include <stdint.h>

/**
 * Clips values of target between min and max
 */
DllExport void clip16(uint16_t* target, uint16_t min, uint16_t max, int len);

/**
 * Clips values of target between min and max, and converts values to 0-255
 */
DllExport void clip32_conv8(uint32_t* target, uint8_t* output, uint16_t min, uint16_t max, int len);

/**
 * Rescales 16 bit pixel values (intensities) in following way:
  - values below min become 0
  - values higher than max become 65535
  - values between min and max get scaled between 0-65535
 */
DllExport void rescale_intensity16(uint16_t* target, uint16_t min, uint16_t max, int len);

/**
 * Composites pixel values from image to target, and colorizes them according to given red, green, blue. 
 */
DllExport void composite16(uint32_t *target, uint16_t* image, float red, float green, float blue, int len);

#endif