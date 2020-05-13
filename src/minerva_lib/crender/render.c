#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <string.h>
#include <math.h>

#define TILE_SIZE 1024
//#define AVX

void clip(float* target, float min, float max, int channels, int tileSize);
void clip16(uint16_t* target, uint16_t min, uint16_t max, int channels, int tileSize);
uint8_t* clip32_conv8(uint32_t* target, uint16_t min, uint16_t max, int channels, int tileSize);
void rescale_intensity(float* target, float imin, float imax, int tileSize);
void rescale_intensity16(uint16_t* target, int imin, int imax, int tileSize);
float* image_as_float(uint16_t* target, int tileSize);
void composite(float *target, float* image, float red, float green, float blue, int tileSize);
void composite16(uint32_t *target, uint16_t* image, float red, float green, float blue, int tileSize);

void print256_num(__m256 var)
{
    float val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %f %f %f %f %f %f %f %f \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7]);
}

void print256i_num(__m256i var)
{
    int val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %d %d %d %d %d %d %d %d \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7]);
}

void print128i_num(__m128i var)
{
    int val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %d %d %d %d \n", 
           val[0], val[1], val[2], val[3]);
}

void print_arr(float *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*8; i<row*8+8; i++) {
            printf("%f ", arr[i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_uarr(uint16_t *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*8; i<row*8+8; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    printf("\n");
}

#ifdef AVX
void composite(float *target, float* image, float red, float green, float blue, int tileSize) {
    int x;

    __m128 i0;
    __m128 c0 = _mm_set_ps(0.0f, blue, green, red);
    __m128 res;
    for (x=0; x<tileSize*tileSize; x+=1) {

        i0 = _mm_broadcast_ss(image+x);
        res = _mm_mul_ps(i0, c0);
        __m128 orig = _mm_load_ps(target+(x*3));
        __m128 comp = _mm_add_ps(res, orig);

        _mm_storeu_ps(target+(x*3), comp);
    }
}
#endif

#ifndef AVX
void composite(float *target, float* image, float red, float green, float blue, int tileSize) {
    int x;
    for (x=0; x<tileSize*tileSize; x++) {
        target[x*3] += (image[x] * red); 
        target[x*3+1] += (image[x] * green); 
        target[x*3+2] += (image[x] * blue);
    }
}
#endif

void composite16(uint32_t *target, uint16_t* image, float red, float green, float blue, int tileSize) {
    int x;
    for (x=0; x<tileSize*tileSize; x++) {
        target[x*3] += image[x] * red;
        target[x*3+1] += image[x] * green;
        target[x*3+2] += image[x] * blue;
    }
}

#ifdef AVX
float* image_as_float(uint16_t* target, int tileSize) {
    float *out = (float *)aligned_alloc(32, tileSize * tileSize * sizeof(float));
    float f = 1 / 65535.0f;
    __m256i ints;
    __m256 floats;
    __m256 res;
    __m256 maxValue = _mm256_broadcast_ss( &f );
    int x, i;

    for (x=0; x<tileSize*tileSize; x+=8) {
        __m128i short_vec = _mm_loadu_si128((__m128i*)(target+x));
        __m256i int_vec =  _mm256_cvtepu16_epi32 (short_vec); // AVX2
        __m256  singleComplete = _mm256_cvtepi32_ps (int_vec);
        // Finally do the math
        __m256 scaledVect = _mm256_mul_ps(singleComplete, maxValue);
        // and puts the result where needed.
        _mm256_store_ps(out+x, scaledVect);
    }
    
    return out;
}
#endif

#ifndef AVX
float* image_as_float(uint16_t* target, int tileSize) {
    float *out = (float *)malloc(tileSize * tileSize * sizeof(float));
    for (int x=0; x<tileSize*tileSize; x++) {
        out[x] = target[x] / 65535.0f;
    }
    return out;
}
#endif

#ifdef AVX
void rescale_intensity(float* target, float imin, float imax, int tileSize) {
    clip(target, imin, imax, 1, tileSize);

    __m256 pixels;
    float factor = 1.0f / (imax - imin);
    __m256 sseFactor = _mm256_broadcast_ss( &factor );
    __m256 sseMin = _mm256_broadcast_ss( &imin );

    for (int x=0; x<tileSize*tileSize; x+=8) {
        pixels = _mm256_load_ps(target+x);
        pixels = _mm256_sub_ps(pixels, sseMin);
        pixels = _mm256_mul_ps(pixels, sseFactor);

        _mm256_store_ps( target+x, pixels );
    }
}
#endif

#ifndef AVX
void rescale_intensity(float* target, float imin, float imax, int tileSize) {
    clip(target, imin, imax, 1, tileSize);

    float factor = 1.0f / (imax - imin);

    for (int x=0; x<tileSize*tileSize; x++) {
        target[x] -= imin;
        target[x] *= factor;
    }
}
#endif
/*
    imin, imax = intensity_range(image, in_range)
    omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))

    np.clip(image, imin, imax, out=image)

    image -= imin
    factor = (omax - omin) / float(imax - imin)
    image *= factor
    image += omin
*/
void rescale_intensity16(uint16_t* target, int imin, int imax, int tileSize) {
    clip16(target, imin, imax, 1, tileSize);

    float factor = 65535.0f / (imax - imin);

    for (int x=0; x<tileSize*tileSize; x++) {
        target[x] -= imin;
        target[x] = (uint16_t)(factor*target[x]);
    }
}

#ifdef AVX
void clip(float* target, float min, float max, int channels, int tileSize) {
    __m256 pixels, clipped;
    __m256 sseFloats;
    int x, y, i;
    __m256 sseMin = _mm256_broadcast_ss( &min );
    __m256 sseMax = _mm256_broadcast_ss( &max );

    for (x=0; x<tileSize*tileSize*channels; x+=8) {
        // load pixel values into register
        pixels = _mm256_load_ps(target+x);
        // clip between min and max
        sseFloats = _mm256_max_ps( pixels, sseMin );
        clipped = _mm256_min_ps( sseFloats, sseMax );

        // copy values from registers to array
        _mm256_store_ps( target+x, clipped );
    }
}
#endif

#ifndef AVX
void clip(float* target, float min, float max, int channels, int tileSize) {
    for (int x=0; x<tileSize*tileSize*channels; x++) {
        const float t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}
#endif

void clip16(uint16_t* target, uint16_t min, uint16_t max, int channels, int tileSize) {
    for (int x=0; x<tileSize*tileSize*channels; x++) {
        const uint16_t t = target[x] < min ? min : target[x];
        target[x] = t > max ? max : t;
    }
}

uint8_t* clip32_conv8(uint32_t* target, uint16_t min, uint16_t max, int channels, int tileSize) {
    uint8_t* res = (uint8_t*)malloc(tileSize * tileSize * channels * sizeof(uint8_t));
    for (int x=0; x<tileSize*tileSize*channels; x++) {
        uint32_t t = target[x] < min ? min : target[x];
        t = t > max ? max : t;
        res[x] = (uint8_t)(t / 256);
    }
    return res;
}

void test_render() {
    int size = 1024*1024;
    uint16_t* intArr = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));
    uint16_t* intArr2 = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));

    float *target = (float *)aligned_alloc(32, size * sizeof(float) * 3);

    int i;
    for (i=0; i<size; i++) {
        intArr[i] = 65535;
        intArr2[i] = 32768;
    }
    for (i=0; i<size*3; i++) {
        target[i] = 0;
    }
    clock_t start = clock() / (CLOCKS_PER_SEC / 1000);

    printf("Initial uint16_t array\n");
    print_uarr(intArr, 3);
    print_uarr(intArr2, 3);

    float* floatArr = image_as_float(intArr, TILE_SIZE);
    float* floatArr2 = image_as_float(intArr2, TILE_SIZE);
    printf("After converting to float\n");
    print_arr(floatArr, 3);
    print_arr(floatArr2, 3);

    rescale_intensity(floatArr, 0.1f, 0.9f, TILE_SIZE);
    rescale_intensity(floatArr2, 0.1f, 0.9f, TILE_SIZE);
    printf("After rescaling intensity\n");
    print_arr(floatArr, 3);
    print_arr(floatArr2, 3);

    composite(target, floatArr, 0.1, 0.2, 0.3, 1024);
    printf("Final target after composition 1\n");
    print_arr(target, 3);
    composite(target, floatArr2, 0.5, 1.0, 0, 1024);
    printf("Final target after composition 2\n");
    print_arr(target, 3);

    clip(target, 0, 1, 1, 1024);
    printf("Final target after clipping \n");
    print_arr(target, 3);

    clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
    clock_t t = end - start;
    printf("TIME: %f \n", (float)t);

    printf("\r\n");
}

void test_render16() {
    int size = 1024*1024;
    uint16_t* intArr = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));
    uint16_t* intArr2 = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));

    uint32_t *target = (uint32_t *)aligned_alloc(32, size * sizeof(uint32_t) * 3);

    int i;
    for (i=0; i<size; i++) {
        intArr[i] = rand() % 65535;
        intArr2[i] = rand() % 65535;
    }
    for (i=0; i<size*3; i++) {
        target[i] = 0;
    }
    clock_t start = clock() / (CLOCKS_PER_SEC / 1000);

    printf("Initial uint16_t array\n");
    print_uarr(intArr, 3);
    print_uarr(intArr2, 3);

    rescale_intensity16(intArr, 2000, 62000, TILE_SIZE);
    rescale_intensity16(intArr2, 12000, 28000, TILE_SIZE);
    printf("After rescaling intensity\n");
    print_uarr(intArr, 3);
    print_uarr(intArr2, 3);

    composite16(target, intArr, 0, 32768, 65535, 1024);
    printf("Final target after composition 1\n");
    //print_uarr(target, 3);
    composite16(target, intArr2, 48000, 65535, 0, 1024);
    printf("Final target after composition 2\n");
    //print_uarr(target, 3);

    //clip16(target, 0, 65535, 1, 1024);
    printf("Final target after clipping \n");
    //print_uarr(target, 3);

    clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
    clock_t t = end - start;
    printf("TIME: %f \n", (float)t);

    printf("\r\n");
}

void test_composite() {
    int size = 1024*1024;
    float* arr1 = (float*)aligned_alloc(32, size * sizeof(float));
    float* arr2 = (float*)aligned_alloc(32, size * sizeof(float));
    float *target = (float *)aligned_alloc(32, size * sizeof(float) * 3);
    for (int i=0; i<size; i++) {
        arr1[i] = rand() / RAND_MAX;
        arr2[i] = rand() / RAND_MAX;
    }
    for (int i=0; i<size*3; i++) {
        target[i] = 0;
    }

    clock_t start = clock() / (CLOCKS_PER_SEC / 1000);
    for (int i=0; i<1000; i++) {
        composite(target, arr1, 0.1, 0.2, 0.3, 1024);
        composite(target, arr2, 0.1, 0.2, 0.3, 1024);
    }
    

    clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
    clock_t t = end - start;
    printf("TIME: %f \n", (float)t);
}

void test_to_float() {
    int size = 8192 * 8192;
    uint16_t* intArr = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));
    for (int i=0; i<size; i++) {
        intArr[i] = rand() % 65535;
    }

    clock_t start = clock() / (CLOCKS_PER_SEC / 1000);

    image_as_float(intArr, 8192);

    clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
    clock_t t = end - start;
    printf("TIME: %f \n", (float)t);
}

void test_clip() {
    int size = 8192*8192;
    float* arr1 = (float*)aligned_alloc(32, size * sizeof(float));
    for (int i=0; i<size; i++) {
        arr1[i] = rand() / RAND_MAX;
    }

    clock_t start = clock() / (CLOCKS_PER_SEC / 1000);

    for (int i=0; i<100; i++) {
        clip(arr1, 0.0f, 1.0f, 1, 8192);
    }

    clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
    clock_t t = end - start;
    printf("TIME: %f \n", (float)t);    
}

int main() {
    test_render16();
    return 0;
}