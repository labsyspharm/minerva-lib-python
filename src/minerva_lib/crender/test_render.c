#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "render.h"

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

void print_uarr8(uint8_t *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*8; i<row*8+8; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_uarr16(uint16_t *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*8; i<row*8+8; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_uarr32(uint32_t *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*8; i<row*8+8; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_rescale(uint16_t* input, uint16_t* result, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*8; i<row*8+8; i++) {
            printf("%d-->%d ", input[i], result[i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_rgb(uint8_t *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*15; i<row*15+15; i+=3) {
            printf("(%d,%d,%d) ", arr[i], arr[i+1], arr[i+2]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_rgb32(uint32_t *arr, int rows) {
    int i, row;
    for (row=0; row<rows; row++) {
        for (i=row*15; i<row*15+15; i+=3) {
            printf("(%d,%d,%d) ", arr[i], arr[i+1], arr[i+2]);
        }
        printf("\n");
    }
    printf("\n");
}

void assert_results(uint8_t* output, uint8_t* expected, int len) {
    int i;
    bool ok = true;
    for (i=0; i<len; i++) {
        if (output[i] != expected[i]) {
            printf("\033[0;31m");
            printf("ERROR - output does not match expected: %d != %d \n", output[i], expected[i]);
            ok = false;
        }
    }
    if (ok) {
        printf("\033[0;32m");
        printf("Output OK\n");
    }
    printf("\033[0m");
}

void test_render16() {
    int size = 1024*1024;
    uint16_t* intArr = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));
    uint16_t* intArr2 = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));
    uint32_t *target = (uint32_t *)aligned_alloc(32, size * sizeof(uint32_t) * 3);
    uint8_t* output = (uint8_t *)aligned_alloc(32, size * sizeof(uint8_t) * 3);

    uint16_t* orig1 = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));
    uint16_t* orig2 = (uint16_t*)aligned_alloc(32, size * sizeof(uint16_t));

    uint16_t min1 = 2000;
    uint16_t max1 = 36000;
    uint16_t min2 = 5500;
    uint16_t max2 = 48000;

    int i;
    for (i=0; i<size; i++) {
        intArr[i] = rand() % 65535;
        intArr2[i] = rand() % 65535;
    }
    // Hardcode a few values to get constant output for test assertion
    intArr[0] = 0;
    intArr[1] = 500;
    intArr[2] = 10000;
    intArr[3] = 32767;
    intArr[4] = 48000;
    intArr[5] = 65535;

    intArr2[0] = 0;
    intArr2[1] = 62000;
    intArr2[2] = 52000;
    intArr2[3] = 32767;
    intArr2[4] = 15000;
    intArr2[5] = 65535;

    // Final expected values for 5 first pixels (R, G, B)
    uint8_t expected[15] = {
        0, 0, 0, 
        136, 0, 162, 
        255, 135, 255, 
        101, 232, 189, 
        21, 141, 69};

    // Create a copy of the original arrays, so we can compare rescaling intensities
    memcpy(orig1, intArr, size * sizeof(uint16_t));
    memcpy(orig2, intArr2, size * sizeof(uint16_t));

    for (i=0; i<size*3; i++) {
        target[i] = 0;
    }
    clock_t start = clock() / (CLOCKS_PER_SEC / 1000);

    printf("Initial uint16_t array\n");
    print_uarr16(intArr, 3);
    print_uarr16(intArr2, 3);

    rescale_intensity16(intArr, min1, max1, size);
    rescale_intensity16(intArr2, min2, max2, size);
    printf("Rescaling intensity MIN: %d MAX: %d\n", min1, max1);
    print_rescale(orig1, intArr, 3);
    printf("Rescaling intensity MIN: %d MAX: %d\n", min2, max2);
    print_rescale(orig2, intArr2, 3);

    composite16(target, intArr, 65535, 65535, 65535, size);
    printf("Final target after composition 1\n");
    print_rgb32(target, 3);
    composite16(target, intArr2, 48000, 65535, 12000, size);
    printf("Final target after composition 2\n");
    print_rgb32(target, 3);

    clip32_conv8(target, output, 0, 65535, size*3);
    printf("Final target after clipping and converting to 8bit \n");
    print_rgb(output, 3);
    assert_results(output, expected, 15);

    clock_t end = clock() / (CLOCKS_PER_SEC / 1000);
    clock_t t = end - start;
    printf("TIME: %f \n", (float)t);

    printf("\r\n");
}

int main() {
    test_render16();
    return 0;
}