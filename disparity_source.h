// disparity_source.h
#ifndef DISPARITY_SOURCE_H
#define DISPARITY_SOURCE_H

#include <stdint.h>

void compute_disparity(
    unsigned char *left, unsigned char *right, signed char *disparity,
    int height, int width, int min_disparity, int max_disparity, int half_block_size,
    int stereo_width, int stereo_height
);

#endif // DISPARITY_SOURCE_H

