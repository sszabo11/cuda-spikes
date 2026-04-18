#include <raylib.h>

#ifndef PERSON_H
#define PERSON_H
typedef struct {
  int width;
  int height;
  int kernel_size;
  int stride;
  float *kernel;   // Matrix of weighted values for each pixel
  float *kernel_x; // Matrix of weighted values for each pixel
  float *kernel_y; // Matrix of weighted values for each pixel
} EyeReceptor;

typedef struct {
  Color *pixels;
  int width;
  int height;
} ImageData;

typedef struct {
  float *output_layer_r;
  float *output_layer_g;
  float *output_layer_b;
  Color *pixels;
  int out_w;
  int out_h;
} ReceptorResponse;

ImageData *get_image_data(char *path);

ReceptorResponse *receptor(EyeReceptor config, ImageData *img);
#endif
