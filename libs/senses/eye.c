#include "eye.h"
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ImageData *get_image_data(char *path) {
  ImageData *data = malloc(sizeof(ImageData));

  Image img = LoadImage(path);
  Color *pixels = LoadImageColors(img);

  data->height = img.height;
  data->width = img.width;
  data->pixels = pixels;

  printf("format: %d", img.format);

  return data;
}

ReceptorResponse *receptor(EyeReceptor config, ImageData *img) {
  int out_h = config.height / config.stride;
  int out_w = config.width / config.stride;
  float *output_layer_r = malloc(sizeof(float) * img->width * img->height);
  float *output_layer_g = malloc(sizeof(float) * img->width * img->height);
  float *output_layer_b = malloc(sizeof(float) * img->width * img->height);
  Color *pixels = malloc(sizeof(Color) * out_w * out_h);

  for (int row = 0; row < config.height; row += config.stride) {
    for (int col = 0; col < config.width; col += config.stride) {

      // printf("\nx: %d | y: %d", col, row);
      float sum_r = 0.0;
      float sum_g = 0.0;
      float sum_b = 0.0;
      for (int kx = 0; kx < config.kernel_size; kx++) {
        for (int ky = 0; ky < config.kernel_size; ky++) {
          // printf("\nx: %d | y: %d", kx, ky);
          Color pixel = img->pixels[(row + ky) * config.width + col + kx];
          float weight = config.kernel[ky * config.kernel_size + kx];

          // printf("\nPixel: %f, weight: %f", (float)pixel.r, weight);

          sum_r += (float)pixel.r * weight;
          sum_g += (float)pixel.g * weight;
          sum_b += (float)pixel.b * weight;
        }
      }
      sum_r /= config.kernel_size * config.kernel_size;
      sum_g /= config.kernel_size * config.kernel_size;
      sum_b /= config.kernel_size * config.kernel_size;

      int out_row = row / config.stride;
      int out_col = col / config.stride;
      int out_idx = out_row * out_w + out_col;
      // output_layer_r[col * img->width + row] = sum_r;
      // output_layer_g[col * img->width + row] = sum_g;
      // output_layer_b[col * img->width + row] = sum_b;
      output_layer_r[out_idx] = sum_r;
      output_layer_g[out_idx] = sum_g;
      output_layer_b[out_idx] = sum_b;
      // pixels[col * img->width + row].r = sum_r;
      // pixels[col * img->width + row].g = sum_g;
      // pixels[col * img->width + row].b = sum_b;
      // pixels[col * img->width + row].a = 255;

      pixels[out_idx] = (Color){(unsigned char)sum_r, (unsigned char)sum_g,
                                (unsigned char)sum_b, 255};
    }
  }

  ReceptorResponse *res = malloc(sizeof(ReceptorResponse));
  res->output_layer_r = output_layer_r;
  res->output_layer_g = output_layer_g;
  res->output_layer_b = output_layer_b;
  res->pixels = pixels;
  res->out_w = out_w;
  res->out_h = out_h;
  return res;
}
