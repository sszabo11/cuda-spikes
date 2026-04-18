#include "eye.h"
#include <math.h>
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

      float sobel_r_x = 0.0;
      float sobel_g_x = 0.0;
      float sobel_b_x = 0.0;

      float sobel_r_y = 0.0;
      float sobel_g_y = 0.0;
      float sobel_b_y = 0.0;

      for (int kx = 0; kx < config.kernel_size; kx++) {
        for (int ky = 0; ky < config.kernel_size; ky++) {
          // printf("\nx: %d | y: %d", kx, ky);
          Color pixel = img->pixels[(row + ky) * config.width + col + kx];
          float weight = config.kernel[ky * config.kernel_size + kx];
          float weight_x = config.kernel_x[ky * config.kernel_size + kx];
          float weight_y = config.kernel_y[ky * config.kernel_size + kx];

          // printf("\nPixel: %f, weight: %f", (float)pixel.r, weight);

          sobel_r_x += (float)pixel.r * weight_x;
          sobel_g_x += (float)pixel.g * weight_x;
          sobel_b_x += (float)pixel.b * weight_x;

          sobel_r_y += (float)pixel.r * weight_y;
          sobel_g_y += (float)pixel.g * weight_y;
          sobel_b_y += (float)pixel.b * weight_y;

          sum_r += (float)pixel.r * weight;
          sum_g += (float)pixel.g * weight;
          sum_b += (float)pixel.b * weight;
        }
      }
      sum_r /= config.kernel_size * config.kernel_size;
      sum_g /= config.kernel_size * config.kernel_size;
      sum_b /= config.kernel_size * config.kernel_size;

      sobel_r_x /= config.kernel_size * config.kernel_size;
      sobel_g_x /= config.kernel_size * config.kernel_size;
      sobel_b_x /= config.kernel_size * config.kernel_size;

      sobel_r_y /= config.kernel_size * config.kernel_size;
      sobel_g_y /= config.kernel_size * config.kernel_size;
      sobel_b_y /= config.kernel_size * config.kernel_size;

      float mag_r = sqrt(pow(sobel_r_x, 2) + pow(sobel_r_y, 2));
      float mag_g = sqrt(pow(sobel_g_x, 2) + pow(sobel_g_y, 2));
      float mag_b = sqrt(pow(sobel_b_x, 2) + pow(sobel_b_y, 2));
      // mag_r /= config.kernel_size * config.kernel_size;
      // mag_g /= config.kernel_size * config.kernel_size;
      // mag_b /= config.kernel_size * config.kernel_size;

      int out_row = row / config.stride;
      int out_col = col / config.stride;
      int out_idx = out_row * out_w + out_col;
      // output_layer_r[out_idx] = mag_r;
      // output_layer_g[out_idx] = mag_g;
      // output_layer_b[out_idx] = mag_b;
      if ((unsigned char)sum_r > 150.0 && (unsigned char)sum_g > 150.0 &&
          (unsigned char)sum_b > 150.0) {
        output_layer_r[out_idx] = sum_r;
        output_layer_g[out_idx] = sum_g;
        output_layer_b[out_idx] = sum_b;
      } else {
        output_layer_r[out_idx] = 0;
        output_layer_g[out_idx] = 0;
        output_layer_b[out_idx] = 0;
      }

      pixels[out_idx] = (Color){(unsigned char)output_layer_r[out_idx],
                                (unsigned char)output_layer_g[out_idx],
                                (unsigned char)output_layer_b[out_idx], 255};
    }
  }

  ReceptorResponse *res = malloc(sizeof(ReceptorResponse));

  res->output_layer_r = output_layer_r;
  res->output_layer_g = output_layer_g;
  res->output_layer_b = output_layer_b;
  res->pixels = pixels;

  // for (int row = 0; row < config.height; row += config.stride) {
  //   for (int col = 0; col < config.width; col += config.stride) {
  //     int out_row = row / config.stride;
  //     int out_col = col / config.stride;
  //     int out_idx = out_row * out_w + out_col;
  //     printf("(%f, %f, %f)", (float)res->pixels[out_idx].r,
  //            (float)res->pixels[out_idx].g, (float)res->pixels[out_idx].b);
  //   }
  // };
  res->out_w = out_w;
  res->out_h = out_h;
  return res;
}
