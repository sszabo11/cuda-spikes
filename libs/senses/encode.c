#include "encode.h"
#include "eye.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Timesteps for input
// Timesteps for computation?
// [
//  [0, 1, 0, 0, 0, 1, 0] T0
//  [0, 1, 0, 1, 0, 0, 0] T1
//  [0, 0, 1, 0, 0, 0, 1] T2
//  [1, 0, 0, 1, 0, 1, 0] T3
// ]
// SpikeTrain *rate_encode(ImageData *img_data, int T) {
//  srand(time(NULL));
//  SpikeTrain *train = malloc(sizeof(SpikeTrain));
//  printf("\nSize of input: %d\n",
//         (int)sizeof(int *) * img_data->width * img_data->height * T);
//  // train->data = malloc(sizeof(int) * img_data->width * img_data->height *
//  T); train->data = malloc(T * sizeof(int *));
//
//  for (int i = 0; i < T; i++) {
//    train->data[i] = malloc(img_data->height * img_data->width * sizeof(int));
//  }
//
//  for (int row = 0; row < img_data->height; row++) {
//    for (int col = 0; col < img_data->width; col++) {
//      Color pixel = img_data->pixels[row * img_data->width + col];
//
//      float grayscale = (float)(pixel.r + pixel.g + pixel.b) / 3 / 255;
//
//      for (int t = 0; t < T; t++) {
//        int idx = row * img_data->width + img_data->height;
//        double random_val = (double)rand() / RAND_MAX;
//        train->data[T][row] = random_val > grayscale ? 1 : 0;
//      }
//    }
//  }
//
//  return train;
//}

SpikeTrain *rate_encode(ImageData *img_data, int T) {
  if (!img_data || T <= 0)
    return NULL;

  SpikeTrain *st = malloc(sizeof(SpikeTrain));
  if (!st)
    return NULL;

  st->width = img_data->width;
  st->height = img_data->height;
  st->T = T;

  // Total size: height * width * T
  size_t total_spikes = (size_t)st->height * st->width * T;

  st->data = calloc(total_spikes, sizeof(uint8_t)); // zeros by default
  if (!st->data) {
    free(st);
    return NULL;
  }

  // Now you can fill it...
  // Example: simple rate encoding
  for (int y = 0; y < st->height; y++) {
    for (int x = 0; x < st->width; x++) {
      // your rate encoding logic here
      Color pixel = img_data->pixels[y * img_data->width + x];

      float grayscale = (float)(pixel.r + pixel.g + pixel.b) / 3 / 255;

      for (int t = 0; t < T; t++) {
        if (rand() / (float)RAND_MAX < grayscale) {
          SPIKE(st, y, x, t) = 1;
        }
      }
    }
  }
}

// TODO: Cuda kernel for rate encoding.
