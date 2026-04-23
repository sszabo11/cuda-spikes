#include "encode.h"
#include "data.h"
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

  st->data = calloc(total_spikes, sizeof(uint8_t));
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

  return st;
}

// TODO: Cuda kernel for rate encoding.
//

// SpikeTrain *encode_mnist2(mnist_dataset_t *dataset, int n, int T) {
//   int n_pixels = 28 * 28; // 784
//
//   for (int img_idx = 0; img_idx < n; img_idx++) {
//     uint8_t *img = dataset->images + img_idx * n_pixels;
//
//     SpikeTrain *st = malloc(sizeof(SpikeTrain));
//     st->width = mnist->cols;
//     st->height = mnist->rows;
//     st->T = T;
//     st->data = calloc(n_pixels * T, sizeof(uint8_t));
//
//     for (int i = 0; i < n_pixels; i++) {
//       float rate = img[i] / 255.0f; // 0.0 - 1.0
//       for (int t = 0; t < T; t++) {
//         if ((float)rand() / RAND_MAX < rate) {
//           SPIKE(st, i / mnist->cols, i % mnist->cols, t) = 1;
//         }
//       }
//     }
//   }
//   return st;
// }

SpikeTrain *encode_mnist(mnist_image_t *img, int T) {
  SpikeTrain *st = malloc(sizeof(SpikeTrain));
  st->width = MNIST_IMAGE_WIDTH;   // 28
  st->height = MNIST_IMAGE_HEIGHT; // 28
  st->T = T;
  st->data =
      calloc(MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT * T, sizeof(uint8_t));

  for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
    float rate = img->pixels[i] / 255.0f;
    int y = i / MNIST_IMAGE_WIDTH;
    int x = i % MNIST_IMAGE_WIDTH;
    for (int t = 0; t < T; t++) {
      if ((float)rand() / RAND_MAX < rate) {
        SPIKE(st, y, x, t) = 1;
      }
    }
  }
  return st;
}

// maps 784 input spikes at timestep t → network pre_spikes
// samples evenly if n_neurons != 784
void load_input_spikes(NetworkData *data, SpikeTrain *st, Config *config,
                       int t) {
  int n_pixels = st->width * st->height; // 784
  for (int i = 0; i < config->n_neurons; i++) {
    int px = (int)((float)i / config->n_neurons * n_pixels);
    int y = px / st->width;
    int x = px % st->width;
    data->pre_spikes[i] = SPIKE(st, y, x, t);
  }
}
