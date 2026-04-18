#include "encode.h"
#include "eye.h"
#include <stdlib.h>
#include <time.h>

SpikeTrain *rate_encode(ImageData *img_data, int T) {
  srand(time(NULL));
  SpikeTrain *train = malloc(sizeof(SpikeTrain));
  train->data = malloc(sizeof(int) * T);

  for (int row = 0; row < img_data->height; row++) {
    for (int col = 0; col < img_data->width; col++) {
      Color pixel = img_data->pixels[row * img_data->width + col];

      float grayscale = (float)(pixel.r + pixel.g + pixel.b) / 3 / 255;

      for (int t = 0; t < T; t++) {
        double random_val = (double)rand() / RAND_MAX;
        train->data[t] = random_val > grayscale ? 1 : 0;
      }
    }
  }

  return train;
}
// TODO: Cuda kernel for rate encoding.
