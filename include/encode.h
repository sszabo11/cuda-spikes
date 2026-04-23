#include "eye.h"
#include "mnist.h"
#include <stdint.h>

#ifndef ENCODE_H
#define ENCODE_H

#define SPIKE(st, y, x, t) (st)->data[((y) * (st)->width + (x)) * (st)->T + (t)]

typedef struct {
  int width;
  int height;
  int T;
  uint8_t *data;
} SpikeTrain;

SpikeTrain *rate_encode(ImageData *img_data, int T);

// SpikeTrain *encode_mnist(mnist_dataset_t *dataset, int T);

//  SpikeTrain *encode_mnist(mnist_image_t *img, int T);
//  void load_input_spikes(NetworkData *data, SpikeTrain *st, Config *config,
//                         int t);
#endif
