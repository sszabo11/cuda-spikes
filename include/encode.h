#include "eye.h"
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
#endif
