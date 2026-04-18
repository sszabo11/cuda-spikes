#include "eye.h"

typedef struct {
  int *data;
} SpikeTrain;

SpikeTrain *rate_encode(ImageData *img_data, int T);
