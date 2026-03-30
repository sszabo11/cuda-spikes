#pragma once

#include "config.h"
#include "data.h"

#include <cuda_runtime.h>

#define checkCuda(err)                                                         \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

cudaError_t run_kernels(Config *config, NetworkData *data);
