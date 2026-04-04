#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

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

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t run_kernels(Config *config, NetworkData *data);
#ifdef __cplusplus
}
#endif

#endif
