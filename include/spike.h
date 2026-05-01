#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include "network.h"

#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdint.h>

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

cudaError_t propagate(Network *net);
cudaError_t update(Network *net);
cudaError_t inject_input(Network *net, int t);
cudaError_t run_network(Network *net, bool learn);
cudaError_t reset_net(Network *net);
#ifdef __cplusplus
}
#endif

#endif
