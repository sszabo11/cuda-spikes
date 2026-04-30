#include "cuda_runtime_api.h"
#include "data.h"
#include "driver_types.h"
#include "encode.h"
#include "logger.h"
#include "network.h"
#include "spike.h"
#include <stdint.h>
#include <stdio.h>

cudaError_t copy_input_to_gpu(Network *net, SpikeTrain *st) {

  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      int pixel_idx = y * 28 + x;
      printf("st: %d\n", GET_SPIKE(st, pixel_idx, 0));
    }
  }
  cudaError_t err =
      cudaMemcpy(net->data->pre_spikes, st->data, st->width * sizeof(uint8_t),
                 cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr, "copy_input_to_gpu failed: %s\n", cudaGetErrorString(err));
  }
  return err;
}

cudaError_t copy_results_back_to_cpu(Network *d_net, Network *h_net) {

  cudaError_t err = cudaMemcpy(
      h_net->data->post_spikes, d_net->data->post_spikes,
      h_net->config->n_neurons * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr, "copy_results_back failed: %s\n", cudaGetErrorString(err));
  }
  return err;
}

cudaError_t compute(Network *d_net, Network *h_net, SpikeTrain *input,
                    int sample_id) {
  // Set pre_spikes to input spikes
  copy_input_to_gpu(d_net, input);

  // Computation steps
  for (int t = 0; t < d_net->config->T; t++) {
    if (sample_id == 0) {
      copy_results_back_to_cpu(d_net, h_net);
      // for (int i = 0; i < h_net->config->n_neurons; i++) {
      //   printf("%d\n", h_net->data->post_spikes[i]);
      // }
      write_to_csv(h_net, t, 1);
    }
    run_kernels(d_net);
  }

  return cudaSuccess;
}
