#include "config.h"
#include "data.h"
#include "process.h"
#include "render.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  srand(2);
  int n_neurons = 2000;
  int n_conns = 10;

  float sparsity = 0.2;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));
  int const T = 1;

  config->n_neurons = n_neurons;
  config->n_conns = n_conns;
  config->tau_minus = 0.95;
  config->tau_plus = 0.8;
  config->beta = 0.9;
  config->a_plus = 0.01;
  config->a_minus = 0.01;
  config->w_min = 0.1;
  config->w_max = 0.8;
  config->base_threshold = 0.5;
  config->sparsity = sparsity;
  config->T = T;

  init_data(config, data);

  for (int i = 0; i < n_neurons; i++) {
    printf("\nNeuron %d: %f", i, data->membranes[i]);
  }
  printf("\n");
  cudaError_t res = process(config, data);

  printf("\nDone %d\n", res);
  // printf("\nDone: %f\n", data->membranes[0]);

  for (int i = 0; i < n_neurons; i++) {
    printf("\nNeuron %d: %f", i, data->membranes[i]);
  }

  render(config, data);
}
