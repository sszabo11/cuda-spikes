#include "config.h"
#include "data.h"
#include "process.h"
#include "render.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));
  int n_neurons = 1000;
  int n_conns = 10;

  float sparsity = 0.02;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));
  int const T = 1;

  config->n_neurons = n_neurons;
  config->n_conns = n_conns;
  config->tau_minus = 0.9;
  config->tau_plus = 0.9;
  config->beta = 0.8;
  config->a_plus = 0.001;
  config->a_minus = 0.001;
  config->w_min = 0.3;
  config->w_max = 0.6;
  config->base_threshold = 0.3;
  config->sparsity = sparsity;
  config->T = T;

  init_data(config, data);

  for (int i = 0; i < n_neurons; i++) {
    printf("\nNeuron %d: %f", i, data->membranes[i]);
  }
  printf("\n");

  // printf("\nDone: %f\n", data->membranes[0]);

  for (int i = 0; i < n_neurons; i++) {
    printf("\nNeuron %d: %f", i, data->membranes[i]);
  }

  // for (int p = 0; p < 10; p++) {
  //   cudaError_t res = process(config, data);
  // }
  render(config, data);
}
