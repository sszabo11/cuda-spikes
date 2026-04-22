#include "config.h"
#include "data.h"
#include "process.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main() {

  srand(time(NULL));
  int n_neurons = 1000000;
  int n_conns = 10;

  float sparsity = 0.05;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));

  int const T = 10;

  config->n_neurons = n_neurons;
  config->n_conns = n_conns;
  config->tau_minus = 0.95;
  config->tau_plus = 0.8;
  config->beta = 0.99;
  config->a_plus = 0.01;
  config->a_minus = 0.01;
  config->w_min = 0.1;
  config->w_max = 0.8;
  config->base_threshold = 0.5;
  config->sparsity = sparsity;
  config->T = T;

  init_data(config, data, NULL);

  printf("\nPrior: %f\n", data->membranes[64]);
  cudaError_t res = process(config, data);

  printf("\nDone: %f\n", data->membranes[64]);
}
