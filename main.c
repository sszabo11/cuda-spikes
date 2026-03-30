#include "config.h"
#include "process.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  int n_neurons = 10000000;
  int n_conns = 100;

  float sparsity = 0.2;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));

  config->n_neurons = n_neurons;
  config->n_conns = n_neurons;
  config->tau_minus = 0.95;
  config->tau_plus = 0.8;
  config->beta = 0.9;
  config->a_plus = 0.01;
  config->a_minus = 0.01;
  config->w_min = 0.1;
  config->w_max = 0.8;
  config->base_threshold = 0.5;
  config->sparsity = sparsity;

  process(config, data);
}
