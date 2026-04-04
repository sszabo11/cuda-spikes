#include "data.h"
#include "config.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>

int init_data(Config *config, NetworkData *data) {
  data->membranes = (float *)malloc(sizeof(float) * config->n_neurons);

  data->conns = generate_connections(config->n_neurons, config->n_conns);

  data->weights = generate_weights(config->n_neurons, config->n_neurons,
                                   config->w_min, config->w_max);

  data->pre_spikes = (int *)malloc(sizeof(int) * config->n_neurons);

  data->post_spikes = (int *)malloc(sizeof(int) * config->n_neurons);

  data->thresholds = (float *)malloc(sizeof(float) * config->n_neurons);

  data->pre_trace = (float *)malloc(sizeof(float) * config->n_neurons);

  data->post_trace = (float *)malloc(sizeof(float) * config->n_neurons);

  data->refactory = (int *)malloc(sizeof(int) * config->n_neurons);

  memset(data->pre_trace, 0, config->n_neurons * sizeof(float));
  memset(data->post_trace, 0, config->n_neurons * sizeof(float));
  memset(data->refactory, 0, config->n_neurons * sizeof(int));

  for (int i = 0; i < config->n_neurons; i++) {
    double random_num = (double)rand() / ((double)RAND_MAX + 1.0);
    double random_num2 = (double)rand() / ((double)RAND_MAX + 1.0);
    double random_num3 = (double)rand() / ((double)RAND_MAX + 1.0);

    data->membranes[i] = random_num;
    data->pre_spikes[i] = random_num > config->sparsity ? 0 : 1;
    data->post_spikes[i] = random_num2 > config->sparsity ? 0 : 1;
    data->thresholds[i] = random_num3;
  }

  int fired = 0;
  for (int i = 0; i < config->n_neurons; i++) {
    // printf("%d", membranes[i]);
    if (data->pre_spikes[i] == 1.0) {
      fired++;
    }
  }
  printf("\nFired: %d from %d. %f %%", fired, config->n_neurons,
         (float)fired / config->n_neurons * 100.0);

  return 0;
}
