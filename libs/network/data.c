#include "data.h"
#include "config.h"
#include "encode.h"
#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int init_network2(Config *config, NetworkData *data, SpikeTrain *input_train) {
  data->membranes = (float *)calloc(config->n_neurons, sizeof(float));

  data->conns = generate_connections(config->n_neurons, config->n_neurons,
                                     config->n_conns);

  data->weights = generate_weights(config->n_neurons, config->n_neurons,
                                   config->w_min, config->w_max);

  data->pre_spikes = (uint8_t *)calloc(config->n_neurons, sizeof(uint8_t));

  data->post_spikes = (uint8_t *)calloc(config->n_neurons, sizeof(uint8_t));

  data->thresholds = (float *)calloc(config->n_neurons, sizeof(float));

  data->pre_trace = (float *)calloc(config->n_neurons, sizeof(float));

  data->post_trace = (float *)calloc(config->n_neurons, sizeof(float));

  data->refactory = (uint8_t *)calloc(config->n_neurons, sizeof(uint8_t));

  // memset(data->pre_trace, 0, config->n_neurons * sizeof(float));
  // memset(data->post_trace, 0, config->n_neurons * sizeof(float));
  // memset(data->refactory, 0, config->n_neurons * sizeof(uint8_t));

  int spikes = 0;
  for (int i = 0; i < config->n_neurons; i++) {
    double random_num = (double)rand() / ((double)RAND_MAX + 1.0);

    data->thresholds[i] = random_num;
  }

  return 0;
}

void free_data2(NetworkData *data) {
  free(data->conns);
  free(data->weights);
  free(data->membranes);
  free(data->post_spikes);
  free(data->pre_spikes);
  free(data->post_trace);
  free(data->pre_trace);
  free(data->refactory);
  free(data->thresholds);
}
