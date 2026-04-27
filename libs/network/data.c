#include "data.h"
#include "config.h"
#include "encode.h"
#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int init_data(Config *config, NetworkData *data, SpikeTrain *input_train) {
  data->membranes = (float *)calloc(config->n_neurons, sizeof(float));

  data->conns = generate_connections(config->n_neurons, config->n_neurons,
                                     config->n_conns);

  data->weights = generate_weights(config->n_neurons, config->n_neurons,
                                   config->w_min, config->w_max);

  data->pre_spikes = (uint8_t *)calloc(config->n_neurons, sizeof(uint8_t));

  data->post_spikes = (uint8_t *)calloc(config->n_neurons, sizeof(uint8_t));

  data->thresholds = (float *)calloc(config->n_neurons, sizeof(float));

  data->pre_trace = (float *)malloc(sizeof(float) * config->n_neurons);

  data->post_trace = (float *)malloc(sizeof(float) * config->n_neurons);

  data->refactory = (uint8_t *)malloc(sizeof(uint8_t) * config->n_neurons);

  memset(data->pre_trace, 0, config->n_neurons * sizeof(float));
  memset(data->post_trace, 0, config->n_neurons * sizeof(float));
  memset(data->refactory, 0, config->n_neurons * sizeof(uint8_t));

  int spikes = 0;
  for (int i = 0; i < config->n_neurons; i++) {
    double random_num = (double)rand() / ((double)RAND_MAX + 1.0);
    double random_num2 = (double)rand() / ((double)RAND_MAX + 1.0);
    double random_num3 = (double)rand() / ((double)RAND_MAX + 1.0);

    // data->membranes[i] = random_num;
    //  int px = i % input_train->width;
    //  int py = i / input_train->width;
    //  int spike = SPIKE(input_train, py, px, 0);
    //  printf("x,y:  (%d, %d) = %d", px, py, spike);
    //  if (spike == 1) {
    //    spikes++;
    //  }
    //  data->pre_spikes[i] = spike;
    //   data->pre_spikes[i] = SPIKE(input_train, i, i, 0);

    // data->pre_spikes[i] = random_num > config->sparsity ? 0 : 1;
    // data->post_spikes[i] = random_num2 > config->sparsity ? 0 : 1;
    data->thresholds[i] = random_num3;
  }
  // printf("\nspikes: %d\n", spikes);

  // int fired = 0;
  // for (int i = 0; i < config->n_neurons; i++) {
  //   // printf("%d", membranes[i]);
  //   if (data->pre_spikes[i] == 1.0) {
  //     fired++;
  //   }
  // }
  // printf("\nFired: %d from %d. %f %%", fired, config->n_neurons,
  //        (float)fired / config->n_neurons * 100.0);

  return 0;
}

void free_data(NetworkData *data) {
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
