#ifndef NET_H
#define NET_H

#include "config.h"
#include <stddef.h>
#include <stdint.h>

typedef struct Data {
  float *membranes;
  size_t *conns;
  float *weights;
  float *thresholds;
  uint8_t *post_spikes;
  uint8_t *pre_spikes;
  uint8_t *refactory;
  float *pre_trace;
  float *post_trace;
  size_t *input_idxs;
  size_t *neuron_idxs;
  size_t *output_idxs;
} Data;

typedef struct Network {
  Data *data;
  Config *config;
  int input_dim;
  int output_dim;
  int n_dim;
} Network;

Network *network_create();
Network *network_destroy(Network *net);

int init_network(Network *net, Config *config);
void free_data(Data *data);

#endif
