#include "network.h"
#include "config.h"
#include "encode.h"
#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

Network *network_create() {
  Network *net = malloc(sizeof(Network));
  if (!net)
    return NULL;

  net->data = malloc(sizeof(Data));
  net->config = malloc(sizeof(Config));

  return net;
}

Network *network_destroy(Network *net) {
  if (!net)
    return NULL;

  free_data(net->data);
  free(net->data);
  free(net->config);
  free(net);

  return net;
}
size_t *allocate_neuron_space(Network *net, int n) {
  size_t *arr = calloc(n, sizeof(size_t));

  for (int i = 0; i < n; i++) {
    // size_t random_num = (size_t)rand() / (size_t)RAND_MAX + 1.0);

    // BUG?: some neurons will be input and output and network
    int max = net->config->n_neurons;
    int min = 0;

    size_t r = (rand() % (max - min + 1)) + min;
    arr[i] = r;
  }
  return arr;
}

// Seed weights, thresholds and init variables
int init_network(Network *net, Config *config) {
  // if (net->config == NULL) {
  //   printf("No config set before `init_network`\n");
  //   return -1;
  // }

  net->config = config;
  net->input_dim = 0;
  net->n_dim = 0;
  net->output_dim = 0;
  net->data->membranes = (float *)calloc(net->config->n_neurons, sizeof(float));

  net->data->conns = generate_connections(
      net->config->n_neurons, net->config->n_neurons, net->config->n_conns);

  net->data->weights =
      generate_weights(net->config->n_neurons, net->config->n_neurons,
                       net->config->w_min, net->config->w_max);

  net->data->pre_spikes =
      (uint8_t *)calloc(net->config->n_neurons, sizeof(uint8_t));

  net->data->post_spikes =
      (uint8_t *)calloc(net->config->n_neurons, sizeof(uint8_t));

  net->data->thresholds =
      (float *)calloc(net->config->n_neurons, sizeof(float));

  net->data->pre_trace = (float *)calloc(net->config->n_neurons, sizeof(float));

  net->data->post_trace =
      (float *)calloc(net->config->n_neurons, sizeof(float));

  net->data->refactory =
      (uint8_t *)calloc(net->config->n_neurons, sizeof(uint8_t));

  net->data->input_idxs = allocate_neuron_space(net, net->input_dim);
  net->data->neuron_idxs = allocate_neuron_space(net, net->n_dim);
  net->data->output_idxs = allocate_neuron_space(net, net->output_dim);

  int spikes = 0;
  for (int i = 0; i < net->config->n_neurons; i++) {
    double random_num = (double)rand() / ((double)RAND_MAX + 1.0);

    net->data->thresholds[i] = random_num;
  }

  return 1;
}

void free_data(Data *data) {
  free(data->conns);
  free(data->weights);
  free(data->membranes);
  free(data->post_spikes);
  free(data->neuron_idxs);
  free(data->output_idxs);
  free(data->input_idxs);
  free(data->pre_spikes);
  free(data->post_trace);
  free(data->pre_trace);
  free(data->refactory);
  free(data->thresholds);
}
