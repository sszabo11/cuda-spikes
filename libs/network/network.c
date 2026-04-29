#include "network.h"
#include "config.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "encode.h"
#include "utils.h"
#include <assert.h>
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

Network *send_network_to_gpu(Network *net) {
  assert(net->data != NULL);
  assert(net->config != NULL);

  Data *data = net->data;
  Config *config = net->config;

  if (!config->T) {
    printf("\nError: Timesteps not specified!\n");
    return NULL;
  }
  if (!config->n_neurons) {
    printf("\nError: Number of neurons not specified!\n");
    return NULL;
  }
  if (!config->n_conns) {
    printf("\nError: Connections not specified!\n");
    return NULL;
  }

  float *d_membranes;
  size_t *d_conns;
  float *d_weights;
  float *d_thresholds;
  float *d_pre_trace;
  float *d_post_trace;
  uint8_t *d_post_spikes;
  uint8_t *d_refactory;
  uint8_t *d_pre_spikes;

  int n_neurons = config->n_neurons;
  int n_conns = config->n_conns;

  Network *d_net = malloc(sizeof(Network));
  Data *d_data = malloc(sizeof(Data));

  cudaMalloc((void **)&d_membranes, n_neurons * sizeof(float));
  cudaMalloc((void **)&d_conns, n_neurons * n_conns * sizeof(size_t));
  cudaMalloc((void **)&d_weights, n_neurons * n_conns * sizeof(float));
  cudaMalloc((void **)&d_pre_spikes, n_neurons * sizeof(uint8_t));
  cudaMalloc((void **)&d_post_spikes, n_neurons * sizeof(uint8_t));
  cudaMalloc((void **)&d_pre_trace, n_neurons * sizeof(float));
  cudaMalloc((void **)&d_post_trace, n_neurons * sizeof(float));
  cudaMalloc((void **)&d_refactory, n_neurons * sizeof(uint8_t));
  cudaMalloc((void **)&d_thresholds, n_neurons * sizeof(float));

  cudaMemcpy(d_membranes, data->membranes, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conns, data->conns, n_neurons * n_conns * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, data->weights, n_neurons * n_conns * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_thresholds, data->thresholds, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_pre_trace, data->pre_trace, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_post_trace, data->post_trace, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_post_spikes, data->post_spikes, n_neurons * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_refactory, data->refactory, n_neurons * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Pointers that live on device
  d_data->membranes = d_membranes;
  d_data->conns = d_conns;
  d_data->weights = d_weights;
  d_data->pre_spikes = d_pre_spikes;
  d_data->post_spikes = d_post_spikes;
  d_data->pre_trace = d_pre_trace;
  d_data->post_trace = d_post_trace;
  d_data->refactory = d_refactory;
  d_data->thresholds = d_thresholds;

  d_net->data = d_data;

  // Doesn't have to live on device. Values are called
  d_net->config = config;
  return d_net;
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
