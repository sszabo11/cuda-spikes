// #include "compute.h"
// #include "cuda_runtime_api.h"
// #include "data.h"
// #include "driver_types.h"
// #include "spike.h"
// #include <assert.h>
// #include <stdio.h>
// #include <stdlib.h>
//
// Network *network_create() {
//   Network *net = malloc(sizeof(Network)); // ← Now this works
//   if (!net)
//     return NULL;
//
//   net->data = malloc(sizeof(NetworkData));
//   net->config = malloc(sizeof(Config));
//
//   return net;
// }
//
// cudaError_t copy_input_to_gpu() { return cudaSuccess; }
// cudaError_t copy_results_back_to_cpu() { return cudaSuccess; }
//
//// Compute one sample through T timesteps.
//// Network data already in GPU
//// Just send input spikes
// cudaError_t compute(Network *d_data, data_mutex_t *obj) {
//   NetworkData *data = obj->back;
//   Config *config = obj->config;
//
//   // Set pre_spikes to input spikes
//   copy_input_to_gpu();
//
//   // Computation steps
//   for (int t = 0; t < obj->T; t++) {
//     run_kernels(config, d_data->data);
//   }
//
//   copy_results_back_to_cpu();
//
//   return cudaSuccess;
// }
//
// Network *send_network_to_gpu(Network *net) {
//   assert(net->data != NULL);
//   assert(net->config != NULL);
//   NetworkData *data = net->data;
//   Config *config = net->config;
//   printf("h: %f\n", data->thresholds[2]);
//
//   if (!config->T) {
//     printf("\nError: Timesteps not specified!\n");
//     return NULL;
//   }
//   if (!config->n_neurons) {
//     printf("\nError: Number of neurons not specified!\n");
//     return NULL;
//   }
//   if (!config->n_conns) {
//     printf("\nError: Connections not specified!\n");
//     return NULL;
//   }
//
//   float *d_membranes;
//   size_t *d_conns;
//   float *d_weights;
//   float *d_thresholds;
//   float *d_pre_trace;
//   float *d_post_trace;
//   uint8_t *d_post_spikes;
//   uint8_t *d_refactory;
//   uint8_t *d_pre_spikes;
//
//   int n_neurons = config->n_neurons;
//   int n_conns = config->n_conns;
//
//   Network *d_net = malloc(sizeof(Network));
//   NetworkData *d_data = malloc(sizeof(NetworkData));
//
//   cudaMalloc((void **)&d_membranes, n_neurons * sizeof(float));
//   cudaMalloc((void **)&d_conns, n_neurons * n_conns * sizeof(size_t));
//   cudaMalloc((void **)&d_weights, n_neurons * n_conns * sizeof(float));
//   cudaMalloc((void **)&d_pre_spikes, n_neurons * sizeof(uint8_t));
//   cudaMalloc((void **)&d_post_spikes, n_neurons * sizeof(uint8_t));
//   cudaMalloc((void **)&d_pre_trace, n_neurons * sizeof(float));
//   cudaMalloc((void **)&d_post_trace, n_neurons * sizeof(float));
//   cudaMalloc((void **)&d_refactory, n_neurons * sizeof(uint8_t));
//   cudaMalloc((void **)&d_thresholds, n_neurons * sizeof(float));
//
//   cudaMemcpy(d_membranes, data->membranes, n_neurons * sizeof(float),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_conns, data->conns, n_neurons * n_conns * sizeof(size_t),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_weights, data->weights, n_neurons * n_conns * sizeof(float),
//              cudaMemcpyHostToDevice);
//
//   cudaMemcpy(d_thresholds, data->thresholds, n_neurons * sizeof(float),
//              cudaMemcpyHostToDevice);
//
//   cudaMemcpy(d_pre_trace, data->pre_trace, n_neurons * sizeof(float),
//              cudaMemcpyHostToDevice);
//
//   cudaMemcpy(d_post_trace, data->post_trace, n_neurons * sizeof(float),
//              cudaMemcpyHostToDevice);
//
//   cudaMemcpy(d_post_spikes, data->post_spikes, n_neurons * sizeof(uint8_t),
//              cudaMemcpyHostToDevice);
//
//   cudaMemcpy(d_refactory, data->refactory, n_neurons * sizeof(uint8_t),
//              cudaMemcpyHostToDevice);
//
//   d_data->membranes = d_membranes;
//   d_data->conns = d_conns;
//   d_data->weights = d_weights;
//   d_data->pre_spikes = d_pre_spikes;
//   d_data->post_spikes = d_post_spikes;
//   d_data->pre_trace = d_pre_trace;
//   d_data->post_trace = d_post_trace;
//   d_data->refactory = d_refactory;
//   d_data->thresholds = d_thresholds;
//
//   d_net->data = d_data;
//   d_net->config = config;
//   return d_net;
// }
//
// cudaError_t compute_from_thread(data_mutex_t *obj) { return cudaSuccess; }
