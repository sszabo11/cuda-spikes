#include "data.h"
#include "spike.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cudaError_t process(Config *config, NetworkData *data) {

  // State of voltage of each neuron at timestep
  // float *h_membranes = (float *)malloc(sizeof(float) * config->n_neurons);

  //// Store connections for each neuron
  // int *h_conns = generate_connections(config->n_neurons, config->n_conns);

  //// Store weights for each neurons connection
  // float *h_weights = generate_weights(config->n_neurons, config->n_neurons,
  //                                     config->w_min, config->w_max);

  //// State of ALL neurons at last timestep
  // int *h_pre_spikes = (int *)malloc(sizeof(int) * config->n_neurons);

  // int *h_post_spikes = (int *)malloc(sizeof(int) * config->n_neurons);

  // float *h_thresholds = (float *)malloc(sizeof(float) * config->n_neurons);

  // float *h_pre_trace = (float *)malloc(sizeof(float) * config->n_neurons);

  // float *h_post_trace = (float *)malloc(sizeof(float) * config->n_neurons);

  // int *h_refactory = (int *)malloc(sizeof(int) * config->n_neurons);

  //  memset(data->pre_trace, 0, config->n_neurons * sizeof(float));
  //  memset(data->post_trace, 0, config->n_neurons * sizeof(float));
  //  memset(data->refactory, 0, config->n_neurons * sizeof(int));

  // for (int i = 0; i < config->n_neurons; i++) {
  //   double random_num = (double)rand() / ((double)RAND_MAX + 1.0);
  //   double random_num2 = (double)rand() / ((double)RAND_MAX + 1.0);
  //   double random_num3 = (double)rand() / ((double)RAND_MAX + 1.0);

  //  data->membranes[i] = random_num;
  //  data->pre_spikes[i] = random_num > config->sparsity ? 0 : 1;
  //  data->post_spikes[i] = random_num2 > config->sparsity ? 0 : 1;
  //  data->thresholds[i] = random_num3;
  //}

  // int fired = 0;
  // for (int i = 0; i < config->n_neurons; i++) {
  //   // printf("%d", membranes[i]);
  //   if (data->pre_spikes[i] == 1.0) {
  //     fired++;
  //   }
  // }

  int const T = 10;

  printf("\n\n%f\n", data->membranes[64]);

  float *d_membranes;
  int *d_conns;
  float *d_weights;
  float *d_thresholds;
  float *d_pre_trace;
  float *d_post_trace;
  int *d_post_spikes;
  int *d_refactory;
  int *d_pre_spikes;

  int n_neurons = config->n_neurons;
  int n_conns = config->n_conns;

  cudaMalloc((void **)&d_membranes, n_neurons * sizeof(float));
  cudaMalloc((void **)&d_conns, n_neurons * n_conns * sizeof(int));
  cudaMalloc((void **)&d_weights, n_neurons * n_conns * sizeof(float));
  cudaMalloc((void **)&d_pre_spikes, n_neurons * sizeof(int));

  cudaMalloc((void **)&d_post_spikes, n_neurons * sizeof(int));
  cudaMalloc((void **)&d_pre_trace, n_neurons * sizeof(float));
  cudaMalloc((void **)&d_post_trace, n_neurons * sizeof(float));
  cudaMalloc((void **)&d_refactory, n_neurons * sizeof(int));
  cudaMalloc((void **)&d_thresholds, n_neurons * sizeof(float));

  cudaMemcpy(d_membranes, data->membranes, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conns, data->conns, n_neurons * n_conns * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, data->weights, n_neurons * n_conns * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_pre_spikes, data->pre_spikes, n_neurons * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_thresholds, data->thresholds, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_pre_trace, data->pre_trace, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_post_trace, data->post_trace, n_neurons * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_post_spikes, data->post_spikes, n_neurons * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_refactory, data->refactory, n_neurons * sizeof(int),
             cudaMemcpyHostToDevice);

  for (int t = 0; t < T; t++) {
    cudaError_t err = run_kernels(config, data);
    printf("err: %d\n", err);
  }
  cudaMemcpy(data->membranes, d_membranes, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->conns, d_conns, n_neurons * n_conns * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->weights, d_weights, n_neurons * n_conns * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->pre_spikes, d_pre_spikes, n_neurons * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->thresholds, d_thresholds, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->pre_trace, d_pre_trace, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->post_trace, d_post_trace, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->post_spikes, d_post_spikes, n_neurons * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->refactory, d_refactory, n_neurons * sizeof(int),
             cudaMemcpyDeviceToHost);

  printf("\n\n%f\n", data->membranes[64]);
  cudaFree(d_membranes);
  cudaFree(d_conns);
  cudaFree(d_pre_spikes);
  cudaFree(d_weights);
  cudaFree(d_refactory);
  cudaFree(d_post_spikes);
  cudaFree(d_pre_trace);
  cudaFree(d_post_trace);
  cudaFree(d_thresholds);

  return cudaSuccess;
}
