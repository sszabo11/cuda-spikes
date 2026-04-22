#include "data.h"
#include "spike.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cudaError_t process(Config *config, NetworkData *data) {
  if (!config->T) {
    printf("Error: Timesteps not specified!");
    return cudaErrorInvalidConfiguration;
  }
  if (!config->n_neurons) {
    printf("Error: Number of neurons not specified!");
    return cudaErrorInvalidConfiguration;
  }
  if (!config->n_conns) {
    printf("Error: Connections not specified!");
    return cudaErrorInvalidConfiguration;
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

  NetworkData d_data;

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
  cudaMemcpy(d_pre_spikes, data->pre_spikes, n_neurons * sizeof(uint8_t),
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

  d_data.membranes = d_membranes;
  d_data.conns = d_conns;
  d_data.weights = d_weights;
  d_data.pre_spikes = d_pre_spikes;
  d_data.post_spikes = d_post_spikes;
  d_data.pre_trace = d_pre_trace;
  d_data.post_trace = d_post_trace;
  d_data.refactory = d_refactory;
  d_data.thresholds = d_thresholds;

  // Input encode one timestep
  // cudaError_t err_i = run_kernels(config, &d_data);

  // Network process one timestep
  cudaError_t err_n = run_kernels(config, &d_data);

  // Output decode on timestep
  // cudaError_t err_o = run_kernels(config, &d_data);

  cudaMemcpy(data->membranes, d_membranes, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->conns, d_conns, n_neurons * n_conns * sizeof(size_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->weights, d_weights, n_neurons * n_conns * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->pre_spikes, d_pre_spikes, n_neurons * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data->thresholds, d_thresholds, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->pre_trace, d_pre_trace, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->post_trace, d_post_trace, n_neurons * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->post_spikes, d_post_spikes, n_neurons * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(data->refactory, d_refactory, n_neurons * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_membranes);
  cudaFree(d_conns);
  cudaFree(d_pre_spikes);
  cudaFree(d_weights);
  cudaFree(d_refactory);
  cudaFree(d_post_spikes);
  cudaFree(d_pre_trace);
  cudaFree(d_post_trace);
  cudaFree(d_thresholds);

  return err_n;
}
