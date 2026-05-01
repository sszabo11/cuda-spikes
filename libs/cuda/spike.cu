
#include "spike.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "network.h"
}

__global__ void inject_input_kernel(uint8_t *input_spikes, size_t *input_idxs,
                                    uint8_t *spikes, int n_neurons, int n_input,
                                    int T, int t) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n_input) {
    return;
  }

  size_t idx = input_idxs[i];

  uint8_t fired = input_spikes[i * T + t] == 1;
  // printf("i: %d, idx: %d, input: %d\n", i, (int)idx, (int)input_spikes[i]);

  spikes[idx] = fired; // Inject the spike into the corrosponding input neuron
}
__global__ void reset_net_kernel(float *membranes, uint8_t *post_spikes,
                                 uint8_t *pre_spikes, uint8_t *refactory,
                                 float *pre_trace, float *post_trace,
                                 int n_neurons) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n_neurons) {
    return;
  }

  membranes[i] = 0.0;
  refactory[i] = 0;
  pre_spikes[i] = 0;
  post_spikes[i] = 0;
  pre_trace[i] = 0.0;
  post_trace[i] = 0.0;
}
__global__ void propagate_kernel(float *membranes, size_t *conns,
                                 float *weights, uint8_t *spikes, int n_neurons,
                                 int n_conns) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n_neurons)
    return;

  float sum = 0.0;
  for (int j = 0; j < n_conns; j++) {
    int pre = conns[i * n_conns + j];
    sum += spikes[pre] * weights[i * n_conns + j];
  }

  membranes[i] += sum;
}

__global__ void update_kernel(float *membranes, size_t *conns, float *weights,
                              float *thresholds, uint8_t *post_spikes,
                              uint8_t *pre_spikes, uint8_t *refactory,
                              float *pre_trace, float *post_trace,
                              float tau_plus, float tau_minus,
                              float base_threshold, float beta, float a_plus,
                              float a_minus, float w_min, float w_max,
                              int n_neurons, int n_conns, bool learn) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n_neurons)
    return;

  pre_trace[i] *= tau_plus;
  post_trace[i] *= tau_minus;

  // Neuron is not inhibited
  if (refactory[i] > 0) {
    refactory[i] -= 1;
    membranes[i] *= beta;
  } else {
    // Decay
    membranes[i] *= beta;
  }
  float thresh = base_threshold + thresholds[i];
  int spiked = membranes[i] >= thresh;
  if (spiked) {
    post_spikes[i] = 1;

    membranes[i] = 0.0; // Reset

    refactory[i] = 5; // Inhibit for 3 timesteps
    post_trace[i] += 1.0;
  } else {
    post_spikes[i] = 0;
  }

  if (spiked)
    pre_trace[i] += 1.0;

  if (learn) {
    for (int j = 0; j < n_conns; j++) {
      int conn = conns[i * n_conns + j];
      float weight = weights[i * n_conns + j];

      if (spiked) {
        weight += a_plus * pre_trace[conn];
      }

      if (pre_spikes[conn]) {
        weight -= a_minus * post_trace[i];
      }

      weights[i * n_conns + j] = fmaxf(w_min, fminf(w_max, weight));
    }
  }
}

cudaError_t inject_input(Network *net, int t) {
  const int THREADS_PER_BLOCK = 512;
  Data *data = net->data;
  Config *config = net->config;

  int n_blocks = (net->input_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  inject_input_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->input_spikes, data->input_idxs, data->pre_spikes, config->n_neurons,
      net->input_dim, net->config->T, t);

  cudaDeviceSynchronize();

  return cudaSuccess;
}

cudaError_t propagate(Network *net) {
  const int THREADS_PER_BLOCK = 512;
  Data *data = net->data;
  Config *config = net->config;

  int n_blocks =
      (config->n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  propagate_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->conns, data->weights, data->pre_spikes,
      config->n_neurons, config->n_conns);

  cudaDeviceSynchronize();
}

cudaError_t run_network(Network *net, bool learn) {

  const int THREADS_PER_BLOCK = 512;
  Data *data = net->data;
  Config *config = net->config;

  int n_blocks =
      (config->n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  propagate_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->conns, data->weights, data->pre_spikes,
      config->n_neurons, config->n_conns);

  cudaDeviceSynchronize();

  update_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->conns, data->weights, data->thresholds,
      data->post_spikes, data->pre_spikes, data->refactory, data->pre_trace,
      data->post_trace, config->tau_plus, config->tau_minus,
      config->base_threshold, config->beta, config->a_plus, config->a_minus,
      config->w_min, config->w_max, config->n_neurons, config->n_conns, learn);
  return cudaSuccess;
}

cudaError_t update(Network *net) {
  const int THREADS_PER_BLOCK = 512;
  Data *data = net->data;
  Config *config = net->config;

  int n_blocks =
      (config->n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  update_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->conns, data->weights, data->thresholds,
      data->post_spikes, data->pre_spikes, data->refactory, data->pre_trace,
      data->post_trace, config->tau_plus, config->tau_minus,
      config->base_threshold, config->beta, config->a_plus, config->a_minus,
      config->w_min, config->w_max, config->n_neurons, config->n_conns, true);

  return cudaSuccess;
}

cudaError_t reset_net(Network *net) {
  const int THREADS_PER_BLOCK = 512;
  Data *data = net->data;
  Config *config = net->config;

  int n_blocks =
      (config->n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  reset_net_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->post_spikes, data->pre_spikes, data->refactory,
      data->pre_trace, data->post_trace, config->n_neurons);

  return cudaSuccess;
}
