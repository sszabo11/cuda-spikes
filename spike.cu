#include "config.h"
#include "data.h"
#include "spike.h"
#include <math.h>
#include <stdlib.h>

__global__ void propagate(float *membranes, int *conns, float *weights,
                          int *spikes, int n_neurons, int n_conns) {
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

__global__ void update(float *membranes, int *conns, float *weights,
                       float *thresholds, int *post_spikes, int *pre_spikes,
                       int *refactory, float *pre_trace, float *post_trace,
                       float tau_plus, float tau_minus, float base_threshold,
                       int beta, float a_plus, float a_minus, float w_min,
                       float w_max, int n_neurons, int n_conns) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n_neurons)
    return;

  pre_trace[i] *= tau_plus;
  post_trace[i] *= tau_minus;

  float new_v;

  // Neuron is not inhibited
  if (refactory[i] > 0) {
    refactory[i] -= 1;
  } else {
    // Decay
    membranes[i] = membranes[i] * beta;
  }
  float thresh = base_threshold + thresholds[i];
  int spiked = membranes[i] >= thresh;
  if (spiked) {
    post_spikes[i] = 1;
    membranes[i] = 0.0; // Reset

    refactory[i] = 3; // Inhibit for 3 timesteps
    post_trace[i] += 1.0;
  }

  if (spiked)
    pre_trace[i] += 1.0;

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

cudaError_t run_kernels(Config *config, NetworkData *data) {

  const int THREADS_PER_BLOCK = 512;
  int n_blocks =
      (config->n_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  propagate<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->conns, data->weights, data->pre_spikes,
      config->n_neurons, config->n_conns);

  cudaDeviceSynchronize();

  update<<<n_blocks, THREADS_PER_BLOCK>>>(
      data->membranes, data->conns, data->weights, data->thresholds,
      data->post_spikes, data->pre_spikes, data->refactory, data->pre_trace,
      data->post_trace, config->tau_plus, config->tau_minus,
      config->base_threshold, config->beta, config->a_plus, config->a_minus,
      config->w_min, config->w_max, config->n_neurons, config->n_conns);

  return cudaSuccess;
}
