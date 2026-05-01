#include "cuda_runtime_api.h"
#include "data.h"
#include "driver_types.h"
#include "encode.h"
#include "logger.h"
#include "mnist.h"
#include "network.h"
#include "spike.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cudaError_t copy_input_to_gpu(Network *net, SpikeTrain *st) {

  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      int pixel_idx = y * 28 + x;
      // printf("st: %d\n", GET_SPIKE(st, pixel_idx, 0));
    }
  }
  // cudaError_t err =
  //     cudaMemcpy(net->data->input_train, st->data, st->width *
  //     sizeof(uint8_t),
  //                cudaMemcpyHostToDevice);
  cudaError_t err =
      cudaMemcpy(net->data->input_spikes, st->data,
                 net->input_dim * sizeof(uint8_t), cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr, "copy_input_to_gpu failed: %s\n", cudaGetErrorString(err));
  }
  return err;
}

cudaError_t copy_results_back_to_cpu(Network *d_net, Network *h_net) {

  cudaError_t err = cudaMemcpy(
      h_net->data->post_spikes, d_net->data->post_spikes,
      h_net->config->n_neurons * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  cudaError_t err2 = cudaMemcpy(
      h_net->data->pre_spikes, d_net->data->pre_spikes,
      h_net->config->n_neurons * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_net->data->weights, d_net->data->weights,
             h_net->config->n_neurons * h_net->config->n_conns * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "copy_results_back failed: %s\n", cudaGetErrorString(err));
  }
  return err;
}

cudaError_t compute(Network *d_net, Network *h_net, SpikeTrain *input,
                    int sample_id, bool learn) {
  // Set pre_spikes to input spikes
  copy_input_to_gpu(d_net, input);

  if (d_net->input_dim == 0) {
    printf("Error: Input dim is set to 0\n");
    return cudaErrorInvalidConfiguration;
  }

  // copy_results_back_to_cpu(d_net, h_net);
  // int fired = 0;
  // for (int i = 0; i < h_net->config->n_neurons; i++) {
  //   if (h_net->data->pre_spikes[i] == 1) {
  //     fired++;
  //   };
  //   // printf("%d\n", h_net->data->pre_spikes[i]);
  // }
  // printf("Injected: %d\n", fired);

  // Computation steps
  for (int t = 0; t < d_net->config->T; t++) {
    inject_input(d_net, t);
    if (sample_id == 0) {
      copy_results_back_to_cpu(d_net, h_net);

      // for (int i = 0; i < h_net->config->n_neurons; i++) {
      //   printf("%d\n", h_net->data->post_spikes[i]);
      // }
      // write_to_csv(h_net, t, 1);
    }
    run_network(d_net, learn);
  }

  return cudaSuccess;
}

cudaError_t test_images(Network *d_net, Network *h_net, mnist_dataset_t *imgs) {
  int n_neurons = h_net->config->n_neurons;

  // Allocate spike count accumulator once, reuse per image
  uint32_t *spike_counts = calloc(n_neurons, sizeof(uint32_t));
  if (!spike_counts) {
    fprintf(stderr, "test_images: failed to allocate spike_counts\n");
    return cudaErrorMemoryAllocation;
  }

  // Write header if digit_responses.csv is new
  init_digit_log(n_neurons);

  for (int img_idx = 0; img_idx < 1000; img_idx++) {
    mnist_image_t *img = &imgs->images[img_idx];
    uint8_t label = imgs->labels[img_idx];

    SpikeTrain *st = encode_mnist(img, h_net->config->T);

    copy_input_to_gpu(d_net, st);

    if (d_net->input_dim == 0) {
      printf("Error: Input dim is set to 0\n");
      free(spike_counts);
      free(st->data);
      free(st);
      return cudaErrorInvalidConfiguration;
    }

    // Zero the accumulator for this image
    memset(spike_counts, 0, n_neurons * sizeof(uint32_t));

    // Run all timesteps, accumulate spikes
    for (int t = 0; t < d_net->config->T; t++) {
      inject_input(d_net, t);
      copy_results_back_to_cpu(d_net, h_net);

      // Accumulate post spikes for this timestep
      for (int i = 0; i < n_neurons; i++) {
        spike_counts[i] += h_net->data->post_spikes[i];
      }

      // write_to_csv(h_net, t, 1);
      run_network(d_net, false);
      if (img_idx == 0) {
        write_to_csv(h_net, t, 1);
      }
    }

    // Write one row per image: img_idx, label, count_per_neuron
    write_digit_response(h_net, img_idx, label, spike_counts);

    reset_net(d_net);
    if (img_idx % 10 == 0) {
      printf("Test image %d / 100 (label: %d)\n", img_idx, label);
    }

    free(st->data);
    free(st);
  }

  free(spike_counts);
  return cudaSuccess;
}
