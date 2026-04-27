#include "mnist.h"
#include "config.h"
#include "data.h"
#include "encode.h"
#include "eye.h"
#include "logger.h"
#include "process.h"
#include "render.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <raylib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const char *train_images_file = "../data/mnist/train-images-idx3-ubyte";
const char *train_labels_file = "../data/mnist/train-labels-idx1-ubyte";
const char *test_images_file = "../data/mnist/t10k-images-idx3-ubyte";
const char *test_labels_file = "../data/mnist/t10k-labels-idx1-ubyte";

int main() {
  srand(time(NULL));
  int n_neurons = 784;
  int n_conns = 10;

  float sparsity = 0.02;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));
  int const T = 300;

  config->n_neurons = n_neurons;
  config->n_conns = n_conns;
  config->tau_minus = 0.95;
  config->tau_plus = 0.95;
  config->beta = 0.8;
  config->a_plus = 0.003;
  config->a_minus = 0.0035;
  config->w_min = 0.1;
  config->w_max = 0.9;
  config->base_threshold = 0.5;
  config->sparsity = sparsity;
  config->T = T;

  mnist_dataset_t *train_dataset, *test_dataset;
  mnist_dataset_t batch;
  float loss, accuracy;
  int i, batches;

  // Read the datasets from the files
  train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
  test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

  // Initialise a new batch
  // mnist_batch(train_dataset, &batch, 100, i % batches);

  init_logs();
  pthread_t render_thread, compute_thread;

  data_mutex_t data_m;
  // data_m.input = malloc(sizeof(SpikeTrain));
  init_data(config, data, NULL);

  for (int img_idx = 0; img_idx < 500; img_idx++) {
    if (img_idx % 10 == 0) {
      printf("Idx: %d\n", img_idx);
    }
    mnist_image_t *img = &train_dataset->images[img_idx];
    SpikeTrain *st = encode_mnist(img, T);
    // printf("Encoded\n");

    // for (int t = 0; t < T; t++) {
    //  load_input_spikes(data, st, config, t);
    data_m.front = data;
    data_m.back = data;
    data_m.config = config;
    data_m.input = st;
    data_m.timestep = 0;
    data_m.samples_done = img_idx;

    // printf("%d\n", *data_m.input[200].data);
    //  data_m.timestep = t;

    // pthread_mutex_init(&data_m.mutex, NULL);
    // pthread_cond_init(&data_m.compute_done, NULL);
    // pthread_cond_init(&data_m.render_done, NULL);
    // data_m.frame_ready = 0;

    // pthread_create(&render_thread, NULL, (void *)render_from_thread,
    // &data_m); pthread_create(&compute_thread, NULL, (void
    // *)process_from_thread,
    //                &data_m);

    process(&data_m);
    // pthread_join(render_thread, NULL);
    // pthread_join(compute_thread, NULL);
    //}
  }

  // render(config, data, encoded_data);

  // pthread_mutex_destroy(&data_m.mutex);
  // pthread_cond_destroy(&data_m.compute_done);
  // pthread_cond_destroy(&data_m.render_done);
  free_data(data);
  free(config);
  mnist_free_dataset(train_dataset);
  mnist_free_dataset(test_dataset);
}
