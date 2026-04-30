#include "mnist.h"
#include "compute.h"
#include "config.h"
#include "encode.h"
#include "eye.h"
#include "logger.h"
#include "network.h"
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

  Network *h_net = network_create();
  init_network(h_net, config);
  printf("Initalized Network\n");

  mnist_dataset_t *train_dataset, *test_dataset;
  mnist_dataset_t batch;
  float loss, accuracy;
  int i, batches;

  // Read the datasets from the files
  train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
  test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

  // Print digits - works
  // for (int i = 0; i < 10; i++) {
  //  mnist_image_t *img = &train_dataset->images[i];

  //  for (int row = 0; row < 28; row++) {
  //    for (int col = 0; col < 28; col++) {
  //      // for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {

  //      float pixel = img->pixels[row * 28 + col];
  //      int y = i / MNIST_IMAGE_WIDTH;
  //      int x = i % MNIST_IMAGE_WIDTH;

  //      // printf("%f", pixel);

  //      if (img->pixels[row * 28 + col] > 200.0) {
  //        printf("██");
  //      } else {
  //        printf("  ");
  //      };
  //    }
  //    printf("\n");
  //  }
  //}
  // return 1;

  init_logs();
  // pthread_t render_thread, compute_thread;

  // Set weights, conns, thresholds etc

  Network *d_net = send_network_to_gpu(h_net);
  printf("Sent to GPU\n");

  // Training
  for (int img_idx = 0; img_idx < 100; img_idx++) {
    if (img_idx % 10 == 0) {
      printf("Idx: %d\n", img_idx);
    }
    mnist_image_t *img = &train_dataset->images[img_idx];
    SpikeTrain *st = encode_mnist(img, T);
    // for (int i = 0; i < 728; i++) {
    //   printf("st: %d\n", st->data[i]);
    // }
    compute(d_net, h_net, st, img_idx);

    free(st->data);
    free(st);
  }

  network_destroy(h_net);
  mnist_free_dataset(train_dataset);
  mnist_free_dataset(test_dataset);
}
