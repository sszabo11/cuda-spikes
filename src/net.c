#include "config.h"
#include "data.h"
#include "encode.h"
#include "eye.h"
#include "process.h"
#include "render.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));
  int n_neurons = 1000;
  int n_conns = 20;

  float sparsity = 0.02;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));
  int const T = 1;

  config->n_neurons = n_neurons;
  config->n_conns = n_conns;
  config->tau_minus = 0.9;
  config->tau_plus = 0.9;
  config->beta = 0.8;
  config->a_plus = 0.001;
  config->a_minus = 0.001;
  config->w_min = 0.3;
  config->w_max = 0.6;
  config->base_threshold = 0.3;
  config->sparsity = sparsity;
  config->T = T;

  ImageData *img_data = get_image_data("../data/desktop1.png");
  EyeReceptor *eye = malloc(sizeof(EyeReceptor));

  eye->kernel_size = 5;
  eye->stride = 1;
  eye->width = img_data->width;
  eye->height = img_data->height;
  eye->kernel = malloc(sizeof(float) * img_data->width * img_data->height);
  eye->kernel_x = malloc(sizeof(float) * img_data->width * img_data->height);
  eye->kernel_y = malloc(sizeof(float) * img_data->width * img_data->height);

  float kernel_laplacian[] = {0.0,  0.0,  -1.0, 0.0,  0.0,  0.0,  -1.0,
                              -2.0, -1.0, 0.0,  -1.0, -2.0, 16.0, -2.0,
                              -1.0, 0.0,  -1.0, -2.0, -1.0, 0.0,  0.0,
                              0.0,  -1.0, 0.0,  0.0};
  float kernel_sobel_x[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
  float kernel_sobel_y[] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};

  Image *output = process_img(eye, kernel_laplacian, kernel_sobel_x,
                              kernel_sobel_y, img_data);
  printf("\nProcessed image.");

  SpikeTrain *encoded_data = rate_encode(img_data, T);
  printf("Rate encoded");

  init_data(config, data, encoded_data);
  printf("Initalized data");

  render(config, data, encoded_data);

  free_data(data);
  free(config);
}
