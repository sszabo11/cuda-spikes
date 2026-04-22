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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));
  int n_neurons = 100;
  int n_conns = 10;

  float sparsity = 0.02;

  Config *config = malloc(sizeof(Config));
  NetworkData *data = malloc(sizeof(NetworkData));
  int const T = 1;

  config->n_neurons = n_neurons;
  config->n_conns = n_conns;
  config->tau_minus = 0.95;
  config->tau_plus = 0.95;
  config->beta = 0.8;
  config->a_plus = 0.01;
  config->a_minus = 0.012;
  config->w_min = 0.1;
  config->w_max = 0.9;
  config->base_threshold = 0.5;
  config->sparsity = sparsity;
  config->T = T;

  ImageData *img_data = get_image_data("../data/desktop1.png");

  EyeReceptor *eye = malloc(sizeof(EyeReceptor));
  eye->kernel_size = 5;
  eye->stride = 1;
  eye->width = img_data->width;
  eye->height = img_data->height;

  size_t ksize = eye->kernel_size * eye->kernel_size;
  eye->kernel = malloc(sizeof(float) * ksize);
  eye->kernel_x = malloc(sizeof(float) * ksize);
  eye->kernel_y = malloc(sizeof(float) * ksize);

  float kernel_laplacian[] = {0.0,  0.0,  -1.0, 0.0,  0.0,  0.0,  -1.0,
                              -2.0, -1.0, 0.0,  -1.0, -2.0, 16.0, -2.0,
                              -1.0, 0.0,  -1.0, -2.0, -1.0, 0.0,  0.0,
                              0.0,  -1.0, 0.0,  0.0};
  float kernel_sobel_x[] = {0, 0, 0, 0,  0, 0, -1, 0, 1, 0, 0, -2, 0,
                            2, 0, 0, -1, 0, 1, 0,  0, 0, 0, 0, 0};

  float kernel_sobel_y[] = {0, 0, 0, 0,  0,  0,  1, 2, 1, 0, 0, 0, 0,
                            0, 0, 0, -1, -2, -1, 0, 0, 0, 0, 0, 0};

  Image *output = process_img(eye, kernel_laplacian, kernel_sobel_x,
                              kernel_sobel_y, img_data);
  printf("\nProcessed image.");

  SpikeTrain *encoded_data = rate_encode(img_data, T);
  printf("Rate encoded");

  init_data(config, data, encoded_data);
  printf("Initalized data");

  init_logs();

  pthread_t render_thread, compute_thread;

  data_mutex_t data_m;
  data_m.front = data;
  data_m.back = data;
  data_m.config = config;
  data_m.input = encoded_data;
  data_m.timestep = 0;

  pthread_mutex_init(&data_m.mutex, NULL);
  pthread_cond_init(&data_m.compute_done, NULL);
  pthread_cond_init(&data_m.render_done, NULL);
  data_m.frame_ready = 0;

  pthread_create(&render_thread, NULL, (void *)render_from_thread, &data_m);
  pthread_create(&compute_thread, NULL, (void *)process_from_thread, &data_m);

  pthread_join(render_thread, NULL);
  pthread_join(compute_thread, NULL);

  // render(config, data, encoded_data);

  pthread_mutex_destroy(&data_m.mutex);
  pthread_cond_destroy(&data_m.compute_done);
  pthread_cond_destroy(&data_m.render_done);
  free_data(data);
  free(config);
  UnloadImageColors(img_data->pixels);
  free(eye->kernel);
  free(eye->kernel_x);
  free(eye->kernel_y);
  free(eye);
}
