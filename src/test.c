#include "config.h"
#include "data.h"
#include "eye.h"
#include "process.h"
#include "render.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SUPPORT_FILEFORMAT_JPG 1

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

  init_data(config, data);

  for (int i = 0; i < n_neurons; i++) {
    printf("\nNeuron %d: %f", i, data->membranes[i]);
  }
  printf("\n");

  // printf("\nDone: %f\n", data->membranes[0]);

  for (int i = 0; i < n_neurons; i++) {
    printf("\nNeuron %d: %f", i, data->membranes[i]);
  }

  ImageData *img_data = get_image_data("../data/Balcony-ja.png");
  EyeReceptor *eye = malloc(sizeof(EyeReceptor));

  eye->kernel_size = 3;
  eye->stride = 1;
  eye->width = img_data->width;
  eye->height = img_data->height;
  eye->kernel = malloc(sizeof(float) * img_data->width * img_data->height);

  float kernel1[] = {0.1, 0.1, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 1.0};
  float kernel2[] = {0.1, 0.1, 0.1, 0.1, 9.0, 0.1, 0.1, 0.1, 0.1};
  float kernel3[] = {0.1, 0.1, -5.0, 0.1, 0.1, -5.0, 0.1, 0.1, -5.0};
  float kernel4[] = {1.1, 0.1, -1.0, 1.1, 0.1, -1.0, 1.1, 0.1, -1.0};
  float kernel5[] = {4.0, -1.0, -1.0, 4.0, -0.0, -0.0, 4.0, -1.0, -0.0};
  float kernel6[] = {1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0};
  for (int kx = 0; kx < eye->kernel_size; kx++) {
    for (int ky = 0; ky < eye->kernel_size; ky++) {
      // eye->kernel[ky * eye->kernel_size + kx] = 1.0;
      eye->kernel[ky * eye->kernel_size + kx] =
          kernel6[ky * eye->kernel_size + kx];
    }
  }

  printf("image w: %d | h: %d\n", img_data->width, img_data->height);

  ReceptorResponse *output = receptor(*eye, img_data);
  printf("\nDone");

  // for (int i = 0; i < eye->width * eye->height; i++) {
  //   printf("\n(%f, %f, %f)", output->output_layer_r[i],
  //          output->output_layer_g[i], output->output_layer_b[i]);
  // }

  Image *processed_img = malloc(sizeof(Image));

  processed_img->width = output->out_w;
  processed_img->height = output->out_h;
  processed_img->format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  processed_img->data = output->pixels;
  processed_img->mipmaps = 1;

  ExportImage(*processed_img, "../data/output3-9.png");

  // for (int p = 0; p < 10; p++) {
  //   cudaError_t res = process(config, data);
  // }
  // render(config, data);

  free_data(data);
  free(config);
}
