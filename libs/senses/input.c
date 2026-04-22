#include "input.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

ConnectedLayer *input_layer(int n_conns, int out_n, int in_n, int T) {
  ConnectedLayer *input = malloc(sizeof(ConnectedLayer));

  input->conns = malloc(n_conns * in_n * sizeof(size_t));
  input->weights = malloc(n_conns * in_n * sizeof(float));
  input->out_n = out_n;
  input->in_n = in_n;
  input->T = T;

  size_t *conns = generate_connections(in_n, out_n, n_conns);
  float *weights = generate_weights(in_n, n_conns, 0.1, 0.8);

  input->conns = conns;
  input->weights = weights;

  printf("Input layer complete!");

  return input;
}
