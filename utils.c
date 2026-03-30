#include <stdlib.h>

int *generate_connections(int n_neurons, int n_conns) {
  int *conns = (int *)malloc(sizeof(int) * n_neurons * n_conns);

  if (conns == NULL) {
    return NULL;
  }

  for (int n = 0; n < n_neurons; n++) {
    for (int c = 0; c < n_conns; c++) {
      conns[n * n_conns + c] = (rand() % (n_neurons - 1 - 0 + 1));
    }
  }
  return conns;
}

float *generate_weights(int n_neurons, int n_conns, float min, float max) {
  float *weights = (float *)malloc(sizeof(float) * n_neurons * n_conns);

  if (weights == NULL) {
    return NULL;
  }

  for (int n = 0; n < n_neurons; n++) {
    for (int c = 0; c < n_conns; c++) {
      float scale = (float)rand() / (float)RAND_MAX;
      weights[n * n_conns + c] = min + scale * (max - min);
    }
  }
  return weights;
}
