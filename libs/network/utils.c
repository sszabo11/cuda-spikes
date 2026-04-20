#include <stdlib.h>

int *generate_connections(int in_n, int out_n, int n_conns) {
  int *conns = (int *)malloc(sizeof(int) * in_n * n_conns);

  if (conns == NULL) {
    return NULL;
  }

  for (int n = 0; n < in_n; n++) {
    for (int c = 0; c < n_conns; c++) {
      conns[n * n_conns + c] = (rand() % (out_n - 1 - 0 + 1));
    }
  }
  return conns;
}

float *generate_weights(int in_n, int n_conns, float min, float max) {
  float *weights = (float *)malloc(sizeof(float) * in_n * n_conns);

  if (weights == NULL) {
    return NULL;
  }

  for (int n = 0; n < in_n; n++) {
    for (int c = 0; c < n_conns; c++) {
      float scale = (float)rand() / (float)RAND_MAX;
      weights[n * n_conns + c] = min + scale * (max - min);
    }
  }
  return weights;
}
