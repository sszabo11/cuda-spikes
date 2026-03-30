#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
  int n_neurons;
  int n_conns;
  float beta;
  float tau_minus;
  float tau_plus;
  float a_plus;
  float a_minus;
  float w_min;
  float w_max;
  float base_threshold;
  float sparsity;
} Config;

#endif
