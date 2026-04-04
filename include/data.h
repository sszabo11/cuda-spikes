#include "config.h"

#ifndef DATA_H
#define DATA_H

typedef struct {
  float *membranes;
  int *conns;
  float *weights;
  float *thresholds;
  int *post_spikes;
  int *pre_spikes;
  int *refactory;
  float *pre_trace;
  float *post_trace;
} NetworkData;

typedef struct {
} InputData;
typedef struct {
} OutputData;

typedef struct {
  NetworkData *data;
  InputData *input_data;
  OutputData *output_data;
} AllData;

#endif
int init_data(Config *config, NetworkData *data);
void free_data(NetworkData *data);
