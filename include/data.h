#include "config.h"
#include "encode.h"
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#ifndef DATA_H
#define DATA_H

typedef struct {
  float *membranes;
  size_t *conns;
  float *weights;
  float *thresholds;
  uint8_t *post_spikes;
  uint8_t *pre_spikes;
  uint8_t *refactory;
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

typedef struct {
  pthread_mutex_t mutex;
  NetworkData *front;
  NetworkData *back;
  Config *config;
  SpikeTrain *input;
  int timestep;
  pthread_cond_t render_done;
  pthread_cond_t compute_done;
  int frame_ready;
} data_mutex_t;

#endif
int init_data(Config *config, NetworkData *data, SpikeTrain *input_train);
void free_data(NetworkData *data);
