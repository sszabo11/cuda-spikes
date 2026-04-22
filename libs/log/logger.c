#include "data.h"
#include <stdio.h>

// Call once before simulation loop
void init_logs() {
  // spikes
  FILE *f = fopen("../logs/spikes.csv", "w");
  fprintf(f, "t,neuron_id,fired\n");
  fclose(f);

  // membranes
  f = fopen("../logs/membranes.csv", "w");
  fprintf(f, "t,neuron_id,v\n");
  fclose(f);

  // weights — log every connection: pre, post, weight
  f = fopen("../logs/weights.csv", "w");
  fprintf(f, "t,pre,post,weight\n");
  fclose(f);

  // traces
  f = fopen("../logs/traces.csv", "w");
  fprintf(f, "t,neuron_id,pre_trace,post_trace\n");
  fclose(f);

  // refractory
  f = fopen("../logs/refractory.csv", "w");
  fprintf(f, "t,neuron_id,refractory\n");
  fclose(f);

  // thresholds
  f = fopen("../logs/thresholds.csv", "w");
  fprintf(f, "t,neuron_id,threshold\n");
  fclose(f);

  // population summary (one row per timestep)
  f = fopen("../logs/population.csv", "w");
  fprintf(f, "t,firing_rate,mean_membrane,mean_weight,mean_pre_trace,mean_post_"
             "trace,n_refractory\n");
  fclose(f);
}

// Call every timestep inside your simulation loop
// Pass log_weights_every = N to only log weights every N steps (they're large)
int write_to_csv(data_mutex_t *obj, int log_weights_every) {
  int t = obj->timestep;
  int n_neurons = obj->config->n_neurons;
  int n_conns = obj->config->n_conns;
  NetworkData *d = obj->front;

  // ── Spikes ────────────────────────────────────────────────────────────────
  FILE *f_spikes = fopen("../logs/spikes.csv", "a");
  if (!f_spikes) {
    printf("Error: spikes.csv\n");
    return 1;
  }

  // ── Membranes ─────────────────────────────────────────────────────────────
  FILE *f_mem = fopen("../logs/membranes.csv", "a");
  if (!f_mem) {
    fclose(f_spikes);
    printf("Error: membranes.csv\n");
    return 1;
  }

  // ── Traces ────────────────────────────────────────────────────────────────
  FILE *f_traces = fopen("../logs/traces.csv", "a");
  if (!f_traces) {
    fclose(f_spikes);
    fclose(f_mem);
    return 1;
  }

  // ── Refractory ────────────────────────────────────────────────────────────
  FILE *f_ref = fopen("../logs/refractory.csv", "a");
  if (!f_ref) {
    fclose(f_spikes);
    fclose(f_mem);
    fclose(f_traces);
    return 1;
  }

  // ── Thresholds ────────────────────────────────────────────────────────────
  FILE *f_thresh = fopen("../logs/thresholds.csv", "a");
  if (!f_thresh) {
    fclose(f_spikes);
    fclose(f_mem);
    fclose(f_traces);
    fclose(f_ref);
    return 1;
  }

  // accumulators for population summary
  int total_fired = 0;
  float sum_membrane = 0.0f;
  float sum_pre_trace = 0.0f;
  float sum_post_trace = 0.0f;
  int n_refractory = 0;

  for (int i = 0; i < n_neurons; i++) {
    uint8_t fired = d->post_spikes[i];
    float v = d->membranes[i];
    float pre_tr = d->pre_trace[i];
    float post_tr = d->post_trace[i];
    uint8_t ref = d->refactory[i];
    float thresh = d->thresholds[i];

    fprintf(f_spikes, "%d,%d,%d\n", t, i, fired);
    fprintf(f_mem, "%d,%d,%.6f\n", t, i, v);
    fprintf(f_traces, "%d,%d,%.6f,%.6f\n", t, i, pre_tr, post_tr);
    fprintf(f_ref, "%d,%d,%d\n", t, i, ref);
    fprintf(f_thresh, "%d,%d,%.6f\n", t, i, thresh);

    total_fired += fired;
    sum_membrane += v;
    sum_pre_trace += pre_tr;
    sum_post_trace += post_tr;
    n_refractory += ref;
  }

  fclose(f_spikes);
  fclose(f_mem);
  fclose(f_traces);
  fclose(f_ref);
  fclose(f_thresh);

  // ── Weights (every N timesteps) ───────────────────────────────────────────
  float sum_weight = 0.0f;
  int weight_count = n_neurons * n_conns;

  if (t % log_weights_every == 0) {
    FILE *f_w = fopen("../logs/weights.csv", "a");
    if (f_w) {
      for (int n = 0; n < n_neurons; n++) {
        for (int c = 0; c < n_conns; c++) {
          int post = (int)d->conns[n * n_conns + c];
          float weight = d->weights[n * n_conns + c];
          fprintf(f_w, "%d,%d,%d,%.6f\n", t, n, post, weight);
          sum_weight += weight;
        }
      }
      fclose(f_w);
    }
  } else {
    // still compute mean weight for population summary
    for (int n = 0; n < n_neurons; n++)
      for (int c = 0; c < n_conns; c++)
        sum_weight += d->weights[n * n_conns + c];
  }

  // ── Population Summary ────────────────────────────────────────────────────
  FILE *f_pop = fopen("../logs/population.csv", "a");
  if (f_pop) {
    fprintf(f_pop, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", t,
            (float)total_fired / n_neurons, sum_membrane / n_neurons,
            sum_weight / weight_count, sum_pre_trace / n_neurons,
            sum_post_trace / n_neurons, n_refractory);
    fclose(f_pop);
  }

  return 0;
}
