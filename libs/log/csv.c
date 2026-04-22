#include "data.h"
#include <stdio.h>

int write_to_csv(data_mutex_t *obj) {
  if (obj->timestep == 1) {
    FILE *fp = fopen("filename.txt", "w"); // Clears the file
    fclose(fp);
  }
  FILE *csv = fopen("../logs/log.csv", "a");

  if (csv == NULL) {
    printf("Error opening file");
    return 1;
  }

  // headers
  if (obj->timestep == 1) {
    fprintf(csv, "t,neuron_idx,fired\n");
  }

  for (int i = 0; i < obj->config->n_neurons; i++) {
    uint8_t spike = obj->front->post_spikes[i];
    int t = obj->timestep;
    fprintf(csv, "%d,%d,%d\n", t, i, spike);
  }

  fclose(csv);

  return 0;
}
