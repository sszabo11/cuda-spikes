
#include "network.h"
#include <stdint.h>

void init_logs();
int write_to_csv(Network *net, int t, int log_weights_every);

void init_digit_log(int n_neurons);
void write_digit_response(Network *h_net, int img_idx, uint8_t label,
                          uint32_t *spike_counts);
