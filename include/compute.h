#ifndef NETWORK_H
#define NETWORK_H
#include "data.h"
#include "network.h"
#include <cuda_runtime.h>

cudaError_t compute(Network *d_net, Network *h_net, SpikeTrain *input,
                    int sample_id);
cudaError_t compute_from_thread(data_mutex_t *obj);

#endif
