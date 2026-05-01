#ifndef NETWORK_H
#define NETWORK_H
#include "data.h"
#include "mnist.h"
#include "network.h"
#include <cuda_runtime.h>

cudaError_t compute(Network *d_net, Network *h_net, SpikeTrain *input,
                    int sample_id, bool learn);
cudaError_t compute_from_thread(data_mutex_t *obj);

cudaError_t test_images(Network *d_net, Network *h_net, mnist_dataset_t *imgs);
#endif
