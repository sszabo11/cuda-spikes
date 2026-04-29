// #ifndef NETWORK_H
// #define NETWORK_H
// #include "data.h"
// #include <cuda_runtime.h>
//
// typedef struct Network {
//   NetworkData *data;
//   Config *config;
// } Network;
//
// Network *network_create();
// cudaError_t compute(Network *d_data, data_mutex_t *obj);
// cudaError_t compute_from_thread(data_mutex_t *obj);
//
// Network *send_network_to_gpu(Network *net);
//
// cudaError_t copy_input_to_gpu();
// cudaError_t copy_results_back_to_cpu();
// #endif
