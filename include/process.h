#include "config.h"
#include "data.h"
#include <cuda_runtime.h>

cudaError_t process(Config *config, NetworkData *data);
cudaError_t process_from_thread(data_mutex_t *obj);
