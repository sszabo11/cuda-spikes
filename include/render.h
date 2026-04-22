
#include "config.h"
#include "data.h"
#include "encode.h"

int render(Config *config, NetworkData *data, SpikeTrain *input_train);

int render_from_thread(data_mutex_t *obj);
