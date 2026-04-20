typedef struct {
  int *conns;
  float *weights;
  int out_n; // The number of neurons to connect to (network)
  int in_n;  // The number of input neurons
  int T;
} ConnectedLayer;

ConnectedLayer *input_layer(int n_conns, int out_n, int in_n, int T);
