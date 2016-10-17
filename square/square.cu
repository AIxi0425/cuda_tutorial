#include <stdio.h>

__global__ void square(float* d_out, float* d_in)
{
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f;
}

int main(int argc, char** grgv)
{
  const int ARRAY_SIZE = 64;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
  
  // generate the input array on the host
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++)
    h_in[i] = float(i);
  
  // declare GPU memory pointers
  float* d_in;
  float* d_out;
  
  // allocate GPU memory
  cudaMalloc((void**) &d_in, ARRAY_SIZE);
  cudaMalloc((void**) &d_out, ARRAY_SIZE);
  
  
}
