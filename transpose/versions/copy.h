#ifndef _TRANSPOSE_
#define _TRANSPOSE_

#include "transpose_util.h"

#define M_TILE 32
#define N_TILE 32

template<typename T> __global__ void copy_kernel(T* input, T* output, const long M, const long N) {
  const long x_0 = blockIdx.x * N_TILE;
  const long y_0 = blockIdx.y * M_TILE;

  const long x = x_0 + threadIdx.x;

  for(long y = y_0; y < y_0+M_TILE; y++) 
    output[y*N+x] = input[y*N+x];
}

// show performance difference when read is strided instead of write


template<typename T> void copy(T* d_input, T* d_output, const long M, const long N) {
  dim3 blockDim(M_TILE, 1);
  dim3 gridDim(M/M_TILE, N/N_TILE);
  copy_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, M, N);
  cudaDeviceSynchronize();
  checkCUDAError();
}
#endif
