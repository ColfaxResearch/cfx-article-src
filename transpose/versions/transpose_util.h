#ifndef _TR_UTIL
#define _TR_UTIL
inline void checkCUDAError() {
  cudaError_t code = cudaPeekAtLastError();
  if (code != cudaSuccess) {
    fprintf(stderr,"GPU Error: %s\n", cudaGetErrorString(code));
    exit(code);
  }
}
#endif
