#ifndef _UTIL
#define _UTIL
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>

// Ran on CPU to make the reference. 
template<typename T>  void naiveTranspose(T* input, T* output, const int M, const int N) {
  #pragma omp parallel for
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < M; y++) {         
      output[x*M+y] = input[y*N+x];
    }
  } 
}

template <typename T> void randomMatrix(T* m, const int M, const int N, T high=1.0, T low=0.0) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> urd(low, high);
  for(int y = 0; y < M; y++) {
    std::generate(&m[y*N], &m[(y+1)*N], [&] () {
      return urd(gen); 
    });
  } 
}

template<typename T>  void printMatrix(T* m,  const int M, const int N) {
  for (int y = 0; y < M; y++) {         
    for (int x = 0; x < N; x++) {
      std::cout << m[y*N+x] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<typename T> bool matrixEqual(T* m1, T* m2, const int M, const int N) {
  for (int y = 0; y < M; y++) 
    for (int x = 0; x < N; x++) 
      if(m1[y*N+x] != m2[y*N+x]) 
        return false;
  return true;
}
#endif
