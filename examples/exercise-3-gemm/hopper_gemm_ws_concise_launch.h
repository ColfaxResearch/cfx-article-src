#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include "hopper_gemm_ws_concise.h"

void gemm_tn_concise_launch(int m, int n, int k, int iterations) {

  using TA = cutlass::half_t;
  using TB = cutlass::half_t;
  using TC = cutlass::half_t;
  using TI = cutlass::half_t;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  // Initialize the tensors
  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
//   for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
//   for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  int ldA = k, ldB = k, ldC = m;

  // One warmup iteration
  hopper_gemm_tn_concise(m, n, k, alpha,
          d_A.data().get(), ldA,
          d_B.data().get(), ldB,
          beta,
          d_C.data().get(), ldC);

  CUTE_CHECK_LAST();

  cudaDeviceSynchronize();

  thrust::host_vector<TC> cute_result = d_C;

  double gflops = (2.0*m*n*k) * 1e-9;
  // Timing iterations
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    hopper_gemm_tn_concise(m, n, k, alpha,
          d_A.data().get(), ldA,
          d_B.data().get(), ldB,
          beta,
          d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

}