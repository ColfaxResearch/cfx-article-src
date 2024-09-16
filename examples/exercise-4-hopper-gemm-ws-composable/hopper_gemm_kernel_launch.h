/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

//
// Adapted from FA3 kernel
//

#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include "kernel_traits.h"
#include "tile_scheduler.hpp"
#include "mainloop_sm90_tma_gmma_ws.hpp"
#include "epilogue_sm90_tma_ws.hpp"
#include "hopper_gemm_kernel.h"
#include "../exercise-3-gemm/hopper_gemm_ws_concise.h"

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void gemm_tn(int M, int N, int K,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC      * C, int ldC,
    cudaStream_t stream = 0) {

    using Kernel_traits = Kernel_traits<128, 256, 64, 12, 4, 1, TA, TC>;

    auto smem_layout_A = typename Kernel_traits::SmemLayoutA{};
    auto smem_layout_B = typename Kernel_traits::SmemLayoutB{};
    auto smem_layout_C = typename Kernel_traits::SmemLayoutC{};
    // print("Smem Layout A: "); print(smem_layout_A); print("\n");
    // print("Smem Layout B: "); print(smem_layout_B); print("\n");
    // print("Smem Layout C: "); print(smem_layout_C); print("\n");

    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    using CollectiveMainloop = cfx::CollectiveMainloop<Kernel_traits>;
    using CollectiveEpilogue = cfx::CollectiveEpilogue<Kernel_traits>;
    // using Scheduler = cfx::StaticPersistentTileScheduler;
    using Scheduler = cfx::SingleTileScheduler;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            A,
            make_layout(make_shape(M, K), make_stride(ldA, Int<1>{})),  // layout_A
            B,
            make_layout(make_shape(N, K), make_stride(ldB, Int<1>{})),  // layout_B
        });
    
    // auto layout_A = mainloop_params.layout_A;
    // print(layout_A);
    // auto tma_A = mainloop_params.tma_load_A;
    // print(tma_A);
    // auto layout_B = mainloop_params.layout_B;
    // print(layout_B);
    // auto tma_B = mainloop_params.tma_load_B;
    // print(tma_B);

    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({
            C,
            make_layout(make_shape(M, N), make_stride(ldC, Int<1>{}))
        });

    // auto layout_C = epilogue_params.layout_C;
    // print(layout_C);
    // auto tma_store = epilogue_params.tma_store;
    // print(tma_store);

    int num_blocks_m = cutlass::ceil_div(M, Kernel_traits::kBlockM);
    // round if using clusters
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    int num_blocks_n = cutlass::ceil_div(N, Kernel_traits::kBlockN);

    typename Scheduler::Arguments scheduler_args = {num_blocks_m, num_blocks_n, 1};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);
    
    // Get the ptr to kernel function.
    void *kernel;
    kernel = (void *)cfx::hopper_gemm_ws<Kernel_traits, Scheduler>;    
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_A = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_A));
    // int smem_size_B = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_B));
    // int smem_size_C = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_C));
    // printf("smem_size = %d, A = %d, B = %d, C = %d.\n", smem_size, smem_size_A, smem_size_B, smem_size_C);
    if (smem_size >= 48 * 1024) {
       cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    // std::cout << grid_dims.x << " " << grid_dims.y << " " << grid_dims.z << std::endl;
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};

#if 1
    cutlass::launch_kernel_on_cluster(
        launch_params, kernel,
        mainloop_params, epilogue_params, scheduler_params);
#endif
}

void gemm_tn_launch(int m, int n, int k, int iterations) {

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

  int ldA = k, ldB = k, ldC = n;

  gemm_tn(m, n, k, alpha,
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
    gemm_tn(m, n, k, alpha,
          d_A.data().get(), ldA,
          d_B.data().get(), ldB,
          beta,
          d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

/*
*  WARNING: This GEMM writes out output in row-major format,
*  but "concise" Hopper GEMM writes out output in column-major format
*  (just like CUTLASS Hopper GEMM tutorial example).
*  Thus for validation, we transpose the reference.
*/

#if 1

  d_C = h_C;

  hopper_gemm_tn_concise(m, n, k, alpha,
          d_A.data().get(), ldA,
          d_B.data().get(), ldB,
          beta,
          d_C.data().get(), /*ldC=*/m);

  CUTE_CHECK_LAST();

  cudaDeviceSynchronize();

  thrust::host_vector<TC> cute_result_ref = d_C;

  auto transposeFunction = make_layout(make_shape(m,n), LayoutRight{});

  float max_diff = 0; int index = 0;
  for (int j = 0; j < m*n; ++j) {
    float diff = std::fabs(float(cute_result[j] - cute_result_ref[transposeFunction(j)]));
    if (diff > max_diff) { max_diff = diff; index = j; }
    // if(j < 100)
    //     std::cout << cute_result[j] << " " << cute_result_ref[j] << " " << cute_result_ref[transposeFunction(j)] << std::endl;
    // std::cout << "diff at  " << j << " equals " << diff << std::endl;
  }

  std::cout << "max diff = " << max_diff << " at index " << index << std::endl;
  std::cout << cute_result[index] << " " << cute_result_ref[transposeFunction(index)] << std::endl;

#endif

}