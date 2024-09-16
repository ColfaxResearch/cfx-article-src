/***************************************************************************************************
 * Copyright (c) 2024 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

//
// Simple refactoring of the CUTLASS Hopper GEMM tutorial into a warp-specialized design
//

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

using namespace cute;

template <int numStages,
          class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorageMyHopperGEMM
{
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
  
  typename cutlass::PipelineTmaAsync<numStages>::SharedStorage pipeline_storage;
};

// Shared Storage for aligned addresses
// template <uint32_t numStages>
// struct SharedStoragePipeline
// {
//   typename cutlass::PipelineTmaAsync<numStages>::SharedStorage storage;
// };

template <int numThreads, int numStages, class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(numThreads, 1)
void
hopper_gemm_concise(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN


  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageMyHopperGEMM<numStages, TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles
  //
  // TUTORIAL:
  //   These are TMA partitionings, which have a dedicated custom partitioner.
  //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
  //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
  //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
  //     with the understanding that the TMA is responsible for everything in mode-0.
  //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
  //

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                       CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);

  //Prepare pipeline
  using MainloopPipeline = typename cutlass::PipelineTmaAsync<numStages>;  
  using PipelineState = typename cutlass::PipelineState<numStages>;  
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;
  constexpr int producerWarpGroupId = 0; // Using the 0th warpgroup to allow for variable consumer WGs

  int warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
  int warp_group_thread_idx = threadIdx.x % 128;

  typename MainloopPipeline::Params params;
  params.transaction_bytes = kTmaTransactionBytes;
  if (warp_group_idx == producerWarpGroupId) { 
    params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  else {
    params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  params.is_leader = warp_group_thread_idx == 0;
  params.num_consumers = numThreads - 128;
  auto cluster_shape = make_shape(Int<2>{},Int<1>{},Int<1>{});
  MainloopPipeline pipeline(smem.pipeline_storage, params, cluster_shape);

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();  
  cute::cluster_wait();

  // Total count of tiles
  int k_tile_count = size<1>(tAgA);

  using LowerRegisterCount = Int<24>;
  using HigherRegisterCount = Int<240>;
 
  if (warp_group_idx == producerWarpGroupId) { 
    cutlass::arch::warpgroup_reg_dealloc<LowerRegisterCount{}>();
    
    int lane_predicate = cute::elect_one_sync();
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();

        copy(tma_a.with(*tmaBar, 0), tAgA(_,k_tile), tAsA(_,stage));
        copy(tma_b.with(*tmaBar, 0), tBgB(_,k_tile), tBsB(_,stage));

        ++smem_pipe_write;
      }
    }
  } else { 
    cutlass::arch::warpgroup_reg_alloc<HigherRegisterCount{}>();
    PipelineState smem_pipe_read;

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x - 128);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
    
    // Allocate accumulators and clear them
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
    clear(tCrC);
    
    // Allocate "fragments"
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      pipeline.consumer_wait(smem_pipe_read);
      auto stage = smem_pipe_read.index();
  
      // MMAs to cover 1 K_TILE
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,stage), tCrB(_,_,_,stage), tCrC);     // (V,M) x (V,N) => (V,M,N)
      warpgroup_commit_batch();
  
      // Wait for all MMAs in a K_TILE to complete
      warpgroup_wait<0>();
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    //
    // Epilogue (unpredicated)
    //
    axpby(alpha, tCrC, beta, tCgC);
  }


}

// 
// Use this method as host entry point.
// Serves as reference for other GEMM kernel in exercise-4 folder
//

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
hopper_gemm_tn_concise(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<256>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  constexpr int numStages = 4;
  auto bP = Int<numStages>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  using TiledMMA = decltype(make_tiled_mma(SM90_64x128x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{}, Layout<Shape<_2,_1,_1>>{}));

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorageMyHopperGEMM<numStages, TA, TB, decltype(sA), decltype(sB)>));
  // std::cout << "smem size = " << smem_size << std::endl;  
  constexpr int num_producers = 128; // 1 warp group  
  constexpr int numThreads = size(TiledMMA{}) + num_producers;
  // std::cout << "num threads = " << numThreads << std::endl;  

  dim3 dimBlock(numThreads);
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &hopper_gemm_concise<numThreads, numStages, decltype(prob_shape), decltype(cta_tiler),
                                           TA, decltype(sA), decltype(tmaA),
                                           TB, decltype(sB), decltype(tmaB),
                                           TC, decltype(dC), TiledMMA,
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, TiledMMA{},
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

