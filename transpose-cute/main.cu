#include "cutlass/util/command_line.h"

#include "include/copy.h"
// #include "include/copy_fma.h"
// #include "include/fma_inner_product.h"
#include "include/fma_inner_product_naive.h"
#include "include/copy_smem.h"
#include "include/copy_direct.h"
#include "include/transpose_naive.h"
#include "include/transpose_smem.h"
#include "include/transpose_tmastore_vectorized.h"
#include "include/util.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  using Element = float;

  int M, N, iterations;
  cmd.get_cmd_line_argument("M", M, 8192*8);
  cmd.get_cmd_line_argument("N", N, 1024);
  cmd.get_cmd_line_argument("iterations", iterations, 1);

  std::cout << "Matrix size: " << M << " x " << N << std::endl;

  printf("Baseline copy with 1x4 vectorization.\n");
  benchmark<Element, false>(copy_baseline<Element>, M, N, iterations);

  // printf("Copy + FMA with 1x4 vectorization. No validation\n");
  // benchmark<Element, false, true>(copy_fma<Element>, M, N, iterations, false);

  // printf("Copy through SMEM with 1x4 vectorization.\n");
  // benchmark<Element, false>(copy_smem<Element>, M, N, iterations);

  // printf("Copy through SMEM with 1x4 vectorization and do FMA inner product.\n");
  // benchmark<Element, false, true>(copy_smem_fma<Element>, M, N, iterations, false);

  printf("Copy through SMEM with 1x4 vectorization and do FMA inner product. No ILP. \n");
  benchmark<Element, false, true>(copy_smem_fma_naive<Element>, M, N, iterations, false);
  
  // printf("Direct GMEM to GMEM copy (no rmem fragment Tensor) with 1x4 vectorization\n");
  // benchmark<Element, false>(copy_direct<Element>, M, N, iterations);
  
  // printf("\nNaive transpose (no tma, no smem, not vectorized):\n");
  // benchmark<Element>(transpose_naive<Element>, M, N);

  // printf("\nSMEM transpose (no tma, smem passthrough, not vectorized, not swizzled):\n");
  // benchmark<Element>(transpose_smem<Element, false>, M, N);

  // printf("\nSwizzle (no tma, smem passthrough, not vectorized, swizzled):\n");
  // benchmark<Element>(transpose_smem<Element, true>, M, N);

  // printf("\nTMA (tma, smem passthrough, vectorized, swizzled):\n");
  // benchmark<Element>(transpose_tma<Element>, M, N);

  return 0;
}
