#include "cutlass/util/command_line.h"

#include "include/copy.h"
#include "include/copy_direct.h"
#include "include/transpose_naive.h"
#include "include/transpose_smem.h"
#include "include/transpose_tmastore_vectorized.h"
#include "include/util.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  using Element = float;

  int M, N;
  cmd.get_cmd_line_argument("M", M, 8192);
  cmd.get_cmd_line_argument("N", N, 8192);

  std::cout << "Matrix size: " << M << " x " << N << std::endl;

  printf("Baseline copy with 1x4 vectorization.\n");
  benchmark<Element, false>(copy_baseline<Element>, M, N);
  
  printf("Direct GMEM to GMEM copy (no rmem fragment Tensor) with 1x4 vectorization\n");
  benchmark<Element, false>(copy_direct<Element>, M, N);
  
  printf("\nNaive transpose (no tma, no smem, not vectorized):\n");
  benchmark<Element>(transpose_naive<Element>, M, N);

  printf("\nSMEM transpose (no tma, smem passthrough, not vectorized, not swizzled):\n");
  benchmark<Element>(transpose_smem<Element, false>, M, N);

  printf("\nSwizzle (no tma, smem passthrough, not vectorized, swizzled):\n");
  benchmark<Element>(transpose_smem<Element, true>, M, N);

  printf("\nTMA (tma, smem passthrough, vectorized, swizzled):\n");
  benchmark<Element>(transpose_tma<Element>, M, N);

  return 0;
}
