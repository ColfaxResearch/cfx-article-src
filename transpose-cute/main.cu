#include "cutlass/util/command_line.h"

#include "copy.h"
#include "transpose_naive.h"
#include "transpose_smem.h"
//#include "transpose_tmastore_vectorized.h"
#include "util.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  using Element = float;

  int M, N;
  cmd.get_cmd_line_argument("M", M, 4096);
  cmd.get_cmd_line_argument("N", N, 4096);

  std::cout << "(M, N): " << M << ", " << N << std::endl;

  printf("Baseline copy; No transpose\n");
  benchmark<Element, false>(copy_baseline<Element>, M, N);
  
  printf("\nNO tma, NO smem, not vectorized\n");
  benchmark<Element>(transpose_naive<Element>, M, N);

  printf("\nNO tma, smem passthrough, not vectorized, not swizzled\n");
  benchmark<Element>(transpose_smem<Element, false>, M, N);

  printf("\nNO tma, smem passthrough, not vectorized, swizzled\n");
  benchmark<Element>(transpose_smem<Element, true>, M, N);

  return 0;
}
