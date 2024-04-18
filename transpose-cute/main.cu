#include "cutlass/util/command_line.h"

#include "transpose_naive.h"
#include "transpose_smem.h"
#include "transpose_smem_bank_conflict.h"
#include "transpose_tmastore_vectorized.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N;
  cmd.get_cmd_line_argument("M", M, 2048);
  cmd.get_cmd_line_argument("N", N, 2048);

  std::cout << "(M, N): " << M << ", " << N << std::endl;

  transpose_host_kernel_naive(M, N);
  transpose_host_kernel_smem<false>(M, N); // not swizzled
  transpose_host_kernel_smem<true>(M, N);  // swizzled
  transpose_host_kernel_tma(M, N);

  return 0;
}