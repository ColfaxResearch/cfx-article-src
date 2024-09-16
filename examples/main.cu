// NOTE: Make sure to use CUTLASS 3.5 to run these examples.
// You can run "git submodule update --init --recursive" after switching to
// this branch, or point the Makefile to your local CUTLASS 3.5 install.

#include "cutlass/util/command_line.h"

// #include "exercise-1-copy/copy_kernel.h"
// #include "exercise-1-copy/copy_kernel_complete.h"
// #include "exercise-1-copy/copy_kernel_smem.h"
// #include "exercise-2-transpose/transpose_kernel_smem.h"
// #include "exercise-3-gemm/hopper_gemm_ws_concise_launch.h"
#include "exercise-4-hopper-gemm-ws-composable/hopper_gemm_kernel_launch.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N, K, iterations;
  cmd.get_cmd_line_argument("M", M, 5120);
  cmd.get_cmd_line_argument("N", N, 5120);
  cmd.get_cmd_line_argument("K", K, 4096);
  cmd.get_cmd_line_argument("iterations", iterations, 10);

  std::cout << "Matrix size: " << M << " x " << N << " x " << K << std::endl;
  
//   copy_host(M, N, iterations);
//   transpose_host(M, N, iterations);
//   swizzle_test();
//   gemm_tn_concise_launch(M, N, K, iterations);
  gemm_tn_launch(M, N, K, iterations);

  return 0;
}
