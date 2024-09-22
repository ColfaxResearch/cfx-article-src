#include "cutlass/util/command_line.h"

#include "hopper-gemm-ws/hopper_gemm_kernel_launch.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N, K, iterations;
  cmd.get_cmd_line_argument("M", M, 8192);
  cmd.get_cmd_line_argument("N", N, 8192);
  cmd.get_cmd_line_argument("K", K, 8192);
  cmd.get_cmd_line_argument("iterations", iterations, 10);

  std::cout << "Matrix size: " << M << " x " << N << " x " << K << std::endl;
  
  gemm_tn_launch(M, N, K, iterations);

  return 0;
}
