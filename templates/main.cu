#include "cutlass/util/command_line.h"

#include "copy_kernel.h

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N, iterations;
  cmd.get_cmd_line_argument("M", M, 4096);
  cmd.get_cmd_line_argument("N", N, 4096);
  cmd.get_cmd_line_argument("iterations", iterations, 1);

  std::cout << "Matrix size: " << M << " x " << N << std::endl;

  copy_host(M, N, iterations);

  return 0;
}
