import torch
import math
import transpose_cute as tc
import copy_cute as cc

from torch.utils.benchmark import Timer
from torch.utils.benchmark import Measurement

M = N = 4096
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, N)).to(device=cuda)

timer = Timer(
    stmt="cc.copy(A)",
    globals={"cc": cc, "A": A},
    num_threads=1,
)

m: Measurement = timer.blocked_autorange(min_run_time=1)
print("Baseline copy:")
print(m)
print("Performance: {:.2f} GB/s".format(2*M*N*A.element_size()/m.mean*pow(10,-9)))
A_copy = cc.copy(A)
print("Validation: {}".format("success" if torch.all(torch.eq(A_copy,A)) else "failed"))
print()

AT_torch = torch.transpose(A, 0, 1)

for ver in [tc.version.naive,tc.version.smem,tc.version.swizzle,tc.version.tma]:
  timer = Timer(
      stmt="tc.transpose(A, version=ver)",
      globals={"tc": tc, "A": A, "ver": ver},
      num_threads=1,
  )

  m: Measurement = timer.blocked_autorange(min_run_time=1)
  print("{}".format(tc.get_version_info(ver)))
  print(m)
  AT = tc.transpose(A, version=ver)
  print("Performance: {:.2f} GB/s".format(2*M*N*A.element_size()/m.mean*pow(10,-9)))
  print("Validation: {}".format("success" if torch.all(torch.eq(AT_torch,AT)) else "failed"))
  print()
