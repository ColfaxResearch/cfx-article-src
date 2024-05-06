import torch
import math
import transpose_cute as tc
import copy_cute as cc

from torch.utils.benchmark import Timer
from torch.utils.benchmark import Measurement

M = N = 32768
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, N)).to(device=cuda)
AT_reference = torch.transpose(A, 0, 1)

def benchmark(stmt, glob, desc): 
  timer = Timer(
      stmt=stmt,
      globals=glob,
      num_threads=1,
  )
  
  m: Measurement = timer.blocked_autorange(min_run_time=3)
  print(desc)
  print("Mean: {{:.{0}g}} ms ({{:.{0}g}} GB/s)".format(m.significant_figures).format(m.mean*pow(10,3),2*M*N*A.element_size()/m.mean*pow(10,-9)))
  print("IQR: {{:.{}g}} us".format(m.significant_figures).format(m.iqr*pow(10,6)))

def validate(res, reference):
  print("Validation: {}".format("success" if torch.all(torch.eq(reference,res)) else "failed"))


benchmark("cc.copy(A)",{"cc": cc, "A": A},"Baseline copy:")
validate(cc.copy(A), A)
print()

benchmark("torch.transpose(A, 0, 1).contiguous()",{"A": A},"Torch transpose:")
print()

for ver in [tc.version.naive,tc.version.smem,tc.version.swizzle,tc.version.tma]:
  benchmark("tc.transpose(A, version=ver)",{"tc": tc, "A": A, "ver": ver},tc.get_version_info(ver))
  validate(tc.transpose(A, version=ver), AT_reference)
  print()
