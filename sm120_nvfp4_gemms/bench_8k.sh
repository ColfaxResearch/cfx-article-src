#!/bin/bash
# Benchmark v1-v6, v7 and clc at 8192^3. Configs live in bench_common.sh.
#
# Usage: ./bench_8k.sh
#        DRY=1 ./bench_8k.sh                  print the commands, run nothing
#        VERSIONS="v6 v7 clc" ./bench_8k.sh   benchmark a subset

BENCH_SIZE=8192
source "$(dirname "$0")/bench_common.sh"
