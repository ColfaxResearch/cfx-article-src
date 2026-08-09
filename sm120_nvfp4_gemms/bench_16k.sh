#!/bin/bash
# Benchmark v1-v6, v7 and clc at 16384^3. Configs live in bench_common.sh.
#
# Usage: ./bench_16k.sh
#        DRY=1 ./bench_16k.sh                  print the commands, run nothing
#        VERSIONS="v6 v7 clc" ./bench_16k.sh   benchmark a subset

BENCH_SIZE=16384
source "$(dirname "$0")/bench_common.sh"
