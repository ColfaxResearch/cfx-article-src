#!/bin/bash
# Benchmark v1-v6, v7 and clc at 32768^3. Configs live in bench_common.sh.
#
# Usage: ./bench_32k.sh
#        DRY=1 ./bench_32k.sh                  print the commands, run nothing
#        VERSIONS="v6 v7 clc" ./bench_32k.sh   benchmark a subset

BENCH_SIZE=32768
source "$(dirname "$0")/bench_common.sh"
