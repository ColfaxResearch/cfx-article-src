#!/bin/bash
# Benchmark v1-v6, v7 and clc at 2048^3. Configs live in bench_common.sh.
#
# Usage: ./bench_2k.sh
#        DRY=1 ./bench_2k.sh                  print the commands, run nothing
#        VERSIONS="v6 v7 clc" ./bench_2k.sh   benchmark a subset

BENCH_SIZE=2048
source "$(dirname "$0")/bench_common.sh"
