#!/bin/bash
# Benchmark v1-v6, v7 and clc at 4096^3. Configs live in bench_common.sh.
#
# Usage: ./bench_4k.sh
#        DRY=1 ./bench_4k.sh                  print the commands, run nothing
#        VERSIONS="v6 v7 clc" ./bench_4k.sh   benchmark a subset

BENCH_SIZE=4096
source "$(dirname "$0")/bench_common.sh"
