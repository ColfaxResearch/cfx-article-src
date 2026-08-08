#!/bin/bash
# Shared implementation behind bench_2k.sh / bench_4k.sh / bench_8k.sh /
# bench_16k.sh / bench_32k.sh, which set BENCH_SIZE and source this file.
#
#   DRY=1               print the commands, run nothing
#   VERSIONS="v1 v6"    benchmark a subset, in this order

set -u

[ -n "${BENCH_SIZE:-}" ] || { echo "bench_common.sh: BENCH_SIZE is not set; run bench_<size>.sh instead" >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PY:-python}
DRY=${DRY:-0}

M=$BENCH_SIZE

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

gpu0_uuid=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [ -n "$gpu0_uuid" ] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -q "$gpu0_uuid"; then
  echo "WARNING: another process is already using GPU ${CUDA_VISIBLE_DEVICES}." >&2
  echo "         Timings will be contended and are not comparable." >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null \
    | grep -e gpu_uuid -e "$gpu0_uuid" >&2
  echo >&2
fi

RUN="--warmup_iterations 3 --iterations 20 --use_cold_l2"

VERSIONS=${VERSIONS:-"v1 v2 v3 v4 v5 v6 v7 clc"}

declare -A KERNEL_FOR=(
  [v1]=nvfp4_gemm_v1.py  [v2]=nvfp4_gemm_v2.py  [v3]=nvfp4_gemm_v3.py
  [v4]=nvfp4_gemm_v4.py  [v5]=nvfp4_gemm_v5.py  [v6]=nvfp4_gemm_v6.py
  [v7]=nvfp4_gemm_v6.py  [clc]=nvfp4_gemm_clc.py
)

# v1 predates the epilogue/pipeline tuning flags and only takes the tile shape.
FLAGS_v1="--tile_shape_mnk 128,128,128"

FLAGS_128="--tile_shape_mnk 128,128,128 --swizzle_size 16 --epi_tile_m 64 \
--epi_tile_n 64 --epi_stage 3 --ab_stage 4"

FLAGS_192="--tile_shape_mnk 192,128,128 --swizzle_size 16 --epi_tile_m 96 \
--epi_tile_n 64 --epi_stage 2 --ab_stage 3"

declare -A FLAGS_v7=(
  [2048]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 64 --raster_along_m --epi_tile_m 96 --epi_tile_n 64 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 32"
  [4096]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 8 --raster_along_m --epi_tile_m 96 --epi_tile_n 32 --epi_stage 3 --sf_tma_internal_type int16 --load_register_requirement 40"
  [8192]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 8 --raster_along_m --epi_tile_m 96 --epi_tile_n 64 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 32"
  [16384]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 64 --raster_along_m --epi_tile_m 96 --epi_tile_n 64 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 40"
  [32768]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 32 --raster_along_m --epi_tile_m 96 --epi_tile_n 32 --epi_stage 1 --sf_tma_internal_type int16 --load_register_requirement 24"
)

declare -A FLAGS_clc=(
  [2048]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 8 --raster_along_m --epi_tile_m 96 --epi_tile_n 64 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 24"
  [4096]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 8 --raster_along_m --epi_tile_m 96 --epi_tile_n 64 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 24"
  [8192]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 8 --raster_along_m --epi_tile_m 96 --epi_tile_n 64 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 24"
  [16384]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 64 --raster_along_m --epi_tile_m 96 --epi_tile_n 128 --epi_stage 1 --sf_tma_internal_type int16 --load_register_requirement 24"
  [32768]="--tile_shape_mnk 192,128,128 --ab_stage 3 --swizzle_size 32 --raster_along_m --epi_tile_m 96 --epi_tile_n 32 --epi_stage 2 --sf_tma_internal_type int16 --load_register_requirement 24"
)

flags_for() {
  case $1 in
    v1)  echo "$FLAGS_v1" ;;
    v6)  echo "$FLAGS_192" ;;
    v7)  echo "${FLAGS_v7[$M]:-}" ;;
    clc) echo "${FLAGS_clc[$M]:-}" ;;
    *)   echo "$FLAGS_128" ;;
  esac
}

RESULTS=$(mktemp)
LOG=$(mktemp)
trap 'rm -f "$RESULTS" "$LOG"' EXIT

for v in $VERSIONS; do
  kernel=${KERNEL_FOR[$v]:-}
  flags=$(flags_for "$v")

  if [ -z "$kernel" ] || [ -z "$flags" ]; then
    printf '  %-4s no config for size %s -- skipped\n' "$v" "$M" >&2
    printf '%s %s\n' "$v" "-" >>"$RESULTS"
    continue
  fi
  if [ ! -f "$kernel" ]; then
    printf '  %-4s %s not found -- skipped\n' "$v" "$kernel" >&2
    printf '%s %s\n' "$v" "-" >>"$RESULTS"
    continue
  fi

  cmd=($PY "$kernel" --mnkl "$M,$M,$M,1" $flags $RUN)

  if [ "$DRY" != "0" ]; then
    printf '  %-4s %s\n' "$v" "${cmd[*]}" >&2
    printf '%s %s\n' "$v" "-" >>"$RESULTS"
    continue
  fi

  printf '  running %-4s ... ' "$v" >&2
  if "${cmd[@]}" >"$LOG" 2>&1; then
    tflops=$(grep -oP 'Throughput\s+:\s+\K[0-9.]+' "$LOG" | tail -1)
    if grep -q '^PASS' "$LOG" && [ -n "$tflops" ]; then
      printf '%s TFLOP/s\n' "$tflops" >&2
    else
      tflops=FAIL
      printf 'FAIL (ref check)\n' >&2
    fi
  else
    tflops=ERR
    printf 'ERR\n' >&2
    tail -3 "$LOG" >&2
  fi
  printf '%s %s\n' "$v" "$tflops" >>"$RESULTS"
done

echo
if [ "$DRY" != "0" ]; then
  echo "DRY=1: nothing was run."
  exit 0
fi