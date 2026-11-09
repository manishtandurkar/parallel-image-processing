#!/usr/bin/env bash
# CUDA benchmark sweep, run ON the Colab VM (see colab/run_gpu.py). Writes
# bench/results/<machine>_cuda_<date>.csv and a .cmd file with the exact command lines.
#
# Usage: bench/run_gpu.sh <machine-tag> [section ...]
#   sections: compare blocks   (default: all)
# Env overrides: BUILD=build  REPS=10  SIZES="512x512 ... 16384x16384"
#   BLOCK_SIZES="4096x4096 7680x4320"  BLOCKS="8x8 16x16 32x8 32x32"
set -euo pipefail

MACHINE=${1:?usage: bench/run_gpu.sh <machine-tag> [compare|blocks ...]}
shift
SECTIONS=${*:-compare blocks}

cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
REPS=${REPS:-10}
SIZES=${SIZES:-"512x512 1024x1024 2048x2048 4096x4096 7680x4320 16384x16384"}
BLOCK_SIZES=${BLOCK_SIZES:-"4096x4096 7680x4320"}
BLOCKS=${BLOCKS:-"8x8 16x16 32x8 32x32"}
OPS="gray blur2d blur_sep sobel pipeline fused"

mkdir -p bench/results
OUT=bench/results/${MACHINE}_cuda_$(date +%Y%m%d-%H%M%S).csv
CMDS=${OUT%.csv}.cmd
GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo unknown)
echo "# $0 $MACHINE $SECTIONS  (REPS=$REPS)  GPU: $GPU" > "$CMDS"

run() {
  local cmd=("$BUILD/pagp_cuda" "$@" --reps "$REPS" --validate --csv "$OUT" --machine "$MACHINE")
  echo "${cmd[*]}" >> "$CMDS"
  "${cmd[@]}"
}

variants() {
  case $1 in
  gray) echo naive ;;
  sobel) echo naive tiled ;;
  fused) echo tiled ;;
  *) echo naive const tiled ;;
  esac
}

for section in $SECTIONS; do
  case $section in
  compare) # every op and variant, every size, default 16x16 blocks
    for size in $SIZES; do
      for op in $OPS; do
        for v in $(variants "$op"); do
          run --gen "$size" --op "$op" --variant "$v" --block 16x16
        done
      done
    done
    ;;
  blocks) # block-shape sweep on the stencil ops
    for size in $BLOCK_SIZES; do
      for op in blur2d sobel pipeline fused; do
        for v in $(variants "$op"); do
          for b in $BLOCKS; do
            run --gen "$size" --op "$op" --variant "$v" --block "$b"
          done
        done
      done
    done
    ;;
  *)
    echo "unknown section: $section" >&2
    exit 2
    ;;
  esac
done

echo "wrote $OUT"
