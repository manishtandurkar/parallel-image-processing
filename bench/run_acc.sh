#!/usr/bin/env bash
# OpenACC GPU benchmark sweep, run ON the Colab VM (see colab/run_gpu.py). Writes
# bench/results/<machine>_acc_<date>.csv and a .cmd file with the exact command lines.
# Only the GPU build is swept: pagp_acc_mc runs on the Colab CPU, which is not used for CPU
# benchmarks (2 weak vCPUs); it is only tested there.
#
# Usage: bench/run_acc.sh <machine-tag>
# The plain kernels variant runs serial GPU kernels for some ops (the compiler cannot prove the
# loops independent; 4 s per rep for fused at 512x512), so it is measured at KERNELS_SIZES only.
#
# Env overrides: BUILD=build-acc  REPS=10  SIZES="512x512 ... 16384x16384"  KERNELS_SIZES=512x512
set -euo pipefail

MACHINE=${1:?usage: bench/run_acc.sh <machine-tag>}

cd "$(dirname "$0")/.."
BUILD=${BUILD:-build-acc}
REPS=${REPS:-10}
SIZES=${SIZES:-"512x512 1024x1024 2048x2048 4096x4096 7680x4320 16384x16384"}
KERNELS_SIZES=${KERNELS_SIZES:-"512x512"}
OPS="gray blur2d blur_sep sobel pipeline fused"

mkdir -p bench/results
OUT=bench/results/${MACHINE}_acc_$(date +%Y%m%d-%H%M%S).csv
CMDS=${OUT%.csv}.cmd
GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo unknown)
NVC=$(nvc++ --version 2>/dev/null | grep -m1 nvc++ || echo unknown)
echo "# $0 $MACHINE  (REPS=$REPS)  GPU: $GPU  compiler: $NVC" > "$CMDS"

run() {
  local cmd=("$BUILD/pagp_acc" "$@" --reps "$REPS" --validate --csv "$OUT" --machine "$MACHINE")
  echo "${cmd[*]}" >> "$CMDS"
  "${cmd[@]}"
}

for size in $SIZES; do
  for op in $OPS; do
    for v in parallel kernels_indep copy; do
      run --gen "$size" --op "$op" --variant "$v"
    done
    if [[ " $KERNELS_SIZES " == *" $size "* ]]; then
      run --gen "$size" --op "$op" --variant kernels
    fi
  done
done

echo "wrote $OUT"
