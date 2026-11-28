#!/usr/bin/env bash
# Nsight Compute metrics for the CUDA stencil kernels, run ON the Colab VM (colab/run_gpu.py, step
# ncu). Explains naive vs const vs tiled: DRAM traffic, L1/L2 hit rates, instructions, occupancy,
# shared-memory bank conflicts and barrier stalls. One raw CSV per op/variant in report/cuda/ncu/;
# summarized locally by scripts/ncu_table.py.
#
# Usage: bench/run_ncu.sh   Env overrides: BUILD=build  SIZE=4096x4096
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
SIZE=${SIZE:-4096x4096}
OUT=report/cuda/ncu
mkdir -p "$OUT"

METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum
METRICS+=,dram__throughput.avg.pct_of_peak_sustained_elapsed
METRICS+=,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct
METRICS+=,sm__warps_active.avg.pct_of_peak_sustained_active
METRICS+=,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum
METRICS+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
METRICS+=,smsp__average_warp_latency_issue_stalled_barrier
METRICS+=,launch__occupancy_limit_shared_mem,launch__registers_per_thread

ncu --version | tail -1
nvidia-smi --query-gpu=name --format=csv,noheader
for spec in "blur2d naive" "blur2d const" "blur2d tiled" "blur_sep naive" "blur_sep const" \
            "blur_sep tiled" "sobel naive" "sobel tiled"; do
  set -- $spec
  # One rep after the 2 warm-ups; profile the last run's kernels (blur_sep launches 2 per run).
  n=1; [ "$1" = blur_sep ] && n=2
  ncu --csv --page raw --metrics "$METRICS" --launch-skip $((2 * n)) --launch-count "$n" \
    "$BUILD/pagp_cuda" --gen "$SIZE" --op "$1" --variant "$2" --reps 1 > "$OUT/$1_$2.log"
  sed -n '/^"ID"/,$p' "$OUT/$1_$2.log" > "$OUT/$1_$2.csv"
  rm "$OUT/$1_$2.log"
  echo "$OUT/$1_$2.csv: $(($(wc -l < "$OUT/$1_$2.csv") - 2)) kernel(s)"
done
