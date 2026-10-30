#!/usr/bin/env bash
# CPU benchmark sweep for one local machine. Writes bench/results/<machine>_<date>.csv and a
# .cmd file next to it with the exact command lines that produced it.
#
# Usage: bench/run_cpu.sh <machine-tag> [section ...]
#   sections: compare scaling sync weak   (default: all)
# Env overrides:
#   BUILD=build  REPS=10  SIZES="512x512 ... 16384x16384"  SCALING_SIZES="2048x2048 7680x4320"
#   THREADS="1 2 4 ..."  (default: 1 2 4 6 8 ... up to the core count)
set -euo pipefail

MACHINE=${1:?usage: bench/run_cpu.sh <machine-tag> [compare|scaling|sync|weak ...]}
shift
SECTIONS=${*:-compare scaling sync weak}

cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
REPS=${REPS:-10}
SIZES=${SIZES:-"512x512 1024x1024 2048x2048 4096x4096 7680x4320 16384x16384"}
SCALING_SIZES=${SCALING_SIZES:-"2048x2048 7680x4320"}
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
if [[ -z ${THREADS:-} ]]; then
  THREADS="1 2"
  for ((t = 4; t <= NCPU; t += 2)); do THREADS+=" $t"; done
  [[ " $THREADS " == *" $NCPU "* ]] || THREADS+=" $NCPU"
fi
OPS="gray blur2d blur_sep sobel pipeline fused"

if command -v pmset >/dev/null && pmset -g batt | grep -q "Battery Power"; then
  echo "WARNING: running on battery; plug in power for benchmark numbers (CLAUDE.md protocol)" >&2
fi

mkdir -p bench/results
OUT=bench/results/${MACHINE}_$(date +%Y%m%d-%H%M%S).csv
CMDS=${OUT%.csv}.cmd
echo "# $0 $MACHINE $SECTIONS  (REPS=$REPS THREADS='$THREADS')" > "$CMDS"

run() {
  local bin=$BUILD/$1
  shift
  [[ -x $bin ]] || return 0
  echo "$bin $* --reps $REPS --validate --csv $OUT --machine $MACHINE" >> "$CMDS"
  "$bin" "$@" --reps "$REPS" --validate --csv "$OUT" --machine "$MACHINE"
}

for section in $SECTIONS; do
  case $section in
  compare) # every backend, every op, every size; omp at all cores with static schedule
    for size in $SIZES; do
      for op in $OPS; do
        run pagp_seq --gen "$size" --op "$op"
        [[ $op == blur2d ]] && run pagp_seq --gen "$size" --op blur2d --variant tiled
        run pagp_autovec --gen "$size" --op "$op"
        run pagp_simd --gen "$size" --op "$op"
        run pagp_omp --gen "$size" --op "$op" --variant static --threads "$NCPU"
      done
    done
    ;;
  scaling) # strong scaling: fixed size, thread sweep, three schedules
    for size in $SCALING_SIZES; do
      for op in $OPS; do
        for sched in static dynamic guided; do
          for t in $THREADS; do
            run pagp_omp --gen "$size" --op "$op" --variant "$sched" --threads "$t"
          done
        done
      done
    done
    ;;
  sync) # false sharing vs padded vs reduction vs atomic
    for v in false_sharing padded padded64 reduction atomic static; do
      for t in $THREADS; do
        run pagp_omp --gen 4096x4096 --op gray --variant "$v" --threads "$t"
      done
    done
    ;;
  weak) # weak scaling: 4096 x 1024 rows per thread
    for t in $THREADS; do
      run pagp_seq --gen "4096x$((1024 * t))" --op pipeline
      run pagp_omp --gen "4096x$((1024 * t))" --op pipeline --variant static --threads "$t"
    done
    ;;
  *)
    echo "unknown section: $section" >&2
    exit 2
    ;;
  esac
done

echo "wrote $OUT"
