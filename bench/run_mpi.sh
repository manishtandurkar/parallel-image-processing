#!/usr/bin/env bash
# MPI benchmark sweep for one local machine (single node). Writes bench/results/<machine>_mpi_<date>.csv
# and a .cmd file next to it with the exact command lines that produced it.
#
# Usage: bench/run_mpi.sh <machine-tag> [section ...]
#   sections: compare scaling weak   (default: all)
# Env overrides:
#   BUILD=build  REPS=10  SIZES="1024x1024 ... 7680x4320"  SCALING_SIZE=7680x4320
#   RANKS="1 2 4 ..."  (default: 1 2 4 6 8 ... up to the core count)
set -euo pipefail

MACHINE=${1:?usage: bench/run_mpi.sh <machine-tag> [compare|scaling|weak ...]}
shift
SECTIONS=${*:-compare scaling weak}

cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
REPS=${REPS:-10}
SIZES=${SIZES:-"1024x1024 2048x2048 4096x4096 7680x4320"}
SCALING_SIZE=${SCALING_SIZE:-7680x4320}
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
if [[ -z ${RANKS:-} ]]; then
  RANKS="1 2"
  for ((r = 4; r <= NCPU; r += 2)); do RANKS+=" $r"; done
  [[ " $RANKS " == *" $NCPU "* ]] || RANKS+=" $NCPU"
fi
OPS="gray blur2d blur_sep sobel pipeline fused"

if command -v pmset >/dev/null && pmset -g batt | grep -q "Battery Power"; then
  echo "WARNING: running on battery; plug in power for benchmark numbers (CLAUDE.md protocol)" >&2
fi
[[ -x $BUILD/pagp_mpi ]] || { echo "error: $BUILD/pagp_mpi not built (-DPAGP_MPI=ON)" >&2; exit 1; }

mkdir -p bench/results
OUT=bench/results/${MACHINE}_mpi_$(date +%Y%m%d-%H%M%S).csv
CMDS=${OUT%.csv}.cmd
echo "# $0 $MACHINE $SECTIONS  (REPS=$REPS RANKS='$RANKS')" > "$CMDS"

run() { # run <np> <args...>
  local np=$1
  shift
  local cmd=(mpirun -np "$np" "$BUILD/pagp_mpi" "$@" --reps "$REPS" --validate --csv "$OUT"
             --machine "$MACHINE")
  echo "${cmd[*]}" >> "$CMDS"
  "${cmd[@]}"
}

seq_run() { # same-file seq baseline for speedup
  local cmd=("$BUILD/pagp_seq" "$@" --reps "$REPS" --validate --csv "$OUT" --machine "$MACHINE")
  echo "${cmd[*]}" >> "$CMDS"
  "${cmd[@]}"
}

variants() { [[ $1 == gray ]] && echo blocking || echo "blocking overlap"; }

for section in $SECTIONS; do
  case $section in
  compare) # every op and variant at all cores, several sizes
    for size in $SIZES; do
      for op in $OPS; do
        seq_run --gen "$size" --op "$op"
        for v in $(variants "$op"); do
          run "$NCPU" --gen "$size" --op "$op" --variant "$v"
        done
      done
    done
    ;;
  scaling) # strong scaling: fixed size, rank sweep
    for op in $OPS; do
      seq_run --gen "$SCALING_SIZE" --op "$op"
      for v in $(variants "$op"); do
        for np in $RANKS; do
          run "$np" --gen "$SCALING_SIZE" --op "$op" --variant "$v"
        done
      done
    done
    ;;
  weak) # weak scaling: 4096 x 1024 rows per rank
    for np in $RANKS; do
      seq_run --gen "4096x$((1024 * np))" --op pipeline
      for v in blocking overlap; do
        run "$np" --gen "4096x$((1024 * np))" --op pipeline --variant "$v"
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
