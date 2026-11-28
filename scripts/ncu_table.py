#!/usr/bin/env python3
"""Summarize the Nsight Compute profiles from bench/run_ncu.sh into a markdown table.

Reads report/cuda/ncu/<op>_<variant>.csv (ncu --csv --page raw: header row, units row, one row per
kernel) and writes report/tables/colab-t4_ncu.md. Multi-kernel ops (blur_sep) are summed for
time/bytes/instructions and time-weighted for percentages.

Usage: python scripts/ncu_table.py [--dir report/cuda/ncu] [--out report/tables/colab-t4_ncu.md]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ORDER = ["blur2d_naive", "blur2d_const", "blur2d_tiled", "blur_sep_naive", "blur_sep_const",
         "blur_sep_tiled", "sobel_naive", "sobel_tiled"]
SUM = {"gpu__time_duration.sum": "time_us", "dram__bytes_read.sum": "dram_read_MB",
       "dram__bytes_write.sum": "dram_write_MB", "smsp__inst_executed.sum": "inst_M",
       "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "smem_bank_conflicts"}
PCT = {"dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_%peak",
       "l1tex__t_sector_hit_rate.pct": "L1_hit_%", "lts__t_sector_hit_rate.pct": "L2_hit_%",
       "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_%",
       "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_%peak",
       "smsp__average_warp_latency_issue_stalled_barrier.ratio": "barrier_stall"}
SCALE = {"gpu__time_duration.sum": {"ns": 1e-3, "us": 1, "usecond": 1, "msecond": 1e3, "ms": 1e3,
                                    "nsecond": 1e-3},
         "dram__bytes_read.sum": {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1, "Gbyte": 1e3},
         "dram__bytes_write.sum": {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1, "Gbyte": 1e3},
         "smsp__inst_executed.sum": {"inst": 1e-6, "Kinst": 1e-3, "Minst": 1, "Ginst": 1e3}}


def num(s) -> float:
    return float(str(s).replace(",", ""))


def summarize(path: Path) -> dict:
    df = pd.read_csv(path, dtype=str)
    units, rows = df.iloc[0], df.iloc[1:]
    row = {"kernels": " + ".join(re.search(r"k_\w+", k).group(0) for k in rows["Kernel Name"])}
    t = rows["gpu__time_duration.sum"].map(num) * SCALE["gpu__time_duration.sum"].get(
        str(units["gpu__time_duration.sum"]), 1)
    for m, name in SUM.items():
        if m not in rows:
            continue
        v = rows[m].map(num) * SCALE.get(m, {}).get(str(units[m]), 1)
        row[name] = v.sum()
    for m, name in PCT.items():
        if m in rows:
            row[name] = (rows[m].map(num) * t).sum() / t.sum()
    row["regs"] = "/".join(rows["launch__registers_per_thread"])
    return row


def markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="report/cuda/ncu")
    ap.add_argument("--out", default="report/tables/colab-t4_ncu.md")
    args = ap.parse_args()
    recs = []
    for name in ORDER:
        p = Path(args.dir) / f"{name}.csv"
        if p.exists():
            op, variant = name.rsplit("_", 1)
            recs.append({"op": op, "variant": variant, **summarize(p)})
    df = pd.DataFrame(recs)
    cols = ["op", "variant", "time_us", "dram_read_MB", "dram_write_MB", "dram_%peak", "L1_hit_%",
            "L2_hit_%", "inst_M", "occupancy_%", "sm_%peak", "smem_bank_conflicts",
            "barrier_stall", "regs", "kernels"]
    df = df[[c for c in cols if c in df]]
    text = ("# Nsight Compute: CUDA stencil variants, Tesla T4, 4096x4096\n\n"
            "Source: report/cuda/ncu/*.csv from bench/run_ncu.sh (one profiled run after 2 "
            "warm-ups; ncu replays each kernel, so times are slightly above the benchmark "
            "medians). blur_sep = blur_h + blur_v kernels: sums, percentages time-weighted. "
            "barrier_stall = smsp__average_warp_latency_issue_stalled_barrier.ratio "
            "(warp latency spent waiting at __syncthreads; 0 when the kernel has none).\n\n" + markdown(df) + "\n")
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
