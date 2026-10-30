#!/usr/bin/env python3
"""Plots and tables from bench/results/*.csv for one machine.

Derived metrics (computed here, never in C++):
  speedup    = seq median_ms / median_ms   (same machine, op, size; seq default variant)
  efficiency = speedup / workers
  throughput = megapixels per second from median_ms

Usage: python scripts/plot.py --machine mac-m5 [--csv FILE ...] [--size 4096x4096]
       [--scaling-size 7680x4320] [--out report]
Writes report/figs/<machine>/*.png and report/tables/<machine>_compare.md.
"""
import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OPS = ["gray", "blur2d", "blur_sep", "sobel", "pipeline", "fused"]
# Fixed colour per entity (never by rank), validated categorical order; markers carry identity too.
BACKEND_STYLE = {
    "seq": ("#2a78d6", "o"),
    "autovec": ("#eb6834", "s"),
    "simd": ("#1baf7a", "^"),
    "omp": ("#eda100", "D"),
    "mpi": ("#e87ba4", "v"),
    "cuda": ("#008300", "P"),
    "acc": ("#6250d6", "X"),
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "figure.dpi": 150,
    })


def load(paths, machine):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df.machine == machine].copy()
    if df.empty:
        raise SystemExit(f"no rows for machine {machine}")
    df["size"] = df.width.astype(str) + "x" + df.height.astype(str)
    df["pixels"] = df.width * df.height
    df["workers_n"] = pd.to_numeric(df.workers, errors="coerce").fillna(1).astype(int)
    # Repeated runs of the same configuration: keep the latest.
    key = ["backend", "variant", "op", "width", "height", "workers"]
    df = df.drop_duplicates(key, keep="last")
    seq = df[(df.backend == "seq") & (df.variant == "naive")][["op", "width", "height", "median_ms"]]
    df = df.merge(seq.rename(columns={"median_ms": "seq_ms"}), on=["op", "width", "height"],
                  how="left")
    df["speedup"] = df.seq_ms / df.median_ms
    df["efficiency"] = df.speedup / df.workers_n
    df["mpix_s"] = df.pixels / 1e6 / (df.median_ms / 1e3)
    return df


def size_label(s):
    w, h = s.split("x")
    return f"{w}²" if w == h else s


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(out / name)


def compare_rows(df, size, ncpu):
    """One row per backend for the comparison: omp only at the full core count, static."""
    d = df[(df["size"] == size) & ~((df.backend == "seq") & (df.variant != "naive"))]
    omp = d.backend == "omp"
    return d[~omp | ((d.variant == "static") & (d.workers_n == ncpu))]


def fig_compare(df, size, ncpu, out):
    d = compare_rows(df, size, ncpu)
    backends = [b for b in BACKEND_STYLE if b in set(d.backend) and b != "seq"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    n = len(backends)
    width = 0.8 / max(n, 1)
    for i, b in enumerate(backends):
        sub = d[d.backend == b].set_index("op").reindex(OPS)
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(OPS))]
        label = f"omp ({ncpu} threads)" if b == "omp" else b
        ax.bar(xs, sub.speedup, width, color=BACKEND_STYLE[b][0], label=label,
               edgecolor=SURFACE, linewidth=1.5)
    ax.axhline(1, color=INK2, linewidth=1, linestyle="--")
    ax.text(len(OPS) - 0.5, 1, " seq = 1", va="bottom", ha="right", color=INK2, fontsize=8)
    ax.set_xticks(range(len(OPS)), OPS)
    ax.set_ylabel("speedup vs seq")
    ax.set_title(f"Speedup over the sequential baseline, {size_label(size)} image")
    ax.legend(ncols=n, loc="upper left")
    save(fig, out, f"compare_{size}.png")


def fig_throughput(df, ncpu, out):
    d = df[~((df.backend == "seq") & (df.variant != "naive"))]
    d = d[(d.backend != "omp") | ((d.variant == "static") & (d.workers_n == ncpu))]
    sizes = d.drop_duplicates("size").sort_values("pixels")["size"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, op in zip(axes.flat, OPS):
        for b, (color, marker) in BACKEND_STYLE.items():
            sub = d[(d.backend == b) & (d.op == op)].sort_values("pixels")
            if not sub.empty:
                ax.plot(sub.pixels / 1e6, sub.mpix_s, color=color, marker=marker, label=b)
        ax.set_xscale("log")
        ax.set_title(op)
    for ax in axes[-1]:
        ax.set_xlabel("image size (megapixels)")
    for ax in axes[:, 0]:
        ax.set_ylabel("throughput (MP/s)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(labels), bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Throughput vs image size (omp at {ncpu} threads); sizes: "
                 + ", ".join(size_label(s) for s in sizes), y=1.08, fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, out, "throughput_vs_size.png")


def fig_scaling(df, size, metric, out):
    d = df[(df.backend == "omp") & (df["size"] == size)
           & df.variant.isin(["static", "dynamic", "guided"])]
    if d.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    tmax = d.workers_n.max()
    for ax, op in zip(axes.flat, OPS):
        for i, sched in enumerate(["static", "dynamic", "guided"]):
            sub = d[(d.op == op) & (d.variant == sched)].sort_values("workers_n")
            ax.plot(sub.workers_n, sub[metric], color=SERIES[i], marker=MARKERS[i], label=sched)
        ideal = [1, tmax] if metric == "speedup" else [1, 1]
        ax.plot([1, tmax], ideal, color=INK2, linewidth=1, linestyle="--", label="ideal")
        ax.set_title(op)
    for ax in axes[-1]:
        ax.set_xlabel("threads")
    for ax in axes[:, 0]:
        ax.set_ylabel("speedup vs seq" if metric == "speedup" else "parallel efficiency")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, out, f"strong_{metric}_{size}.png")


def fig_sync(df, out):
    variants = ["false_sharing", "padded64", "padded", "reduction", "atomic", "static"]
    d = df[(df.backend == "omp") & (df.op == "gray") & df.variant.isin(variants[:5])]
    if d.empty:
        return
    size = d["size"].iloc[0]
    d = df[(df.backend == "omp") & (df.op == "gray") & (df["size"] == size)
           & df.variant.isin(variants)]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {"static": "no counter (plain gray)"}
    for i, v in enumerate(variants):
        sub = d[d.variant == v].sort_values("workers_n")
        ax.plot(sub.workers_n, sub.median_ms, color=SERIES[i], marker=MARKERS[i],
                label=labels.get(v, v))
    ax.set_yscale("log")
    ax.set_xlabel("threads")
    ax.set_ylabel("median time (ms, log scale)")
    ax.set_title(f"Per-thread counters: false sharing vs padding vs reduction vs atomic, "
                 f"{size_label(size)}")
    ax.legend()
    save(fig, out, "sync_counters.png")


def fig_weak(df, out):
    d = df[(df.backend == "omp") & (df.op == "pipeline") & (df.variant == "static")
           & (df.width == 4096) & (df.height == 1024 * df.workers_n)].sort_values("workers_n")
    if len(d) < 2 or 1 not in set(d.workers_n):
        return
    t1 = d[d.workers_n == 1].median_ms.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(d.workers_n, t1 / d.median_ms, color=SERIES[3], marker=MARKERS[3], label="omp static")
    ax.axhline(1, color=INK2, linewidth=1, linestyle="--", label="ideal")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("threads (image = 4096 x 1024·threads)")
    ax.set_ylabel("weak-scaling efficiency  T1 / Tp")
    ax.set_title("Weak scaling, pipeline")
    ax.legend()
    save(fig, out, "weak_pipeline.png")


def table_compare(df, size, ncpu, path):
    d = compare_rows(df, size, ncpu)
    backends = [b for b in BACKEND_STYLE if b in set(d.backend)]
    lines = [f"Median time in ms (speedup vs seq), {size}, omp = static at {ncpu} threads.", "",
             "| op | " + " | ".join(backends) + " |",
             "|---|" + "---:|" * len(backends)]
    for op in OPS:
        cells = []
        for b in backends:
            r = d[(d.backend == b) & (d.op == op)]
            cells.append("" if r.empty else
                         f"{r.median_ms.iloc[0]:.2f} ({r.speedup.iloc[0]:.2f}x)")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--csv", nargs="*", help="default: bench/results/<machine>_*.csv")
    ap.add_argument("--size", default="4096x4096", help="size for the comparison bar chart/table")
    ap.add_argument("--scaling-size", default="7680x4320")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()

    paths = args.csv or sorted(glob.glob(f"bench/results/{args.machine}_*.csv"))
    if not paths:
        raise SystemExit("no CSV files found")
    style()
    df = load(paths, args.machine)
    ncpu = int(df[df.backend == "omp"].workers_n.max()) if (df.backend == "omp").any() else 1
    figs = Path(args.out) / "figs" / args.machine
    tables = Path(args.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    bad = df[df.max_abs_diff.notna() & (df.max_abs_diff > 1)]
    if not bad.empty:
        print(f"WARNING: {len(bad)} rows exceed max_abs_diff 1")

    if (df["size"] == args.size).any():
        fig_compare(df, args.size, ncpu, figs)
        table_compare(df, args.size, ncpu, tables / f"{args.machine}_compare.md")
    fig_throughput(df, ncpu, figs)
    for metric in ["speedup", "efficiency"]:
        fig_scaling(df, args.scaling_size, metric, figs)
    fig_sync(df, figs)
    fig_weak(df, figs)


if __name__ == "__main__":
    main()
