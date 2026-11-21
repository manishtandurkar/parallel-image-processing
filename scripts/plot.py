#!/usr/bin/env python3
"""Plots and tables from bench/results/*.csv for one machine.

Derived metrics (computed here, never in C++):
  speedup    = seq median_ms / median_ms   (same machine, op, size; seq default variant)
  efficiency = speedup / workers
  throughput = megapixels per second from median_ms

Usage: python scripts/plot.py --machine mac-m5 [--csv FILE ...] [--size 4096x4096]
       [--scaling-size 7680x4320] [--out report]
Writes report/figs/<machine>/*.png and report/tables/<machine>_compare.md.

GPU machines (cuda/acc rows, no seq): python scripts/plot.py --machine colab-t4 --cpu-machine mac-m5
GPU times: median_ms is compute only, end-to-end adds h2d_ms + d2h_ms (acc copy: median_ms already
includes its transfers). The GPU vs CPU table compares two different machines and says so.
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
    df = pd.concat([pd.read_csv(p).assign(src=p) for p in paths], ignore_index=True)
    df = df[df.machine == machine].copy()
    if df.empty:
        raise SystemExit(f"no rows for machine {machine}")
    df["size"] = df.width.astype(str) + "x" + df.height.astype(str)
    df["pixels"] = df.width * df.height
    df["workers_n"] = pd.to_numeric(df.workers, errors="coerce").fillna(1).astype(int)
    # Repeated runs of the same configuration: keep the latest. Speedup uses the seq run from the
    # same CSV file (run_mpi.sh records its own seq baseline next to the mpi rows).
    key = ["backend", "variant", "op", "width", "height", "workers"]
    df = df.drop_duplicates(key + ["src"], keep="last")
    seq = df[(df.backend == "seq") & (df.variant == "naive")]
    seq = seq[["src", "op", "width", "height", "median_ms"]].rename(columns={"median_ms": "seq_ms"})
    df = df.merge(seq, on=["src", "op", "width", "height"], how="left")
    # One seq row per configuration for display: the first file's.
    df = df[(df.backend != "seq") | ~df.duplicated(key, keep="first")]
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


def headline(d, ncpu):
    """One row per backend and config: seq naive, omp static and mpi blocking at the core count."""
    d = d[~((d.backend == "seq") & (d.variant != "naive"))]
    keep = {"omp": "static", "mpi": "blocking"}
    multi = d.backend.isin(list(keep))
    ok = (d.variant == d.backend.map(keep)) & (d.workers_n == ncpu)
    return d[~multi | ok]


def worker_label(b, ncpu):
    return {"omp": f"omp ({ncpu} threads)", "mpi": f"mpi ({ncpu} ranks)"}.get(b, b)


def compare_rows(df, size, ncpu):
    return headline(df[df["size"] == size], ncpu)


def fig_compare(df, size, ncpu, out):
    d = compare_rows(df, size, ncpu)
    backends = [b for b in BACKEND_STYLE if b in set(d.backend) and b != "seq"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    n = len(backends)
    width = 0.8 / max(n, 1)
    for i, b in enumerate(backends):
        sub = d[d.backend == b].set_index("op").reindex(OPS)
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(OPS))]
        label = worker_label(b, ncpu)
        ax.bar(xs, sub.speedup, width, color=BACKEND_STYLE[b][0], label=label,
               edgecolor=SURFACE, linewidth=1.5)
    ax.axhline(1, color=INK2, linewidth=1, linestyle="--")
    ax.text(len(OPS) - 0.5, 1, " seq = 1", va="bottom", ha="right", color=INK2, fontsize=8)
    ax.set_xticks(range(len(OPS)), OPS)
    ax.set_ylim(0, d.speedup.max() * 1.2)  # headroom for the legend
    ax.set_ylabel("speedup vs seq")
    ax.set_title(f"Speedup over the sequential baseline, {size_label(size)} image")
    ax.legend(ncols=n, loc="upper left")
    save(fig, out, f"compare_{size}.png")


def fig_throughput(df, ncpu, out):
    d = headline(df, ncpu)
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
    fig.suptitle(f"Throughput vs image size (omp/mpi at {ncpu} workers); sizes: "
                 + ", ".join(size_label(s) for s in sizes), y=1.08, fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, out, "throughput_vs_size.png")


def fig_scaling(df, size, metric, out, backend="omp", variants=("static", "dynamic", "guided"),
                xlabel="threads", prefix=""):
    d = df[(df.backend == backend) & (df["size"] == size) & df.variant.isin(variants)]
    if d.workers_n.nunique() < 2:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    tmax = d.workers_n.max()
    for ax, op in zip(axes.flat, OPS):
        for i, sched in enumerate(variants):
            sub = d[(d.op == op) & (d.variant == sched)].sort_values("workers_n")
            ax.plot(sub.workers_n, sub[metric], color=SERIES[i], marker=MARKERS[i], label=sched)
        ideal = [1, tmax] if metric == "speedup" else [1, 1]
        ax.plot([1, tmax], ideal, color=INK2, linewidth=1, linestyle="--", label="ideal")
        ax.set_title(op)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("speedup vs seq" if metric == "speedup" else "parallel efficiency")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, out, f"{prefix}strong_{metric}_{size}.png")


def fig_mpi_comm(df, size, out):
    """Where MPI time goes: time inside MPI calls vs the rest, per rank count (blocking)."""
    d = df[(df.backend == "mpi") & (df["size"] == size) & (df.variant == "blocking")]
    if d.workers_n.nunique() < 2:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, op in zip(axes.flat, OPS):
        sub = d[d.op == op].sort_values("workers_n")
        xs = range(len(sub))
        comm = sub.comm_ms.fillna(0)
        ax.bar(xs, sub.median_ms - comm, color=SERIES[0], label="outside MPI calls",
               edgecolor=SURFACE, linewidth=1)
        ax.bar(xs, comm, bottom=sub.median_ms - comm, color=SERIES[1],
               label="inside MPI calls (scatter, halo, gather, waiting)", edgecolor=SURFACE,
               linewidth=1)
        ax.set_xticks(list(xs), sub.workers_n.tolist())
        ax.set_title(op)
    for ax in axes[-1]:
        ax.set_xlabel("ranks")
    for ax in axes[:, 0]:
        ax.set_ylabel("median time (ms)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"MPI time breakdown, blocking halo exchange, {size_label(size)} "
                 "(slowest rank's time in MPI calls)", y=1.07, fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, out, f"mpi_comm_{size}.png")


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
    series = [("omp", "static", 3), ("mpi", "blocking", 4), ("mpi", "overlap", 6)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    plotted = 0
    for b, v, i in series:
        d = df[(df.backend == b) & (df.op == "pipeline") & (df.variant == v) & (df.width == 4096)
               & (df.height == 1024 * df.workers_n)].sort_values("workers_n")
        if len(d) < 2 or 1 not in set(d.workers_n):
            continue
        t1 = d[d.workers_n == 1].median_ms.iloc[0]
        ax.plot(d.workers_n, t1 / d.median_ms, color=SERIES[i], marker=MARKERS[i], label=f"{b} {v}")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(1, color=INK2, linewidth=1, linestyle="--", label="ideal")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("threads or ranks (image = 4096 x 1024·workers)")
    ax.set_ylabel("weak-scaling efficiency  T1 / Tp")
    ax.set_title("Weak scaling, pipeline")
    ax.legend()
    save(fig, out, "weak_pipeline.png")


# Roofline work model per output pixel, from pixel.hpp (conversions and compares not counted).
# ops: arithmetic operations; float only except sobel, whose gradient is integer work
#   gray 3 mul + 2 add + 1 add (rounding) = 6
#   blur2d 25 mul + 25 add + 1 = 51;  blur_sep h 5+5, v 5+5+1 = 21
#   sobel gx, gy 2 x (5 add + 2 mul) + gx*gx+gy*gy (2 mul + 1 add) + sqrt + rounding add = 19
#   pipeline = fused = 6 + 21 + 19 = 46
# bytes: compulsory DRAM traffic, each buffer read or written once (no write-allocate):
#   gray 3 in + 1 out; blur2d, sobel 1 + 1; blur_sep 1 + 4 (float tmp written) + 4 (read) + 1;
#   pipeline 4 + 10 + 2 (all intermediates round-trip); fused 3 + 1 (intermediates stay in cache)
OPS_PER_PX = {"gray": 6, "blur2d": 51, "blur_sep": 21, "sobel": 19, "pipeline": 46, "fused": 46}
BYTES_PER_PX = {"gray": 4, "blur2d": 2, "blur_sep": 10, "sobel": 2, "pipeline": 16, "fused": 4}


def load_ceilings(machine):
    paths = sorted(glob.glob(f"bench/results/ceilings/{machine}_*.csv"))
    if not paths:
        return None
    c = pd.read_csv(paths[-1])
    return {(r.kind, int(r.threads)): r.value for r in c.itertuples()}


def roofline_points(df, size, ncpu):
    d = headline(df[(df["size"] == size) & df.backend.isin(["seq", "autovec", "simd", "omp"])], ncpu)
    d = d[d.op.isin(OPS)].copy()
    d["ai"] = d.op.map(OPS_PER_PX) / d.op.map(BYTES_PER_PX)
    d["gops"] = d.op.map(OPS_PER_PX) * d.pixels / d.median_ms / 1e6
    d["gbs"] = d.op.map(BYTES_PER_PX) * d.pixels / d.median_ms / 1e6
    return d


def fig_roofline(df, ceil, size, ncpu, out):
    import numpy as np
    d = roofline_points(df, size, ncpu)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ai = np.logspace(-1, 2.3, 200)
    for t, ls in [(1, "--"), (ncpu, "-")]:
        bw, pk = ceil.get(("triad_bw", t)), ceil.get(("peak_flops", t))
        if bw is None or pk is None:
            continue
        ax.plot(ai, np.minimum(pk, ai * bw), color=INK2, linestyle=ls, linewidth=1.5,
                label=f"{t} thread{'s' if t > 1 else ''}: {bw:.0f} GB/s, {pk:.0f} GFLOP/s")
    op_markers = dict(zip(OPS, MARKERS))
    for b in ["seq", "autovec", "simd", "omp"]:
        sub = d[d.backend == b]
        for r in sub.itertuples():
            ax.scatter(r.ai, r.gops, color=BACKEND_STYLE[b][0], marker=op_markers[r.op], s=55,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for op in OPS:  # op names once, next to the seq point
        r = d[(d.backend == "seq") & (d.op == op)]
        if len(r):
            ax.annotate(op, (r.ai.iloc[0], r.gops.iloc[0]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=BACKEND_STYLE[b][0], marker="o", linestyle="")
                for b in ["seq", "autovec", "simd", "omp"]]
    labels += ["seq", "autovec", "simd (NEON)", f"omp static ({ncpu} threads)"]
    handles += [Line2D([], [], color=INK2, marker=op_markers[o], linestyle="") for o in OPS]
    labels += OPS
    ax.legend(handles, labels, fontsize=8, ncols=2, loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (ops per byte of compulsory DRAM traffic)")
    ax.set_ylabel("attained performance (G ops/s)")
    ax.set_title(f"Roofline, {size_label(size)} image; measured ceilings (STREAM triad, "
                 "mul+add peak)", fontsize=10)
    save(fig, out, f"roofline_{size}.png")


def table_roofline(df, ceil, size, ncpu, path):
    d = roofline_points(df, size, ncpu)
    bw1, pk1 = ceil[("triad_bw", 1)], ceil[("peak_flops", 1)]
    bwn, pkn = ceil[("triad_bw", ncpu)], ceil[("peak_flops", ncpu)]
    lines = [f"Roofline at {size}. Ceilings measured with bench/ceilings.cpp: 1 thread "
             f"{bw1:.0f} GB/s, {pk1:.0f} GFLOP/s; {ncpu} threads {bwn:.0f} GB/s, {pkn:.0f} GFLOP/s.",
             "Attainable = min(peak, AI x bandwidth) for the backend's thread count; "
             "% = attained / attainable. Ridge point (peak / bandwidth): "
             f"{pk1 / bw1:.1f} ops/byte at 1 thread, {pkn / bwn:.1f} at {ncpu}. Sobel ops are mostly "
             "integer (8 x int16 per NEON instruction) measured against a float peak, so simd can "
             "exceed 100%.",
             "", f"| op | ops/px | bytes/px | AI | bound at 1 / {ncpu} threads | seq | simd | "
             f"omp {ncpu}t | omp effective GB/s |", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for op in OPS:
        ai = OPS_PER_PX[op] / BYTES_PER_PX[op]
        bound = " / ".join("memory" if ai * bw < pk else "compute" for bw, pk in
                           [(bw1, pk1), (bwn, pkn)])
        cells = []
        for b, bw, pk in [("seq", bw1, pk1), ("simd", bw1, pk1), ("omp", bwn, pkn)]:
            r = d[(d.backend == b) & (d.op == op)]
            if r.empty:
                cells.append("")
                continue
            att = min(pk, ai * bw)
            cells.append(f"{r.gops.iloc[0]:.1f} G ({100 * r.gops.iloc[0] / att:.0f}%)")
        o = d[(d.backend == "omp") & (d.op == op)]
        cells.append(f"{o.gbs.iloc[0]:.0f}" if len(o) else "")
        lines.append(f"| {op} | {OPS_PER_PX[op]} | {BYTES_PER_PX[op]} | {ai:.1f} | {bound} | "
                     + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(path)


GPU_OPS = OPS
TRANSFER_STYLE = {"h2d": SERIES[0], "compute": SERIES[1], "d2h": SERIES[2]}


def gpu_rows(df):
    d = df[df.backend.isin(["cuda", "acc"])].copy()
    d["h2d"] = d.h2d_ms.fillna(0)
    d["d2h"] = d.d2h_ms.fillna(0)
    d["e2e_ms"] = d.median_ms + d.h2d + d.d2h
    return d


def pick(d, backend, variant, size, op, workers=None):
    r = d[(d.backend == backend) & (d.variant == variant) & (d["size"] == size) & (d.op == op)]
    if workers is not None:
        r = r[r.workers.astype(str) == workers]
    return r.iloc[0] if len(r) else None


def fig_gpu_breakdown(d, size, out):
    """Stacked h2d / compute / d2h per op: CUDA naive (16x16) next to OpenACC parallel."""
    configs = [("cuda", "naive", "16x16", "CUDA naive"), ("acc", "parallel", None, "OpenACC parallel")]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.38
    for i, (b, v, wk, label) in enumerate(configs):
        for j, op in enumerate(GPU_OPS):
            r = pick(d, b, v, size, op, wk) if op != "fused" or b == "acc" else \
                pick(d, "cuda", "tiled", size, op, "16x16")
            if r is None:
                continue
            x = j + (i - 0.5) * width
            bottom = 0
            for part, val in [("h2d", r.h2d), ("compute", r.median_ms), ("d2h", r.d2h)]:
                ax.bar(x, val, width, bottom=bottom, color=TRANSFER_STYLE[part],
                       edgecolor=SURFACE, linewidth=1, hatch="//" if b == "acc" else None,
                       label=f"{part}" if j == 0 and i == 0 else None)
                bottom += val
    ax.set_xticks(range(len(GPU_OPS)), GPU_OPS)
    ax.set_ylabel("median time (ms)")
    ax.set_title(f"GPU time breakdown, {size_label(size)}: left bar CUDA naive 16x16 (fused: tiled), "
                 "right bar (hatched) OpenACC parallel", fontsize=9)
    ax.legend(loc="upper left")
    save(fig, out, f"gpu_breakdown_{size}.png")


def fig_cuda_variants(d, size, out):
    c = d[(d.backend == "cuda") & (d["size"] == size) & (d.workers.astype(str) == "16x16")]
    if c.empty:
        return
    variants = ["naive", "const", "tiled"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.27
    for i, v in enumerate(variants):
        sub = c[c.variant == v].set_index("op").reindex(GPU_OPS)
        xs = [j + (i - 1) * width for j in range(len(GPU_OPS))]
        ax.bar(xs, sub.median_ms, width, color=SERIES[i], label=v, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(range(len(GPU_OPS)), GPU_OPS)
    ax.set_ylabel("kernel time (ms)")
    ax.set_title(f"CUDA memory variants, kernel time only, 16x16 blocks, {size_label(size)}")
    ax.legend(ncols=3, loc="upper left")
    save(fig, out, f"cuda_variants_{size}.png")


def fig_cuda_blocks(d, size, out):
    c = d[(d.backend == "cuda") & (d["size"] == size)]
    blocks = ["8x8", "16x16", "32x8", "32x32"]
    ops = [o for o in ["blur2d", "sobel", "pipeline", "fused"] if (c.op == o).any()]
    if c.workers.nunique() < 2 or not ops:
        return
    fig, axes = plt.subplots(1, len(ops), figsize=(3 * len(ops), 3.4), sharey=False)
    for ax, op in zip(axes, ops):
        for i, v in enumerate(["naive", "const", "tiled"]):
            sub = c[(c.op == op) & (c.variant == v)]
            sub = sub.set_index(sub.workers.astype(str)).reindex(blocks)
            if sub.median_ms.notna().sum() > 1:
                ax.plot(blocks, sub.median_ms, color=SERIES[i], marker=MARKERS[i], label=v)
        ax.set_title(op)
        ax.set_xlabel("block (x by y threads)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("kernel time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"CUDA thread-block shape, {size_label(size)}", y=1.12, fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, out, f"cuda_blocks_{size}.png")


def fig_acc_variants(d, size, small, out):
    """Left: end-to-end time of the data strategies. Right: plain kernels vs parallel (compute)."""
    a = d[d.backend == "acc"]
    if a.empty:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    variants = [("parallel", "parallel + data region"), ("kernels_indep", "kernels + loop independent"),
                ("copy", "copy clauses, no data region")]
    width = 0.27
    for i, (v, label) in enumerate(variants):
        sub = a[(a.variant == v) & (a["size"] == size)].set_index("op").reindex(GPU_OPS)
        xs = [j + (i - 1) * width for j in range(len(GPU_OPS))]
        ax.bar(xs, sub.e2e_ms, width, color=SERIES[i], label=label, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(range(len(GPU_OPS)), GPU_OPS)
    ax.set_ylabel("end-to-end time incl. transfers (ms)")
    ax.set_title(f"OpenACC data strategies, {size_label(size)}")
    ax.legend(loc="upper left")
    k = a[a["size"] == small]
    if (k.variant == "kernels").any():
        for i, v in enumerate(["parallel", "kernels"]):
            sub = k[k.variant == v].set_index("op").reindex(GPU_OPS)
            xs = [j + (i - 0.5) * 0.38 for j in range(len(GPU_OPS))]
            ax2.bar(xs, sub.median_ms, 0.38, color=SERIES[3 + i], edgecolor=SURFACE, linewidth=1.5,
                    label="parallel loop" if v == "parallel" else "kernels (compiler decides)")
        ax2.set_yscale("log")
        ax2.set_xticks(range(len(GPU_OPS)), GPU_OPS, rotation=30)
        ax2.set_ylabel("compute time (ms, log scale)")
        ax2.set_title(f"kernels without independence hint, {size_label(small)}")
        ax2.legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, f"acc_variants_{size}.png")


def fig_gpu_throughput(d, out):
    configs = [("cuda", "naive", "16x16"), ("acc", "parallel", None)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, op in zip(axes.flat, GPU_OPS):
        for i, (b, v, wk) in enumerate(configs):
            vv = "tiled" if (b == "cuda" and op == "fused") else v
            sub = d[(d.backend == b) & (d.variant == vv) & (d.op == op)]
            if wk:
                sub = sub[sub.workers.astype(str) == wk]
            sub = sub.sort_values("pixels")
            color, marker = BACKEND_STYLE[b]
            ax.plot(sub.pixels / 1e6, sub.pixels / 1e3 / sub.median_ms, color=color, marker=marker,
                    label=f"{b} {vv}, compute only")
            ax.plot(sub.pixels / 1e6, sub.pixels / 1e3 / sub.e2e_ms, color=color, marker=marker,
                    linestyle="--", label=f"{b} {vv}, incl. transfers")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(op)
    for ax in axes[-1]:
        ax.set_xlabel("image size (megapixels)")
    for ax in axes[:, 0]:
        ax.set_ylabel("throughput (MP/s, log)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("GPU throughput: compute only (solid) vs including host-device transfers (dashed); "
                 "fused panel uses cuda tiled",
                 y=1.1, fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, out, "gpu_throughput_vs_size.png")


def table_gpu_vs_cpu(d, cpu, size, gpu_machine, cpu_machine, path):
    ncpu = int(cpu[cpu.backend == "omp"].workers_n.max())
    cols = ["seq", f"omp {ncpu}t", "cuda kernel", "cuda end-to-end", "acc compute",
            "acc end-to-end"]
    lines = [f"Median time in ms at {size}. seq/omp: {cpu_machine}; cuda/acc: {gpu_machine}. "
             "Different machines: GPU speedup is only indicative.",
             "cuda = naive 16x16 (fused: tiled), acc = parallel. Speedup vs seq in brackets.", "",
             "| op | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for op in OPS:
        s = cpu[(cpu.backend == "seq") & (cpu.variant == "naive") & (cpu["size"] == size)
                & (cpu.op == op)]
        o = cpu[(cpu.backend == "omp") & (cpu.variant == "static") & (cpu["size"] == size)
                & (cpu.op == op) & (cpu.workers_n == ncpu)]
        c = pick(d, "cuda", "tiled" if op == "fused" else "naive", size, op, "16x16")
        a = pick(d, "acc", "parallel", size, op)
        base = s.median_ms.iloc[0] if len(s) else None
        vals = [base, o.median_ms.iloc[0] if len(o) else None,
                c.median_ms if c is not None else None, c.e2e_ms if c is not None else None,
                a.median_ms if a is not None else None, a.e2e_ms if a is not None else None]
        cells = ["" if v is None else (f"{v:.2f}" + (f" ({base / v:.1f}x)" if base else ""))
                 for v in vals]
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(path)


def gpu_main(df, args, figs, tables):
    d = gpu_rows(df)
    fig_gpu_breakdown(d, args.gpu_size, figs)
    fig_cuda_variants(d, args.gpu_size, figs)
    fig_cuda_blocks(d, args.scaling_size, figs)
    fig_acc_variants(d, args.gpu_size, "512x512", figs)
    fig_gpu_throughput(d, figs)
    if args.cpu_machine:
        cpu_paths = sorted(glob.glob(f"bench/results/{args.cpu_machine}_*.csv"))
        cpu = load(cpu_paths, args.cpu_machine)
        table_gpu_vs_cpu(d, cpu, args.size, args.machine, args.cpu_machine,
                         tables / f"{args.machine}_vs_{args.cpu_machine}.md")


def table_compare(df, size, ncpu, path):
    d = compare_rows(df, size, ncpu)
    backends = [b for b in BACKEND_STYLE if b in set(d.backend)]
    lines = [f"Median time in ms (speedup vs seq), {size}, omp = static at {ncpu} threads, "
             f"mpi = blocking at {ncpu} ranks.", "",
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
    ap.add_argument("--gpu-size", default="16384x16384", help="size for the GPU bar charts")
    ap.add_argument("--cpu-machine", help="GPU machines: CPU machine for the GPU vs CPU table")
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
    if not (df.backend == "seq").any():  # GPU machine: no same-machine seq baseline
        gpu_main(df, args, figs, tables)
        return

    if (df["size"] == args.size).any():
        fig_compare(df, args.size, ncpu, figs)
        table_compare(df, args.size, ncpu, tables / f"{args.machine}_compare.md")
    fig_throughput(df, ncpu, figs)
    for metric in ["speedup", "efficiency"]:
        fig_scaling(df, args.scaling_size, metric, figs)
        fig_scaling(df, args.scaling_size, metric, figs, backend="mpi",
                    variants=("blocking", "overlap"), xlabel="ranks", prefix="mpi_")
    fig_mpi_comm(df, args.scaling_size, figs)
    fig_sync(df, figs)
    fig_weak(df, figs)
    ceil = load_ceilings(args.machine)
    if ceil and (df["size"] == args.scaling_size).any():
        fig_roofline(df, ceil, args.scaling_size, ncpu, figs)
        table_roofline(df, ceil, args.scaling_size, ncpu, tables / f"{args.machine}_roofline.md")


if __name__ == "__main__":
    main()
