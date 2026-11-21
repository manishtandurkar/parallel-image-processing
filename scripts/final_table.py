#!/usr/bin/env python3
"""Final cross-backend comparison table: report/tables/final_comparison.md.

One row per backend at one image size (default 4096x4096, the size every backend ran at):
  - configuration = the backend's fastest variant (and worker count / block) for the pipeline op;
    GPU backends are ranked by end-to-end time (h2d + compute + d2h)
  - speedup vs seq on the CPU machine, from the same CSV file as the backend's row (run_mpi.sh
    records its own seq baseline); GPU rows use the main CPU sweep and compare two different
    machines (marked)
  - efficiency = speedup / workers, only where workers are threads or ranks
  - geomean speedup over the six ops, same configuration (fused falls back to the backend's
    fastest fused variant if the configuration has none, e.g. cuda naive)
  - code size: non-blank, non-comment lines of the backend's own source (kernels + backend glue,
    driver separately), and directive lines (#pragma omp / acc). The backends are separate files,
    so this stands in for "lines changed vs seq".

Usage: python scripts/final_table.py [--cpu-machine mac-m5] [--gpu-machine colab-t4]
       [--size 4096x4096] [--out report/tables/final_comparison.md]
"""
import argparse
import glob
import math
import re
from pathlib import Path

import pandas as pd

OPS = ["gray", "blur2d", "blur_sep", "sobel", "pipeline", "fused"]

# (backend, model, own source files, driver files)
BACKENDS = [
    ("seq", "sequential reference", ["src/seq/seq.cpp", "src/seq/backend.cpp"], ["src/seq/main.cpp"]),
    ("autovec", "compiler auto-vectorization", ["src/simd/autovec.cpp"], ["src/simd/main_autovec.cpp"]),
    ("simd", "NEON intrinsics", ["src/simd/neon.cpp"], ["src/simd/main_simd.cpp"]),
    ("omp", "OpenMP threads", ["src/omp/omp.cpp"], ["src/omp/main.cpp"]),
    ("mpi", "MPI processes", ["src/mpi/mpi.cpp", "src/mpi/mpi_backend.hpp"], ["src/mpi/main.cpp"]),
    ("cuda", "CUDA kernels", ["src/cuda/cuda.cu", "src/cuda/cuda_backend.hpp"], ["src/cuda/main.cpp"]),
    ("acc", "OpenACC directives (GPU)", ["src/acc/acc.cpp", "src/acc/acc_backend.hpp"],
     ["src/acc/main.cpp"]),
]
SHARED = ["include/pagp/pixel.hpp", "include/pagp/rows.hpp", "src/core/rows.cpp"]
GPU = {"cuda", "acc"}


def sloc(paths):
    """Non-blank lines that are not only a comment (// or /* */ blocks)."""
    n = 0
    for p in paths:
        in_block = False
        for line in Path(p).read_text().splitlines():
            s = line.strip()
            if in_block:
                if "*/" in s:
                    in_block = False
                    s = s.split("*/", 1)[1].strip()
                else:
                    continue
            if s.startswith("/*"):
                if "*/" not in s:
                    in_block = True
                continue
            if s and not s.startswith("//"):
                n += 1
    return n


def directives(paths):
    pat = re.compile(r"#\s*pragma\s+(omp|acc)\b|PAGP_ACC_PRAGMA\(acc")
    return sum(len(pat.findall(Path(p).read_text())) for p in paths)


def load(machine):
    paths = sorted(glob.glob(f"bench/results/{machine}_*.csv"))
    df = pd.concat([pd.read_csv(p).assign(src=p) for p in paths], ignore_index=True)
    df = df[df.machine == machine].copy()
    df["size"] = df.width.astype(str) + "x" + df.height.astype(str)
    df["workers_n"] = pd.to_numeric(df.workers, errors="coerce")
    df["e2e_ms"] = df.median_ms + df.h2d_ms.fillna(0) + df.d2h_ms.fillna(0)
    # Repeated configurations within a file: keep the latest run.
    return df.drop_duplicates(["src", "backend", "variant", "op", "size", "workers"], keep="last")


def best_config(d, backend, size):
    """Fastest pipeline configuration (variant, workers) at the backend's largest worker count."""
    p = d[(d.backend == backend) & (d["size"] == size) & (d.op == "pipeline")]
    if backend == "seq":  # seq appears in several files; the main CPU sweep is the reference
        p = p[(p.variant == "naive") & (p.src == d[d.backend == "omp"].src.iloc[0])]
    if p.empty:
        return None
    if p.workers_n.notna().any() and backend in ("omp", "mpi"):
        p = p[p.workers_n == p.workers_n.max()]
    key = "e2e_ms" if backend in GPU else "median_ms"
    return p.sort_values(key).iloc[0]


def op_row(d, cfg, op, size):
    r = d[(d.backend == cfg.backend) & (d["size"] == size) & (d.op == op)
          & (d.variant == cfg.variant) & (d.workers.astype(str) == str(cfg.workers))]
    if r.empty:  # configuration lacks this op (e.g. cuda fused exists only as tiled)
        r = d[(d.backend == cfg.backend) & (d["size"] == size) & (d.op == op)]
        if cfg.backend in ("omp", "mpi"):
            r = r[r.workers_n == cfg.workers_n]
        r = r.sort_values("e2e_ms" if cfg.backend in GPU else "median_ms")
    return r.iloc[0] if len(r) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cpu-machine", default="mac-m5")
    ap.add_argument("--gpu-machine", default="colab-t4")
    ap.add_argument("--size", default="4096x4096")
    ap.add_argument("--out", default="report/tables/final_comparison.md")
    a = ap.parse_args()

    cpu, gpu = load(a.cpu_machine), load(a.gpu_machine)
    s = cpu[(cpu.backend == "seq") & (cpu.variant == "naive") & (cpu["size"] == a.size)]
    seq_by_src = {src: dict(zip(g.op, g.median_ms)) for src, g in s.groupby("src")}
    main_src = cpu[cpu.backend == "omp"].src.iloc[0]  # the main CPU sweep: baseline for GPU rows
    seq = seq_by_src[main_src]
    shared = sloc(SHARED)
    seq_sloc = sloc(BACKENDS[0][2])

    head = ["backend", "model", "machine", "configuration", "pipeline ms", "speedup", "efficiency",
            "geomean speedup (6 ops)", "own SLOC", "vs seq", "driver SLOC", "directives"]
    rows, per_op = [], []
    for name, model, src, drv in BACKENDS:
        d = gpu if name in GPU else cpu
        machine = a.gpu_machine if name in GPU else a.cpu_machine
        cfg = best_config(d, name, a.size)
        if cfg is None:
            continue
        workers = str(cfg.workers)
        conf = cfg.variant
        if name in ("omp", "mpi"):
            conf += f", {int(cfg.workers_n)} {'threads' if name == 'omp' else 'ranks'}"
        elif name == "cuda":
            conf += f", block {workers}"
        ops = {op: op_row(d, cfg, op, a.size) for op in OPS}
        seq = seq_by_src.get(cfg.src, seq_by_src[main_src])
        if name in GPU:
            t = f"{cfg.median_ms:.2f} kernel / {cfg.e2e_ms:.2f} e2e"
            sp = f"{seq['pipeline'] / cfg.median_ms:.1f}x / {seq['pipeline'] / cfg.e2e_ms:.1f}x †"
            gm = [math.log(seq[o] / ops[o].e2e_ms) for o in OPS if ops[o] is not None]
            gms = f"{math.exp(sum(gm) / len(gm)):.1f}x e2e †"
        else:
            t = f"{cfg.median_ms:.2f}"
            sp = f"{seq['pipeline'] / cfg.median_ms:.2f}x"
            gm = [math.log(seq[o] / ops[o].median_ms) for o in OPS if ops[o] is not None]
            gms = f"{math.exp(sum(gm) / len(gm)):.2f}x"
        eff = (f"{seq['pipeline'] / cfg.median_ms / cfg.workers_n:.0%}"
               if name in ("omp", "mpi") else "")
        own = sloc(src)
        rows.append([name, model, machine, conf, t, sp, eff, gms, str(own),
                     "" if name == "seq" else f"{own / seq_sloc:.1f}x", str(sloc(drv)),
                     str(directives(src)) if name in ("omp", "acc") else ""])
        cells = []
        for op in OPS:
            r = ops[op]
            if r is None:
                cells.append("")
            elif name in GPU:
                cells.append(f"{seq[op] / r.e2e_ms:.1f}x ({seq[op] / r.median_ms:.0f}x) †")
            else:
                cells.append(f"{seq[op] / r.median_ms:.2f}x")
        per_op.append([f"{name} ({conf})"] + cells)

    lines = [
        f"# Final comparison, {a.size}",
        "",
        f"Speedup = {a.cpu_machine} seq time / backend time, per op, with seq from the same result "
        f"file as the backend (main CPU sweep seq {a.size} pipeline: "
        f"{seq_by_src[main_src]['pipeline']:.2f} ms). Each backend uses its fastest configuration for pipeline "
        "(GPU: by end-to-end time).",
        f"† GPU rows ran on {a.gpu_machine}, a different machine from seq ({a.cpu_machine}): "
        "those speedups are indicative only. GPU speedups are given as kernel-only / end-to-end "
        "(end-to-end = host-to-device + compute + device-to-host).",
        "Efficiency = speedup / workers (threads or ranks). Code size: non-blank, non-comment lines "
        "of each backend's own source (backend files are separate, so this stands in for lines "
        f"changed vs seq); all CPU backends also reuse the shared row drivers and per-pixel "
        f"kernels ({shared} SLOC: pixel.hpp, rows.hpp, rows.cpp). Directives = #pragma omp/acc "
        "lines.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * 4 + ["---:"] * (len(head) - 4)) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", "## Speedup per op (same configuration; GPU: end-to-end with kernel-only in "
              "brackets)", "",
              "| backend | " + " | ".join(OPS) + " |", "|---|" + "---:|" * len(OPS)]
    lines += ["| " + " | ".join(r) + " |" for r in per_op]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
