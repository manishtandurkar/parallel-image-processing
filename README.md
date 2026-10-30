# Parallel Image Processing: A Comparative Study of Parallel Programming Models

**PAGP (CS372IA) Experiential Learning project, RVCE, 7th semester.**

We implement three image filters (grayscale, Gaussian blur, Sobel edge detection) in C++17 using several
parallel programming models, check that every version produces the same output as a sequential reference,
and benchmark them to explain *why* each one is fast or slow: cache locality, memory bandwidth,
synchronization, load balance, and communication or host-device transfer overhead.

The filters are the vehicle; the point of the project is the **measured analysis**.

## Backends

| Binary         | Model                                           | Runs on              | Status        |
|----------------|-------------------------------------------------|----------------------|---------------|
| `pagp_seq`     | Sequential reference (source of truth)          | Mac, WSL             | Done          |
| `pagp_autovec` | Vectorizer-friendly scalar code (compiler SIMD) | Mac, WSL             | Done          |
| `pagp_simd`    | Hand-written intrinsics: NEON (arm64) / AVX2 (x86-64) | Mac, WSL       | NEON done, AVX2 planned |
| `pagp_omp`     | OpenMP multithreading                           | Mac, WSL             | Done          |
| `pagp_mpi`     | MPI row-block decomposition + halo exchange     | Mac, WSL             | Planned       |
| `pagp_cuda`    | CUDA kernels (naive / tiled / const / fused)    | Google Colab T4      | Planned       |
| `pagp_acc`     | OpenACC (`nvc++ -acc=gpu` and `-acc=multicore`) | Google Colab T4      | Planned       |

## Filters

All backends follow one spec, defined once in [`include/pagp/filters.hpp`](include/pagp/filters.hpp):

- **Input:** 8-bit interleaved RGB; processing is single-channel after grayscale.
- **`gray`:** `Y = 0.299 R + 0.587 G + 0.114 B` in `float`, rounded and clamped to `[0, 255]`.
- **`blur2d` / `blur_sep`:** 5×5 Gaussian, σ = 1.0, applied directly (`blur2d`) or as separable
  1×5 + 5×1 passes with a float intermediate (`blur_sep`).
- **`sobel`:** 3×3 `Gx`, `Gy`; output `min(255, round(sqrt(Gx² + Gy²)))`.
- **`pipeline`:** `gray → blur_sep → sobel` as three passes. **`fused`:** the same in one pass, to
  compare memory traffic.
- **Borders:** clamp-to-edge everywhere.
- **Tolerance vs `seq`:** `gray` and `sobel` must match exactly; blur may differ by at most 1 per pixel.

## Quick start

### Prerequisites

macOS (Apple clang has no OpenMP, so use Homebrew LLVM):

```bash
brew install llvm libomp open-mpi cmake python
```

```bash
export CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++
```

Ubuntu / WSL2 (build inside WSL, never with native Windows/MSVC):

```bash
sudo apt install build-essential cmake ninja-build libopenmpi-dev openmpi-bin python3-pip linux-tools-generic
```

Python, for test images and plots:

```bash
pip install numpy pandas matplotlib pillow
```

### Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPAGP_SIMD=ON -DPAGP_OMP=ON -DPAGP_MPI=ON
```

```bash
cmake --build build -j
```

```bash
ctest --test-dir build --output-on-failure
```

Each backend is a CMake option (`PAGP_SIMD`, `PAGP_OMP`, `PAGP_MPI`, `PAGP_CUDA`, `PAGP_ACC`), so a missing
toolchain never breaks the rest of the build. `-DPAGP_AUTOVEC_REPORT=ON` prints the compiler's
vectorization report for the `autovec` target (saved reports live in [`report/autovec/`](report/autovec)).

Compiler flags: `-O3 -ffp-contract=off` on every target (no FMA contraction, so float rounding is identical
across compilers). `-march=native` / `-mcpu=native` is added **only** to `simd` and `autovec`, so the
scalar baseline stays comparable.

### Run

Every binary shares one CLI:

```bash
./build/pagp_omp --gen 4096x4096 --op pipeline --variant dynamic --threads 8 --reps 10 --validate
```

```
--input <path> | --gen <WxH>   load an image or generate a synthetic one
--op gray|blur2d|blur_sep|sobel|pipeline|fused
--variant <name>               backend-specific (see below)
--reps N                       timed runs after 2 untimed warm-ups (default 10)
--threads N                    omp worker threads
--validate                     compare with the seq reference, record max_abs_diff
--save <path>                  write the output image (PNG)
--csv <path>                   append one result row
--machine <tag>                e.g. mac-m5, wsl-x86, colab-t4
```

Variants available today:

| Backend   | Variants |
|-----------|----------|
| `seq`     | `naive`; `blur2d` also has `tiled` (cache blocking) |
| `autovec` | `autovec` |
| `simd`    | `neon` |
| `omp`     | `static`, `dynamic`, `guided`; `gray` also has `false_sharing`, `padded` (128 B), `padded64`, `reduction`, `atomic` |

## Benchmarking

```bash
python scripts/gen_images.py
```

```bash
bench/run_cpu.sh mac-m5
```

`run_cpu.sh <machine-tag> [compare|scaling|sync|weak ...]` runs the CPU sweep and writes
`bench/results/<machine>_<timestamp>.csv` plus a `.cmd` file with the exact commands that produced it.
Override defaults with `BUILD`, `REPS`, `SIZES`, `SCALING_SIZES`, `THREADS`.

```bash
python scripts/plot.py --machine mac-m5
```

`plot.py` computes the derived metrics (speedup vs `seq` on the same machine, efficiency, throughput in
MP/s, strong and weak scaling) and writes figures to `report/figs/<machine>/` and a comparison table to
`report/tables/<machine>_compare.md`.

Protocol highlights:

- Sizes 512², 1024², 2048², 4096², 7680×4320 (8K), 16384².
- Two warm-ups, then `--reps` timed runs; report **median** and min, never a single run.
- Only the compute region is timed. CUDA reports H2D / kernel / D2H separately; MPI reports compute and
  communication separately.
- Raw times are only compared within one machine (specs in [`bench/machines.md`](bench/machines.md)).
- CPU benchmarks never run on Colab.

CSV schema:

```
machine,backend,variant,op,width,height,workers,reps,median_ms,min_ms,h2d_ms,d2h_ms,comm_ms,max_abs_diff
```

## Hardware

| Tag       | Machine                                   | Used for |
|-----------|-------------------------------------------|----------|
| `mac-m5`  | MacBook, Apple M5 (4 P + 6 E cores), arm64 | seq, NEON, autovec, OpenMP, MPI |
| `wsl-x86` | Windows laptop, Ubuntu on WSL2, x86-64     | seq, AVX2, autovec, OpenMP, MPI |
| `colab-t4`| Google Colab free-tier NVIDIA T4           | CUDA and OpenACC only, driven via the Colab CLI |

The M5's mix of performance and efficiency cores makes `schedule(static)` imbalanced. That is one of the
load-balancing findings, not a bug.

## Repository layout

```
include/pagp/     shared headers: image, filter spec, CLI, timer, CSV, validation, backend interface
src/core/         image I/O, CLI parsing, CSV, validation, shared runner (main driver)
src/seq/          sequential reference
src/simd/         autovec.cpp, neon.cpp (avx2.cpp planned)
src/omp/          OpenMP backend
tests/            every backend vs seq on small and odd-sized images
scripts/          gen_images.py (synthetic inputs), plot.py (figures and tables)
bench/            run_cpu.sh sweep, machines.md, results/ CSVs
report/           figures, tables, compiler vectorization reports
third_party/stb/  stb_image / stb_image_write (vendored)
```

## What the study covers

| Concept | Where it is measured |
|---|---|
| Speedup, efficiency, Amdahl / Gustafson | every benchmark, strong and weak scaling sweeps |
| Cache locality | naive vs tiled `blur2d`, `blur2d` vs `blur_sep` |
| SIMD / data-level parallelism | NEON and AVX2 intrinsics vs compiler auto-vectorization |
| Synchronization, false sharing | OpenMP `false_sharing` vs `padded`, `reduction` vs `atomic` |
| Scheduling and load balance | OpenMP `static` / `dynamic` / `guided` on heterogeneous cores |
| Memory-bound vs compute-bound | roofline: `gray` vs `blur` / `sobel`, `pipeline` vs `fused` |
| Message passing and overlap | MPI blocking vs overlapped halo exchange |
| GPU memory hierarchy | CUDA naive vs shared-memory tiled vs constant memory, transfer breakdown |
| Directive-based offload | OpenACC `kernels` vs `parallel loop`, explicit vs implicit data regions |

See [`CLAUDE.md`](CLAUDE.md) for the full project spec, backend design notes, Colab CLI workflow, and
coding conventions.

## Team

Two-person team: Person A (Mac M5) owns the core library, `seq`, NEON, OpenMP, and plotting; Person B
(WSL x86) owns AVX2, MPI, CUDA, and OpenACC.
