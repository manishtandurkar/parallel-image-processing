# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

**A Comparative Study of Parallel Programming Models for Image Processing on Multicore CPUs and GPUs**
PAGP (CS372IA) Experiential Learning project, RVCE, 7th semester. Team of 2.

We implement three image filters (grayscale, Gaussian blur, Sobel edge detection) in C++ using several
parallel programming models, validate that every version produces the same output as the sequential
reference, and benchmark them to explain *why* each is fast or slow (cache locality, memory bandwidth,
synchronization, load balance, communication and transfer overhead).

Backends:

| Backend   | Model                          | Runs on                          |
|-----------|--------------------------------|----------------------------------|
| `seq`     | Sequential baseline (reference)| Mac, WSL                         |
| `simd`    | NEON (arm64) / AVX2 (x86-64) intrinsics + auto-vectorized scalar | Mac (NEON), WSL (AVX2) |
| `omp`     | OpenMP multithreading          | Mac, WSL                         |
| `mpi`     | MPI distributed (row-block + halo exchange) | Mac, WSL, optionally both together |
| `cuda`    | CUDA kernels                   | Google Colab only                |
| `acc`     | OpenACC (NVIDIA HPC SDK `nvc++`) | Google Colab only (`-acc=gpu` and `-acc=multicore`) |

The point of the project is the **analysis**, not the filters. Every optimization should be measurable and
every claim in the report should be backed by a benchmark or counter.

### Project description (as submitted)

This project develops a parallel image-processing system in C++ that performs grayscale conversion, Gaussian
blur, and Sobel edge detection, and compares a sequential baseline against SIMD vectorization (NEON/AVX2
intrinsics and compiler auto-vectorization), OpenMP multithreading, MPI distributed processing with halo
exchange, OpenACC directive-based parallelism, and CUDA GPU execution on Google Colab. Each implementation
processes images of increasing size up to 8K and beyond, is validated against the sequential output for
correctness, and is evaluated on execution time, speedup, parallel efficiency, throughput, strong and weak
scalability, and communication or data-transfer overhead. The study examines key parallel computing concepts
through measurement rather than theory alone: cache locality using tiled and separable blur implementations,
memory-bound versus compute-bound behavior using roofline analysis, thread synchronization and false sharing,
OpenMP scheduling strategies, task decomposition and load balancing in MPI, and the GPU memory hierarchy through
naive versus shared-memory-tiled CUDA kernels with separate host-device transfer timings. Hardware performance
counters (IPC, cache misses, branch mispredictions) support the analysis where available, and Python scripts
generate performance graphs, comparison tables, and processed output images. The project uses GCC/Clang,
OpenMP, MPI, the NVIDIA HPC SDK, CUDA, and CMake, and demonstrates how different parallel programming models
exploit shared-memory, distributed-memory, and GPU architectures to accelerate data-parallel workloads.

### In simple terms

We take three common photo effects (black-and-white, blur, edge detection) and write the same program several
ways to see which runs fastest and why. An image is millions of pixels that can mostly be processed
independently, so it is a natural fit for parallelism.

- **Sequential:** one pixel after another. The baseline everything is compared against.
- **SIMD:** one core processes several pixels in a single instruction (e.g. 16 at once).
- **OpenMP:** the image is split among all cores of one laptop.
- **MPI:** the image is split across processes (and optionally laptops); they exchange border pixels.
- **OpenACC:** compiler directives on the normal loops; the compiler parallelizes them.
- **CUDA:** thousands of GPU threads process pixels at once (on a free Colab T4).

For each version we measure speedup, scaling as cores/processes grow, and where time is lost (waiting for
memory, threads waiting on each other, network or GPU transfers), and we check every version produces the
same image as the sequential one.

### Objectives

1. Implement the three filters (plus a fused pipeline) in every backend, following one shared spec.
2. Validate every backend against the sequential reference.
3. Benchmark across image sizes, thread/rank counts, and variants with a fixed protocol.
4. Explain the results with measurements: cache behavior, memory bandwidth, synchronization, load balance,
   communication and transfer overhead.
5. Present plots, a comparison table, and processed images in the final report and viva.

### How this maps to the PAGP (CS372IA) syllabus

The goal is to demonstrate the important PAGP concepts, not to cover every unit exhaustively.

| PAGP concept | Unit | Where it shows up in this project |
|---|---|---|
| Performance measurement, speedup, efficiency, Amdahl/Gustafson | I | Every benchmark; `plot.py` derived metrics |
| Pipelining, ILP, branch prediction | I | `perf` counters (IPC, branch misses), loop unrolling effects |
| Shared-memory multiprocessors, caches | II | `omp` backend; cache-tiled vs naive `blur2d`; cache-miss counts |
| Synchronization, false sharing, memory consistency | II | `omp` `false_sharing` demo, reductions vs atomics, schedules |
| Data-level parallelism, SIMD extensions | III | `simd` backend (NEON/AVX2) vs `autovec` |
| Detecting loop-level parallelism | III | Compiler vectorization reports (`PAGP_AUTOVEC_REPORT`) |
| CUDA threads/blocks/grids, SMs and warps | III | `cuda` backend, block-size sweep |
| CUDA memory hierarchy (global, shared, constant) | III | `naive` vs `tiled` vs `const` CUDA variants, `ncu` metrics |
| Decomposition, mapping, load balancing | IV | MPI row-block decomposition; OpenMP schedules on M5 P/E cores |
| Containing interaction overhead | IV | Halo size, fused pipeline vs separate passes |
| Message passing: MPI, collectives, overlap | IV | `Scatterv`/`Gatherv`, blocking vs `overlap` halo exchange |
| OpenACC compute constructs, data environment | V | `acc` backend: `kernels` vs `parallel loop`, data regions |

### Deliverables

- Source code for all backends in this repo, passing `tests/`.
- Benchmark CSVs in `bench/results/` with the commands that produced them.
- Plots and comparison tables in `report/` (speedup, efficiency, scaling, roofline, CUDA transfer breakdown).
- Sample processed images (input, grayscale, blur, Sobel) for the presentation.
- Final report and viva slides explaining each result in terms of the concepts above.

## Hardware and environments

- **Machine A: MacBook (Apple M5, arm64, macOS).** SIMD path = NEON. Heterogeneous P-cores + E-cores:
  this makes OpenMP `schedule(static)` imbalanced. Treat this as a load-balancing finding, not a bug.
- **Machine B: Windows laptop running Ubuntu under WSL2 (x86-64).** SIMD path = AVX2.
  Do all builds inside WSL, never native Windows/MSVC.
- **Google Colab free-tier T4 (Linux x86-64 + NVIDIA GPU), accessed only through the Colab CLI.**
  CUDA and OpenACC only. Colab CPUs are weak (~2 vCPUs),
  so **never use Colab for CPU benchmarks**. Always record the GPU model from `nvidia-smi`.

Record machine specs (CPU model, core counts, RAM, OS, compiler versions) in `bench/machines.md`.
Never compare raw times across different machines without saying so.

## Setup

macOS (Apple clang has no OpenMP, so use Homebrew LLVM):
```sh
brew install llvm libomp open-mpi cmake python
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
```

WSL2 Ubuntu:
```sh
sudo apt update
sudo apt install build-essential cmake ninja-build libopenmpi-dev openmpi-bin python3-pip linux-tools-generic
```

Python (for plots), either machine:
```sh
pip install numpy pandas matplotlib pillow
```

### Colab (GPU work): CLI only

All GPU work runs on a free-tier Colab T4 driven **entirely from the local terminal with the Google Colab CLI**.
Never use the Colab website or browser notebook UI, and never upload files by hand. Code is written and edited
locally; only execution happens on the Colab VM. No Colab subscription is used.

One-time setup (on either laptop; on Windows, inside WSL):
```sh
uv tool install google-colab-cli      # or: pip install google-colab-cli (needs Python 3.13+)
colab sessions                        # first call opens a browser OAuth consent once (token cached in
                                      # ~/.config/colab-cli/token.json); `colab whoami` shows the account
colab version
```

Standard GPU session (named `pagp` so every command targets the same VM):
```sh
colab new --gpu T4 -s pagp                       # provision a free T4 runtime
colab upload -s pagp pagp.tar.gz /content/pagp.tar.gz              # working tree (see colab/README.md)
colab exec -s pagp --env PAGP_STEPS=build -f colab/run_gpu.py      # extract, build, ctest
colab exec -s pagp --env PAGP_STEPS=sweep -f colab/run_gpu.py      # bench/run_gpu.sh colab-t4
colab download -s pagp /content/pagp/bench/results/<file> bench/results/<file>
colab log -s pagp -o colab/logs/<date>.md        # save the run log for the report
colab stop -s pagp                               # ALWAYS release the VM when done (free quota)
```

Rules:
- `colab auth` does NOT log the CLI in: it injects GCP credentials into the VM kernel (and needs a TTY). CLI
  login is the OAuth flow above (or `--auth=adc` with gcloud). Run `colab skill` for the CLI's own guide.
- Before the first CLI use, read the `COLAB_SKILL.md` bundled with the CLI and `colab --help`. If command names or
  flags differ from this file, follow the CLI's own docs and update this section.
- `colab/pagp_gpu.ipynb` is edited locally (Jupyter / VS Code). `colab exec` sends only the notebook, so its first
  cells must get the code onto the VM: `git clone` the repo (preferred), or use the CLI's upload command for files
  not in git (e.g. large test images). Then `!nvidia-smi`, `!nvcc -arch=sm_75 ...`, and run the benchmark binaries.
- Plain scripts also work and are what we use: `colab/run_gpu.py`, with the source uploaded as a tarball so
  untested local changes can run on the GPU before they are committed.
- Interactive debugging stays in the terminal: `colab repl -s pagp` or `colab console -s pagp`. Not the browser.
- Keep each session short: run the sweep, download results, stop. Free-tier sessions can be preempted or
  disconnected, so download results after every benchmark batch rather than at the end.
- If `colab new --gpu T4` fails (no free GPU available), report it and stop; do not retry in a loop.
- CUDA (`nvcc`) is preinstalled on the runtime. The NVIDIA HPC SDK (`nvc++` for OpenACC) is **not**: a notebook
  cell installs it via NVIDIA's apt repo each session (large download, do it once per session, only when `acc` runs).
- Record the GPU model from `nvidia-smi` in every result CSV (`--machine colab-t4`) and in `bench/machines.md`.

## Build

CMake, C++17, out-of-source builds. Each backend is a CMake option so a missing toolchain never breaks the build.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DPAGP_SIMD=ON -DPAGP_OMP=ON -DPAGP_MPI=ON -DPAGP_CUDA=OFF -DPAGP_ACC=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- Release flags: `-O3`, plus `-ffp-contract=off` on every target (no FMA contraction, so float rounding
  in gray/sobel is identical across compilers; GPU builds need the equivalent, e.g. `nvcc --fmad=false`).
  Add `-march=native` on WSL and `-mcpu=native` on Mac **only** in the `simd` and `autovec` targets, so the scalar baseline stays comparable. Document any flag change in the report.
- `PAGP_AUTOVEC_REPORT=ON` adds `-fopt-info-vec-all` (GCC) or `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize`
  (Clang) to show which loops did and did not vectorize. Save these reports; they go in the report.
- SIMD dispatch is compile-time: `#if defined(__aarch64__)` → NEON, `#elif defined(__AVX2__)` → AVX2,
  else scalar fallback. Never mix intrinsics into the shared core.

## Repository layout

```
CMakeLists.txt
CLAUDE.md
include/pagp/
  image.hpp        # Image struct, load/save (wraps stb), test-image generator
  filters.hpp      # filter parameters shared by ALL backends (kernel, weights, border rule)
  validate.hpp     # compare against reference, max-abs-diff, PSNR
  timer.hpp        # steady_clock timer, median/min over repetitions
  cli.hpp          # common command-line parsing
  csv.hpp          # result row writer
  seq.hpp          # seq reference API + Workspace (scratch buffers sized outside the timed region)
  backend.hpp      # Backend table (name, variants per op, run); run_main() is the shared CLI driver
src/
  core/            # image I/O, validation, timing, CSV, CLI, runner.cpp (shared main driver)
  seq/             # reference implementation (the source of truth); backend.cpp = table, main.cpp = 1 line
  simd/            # neon.cpp, avx2.cpp, autovec.cpp
  omp/
  mpi/
  cuda/            # .cu files, built on Colab
  acc/             # OpenACC, built on Colab
third_party/stb/   # stb_image.h, stb_image_write.h (vendored, unmodified)
tests/             # correctness tests: every backend vs seq on small images
data/              # input images (generated ones are gitignored)
bench/
  run_cpu.sh       # benchmark sweep for local machines
  run_mpi.sh
  results/         # CSV output, one file per machine per run
  machines.md
scripts/
  gen_images.py    # synthetic test images at fixed sizes
  plot.py          # speedup, efficiency, scaling, roofline plots
colab/
  pagp_gpu.ipynb   # edited locally, executed remotely with `colab exec`
  run_gpu.py       # optional script version of the same GPU sweep
  logs/            # `colab log` output per session
  README.md        # Colab CLI session commands
report/            # figures and write-up
```

## Filter specification (must be identical in every backend)

All backends must follow this exactly, or validation will fail.

- **Input:** 8-bit RGB (`uint8_t`, interleaved RGB, row-major). Internal processing is single-channel.
- **Grayscale:** `Y = 0.299 R + 0.587 G + 0.114 B` computed in `float`, rounded to nearest, clamped to `[0,255]`.
- **Gaussian blur:** 5×5 kernel, σ = 1.0, weights normalized to sum to 1, stored once in `filters.hpp`.
  Two variants: `blur2d` (direct 5×5) and `blur_sep` (horizontal 1×5 then vertical 5×1 with a float
  intermediate buffer).
- **Sobel:** standard 3×3 `Gx`, `Gy`; output `min(255, round(sqrt(Gx² + Gy²)))`.
- **Borders:** clamp-to-edge for every filter in every backend (MPI halos must reproduce this at image edges).
- **Pipeline op:** `gray → blur_sep → sobel`. Also implement a `fused` variant where possible to compare
  memory traffic against three separate passes.
- **Tolerance:** `gray` and `sobel` on identical input must match exactly. `blur` may differ by at most 1
  per pixel (float summation order differs across SIMD/GPU). Tests enforce `max_abs_diff <= 1`.

## Common CLI (every backend binary)

Binaries: `pagp_seq`, `pagp_simd`, `pagp_autovec`, `pagp_omp`, `pagp_mpi`, `pagp_cuda`, `pagp_acc`.

```
--input <path> | --gen <WxH>   # load an image or generate a synthetic one
--op gray|blur2d|blur_sep|sobel|pipeline|fused
--variant <name>               # backend-specific, e.g. omp: static|dynamic|guided, cuda: naive|tiled
--reps N                       # default 10, plus 2 untimed warm-up runs
--threads N                    # omp
--validate                     # compare with seq reference and record max_abs_diff
--save <path>                  # write output image
--csv <path>                   # append one result row
--machine <tag>                # e.g. mac-m5, wsl-x86, colab-t4
```

## Benchmark protocol

- Image sizes: 512², 1024², 2048², 4096², 7680×4320 (8K), 16384². Generated by `scripts/gen_images.py`.
- Two warm-up runs, then `--reps` timed runs. Report **median** and min. Never report a single run.
- Time only the compute region. CUDA reports `h2d_ms`, `kernel_ms`, `d2h_ms` separately using `cudaEvent_t`.
  MPI reports `compute_ms` and `comm_ms` separately (wrap communication calls with `MPI_Wtime`).
- Close other heavy apps. On the Mac, plug in power (thermal and power throttling skew results).
- CSV schema (header written once per file):
  ```
  machine,backend,variant,op,width,height,workers,reps,median_ms,min_ms,h2d_ms,d2h_ms,comm_ms,max_abs_diff
  ```
  `workers` = threads (omp), ranks (mpi), block size as `16x16` (cuda), 1 otherwise. Unused columns = empty.
- Derived metrics (computed in `plot.py`, not in C++): speedup vs `seq` on the same machine, efficiency,
  throughput in megapixels/s, strong and weak scaling.

## Backend design notes

**seq.** Plain, readable loops. This is the correctness reference; do not optimize it beyond `-O3`.
A separate `naive` vs `tiled` cache-blocking variant for `blur2d` lives here for the cache-locality study.

**simd.** One file per ISA. Process 16 pixels per iteration where the ISA allows (NEON `uint8x16_t`,
AVX2 `__m256i` widened to 16-bit/32-bit as needed). Handle the remainder with scalar code.
`autovec.cpp` is the same scalar code written to be vectorizer-friendly (`__restrict`, no aliasing,
contiguous inner loop) so we can compare hand-written vs compiler-generated SIMD.

**omp.** `parallel for` over rows. Variants: `static`, `dynamic`, `guided` schedules, and a deliberate
`false_sharing` demo (per-thread counters in a contiguous array vs padded to 128 bytes, plus
`padded64`; Apple M-series cache lines are 128 B, x86 lines are 64 B). The counters are relaxed atomics:
with plain volatile stores the M5 showed no false-sharing effect (hypothesis: store-buffer merging on
weakly ordered arm64; unverified, compare on WSL x86 before claiming it). Also `reduction` and `atomic`.
Thread sweep: 1, 2, 4, ... up to the core count. On the Mac, also compare P-core-only counts.

**mpi.** Row-block decomposition with `MPI_Scatterv` / `MPI_Gatherv`. Each rank holds its rows plus
`radius` halo rows on each side (2 for 5×5 blur, 1 for Sobel). Two halo-exchange variants:
`blocking` (`MPI_Sendrecv`) and `overlap` (`MPI_Isend`/`MPI_Irecv`, compute interior rows, `MPI_Waitall`,
then compute boundary rows). Ranks at the image edge apply clamp-to-edge instead of receiving a halo.
Run with `mpirun -np N` on one machine first. A two-machine run (Mac arm64 + WSL x86) is an optional
stretch goal: it needs identical Open MPI versions, SSH between machines, and WSL2 mirrored networking,
and heterogeneous-architecture MPI may simply not work. If it fails, document the attempt and why.

**cuda.** One thread per output pixel, 2D blocks (default 16×16, sweep 8×8/16×16/32×8/32×32).
Variants: `naive` (global memory only), `tiled` (shared-memory tile with halo, `__syncthreads()`),
`const` (Gaussian weights in `__constant__` memory), and `fused` (all three stages in one kernel).
Check every CUDA call with a `CUDA_CHECK` macro. Compile with `-arch=sm_XX` matching the Colab GPU.

**acc.** Same loops as `seq` with directives. Variants: `kernels` vs `parallel loop`, and explicit
`data copyin/copyout` regions vs letting the compiler manage data. Build twice: `-acc=gpu` and `-acc=multicore`.
Keep `-Minfo=accel` output for the report.

## Hardware counters

- WSL2: try `perf stat -e cycles,instructions,cache-misses,branch-misses`. Hardware counters are often
  unavailable under WSL2; if so, record that and fall back to timing-only analysis plus `valgrind --tool=cachegrind`
  for cache-miss estimates.
- macOS: no `perf`. Use Instruments (CPU Counters template) or skip counters on the Mac.
- Colab GPU: `ncu` (Nsight Compute CLI) for occupancy, memory throughput, shared-memory bank conflicts,
  if it is available in the session.

## Coding conventions

- C++17, no exceptions across backend boundaries; return status codes or abort with a clear message.
- Images: `struct Image { int w, h, c; std::vector<uint8_t> data; }`. Single-channel intermediates use
  `std::vector<float>` or `uint8_t` as the spec above says.
- No global mutable state. No `using namespace std;` in headers.
- `clang-format` with the repo's `.clang-format` (LLVM style, 100 columns).
- Every backend must pass `tests/` before its numbers go into `bench/results/`.
- Commit benchmark CSVs and the exact command that produced them. Do not commit generated images
  larger than 1 MB.

## Work split (2 people)

- **Person A (Mac M5):** core library, `seq`, NEON `simd`, `omp`, plotting scripts.
- **Person B (WSL x86):** AVX2 `simd`, `mpi`, `cuda` and `acc` (run via the Colab CLI).
- Both run the full CPU benchmark sweep on their own machine, which also gives an ARM vs x86 comparison.

## Milestones

1. Core library, `seq`, tests, image generator. All filters validated on small images.
2. `simd` (NEON + AVX2 + autovec) and `omp`, with benchmark sweep and first plots.
3. `mpi` with blocking and overlap halo exchange.
4. Colab: `cuda` variants and `acc`, with transfer-time breakdown.
5. Analysis: speedup/efficiency/scaling plots, roofline (memory-bound `gray` vs compute-heavier `blur`/`sobel`),
   counter data, final comparison table (time, speedup, efficiency, lines of code changed vs `seq`).

## Things to avoid

- Don't benchmark CPU backends on Colab.
- Don't use the Colab website or browser UI; everything goes through the Colab CLI.
- Don't leave a Colab session running: always `colab stop -s pagp` at the end.
- Don't change the filter spec in one backend only; change `filters.hpp` and every backend together.
- Don't report GPU speedup without including transfer time alongside kernel-only time.
- Don't claim a concept (e.g. memory consistency, ILP) in the report unless a measurement in `bench/results/`
  supports it.
