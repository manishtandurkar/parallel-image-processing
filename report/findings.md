# Findings from the final Mac-side analysis (2026-11-28)

Each claim below is backed by a file in `bench/results/` or `report/`.

## 1. MPI anomaly: 8K pipeline at 4 ranks is a memory-layout cliff, not communication

Symptom (`bench/results/mac-m5_mpi_20261121-193154.csv`): pipeline 7680x4320 takes 65 ms at
np=4 with comm_ms = 40 ms, slower than np=3 (44 ms) and np=5 (31 ms). Both variants, both the
shared-memory and TCP transports, not at 4096², not for `fused`.

Diagnosis (temporary per-rank, per-phase timers in `src/mpi/mpi.cpp`, not committed):

| np | rank | gray | blur_sep | sobel |
|---|---|---|---|---|
| 4 | 0 | 2.3 ms | 11.8 ms | 47.7 ms |
| 4 | 1-3 | 2.3-2.4 ms | 11.7-12.2 ms | 10.8-11.1 ms |
| 3 | 0-2 | 3.0 ms | 13.7 ms | 24.9-25.1 ms |

- Only Sobel is slow, and only on some ranks. The time is compute, not messaging: the other
  ranks finish and wait in `MPI_Gatherv`, and because comm_ms is the slowest rank's time inside
  MPI calls, that wait shows up as "communication". This is load imbalance from one slow rank,
  reported as comm time.
- Running the same Sobel again on the same buffers is just as slow, but with a freshly allocated
  output buffer the same rows take 11-13 ms on every rank. So it depends on where the buffers sit
  in memory, not on the data.
- `bench/alias_sobel.cpp` (`build/pagp_alias_sobel`) isolates it: one rank's block (7680 x 1080),
  seq Sobel row kernel, output buffer fixed, input placed at a chosen distance. Result in
  `bench/results/alias/mac-m5_20261128-113317.csv`:
  - normal: 8.7 ms at nearly every distance (1000, 1024, 1086, 1090, 1152 rows ...)
  - slow: 39-49 ms when the distance is 544, 1088, 2176, 3264 or 4352 rows, and also at +-1 row
    from those (Sobel reads input rows y-1, y, y+1 while writing output row y).
  - 544 rows x 7680 B = 4,177,920 B = 255 x 16 KiB pages. Shifting the distance by only 64 B or
    4 KiB removes the slowdown completely.
  - In the np=4 run, rank 0's Sobel input (`b`) and output (`out`) buffers happened to be exactly
    1088 rows apart; at np=3 the distance (1640.5 rows) gives a milder slowdown (16-25 ms).
- Interpretation (hypothesis, no hardware counters on macOS): a load from an input row whose
  address matches a just-written output row under some hashed address comparison (the 255-page
  period points to a hash that folds page-number bits) is treated as possibly dependent on that
  store and is delayed or replayed. This is the M5 counterpart of x86 "4K aliasing". Verifying it
  needs Instruments CPU counters or perf on another arm64 machine.
- Consequence for the report: the np=4 point on the 8K MPI pipeline curve is an outlier caused by
  buffer placement; it is kept in the data and explained, not removed. Lesson: "comm time" taken
  as wall time inside MPI calls includes waiting for the slowest rank.

## 2. Why the CUDA tiled kernels are slower than naive (Nsight Compute)

Source: `report/cuda/ncu/*.csv` (`bench/run_ncu.sh`), summary `report/tables/colab-t4_ncu.md`,
Tesla T4, 4096², 16x16 blocks.

| kernel | time | DRAM read | L1 hit | instructions | barrier stall |
|---|---|---|---|---|---|
| sobel naive | 1.04 ms | 63 MB | 75% | 62.5 M | 0 |
| sobel tiled | 1.59 ms | 67 MB | 16% | 98.2 M | 370 |
| blur_sep naive | 1.87 ms | 162 MB | 70% | 89.1 M | 0 |
| blur_sep tiled | 3.01 ms | 173 MB | 6% | 179 M | 447 |
| blur2d naive | 3.54 ms | 76 MB | 93% | 133 M | 0 |
| blur2d const | 2.45 ms | 76 MB | 90% | 126 M | 0 |
| blur2d tiled | 2.19 ms | 71 MB | 16% | 135 M | 348 |

- Tiling is meant to cut global-memory traffic, but DRAM bytes are the same for naive and tiled:
  in the naive kernels, neighbouring threads' overlapping reads already hit in L1 (70-93% hit
  rate) and L2, so there is no traffic left to save.
- Tiling instead adds work: the cooperative tile + halo load with clamped indices raises executed
  instructions by 1.6x (sobel) to 2x (blur_sep), and warps now stall at `__syncthreads()`.
  There are no shared-memory bank conflicts (0 in every kernel), so that is not the cause.
- blur2d is the exception: 25 taps per pixel give more reuse per loaded pixel, so tiled (2.19 ms)
  beats naive (3.54 ms). Most of that gain is the weights: naive reads them from a global array
  (58 registers/thread), `const` puts them in `__constant__` memory (2.45 ms), and tiled adds a
  further ~10% on top of const.
- Lesson: on GPUs with L1/L2 caches for global loads (Volta/Turing and later), shared-memory
  tiling pays off only when reuse per pixel is high; for small stencils the hardware cache
  already does the job.

## 3. Pinned host memory cuts transfer time 2.6-3x

Source: `bench/results/colab-t4_cuda_20261128-060544.csv` (`bench/run_gpu.sh colab-t4 pinned`,
`--pinned` flag: `cudaMallocHost` buffers), figure `report/figs/colab-t4/gpu_pinned_bandwidth.png`.

- Effective bandwidth for large images: pageable h2d ~4.6 GB/s, d2h ~4.4 GB/s; pinned h2d
  ~12.4 GB/s, d2h ~13.1 GB/s (close to what PCIe 3.0 x16 delivers in practice). Pageable memory is
  copied through a driver staging buffer first; pinned memory is transferred directly by DMA.
- Pipeline end-to-end: 4096² 17.5 -> 8.0 ms, 7680x4320 35.0 -> 16.7 ms, 16384² 266 -> 114 ms.
  Kernel times are unchanged.
- Even pinned, transfers are still 2-4x the pipeline kernel time, so the conclusion stands that a
  single filter pass is transfer-bound on the GPU. Pinned memory narrows the gap; keeping data
  resident on the GPU across stages (the fused kernel, OpenACC data regions) removes it.

## 4. Validation tolerance for pipeline and fused

The spec fixes tolerance only for the single filters (gray/sobel exact, blur <= 1). Pipeline and
fused keep a tolerance of 1 in `src/core/runner.cpp` (a +-1 blur difference could in principle
grow through Sobel), but every backend measured max_abs_diff = 0 on every op and size in every
result CSV, so in practice all backends are bit-exact with seq.
