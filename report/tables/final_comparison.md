# Final comparison, 4096x4096

Speedup = mac-m5 seq time / backend time, per op, with seq from the same result file as the backend (main CPU sweep seq 4096x4096 pipeline: 43.57 ms). Each backend uses its fastest configuration for pipeline (GPU: by end-to-end time).
† GPU rows ran on colab-t4, a different machine from seq (mac-m5): those speedups are indicative only. GPU speedups are given as kernel-only / end-to-end (end-to-end = host-to-device + compute + device-to-host).
Efficiency = speedup / workers (threads or ranks). Code size: non-blank, non-comment lines of each backend's own source (backend files are separate, so this stands in for lines changed vs seq); all CPU backends also reuse the shared row drivers and per-pixel kernels (173 SLOC: pixel.hpp, rows.hpp, rows.cpp). Directives = #pragma omp/acc lines.

| backend | model | machine | configuration | pipeline ms | speedup | efficiency | geomean speedup (6 ops) | own SLOC | vs seq | driver SLOC | directives |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seq | sequential reference | mac-m5 | naive | 43.57 | 1.00x |  | 1.00x | 107 |  | 2 |  |
| autovec | compiler auto-vectorization | mac-m5 | autovec | 20.75 | 2.10x |  | 1.89x | 116 | 1.1x | 2 |  |
| simd | NEON intrinsics | mac-m5 | neon | 16.73 | 2.61x |  | 2.46x | 156 | 1.5x | 2 |  |
| omp | OpenMP threads | mac-m5 | static, 10 threads | 9.59 | 4.54x | 45% | 4.87x | 160 | 1.5x | 2 | 15 |
| mpi | MPI processes | mac-m5 | overlap, 10 ranks | 17.13 | 2.96x | 30% | 1.75x | 215 | 2.0x | 123 |  |
| cuda | CUDA kernels | colab-t4 | const, block 16x16 | 1.96 kernel / 16.22 e2e | 22.2x / 2.7x † |  | 2.2x e2e † | 415 | 3.9x | 108 |  |
| acc | OpenACC directives (GPU) | colab-t4 | kernels_indep | 2.62 kernel / 17.44 e2e | 16.6x / 2.5x † |  | 1.8x e2e † | 351 | 3.3x | 100 | 22 |

## Speedup per op (same configuration; GPU: end-to-end with kernel-only in brackets)

| backend | gray | blur2d | blur_sep | sobel | pipeline | fused |
|---|---:|---:|---:|---:|---:|---:|
| seq (naive) | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| autovec (autovec) | 0.80x | 2.32x | 1.78x | 2.95x | 2.10x | 2.21x |
| simd (neon) | 1.01x | 2.60x | 2.29x | 5.05x | 2.61x | 2.79x |
| omp (static, 10 threads) | 5.67x | 4.60x | 4.67x | 4.87x | 4.54x | 4.96x |
| mpi (overlap, 10 ranks) | 0.38x | 2.89x | 1.87x | 1.67x | 2.96x | 2.80x |
| cuda (const, block 16x16) | 0.3x (9x) † | 9.2x (54x) † | 2.5x (18x) † | 2.5x (24x) † | 2.7x (22x) † | 2.5x (12x) † |
| acc (kernels_indep) | 0.3x (11x) † | 8.8x (54x) † | 2.3x (13x) † | 2.3x (22x) † | 2.5x (17x) † | 1.1x (2x) † |

Pinned host memory (separate colab-t4 session, bench/results/colab-t4_cuda_20261128-060544.csv): CUDA naive+pinned pipeline 7.96 ms end-to-end vs 17.50 ms pageable in the same session (5.5x vs 2.5x seq †); transfers 5.37 vs 14.73 ms.
