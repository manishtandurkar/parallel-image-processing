# Benchmark machines

Raw times are only comparable within one machine. Speedups are always vs `seq` on the same machine.

## mac-m5 (Person A)

| Item | Value |
|---|---|
| CPU | Apple M5, arm64, 10 cores (4 performance + 6 efficiency) |
| RAM | 16 GB |
| OS | macOS 27.2 (build 26B123) |
| Compiler | Homebrew clang 23.1.5 (`/opt/homebrew/opt/llvm`; 23.1.3 before 2026-11-09), OpenMP via LLVM's bundled libomp |
| MPI | Open MPI 5.0.11 (Homebrew) |
| CMake | 4.4.4 |
| Python | 3.13 (project `.venv`: numpy, pandas, matplotlib, pillow) |
| Flags | `-O3 -ffp-contract=off` for all targets; `-mcpu=native` only on simd/autovec |
| Notes | Run plugged in, other heavy apps closed. Heterogeneous cores: `schedule(static)` is expected to be imbalanced. |

## wsl-x86 (Person B)

TODO: CPU model, core count, RAM, Windows + WSL2 kernel, Ubuntu version, gcc version, Open MPI version.

## colab-t4

| Item | Value |
|---|---|
| GPU | Tesla T4, 15360 MiB, compute capability 7.5 (`nvidia-smi`, 2026-11-09) |
| Driver | 580.82.07 |
| CUDA | nvcc 13.0 (V13.0.88), built for `sm_75` with `--fmad=false` |
| Host compiler | g++ 13.3.0 (Ubuntu 24.04), CMake 3.31.10 |
| nvc++ | TODO (OpenACC not run yet) |
| Notes | Free tier via the Colab CLI. Host buffers are pageable (`std::vector`), so h2d/d2h are not pinned-memory rates. CPU numbers from this VM are never used. |
