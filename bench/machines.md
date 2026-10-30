# Benchmark machines

Raw times are only comparable within one machine. Speedups are always vs `seq` on the same machine.

## mac-m5 (Person A)

| Item | Value |
|---|---|
| CPU | Apple M5, arm64, 10 cores (4 performance + 6 efficiency) |
| RAM | 16 GB |
| OS | macOS 27.2 (build 26B123) |
| Compiler | Homebrew clang 23.1.3 (`/opt/homebrew/opt/llvm`), OpenMP via LLVM's bundled libomp |
| MPI | Open MPI 5.0.11 (Homebrew) |
| CMake | 4.4.4 |
| Python | 3.13 (project `.venv`: numpy, pandas, matplotlib, pillow) |
| Flags | `-O3 -ffp-contract=off` for all targets; `-mcpu=native` only on simd/autovec |
| Notes | Run plugged in, other heavy apps closed. Heterogeneous cores: `schedule(static)` is expected to be imbalanced. |

## wsl-x86 (Person B)

TODO: CPU model, core count, RAM, Windows + WSL2 kernel, Ubuntu version, gcc version, Open MPI version.

## colab-t4

TODO: GPU from `nvidia-smi`, driver, CUDA version, nvc++ version.
