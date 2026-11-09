# GPU job, executed on the Colab VM with `colab exec -s pagp -f colab/run_gpu.py`.
# Expects the source tree uploaded as /content/pagp.tar.gz (see colab/README.md). Steps are chosen
# with the PAGP_STEPS env var (`colab exec --env PAGP_STEPS=...`), space separated:
#   build   extract, configure (CUDA only), build, ctest
#   sweep   bench/run_gpu.sh colab-t4  -> bench/results/colab-t4_cuda_<date>.csv
# Kernel state persists between exec calls, but every step re-derives what it needs from disk.
import os
import subprocess

SRC = "/content/pagp"
STEPS = os.environ.get("PAGP_STEPS", "build").split()


def sh(cmd, cwd=SRC):
    print(f"$ {cmd}", flush=True)
    # Stream line by line: in a Jupyter kernel sys.stdout has no file descriptor to hand over.
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", cwd=cwd, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise SystemExit(f"command failed ({p.returncode}): {cmd}")


if "build" in STEPS:
    sh(f"rm -rf {SRC} && mkdir -p {SRC} && tar -xzf /content/pagp.tar.gz -C {SRC}", cwd="/content")
    sh("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv")
    sh("nvcc --version | tail -2; cmake --version | head -1; g++ --version | head -1")
    sh("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPAGP_CUDA=ON "
       "-DPAGP_SIMD=OFF -DPAGP_OMP=OFF -DPAGP_MPI=OFF > /dev/null")
    sh("cmake --build build -j 2>&1 | grep -E 'error|warning|Built target' ; "
       "test ${PIPESTATUS[0]} -eq 0")
    sh("ctest --test-dir build --output-on-failure")

if "sweep" in STEPS:
    sh("bench/run_gpu.sh colab-t4")
    sh("ls -la bench/results/")
