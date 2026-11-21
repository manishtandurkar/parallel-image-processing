# GPU job, executed on the Colab VM with `colab exec -s pagp -f colab/run_gpu.py`.
# Expects the source tree uploaded as /content/pagp.tar.gz (see colab/README.md). Steps are chosen
# with the PAGP_STEPS env var (`colab exec --env PAGP_STEPS=...`), space separated:
#   build   extract, configure (CUDA only), build, ctest
#   sweep   bench/run_gpu.sh colab-t4  -> bench/results/colab-t4_cuda_<date>.csv
#   acc-install  NVIDIA HPC SDK (nvc++) from NVIDIA's apt repo; skipped if already installed
#   acc-build    configure build-acc with nvc++, build GPU + multicore, ctest; -Minfo=accel output
#                -> report/acc/minfo_colab-t4.txt
#   acc-sweep    bench/run_acc.sh colab-t4  -> bench/results/colab-t4_acc_<date>.csv
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

# Newest installed HPC SDK compilers directory (empty if none).
NVHPC_BIN = ("$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/[0-9]*/compilers/bin 2>/dev/null "
             "| sort -V | tail -1)")
ACC_ENV = f'export PATH="{NVHPC_BIN}:$PATH"; '

if "acc-install" in STEPS:
    sh(f'if [ -n "{NVHPC_BIN}" ]; then echo "HPC SDK already installed: {NVHPC_BIN}"; else '
       "curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK"
       " | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg && "
       "echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] "
       "https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' "
       "> /etc/apt/sources.list.d/nvhpc.list && apt-get update -qq && "
       "PKG=$(apt-cache pkgnames nvhpc- | grep -E '^nvhpc-[0-9]+-[0-9]+$' | sort -V | tail -1) && "
       'echo "installing $PKG" && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$PKG" '
       "> /dev/null; fi", cwd="/content")
    sh(ACC_ENV + "nvc++ --version | head -3; nvaccelinfo | grep -E 'Device Name|CUDA Driver|Default "
       "Target' | head -3", cwd="/content")

if "acc-build" in STEPS:
    sh(ACC_ENV + "CXX=nvc++ cmake -S . -B build-acc -DCMAKE_BUILD_TYPE=Release -DPAGP_ACC=ON "
       "-DPAGP_SIMD=OFF -DPAGP_OMP=OFF -DPAGP_MPI=OFF -DPAGP_CUDA=OFF > /dev/null")
    # Compiler feedback for the report: what was parallelized, how, and the implicit data moves.
    sh(ACC_ENV + "mkdir -p report/acc && cmake --build build-acc --target pagp_core > /dev/null && "
       "touch src/acc/acc.cpp && "
       "cmake --build build-acc -j1 --target pagp_acc_lib pagp_acc_mc_lib 2>&1 "
       "| tee report/acc/minfo_colab-t4.txt")
    sh(ACC_ENV + "cmake --build build-acc -j 2>&1 | grep -E 'error|Built target' ; "
       "test ${PIPESTATUS[0]} -eq 0")
    sh(ACC_ENV + "ctest --test-dir build-acc --output-on-failure")

if "acc-sweep" in STEPS:
    sh(ACC_ENV + "bench/run_acc.sh colab-t4")
    sh("ls -la bench/results/")

if "sweep" in STEPS:
    sh("bench/run_gpu.sh colab-t4")
    sh("ls -la bench/results/")
