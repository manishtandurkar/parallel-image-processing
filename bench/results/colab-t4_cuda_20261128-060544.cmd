# bench/run_gpu.sh colab-t4 pinned  (REPS=10)  GPU: Tesla T4, 580.82.07, 15360 MiB
build/pagp_cuda --gen 512x512 --op gray --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 512x512 --op gray --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 512x512 --op sobel --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 512x512 --op sobel --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 512x512 --op pipeline --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 512x512 --op pipeline --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 1024x1024 --op gray --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 1024x1024 --op gray --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 1024x1024 --op sobel --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 1024x1024 --op sobel --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 1024x1024 --op pipeline --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 1024x1024 --op pipeline --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 2048x2048 --op gray --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 2048x2048 --op gray --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 2048x2048 --op sobel --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 2048x2048 --op sobel --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 2048x2048 --op pipeline --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 2048x2048 --op pipeline --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 4096x4096 --op gray --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 4096x4096 --op gray --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 4096x4096 --op sobel --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 4096x4096 --op sobel --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 4096x4096 --op pipeline --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 4096x4096 --op pipeline --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 7680x4320 --op gray --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 7680x4320 --op gray --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 7680x4320 --op sobel --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 7680x4320 --op sobel --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 7680x4320 --op pipeline --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 7680x4320 --op pipeline --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 16384x16384 --op gray --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 16384x16384 --op gray --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 16384x16384 --op sobel --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 16384x16384 --op sobel --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 16384x16384 --op pipeline --variant naive --block 16x16 --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
build/pagp_cuda --gen 16384x16384 --op pipeline --variant naive --block 16x16 --pinned --reps 10 --validate --csv bench/results/colab-t4_cuda_20261128-060544.csv --machine colab-t4
