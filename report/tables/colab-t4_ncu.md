# Nsight Compute: CUDA stencil variants, Tesla T4, 4096x4096

Source: report/cuda/ncu/*.csv from bench/run_ncu.sh (one profiled run after 2 warm-ups; ncu replays each kernel, so times are slightly above the benchmark medians). blur_sep = blur_h + blur_v kernels: sums, percentages time-weighted. barrier_stall = smsp__average_warp_latency_issue_stalled_barrier.ratio (warp latency spent waiting at __syncthreads; 0 when the kernel has none).

| op | variant | time_us | dram_read_MB | dram_write_MB | dram_%peak | L1_hit_% | L2_hit_% | inst_M | occupancy_% | sm_%peak | smem_bank_conflicts | barrier_stall | regs | kernels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur2d | naive | 3.54e+03 | 76.1 | 19.8 | 8.46 | 92.8 | 77.4 | 133 | 93.7 | 65.3 | 0 | 0 | 58 | k_blur2d |
| blur2d | const | 2.45e+03 | 75.6 | 19.7 | 11.8 | 90.4 | 78.3 | 126 | 83.6 | 53.7 | 0 | 0 | 38 | k_blur2d |
| blur2d | tiled | 2.19e+03 | 70.7 | 19.6 | 12.9 | 15.9 | 76.6 | 135 | 93.2 | 64.5 | 0 | 348 | 36 | k_blur2d_tiled |
| blur_sep | naive | 1.87e+03 | 162 | 99.9 | 43.6 | 69.7 | 60.5 | 89.1 | 81.4 | 60.6 | 0 | 0 | 20/24 | k_blur_h + k_blur_v |
| blur_sep | const | 1.62e+03 | 150 | 98.2 | 48.4 | 62 | 60.2 | 82.8 | 81.1 | 53.8 | 0 | 0 | 17/22 | k_blur_h + k_blur_v |
| blur_sep | tiled | 3.01e+03 | 173 | 99 | 28.5 | 5.8 | 59.7 | 179 | 91.4 | 62.4 | 0 | 447 | 38/38 | k_blur_h_tiled + k_blur_v_tiled |
| sobel | naive | 1.04e+03 | 63 | 19.2 | 24 | 75.3 | 75.4 | 62.5 | 83.2 | 62.4 | 0 | 0 | 30 | k_sobel |
| sobel | tiled | 1.59e+03 | 66.7 | 19.5 | 16.9 | 15.5 | 74.8 | 98.2 | 91.1 | 64.2 | 0 | 370 | 36 | k_sobel_tiled |
