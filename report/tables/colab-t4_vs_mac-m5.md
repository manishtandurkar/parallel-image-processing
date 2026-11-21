Median time in ms at 4096x4096. seq/omp: mac-m5; cuda/acc: colab-t4. Different machines: GPU speedup is only indicative.
cuda = naive 16x16 (fused: tiled), acc = parallel. Speedup vs seq in brackets.

| op | seq | omp 10t | cuda kernel | cuda end-to-end | acc compute | acc end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| gray | 4.53 (1.0x) | 0.80 (5.7x) | 0.51 (8.9x) | 14.85 (0.3x) | 0.42 (10.8x) | 15.36 (0.3x) |
| blur2d | 76.60 (1.0x) | 16.67 (4.6x) | 3.02 (25.3x) | 11.09 (6.9x) | 1.56 (49.1x) | 8.87 (8.6x) |
| blur_sep | 20.52 (1.0x) | 4.40 (4.7x) | 1.23 (16.6x) | 8.27 (2.5x) | 1.92 (10.7x) | 9.26 (2.2x) |
| sobel | 19.57 (1.0x) | 4.02 (4.9x) | 0.91 (21.6x) | 7.93 (2.5x) | 0.94 (20.8x) | 9.15 (2.1x) |
| pipeline | 43.57 (1.0x) | 9.59 (4.5x) | 2.10 (20.8x) | 16.38 (2.7x) | 2.77 (15.7x) | 17.55 (2.5x) |
| fused | 45.28 (1.0x) | 9.13 (5.0x) | 3.91 (11.6x) | 18.53 (2.4x) | 24.36 (1.9x) | 39.18 (1.2x) |
