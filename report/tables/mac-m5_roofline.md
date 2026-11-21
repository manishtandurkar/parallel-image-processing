Roofline at 7680x4320. Ceilings measured with bench/ceilings.cpp: 1 thread 117 GB/s, 71 GFLOP/s; 10 threads 124 GB/s, 437 GFLOP/s.
Attainable = min(peak, AI x bandwidth) for the backend's thread count; % = attained / attainable. Ridge point (peak / bandwidth): 0.6 ops/byte at 1 thread, 3.5 at 10. Sobel ops are mostly integer (8 x int16 per NEON instruction) measured against a float peak, so simd can exceed 100%.

| op | ops/px | bytes/px | AI | bound at 1 / 10 threads | seq | simd | omp 10t | omp effective GB/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| gray | 6 | 4 | 1.5 | compute / memory | 22.3 G (31%) | 23.6 G (33%) | 134.1 G (72%) | 89 |
| blur2d | 51 | 2 | 25.5 | compute / compute | 11.2 G (16%) | 29.6 G (42%) | 52.6 G (12%) | 2 |
| blur_sep | 21 | 10 | 2.1 | compute / memory | 17.9 G (25%) | 40.1 G (56%) | 84.4 G (32%) | 40 |
| sobel | 19 | 2 | 9.5 | compute / compute | 16.5 G (23%) | 88.5 G (124%) | 80.0 G (18%) | 8 |
| pipeline | 46 | 16 | 2.9 | compute / memory | 19.0 G (27%) | 46.6 G (66%) | 67.4 G (19%) | 23 |
| fused | 46 | 4 | 11.5 | compute / compute | 18.9 G (27%) | 47.5 G (67%) | 80.3 G (18%) | 7 |
