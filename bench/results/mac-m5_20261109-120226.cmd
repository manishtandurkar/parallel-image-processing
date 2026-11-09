# bench/run_cpu.sh mac-m5 compare scaling sync weak  (REPS=10 THREADS='1 2 4 6 8 10')
build/pagp_seq --gen 512x512 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 512x512 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 512x512 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 512x512 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 512x512 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 512x512 --op blur2d --variant tiled --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 512x512 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 512x512 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 512x512 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 512x512 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 512x512 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 512x512 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 512x512 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 512x512 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 512x512 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 512x512 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 512x512 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 512x512 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 512x512 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 512x512 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 512x512 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 512x512 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 512x512 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 512x512 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 512x512 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 1024x1024 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 1024x1024 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 1024x1024 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op blur2d --variant tiled --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 1024x1024 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 1024x1024 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 1024x1024 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 1024x1024 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 1024x1024 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 1024x1024 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 1024x1024 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 1024x1024 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 1024x1024 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 1024x1024 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 1024x1024 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 1024x1024 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 1024x1024 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 1024x1024 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 1024x1024 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 2048x2048 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 2048x2048 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op blur2d --variant tiled --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 2048x2048 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 2048x2048 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 2048x2048 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 2048x2048 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 2048x2048 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 2048x2048 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 2048x2048 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 2048x2048 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 2048x2048 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 2048x2048 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 4096x4096 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 4096x4096 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op blur2d --variant tiled --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 4096x4096 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 4096x4096 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 4096x4096 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 4096x4096 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 4096x4096 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 4096x4096 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 4096x4096 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 4096x4096 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 4096x4096 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 4096x4096 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 7680x4320 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 7680x4320 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur2d --variant tiled --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 7680x4320 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 7680x4320 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 7680x4320 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 7680x4320 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 7680x4320 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 7680x4320 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 7680x4320 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 7680x4320 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 7680x4320 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 7680x4320 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 16384x16384 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 16384x16384 --op gray --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 16384x16384 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op blur2d --variant tiled --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 16384x16384 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 16384x16384 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 16384x16384 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 16384x16384 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 16384x16384 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 16384x16384 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 16384x16384 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 16384x16384 --op sobel --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 16384x16384 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 16384x16384 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 16384x16384 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 16384x16384 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 16384x16384 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_autovec --gen 16384x16384 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_simd --gen 16384x16384 --op fused --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 16384x16384 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op gray --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur2d --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op blur_sep --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op sobel --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op pipeline --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 2048x2048 --op fused --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op gray --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur2d --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op blur_sep --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op sobel --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op pipeline --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant dynamic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant dynamic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant dynamic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant dynamic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant dynamic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant dynamic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant guided --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant guided --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant guided --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant guided --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant guided --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 7680x4320 --op fused --variant guided --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant false_sharing --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant false_sharing --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant false_sharing --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant false_sharing --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant false_sharing --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant false_sharing --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded64 --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded64 --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded64 --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded64 --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded64 --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant padded64 --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant reduction --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant reduction --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant reduction --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant reduction --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant reduction --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant reduction --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant atomic --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant atomic --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant atomic --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant atomic --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant atomic --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant atomic --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op gray --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x1024 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x1024 --op pipeline --variant static --threads 1 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x2048 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x2048 --op pipeline --variant static --threads 2 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x4096 --op pipeline --variant static --threads 4 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x6144 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x6144 --op pipeline --variant static --threads 6 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x8192 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x8192 --op pipeline --variant static --threads 8 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_seq --gen 4096x10240 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
build/pagp_omp --gen 4096x10240 --op pipeline --variant static --threads 10 --reps 10 --validate --csv bench/results/mac-m5_20261109-120226.csv --machine mac-m5
