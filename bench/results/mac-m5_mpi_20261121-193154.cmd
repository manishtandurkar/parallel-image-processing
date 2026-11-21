# bench/run_mpi.sh mac-m5 compare scaling weak  (REPS=10 RANKS='1 2 4 6 8 10')
build/pagp_seq --gen 1024x1024 --op gray --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op sobel --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 1024x1024 --op fused --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 1024x1024 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op gray --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op sobel --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 2048x2048 --op fused --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 2048x2048 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op gray --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op sobel --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op fused --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x4096 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op gray --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op sobel --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op fused --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op gray --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op gray --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur2d --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur2d --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur2d --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op blur_sep --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op blur_sep --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op sobel --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op sobel --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op sobel --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 7680x4320 --op fused --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op fused --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 7680x4320 --op fused --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x1024 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 4096x1024 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 1 build/pagp_mpi --gen 4096x1024 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x2048 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 4096x2048 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 2 build/pagp_mpi --gen 4096x2048 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x4096 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 4096x4096 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 4 build/pagp_mpi --gen 4096x4096 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x6144 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 4096x6144 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 6 build/pagp_mpi --gen 4096x6144 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x8192 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 4096x8192 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 8 build/pagp_mpi --gen 4096x8192 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
build/pagp_seq --gen 4096x10240 --op pipeline --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x10240 --op pipeline --variant blocking --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
mpirun -np 10 build/pagp_mpi --gen 4096x10240 --op pipeline --variant overlap --reps 10 --validate --csv bench/results/mac-m5_mpi_20261121-193154.csv --machine mac-m5
