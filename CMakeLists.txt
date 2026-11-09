cmake_minimum_required(VERSION 3.20)
project(pagp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(PAGP_SIMD "Build the simd and autovec backends" ON)
option(PAGP_OMP "Build the OpenMP backend" ON)
option(PAGP_MPI "Build the MPI backend" ON)
option(PAGP_CUDA "Build the CUDA backend (Colab)" OFF)
option(PAGP_ACC "Build the OpenACC backend (Colab, nvc++)" OFF)
option(PAGP_AUTOVEC_REPORT "Emit compiler vectorization reports for autovec" OFF)

# No FMA contraction anywhere: gray and sobel must match seq bit-for-bit, and contraction
# would make the rounding of float expressions depend on the compiler and target.
# (CXX only: nvcc gets the equivalent --fmad=false on the CUDA target.)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-ffp-contract=off>")

# Core library: image I/O, CLI, timing, CSV, validation, shared driver, and the seq reference
# (every binary links seq so --validate can compare against it).
add_library(pagp_core STATIC
  src/core/image.cpp
  src/core/stb_impl.cpp
  src/core/validate.cpp
  src/core/cli.cpp
  src/core/csv.cpp
  src/core/runner.cpp
  src/core/rows.cpp
  src/seq/seq.cpp
  src/seq/backend.cpp)
target_include_directories(pagp_core PUBLIC include)
target_include_directories(pagp_core SYSTEM PRIVATE third_party/stb)

add_executable(pagp_seq src/seq/main.cpp)
target_link_libraries(pagp_seq PRIVATE pagp_core)

enable_testing()
add_executable(test_seq tests/test_seq.cpp)
target_link_libraries(test_seq PRIVATE pagp_core)
add_test(NAME seq_unit COMMAND test_seq)
add_test(NAME seq_cli_smoke
  COMMAND pagp_seq --gen 97x61 --op pipeline --reps 1 --validate)

# Backends. Each is a static library (so tests can link it) plus a thin executable.
function(pagp_backend name)
  cmake_parse_arguments(B "" "MAIN" "SOURCES" ${ARGN})
  add_library(pagp_${name}_lib STATIC ${B_SOURCES})
  target_link_libraries(pagp_${name}_lib PUBLIC pagp_core)
  add_executable(pagp_${name} ${B_MAIN})
  target_link_libraries(pagp_${name} PRIVATE pagp_${name}_lib)
  add_executable(test_${name} tests/test_backend.cpp)
  target_compile_definitions(test_${name} PRIVATE PAGP_TEST_BACKEND=${name}_backend)
  target_link_libraries(test_${name} PRIVATE pagp_${name}_lib)
endfunction()

# -mcpu=native / -march=native only on the simd and autovec targets, so seq stays comparable.
# -fno-math-errno lets sqrtf vectorize (already the default on Darwin, not on Linux/GCC).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(PAGP_NATIVE_FLAGS -mcpu=native -fno-math-errno)
else()
  set(PAGP_NATIVE_FLAGS -march=native -fno-math-errno)
endif()

if(PAGP_SIMD)
  pagp_backend(autovec MAIN src/simd/main_autovec.cpp SOURCES src/simd/autovec.cpp)
  target_compile_options(pagp_autovec_lib PRIVATE ${PAGP_NATIVE_FLAGS})
  if(PAGP_AUTOVEC_REPORT)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(pagp_autovec_lib PRIVATE
        -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
    else()
      target_compile_options(pagp_autovec_lib PRIVATE -fopt-info-vec-all)
    endif()
  endif()
  add_test(NAME autovec_vs_seq COMMAND test_autovec)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    pagp_backend(simd MAIN src/simd/main_simd.cpp SOURCES src/simd/neon.cpp)
    target_compile_options(pagp_simd_lib PRIVATE ${PAGP_NATIVE_FLAGS})
    add_test(NAME simd_vs_seq COMMAND test_simd)
  else()
    message(STATUS "pagp_simd: AVX2 kernels (src/simd/avx2.cpp) not written yet; skipping")
  endif()
endif()

if(PAGP_OMP)
  find_package(OpenMP REQUIRED)
  pagp_backend(omp MAIN src/omp/main.cpp SOURCES src/omp/omp.cpp)
  target_link_libraries(pagp_omp_lib PUBLIC OpenMP::OpenMP_CXX)
  add_test(NAME omp_vs_seq COMMAND test_omp 1 2 3 10)
endif()

if(PAGP_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  # Own driver (src/mpi/main.cpp) instead of run_main: timing and I/O are collective.
  add_library(pagp_mpi_lib STATIC src/mpi/mpi.cpp)
  target_include_directories(pagp_mpi_lib PUBLIC src/mpi)
  target_link_libraries(pagp_mpi_lib PUBLIC pagp_core MPI::MPI_CXX)
  add_executable(pagp_mpi src/mpi/main.cpp)
  target_link_libraries(pagp_mpi PRIVATE pagp_mpi_lib)
  add_executable(test_mpi tests/test_mpi.cpp)
  target_link_libraries(test_mpi PRIVATE pagp_mpi_lib)
  foreach(np 1 3 4)
    add_test(NAME mpi_vs_seq_np${np}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:test_mpi> ${MPIEXEC_POSTFLAGS})
  endforeach()
  add_test(NAME mpi_cli_smoke
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:pagp_mpi> --gen 97x61 --op pipeline --variant overlap --reps 1 --validate)
endif()

if(PAGP_CUDA)
  # Colab T4 = sm_75; override with -DCMAKE_CUDA_ARCHITECTURES=<nn> for another GPU.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  add_library(pagp_cuda_lib STATIC src/cuda/cuda.cu)
  # --fmad=false: no fused multiply-add, so float results match seq bit-for-bit.
  target_compile_options(pagp_cuda_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -Xcompiler=-ffp-contract=off>)
  target_include_directories(pagp_cuda_lib PUBLIC src/cuda)
  target_link_libraries(pagp_cuda_lib PUBLIC pagp_core CUDA::cudart)
  add_executable(pagp_cuda src/cuda/main.cpp)
  target_link_libraries(pagp_cuda PRIVATE pagp_cuda_lib)
  add_executable(test_cuda tests/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE pagp_cuda_lib)
  add_test(NAME cuda_vs_seq COMMAND test_cuda)
  add_test(NAME cuda_cli_smoke
    COMMAND pagp_cuda --gen 97x61 --op fused --block 8x8 --reps 1 --validate)
endif()
