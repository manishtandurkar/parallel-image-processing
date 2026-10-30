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
add_compile_options(-Wall -Wextra -ffp-contract=off)

# Core library: image I/O, CLI, timing, CSV, validation, shared driver, and the seq reference
# (every binary links seq so --validate can compare against it).
add_library(pagp_core STATIC
  src/core/image.cpp
  src/core/stb_impl.cpp
  src/core/validate.cpp
  src/core/cli.cpp
  src/core/csv.cpp
  src/core/runner.cpp
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
