// Interface each backend binary implements; the shared driver (runner.cpp) does CLI, timing,
// validation, saving and CSV output.
#pragma once

#include <string>
#include <vector>

#include "pagp/cli.hpp"
#include "pagp/image.hpp"
#include "pagp/seq.hpp"

namespace pagp {

struct Backend {
  const char *name;
  // Supported variants for an op; the first is the default. Empty = op not supported.
  std::vector<std::string> (*variants)(Op op);
  // Runs op on src (RGB for op_takes_rgb ops, otherwise gray) into out (gray, same w/h).
  // Must not allocate full-image buffers itself when ws already provides them.
  void (*run)(Op op, const std::string &variant, const Options &opts, const Image &src, Image &out,
              Workspace &ws);
  // Value for the CSV workers column; nullptr = --threads if given, else 1.
  int (*workers)(const Options &opts) = nullptr;
};

// Allowed max_abs_diff against seq for an op.
int tolerance(Op op);

// Runs the seq reference (default variant) for op.
void run_reference(Op op, const Image &src, Image &out, Workspace &ws);

// Backend tables, one per backend library (each defined in src/<backend>/backend.cpp).
const Backend &seq_backend();
const Backend &autovec_backend();
const Backend &simd_backend(); // NEON on arm64 (AVX2 on x86-64: not written yet)
const Backend &omp_backend();

// Full CLI driver. Returns the process exit code.
int run_main(int argc, char **argv, const Backend &backend);

} // namespace pagp
