// CUDA backend interface. Plain C++ (no CUDA headers) so the driver and tests compile with the
// host compiler; everything CUDA lives in cuda.cu.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pagp/cli.hpp"

namespace pagp::cuda {

// Variants (first = default):
//   naive  global memory only; Gaussian weights read from a global-memory array
//   const  as naive, weights in __constant__ memory
//   tiled  shared-memory tile with halo (+ __syncthreads), weights in __constant__ memory
// fused (op) is one kernel doing gray -> blur_h -> blur_v -> sobel in shared memory: "tiled".
std::vector<std::string> variants(Op op);

// Name and compute capability of the current device, e.g. "Tesla T4 (sm_75)".
std::string device_name();

struct Timing {
  float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
};

// Device buffers for one image size and op, allocated once outside the timed region.
class Context {
public:
  Context(int w, int h, Op op);
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

// Copies src (RGB for op_takes_rgb ops, otherwise gray) to the device, runs op with
// block_w x block_h thread blocks, and copies the gray result back to out. Each phase is timed
// separately with cudaEvent_t.
Timing run(Op op, const std::string &variant, int block_w, int block_h, Context &ctx,
           const uint8_t *src, uint8_t *out);

} // namespace pagp::cuda
