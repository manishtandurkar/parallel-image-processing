// OpenACC backend interface. The same source builds twice: -acc=gpu (pagp_acc) and
// -acc=multicore (pagp_acc_mc). Without _OPENACC the pragmas are ignored and it runs sequentially,
// which lets the loops be checked against seq with any compiler.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pagp/cli.hpp"

namespace pagp::acc {

// Variants (first = default), the same for every op:
//   parallel  `parallel loop collapse(2)`; image buffers stay in a device data region
//   kernels   `kernels` region; the compiler decides what to parallelize; same data region.
//             nvc++ 26.11 cannot prove the pixel loops independent: gray, sobel and fused become
//             serial GPU kernels and the blur stages parallelize only their 5-tap inner loops
//             (see report/acc/minfo_colab-t4.txt)
//   kernels_indep  as kernels, plus `loop independent collapse(2)` asserting what the compiler
//             could not prove
//   copy      copyin/copyout clauses on each compute region and no data region, so every stage
//             of pipeline moves its input and output between host and device
// fused (op) is one compute region that recomputes gray and blur for each output pixel.
std::vector<std::string> variants(Op op);

// "acc" for a GPU build, "acc_mc" for a multicore build (or a build without OpenACC).
const char *backend_name();
// Device description, e.g. "Tesla T4" or "host multicore (2 threads)".
std::string device_name();
// CSV workers column: 1 on a GPU, the thread count on multicore.
int workers();

struct Timing {
  // h2d/d2h are -1 for the copy variant: its transfers happen inside kernel_ms.
  double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
};

// Host staging buffers (and, unless the variant is copy, their device copies) for one image size
// and op, allocated once outside the timed region.
class Context {
public:
  Context(int w, int h, Op op, const std::string &variant);
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

// Runs op on src (RGB for op_takes_rgb ops, otherwise gray) and writes the gray result to out.
// variant must be the one ctx was created with.
Timing run(Op op, const std::string &variant, Context &ctx, const uint8_t *src, uint8_t *out);

} // namespace pagp::acc
