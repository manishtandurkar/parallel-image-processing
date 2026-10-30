// Shared correctness check: every op x variant of a backend against the seq reference.
#pragma once

#include <cstdio>

#include "pagp/backend.hpp"
#include "pagp/validate.hpp"

namespace pagp {

// Returns the number of failing (op, variant, size) cases. exact = require max_abs_diff == 0.
inline int check_backend(const Backend &be, bool exact, int threads = 0) {
  static const int kSizes[][2] = {{1, 1},   {2, 3},    {3, 7},    {5, 5},  {16, 16},
                                  {64, 64}, {257, 131}, {640, 37}, {33, 300}};
  static const Op kOps[] = {Op::Gray, Op::Blur2d, Op::BlurSep, Op::Sobel, Op::Pipeline, Op::Fused};
  Options opts;
  opts.threads = threads;
  int failures = 0, cases = 0;
  for (auto &sz : kSizes) {
    Image rgb = generate_image(sz[0], sz[1], 7);
    Image gray(rgb.w, rgb.h, 1);
    seq::gray(rgb.data.data(), gray.data.data(), rgb.w, rgb.h);
    Workspace ws;
    ws.reserve(rgb.w, rgb.h);
    for (Op op : kOps) {
      const Image &src = op_takes_rgb(op) ? rgb : gray;
      Image ref(src.w, src.h, 1);
      run_reference(op, src, ref, ws);
      for (const std::string &v : be.variants(op)) {
        Image out(src.w, src.h, 1);
        be.run(op, v, opts, src, out, ws);
        int d = max_abs_diff(out, ref);
        int tol = exact ? 0 : tolerance(op);
        ++cases;
        if (d < 0 || d > tol) {
          ++failures;
          std::printf("FAIL %s %s/%s %dx%d max_abs_diff=%d tol=%d\n", be.name, op_name(op),
                      v.c_str(), src.w, src.h, d, tol);
        }
      }
    }
  }
  std::printf("%s: %d/%d cases passed\n", be.name, cases - failures, cases);
  return failures;
}

} // namespace pagp
