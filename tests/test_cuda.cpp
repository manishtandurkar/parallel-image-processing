// CUDA backend vs seq: every op x variant x size x block shape must match seq exactly
// (same arithmetic order, built with --fmad=false). Odd sizes exercise partial edge blocks and
// images smaller than one block.
#include <cstdio>

#include "cuda_backend.hpp"
#include "pagp/backend.hpp"
#include "pagp/validate.hpp"

int main() {
  using namespace pagp;
  static const int kSizes[][2] = {{1, 1},   {2, 3},     {3, 7},    {5, 5},   {16, 16},
                                  {64, 64}, {257, 131}, {640, 37}, {33, 300}};
  static const int kBlocks[][2] = {{16, 16}, {8, 8}, {32, 8}, {32, 32}, {7, 5}};
  static const Op kOps[] = {Op::Gray, Op::Blur2d, Op::BlurSep, Op::Sobel, Op::Pipeline, Op::Fused};
  std::printf("device: %s\n", cuda::device_name().c_str());
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
      cuda::Context ctx(src.w, src.h, op);
      for (const std::string &v : cuda::variants(op))
        for (auto &bl : kBlocks) {
          Image out(src.w, src.h, 1);
          cuda::run(op, v, bl[0], bl[1], ctx, src.data.data(), out.data.data());
          int d = max_abs_diff(out, ref);
          ++cases;
          if (d != 0) {
            ++failures;
            std::printf("FAIL cuda %s/%s %dx%d block %dx%d max_abs_diff=%d\n", op_name(op),
                        v.c_str(), src.w, src.h, bl[0], bl[1], d);
          }
        }
    }
  }
  std::printf("cuda: %d/%d cases passed (exact)\n", cases - failures, cases);
  return failures ? 1 : 0;
}
