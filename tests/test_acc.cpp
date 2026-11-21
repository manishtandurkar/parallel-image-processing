// OpenACC backend vs seq: every op x variant x size must match seq exactly (same arithmetic order,
// FMA disabled). Linked once against the GPU build and once against the multicore build.
#include <cstdio>

#include "acc_backend.hpp"
#include "pagp/backend.hpp"
#include "pagp/validate.hpp"

int main() {
  using namespace pagp;
  static const int kSizes[][2] = {{1, 1},   {2, 3},     {3, 7},    {5, 5},   {16, 16},
                                  {64, 64}, {257, 131}, {640, 37}, {33, 300}};
  static const Op kOps[] = {Op::Gray, Op::Blur2d, Op::BlurSep, Op::Sobel, Op::Pipeline, Op::Fused};
  std::printf("%s on %s\n", acc::backend_name(), acc::device_name().c_str());
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
      for (const std::string &v : acc::variants(op)) {
        acc::Context ctx(src.w, src.h, op, v);
        Image out(src.w, src.h, 1);
        acc::run(op, v, ctx, src.data.data(), out.data.data());
        int d = max_abs_diff(out, ref);
        ++cases;
        if (d != 0) {
          ++failures;
          std::printf("FAIL %s %s/%s %dx%d max_abs_diff=%d\n", acc::backend_name(), op_name(op),
                      v.c_str(), src.w, src.h, d);
        }
      }
    }
  }
  std::printf("%s: %d/%d cases passed (exact)\n", acc::backend_name(), cases - failures, cases);
  return failures ? 1 : 0;
}
