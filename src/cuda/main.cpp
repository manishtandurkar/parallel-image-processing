// pagp_cuda driver: same CLI as the other backends plus --block WxH. Its own driver (not run_main)
// because each rep has three separately timed phases. CSV: median_ms/min_ms = kernel time only;
// h2d_ms/d2h_ms = median transfer times (add them for end-to-end time); workers = block size.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "cuda_backend.hpp"
#include "pagp/backend.hpp"
#include "pagp/csv.hpp"
#include "pagp/timer.hpp"
#include "pagp/validate.hpp"

int main(int argc, char **argv) {
  using namespace pagp;
  Options o;
  if (!parse_cli(argc, argv, o))
    return 2;
  std::vector<std::string> vs = cuda::variants(o.op);
  std::string variant = o.variant.empty() ? vs.front() : o.variant;
  if (std::find(vs.begin(), vs.end(), variant) == vs.end()) {
    std::fprintf(stderr, "error: backend cuda has no variant '%s' for op %s\n", variant.c_str(),
                 op_name(o.op));
    return 2;
  }

  Image rgb;
  if (!o.input.empty()) {
    if (!load_image(o.input, rgb)) {
      std::fprintf(stderr, "error: cannot load '%s'\n", o.input.c_str());
      return 1;
    }
  } else {
    rgb = generate_image(o.gen_w, o.gen_h);
  }
  Image src;
  if (op_takes_rgb(o.op)) {
    src = std::move(rgb);
  } else {
    src = Image(rgb.w, rgb.h, 1);
    seq::gray(rgb.data.data(), src.data.data(), rgb.w, rgb.h);
  }
  Image out(src.w, src.h, 1);

  cuda::Context ctx(src.w, src.h, o.op);
  std::vector<double> h2d, kern, d2h, total;
  for (int i = 0; i < o.warmup + o.reps; ++i) {
    cuda::Timing t =
        cuda::run(o.op, variant, o.block_w, o.block_h, ctx, src.data.data(), out.data.data());
    if (i >= o.warmup) {
      h2d.push_back(t.h2d_ms);
      kern.push_back(t.kernel_ms);
      d2h.push_back(t.d2h_ms);
      total.push_back(double(t.h2d_ms) + t.kernel_ms + t.d2h_ms);
    }
  }
  TimingStats tk = summarize(kern), th = summarize(h2d), td = summarize(d2h), tt = summarize(total);

  int diff = -1;
  if (o.validate) {
    Workspace ws;
    ws.reserve(src.w, src.h);
    Image ref(src.w, src.h, 1);
    run_reference(o.op, src, ref, ws);
    diff = max_abs_diff(out, ref);
  }

  std::string block = std::to_string(o.block_w) + "x" + std::to_string(o.block_h);
  std::printf("cuda %s/%s %dx%d block=%s reps=%d kernel median=%.3f ms min=%.3f ms "
              "h2d=%.3f ms d2h=%.3f ms total=%.3f ms",
              op_name(o.op), variant.c_str(), src.w, src.h, block.c_str(), o.reps, tk.median_ms,
              tk.min_ms, th.median_ms, td.median_ms, tt.median_ms);
  if (diff >= 0)
    std::printf(" max_abs_diff=%d", diff);
  std::printf("  [%s]\n", cuda::device_name().c_str());

  if (!o.save.empty() && !save_image(o.save, out)) {
    std::fprintf(stderr, "error: cannot write '%s'\n", o.save.c_str());
    return 1;
  }
  if (!o.csv.empty()) {
    ResultRow r;
    r.machine = o.machine;
    r.backend = "cuda";
    r.variant = variant;
    r.op = op_name(o.op);
    r.width = src.w;
    r.height = src.h;
    r.workers = block;
    r.reps = o.reps;
    r.median_ms = tk.median_ms;
    r.min_ms = tk.min_ms;
    r.h2d_ms = th.median_ms;
    r.d2h_ms = td.median_ms;
    r.max_abs_diff = diff;
    if (!append_csv(o.csv, r)) {
      std::fprintf(stderr, "error: cannot write '%s'\n", o.csv.c_str());
      return 1;
    }
  }
  if (diff > tolerance(o.op)) {
    std::fprintf(stderr, "FAIL: max_abs_diff %d exceeds tolerance %d\n", diff, tolerance(o.op));
    return 3;
  }
  return 0;
}
