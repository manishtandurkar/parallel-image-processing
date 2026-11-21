// pagp_acc / pagp_acc_mc driver: same CLI as the other backends. Its own driver (not run_main)
// because each rep has separately timed phases, like pagp_cuda. CSV: median_ms/min_ms = compute
// time only, h2d_ms/d2h_ms = median transfer times; for the copy variant transfers happen inside
// each compute region, so median_ms is end-to-end and h2d_ms/d2h_ms are empty.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "acc_backend.hpp"
#include "pagp/backend.hpp"
#include "pagp/csv.hpp"
#include "pagp/timer.hpp"
#include "pagp/validate.hpp"

int main(int argc, char **argv) {
  using namespace pagp;
  Options o;
  if (!parse_cli(argc, argv, o))
    return 2;
  const char *name = acc::backend_name();
  std::vector<std::string> vs = acc::variants(o.op);
  std::string variant = o.variant.empty() ? vs.front() : o.variant;
  if (std::find(vs.begin(), vs.end(), variant) == vs.end()) {
    std::fprintf(stderr, "error: backend %s has no variant '%s' for op %s\n", name,
                 variant.c_str(), op_name(o.op));
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

  acc::Context ctx(src.w, src.h, o.op, variant);
  std::vector<double> h2d, kern, d2h;
  bool transfers = true;
  for (int i = 0; i < o.warmup + o.reps; ++i) {
    acc::Timing t = acc::run(o.op, variant, ctx, src.data.data(), out.data.data());
    transfers = t.h2d_ms >= 0;
    if (i >= o.warmup) {
      h2d.push_back(t.h2d_ms);
      kern.push_back(t.kernel_ms);
      d2h.push_back(t.d2h_ms);
    }
  }
  TimingStats tk = summarize(kern), th = summarize(h2d), td = summarize(d2h);

  int diff = -1;
  if (o.validate) {
    Workspace ws;
    ws.reserve(src.w, src.h);
    Image ref(src.w, src.h, 1);
    run_reference(o.op, src, ref, ws);
    diff = max_abs_diff(out, ref);
  }

  std::printf("%s %s/%s %dx%d reps=%d compute median=%.3f ms min=%.3f ms", name, op_name(o.op),
              variant.c_str(), src.w, src.h, o.reps, tk.median_ms, tk.min_ms);
  if (transfers)
    std::printf(" h2d=%.3f ms d2h=%.3f ms", th.median_ms, td.median_ms);
  else
    std::printf(" (transfers included)");
  if (diff >= 0)
    std::printf(" max_abs_diff=%d", diff);
  std::printf("  [%s]\n", acc::device_name().c_str());

  if (!o.save.empty() && !save_image(o.save, out)) {
    std::fprintf(stderr, "error: cannot write '%s'\n", o.save.c_str());
    return 1;
  }
  if (!o.csv.empty()) {
    ResultRow r;
    r.machine = o.machine;
    r.backend = name;
    r.variant = variant;
    r.op = op_name(o.op);
    r.width = src.w;
    r.height = src.h;
    r.workers = std::to_string(acc::workers());
    r.reps = o.reps;
    r.median_ms = tk.median_ms;
    r.min_ms = tk.min_ms;
    if (transfers) {
      r.h2d_ms = th.median_ms;
      r.d2h_ms = td.median_ms;
    }
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
