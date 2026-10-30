// Shared main driver for every backend binary.
#include "pagp/backend.hpp"

#include <algorithm>
#include <cstdio>

#include "pagp/csv.hpp"
#include "pagp/timer.hpp"
#include "pagp/validate.hpp"

namespace pagp {

int tolerance(Op op) {
  switch (op) {
  case Op::Gray:
  case Op::Sobel:
    return 0;
  case Op::Blur2d:
  case Op::BlurSep:
    return 1;
  case Op::Pipeline:
  case Op::Fused:
    // Not fixed by the spec: a +-1 blur difference can grow through Sobel. Revisit with SIMD/GPU data.
    return 1;
  }
  return 0;
}

void run_reference(Op op, const Image &src, Image &out, Workspace &ws) {
  const uint8_t *in = src.data.data();
  uint8_t *o = out.data.data();
  int w = src.w, h = src.h;
  switch (op) {
  case Op::Gray:
    seq::gray(in, o, w, h);
    break;
  case Op::Blur2d:
    seq::blur2d(in, o, w, h);
    break;
  case Op::BlurSep:
    seq::blur_sep(in, o, ws.f32.data(), w, h);
    break;
  case Op::Sobel:
    seq::sobel(in, o, w, h);
    break;
  case Op::Pipeline:
  case Op::Fused:
    seq::pipeline(in, o, ws, w, h);
    break;
  }
}

int run_main(int argc, char **argv, const Backend &be) {
  Options o;
  if (!parse_cli(argc, argv, o))
    return 2;

  std::vector<std::string> vs = be.variants(o.op);
  if (vs.empty()) {
    std::fprintf(stderr, "error: backend %s does not implement op %s\n", be.name, op_name(o.op));
    return 2;
  }
  std::string variant = o.variant.empty() ? vs.front() : o.variant;
  if (std::find(vs.begin(), vs.end(), variant) == vs.end()) {
    std::fprintf(stderr, "error: backend %s has no variant '%s' for op %s (have:", be.name,
                 variant.c_str(), op_name(o.op));
    for (auto &v : vs)
      std::fprintf(stderr, " %s", v.c_str());
    std::fprintf(stderr, ")\n");
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

  Workspace ws;
  ws.reserve(rgb.w, rgb.h);
  // Single-channel ops get a grayscale input, prepared outside the timed region.
  Image src = rgb;
  if (!op_takes_rgb(o.op)) {
    src = Image(rgb.w, rgb.h, 1);
    seq::gray(rgb.data.data(), src.data.data(), rgb.w, rgb.h);
  }
  Image out(src.w, src.h, 1);

  TimingStats t = time_reps([&] { be.run(o.op, variant, o, src, out, ws); }, o.warmup, o.reps);

  int diff = -1;
  if (o.validate) {
    Image ref(src.w, src.h, 1);
    run_reference(o.op, src, ref, ws);
    diff = max_abs_diff(out, ref);
  }

  std::printf("%s %s/%s %dx%d reps=%d median=%.3f ms min=%.3f ms", be.name, op_name(o.op),
              variant.c_str(), src.w, src.h, o.reps, t.median_ms, t.min_ms);
  if (diff >= 0)
    std::printf(" max_abs_diff=%d", diff);
  std::printf("\n");

  if (!o.save.empty() && !save_image(o.save, out)) {
    std::fprintf(stderr, "error: cannot write '%s'\n", o.save.c_str());
    return 1;
  }
  if (!o.csv.empty()) {
    ResultRow r;
    r.machine = o.machine;
    r.backend = be.name;
    r.variant = variant;
    r.op = op_name(o.op);
    r.width = src.w;
    r.height = src.h;
    if (o.threads > 0)
      r.workers = std::to_string(o.threads);
    r.reps = o.reps;
    r.median_ms = t.median_ms;
    r.min_ms = t.min_ms;
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

} // namespace pagp
