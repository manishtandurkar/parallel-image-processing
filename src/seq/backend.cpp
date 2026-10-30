// Backend table for the sequential reference (pagp_seq).
#include "pagp/backend.hpp"

namespace {

constexpr int kDefaultTile = 64;

std::vector<std::string> variants(pagp::Op op) {
  using pagp::Op;
  switch (op) {
  case Op::Blur2d:
    return {"naive", "tiled"};
  default:
    return {"naive"};
  }
}

void run(pagp::Op op, const std::string &variant, const pagp::Options &, const pagp::Image &src,
         pagp::Image &out, pagp::Workspace &ws) {
  using pagp::Op;
  const uint8_t *in = src.data.data();
  uint8_t *o = out.data.data();
  int w = src.w, h = src.h;
  switch (op) {
  case Op::Blur2d:
    if (variant == "tiled")
      pagp::seq::blur2d_tiled(in, o, w, h, kDefaultTile);
    else
      pagp::seq::blur2d(in, o, w, h);
    break;
  case Op::Fused:
    pagp::seq::fused(in, o, w, h);
    break;
  default:
    pagp::run_reference(op, src, out, ws);
    break;
  }
}

} // namespace

const pagp::Backend &pagp::seq_backend() {
  static const Backend b{"seq", variants, run};
  return b;
}
