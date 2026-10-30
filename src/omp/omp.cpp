// OpenMP backend: parallel for over rows using the seq scalar row kernels, so speedup vs seq
// isolates threading. Schedules (static|dynamic|guided) go through schedule(runtime).
// gray also has synchronization demo variants that count bright pixels (Y >= 128) per thread:
//   false_sharing  per-thread counters adjacent in one array (same cache line)
//   padded         per-thread counters each on its own cache line (128 B: Apple M-series lines are
//                  128 B, x86 lines are 64 B, so 128 is safe on both)
//   padded64       counters 64 B apart: enough on x86, still false sharing on 128 B lines
//   reduction      reduction(+:count)
//   atomic         one shared counter, #pragma omp atomic per pixel
#include <omp.h>

#include <atomic>
#include <vector>

#include "pagp/backend.hpp"
#include "pagp/pixel.hpp"
#include "pagp/rows.hpp"

namespace {

using namespace pagp;

constexpr int kFusedBlockRows = 64;
constexpr int kBright = 128;
constexpr int kCacheLine = 128;

std::vector<std::string> variants(Op op) {
  if (op == Op::Gray)
    return {"static", "dynamic",  "guided",    "false_sharing",
            "padded", "padded64", "reduction", "atomic"};
  return {"static", "dynamic", "guided"};
}

bool set_schedule(const std::string &v) {
  if (v == "dynamic")
    omp_set_schedule(omp_sched_dynamic, 0);
  else if (v == "guided")
    omp_set_schedule(omp_sched_guided, 0);
  else if (v == "static")
    omp_set_schedule(omp_sched_static, 0);
  else
    return false;
  return true;
}

void gray_counting(const std::string &v, const uint8_t *rgb, uint8_t *out, int w, int h) {
  int nt = omp_get_max_threads();
  volatile long sink = 0;
  if (v == "false_sharing" || v == "padded" || v == "padded64") {
    // Relaxed atomic increments: each one is a real read-modify-write on the cache line. Plain
    // (volatile) stores can be merged in the core's store buffer, which hides false sharing on
    // weakly ordered arm64. Both variants pay the same instruction cost; only the layout differs.
    struct alignas(kCacheLine) Padded {
      std::atomic<long> v{0};
    };
    struct alignas(64) Padded64 {
      std::atomic<long> v{0};
    };
    std::vector<std::atomic<long>> packed(nt);
    std::vector<Padded> padded(nt);
    std::vector<Padded64> padded64(nt);
#pragma omp parallel
    {
      int t = omp_get_thread_num();
      std::atomic<long> *c = v == "padded"     ? &padded[t].v
                             : v == "padded64" ? &padded64[t].v
                                               : &packed[t];
#pragma omp for schedule(static)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          size_t i = size_t(y) * w + x;
          uint8_t g = px::gray(rgb + 3 * i);
          out[i] = g;
          if (g >= kBright)
            c->fetch_add(1, std::memory_order_relaxed);
        }
    }
    long total = 0;
    for (int t = 0; t < nt; ++t)
      total += packed[t] + padded[t].v + padded64[t].v;
    sink = total;
  } else if (v == "reduction") {
    long count = 0;
#pragma omp parallel for schedule(static) reduction(+ : count)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        size_t i = size_t(y) * w + x;
        uint8_t g = px::gray(rgb + 3 * i);
        out[i] = g;
        count += g >= kBright;
      }
    sink = count;
  } else { // atomic
    long count = 0;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        size_t i = size_t(y) * w + x;
        uint8_t g = px::gray(rgb + 3 * i);
        out[i] = g;
        if (g >= kBright) {
#pragma omp atomic
          ++count;
        }
      }
    sink = count;
  }
  (void)sink;
}

void run(Op op, const std::string &v, const Options &opts, const Image &src, Image &out,
         Workspace &ws) {
  if (opts.threads > 0)
    omp_set_num_threads(opts.threads);
  const RowKernels &k = seq::row_kernels();
  const uint8_t *in = src.data.data();
  uint8_t *o = out.data.data();
  int w = src.w, h = src.h;

  if (!set_schedule(v)) {
    gray_counting(v, in, o, w, h);
    return;
  }

  switch (op) {
  case Op::Gray:
#pragma omp parallel for schedule(runtime)
    for (int y = 0; y < h; ++y)
      gray_rows(k, in, o, w, y, y + 1);
    break;
  case Op::Blur2d:
#pragma omp parallel for schedule(runtime)
    for (int y = 0; y < h; ++y)
      blur2d_rows(k, in, o, w, h, y, y + 1);
    break;
  case Op::BlurSep:
  case Op::Pipeline: {
    float *tmp = ws.f32.data();
    const uint8_t *bin = op == Op::Pipeline ? ws.u8a.data() : in;
    uint8_t *bout = op == Op::Pipeline ? ws.u8b.data() : o;
#pragma omp parallel
    {
      if (op == Op::Pipeline) {
#pragma omp for schedule(runtime)
        for (int y = 0; y < h; ++y)
          gray_rows(k, in, ws.u8a.data(), w, y, y + 1);
      }
#pragma omp for schedule(runtime)
      for (int y = 0; y < h; ++y)
        blur_h_rows(k, bin, tmp, w, y, y + 1);
#pragma omp for schedule(runtime)
      for (int y = 0; y < h; ++y)
        blur_v_rows(k, tmp, bout, w, h, y, y + 1);
      if (op == Op::Pipeline) {
#pragma omp for schedule(runtime)
        for (int y = 0; y < h; ++y)
          sobel_rows(k, bout, o, w, h, y, y + 1);
      }
    }
    break;
  }
  case Op::Sobel:
#pragma omp parallel for schedule(runtime)
    for (int y = 0; y < h; ++y)
      sobel_rows(k, in, o, w, h, y, y + 1);
    break;
  case Op::Fused: {
    // Each block of rows streams independently; rows bordering a block are recomputed.
    int blocks = (h + kFusedBlockRows - 1) / kFusedBlockRows;
#pragma omp parallel for schedule(runtime)
    for (int b = 0; b < blocks; ++b) {
      int y0 = b * kFusedBlockRows, y1 = y0 + kFusedBlockRows < h ? y0 + kFusedBlockRows : h;
      fused_rows(k, in, o, w, h, y0, y1);
    }
    break;
  }
  }
}

int workers(const Options &opts) { return opts.threads > 0 ? opts.threads : omp_get_max_threads(); }

} // namespace

const pagp::Backend &pagp::omp_backend() {
  static const Backend b{"omp", variants, run, workers};
  return b;
}
