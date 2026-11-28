// Memory-layout cliff behind the MPI pipeline anomaly (8K, 4 ranks): the seq Sobel row kernel
// on one rank's block (7680 x 1080 rows) is ~4-5x slower when its input buffer sits a particular
// distance from its output buffer. This sweeps that distance (in whole rows, plus a few non-row
// offsets) with everything else fixed: same kernel, same data, page-aligned output.
//
// Usage: pagp_alias_sobel --machine <tag> [--csv <path>]
// CSV: machine,dist_rows,dist_bytes,median_ms,min_ms   (dist = input - output, 9 timed reps)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "pagp/rows.hpp"
#include "pagp/seq.hpp"

namespace {

constexpr int kW = 7680, kRows = 1080, kReps = 9;

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char **argv) {
  std::string machine = "unknown", csv;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--machine"))
      machine = argv[i + 1];
    else if (!std::strcmp(argv[i], "--csv"))
      csv = argv[i + 1];
  }
  const pagp::RowKernels &k = pagp::seq::row_kernels();
  const size_t span = size_t(kRows) * kW;

  std::vector<long> dists;
  for (long r : {-4352, -3264, -2176, -1152, -1090, -1089, -1088, -1087, -1086, -1024, -1000, -545,
                 -544, -543, -272, 1088, 2176})
    dists.push_back(r * kW);
  for (long b : {4096L, 8192L, 64L})
    dists.push_back(-1088L * kW + b);

  // Output in the middle of one big allocation so input can sit on either side.
  const size_t half = size_t(48) << 20;
  std::vector<uint8_t> big(2 * half + span);
  uint8_t *out = big.data() + half;
  out -= uintptr_t(out) & 16383; // page-aligned, as large malloc blocks are on macOS
  uint32_t s = 12345;
  std::vector<uint8_t> src(span);
  for (auto &v : src)
    v = uint8_t((s = s * 1664525u + 1013904223u) >> 24);

  FILE *f = nullptr;
  if (!csv.empty()) {
    bool header = !std::fopen(csv.c_str(), "r");
    f = std::fopen(csv.c_str(), "a");
    if (header)
      std::fprintf(f, "machine,dist_rows,dist_bytes,median_ms,min_ms\n");
  }
  for (long d : dists) {
    uint8_t *in = out + d;
    std::copy(src.begin(), src.end(), in);
    std::vector<double> t;
    for (int r = 0; r < kReps + 2; ++r) {
      double t0 = now_ms();
      pagp::sobel_rows(k, in, out, kW, kRows, 0, kRows);
      if (r >= 2)
        t.push_back(now_ms() - t0);
    }
    std::sort(t.begin(), t.end());
    double rows = double(d) / kW;
    std::printf("dist %+9.2f rows (%+10ld B): median %6.2f ms  min %6.2f ms\n", rows, d,
                t[kReps / 2], t[0]);
    if (f)
      std::fprintf(f, "%s,%.4f,%ld,%.4f,%.4f\n", machine.c_str(), rows, d, t[kReps / 2], t[0]);
  }
  if (f)
    std::fclose(f);
  return 0;
}
