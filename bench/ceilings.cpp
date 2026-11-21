// Measured roofline ceilings for one machine: memory bandwidth (STREAM-style triad) and peak
// arithmetic rate (independent multiply + add chains), each at several thread counts.
//
// Usage: pagp_ceilings --machine <tag> [--threads 1,4,10] [--csv <path>]
// CSV: machine,kind,threads,value,unit   (kind = triad_bw | peak_flops; best of the repetitions)
//
// triad: a[i] = b[i] + s * c[i] on float arrays far larger than the caches; bytes counted as
// 3 * 4 per element (STREAM convention, write-allocate traffic not counted).
// peak: each thread updates kAcc independent 8-float vectors with acc = acc * x + y (2 flops per
// lane; built with -ffp-contract=off so it stays a separate multiply and add, like the filters).
// GCC/Clang vector types: two 128-bit NEON registers per vector on arm64, one AVX register on x86.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kN = size_t(1) << 26; // 64 Mi floats = 256 MiB per array
constexpr int kReps = 10;
using vf = float __attribute__((vector_size(32)));
constexpr int kLanes = 8;
constexpr int kAcc = 12;               // independent vectors per thread (24 NEON registers)
constexpr long kIters = 20'000'000;    // per thread

double triad_gbs(int threads, std::vector<float> &a, const std::vector<float> &b,
                 const std::vector<float> &c) {
  const float s = 1.0001f;
  float *pa = a.data();
  const float *pb = b.data(), *pc = c.data();
  double best = 0;
  for (int r = 0; r < kReps; ++r) {
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < kN; ++i)
      pa[i] = pb[i] + s * pc[i];
    double t = omp_get_wtime() - t0;
    best = std::max(best, 3.0 * sizeof(float) * kN / t / 1e9);
  }
  return best;
}

double peak_gflops(int threads) {
  volatile float vx = 0.9999999f, vy = 1e-7f;
  double best = 0;
  float sink = 0;
  for (int r = 0; r < 3; ++r) {
    double t0 = omp_get_wtime();
#pragma omp parallel num_threads(threads) reduction(+ : sink)
    {
      const float xs = vx, ys = vy;
      const vf x = xs - vf{}, y = ys - vf{}; // broadcast
      vf acc[kAcc];
      for (int j = 0; j < kAcc; ++j)
        acc[j] = float(j + omp_get_thread_num()) - vf{};
      for (long it = 0; it < kIters; ++it)
        for (int j = 0; j < kAcc; ++j)
          acc[j] = acc[j] * x + y;
      for (int j = 0; j < kAcc; ++j)
        for (int l = 0; l < kLanes; ++l)
          sink += acc[j][l];
    }
    double t = omp_get_wtime() - t0;
    best = std::max(best, 2.0 * kAcc * kLanes * kIters * threads / t / 1e9);
  }
  if (sink == 42.0f) // keep the result live
    std::puts("");
  return best;
}

} // namespace

int main(int argc, char **argv) {
  std::string machine = "unknown", csv;
  std::vector<int> threads;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--machine" && i + 1 < argc)
      machine = argv[++i];
    else if (a == "--csv" && i + 1 < argc)
      csv = argv[++i];
    else if (a == "--threads" && i + 1 < argc) {
      std::string list = argv[++i];
      for (size_t p = 0; p < list.size();) {
        size_t q = list.find(',', p);
        threads.push_back(std::atoi(list.substr(p, q - p).c_str()));
        p = q == std::string::npos ? list.size() : q + 1;
      }
    } else {
      std::fprintf(stderr, "usage: %s --machine <tag> [--threads 1,4,10] [--csv <path>]\n",
                   argv[0]);
      return 2;
    }
  }
  if (threads.empty())
    threads = {1, omp_get_max_threads()};

  std::vector<float> a(kN), b(kN, 1.0f), c(kN, 2.0f);
#pragma omp parallel for schedule(static) // first touch spread over threads
  for (size_t i = 0; i < kN; ++i)
    a[i] = 0.0f;

  FILE *f = nullptr;
  if (!csv.empty()) {
    bool fresh = true;
    if (FILE *g = std::fopen(csv.c_str(), "r")) {
      fresh = std::fgetc(g) == EOF;
      std::fclose(g);
    }
    f = std::fopen(csv.c_str(), "a");
    if (!f) {
      std::fprintf(stderr, "error: cannot write '%s'\n", csv.c_str());
      return 1;
    }
    if (fresh)
      std::fputs("machine,kind,threads,value,unit\n", f);
  }
  for (int t : threads) {
    double bw = triad_gbs(t, a, b, c), fl = peak_gflops(t);
    std::printf("%s threads=%d triad=%.1f GB/s peak=%.1f GFLOP/s\n", machine.c_str(), t, bw, fl);
    if (f) {
      std::fprintf(f, "%s,triad_bw,%d,%.2f,GB/s\n", machine.c_str(), t, bw);
      std::fprintf(f, "%s,peak_flops,%d,%.2f,GFLOP/s\n", machine.c_str(), t, fl);
    }
  }
  if (f)
    std::fclose(f);
  return 0;
}
