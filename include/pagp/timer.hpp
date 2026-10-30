// steady_clock timing: warm-up runs, then timed repetitions summarized by median and min.
#pragma once

#include <algorithm>
#include <chrono>
#include <vector>

namespace pagp {

struct TimingStats {
  double median_ms = 0, min_ms = 0;
};

inline TimingStats summarize(std::vector<double> ms) {
  TimingStats s;
  if (ms.empty())
    return s;
  std::sort(ms.begin(), ms.end());
  size_t n = ms.size();
  s.median_ms = n % 2 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
  s.min_ms = ms.front();
  return s;
}

template <class F> TimingStats time_reps(F &&f, int warmup, int reps) {
  using clock = std::chrono::steady_clock;
  for (int i = 0; i < warmup; ++i)
    f();
  std::vector<double> ms;
  ms.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    auto t0 = clock::now();
    f();
    auto t1 = clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return summarize(ms);
}

} // namespace pagp
