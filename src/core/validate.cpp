#include "pagp/validate.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace pagp {

int max_abs_diff(const Image &a, const Image &b) {
  if (a.w != b.w || a.h != b.h || a.c != b.c)
    return -1;
  int m = 0;
  for (size_t i = 0; i < a.data.size(); ++i) {
    int d = std::abs(int(a.data[i]) - int(b.data[i]));
    if (d > m)
      m = d;
  }
  return m;
}

double psnr(const Image &a, const Image &b) {
  if (a.data.size() != b.data.size() || a.data.empty())
    return 0.0;
  double se = 0.0;
  for (size_t i = 0; i < a.data.size(); ++i) {
    double d = double(a.data[i]) - double(b.data[i]);
    se += d * d;
  }
  if (se == 0.0)
    return std::numeric_limits<double>::infinity();
  double mse = se / double(a.data.size());
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

} // namespace pagp
