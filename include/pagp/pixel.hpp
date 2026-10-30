// Scalar per-pixel filter kernels with clamp-to-edge borders. These fix the canonical arithmetic
// and summation order; SIMD/vectorized row kernels use them for border and remainder pixels and
// must reproduce the same order in their vector paths.
#pragma once

#include <cmath>
#include <cstdint>

#include "pagp/filters.hpp"

namespace pagp {
namespace px {

inline uint8_t gray(const uint8_t *p) {
  return to_u8(kGrayR * p[0] + kGrayG * p[1] + kGrayB * p[2]);
}

// rows[k] = input row clamp(y + k - 2).
inline uint8_t blur2d(const uint8_t *const rows[kBlurSize], int w, int x) {
  float sum = 0.0f;
  for (int ky = 0; ky < kBlurSize; ++ky)
    for (int kx = 0; kx < kBlurSize; ++kx)
      sum += kGauss2D.w[ky][kx] * rows[ky][clamp_idx(x + kx - kBlurRadius, w)];
  return to_u8(sum);
}

inline float blur_h(const uint8_t *in, int w, int x) {
  float sum = 0.0f;
  for (int k = 0; k < kBlurSize; ++k)
    sum += kGauss1D[k] * in[clamp_idx(x + k - kBlurRadius, w)];
  return sum;
}

// rows[k] = horizontally blurred row clamp(y + k - 2).
inline uint8_t blur_v(const float *const rows[kBlurSize], int x) {
  float sum = 0.0f;
  for (int k = 0; k < kBlurSize; ++k)
    sum += kGauss1D[k] * rows[k][x];
  return to_u8(sum);
}

inline uint8_t sobel_mag(int gx, int gy) { return to_u8(std::sqrt(float(gx * gx + gy * gy))); }

// rows[k] = input row clamp(y + k - 1).
inline uint8_t sobel(const uint8_t *const rows[3], int w, int x) {
  int gx = 0, gy = 0;
  for (int dy = 0; dy < 3; ++dy)
    for (int dx = 0; dx < 3; ++dx) {
      int v = rows[dy][clamp_idx(x + dx - 1, w)];
      gx += kSobelGx[dy][dx] * v;
      gy += kSobelGy[dy][dx] * v;
    }
  return sobel_mag(gx, gy);
}

} // namespace px
} // namespace pagp
