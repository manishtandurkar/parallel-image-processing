// Filter parameters shared by ALL backends. Change them here and in every backend together.
#pragma once

#include <cstdint>

namespace pagp {

// Grayscale: Y = 0.299 R + 0.587 G + 0.114 B in float, rounded to nearest, clamped to [0,255].
constexpr float kGrayR = 0.299f;
constexpr float kGrayG = 0.587f;
constexpr float kGrayB = 0.114f;

// Gaussian 5x5, sigma = 1.0. 1D weights exp(-i^2/2), i = -2..2, normalized to sum to 1.
// The 2D kernel is the outer product kGauss1D[i] * kGauss1D[j] (also sums to 1).
constexpr int kBlurRadius = 2;
constexpr int kBlurSize = 2 * kBlurRadius + 1;
constexpr float kGauss1D[kBlurSize] = {0.0544886845f, 0.2442013420f, 0.4026199469f, 0.2442013420f,
                                       0.0544886845f};

struct Gauss2D {
  float w[kBlurSize][kBlurSize];
  constexpr Gauss2D() : w{} {
    for (int i = 0; i < kBlurSize; ++i)
      for (int j = 0; j < kBlurSize; ++j)
        w[i][j] = kGauss1D[i] * kGauss1D[j];
  }
};
constexpr Gauss2D kGauss2D{};

// Sobel 3x3, indexed [dy+1][dx+1].
constexpr int kSobelRadius = 1;
constexpr int kSobelGx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
constexpr int kSobelGy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Clamp-to-edge border rule used by every filter in every backend.
inline int clamp_idx(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// Round to nearest (halves round up) and clamp to [0,255].
inline uint8_t to_u8(float v) {
  v = v + 0.5f;
  if (v < 0.0f)
    return 0;
  if (v >= 255.0f)
    return 255;
  return static_cast<uint8_t>(v);
}

} // namespace pagp
