// Sequential reference. Plain loops; the summation order here is the canonical one.
#include "pagp/seq.hpp"

#include <cmath>
#include <vector>

#include "pagp/filters.hpp"

namespace pagp {
namespace seq {

void gray(const uint8_t *rgb, uint8_t *out, int w, int h) {
  size_t n = size_t(w) * h;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t *p = rgb + 3 * i;
    float y = kGrayR * p[0] + kGrayG * p[1] + kGrayB * p[2];
    out[i] = to_u8(y);
  }
}

static uint8_t blur2d_pixel(const uint8_t *in, int w, int h, int x, int y) {
  float sum = 0.0f;
  for (int ky = -kBlurRadius; ky <= kBlurRadius; ++ky) {
    const uint8_t *row = in + size_t(clamp_idx(y + ky, h)) * w;
    for (int kx = -kBlurRadius; kx <= kBlurRadius; ++kx)
      sum += kGauss2D.w[ky + kBlurRadius][kx + kBlurRadius] * row[clamp_idx(x + kx, w)];
  }
  return to_u8(sum);
}

void blur2d(const uint8_t *in, uint8_t *out, int w, int h) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      out[size_t(y) * w + x] = blur2d_pixel(in, w, h, x, y);
}

void blur2d_tiled(const uint8_t *in, uint8_t *out, int w, int h, int tile) {
  for (int ty = 0; ty < h; ty += tile)
    for (int tx = 0; tx < w; tx += tile) {
      int ye = ty + tile < h ? ty + tile : h;
      int xe = tx + tile < w ? tx + tile : w;
      for (int y = ty; y < ye; ++y)
        for (int x = tx; x < xe; ++x)
          out[size_t(y) * w + x] = blur2d_pixel(in, w, h, x, y);
    }
}

// One horizontal 1x5 row: in (uint8) -> out (float).
static void blur_h_row(const uint8_t *in, float *out, int w) {
  for (int x = 0; x < w; ++x) {
    float sum = 0.0f;
    for (int k = -kBlurRadius; k <= kBlurRadius; ++k)
      sum += kGauss1D[k + kBlurRadius] * in[clamp_idx(x + k, w)];
    out[x] = sum;
  }
}

// One vertical 5x1 row from five (already clamped) float rows.
static void blur_v_row(const float *const rows[kBlurSize], uint8_t *out, int w) {
  for (int x = 0; x < w; ++x) {
    float sum = 0.0f;
    for (int k = 0; k < kBlurSize; ++k)
      sum += kGauss1D[k] * rows[k][x];
    out[x] = to_u8(sum);
  }
}

void blur_sep(const uint8_t *in, uint8_t *out, float *tmp, int w, int h) {
  for (int y = 0; y < h; ++y)
    blur_h_row(in + size_t(y) * w, tmp + size_t(y) * w, w);
  for (int y = 0; y < h; ++y) {
    const float *rows[kBlurSize];
    for (int k = 0; k < kBlurSize; ++k)
      rows[k] = tmp + size_t(clamp_idx(y + k - kBlurRadius, h)) * w;
    blur_v_row(rows, out + size_t(y) * w, w);
  }
}

// One Sobel output row from three (already clamped) input rows.
static void sobel_row(const uint8_t *const rows[3], uint8_t *out, int w) {
  for (int x = 0; x < w; ++x) {
    int gx = 0, gy = 0;
    for (int dy = 0; dy < 3; ++dy)
      for (int dx = 0; dx < 3; ++dx) {
        int v = rows[dy][clamp_idx(x + dx - 1, w)];
        gx += kSobelGx[dy][dx] * v;
        gy += kSobelGy[dy][dx] * v;
      }
    out[x] = to_u8(std::sqrt(float(gx * gx + gy * gy)));
  }
}

void sobel(const uint8_t *in, uint8_t *out, int w, int h) {
  for (int y = 0; y < h; ++y) {
    const uint8_t *rows[3];
    for (int k = 0; k < 3; ++k)
      rows[k] = in + size_t(clamp_idx(y + k - 1, h)) * w;
    sobel_row(rows, out + size_t(y) * w, w);
  }
}

void pipeline(const uint8_t *rgb, uint8_t *out, Workspace &ws, int w, int h) {
  gray(rgb, ws.u8a.data(), w, h);
  blur_sep(ws.u8a.data(), ws.u8b.data(), ws.f32.data(), w, h);
  sobel(ws.u8b.data(), out, w, h);
}

// Streams rows through two small ring buffers: horizontally blurred rows (float, 8 slots) and
// fully blurred rows (uint8, 4 slots). Rows are requested in non-decreasing windows of 5 and 3
// consecutive indices, so a slot is never evicted while still in use.
void fused(const uint8_t *rgb, uint8_t *out, int w, int h) {
  constexpr int kHSlots = 8, kBSlots = 4;
  std::vector<uint8_t> grow(w);
  std::vector<float> hbuf(size_t(kHSlots) * w);
  std::vector<uint8_t> bbuf(size_t(kBSlots) * w);
  int htag[kHSlots], btag[kBSlots];
  for (int &t : htag)
    t = -1;
  for (int &t : btag)
    t = -1;

  auto hrow = [&](int r) -> const float * {
    int s = r % kHSlots;
    float *dst = hbuf.data() + size_t(s) * w;
    if (htag[s] != r) {
      gray(rgb + size_t(r) * w * 3, grow.data(), w, 1);
      blur_h_row(grow.data(), dst, w);
      htag[s] = r;
    }
    return dst;
  };
  auto brow = [&](int r) -> const uint8_t * {
    int s = r % kBSlots;
    uint8_t *dst = bbuf.data() + size_t(s) * w;
    if (btag[s] != r) {
      const float *rows[kBlurSize];
      for (int k = 0; k < kBlurSize; ++k)
        rows[k] = hrow(clamp_idx(r + k - kBlurRadius, h));
      blur_v_row(rows, dst, w);
      btag[s] = r;
    }
    return dst;
  };

  for (int y = 0; y < h; ++y) {
    const uint8_t *rows[3];
    for (int k = 0; k < 3; ++k)
      rows[k] = brow(clamp_idx(y + k - 1, h));
    sobel_row(rows, out + size_t(y) * w, w);
  }
}

} // namespace seq
} // namespace pagp
