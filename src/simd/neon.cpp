// Hand-written NEON row kernels (arm64), 16 pixels per iteration. Floats use separate vmulq/vaddq
// (no fused multiply-add) in the same order as seq, so results are bit-identical. Border and
// remainder pixels use the scalar kernels from pixel.hpp.
#if defined(__aarch64__)

#include <arm_neon.h>

#include "pagp/backend.hpp"
#include "pagp/pixel.hpp"
#include "pagp/rows.hpp"

namespace {

using namespace pagp;

// 16 x u8 -> 4 x (4 x f32), lanes in order.
inline void widen_f32(uint8x16_t v, float32x4_t f[4]) {
  uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_high_u8(v);
  f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
  f[1] = vcvtq_f32_u32(vmovl_high_u16(lo));
  f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
  f[3] = vcvtq_f32_u32(vmovl_high_u16(hi));
}

// Vector to_u8: +0.5, clamp to 255, truncate (vcvtq_u32_f32 saturates negatives to 0).
inline uint8x16_t narrow_u8(const float32x4_t f[4]) {
  const float32x4_t half = vdupq_n_f32(0.5f), top = vdupq_n_f32(255.0f);
  uint32x4_t u[4];
  for (int j = 0; j < 4; ++j)
    u[j] = vcvtq_u32_f32(vminq_f32(vaddq_f32(f[j], half), top));
  uint16x8_t a = vcombine_u16(vmovn_u32(u[0]), vmovn_u32(u[1]));
  uint16x8_t b = vcombine_u16(vmovn_u32(u[2]), vmovn_u32(u[3]));
  return vcombine_u8(vmovn_u16(a), vmovn_u16(b));
}

void gray_row(const uint8_t *rgb, uint8_t *out, int n) {
  const float32x4_t R = vdupq_n_f32(kGrayR), G = vdupq_n_f32(kGrayG), B = vdupq_n_f32(kGrayB);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16x3_t p = vld3q_u8(rgb + 3 * i); // de-interleaves R, G, B
    float32x4_t r[4], g[4], b[4], y[4];
    widen_f32(p.val[0], r);
    widen_f32(p.val[1], g);
    widen_f32(p.val[2], b);
    for (int j = 0; j < 4; ++j)
      y[j] = vaddq_f32(vaddq_f32(vmulq_f32(R, r[j]), vmulq_f32(G, g[j])), vmulq_f32(B, b[j]));
    vst1q_u8(out + i, narrow_u8(y));
  }
  for (; i < n; ++i)
    out[i] = px::gray(rgb + 3 * i);
}

void blur2d_row(const uint8_t *const rows_in[kBlurSize], uint8_t *out, int w) {
  const uint8_t *rows[kBlurSize] = {rows_in[0], rows_in[1], rows_in[2], rows_in[3], rows_in[4]};
  int x = 0;
  for (; x < kBlurRadius && x < w; ++x)
    out[x] = px::blur2d(rows, w, x);
  for (; x + 16 <= w - kBlurRadius; x += 16) {
    float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                          vdupq_n_f32(0.0f)};
    for (int ky = 0; ky < kBlurSize; ++ky)
      for (int kx = 0; kx < kBlurSize; ++kx) {
        float32x4_t k = vdupq_n_f32(kGauss2D.w[ky][kx]), f[4];
        widen_f32(vld1q_u8(rows[ky] + x + kx - kBlurRadius), f);
        for (int j = 0; j < 4; ++j)
          acc[j] = vaddq_f32(acc[j], vmulq_f32(k, f[j]));
      }
    vst1q_u8(out + x, narrow_u8(acc));
  }
  for (; x < w; ++x)
    out[x] = px::blur2d(rows, w, x);
}

void blur_h_row(const uint8_t *in, float *out, int w) {
  int x = 0;
  for (; x < kBlurRadius && x < w; ++x)
    out[x] = px::blur_h(in, w, x);
  for (; x + 16 <= w - kBlurRadius; x += 16) {
    float32x4_t acc[4], f[4];
    widen_f32(vld1q_u8(in + x - 2), f);
    float32x4_t g = vdupq_n_f32(kGauss1D[0]);
    for (int j = 0; j < 4; ++j)
      acc[j] = vmulq_f32(g, f[j]); // == 0.0f + g * v
    for (int k = 1; k < kBlurSize; ++k) {
      g = vdupq_n_f32(kGauss1D[k]);
      widen_f32(vld1q_u8(in + x + k - 2), f);
      for (int j = 0; j < 4; ++j)
        acc[j] = vaddq_f32(acc[j], vmulq_f32(g, f[j]));
    }
    for (int j = 0; j < 4; ++j)
      vst1q_f32(out + x + 4 * j, acc[j]);
  }
  for (; x < w; ++x)
    out[x] = px::blur_h(in, w, x);
}

void blur_v_row(const float *const rows_in[kBlurSize], uint8_t *out, int w) {
  const float *rows[kBlurSize] = {rows_in[0], rows_in[1], rows_in[2], rows_in[3], rows_in[4]};
  int x = 0;
  for (; x + 16 <= w; x += 16) {
    float32x4_t acc[4];
    float32x4_t g = vdupq_n_f32(kGauss1D[0]);
    for (int j = 0; j < 4; ++j)
      acc[j] = vmulq_f32(g, vld1q_f32(rows[0] + x + 4 * j));
    for (int k = 1; k < kBlurSize; ++k) {
      g = vdupq_n_f32(kGauss1D[k]);
      for (int j = 0; j < 4; ++j)
        acc[j] = vaddq_f32(acc[j], vmulq_f32(g, vld1q_f32(rows[k] + x + 4 * j)));
    }
    vst1q_u8(out + x, narrow_u8(acc));
  }
  for (; x < w; ++x)
    out[x] = px::blur_v(rows, x);
}

inline int16x8_t s16(uint8x8_t v) { return vreinterpretq_s16_u16(vmovl_u8(v)); }

// Sobel magnitude for 8 pixels given left/centre/right neighbours of the three rows.
inline void sobel8(const uint8x8_t l[3], const uint8x8_t c[3], const uint8x8_t r[3],
                   float32x4_t mag[2]) {
  int16x8_t gx = vaddq_s16(
      vaddq_s16(vsubq_s16(s16(r[0]), s16(l[0])), vshlq_n_s16(vsubq_s16(s16(r[1]), s16(l[1])), 1)),
      vsubq_s16(s16(r[2]), s16(l[2])));
  int16x8_t top = vaddq_s16(vaddq_s16(s16(l[0]), vshlq_n_s16(s16(c[0]), 1)), s16(r[0]));
  int16x8_t bot = vaddq_s16(vaddq_s16(s16(l[2]), vshlq_n_s16(s16(c[2]), 1)), s16(r[2]));
  int16x8_t gy = vsubq_s16(bot, top);
  int32x4_t lo = vaddq_s32(vmull_s16(vget_low_s16(gx), vget_low_s16(gx)),
                           vmull_s16(vget_low_s16(gy), vget_low_s16(gy)));
  int32x4_t hi = vaddq_s32(vmull_high_s16(gx, gx), vmull_high_s16(gy, gy));
  mag[0] = vsqrtq_f32(vcvtq_f32_s32(lo));
  mag[1] = vsqrtq_f32(vcvtq_f32_s32(hi));
}

void sobel_row(const uint8_t *const rows_in[3], uint8_t *out, int w) {
  const uint8_t *rows[3] = {rows_in[0], rows_in[1], rows_in[2]};
  int x = 0;
  for (; x < 1 && x < w; ++x)
    out[x] = px::sobel(rows, w, x);
  for (; x + 16 <= w - 1; x += 16) {
    uint8x16_t l[3], c[3], r[3];
    for (int k = 0; k < 3; ++k) {
      l[k] = vld1q_u8(rows[k] + x - 1);
      c[k] = vld1q_u8(rows[k] + x);
      r[k] = vld1q_u8(rows[k] + x + 1);
    }
    uint8x8_t l8[3], c8[3], r8[3];
    float32x4_t mag[4];
    for (int k = 0; k < 3; ++k) {
      l8[k] = vget_low_u8(l[k]);
      c8[k] = vget_low_u8(c[k]);
      r8[k] = vget_low_u8(r[k]);
    }
    sobel8(l8, c8, r8, mag);
    for (int k = 0; k < 3; ++k) {
      l8[k] = vget_high_u8(l[k]);
      c8[k] = vget_high_u8(c[k]);
      r8[k] = vget_high_u8(r[k]);
    }
    sobel8(l8, c8, r8, mag + 2);
    vst1q_u8(out + x, narrow_u8(mag));
  }
  for (; x < w; ++x)
    out[x] = px::sobel(rows, w, x);
}

const RowKernels kKernels{gray_row, blur2d_row, blur_h_row, blur_v_row, sobel_row};

std::vector<std::string> variants(Op) { return {"neon"}; }

void run(Op op, const std::string &, const Options &, const Image &src, Image &out, Workspace &ws) {
  run_with_kernels(kKernels, op, src, out, ws);
}

} // namespace

const pagp::Backend &pagp::simd_backend() {
  static const Backend b{"simd", variants, run};
  return b;
}

#endif // __aarch64__
