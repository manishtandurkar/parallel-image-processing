// Scalar row kernels written for the auto-vectorizer: __restrict pointers, row pointers in locals,
// no clamping in the inner loops (border pixels use the scalar kernels from pixel.hpp).
// Same per-pixel arithmetic order as seq, so results are bit-identical.
#include "pagp/backend.hpp"
#include "pagp/pixel.hpp"
#include "pagp/rows.hpp"

namespace {

using namespace pagp;

void gray_row(const uint8_t *__restrict rgb, uint8_t *__restrict out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = to_u8(kGrayR * rgb[3 * i] + kGrayG * rgb[3 * i + 1] + kGrayB * rgb[3 * i + 2]);
}

void blur2d_row(const uint8_t *const rows[kBlurSize], uint8_t *__restrict out, int w) {
  const uint8_t *__restrict r0 = rows[0], *__restrict r1 = rows[1], *__restrict r2 = rows[2],
                            *__restrict r3 = rows[3], *__restrict r4 = rows[4];
  const uint8_t *r[kBlurSize] = {r0, r1, r2, r3, r4};
  if (w < kBlurSize) {
    for (int x = 0; x < w; ++x)
      out[x] = px::blur2d(r, w, x);
    return;
  }
  for (int x = 0; x < kBlurRadius; ++x) {
    out[x] = px::blur2d(r, w, x);
    out[w - 1 - x] = px::blur2d(r, w, w - 1 - x);
  }
  const auto &K = kGauss2D.w;
  for (int x = kBlurRadius; x < w - kBlurRadius; ++x) {
    float sum = 0.0f;
    sum += K[0][0] * r0[x - 2];
    sum += K[0][1] * r0[x - 1];
    sum += K[0][2] * r0[x];
    sum += K[0][3] * r0[x + 1];
    sum += K[0][4] * r0[x + 2];
    sum += K[1][0] * r1[x - 2];
    sum += K[1][1] * r1[x - 1];
    sum += K[1][2] * r1[x];
    sum += K[1][3] * r1[x + 1];
    sum += K[1][4] * r1[x + 2];
    sum += K[2][0] * r2[x - 2];
    sum += K[2][1] * r2[x - 1];
    sum += K[2][2] * r2[x];
    sum += K[2][3] * r2[x + 1];
    sum += K[2][4] * r2[x + 2];
    sum += K[3][0] * r3[x - 2];
    sum += K[3][1] * r3[x - 1];
    sum += K[3][2] * r3[x];
    sum += K[3][3] * r3[x + 1];
    sum += K[3][4] * r3[x + 2];
    sum += K[4][0] * r4[x - 2];
    sum += K[4][1] * r4[x - 1];
    sum += K[4][2] * r4[x];
    sum += K[4][3] * r4[x + 1];
    sum += K[4][4] * r4[x + 2];
    out[x] = to_u8(sum);
  }
}

void blur_h_row(const uint8_t *__restrict in, float *__restrict out, int w) {
  if (w < kBlurSize) {
    for (int x = 0; x < w; ++x)
      out[x] = px::blur_h(in, w, x);
    return;
  }
  for (int x = 0; x < kBlurRadius; ++x) {
    out[x] = px::blur_h(in, w, x);
    out[w - 1 - x] = px::blur_h(in, w, w - 1 - x);
  }
  const float g0 = kGauss1D[0], g1 = kGauss1D[1], g2 = kGauss1D[2], g3 = kGauss1D[3],
              g4 = kGauss1D[4];
  for (int x = kBlurRadius; x < w - kBlurRadius; ++x) {
    float sum = 0.0f;
    sum += g0 * in[x - 2];
    sum += g1 * in[x - 1];
    sum += g2 * in[x];
    sum += g3 * in[x + 1];
    sum += g4 * in[x + 2];
    out[x] = sum;
  }
}

void blur_v_row(const float *const rows[kBlurSize], uint8_t *__restrict out, int w) {
  const float *__restrict r0 = rows[0], *__restrict r1 = rows[1], *__restrict r2 = rows[2],
                          *__restrict r3 = rows[3], *__restrict r4 = rows[4];
  const float g0 = kGauss1D[0], g1 = kGauss1D[1], g2 = kGauss1D[2], g3 = kGauss1D[3],
              g4 = kGauss1D[4];
  for (int x = 0; x < w; ++x) {
    float sum = 0.0f;
    sum += g0 * r0[x];
    sum += g1 * r1[x];
    sum += g2 * r2[x];
    sum += g3 * r3[x];
    sum += g4 * r4[x];
    out[x] = to_u8(sum);
  }
}

void sobel_row(const uint8_t *const rows[3], uint8_t *__restrict out, int w) {
  const uint8_t *__restrict r0 = rows[0], *__restrict r1 = rows[1], *__restrict r2 = rows[2];
  const uint8_t *r[3] = {r0, r1, r2};
  if (w < 3) {
    for (int x = 0; x < w; ++x)
      out[x] = px::sobel(r, w, x);
    return;
  }
  out[0] = px::sobel(r, w, 0);
  out[w - 1] = px::sobel(r, w, w - 1);
  for (int x = 1; x < w - 1; ++x) {
    int gx = (r0[x + 1] - r0[x - 1]) + 2 * (r1[x + 1] - r1[x - 1]) + (r2[x + 1] - r2[x - 1]);
    int gy = (r2[x - 1] + 2 * r2[x] + r2[x + 1]) - (r0[x - 1] + 2 * r0[x] + r0[x + 1]);
    out[x] = px::sobel_mag(gx, gy);
  }
}

const RowKernels kKernels{gray_row, blur2d_row, blur_h_row, blur_v_row, sobel_row};

std::vector<std::string> variants(Op) { return {"autovec"}; }

void run(Op op, const std::string &, const Options &, const Image &src, Image &out, Workspace &ws) {
  run_with_kernels(kKernels, op, src, out, ws);
}

} // namespace

const pagp::Backend &pagp::autovec_backend() {
  static const Backend b{"autovec", variants, run};
  return b;
}
