// Sequential reference. Plain loops over the per-pixel kernels in pixel.hpp, which fix the
// canonical arithmetic and summation order.
#include "pagp/seq.hpp"

#include "pagp/pixel.hpp"
#include "pagp/rows.hpp"

namespace pagp {
namespace seq {

static void gray_row(const uint8_t *rgb, uint8_t *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = px::gray(rgb + 3 * i);
}

// The row-pointer arrays are copied to locals: out is uint8_t* and may alias them, so the
// compiler would otherwise reload every pointer after each store.
static void blur2d_row(const uint8_t *const rows_in[kBlurSize], uint8_t *out, int w) {
  const uint8_t *rows[kBlurSize] = {rows_in[0], rows_in[1], rows_in[2], rows_in[3], rows_in[4]};
  for (int x = 0; x < w; ++x)
    out[x] = px::blur2d(rows, w, x);
}

static void blur_h_row(const uint8_t *in, float *out, int w) {
  for (int x = 0; x < w; ++x)
    out[x] = px::blur_h(in, w, x);
}

static void blur_v_row(const float *const rows_in[kBlurSize], uint8_t *out, int w) {
  const float *rows[kBlurSize] = {rows_in[0], rows_in[1], rows_in[2], rows_in[3], rows_in[4]};
  for (int x = 0; x < w; ++x)
    out[x] = px::blur_v(rows, x);
}

static void sobel_row(const uint8_t *const rows_in[3], uint8_t *out, int w) {
  const uint8_t *rows[3] = {rows_in[0], rows_in[1], rows_in[2]};
  for (int x = 0; x < w; ++x)
    out[x] = px::sobel(rows, w, x);
}

const RowKernels &row_kernels() {
  static const RowKernels k{gray_row, blur2d_row, blur_h_row, blur_v_row, sobel_row};
  return k;
}

void gray(const uint8_t *rgb, uint8_t *out, int w, int h) {
  gray_rows(row_kernels(), rgb, out, w, 0, h);
}

void blur2d(const uint8_t *in, uint8_t *out, int w, int h) {
  blur2d_rows(row_kernels(), in, out, w, h, 0, h);
}

void blur2d_tiled(const uint8_t *in, uint8_t *out, int w, int h, int tile) {
  for (int ty = 0; ty < h; ty += tile)
    for (int tx = 0; tx < w; tx += tile) {
      int ye = ty + tile < h ? ty + tile : h;
      int xe = tx + tile < w ? tx + tile : w;
      for (int y = ty; y < ye; ++y) {
        const uint8_t *rows[kBlurSize];
        for (int k = 0; k < kBlurSize; ++k)
          rows[k] = in + size_t(clamp_idx(y + k - kBlurRadius, h)) * w;
        for (int x = tx; x < xe; ++x)
          out[size_t(y) * w + x] = px::blur2d(rows, w, x);
      }
    }
}

void blur_sep(const uint8_t *in, uint8_t *out, float *tmp, int w, int h) {
  blur_h_rows(row_kernels(), in, tmp, w, 0, h);
  blur_v_rows(row_kernels(), tmp, out, w, h, 0, h);
}

void sobel(const uint8_t *in, uint8_t *out, int w, int h) {
  sobel_rows(row_kernels(), in, out, w, h, 0, h);
}

void pipeline(const uint8_t *rgb, uint8_t *out, Workspace &ws, int w, int h) {
  gray(rgb, ws.u8a.data(), w, h);
  blur_sep(ws.u8a.data(), ws.u8b.data(), ws.f32.data(), w, h);
  sobel(ws.u8b.data(), out, w, h);
}

void fused(const uint8_t *rgb, uint8_t *out, int w, int h) {
  fused_rows(row_kernels(), rgb, out, w, h, 0, h);
}

} // namespace seq
} // namespace pagp
