// Row kernels and the image-level drivers built on them. A backend that only changes how a row is
// computed (seq, autovec, NEON) supplies a RowKernels table; backends that change how rows are
// distributed (OpenMP) call the drivers on row ranges.
#pragma once

#include <cstdint>

#include "pagp/cli.hpp"
#include "pagp/filters.hpp"
#include "pagp/image.hpp"
#include "pagp/seq.hpp"

namespace pagp {

struct RowKernels {
  // n consecutive RGB pixels -> n gray pixels.
  void (*gray)(const uint8_t *rgb, uint8_t *out, int n);
  // rows[k] = input row clamp(y + k - 2).
  void (*blur2d)(const uint8_t *const rows[kBlurSize], uint8_t *out, int w);
  void (*blur_h)(const uint8_t *in, float *out, int w);
  // rows[k] = horizontally blurred row clamp(y + k - 2).
  void (*blur_v)(const float *const rows[kBlurSize], uint8_t *out, int w);
  // rows[k] = input row clamp(y + k - 1).
  void (*sobel)(const uint8_t *const rows[3], uint8_t *out, int w);
};

// Drivers over output rows [y0, y1) of a w x h image.
void gray_rows(const RowKernels &k, const uint8_t *rgb, uint8_t *out, int w, int y0, int y1);
void blur2d_rows(const RowKernels &k, const uint8_t *in, uint8_t *out, int w, int h, int y0,
                 int y1);
void blur_h_rows(const RowKernels &k, const uint8_t *in, float *tmp, int w, int y0, int y1);
void blur_v_rows(const RowKernels &k, const float *tmp, uint8_t *out, int w, int h, int y0, int y1);
void sobel_rows(const RowKernels &k, const uint8_t *in, uint8_t *out, int w, int h, int y0, int y1);
// gray -> blur_sep -> sobel for rows [y0, y1) in one streaming pass with small ring buffers.
// Rows just outside the range are recomputed, so disjoint ranges can run in parallel.
void fused_rows(const RowKernels &k, const uint8_t *rgb, uint8_t *out, int w, int h, int y0,
                int y1);

// Runs any op single-threaded over the whole image with the given kernels.
void run_with_kernels(const RowKernels &k, Op op, const Image &src, Image &out, Workspace &ws);

} // namespace pagp
