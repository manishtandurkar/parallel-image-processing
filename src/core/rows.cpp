#include "pagp/rows.hpp"

#include <vector>

namespace pagp {

void gray_rows(const RowKernels &k, const uint8_t *rgb, uint8_t *out, int w, int y0, int y1) {
  size_t off = size_t(y0) * w;
  k.gray(rgb + 3 * off, out + off, w * (y1 - y0));
}

void blur2d_rows(const RowKernels &k, const uint8_t *in, uint8_t *out, int w, int h, int y0,
                 int y1) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t *rows[kBlurSize];
    for (int i = 0; i < kBlurSize; ++i)
      rows[i] = in + size_t(clamp_idx(y + i - kBlurRadius, h)) * w;
    k.blur2d(rows, out + size_t(y) * w, w);
  }
}

void blur_h_rows(const RowKernels &k, const uint8_t *in, float *tmp, int w, int y0, int y1) {
  for (int y = y0; y < y1; ++y)
    k.blur_h(in + size_t(y) * w, tmp + size_t(y) * w, w);
}

void blur_v_rows(const RowKernels &k, const float *tmp, uint8_t *out, int w, int h, int y0,
                 int y1) {
  for (int y = y0; y < y1; ++y) {
    const float *rows[kBlurSize];
    for (int i = 0; i < kBlurSize; ++i)
      rows[i] = tmp + size_t(clamp_idx(y + i - kBlurRadius, h)) * w;
    k.blur_v(rows, out + size_t(y) * w, w);
  }
}

void sobel_rows(const RowKernels &k, const uint8_t *in, uint8_t *out, int w, int h, int y0,
                int y1) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t *rows[3];
    for (int i = 0; i < 3; ++i)
      rows[i] = in + size_t(clamp_idx(y + i - 1, h)) * w;
    k.sobel(rows, out + size_t(y) * w, w);
  }
}

// Two ring buffers: horizontally blurred rows (float, 8 slots) and fully blurred rows (uint8,
// 4 slots). Rows are requested in non-decreasing windows of 5 and 3 consecutive indices, so a slot
// is never evicted while still in use.
void fused_rows(const RowKernels &k, const uint8_t *rgb, uint8_t *out, int w, int h, int y0,
                int y1) {
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
      k.gray(rgb + size_t(r) * w * 3, grow.data(), w);
      k.blur_h(grow.data(), dst, w);
      htag[s] = r;
    }
    return dst;
  };
  auto brow = [&](int r) -> const uint8_t * {
    int s = r % kBSlots;
    uint8_t *dst = bbuf.data() + size_t(s) * w;
    if (btag[s] != r) {
      const float *rows[kBlurSize];
      for (int i = 0; i < kBlurSize; ++i)
        rows[i] = hrow(clamp_idx(r + i - kBlurRadius, h));
      k.blur_v(rows, dst, w);
      btag[s] = r;
    }
    return dst;
  };

  for (int y = y0; y < y1; ++y) {
    const uint8_t *rows[3];
    for (int i = 0; i < 3; ++i)
      rows[i] = brow(clamp_idx(y + i - 1, h));
    k.sobel(rows, out + size_t(y) * w, w);
  }
}

void run_with_kernels(const RowKernels &k, Op op, const Image &src, Image &out, Workspace &ws) {
  const uint8_t *in = src.data.data();
  uint8_t *o = out.data.data();
  int w = src.w, h = src.h;
  switch (op) {
  case Op::Gray:
    gray_rows(k, in, o, w, 0, h);
    break;
  case Op::Blur2d:
    blur2d_rows(k, in, o, w, h, 0, h);
    break;
  case Op::BlurSep:
    blur_h_rows(k, in, ws.f32.data(), w, 0, h);
    blur_v_rows(k, ws.f32.data(), o, w, h, 0, h);
    break;
  case Op::Sobel:
    sobel_rows(k, in, o, w, h, 0, h);
    break;
  case Op::Pipeline:
    gray_rows(k, in, ws.u8a.data(), w, 0, h);
    blur_h_rows(k, ws.u8a.data(), ws.f32.data(), w, 0, h);
    blur_v_rows(k, ws.f32.data(), ws.u8b.data(), w, h, 0, h);
    sobel_rows(k, ws.u8b.data(), o, w, h, 0, h);
    break;
  case Op::Fused:
    fused_rows(k, in, o, w, h, 0, h);
    break;
  }
}

} // namespace pagp
