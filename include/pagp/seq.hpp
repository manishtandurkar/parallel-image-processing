// Sequential reference implementation: the source of truth every backend is validated against.
#pragma once

#include <cstdint>
#include <vector>

namespace pagp {

// Scratch buffers for multi-stage ops, sized once outside the timed region.
struct Workspace {
  std::vector<uint8_t> u8a, u8b;
  std::vector<float> f32;
  void reserve(int w, int h) {
    size_t n = size_t(w) * h;
    u8a.resize(n);
    u8b.resize(n);
    f32.resize(n);
  }
};

namespace seq {

void gray(const uint8_t *rgb, uint8_t *out, int w, int h);
void blur2d(const uint8_t *in, uint8_t *out, int w, int h);
// Same result as blur2d, computed in tile x tile blocks for the cache-locality study.
void blur2d_tiled(const uint8_t *in, uint8_t *out, int w, int h, int tile);
// Horizontal 1x5 into a float buffer (w*h), then vertical 5x1.
void blur_sep(const uint8_t *in, uint8_t *out, float *tmp, int w, int h);
void sobel(const uint8_t *in, uint8_t *out, int w, int h);
// gray -> blur_sep -> sobel as three full-image passes.
void pipeline(const uint8_t *rgb, uint8_t *out, Workspace &ws, int w, int h);
// Same result as pipeline in one streaming pass with small row buffers (no full intermediates).
void fused(const uint8_t *rgb, uint8_t *out, int w, int h);

} // namespace seq
} // namespace pagp
