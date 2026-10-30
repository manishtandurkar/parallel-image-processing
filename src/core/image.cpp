#include "pagp/image.hpp"

#include <cstring>

#include "stb_image.h"
#include "stb_image_write.h"

namespace pagp {

bool load_image(const std::string &path, Image &out) {
  int w, h, n;
  unsigned char *p = stbi_load(path.c_str(), &w, &h, &n, 3);
  if (!p)
    return false;
  out = Image(w, h, 3);
  std::memcpy(out.data.data(), p, out.data.size());
  stbi_image_free(p);
  return true;
}

bool save_image(const std::string &path, const Image &img) {
  return stbi_write_png(path.c_str(), img.w, img.h, img.c, img.data.data(), img.w * img.c) != 0;
}

static uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

Image generate_image(int w, int h, uint32_t seed) {
  Image img(w, h, 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Smooth background gradients.
      int r = w > 1 ? 255 * x / (w - 1) : 0;
      int g = h > 1 ? 255 * y / (h - 1) : 0;
      int b = 128;
      // Checkerboard blocks and a disc give sharp edges for Sobel.
      int cell = (w < h ? w : h) / 8 + 1;
      if (((x / cell) + (y / cell)) % 2 == 0)
        b = 220;
      long dx = x - w / 2, dy = y - h / 2, rad = (w < h ? w : h) / 3;
      if (dx * dx + dy * dy < rad * rad) {
        r = 255 - r;
        g = 255 - g;
      }
      // Low-amplitude deterministic noise so blur has something to smooth.
      int noise = int(hash32(uint32_t(y) * 73856093U ^ uint32_t(x) * 19349663U ^ seed) % 33) - 16;
      uint8_t *p = &img.data[(size_t(y) * w + x) * 3];
      auto clip = [](int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); };
      p[0] = clip(r + noise);
      p[1] = clip(g + noise);
      p[2] = clip(b + noise);
    }
  }
  return img;
}

} // namespace pagp
