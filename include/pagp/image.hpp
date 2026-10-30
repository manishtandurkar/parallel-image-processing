// Image container, load/save (wraps stb), deterministic test-image generator.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pagp {

// Interleaved, row-major, 8 bits per channel. c = 3 (RGB) or 1 (gray).
struct Image {
  int w = 0, h = 0, c = 0;
  std::vector<uint8_t> data;

  Image() = default;
  Image(int w_, int h_, int c_) : w(w_), h(h_), c(c_), data(size_t(w_) * h_ * c_) {}
  size_t pixels() const { return size_t(w) * h; }
};

// Loads any stb-supported file as RGB. Returns false on failure.
bool load_image(const std::string &path, Image &out);
// Writes a PNG with img.c channels. Returns false on failure.
bool save_image(const std::string &path, const Image &img);
// Deterministic synthetic RGB image (gradients, shapes with sharp edges, noise).
Image generate_image(int w, int h, uint32_t seed = 1);

} // namespace pagp
