// Seq reference unit tests on hand-checkable inputs, plus variant consistency (tiled, fused).
#include <cstdio>

#include "check_backend.hpp"
#include "pagp/filters.hpp"

using namespace pagp;

static int g_failures = 0;
#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::printf("CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond);                          \
      ++g_failures;                                                                                \
    }                                                                                              \
  } while (0)

static void test_weights() {
  float s1 = 0, s2 = 0;
  for (int i = 0; i < kBlurSize; ++i) {
    s1 += kGauss1D[i];
    for (int j = 0; j < kBlurSize; ++j)
      s2 += kGauss2D.w[i][j];
  }
  CHECK(s1 > 0.9999f && s1 < 1.0001f);
  CHECK(s2 > 0.9999f && s2 < 1.0001f);
  CHECK(kGauss1D[0] == kGauss1D[4] && kGauss1D[1] == kGauss1D[3]);
}

static void test_gray() {
  // Pure red, green, blue, white, black.
  const uint8_t rgb[] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0};
  uint8_t out[5];
  seq::gray(rgb, out, 5, 1);
  CHECK(out[0] == 76);  // 76.245
  CHECK(out[1] == 150); // 149.685
  CHECK(out[2] == 29);  // 29.07
  CHECK(out[3] == 255);
  CHECK(out[4] == 0);
}

static void test_constant_image() {
  // Blur of a constant image is the same constant (clamp-to-edge); Sobel of it is 0.
  const int w = 13, h = 9;
  std::vector<uint8_t> in(w * h, 100), out(w * h), out2(w * h);
  std::vector<float> tmp(w * h);
  seq::blur2d(in.data(), out.data(), w, h);
  seq::blur_sep(in.data(), out2.data(), tmp.data(), w, h);
  for (int i = 0; i < w * h; ++i) {
    CHECK(out[i] == 100);
    CHECK(out2[i] == 100);
  }
  seq::sobel(in.data(), out.data(), w, h);
  for (int i = 0; i < w * h; ++i)
    CHECK(out[i] == 0);
}

static void test_sobel_step() {
  // Vertical step edge 0 | 100 between columns 3 and 4: |Gx| = 4*100 = 400 -> 255 at the edge,
  // 0 elsewhere. Rows are identical so Gy = 0.
  const int w = 8, h = 4;
  std::vector<uint8_t> in(w * h), out(w * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      in[y * w + x] = x < 4 ? 0 : 100;
  seq::sobel(in.data(), out.data(), w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      CHECK(out[y * w + x] == ((x == 3 || x == 4) ? 255 : 0));

  // Smaller step (0 | 40): magnitude 160, not saturated.
  for (auto &v : in)
    v = v ? 40 : 0;
  seq::sobel(in.data(), out.data(), w, h);
  CHECK(out[3] == 160 && out[4] == 160 && out[0] == 0 && out[7] == 0);
}

static void test_blur_impulse() {
  // A single 255 impulse in the middle reproduces the kernel scaled by 255.
  const int w = 9, h = 9;
  std::vector<uint8_t> in(w * h, 0), out(w * h);
  in[4 * w + 4] = 255;
  seq::blur2d(in.data(), out.data(), w, h);
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx)
      CHECK(out[(4 + dy) * w + 4 + dx] == to_u8(255.0f * kGauss2D.w[dy + 2][dx + 2]));
  CHECK(out[0] == 0);
}

int main() {
  test_weights();
  test_gray();
  test_constant_image();
  test_sobel_step();
  test_blur_impulse();
  // tiled blur2d and fused must be bit-identical to the reference.
  g_failures += check_backend(seq_backend(), /*exact=*/true);
  if (g_failures)
    std::printf("%d failure(s)\n", g_failures);
  else
    std::printf("all seq tests passed\n");
  return g_failures ? 1 : 0;
}
