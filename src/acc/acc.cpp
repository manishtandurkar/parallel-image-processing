// OpenACC backend: the seq loops over (y, x) with directives. The per-pixel arithmetic repeats
// pixel.hpp operation for operation (same summation order, round-to-nearest, clamp-to-edge) and
// the build disables FMA (-Mnofma, -gpu=nofma), so results match seq exactly.
#include "acc_backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _OPENACC
#include <openacc.h>
#endif

#include "pagp/filters.hpp"

namespace pagp::acc {

namespace {

constexpr int kB = kBlurSize, kR = kBlurRadius;
constexpr int kNumWeights = kB + kB * kB; // g[0..4] = kGauss1D, g[5..29] = kGauss2D row-major

enum class Mode { Parallel, Kernels, KernelsIndep, Copy };

// Per-pixel routines. Weights come from a device-resident array rather than the constexpr globals
// in filters.hpp, which would need `declare` directives to be visible on the device.

#pragma acc routine seq
inline int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

#pragma acc routine seq
inline uint8_t to_u8a(float v) { // pagp::to_u8
  v = v + 0.5f;
  if (v < 0.0f)
    return 0;
  if (v >= 255.0f)
    return 255;
  return static_cast<uint8_t>(v);
}

#pragma acc routine seq
inline uint8_t gray_px(const uint8_t *p) {
  return to_u8a(kGrayR * p[0] + kGrayG * p[1] + kGrayB * p[2]);
}

#pragma acc routine seq
inline uint8_t blur2d_px(const uint8_t *in, const float *g, int w, int h, int x, int y) {
  float sum = 0.0f;
  for (int ky = 0; ky < kB; ++ky) {
    const uint8_t *row = in + size_t(clampi(y + ky - kR, h)) * w;
    for (int kx = 0; kx < kB; ++kx)
      sum += g[kB + ky * kB + kx] * row[clampi(x + kx - kR, w)];
  }
  return to_u8a(sum);
}

#pragma acc routine seq
inline float blur_h_px(const uint8_t *row, const float *g, int w, int x) {
  float sum = 0.0f;
  for (int k = 0; k < kB; ++k)
    sum += g[k] * row[clampi(x + k - kR, w)];
  return sum;
}

#pragma acc routine seq
inline uint8_t blur_v_px(const float *tmp, const float *g, int w, int h, int x, int y) {
  float sum = 0.0f;
  for (int k = 0; k < kB; ++k)
    sum += g[k] * tmp[size_t(clampi(y + k - kR, h)) * w + x];
  return to_u8a(sum);
}

#pragma acc routine seq
inline uint8_t sobel_mag(int gx, int gy) { return to_u8a(sqrtf(float(gx * gx + gy * gy))); }

#pragma acc routine seq
inline uint8_t sobel_px(const uint8_t *in, int w, int h, int x, int y) {
  const uint8_t *r0 = in + size_t(clampi(y - 1, h)) * w, *r1 = in + size_t(y) * w,
                *r2 = in + size_t(clampi(y + 1, h)) * w;
  int xl = clampi(x - 1, w), xr = clampi(x + 1, w);
  // kSobelGx / kSobelGy written out as differences; integer sums are exact in any order. No unary
  // minus on a byte: nvc++ 26.11 narrows `-r0[xl]` to an i8 subtraction and then crashes in NVVM.
  int a00 = r0[xl], a01 = r0[x], a02 = r0[xr], a10 = r1[xl], a12 = r1[xr], a20 = r2[xl],
      a21 = r2[x], a22 = r2[xr];
  int gx = (a02 - a00) + 2 * (a12 - a10) + (a22 - a20);
  int gy = (a20 - a00) + 2 * (a21 - a01) + (a22 - a02);
  return sobel_mag(gx, gy);
}

// fused: blurred pixel (x, y) straight from RGB, recomputing gray and the horizontal pass.
#pragma acc routine seq
inline int blur_from_rgb(const uint8_t *rgb, const float *g, int w, int h, int x, int y) {
  float sum = 0.0f;
  for (int k = 0; k < kB; ++k) {
    const uint8_t *row = rgb + 3 * size_t(clampi(y + k - kR, h)) * w;
    float hs = 0.0f;
    for (int j = 0; j < kB; ++j)
      hs += g[j] * gray_px(row + 3 * clampi(x + j - kR, w));
    sum += g[k] * hs;
  }
  return to_u8a(sum);
}

#pragma acc routine seq
inline uint8_t fused_px(const uint8_t *rgb, const float *g, int w, int h, int x, int y) {
  int xl = clampi(x - 1, w), xr = clampi(x + 1, w), yu = clampi(y - 1, h), yd = clampi(y + 1, h);
  int b00 = blur_from_rgb(rgb, g, w, h, xl, yu), b01 = blur_from_rgb(rgb, g, w, h, x, yu),
      b02 = blur_from_rgb(rgb, g, w, h, xr, yu), b10 = blur_from_rgb(rgb, g, w, h, xl, y),
      b12 = blur_from_rgb(rgb, g, w, h, xr, y), b20 = blur_from_rgb(rgb, g, w, h, xl, yd),
      b21 = blur_from_rgb(rgb, g, w, h, x, yd), b22 = blur_from_rgb(rgb, g, w, h, xr, yd);
  int gx = (b02 - b00) + 2 * (b12 - b10) + (b22 - b20); // as in sobel_px
  int gy = (b20 - b00) + 2 * (b21 - b01) + (b22 - b02);
  return sobel_mag(gx, gy);
}

// One stage = a loop nest over every pixel, emitted once per mode. COPY_DIRECTIVE is the full
// directive for the copy variant (its data clauses differ per stage). Each stage function names
// its buffers in, o, g with sizes n_in, n.
#define PAGP_ACC_PRAGMA(x) _Pragma(#x)
#define PAGP_ACC_STAGE(mode, COPY_DIRECTIVE, BODY)                                                 \
  (void)n; /* sizes appear only in directives */                                                   \
  switch (mode) {                                                                                  \
  case Mode::Parallel:                                                                             \
    PAGP_ACC_PRAGMA(acc parallel loop collapse(2) default(present))                                \
    for (int y = 0; y < h; ++y)                                                                    \
      for (int x = 0; x < w; ++x) {                                                                \
        BODY;                                                                                      \
      }                                                                                            \
    break;                                                                                         \
  case Mode::Kernels:                                                                              \
    PAGP_ACC_PRAGMA(acc kernels default(present))                                                  \
    for (int y = 0; y < h; ++y)                                                                    \
      for (int x = 0; x < w; ++x) {                                                                \
        BODY;                                                                                      \
      }                                                                                            \
    break;                                                                                         \
  case Mode::KernelsIndep:                                                                         \
    PAGP_ACC_PRAGMA(acc kernels default(present))                                                  \
    PAGP_ACC_PRAGMA(acc loop independent collapse(2))                                              \
    for (int y = 0; y < h; ++y)                                                                    \
      for (int x = 0; x < w; ++x) {                                                                \
        BODY;                                                                                      \
      }                                                                                            \
    break;                                                                                         \
  case Mode::Copy:                                                                                 \
    PAGP_ACC_PRAGMA(COPY_DIRECTIVE)                                                                \
    for (int y = 0; y < h; ++y)                                                                    \
      for (int x = 0; x < w; ++x) {                                                                \
        BODY;                                                                                      \
      }                                                                                            \
    break;                                                                                         \
  }

#define PAGP_ACC_COPY(...) acc parallel loop collapse(2) __VA_ARGS__

void gray(Mode m, const uint8_t *__restrict in, uint8_t *__restrict o, int w, int h) {
  const size_t n = size_t(w) * h, n_in = 3 * n;
  (void)n_in;
  PAGP_ACC_STAGE(m, PAGP_ACC_COPY(copyin(in[0:n_in]) copyout(o[0:n])),
                 size_t i = size_t(y) * w + x;
                 o[i] = gray_px(in + 3 * i))
}

void blur2d(Mode m, const uint8_t *__restrict in, uint8_t *__restrict o, const float *__restrict g,
            int w, int h) {
  const size_t n = size_t(w) * h;
  PAGP_ACC_STAGE(m, PAGP_ACC_COPY(copyin(in[0:n], g[0:kNumWeights]) copyout(o[0:n])),
                 o[size_t(y) * w + x] = blur2d_px(in, g, w, h, x, y))
}

void blur_h(Mode m, const uint8_t *__restrict in, float *__restrict o, const float *__restrict g,
            int w, int h) {
  const size_t n = size_t(w) * h;
  PAGP_ACC_STAGE(m, PAGP_ACC_COPY(copyin(in[0:n], g[0:kNumWeights]) copyout(o[0:n])),
                 o[size_t(y) * w + x] = blur_h_px(in + size_t(y) * w, g, w, x))
}

void blur_v(Mode m, const float *__restrict in, uint8_t *__restrict o, const float *__restrict g,
            int w, int h) {
  const size_t n = size_t(w) * h;
  PAGP_ACC_STAGE(m, PAGP_ACC_COPY(copyin(in[0:n], g[0:kNumWeights]) copyout(o[0:n])),
                 o[size_t(y) * w + x] = blur_v_px(in, g, w, h, x, y))
}

void sobel(Mode m, const uint8_t *__restrict in, uint8_t *__restrict o, int w, int h) {
  const size_t n = size_t(w) * h;
  PAGP_ACC_STAGE(m, PAGP_ACC_COPY(copyin(in[0:n]) copyout(o[0:n])),
                 o[size_t(y) * w + x] = sobel_px(in, w, h, x, y))
}

void fused(Mode m, const uint8_t *__restrict in, uint8_t *__restrict o, const float *__restrict g,
           int w, int h) {
  const size_t n = size_t(w) * h, n_in = 3 * n;
  (void)n_in;
  PAGP_ACC_STAGE(m, PAGP_ACC_COPY(copyin(in[0:n_in], g[0:kNumWeights]) copyout(o[0:n])),
                 o[size_t(y) * w + x] = fused_px(in, g, w, h, x, y))
}

Mode parse_mode(const std::string &v) {
  if (v == "kernels")
    return Mode::Kernels;
  if (v == "kernels_indep")
    return Mode::KernelsIndep;
  return v == "copy" ? Mode::Copy : Mode::Parallel;
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

struct Context::Impl {
  int w = 0, h = 0;
  bool resident = false; // device data region open (parallel / kernels)
  size_t n = 0, n_in = 0;
  uint8_t *in = nullptr, *out = nullptr, *a = nullptr, *b = nullptr;
  float *tmp = nullptr, *g = nullptr;
};

Context::Context(int w, int h, Op op, const std::string &variant) : impl(new Impl) {
  Impl &c = *impl;
  c.w = w;
  c.h = h;
  c.resident = parse_mode(variant) != Mode::Copy;
  c.n = size_t(w) * h;
  c.n_in = c.n * (op_takes_rgb(op) ? 3 : 1);
  c.in = new uint8_t[c.n_in];
  c.out = new uint8_t[c.n];
  if (op == Op::Pipeline) {
    c.a = new uint8_t[c.n];
    c.b = new uint8_t[c.n];
  }
  if (op == Op::BlurSep || op == Op::Pipeline)
    c.tmp = new float[c.n];
  c.g = new float[kNumWeights];
  for (int i = 0; i < kB; ++i)
    c.g[i] = kGauss1D[i];
  for (int i = 0; i < kB; ++i)
    for (int j = 0; j < kB; ++j)
      c.g[kB + i * kB + j] = kGauss2D.w[i][j];

  if (!c.resident)
    return;
  uint8_t *in = c.in, *out = c.out, *a = c.a, *b = c.b;
  float *tmp = c.tmp, *g = c.g;
  size_t n = c.n, n_in = c.n_in;
#pragma acc enter data create(in[0:n_in], out[0:n]) copyin(g[0:kNumWeights])
  if (a) {
#pragma acc enter data create(a[0:n], b[0:n])
  }
  if (tmp) {
#pragma acc enter data create(tmp[0:n])
  }
  (void)in, (void)out, (void)a, (void)b, (void)tmp, (void)g, (void)n, (void)n_in;
}

Context::~Context() {
  Impl &c = *impl;
  if (c.resident) {
    uint8_t *in = c.in, *out = c.out, *a = c.a, *b = c.b;
    float *tmp = c.tmp, *g = c.g;
    size_t n = c.n, n_in = c.n_in;
    (void)in, (void)out, (void)a, (void)b, (void)tmp, (void)g, (void)n, (void)n_in;
    if (tmp) {
#pragma acc exit data delete(tmp[0:n])
    }
    if (a) {
#pragma acc exit data delete(a[0:n], b[0:n])
    }
#pragma acc exit data delete(in[0:n_in], out[0:n], g[0:kNumWeights])
  }
  delete[] c.in;
  delete[] c.out;
  delete[] c.a;
  delete[] c.b;
  delete[] c.tmp;
  delete[] c.g;
}

std::vector<std::string> variants(Op) {
  return {"parallel", "kernels", "kernels_indep", "copy"};
}

#ifdef _OPENACC
static bool on_host() { return acc_get_device_type() == acc_device_host; }
#else
static bool on_host() { return true; }
#endif

const char *backend_name() { return on_host() ? "acc_mc" : "acc"; }

int workers() {
  if (!on_host())
    return 1;
#ifdef _OPENACC
  if (const char *s = std::getenv("ACC_NUM_CORES"))
    return std::atoi(s);
  return int(std::thread::hardware_concurrency());
#else
  return 1;
#endif
}

std::string device_name() {
#ifdef _OPENACC
  if (!on_host()) {
    acc_device_t t = acc_get_device_type();
    const char *name = acc_get_property_string(acc_get_device_num(t), t, acc_property_name);
    return name ? name : "unknown GPU";
  }
  return "host multicore (" + std::to_string(workers()) + " threads)";
#else
  return "host, no OpenACC (sequential)";
#endif
}

Timing run(Op op, const std::string &variant, Context &ctx, const uint8_t *src, uint8_t *dst) {
  Context::Impl &c = *ctx.impl;
  const Mode m = parse_mode(variant);
  const int w = c.w, h = c.h;
  uint8_t *in = c.in, *out = c.out, *a = c.a, *b = c.b;
  float *tmp = c.tmp, *g = c.g;
  const size_t n = c.n, n_in = c.n_in;
  (void)n;
  std::memcpy(in, src, n_in); // host staging, untimed

  Timing t;
  double t0 = now_ms();
  if (c.resident) {
#pragma acc update device(in[0:n_in])
  }
  double t1 = now_ms();
  switch (op) { // compute regions are synchronous (no async clauses)
  case Op::Gray:
    gray(m, in, out, w, h);
    break;
  case Op::Blur2d:
    blur2d(m, in, out, g, w, h);
    break;
  case Op::BlurSep:
    blur_h(m, in, tmp, g, w, h);
    blur_v(m, tmp, out, g, w, h);
    break;
  case Op::Sobel:
    sobel(m, in, out, w, h);
    break;
  case Op::Pipeline:
    gray(m, in, a, w, h);
    blur_h(m, a, tmp, g, w, h);
    blur_v(m, tmp, b, g, w, h);
    sobel(m, b, out, w, h);
    break;
  case Op::Fused:
    fused(m, in, out, g, w, h);
    break;
  }
  double t2 = now_ms();
  if (c.resident) {
#pragma acc update self(out[0:n])
  }
  double t3 = now_ms();
  std::memcpy(dst, out, n);

  if (c.resident) {
    t.h2d_ms = t1 - t0;
    t.kernel_ms = t2 - t1;
    t.d2h_ms = t3 - t2;
  } else {
    t.h2d_ms = t.d2h_ms = -1;
    t.kernel_ms = t3 - t0;
  }
  return t;
}

} // namespace pagp::acc
