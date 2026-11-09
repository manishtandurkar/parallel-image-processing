// CUDA backend: one thread per output pixel, 2D thread blocks. The per-pixel arithmetic repeats
// pixel.hpp operation for operation (same summation order, round-to-nearest, clamp-to-edge), and
// the build uses --fmad=false, so results match seq exactly.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "cuda_backend.hpp"
#include "pagp/filters.hpp"

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err_ = (call);                                                                     \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "CUDA error at %s:%d: %s: %s\n", __FILE__, __LINE__, #call,             \
                   cudaGetErrorString(err_));                                                      \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

namespace pagp::cuda {

namespace {

constexpr int kB = kBlurSize, kR = kBlurRadius;

__constant__ float c_w1[kB];      // kGauss1D
__constant__ float c_w2[kB * kB]; // kGauss2D, row-major
__constant__ int c_sgx[9], c_sgy[9];

__device__ __forceinline__ int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

__device__ __forceinline__ uint8_t to_u8d(float v) {
  v = v + 0.5f;
  if (v < 0.0f)
    return 0;
  if (v >= 255.0f)
    return 255;
  return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint8_t gray_px(const uint8_t *p) {
  return to_u8d(kGrayR * p[0] + kGrayG * p[1] + kGrayB * p[2]);
}

__device__ __forceinline__ uint8_t sobel_mag(int gx, int gy) {
  return to_u8d(sqrtf(float(gx * gx + gy * gy)));
}

// Gaussian weight k from global memory (naive) or constant memory (const/tiled).
template <bool Const> __device__ __forceinline__ float w1(const float *g, int k) {
  return Const ? c_w1[k] : g[k];
}
template <bool Const> __device__ __forceinline__ float w2(const float *g, int k) {
  return Const ? c_w2[k] : g[k];
}

#define PIXEL_XY                                                                                   \
  int x = blockIdx.x * blockDim.x + threadIdx.x;                                                   \
  int y = blockIdx.y * blockDim.y + threadIdx.y;

// ---- naive / const: global memory only -------------------------------------------------------

__global__ void k_gray(const uint8_t *rgb, uint8_t *out, int w, int h) {
  PIXEL_XY
  if (x < w && y < h)
    out[size_t(y) * w + x] = gray_px(rgb + 3 * (size_t(y) * w + x));
}

template <bool Const>
__global__ void k_blur2d(const uint8_t *in, uint8_t *out, int w, int h, const float *gw2) {
  PIXEL_XY
  if (x >= w || y >= h)
    return;
  float sum = 0.0f;
#pragma unroll
  for (int ky = 0; ky < kB; ++ky) {
    const uint8_t *row = in + size_t(clampi(y + ky - kR, h)) * w;
#pragma unroll
    for (int kx = 0; kx < kB; ++kx)
      sum += w2<Const>(gw2, ky * kB + kx) * row[clampi(x + kx - kR, w)];
  }
  out[size_t(y) * w + x] = to_u8d(sum);
}

template <bool Const>
__global__ void k_blur_h(const uint8_t *in, float *tmp, int w, int h, const float *gw1) {
  PIXEL_XY
  if (x >= w || y >= h)
    return;
  const uint8_t *row = in + size_t(y) * w;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kB; ++k)
    sum += w1<Const>(gw1, k) * row[clampi(x + k - kR, w)];
  tmp[size_t(y) * w + x] = sum;
}

template <bool Const>
__global__ void k_blur_v(const float *tmp, uint8_t *out, int w, int h, const float *gw1) {
  PIXEL_XY
  if (x >= w || y >= h)
    return;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kB; ++k)
    sum += w1<Const>(gw1, k) * tmp[size_t(clampi(y + k - kR, h)) * w + x];
  out[size_t(y) * w + x] = to_u8d(sum);
}

__global__ void k_sobel(const uint8_t *in, uint8_t *out, int w, int h) {
  PIXEL_XY
  if (x >= w || y >= h)
    return;
  int gx = 0, gy = 0;
#pragma unroll
  for (int dy = 0; dy < 3; ++dy) {
    const uint8_t *row = in + size_t(clampi(y + dy - 1, h)) * w;
#pragma unroll
    for (int dx = 0; dx < 3; ++dx) {
      int v = row[clampi(x + dx - 1, w)];
      gx += c_sgx[dy * 3 + dx] * v;
      gy += c_sgy[dy * 3 + dx] * v;
    }
  }
  out[size_t(y) * w + x] = sobel_mag(gx, gy);
}

// ---- tiled: block loads its input tile + halo into shared memory once ------------------------
// Tile element (r, c) holds input (clamp(y0 + r - halo), clamp(x0 + c - halo)), so a thread's
// stencil reads tile (ty + k, tx + k) = input clamp(y + k - halo): the same values as naive.

#define TILE_SETUP                                                                                 \
  int tx = threadIdx.x, ty = threadIdx.y, bw = blockDim.x, bh = blockDim.y;                        \
  int x0 = blockIdx.x * bw, y0 = blockIdx.y * bh, x = x0 + tx, y = y0 + ty;                        \
  int tid = ty * bw + tx, nt = bw * bh;

__global__ void k_blur2d_tiled(const uint8_t *in, uint8_t *out, int w, int h) {
  extern __shared__ uint8_t sm[];
  TILE_SETUP
  int tw = bw + 2 * kR, th = bh + 2 * kR;
  for (int i = tid; i < tw * th; i += nt)
    sm[i] = in[size_t(clampi(y0 + i / tw - kR, h)) * w + clampi(x0 + i % tw - kR, w)];
  __syncthreads();
  if (x >= w || y >= h)
    return;
  float sum = 0.0f;
#pragma unroll
  for (int ky = 0; ky < kB; ++ky)
#pragma unroll
    for (int kx = 0; kx < kB; ++kx)
      sum += c_w2[ky * kB + kx] * sm[(ty + ky) * tw + tx + kx];
  out[size_t(y) * w + x] = to_u8d(sum);
}

__global__ void k_blur_h_tiled(const uint8_t *in, float *tmp, int w, int h) {
  extern __shared__ uint8_t sm[];
  TILE_SETUP
  int tw = bw + 2 * kR;
  for (int i = tid; i < tw * bh; i += nt) {
    int r = y0 + i / tw;
    if (r < h)
      sm[i] = in[size_t(r) * w + clampi(x0 + i % tw - kR, w)];
  }
  __syncthreads();
  if (x >= w || y >= h)
    return;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kB; ++k)
    sum += c_w1[k] * sm[ty * tw + tx + k];
  tmp[size_t(y) * w + x] = sum;
}

__global__ void k_blur_v_tiled(const float *tmp, uint8_t *out, int w, int h) {
  extern __shared__ float smf[];
  TILE_SETUP
  int th = bh + 2 * kR;
  for (int i = tid; i < bw * th; i += nt) {
    int c = x0 + i % bw;
    if (c < w)
      smf[i] = tmp[size_t(clampi(y0 + i / bw - kR, h)) * w + c];
  }
  __syncthreads();
  if (x >= w || y >= h)
    return;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kB; ++k)
    sum += c_w1[k] * smf[(ty + k) * bw + tx];
  out[size_t(y) * w + x] = to_u8d(sum);
}

__global__ void k_sobel_tiled(const uint8_t *in, uint8_t *out, int w, int h) {
  extern __shared__ uint8_t sm[];
  TILE_SETUP
  int tw = bw + 2;
  for (int i = tid; i < tw * (bh + 2); i += nt)
    sm[i] = in[size_t(clampi(y0 + i / tw - 1, h)) * w + clampi(x0 + i % tw - 1, w)];
  __syncthreads();
  if (x >= w || y >= h)
    return;
  int gx = 0, gy = 0;
#pragma unroll
  for (int dy = 0; dy < 3; ++dy)
#pragma unroll
    for (int dx = 0; dx < 3; ++dx) {
      int v = sm[(ty + dy) * tw + tx + dx];
      gx += c_sgx[dy * 3 + dx] * v;
      gy += c_sgy[dy * 3 + dx] * v;
    }
  out[size_t(y) * w + x] = sobel_mag(gx, gy);
}

// ---- fused: gray -> blur_h -> blur_v -> sobel in one kernel ----------------------------------
// Stages chain through clamped coordinates (sobel reads blurred pixel clamp(y-1), which blurs
// rows clamp(clamp(y-1)-2), ...), so each stage's buffer is indexed by the clamped global
// coordinate minus the buffer's origin, not by tile offsets.
//   G  gray   rows y0-3 .. y0+bh+2, cols x0-3 .. x0+bw+2
//   H  blur_h rows y0-3 .. y0+bh+2, cols x0-1 .. x0+bw   (float)
//   Bl blur   rows y0-1 .. y0+bh,   cols x0-1 .. x0+bw
constexpr int kFH = kR + 1; // fused halo: blur radius + sobel radius

size_t fused_smem(int bw, int bh) {
  return sizeof(float) * (bh + 2 * kFH) * (bw + 2) + size_t(bh + 2 * kFH) * (bw + 2 * kFH) +
         size_t(bh + 2) * (bw + 2);
}

__global__ void k_fused(const uint8_t *rgb, uint8_t *out, int w, int h) {
  extern __shared__ float smf[];
  TILE_SETUP
  int gw = bw + 2 * kFH, gh = bh + 2 * kFH, hw = bw + 2, bbw = bw + 2;
  float *H = smf;
  uint8_t *G = reinterpret_cast<uint8_t *>(H + gh * hw);
  uint8_t *Bl = G + gh * gw;
  int gy0 = y0 - kFH, gx0 = x0 - kFH, hx0 = x0 - 1, by0 = y0 - 1, bx0 = x0 - 1;

  for (int i = tid; i < gw * gh; i += nt) {
    int r = clampi(gy0 + i / gw, h), c = clampi(gx0 + i % gw, w);
    G[i] = gray_px(rgb + 3 * (size_t(r) * w + c));
  }
  __syncthreads();
  for (int i = tid; i < hw * gh; i += nt) {
    int r = gy0 + i / hw, c = hx0 + i % hw;
    if (r < 0 || r >= h || c < 0 || c >= w)
      continue; // never read: later stages only use in-image coordinates
    const uint8_t *grow = G + (r - gy0) * gw;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kB; ++k)
      sum += c_w1[k] * grow[clampi(c + k - kR, w) - gx0];
    H[i] = sum;
  }
  __syncthreads();
  for (int i = tid; i < bbw * (bh + 2); i += nt) {
    int r = by0 + i / bbw, c = bx0 + i % bbw;
    if (r < 0 || r >= h || c < 0 || c >= w)
      continue;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kB; ++k)
      sum += c_w1[k] * H[(clampi(r + k - kR, h) - gy0) * hw + (c - hx0)];
    Bl[i] = to_u8d(sum);
  }
  __syncthreads();
  if (x >= w || y >= h)
    return;
  int gx = 0, gy = 0;
#pragma unroll
  for (int dy = 0; dy < 3; ++dy) {
    const uint8_t *brow = Bl + (clampi(y + dy - 1, h) - by0) * bbw;
#pragma unroll
    for (int dx = 0; dx < 3; ++dx) {
      int v = brow[clampi(x + dx - 1, w) - bx0];
      gx += c_sgx[dy * 3 + dx] * v;
      gy += c_sgy[dy * 3 + dx] * v;
    }
  }
  out[size_t(y) * w + x] = sobel_mag(gx, gy);
}

void upload_constants() {
  static bool done = false;
  if (done)
    return;
  int sgx[9], sgy[9];
  for (int i = 0; i < 9; ++i) {
    sgx[i] = kSobelGx[i / 3][i % 3];
    sgy[i] = kSobelGy[i / 3][i % 3];
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_w1, kGauss1D, sizeof(kGauss1D)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_w2, kGauss2D.w, sizeof(kGauss2D.w)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sgx, sgx, sizeof(sgx)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sgy, sgy, sizeof(sgy)));
  done = true;
}

template <class T> T *dev_alloc(size_t n) {
  T *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

} // namespace

struct Context::Impl {
  int w = 0, h = 0;
  size_t in_bytes = 0;
  uint8_t *in = nullptr, *g = nullptr, *b = nullptr, *out = nullptr;
  float *f = nullptr, *gw1 = nullptr, *gw2 = nullptr;
  cudaEvent_t ev[4];
};

Context::Context(int w, int h, Op op) : impl(new Impl) {
  upload_constants();
  Impl &c = *impl;
  c.w = w;
  c.h = h;
  size_t n = size_t(w) * h;
  c.in_bytes = n * (op_takes_rgb(op) ? 3 : 1);
  c.in = dev_alloc<uint8_t>(c.in_bytes);
  c.out = dev_alloc<uint8_t>(n);
  if (op == Op::Pipeline) {
    c.g = dev_alloc<uint8_t>(n);
    c.b = dev_alloc<uint8_t>(n);
  }
  if (op == Op::BlurSep || op == Op::Pipeline)
    c.f = dev_alloc<float>(n);
  c.gw1 = dev_alloc<float>(kB);
  c.gw2 = dev_alloc<float>(kB * kB);
  CUDA_CHECK(cudaMemcpy(c.gw1, kGauss1D, sizeof(kGauss1D), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c.gw2, kGauss2D.w, sizeof(kGauss2D.w), cudaMemcpyHostToDevice));
  for (cudaEvent_t &e : c.ev)
    CUDA_CHECK(cudaEventCreate(&e));
}

Context::~Context() {
  Impl &c = *impl;
  for (void *p : {(void *)c.in, (void *)c.g, (void *)c.b, (void *)c.out, (void *)c.f, (void *)c.gw1,
                  (void *)c.gw2})
    cudaFree(p);
  for (cudaEvent_t e : c.ev)
    cudaEventDestroy(e);
}

std::vector<std::string> variants(Op op) {
  switch (op) {
  case Op::Gray:
    return {"naive"};
  case Op::Sobel:
    return {"naive", "tiled"}; // no weights, so no const variant
  case Op::Fused:
    return {"tiled"};
  default:
    return {"naive", "const", "tiled"};
  }
}

std::string device_name() {
  int dev = 0;
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  char buf[320];
  std::snprintf(buf, sizeof buf, "%s (sm_%d%d)", p.name, p.major, p.minor);
  return buf;
}

Timing run(Op op, const std::string &v, int bw, int bh, Context &ctx, const uint8_t *src,
           uint8_t *out) {
  Context::Impl &c = *ctx.impl;
  const int w = c.w, h = c.h;
  const dim3 block(bw, bh), grid((w + bw - 1) / bw, (h + bh - 1) / bh);
  const bool tiled = v == "tiled", cst = v == "const";

  auto blur_sep = [&](const uint8_t *in, uint8_t *o) {
    if (tiled) {
      k_blur_h_tiled<<<grid, block, size_t(bh) * (bw + 2 * kR)>>>(in, c.f, w, h);
      k_blur_v_tiled<<<grid, block, sizeof(float) * (bh + 2 * kR) * bw>>>(c.f, o, w, h);
    } else if (cst) {
      k_blur_h<true><<<grid, block>>>(in, c.f, w, h, c.gw1);
      k_blur_v<true><<<grid, block>>>(c.f, o, w, h, c.gw1);
    } else {
      k_blur_h<false><<<grid, block>>>(in, c.f, w, h, c.gw1);
      k_blur_v<false><<<grid, block>>>(c.f, o, w, h, c.gw1);
    }
  };
  auto sobel = [&](const uint8_t *in, uint8_t *o) {
    if (tiled)
      k_sobel_tiled<<<grid, block, size_t(bh + 2) * (bw + 2)>>>(in, o, w, h);
    else
      k_sobel<<<grid, block>>>(in, o, w, h);
  };

  CUDA_CHECK(cudaEventRecord(c.ev[0]));
  CUDA_CHECK(cudaMemcpy(c.in, src, c.in_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(c.ev[1]));
  switch (op) {
  case Op::Gray:
    k_gray<<<grid, block>>>(c.in, c.out, w, h);
    break;
  case Op::Blur2d:
    if (tiled)
      k_blur2d_tiled<<<grid, block, size_t(bh + 2 * kR) * (bw + 2 * kR)>>>(c.in, c.out, w, h);
    else if (cst)
      k_blur2d<true><<<grid, block>>>(c.in, c.out, w, h, c.gw2);
    else
      k_blur2d<false><<<grid, block>>>(c.in, c.out, w, h, c.gw2);
    break;
  case Op::BlurSep:
    blur_sep(c.in, c.out);
    break;
  case Op::Sobel:
    sobel(c.in, c.out);
    break;
  case Op::Pipeline:
    k_gray<<<grid, block>>>(c.in, c.g, w, h);
    blur_sep(c.g, c.b);
    sobel(c.b, c.out);
    break;
  case Op::Fused:
    k_fused<<<grid, block, fused_smem(bw, bh)>>>(c.in, c.out, w, h);
    break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(c.ev[2]));
  CUDA_CHECK(cudaMemcpy(out, c.out, size_t(w) * h, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(c.ev[3]));
  CUDA_CHECK(cudaEventSynchronize(c.ev[3]));

  Timing t;
  CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, c.ev[0], c.ev[1]));
  CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, c.ev[1], c.ev[2]));
  CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, c.ev[2], c.ev[3]));
  return t;
}

} // namespace pagp::cuda
