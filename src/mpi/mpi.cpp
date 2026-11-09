// MPI backend: rank 0 scatters row blocks (MPI_Scatterv), each rank receives `radius` halo rows
// from its neighbours, computes its rows with the seq row kernels, and rank 0 gathers the result
// (MPI_Gatherv). Ranks at the image edge get no halo there: the row drivers clamp to the local
// buffer, whose first/last row is then the image's first/last row, which is clamp-to-edge.
//
// Variants:
//   blocking  MPI_Sendrecv halo exchange, then compute all owned rows
//   overlap   MPI_Isend/MPI_Irecv, compute rows that need no halo, MPI_Waitall, compute the rest
//
// Halo radius per exchange: blur2d/blur_sep 2, sobel 1, pipeline 2 (gray) then 1 (blurred),
// fused 3 (RGB rows, no second exchange: the extra rows are recomputed locally).
#include "mpi_backend.hpp"

#include <algorithm>

#include "pagp/rows.hpp"
#include "pagp/seq.hpp"

namespace pagp::mpi {

namespace {

constexpr int kTagUp = 0;   // rows sent to the rank above (it stores them as its bottom halo)
constexpr int kTagDown = 1; // rows sent to the rank below (its top halo)

int halo_top(const Plan &p, int r) { return p.up != MPI_PROC_NULL ? r : 0; }
int halo_bottom(const Plan &p, int r) { return p.down != MPI_PROC_NULL ? r : 0; }

// Counts and displacements (bytes) for Scatterv/Gatherv of `ch`-channel rows.
void counts(const Plan &p, int ch, std::vector<int> &cnt, std::vector<int> &dsp) {
  cnt.resize(p.nprocs);
  dsp.resize(p.nprocs);
  for (int r = 0; r < p.nprocs; ++r) {
    cnt[r] = p.rows[r] * p.w * ch;
    dsp[r] = p.first[r] * p.w * ch;
  }
}

} // namespace

std::vector<std::string> variants(Op op) {
  if (op == Op::Gray) // no halo, so nothing to overlap
    return {"blocking"};
  return {"blocking", "overlap"};
}

Plan make_plan(int w, int h, Op op, MPI_Comm comm) {
  Plan p;
  p.w = w;
  p.h = h;
  MPI_Comm_rank(comm, &p.rank);
  MPI_Comm_size(comm, &p.nprocs);
  // Every active rank owns at least kMargin rows, so a halo always comes from the next rank only.
  p.active = std::min(p.nprocs, std::max(1, h / kMargin));
  p.rows.assign(p.nprocs, 0);
  p.first.assign(p.nprocs, h);
  for (int r = 0, y = 0; r < p.active; ++r) {
    p.rows[r] = h / p.active + (r < h % p.active ? 1 : 0);
    p.first[r] = y;
    y += p.rows[r];
  }
  p.y0 = p.first[p.rank];
  p.n = p.rows[p.rank];
  if (p.rank < p.active) {
    if (p.rank > 0)
      p.up = p.rank - 1;
    if (p.rank + 1 < p.active)
      p.down = p.rank + 1;
  }
  counts(p, 1, p.cnt1, p.dsp1);
  counts(p, 3, p.cnt3, p.dsp3);

  size_t px = size_t(p.n + 2 * kMargin) * w;
  if (op_takes_rgb(op))
    p.rgb.resize(3 * px);
  if (op == Op::Blur2d || op == Op::BlurSep || op == Op::Sobel || op == Op::Pipeline)
    p.g.resize(px);
  if (op == Op::BlurSep || op == Op::Pipeline)
    p.f.resize(px);
  if (op == Op::Pipeline)
    p.b.resize(px);
  p.out.resize(px);
  return p;
}

void run(Op op, const std::string &variant, Plan &p, const uint8_t *src, uint8_t *out,
         double &comm_ms, MPI_Comm comm) {
  const RowKernels &k = seq::row_kernels();
  const bool overlap = variant == "overlap";
  const int w = p.w, n = p.n;
  MPI_Request req[4];

  auto timed = [&](auto f) {
    double t0 = MPI_Wtime();
    f();
    comm_ms += (MPI_Wtime() - t0) * 1e3;
  };

  // Starts (overlap) or completes (blocking) the exchange of r halo rows of `rowbytes` bytes in a
  // buffer laid out with kMargin spare rows on each side.
  auto halo = [&](uint8_t *buf, size_t rowbytes, int r) {
    uint8_t *top_halo = buf + (kMargin - r) * rowbytes;
    uint8_t *own_top = buf + kMargin * rowbytes;
    uint8_t *own_bottom = buf + (kMargin + n - r) * rowbytes;
    uint8_t *bottom_halo = buf + (kMargin + n) * rowbytes;
    int bytes = int(r * rowbytes);
    timed([&] {
      if (overlap) {
        MPI_Irecv(top_halo, bytes, MPI_BYTE, p.up, kTagDown, comm, &req[0]);
        MPI_Irecv(bottom_halo, bytes, MPI_BYTE, p.down, kTagUp, comm, &req[1]);
        MPI_Isend(own_top, bytes, MPI_BYTE, p.up, kTagUp, comm, &req[2]);
        MPI_Isend(own_bottom, bytes, MPI_BYTE, p.down, kTagDown, comm, &req[3]);
      } else {
        MPI_Sendrecv(own_top, bytes, MPI_BYTE, p.up, kTagUp, bottom_halo, bytes, MPI_BYTE, p.down,
                     kTagUp, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(own_bottom, bytes, MPI_BYTE, p.down, kTagDown, top_halo, bytes, MPI_BYTE, p.up,
                     kTagDown, comm, MPI_STATUS_IGNORE);
      }
    });
  };
  auto wait = [&] {
    if (overlap)
      timed([&] { MPI_Waitall(4, req, MPI_STATUSES_IGNORE); });
  };

  // Computes owned rows [ht, ht + n) of a radius-r stencil with compute(y0, y1) in local rows.
  // overlap: rows that need no halo first, then wait, then the boundary rows.
  auto stencil = [&](int ht, int hb, auto compute) {
    int lo = ht, hi = ht + n;
    if (!overlap) {
      compute(lo, hi);
      return;
    }
    int a = std::min(lo + ht, hi), b = std::max(hi - hb, a);
    compute(a, b);
    wait();
    compute(lo, a);
    compute(b, hi);
  };

  // blur_sep on local buffers g -> o with a radius-2 halo already started/exchanged on g.
  auto blur_sep = [&](uint8_t *g, uint8_t *o) {
    int ht = halo_top(p, kBlurRadius), hb = halo_bottom(p, kBlurRadius), lh = ht + n + hb;
    const uint8_t *gv = g + size_t(kMargin - ht) * w;
    float *fv = p.f.data() + size_t(kMargin - ht) * w;
    uint8_t *ov = o + size_t(kMargin - ht) * w;
    if (!overlap) {
      blur_h_rows(k, gv, fv, w, 0, lh);
      blur_v_rows(k, fv, ov, w, lh, ht, ht + n);
      return;
    }
    int a = std::min(2 * ht, ht + n), b = std::max(ht + n - hb, a);
    blur_h_rows(k, gv, fv, w, ht, ht + n);
    blur_v_rows(k, fv, ov, w, lh, a, b);
    wait();
    blur_h_rows(k, gv, fv, w, 0, ht);
    blur_h_rows(k, gv, fv, w, ht + n, lh);
    blur_v_rows(k, fv, ov, w, lh, ht, a);
    blur_v_rows(k, fv, ov, w, lh, b, ht + n);
  };

  const bool rgb_in = op_takes_rgb(op);
  uint8_t *in = rgb_in ? p.rgb.data() : p.g.data();
  size_t in_row = size_t(w) * (rgb_in ? 3 : 1);
  timed([&] {
    MPI_Scatterv(src, rgb_in ? p.cnt3.data() : p.cnt1.data(),
                 rgb_in ? p.dsp3.data() : p.dsp1.data(), MPI_BYTE, in + kMargin * in_row,
                 int(n * in_row), MPI_BYTE, 0, comm);
  });

  if (n > 0) {
    uint8_t *o = p.out.data();
    switch (op) {
    case Op::Gray:
      gray_rows(k, in + kMargin * in_row, o + size_t(kMargin) * w, w, 0, n);
      break;
    case Op::Blur2d:
    case Op::Sobel: {
      int r = op == Op::Blur2d ? kBlurRadius : kSobelRadius;
      int ht = halo_top(p, r), hb = halo_bottom(p, r), lh = ht + n + hb;
      const uint8_t *iv = in + size_t(kMargin - ht) * w;
      uint8_t *ov = o + size_t(kMargin - ht) * w;
      halo(in, w, r);
      stencil(ht, hb, [&](int y0, int y1) {
        if (op == Op::Blur2d)
          blur2d_rows(k, iv, ov, w, lh, y0, y1);
        else
          sobel_rows(k, iv, ov, w, lh, y0, y1);
      });
      break;
    }
    case Op::BlurSep:
      halo(in, w, kBlurRadius);
      blur_sep(in, o);
      break;
    case Op::Pipeline: {
      uint8_t *g = p.g.data(), *b = p.b.data();
      gray_rows(k, in + kMargin * in_row, g + size_t(kMargin) * w, w, 0, n);
      halo(g, w, kBlurRadius);
      blur_sep(g, b);
      int ht = halo_top(p, kSobelRadius), hb = halo_bottom(p, kSobelRadius), lh = ht + n + hb;
      const uint8_t *bv = b + size_t(kMargin - ht) * w;
      uint8_t *ov = o + size_t(kMargin - ht) * w;
      halo(b, w, kSobelRadius);
      stencil(ht, hb, [&](int y0, int y1) { sobel_rows(k, bv, ov, w, lh, y0, y1); });
      break;
    }
    case Op::Fused: {
      int r = kBlurRadius + kSobelRadius;
      int ht = halo_top(p, r), hb = halo_bottom(p, r), lh = ht + n + hb;
      const uint8_t *iv = in + size_t(kMargin - ht) * in_row;
      uint8_t *ov = o + size_t(kMargin - ht) * w;
      halo(in, in_row, r);
      stencil(ht, hb, [&](int y0, int y1) {
        if (y0 < y1)
          fused_rows(k, iv, ov, w, lh, y0, y1);
      });
      break;
    }
    }
  }

  timed([&] {
    MPI_Gatherv(p.out.data() + size_t(kMargin) * w, n * w, MPI_BYTE, out, p.cnt1.data(),
                p.dsp1.data(), MPI_BYTE, 0, comm);
  });
}

} // namespace pagp::mpi
