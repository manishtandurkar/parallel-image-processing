// MPI backend: row-block decomposition with MPI_Scatterv / MPI_Gatherv and halo exchange.
// Uses the seq scalar row kernels, so speedup vs seq isolates the distribution.
#pragma once

#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

#include "pagp/cli.hpp"

namespace pagp::mpi {

// Every local buffer keeps this many spare rows above and below the owned rows: the largest halo
// any op needs (fused: blur radius 2 + sobel radius 1).
constexpr int kMargin = 3;

// Supported variants for an op; the first is the default.
std::vector<std::string> variants(Op op);

// Row decomposition and per-rank buffers for one image size. Built outside the timed region.
struct Plan {
  int w = 0, h = 0, rank = 0, nprocs = 1;
  int active = 1;         // ranks that own rows (each owns >= kMargin rows); the rest idle
  int y0 = 0, n = 0;      // this rank's owned rows [y0, y0 + n)
  int up = MPI_PROC_NULL; // neighbour owning the rows above (or MPI_PROC_NULL)
  int down = MPI_PROC_NULL;
  std::vector<int> rows, first;            // per rank: owned row count and first row
  std::vector<int> cnt1, dsp1, cnt3, dsp3; // Scatterv/Gatherv bytes for gray and RGB rows
  // Local buffers: (n + 2 * kMargin) rows, owned rows start at row kMargin. Only the ones the op
  // needs are allocated.
  std::vector<uint8_t> rgb, g, b, out;
  std::vector<float> f;
};

Plan make_plan(int w, int h, Op op, MPI_Comm comm);

// Runs op collectively: every rank calls it. src (gray, or RGB for op_takes_rgb ops) and out are
// only read/written on rank 0 and may be null elsewhere. Adds time spent in MPI calls to comm_ms.
void run(Op op, const std::string &variant, Plan &p, const uint8_t *src, uint8_t *out,
         double &comm_ms, MPI_Comm comm);

} // namespace pagp::mpi
