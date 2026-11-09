// MPI backend vs seq, run under mpirun with any rank count. Every op x variant x size must match
// seq exactly (MPI uses the seq row kernels; only the distribution differs). Sizes include images
// with fewer rows than ranks need, so some ranks own nothing.
#include <cstdio>

#include "mpi_backend.hpp"
#include "pagp/backend.hpp"
#include "pagp/validate.hpp"

int main(int argc, char **argv) {
  using namespace pagp;
  MPI_Init(&argc, &argv);
  int rank = 0, nprocs = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  static const int kSizes[][2] = {{1, 1},   {2, 3},     {3, 7},    {5, 5},   {16, 16},
                                  {64, 64}, {257, 131}, {640, 37}, {33, 300}};
  static const Op kOps[] = {Op::Gray, Op::Blur2d, Op::BlurSep, Op::Sobel, Op::Pipeline, Op::Fused};
  int failures = 0, cases = 0;
  for (auto &sz : kSizes) {
    // Every rank generates the (deterministic) input; only rank 0's copy is scattered.
    Image rgb = generate_image(sz[0], sz[1], 7);
    Image gray(rgb.w, rgb.h, 1);
    seq::gray(rgb.data.data(), gray.data.data(), rgb.w, rgb.h);
    Workspace ws;
    ws.reserve(rgb.w, rgb.h);
    for (Op op : kOps) {
      const Image &src = op_takes_rgb(op) ? rgb : gray;
      Image ref(src.w, src.h, 1);
      run_reference(op, src, ref, ws);
      for (const std::string &v : mpi::variants(op)) {
        mpi::Plan plan = mpi::make_plan(src.w, src.h, op, MPI_COMM_WORLD);
        Image out(src.w, src.h, 1);
        double comm_ms = 0;
        mpi::run(op, v, plan, src.data.data(), out.data.data(), comm_ms, MPI_COMM_WORLD);
        if (rank != 0)
          continue;
        int d = max_abs_diff(out, ref);
        ++cases;
        if (d != 0) {
          ++failures;
          std::printf("FAIL mpi np=%d %s/%s %dx%d max_abs_diff=%d\n", nprocs, op_name(op),
                      v.c_str(), src.w, src.h, d);
        }
      }
    }
  }
  if (rank == 0)
    std::printf("mpi np=%d: %d/%d cases passed (exact)\n", nprocs, cases - failures, cases);
  MPI_Finalize();
  return failures ? 1 : 0;
}
