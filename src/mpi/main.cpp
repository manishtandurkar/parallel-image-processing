// pagp_mpi driver: same CLI as the other backends, run as `mpirun -np N pagp_mpi ...`.
// Its own driver (not run_main) because every rank takes part in each repetition and timing is
// collective: each rep starts after MPI_Barrier, and the reported time is the slowest rank's
// (MPI_Reduce MAX). comm_ms is the slowest rank's time inside MPI calls (scatter, halo exchange,
// gather), so it includes waiting for other ranks. Only rank 0 loads, validates, saves and
// writes CSV.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "mpi_backend.hpp"
#include "pagp/backend.hpp"
#include "pagp/csv.hpp"
#include "pagp/timer.hpp"
#include "pagp/validate.hpp"

namespace {

using namespace pagp;

int drive(int argc, char **argv) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0, nprocs = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  const bool root = rank == 0;

  Options o;
  if (!parse_cli(argc, argv, o))
    return 2;
  std::vector<std::string> vs = mpi::variants(o.op);
  std::string variant = o.variant.empty() ? vs.front() : o.variant;
  if (std::find(vs.begin(), vs.end(), variant) == vs.end()) {
    if (root)
      std::fprintf(stderr, "error: backend mpi has no variant '%s' for op %s\n", variant.c_str(),
                   op_name(o.op));
    return 2;
  }

  // Rank 0 holds the full input; the others only learn its size.
  Image src;
  int dims[3] = {0, 0, 1}; // w, h, ok
  if (root) {
    Image rgb;
    if (!o.input.empty())
      dims[2] = load_image(o.input, rgb);
    else
      rgb = generate_image(o.gen_w, o.gen_h);
    if (dims[2]) {
      if (op_takes_rgb(o.op)) {
        src = std::move(rgb);
      } else {
        src = Image(rgb.w, rgb.h, 1);
        seq::gray(rgb.data.data(), src.data.data(), rgb.w, rgb.h);
      }
      dims[0] = src.w;
      dims[1] = src.h;
    } else {
      std::fprintf(stderr, "error: cannot load '%s'\n", o.input.c_str());
    }
  }
  MPI_Bcast(dims, 3, MPI_INT, 0, comm);
  if (!dims[2])
    return 1;
  const int w = dims[0], h = dims[1];

  mpi::Plan plan = mpi::make_plan(w, h, o.op, comm);
  Image out;
  if (root)
    out = Image(w, h, 1);

  std::vector<double> total_ms, comm_ms;
  for (int i = 0; i < o.warmup + o.reps; ++i) {
    double c = 0;
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    mpi::run(o.op, variant, plan, root ? src.data.data() : nullptr,
             root ? out.data.data() : nullptr, c, comm);
    double local[2] = {(MPI_Wtime() - t0) * 1e3, c}, slowest[2] = {0, 0};
    MPI_Reduce(local, slowest, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (i >= o.warmup) {
      total_ms.push_back(slowest[0]);
      comm_ms.push_back(slowest[1]);
    }
  }
  if (!root)
    return 0;

  TimingStats t = summarize(total_ms), tc = summarize(comm_ms);
  int diff = -1;
  if (o.validate) {
    Workspace ws;
    ws.reserve(w, h);
    Image ref(w, h, 1);
    run_reference(o.op, src, ref, ws);
    diff = max_abs_diff(out, ref);
  }

  std::printf("mpi %s/%s %dx%d ranks=%d reps=%d median=%.3f ms min=%.3f ms comm=%.3f ms",
              op_name(o.op), variant.c_str(), w, h, nprocs, o.reps, t.median_ms, t.min_ms,
              tc.median_ms);
  if (diff >= 0)
    std::printf(" max_abs_diff=%d", diff);
  std::printf("\n");

  if (!o.save.empty() && !save_image(o.save, out)) {
    std::fprintf(stderr, "error: cannot write '%s'\n", o.save.c_str());
    return 1;
  }
  if (!o.csv.empty()) {
    ResultRow r;
    r.machine = o.machine;
    r.backend = "mpi";
    r.variant = variant;
    r.op = op_name(o.op);
    r.width = w;
    r.height = h;
    r.workers = std::to_string(nprocs);
    r.reps = o.reps;
    r.median_ms = t.median_ms;
    r.min_ms = t.min_ms;
    r.comm_ms = tc.median_ms;
    r.max_abs_diff = diff;
    if (!append_csv(o.csv, r)) {
      std::fprintf(stderr, "error: cannot write '%s'\n", o.csv.c_str());
      return 1;
    }
  }
  if (diff > tolerance(o.op)) {
    std::fprintf(stderr, "FAIL: max_abs_diff %d exceeds tolerance %d\n", diff, tolerance(o.op));
    return 3;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rc = drive(argc, argv);
  MPI_Finalize();
  return rc;
}
