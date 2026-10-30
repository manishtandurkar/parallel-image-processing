#include "pagp/csv.hpp"

#include <cstdio>

namespace pagp {

static void put_ms(FILE *f, double v) {
  if (v >= 0)
    std::fprintf(f, ",%.4f", v);
  else
    std::fputc(',', f);
}

bool append_csv(const std::string &path, const ResultRow &r) {
  FILE *f = std::fopen(path.c_str(), "a");
  if (!f)
    return false;
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fprintf(f, "machine,backend,variant,op,width,height,workers,reps,median_ms,min_ms,"
                    "h2d_ms,d2h_ms,comm_ms,max_abs_diff\n");
  std::fprintf(f, "%s,%s,%s,%s,%d,%d,%s,%d", r.machine.c_str(), r.backend.c_str(),
               r.variant.c_str(), r.op.c_str(), r.width, r.height, r.workers.c_str(), r.reps);
  put_ms(f, r.median_ms);
  put_ms(f, r.min_ms);
  put_ms(f, r.h2d_ms);
  put_ms(f, r.d2h_ms);
  put_ms(f, r.comm_ms);
  if (r.max_abs_diff >= 0)
    std::fprintf(f, ",%d\n", r.max_abs_diff);
  else
    std::fprintf(f, ",\n");
  return std::fclose(f) == 0;
}

} // namespace pagp
