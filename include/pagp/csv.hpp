// Benchmark result row writer (schema fixed in CLAUDE.md).
#pragma once

#include <string>

namespace pagp {

// Columns left negative (or empty strings) are written as empty cells.
struct ResultRow {
  std::string machine, backend, variant, op;
  int width = 0, height = 0;
  std::string workers = "1";
  int reps = 0;
  double median_ms = 0, min_ms = 0;
  double h2d_ms = -1, d2h_ms = -1, comm_ms = -1;
  int max_abs_diff = -1;
};

// Appends one row; writes the header first if the file is new or empty. Returns false on I/O error.
bool append_csv(const std::string &path, const ResultRow &row);

} // namespace pagp
