// Command-line options common to every backend binary.
#pragma once

#include <string>

namespace pagp {

enum class Op { Gray, Blur2d, BlurSep, Sobel, Pipeline, Fused };

const char *op_name(Op op);
bool parse_op(const std::string &s, Op &op);
// Ops whose input is RGB; the others take a single-channel (grayscale) image.
inline bool op_takes_rgb(Op op) { return op == Op::Gray || op == Op::Pipeline || op == Op::Fused; }

struct Options {
  std::string input;      // --input <path>
  int gen_w = 0, gen_h = 0; // --gen WxH
  Op op = Op::Pipeline;   // --op
  std::string variant;    // --variant (empty = backend default)
  int reps = 10;          // --reps
  int warmup = 2;         // fixed by the benchmark protocol
  int threads = 0;        // --threads (0 = backend default)
  bool validate = false;  // --validate
  std::string save;       // --save <path>
  std::string csv;        // --csv <path>
  std::string machine = "unknown"; // --machine <tag>
};

// Parses argv into opts. On error or --help prints usage and returns false.
bool parse_cli(int argc, char **argv, Options &opts);

} // namespace pagp
