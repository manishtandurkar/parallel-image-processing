#include "pagp/cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace pagp {

static const char *kOpNames[] = {"gray", "blur2d", "blur_sep", "sobel", "pipeline", "fused"};

const char *op_name(Op op) { return kOpNames[int(op)]; }

bool parse_op(const std::string &s, Op &op) {
  for (int i = 0; i < 6; ++i) {
    if (s == kOpNames[i]) {
      op = Op(i);
      return true;
    }
  }
  return false;
}

static void usage(const char *prog) {
  std::fprintf(stderr,
               "usage: %s (--input <path> | --gen <WxH>) [options]\n"
               "  --op gray|blur2d|blur_sep|sobel|pipeline|fused   (default pipeline)\n"
               "  --variant <name>   backend-specific variant (default: backend default)\n"
               "  --reps N           timed repetitions after 2 warm-ups (default 10)\n"
               "  --threads N        worker threads (omp)\n"
               "  --block WxH        thread block size (cuda, default 16x16)\n"
               "  --validate         compare with the seq reference\n"
               "  --save <path>      write the output image (PNG)\n"
               "  --csv <path>       append one result row\n"
               "  --machine <tag>    e.g. mac-m5, wsl-x86, colab-t4\n",
               prog);
}

bool parse_cli(int argc, char **argv, Options &o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char *&v) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
        return false;
      }
      v = argv[++i];
      return true;
    };
    const char *v = nullptr;
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return false;
    } else if (a == "--validate") {
      o.validate = true;
    } else if (a == "--input") {
      if (!next(v))
        return false;
      o.input = v;
    } else if (a == "--gen") {
      if (!next(v))
        return false;
      if (std::sscanf(v, "%dx%d", &o.gen_w, &o.gen_h) != 2 || o.gen_w <= 0 || o.gen_h <= 0) {
        std::fprintf(stderr, "error: --gen expects WxH, got '%s'\n", v);
        return false;
      }
    } else if (a == "--op") {
      if (!next(v))
        return false;
      if (!parse_op(v, o.op)) {
        std::fprintf(stderr, "error: unknown op '%s'\n", v);
        return false;
      }
    } else if (a == "--variant") {
      if (!next(v))
        return false;
      o.variant = v;
    } else if (a == "--reps") {
      if (!next(v))
        return false;
      o.reps = std::atoi(v);
      if (o.reps < 1) {
        std::fprintf(stderr, "error: --reps must be >= 1\n");
        return false;
      }
    } else if (a == "--threads") {
      if (!next(v))
        return false;
      o.threads = std::atoi(v);
    } else if (a == "--block") {
      if (!next(v))
        return false;
      if (std::sscanf(v, "%dx%d", &o.block_w, &o.block_h) != 2 || o.block_w <= 0 ||
          o.block_h <= 0 || o.block_w * o.block_h > 1024) {
        std::fprintf(stderr, "error: --block expects WxH with W*H <= 1024, got '%s'\n", v);
        return false;
      }
    } else if (a == "--save") {
      if (!next(v))
        return false;
      o.save = v;
    } else if (a == "--csv") {
      if (!next(v))
        return false;
      o.csv = v;
    } else if (a == "--machine") {
      if (!next(v))
        return false;
      o.machine = v;
    } else {
      std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
      usage(argv[0]);
      return false;
    }
  }
  if (o.input.empty() == (o.gen_w == 0)) {
    std::fprintf(stderr, "error: give exactly one of --input or --gen\n");
    usage(argv[0]);
    return false;
  }
  return true;
}

} // namespace pagp
