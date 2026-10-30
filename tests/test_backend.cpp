// Every op x variant of one backend vs seq. Built once per backend with
// PAGP_TEST_BACKEND=<name>_backend. Optional args: thread counts to test (default: backend
// default).
#include <cstdlib>

#include "check_backend.hpp"

int main(int argc, char **argv) {
  const pagp::Backend &be = pagp::PAGP_TEST_BACKEND();
  int failures = 0;
  if (argc < 2)
    return pagp::check_backend(be, /*exact=*/false) ? 1 : 0;
  for (int i = 1; i < argc; ++i) {
    std::printf("threads=%s\n", argv[i]);
    failures += pagp::check_backend(be, /*exact=*/false, std::atoi(argv[i]));
  }
  return failures ? 1 : 0;
}
