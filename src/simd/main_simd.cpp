#include "pagp/backend.hpp"

int main(int argc, char **argv) { return pagp::run_main(argc, argv, pagp::simd_backend()); }
