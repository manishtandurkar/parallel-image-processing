#include "pagp/backend.hpp"

int main(int argc, char **argv) { return pagp::run_main(argc, argv, pagp::autovec_backend()); }
