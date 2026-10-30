// Comparison against the sequential reference.
#pragma once

#include "pagp/image.hpp"

namespace pagp {

// Largest per-sample absolute difference, or -1 if the shapes differ.
int max_abs_diff(const Image &a, const Image &b);
// Peak signal-to-noise ratio in dB; +inf for identical images.
double psnr(const Image &a, const Image &b);

} // namespace pagp
