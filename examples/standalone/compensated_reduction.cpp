// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// USAGE EXAMPLE — not a benchmark.
//
// Compensated reduction with xp::DoubleDouble: fold many FP64 terms into a
// double-double accumulator. Each `+` goes through Knuth's TwoSum, so the
// rounding error that a plain `double` sum would drop is retained in the low
// word.
//
// Build against an *installed* xpmath (this directory's CMakeLists.txt):
//
//   cmake -S . -B build -DCMAKE_PREFIX_PATH=<xpmath-install-prefix>
//   cmake --build build && ./build/compensated_reduction
//
// This is a smoke / illustration, not an accuracy gate. Correctness of the
// library is judged only by validation/sweep/ (docs/CORRECTNESS.md).

#include <xp/dd_math.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Pathological input: one large value followed by many 1.0s. In FP64 the 1.0s
// are smaller than ulp(1e16) and vanish from a naive sum. Double-double keeps
// them. Exact answer is 1e16 + (N - 1).
constexpr std::size_t kN     = 1000;
constexpr double      kLarge = 1.0e16;

double naive_sum(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) s += v;
  return s;
}

xp::DoubleDouble compensated_sum(const std::vector<double>& x) {
  xp::DoubleDouble s(0.0);
  for (double v : x) s = s + v;
  return s;
}

}  // namespace

int main() {
  std::vector<double> x(kN);
  x[0] = kLarge;
  for (std::size_t i = 1; i < kN; ++i) x[i] = 1.0;

  const double ones   = static_cast<double>(kN - 1);  // 999: exact in FP64
  const double naive  = naive_sum(x);
  const xp::DoubleDouble dd = compensated_sum(x);
  // 1e16 + 999 is not an FP64 number (ulp(1e16) == 2), so do not collapse the
  // DD sum to double and call that "exact". Recover the ones in the DD limbs:
  // (hi - 1e16) is exact here, then + lo brings back what naive lost.
  const double recovered = (dd.hi - kLarge) + dd.lo;

  std::printf("xpmath standalone example: compensated reduction (host)\n");
  std::printf("  N = %zu terms: 1e16 + 1.0 * %zu\n", kN, kN - 1);
  std::printf("  naive double sum      = %.17g   (lost the ones: %s)\n", naive,
              naive == kLarge ? "yes" : "no");
  std::printf("  DoubleDouble limbs    = hi=%.17g lo=%.17g\n", dd.hi, dd.lo);
  std::printf("  ones recovered by DD  = %.17g   (want %.17g)\n", recovered,
              ones);

  // Coarse smoke only: naive must drop every 1.0; DD must recover their count.
  // Not a competing accuracy verdict — see docs/CORRECTNESS.md.
  const bool naive_bad = (naive == kLarge);
  const bool dd_ok     = std::fabs(recovered - ones) < 0.5;
  if (!dd_ok || !naive_bad) {
    std::printf("RESULT: FAIL (smoke expectations not met)\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
