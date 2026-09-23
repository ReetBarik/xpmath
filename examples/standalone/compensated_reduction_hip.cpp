// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// USAGE EXAMPLE — not a benchmark.
//
// HIP variant of the compensated-reduction example. Same numerical story as
// compensated_reduction.cpp (1e16 + many 1.0s), but the per-chunk DoubleDouble
// folds run through tests/device_harness.hpp under hipcc. The harness selects
// its HIP backend when compiled as HIP (__HIPCC__); under a plain host
// compiler it falls back to the serial path, so this file is also a usable
// illustration without a GPU.
//
// xp::DoubleDouble is not trivially copyable (user-declared copy ctor), so
// device results travel as separate hi/lo double buffers — the same pattern
// the device tests use.
//
// Compile from this directory after installing xpmath (harness is NOT part of
// the install; the relative include reaches it in the source tree):
//
//   hipcc -std=c++17 -O2 \
//     -I<xpmath-install-prefix>/include \
//     compensated_reduction_hip.cpp -o compensated_reduction_hip
//
// This is a smoke / illustration, not an accuracy gate. Correctness of the
// library is judged only by validation/sweep/ (docs/CORRECTNESS.md).

#include "../../tests/device_harness.hpp"

#include <xp/dd_math.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kN       = 1000;
constexpr std::size_t kChunks  = 32;
constexpr double      kLarge   = 1.0e16;

// One thread owns a strided slice, folds it with DoubleDouble, writes limbs.
struct ChunkReduce {
  const double* in;
  double*       out_hi;
  double*       out_lo;
  std::size_t   n;
  std::size_t   nchunks;

  XPMATH_INLINE_FUNCTION void operator()(std::size_t tid) const {
    xp::DoubleDouble s(0.0);
    for (std::size_t i = tid; i < n; i += nchunks) s = s + in[i];
    out_hi[tid] = s.hi;
    out_lo[tid] = s.lo;
  }
};

}  // namespace

int main() {
  std::printf("xpmath standalone example: compensated reduction (HIP / %s)\n",
              xpt::where_name());

  std::vector<double> x(kN);
  x[0] = kLarge;
  for (std::size_t i = 1; i < kN; ++i) x[i] = 1.0;

  xpt::buffer<double> in(kN);
  xpt::buffer<double> part_hi(kChunks);
  xpt::buffer<double> part_lo(kChunks);
  for (std::size_t i = 0; i < kN; ++i) in.host()[i] = x[i];
  for (std::size_t t = 0; t < kChunks; ++t) {
    part_hi.host()[t] = 0.0;
    part_lo.host()[t] = 0.0;
  }
  in.to_device();
  part_hi.to_device();
  part_lo.to_device();

  xpt::parallel_for_n(
      kChunks, ChunkReduce{in.device(), part_hi.device(), part_lo.device(), kN,
                           kChunks});
  part_hi.from_device();
  part_lo.from_device();

  if (xpt::last_error() != 0) {
    std::printf("RESULT: FAIL (device harness last_error=%d)\n",
                xpt::last_error());
    return 1;
  }

  xp::DoubleDouble total(0.0);
  for (std::size_t t = 0; t < kChunks; ++t)
    total = total + xp::DoubleDouble(part_hi.host()[t], part_lo.host()[t]);

  double naive = 0.0;
  for (double v : x) naive += v;

  // 1e16 + 999 is not an FP64 number (ulp(1e16) == 2). Recover the ones from
  // the DD limbs instead of collapsing hi+lo to double.
  const double ones      = static_cast<double>(kN - 1);
  const double recovered = (total.hi - kLarge) + total.lo;

  std::printf("  N = %zu terms, %zu device chunks\n", kN, kChunks);
  std::printf("  naive double sum      = %.17g   (lost the ones: %s)\n", naive,
              naive == kLarge ? "yes" : "no");
  std::printf("  DoubleDouble limbs    = hi=%.17g lo=%.17g\n", total.hi,
              total.lo);
  std::printf("  ones recovered by DD  = %.17g   (want %.17g)\n", recovered,
              ones);

  const bool naive_bad = (naive == kLarge);
  const bool dd_ok     = std::fabs(recovered - ones) < 0.5;
  if (!dd_ok || !naive_bad) {
    std::printf("RESULT: FAIL (smoke expectations not met)\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
