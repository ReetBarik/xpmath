// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// S5 Phase 4 deliverable (ff_complex.hpp converted) — NO-KOKKOS COMPILE SMOKE.
//
// The claim this file exists to defend: the standalone float-float complex
// core is standalone. Not "we removed the obvious Kokkos includes" — actually
// compilable, and runnable, by a plain C++17 toolchain on a machine where
// Kokkos is not installed and is nowhere on the include path.
//
// Build and run it with scripts/check_standalone_no_kokkos.sh, which compiles
// this TU with an include path containing ONLY `include/` — notably NOT
// third_party/include/, so the Kokkos compat wrapper cannot be reached even by
// accident — and additionally greps the preprocessed output for the token
// "Kokkos", so an indirect include would be caught rather than assumed absent.
//
// Deliberately NOT registered with ctest, for the same reason as the real smoke:
// the gate is "full ctest 23/23", and changing that number mid-phase would be
// confusing. Wiring it into the build system belongs with the CMake packaging
// work (a later sub-plan).

#include <xp/ff_complex.hpp>  // the ONLY project header this TU may include

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace {

int failures = 0;

// Coarse check: agreement with the FP32 complex result to ~1e-6 relative.
// A FF complex implementation that is merely *working* clears this by many
// orders of magnitude; the point is to detect "returns 0" or "returns NaN",
// not to measure digits.
void near_complex(const char* what, xp::FloatFloatComplex got, float want_re, float want_im) {
  const float g_re = got.re.hi + got.re.lo;
  const float g_im = got.im.hi + got.im.lo;
  const float scale_re = (std::fabs(want_re) > 1.0f) ? std::fabs(want_re) : 1.0f;
  const float scale_im = (std::fabs(want_im) > 1.0f) ? std::fabs(want_im) : 1.0f;
  const bool ok_re = std::isfinite(g_re) && std::fabs(g_re - want_re) <= 1.0e-6f * scale_re;
  const bool ok_im = std::isfinite(g_im) && std::fabs(g_im - want_im) <= 1.0e-6f * scale_im;
  const bool ok = ok_re && ok_im;
  std::printf("  %-28s %-9s got (%.6g, %.6g) want (%.6g, %.6g)\n", what, ok ? "[ ok ]" : "[FAIL]",
              g_re, g_im, want_re, want_im);
  if (!ok) ++failures;
}

void near_real(const char* what, xp::FloatFloat got, float want) {
  const float g = got.hi + got.lo;
  const float scale = (std::fabs(want) > 1.0f) ? std::fabs(want) : 1.0f;
  const bool ok = std::isfinite(g) && std::fabs(g - want) <= 1.0e-6f * scale;
  std::printf("  %-28s %-9s got %.6g want %.6g\n", what, ok ? "[ ok ]" : "[FAIL]",
              g, want);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  using xp::FloatFloat;
  using xp::FloatFloatComplex;

  std::printf("standalone no-Kokkos smoke: xp::FloatFloatComplex\n");
  std::printf("  XPMATH_INLINE_FUNCTION and the scalar-math dispatch resolved\n"
              "  without any Kokkos header being reachable.\n\n");

  // --- construction ---------------------------------------------------
  const FloatFloatComplex z1(1.0f, 2.0f);
  near_complex("construction (1, 2)", z1, 1.0f, 2.0f);

  // --- primitive arithmetic through the operators ---------------------
  const FloatFloatComplex a(3.0f, 4.0f);
  const FloatFloatComplex b(1.0f, -1.0f);
  near_complex("a + b", a + b, 4.0f, 3.0f);
  near_complex("a - b", a - b, 2.0f, 5.0f);
  near_complex("a * b", a * b, 7.0f, 1.0f);  // (3+4i)*(1-i) = 3-3i+4i+4 = 7+i
  near_complex("a / b", a / b, -0.5f, 3.5f);  // (3+4i)/(1-i) = (3+4i)*(1+i)/2 = (-1+7i)/2

  // --- the two-word property for complex actually holds ---------------
  // (1+i)/(3+4i) has components that are not exactly representable in FP32,
  // so a genuine FF complex carries non-zero lo words. The result is
  // (7+i)/25 = (0.28 + 0.04i), where 0.28 and 0.04 have non-trivial binary
  // representations. A degenerate build would give lo == 0.
  const FloatFloatComplex c1(1.0f, 1.0f);
  const FloatFloatComplex c2(3.0f, 4.0f);
  const FloatFloatComplex ratio = c1 / c2;
  const bool two_word = (ratio.re.lo != 0.0f || ratio.im.lo != 0.0f);
  std::printf("  %-28s %-9s re.lo = %.5g, im.lo = %.5g\n", "(1+i)/(3+4i) has lo words",
              two_word ? "[ ok ]" : "[FAIL]", ratio.re.lo, ratio.im.lo);
  if (!two_word) ++failures;

  // --- basic complex operations ---------------------------------------
  const FloatFloatComplex z(3.0f, 4.0f);
  near_real("abs(3+4i)", xp::abs(z), 5.0f);
  near_complex("conj(3+4i)", xp::conj(z), 3.0f, -4.0f);

  // --- sqrt of complex ------------------------------------------------
  const FloatFloatComplex sqrt_z = xp::sqrt(z);
  const FloatFloatComplex sqrt_z2 = sqrt_z * sqrt_z;
  near_complex("sqrt(3+4i)^2", sqrt_z2, 3.0f, 4.0f);

  // --- transcendentals: exp, log --------------------------------------
  const FloatFloatComplex w(1.0f, float(M_PI)/2.0f);
  const FloatFloatComplex exp_w = xp::exp(w);
  near_complex("exp(1 + pi/2 i)", exp_w, 0.0f, std::exp(1.0f));  // e^(1+πi/2) = e·i

  const FloatFloatComplex log_exp = xp::log(exp_w);
  near_complex("log(exp(1 + pi/2 i))", log_exp, 1.0f, float(M_PI)/2.0f);

  // --- trig functions -------------------------------------------------
  const FloatFloatComplex sin_i = xp::sin(FloatFloatComplex(0.0f, 1.0f));
  near_complex("sin(i)", sin_i, 0.0f, std::sinh(1.0f));

  const FloatFloatComplex cos_i = xp::cos(FloatFloatComplex(0.0f, 1.0f));
  near_complex("cos(i)", cos_i, std::cosh(1.0f), 0.0f);

  // --- ADL and the host-only ostream overload -------------------------
  std::ostringstream os;
  os << z;
  const bool streamed = !os.str().empty();
  std::printf("  %-28s %-9s %s\n", "operator<< via ADL",
              streamed ? "[ ok ]" : "[FAIL]", os.str().c_str());
  if (!streamed) ++failures;

  std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
