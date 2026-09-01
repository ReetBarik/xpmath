// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// S5 Phase 1 deliverable (ff_math.hpp converted) — NO-KOKKOS COMPILE SMOKE.
//
// Identical to dd_no_kokkos_smoke.cpp, for FloatFloat instead of DoubleDouble.
// The claim this file exists to defend: the standalone float-float core is
// standalone. Not "we removed the obvious Kokkos includes" — actually
// compilable, and runnable, by a plain C++17 toolchain on a machine where
// Kokkos is not installed and is nowhere on the include path.
//
// Build and run it with scripts/check_standalone_no_kokkos.sh, which compiles
// this TU with an include path containing ONLY `include/` — notably NOT
// third_party/include/, so the Kokkos compat wrapper cannot be reached even by
// accident — and additionally greps the preprocessed output for the token
// "Kokkos", so an indirect include would be caught rather than assumed absent.
//
// Deliberately NOT registered with ctest, for the same reason as the DD smoke:
// the gate is "full ctest 23/23", and changing that number mid-phase would be
// confusing. Wiring it into the build system belongs with the CMake packaging
// work (a later sub-plan).

#include <xp/ff_math.hpp>  // the ONLY project header this TU may include

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace {

int failures = 0;

// Coarse check: agreement with the FP32 result to ~1e-6 relative. An FF
// implementation that is merely *working* clears this by 8 orders of
// magnitude; the point is to detect "returns 0" or "returns NaN", not to
// measure digits.
void near(const char* what, xp::FloatFloat got, float want) {
  const float g = got.hi + got.lo;
  const float scale = (std::fabs(want) > 1.0f) ? std::fabs(want) : 1.0f;
  const bool ok = std::isfinite(g) && std::fabs(g - want) <= 1.0e-6f * scale;
  std::printf("  %-28s %-9s got %.8g want %.8g\n", what, ok ? "[ ok ]" : "[FAIL]",
              g, want);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  using xp::FloatFloat;

  std::printf("standalone no-Kokkos smoke: xp::FloatFloat\n");
  std::printf("  XPMATH_INLINE_FUNCTION and the scalar-math dispatch resolved\n"
              "  without any Kokkos header being reachable.\n\n");

  // --- construction, the bit-pattern factory, and the constants -----------
  const FloatFloat pi = xp::FloatFloat_pi();
  near("FloatFloat_pi()", pi, (float)M_PI);
  near("from_bits round trip", FloatFloat::from_bits(0x3f800000U, 0x0U), 1.0f);

  // --- primitive arithmetic through the operators -------------------------
  const FloatFloat a(1.0f), b(3.0f);
  near("1/3 * 3", (a / b) * b, 1.0f);
  near("operator chain", (a + b) * FloatFloat(2.0f) - b, 5.0f);

  // --- the two-word property actually holds -------------------------------
  // 1/3 is not representable in FP32, so a genuine FF carries a non-zero lo
  // word. A degenerate build that silently dropped to single-float would
  // give lo == 0 here and pass every check above.
  const FloatFloat third = a / b;
  const bool two_word = (third.lo != 0.0f);
  std::printf("  %-28s %-9s lo = %.5g\n", "1/3 has a lo word",
              two_word ? "[ ok ]" : "[FAIL]", third.lo);
  if (!two_word) ++failures;

  // --- transcendentals: the scalar-math dispatch under load ---------------
  // sqrt/log/copysign/atan2 are the scalar calls the FF header makes; exp, log
  // and sqrt below route through most of them.
  near("sqrt(2)", xp::sqrt(FloatFloat(2.0f)), std::sqrt(2.0f));
  near("exp(1)", xp::exp(FloatFloat(1.0f)), std::exp(1.0f));
  near("log(exp(2))", xp::log(xp::exp(FloatFloat(2.0f))), 2.0f);
  near("sin(pi/6)", xp::sin(xp::divide_scalar(pi, 6.0f)), 0.5f);
  near("atan2(1,1)", xp::atan2(FloatFloat(1.0f), FloatFloat(1.0f)), (float)(M_PI / 4.0));
  near("tgamma(5)", xp::tgamma(FloatFloat(5.0f)), 24.0f);
  near("erf(1)", xp::erf(FloatFloat(1.0f)), std::erf(1.0f));

  // --- ADL and the host-only ostream overload -----------------------------
  std::ostringstream os;
  os << pi;
  const bool streamed = !os.str().empty();
  std::printf("  %-28s %-9s %s\n", "operator<< via ADL",
              streamed ? "[ ok ]" : "[FAIL]", os.str().c_str());
  if (!streamed) ++failures;

  std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
