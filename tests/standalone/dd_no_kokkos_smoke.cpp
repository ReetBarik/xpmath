// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// S2 deliverable 4 — NO-KOKKOS COMPILE SMOKE.
//
// The claim this file exists to defend: the standalone double-double core is
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
// Deliberately NOT registered with ctest. The S2 gate is "full ctest 23/23",
// and quietly making it 24 would make that number mean something different
// before and after the restructure. Wiring it into the build system belongs
// with the CMake packaging work (S5/S6).
//
// This is a compile-and-link smoke test, not an accuracy test: numerical
// behaviour is covered by the 23-test suite and by the byte-identical demo
// gate, both of which run the same code through the Kokkos path. The checks
// below only need to be strong enough that the compiler cannot optimize the
// library away, and coarse enough that they never disagree with the real
// accuracy suite.

#include <xp/dd_math.hpp>  // the ONLY project header this TU may include

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace {

int failures = 0;

// Coarse check: agreement with the FP64 result to ~1e-12 relative. A DD
// implementation that is merely *working* clears this by 19 orders of
// magnitude; the point is to detect "returns 0" or "returns NaN", not to
// measure digits.
void near(const char* what, xp::DoubleDouble got, double want) {
  const double g = got.hi + got.lo;
  const double scale = (std::fabs(want) > 1.0) ? std::fabs(want) : 1.0;
  const bool ok = std::isfinite(g) && std::fabs(g - want) <= 1.0e-12 * scale;
  std::printf("  %-28s %-9s got %.15g want %.15g\n", what, ok ? "[ ok ]" : "[FAIL]",
              g, want);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  using xp::DoubleDouble;

  std::printf("standalone no-Kokkos smoke: xp::DoubleDouble\n");
  std::printf("  XPMATH_INLINE_FUNCTION and the scalar-math dispatch resolved\n"
              "  without any Kokkos header being reachable.\n\n");

  // --- construction, the bit-pattern factory, and the constants -----------
  const DoubleDouble pi = xp::DoubleDouble_pi();
  near("DoubleDouble_pi()", pi, M_PI);
  near("from_bits round trip", DoubleDouble::from_bits(0x3ff0000000000000ULL, 0x0ULL), 1.0);

  // --- primitive arithmetic through the operators -------------------------
  const DoubleDouble a(1.0), b(3.0);
  near("1/3 * 3", (a / b) * b, 1.0);
  near("operator chain", (a + b) * DoubleDouble(2.0) - b, 5.0);

  // --- the two-word property actually holds -------------------------------
  // 1/3 is not representable in FP64, so a genuine DD carries a non-zero lo
  // word. A degenerate build that silently dropped to single-double would
  // give lo == 0 here and pass every check above.
  const DoubleDouble third = a / b;
  const bool two_word = (third.lo != 0.0);
  std::printf("  %-28s %-9s lo = %.5g\n", "1/3 has a lo word",
              two_word ? "[ ok ]" : "[FAIL]", third.lo);
  if (!two_word) ++failures;

  // --- transcendentals: the scalar-math dispatch under load ---------------
  // sqrt/log/copysign/atan2/ldexp are the five host-or-device scalar calls the
  // header makes; exp, log and sqrt below route through all of them.
  near("sqrt(2)", xp::sqrt(DoubleDouble(2.0)), std::sqrt(2.0));
  near("exp(1)", xp::exp(DoubleDouble(1.0)), std::exp(1.0));
  near("log(exp(2))", xp::log(xp::exp(DoubleDouble(2.0))), 2.0);
  near("sin(pi/6)", xp::sin(xp::divide_scalar(pi, 6.0)), 0.5);
  near("atan2(1,1)", xp::atan2(DoubleDouble(1.0), DoubleDouble(1.0)), M_PI / 4.0);
  near("tgamma(5)", xp::tgamma(DoubleDouble(5.0)), 24.0);
  near("erf(1)", xp::erf(DoubleDouble(1.0)), std::erf(1.0));

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
