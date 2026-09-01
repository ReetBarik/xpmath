// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// S5 Phase 2 deliverable (qf_math.hpp converted) — NO-KOKKOS COMPILE SMOKE.
//
// Identical to dd_no_kokkos_smoke.cpp / ff_no_kokkos_smoke.cpp, for QuadFloat
// instead of DoubleDouble/FloatFloat. The claim this file exists to defend: the
// standalone quad-float core is standalone. Not "we removed the obvious Kokkos
// includes" — actually compilable, and runnable, by a plain C++17 toolchain on
// a machine where Kokkos is not installed and is nowhere on the include path.
//
// Build and run it with scripts/check_standalone_no_kokkos.sh, which compiles
// this TU with an include path containing ONLY `include/` — notably NOT
// third_party/include/, so the Kokkos compat wrapper cannot be reached even by
// accident — and additionally greps the preprocessed output for the token
// "Kokkos", so an indirect include would be caught rather than assumed absent.
//
// Deliberately NOT registered with ctest, for the same reason as the DD/FF smoke:
// the gate is "full ctest 23/23", and changing that number mid-phase would be
// confusing. Wiring it into the build system belongs with the CMake packaging
// work (a later sub-plan).

#include <xp/qf_math.hpp>  // the ONLY project header this TU may include

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace {

int failures = 0;

// Coarse check: agreement with the FP32 result to ~1e-6 relative. A QF
// implementation that is merely *working* clears this by 23 orders of
// magnitude; the point is to detect "returns 0" or "returns NaN", not to
// measure digits.
void near(const char* what, xp::QuadFloat got, float want) {
  const float g = got.f0 + got.f1 + got.f2 + got.f3;
  const float scale = (std::fabs(want) > 1.0f) ? std::fabs(want) : 1.0f;
  const bool ok = std::isfinite(g) && std::fabs(g - want) <= 1.0e-6f * scale;
  std::printf("  %-28s %-9s got %.8g want %.8g\n", what, ok ? "[ ok ]" : "[FAIL]",
              g, want);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  using xp::QuadFloat;

  std::printf("standalone no-Kokkos smoke: xp::QuadFloat\n");
  std::printf("  XPMATH_INLINE_FUNCTION and the scalar-math dispatch resolved\n"
              "  without any Kokkos header being reachable.\n\n");

  // --- construction, the bit-pattern factory, and the constants -----------
  const QuadFloat pi = xp::QuadFloat_pi();
  near("QuadFloat_pi()", pi, (float)M_PI);
  near("from_bits round trip", QuadFloat::from_bits(0x3f800000U, 0x0U, 0x0U, 0x0U), 1.0f);

  // --- primitive arithmetic through the operators -------------------------
  const QuadFloat a(1.0f), b(3.0f);
  near("1/3 * 3", (a / b) * b, 1.0f);
  near("operator chain", (a + b) * QuadFloat(2.0f) - b, 5.0f);

  // --- the four-word property actually holds -------------------------------
  // 1/3 is not representable in FP32, so a genuine QF carries non-zero words
  // beyond f0. A degenerate build that silently dropped to single-float would
  // give f1/f2/f3 == 0 here and pass every check above.
  const QuadFloat third = a / b;
  const bool four_word = (third.f1 != 0.0f || third.f2 != 0.0f || third.f3 != 0.0f);
  std::printf("  %-28s %-9s f1=%.5g f2=%.5g f3=%.5g\n", "1/3 has beyond-f0 words",
              four_word ? "[ ok ]" : "[FAIL]", third.f1, third.f2, third.f3);
  if (!four_word) ++failures;

  // --- transcendentals: the scalar-math dispatch under load ---------------
  // fabs/floor/ceil/copysign/sqrt/log/atan2 are the scalar calls the QF header
  // makes; exp, log, sqrt and the trig functions route through most of them.
  near("sqrt(2)", xp::sqrt(QuadFloat(2.0f)), std::sqrt(2.0f));
  near("exp(1)", xp::exp(QuadFloat(1.0f)), std::exp(1.0f));
  near("log(exp(2))", xp::log(xp::exp(QuadFloat(2.0f))), 2.0f);
  near("sin(pi/6)", xp::sin(xp::divide_scalar(pi, 6.0f)), 0.5f);
  near("atan2(1,1)", xp::atan2(QuadFloat(1.0f), QuadFloat(1.0f)), (float)(M_PI / 4.0));
  near("sinh(0.5)", xp::sinh(QuadFloat(0.5f)), std::sinh(0.5f));
  near("asinh(1)", xp::asinh(QuadFloat(1.0f)), std::asinh(1.0f));

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
