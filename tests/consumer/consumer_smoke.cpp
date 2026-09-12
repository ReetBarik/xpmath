// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// CONSUMER SMOKE TEST for the INSTALLED xpmath package.
//
// This file is NOT part of the main build. tests/consumer/ is configured as
// its OWN CMake project against an installed xpmath via find_package(xpmath),
// from OUTSIDE the build tree. That separation is the whole point: the main
// build never exercises the exported target, the installed header layout, or
// the generated version file, so all three can be wrong while every existing
// test is green.
//
// WHAT THIS PROVES, and deliberately nothing more:
//   1. find_package(xpmath) resolves from an install prefix.
//   2. The exported target xpmath::xpmath carries a usable include path.
//   3. The installed headers compile with plain C++17 -- no Kokkos, no
//      libquadmath, no MPFR.
//   4. All four backends compute.
//
// It is a COMPILE-LINK-AND-COMPUTE check, not an accuracy check. Accuracy is
// the sweep's job. The tolerances below are deliberately coarse -- a working
// backend clears them by many orders of magnitude -- so that this file can
// never disagree with the accuracy suite. One measurement, one verdict; this
// is not a second opinion on digits.
//
// Same posture as tests/standalone/*_no_kokkos_smoke.cpp, which checks the
// headers in-tree. This checks them after install.

#include <cmath>
#include <cstdio>

#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>
#include <xp/qf_math.hpp>
#include <xp/tf_math.hpp>

namespace {

int failures = 0;

// The backends expose their components as plain fields and carry no
// operator double, so the value is recovered by summing the expansion. That
// is exact enough for a smoke check: the leading word alone already carries
// the answer to FP32/FP64 precision.
double value_of(const xp::DoubleDouble& v) { return v.hi + v.lo; }
double value_of(const xp::FloatFloat& v) {
  return static_cast<double>(v.hi) + static_cast<double>(v.lo);
}
double value_of(const xp::QuadFloat& v) {
  return ((static_cast<double>(v.f0) + static_cast<double>(v.f1)) +
          static_cast<double>(v.f2)) + static_cast<double>(v.f3);
}
double value_of(const xp::TripleFloat& v) {
  return (static_cast<double>(v.f0) + static_cast<double>(v.f1)) +
         static_cast<double>(v.f2);
}

void near(const char* what, double got, double want, double tol) {
  const double scale = (std::fabs(want) > 1.0) ? std::fabs(want) : 1.0;
  const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol * scale;
  std::printf("  %-24s %-8s got %.17g  want %.17g\n", what,
              ok ? "[ ok ]" : "[FAIL]", got, want);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  std::printf("xpmath consumer smoke test (installed package)\n");
  std::printf("  find_package(xpmath) resolved; xpmath::xpmath include path\n"
              "  usable; headers compiled with no Kokkos present.\n\n");

  // (1/3)*3 round trip on each backend. Detects "returns 0" / "returns NaN" /
  // a header set that is internally inconsistent after install. FP32-based
  // backends are checked at FP32-ish tolerance because value_of() collapses
  // the expansion into a double.
  {
    const xp::DoubleDouble a =
        xp::DoubleDouble(1.0) / xp::DoubleDouble(3.0);
    near("DD (1/3)*3", value_of(a * xp::DoubleDouble(3.0)), 1.0, 1e-12);
  }
  {
    const xp::FloatFloat a = xp::FloatFloat(1.0f) / xp::FloatFloat(3.0f);
    near("FF (1/3)*3", value_of(a * xp::FloatFloat(3.0f)), 1.0, 1e-6);
  }
  {
    const xp::QuadFloat a = xp::QuadFloat(1.0f) / xp::QuadFloat(3.0f);
    near("QF (1/3)*3", value_of(a * xp::QuadFloat(3.0f)), 1.0, 1e-6);
  }
  {
    const xp::TripleFloat a = xp::TripleFloat(1.0f) / xp::TripleFloat(3.0f);
    near("TF (1/3)*3", value_of(a * xp::TripleFloat(3.0f)), 1.0, 1e-6);
  }

  // sqrt on each backend: the most recently rewritten op (correctly rounded
  // seed, f81a98c), so it is the one most likely to expose a stale or
  // partially-installed header set.
  near("DD sqrt(2)", value_of(xp::sqrt(xp::DoubleDouble(2.0))),
       1.4142135623730951, 1e-12);
  near("FF sqrt(2)", value_of(xp::sqrt(xp::FloatFloat(2.0f))),
       1.4142135623730951, 1e-6);
  near("QF sqrt(2)", value_of(xp::sqrt(xp::QuadFloat(2.0f))),
       1.4142135623730951, 1e-6);
  near("TF sqrt(2)", value_of(xp::sqrt(xp::TripleFloat(2.0f))),
       1.4142135623730951, 1e-6);

  std::printf("\n");
  if (failures) {
    std::printf("RESULT: FAIL (%d check%s)\n", failures,
                failures == 1 ? "" : "s");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
