// ============================================================================
// device_harness_test.cpp — the self-test for tests/device_harness.hpp
// ============================================================================
// Added by CORE_PLAN section C3.
//
// THIS GATE MUST PROVE IT RAN, and that is the whole design of the file.
//
// A device self-test whose pass condition is "nothing complained" is
// indistinguishable from one where the kernel never launched: an empty result,
// a no-op launch and a silently skipped test all read as success. CORE_PLAN §0
// forbids that shape. So the output buffer is filled with a POISON value on
// BOTH sides before the launch —
//
//   out.host()[i] = kPoison;  out.to_device();   // poison is now on the device
//
// — and the kernel writes a value DERIVED FROM THE INDEX that lies in a range
// the poison is provably outside of. Then:
//
//   * if the kernel never runs,      the device still holds poison  -> RED
//   * if from_device() is a no-op,   the host   still holds poison  -> RED
//   * if either writes the wrong thing, the exact-value check       -> RED
//
// The poison is -12345.0. The kernel computes out[i] = in[i]*2 + 1 with
// in[i] = 0.5*i, so every legitimate output is exactly i + 1.0 and therefore
// >= 1.0. The test asserts that separation at runtime rather than asserting it
// in a comment, because a later edit to either constant could quietly make the
// poison reachable and turn the strongest check here into a tautology.
//
// Exact equality is the right comparison, not a tolerance: 0.5*i is exact for
// every i in range, doubling is exact, and adding 1.0 is exact below 2^53. Any
// difference at all is a transport or launch defect, not rounding. There is no
// ulp arithmetic here and there must not be — docs/CORRECTNESS.md allows one
// scorer and this is not it.
//
// Registered in the KOKKOS-FREE ctest set, so it runs in both the default and
// the -DXPMATH_WITH_KOKKOS=OFF configurations. On a login node it exercises the
// serial fallback; under nvcc or hipcc it exercises the real thing. Same
// source, same functor, same assertions.
// ============================================================================

#include "device_harness.hpp"

#include <cstddef>
#include <cstdio>

namespace {

// Outside the reachable output range [1.0, n], and negative, so no plausible
// arithmetic slip lands on it.
constexpr double kPoison = -12345.0;

constexpr std::size_t kN = 4096;

// A plain struct, not a lambda: passed BY VALUE into the kernel, so it must be
// trivially copyable, and XPMATH_INLINE_FUNCTION gives it the __host__
// __device__ annotation on the vendor backends.
struct AxPlusB {
  const double* in;
  double*       out;

  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    out[i] = in[i] * 2.0 + 1.0;
  }
};

int failures = 0;

void check(const char* what, bool ok, const char* detail) {
  std::printf("  %-38s %-9s %s\n", what, ok ? "[ ok ]" : "[FAIL]", detail);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  std::printf("device harness self-test: backend = %s\n", xpt::where_name());

  xpt::buffer<double> in(kN);
  xpt::buffer<double> out(kN);

  check("allocation reported no vendor error", xpt::last_error() == 0,
        xpt::last_error() == 0 ? "last_error() == 0" : "last_error() != 0");
  check("buffer::size() round-trips",
        in.size() == kN && out.size() == kN, "4096 elements each");

  // The premise of the poison check, asserted rather than assumed. If a later
  // edit makes a legitimate output equal kPoison, the check below stops proving
  // anything, and this is where that is caught.
  bool poison_unreachable = true;
  for (std::size_t i = 0; i < kN; ++i) {
    if (static_cast<double>(i) + 1.0 == kPoison) poison_unreachable = false;
  }
  check("poison is outside the reachable range", poison_unreachable,
        "no expected value equals -12345.0");

  for (std::size_t i = 0; i < kN; ++i) {
    in.host()[i]  = 0.5 * static_cast<double>(i);
    out.host()[i] = kPoison;
  }

  in.to_device();
  out.to_device();  // poison the DEVICE side too, so a no-op kernel is visible

  xpt::parallel_for_n(kN, AxPlusB{in.device(), out.device()});
  xpt::fence();     // redundant — parallel_for_n already fenced — but the
                    // public entry point should be exercised by its own test
  out.from_device();

  check("launch reported no vendor error", xpt::last_error() == 0,
        xpt::last_error() == 0 ? "last_error() == 0" : "last_error() != 0");

  std::size_t survived_poison = 0;
  std::size_t wrong_value     = 0;
  std::size_t first_wrong     = kN;
  for (std::size_t i = 0; i < kN; ++i) {
    const double got  = out.host()[i];
    const double want = static_cast<double>(i) + 1.0;
    if (got == kPoison) ++survived_poison;
    if (got != want) {
      ++wrong_value;
      if (first_wrong == kN) first_wrong = i;
    }
  }

  char buf[160];
  std::snprintf(buf, sizeof(buf), "%zu of %zu elements still hold the poison",
                survived_poison, kN);
  check("the kernel actually wrote every element", survived_poison == 0, buf);

  if (wrong_value == 0) {
    std::snprintf(buf, sizeof(buf), "all %zu elements exactly equal i + 1", kN);
  } else {
    std::snprintf(buf, sizeof(buf),
                  "%zu wrong; first at i=%zu: got %.17g want %.17g",
                  wrong_value, first_wrong, out.host()[first_wrong],
                  static_cast<double>(first_wrong) + 1.0);
  }
  check("every element is exactly i + 1", wrong_value == 0, buf);

  std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
