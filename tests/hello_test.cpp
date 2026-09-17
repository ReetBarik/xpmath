// ============================================================================
// hello_test.cpp — the HOST half of the T0.1 harness smoke test.
// ============================================================================
//
// This is INFRASTRUCTURE, not a test of DD arithmetic correctness. It exercises
// the harness plumbing (input generation, oracle comparison, pass/fail
// reporting) on the most trivial possible identity:
//
//   for 10^6 random FP64 inputs x:
//     dd::DoubleDouble(x) round-trips to binary128 as exactly x, i.e.
//     OracleTraits<DD>::to_quad(dd::DoubleDouble(x)) == (__float128)x
//
// This holds by construction: dd::DoubleDouble(x) stores {hi=x, lo=0}, and
// to_quad = (float128)hi + (float128)lo = (float128)x exactly. So a passing run
// proves the host half of the harness end-to-end WITHOUT depending on any DD op
// being correct. Real DD correctness coverage begins in Phase 1 (T1.1..T1.6).
//
// SPLIT IN CORE_PLAN C4 step 4 (chunk E), AND IT WAS THE MIRROR CHECK THAT
// FOUND IT. C4 chunk A took this file off Kokkos -- its second half used to call
// kokkos_ep::run_unary_op, which is Kokkos::View + parallel_for, and that one
// call was the whole reason it linked Kokkos -- and re-pointed the launch at
// tests/device_harness.hpp. But it stayed MIXED: a TU including
// tests/test_utils_host.hpp (binary128) that also launches a kernel is exactly
// the shape the eight named splits exist to eliminate, and the shape nvcc
// rejects (S6). Chunk E's new mirror check in
// scripts/check_device_tu_purity.sh -- "no host-side test TU may contain a
// kernel launch" -- went red on line 130 of the old file on its first run. It
// was NOT a false positive and the gate was not weakened to accommodate it:
// this file is the host half and the launch is now tests/hello_test_device.cpp.
//
// NOTHING IS LOST BY THE SPLIT, and the reasoning is worth stating because the
// old device pass scored against the oracle and this one cannot. The old
// assertion was `digits_of_accuracy(device output, (float128)x) >= max_digits`.
// The device half now asserts instead that the returned limbs are BIT-IDENTICAL
// to the limbs that went in, which is strictly stronger than "agrees to
// max_digits" -- and this file still proves, over the same 10^6 inputs, that
// DD(x) -> binary128 is exactly x. Compose the two and you have the original
// claim, with the oracle on the side of the split that can carry it.
// ============================================================================

#include "test_utils_host.hpp"

#include <cstddef>

using namespace kokkos_ep;

int main() {
  int rc = 0;
  {
    constexpr int      n    = 1'000'000;
    constexpr uint64_t seed = 12345ULL;

    // Generate inputs on host (same engine the runners use), round-trip each
    // through DD on host, and assert bit-exact equality against the oracle.
    // This checks the DD<->quad conversion identity that every scored layer
    // relies on, independent of any device kernel.
    std::mt19937_64 gen(seed);
    InputDist dist = uniform(-1e8, 1e8);

    int mism = 0;
    for (int i = 0; i < n; ++i) {
      double   x    = dist(gen);
      dd::DoubleDouble x_dd(x);
      float128 back = OracleTraits<DD>::to_quad(x_dd);
      if (back != (float128)x) {
        if (mism < 5) {
          std::printf("MISMATCH i=%d  x=%.17g  back!=x\n", i, x);
        }
        ++mism;
      }
    }

    KOKKOS_EP_ASSERT(mism == 0, "DD round-trip to binary128 was not bit-exact");
    std::printf("hello_test: DD round-trip identity  %d/%d passed\n", n - mism, n);

    rc = ep_exit_code();
  }
  return rc;
}
