// ============================================================================
// hello_test.cpp — end-to-end smoke test for the T0.1 harness.
// ============================================================================
//
// This is INFRASTRUCTURE, not a test of DD arithmetic correctness. It exercises
// the harness plumbing (Kokkos init, input generation, host<->device copy,
// oracle comparison, pass/fail reporting) on the most trivial possible identity:
//
//   for 10^6 random FP64 inputs x:
//     dd::DoubleDouble(x) round-trips to binary128 as exactly x, i.e.
//     BackendTraits<DD>::to_quad(dd::DoubleDouble(x)) == (__float128)x
//
// This holds by construction: dd::DoubleDouble(x) stores {hi=x, lo=0}, and
// to_quad = (float128)hi + (float128)lo = (float128)x exactly. So a passing run
// proves the harness end-to-end WITHOUT depending on any DD op being correct.
// Real DD correctness coverage begins in Phase 1 (T1.1..T1.6).
// ============================================================================

#include "test_utils.hpp"

using namespace kokkos_ep;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    constexpr int      n    = 1'000'000;
    constexpr uint64_t seed = 12345ULL;

    // Generate inputs on host (same engine the runners use), round-trip each
    // through DD on host, and assert bit-exact equality against the oracle.
    // Kept on host deliberately: this checks the DD<->quad conversion identity
    // that the device runners rely on, independent of any device kernel.
    std::mt19937_64 gen(seed);
    InputDist dist = uniform(-1e8, 1e8);

    int mism = 0;
    for (int i = 0; i < n; ++i) {
      double   x    = dist(gen);
      dd::DoubleDouble x_dd(x);
      float128 back = BackendTraits<DD>::to_quad(x_dd);
      if (back != (float128)x) {
        if (mism < 5) {
          std::printf("MISMATCH i=%d  x=%.17g  back!=x\n", i, x);
        }
        ++mism;
      }
    }

    KOKKOS_EP_ASSERT(mism == 0, "DD round-trip to binary128 was not bit-exact");
    std::printf("hello_test: DD round-trip identity  %d/%d passed\n", n - mism, n);

    // Also exercise the device runner primitive so the smoke test covers the
    // host->device->host path the accuracy layers depend on. Device op is the
    // identity; host oracle is the identity; expect max_digits everywhere.
    AccStats st = run_unary_op<DD>(
        n, seed, uniform(-1e8, 1e8),
        [](float128 x) { return x; },                       // host oracle: identity
        KOKKOS_LAMBDA(dd::DoubleDouble x) { return x; });        // device op: identity
    print_stats("run_unary_op identity", st);
    KOKKOS_EP_ASSERT(st.min >= (double)BackendTraits<DD>::max_digits,
                     "device-runner identity did not reach full digits");

    rc = ep_exit_code();
  }
  Kokkos::finalize();
  return rc;
}
