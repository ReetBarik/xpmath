// ============================================================================
// hello_test.cpp — end-to-end smoke test for the T0.1 harness.
// ============================================================================
//
// This is INFRASTRUCTURE, not a test of DD arithmetic correctness. It exercises
// the harness plumbing (input generation, host<->device copy, oracle
// comparison, pass/fail reporting) on the most trivial possible identity:
//
//   for 10^6 random FP64 inputs x:
//     dd::DoubleDouble(x) round-trips to binary128 as exactly x, i.e.
//     OracleTraits<DD>::to_quad(dd::DoubleDouble(x)) == (__float128)x
//
// This holds by construction: dd::DoubleDouble(x) stores {hi=x, lo=0}, and
// to_quad = (float128)hi + (float128)lo = (float128)x exactly. So a passing run
// proves the harness end-to-end WITHOUT depending on any DD op being correct.
// Real DD correctness coverage begins in Phase 1 (T1.1..T1.6).
//
// NO KOKKOS, AS OF CORE_PLAN C4. The second half below used to call
// kokkos_ep::run_unary_op, which is Kokkos::View + parallel_for, and that one
// call was the whole reason this file linked Kokkos. It now launches through
// tests/device_harness.hpp (C3), which is the Kokkos-free CUDA/HIP/serial
// launch layer the rest of the C4-C8 arc measures through. The assertion is
// unchanged and so is what it proves: the host -> device -> host round trip
// preserves every element exactly.
//
// Built by g++ here, so what runs is the harness's SERIAL backend -- the same
// thing Kokkos's Serial execution space gave this test before. A real GPU
// launch of the same shape is device_harness_test's job.
//
// WHY THE BUFFERS ARE double AND NOT dd::DoubleDouble.
// xpt::buffer static_asserts std::is_trivially_copyable<T>, because it moves
// bytes with memcpy/cudaMemcpy. xp::DoubleDouble declares its own copy
// constructor and copy assignment -- both of which do exactly what the implicit
// ones would -- and a USER-PROVIDED copy member makes a type non-trivially-
// copyable by rule, whatever its body says. So the type cannot ride in a
// buffer<> today. The assertion is right and is left alone: it is guarding a
// real hazard, and the alternative (=default those members across the four
// backends) is a library change with device-codegen consequences that has no
// business inside this chunk. See the report.
//
// The components travel instead, one buffer per limb, and the kernel builds a
// real dd::DoubleDouble from them and writes its limbs back. Nothing about the
// round trip is weakened -- the same 2*10^6 doubles cross the bus -- and the
// test makes no layout assumption about the struct in exchange.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_host.hpp"

#include <cstddef>

using namespace kokkos_ep;

// The functor xpt::parallel_for_n launches. A plain struct of raw device
// pointers, not a lambda: the harness passes F by value into the kernel and
// deliberately does not require nvcc --extended-lambda of its callers.
struct IdentityOp {
  const double* in_hi;
  const double* in_lo;
  double*       out_hi;
  double*       out_lo;
  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    const dd::DoubleDouble v(in_hi[i], in_lo[i]);
    out_hi[i] = v.hi;
    out_lo[i] = v.lo;
  }
};

int main() {
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

    // Also exercise the launch harness so the smoke test covers the
    // host->device->host path the accuracy layers depend on. Device op is the
    // identity; host oracle is the identity; expect max_digits everywhere.
    std::printf("hello_test: launch backend = %s\n", xpt::where_name());

    xpt::buffer<double>   in_hi(n), in_lo(n), out_hi(n), out_lo(n);
    std::vector<float128> href(n);
    {
      std::mt19937_64 g2(seed);
      InputDist       d2 = uniform(-1e8, 1e8);
      for (int i = 0; i < n; ++i) {
        const double           x = d2(g2);
        const dd::DoubleDouble v(x);
        in_hi.host()[i] = v.hi;
        in_lo.host()[i] = v.lo;
        href[i]         = (float128)x;   // host oracle: identity
      }
    }
    in_hi.to_device();
    in_lo.to_device();

    // POISON THE OUTPUT ON BOTH SIDES. A kernel that never launched, or a
    // from_device() that copied nothing, must be RED rather than quiet -- and
    // reading back whatever malloc happened to leave in the buffer is exactly
    // how that goes quiet. The poison is not in the input range, so any element
    // that survives scores 0 digits and trips the min assertion below; the
    // explicit count makes the diagnostic say which failure it was.
    const double kPoison = -1.0e300;
    for (int i = 0; i < n; ++i) { out_hi.host()[i] = kPoison; out_lo.host()[i] = kPoison; }
    out_hi.to_device();
    out_lo.to_device();

    xpt::parallel_for_n(static_cast<std::size_t>(n),
                        IdentityOp{in_hi.device(), in_lo.device(),
                                   out_hi.device(), out_lo.device()});
    out_hi.from_device();
    out_lo.from_device();

    KOKKOS_EP_ASSERT(xpt::last_error() == 0,
                     "device harness reported a nonzero vendor error");

    int survived = 0;
    std::vector<double> digs(n), ulps(n);
    for (int i = 0; i < n; ++i) {
      if (out_hi.host()[i] == kPoison) ++survived;
      const dd::DoubleDouble got_dd(out_hi.host()[i], out_lo.host()[i]);
      const float128         got = OracleTraits<DD>::to_quad(got_dd);
      digs[i] = digits_of_accuracy<DD>(got, href[i]);
      ulps[i] = ulp_error<DD>(got, href[i]);
    }
    KOKKOS_EP_ASSERT(survived == 0,
                     "poison survived the launch: the kernel or the copy-back "
                     "did not run");

    AccStats st = compute_stats(digs.data(), ulps.data(), n);
    print_stats("harness identity", st);
    KOKKOS_EP_ASSERT(st.min >= (double)BackendTraits<DD>::max_digits,
                     "device-runner identity did not reach full digits");

    rc = ep_exit_code();
  }
  return rc;
}
