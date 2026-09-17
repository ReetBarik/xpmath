// ============================================================================
// hello_test_device.cpp — the DEVICE half of the T0.1 harness smoke test.
// ============================================================================
// Added by CORE_PLAN section C4 step 4 (chunk E).
//
// WHY THIS FILE EXISTS, AND WHY IT WAS NOT IN C4's LIST OF EIGHT
//   The plan names eight mixed TUs to split. hello_test was not one of them --
//   it was in the "seven that link Kokkos and barely use it" list instead, on
//   the grounds that its only Kokkos was a launch it could be lifted off. Chunk
//   A did exactly that and the file kept building and passing, so nothing
//   complained. But the result was still a TU that includes
//   tests/test_utils_host.hpp (binary128) AND launches a kernel, which is the
//   definition of mixed and precisely what nvcc rejects (S6): it walks STL
//   member signatures during the device pass and refuses
//   std::vector<__float128>.
//
//   Chunk E's mirror check -- "no host-side test TU may contain a kernel
//   launch", the other half of scripts/check_device_tu_purity.sh -- found it on
//   its first run, on line 130 of the pre-split file. That is the gate working,
//   not a false positive, and it is why the mirror was worth writing: the
//   forward check (no binary128 in a device TU) cannot see a host TU that
//   quietly acquired a launch.
//
// WHAT IT CHECKS, AND WHY IT NEEDS NO ORACLE
//   The op is the identity: a kernel reads (hi, lo) limb pairs, rebuilds a
//   dd::DoubleDouble from them, and writes its limbs back. So the correct output
//   is the input, BIT FOR BIT -- not "to within max_digits of a binary128
//   reference", which is what the pre-split file asserted. Bit-equality is
//   strictly stronger and needs no reference value at all, so the whole
//   assertion fits in a device-pure TU. The host half still proves separately,
//   over the same 10^6 inputs, that DD(x) -> binary128 is exactly x; compose the
//   two and the original claim is intact.
//
//   NOTHING HERE SCORES. Every check below is `==` on a pair of doubles.
//   docs/CORRECTNESS.md permits one verdict per point and the sweep issues it.
//
// WHY THE BUFFERS ARE double AND NOT dd::DoubleDouble
//   xpt::buffer static_asserts std::is_trivially_copyable<T>, because it moves
//   bytes with memcpy/cudaMemcpy. xp::DoubleDouble declares its own copy
//   constructor and copy assignment -- both of which do exactly what the
//   implicit ones would -- and a USER-PROVIDED copy member makes a type
//   non-trivially-copyable by rule, whatever its body says. So the type cannot
//   ride in a buffer<> today. The assertion is right and is left alone: it is
//   guarding a real hazard, and the alternative (=default those members across
//   the four backends) is a library change with device-codegen consequences.
//
//   The components travel instead, one buffer per limb, and the kernel builds a
//   real dd::DoubleDouble from them and writes its limbs back. Nothing about the
//   round trip is weakened -- the same 2*10^6 doubles cross the bus -- and the
//   test makes no layout assumption about the struct in exchange.
//
// Built by g++ here, so what runs is the harness's SERIAL backend -- the same
// thing Kokkos's Serial execution space gave this test before. C5 gives it a
// device tree and C7/C8 run it on real hardware.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include <xp/dd_math.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

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
    const xp::DoubleDouble v(in_hi[i], in_lo[i]);
    out_hi[i] = v.hi;
    out_lo[i] = v.lo;
  }
};

int main() {
  int rc = 0;
  {
    constexpr int      n    = 1'000'000;
    constexpr uint64_t seed = 12345ULL;

    std::printf("hello_test_device: launch backend = %s\n", xpt::where_name());

    xpt::buffer<double> in_hi(n), in_lo(n), out_hi(n), out_lo(n);

    // The SAME generator and distribution the host half uses, so the two halves
    // cover the same 10^6 inputs even though they are separate processes.
    std::vector<double> ref_hi(n), ref_lo(n);
    {
      // Constructed INSIDE the loop, matching test_utils_host.hpp's uniform()
      // lambda exactly, so the two halves really do draw the same sequence
      // rather than merely the same distribution.
      std::mt19937_64 g(seed);
      for (int i = 0; i < n; ++i) {
        std::uniform_real_distribution<double> d(-1e8, 1e8);
        const double           x = d(g);
        const xp::DoubleDouble v(x);
        in_hi.host()[i] = v.hi;
        in_lo.host()[i] = v.lo;
        ref_hi[i]       = v.hi;   // identity: the expected output IS the input
        ref_lo[i]       = v.lo;
      }
    }
    in_hi.to_device();
    in_lo.to_device();

    // POISON THE OUTPUT ON BOTH SIDES. A kernel that never launched, or a
    // from_device() that copied nothing, must be RED rather than quiet -- and
    // reading back whatever malloc happened to leave in the buffer is exactly
    // how that goes quiet. The poison is outside the input range, so any element
    // that survives fails the bit-equality below; the explicit count makes the
    // diagnostic say which failure it was.
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
    int mism     = 0;
    for (int i = 0; i < n; ++i) {
      if (out_hi.host()[i] == kPoison && out_lo.host()[i] == kPoison) ++survived;
      if (out_hi.host()[i] != ref_hi[i] || out_lo.host()[i] != ref_lo[i]) {
        if (mism < 5) {
          std::printf("MISMATCH i=%d  in=(%.17g,%.17g)  out=(%.17g,%.17g)\n",
                      i, ref_hi[i], ref_lo[i], out_hi.host()[i], out_lo.host()[i]);
        }
        ++mism;
      }
    }

    KOKKOS_EP_ASSERT(survived == 0,
                     "poison survived the launch: the kernel or the copy-back "
                     "did not run");
    KOKKOS_EP_ASSERT(mism == 0,
                     "device identity did not return its input bit for bit");
    std::printf("hello_test_device: harness identity  %d/%d bit-exact\n",
                n - mism, n);

    rc = ep_exit_code();
  }
  return rc;
}
