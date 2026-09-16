// ============================================================================
// tests/test_utils_device.hpp — the DEVICE-SAFE half of the test harness
// ============================================================================
// Split out of tests/test_utils.hpp by CORE_PLAN section C2.
//
// WHAT BELONGS HERE, AND THE ONE RULE THAT DECIDES IT
// Anything a DEVICE translation unit may include. Concretely that means this
// header must never name `__float128`, and must never instantiate a template
// on it. `std::vector<__float128>` is what nvcc rejects -- it walks STL member
// signatures during the device pass and refuses the 128-bit float -- and the
// unsplit tests/test_utils.hpp put that instantiation into all 21 of the test
// TUs that included it, whether or not they used the oracle.
//
// WHAT THE SPLIT DOES NOT DO, MEASURED IN C2. It does not by itself rescue any
// currently-registered test for device. All 16 host-classified TUs name an
// oracle symbol directly, hello_test among them -- it uses float128 and
// OracleTraits at hello_test.cpp:41 and AccStats at :56, so the C2 plan's
// description of it as containing no oracle code of its own is wrong, and was
// wrong before the split. What the split buys is an HONEST boundary: the five
// device-classified TUs now include no oracle at all, and C4's device TUs have
// something to include that cannot silently drag one in.
//
// Plain <vector> is NOT the problem and is not banned: Kokkos uses it
// throughout and compiles under nvcc. The banned thing is the 128-bit type.
//
// The companion is tests/test_utils_host.hpp, which includes THIS header and
// adds the oracle surface. There is deliberately no forwarding shim at the old
// name: a shim would let a device TU pull in the oracle by accident, which is
// the exact failure this split exists to prevent.
//
// Enforced by scripts/check_device_tu_purity.sh (ctest target device_tu_purity).
// ============================================================================

#pragma once

// The xp CORE headers, NOT the Kokkos compat wrappers in third_party/include.
// This header must preprocess with -Iinclude alone, so that the purity gate
// (scripts/check_device_tu_purity.sh) can run in the Kokkos-free build rather
// than only where a Kokkos install happens to be.
//
// This costs nothing in type identity: third_party/include/dd_math.hpp says
// `using DoubleDouble = xp::DoubleDouble` -- a true alias, not a distinct
// wrapper type -- so naming the core type here names exactly the type
// `Kokkos::Experimental::DoubleDouble` names. A TU that additionally wants the
// Kokkos:: spellings, or the Kokkos runtime, includes those headers itself.
#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace kokkos_ep {

// ============================================================================
// Backend tags and traits
// ============================================================================
// Tag types (NOT the arithmetic types). A test file instantiates its logic on a
// tag; BackendTraits<Tag> maps the tag to the concrete type + metadata.

namespace dd = xp;

struct DD {};  // double-double (2 x FP64)
struct FF {};  // float-float  (2 x FP32), backend merged onto main in T2.0
// TODO(Phase 3): struct QF {};  // quad-float  (4 x FP32), backend on qffunKokkos

template <typename Backend>
struct BackendTraits;  // primary template intentionally undefined

template <>
struct BackendTraits<DD> {
  using type = dd::DoubleDouble;

  // u = 2^-53 (FP64 unit roundoff); DD carries ~2u^2 worth of tail, so the
  // relevant scale for double-word error bounds is u^2 = 2^-106.
  static constexpr double u_squared = 1.0 / 9007199254740992.0    // 2^-53
                                    / 9007199254740992.0;         // * 2^-53 = 2^-106

  // The __float128 oracle has ~34 digits; DD targets ~31.9. Cap digit counts at 31
  // to avoid reporting oracle noise as accuracy. Matches kMaxDigits_dd in
  // src/demo_real.cpp.
  static constexpr int max_digits = 31;

  // p in ulp(true) = |true| * 2^-p. 2 x FP64 significand (53 bits each,
  // hidden bit included). See the ULP ERROR section below.
  static constexpr int sig_bits = 106;

  static const char* name() { return "DD"; }

};

// ---------------------------------------------------------------------------
// FF (float-float, 2 x FP32). Merged onto main in T2.0. Mirrors the DD entry
// above field-for-field so the T2.2/T2.3/T2.4/T2.6 tests can instantiate the
// same runner/accuracy machinery against FF with zero source duplication.
//
// This entry is what T2.1 is required to add even though the T2.1 EFT test is
// self-contained (it does NOT call the run_* runners or digits_of_accuracy —
// its oracle is exact FP64, not the __float128 oracle these helpers
// use). Provided here so the later FF layers are unblocked.
// ---------------------------------------------------------------------------
template <>
struct BackendTraits<FF> {
  using type = xp::FloatFloat;

  // u = 2^-24 (FP32 unit roundoff). FF carries two FP32 limbs, so the relevant
  // scale for double-word (float-float) error bounds is u^2 = 2^-48. (The DD
  // analogue is u = 2^-53, u^2 = 2^-106.)
  static constexpr double u          = 1.0 / 16777216.0;              // 2^-24
  static constexpr double u_squared  = (1.0 / 16777216.0)            // 2^-24
                                     * (1.0 / 16777216.0);           // * 2^-24 = 2^-48

  // FP32 mantissa is 24 bits; two limbs give ~48 bits ~= -log10(2^-48) ~= 14.45
  // decimal digits. Cap digit counts at 14 to avoid reporting oracle noise as
  // accuracy. Matches kMaxDigits (14.0) in src/demo_ff_real.cpp.
  static constexpr int max_digits = 14;

  // p in ulp(true) = |true| * 2^-p. 2 x FP32 significand (24 bits each).
  static constexpr int sig_bits = 48;

  static const char* name() { return "FF"; }

};


// ============================================================================
// Expected-min-drop registry (PORT_NOTES §5 conditioning limits)
// ============================================================================
// Some ops legitimately show a low MIN digit count that is NOT a regression: the
// operation is conditioning-limited and no fixed-precision algorithm can do
// better (PORT_NOTES.md §5 on branch fffunKokkos). Tests fail-gate on the MEAN
// column but must NOT fail on the min for these ops; instead they report
// "expected-min-drop: OK". This registry lets a test ask, per op, whether a low
// min is expected and how low is tolerable.
//
// A test's reporting logic should:
//   const auto* ann = lookup_expected_min_drop(op_name);
//   if (ann && stats.min >= ann->min_digits_allowed) -> "expected-min-drop: OK"
//   else fail-gate on mean as usual.

struct ExpectedMinDropAnnotation {
  const char* op_name;
  const char* reason;
  double      min_digits_allowed;  // min may drop this low without being a regression
};

// Preloaded from PORT_NOTES §5. min_digits_allowed = 0.0 means "min may hit the
// floor" (pure conditioning; e.g. exact cancellation, derivative -> inf). Chosen
// as a static constexpr table + linear scan: the set is tiny and fixed at compile
// time, so a table is simpler and allocation-free next to std::map, and it reads
// as data rather than control flow next to an if/else chain.
inline const ExpectedMinDropAnnotation* lookup_expected_min_drop(const char* op_name) {
  static const ExpectedMinDropAnnotation kTable[] = {
    {"sub",       "near-cancellation loses leading digits (matches FP64); PORT_NOTES §5", 0.0},
    {"fdim",      "near-cancellation loses leading digits (matches FP64); PORT_NOTES §5", 0.0},
    // Ratcheted from 0.0 with KI-38 (exact product expansion in fma). fma no
    // longer loses digits to near-cancellation, so the old "matches FP64"
    // rationale was wrong on both counts. Observed min is now the cap on three
    // backends -- DD 31.00 / TF 21.70 / QF 29.00 -- and FF 12.69. FF is the
    // only one that drops, and not from algorithm error: FF's 48 bits cannot
    // hold the 53-bit test inputs, so its min is the input-storage floor.
    // 7.5 keeps a 5-digit margin under FF, matching the acos precedent below,
    // while no longer sanctioning a total loss.
    {"fma",       "min is FF's input-storage floor, not algorithm error (KI-38)",         7.5},
    {"asin",      "derivative 1/sqrt(1-a^2) -> inf near |a|=1; PORT_NOTES §5",            0.0},
    // Ratcheted from 0.0 with KI-4 (joint-doubling sincos). acos runs through
    // angle()->sincos, and the observed min rose to DD 29.96 / FF 10.08 / QF 25.91;
    // 5.0 keeps a 5-digit margin under the worst backend while no longer
    // sanctioning a total loss.
    {"acos",      "derivative 1/sqrt(1-a^2) -> inf near |a|=1; PORT_NOTES §5",            5.0},
    {"atanh",     "1/(1-a^2) blows up near |a|=1; PORT_NOTES §5",                          0.0},
    {"remainder", "a - b*nint(a/b) -> 0 with fixed abs error near multiples of b; PORT_NOTES §5", 0.0},
    {"exp",       "output denormal range: lo falls into subnormal, loses bits; PORT_NOTES §5", 0.0},
    // sin/cos ratcheted from 0.0 with KI-4. Conditioning near +/-pi still costs
    // roughly half the digits, but a *total* loss there was KI-4's sign flip, not
    // conditioning. Observed min DD 15.42 / FF 6.06 / QF 21.73 (sin) and DD 15.24 /
    // FF 6.02 / QF 21.87 (cos); 3.0 sits well under FF, the binding backend.
    // tan and asin stay at 0.0: QF's min is genuinely -0.00 / 0.00 on those.
    {"sin",       "near +/-pi needs triple-float arg reduction (out of scope); PORT_NOTES §5", 3.0},
    {"cos",       "near +/-pi needs triple-float arg reduction (out of scope); PORT_NOTES §5", 3.0},
    {"tan",       "near +/-pi needs triple-float arg reduction (out of scope); PORT_NOTES §5", 0.0},
  };
  for (const auto& e : kTable) {
    // std::strcmp without pulling <cstring> into every TU: compare inline.
    const char* a = op_name;
    const char* b = e.op_name;
    while (*a && (*a == *b)) { ++a; ++b; }
    if (*a == *b) return &e;  // both hit '\0' -> equal
  }
  return nullptr;
}


// ============================================================================
// Assertion macro
// ============================================================================
// Prints file:line + message on failure and flips a caller-visible failure
// flag. The test's main() returns nonzero iff any assertion failed. Deliberately
// minimal — see the framework-choice note at the top of this header.

// Each test file defines exactly one: `int g_ep_failures = 0;` at file scope.
#define KOKKOS_EP_ASSERT(cond, msg)                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("ASSERT FAILED %s:%d: %s\n", __FILE__, __LINE__, (msg));     \
      ++::kokkos_ep::detail::ep_failure_count();                              \
    }                                                                          \
  } while (0)

namespace detail {
// Single translation-unit-local failure counter reachable from the macro
// without each test having to declare a global. Defined inline (C++17) so it is
// shared per-TU without ODR issues.
inline int& ep_failure_count() {
  static int count = 0;
  return count;
}
}  // namespace detail

// Convenience: final exit code for a test's main().
inline int ep_exit_code() {
  return detail::ep_failure_count() == 0 ? 0 : 1;
}

}  // namespace kokkos_ep
