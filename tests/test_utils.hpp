#pragma once
// ============================================================================
// test_utils.hpp — shared harness for the extended-precision test suite (T0.1)
// ============================================================================
//
// Framework choice: CTest + this lightweight header. No GoogleTest / Catch2.
//   Rationale: adding a unit-test framework is scope creep for a skeleton.
//   CTest drives the binaries; KOKKOS_EP_ASSERT below is the only assertion
//   primitive a test file needs. If the layered tests (Phase 1+) grow complex
//   enough to want fixtures/parameterization from a real framework, we revisit
//   then — see docs/TEST_SUITE_PLAN.md T0.1 ("Framework decision is part of
//   T0.1"). This mirrors the printf+return-code posture of the seed example
//   scripts/test_ffmul.cpp (on branch fffunKokkos), generalized so it scales to
//   6 layers x 3 backends without duplicating the device-runner plumbing.
//
// Backend abstraction: TRAITS TEMPLATE (BackendTraits<Backend>), NOT CRTP or
//   per-file #ifdef.
//   Rationale: a single test file (e.g. a T1.4 accuracy test) can be written
//   once against BackendTraits<Backend> and instantiated for DD today and FF/QF
//   in Phase 2/3 with zero source duplication. CRTP would force the backends to
//   share a base class they don't have; #ifdef per file would fork the source
//   per backend, which is exactly what this harness exists to avoid.
//
// Oracle: THERE IS NO LONGER A LIBQUADMATH DEPENDENCE, AND SO NO SKIP.
//   This header used to reach the __float128 oracle through Kokkos's quadmath
//   overloads (impl/Kokkos_QuadPrecisionMath.hpp), which exist only when Kokkos
//   was built with Kokkos_ENABLE_LIBQUADMATH=ON. tests/CMakeLists.txt defined
//   KOKKOS_EP_HAVE_QUADMATH iff LIBQUADMATH was in Kokkos_TPLS; without it every
//   oracle-dependent region was #ifdef'd out and the test's main() returned
//   KOKKOS_EP_SKIP (77) -> CTest "Skipped".
//
//   That was graceful degradation while the oracle genuinely needed libquadmath.
//   It stopped being graceful the moment it didn't: a skip is indistinguishable
//   from a pass in a summary line, so a machine whose Kokkos happens to lack the
//   TPL reported a green suite that had checked nothing. The MI250 install is
//   exactly such a machine.
//
//   __float128 SURVIVES AS THE CARRIER — it is a COMPILER type, and its
//   arithmetic comes from libgcc (__addtf3 and friends), not from libquadmath.
//   What libquadmath supplied was only the `*q` spellings of the elementary
//   functions, and every one this header needed is either an exact bit operation
//   or an IEEE-mandated correctly-rounded one, so all of them move to glibc's
//   `*f128` entry points via __builtin_. See the q_* block below, and
//   scripts/sweep_accuracy.cpp's "WHAT <quadmath.h> USED TO SUPPLY" section for
//   the same taxonomy applied to the sweep.
// ============================================================================

#include <Kokkos_Core.hpp>

#include <dd_math.hpp>
#include <ff_math.hpp>

// Corner-case corpus (T0.2). Included at file scope (outside namespace
// kokkos_ep) because corpus.hpp declares its own namespace kokkos_ep::corpus;
// the actual integration note lives at the extension point further down.
#include "corpus.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace kokkos_ep {

using float128 = __float128;

// ---------------------------------------------------------------------------
// WHAT THE Kokkos:: __float128 OVERLOADS USED TO SUPPLY, AND WHERE IT COMES
// FROM NOW
// ---------------------------------------------------------------------------
// Every call below is either a BIT operation on binary128 (sign, magnitude, the
// classification predicates) or an operation IEEE 754 requires to be correctly
// rounded. None of them is an approximation, so none of them is a second
// opinion about what the right answer is: there is exactly one, and the
// compiler knows it. That is the same test scripts/sweep_accuracy.cpp applies,
// and it is why these can move to __builtin_ without a measurement while a
// genuine transcendental REFERENCE (exp_reduction_test's ln2) had to go to MPFR
// instead.
//
// q_abs / q_isnan / q_isinf compile to inline bit arithmetic with no call at
// all. q_sqrt and q_log10 resolve to glibc's binary128 entry points in libm.
//
// NAMED q_* AND NOT abs/isnan. Keeping the Kokkos:: spellings would have made
// the diff smaller and every call site misleading: a reader who sees
// Kokkos::abs(float128) reasonably concludes Kokkos_QuadPrecisionMath.hpp is
// included and therefore that LIBQUADMATH is in the link line. The rename is
// the point. (Mirrors the q_* block in scripts/sweep_accuracy.cpp, which made
// the same move for the same reason.)
inline float128 q_abs  (float128 v)   { return __builtin_fabsq(v); }
inline bool     q_isnan(float128 v)   { return __builtin_isnan(v) != 0; }
inline bool     q_isinf(float128 v)   { return __builtin_isinf(v) != 0; }

// IEEE 754 REQUIRES sqrt to be correctly rounded, so glibc's sqrtf128 and a
// 400-bit MPFR sqrt rounded to binary128 are the same number by definition.
// That matters here because this one IS used as a reference (the K1 kernels'
// truth value), unlike the rest of this block.
//
// MEASURED, AND IT IS NOT A WASH: over 50,000 random positive binary128 inputs
// spanning 2^-300..2^300, arbitrated against mpfr_sqrt at 400 bits rounded once
// to 113, __builtin_sqrtf128 was bit-exact in 50,000 of 50,000 -- and sqrtq,
// the call this replaced, was WRONG IN 12,504 of them (25%, all by one ulp of
// binary128). libquadmath's sqrtq is not correctly rounded. So this swap did
// not downgrade the reference to keep the build simple; it removed a 1-ulp
// error that the old reference had been injecting into the K1 truth value.
inline float128 q_sqrt (float128 v)   { return __builtin_sqrtf128(v); }

// log10 of a magnitude, used only to COUNT DECIMAL DIGITS of agreement. Every
// caller casts straight to double, so the binary128 tail was never observable
// and this is not a reference. Measured anyway, on the same 200,000-input
// sweep: __builtin_log10f128 and log10q were BIT-IDENTICAL in every one, so
// the digit counts this feeds are unchanged even before the cast.
inline float128 q_log10(float128 v)   { return __builtin_log10f128(v); }

// Exact: scale by a power of two. Used by ulp_error to apply 2^p, and to build
// the exactly-representable 2^-k bounds in qf_eft_test.
inline float128 q_ldexp(float128 v, int e) { return __builtin_ldexpf128(v, e); }

// ============================================================================
// Backend tags and traits
// ============================================================================
// Tag types (NOT the arithmetic types). A test file instantiates its logic on a
// tag; BackendTraits<Tag> maps the tag to the concrete type + metadata.

namespace dd = Kokkos::Experimental;

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

  // Widen a DD value to the oracle type. Bit-exact: hi + lo with no rounding
  // because |lo| <= 1/2 ulp(hi) and __float128 has far more mantissa. Mirrors
  // dd_to_q() in src/demo_real.cpp.
  static float128 to_quad(type x) {
    return (float128)x.hi + (float128)x.lo;
  }
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
  using type = Kokkos::Experimental::FloatFloat;

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

  // Widen an FF value to the oracle type. Bit-exact: hi + lo with no rounding
  // because |lo| <= 1/2 ulp(hi) and __float128 has far more mantissa. Mirrors
  // ff_to_q() in src/demo_ff_real.cpp.
  static float128 to_quad(type x) {
    return (float128)x.hi + (float128)x.lo;
  }
};

// ============================================================================
// RNG-seeded input generators
// ============================================================================
// A generator is any callable double(std::mt19937_64&). Test files pass one to
// the device runners; the runner owns the seeded engine so runs are
// reproducible from (seed, n) alone.

using InputDist = std::function<double(std::mt19937_64&)>;

// Uniform real on [lo, hi].
inline InputDist uniform(double lo, double hi) {
  return [lo, hi](std::mt19937_64& g) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(g);
  };
}

// ---------------------------------------------------------------------------
// Corner-case corpus (T0.2). corpus.hpp provides deterministic corner-case
// inputs (subnormals, +/-0, +/-inf, NaN, powers of two, nextafter neighbors,
// near-cancellation pairs, huge/tiny mixes, half-integer boundaries) plus the
// explicit PORT_NOTES §3/§4 regression accessors. It returns MATERIALIZED
// vectors (std::vector<T> / std::vector<std::pair<T,T>>), not InputDist
// generators, because corpus entries are fixed constants rather than random
// draws. The random-pass runners above stay generator-driven. The corpus-pass
// runners that used to sit below consumed the
// corpus vectors. A full accuracy test runs BOTH passes: random for breadth,
// corpus for the pathological inputs uniform random misses.
// (corpus.hpp is #included at the top of this file, at file scope.)
// ---------------------------------------------------------------------------

// ============================================================================
// Accuracy: relative error and digits-of-accuracy vs the oracle
// ============================================================================

// Relative error of a device-under-test result (already widened to float128 by
// the caller via BackendTraits<Backend>::to_quad) against the oracle reference.
// Returns 0 for an exact match; +inf sentinel handling is left to
// digits_of_accuracy, which is what tests should use.
template <typename Backend>
inline float128 rel_err(float128 dut_quad, float128 ref) {
  if (ref == (float128)0.0) {
    return (dut_quad == (float128)0.0) ? (float128)0.0 : (float128)1.0;
  }
  return q_abs((dut_quad - ref) / ref);
}

// Digits of accuracy = -log10(rel_err), clamped to [0, max_digits]. Extracted
// verbatim in spirit from element_digits() in src/demo_real.cpp so the harness
// and the demo agree on the definition (NaN/inf/zero handling included).
template <typename Backend>
inline double digits_of_accuracy(float128 dut_quad, float128 ref) {
  const double max_digits = (double)BackendTraits<Backend>::max_digits;
  if (q_isnan(dut_quad) || q_isnan(ref)) return 0.0;
  if (q_isinf(ref)) {
    return (q_isinf(dut_quad) && (dut_quad > 0) == (ref > 0)) ? max_digits : 0.0;
  }
  if (ref == (float128)0.0) {
    return (dut_quad == (float128)0.0) ? max_digits : 0.0;
  }
  float128 rel = q_abs((dut_quad - ref) / ref);
  if (rel == (float128)0.0) return max_digits;
  double d = -(double)q_log10(rel);
  return d < 0.0 ? 0.0 : (d > max_digits ? max_digits : d);
}

// ============================================================================
// ULP ERROR — the second metric (see docs/ULP_METRIC.md)
// ============================================================================
// digits_of_accuracy above is a RELATIVE error, and a relative error goes
// vacuous exactly where the bugs are: near a zero of the function. DD sin(3*pi)
// scores 15.90 digits — a 21-digit gate passes it — while sitting 1.0e16 ulps
// from the true value. KI-4 (DD sin returning the wrong SIGN near odd multiples
// of pi) lived in that blind spot and these tests never saw it.
//
// So every scored element ALSO gets an ulp error:
//
//     ulp(true) = |true| * 2^-p       p = the expansion's significand bits
//     ulps      = |got - true| / ulp(true)
//
// BOTH metrics are reported. The digit score is NOT retired: docs/DOMAINS.md and
// the entire KI history are written in digits, and discarding it orphans all of
// that.
//
// p = limbs x word significand (FP64 53, FP32 24, hidden bit included):
//     DD 106 (2xFP64)   QF 96 (4xFP32)   TF 72 (3xFP32)   FF 48 (2xFP32)
// which is the same arithmetic that makes widening to binary128 exact.
//
// ZERO / NON-FINITE REFERENCE. ulp(0) is not defined, so the element is marked
// UNSCORABLE and excluded from the ulp statistic rather than being scored
// against a zero ulp (which would make every miss infinite and every op with a
// zero in its range unreportable). The digit score already handles those cells.
// A non-finite `got` against a finite `ref` is a real, reportable +inf.
constexpr double kUlpUnscorable = -1.0;   // sentinel

template <typename Backend>
inline double ulp_error(float128 dut_quad, float128 ref) {
  const int p = BackendTraits<Backend>::sig_bits;
  if (q_isnan(ref) || q_isinf(ref)) return kUlpUnscorable;
  if (ref == (float128)0.0) return (dut_quad == (float128)0.0) ? 0.0 : kUlpUnscorable;
  if (q_isnan(dut_quad) || q_isinf(dut_quad)) return HUGE_VAL;
  if (dut_quad == ref) return 0.0;
  // (|got-ref|/|ref|) * 2^p, scaled last so the ratio cannot overflow binary128.
  const float128 rel = q_abs((dut_quad - ref) / ref);
  return (double)q_ldexp(rel, p);
}


// ============================================================================
// Stat reporting
// ============================================================================

struct AccStats {
  double min = 0, max = 0, mean = 0, median = 0;
  int    n   = 0;
  // Second metric, carried alongside. ulp_max is the WORST point, never a mean:
  // an op that is exact at 1600 points and wrong at 52 still averages clean.
  double ulp_max        = 0.0;
  int    n_ulp_scored   = 0;
  int    n_ulp_unscored = 0;   // ref was zero / non-finite; see ulp_error()
};

// Reduce a parallel array of per-element ulp errors into an AccStats. Split out
// so the bespoke runners in the QF/TF/complex tests can call it directly.
inline void accumulate_ulps(AccStats& s, const double* ulps, int n) {
  for (int i = 0; i < n; ++i) {
    if (ulps[i] == kUlpUnscorable) { ++s.n_ulp_unscored; continue; }
    ++s.n_ulp_scored;
    if (ulps[i] > s.ulp_max) s.ulp_max = ulps[i];
  }
}

// Worst-point union of two ulp summaries (random pass + corpus pass).
inline void merge_ulps(AccStats& into, const AccStats& from) {
  if (from.ulp_max > into.ulp_max) into.ulp_max = from.ulp_max;
  into.n_ulp_scored   += from.n_ulp_scored;
  into.n_ulp_unscored += from.n_ulp_unscored;
}

// Compute min/max/mean/median over a digit array. Sorts a local copy.
inline AccStats compute_stats(const double* digits, int n) {
  AccStats s;
  s.n = n;
  if (n <= 0) return s;
  std::vector<double> v(digits, digits + n);
  std::sort(v.begin(), v.end());
  s.min    = v.front();
  s.max    = v.back();
  s.mean   = std::accumulate(v.begin(), v.end(), 0.0) / (double)n;
  size_t m = v.size();
  s.median = (m % 2 == 1) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
  return s;
}

// Overload that also folds in the ulp array. Same digit stats, byte for byte.
inline AccStats compute_stats(const double* digits, const double* ulps, int n) {
  AccStats s = compute_stats(digits, n);
  if (n > 0) accumulate_ulps(s, ulps, n);
  return s;
}

inline void print_stats(const char* label, const AccStats& s) {
  std::printf("  %-24s  n=%d  min=%.3f  mean=%.3f  median=%.3f  max=%.3f  (digits)"
              "  worst=%.4g ulp (n=%d, unscorable=%d)\n",
              label, s.n, s.min, s.mean, s.median, s.max,
              s.ulp_max, s.n_ulp_scored, s.n_ulp_unscored);
}

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
// Kokkos device runners
// ============================================================================
// These are the primitives every T*.4 accuracy test calls. Each:
//   1. generates n host inputs (double) from input_dist(seed),
//   2. deep-copies them into a device View of BackendTraits<Backend>::type,
//   3. runs a parallel_for applying device_op to each element on device,
//   4. copies results back to host,
//   5. widens each result via BackendTraits<Backend>::to_quad and compares to
//      host_oracle(input) with digits_of_accuracy,
//   6. returns AccStats over the per-element digit counts.
//
// device_op MUST be a device-callable functor (KOKKOS_LAMBDA / KOKKOS_FUNCTION)
// so it can be captured by value into the kernel. host_oracle runs on host only.

template <typename Backend, typename DeviceOp>
AccStats run_unary_op(int n, uint64_t seed,
                      const InputDist& input_dist,
                      const std::function<float128(float128)>& host_oracle,
                      DeviceOp device_op) {
  using T = typename BackendTraits<Backend>::type;
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_t     = Kokkos::View<T*,      Kokkos::LayoutRight, exec_space>;

  // 1. host inputs
  std::vector<double>   hin(n);
  std::vector<float128> href(n);
  {
    std::mt19937_64 gen(seed);
    for (int i = 0; i < n; ++i) hin[i] = input_dist(gen);
  }
  for (int i = 0; i < n; ++i) href[i] = host_oracle((float128)hin[i]);

  // 2. inputs -> device
  view_t din("din", n), dout("dout", n);
  auto hmir = Kokkos::create_mirror_view(din);
  for (int i = 0; i < n; ++i) hmir(i) = T(hin[i]);
  Kokkos::deep_copy(din, hmir);

  // 3. run op on device
  Kokkos::parallel_for("run_unary_op", Kokkos::RangePolicy<exec_space>(0, n),
                       KOKKOS_LAMBDA(int i) { dout(i) = device_op(din(i)); });
  Kokkos::fence();

  // 4. results -> host
  auto rmir = Kokkos::create_mirror_view(dout);
  Kokkos::deep_copy(rmir, dout);

  // 5. per-element accuracy — BOTH metrics, from the same widened result.
  std::vector<double> digs(n), ulps(n);
  for (int i = 0; i < n; ++i) {
    float128 got = BackendTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
    ulps[i] = ulp_error<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), ulps.data(), n);
}

template <typename Backend, typename DeviceOp>
AccStats run_binary_op(int n, uint64_t seed,
                       const InputDist& input_dist_a,
                       const InputDist& input_dist_b,
                       const std::function<float128(float128, float128)>& host_oracle,
                       DeviceOp device_op) {
  using T = typename BackendTraits<Backend>::type;
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_t     = Kokkos::View<T*,      Kokkos::LayoutRight, exec_space>;

  // 1. host inputs. One engine drives both streams (a then b per element) so a
  //    run is fully reproducible from (seed, n).
  std::vector<double>   ha(n), hb(n);
  std::vector<float128> href(n);
  {
    std::mt19937_64 gen(seed);
    for (int i = 0; i < n; ++i) { ha[i] = input_dist_a(gen); hb[i] = input_dist_b(gen); }
  }
  for (int i = 0; i < n; ++i) href[i] = host_oracle((float128)ha[i], (float128)hb[i]);

  // 2. inputs -> device
  view_t da("da", n), db("db", n), dout("dout", n);
  auto hma = Kokkos::create_mirror_view(da);
  auto hmb = Kokkos::create_mirror_view(db);
  for (int i = 0; i < n; ++i) { hma(i) = T(ha[i]); hmb(i) = T(hb[i]); }
  Kokkos::deep_copy(da, hma);
  Kokkos::deep_copy(db, hmb);

  // 3. run op on device
  Kokkos::parallel_for("run_binary_op", Kokkos::RangePolicy<exec_space>(0, n),
                       KOKKOS_LAMBDA(int i) { dout(i) = device_op(da(i), db(i)); });
  Kokkos::fence();

  // 4. results -> host
  auto rmir = Kokkos::create_mirror_view(dout);
  Kokkos::deep_copy(rmir, dout);

  // 5. per-element accuracy — BOTH metrics, from the same widened result.
  std::vector<double> digs(n), ulps(n);
  for (int i = 0; i < n; ++i) {
    float128 got = BackendTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
    ulps[i] = ulp_error<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), ulps.data(), n);
}

// --- Corpus-pass runners ---------------------------------------------------
// Same host->device->host->oracle pipeline as run_unary_op/run_binary_op, but
// driven by a caller-supplied deterministic input vector (from corpus.hpp)
// instead of (seed, n) + generator. Tests call these for the corpus pass; the
// generator-based runners above are unchanged for the random pass.




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
