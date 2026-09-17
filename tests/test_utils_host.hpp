// ============================================================================
// tests/test_utils_host.hpp — the HOST-ORACLE half of the test harness
// ============================================================================
// Split out of tests/test_utils.hpp by CORE_PLAN section C2.
//
// This header carries everything that touches the __float128 oracle: the
// float128 alias, the q_* helpers, OracleTraits<B>::to_quad, the digit and ulp
// metrics, the statistics structs, and the Kokkos runners (whose host_oracle
// parameter is a std::function<float128(float128)>).
//
// A TU that includes this header is a HOST translation unit and cannot be
// compiled for a device target. If you are writing device-side test code,
// include tests/test_utils_device.hpp instead -- it is included below, so
// everything device-safe is available here too.
//
// WHY to_quad MOVED. It was BackendTraits<B>::to_quad, and it was the ONLY
// float128 member of that struct; everything else (type, u, u_squared,
// max_digits, sig_bits, name) is device-safe. Keeping one oracle method in an
// otherwise device-safe trait is what made the whole trait -- and so the whole
// header -- host-only. It now lives in OracleTraits<B>, which inherits
// BackendTraits<B>, so host code reads OracleTraits<Backend>::to_quad(x) and
// device code keeps the unqualified trait.
// ============================================================================

#pragma once

#include "test_utils_device.hpp"

// Kokkos, and the compat wrappers, live on THIS side of the split. The device
// header deliberately names only the xp core so it stays preprocessable without
// a Kokkos install; the runners at the bottom of this file use Kokkos::View /
// parallel_for, and the Kokkos-linked host TUs expect the Kokkos:: math
// spellings the unsplit tests/test_utils.hpp used to hand them.
//
// BUT HOST-ORACLE IS NOT THE SAME THING AS KOKKOS-LINKED, and until CORE_PLAN
// C4 this header conflated them. `corpus_test` references zero Kokkos APIs and
// still could not compile without a Kokkos install, purely because it needed
// float128 and ulp_error. So the Kokkos surface is now opt-in:
// XPMATH_TEST_HAVE_KOKKOS is defined by the kokkos_ep_add_test* helpers in
// tests/CMakeLists.txt and by nothing else. Without it this header is the
// oracle and the statistics, and needs no Kokkos on the include path.
//
// It is a POSITIVE define on the linked targets rather than a negative one on
// the unlinked targets on purpose: a new test that forgets the define fails to
// compile the moment it names Kokkos, whereas a forgotten
// XPMATH_TEST_NO_KOKKOS would silently pull the runtime back in.
#if defined(XPMATH_TEST_HAVE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <dd_math.hpp>
#include <ff_math.hpp>
#endif

// Corner-case corpus (T0.2). Included at file scope (outside namespace
// kokkos_ep) because corpus.hpp declares its own namespace kokkos_ep::corpus;
// the actual integration note lives at the extension point further down.
#include "corpus.hpp"

#include <algorithm>
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
// OracleTraits — the host-only extension of BackendTraits
// ============================================================================
// BackendTraits<B> lives in test_utils_device.hpp and is device-safe. This adds
// the one member that is not: widening to the binary128 oracle type. Inherits,
// so OracleTraits<B>::sig_bits and friends resolve exactly as before.
//
// Bit-exact widening: hi + lo with no rounding, because |lo| <= 1/2 ulp(hi) and
// __float128 has far more mantissa than either limb pair.

template <typename Backend>
struct OracleTraits;  // primary template intentionally undefined

template <>
struct OracleTraits<DD> : BackendTraits<DD> {
  static float128 to_quad(type x) {
    return (float128)x.hi + (float128)x.lo;
  }
};

template <>
struct OracleTraits<FF> : BackendTraits<FF> {
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
// the caller via OracleTraits<Backend>::to_quad) against the oracle reference.
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
// Kokkos device runners
// ============================================================================
// These are the primitives every T*.4 accuracy test calls. Each:
//   1. generates n host inputs (double) from input_dist(seed),
//   2. deep-copies them into a device View of BackendTraits<Backend>::type,
//   3. runs a parallel_for applying device_op to each element on device,
//   4. copies results back to host,
//   5. widens each result via OracleTraits<Backend>::to_quad and compares to
//      host_oracle(input) with digits_of_accuracy,
//   6. returns AccStats over the per-element digit counts.
//
// device_op MUST be a device-callable functor (KOKKOS_LAMBDA / KOKKOS_FUNCTION)
// so it can be captured by value into the kernel. host_oracle runs on host only.
//
// Compiled only for the Kokkos-linked targets (see the XPMATH_TEST_HAVE_KOKKOS
// note at the top). CORE_PLAN C4 migrates the remaining callers onto
// tests/device_harness.hpp, after which this block goes away entirely.
#if defined(XPMATH_TEST_HAVE_KOKKOS)

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
    float128 got = OracleTraits<Backend>::to_quad(rmir(i));
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
    float128 got = OracleTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
    ulps[i] = ulp_error<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), ulps.data(), n);
}

#endif  // XPMATH_TEST_HAVE_KOKKOS

// --- Corpus-pass runners ---------------------------------------------------
// Same host->device->host->oracle pipeline as run_unary_op/run_binary_op, but
// driven by a caller-supplied deterministic input vector (from corpus.hpp)
// instead of (seed, n) + generator. Tests call these for the corpus pass; the
// generator-based runners above are unchanged for the random pass.





}  // namespace kokkos_ep
