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
// Oracle: the host __float128 oracle is provided by Kokkos's quadmath overloads
//   (impl/Kokkos_QuadPrecisionMath.hpp), available only when Kokkos was built
//   with Kokkos_ENABLE_LIBQUADMATH=ON. The build system (tests/CMakeLists.txt)
//   defines KOKKOS_EP_HAVE_QUADMATH iff LIBQUADMATH is in Kokkos_TPLS. When it
//   is absent, oracle-dependent code is #ifdef'd out and a test's main() should
//   return KOKKOS_EP_SKIP (77), which CTest reports as "Skipped" (see
//   SKIP_RETURN_CODE in tests/CMakeLists.txt). Graceful-degradation choice:
//   SKIP, not fail-loud — a legitimately quadmath-less Kokkos config should not
//   turn the whole suite red; a visible "Skipped" is the honest signal. Same
//   posture T0.0/T0.3 used for the demos.
// ============================================================================

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_EP_HAVE_QUADMATH
// Pulls in <quadmath.h> and the Kokkos:: __float128 math overloads used as the
// oracle. Host-only (inline, not KOKKOS_INLINE_FUNCTION) — perfect for oracle
// computation, which happens on host after results are copied back.
#  include <impl/Kokkos_QuadPrecisionMath.hpp>
#endif

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

// Exit code CTest maps to "Skipped" (see SKIP_RETURN_CODE in CMakeLists.txt).
constexpr int KOKKOS_EP_SKIP = 77;

#ifdef KOKKOS_EP_HAVE_QUADMATH
using float128 = __float128;
#endif

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

  // Oracle (quadmath) has ~34 digits; DD targets ~31.9. Cap digit counts at 31
  // to avoid reporting oracle noise as accuracy. Matches kMaxDigits_dd in
  // src/demo_real.cpp.
  static constexpr int max_digits = 31;

  static const char* name() { return "DD"; }

#ifdef KOKKOS_EP_HAVE_QUADMATH
  // Widen a DD value to the oracle type. Bit-exact: hi + lo with no rounding
  // because |lo| <= 1/2 ulp(hi) and __float128 has far more mantissa. Mirrors
  // dd_to_q() in src/demo_real.cpp.
  static float128 to_quad(type x) {
    return (float128)x.hi + (float128)x.lo;
  }
#endif
};

// ---------------------------------------------------------------------------
// FF (float-float, 2 x FP32). Merged onto main in T2.0. Mirrors the DD entry
// above field-for-field so the T2.2/T2.3/T2.4/T2.6 tests can instantiate the
// same runner/accuracy machinery against FF with zero source duplication.
//
// This entry is what T2.1 is required to add even though the T2.1 EFT test is
// self-contained (it does NOT call the run_* runners or digits_of_accuracy —
// its oracle is exact FP64, not the __float128 quadmath oracle these helpers
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

  static const char* name() { return "FF"; }

#ifdef KOKKOS_EP_HAVE_QUADMATH
  // Widen an FF value to the oracle type. Bit-exact: hi + lo with no rounding
  // because |lo| <= 1/2 ulp(hi) and __float128 has far more mantissa. Mirrors
  // ff_to_q() in src/demo_ff_real.cpp.
  static float128 to_quad(type x) {
    return (float128)x.hi + (float128)x.lo;
  }
#endif
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
// draws. The random-pass runners above stay generator-driven; the corpus-pass
// runners below (run_unary_op_on_corpus / run_binary_op_on_corpus) consume the
// corpus vectors. A full accuracy test runs BOTH passes: random for breadth,
// corpus for the pathological inputs uniform random misses.
// (corpus.hpp is #included at the top of this file, at file scope.)
// ---------------------------------------------------------------------------

// ============================================================================
// Accuracy: relative error and digits-of-accuracy vs the oracle
// ============================================================================

#ifdef KOKKOS_EP_HAVE_QUADMATH

// Relative error of a device-under-test result (already widened to float128 by
// the caller via BackendTraits<Backend>::to_quad) against the oracle reference.
// Returns 0 for an exact match; +inf sentinel handling is left to
// digits_of_accuracy, which is what tests should use.
template <typename Backend>
inline float128 rel_err(float128 dut_quad, float128 ref) {
  if (ref == (float128)0.0) {
    return (dut_quad == (float128)0.0) ? (float128)0.0 : (float128)1.0;
  }
  return Kokkos::abs((dut_quad - ref) / ref);
}

// Digits of accuracy = -log10(rel_err), clamped to [0, max_digits]. Extracted
// verbatim in spirit from element_digits() in src/demo_real.cpp so the harness
// and the demo agree on the definition (NaN/inf/zero handling included).
template <typename Backend>
inline double digits_of_accuracy(float128 dut_quad, float128 ref) {
  const double max_digits = (double)BackendTraits<Backend>::max_digits;
  if (Kokkos::isnan(dut_quad) || Kokkos::isnan(ref)) return 0.0;
  if (Kokkos::isinf(ref)) {
    return (Kokkos::isinf(dut_quad) && (dut_quad > 0) == (ref > 0)) ? max_digits : 0.0;
  }
  if (ref == (float128)0.0) {
    return (dut_quad == (float128)0.0) ? max_digits : 0.0;
  }
  float128 rel = Kokkos::abs((dut_quad - ref) / ref);
  if (rel == (float128)0.0) return max_digits;
  double d = -(double)Kokkos::log10(rel);
  return d < 0.0 ? 0.0 : (d > max_digits ? max_digits : d);
}

#endif  // KOKKOS_EP_HAVE_QUADMATH

// ============================================================================
// Stat reporting
// ============================================================================

struct AccStats {
  double min = 0, max = 0, mean = 0, median = 0;
  int    n   = 0;
};

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

inline void print_stats(const char* label, const AccStats& s) {
  std::printf("  %-24s  n=%d  min=%.3f  mean=%.3f  median=%.3f  max=%.3f  (digits)\n",
              label, s.n, s.min, s.mean, s.median, s.max);
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
    {"fma",       "near-cancellation loses leading digits (matches FP64); PORT_NOTES §5", 0.0},
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

#ifdef KOKKOS_EP_HAVE_QUADMATH

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

  // 5. per-element accuracy
  std::vector<double> digs(n);
  for (int i = 0; i < n; ++i) {
    float128 got = BackendTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), n);
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

  // 5. per-element accuracy
  std::vector<double> digs(n);
  for (int i = 0; i < n; ++i) {
    float128 got = BackendTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), n);
}

// --- Corpus-pass runners ---------------------------------------------------
// Same host->device->host->oracle pipeline as run_unary_op/run_binary_op, but
// driven by a caller-supplied deterministic input vector (from corpus.hpp)
// instead of (seed, n) + generator. Tests call these for the corpus pass; the
// generator-based runners above are unchanged for the random pass.

template <typename Backend, typename DeviceOp>
AccStats run_unary_op_on_corpus(
    const std::vector<double>& inputs,
    const std::function<float128(float128)>& host_oracle,
    DeviceOp device_op) {
  using T = typename BackendTraits<Backend>::type;
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_t     = Kokkos::View<T*, Kokkos::LayoutRight, exec_space>;

  const int n = (int)inputs.size();
  if (n <= 0) return AccStats{};

  // 1. host oracle reference from the corpus inputs.
  std::vector<float128> href(n);
  for (int i = 0; i < n; ++i) href[i] = host_oracle((float128)inputs[i]);

  // 2. inputs -> device
  view_t din("din", n), dout("dout", n);
  auto hmir = Kokkos::create_mirror_view(din);
  for (int i = 0; i < n; ++i) hmir(i) = T(inputs[i]);
  Kokkos::deep_copy(din, hmir);

  // 3. run op on device
  Kokkos::parallel_for("run_unary_op_on_corpus", Kokkos::RangePolicy<exec_space>(0, n),
                       KOKKOS_LAMBDA(int i) { dout(i) = device_op(din(i)); });
  Kokkos::fence();

  // 4. results -> host
  auto rmir = Kokkos::create_mirror_view(dout);
  Kokkos::deep_copy(rmir, dout);

  // 5. per-element accuracy
  std::vector<double> digs(n);
  for (int i = 0; i < n; ++i) {
    float128 got = BackendTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), n);
}

template <typename Backend, typename DeviceOp>
AccStats run_binary_op_on_corpus(
    const std::vector<std::pair<double, double>>& inputs,
    const std::function<float128(float128, float128)>& host_oracle,
    DeviceOp device_op) {
  using T = typename BackendTraits<Backend>::type;
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_t     = Kokkos::View<T*, Kokkos::LayoutRight, exec_space>;

  const int n = (int)inputs.size();
  if (n <= 0) return AccStats{};

  // 1. host oracle reference from the corpus pairs.
  std::vector<float128> href(n);
  for (int i = 0; i < n; ++i)
    href[i] = host_oracle((float128)inputs[i].first, (float128)inputs[i].second);

  // 2. inputs -> device
  view_t da("da", n), db("db", n), dout("dout", n);
  auto hma = Kokkos::create_mirror_view(da);
  auto hmb = Kokkos::create_mirror_view(db);
  for (int i = 0; i < n; ++i) { hma(i) = T(inputs[i].first); hmb(i) = T(inputs[i].second); }
  Kokkos::deep_copy(da, hma);
  Kokkos::deep_copy(db, hmb);

  // 3. run op on device
  Kokkos::parallel_for("run_binary_op_on_corpus", Kokkos::RangePolicy<exec_space>(0, n),
                       KOKKOS_LAMBDA(int i) { dout(i) = device_op(da(i), db(i)); });
  Kokkos::fence();

  // 4. results -> host
  auto rmir = Kokkos::create_mirror_view(dout);
  Kokkos::deep_copy(rmir, dout);

  // 5. per-element accuracy
  std::vector<double> digs(n);
  for (int i = 0; i < n; ++i) {
    float128 got = BackendTraits<Backend>::to_quad(rmir(i));
    digs[i] = digits_of_accuracy<Backend>(got, href[i]);
  }

  // 6. stats
  return compute_stats(digs.data(), n);
}

#endif  // KOKKOS_EP_HAVE_QUADMATH

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
