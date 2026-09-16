// ============================================================================
// dd_invariant_test.cpp — Layer 2 (output-invariant tests) for DD.  Plan T1.2.
// ============================================================================
//
// WHAT THIS LAYER CHECKS AND WHY IT IS ORACLE-INDEPENDENT
// ------------------------------------------------------
// Layer 1 (dd_eft_test.cpp, T1.1) proved the two atoms — twoSum and the Dekker
// twoProduct — are bit-exact. This layer checks a STRUCTURAL property of every
// higher-level op's OUTPUT: a double-double (hi, lo) must be NON-OVERLAPPING,
// i.e. lo carries only bits BELOW the last bit of hi. The canonical, bit-exact
// statement of that is
//
//     fl(hi + lo) == hi            (equivalently  |lo| <= 1/2 ulp(hi))
//
// evaluated in RAW FP64 (a single hardware add + compare). If lo held any bit at
// or above hi's ulp, the rounded sum hi+lo would land on a different double and
// the equality would fail. Every DD op is supposed to return a renormalized,
// non-overlapping pair; a violation localizes a normalization bug to that exact
// op — no reference value and no wider type is needed to see it. That is why this
// test carries NO __float128 oracle at all: it runs (and must run) on the
// narrowest possible toolchain. (It never carried the KOKKOS_EP_HAVE_QUADMATH
// guard either; that guard is now gone from the tests that DID carry it, so the
// distinction no longer separates this file from them.) Accuracy-vs-oracle is a separate
// concern handled in T1.4.
//
// WHY RAW FP64 EQUALITY, NOT A float128 PROMOTION
// -----------------------------------------------
// The invariant is a statement ABOUT FP64 rounding: "adding lo back into hi does
// not move hi". Promoting hi/lo to __float128 and comparing would test a DIFFERENT
// thing (the exact real sum), and would report a perfectly-normalized pair with a
// nonzero lo as "unequal". So the check is deliberately `(d.hi + d.lo) == d.hi`
// in double. (This TU is built with the project's normal flags — no contraction
// pragma needed: a lone add-then-compare has nothing to fuse.)
//
// SKIP (not FAIL) CRITERIA — a result outside the invariant's domain
//   * hi is NaN            — non-overlap is undefined; op fed an out-of-domain input
//   * hi is +/-inf         — overflowed; lo is meaningless
//   * hi is subnormal      — below the smallest normal, ulp(hi) is the fixed
//                            subnormal step and the "1/2 ulp" argument degrades;
//                            a nonzero lo there is not a normalization defect
//   * hi in the underflow tail (|hi| < 2^-967, i.e. 1/2 ulp(hi) itself subnormal)
//                            — the residual lo is forced onto quantized subnormal
//                            values that can land EXACTLY on the 1/2 ulp(hi) tie,
//                            where round-to-even flips fl(hi+lo) off hi by one ulp
//                            even though non-overlap |lo| <= 1/2 ulp(hi) still
//                            holds. A property of double-word arithmetic in the
//                            denormal tail, NOT a dd_math.hpp defect. See
//                            result_checkable's kUnderflowTail.
//   * input out of the op's mathematical domain (log of <=0, asin of |x|>1, ...)
//                            — also avoids the domain-guard diagnostics dd_math.hpp
//                            prints, which would otherwise spam the log
//
// TEST STRUCTURE
//   Test A — every op, two passes each: 10^6 op-appropriate random inputs, then
//            the full corner-case corpus (corpus.hpp). Per-op tested/skipped/
//            failures; a summary table at the end.
//   Test B — device tripwire: 5 representative ops (add, multiply, sqrt, exp, sin)
//            run the SAME invariant on 10^5 random inputs inside a parallel_for,
//            results copied back and checked on host. Catches a Serial->CUDA
//            regression (subnormal flush, contraction) the host pass cannot see.
//   Test C — explicit PORT_NOTES §4 regressions as named cases (documents that DD
//            has no equivalent of the FF bugs): exp in the 79.5..88.72 band,
//            round_to_nearest_int at the 19.4999993 / k+0.5 boundaries, and
//            remainder(68.379, 3.5066) with its expected positive sign.
//
// SCOPE: real DD ops only (dd_complex.hpp is out of scope). If a violation is
// found this test REPORTS it (op, counts, first offending bit patterns) and fails
// — it does not attempt any fix to dd_math.hpp.
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 1, "T1.2: non-overlap invariant
// checks for DD" and "The six test layers" layer 2.
// ============================================================================

#include "test_utils_host.hpp"
#include "corpus.hpp"
#include <dd_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace kokkos_ep;

// ----------------------------------------------------------------------------
// The invariant and its domain, on a raw DD value.
// ----------------------------------------------------------------------------

// The non-overlap invariant itself: adding lo back into hi must not move hi.
// Raw FP64 — a single add and a bit-for-bit double comparison.
inline bool invariant_holds(const dd::DoubleDouble& d) {
  return (d.hi + d.lo) == d.hi;
}

// Is this result in the invariant's domain (i.e. should we CHECK it, vs SKIP it)?
// NaN / inf / subnormal hi are out of domain — see the SKIP criteria header note.
// UNDERFLOW-TAIL guard (kUnderflowTail): the strict bit-exact form fl(hi+lo)==hi
// is only WELL-POSED when the tie point 1/2 ulp(hi) is itself a NORMAL double. For
// hi with unbiased exponent e, 1/2 ulp(hi) = 2^(e-53); this is subnormal once
// e < -969, i.e. |hi| < 2^-969 (~2.0e-292). In that near-underflow regime the
// residual lo is forced onto quantized subnormal values that land EXACTLY on the
// 1/2 ulp(hi) tie, where round-to-even flips fl(hi+lo) off hi by one ulp even
// though the mathematical non-overlap |lo| <= 1/2 ulp(hi) still holds.
//
// This is the DD sibling of ff_invariant_test's guard (T2.2), added per the
// KNOWN-LURKING note in TEST_SUITE_PLAN.md §T2.2: the hole is a property of the
// fl(hi+lo)==hi EVALUATION, not of FF, so DD can hit it too.
//
// IT IS REACHED TODAY — 86 results, not zero. That note predicted DD was out of
// reach because FP64's exponent range is ~6x wider, and that reasoning is right
// about OUTPUTS: dd exp is guarded at |arg| <= 300, bottoming out near 5e-131,
// 161 orders above this gate, so no DD op COMPUTES its way into the tail. But it
// misses the INPUT path. corpus.hpp's power-of-two ladder emits 2^-1022 (min
// normal), 2^-1021 and 2^-1018 with both signs — normal doubles that sit below
// this gate — and ~14 ops satisfy f(x) ~ x for tiny x (abs, negate, expm1, log1p,
// asin, atan, sinh, tanh, asinh, ...), so those corpus values pass straight
// through to the output unchanged. Measured: 86 results across 14 of 50 ops move
// from tested to skipped (tested 50,005,279 -> 50,005,193, a 0.00017% reduction;
// max 8 per op). Failures are 0 before AND after — these results were PASSING,
// they were simply being checked in a regime where the check is not well-posed.
// The same corpus entries are already skipped on the FF side for the same reason.
//
// Derivation mirrors FF's exactly, retargeted from 24-bit to 53-bit mantissa and
// from the FP32 to the FP64 minimum normal exponent:
//        FF   1/2 ulp = 2^(e-24), subnormal below 2^-126  ->  gate 2^-100
//        DD   1/2 ulp = 2^(e-53), subnormal below 2^-1022 ->  gate 2^-967
// Same 4x (two-binade) margin above the ill-posed boundary. This does NOT mask a
// genuine overlap: any normal-range hi with |lo| > 1/2 ulp(hi) is still checked.
static constexpr double kUnderflowTail = 0x1p-967;  // ~8.0e-292; 4x above 2^-969
inline bool result_checkable(const dd::DoubleDouble& d) {
  if (std::isnan(d.hi)) return false;
  if (std::isinf(d.hi)) return false;
  // subnormal hi: nonzero, finite, magnitude below the smallest normal double.
  if (d.hi != 0.0 && std::isfinite(d.hi) &&
      std::fabs(d.hi) < std::numeric_limits<double>::min())
    return false;
  // near-underflow tail: 1/2 ulp(hi) is subnormal, so fl(hi+lo)==hi is ill-posed.
  if (d.hi != 0.0 && std::fabs(d.hi) < kUnderflowTail)
    return false;
  return true;
}

// ----------------------------------------------------------------------------
// Per-op counters and failure-sample printers (bit patterns, probe_op.cpp style).
// ----------------------------------------------------------------------------
struct InvCount { long tested = 0; long skipped = 0; long failures = 0; };

struct InvSummary {
  std::string name;
  long tested = 0, skipped = 0, failures = 0;
};

static uint64_t dbits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(double));
  return b;
}

static void print_fail_unary(const char* op, double x, const dd::DoubleDouble& d) {
  std::printf("    FAIL %-10s x=%.17g (0x%016llx)  hi=%.17g (0x%016llx)  "
              "lo=%.17g (0x%016llx)  fl(hi+lo)-hi=%.3g\n",
              op, x, (unsigned long long)dbits(x),
              d.hi, (unsigned long long)dbits(d.hi),
              d.lo, (unsigned long long)dbits(d.lo),
              (d.hi + d.lo) - d.hi);
}

static void print_fail_binary(const char* op, double a, double b,
                              const dd::DoubleDouble& d) {
  std::printf("    FAIL %-10s a=%.17g (0x%016llx)  b=%.17g (0x%016llx)  "
              "hi=%.17g (0x%016llx)  lo=%.17g (0x%016llx)\n",
              op, a, (unsigned long long)dbits(a), b, (unsigned long long)dbits(b),
              d.hi, (unsigned long long)dbits(d.hi),
              d.lo, (unsigned long long)dbits(d.lo));
}

// ----------------------------------------------------------------------------
// Op registries. An op carries an input-domain predicate (gates BOTH the random
// generator's rare escapes and every corpus value, so out-of-domain inputs are
// skipped before the call — which also suppresses dd_math.hpp's domain-guard
// prints), a random generator, and the op itself. All dd ops are
// KOKKOS_INLINE_FUNCTION and thus host-callable directly in these loops.
// ----------------------------------------------------------------------------
using DDUnary  = std::function<dd::DoubleDouble(double)>;
using DDBinary = std::function<dd::DoubleDouble(double, double)>;
using Dom1     = std::function<bool(double)>;
using Dom2     = std::function<bool(double, double)>;

struct UnaryOp {
  const char* name;
  Dom1        in_domain;
  InputDist   gen;
  DDUnary     apply;
};

struct BinaryOp {
  const char* name;
  Dom2        in_domain;
  InputDist   gen_a;
  InputDist   gen_b;
  DDBinary    apply;
};

// Common domain predicates.
static const Dom1 dom_any   = [](double x) { return std::isfinite(x); };
static const Dom1 dom_nonneg = [](double x) { return std::isfinite(x) && x >= 0.0; };
static const Dom1 dom_pos    = [](double x) { return std::isfinite(x) && x > 0.0; };
static const Dom1 dom_abs_le1 = [](double x) { return std::isfinite(x) && std::fabs(x) <= 1.0; };

// ----------------------------------------------------------------------------
// One input check (shared by random and corpus passes).
// ----------------------------------------------------------------------------
static void check_unary(const UnaryOp& op, double x, InvCount& c, int& samples_left) {
  if (!op.in_domain(x)) { ++c.skipped; return; }
  dd::DoubleDouble d = op.apply(x);
  if (!result_checkable(d)) { ++c.skipped; return; }
  ++c.tested;
  if (!invariant_holds(d)) {
    ++c.failures;
    if (samples_left > 0) { print_fail_unary(op.name, x, d); --samples_left; }
  }
}

static void check_binary(const BinaryOp& op, double a, double b,
                         InvCount& c, int& samples_left) {
  if (!op.in_domain(a, b)) { ++c.skipped; return; }
  dd::DoubleDouble d = op.apply(a, b);
  if (!result_checkable(d)) { ++c.skipped; return; }
  ++c.tested;
  if (!invariant_holds(d)) {
    ++c.failures;
    if (samples_left > 0) { print_fail_binary(op.name, a, b, d); --samples_left; }
  }
}

// Corpus flags per spec: zero on, inf off, nan off (subnormals default on).
static corpus::CorpusFlags corpus_flags() {
  corpus::CorpusFlags f;
  f.include_zero = true;
  f.include_inf  = false;
  f.include_nan  = false;
  // f.include_subnormals stays true (default) — subnormal INPUTS are valid; a
  // subnormal RESULT is filtered by result_checkable, not here.
  return f;
}

static constexpr int kRandomN = 1'000'000;  // 10^6 random inputs per op (spec)

static InvSummary run_unary(const UnaryOp& op, uint64_t seed) {
  InvCount c;
  int samples_left = 3;

  // Pass (a): 10^6 op-appropriate random inputs.
  {
    std::mt19937_64 gen(seed);
    for (int i = 0; i < kRandomN; ++i) check_unary(op, op.gen(gen), c, samples_left);
  }
  // Pass (b): full corner-case corpus (already folds in the PORT_NOTES §3/§4
  // unary regression families — exp_overflow, nint_half_integer, atanh_small,
  // sinh_cosh_small, trig_near_pi — see corpus.hpp unary()).
  {
    std::vector<double> xs = corpus::unary<double>(corpus_flags());
    for (double x : xs) check_unary(op, x, c, samples_left);
  }

  std::printf("  %-14s tested=%-9ld skipped=%-9ld failures=%ld\n",
              op.name, c.tested, c.skipped, c.failures);
  return InvSummary{op.name, c.tested, c.skipped, c.failures};
}

static InvSummary run_binary(const BinaryOp& op, uint64_t seed) {
  InvCount c;
  int samples_left = 3;

  // Pass (a): 10^6 random (a, b) pairs from one reproducible engine.
  {
    std::mt19937_64 gen(seed);
    for (int i = 0; i < kRandomN; ++i) {
      double a = op.gen_a(gen), b = op.gen_b(gen);
      check_binary(op, a, b, c, samples_left);
    }
  }
  // Pass (b): full corner-case corpus (folds in remainder_regression).
  {
    std::vector<std::pair<double, double>> ps = corpus::binary<double>(corpus_flags());
    for (auto& p : ps) check_binary(op, p.first, p.second, c, samples_left);
  }

  std::printf("  %-14s tested=%-9ld skipped=%-9ld failures=%ld\n",
              op.name, c.tested, c.skipped, c.failures);
  return InvSummary{op.name, c.tested, c.skipped, c.failures};
}

// ----------------------------------------------------------------------------
// Device tripwire (Test B). Same invariant, computed on device for 5 ops.
// A custom runner is required: test_utils_host.hpp's run_unary_op/run_binary_op return
// digits-of-accuracy AccStats (and are scored against __float128), not the raw DD outputs
// this test needs — so we mirror their host->device->host View plumbing but ship
// hi/lo back and check non-overlap on host.
// ----------------------------------------------------------------------------
template <typename DeviceOp>
static InvSummary device_unary(const char* name, int n, uint64_t seed,
                               const InputDist& gen, const Dom1& in_domain,
                               DeviceOp op) {
  using exec_space = Kokkos::DefaultExecutionSpace;

  std::vector<double> hx(n);
  { std::mt19937_64 g(seed); for (int i = 0; i < n; ++i) hx[i] = gen(g); }

  Kokkos::View<double*, exec_space> dx("dx", n), dhi("dhi", n), dlo("dlo", n);
  auto hmx = Kokkos::create_mirror_view(dx);
  for (int i = 0; i < n; ++i) hmx(i) = hx[i];
  Kokkos::deep_copy(dx, hmx);

  Kokkos::parallel_for("dd_inv_dev_unary", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      dd::DoubleDouble d = op(dd::DoubleDouble(dx(i)));
      dhi(i) = d.hi; dlo(i) = d.lo;
    });
  Kokkos::fence();

  auto hhi = Kokkos::create_mirror_view(dhi);
  auto hlo = Kokkos::create_mirror_view(dlo);
  Kokkos::deep_copy(hhi, dhi);
  Kokkos::deep_copy(hlo, dlo);

  InvCount c; int samples_left = 3;
  for (int i = 0; i < n; ++i) {
    if (!in_domain(hx[i])) { ++c.skipped; continue; }
    dd::DoubleDouble d(hhi(i), hlo(i));
    if (!result_checkable(d)) { ++c.skipped; continue; }
    ++c.tested;
    if (!invariant_holds(d)) {
      ++c.failures;
      if (samples_left > 0) { print_fail_unary(name, hx[i], d); --samples_left; }
    }
  }
  std::printf("  [device] %-12s tested=%-8ld skipped=%-8ld failures=%ld\n",
              name, c.tested, c.skipped, c.failures);
  return InvSummary{std::string("device:") + name, c.tested, c.skipped, c.failures};
}

template <typename DeviceOp>
static InvSummary device_binary(const char* name, int n, uint64_t seed,
                                const InputDist& gen_a, const InputDist& gen_b,
                                const Dom2& in_domain, DeviceOp op) {
  using exec_space = Kokkos::DefaultExecutionSpace;

  std::vector<double> ha(n), hb(n);
  { std::mt19937_64 g(seed);
    for (int i = 0; i < n; ++i) { ha[i] = gen_a(g); hb[i] = gen_b(g); } }

  Kokkos::View<double*, exec_space> da("da", n), db("db", n), dhi("dhi", n), dlo("dlo", n);
  auto hma = Kokkos::create_mirror_view(da);
  auto hmb = Kokkos::create_mirror_view(db);
  for (int i = 0; i < n; ++i) { hma(i) = ha[i]; hmb(i) = hb[i]; }
  Kokkos::deep_copy(da, hma);
  Kokkos::deep_copy(db, hmb);

  Kokkos::parallel_for("dd_inv_dev_binary", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      dd::DoubleDouble d = op(dd::DoubleDouble(da(i)), dd::DoubleDouble(db(i)));
      dhi(i) = d.hi; dlo(i) = d.lo;
    });
  Kokkos::fence();

  auto hhi = Kokkos::create_mirror_view(dhi);
  auto hlo = Kokkos::create_mirror_view(dlo);
  Kokkos::deep_copy(hhi, dhi);
  Kokkos::deep_copy(hlo, dlo);

  InvCount c; int samples_left = 3;
  for (int i = 0; i < n; ++i) {
    if (!in_domain(ha[i], hb[i])) { ++c.skipped; continue; }
    dd::DoubleDouble d(hhi(i), hlo(i));
    if (!result_checkable(d)) { ++c.skipped; continue; }
    ++c.tested;
    if (!invariant_holds(d)) {
      ++c.failures;
      if (samples_left > 0) { print_fail_binary(name, ha[i], hb[i], d); --samples_left; }
    }
  }
  std::printf("  [device] %-12s tested=%-8ld skipped=%-8ld failures=%ld\n",
              name, c.tested, c.skipped, c.failures);
  return InvSummary{std::string("device:") + name, c.tested, c.skipped, c.failures};
}

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("=== dd_invariant_test (T1.2): non-overlap invariant fl(hi+lo)==hi "
                "for every DD op ===\n");
    std::printf("Oracle-independent (raw FP64 check). Execution space: %s\n\n",
                Kokkos::DefaultExecutionSpace::name());

    std::vector<InvSummary> summary;

    // ---- Unary op registry -------------------------------------------------
    // Ranges are chosen op-appropriate: wide enough for broad coverage, bounded
    // to stay inside each op's mathematical domain and normal output range so
    // inputs are exercised rather than skipped, and so dd_math.hpp's domain
    // guards (which print) are never tripped.
    const std::vector<UnaryOp> unary_ops = {
      {"abs",       dom_any,     uniform(-1e8, 1e8),   [](double x){ return dd::abs(dd::DoubleDouble(x)); }},
      {"negate",    dom_any,     uniform(-1e8, 1e8),   [](double x){ return dd::negate(dd::DoubleDouble(x)); }},
      {"sqrt",      dom_nonneg,  uniform(0.0, 1e8),    [](double x){ return dd::sqrt(dd::DoubleDouble(x)); }},
      // round-family: |x| < 2^105 (nint guard). Keep to |x|<1e15 so results are
      // exact integers well inside the normal range.
      {"round_to_nearest_int", [](double x){ return std::isfinite(x) && std::fabs(x) < 1e15; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::round_to_nearest_int(dd::DoubleDouble(x)); }},
      {"ceil",      [](double x){ return std::isfinite(x) && std::fabs(x) < 1e15; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::ceil(dd::DoubleDouble(x)); }},
      {"floor",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e15; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::floor(dd::DoubleDouble(x)); }},
      {"round",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e15; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::round(dd::DoubleDouble(x)); }},
      {"trunc",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e15; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::trunc(dd::DoubleDouble(x)); }},
      // exp-family: keep the argument to exp below its 300 guard.
      {"exp",       [](double x){ return std::isfinite(x) && x < 300.0; },
                    uniform(-300.0, 299.0), [](double x){ return dd::exp(dd::DoubleDouble(x)); }},
      {"exp2",      [](double x){ return std::isfinite(x) && std::fabs(x) < 400.0; },
                    uniform(-100.0, 100.0), [](double x){ return dd::exp2(dd::DoubleDouble(x)); }},
      {"exp10",     [](double x){ return std::isfinite(x) && std::fabs(x) < 120.0; },
                    uniform(-80.0, 80.0),   [](double x){ return dd::exp10(dd::DoubleDouble(x)); }},
      {"expm1",     [](double x){ return std::isfinite(x) && x < 300.0; },
                    uniform(-5.0, 5.0),     [](double x){ return dd::expm1(dd::DoubleDouble(x)); }},
      // log-family domain: 1e-100 <= x <= 1e100 AND x normal. dd_math's log refines
      // via an INTERNAL exp(b) with b ~ log(x); that Taylor series only converges in
      // its iteration budget while |b| stays moderate. Both tails break it (the
      // result stays invariant-clean, but the input is outside the op's useful range
      // and the internal guard prints a diagnostic): for x above e^300 (~1.9e130) the
      // exp trips its own 300 guard, and for x below ~e^-230 (a tiny-but-normal or
      // subnormal x, log(x) ~ -230..-744) the exp hits its iteration limit. Bounding
      // |x| to [1e-100, 1e100] keeps |log x| < ~230, inside the converging range.
      // Random draws never approach either tail; these gate the corpus extremes
      // (maxT, the smallest normals/subnormals, top powers of two).
      {"log",       [](double x){ return std::isnormal(x) && x >= 1e-100 && x <= 1e100; },
                    uniform(1e-6, 1e6),   [](double x){ return dd::log(dd::DoubleDouble(x)); }},
      {"log2",      [](double x){ return std::isnormal(x) && x >= 1e-100 && x <= 1e100; },
                    uniform(1e-6, 1e6),   [](double x){ return dd::log2(dd::DoubleDouble(x)); }},
      {"log10",     [](double x){ return std::isnormal(x) && x >= 1e-100 && x <= 1e100; },
                    uniform(1e-6, 1e6),   [](double x){ return dd::log10(dd::DoubleDouble(x)); }},
      // log1p(x)=log(1+x): 1+x is ~1 for tiny x, so no subnormal-log hazard; just
      // bound the upper magnitude like the log family.
      {"log1p",     [](double x){ return std::isfinite(x) && x > -1.0 && x < 1e100; },
                    uniform(-0.9, 1e6),   [](double x){ return dd::log1p(dd::DoubleDouble(x)); }},
      // trig: |x| < 1e60 (sincos guard). Bounded so many periods are covered.
      {"sin",       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e6; },
                    uniform(-1000.0, 1000.0), [](double x){ return dd::sin(dd::DoubleDouble(x)); }},
      {"cos",       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e6; },
                    uniform(-1000.0, 1000.0), [](double x){ return dd::cos(dd::DoubleDouble(x)); }},
      {"tan",       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e6; },
                    uniform(-1000.0, 1000.0), [](double x){ return dd::tan(dd::DoubleDouble(x)); }},
      {"asin",      dom_abs_le1, uniform(-1.0, 1.0),   [](double x){ return dd::asin(dd::DoubleDouble(x)); }},
      {"acos",      dom_abs_le1, uniform(-1.0, 1.0),   [](double x){ return dd::acos(dd::DoubleDouble(x)); }},
      // atan(a)=angle(1,a); angle forms r=sqrt(x^2+y^2). For |a| >~ 1e154, a*a
      // overflows to inf, r=inf, the normalized (nx,ny) collapse to 0, and angle's
      // sincos-based Newton diverges (sincos then prints its iteration-limit
      // diagnostic). atan has saturated to +/-pi/2 to full DD precision long before
      // then, so cap |a| < 1e150 (a*a stays finite) — excludes maxT / top powers of
      // two from the corpus.
      {"atan",      [](double x){ return std::isfinite(x) && std::fabs(x) < 1e150; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::atan(dd::DoubleDouble(x)); }},
      {"sinh",      [](double x){ return std::isfinite(x) && std::fabs(x) < 300.0; },
                    uniform(-100.0, 100.0), [](double x){ return dd::sinh(dd::DoubleDouble(x)); }},
      {"cosh",      [](double x){ return std::isfinite(x) && std::fabs(x) < 300.0; },
                    uniform(-100.0, 100.0), [](double x){ return dd::cosh(dd::DoubleDouble(x)); }},
      // tanh(x)=expm1(2x)/(...) drives an internal exp; cap |x|<300 (tanh has
      // already saturated to +/-1 to full DD precision well before then).
      {"tanh",      [](double x){ return std::isfinite(x) && std::fabs(x) < 300.0; },
                    uniform(-50.0, 50.0), [](double x){ return dd::tanh(dd::DoubleDouble(x)); }},
      // asinh/acosh reduce to log(x + sqrt(x^2 +/- 1)); cap |x|<1e100 to keep the
      // internal log's exp in range and to avoid x*x overflowing to inf.
      {"asinh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e100; },
                    uniform(-1e6, 1e6),   [](double x){ return dd::asinh(dd::DoubleDouble(x)); }},
      {"acosh",     [](double x){ return std::isfinite(x) && x >= 1.0 && x < 1e100; },
                    uniform(1.0, 1e6),    [](double x){ return dd::acosh(dd::DoubleDouble(x)); }},
      {"atanh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1.0; },
                    uniform(-0.999, 0.999), [](double x){ return dd::atanh(dd::DoubleDouble(x)); }},
      {"erf",       dom_any,     uniform(-10.0, 10.0), [](double x){ return dd::erf(dd::DoubleDouble(x)); }},
      {"erfc",      dom_any,     uniform(-10.0, 10.0), [](double x){ return dd::erfc(dd::DoubleDouble(x)); }},
      // tgamma: bounded to (0, 50) to stay below overflow (~171) and away from the
      // non-positive-integer poles (random never lands on one exactly, but keeping
      // to positive args keeps results clean and in-range).
      {"tgamma",    [](double x){ return std::isfinite(x) && x > 0.0 && x < 50.0; },
                    uniform(0.1, 50.0),   [](double x){ return dd::tgamma(dd::DoubleDouble(x)); }},
    };

    // Two-output ops, tested as one "op" per output component.
    // sincos(a, cos, sin); sinhcosh(a, cosh, sinh)  (see dd_math.hpp).
    const std::vector<UnaryOp> two_out_ops = {
      {"sincos.cos",   [](double x){ return std::isfinite(x) && std::fabs(x) < 1e6; },
                       uniform(-1000.0, 1000.0),
                       [](double x){ dd::DoubleDouble c, s; dd::sincos(dd::DoubleDouble(x), c, s); return c; }},
      {"sincos.sin",   [](double x){ return std::isfinite(x) && std::fabs(x) < 1e6; },
                       uniform(-1000.0, 1000.0),
                       [](double x){ dd::DoubleDouble c, s; dd::sincos(dd::DoubleDouble(x), c, s); return s; }},
      {"sinhcosh.cosh",[](double x){ return std::isfinite(x) && std::fabs(x) < 300.0; },
                       uniform(-100.0, 100.0),
                       [](double x){ dd::DoubleDouble c, s; dd::sinhcosh(dd::DoubleDouble(x), c, s); return c; }},
      {"sinhcosh.sinh",[](double x){ return std::isfinite(x) && std::fabs(x) < 300.0; },
                       uniform(-100.0, 100.0),
                       [](double x){ dd::DoubleDouble c, s; dd::sinhcosh(dd::DoubleDouble(x), c, s); return s; }},
    };

    // ---- Binary op registry ------------------------------------------------
    const Dom2 dom2_any    = [](double a, double b){ return std::isfinite(a) && std::isfinite(b); };
    // fmod/remainder form q = a/b then nint/trunc(q); |q| >= 2^105 trips the nint
    // guard (dd_math prints "DDNINT: argument too large"). Require |a| < |b|*1e30
    // (1e30 < 2^105 ~ 4e31) so the quotient stays in nint's range — extreme-ratio
    // pairs (corpus maxT/1, huge_tiny) are skipped rather than fed.
    const Dom2 dom2_modbnz = [](double a, double b){
      return std::isfinite(a) && std::isfinite(b) && b != 0.0 &&
             std::fabs(a) < std::fabs(b) * 1e30;
    };
    const Dom2 dom2_bnz    = [](double a, double b){ return std::isfinite(a) && std::isfinite(b) && b != 0.0; };
    // pow(a,b)=exp(b*log(a)); the FINAL exp trips its 300 guard whenever
    // |b*ln(a)| >= 300 (e.g. corpus near-cancellation with base ~1e6 -> b ~ 1e6,
    // ln(a) ~ 13.8). Gate on the predicted exponent so only pairs whose result is
    // representable are fed. a>0 is the mathematical domain; restrict the base to
    // the same normal [1e-100, 1e100] window as the log family, because pow's
    // INTERNAL log(a) hits the very same exp iteration limit for tiny/subnormal a
    // (independently of the final-exp guard above).
    const Dom2 dom2_powpos = [](double a, double b){
      return std::isnormal(a) && std::isfinite(b) && a >= 1e-100 && a <= 1e100 &&
             std::fabs(b * std::log(a)) < 300.0;
    };

    const std::vector<BinaryOp> binary_ops = {
      {"add",       dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return dd::add(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"subtract",  dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return dd::subtract(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"multiply",  dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return dd::multiply(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"divide",    dom2_bnz, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return dd::divide(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"pow",       dom2_powpos, uniform(0.1, 20.0), uniform(-6.0, 6.0),
                    [](double a, double b){ return dd::pow(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      // atan2(y,x)=angle(x,y), which forms r=sqrt(x^2+y^2) and drives an internal
      // sincos Newton. Two magnitude hazards bound the domain (result stays
      // invariant-clean either way; the internal guard just prints): x^2+y^2 must
      // stay finite (cap the larger magnitude below 1e150) AND must not UNDERFLOW to
      // zero. When BOTH operands are subnormal or the smallest normal (~2.2e-308),
      // each square rounds to 0, r=0, and sincos gets a degenerate angle -> its
      // iteration limit. Gate on the larger magnitude m: m=0 is atan2(0,0) (defined,
      // no sincos hazard); otherwise require 1e-150 <= m <= 1e150 so r is a normal
      // number. Random draws (|.|<1e3) never approach either tail; this gates the
      // corpus's smallest-normal / subnormal pairs.
      {"atan2",     [](double a, double b){
                      if (!(std::isfinite(a) && std::isfinite(b))) return false;
                      double m = std::fmax(std::fabs(a), std::fabs(b));
                      if (m == 0.0) return true;
                      return m >= 1e-150 && m <= 1e150; },
                    uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](double a, double b){ return dd::atan2(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"hypot",     dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return dd::hypot(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"fmod",      dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](double a, double b){ return dd::fmod(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"remainder", dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](double a, double b){ return dd::remainder(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"copysign",  dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return dd::copysign(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"fmax",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return dd::fmax(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"fmin",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return dd::fmin(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
      {"fdim",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return dd::fdim(dd::DoubleDouble(a), dd::DoubleDouble(b)); }},
    };

    // -- Test A: every op ----------------------------------------------------
    std::printf("[Test A] per-op invariant (10^6 random + full corpus)\n");
    uint64_t seed = 12345ULL;
    for (const auto& op : unary_ops)   summary.push_back(run_unary(op,  seed++));
    for (const auto& op : two_out_ops) summary.push_back(run_unary(op,  seed++));
    for (const auto& op : binary_ops)  summary.push_back(run_binary(op, seed++));

    // Ternary fma(a, b, c): invariant on the fused result. c drawn from the same
    // range; a third stream off the same engine keeps the run reproducible.
    {
      const Dom2 fdom = [](double a, double b){ return std::isfinite(a) && std::isfinite(b); };
      InvCount c; int samples_left = 3;
      InputDist g = uniform(-1e4, 1e4);
      {
        std::mt19937_64 gen(seed++);
        for (int i = 0; i < kRandomN; ++i) {
          double a = g(gen), b = g(gen), cc = g(gen);
          if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(cc))) { ++c.skipped; continue; }
          dd::DoubleDouble d = dd::fma(dd::DoubleDouble(a), dd::DoubleDouble(b), dd::DoubleDouble(cc));
          if (!result_checkable(d)) { ++c.skipped; continue; }
          ++c.tested;
          if (!invariant_holds(d)) {
            ++c.failures;
            if (samples_left > 0) {
              std::printf("    FAIL fma        a=%.17g b=%.17g c=%.17g  hi=%.17g lo=%.17g\n",
                          a, b, cc, d.hi, d.lo);
              --samples_left;
            }
          }
        }
      }
      std::printf("  %-14s tested=%-9ld skipped=%-9ld failures=%ld\n",
                  "fma", c.tested, c.skipped, c.failures);
      summary.push_back(InvSummary{"fma", c.tested, c.skipped, c.failures});
      (void)fdom;
    }

    // pow_int(a, n): integer exponent. Domain: not 0^negative (nint/guard print).
    {
      InvCount c; int samples_left = 3;
      InputDist gx = uniform(-5.0, 5.0);
      std::mt19937_64 gen(seed++);
      std::uniform_int_distribution<int> dn(-20, 20);
      for (int i = 0; i < kRandomN; ++i) {
        double x = gx(gen);
        int    n = dn(gen);
        if (!std::isfinite(x) || (x == 0.0 && n < 0)) { ++c.skipped; continue; }
        dd::DoubleDouble d = dd::pow_int(dd::DoubleDouble(x), n);
        if (!result_checkable(d)) { ++c.skipped; continue; }
        ++c.tested;
        if (!invariant_holds(d)) {
          ++c.failures;
          if (samples_left > 0) {
            std::printf("    FAIL pow_int    x=%.17g n=%d  hi=%.17g lo=%.17g\n", x, n, d.hi, d.lo);
            --samples_left;
          }
        }
      }
      std::printf("  %-14s tested=%-9ld skipped=%-9ld failures=%ld\n",
                  "pow_int", c.tested, c.skipped, c.failures);
      summary.push_back(InvSummary{"pow_int", c.tested, c.skipped, c.failures});
    }

    // -- Test B: device tripwire (5 representative ops) ----------------------
    std::printf("\n[Test B] device tripwire (5 ops, 10^5 random on %s)\n",
                Kokkos::DefaultExecutionSpace::name());
    const int nd = 100'000;
    summary.push_back(device_binary("add", nd, 55501ULL,
        uniform(-1e8, 1e8), uniform(-1e8, 1e8), dom2_any,
        KOKKOS_LAMBDA(dd::DoubleDouble a, dd::DoubleDouble b){ return dd::add(a, b); }));
    summary.push_back(device_binary("multiply", nd, 55502ULL,
        uniform(-1e6, 1e6), uniform(-1e6, 1e6), dom2_any,
        KOKKOS_LAMBDA(dd::DoubleDouble a, dd::DoubleDouble b){ return dd::multiply(a, b); }));
    summary.push_back(device_unary("sqrt", nd, 55503ULL,
        uniform(0.0, 1e8), dom_nonneg,
        KOKKOS_LAMBDA(dd::DoubleDouble x){ return dd::sqrt(x); }));
    summary.push_back(device_unary("exp", nd, 55504ULL,
        uniform(-300.0, 299.0), [](double x){ return std::isfinite(x) && x < 300.0; },
        KOKKOS_LAMBDA(dd::DoubleDouble x){ return dd::exp(x); }));
    summary.push_back(device_unary("sin", nd, 55505ULL,
        uniform(-1000.0, 1000.0), [](double x){ return std::isfinite(x) && std::fabs(x) < 1e6; },
        KOKKOS_LAMBDA(dd::DoubleDouble x){ return dd::sin(x); }));

    // -- Test C: explicit PORT_NOTES §4 regressions --------------------------
    // These document that DD has NO equivalent of the FF bugs recorded in
    // PORT_NOTES §4 (branch fffunKokkos): DD's exp splitter does not overflow in
    // the 79.5..88.72 band, DD's nint uses the 2^105 magic-constant trick (safe on
    // FP64's 53-bit mantissa, unlike FP32's 24-bit), and DD's remainder inherits
    // the correct sign. Each asserts the invariant AND the FF-bug-specific outcome.
    std::printf("\n[Test C] PORT_NOTES §4 named regressions (DD has no equivalent bug)\n");
    int c_pass = 0, c_total = 0;

    auto case_exp = [&](double x) {
      ++c_total;
      dd::DoubleDouble d = dd::exp(dd::DoubleDouble(x));
      bool ok = !std::isnan(d.hi) && !std::isinf(d.hi) && invariant_holds(d);
      std::printf("    exp(%.5g): hi=%.17g lo=%.5g  %s\n", x, d.hi, d.lo, ok ? "PASS" : "FAIL");
      if (ok) ++c_pass; else print_fail_unary("exp", x, d);
    };
    for (double x : corpus::exp_overflow<double>()) case_exp(x);   // 79.5,80,85,88.7,88.72

    auto case_nint = [&](double x) {
      ++c_total;
      dd::DoubleDouble d = dd::round_to_nearest_int(dd::DoubleDouble(x));
      bool ok = result_checkable(d) && invariant_holds(d);
      std::printf("    nint(%.10g): hi=%.17g lo=%.5g  %s\n", x, d.hi, d.lo, ok ? "PASS" : "FAIL");
      if (ok) ++c_pass; else print_fail_unary("nint", x, d);
    };
    // The literal FF offender plus the k+/-0.5 boundary family.
    for (double x : corpus::nint_half_integer<double>()) case_nint(x);

    // remainder(68.379, 3.5066): invariant AND the CORRECT (precision-appropriate)
    // sign — DD must match the IEEE reference remainder() for FP64.
    //
    // PORT_NOTES §4b records this input as an FP32 float-float bug: ffnint's
    // 2^47 magic-constant trick rounded a/b UP to 20 when it should have been 19,
    // flipping the sign of the result. Crucially, "correct" there was POSITIVE:
    // in FP32 the literals round so that a/b ~ 19.4999993 (< 19.5), giving
    // nint=19 and remainder = +1.7536.
    //
    // At FP64/DD precision the SAME literals round to the other side of 19.5:
    // a/b = 19.500085552957... (> 19.5), so the IEEE-correct nint is 20 and the
    // correct remainder is NEGATIVE, -1.7529999999999983. Verified against both
    // libm remainder() and quadmath remainderq() (both == -1.75299999999999833).
    // DD reproduces that value exactly. So the DD-correct outcome is the opposite
    // sign from the FP32 note — not because DD is wrong, but because the true
    // quotient sits on the far side of the half-integer boundary at this precision.
    // We therefore gate DD's sign against the FP64 IEEE oracle (std::remainder,
    // always available — no wide type needed), which documents "no equivalent DD
    // bug": DD lands on the precision-appropriate sign the FF path missed.
    {
      ++c_total;
      double a = 68.379, b = 3.5066;
      dd::DoubleDouble d = dd::remainder(dd::DoubleDouble(a), dd::DoubleDouble(b));
      double ref = std::remainder(a, b);                 // IEEE FP64 oracle (negative here)
      bool inv = invariant_holds(d);
      bool sign_ok = (d.hi < 0.0) == (ref < 0.0);        // DD sign matches IEEE
      bool ok = inv && sign_ok;
      std::printf("    remainder(%.5g, %.5g): hi=%.17g lo=%.5g  sign=%s "
                  "IEEEref=%.17g inv=%s  %s\n",
                  a, b, d.hi, d.lo, d.hi < 0.0 ? "-" : "+", ref, inv ? "ok" : "BAD",
                  ok ? "PASS" : "FAIL");
      if (ok) ++c_pass; else print_fail_binary("remainder", a, b, d);
    }
    std::printf("  Test C: %d/%d passed\n", c_pass, c_total);

    // -- Summary table -------------------------------------------------------
    std::printf("\n=== Summary (op : tested / skipped / failures : status) ===\n");
    long total_tested = 0, total_skipped = 0, total_failures = 0;
    for (const auto& s : summary) {
      total_tested += s.tested; total_skipped += s.skipped; total_failures += s.failures;
      std::printf("  %-22s %12ld %12ld %10ld   %s\n",
                  s.name.c_str(), s.tested, s.skipped, s.failures,
                  s.failures == 0 ? "OK" : "FAIL");
    }
    std::printf("  %-22s %12ld %12ld %10ld\n",
                "TOTAL", total_tested, total_skipped, total_failures);

    KOKKOS_EP_ASSERT(total_failures == 0,
                     "one or more DD ops produced an overlapping (hi, lo) result");
    KOKKOS_EP_ASSERT(c_pass == c_total, "a PORT_NOTES §4 named regression case failed");

    rc = ep_exit_code();
    std::printf("\n=== dd_invariant_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();
  return rc;
}
