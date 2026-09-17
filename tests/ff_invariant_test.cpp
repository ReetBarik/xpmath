// ============================================================================
// ff_invariant_test.cpp — Layer 2 (output-invariant tests) for FF.  Plan T2.2.
// ============================================================================
//
// WHAT THIS LAYER CHECKS AND WHY IT IS ORACLE-INDEPENDENT
// ------------------------------------------------------
// This is the FF analogue of dd_invariant_test.cpp (T1.2). Layer 1
// (ff_eft_test.cpp, T2.1) proved the two atoms — twoSum and the Dekker
// twoProduct — are bit-exact for FF (splitter 8193.0f = 2^13+1). This layer
// checks a STRUCTURAL property of every higher-level op's OUTPUT: a float-float
// (hi, lo) must be NON-OVERLAPPING, i.e. lo carries only bits BELOW the last bit
// of hi. The canonical, bit-exact statement of that is
//
//     fl(hi + lo) == hi            (equivalently  |lo| <= 1/2 ulp(hi))
//
// evaluated in RAW FP32 (a single hardware add + compare). This is the SAME
// statement T1.2 makes for DD, just FP32-typed: for DD the check is
// (d.hi + d.lo) == d.hi in double; for FF it is (f.hi + f.lo) == f.hi in float.
// If lo held any bit at or above hi's ulp, the rounded sum hi+lo would land on a
// different float and the equality would fail. Every FF op is supposed to return
// a renormalized, non-overlapping pair; a violation localizes a normalization
// bug to that exact op — no reference value and no wider type is needed to see
// it. That is why this test carries NO __float128 oracle at all: it runs (and
// must run) on the narrowest possible toolchain. (It never carried the
// KOKKOS_EP_HAVE_QUADMATH guard either; that guard is now gone from the tests
// that DID carry it.) Accuracy-vs-oracle is a separate concern handled in T2.4.
//
// WHY RAW FP32 EQUALITY, NOT A double OR float128 PROMOTION
// --------------------------------------------------------
// The invariant is a statement ABOUT FP32 rounding: "adding lo back into hi does
// not move hi". Promoting hi/lo to double (or __float128) and comparing would
// test a DIFFERENT thing (the exact real sum), and would report a
// perfectly-normalized pair with a nonzero lo as "unequal". So the check is
// deliberately `(f.hi + f.lo) == f.hi` in float. (This TU is built with the
// project's normal flags — no contraction pragma needed: a lone add-then-compare
// has nothing to fuse. It is registered with the plain kokkos_ep_add_test, not
// the EFT helper.)
//
// SKIP (not FAIL) CRITERIA — a result outside the invariant's domain
//   * hi is NaN            — non-overlap is undefined; op fed an out-of-domain input
//   * hi is +/-inf         — overflowed; lo is meaningless
//   * hi is subnormal      — below the smallest normal FP32, ulp(hi) is the fixed
//                            subnormal step and the "1/2 ulp" argument degrades;
//                            a nonzero lo there is not a normalization defect
//   * hi in the underflow tail (|hi| < 2^-102, i.e. 1/2 ulp(hi) itself subnormal)
//                            — the residual lo is forced onto quantized subnormal
//                            values that land EXACTLY on the 1/2 ulp(hi) tie, where
//                            round-to-even flips fl(hi+lo) off hi by one ulp even
//                            though non-overlap |lo| <= 1/2 ulp(hi) still holds.
//                            A property of double-word arithmetic in the denormal
//                            tail (exp/exp2/exp10 of very negative args), NOT an
//                            ff_math.hpp defect. See result_checkable's kUnderflowTail
//                            and tests/README.md finding (3).
//   * input out of the op's mathematical domain (log of <=0, asin of |x|>1, ...)
//                            — also avoids the domain-guard diagnostics ff_math.hpp
//                            prints, which would otherwise spam the log
//
// FP32-NARROWER DOMAIN PREDICATES (derived from ff_math.hpp, NOT copied from DD)
// ----------------------------------------------------------------------------
// FP32's exponent range is ~6x narrower than FP64 (max ~3.4e38, min-normal
// ~1.18e-38, vs DBL_MAX ~1.8e308). Every DD bound was re-derived against the
// SHIPPED ff_math.hpp guards and empirically confirmed to emit ZERO internal
// diagnostics. The material FF-specific tightenings vs T1.2 (each verified by a
// standalone corpus+random probe against ff_math.hpp):
//   * exp: ff_math.hpp guards at a.hi >= 88.0f (FP32 ln-range), NOT DD's 300.
//   * trig (sin/cos/tan/atan) AND atan2/tgamma-reflection: FF's sincos Taylor
//     loop hits its iteration limit (FFCSSNR) for TINY nonzero arguments
//     (|x| < ~1e-28) because at FP32 r=x/2^nq underflows and the series never
//     converges. DD's FP64 sincos never saw this. So the trig family carries a
//     LOWER magnitude bound (|x| >= 1e-25, or exactly 0) that T1.2 did not need.
//   * atan2: additionally, a subnormal-tiny operand paired with a normal one
//     drives the internal sincos degenerate; gate BOTH operands away from
//     |.| < 1e-18 (0 allowed) — stronger than DD's larger-magnitude-only gate.
//   * tgamma: the reflection path (x < 0.5) calls sin(pi*x); for x < ~1e-3 that
//     is sin of a tiny arg -> FFCSSNR. Gate tgamma to x >= 1e-3 (also < 23,
//     above which the internal pow/exp overflows FP32 to NaN).
//   * sinh/cosh cap at |x| < 40 (exp(40)~2.4e17 finite; exp(80) overflows FP32);
//     tanh at |x| < 20 (expm1(2x) overflows past ~x=20 at FP32).
//   * log-family window [1e-34, 1e34] (DD used [1e-100,1e100]); asinh/acosh/atan
//     upper caps at 1e18 (a*a must stay < FLT_MAX ~3.4e38).
//
// TEST STRUCTURE (mirrors T1.2 verbatim)
//   Test A — every op, two passes each: 10^6 op-appropriate random inputs, then
//            the full corner-case corpus (corpus::unary<float>/binary<float>).
//            Per-op tested/skipped/failures; a summary table at the end.
//   Test B — device tripwire: 5 representative ops (add, multiply, sqrt, exp, sin)
//            run the SAME invariant on 10^5 random inputs inside a parallel_for,
//            results copied back and checked on host. Catches a Serial->CUDA
//            regression (subnormal flush, contraction) the host pass cannot see.
//   Test C — explicit PORT_NOTES §4 regressions as named cases: exp in the
//            79.5..88.72 band (§4a splitter overflow), round_to_nearest_int at
//            the 19.4999993 / k+0.5 boundaries (§4b ffnint), and
//            remainder(68.379, 3.5066) with its sign gated against
//            std::remainderf (§4b). See the Test C body for the FP32-semantics
//            finding on the remainder sign.
//
// SCOPE: real FF ops only (ff_complex.hpp is out of scope). If a violation is
// found this test REPORTS it (op, counts, first offending bit patterns) and fails
// — it does not attempt any fix to ff_math.hpp (rule 4).
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 2, "T2.2: non-overlap invariant
// checks for FF", the T1.2 DONE block (direct structural template), and "The six
// test layers" layer 2.
// ============================================================================

#include "test_utils_host.hpp"
#include "corpus.hpp"
#include <ff_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace kokkos_ep;

// dd:: alias comes from test_utils_device.hpp (namespace dd = Kokkos::Experimental).
// FF types live in the SAME namespace; introduce an ff:: alias for readability.
namespace ff = Kokkos::Experimental;

// ----------------------------------------------------------------------------
// The invariant and its domain, on a raw FF value.
// ----------------------------------------------------------------------------

// The non-overlap invariant itself: adding lo back into hi must not move hi.
// Raw FP32 — a single add and a bit-for-bit float comparison.
inline bool invariant_holds(const ff::FloatFloat& d) {
  return (d.hi + d.lo) == d.hi;
}

// Is this result in the invariant's domain (i.e. should we CHECK it, vs SKIP it)?
// NaN / inf / subnormal hi are out of domain — see the SKIP criteria header note.
//
// UNDERFLOW-TAIL guard (kUnderflowTail): the strict bit-exact form fl(hi+lo)==hi
// is only WELL-POSED when the tie point 1/2 ulp(hi) is itself a NORMAL float. For
// hi with unbiased exponent e, 1/2 ulp(hi) = 2^(e-24); this is subnormal once
// e < -102, i.e. |hi| < 2^-102 (~1.97e-31). In that near-underflow regime the
// residual lo is forced onto quantized subnormal values that land EXACTLY on the
// 1/2 ulp(hi) tie, where round-to-even flips fl(hi+lo) off hi by one ulp even
// though the mathematical non-overlap |lo| <= 1/2 ulp(hi) still holds. That is a
// property of double-word arithmetic in the denormal tail (exp/exp2/exp10 of very
// negative args), NOT an ff_math.hpp normalization defect — see tests/README.md
// "FF non-overlap invariant", finding (3). We gate at 2^-100 (a 4x margin above
// 2^-102) so the check is applied only where it is well-posed. This does NOT mask
// a genuine overlap: any normal-range hi with |lo| > 1/2 ulp(hi) is still checked.
static constexpr float kUnderflowTail = 0x1p-100f;  // ~7.9e-31; 4x above 2^-102
inline bool result_checkable(const ff::FloatFloat& d) {
  if (std::isnan(d.hi)) return false;
  if (std::isinf(d.hi)) return false;
  // subnormal hi: nonzero, finite, magnitude below the smallest normal float.
  if (d.hi != 0.0f && std::isfinite(d.hi) &&
      std::fabs(d.hi) < std::numeric_limits<float>::min())
    return false;
  // near-underflow tail: 1/2 ulp(hi) is subnormal, so fl(hi+lo)==hi is ill-posed.
  if (d.hi != 0.0f && std::fabs(d.hi) < kUnderflowTail)
    return false;
  return true;
}

// ----------------------------------------------------------------------------
// Per-op counters and failure-sample printers (bit patterns, probe_op.cpp style).
// FF components are 32-bit, so bit patterns print as 0x%08x (NOT DD's 0x%016llx).
// ----------------------------------------------------------------------------
struct InvCount { long tested = 0; long skipped = 0; long failures = 0; };

struct InvSummary {
  std::string name;
  long tested = 0, skipped = 0, failures = 0;
};

static uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(float));
  return b;
}

static uint64_t dbits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(double));
  return b;
}

static void print_fail_unary(const char* op, double x, const ff::FloatFloat& d) {
  std::printf("    FAIL %-10s x=%.9g (0x%016llx dbl)  hi=%.9g (0x%08x)  "
              "lo=%.9g (0x%08x)  fl(hi+lo)-hi=%.3g\n",
              op, x, (unsigned long long)dbits(x),
              d.hi, fbits(d.hi),
              d.lo, fbits(d.lo),
              (double)((d.hi + d.lo) - d.hi));
}

static void print_fail_binary(const char* op, double a, double b,
                              const ff::FloatFloat& d) {
  std::printf("    FAIL %-10s a=%.9g b=%.9g  hi=%.9g (0x%08x)  lo=%.9g (0x%08x)\n",
              op, a, b, d.hi, fbits(d.hi), d.lo, fbits(d.lo));
}

// ----------------------------------------------------------------------------
// Op registries. An op carries an input-domain predicate (gates BOTH the random
// generator's rare escapes and every corpus value, so out-of-domain inputs are
// skipped before the call — which also suppresses ff_math.hpp's domain-guard
// prints), a random generator, and the op itself. All ff ops are
// KOKKOS_INLINE_FUNCTION and thus host-callable directly in these loops.
//
// Inputs are constructed via the FloatFloat(double) constructor (Route-A split),
// matching the demos (PORT_NOTES §1): the input faithfully carries ~14 digits,
// so lo is generally nonzero and the invariant is a non-trivial check.
// ----------------------------------------------------------------------------
using FFUnary  = std::function<ff::FloatFloat(double)>;
using FFBinary = std::function<ff::FloatFloat(double, double)>;
using Dom1     = std::function<bool(double)>;
using Dom2     = std::function<bool(double, double)>;

struct UnaryOp {
  const char* name;
  Dom1        in_domain;
  InputDist   gen;
  FFUnary     apply;
};

struct BinaryOp {
  const char* name;
  Dom2        in_domain;
  InputDist   gen_a;
  InputDist   gen_b;
  FFBinary    apply;
};

// Common domain predicates.
static const Dom1 dom_any    = [](double x) { return std::isfinite(x); };
static const Dom1 dom_nonneg = [](double x) { return std::isfinite(x) && x >= 0.0; };

// Trig family lower bound: FF sincos's Taylor loop hits its iteration limit for
// tiny nonzero arguments (FFCSSNR), a purely FP32 hazard (DD never saw it — its
// FP64 r=x/2^nq stays representable). 0 is fine (early return); otherwise require
// |x| >= 1e-25. Upper bound |x| < 1e6 matches the sincos "argument too large"
// guard (1e30) with margin so many periods are covered.
static const Dom1 dom_trig = [](double x) {
  return std::isfinite(x) && (x == 0.0 || (std::fabs(x) >= 1e-25 && std::fabs(x) < 1e6));
};

// ----------------------------------------------------------------------------
// One input check (shared by random and corpus passes).
// ----------------------------------------------------------------------------
static void check_unary(const UnaryOp& op, double x, InvCount& c, int& samples_left) {
  if (!op.in_domain(x)) { ++c.skipped; return; }
  ff::FloatFloat d = op.apply(x);
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
  ff::FloatFloat d = op.apply(a, b);
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
    std::vector<float> xs = corpus::unary<float>(corpus_flags());
    for (float x : xs) check_unary(op, (double)x, c, samples_left);
  }

  std::printf("  %-16s tested=%-9ld skipped=%-9ld failures=%ld\n",
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
    std::vector<std::pair<float, float>> ps = corpus::binary<float>(corpus_flags());
    for (auto& p : ps) check_binary(op, (double)p.first, (double)p.second, c, samples_left);
  }

  std::printf("  %-16s tested=%-9ld skipped=%-9ld failures=%ld\n",
              op.name, c.tested, c.skipped, c.failures);
  return InvSummary{op.name, c.tested, c.skipped, c.failures};
}

// ----------------------------------------------------------------------------
// Device tripwire (Test B). Same invariant, computed on device for 5 ops.
// A custom runner is required: test_utils_host.hpp's run_unary_op/run_binary_op return
// digits-of-accuracy AccStats (and are scored against __float128), not the raw FF outputs
// this test needs — so we mirror their host->device->host View plumbing but ship
// hi/lo back and check non-overlap on host. Mirrors T1.2's device_unary/binary.
// ----------------------------------------------------------------------------
template <typename DeviceOp>
static InvSummary device_unary(const char* name, int n, uint64_t seed,
                               const InputDist& gen, const Dom1& in_domain,
                               DeviceOp op) {
  using exec_space = Kokkos::DefaultExecutionSpace;

  std::vector<double> hx(n);
  { std::mt19937_64 g(seed); for (int i = 0; i < n; ++i) hx[i] = gen(g); }

  Kokkos::View<float*, exec_space> dx("dx", n), dhi("dhi", n), dlo("dlo", n);
  auto hmx = Kokkos::create_mirror_view(dx);
  for (int i = 0; i < n; ++i) hmx(i) = (float)hx[i];
  Kokkos::deep_copy(dx, hmx);

  Kokkos::parallel_for("ff_inv_dev_unary", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      ff::FloatFloat d = op(ff::FloatFloat((double)dx(i)));
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
    ff::FloatFloat d(hhi(i), hlo(i));
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

  Kokkos::View<float*, exec_space> da("da", n), db("db", n), dhi("dhi", n), dlo("dlo", n);
  auto hma = Kokkos::create_mirror_view(da);
  auto hmb = Kokkos::create_mirror_view(db);
  for (int i = 0; i < n; ++i) { hma(i) = (float)ha[i]; hmb(i) = (float)hb[i]; }
  Kokkos::deep_copy(da, hma);
  Kokkos::deep_copy(db, hmb);

  Kokkos::parallel_for("ff_inv_dev_binary", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      ff::FloatFloat d = op(ff::FloatFloat((double)da(i)), ff::FloatFloat((double)db(i)));
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
    ff::FloatFloat d(hhi(i), hlo(i));
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
    std::printf("=== ff_invariant_test (T2.2): non-overlap invariant fl(hi+lo)==hi "
                "for every FF op ===\n");
    std::printf("Oracle-independent (raw FP32 check). Execution space: %s\n\n",
                Kokkos::DefaultExecutionSpace::name());

    std::vector<InvSummary> summary;

    // ---- Unary op registry -------------------------------------------------
    // Ranges are chosen op-appropriate for FP32: wide enough for broad coverage,
    // bounded to stay inside each op's mathematical domain and normal output
    // range so inputs are exercised rather than skipped, and so ff_math.hpp's
    // domain guards (which print) are never tripped. Bounds derived from
    // ff_math.hpp guards, NOT copied from T1.2 (FP32 is ~6x narrower than FP64).
    const std::vector<UnaryOp> unary_ops = {
      {"abs",       dom_any,     uniform(-1e8, 1e8),   [](double x){ return ff::abs(ff::FloatFloat(x)); }},
      {"negate",    dom_any,     uniform(-1e8, 1e8),   [](double x){ return ff::negate(ff::FloatFloat(x)); }},
      {"sqrt",      dom_nonneg,  uniform(0.0, 1e8),    [](double x){ return ff::sqrt(ff::FloatFloat(x)); }},
      // round-family: nint guards at |total| >= 2^47 (~1.4e14). Keep to |x|<1e13
      // so results are exact integers well inside the normal range.
      {"round_to_nearest_int", [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                    uniform(-1e6, 1e6),   [](double x){ return ff::round_to_nearest_int(ff::FloatFloat(x)); }},
      {"ceil",      [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                    uniform(-1e6, 1e6),   [](double x){ return ff::ceil(ff::FloatFloat(x)); }},
      {"floor",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                    uniform(-1e6, 1e6),   [](double x){ return ff::floor(ff::FloatFloat(x)); }},
      {"round",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                    uniform(-1e6, 1e6),   [](double x){ return ff::round(ff::FloatFloat(x)); }},
      {"trunc",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                    uniform(-1e6, 1e6),   [](double x){ return ff::trunc(ff::FloatFloat(x)); }},
      // exp-family: ff_math.hpp exp guards at a.hi >= 88.0f (FP32 ln-range), NOT
      // DD's 300. Keep the argument strictly below 88.
      {"exp",       [](double x){ return std::isfinite(x) && x < 88.0; },
                    uniform(-88.0, 87.5), [](double x){ return ff::exp(ff::FloatFloat(x)); }},
      // exp2(a)=exp(a*ln2): a*ln2 must stay < 88 -> |a| < ~127. Cap |a|<126.
      {"exp2",      [](double x){ return std::isfinite(x) && std::fabs(x) < 126.0; },
                    uniform(-120.0, 120.0), [](double x){ return ff::exp2(ff::FloatFloat(x)); }},
      // exp10(a)=exp(a*ln10): a*ln10 < 88 -> |a| < ~38. Cap |a|<38.
      {"exp10",     [](double x){ return std::isfinite(x) && std::fabs(x) < 38.0; },
                    uniform(-37.0, 37.0),   [](double x){ return ff::exp10(ff::FloatFloat(x)); }},
      {"expm1",     [](double x){ return std::isfinite(x) && x < 88.0; },
                    uniform(-5.0, 5.0),     [](double x){ return ff::expm1(ff::FloatFloat(x)); }},
      // log-family domain: 1e-34 <= x <= 1e34 AND x normal. ff_math's log refines
      // via an INTERNAL exp(b) with b ~ log(x); that Newton step only stays inside
      // exp's 88-guard while |b| = |log x| < ~78 (log(1e34)~78.3). Beyond ~1e37 the
      // internal exp trips its iteration limit (empirically confirmed: log(1e37)
      // prints FFEXP and returns NaN). Bounding |x| to [1e-34, 1e34] keeps
      // |log x| < ~78, safely inside exp's range. Random draws never approach
      // either tail; these gate the corpus extremes (maxT, smallest normals).
      {"log",       [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
                    uniform(1e-6, 1e6),   [](double x){ return ff::log(ff::FloatFloat(x)); }},
      {"log2",      [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
                    uniform(1e-6, 1e6),   [](double x){ return ff::log2(ff::FloatFloat(x)); }},
      {"log10",     [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
                    uniform(1e-6, 1e6),   [](double x){ return ff::log10(ff::FloatFloat(x)); }},
      // log1p(x)=log(1+x): 1+x is ~1 for tiny x, so no subnormal-log hazard; just
      // bound the upper magnitude like the log family.
      {"log1p",     [](double x){ return std::isfinite(x) && x > -1.0 && x < 1e34; },
                    uniform(-0.9, 1e6),   [](double x){ return ff::log1p(ff::FloatFloat(x)); }},
      // trig: FP32 sincos trips FFCSSNR for tiny nonzero args (see dom_trig).
      // 0 allowed; else 1e-25 <= |x| < 1e6.
      {"sin",       dom_trig, uniform(-1000.0, 1000.0), [](double x){ return ff::sin(ff::FloatFloat(x)); }},
      {"cos",       dom_trig, uniform(-1000.0, 1000.0), [](double x){ return ff::cos(ff::FloatFloat(x)); }},
      {"tan",       dom_trig, uniform(-1000.0, 1000.0), [](double x){ return ff::tan(ff::FloatFloat(x)); }},
      // asin/acos: |a|<=1 AND (0 or |a|>=1e-25) — asin(a)=angle(sqrt(1-a^2),a)
      // whose internal sincos Newton trips FFCSSNR on subnormal-tiny a.
      {"asin",      [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0 && (x == 0.0 || std::fabs(x) >= 1e-25); },
                    uniform(-1.0, 1.0),   [](double x){ return ff::asin(ff::FloatFloat(x)); }},
      {"acos",      [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0 && (x == 0.0 || std::fabs(x) >= 1e-25); },
                    uniform(-1.0, 1.0),   [](double x){ return ff::acos(ff::FloatFloat(x)); }},
      // atan(a)=angle(1,a): FF's angle drives an internal sincos Newton that trips
      // FFCSSNR for subnormal-tiny a (0 allowed). Cap |a|<1e18 so a*a stays finite
      // (angle forms r=sqrt(1+a^2); a*a must be < FLT_MAX ~3.4e38).
      {"atan",      [](double x){ return std::isfinite(x) && (x == 0.0 || (std::fabs(x) >= 1e-25 && std::fabs(x) < 1e18)); },
                    uniform(-1e6, 1e6),   [](double x){ return ff::atan(ff::FloatFloat(x)); }},
      // sinh/cosh via (e^a +/- e^-a)/2 (|a|>=0.5) or Taylor (|a|<0.5). exp guards
      // at 88; keep |a|<40 (exp(40)~2.4e17 finite; the |a|<0.5 Taylor is fine).
      {"sinh",      [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
                    uniform(-39.0, 39.0), [](double x){ return ff::sinh(ff::FloatFloat(x)); }},
      {"cosh",      [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
                    uniform(-39.0, 39.0), [](double x){ return ff::cosh(ff::FloatFloat(x)); }},
      // tanh(x)=expm1(2x)/(expm1(2x)+2): expm1 calls exp(2x); 2*|x|<88 -> |x|<44,
      // but expm1's intermediate overflows FP32 to NaN well before then
      // (empirically ~x=40). Cap |x|<20 (tanh has saturated to +/-1 to full FF
      // precision by ~x=10 anyway).
      {"tanh",      [](double x){ return std::isfinite(x) && std::fabs(x) < 20.0; },
                    uniform(-19.0, 19.0), [](double x){ return ff::tanh(ff::FloatFloat(x)); }},
      // asinh/acosh reduce to log(x + sqrt(x^2 +/- 1)); cap |x|<1e18 so x*x stays
      // finite in FP32 (empirically FFEXP iteration limit past ~2e19).
      {"asinh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e18; },
                    uniform(-1e6, 1e6),   [](double x){ return ff::asinh(ff::FloatFloat(x)); }},
      {"acosh",     [](double x){ return std::isfinite(x) && x >= 1.0 && x < 1e18; },
                    uniform(1.0, 1e6),    [](double x){ return ff::acosh(ff::FloatFloat(x)); }},
      {"atanh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1.0; },
                    uniform(-0.999, 0.999), [](double x){ return ff::atanh(ff::FloatFloat(x)); }},
      // erf/erfc: ff_math returns +/-1 immediately past |z|=6; the series/asymptotic
      // branches call exp(z^2) with z^2 < 36 < 88, so no guard trips for any finite z.
      {"erf",       dom_any,     uniform(-6.0, 6.0), [](double x){ return ff::erf(ff::FloatFloat(x)); }},
      {"erfc",      dom_any,     uniform(-6.0, 6.0), [](double x){ return ff::erfc(ff::FloatFloat(x)); }},
      // tgamma: x >= 1e-3 (the reflection path x<0.5 calls sin(pi*x); for x<~1e-3
      // that is sin of a tiny arg -> FFCSSNR) and x < 23 (the internal
      // pow(t,x+0.5)*exp(-t) overflows FP32 to NaN at x~24). Positive args only
      // (avoids the non-positive-integer poles).
      {"tgamma",    [](double x){ return std::isfinite(x) && x >= 1e-3 && x < 23.0; },
                    uniform(0.1, 20.0),   [](double x){ return ff::tgamma(ff::FloatFloat(x)); }},
    };

    // Two-output ops, tested as one "op" per output component.
    // sincos(a, cos, sin); sinhcosh(a, cosh, sinh)  (see ff_math.hpp).
    const std::vector<UnaryOp> two_out_ops = {
      {"sincos.cos",   dom_trig, uniform(-1000.0, 1000.0),
                       [](double x){ ff::FloatFloat c, s; ff::sincos(ff::FloatFloat(x), c, s); return c; }},
      {"sincos.sin",   dom_trig, uniform(-1000.0, 1000.0),
                       [](double x){ ff::FloatFloat c, s; ff::sincos(ff::FloatFloat(x), c, s); return s; }},
      {"sinhcosh.cosh",[](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
                       uniform(-39.0, 39.0),
                       [](double x){ ff::FloatFloat c, s; ff::sinhcosh(ff::FloatFloat(x), c, s); return c; }},
      {"sinhcosh.sinh",[](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
                       uniform(-39.0, 39.0),
                       [](double x){ ff::FloatFloat c, s; ff::sinhcosh(ff::FloatFloat(x), c, s); return s; }},
    };

    // ---- Binary op registry ------------------------------------------------
    const Dom2 dom2_any    = [](double a, double b){ return std::isfinite(a) && std::isfinite(b); };
    // fmod/remainder form q = a/b then trunc/nint(q); |q| >= 2^47 (~1.4e14) trips
    // the nint guard. Require |a| < |b|*1e13 so the quotient stays in range —
    // extreme-ratio pairs (corpus maxT/1, huge_tiny) are skipped rather than fed.
    const Dom2 dom2_modbnz = [](double a, double b){
      return std::isfinite(a) && std::isfinite(b) && b != 0.0 &&
             std::fabs(a) < std::fabs(b) * 1e13;
    };
    const Dom2 dom2_bnz    = [](double a, double b){ return std::isfinite(a) && std::isfinite(b) && b != 0.0; };
    // pow(a,b)=exp(b*log(a)); the FINAL exp trips its 88 guard whenever
    // |b*ln(a)| >= 88. Gate on the predicted exponent so only pairs whose result
    // is representable are fed. a>0 is the mathematical domain; restrict the base
    // to the same normal [1e-34, 1e34] window as the log family (pow's INTERNAL
    // log(a) hits the same limit for tiny/subnormal a).
    const Dom2 dom2_powpos = [](double a, double b){
      return std::isnormal(a) && std::isfinite(b) && a >= 1e-34 && a <= 1e34 &&
             std::fabs(b * std::log(a)) < 88.0;
    };
    // atan2(y,x)=angle(x,y), which forms r=sqrt(x^2+y^2) and drives an internal
    // sincos Newton. FP32 hazards bound the domain (result stays invariant-clean
    // either way; the internal guard just prints): x^2+y^2 must stay finite (cap
    // the larger magnitude below 1e18) AND neither operand may be subnormal-tiny.
    // In FP32, a subnormal operand paired with a normal one drives the internal
    // sincos degenerate -> FFCSSNR (confirmed on corpus pairs like
    // (1.4e-45, 1)); DD's FP64 sincos absorbed this, so this per-operand floor is
    // an FF-specific tightening of T1.2's larger-magnitude-only gate. m=0 is
    // atan2(0,0) (defined, no hazard).
    const Dom2 dom2_atan2 = [](double a, double b){
      if (!(std::isfinite(a) && std::isfinite(b))) return false;
      double m = std::fmax(std::fabs(a), std::fabs(b));
      if (m == 0.0) return true;
      if (m > 1e18) return false;
      if (a != 0.0 && std::fabs(a) < 1e-18) return false;
      if (b != 0.0 && std::fabs(b) < 1e-18) return false;
      return true;
    };

    const std::vector<BinaryOp> binary_ops = {
      {"add",       dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return ff::add(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"subtract",  dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return ff::subtract(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"multiply",  dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return ff::multiply(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"divide",    dom2_bnz, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return ff::divide(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"pow",       dom2_powpos, uniform(0.1, 20.0), uniform(-6.0, 6.0),
                    [](double a, double b){ return ff::pow(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"atan2",     dom2_atan2, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](double a, double b){ return ff::atan2(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"hypot",     dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return ff::hypot(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"fmod",      dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](double a, double b){ return ff::fmod(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"remainder", dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](double a, double b){ return ff::remainder(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"copysign",  dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](double a, double b){ return ff::copysign(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"fmax",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return ff::fmax(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"fmin",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return ff::fmin(ff::FloatFloat(a), ff::FloatFloat(b)); }},
      {"fdim",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](double a, double b){ return ff::fdim(ff::FloatFloat(a), ff::FloatFloat(b)); }},
    };

    // -- Test A: every op ----------------------------------------------------
    std::printf("[Test A] per-op invariant (10^6 random + full corpus)\n");
    uint64_t seed = 12345ULL;
    for (const auto& op : unary_ops)   summary.push_back(run_unary(op,  seed++));
    for (const auto& op : two_out_ops) summary.push_back(run_unary(op,  seed++));
    for (const auto& op : binary_ops)  summary.push_back(run_binary(op, seed++));

    // Ternary fma(a, b, c): invariant on the fused result. All three streams off
    // one engine keep the run reproducible. Ranges kept modest so a*b+c stays
    // finite in FP32 (1e4*1e4 = 1e8 << FLT_MAX).
    {
      InvCount c; int samples_left = 3;
      InputDist g = uniform(-1e4, 1e4);
      {
        std::mt19937_64 gen(seed++);
        for (int i = 0; i < kRandomN; ++i) {
          double a = g(gen), b = g(gen), cc = g(gen);
          if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(cc))) { ++c.skipped; continue; }
          ff::FloatFloat d = ff::fma(ff::FloatFloat(a), ff::FloatFloat(b), ff::FloatFloat(cc));
          if (!result_checkable(d)) { ++c.skipped; continue; }
          ++c.tested;
          if (!invariant_holds(d)) {
            ++c.failures;
            if (samples_left > 0) {
              std::printf("    FAIL fma        a=%.9g b=%.9g c=%.9g  hi=%.9g lo=%.9g\n",
                          a, b, cc, d.hi, d.lo);
              --samples_left;
            }
          }
        }
      }
      std::printf("  %-16s tested=%-9ld skipped=%-9ld failures=%ld\n",
                  "fma", c.tested, c.skipped, c.failures);
      summary.push_back(InvSummary{"fma", c.tested, c.skipped, c.failures});
    }

    // pow_int(a, n): integer exponent. Domain: not 0^negative (guard print), and
    // keep |a|^|n| finite in FP32 — base in [-5,5], n in [-20,20] gives at most
    // 5^20 ~= 9.5e13 << FLT_MAX.
    {
      InvCount c; int samples_left = 3;
      InputDist gx = uniform(-5.0, 5.0);
      std::mt19937_64 gen(seed++);
      std::uniform_int_distribution<int> dn(-20, 20);
      for (int i = 0; i < kRandomN; ++i) {
        double x = gx(gen);
        int    n = dn(gen);
        if (!std::isfinite(x) || (x == 0.0 && n < 0)) { ++c.skipped; continue; }
        ff::FloatFloat d = ff::pow_int(ff::FloatFloat(x), n);
        if (!result_checkable(d)) { ++c.skipped; continue; }
        ++c.tested;
        if (!invariant_holds(d)) {
          ++c.failures;
          if (samples_left > 0) {
            std::printf("    FAIL pow_int    x=%.9g n=%d  hi=%.9g lo=%.9g\n", x, n, d.hi, d.lo);
            --samples_left;
          }
        }
      }
      std::printf("  %-16s tested=%-9ld skipped=%-9ld failures=%ld\n",
                  "pow_int", c.tested, c.skipped, c.failures);
      summary.push_back(InvSummary{"pow_int", c.tested, c.skipped, c.failures});
    }

    // -- Test B: device tripwire (5 representative ops) ----------------------
    std::printf("\n[Test B] device tripwire (5 ops, 10^5 random on %s)\n",
                Kokkos::DefaultExecutionSpace::name());
    const int nd = 100'000;
    summary.push_back(device_binary("add", nd, 55501ULL,
        uniform(-1e8, 1e8), uniform(-1e8, 1e8), dom2_any,
        KOKKOS_LAMBDA(ff::FloatFloat a, ff::FloatFloat b){ return ff::add(a, b); }));
    summary.push_back(device_binary("multiply", nd, 55502ULL,
        uniform(-1e6, 1e6), uniform(-1e6, 1e6), dom2_any,
        KOKKOS_LAMBDA(ff::FloatFloat a, ff::FloatFloat b){ return ff::multiply(a, b); }));
    summary.push_back(device_unary("sqrt", nd, 55503ULL,
        uniform(0.0, 1e8), dom_nonneg,
        KOKKOS_LAMBDA(ff::FloatFloat x){ return ff::sqrt(x); }));
    summary.push_back(device_unary("exp", nd, 55504ULL,
        uniform(-88.0, 87.5), [](double x){ return std::isfinite(x) && x < 88.0; },
        KOKKOS_LAMBDA(ff::FloatFloat x){ return ff::exp(x); }));
    summary.push_back(device_unary("sin", nd, 55505ULL,
        uniform(-1000.0, 1000.0), dom_trig,
        KOKKOS_LAMBDA(ff::FloatFloat x){ return ff::sin(x); }));

    // -- Test C: explicit PORT_NOTES §4 regressions --------------------------
    // These are the FF-side counterparts to T1.2's Test C. Where T1.2 documented
    // that DD has NO equivalent of the FF bugs, T2.2 exercises the FF ops on the
    // very inputs the historical bugs broke, asserting the invariant AND the
    // bug-specific outcome.
    std::printf("\n[Test C] PORT_NOTES §4 named regressions (historical FF bugs)\n");
    int c_pass = 0, c_total = 0;

    // §4a splits into TWO DISTINCT roles — do NOT conflate them (they test
    // different things, and only the first set is the actual bug regression):
    //
    //  * 79.5 / 80 / 85  — BUG-REGRESSION cases. The historical §4a bug was
    //    NaN-from-splitter-overflow: exp's internal Dekker split did `b * 8193.0f`,
    //    which OVERFLOWED FP32 for large b and produced NaN. These three are the
    //    LOAD-BEARING regression cases — pre-fix they returned NaN; post-fix
    //    (direct scaling) they must return a FINITE, invariant-clean result. This
    //    is what proves the §4a fix is in place. No diagnostic is expected here.
    //
    //  * 88.7 / 88.72  — EDGE-OF-SATURATION GUARD cases. These sit PAST
    //    ff_math.hpp's a.hi >= 88.0 guard (FP32 ln-range). They do NOT re-test the
    //    §4a bug; they test that the guard FIRES SENSIBLY at the edge — saturating
    //    to +0 (invariant trivially holds; not-NaN) instead of producing garbage.
    //    Each emits ONE "FFEXP: argument too large" diagnostic; those 2 prints are
    //    EXPECTED and NORMAL (documented safety-guard behavior, not a bug — so this
    //    is a PASS, not a report-and-stop). They are the ONLY internal ff_math.hpp
    //    diagnostics the whole run produces; Test A/B are diagnostic-clean.
    auto case_exp = [&](double x) {
      ++c_total;
      ff::FloatFloat d = ff::exp(ff::FloatFloat(x));
      bool guarded = (x >= 88.0);  // past ff_math.hpp's a.hi >= 88 exp guard
      bool ok = !std::isnan(d.hi) && !std::isinf(d.hi) && invariant_holds(d);
      std::printf("    exp(%.5g) [%s]: hi=%.9g lo=%.5g  %s%s\n", x,
                  guarded ? "edge-of-saturation guard" : "bug-regression (NaN pre-fix)",
                  d.hi, d.lo, ok ? "PASS" : "FAIL",
                  guarded ? " (saturates to +0; 1 FFEXP diagnostic expected+normal)" : "");
      if (ok) ++c_pass; else print_fail_unary("exp", x, d);
    };
    for (float x : corpus::exp_overflow<float>()) case_exp((double)x);  // 79.5,80,85,88.7,88.72

    // §4b: ffnint off-by-one. The historical bug had ffnint(19.4999993...) return
    // 20 instead of 19. There is an FP32 SUBTLETY worth recording: the literal
    // 19.4999993f rounds to EXACTLY 19.5f at FP32 (lo=0), so round_to_nearest_int
    // of the pure-float value correctly returns 20. The 19 case only appears when
    // the FULL-PRECISION value 19.4999993 is carried in the FF pair via the
    // Route-A double split (hi=19.5, lo=-7e-7, total < 19.5), where the FIXED
    // ffnint (which rounds hi+lo in FP64) returns 19. We therefore check the
    // invariant on both constructions AND, for the Route-A value, that the fixed
    // nint lands on 19 — the historical off-by-one would have given 20.
    auto case_nint_inv = [&](double x) {
      ++c_total;
      ff::FloatFloat d = ff::round_to_nearest_int(ff::FloatFloat(x));
      bool ok = result_checkable(d) && invariant_holds(d);
      std::printf("    nint(%.10g): hi=%.9g lo=%.5g  %s\n", x, d.hi, d.lo, ok ? "PASS" : "FAIL");
      if (ok) ++c_pass; else print_fail_unary("nint", x, d);
    };
    // The literal FF offender plus the k+/-0.5 boundary family (invariant only).
    for (float x : corpus::nint_half_integer<float>()) case_nint_inv((double)x);

    // The §4b value carried at full precision: fixed ffnint must return 19.
    {
      ++c_total;
      ff::FloatFloat d = ff::round_to_nearest_int(ff::FloatFloat(19.4999993));  // Route-A split
      bool inv = invariant_holds(d);
      bool nint_ok = (d.hi == 19.0f);  // fixed ffnint: total 19.4999993 < 19.5 -> 19
      bool ok = inv && nint_ok;
      std::printf("    nint(19.4999993 via double-ctor): hi=%.9g lo=%.5g  nint=%s inv=%s  %s\n",
                  d.hi, d.lo, nint_ok ? "19(fixed)" : "NOT-19", inv ? "ok" : "BAD",
                  ok ? "PASS" : "FAIL");
      if (ok) ++c_pass; else print_fail_unary("nint", 19.4999993, d);
    }

    // §4b: remainder(68.379, 3.5066): invariant AND sign, gated against
    // std::remainderf (single-precision libm — the SAME-PRECISION oracle for FF).
    //
    // FINDING (reported regardless of outcome, per the T2.2 prompt): PORT_NOTES
    // §4b's text says the FP32-correct answer here is POSITIVE (+1.7533), on the
    // premise that a/b ~ 19.4999993 < 19.5 at FP32, so nint=19. That premise does
    // NOT hold for the corpus literals: at FP32, 68.379f/3.5066f = 19.5000858
    // (> 19.5), so the CORRECT nint is 20 and the CORRECT remainder is NEGATIVE
    // (-1.75300026). std::remainderf(68.379f, 3.5066f) agrees (-1.75300026), and
    // the shipped FF remainder reproduces it exactly. So the "+1.7533" in
    // PORT_NOTES §4b describes a DIFFERENT rounding of a/b than these specific
    // literals produce — the FP32 quotient here sits just ABOVE 19.5, not below.
    // The historical ffnint BUG was a real off-by-one (it rounded a/b to the wrong
    // integer); the FIXED ffnint now agrees with libm. We gate FF's sign against
    // std::remainderf (same precision — NOT std::remainder, which would test the
    // FP64 answer against an FP32 op), which is the correct oracle and passes.
    {
      ++c_total;
      float a = 68.379f, b = 3.5066f;
      ff::FloatFloat d = ff::remainder(ff::FloatFloat((double)a), ff::FloatFloat((double)b));
      float ref = std::remainderf(a, b);                 // FP32 IEEE oracle
      bool inv = invariant_holds(d);
      bool sign_ok = (d.hi < 0.0f) == (ref < 0.0f);      // FF sign matches FP32 libm
      bool ok = inv && sign_ok;
      std::printf("    remainder(%.5g, %.5g): hi=%.9g lo=%.5g  sign=%s "
                  "remainderf=%.9g inv=%s  %s\n",
                  (double)a, (double)b, d.hi, d.lo, d.hi < 0.0f ? "-" : "+",
                  (double)ref, inv ? "ok" : "BAD", ok ? "PASS" : "FAIL");
      std::printf("      [finding] FP32 a/b = %.9g (%s 19.5) -> nint=%d -> sign %s; "
                  "PORT_NOTES §4b '+1.7533' premise (a/b<19.5) does NOT hold for these literals\n",
                  (double)(a / b), (a / b) > 19.5f ? ">" : "<",
                  (int)std::nearbyint(a / b), ref < 0.0f ? "NEGATIVE" : "POSITIVE");
      if (ok) ++c_pass; else print_fail_binary("remainder", (double)a, (double)b, d);
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
                     "one or more FF ops produced an overlapping (hi, lo) result");
    KOKKOS_EP_ASSERT(c_pass == c_total, "a PORT_NOTES §4 named regression case failed");

    rc = ep_exit_code();
    std::printf("\n=== ff_invariant_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();
  return rc;
}
