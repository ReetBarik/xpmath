// ============================================================================
// ff_property_test.cpp — Layer 3 (property / identity tests) for FF.  Plan T2.3.
// ============================================================================
//
// FF analogue of dd_property_test.cpp (T1.3). The structure is mirrored end to
// end; the mechanical change is the PRECISION SCALE:
//
//     DD:  u = 2^-53, u^2 = 2^-106  -> tolerance_digits ~= 25.91 at N=10^6
//     FF:  u = 2^-24, u^2 = 2^-48   -> tolerance_digits ~=  8.45 at N=10^6
//                                       (= -log10(N * u^2), computed at runtime)
//
// WHAT THIS LAYER CHECKS
// ----------------------
// Layer 1 (ff_eft_test, T2.1) proved twoSum + Dekker twoProduct are bit-exact;
// Layer 2 (ff_invariant_test, T2.2) proved every FF op returns a non-overlapping
// (hi, lo) with fl(hi+lo)==hi. This layer checks ALGEBRAIC identities the composed
// ops must satisfy. Three flavours:
//
//   GROUP A — bit-exact identities.  Pure sign/structure identities that MUST hold
//     to the last FP32 bit with NO tolerance and NO oracle: additive inverse,
//     self-subtraction, multiply by +/-1, |a| sign branches, double negation, and
//     addition commutativity. Gated by exact FP32 equality on BOTH components.
//     Group A runs unconditionally — it needs no __float128.
//
//   GROUP B — approximate identities.  Round-trips (sqrt/square, exp/log),
//     Pythagorean sin^2+cos^2, sin/cos symmetry, tan*cos, double-angle, and the
//     DEMOTED multiply-commutativity (B0). Reported in "digits of accuracy" via the
//     __float128 machinery in test_utils.hpp. Group B and Test C used to be
//     #ifdef'd on KOKKOS_EP_HAVE_QUADMATH and to SKIP (77) without it; they now
//     run unconditionally, because __float128 needs no TPL.
//
//   TEST C — named-constant regressions.  sin(pi)~=0, log(e)~=1, exp(log2)~=2,
//     sqrt2*sqrt2~=2, log(10)~=ln(10) constant — each to >= a named-min scaled to
//     FF's 14-digit cap.
//
// FF-SPECIFIC NOTES (beyond the pure DD->FF scale change)
// -------------------------------------------------------
//   * Operand construction. DD's Group A fed SINGLE-DOUBLE operands (DoubleDouble
//     from a plain double has lo==0). The FF analogue of "single-double" is
//     "single-float" (one FP32 with lo==0). A1-A6 use full Route-A FF operands
//     (FloatFloat(double), lo generally NONZERO) because those identities are
//     bit-exact regardless of lo (pure sign, or multiply-by-+/-1 which reconstructs
//     a non-overlapping pair exactly). A8 (add commutativity) uses SINGLE-FLOAT
//     operands (lo==0): Knuth twoSum's error term is exact and order-independent,
//     but add()'s trailing "+ a.lo + b.lo" reorders under operand swap and would
//     lose bit-exactness with a nonzero lo — so A8 is claimed bit-exact only for
//     single-float operands, exactly the way DD claims it only for single-double.
//
//   * A7 mul-commutativity is DEMOTED to Group B (B0), same reason as DD:
//     multiply()'s Dekker cross-term sum c21 adds a1*b2 and a2*b1 in a FIXED
//     a-first order (in `multiply`), so swapping operands reorders two
//     addends and FP addition is not associative.
//
//   * Denormal-tail audit. Group A compares raw FP32 (hi,lo) pairs, so a result
//     landing in the FP32 denormal tail could re-trip the round-to-even hole the
//     T2.2 invariant guards with kUnderflowTail = 2^-100. The multiply-based
//     A3/A4 are additionally gated by dom_dekker (T2.1 split_safe_max upper bound
//     + kUnderflowTail lower bound); any residual tail mismatch is counted as
//     SKIPPED, not FAILED.
//
//   * exp guard + the B8 divide fix. ff_math.hpp exp returns 0 (and prints) for
//     a.hi >= 88.0f, so the exp round-trips narrow away from DD's +/-290 range. A
//     SECOND, tighter limit was discovered during T2.3: exp round-trips stalled at
//     the 60-iteration cap ("FFEXP: iteration limit") on ~[70,85] arguments. T2.3
//     hypothesized the Taylor eps (1e-15f, finer than FloatFloat's ~3.55e-15) and
//     narrowed B3/B9 to +/-69 pending bug task B4. B4's investigation FALSIFIED the
//     eps theory: the real defect was a Dekker splitter overflow in divide()
//     (b.hi * split -> inf, inf - inf = NaN) poisoning log()'s Newton exp for
//     x ≳ e^79.7. B8 fixed divide() with a scaled splitter, so B3/B9 are RESTORED to
//     +/-79 (one integer under the ~79.7 empirical clean ceiling; not +/-85). B2
//     (exp arg = log(x), |.|<=69) was already clean and stays on [1e-30,1e30]. See
//     docs/TEST_SUITE_PLAN.md §B4/§B8.
//
//   * Large-argument trig. Per PORT_NOTES §5 sin/cos near +/-pi*k need triple-float
//     argument reduction (out of scope). sin^2+cos^2 (B4) and the symmetry checks
//     (B5/B6) are robust to reduction error (identities of the reduced angle), but
//     the double-angle check B8 compares sin(2a) against 2*sin(a)*cos(a) — two
//     DIFFERENT reduced angles — so B8 narrows to |a| < 3 where reduction is clean.
//
// DELIBERATELY NOT TESTED (anti-tests) — see the block near the bottom of main():
// associativity of add and distributivity across large-magnitude differences. Both
// are FALSE for every finite-precision format, FF included.
//
// SCOPE: real FF ops only (ff_complex.hpp out of scope). ff_math.hpp is NOT
// modified (rule 4): if an identity failed unexpectedly this test would REPORT
// (bit patterns) and fail, not patch the library.
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 2, "T2.3: Property/identity
// tests for FF" and "The six test layers" layer 3.
// ============================================================================

#include "test_utils.hpp"
#include "corpus.hpp"
#include <ff_math.hpp>

#include <algorithm>
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
namespace ff = Kokkos::Experimental;

// ----------------------------------------------------------------------------
// Bit-pattern helpers (ff_eft_test.cpp / ff_invariant_test.cpp hex format).
// ----------------------------------------------------------------------------
static uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(float));
  return b;
}

// Exact FP32 equality on both components. NOTE: uses value equality (==), so
// +0.0 and -0.0 compare equal — intended, since the zero-result identities are
// specified as "hi==0.0f && lo==0.0f", i.e. the value zero regardless of sign bit.
static bool ff_eq(const ff::FloatFloat& x, const ff::FloatFloat& y) {
  return x.hi == y.hi && x.lo == y.lo;
}
static bool ff_is_zero(const ff::FloatFloat& x) {
  return x.hi == 0.0f && x.lo == 0.0f;
}

// Denormal-tail guard (T2.2): the strict bit-exact comparison can trip the FP32
// round-to-even hole when a result's hi limb falls into the subnormal tail. Below
// this magnitude a mismatch is counted as SKIPPED rather than FAILED (a domain
// limit of FP32, not an ff_math.hpp defect). 2^-100 is 4x above the 2^-102 floor.
static constexpr double kUnderflowTail = 0x1p-100;  // ~7.9e-31
static bool in_underflow_tail(const ff::FloatFloat& v) {
  return v.hi != 0.0f && std::fabs((double)v.hi) < kUnderflowTail;
}

// T2.1 split_safe_max: the Dekker (Veltkamp) split of an operand x computes
// x*8193; that product overflows to inf (-> inf-inf = NaN in the split) for
// |x| >= FLT_MAX/8193 ~= 2^114.9998 ~= 4.15e34, REGARDLESS of the other operand.
// Such inputs are OUT OF multiply's domain and are SKIPPED, not failed (same limit
// documented in ff_eft_test, T2.1). Redefined locally (the T2.1 helper is static
// to that TU) so this file consumes the identical bound.
static float split_safe_max() {
  return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

static void print_fail_unary(const char* id, double x, const ff::FloatFloat& got,
                             const ff::FloatFloat& want) {
  std::printf("    FAIL %-12s x=%.9g  got.hi=%.9g (0x%08x) got.lo=%.9g (0x%08x)  "
              "want.hi=%.9g (0x%08x) want.lo=%.9g (0x%08x)\n",
              id, x,
              got.hi, fbits(got.hi), got.lo, fbits(got.lo),
              want.hi, fbits(want.hi), want.lo, fbits(want.lo));
}
static void print_fail_binary(const char* id, double a, double b,
                              const ff::FloatFloat& got, const ff::FloatFloat& want) {
  std::printf("    FAIL %-12s a=%.9g b=%.9g  got.hi=%.9g (0x%08x) got.lo=%.9g (0x%08x)  "
              "want.hi=%.9g (0x%08x) want.lo=%.9g (0x%08x)\n",
              id, a, b,
              got.hi, fbits(got.hi), got.lo, fbits(got.lo),
              want.hi, fbits(want.hi), want.lo, fbits(want.lo));
}

// ----------------------------------------------------------------------------
// Uniform report line, shared by Group A and Group B.
// ----------------------------------------------------------------------------
static void report_line(const char* name, long n, double min_d, double mean_d,
                        double tol_d, bool pass) {
  std::printf("  %-16s n=%-9ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, min_d, mean_d, tol_d, pass ? "PASS" : "FAIL");
}

// Corpus flags: zeros ON, inf OFF, nan OFF (subnormals default ON) — matches the
// invariant test. Sign-flip / additive identities hold for every finite input;
// inf/nan are excluded because e.g. inf + (-inf) = nan is not the zero identity.
static corpus::CorpusFlags corpus_flags() {
  corpus::CorpusFlags f;
  f.include_zero = true;
  f.include_inf  = false;
  f.include_nan  = false;
  return f;
}

static constexpr int    kRandomN = 1'000'000;  // 10^6 random inputs per identity
static constexpr int    kDeviceN = 100'000;    // 10^5 for the device pass
static constexpr double kMaxDig  = (double)BackendTraits<FF>::max_digits;  // 14

// ============================================================================
// GROUP A — bit-exact identities (no oracle, no tolerance)
// ============================================================================
// Each identity is a predicate on an FF value that must hold to the bit. A runner
// sweeps 10^6 random inputs + the full finite corpus, counts failures, and dumps
// the first <=3 with FP32 bit patterns. On zero failures the reported min/mean are
// the max_digits cap (bit-exact => "infinite" agreement, clamped to 14).

struct GroupAResult { std::string name; long n; long skipped; long failures; };

// Domain predicates for Group A. Most sign/additive identities hold for every
// finite input (dom_all). The multiply-based ones (A3, A4) go through Dekker's
// Veltkamp split, so they are gated by dom_dekker: finite, |x| below the splitter-
// overflow bound (split_safe_max), and |x| out of the denormal tail (>= kUnderflow
// Tail, or exactly 0). Random |x| < 1e8 never approaches either bound; this only
// gates the corpus's maxT / subnormal / huge entries.
static const std::function<bool(double)> dom_all = [](double x){ return std::isfinite(x); };
static const std::function<bool(double)> dom_dekker = [](double x){
  if (!std::isfinite(x)) return false;
  double ax = std::fabs(x);
  if (ax == 0.0) return true;                       // exact: a*(+/-1) with a==0
  if (ax >= (double)split_safe_max()) return false; // Dekker splitter overflow
  if (ax < kUnderflowTail) return false;            // denormal-tail / subnormal
  return true;
};

// A unary identity: given a (as FF), return (got, want); ff_eq(got,want) must hold.
using UnaryIdentity = std::function<void(const ff::FloatFloat& a,
                                         ff::FloatFloat& got, ff::FloatFloat& want)>;

static GroupAResult run_group_a_unary(const char* name, uint64_t seed,
                                      const InputDist& gen, const UnaryIdentity& id,
                                      const std::function<bool(double)>& in_domain = dom_all) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  auto one = [&](double x) {
    if (!in_domain(x)) { ++skipped; return; }
    ff::FloatFloat a(x), got, want;   // Route-A split: lo generally nonzero
    id(a, got, want);
    // Denormal-tail audit: a mismatch whose limbs are in the FP32 subnormal tail
    // is a domain limit (round-to-even hole), SKIP not FAIL.
    if (!ff_eq(got, want) && (in_underflow_tail(got) || in_underflow_tail(want))) {
      ++skipped; return;
    }
    ++n;
    if (!ff_eq(got, want)) {
      ++fails;
      if (samples_left > 0) { print_fail_unary(name, x, got, want); --samples_left; }
    }
  };
  { std::mt19937_64 g(seed); for (int i = 0; i < kRandomN; ++i) one(gen(g)); }
  for (float x : corpus::unary<float>(corpus_flags())) one((double)x);

  bool pass = (fails == 0);
  std::printf("  %-16s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

// A binary identity (only A8 add-commutativity here). SINGLE-FLOAT operands
// (lo==0) — the FF analogue of DD's single-double convention (see file header):
// with lo==0 the add() tail "+a.lo+b.lo" vanishes and Knuth twoSum is bit-exact
// under operand swap.
using BinaryIdentity = std::function<void(const ff::FloatFloat& a, const ff::FloatFloat& b,
                                          ff::FloatFloat& got, ff::FloatFloat& want)>;

static GroupAResult run_group_a_binary(const char* name, uint64_t seed,
                                       const InputDist& ga, const InputDist& gb,
                                       const BinaryIdentity& id) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  // add() does not Veltkamp-split, so it has no splitter-overflow limit; the only
  // out-of-domain case for A8 is a non-finite operand (skip, don't fail).
  auto one = [&](double av, double bv) {
    if (!(std::isfinite(av) && std::isfinite(bv))) { ++skipped; return; }
    ff::FloatFloat a((float)av), b((float)bv), got, want;   // single-float: lo==0
    id(a, b, got, want);
    ++n;
    if (!ff_eq(got, want)) {
      ++fails;
      if (samples_left > 0) { print_fail_binary(name, av, bv, got, want); --samples_left; }
    }
  };
  { std::mt19937_64 g(seed);
    for (int i = 0; i < kRandomN; ++i) { double a = ga(g), b = gb(g); one(a, b); } }
  for (auto& p : corpus::binary<float>(corpus_flags())) one((double)p.first, (double)p.second);

  bool pass = (fails == 0);
  std::printf("  %-16s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

// ============================================================================
// GROUP S — IEEE special-value semantics for division (no oracle, no tolerance)
// ============================================================================
// Group A sweeps a corpus with inf/nan deliberately OFF (see corpus_flags), and
// ff_invariant_test SKIPS non-finite hi by construction, so nothing in the FF
// suite pinned down what divide() returns for a non-finite DIVISOR. It returned
// NaN for every one of them: B8's divisor guard classifies |b.hi| > thresh, which
// is true for +/-inf, scales by 2^-64 (inf stays inf), and the Dekker split then
// evaluates inf - inf = NaN. IEEE 754-2019 §6.1 requires x / +/-inf = +/-0 for
// finite x. See PORT_NOTES §4l.
//
// Written as an explicit expectation TABLE rather than a computed reference: the
// point is to encode what IEEE mandates independently, not to recompute the
// quotient with the same primitive under test.
//
// The divide-by-ZERO rows are regression guards, not new behavior — that path is
// B10's Zone C and is deliberately untouched by the §4l fix.

enum class SpecialWant { PosZero, NegZero, NaN, PosInf, NegInf };

static const char* want_name(SpecialWant w) {
  switch (w) {
    case SpecialWant::PosZero: return "+0";
    case SpecialWant::NegZero: return "-0";
    case SpecialWant::NaN:     return "NaN";
    case SpecialWant::PosInf:  return "+inf";
    default:                   return "-inf";
  }
}

// Classify an FF result. For a zero/inf expectation the value must be EXACT, so
// lo is required to be +/-0 as well; for NaN only hi's NaN-ness is checked, since
// IEEE leaves a NaN's sign and payload unspecified.
static bool special_matches(ff::FloatFloat got, SpecialWant w) {
  const bool lo_is_zero = (got.lo == 0.0f);
  switch (w) {
    case SpecialWant::PosZero: return got.hi == 0.0f && !std::signbit(got.hi) && lo_is_zero;
    case SpecialWant::NegZero: return got.hi == 0.0f &&  std::signbit(got.hi) && lo_is_zero;
    case SpecialWant::NaN:     return std::isnan(got.hi);
    case SpecialWant::PosInf:  return std::isinf(got.hi) && !std::signbit(got.hi) && lo_is_zero;
    default:                   return std::isinf(got.hi) &&  std::signbit(got.hi) && lo_is_zero;
  }
}

static const char* describe(float v) {
  static char buf[8][32]; static int slot = 0;
  char* b = buf[slot = (slot + 1) & 7];
  if (std::isnan(v))      std::snprintf(b, 32, "NaN");
  else if (std::isinf(v)) std::snprintf(b, 32, "%cinf", std::signbit(v) ? '-' : '+');
  else if (v == 0.0f)     std::snprintf(b, 32, "%c0",   std::signbit(v) ? '-' : '+');
  else                    std::snprintf(b, 32, "%g", (double)v);
  return b;
}

struct SpecialCase { float num; float den; SpecialWant want; };

// Returns the number of failures; prints one line per failing case.
static long run_group_s(const std::vector<SpecialCase>& cases, long& checked) {
  long fails = 0;
  for (const auto& c : cases) {
    const ff::FloatFloat a(c.num);
    // divide(FF, FF)
    {
      ++checked;
      ff::FloatFloat got = ff::divide(a, ff::FloatFloat(c.den));
      if (!special_matches(got, c.want)) {
        ++fails;
        std::printf("    FAIL divide(%s, %s): want %s, got (%g, %g)\n",
                    describe(c.num), describe(c.den), want_name(c.want),
                    (double)got.hi, (double)got.lo);
      }
    }
    // divide_scalar(FF, float) — same expectation, scalar divisor.
    {
      ++checked;
      ff::FloatFloat got = ff::divide_scalar(a, c.den);
      if (!special_matches(got, c.want)) {
        ++fails;
        std::printf("    FAIL divide_scalar(%s, %s): want %s, got (%g, %g)\n",
                    describe(c.num), describe(c.den), want_name(c.want),
                    (double)got.hi, (double)got.lo);
      }
    }
  }
  return fails;
}

// ============================================================================
// GROUP B — approximate identities (tolerance-based, needs the __float128 path)
// ============================================================================
// Each identity produces, per input, a (computed_quad, reference_quad) pair and
// scores digits_of_accuracy against it (capped at FF's 14 max_digits). We report
// min AND mean, and gate on the MEAN clearing the statistical threshold
//     tolerance_digits = -log10(N * u^2),  u = 2^-24, u^2 = 2^-48,
// which is the worst-of-N floor for a per-sample error at the u^2 scale (the
// per-identity theoretical bound, 2u^2/4u^2/10u^2, is cited at each call site).
// Gating on the mean — not the min — lets conditioning-limited samples (sin^2+cos^2
// far from a small argument, exp output near the FP32 denormal edge, etc.) dip
// without red-flagging a correct implementation; low mins are annotated instead.

struct GroupBResult { std::string name; long n; double min_d, mean_d, tol_d; bool pass; };

static double threshold_digits(long n) {
  // -log10(N * u^2): statistical floor over N samples at the u^2 error scale.
  return -std::log10((double)n * BackendTraits<FF>::u_squared);
}

// Per-sample producers. Return computed & reference already widened to float128.
using BSampleU = std::function<void(double x, float128& computed, float128& ref)>;
using BSampleB = std::function<void(double a, double b, float128& computed, float128& ref)>;

static GroupBResult run_group_b_unary(const char* name, long N, uint64_t seed,
                                      const InputDist& gen, const BSampleU& fn) {
  std::vector<double> digs; digs.reserve(N);
  std::mt19937_64 g(seed);
  for (long i = 0; i < N; ++i) {
    double x = gen(g);
    float128 c, r; fn(x, c, r);
    digs.push_back(digits_of_accuracy<FF>(c, r));
  }
  AccStats s = compute_stats(digs.data(), (int)digs.size());
  double tol = threshold_digits((long)digs.size());
  bool pass = s.mean >= tol;
  report_line(name, s.n, s.min, s.mean, tol, pass);
  return GroupBResult{name, s.n, s.min, s.mean, tol, pass};
}

static GroupBResult run_group_b_binary(const char* name, long N, uint64_t seed,
                                       const InputDist& ga, const InputDist& gb,
                                       const BSampleB& fn) {
  std::vector<double> digs; digs.reserve(N);
  std::mt19937_64 g(seed);
  for (long i = 0; i < N; ++i) {
    double a = ga(g), b = gb(g);
    float128 c, r; fn(a, b, c, r);
    digs.push_back(digits_of_accuracy<FF>(c, r));
  }
  AccStats s = compute_stats(digs.data(), (int)digs.size());
  double tol = threshold_digits((long)digs.size());
  bool pass = s.mean >= tol;
  report_line(name, s.n, s.min, s.mean, tol, pass);
  return GroupBResult{name, s.n, s.min, s.mean, tol, pass};
}

// Log-uniform generator: 10^u, u ~ Uniform[explo, exphi]. Spans magnitudes so the
// round-trip identities exercise the exponent range, not just [0,1].
static InputDist loguniform(double explo, double exphi) {
  return [explo, exphi](std::mt19937_64& g) {
    std::uniform_real_distribution<double> d(explo, exphi);
    return std::pow(10.0, d(g));
  };
}

// ============================================================================
// Device pass. 3 Group A (A1, A3, A5) + 2 Group B (B1, B4) on 10^5 inputs. A
// generic runner ships one FF result per input back to host; the caller does the
// (bit-exact for A / oracle for B) comparison. Catches a Serial->CUDA regression
// the host pass cannot see. The op receives the raw double input and constructs
// its own FF operand, so A/B can pick single-float vs Route-A as needed.
// ============================================================================
template <typename DeviceOp>
static void device_run(int n, uint64_t seed, const InputDist& gen, DeviceOp op,
                       std::vector<double>& x_out,
                       std::vector<float>& hi_out, std::vector<float>& lo_out) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  std::vector<double> hx(n);
  { std::mt19937_64 g(seed); for (int i = 0; i < n; ++i) hx[i] = gen(g); }

  Kokkos::View<double*, exec_space> dx("dx", n);
  Kokkos::View<float*,  exec_space> dhi("dhi", n), dlo("dlo", n);
  auto hmx = Kokkos::create_mirror_view(dx);
  for (int i = 0; i < n; ++i) hmx(i) = hx[i];
  Kokkos::deep_copy(dx, hmx);

  Kokkos::parallel_for("ff_prop_dev", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      ff::FloatFloat d = op(dx(i));
      dhi(i) = d.hi; dlo(i) = d.lo;
    });
  Kokkos::fence();

  auto hhi = Kokkos::create_mirror_view(dhi);
  auto hlo = Kokkos::create_mirror_view(dlo);
  Kokkos::deep_copy(hhi, dhi);
  Kokkos::deep_copy(hlo, dlo);

  x_out.assign(hx.begin(), hx.end());
  hi_out.resize(n); lo_out.resize(n);
  for (int i = 0; i < n; ++i) { hi_out[i] = hhi(i); lo_out[i] = hlo(i); }
}

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("=== ff_property_test (T2.3): algebraic identities for FF ===\n");
    std::printf("Execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("Group A = bit-exact (no oracle); Group B / Test C = tolerance "
                "(needs __float128 oracle).\n\n");

    long groupA_failures = 0;

    // ------------------------------------------------------------------------
    // GROUP A — bit-exact identities (unconditional).
    // ------------------------------------------------------------------------
    std::printf("[Group A] bit-exact identities (10^6 random + full finite corpus)\n");
    std::vector<GroupAResult> ga;
    const InputDist wide = uniform(-1e8, 1e8);
    uint64_t seed = 12345ULL;

    // A1: a + (-a) == 0.  twoSum of x and -x gives +0.0 exactly.
    ga.push_back(run_group_a_unary("A1_add_neg", seed++, wide,
      [](const ff::FloatFloat& a, ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::add(a, ff::negate(a)); want = ff::FloatFloat(0.0f, 0.0f);
      }));
    // A2: a - a == 0, via operator-.
    ga.push_back(run_group_a_unary("A2_self_sub", seed++, wide,
      [](const ff::FloatFloat& a, ff::FloatFloat& got, ff::FloatFloat& want) {
        got = a - a; want = ff::FloatFloat(0.0f, 0.0f);
      }));
    // A3: a * 1 == a.  Dekker with b=(1,0) leaves both components unchanged for a
    //     non-overlapping pair. (Dekker split domain — see dom_dekker.)
    ga.push_back(run_group_a_unary("A3_mul_one", seed++, wide,
      [](const ff::FloatFloat& a, ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::multiply(a, ff::FloatFloat(1.0f)); want = a;
      }, dom_dekker));
    // A4: a * (-1) == -a.  (Dekker split domain — see dom_dekker.)
    ga.push_back(run_group_a_unary("A4_mul_negone", seed++, wide,
      [](const ff::FloatFloat& a, ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::multiply(a, ff::FloatFloat(-1.0f)); want = ff::negate(a);
      }, dom_dekker));
    // A5: |a| == a when a.hi>=0 else -a. Exercises BOTH sign branches (the wide
    //     symmetric range + corpus +/-0/+/-powers hit a.hi>=0 and a.hi<0).
    ga.push_back(run_group_a_unary("A5_abs_branch", seed++, wide,
      [](const ff::FloatFloat& a, ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::abs(a); want = (a.hi >= 0.0f) ? a : ff::negate(a);
      }));
    // A6: -(-a) == a.
    ga.push_back(run_group_a_unary("A6_double_neg", seed++, wide,
      [](const ff::FloatFloat& a, ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::negate(ff::negate(a)); want = a;
      }));
    // A8: add(a,b) == add(b,a).  Bit-exact for SINGLE-FLOAT operands — see the
    //     classification note in the file header (Knuth twoSum error is exact and
    //     order-independent; the +a.lo+b.lo tail is zero here).
    ga.push_back(run_group_a_binary("A8_add_comm", seed++, wide, uniform(-1e8, 1e8),
      [](const ff::FloatFloat& a, const ff::FloatFloat& b,
         ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::add(a, b); want = ff::add(b, a);
      }));

    for (const auto& r : ga) groupA_failures += r.failures;

    // ------------------------------------------------------------------------
    // GROUP S — IEEE special-value semantics for division (unconditional).
    // ------------------------------------------------------------------------
    std::printf("\n[Group S] IEEE special-value semantics for divide / divide_scalar\n");
    const float kInf    = std::numeric_limits<float>::infinity();
    const float kNaN    = std::numeric_limits<float>::quiet_NaN();
    const float kFltMax = std::numeric_limits<float>::max();
    const float kFltMin = std::numeric_limits<float>::min();       // smallest normal
    const float kDenorm = std::numeric_limits<float>::denorm_min();

    const std::vector<SpecialCase> s_cases = {
      // x / +/-inf = +/-0, sign = sign(x) XOR sign(inf). Was NaN before §4l.
      { 1.0f,     kInf,  SpecialWant::PosZero},
      {-1.0f,     kInf,  SpecialWant::NegZero},
      { 1.0f,    -kInf,  SpecialWant::NegZero},
      {-1.0f,    -kInf,  SpecialWant::PosZero},
      // Sign of a ZERO numerator is IEEE-preserved through division by inf.
      { 0.0f,     kInf,  SpecialWant::PosZero},
      {-0.0f,     kInf,  SpecialWant::NegZero},
      { 0.0f,    -kInf,  SpecialWant::NegZero},
      {-0.0f,    -kInf,  SpecialWant::PosZero},
      // A spread of finite magnitudes, spanning subnormal to FLT_MAX.
      { kDenorm,  kInf,  SpecialWant::PosZero},
      {-kDenorm,  kInf,  SpecialWant::NegZero},
      { kFltMin,  kInf,  SpecialWant::PosZero},
      { 1e-30f,   kInf,  SpecialWant::PosZero},
      { 1.0f,     kInf,  SpecialWant::PosZero},
      { 1e30f,    kInf,  SpecialWant::PosZero},
      { 1e35f,    kInf,  SpecialWant::PosZero},   // numerator inside B10's Zone B band
      { kFltMax,  kInf,  SpecialWant::PosZero},
      { kFltMax, -kInf,  SpecialWant::NegZero},
      // inf / inf is invalid — must stay NaN, in all four sign combinations.
      { kInf,     kInf,  SpecialWant::NaN},
      { kInf,    -kInf,  SpecialWant::NaN},
      {-kInf,     kInf,  SpecialWant::NaN},
      {-kInf,    -kInf,  SpecialWant::NaN},
      // NaN propagates from either side.
      { 1.0f,     kNaN,  SpecialWant::NaN},
      {-1.0f,     kNaN,  SpecialWant::NaN},
      { 0.0f,     kNaN,  SpecialWant::NaN},
      { kNaN,     kInf,  SpecialWant::NaN},
      { kNaN,     kNaN,  SpecialWant::NaN},
      // Divide-by-zero: B10 Zone C, untouched by §4l. Regression guards.
      { 1.0f,     0.0f,  SpecialWant::PosInf},
      {-1.0f,     0.0f,  SpecialWant::NegInf},
      { 1.0f,    -0.0f,  SpecialWant::NegInf},
      {-1.0f,    -0.0f,  SpecialWant::PosInf},
      { 0.0f,     0.0f,  SpecialWant::NaN},
    };

    long groupS_checked = 0;
    const long groupS_failures = run_group_s(s_cases, groupS_checked);
    std::printf("  %-16s n=%-9ld failures=%ld status=%s\n", "S1_div_special",
                groupS_checked, groupS_failures, groupS_failures == 0 ? "PASS" : "FAIL");

    // ------------------------------------------------------------------------
    // GROUP B — approximate identities (tolerance, digits vs oracle).
    // ------------------------------------------------------------------------
    std::printf("\n[Group B] approximate identities (digits of accuracy vs __float128)\n");
    std::vector<GroupBResult> gb;

    // A7 (DEMOTED): multiply(a,b) ~= multiply(b,a). Reference = one ordering,
    // computed = the other; digits measure how close to bit-symmetric the Dekker
    // cross-term sum is. Bound: ~2u^2 (a single reordered add in c21).
    gb.push_back(run_group_b_binary("B0_mul_comm", kRandomN, seed++,
      uniform(-1e6, 1e6), uniform(-1e6, 1e6),
      [](double a, double b, float128& c, float128& r) {
        r = BackendTraits<FF>::to_quad(ff::multiply(ff::FloatFloat(a), ff::FloatFloat(b)));
        c = BackendTraits<FF>::to_quad(ff::multiply(ff::FloatFloat(b), ff::FloatFloat(a)));
      }));

    // B1: sqrt(a)^2 ~= a, a>0 in [1e-30,1e30]. Bound: 4u^2 (sqrt double-word + one
    // double-word square).
    gb.push_back(run_group_b_unary("B1_sqrt_sq", kRandomN, seed++, loguniform(-30, 30),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat s = ff::sqrt(ff::FloatFloat(x));
        c = BackendTraits<FF>::to_quad(ff::multiply(s, s));
        r = (float128)x;
      }));
    // B2: exp(log(a)) ~= a, a>0 in [1e-30,1e30]. DEVIATION from DD's [1e-100,1e100]:
    // ff_math.hpp exp guards a.hi>=88, so ln(a) must stay < 88 => |log10(a)| < 38;
    // [1e-30,1e30] keeps ln(a) in +/-69, well inside. Bound: 10u^2.
    gb.push_back(run_group_b_unary("B2_exp_log", kRandomN, seed++, loguniform(-30, 30),
      [](double x, float128& c, float128& r) {
        c = BackendTraits<FF>::to_quad(ff::exp(ff::log(ff::FloatFloat(x))));
        r = (float128)x;
      }));
    // B3: log(exp(a)) ~= a. Bound: 10u^2.
    // Domain [-79, 79], RESTORED from the temporary [-69, 69] narrowing once B8
    // landed. B4's originally-hypothesized eps root cause was empirically falsified;
    // the real defect was a Dekker splitter overflow in divide() (b.hi * split -> inf,
    // inf - inf = NaN) that poisoned log()'s Newton exp for x ≳ e^79.7. B8 fixes
    // divide() with a scaled splitter, so the "FFEXP: iteration limit" stall no longer
    // fires in this range. The ceiling is 79 (safety-margined one integer under the
    // empirical clean limit of ~79.7); NOT pushed to [-85, 85]. See ff_math.hpp
    // divide() B8 comment + docs/TEST_SUITE_PLAN.md §B4/§B8.
    gb.push_back(run_group_b_unary("B3_log_exp", kRandomN, seed++, uniform(-79.0, 79.0),
      [](double x, float128& c, float128& r) {
        c = BackendTraits<FF>::to_quad(ff::log(ff::exp(ff::FloatFloat(x))));
        r = (float128)x;
      }));
    // B4: sin^2(a) + cos^2(a) ~= 1, a in [-100,100]. Bound: 10u^2. Robust to arg-
    // reduction error (identity of the reduced angle); the min may still dip near
    // a=+/-pi*k and is annotated below.
    gb.push_back(run_group_b_unary("B4_pythag", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat cc, ss; ff::sincos(ff::FloatFloat(x), cc, ss);
        c = BackendTraits<FF>::to_quad(ff::add(ff::multiply(ss, ss), ff::multiply(cc, cc)));
        r = (float128)1.0;
      }));
    // B5: sin(-a) ~= -sin(a), a in [-100,100]. Placed in Group B (not asserted
    // bit-exact) — see header. Bound: 2u^2 (one sign flip; expected near-exact).
    gb.push_back(run_group_b_unary("B5_sin_odd", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        c = BackendTraits<FF>::to_quad(ff::sin(ff::negate(ff::FloatFloat(x))));
        r = BackendTraits<FF>::to_quad(ff::negate(ff::sin(ff::FloatFloat(x))));
      }));
    // B6: cos(-a) ~= cos(a). Bound: 2u^2.
    gb.push_back(run_group_b_unary("B6_cos_even", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        c = BackendTraits<FF>::to_quad(ff::cos(ff::negate(ff::FloatFloat(x))));
        r = BackendTraits<FF>::to_quad(ff::cos(ff::FloatFloat(x)));
      }));
    // B7: tan(a)*cos(a) ~= sin(a), a in [-1.3,1.3] (avoids cos~=0 near +/-pi/2).
    // Bound: 10u^2.
    gb.push_back(run_group_b_unary("B7_tan_cos", kRandomN, seed++, uniform(-1.3, 1.3),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat a(x);
        c = BackendTraits<FF>::to_quad(ff::multiply(ff::tan(a), ff::cos(a)));
        r = BackendTraits<FF>::to_quad(ff::sin(a));
      }));
    // B8: 2*sin(a)*cos(a) ~= sin(2a), a in [-3,3], 10^5 samples. DEVIATION from DD's
    // [-100,100]: sin(2a) and 2*sin*cos reduce DIFFERENT arguments, and FF's double-
    // float arg reduction degrades for large |a| (PORT_NOTES §5); |a|<3 keeps 2a
    // inside ~one period so reduction is clean. Bound: 10u^2.
    gb.push_back(run_group_b_unary("B8_double_angle", 100'000, seed++, uniform(-3.0, 3.0),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat a(x);
        c = BackendTraits<FF>::to_quad(ff::multiply_scalar(ff::multiply(ff::sin(a), ff::cos(a)), 2.0f));
        r = BackendTraits<FF>::to_quad(ff::sin(ff::multiply_scalar(a, 2.0f)));
      }));
    // B9: exp(a)*exp(-a) ~= 1. Bound: 4u^2.
    // Domain [-79, 79], RESTORED from the temporary [-69, 69] narrowing once B8
    // landed (see B3 above for the full B4->B8 root-cause history — the real defect
    // was divide()'s Dekker splitter overflow, fixed by B8's scaled splitter). The
    // ceiling is matched to B3 (one integer under the ~79.7 empirical clean limit);
    // NOT pushed to [-85, 85]. See ff_math.hpp divide() B8 comment +
    // docs/TEST_SUITE_PLAN.md §B4/§B8.
    gb.push_back(run_group_b_unary("B9_exp_prod", kRandomN, seed++, uniform(-79.0, 79.0),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat a(x);
        c = BackendTraits<FF>::to_quad(ff::multiply(ff::exp(a), ff::exp(ff::negate(a))));
        r = (float128)1.0;
      }));
    // B10: hypot(a,b)^2 ~= a^2+b^2, a,b in [-1e3,1e3]. Bound: 10u^2.
    gb.push_back(run_group_b_binary("B10_hypot_sq", kRandomN, seed++,
      uniform(-1e3, 1e3), uniform(-1e3, 1e3),
      [](double a, double b, float128& c, float128& r) {
        ff::FloatFloat da(a), db(b);
        ff::FloatFloat h = ff::hypot(da, db);
        c = BackendTraits<FF>::to_quad(ff::multiply(h, h));
        r = BackendTraits<FF>::to_quad(ff::add(ff::multiply(da, da), ff::multiply(db, db)));
      }));
    // B11: pow(a,2) ~= a*a, a>0 in [1e-15,1e15] (keeps 2*ln(a) inside exp's 88
    // guard: 2*ln(1e15)~=69). Bound: 4u^2 (pow = exp(2*log a) vs one double-word
    // square).
    gb.push_back(run_group_b_unary("B11_pow_two", kRandomN, seed++, loguniform(-15, 15),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat a(x);
        c = BackendTraits<FF>::to_quad(ff::pow(a, ff::FloatFloat(2.0)));
        r = BackendTraits<FF>::to_quad(ff::multiply(a, a));
      }));
    // B12: atanh(a) ~= 0.5*(log(1+a) - log(1-a)), |a|<0.5. Reformulated as an
    // equivalence (not "difference ~= 0") so the digit metric is well-defined.
    // Bound: 10u^2.
    gb.push_back(run_group_b_unary("B12_atanh_log", kRandomN, seed++, uniform(-0.5, 0.5),
      [](double x, float128& c, float128& r) {
        ff::FloatFloat a(x);
        c = BackendTraits<FF>::to_quad(ff::atanh(a));
        ff::FloatFloat lhs = ff::log(ff::add(ff::FloatFloat(1.0), a));
        ff::FloatFloat rhs = ff::log(ff::subtract(ff::FloatFloat(1.0), a));
        r = BackendTraits<FF>::to_quad(ff::multiply_scalar(ff::subtract(lhs, rhs), 0.5f));
      }));

    // B4 min annotation (conditioning near +/-pi*k). The identity is well-behaved
    // so its mean stays high; we surface any low min honestly rather than hiding it.
    {
      const auto* ann = lookup_expected_min_drop("sin");
      const GroupBResult& b4 = *std::find_if(gb.begin(), gb.end(),
          [](const GroupBResult& g){ return g.name == "B4_pythag"; });
      std::printf("    note[B4_pythag]: min_digits=%.2f — dips are argument-reduction "
                  "conditioning near a=+/-pi*k (%s)\n", b4.min_d,
                  ann ? ann->reason : "sin near +/-pi; PORT_NOTES §5");
    }

    // ------------------------------------------------------------------------
    // Device pass: 3 Group A (bit-exact) + 2 Group B (tolerance) on 10^5 inputs.
    // ------------------------------------------------------------------------
    std::printf("\n[Device] 3 Group A (bit-exact) + 2 Group B on %d inputs (%s)\n",
                kDeviceN, Kokkos::DefaultExecutionSpace::name());
    long device_failures = 0;

    // Device A1: a + (-a) == 0.  (Route-A operands; result is exactly 0 regardless
    // of lo.)
    {
      std::vector<double> x; std::vector<float> hi, lo;
      device_run(kDeviceN, 700001ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(double xv){ ff::FloatFloat a(xv); return ff::add(a, ff::negate(a)); },
        x, hi, lo);
      long f = 0; for (int i = 0; i < kDeviceN; ++i) if (!(hi[i] == 0.0f && lo[i] == 0.0f)) ++f;
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A1_add_neg", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }
    // Device A3: a * 1 == a (compare to the Route-A split of the input on host).
    {
      std::vector<double> x; std::vector<float> hi, lo;
      device_run(kDeviceN, 700002ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(double xv){ ff::FloatFloat a(xv); return ff::multiply(a, ff::FloatFloat(1.0f)); },
        x, hi, lo);
      long f = 0;
      for (int i = 0; i < kDeviceN; ++i) {
        ff::FloatFloat a(x[i]);
        if (!(hi[i] == a.hi && lo[i] == a.lo)) ++f;
      }
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A3_mul_one", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }
    // Device A5: |a| == (a.hi>=0 ? a : -a).
    {
      std::vector<double> x; std::vector<float> hi, lo;
      device_run(kDeviceN, 700003ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(double xv){ ff::FloatFloat a(xv); return ff::abs(a); }, x, hi, lo);
      long f = 0;
      for (int i = 0; i < kDeviceN; ++i) {
        ff::FloatFloat a(x[i]);
        ff::FloatFloat w = (a.hi >= 0.0f) ? a : ff::FloatFloat(-a.hi, -a.lo);
        if (!(hi[i] == w.hi && lo[i] == w.lo)) ++f;
      }
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A5_abs_branch", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }

    // Device B1: sqrt(a)^2 ~= a.
    {
      std::vector<double> x; std::vector<float> hi, lo;
      device_run(kDeviceN, 700004ULL, loguniform(-30, 30),
        KOKKOS_LAMBDA(double xv){ ff::FloatFloat a(xv); ff::FloatFloat s = ff::sqrt(a); return ff::multiply(s, s); },
        x, hi, lo);
      std::vector<double> digs(kDeviceN);
      for (int i = 0; i < kDeviceN; ++i)
        digs[i] = digits_of_accuracy<FF>(BackendTraits<FF>::to_quad(ff::FloatFloat(hi[i], lo[i])),
                                         (float128)x[i]);
      AccStats s = compute_stats(digs.data(), kDeviceN);
      double tol = threshold_digits(kDeviceN); bool pass = s.mean >= tol;
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  "B1_sqrt_sq", kDeviceN, s.min, s.mean, tol, pass ? "PASS" : "FAIL");
    }
    // Device B4: sin^2+cos^2 ~= 1.
    {
      std::vector<double> x; std::vector<float> hi, lo;
      device_run(kDeviceN, 700005ULL, uniform(-100.0, 100.0),
        KOKKOS_LAMBDA(double xv){
          ff::FloatFloat a(xv);
          ff::FloatFloat cc, ss; ff::sincos(a, cc, ss);
          return ff::add(ff::multiply(ss, ss), ff::multiply(cc, cc));
        }, x, hi, lo);
      std::vector<double> digs(kDeviceN);
      for (int i = 0; i < kDeviceN; ++i)
        digs[i] = digits_of_accuracy<FF>(BackendTraits<FF>::to_quad(ff::FloatFloat(hi[i], lo[i])),
                                         (float128)1.0);
      AccStats s = compute_stats(digs.data(), kDeviceN);
      double tol = threshold_digits(kDeviceN); bool pass = s.mean >= tol;
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  "B4_pythag", kDeviceN, s.min, s.mean, tol, pass ? "PASS" : "FAIL");
    }

    // ------------------------------------------------------------------------
    // TEST C — named-constant regressions.
    // ------------------------------------------------------------------------
    // Scaled to FF's 14-digit cap: DD used >=30 of 31; FF uses >=12 of 14 (two
    // digits of conditioning headroom for the FP32 transcendental round-trips).
    std::printf("\n[Test C] named-constant regressions (target >=12 digits)\n");
    int c_pass = 0, c_total = 0;
    const double kNamedMin = 12.0;

    auto case_digits = [&](const char* name, float128 computed, float128 ref) {
      ++c_total;
      double d = digits_of_accuracy<FF>(computed, ref);
      bool ok = d >= kNamedMin;
      std::printf("    %-14s digits=%6.2f  %s\n", name, d, ok ? "PASS" : "FAIL");
      if (ok) ++c_pass;
    };

    // C1: sin(pi) ~= 0. digits-of-accuracy against a nonzero ref is ill-defined at
    // ref=0, so measure "digits of zero-ness" = -log10(|sin(pi)|). SOFTENED by the
    // sin conditioning note near +/-pi (lookup_expected_min_drop("sin")): the FF pi
    // constant carries a ~u^2 absolute error, so |sin(pi_ff)| ~= that error and the
    // achievable zero-digits is ~ the 14-digit cap minus reduction loss. Gate at a
    // conditioning-aware floor rather than the full kNamedMin.
    {
      ++c_total;
      ff::FloatFloat sp = ff::sin(ff::FloatFloat_pi());
      float128 v = BackendTraits<FF>::to_quad(sp);
      double zero_digits = (v == (float128)0.0) ? kMaxDig
                           : -(double)q_log10(q_abs(v));
      const auto* ann = lookup_expected_min_drop("sin");
      const double kSinPiFloor = 6.0;   // arg-reduction near pi (PORT_NOTES §5)
      bool ok = zero_digits >= kSinPiFloor;
      std::printf("    %-14s |sin(pi)|=%.3e zero_digits=%6.2f  %s  (softened floor=%.1f; %s)\n",
                  "C1_sin_pi", (double)q_abs(v), zero_digits,
                  ok ? "PASS" : "FAIL", kSinPiFloor, ann ? ann->reason : "sin near pi");
      if (ok) ++c_pass;
    }
    // C2: log(e) ~= 1.
    case_digits("C2_log_e",
                BackendTraits<FF>::to_quad(ff::log(ff::FloatFloat_e())), (float128)1.0);
    // C3: exp(log2) ~= 2.
    case_digits("C3_exp_log2",
                BackendTraits<FF>::to_quad(ff::exp(ff::FloatFloat_log2())), (float128)2.0);
    // C4: sqrt2 * sqrt2 ~= 2.
    case_digits("C4_sqrt2_sq",
                BackendTraits<FF>::to_quad(ff::multiply(ff::FloatFloat_sqrt2(),
                                                        ff::FloatFloat_sqrt2())), (float128)2.0);
    // C5: log(10) ~= the FloatFloat_log10() constant (which stores ln(10)).
    case_digits("C5_log_ten",
                BackendTraits<FF>::to_quad(ff::log(ff::FloatFloat(10.0))),
                BackendTraits<FF>::to_quad(ff::FloatFloat_log10()));
    // C6: euler_gamma / digamma — SKIPPED: no digamma in ff_math.hpp, and the stored
    // euler_gamma constant has no independent FF op to check it against (would be a
    // tautology). Documented, not silently dropped.
    std::printf("    %-14s SKIPPED (no digamma op; constant has no independent "
                "FF oracle in ff_math.hpp)\n", "C6_gamma");

    std::printf("  Test C: %d/%d passed\n", c_pass, c_total);

    // ------------------------------------------------------------------------
    // ANTI-TESTS — identities DELIBERATELY NOT tested, and why.
    // ------------------------------------------------------------------------
    // These are FALSE for every finite-precision floating format, FF included;
    // asserting them would test IEEE arithmetic's known non-properties, not FF.
    //
    //  * Associativity of addition:  add(add(a,b),c) == add(a,add(b,c)).
    //    FP (and FF) addition is COMMUTATIVE but NOT ASSOCIATIVE — the two groupings
    //    round intermediate sums differently. By design, not a bug; T2.4 measures FF
    //    add's accuracy vs the oracle.
    //
    //  * Distributivity across large-magnitude differences:
    //    multiply(a, subtract(b,c)) == subtract(multiply(a,b), multiply(a,c)).
    //    When b and c are close and large, subtract(b,c) cancels to a small exact
    //    value while multiply(a,b) and multiply(a,c) are large and individually
    //    rounded; their difference carries the two roundings and cannot match the LHS
    //    bit-for-bit. Again a property of finite precision, not of FF.
    std::printf("\n[Anti-tests] associativity of add and distributivity across large\n"
                "  cancellations are NOT tested: both are false for any finite-precision\n"
                "  format (rounding is grouping-dependent). See the comment block in main().\n");

    // ------------------------------------------------------------------------
    // Summary + gates.
    // ------------------------------------------------------------------------
    std::printf("\n=== Summary ===\n");
    std::printf("  Group A: %zu identities, total failures=%ld\n", ga.size(), groupA_failures);
    KOKKOS_EP_ASSERT(groupA_failures == 0,
                     "a Group A bit-exact identity did not hold to the last bit");

    std::printf("  Group S: %ld special-value checks, failures=%ld\n",
                groupS_checked, groupS_failures);
    KOKKOS_EP_ASSERT(groupS_failures == 0,
                     "divide/divide_scalar violated an IEEE special-value requirement");

    long groupB_failed = 0; for (const auto& r : gb) if (!r.pass) ++groupB_failed;
    std::printf("  Group B: %zu identities, mean below tolerance=%ld\n", gb.size(), groupB_failed);
    KOKKOS_EP_ASSERT(groupB_failed == 0,
                     "a Group B identity's MEAN digits fell below the -log10(N*u^2) tolerance");
    KOKKOS_EP_ASSERT(c_pass == c_total, "a Test C named-constant regression fell below its floor");

    std::printf("  Device: total failures=%ld\n", device_failures);
    KOKKOS_EP_ASSERT(device_failures == 0, "a device identity check failed");

    rc = ep_exit_code();
    std::printf("\n=== ff_property_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();

  return rc;
}
