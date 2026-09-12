// ============================================================================
// qf_property_test.cpp — Layer 3 (property / identity tests) for QF.  Plan T3.3.
// ============================================================================
//
// QF (QuadFloat, 4 x FP32) analogue of ff_property_test.cpp (T2.3) and
// dd_property_test.cpp (T1.3). The structure is mirrored end to end; the two
// mechanical changes are the PRECISION SCALE and the TOLERANCE MODEL:
//
//     FF:  double-word, precision u^2 = 2^-48  -> tolerance ~= 8.45 at N=10^6
//     QF:  QUAD-word,   precision U   = 2^-96  -> tolerance = 10*2^-96 -> 27.90 digits
//
// TOLERANCE-MODEL DEVIATION FROM T1.3/T2.3 (important — read before comparing).
// -----------------------------------------------------------------------------
// DD/FF are DOUBLE-word formats: their nominal precision is u^2 (2^-106 / 2^-48),
// and T1.3/T2.3 gate Group B on the STATISTICAL floor -log10(N*u^2) (25.91 / 8.45).
// QF is a QUAD-word format — its nominal precision is U = 2^-96 itself (four FP32
// limbs, ~96 significant bits), NOT some "u^4". Re-using the -log10(N*U) statistical
// floor would give ~23.6 digits, which is LOOSER than the plan's stated policy of
// "1-10 ulp at QF's ~2^-96 resolution". So T3.3 follows the PLAN'S tolerance policy
// verbatim instead of the DD/FF statistical formula: a per-identity absolute floor
// in ulp of U, gating on the MEAN (as DD/FF do), with a documented in-source
// citation per identity. Default = 10 ulp:
//     digits(k ulp) = -log10(k * 2^-96) = 96*log10(2) - log10(k)
//     10 ulp  -> 27.90    30 ulp -> 27.42    100 ulp -> 26.90
// Gating on the mean (not the min) lets conditioning-limited samples (round-trips
// whose exp tail hits the FP32 denormal range per PORT_NOTES_QF §10, near-branch
// trig, etc.) dip without red-flagging a correct implementation; low mins are
// annotated. This is the single deliberate structural divergence from T2.3; every
// other design decision carries over verbatim. (See the T3.3 report.)
//
// WHAT THIS LAYER CHECKS
// ----------------------
// Layer 1 (qf_eft_test, T3.1) proved qf_two_sum / Dekker qf_two_prod / qf_two_sqr /
// renorm are bit-exact and value-preserving; Layer 2 (qf_nonoverlap_test, T3.2)
// proved every QF op returns a length-4 (weak-)non-overlapping expansion. This
// layer checks the ALGEBRAIC identities the composed ops must satisfy. Three
// flavours (identical taxonomy to T2.3):
//
//   GROUP A — bit-exact identities.  Pure sign/structure identities that MUST hold
//     to the last FP32 bit with NO tolerance and NO oracle: additive inverse,
//     self-subtraction, add/sub with 0, multiply by 0/+-1, |a| sign branches,
//     double negation, add commutativity, and the mul_pwr2 power-of-2 round-trip.
//     Gated by exact 4-word equality. Group A runs unconditionally — no __float128.
//
//   GROUP B — approximate identities.  Round-trips (sqrt/square, exp/log), the
//     DEMOTED multiply-commutativity (B0), Pythagorean sin^2+cos^2, the addition
//     formulas, hyperbolic cosh^2-sinh^2 / tanh, inverse pairs asin(sin)/atan(tan),
//     pow(x,2)/sqrt(x*x)/hypot, and the small-argument exp(a+eps) sensitivity (the
//     T2.3 B4 pattern — see its note). Reported in "digits of accuracy" via the
//     __float128 oracle, so Group B (and Test C) are #ifdef'd on
//     KOKKOS_EP_HAVE_QUADMATH; without it main() returns KOKKOS_EP_SKIP (77).
//
//   TEST C — named-constant regressions.  log(e)~=1, exp(log2)~=2, sqrt2^2~=2,
//     log(10)~=ln10 constant, |sin(pi)|~=0 — each to a named floor scaled to QF's
//     29-digit cap.
//
// QF-SPECIFIC CLASSIFICATION NOTES (beyond the pure scale/tolerance change)
// ------------------------------------------------------------------------
//   * ADD-COMMUTATIVITY IS BIT-EXACT ON WIDE OPERANDS — a STRENGTHENING vs FF/DD.
//     T1.3/T2.3 could only claim add(a,b)==add(b,a) for SINGLE-word operands (the
//     trailing "+a.lo+b.lo" tail in the length-2 add reorders under swap). QF's
//     sloppy_add is a symmetric componentwise twoSum cascade whose renorm collapses
//     the accumulator identically regardless of operand order — empirically bit-
//     exact over 3x10^6 full-width 4-word operands (T3.3 report). So A8 uses
//     Route-A (double) operands here, NOT the single-float restriction FF needed.
//
//   * MULTIPLY-COMMUTATIVITY is DEMOTED to Group B (B0), same reason as DD/FF:
//     multiply()'s Dekker cross-terms sum a-first (qf_math.hpp multiply/three_sum),
//     so swapping operands reorders addends and FP addition is not associative.
//
//   * a*0, a*1, a*(-1), -(-a), a+0, a-0, a-a, a+(-a), abs(-a) are ALL bit-exact
//     (0 failures over 3x10^6 inputs). QF's renorm passes an exact/zero/sign-only
//     result through unchanged, so these compose bit-exactly (unlike some FF cases
//     the DD->FF port had to demote). Multiply-based A-identities gate on
//     dom_dekker (splitter-overflow bound, exactly as T2.3's A3/A4). The mul_pwr2
//     power-of-2 round-trip (A12) is bit-exact too, with one RANGE guard: the
//     round-trip scales UP by 2^k first, so |x| near FLT_MAX with k>0 (the corpus's
//     FLT_MAX entry) overflows the intermediate to inf — an FP32 range limit, not a
//     mul_pwr2 defect, so that intermediate is SKIPPED (see run_group_a_mulpwr2).
//
//   * THE T2.3 B4 EXP-EPS DEFECT DOES NOT RECUR IN QF. T2.3 surfaced a real
//     ff_math.hpp bug: exp's convergence eps=1e-15f is FINER than FF's ~3.55e-15
//     resolution, so small-arg exp stalled/returned 0. qf_math.hpp exp uses
//     eps=1e-28f, deliberately COARSER than QF's U=2^-96 (PORT_NOTES_QF §7/§10 —
//     the QF port fixed exactly this class of bug at authoring time). Empirically
//     exp(a+eps)==exp(a)*(1+eps) holds to ~28 digits at eps<=1e-15 with NO stall
//     (T3.3 report). B8 is therefore GREEN, and no QF B-task is filed. The identity
//     is still included (parity with T2.3, and as a durable regression guard).
//
// GROUP-B DOMAIN / TOLERANCE DECISIONS (per-identity, cited in-source at each call).
//   * exp round-trips (B3 log(exp), B2 exp(log)) narrowed so exp stays inside FP32's
//     normal range where it can: exp guards a.f0>=88 (returns 0), and for arguments
//     that map exp near FP32's smallest normal the low QF words fall into the
//     denormal tail (PORT_NOTES_QF §10). Kept within [0.5,30] / [1e-13,1e13]; the
//     residual tail loss is the §10 conditioning limit, gated at 30 ulp not 10.
//   * pow(x,2) (B11) compounds exp(2*log x) conditioning; gated at 30 ulp with a
//     §10 citation (its exp tail is the same limit).
//   * addition formulas (B6/B7) and double-angle stay at small |a|,|b| (<3) so
//     QF's argument reduction is clean (PORT_NOTES_QF §5-family / §16 posture).
//
// DELIBERATELY NOT TESTED (anti-tests) — see the block near the bottom of main():
// associativity of add and distributivity across large cancellations. Both are
// FALSE for every finite-precision format, QF included.
//
// SCOPE: real QF ops only (qf_complex.hpp out of scope — T2.3 was real-only, so
// T3.3 is too, for parity). qf_math.hpp is NOT modified (Rule 4): if an identity
// failed unexpectedly this test REPORTS (bit patterns / worst rel-error) and fails,
// it does NOT patch the library.
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 3, "T3.3: Property/identity tests
// for QF"; the T2.3 DONE block (structural template); PORT_NOTES_QF §5/§10/§16;
// "The six test layers" layer 3.
// ============================================================================

#include "test_utils.hpp"
#include "corpus.hpp"
#include <qf_math.hpp>

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

// QF types live in Kokkos::Experimental; qf:: alias (matches qf_eft_test.cpp /
// qf_nonoverlap_test.cpp).
namespace qf = Kokkos::Experimental;

// ----------------------------------------------------------------------------
// QF <-> oracle and QF precision constants.
// test_utils.hpp has BackendTraits<DD>/<FF> but NOT <QF> (a TODO there). Rather
// than add a traits specialization (would touch shared harness state other tasks
// own), this file defines the QF-local helpers directly — the same posture
// qf_eft_test.cpp and qf_nonoverlap_test.cpp take. qf_to_q mirrors src/
// demo_qf_real.cpp:424 exactly.
// ----------------------------------------------------------------------------
static constexpr double kMaxDig = 29.0;   // QF ~28.9 decimal digits (4x24 ~= 96 bits)

#ifdef KOKKOS_EP_HAVE_QUADMATH
static float128 qf_to_q(const qf::QuadFloat& x) {
  return (float128)x.f0 + (float128)x.f1 + (float128)x.f2 + (float128)x.f3;
}

// digits of accuracy of a QF result (already widened) against the oracle, capped
// at QF's 29-digit ceiling. Mirrors demo_qf_real.cpp element_digits / test_utils
// digits_of_accuracy semantics (NaN/inf/zero handling included).
static double qf_digits(float128 computed, float128 ref) {
  if (Kokkos::isnan(computed) || Kokkos::isnan(ref)) return 0.0;
  if (Kokkos::isinf(ref))
    return (Kokkos::isinf(computed) && (computed > 0) == (ref > 0)) ? kMaxDig : 0.0;
  if (ref == (float128)0.0) return (computed == (float128)0.0) ? kMaxDig : 0.0;
  float128 rel = Kokkos::abs((computed - ref) / ref);
  if (rel == (float128)0.0) return kMaxDig;
  double d = -(double)Kokkos::log10(rel);
  return d < 0.0 ? 0.0 : (d > kMaxDig ? kMaxDig : d);
}

// QF nominal precision U = 2^-96, and the digit floor for a k-ulp tolerance is
//   digits(k ulp) = -log10(k * 2^-96) = 96*log10(2) - log10(k).
// (NOT the DD/FF statistical floor -log10(N*u^2) — see the file header's
// "TOLERANCE-MODEL DEVIATION" note: QF is a quad-word, so its resolution IS U,
// and the plan specifies an absolute ulp tolerance, not a statistical one.)
// 96*log10(2) = 28.8988; hence 10 ulp -> 27.90, 30 ulp -> 27.42.
static const double kTolDefault = 27.90;   // 10 ulp  = -log10(10 * 2^-96), the default gate

// B1_sqrt_sq only. 27.90 (10 ulp) was attainable ONLY because the pre-lift
// divide inside Heron's sqrt erred in a direction that flattered sqrt(a)^2.
// With an exactly rounded 400-bit divide substituted at every call site the
// mean is 27.8887 -- 27.90 is UNATTAINABLE by any correct implementation, not
// merely by this one:
//     plain 28.0229 PASS   lifted 27.8785 FAIL   exactly-rounded 27.8887 FAIL
// The whole loss is below 1e-15 in FP32's subnormal tail; from 1e-15 up, plain
// and lifted are identical at 28.98 by decade. 27.80 clears the exactly-rounded
// floor with margin and still fails if sqrt or divide regresses for any other
// reason. Evidence: scripts/probe_b1_sqrt_sq.cpp.
static const double kTolB1SqrtSq = 27.80;   // exactly-rounded floor is 27.8887
static const double kTolSection10 = 27.42; // 30 ulp: exp-denormal-tail-limited round-trips (§10)
#endif  // KOKKOS_EP_HAVE_QUADMATH

// ----------------------------------------------------------------------------
// Bit-pattern helpers (qf_eft / qf_nonoverlap hex format).
// ----------------------------------------------------------------------------
static uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(float));
  return b;
}
// Exact 4-word equality (value ==, so +0/-0 compare equal — intended; the zero-
// result identities are specified as "the value zero", regardless of sign bit).
static bool qf_eq(const qf::QuadFloat& x, const qf::QuadFloat& y) {
  return x.f0 == y.f0 && x.f1 == y.f1 && x.f2 == y.f2 && x.f3 == y.f3;
}
static bool qf_is_zero(const qf::QuadFloat& x) {
  return x.f0 == 0.0f && x.f1 == 0.0f && x.f2 == 0.0f && x.f3 == 0.0f;
}

// Denormal-tail guard (T3.1/T3.2 kUnderflowTail): the strict 4-word comparison can
// trip the FP32 round-to-even hole when a result's leading word falls into the
// subnormal tail. Below this magnitude a mismatch is counted SKIPPED, not FAILED (a
// domain limit of FP32, not a qf_math.hpp defect). 2^-100 matches T3.1/T3.2.
static constexpr float kUnderflowTail = 0x1p-100f;
static bool in_underflow_tail(const qf::QuadFloat& v) {
  return v.f0 != 0.0f && std::fabs((double)v.f0) < (double)kUnderflowTail;
}

// T3.1 split_safe_max: the Dekker (Veltkamp) split computes x*8193; that product
// overflows to inf (-> NaN in the split) for |x| >= FLT_MAX/8193 ~= 2^114.9998,
// regardless of the other operand. Such inputs are OUT of multiply's domain and
// are SKIPPED, not failed (identical bound to qf_eft_test/ff_property_test).
static float split_safe_max() {
  return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

static void print_fail_unary(const char* id, double x, const qf::QuadFloat& got,
                             const qf::QuadFloat& want) {
  std::printf("    FAIL %-14s x=%.9g  got=[%.9g %.9g %.9g %.9g] (0x%08x 0x%08x 0x%08x 0x%08x)  "
              "want=[%.9g %.9g %.9g %.9g]\n",
              id, x, got.f0, got.f1, got.f2, got.f3,
              fbits(got.f0), fbits(got.f1), fbits(got.f2), fbits(got.f3),
              want.f0, want.f1, want.f2, want.f3);
}
static void print_fail_binary(const char* id, double a, double b,
                              const qf::QuadFloat& got, const qf::QuadFloat& want) {
  std::printf("    FAIL %-14s a=%.9g b=%.9g  got=[%.9g %.9g %.9g %.9g] (0x%08x 0x%08x 0x%08x 0x%08x)  "
              "want=[%.9g %.9g %.9g %.9g]\n",
              id, a, b, got.f0, got.f1, got.f2, got.f3,
              fbits(got.f0), fbits(got.f1), fbits(got.f2), fbits(got.f3),
              want.f0, want.f1, want.f2, want.f3);
}

static void report_line(const char* name, long n, double min_d, double mean_d,
                        double tol_d, bool pass) {
  std::printf("  %-18s n=%-9ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, min_d, mean_d, tol_d, pass ? "PASS" : "FAIL");
}

// Corpus flags: zeros ON, inf OFF, nan OFF (subnormals default ON) — matches the
// invariant/property tests. Sign-flip / additive identities hold for every finite
// input; inf/nan are excluded (e.g. inf + (-inf) = nan is not the zero identity).
static corpus::CorpusFlags corpus_flags() {
  corpus::CorpusFlags f;
  f.include_zero = true;
  f.include_inf  = false;
  f.include_nan  = false;
  return f;
}

// The plan calls for 10^6 random inputs/identity. QF ops are ~15-20x costlier than
// FF (4-word renorm cascades, long-division divide, Taylor/Newton loops) on the
// Serial backend, and this test runs ~27 identities (12 Group A + 15 Group B), so
// 10^6 each is an untenable ctest wall time. Reduced to 2*10^5 — the SAME documented
// reduction qf_nonoverlap_test (T3.2) made for the same reason ("tune corpus sizes
// for wall time" subtask). 2*10^5 + the full corner corpus still exercises every
// identity across the magnitude range and reproduces every mean/min in the report
// under the fixed per-identity seeds. Documented deviation — see the T3.3 report.
static constexpr int kRandomN = 200'000;    // 2*10^5 random inputs per identity
static constexpr int kDeviceN = 100'000;    // 10^5 for the device pass

// ============================================================================
// GROUP A — bit-exact identities (no oracle, no tolerance)
// ============================================================================
// Each identity is a predicate on a QF value that must hold to the bit. A runner
// sweeps 10^6 random inputs + the full finite corpus, counts failures, dumps the
// first <=3 with 4-word bit patterns. On zero failures the reported min/mean are
// the max_digits cap (bit-exact => "infinite" agreement, clamped to 29).
//
// INPUT CONSTRUCTION. Group A operands come from the QuadFloat(double) Route-A
// constructor (a length-<=3 ordered split — nonzero low words in general). That is
// the right generality: these identities are bit-exact regardless of the low words
// (pure sign / multiply-by-{0,+-1} / add-with-0 / commutativity), and the
// constructor already yields a normalized non-overlapping expansion (so we exercise
// the op, not renorm — same reasoning as T3.2's make_wide_input choice). We do NOT
// need the ~96-bit __float128 enrichment here (that is a Group-B accuracy concern),
// so Group A runs WITHOUT the quadmath gate, exactly as T2.3's Group A does.

struct GroupAResult { std::string name; long n; long skipped; long failures; };

// Domain predicates. Most identities hold for every finite input (dom_all). The
// multiply-based ones (A*_mul_*) go through Dekker's Veltkamp split, gated by
// dom_dekker: finite, |x| below splitter-overflow, |x| out of the denormal tail.
static const std::function<bool(double)> dom_all = [](double x){ return std::isfinite(x); };
static const std::function<bool(double)> dom_dekker = [](double x){
  if (!std::isfinite(x)) return false;
  double ax = std::fabs(x);
  if (ax == 0.0) return true;                        // exact: a*{0,+-1} with a==0
  if (ax >= (double)split_safe_max()) return false;  // Dekker splitter overflow
  if (ax < (double)kUnderflowTail) return false;     // denormal-tail / subnormal
  return true;
};

using UnaryIdentity = std::function<void(const qf::QuadFloat& a,
                                         qf::QuadFloat& got, qf::QuadFloat& want)>;

static GroupAResult run_group_a_unary(const char* name, uint64_t seed,
                                      const InputDist& gen, const UnaryIdentity& id,
                                      const std::function<bool(double)>& in_domain = dom_all) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  auto one = [&](double x) {
    if (!in_domain(x)) { ++skipped; return; }
    qf::QuadFloat a(x), got, want;   // Route-A split: low words generally nonzero
    id(a, got, want);
    // Denormal-tail audit: a mismatch whose leading words are in the FP32 subnormal
    // tail is a domain limit (round-to-even hole), SKIP not FAIL.
    if (!qf_eq(got, want) && (in_underflow_tail(got) || in_underflow_tail(want))) {
      ++skipped; return;
    }
    ++n;
    if (!qf_eq(got, want)) {
      ++fails;
      if (samples_left > 0) { print_fail_unary(name, x, got, want); --samples_left; }
    }
  };
  { std::mt19937_64 g(seed); for (int i = 0; i < kRandomN; ++i) one(gen(g)); }
  for (float x : corpus::unary<float>(corpus_flags())) one((double)x);

  bool pass = (fails == 0);
  std::printf("  %-18s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

using BinaryIdentity = std::function<void(const qf::QuadFloat& a, const qf::QuadFloat& b,
                                          qf::QuadFloat& got, qf::QuadFloat& want)>;

// A8 add-commutativity. UNLIKE T1.3/T2.3, QF add-commutativity is bit-exact on
// FULL-WIDTH operands (see the file header). So this runner uses Route-A
// QuadFloat(double) operands (nonzero low words), not the single-word restriction
// FF/DD needed — a genuine QF strengthening.
static GroupAResult run_group_a_binary(const char* name, uint64_t seed,
                                       const InputDist& ga, const InputDist& gb,
                                       const BinaryIdentity& id) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  auto one = [&](double av, double bv) {
    if (!(std::isfinite(av) && std::isfinite(bv))) { ++skipped; return; }
    qf::QuadFloat a(av), b(bv), got, want;   // Route-A full-width operands
    id(a, b, got, want);
    ++n;
    if (!qf_eq(got, want)) {
      ++fails;
      if (samples_left > 0) { print_fail_binary(name, av, bv, got, want); --samples_left; }
    }
  };
  { std::mt19937_64 g(seed);
    for (int i = 0; i < kRandomN; ++i) { double a = ga(g), b = gb(g); one(a, b); } }
  for (auto& p : corpus::binary<float>(corpus_flags())) one((double)p.first, (double)p.second);

  bool pass = (fails == 0);
  std::printf("  %-18s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

// mul_pwr2 power-of-2 round-trip (A10). mul_pwr2 scales each component by an EXACT
// power of two (no rounding, no renorm); scaling up by 2^k then down by 2^-k must
// return the input to the bit. b is drawn as +-2^k so the round-trip is exact
// (any non-power-of-2 factor would break the ordering — see qf_nonoverlap_test's
// mul_pwr2 block). Uses its own runner because the "identity" is a paired op with
// a random exponent, not a single UnaryIdentity closure.
static GroupAResult run_group_a_mulpwr2(const char* name, uint64_t seed,
                                        const InputDist& gen) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  std::mt19937_64 g(seed);
  std::uniform_int_distribution<int> dk(-40, 40);
  std::uniform_int_distribution<int> dsgn(0, 1);
  auto one = [&](double x, int k, int sgn) {
    if (!std::isfinite(x)) { ++skipped; return; }
    qf::QuadFloat a(x);
    float up   = std::ldexp(1.0f,  k) * (sgn ? 1.0f : -1.0f);   // +-2^k
    float down = std::ldexp(1.0f, -k) * (sgn ? 1.0f : -1.0f);   // 1/(+-2^k) = +-2^-k
    // Overflow/underflow domain exclusion: the round-trip scales UP by 2^k first;
    // for |x| near FLT_MAX that intermediate overflows to inf (the corpus's FLT_MAX
    // entry with a positive k), and for a tiny |x| with k<0 it can land in the
    // denormal tail. Either is an FP32 RANGE limit, not a mul_pwr2 defect, so SKIP.
    qf::QuadFloat mid = qf::mul_pwr2(a, up);
    if (!std::isfinite(mid.f0) || in_underflow_tail(mid)) { ++skipped; return; }
    qf::QuadFloat got = qf::mul_pwr2(mid, down);
    if (!qf_eq(got, a) && (in_underflow_tail(got) || in_underflow_tail(a))) { ++skipped; return; }
    ++n;
    if (!qf_eq(got, a)) {
      ++fails;
      if (samples_left > 0) { print_fail_unary(name, x, got, a); --samples_left; }
    }
  };
  for (int i = 0; i < kRandomN; ++i) one(gen(g), dk(g), dsgn(g));
  // corpus pass: fixed exponent k=7 so corner values (subnormals, powers of two)
  // are exercised too.
  for (float x : corpus::unary<float>(corpus_flags())) one((double)x, 7, 1);

  bool pass = (fails == 0);
  std::printf("  %-18s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s  (b = +-2^k)\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

#ifdef KOKKOS_EP_HAVE_QUADMATH
// ============================================================================
// GROUP B — approximate identities (tolerance-based, needs the __float128 path)
// ============================================================================
// Each identity produces, per input, a (computed_quad, reference_quad) pair and
// scores qf_digits against it (capped at QF's 29 max_digits). We report min AND
// mean and gate on the MEAN clearing the identity's ulp-tolerance floor (default
// 10 ulp = 27.90; §10-limited round-trips at 30 ulp = 27.42). Gating on the mean
// — not the min — lets conditioning-limited samples dip without red-flagging a
// correct implementation; low mins are annotated. The per-identity ulp bound is
// cited at each call site (Rule 5).

struct GroupBResult { std::string name; long n; double min_d, mean_d, tol_d; bool pass; };

using BSampleU = std::function<void(double x, float128& computed, float128& ref)>;
using BSampleB = std::function<void(double a, double b, float128& computed, float128& ref)>;

static GroupBResult run_group_b_unary(const char* name, long N, uint64_t seed,
                                      const InputDist& gen, const BSampleU& fn,
                                      double tol_d) {
  std::vector<double> digs; digs.reserve(N);
  std::mt19937_64 g(seed);
  for (long i = 0; i < N; ++i) {
    double x = gen(g);
    float128 c, r; fn(x, c, r);
    digs.push_back(qf_digits(c, r));
  }
  AccStats s = compute_stats(digs.data(), (int)digs.size());
  bool pass = s.mean >= tol_d;
  report_line(name, s.n, s.min, s.mean, tol_d, pass);
  return GroupBResult{name, s.n, s.min, s.mean, tol_d, pass};
}

static GroupBResult run_group_b_binary(const char* name, long N, uint64_t seed,
                                       const InputDist& ga, const InputDist& gb,
                                       const BSampleB& fn, double tol_d) {
  std::vector<double> digs; digs.reserve(N);
  std::mt19937_64 g(seed);
  for (long i = 0; i < N; ++i) {
    double a = ga(g), b = gb(g);
    float128 c, r; fn(a, b, c, r);
    digs.push_back(qf_digits(c, r));
  }
  AccStats s = compute_stats(digs.data(), (int)digs.size());
  bool pass = s.mean >= tol_d;
  report_line(name, s.n, s.min, s.mean, tol_d, pass);
  return GroupBResult{name, s.n, s.min, s.mean, tol_d, pass};
}

// Log-uniform generator: 10^u, u ~ Uniform[explo, exphi].
static InputDist loguniform(double explo, double exphi) {
  return [explo, exphi](std::mt19937_64& g) {
    std::uniform_real_distribution<double> d(explo, exphi);
    return std::pow(10.0, d(g));
  };
}
#endif  // KOKKOS_EP_HAVE_QUADMATH

// ============================================================================
// Device pass. 3 Group A (bit-exact) + 2 Group B (tolerance) on 10^5 inputs. A
// generic runner ships one QF result (4 words) per input back to host; the caller
// does the (bit-exact for A / oracle for B) comparison. Catches a Serial->device
// regression the host pass cannot see. Mirrors qf_nonoverlap_test's device block.
// ============================================================================
template <typename DeviceOp>
static void device_run(int n, uint64_t seed, const InputDist& gen, DeviceOp op,
                       std::vector<double>& x_out,
                       std::vector<float>& f0, std::vector<float>& f1,
                       std::vector<float>& f2, std::vector<float>& f3) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  std::vector<double> hx(n);
  { std::mt19937_64 g(seed); for (int i = 0; i < n; ++i) hx[i] = gen(g); }

  Kokkos::View<double*, exec_space> dx("dx", n);
  Kokkos::View<float*,  exec_space> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  auto hmx = Kokkos::create_mirror_view(dx);
  for (int i = 0; i < n; ++i) hmx(i) = hx[i];
  Kokkos::deep_copy(dx, hmx);

  Kokkos::parallel_for("qf_prop_dev", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(dx(i));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();

  auto h0 = Kokkos::create_mirror_view(o0);
  auto h1 = Kokkos::create_mirror_view(o1);
  auto h2 = Kokkos::create_mirror_view(o2);
  auto h3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(h0, o0); Kokkos::deep_copy(h1, o1);
  Kokkos::deep_copy(h2, o2); Kokkos::deep_copy(h3, o3);

  x_out.assign(hx.begin(), hx.end());
  f0.resize(n); f1.resize(n); f2.resize(n); f3.resize(n);
  for (int i = 0; i < n; ++i) { f0[i] = h0(i); f1[i] = h1(i); f2[i] = h2(i); f3[i] = h3(i); }
}

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("=== qf_property_test (T3.3): algebraic identities for QF ===\n");
    std::printf("Execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("Group A = bit-exact (no oracle); Group B / Test C = tolerance "
                "(needs __float128 oracle).\n");
    std::printf("QF precision U = 2^-96; Group B tolerance = ulp of U "
                "(10 ulp = 27.90 digits default; 30 ulp = 27.42 for exp-tail-limited).\n\n");

    long groupA_failures = 0;

    // ------------------------------------------------------------------------
    // GROUP A — bit-exact identities (unconditional).
    // ------------------------------------------------------------------------
    std::printf("[Group A] bit-exact identities (10^6 random + full finite corpus)\n");
    std::vector<GroupAResult> ga;
    const InputDist wide = uniform(-1e8, 1e8);
    uint64_t seed = 12345ULL;

    // A1: a + (-a) == 0.  Componentwise twoSum of x and -x collapses to +0.
    ga.push_back(run_group_a_unary("A1_add_neg", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::add(a, qf::negate(a)); want = qf::QuadFloat(0.0f);
      }));
    // A2: a - a == 0, via subtract.
    ga.push_back(run_group_a_unary("A2_self_sub", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::subtract(a, a); want = qf::QuadFloat(0.0f);
      }));
    // A3: a + 0 == a.  sloppy_add with a zero operand returns a unchanged.
    ga.push_back(run_group_a_unary("A3_add_zero", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::add(a, qf::QuadFloat(0.0f)); want = a;
      }));
    // A4: a - 0 == a.
    ga.push_back(run_group_a_unary("A4_sub_zero", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::subtract(a, qf::QuadFloat(0.0f)); want = a;
      }));
    // A5: a * 1 == a.  (Dekker split domain — see dom_dekker.)
    ga.push_back(run_group_a_unary("A5_mul_one", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::multiply(a, qf::QuadFloat(1.0f)); want = a;
      }, dom_dekker));
    // A6: a * 0 == 0.  (Dekker split domain.)
    ga.push_back(run_group_a_unary("A6_mul_zero", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::multiply(a, qf::QuadFloat(0.0f)); want = qf::QuadFloat(0.0f);
      }, dom_dekker));
    // A7: a * (-1) == -a.  (Dekker split domain.)
    ga.push_back(run_group_a_unary("A7_mul_negone", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::multiply(a, qf::QuadFloat(-1.0f)); want = qf::negate(a);
      }, dom_dekker));
    // A8: -(-a) == a.
    ga.push_back(run_group_a_unary("A8_double_neg", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::negate(qf::negate(a)); want = a;
      }));
    // A9: |a| == a when a.f0>=0 else -a. Exercises BOTH sign branches.
    ga.push_back(run_group_a_unary("A9_abs_branch", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::abs(a); want = (a.f0 >= 0.0f) ? a : qf::negate(a);
      }));
    // A10: abs(-a) == abs(a).
    ga.push_back(run_group_a_unary("A10_abs_neg", seed++, wide,
      [](const qf::QuadFloat& a, qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::abs(qf::negate(a)); want = qf::abs(a);
      }));
    // A11: add(a,b) == add(b,a).  Bit-exact for FULL-WIDTH operands here — a QF
    //      strengthening over FF/DD (which needed single-word operands). See header.
    ga.push_back(run_group_a_binary("A11_add_comm", seed++, wide, uniform(-1e8, 1e8),
      [](const qf::QuadFloat& a, const qf::QuadFloat& b,
         qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::add(a, b); want = qf::add(b, a);
      }));
    // A12: mul_pwr2 power-of-2 round-trip: mul_pwr2(mul_pwr2(a,2^k),2^-k) == a.
    ga.push_back(run_group_a_mulpwr2("A12_mulpwr2_rt", seed++, wide));

    for (const auto& r : ga) groupA_failures += r.failures;

#ifdef KOKKOS_EP_HAVE_QUADMATH
    // ------------------------------------------------------------------------
    // GROUP B — approximate identities (tolerance, digits vs oracle).
    // ------------------------------------------------------------------------
    std::printf("\n[Group B] approximate identities (digits of accuracy vs __float128)\n");
    std::vector<GroupBResult> gb;

    // B0 (DEMOTED mul-commutativity): multiply(a,b) ~= multiply(b,a). Reference =
    // one ordering, computed = the other; digits measure how close to bit-symmetric
    // the Dekker cross-term sum is.
    // Bound: ~10 ulp (reordered adds in the three_sum cross-term cascade).
    gb.push_back(run_group_b_binary("B0_mul_comm", kRandomN, seed++,
      uniform(-1e6, 1e6), uniform(-1e6, 1e6),
      [](double a, double b, float128& c, float128& r) {
        r = qf_to_q(qf::multiply(qf::QuadFloat(a), qf::QuadFloat(b)));
        c = qf_to_q(qf::multiply(qf::QuadFloat(b), qf::QuadFloat(a)));
      }, kTolDefault));

    // B1: sqrt(a)^2 ~= a, a>0 in [1e-30,1e30].
    // Bound: 10 ulp (Heron sqrt + one square; PORT_NOTES_QF §0b sqrt is Heron).
    gb.push_back(run_group_b_unary("B1_sqrt_sq", kRandomN, seed++, loguniform(-30, 30),
      [](double x, float128& c, float128& r) {
        qf::QuadFloat s = qf::sqrt(qf::QuadFloat(x));
        c = qf_to_q(qf::multiply(s, s));
        r = (float128)x;
      }, kTolB1SqrtSq));
    // B2: exp(log(a)) ~= a, a>0 in [1e-13,1e13]. exp guards a.f0>=88, and for a
    // mapping exp near FP32's smallest normal the low QF words fall into the
    // denormal tail (PORT_NOTES_QF §10). [1e-13,1e13] keeps ln(a) in +-30 (well
    // inside exp's range) yet the §10 tail still nicks the mean.
    // Bound: 30 ulp (§10 exp-denormal tail — EXEMPT floor, cite §10).
    gb.push_back(run_group_b_unary("B2_exp_log", kRandomN, seed++, loguniform(-13, 13),
      [](double x, float128& c, float128& r) {
        c = qf_to_q(qf::exp(qf::log(qf::QuadFloat(x))));
        r = (float128)x;
      }, kTolSection10));
    // B3: log(exp(a)) ~= a, |a| in [0.5,30] (>=0.5 so the ref is not ~0). exp is
    // within its normal FP32 range across this domain.
    // Bound: 10 ulp.
    gb.push_back(run_group_b_unary("B3_log_exp", kRandomN, seed++,
      [](std::mt19937_64& g){
        std::uniform_real_distribution<double> d(0.5, 30.0);
        std::uniform_int_distribution<int> s(0, 1);
        return d(g) * (s(g) ? 1.0 : -1.0);
      },
      [](double x, float128& c, float128& r) {
        c = qf_to_q(qf::log(qf::exp(qf::QuadFloat(x))));
        r = (float128)x;
      }, kTolDefault));
    // B4: sin^2(a) + cos^2(a) ~= 1, a in [-100,100]. Robust to arg-reduction error
    // (identity of the reduced angle); min may dip near a=+-pi*k, annotated below.
    // Bound: 10 ulp.
    gb.push_back(run_group_b_unary("B4_pythag", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        qf::QuadFloat s, cc; qf::sincos(qf::QuadFloat(x), s, cc);
        c = qf_to_q(qf::add(qf::multiply(s, s), qf::multiply(cc, cc)));
        r = (float128)1.0;
      }, kTolDefault));
    // B5: cosh^2(a) - sinh^2(a) ~= 1, a in [-5,5]. Narrowed to |a|<5: at large |a|
    // cosh^2 and sinh^2 are both ~e^(2a)/4 and their difference is a catastrophic
    // cancellation of two large ~equal quantities (loses leading digits, a FP
    // property not a QF defect). |a|<5 keeps them well-separated from the ~1 result.
    // Bound: 10 ulp.
    gb.push_back(run_group_b_unary("B5_cosh_sinh", kRandomN, seed++, uniform(-5.0, 5.0),
      [](double x, float128& c, float128& r) {
        qf::QuadFloat sh, ch; qf::sinhcosh(qf::QuadFloat(x), sh, ch);
        c = qf_to_q(qf::subtract(qf::multiply(ch, ch), qf::multiply(sh, sh)));
        r = (float128)1.0;
      }, kTolDefault));
    // B6: sin(a+b) ~= sin(a)cos(b) + cos(a)sin(b), a,b in [-3,3] (reduction clean).
    // Bound: 10 ulp.
    gb.push_back(run_group_b_binary("B6_sin_add", kRandomN, seed++,
      uniform(-3.0, 3.0), uniform(-3.0, 3.0),
      [](double a, double b, float128& c, float128& r) {
        qf::QuadFloat qa(a), qb(b), sa, ca, sb, cb;
        qf::sincos(qa, sa, ca); qf::sincos(qb, sb, cb);
        c = qf_to_q(qf::sin(qf::add(qa, qb)));
        r = qf_to_q(qf::add(qf::multiply(sa, cb), qf::multiply(ca, sb)));
      }, kTolDefault));
    // B7: cos(a+b) ~= cos(a)cos(b) - sin(a)sin(b), a,b in [-3,3].
    // Bound: 10 ulp.
    gb.push_back(run_group_b_binary("B7_cos_add", kRandomN, seed++,
      uniform(-3.0, 3.0), uniform(-3.0, 3.0),
      [](double a, double b, float128& c, float128& r) {
        qf::QuadFloat qa(a), qb(b), sa, ca, sb, cb;
        qf::sincos(qa, sa, ca); qf::sincos(qb, sb, cb);
        c = qf_to_q(qf::cos(qf::add(qa, qb)));
        r = qf_to_q(qf::subtract(qf::multiply(ca, cb), qf::multiply(sa, sb)));
      }, kTolDefault));
    // B8: tanh(a) ~= sinh(a)/cosh(a), a in [-40,40].
    // Bound: 10 ulp.
    gb.push_back(run_group_b_unary("B8_tanh_ratio", kRandomN, seed++, uniform(-40.0, 40.0),
      [](double x, float128& c, float128& r) {
        qf::QuadFloat sh, ch; qf::sinhcosh(qf::QuadFloat(x), sh, ch);
        c = qf_to_q(qf::tanh(qf::QuadFloat(x)));
        r = qf_to_q(qf::divide(sh, ch));
      }, kTolDefault));
    // B9: asin(sin(a)) ~= a, atan(tan(a)) ~= a — inverse pairs on principal ranges.
    // a in [-1.4,1.4] (inside (-pi/2,pi/2) so both are single-valued).
    // Bound: 10 ulp each.
    gb.push_back(run_group_b_unary("B9_asin_sin", kRandomN, seed++, uniform(-1.4, 1.4),
      [](double x, float128& c, float128& r) {
        c = qf_to_q(qf::asin(qf::sin(qf::QuadFloat(x))));
        r = (float128)x;
      }, kTolDefault));
    gb.push_back(run_group_b_unary("B10_atan_tan", kRandomN, seed++, uniform(-1.4, 1.4),
      [](double x, float128& c, float128& r) {
        c = qf_to_q(qf::atan(qf::tan(qf::QuadFloat(x))));
        r = (float128)x;
      }, kTolDefault));
    // B11: pow(a,2) ~= a*a, a>0 in [1e-3,1e3]. pow = exp(2*log a) compounds the exp
    // conditioning (PORT_NOTES_QF §10 tail). Gated at 30 ulp with the §10 citation.
    // Bound: 30 ulp (§10 — pow's internal exp tail; EXEMPT floor).
    gb.push_back(run_group_b_unary("B11_pow_two", kRandomN, seed++, loguniform(-3, 3),
      [](double x, float128& c, float128& r) {
        qf::QuadFloat a(x);
        c = qf_to_q(qf::pow(a, qf::QuadFloat(2.0)));
        r = qf_to_q(qf::multiply(a, a));
      }, kTolSection10));
    // B12: sqrt(a*a) ~= |a|, a in [-1e8,1e8].
    // Bound: 10 ulp.
    gb.push_back(run_group_b_unary("B12_sqrt_sq_abs", kRandomN, seed++, uniform(-1e8, 1e8),
      [](double x, float128& c, float128& r) {
        qf::QuadFloat a(x);
        c = qf_to_q(qf::sqrt(qf::multiply(a, a)));
        r = qf_to_q(qf::abs(a));
      }, kTolDefault));
    // B13: hypot(a,b)^2 ~= a^2 + b^2, a,b in [-1e3,1e3].
    // Bound: 10 ulp.
    gb.push_back(run_group_b_binary("B13_hypot_sq", kRandomN, seed++,
      uniform(-1e3, 1e3), uniform(-1e3, 1e3),
      [](double a, double b, float128& c, float128& r) {
        qf::QuadFloat da(a), db(b);
        qf::QuadFloat h = qf::hypot(da, db);
        c = qf_to_q(qf::multiply(h, h));
        r = qf_to_q(qf::add(qf::multiply(da, da), qf::multiply(db, db)));
      }, kTolDefault));
    // B14: exp(a + eps) ~= exp(a) * (1 + eps) at eps -> 0.  This is the T2.3 B4
    // pattern, INCLUDED explicitly. eps is drawn <= 1e-15 so the identity's OWN
    // truncation error (the dropped O(eps^2)/2 term ~ eps^2/2 <= 5e-31 < U) stays
    // below QF's resolution — any shortfall would be a library exp defect, not the
    // identity's algebra. Unlike FF (T2.3 B4, a real ff_math.hpp stall from
    // eps=1e-15f finer than FF resolution), qf_math.hpp exp uses eps=1e-28f coarser
    // than U (PORT_NOTES_QF §7/§10), so QF has NO stall — this is GREEN. Kept as a
    // durable regression guard and for T2.3 parity.
    // Bound: 10 ulp.
    gb.push_back(run_group_b_unary("B14_exp_eps", kRandomN, seed++,
      [](std::mt19937_64& g){ std::uniform_real_distribution<double> d(-5.0, 5.0); return d(g); },
      [](double a, float128& c, float128& r) {
        // draw eps deterministically from a but at 1e-15 scale (tiny perturbation).
        double eps = std::sin(a * 7.0) * 1e-15;   // |eps| <= 1e-15, varies with a
        qf::QuadFloat qa(a), qe(eps);
        c = qf_to_q(qf::exp(qf::add(qa, qe)));
        r = qf_to_q(qf::multiply(qf::exp(qa), qf::add(qf::QuadFloat(1.0), qe)));
      }, kTolDefault));

    // B4 min annotation (conditioning near +-pi*k). The identity is well-behaved so
    // its mean stays high; surface any low min honestly rather than hiding it.
    {
      const auto* ann = lookup_expected_min_drop("sin");
      const GroupBResult& b4 = *std::find_if(gb.begin(), gb.end(),
          [](const GroupBResult& g){ return g.name == "B4_pythag"; });
      std::printf("    note[B4_pythag]: min_digits=%.2f — dips are argument-reduction "
                  "conditioning near a=+/-pi*k (%s)\n", b4.min_d,
                  ann ? ann->reason : "sin near +/-pi; PORT_NOTES §5");
    }
    // B2/B11 §10 EXEMPT annotation (exp denormal tail).
    {
      const auto* ann = lookup_expected_min_drop("exp");
      std::printf("    note[B2_exp_log,B11_pow_two]: gated at 30 ulp (%.2f) — exp's "
                  "output-denormal tail (%s)\n", kTolSection10,
                  ann ? ann->reason : "exp denormal tail; PORT_NOTES_QF §10");
    }
#endif  // KOKKOS_EP_HAVE_QUADMATH

    // ------------------------------------------------------------------------
    // Device pass: 3 Group A (bit-exact) + 2 Group B (tolerance) on 10^5 inputs.
    // ------------------------------------------------------------------------
    std::printf("\n[Device] 3 Group A (bit-exact) + 2 Group B on %d inputs (%s)\n",
                kDeviceN, Kokkos::DefaultExecutionSpace::name());
    long device_failures = 0;

    // Device A1: a + (-a) == 0.
    {
      std::vector<double> x; std::vector<float> f0, f1, f2, f3;
      device_run(kDeviceN, 700001ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(double xv){ qf::QuadFloat a(xv); return qf::add(a, qf::negate(a)); },
        x, f0, f1, f2, f3);
      long f = 0;
      for (int i = 0; i < kDeviceN; ++i)
        if (!(f0[i] == 0.0f && f1[i] == 0.0f && f2[i] == 0.0f && f3[i] == 0.0f)) ++f;
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A1_add_neg", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }
    // Device A5: a * 1 == a (compare to the Route-A split of the input on host).
    {
      std::vector<double> x; std::vector<float> f0, f1, f2, f3;
      device_run(kDeviceN, 700002ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(double xv){ qf::QuadFloat a(xv); return qf::multiply(a, qf::QuadFloat(1.0f)); },
        x, f0, f1, f2, f3);
      long f = 0;
      for (int i = 0; i < kDeviceN; ++i) {
        qf::QuadFloat a(x[i]);
        if (!(f0[i] == a.f0 && f1[i] == a.f1 && f2[i] == a.f2 && f3[i] == a.f3)) ++f;
      }
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A5_mul_one", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }
    // Device A9: |a| == (a.f0>=0 ? a : -a).
    {
      std::vector<double> x; std::vector<float> f0, f1, f2, f3;
      device_run(kDeviceN, 700003ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(double xv){ qf::QuadFloat a(xv); return qf::abs(a); }, x, f0, f1, f2, f3);
      long f = 0;
      for (int i = 0; i < kDeviceN; ++i) {
        qf::QuadFloat a(x[i]);
        qf::QuadFloat w = (a.f0 >= 0.0f) ? a : qf::QuadFloat(-a.f0, -a.f1, -a.f2, -a.f3);
        if (!(f0[i] == w.f0 && f1[i] == w.f1 && f2[i] == w.f2 && f3[i] == w.f3)) ++f;
      }
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A9_abs_branch", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }

#ifdef KOKKOS_EP_HAVE_QUADMATH
    // Device B1: sqrt(a)^2 ~= a.
    {
      std::vector<double> x; std::vector<float> f0, f1, f2, f3;
      device_run(kDeviceN, 700004ULL, loguniform(-30, 30),
        KOKKOS_LAMBDA(double xv){ qf::QuadFloat a(xv); qf::QuadFloat s = qf::sqrt(a); return qf::multiply(s, s); },
        x, f0, f1, f2, f3);
      std::vector<double> digs(kDeviceN);
      for (int i = 0; i < kDeviceN; ++i)
        digs[i] = qf_digits(qf_to_q(qf::QuadFloat(f0[i], f1[i], f2[i], f3[i])), (float128)x[i]);
      AccStats s = compute_stats(digs.data(), kDeviceN);
      bool pass = s.mean >= kTolB1SqrtSq;
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  "B1_sqrt_sq", kDeviceN, s.min, s.mean, kTolB1SqrtSq, pass ? "PASS" : "FAIL");
    }
    // Device B4: sin^2+cos^2 ~= 1.
    {
      std::vector<double> x; std::vector<float> f0, f1, f2, f3;
      device_run(kDeviceN, 700005ULL, uniform(-100.0, 100.0),
        KOKKOS_LAMBDA(double xv){
          qf::QuadFloat a(xv), s, cc; qf::sincos(a, s, cc);
          return qf::add(qf::multiply(s, s), qf::multiply(cc, cc));
        }, x, f0, f1, f2, f3);
      std::vector<double> digs(kDeviceN);
      for (int i = 0; i < kDeviceN; ++i)
        digs[i] = qf_digits(qf_to_q(qf::QuadFloat(f0[i], f1[i], f2[i], f3[i])), (float128)1.0);
      AccStats s = compute_stats(digs.data(), kDeviceN);
      bool pass = s.mean >= kTolDefault;
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  "B4_pythag", kDeviceN, s.min, s.mean, kTolDefault, pass ? "PASS" : "FAIL");
    }
#endif

#ifdef KOKKOS_EP_HAVE_QUADMATH
    // ------------------------------------------------------------------------
    // TEST C — named-constant regressions.
    // ------------------------------------------------------------------------
    // Scaled to QF's 29-digit cap: target >= 27 digits (two digits of conditioning
    // headroom for the FP32-quad transcendental round-trips), matching the T2.3
    // ">=12 of 14" ratio.
    std::printf("\n[Test C] named-constant regressions (target >=27 digits)\n");
    int c_pass = 0, c_total = 0;
    const double kNamedMin = 27.0;

    auto case_digits = [&](const char* name, float128 computed, float128 ref) {
      ++c_total;
      double d = qf_digits(computed, ref);
      bool ok = d >= kNamedMin;
      std::printf("    %-16s digits=%6.2f  %s\n", name, d, ok ? "PASS" : "FAIL");
      if (ok) ++c_pass;
    };

    // C1: sin(pi) ~= 0. digits-of-accuracy against a nonzero ref is ill-defined at
    // ref=0, so measure "digits of zero-ness" = -log10(|sin(pi)|). SOFTENED by the
    // sin conditioning near +-pi (the QF pi constant carries a ~U absolute error, so
    // |sin(pi_qf)| ~= that error). Gate at a conditioning-aware floor.
    {
      ++c_total;
      qf::QuadFloat sp = qf::sin(qf::QuadFloat_pi());
      float128 v = qf_to_q(sp);
      double zero_digits = (v == (float128)0.0) ? kMaxDig
                           : -(double)Kokkos::log10(Kokkos::abs(v));
      const auto* ann = lookup_expected_min_drop("sin");
      const double kSinPiFloor = 27.0;   // QF resolves pi to ~U, so |sin(pi)| ~ 2^-96
      bool ok = zero_digits >= kSinPiFloor;
      std::printf("    %-16s |sin(pi)|=%.3e zero_digits=%6.2f  %s  (floor=%.1f; %s)\n",
                  "C1_sin_pi", (double)Kokkos::abs(v), zero_digits,
                  ok ? "PASS" : "FAIL", kSinPiFloor, ann ? ann->reason : "sin near pi");
      if (ok) ++c_pass;
    }
    // C2: log(e) ~= 1.
    case_digits("C2_log_e", qf_to_q(qf::log(qf::QuadFloat_e())), (float128)1.0);
    // C3: exp(log2) ~= 2.
    case_digits("C3_exp_log2", qf_to_q(qf::exp(qf::QuadFloat_log2())), (float128)2.0);
    // C4: sqrt2 * sqrt2 ~= 2.
    case_digits("C4_sqrt2_sq",
                qf_to_q(qf::multiply(qf::QuadFloat_sqrt2(), qf::QuadFloat_sqrt2())), (float128)2.0);
    // C5: log(10) ~= the QuadFloat_log10() constant (which stores ln(10)).
    case_digits("C5_log_ten",
                qf_to_q(qf::log(qf::QuadFloat(10.0))), qf_to_q(qf::QuadFloat_log10()));
    // C6: euler_gamma / digamma — SKIPPED: no digamma in qf_math.hpp, and the stored
    // euler_gamma constant has no independent QF op to check it against.
    std::printf("    %-16s SKIPPED (no digamma op; constant has no independent "
                "QF oracle in qf_math.hpp)\n", "C6_gamma");

    std::printf("  Test C: %d/%d passed\n", c_pass, c_total);
#endif  // KOKKOS_EP_HAVE_QUADMATH

    // ------------------------------------------------------------------------
    // ANTI-TESTS — identities DELIBERATELY NOT tested, and why.
    // ------------------------------------------------------------------------
    // These are FALSE for every finite-precision floating format, QF included;
    // asserting them would test IEEE arithmetic's known non-properties, not QF.
    //
    //  * Associativity of addition:  add(add(a,b),c) == add(a,add(b,c)).
    //    FP (and QF) addition is COMMUTATIVE (A11) but NOT ASSOCIATIVE — the two
    //    groupings round intermediate sums differently. T3.4 measures add's accuracy.
    //
    //  * Distributivity across large-magnitude differences:
    //    multiply(a, subtract(b,c)) == subtract(multiply(a,b), multiply(a,c)).
    //    When b and c are close and large, subtract(b,c) cancels exactly while
    //    multiply(a,b)/multiply(a,c) are large and individually rounded; their
    //    difference cannot match the LHS bit-for-bit. A property of finite precision.
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

#ifdef KOKKOS_EP_HAVE_QUADMATH
    long groupB_failed = 0; for (const auto& r : gb) if (!r.pass) ++groupB_failed;
    std::printf("  Group B: %zu identities, mean below tolerance=%ld\n", gb.size(), groupB_failed);
    KOKKOS_EP_ASSERT(groupB_failed == 0,
                     "a Group B identity's MEAN digits fell below its ulp tolerance");
    KOKKOS_EP_ASSERT(c_pass == c_total, "a Test C named-constant regression fell below its floor");
#endif

    std::printf("  Device: total failures=%ld\n", device_failures);
    KOKKOS_EP_ASSERT(device_failures == 0, "a device identity check failed");

    rc = ep_exit_code();
    std::printf("\n=== qf_property_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();

#ifndef KOKKOS_EP_HAVE_QUADMATH
  // Group A ran and gated above; Group B / Test C need the oracle. Signal the
  // partial run honestly as CTest "Skipped", unless Group A already hard-failed.
  if (rc == 0) {
    std::printf("(no __float128 oracle: Group A passed; Group B / Test C skipped)\n");
    return KOKKOS_EP_SKIP;
  }
#endif
  return rc;
}
