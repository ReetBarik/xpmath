// ============================================================================
// tf_accuracy_test.cpp — Layer 4 (differential accuracy vs quadmath) for TF.
//                        Plan S10 Phase 2.
// ============================================================================
//
// TF (TripleFloat, 3 x FP32) analogue of qf_accuracy_test.cpp (T3.4),
// ff_accuracy_test.cpp (T2.4), and dd_accuracy_test.cpp (T1.4). Structure
// mirrored end to end; mechanical changes are PRECISION SCALE and TOLERANCE
// MODEL:
//
//     FF:  double-word, precision u^2 = 2^-48  -> tolerance ~= 8.45 at N=10^6
//     QF:  quad-word,   precision U   = 2^-96  -> tolerance = 10*2^-96 -> 27.90 digits
//     TF:  TRIPLE-word, precision U   = 2^-72  -> tolerance = 10*2^-72 -> 19.63 digits
//
// TOLERANCE MODEL (TF precision scale, QF's absolute ulp-of-U pattern)
// ---------------------------------------------------------------------
// TF is a TRIPLE-word format — its resolution IS U = 2^-72 (three FP32 limbs,
// ~72 bits), NOT "u^3" — so this test follows the plan's per-op ABSOLUTE
// ulp-of-U tolerance, gating on the MEAN:
//     digits(k ulp) = -log10(k * 2^-72) = 72*log10(2) - log10(k)
//     10 ulp -> 19.63 (kTolDefault, the default gate)
//     30 ulp -> 19.16 (kTol30, for exp-family output-denormal tail)
//
// THE ORACLE, AND WIDENED-INPUT REFERENCE (same as QF)
// -----------------------------------------------------
// The host __float128 (quadmath) overloads carry ~34 digits — ~12 digits of
// headroom over TF's ~21.7 — so they are a legitimate ground-truth reference.
// The whole file is #ifdef KOKKOS_EP_HAVE_QUADMATH; without it main() returns
// KOKKOS_EP_SKIP (77 -> CTest "Skipped").
//
// Like QF, TF claims ~21.7 digits (~72 bits) — FINER than a double — so at TF
// precision the gap between (float128)x and the backend's actual stored value
// MATTERS for the broad regime. This test evaluates the oracle at the EXACT
// widened value tf_to_q(input), not at the nominal double. For narrow (Route-A)
// regime the two coincide (a double fits losslessly in 3 FP32 limbs), so this
// only changes the broad regime — where it is the only honest choice.
//
// INPUT REGIMES (three passes, combined per op)
//   (1) NARROW random  — Route-A TripleFloat(double) over well-conditioned domain
//   (2) BROAD random   — SAME domain but each input enriched to full ~72-bit via
//       make_wide_input (three-word mantissa width)
//   (3) CORPUS / near-edge — deterministic corner-case inputs random misses
//
// MEASURED ACCURACY (Phase 1.5, 300 log-uniform samples, __float128 oracle)
// --------------------------------------------------------------------------
//   add 24.61 | sub 24.57 | mul 23.45 | div 22.79 | sqrt 22.97
//   exp 21.35 | log 22.01 | sin 22.17 | cos 22.68
//
//   S10 Phase 3, same harness: atan 22.56 | asin 21.87 | acos 22.27 |
//   atan2 22.48 | angle 22.48 — replacing the 7.76 the FP32 placeholder scored.
//
// TOLERANCE TABLE (per-op, derived from measurement with margin)
// ---------------------------------------------------------------
// The phase-1 multiply bug measured 7.87 digits where ~23 was expected, and
// self-reporting missed it. This table must make a regression of that size fail
// loudly. Set each tolerance from measurement with a sensible margin.
//
// KNOWN-SHORT OPS (documented in PORT_NOTES_TF.md — NOT defects)
// ---------------------------------------------------------------
//   * atan, asin, acos, atan2, angle: SCORED as of S10 Phase 3 — no longer
//     placeholders. Real k=3 Newton-on-sincos port of qd_real::atan2. All five
//     are in the inventory below at tol 20.5, matching the forward trig family.
//     atan and asin carry a 2^-100 subnormal-tail domain bound (a sincos range
//     limit, characterized at the predicate); see PORT_NOTES_TF.md §11.
//   * expm1 (min 16.62), log1p (min 14.79), asinh (min 16.74): means at target,
//     minima low from argument conditioning; DD/FF/QF show the same shape.
//   * exp/exp2/exp10/pow at ~21.0-21.4 rather than 21.7: documented phase-1
//     trade (nq=5 squaring tail, deliberately coarse eps). Gated at 19.0 (2 digits
//     of margin below measured mean).

#include "test_utils.hpp"
#include "corpus.hpp"
#include <xp/tf_math.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace kokkos_ep;

namespace tf = xp;

static constexpr double kMaxDig = 21.7;   // TF ~21.68 decimal digits (3x24 ~= 72 bits)

#ifdef KOKKOS_EP_HAVE_QUADMATH

static float128 tf_to_q(const tf::TripleFloat& x) {
  return (float128)x.f0 + (float128)x.f1 + (float128)x.f2;
}

static double tf_digits(const tf::TripleFloat& got, const float128& truth) {
  if (truth == 0.0Q) return (got.f0 == 0.0f && got.f1 == 0.0f && got.f2 == 0.0f) ? kMaxDig : 0.0;
  float128 diff = tf_to_q(got) - truth;
  if (diff == 0.0Q) return kMaxDig;
  float128 rel = diff / truth;
  if (rel < 0.0Q) rel = -rel;
  double d = -log10q(rel);
  if (d < 0.0) return 0.0;
  if (d > kMaxDig) return kMaxDig;
  return d;
}

// ULP error, the second metric (see docs/ULP_METRIC.md). TF is 3 x FP32, so
// p = 72 significand bits and ulp(true) = |true| * 2^-72. A relative-error score
// goes vacuous near a zero of the function; this one does not. Reported, not
// gated — the condition-aware verdict needs kappa and a region label, and both
// live in scripts/sweep_accuracy --ulp.
static const int kTfSigBits = 72;                 // 3 x 24
static const double kUlpUnscorableTf = -1.0;      // ref zero / non-finite

static double tf_ulps(const tf::TripleFloat& got, const float128& truth) {
  if (isnanq(truth) || !finiteq(truth)) return kUlpUnscorableTf;
  const float128 g = tf_to_q(got);
  if (truth == 0.0Q) return (g == 0.0Q) ? 0.0 : kUlpUnscorableTf;
  if (isnanq(g) || !finiteq(g)) return HUGE_VAL;
  if (g == truth) return 0.0;
  return (double)ldexpq(fabsq(g - truth) / fabsq(truth), kTfSigBits);
}

// ----------------------------------------------------------------------------
// Wide input construction (reused from qf tests)
// ----------------------------------------------------------------------------
static tf::TripleFloat make_wide_input(double x, std::mt19937_64& gen) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  double r = x;
  float f0 = (float)r; r -= (double)f0;
  r += d(gen) * std::ldexp(1.0, -25);
  float f1 = (float)r; r -= (double)f1;
  r += d(gen) * std::ldexp(1.0, -50);
  float f2 = (float)r;
  // S10 Phase 3 fix: this was `from_bits(f0, f1, f2)`. `from_bits` takes three
  // uint32_t IEEE-754 BIT PATTERNS, not three floats, so every broad-regime
  // input was a float-to-uint32 conversion (0 or UB for negatives and for
  // anything past 2^32) REINTERPRETED as a float — garbage, frequently NaN.
  // That is what produced `mean nan` on 33 of the 45 scored rows. The three
  // words are ordinary component values here, so the 3-component constructor is
  // what this wants. See PORT_NOTES_TF.md §11d.
  return tf::TripleFloat(f0, f1, f2);
}

// ----------------------------------------------------------------------------
// Tolerance constants
// ----------------------------------------------------------------------------
static constexpr double kTolDefault = 19.63;  // 10 ulp of 2^-72
static constexpr double kTol30 = 19.16;       // 30 ulp (exp-family denormal tail)

// Per-op tolerance table (derived from measurement with margin)
struct OpTolerance {
  const char* name;
  double tol;
};

// S10 Phase 3.5 (DEFECT 2) — THE CAP RULE.
// -----------------------------------------------------------------------------
// No tolerance may sit at or above kMaxDig (21.70). tf_digits() CLAMPS every
// per-sample score to kMaxDig, so a gate of 22.00 is unreachable by
// construction: a bit-exact op scores a perfect 21.70 on every sample and is
// then marked FAIL. Eleven rows carried 22.00 and accounted for eleven of the
// fourteen pre-Phase-3.5 failures. They are corrected below, per row, with the
// value this harness measures.
//
// MARGIN CONVENTION. The inexact rows keep the file's existing convention
// (measured mean minus a 1.0-2.6 digit margin). For the EXACT rows there is no
// measurement spread to margin against — min == mean == 21.70 across all 409
// samples — so the only thing a margin buys is tolerance for a regression, and
// the correct gate is as close under the cap as the harness allows: 21.50, i.e.
// 0.20 below. That is TIGHTER than the file's other exact rows (fdim, fma,
// pow_int all score the cap and are gated at 19.0-20.0); those are left alone.
static const OpTolerance kOpTols[] = {
  // Arithmetic (measured means: add 21.70, sub 21.70, mul 21.08, div 21.69,
  // sqrt 21.52, sqr 21.53 post-domain-fix)
  {"add", 21.5},         // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact
                         //       (min == mean, all 409 samples). Gate 0.20 under.
  {"subtract", 21.5},    // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact.
  {"multiply", 21.0},    // unchanged; measured 21.08
  {"divide", 20.5},      // unchanged; measured 21.69
  {"sqrt", 20.5},        // unchanged; measured 21.52
  {"sqr", 21.0},         // unchanged; measured 21.53 once the FP32-underflow
                         // corpus inputs are excluded by domain (see sqr_in_domain)
  {"abs", 21.5},         // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact.
  {"negate", 21.5},      // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact.

  // Transcendentals (measured: exp 21.35, log 22.01, sin 22.17, cos 22.68)
  {"exp", 19.0},         // 2.4 margin; known coarse (~21.0-21.4 per PORT_NOTES)
  {"exp2", 19.0},        // same family
  {"exp10", 19.0},       // same family
  {"expm1", 19.0},       // min 16.62 from conditioning, mean at target
  {"log", 20.0},         // 2.0 margin below 22.01
  {"log2", 20.0},        // same family
  {"log10", 20.0},       // same family
  {"log1p", 19.0},       // min 14.79 from conditioning
  {"sin", 20.0},         // 2.2 margin below 22.17
  {"cos", 20.5},         // 2.2 margin below 22.68
  {"tan", 20.0},         // derived from sin/cos
  {"sinh", 20.0},        // hyperbolic analogue
  {"cosh", 20.0},        // hyperbolic analogue
  {"tanh", 20.0},        // hyperbolic analogue
  {"asinh", 19.0},       // min 16.74 from conditioning
  {"acosh", 19.0},       // inverse hyperbolic
  {"atanh", 19.0},       // inverse hyperbolic

  // Inverse trigonometric (S10 Phase 3). Real k=3 Newton-on-sincos, no longer
  // FP32 placeholders. Tolerances set from THIS test's measurement with margin,
  // same as every other row: measured means are 21.61-21.69 and measured minima
  // 20.80-20.99, so 20.50 leaves ~1.1 digits on the mean — the same margin the
  // FORWARD trig family carries (cos: mean 21.44 at tol 20.50), and the gate
  // still sits above every measured minimum. See PORT_NOTES_TF.md §11.
  {"atan", 20.5},
  {"asin", 20.5},
  {"acos", 20.5},
  {"atan2", 20.5},
  {"angle", 20.5},
  {"pow", 19.0},         // exp-family, ~21.0-21.4 measured
  {"pow_int", 20.0},     // repeated multiply
  {"hypot", 20.88},      // KI-8: ratcheted from 20.00. Measured mean 21.18 on
                         //       [-100,100]^2 -- inside the scaling gate, so the
                         //       KI-8 fix leaves this domain bit-identical and
                         //       the old 1.18-digit slack was never earned.

  // Binary ops
  {"fmod", 21.5},        // KI-10/KI-15: ratcheted from 20.00. The exact
                         //       iterative reduction returns the correctly
                         //       rounded remainder, so this domain measures
                         //       21.70 = cap; the old 1.70-digit slack was
                         //       slack for a defect, not for conditioning.
  {"remainder", 21.5},   // KI-10/KI-15: ratcheted from 19.00, same reason.
  {"fdim", 20.0},
  {"fmax", 21.5},        // P3.5: was 22.00 (> cap). Exact selection, measured
                         //       21.70 = cap on all 201 samples.
  {"fmin", 21.5},        // P3.5: was 22.00 (> cap). Exact selection, 21.70 = cap.
  {"copysign", 21.5},    // P6: exact sign manipulation (to be measured)
  {"fma", 19.0},         // triple composition
  {"multiply_scalar", 21.0},  // unchanged; 21.70 once the oracle uses the FP32
                              // constant the op is actually given (P3.5)

  // Rounding family. All five were at 22.00 (> cap).
  {"round", 21.5},       // P3.5: 21.70 = cap once exact half-integers and their
                         //       FP32 neighbours leave the corpus (round_in_domain)
  {"ceil", 21.5},        // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact.
  {"floor", 21.5},       // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact.
  {"trunc", 21.5},       // P3.5: was 22.00 (> cap). Measured 21.70 = cap, exact.
  {"round_to_nearest_int", 21.5},  // P3.5: as round (same function).
};

static double lookup_tolerance(const char* name) {
  for (auto& t : kOpTols) {
    if (std::strcmp(t.name, name) == 0) return t.tol;
  }
  return kTolDefault;  // fallback
}

// ----------------------------------------------------------------------------
// OpResult accumulator
// ----------------------------------------------------------------------------
struct OpResult {
  double sum = 0.0;
  double min_dig = kMaxDig;
  int n = 0;
  int skip = 0;
  // Second metric. WORST point over all three passes, never a mean.
  double worst_ulp = 0.0;
  int n_ulp_scored = 0;
  int n_ulp_unscored = 0;
};

// Fold one element's ulp error into an OpResult. Called next to every
// tf_digits() so the two metrics always cover the same element set.
static void tf_note_ulp(OpResult& R, const tf::TripleFloat& got,
                        const float128& truth) {
  const double u = tf_ulps(got, truth);
  if (u == kUlpUnscorableTf) { ++R.n_ulp_unscored; return; }
  ++R.n_ulp_scored;
  if (u > R.worst_ulp) R.worst_ulp = u;
}

// ----------------------------------------------------------------------------
// Unary op runner (three passes: narrow, broad, corpus)
// ----------------------------------------------------------------------------
template<typename UnaryOp, typename Domain>
static OpResult test_unary(const char* name, UnaryOp op,
                            float128 (*oracle)(float128), Domain dom) {
  OpResult R;
  std::mt19937_64 gen(12345);

  // Pass 1: narrow (Route-A)
  std::uniform_real_distribution<double> d_narrow(dom.lo, dom.hi);
  for (int i = 0; i < 150; ++i) {
    double x = d_narrow(gen);
    if (!dom.valid(x)) { ++R.skip; continue; }
    auto tf_x = tf::TripleFloat(x);
    auto got = op(tf_x);
    float128 truth = oracle(tf_to_q(tf_x));
    double dig = tf_digits(got, truth);
    tf_note_ulp(R, got, truth);
    R.sum += dig; ++R.n;
    if (dig < R.min_dig) R.min_dig = dig;
  }

  // Pass 2: broad (wide inputs)
  for (int i = 0; i < 100; ++i) {
    double x = d_narrow(gen);
    if (!dom.valid(x)) { ++R.skip; continue; }
    auto tf_x = make_wide_input(x, gen);
    auto got = op(tf_x);
    float128 truth = oracle(tf_to_q(tf_x));
    double dig = tf_digits(got, truth);
    tf_note_ulp(R, got, truth);
    R.sum += dig; ++R.n;
    if (dig < R.min_dig) R.min_dig = dig;
  }

  // Pass 3: corpus
  corpus::CorpusFlags flags;
  auto xs = corpus::unary<float>(flags);
  for (float xf : xs) {
    double x = (double)xf;
    if (!dom.valid(x)) { ++R.skip; continue; }
    auto tf_x = tf::TripleFloat(x);
    auto got = op(tf_x);
    float128 truth = oracle(tf_to_q(tf_x));
    double dig = tf_digits(got, truth);
    tf_note_ulp(R, got, truth);
    R.sum += dig; ++R.n;
    if (dig < R.min_dig) R.min_dig = dig;
  }

  return R;
}

// Binary op runner
template<typename BinaryOp, typename Domain>
static OpResult test_binary(const char* name, BinaryOp op,
                             float128 (*oracle)(float128, float128), Domain dom) {
  OpResult R;
  std::mt19937_64 gen(23456);

  // Narrow
  std::uniform_real_distribution<double> da(dom.lo_a, dom.hi_a);
  std::uniform_real_distribution<double> db(dom.lo_b, dom.hi_b);
  for (int i = 0; i < 100; ++i) {
    double xa = da(gen), xb = db(gen);
    if (!dom.valid(xa, xb)) { ++R.skip; continue; }
    auto tfa = tf::TripleFloat(xa), tfb = tf::TripleFloat(xb);
    auto got = op(tfa, tfb);
    float128 truth = oracle(tf_to_q(tfa), tf_to_q(tfb));
    double dig = tf_digits(got, truth);
    tf_note_ulp(R, got, truth);
    R.sum += dig; ++R.n;
    if (dig < R.min_dig) R.min_dig = dig;
  }

  // Broad
  for (int i = 0; i < 50; ++i) {
    double xa = da(gen), xb = db(gen);
    if (!dom.valid(xa, xb)) { ++R.skip; continue; }
    auto tfa = make_wide_input(xa, gen), tfb = make_wide_input(xb, gen);
    auto got = op(tfa, tfb);
    float128 truth = oracle(tf_to_q(tfa), tf_to_q(tfb));
    double dig = tf_digits(got, truth);
    tf_note_ulp(R, got, truth);
    R.sum += dig; ++R.n;
    if (dig < R.min_dig) R.min_dig = dig;
  }

  // Corpus
  corpus::CorpusFlags flags;
  auto xs = corpus::binary<float>(flags);
  for (auto [af, bf] : xs) {
    double xa = (double)af, xb = (double)bf;
    if (!dom.valid(xa, xb)) { ++R.skip; continue; }
    auto tfa = tf::TripleFloat(xa), tfb = tf::TripleFloat(xb);
    auto got = op(tfa, tfb);
    float128 truth = oracle(tf_to_q(tfa), tf_to_q(tfb));
    double dig = tf_digits(got, truth);
    tf_note_ulp(R, got, truth);
    R.sum += dig; ++R.n;
    if (dig < R.min_dig) R.min_dig = dig;
  }

  return R;
}

// ----------------------------------------------------------------------------
// Domain predicates
// ----------------------------------------------------------------------------
struct UnaryDomain {
  double lo, hi;
  bool (*valid_fn)(double) = nullptr;
  bool valid(double x) const {
    if (x < lo || x > hi) return false;
    if (valid_fn && !valid_fn(x)) return false;
    return std::isfinite(x);
  }
};

struct BinaryDomain {
  double lo_a, hi_a, lo_b, hi_b;
  bool (*valid_fn)(double, double) = nullptr;
  bool valid(double a, double b) const {
    if (a < lo_a || a > hi_a || b < lo_b || b > hi_b) return false;
    if (valid_fn && !valid_fn(a, b)) return false;
    return std::isfinite(a) && std::isfinite(b);
  }
};

// atan2(y, x) / angle(x, y): both operands finite, neither in the FP32
// subnormal tail, and the larger magnitude below the point where x^2 + y^2
// overflows an FP32 word. Ported from qf_accuracy_test.cpp:793-800 (dom2_atan2)
// — the same bounds transfer verbatim because QF's words are FP32 too. The
// both-zero pair is excluded: it is xp::angle's diagnostic path, not a value.
static bool atan2_in_domain(double a, double b) {
  if (!(std::isfinite(a) && std::isfinite(b))) return false;
  double m = std::fmax(std::fabs(a), std::fabs(b));
  if (m == 0.0) return false;
  if (m > 1e18) return false;
  if (a != 0.0 && std::fabs(a) < 1e-18) return false;
  if (b != 0.0 && std::fabs(b) < 1e-18) return false;
  return true;
}

// angle(x, y) == atan2(y, x): the oracle takes the arguments in the opposite
// order, so it needs its own free function (test_binary wants a plain pointer).
static float128 angle_oracle(float128 x, float128 y) { return atan2q(y, x); }

// atan / asin: exclude the FP32 subnormal tail. The bound is 2^-100, matching
// tf_property_test.cpp:137's kUnderflowTail rather than the coarser 1e-18 the
// binary predicate above inherits from qf_accuracy_test.cpp:799 — 2^-100 is what
// the defect actually needs, and a wider exclusion would discard well-behaved
// inputs (atan(1e-25) scores the cap) for nothing.
//
// Cause, measured: for a subnormal argument the Newton loop's sincos underflows.
// sincos divides its reduced residual by 2^nq (tf_math.hpp:832), and
// denorm_min/16 flushes to zero, so sin(z) returns 0 instead of z. Newton then
// adds the full residual every step, and atan(1.4e-45) comes back as 4.2e-45 —
// exactly 3x the input, one term per iteration plus the seed. This is a range
// limit of TF's sincos in the subnormal tail, not of the inverse trig: the
// arithmetic is fine, the argument reduction has no bits left to work with.
// It bounds the domain rather than the accuracy, so it is stated here as a
// domain, exactly as PORT_NOTES_TF.md §10d argues for B1's sqrt bound.
// acos is NOT excluded: acos(tiny) -> pi/2 is well-conditioned and scores 20.99.
static bool inv_trig_in_domain(double x) {
  return x == 0.0 || std::fabs(x) >= 0x1p-100;
}

// sqr: exclude inputs whose SQUARE is not an FP32 normal. S10 Phase 3.5.
// The corpus carries FLT_MIN, FLT_TRUE_MIN and neighbours; x*x for those is
// 1e-76 .. 1e-90, far below FP32's 2^-149 smallest subnormal, so every word of
// the product flushes to zero and the sample scores 0 digits. Fourteen such
// samples pulled the sqr mean from 21.53 to 20.85. This is an EXPONENT-RANGE
// limit of the FP32 word, not an accuracy limit of the k=3 algorithm — the same
// argument inv_trig_in_domain above makes for the subnormal tail, and the one
// PORT_NOTES_TF.md §10d makes for B1's sqrt bound — so it bounds the domain
// rather than the tolerance. |x| >= 2^-63 makes x*x >= 2^-126 = FLT_MIN.
// (multiply is not given the same predicate: it already passes, and Phase 3.5
// was told not to move rows it does not have to.)
static bool sqr_in_domain(double x) {
  double f = std::fabs(x);
  return f == 0.0 || f >= 0x1p-63;
}

// round family: exclude exact half-integers and their FP32 neighbours.
// S10 Phase 3.5. TF's round_to_nearest_int is QD's nint, floor(x + 0.5)
// (qd_inline.h:975 / qd.cpp nint), which breaks ties toward +infinity;
// quadmath's roundq breaks them AWAY FROM ZERO. The two disagree on every
// negative half-integer — nint(-1.5) = -1, roundq(-1.5) = -2 — and the corpus
// supplies -0.5, -1.5, -2.5, -10.5, -19.5, -100.5, -1000.5 among others, each
// scoring ~0 digits and dragging the mean to 21.18.
//
// This is the tie-semantics question qf_accuracy_test.cpp:104-113 already
// declares out of scope, and it handles it the same way: keep Kokkos::round /
// roundq as the oracle and keep exact ties out of the input set (QF excludes the
// nint_half_integer corpus family; TF's corpus has no such switch, so the
// predicate does it).
//
// TF needs ONE case QF does not: `floor(f0 + 0.5f)` is evaluated in FP32, so an
// input just under a tie can round UP onto it in the ADD and then land on the
// far side. 0.49999997 is the corpus case — 0.49999997f + 0.5f is 0.99999997,
// which is half an FP32 ulp below 1 and ties-to-even up to exactly 1.0f, so nint
// returns 1 where the nearest integer is 0. That is QD's own double-rounding
// hazard evaluated at FP32 resolution, not a TF port error.
//
// The exclusion is deliberately NARROW: only exact ties and that inexact-add
// case. It does NOT exclude a leading word that merely sits on a half-integer
// with a nonzero tail — tf_math.hpp:701-702 ports QD's guard for that
// (`|f0 - a.f0| == 0.5f && a.f1 < 0.0f -> f0 -= 1`) and gets those right. An
// earlier one-ulp-band version of this predicate skipped 85 of 405 samples,
// most of them cases TF handles correctly; this one skips the same 12 the
// unfiltered domain did, plus the ties.
static bool round_in_domain(double x) {
  if (!std::isfinite(x)) return false;
  if (std::fabs(x) >= 1e13) return false;              // nint's guard, as QF's rows
  // (a) nint's FP32 add must not CHANGE the decision. Testing the add for
  //     exactness alone is too broad — for |x| << 1 the add is inexact but
  //     floor() lands on 0 either way (it skipped 38 harmless samples). Compare
  //     the two floors instead, so only genuine flips are excluded.
  float f0 = (float)x;
  if (std::floor((double)(f0 + 0.5f)) != std::floor((double)f0 + 0.5)) return false;
  // (b) exact tie: nint is half-UP, roundq is half-AWAY-FROM-ZERO. They disagree
  //     on every negative half-integer; convention, not accuracy.
  return std::fabs(x - std::floor(x)) != 0.5;
}

// multiply_scalar's scalar operand. Declared once so the OP and the ORACLE use
// the SAME value. S10 Phase 3.5: the op was called with 3.14159f and the oracle
// with the literal 3.14159Q, and float(3.14159) = 3.141590118408203125 differs
// from the decimal by 3.77e-8 relative — 7.42 digits. That constant mismatch,
// not the header, is the whole of the 7.50 this row used to score; with the
// oracle corrected the row is bit-exact at the 21.70 cap. qf_accuracy_test.cpp
// (:880, "Oracle: qf_to_q(a) * (float128)b") already widens the FP32 operand.
static constexpr float kMulScalarB = 3.14159f;

// multiply_scalar: exclude inputs whose PRODUCT is not an FP32 normal — the same
// exponent-range argument sqr_in_domain makes. With the oracle corrected the row
// measured 21.26 mean / 1.35 min, and the min is FLT_TRUE_MIN: 1.4013e-45 *
// 3.14159 = 4.4023e-45, whose nearest FP32 subnormal is 4.2039e-45 (three times
// FLT_TRUE_MIN), a 4.5% relative error = 1.35 digits. Two bits of mantissa is
// all the format has left there. |x| >= 2^-124 keeps |x*b| above FLT_MIN.
static bool mul_scalar_in_domain(double x) {
  double f = std::fabs(x);
  return f == 0.0 || f >= 0x1p-124;
}

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("=== tf_accuracy_test: TF differential accuracy vs __float128 oracle ===\n");
    std::printf("TF precision: ~21.7 decimal digits (3×FP32, ~72 bits)\n");
    std::printf("Tolerance: per-op table derived from measurement with margin\n");
    std::printf("Regimes: narrow (Route-A) + broad (wide) + corpus\n\n");

    std::vector<std::pair<const char*, double>> results;
    int total_ops = 0;
    int passed = 0;

    auto report = [&](const char* name, const OpResult& R) {
      double mean = R.n > 0 ? R.sum / R.n : 0.0;
      double tol = lookup_tolerance(name);
      bool ok = mean >= tol;
      if (ok) ++passed;
      ++total_ops;
      results.push_back({name, mean});
      // Both metrics: the digit score the KI history is written in, and the
      // worst-point ulp error the digit score cannot see near a zero of f.
      std::printf("  %-20s: mean %6.2f, min %6.2f, n=%5d, skip=%4d [tol %.2f] %s"
                  "  worst_ulp=%-11.4g ulp_n=%-5d ulp_unscorable=%d\n",
                  name, mean, R.min_dig, R.n, R.skip, tol, ok ? "PASS" : "FAIL",
                  R.worst_ulp, R.n_ulp_scored, R.n_ulp_unscored);
      return ok;
    };

    // Arithmetic
    std::printf("[Arithmetic]\n");
    report("add", test_unary("add", [](auto x) { return tf::add(x, tf::TripleFloat(1.0)); },
                              [](auto x) { return x + 1.0Q; }, UnaryDomain{-1e15, 1e15}));
    report("subtract", test_unary("subtract", [](auto x) { return tf::subtract(x, tf::TripleFloat(0.5)); },
                                   [](auto x) { return x - 0.5Q; }, UnaryDomain{-1e15, 1e15}));
    report("multiply", test_binary("multiply", [](auto a, auto b) { return tf::multiply(a, b); },
                                    [](auto a, auto b) { return a * b; },
                                    BinaryDomain{-1e15, 1e15, -1e15, 1e15}));
    report("divide", test_binary("divide", [](auto a, auto b) { return tf::divide(a, b); },
                                  [](auto a, auto b) { return a / b; },
                                  BinaryDomain{-1e15, 1e15, 0.01, 1e15}));
    report("sqrt", test_unary("sqrt", [](auto x) { return tf::sqrt(x); }, sqrtq, UnaryDomain{0.0, 1e30}));
    report("sqr", test_unary("sqr", [](auto x) { return tf::sqr(x); }, [](auto x) { return x * x; },
                              UnaryDomain{-1e15, 1e15, sqr_in_domain}));
    report("abs", test_unary("abs", [](auto x) { return tf::abs(x); }, fabsq, UnaryDomain{-1e30, 1e30}));
    report("negate", test_unary("negate", [](auto x) { return tf::negate(x); }, [](auto x) { return -x; }, UnaryDomain{-1e30, 1e30}));

    // Exponential family
    std::printf("\n[Exponential family]\n");
    report("exp", test_unary("exp", [](auto x) { return xp::exp(x); }, expq, UnaryDomain{-35.0, 80.0}));
    report("exp2", test_unary("exp2", [](auto x) { return xp::exp2(x); }, exp2q, UnaryDomain{-50.0, 120.0}));
    report("exp10", test_unary("exp10", [](auto x) { return xp::exp10(x); },
                                [](auto x) { return powq(10.0Q, x); }, UnaryDomain{-15.0, 30.0}));
    report("expm1", test_unary("expm1", [](auto x) { return xp::expm1(x); }, expm1q, UnaryDomain{-1.0, 1.0}));

    // Logarithm family
    std::printf("\n[Logarithm family]\n");
    report("log", test_unary("log", [](auto x) { return xp::log(x); }, logq, UnaryDomain{1e-13, 1e13}));
    report("log2", test_unary("log2", [](auto x) { return xp::log2(x); }, log2q, UnaryDomain{1e-13, 1e13}));
    report("log10", test_unary("log10", [](auto x) { return xp::log10(x); }, log10q, UnaryDomain{1e-13, 1e13}));
    report("log1p", test_unary("log1p", [](auto x) { return xp::log1p(x); }, log1pq, UnaryDomain{-0.5, 2.0}));

    // Trigonometric
    std::printf("\n[Trigonometric]\n");
    report("sin", test_unary("sin", [](auto x) { return xp::sin(x); }, sinq, UnaryDomain{-3.0, 3.0}));
    report("cos", test_unary("cos", [](auto x) { return xp::cos(x); }, cosq, UnaryDomain{-3.0, 3.0}));
    report("tan", test_unary("tan", [](auto x) { return xp::tan(x); }, tanq, UnaryDomain{-1.0, 1.0}));

    // Inverse trigonometric — SCORED as of S10 Phase 3. These were excluded
    // through Phase 2.5 as FP32 scalar placeholders (~7.5 digits); Phase 3
    // replaced them with the real k=3 Newton-on-sincos port of qd_real::atan2
    // (QD 2.3.24 qd_real.cpp:2393-2458). See PORT_NOTES_TF.md §11.
    std::printf("\n[Inverse trigonometric]\n");
    report("atan", test_unary("atan", [](auto x) { return xp::atan(x); }, atanq,
                               UnaryDomain{-1e18, 1e18, inv_trig_in_domain}));
    report("asin", test_unary("asin", [](auto x) { return xp::asin(x); }, asinq,
                               UnaryDomain{-1.0, 1.0, inv_trig_in_domain}));
    report("acos", test_unary("acos", [](auto x) { return xp::acos(x); }, acosq,
                               UnaryDomain{-1.0, 1.0}));
    // atan2(y, x): quadmath's atan2q takes the same (y, x) order.
    report("atan2", test_binary("atan2", [](auto a, auto b) { return xp::atan2(a, b); }, atan2q,
                                 BinaryDomain{-1e3, 1e3, -1e3, 1e3, atan2_in_domain}));
    // angle(x, y) == atan2(y, x): the oracle swaps the arguments back.
    report("angle", test_binary("angle", [](auto a, auto b) { return xp::angle(a, b); }, angle_oracle,
                                 BinaryDomain{-1e3, 1e3, -1e3, 1e3, atan2_in_domain}));

    // Hyperbolic
    std::printf("\n[Hyperbolic]\n");
    report("sinh", test_unary("sinh", [](auto x) { return xp::sinh(x); }, sinhq, UnaryDomain{-1.0, 1.0}));
    report("cosh", test_unary("cosh", [](auto x) { return xp::cosh(x); }, coshq, UnaryDomain{-1.0, 1.0}));
    report("tanh", test_unary("tanh", [](auto x) { return xp::tanh(x); }, tanhq, UnaryDomain{-2.0, 2.0}));
    report("asinh", test_unary("asinh", [](auto x) { return xp::asinh(x); }, asinhq, UnaryDomain{-10.0, 10.0}));
    report("acosh", test_unary("acosh", [](auto x) { return xp::acosh(x); }, acoshq, UnaryDomain{1.0, 100.0}));
    report("atanh", test_unary("atanh", [](auto x) { return xp::atanh(x); }, atanhq, UnaryDomain{-0.9, 0.9}));

    // Power
    std::printf("\n[Power]\n");
    report("pow", test_binary("pow", [](auto a, auto b) { return xp::pow(a, b); }, powq,
                               BinaryDomain{0.5, 100.0, -2.0, 2.0}));
    report("pow_int", test_unary("pow_int", [](auto x) { return xp::pow_int(x, 7); },
                                  [](auto x) { return powq(x, 7.0Q); }, UnaryDomain{0.1, 10.0}));
    report("hypot", test_binary("hypot", [](auto a, auto b) { return xp::hypot(a, b); },
                                 hypotq, BinaryDomain{-100.0, 100.0, -100.0, 100.0}));

    // Binary ops
    std::printf("\n[Binary operations]\n");
    report("fmod", test_binary("fmod", [](auto a, auto b) { return xp::fmod(a, b); }, fmodq,
                                BinaryDomain{-100.0, 100.0, 1.0, 10.0}));
    report("remainder", test_binary("remainder", [](auto a, auto b) { return xp::remainder(a, b); }, remainderq,
                                     BinaryDomain{-100.0, 100.0, 1.0, 10.0}));
    report("fdim", test_binary("fdim", [](auto a, auto b) { return xp::fdim(a, b); }, fdimq,
                                BinaryDomain{-100.0, 100.0, -100.0, 100.0}));
    report("fmax", test_binary("fmax", [](auto a, auto b) { return xp::fmax(a, b); }, fmaxq,
                                BinaryDomain{-100.0, 100.0, -100.0, 100.0}));
    report("fmin", test_binary("fmin", [](auto a, auto b) { return xp::fmin(a, b); }, fminq,
                                BinaryDomain{-100.0, 100.0, -100.0, 100.0}));
    report("copysign", test_binary("copysign", [](auto a, auto b) { return xp::copysign(a, b); }, copysignq,
                                    BinaryDomain{-100.0, 100.0, -100.0, 100.0}));
    report("fma", test_binary("fma", [](auto a, auto b) { return xp::fma(a, b, tf::TripleFloat(1.0)); },
                               [](auto a, auto b) { return fmaq(a, b, 1.0Q); }, BinaryDomain{-10.0, 10.0, -10.0, 10.0}));
    report("multiply_scalar", test_unary("multiply_scalar",
                                          [](auto x) { return xp::multiply_scalar(x, kMulScalarB); },
                                          [](auto x) { return x * (float128)kMulScalarB; },
                                          UnaryDomain{-1e15, 1e15, mul_scalar_in_domain}));

    // Rounding family
    std::printf("\n[Rounding]\n");
    // round / round_to_nearest_int carry round_in_domain (nint-vs-roundq tie
    // semantics, out of scope per qf_accuracy_test.cpp:104-113). ceil/floor/trunc
    // are directed roundings with no tie to break, so they keep the plain domain.
    report("round", test_unary("round", [](auto x) { return tf::round(x); }, roundq,
                                UnaryDomain{-1e6, 1e6, round_in_domain}));
    report("ceil", test_unary("ceil", [](auto x) { return tf::ceil(x); }, ceilq, UnaryDomain{-1e6, 1e6}));
    report("floor", test_unary("floor", [](auto x) { return tf::floor(x); }, floorq, UnaryDomain{-1e6, 1e6}));
    report("trunc", test_unary("trunc", [](auto x) { return tf::trunc(x); }, truncq, UnaryDomain{-1e6, 1e6}));
    report("round_to_nearest_int", test_unary("round_to_nearest_int",
                                               [](auto x) { return tf::round_to_nearest_int(x); },
                                               roundq, UnaryDomain{-1e6, 1e6, round_in_domain}));

    std::printf("\n=== Summary ===\n");
    std::printf("Total ops tested: %d\n", total_ops);
    std::printf("Passed: %d\n", passed);
    std::printf("Failed: %d\n", total_ops - passed);

    if (passed < total_ops) {
      std::printf("\nFailed ops:\n");
      for (auto [name, mean] : results) {
        double tol = lookup_tolerance(name);
        if (mean < tol) {
          std::printf("  %s: %.2f < %.2f\n", name, mean, tol);
        }
      }
    }

    // S10 Phase 3.5 (DEFECT 1). This assert is what makes the test able to fail.
    // Until now main() went straight to `rc = ep_exit_code()`, and
    // ep_exit_code() (test_utils.hpp:530) reads detail::ep_failure_count() —
    // which NOTHING in this file incremented. The per-op verdicts above were
    // printed, never registered, so the binary exited 0 while its own summary
    // said `Failed: 14`, and its ctest green meant only that it ran.
    // Mechanism copied from qf_accuracy_test.cpp:1114-1121: assert on the
    // aggregate failure count FIRST, then read the exit code.
    KOKKOS_EP_ASSERT(passed == total_ops,
                     "one or more TF ops fell below their accuracy tolerance "
                     "(mean < per-op gate) — see the FAIL rows above");
    rc = ep_exit_code();

    std::printf("\n=== tf_accuracy_test: %s ===\n", rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();
  return rc;
}

#else // !KOKKOS_EP_HAVE_QUADMATH

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  std::printf("=== tf_accuracy_test: SKIPPED (no __float128 oracle) ===\n");
  Kokkos::finalize();
  return KOKKOS_EP_SKIP;
}

#endif // KOKKOS_EP_HAVE_QUADMATH
