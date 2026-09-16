// ============================================================================
// qf_nonoverlap_test.cpp — Layer 2 (output-invariant tests) for QF.  Plan T3.2.
// ============================================================================
//
// WHAT THIS LAYER CHECKS AND WHY IT IS ORACLE-INDEPENDENT
// ------------------------------------------------------
// This is the QF (QuadFloat, 4 x FP32) analogue of ff_invariant_test.cpp (T2.2)
// and dd_invariant_test.cpp (T1.2).  Layer 1 (qf_eft_test.cpp, T3.1) proved the
// atoms — qf_two_sum, the Dekker qf_two_prod / qf_two_sqr, and the renorm family
// — are bit-exact and non-overlap-preserving.  This layer checks a STRUCTURAL
// property of every higher-level op's OUTPUT: a QuadFloat (f0, f1, f2, f3) must
// be a NON-OVERLAPPING length-4 expansion — each successive word carries only
// bits BELOW the previous word's last bit.  The canonical statement of that is
// Priest's non-overlap invariant
//
//     |f_{i+1}| <= 1/2 ulp(f_i)   for i = 0, 1, 2
//
// evaluated in the MATHEMATICAL 1/2-ulp form (via frexp), NOT the length-2
// bit-form fl(hi+lo)==hi that T1.2/T2.2 use.  Two reasons the bit-form does not
// carry over: (1) it only speaks about a PAIR — QuadFloat is length-4, so we
// need the per-word chain — and (2) the bit-form fl(f_i + f_{i+1}) == f_i has
// round-to-even FALSE POSITIVES at exact ties (a word sitting exactly on the
// 1/2 ulp boundary rounds back onto f_i even though it is a genuine overlap).
// The frexp half_ulp form is exact at the tie.  This is the SAME invariant and
// the SAME half_ulp / pair_checkable machinery T3.1 uses on the renorm outputs;
// T3.2 applies it to the output of EVERY QF op.
//
// If any word violated |f_{i+1}| <= 1/2 ulp(f_i), the expansion would carry
// redundant / contradictory bits and the ~29-digit precision claim would be
// false for that op.  A violation localizes a normalization bug to that exact op
// — no reference value and no wider type is needed to SEE it.  So this test
// carries NO oracle and runs unconditionally (matching T2.2).  Accuracy-vs-
// oracle is the separate concern of T3.4.  The ONLY use of __float128 here is to
// ENRICH the inputs to full ~96-bit width (see below); that enrichment used to
// be KOKKOS_EP_HAVE_QUADMATH-gated while the invariant check was not, so on a
// Kokkos without the TPL this test ran but tested narrower inputs than it
// claimed.  The gate is gone; the enrichment always happens.
//
// TWO-TIER CLASSIFICATION (strict Priest 1/2-ulp vs QD-weak <=ulp)
// ---------------------------------------------------------------
// A subtlety unique to QF: qf_math's renorm / renorm_4 (Hida-Li-Bailey, the QD
// 2.3.24 normalization) is a quick_two_sum CASCADE.  That cascade provably yields
// only the WEAKER Shewchuk non-overlap |f_{i+1}| <= ulp(f_i), NOT the strict
// Priest |f_{i+1}| <= 1/2 ulp(f_i) (see Joldes/Muller/Popescu, "Tight & rigorous
// error bounds for basic building blocks of double-word arithmetic").  So a
// handful of renorming ops legitimately land a word in the half-open band
// (1/2 ulp, ulp] — strictly Priest-violating, but NOT corruption.
//
// classify_nonoverlap() therefore returns THREE classes:
//   NOVL_OK   — |f_{i+1}| <= 1/2 ulp(f_i) for all i (strict Priest holds)
//   NOVL_WEAK — some word in (1/2 ulp, ulp]  (QD weak-normalization band)
//   NOVL_FAIL — some word > ulp(f_i), OR a packing break (nonzero after a zero),
//               OR NaN/inf in a checkable slot  (genuine corruption — always fatal)
// NOVL_FAIL fails under EITHER gate.  Whether NOVL_WEAK fails is controlled by the
// single flag kStrictPriestGate (see its definition below): false (default) =
// Shewchuk-weak, WEAK tolerated (the posture adopted after Reet's review — QD's
// renorm is a quick_two_sum cascade and provably delivers only the weak bound, so
// testing for strict Priest would demand a stronger library than qf_math.hpp, a
// faithful QD port, claims to be; see PORT_NOTES_QF §16); true = strict Priest,
// WEAK counts as failure (the literal reading of the original T3.2 spec, kept as a
// diagnostic switch).  To CHANGE the posture, flip that one flag — do NOT loosen
// the check itself.  Every WEAK deviation is counted and its worst overlap ratio
// (1.0 = exactly 1/2 ulp, 2.0 = exactly ulp) is reported per-op regardless of the
// gate, so any drift into the (1/2 ulp, ulp] band stays visible in the log.
//
// WHY INPUTS ARE BUILT BY ORDERED FP32 DECOMPOSITION (T3.1's construction)
// -----------------------------------------------------------------------
// Successive round-to-nearest FP32 decomposition of a wider value produces a
// magnitude-ordered, non-overlapping expansion: with r_0 = v and
// w_k = fl32(r_k), r_{k+1} = r_k - w_k, each residual satisfies
// |r_{k+1}| <= 1/2 ulp(w_k), so the words are already Priest-valid.  This is
// exactly the shape add/multiply/QuadFloat(double) produce and exactly what
// renorm's quick_two_sum cascade assumes.  It matters here for a SPECIFIC
// reason: the non-renorming ops (negate, abs, mul_pwr2, copysign, fmax, fmin)
// pass their input words through UNCHANGED (up to sign / power-of-2 scaling), so
// their output is non-overlapping IF AND ONLY IF the input already was.  Feeding
// them an un-normalized input would test renorm (T3.1's job) at the same time as
// the op and could raise a spurious failure on a perfectly-correct passthrough.
// So the random pass builds each input with make_wide_input() — a ~96-bit
// __float128 value (nominal x plus sub-leading-ulp tail terms) decomposed into
// four ordered FP32 words — and the corpus pass uses the QuadFloat(double)
// constructor (itself an ordered 2–3-word decomposition), which preserves the
// exact corner-case value while still being normalized.  make_wide_input keeps
// the leading word f0 == (float)x (the tail is below 1/2 ulp of f0), so every
// domain predicate below — written against the nominal x — stays valid and QF's
// domain-guard diagnostics (which print) are never tripped.
//
// SKIP (not FAIL) CRITERIA — a result outside the invariant's domain
//   * f0 is NaN / +-inf     — non-overlap is undefined; op fed an out-of-domain
//                             input, or overflowed
//   * f0 is subnormal       — below the smallest normal FP32, the "1/2 ulp"
//                             argument degrades; a nonzero tail there is not a
//                             normalization defect
//   * f0 in the underflow tail (|f0| < 2^-100) — 1/2 ulp(f0) is itself subnormal,
//                             so the check is ill-posed (same reasoning as T3.1's
//                             kUnderflowTail and T2.2's result_checkable)
//   * input out of the op's mathematical domain (log of <=0, asin of |x|>1, ...)
//                             — skipped before the call, which also suppresses the
//                             QF domain-guard prints
//
// QF-SPECIFIC DOMAIN PREDICATES (derived from qf_math.hpp guards)
// --------------------------------------------------------------
// FP32's exponent range bounds every op (exp guards at a.f0 >= 88, log window,
// sinh/cosh/tanh caps, etc.).  ONE material RELAXATION vs T2.2: FF's sincos hit
// its Taylor iteration limit (FFCSSNR) for tiny nonzero arguments and forced a
// LOWER trig bound (|x| >= 1e-25).  QF's sincos uses eps = 1e-28 (coarser than
// u = 2^-96) and converges in ~9 terms at nq=5 with NO return-0 on the iteration
// cap (see `sincos` in qf_math.hpp), so QF has NO tiny-argument stall — the QF trig
// family needs no lower bound (0 and arbitrarily-small |x| are both fine); the
// only bound is the |a.f0| >= 1e30 "argument too large" guard.
//
// TEST STRUCTURE (mirrors T2.2)
//   Test A — every QF op returning a QuadFloat, two passes each: kRandomN
//            op-appropriate random inputs (enriched to ~96-bit width; see the
//            kRandomN note for why it is 2*10^5 not 10^6), then the
//            full corner-case corpus (corpus::unary/binary<float>, exact ctor).
//            Covers arithmetic (add/subtract/multiply/divide + ieee_add/
//            sloppy_add/divide_accurate), sqr, sqrt, the transcendentals, the
//            joint sincos/sinhcosh components, angle, and the utility ops
//            (round family, copysign/fmax/fmin/fdim, fmod/remainder, hypot).
//            The QF-only scalar ops (multiply_scalar, mul_pwr2), the ternary fma,
//            and integer pow_int run in dedicated loops after the registries
//            (mul_pwr2 is tested ONLY with b = +-2^k — it does componentwise
//            multiply with NO renorm, exact only for powers of two).
//   Test B — device tripwire: 5 representative ops (add, multiply, sqrt, exp,
//            sin) run the SAME invariant inside a Kokkos::parallel_for, results
//            copied back and checked on host.  Catches a Serial->device
//            regression the host pass cannot see.
//   Test C — corner cases as named asserts: zero, +-ulp, subnormals, +-inf, NaN.
//
// SCOPE: real QF ops only (qf_complex.hpp is out of scope — that is T3.x complex
// work).  If a violation is found this test REPORTS it (op, counts, first
// offending 4-word bit patterns) and fails — it does NOT touch qf_math.hpp
// (Rule 4: test authors do not patch libraries).
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 3, "T3.2: non-overlap
// invariant checks for QF"; the T3.1 DONE block (source of half_ulp /
// nonoverlap_holds / ordered-decomposition) and the T2.2 DONE block (structural
// template); "The six test layers" layer 2.
// ============================================================================

#include "test_utils.hpp"
#include "corpus.hpp"
#include <qf_math.hpp>

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

// QF types live in Kokkos::Experimental; introduce a qf:: alias for readability
// (matches qf_eft_test.cpp).
namespace qf = Kokkos::Experimental;

// ----------------------------------------------------------------------------
// The Priest length-4 non-overlap invariant and its domain.
// Ported verbatim from qf_eft_test.cpp (T3.1) so both layers speak the SAME
// invariant on the SAME machinery (a mirror could drift; a shared form cannot —
// but these are header-free helpers, so we copy rather than cross-include a .cpp).
// ----------------------------------------------------------------------------

// Mathematical 1/2 ulp of a normal float, from its binade exponent. frexp writes
// x = m * 2^e with m in [0.5, 1); for FP32's 24-bit significand ulp(x) = 2^(e-24),
// so 1/2 ulp = 2^(e-25). NOT the bit-form fl(f_i+f_{i+1})==f_i (round-to-even
// false positives at ties) — the plan specifies the mathematical form.
inline double half_ulp(float x) {
  if (x == 0.0f) return 0.0;
  int e;
  std::frexp((double)x, &e);
  return std::ldexp(1.0, e - 25);
}

// UNDERFLOW-TAIL gate: |f_i| >= 2^-102 is required for 1/2 ulp(f_i) to be a
// normal float; below that subnormal quantization makes the comparison ill-posed.
// Gated a hair higher at 2^-100 for margin (identical to T3.1 / T2.2).
static constexpr float kUnderflowTail = 0x1p-100f;
inline bool pair_checkable(float hi) {
  if (!std::isfinite(hi)) return false;
  if (hi == 0.0f) return false;                                       // trailing zero
  if (std::fabs(hi) < std::numeric_limits<float>::min()) return false;  // subnormal
  if (std::fabs(hi) < kUnderflowTail) return false;                  // underflow tail
  return true;
}

// GATE POSTURE (strict Priest vs QD weak-normalization). The plan specifies the
// STRICT Priest length-4 invariant |f_{i+1}| <= 1/2 ulp(f_i). Empirically (see the
// T3.2 report), qf_math's QD renorm (Hida-Li-Bailey renorm / renorm_4) provably
// delivers only the WEAKER Shewchuk non-overlap |f_{i+1}| <= ulp(f_i): a handful
// of ops (add, subtract, divide_accurate, fmod, remainder, fdim, multiply_scalar,
// pow_int, sqr, log, cos, sincos) occasionally (~1e-6) land a word in (1/2 ulp,
// ulp]. All such deviations stay within ulp (ratio < 2.0); NONE exceed it and NONE
// break packing. This is inherent to QD's fast renormalization, not a per-op bug,
// and cannot be tightened without replacing the renorm (a library change barred by
// Rule 4). See PORT_NOTES_QF §16 for the algorithmic proof and the posture ruling.
//
// kStrictPriestGate selects which invariant is FATAL:
//   false (default, per Reet's review -- PORT_NOTES_QF §16): QD weak-normalization
//         is the correct posture. renorm is a quick_two_sum cascade that cannot
//         beat |f_{i+1}| <= ulp(f_i), so a (1/2 ulp, ulp] word is expected, not a
//         defect; only >ulp overlaps / packing breaks / NaN·inf leaks are fatal.
//         Strict deviations are still counted and reported, just non-fatal. ctest
//         is GREEN.
//   true  (diagnostic switch, literal reading of the original T3.2 spec): strict
//         Priest -- a (1/2 ulp, ulp] word is a failure. ctest is RED on the QD
//         weak-normalization ops above (kept for regression diagnostics only).
// Either way the strict 1/2-ulp check runs on every output and every deviation is
// tallied with its overlap ratio -- nothing is hidden. Do NOT silently loosen: to
// change the gate, flip this one flag (a visible, reviewed decision), not the check.
static constexpr bool kStrictPriestGate = false;

// Three-level non-overlap classification on (b0,b1,b2,b3):
//   NOVL_OK    every checkable pair satisfies strict Priest |f_{i+1}| <= 1/2 ulp(f_i)
//   NOVL_WEAK  some pair is in (1/2 ulp, ulp]  -- QD weak-normalization (see above)
//   NOVL_FAIL  some pair exceeds ulp, OR the expansion is not packed -- corruption
// *skips counts pairs whose leading word is out of the well-posed domain. worst_ratio
// (if non-null) accumulates max |f_{i+1}| / (1/2 ulp(f_i)) over checkable pairs:
// 1.0 == exactly 1/2 ulp, 2.0 == exactly ulp. worst_idx receives the offending i.
enum NovlClass { NOVL_OK = 0, NOVL_WEAK = 1, NOVL_FAIL = 2 };
inline NovlClass classify_nonoverlap(float b0, float b1, float b2, float b3,
                                     int* skips, double* worst_ratio, int* worst_idx) {
  const float b[4] = {b0, b1, b2, b3};
  // packing: no nonzero word may follow a zero word.
  bool seen_zero = false;
  for (int i = 0; i < 4; ++i) {
    if (b[i] == 0.0f) seen_zero = true;
    else if (seen_zero) return NOVL_FAIL;   // nonzero after zero -> not packed
  }
  NovlClass cls = NOVL_OK;
  for (int i = 0; i < 3; ++i) {
    if (b[i] == 0.0f) break;                 // trailing zeros: invariant trivially holds
    if (!pair_checkable(b[i])) { if (skips) ++*skips; continue; }
    const double hu  = half_ulp(b[i]);
    const double nxt = std::fabs((double)b[i + 1]);
    if (nxt > hu) {
      const double ratio = nxt / hu;         // 1.0 = 1/2 ulp, 2.0 = ulp
      if (worst_ratio && ratio > *worst_ratio) { *worst_ratio = ratio;
                                                 if (worst_idx) *worst_idx = i; }
      if (nxt > 2.0 * hu) return NOVL_FAIL;  // exceeds ulp -> genuine overlap
      cls = NOVL_WEAK;                        // in (1/2 ulp, ulp] -> QD weak-normalization
    }
  }
  return cls;
}

// Whole-QuadFloat gate: should we CHECK this result (vs SKIP it)? Mirrors T2.2's
// result_checkable, keyed on the leading word f0. A pure-zero result (f0==0,
// all-zero) IS checked and trivially passes; a packing violation with f0==0 but
// a nonzero lower word is a genuine failure (NOVL_FAIL) caught by classify_nonoverlap.
inline bool result_checkable(const qf::QuadFloat& d) {
  if (std::isnan(d.f0)) return false;
  if (std::isinf(d.f0)) return false;
  if (d.f0 != 0.0f && std::isfinite(d.f0) &&
      std::fabs(d.f0) < std::numeric_limits<float>::min())
    return false;                       // subnormal leading word
  if (d.f0 != 0.0f && std::fabs(d.f0) < kUnderflowTail)
    return false;                       // near-underflow tail: check ill-posed
  return true;
}

inline NovlClass classify_result(const qf::QuadFloat& d, double* ratio, int* idx) {
  return classify_nonoverlap(d.f0, d.f1, d.f2, d.f3, nullptr, ratio, idx);
}

// Gate-aware pass/fail (used by the device tripwire and corner-case checks, which
// don't need the per-op weak breakdown). Under the strict gate a WEAK result fails;
// under the weak gate only genuine corruption (NOVL_FAIL) fails.
inline bool invariant_holds(const qf::QuadFloat& d) {
  NovlClass c = classify_result(d, nullptr, nullptr);
  return kStrictPriestGate ? (c == NOVL_OK) : (c != NOVL_FAIL);
}

// ----------------------------------------------------------------------------
// Input construction. Both paths yield an ORDERED, non-overlapping expansion so
// non-renorming ops are exercised, not renorm (T3.1's job). See block header.
// ----------------------------------------------------------------------------

// Enrich a nominal double x to a full ~96-bit ordered QuadFloat: add sub-leading
// -ulp tail terms (relative 2^-28 / 2^-56 / 2^-84, each below 1/2 ulp of f0 so
// f0 stays == (float)x), then decompose the wide value into four ordered FP32
// words by successive round-to-nearest (the T3.1 construction). This used to have
// a no-__float128 fallback to the ordered QuadFloat(double) split (~2-3 words);
// the enrichment is unconditional now, so the fallback is gone with the gate.
inline qf::QuadFloat make_wide_input(double x, std::mt19937_64& g) {
  std::uniform_real_distribution<double> dt(-1.0, 1.0);
  float128 v = (float128)x;
  v += (float128)x * (float128)(dt(g) * 0x1p-28);   // ~2^-28 relative (fills word 1)
  v += (float128)x * (float128)(dt(g) * 0x1p-56);   // ~2^-56 relative (fills word 2)
  v += (float128)x * (float128)(dt(g) * 0x1p-84);   // ~2^-84 relative (fills word 3)
  float w0 = (float)v;             float128 r = v - (float128)w0;
  float w1 = (float)r;             r -= (float128)w1;
  float w2 = (float)r;             r -= (float128)w2;
  float w3 = (float)r;
  return qf::QuadFloat(w0, w1, w2, w3);
}

// ----------------------------------------------------------------------------
// Per-op counters and failure-sample printers (4-word bit patterns, 0x%08x each).
// ----------------------------------------------------------------------------
// failures = NOVL_FAIL (>ulp overlap or packing break): fatal under EITHER gate.
// weak     = NOVL_WEAK (in (1/2 ulp, ulp]): QD weak-normalization; fatal only under
//            the strict Priest gate. worst_ratio/worst_idx: worst weak deviation.
struct InvCount { long tested = 0; long skipped = 0; long failures = 0;
                  long weak = 0; double worst_ratio = 0.0; int worst_idx = -1; };

struct InvSummary {
  std::string name;
  long tested = 0, skipped = 0, failures = 0, weak = 0;
  double worst_ratio = 0.0; int worst_idx = -1;
};

// Grade one checkable result into the per-op counters; return its class and (via
// out-params) the worst weak-overlap ratio / index for sample printing. Used by
// the special-form loops (multiply_scalar / mul_pwr2 / fma / pow_int) whose bespoke
// input plumbing doesn't fit the UnaryOp/BinaryOp registries.
static NovlClass grade(const qf::QuadFloat& d, InvCount& c, double* ratio, int* idx) {
  double r = 0.0; int i = -1;
  NovlClass cl = classify_result(d, &r, &i);
  if (cl == NOVL_FAIL) ++c.failures;
  else if (cl == NOVL_WEAK) {
    ++c.weak;
    if (r > c.worst_ratio) { c.worst_ratio = r; c.worst_idx = i; }
  }
  if (ratio) *ratio = r;
  if (idx)   *idx = i;
  return cl;
}

static uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(float));
  return b;
}

static void print_fail_unary(const char* op, double x, const qf::QuadFloat& d) {
  std::printf("    FAIL %-12s x=%.9g  f=[%.9g %.9g %.9g %.9g]  "
              "bits=[0x%08x 0x%08x 0x%08x 0x%08x]\n",
              op, x, d.f0, d.f1, d.f2, d.f3,
              fbits(d.f0), fbits(d.f1), fbits(d.f2), fbits(d.f3));
}

static void print_fail_binary(const char* op, double a, double b,
                              const qf::QuadFloat& d) {
  std::printf("    FAIL %-12s a=%.9g b=%.9g  f=[%.9g %.9g %.9g %.9g]  "
              "bits=[0x%08x 0x%08x 0x%08x 0x%08x]\n",
              op, a, b, d.f0, d.f1, d.f2, d.f3,
              fbits(d.f0), fbits(d.f1), fbits(d.f2), fbits(d.f3));
}

// Sample line for a QD weak-normalization deviation (in (1/2 ulp, ulp]): tagged
// WEAK, prefixed by whether the strict gate treats it as a failure.
static void print_weak(const char* op, double x, const qf::QuadFloat& d,
                       double ratio, int idx) {
  std::printf("    %s %-12s x=%.9g  ratio=%.6f @idx%d (1.0=1/2ulp,2.0=ulp)  "
              "f=[%.9g %.9g %.9g %.9g]\n",
              kStrictPriestGate ? "FAIL(weak-norm)" : "WEAK-NORM",
              op, x, ratio, idx, d.f0, d.f1, d.f2, d.f3);
}

// ----------------------------------------------------------------------------
// Op registries. apply takes a (pre-built, normalized) QuadFloat and returns the
// op's QuadFloat result; in_domain gates on the nominal double x (suppressing
// out-of-domain calls and thus QF's domain-guard prints); gen draws the nominal
// input. All QF ops are KOKKOS_INLINE_FUNCTION, host-callable directly.
// ----------------------------------------------------------------------------
using QFUnary  = std::function<qf::QuadFloat(qf::QuadFloat)>;
using QFBinary = std::function<qf::QuadFloat(qf::QuadFloat, qf::QuadFloat)>;
using Dom1     = std::function<bool(double)>;
using Dom2     = std::function<bool(double, double)>;

struct UnaryOp {
  const char* name;
  Dom1        in_domain;
  InputDist   gen;
  QFUnary     apply;
};

struct BinaryOp {
  const char* name;
  Dom2        in_domain;
  InputDist   gen_a;
  InputDist   gen_b;
  QFBinary    apply;
};

// Common domain predicates.
static const Dom1 dom_any     = [](double x) { return std::isfinite(x); };
static const Dom1 dom_nonneg  = [](double x) { return std::isfinite(x) && x >= 0.0; };
// a*a must stay finite in FP32 (FLT_MAX ~3.4e38 -> |x| < ~1.8e19); cap at 1e18.
static const Dom1 dom_sqfinite = [](double x) { return std::isfinite(x) && std::fabs(x) < 1e18; };
// Trig: QF sincos has NO tiny-argument stall (see header) — no lower bound.
// Only the |a.f0| >= 1e30 "too large" guard bounds it; cap at 1e29 with margin.
static const Dom1 dom_trig    = [](double x) { return std::isfinite(x) && std::fabs(x) < 1e29; };

// ----------------------------------------------------------------------------
// One input check (shared by random and corpus passes). q is the pre-built input.
// ----------------------------------------------------------------------------
static void check_unary(const UnaryOp& op, double x, const qf::QuadFloat& q,
                        InvCount& c, int& samples_left) {
  if (!op.in_domain(x)) { ++c.skipped; return; }
  qf::QuadFloat d = op.apply(q);
  if (!result_checkable(d)) { ++c.skipped; return; }
  ++c.tested;
  double ratio = 0.0; int idx = -1;
  NovlClass cl = classify_result(d, &ratio, &idx);
  if (cl == NOVL_FAIL) {
    ++c.failures;
    if (samples_left > 0) { print_fail_unary(op.name, x, d); --samples_left; }
  } else if (cl == NOVL_WEAK) {
    ++c.weak; if (ratio > c.worst_ratio) { c.worst_ratio = ratio; c.worst_idx = idx; }
    if (samples_left > 0) { print_weak(op.name, x, d, ratio, idx); --samples_left; }
  }
}

static void check_binary(const BinaryOp& op, double a, double b,
                         const qf::QuadFloat& qa, const qf::QuadFloat& qb,
                         InvCount& c, int& samples_left) {
  if (!op.in_domain(a, b)) { ++c.skipped; return; }
  qf::QuadFloat d = op.apply(qa, qb);
  if (!result_checkable(d)) { ++c.skipped; return; }
  ++c.tested;
  double ratio = 0.0; int idx = -1;
  NovlClass cl = classify_result(d, &ratio, &idx);
  if (cl == NOVL_FAIL) {
    ++c.failures;
    if (samples_left > 0) { print_fail_binary(op.name, a, b, d); --samples_left; }
  } else if (cl == NOVL_WEAK) {
    ++c.weak; if (ratio > c.worst_ratio) { c.worst_ratio = ratio; c.worst_idx = idx; }
    if (samples_left > 0) { print_weak(op.name, a, d, ratio, idx); --samples_left; }
  }
}

// Corpus flags per spec: zero on, inf off, nan off (subnormals default on).
static corpus::CorpusFlags corpus_flags() {
  corpus::CorpusFlags f;
  f.include_zero = true;
  f.include_inf  = false;
  f.include_nan  = false;
  return f;
}

// The plan calls for 10^6 random inputs/op. QF ops are ~15-20x costlier than FF
// (4-word renorm cascades, long-division divide, Taylor/Newton loops) on the Serial
// backend, so 10^6 gives a ~13.5 min ctest -- untenable. Reduced to 2*10^5 (the
// plan's own "tune corpus sizes for wall time" subtask): ~5.5 min, a 2.5x speedup
// that still surfaces the systemic QD weak-normalization across 12 ops -- the
// higher-rate ops remainder (~3.5e-5) and fmod (~2e-5) hit deterministically under
// the fixed per-op seeds, so the strict-gate RED verdict is reproducible. Documented
// deviation -- see the T3.2 report.
static constexpr int kRandomN = 200'000;

static InvSummary run_unary(const UnaryOp& op, uint64_t seed) {
  InvCount c;
  int samples_left = 3;

  // Pass (a): kRandomN op-appropriate random inputs, enriched to ~96-bit width.
  {
    std::mt19937_64 gen(seed);
    for (int i = 0; i < kRandomN; ++i) {
      double x = op.gen(gen);
      qf::QuadFloat q = make_wide_input(x, gen);
      check_unary(op, x, q, c, samples_left);
    }
  }
  // Pass (b): full corner-case corpus (exact ordered QuadFloat(double), so corner
  // semantics are preserved). Folds in the PORT_NOTES regression families.
  {
    std::vector<float> xs = corpus::unary<float>(corpus_flags());
    for (float x : xs) check_unary(op, (double)x, qf::QuadFloat((double)x), c, samples_left);
  }

  std::printf("  %-18s tested=%-9ld skipped=%-9ld failures=%ld weak=%ld worst=%.4f\n",
              op.name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
  return InvSummary{op.name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio, c.worst_idx};
}

static InvSummary run_binary(const BinaryOp& op, uint64_t seed) {
  InvCount c;
  int samples_left = 3;

  // Pass (a): kRandomN random (a, b) pairs from one reproducible engine.
  {
    std::mt19937_64 gen(seed);
    for (int i = 0; i < kRandomN; ++i) {
      double a = op.gen_a(gen);
      qf::QuadFloat qa = make_wide_input(a, gen);
      double b = op.gen_b(gen);
      qf::QuadFloat qb = make_wide_input(b, gen);
      check_binary(op, a, b, qa, qb, c, samples_left);
    }
  }
  // Pass (b): full corner-case corpus (exact ordered ctor).
  {
    std::vector<std::pair<float, float>> ps = corpus::binary<float>(corpus_flags());
    for (auto& p : ps)
      check_binary(op, (double)p.first, (double)p.second,
                   qf::QuadFloat((double)p.first), qf::QuadFloat((double)p.second),
                   c, samples_left);
  }

  std::printf("  %-18s tested=%-9ld skipped=%-9ld failures=%ld weak=%ld worst=%.4f\n",
              op.name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
  return InvSummary{op.name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio, c.worst_idx};
}

// ----------------------------------------------------------------------------
// Device tripwire (Test B). Same invariant, computed on device for 5 ops. Input
// words are built on host (make_wide_input) and shipped as four Views; output
// words copied back and checked on host. Mirrors qf_eft_test.cpp's device block.
// ----------------------------------------------------------------------------
template <typename DeviceOp>
static InvSummary device_unary(const char* name, int n, uint64_t seed,
                               const InputDist& gen, const Dom1& in_domain,
                               DeviceOp op) {
  using exec_space = Kokkos::DefaultExecutionSpace;

  std::vector<double> hx(n);
  Kokkos::View<float*, exec_space> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
  auto ha0 = Kokkos::create_mirror_view(a0);
  auto ha1 = Kokkos::create_mirror_view(a1);
  auto ha2 = Kokkos::create_mirror_view(a2);
  auto ha3 = Kokkos::create_mirror_view(a3);
  {
    std::mt19937_64 g(seed);
    for (int i = 0; i < n; ++i) {
      hx[i] = gen(g);
      qf::QuadFloat q = make_wide_input(hx[i], g);
      ha0(i) = q.f0; ha1(i) = q.f1; ha2(i) = q.f2; ha3(i) = q.f3;
    }
  }
  Kokkos::deep_copy(a0, ha0); Kokkos::deep_copy(a1, ha1);
  Kokkos::deep_copy(a2, ha2); Kokkos::deep_copy(a3, ha3);

  Kokkos::View<float*, exec_space> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  Kokkos::parallel_for("qf_nonoverlap_dev_unary", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(qf::QuadFloat(a0(i), a1(i), a2(i), a3(i)));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();

  auto ho0 = Kokkos::create_mirror_view(o0);
  auto ho1 = Kokkos::create_mirror_view(o1);
  auto ho2 = Kokkos::create_mirror_view(o2);
  auto ho3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(ho0, o0); Kokkos::deep_copy(ho1, o1);
  Kokkos::deep_copy(ho2, o2); Kokkos::deep_copy(ho3, o3);

  InvCount c; int samples_left = 3;
  for (int i = 0; i < n; ++i) {
    if (!in_domain(hx[i])) { ++c.skipped; continue; }
    qf::QuadFloat d(ho0(i), ho1(i), ho2(i), ho3(i));
    if (!result_checkable(d)) { ++c.skipped; continue; }
    ++c.tested;
    double ratio = 0.0; int idx = -1;
    NovlClass cl = grade(d, c, &ratio, &idx);
    if (cl == NOVL_FAIL && samples_left > 0) { print_fail_unary(name, hx[i], d); --samples_left; }
    else if (cl == NOVL_WEAK && samples_left > 0) { print_weak(name, hx[i], d, ratio, idx); --samples_left; }
  }
  std::printf("  [device] %-12s tested=%-8ld skipped=%-8ld failures=%ld weak=%ld worst=%.4f\n",
              name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
  return InvSummary{std::string("device:") + name, c.tested, c.skipped, c.failures,
                    c.weak, c.worst_ratio, c.worst_idx};
}

template <typename DeviceOp>
static InvSummary device_binary(const char* name, int n, uint64_t seed,
                                const InputDist& gen_a, const InputDist& gen_b,
                                const Dom2& in_domain, DeviceOp op) {
  using exec_space = Kokkos::DefaultExecutionSpace;

  std::vector<double> ha(n), hb(n);
  Kokkos::View<float*, exec_space> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
  Kokkos::View<float*, exec_space> b0("b0", n), b1("b1", n), b2("b2", n), b3("b3", n);
  auto ma0 = Kokkos::create_mirror_view(a0); auto ma1 = Kokkos::create_mirror_view(a1);
  auto ma2 = Kokkos::create_mirror_view(a2); auto ma3 = Kokkos::create_mirror_view(a3);
  auto mb0 = Kokkos::create_mirror_view(b0); auto mb1 = Kokkos::create_mirror_view(b1);
  auto mb2 = Kokkos::create_mirror_view(b2); auto mb3 = Kokkos::create_mirror_view(b3);
  {
    std::mt19937_64 g(seed);
    for (int i = 0; i < n; ++i) {
      ha[i] = gen_a(g); qf::QuadFloat qa = make_wide_input(ha[i], g);
      hb[i] = gen_b(g); qf::QuadFloat qb = make_wide_input(hb[i], g);
      ma0(i) = qa.f0; ma1(i) = qa.f1; ma2(i) = qa.f2; ma3(i) = qa.f3;
      mb0(i) = qb.f0; mb1(i) = qb.f1; mb2(i) = qb.f2; mb3(i) = qb.f3;
    }
  }
  Kokkos::deep_copy(a0, ma0); Kokkos::deep_copy(a1, ma1);
  Kokkos::deep_copy(a2, ma2); Kokkos::deep_copy(a3, ma3);
  Kokkos::deep_copy(b0, mb0); Kokkos::deep_copy(b1, mb1);
  Kokkos::deep_copy(b2, mb2); Kokkos::deep_copy(b3, mb3);

  Kokkos::View<float*, exec_space> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  Kokkos::parallel_for("qf_nonoverlap_dev_binary", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(qf::QuadFloat(a0(i), a1(i), a2(i), a3(i)),
                           qf::QuadFloat(b0(i), b1(i), b2(i), b3(i)));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();

  auto ho0 = Kokkos::create_mirror_view(o0);
  auto ho1 = Kokkos::create_mirror_view(o1);
  auto ho2 = Kokkos::create_mirror_view(o2);
  auto ho3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(ho0, o0); Kokkos::deep_copy(ho1, o1);
  Kokkos::deep_copy(ho2, o2); Kokkos::deep_copy(ho3, o3);

  InvCount c; int samples_left = 3;
  for (int i = 0; i < n; ++i) {
    if (!in_domain(ha[i], hb[i])) { ++c.skipped; continue; }
    qf::QuadFloat d(ho0(i), ho1(i), ho2(i), ho3(i));
    if (!result_checkable(d)) { ++c.skipped; continue; }
    ++c.tested;
    double ratio = 0.0; int idx = -1;
    NovlClass cl = grade(d, c, &ratio, &idx);
    if (cl == NOVL_FAIL && samples_left > 0) { print_fail_binary(name, ha[i], hb[i], d); --samples_left; }
    else if (cl == NOVL_WEAK && samples_left > 0) { print_weak(name, ha[i], d, ratio, idx); --samples_left; }
  }
  std::printf("  [device] %-12s tested=%-8ld skipped=%-8ld failures=%ld weak=%ld worst=%.4f\n",
              name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
  return InvSummary{std::string("device:") + name, c.tested, c.skipped, c.failures,
                    c.weak, c.worst_ratio, c.worst_idx};
}

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("=== qf_nonoverlap_test (T3.2): Priest length-4 non-overlap "
                "|f_{i+1}| <= 1/2 ulp(f_i) for every QF op ===\n");
    std::printf("Oracle-independent (mathematical 1/2-ulp check). Execution space: %s\n",
                Kokkos::DefaultExecutionSpace::name());
    std::printf("Inputs enriched to ~96-bit width via __float128 ordered decomposition.\n\n");

    std::vector<InvSummary> summary;

    // ---- Unary op registry -------------------------------------------------
    // Bounds derived from qf_math.hpp guards. NOTE the trig relaxation vs T2.2:
    // dom_trig has NO lower magnitude bound (QF sincos has no tiny-arg stall).
    const std::vector<UnaryOp> unary_ops = {
      {"negate",   dom_any,      uniform(-1e8, 1e8),  [](qf::QuadFloat a){ return qf::negate(a); }},
      {"abs",      dom_any,      uniform(-1e8, 1e8),  [](qf::QuadFloat a){ return qf::abs(a); }},
      {"sqr",      dom_sqfinite, uniform(-1e6, 1e6),  [](qf::QuadFloat a){ return qf::sqr(a); }},
      {"sqrt",     dom_nonneg,   uniform(0.0, 1e8),   [](qf::QuadFloat a){ return qf::sqrt(a); }},
      // round family: keep |x| < 1e13 so results are exact integers well in range.
      {"round_to_nearest_int", [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                   uniform(-1e6, 1e6),  [](qf::QuadFloat a){ return qf::round_to_nearest_int(a); }},
      {"ceil",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                   uniform(-1e6, 1e6),  [](qf::QuadFloat a){ return qf::ceil(a); }},
      {"floor",    [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                   uniform(-1e6, 1e6),  [](qf::QuadFloat a){ return qf::floor(a); }},
      {"trunc",    [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                   uniform(-1e6, 1e6),  [](qf::QuadFloat a){ return qf::trunc(a); }},
      {"round",    [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
                   uniform(-1e6, 1e6),  [](qf::QuadFloat a){ return qf::round(a); }},
      // exp-family: qf exp guards at a.f0 >= 88 (FP32 ln-range); below -88 returns 0.
      {"exp",      [](double x){ return std::isfinite(x) && x < 88.0; },
                   uniform(-88.0, 87.5),   [](qf::QuadFloat a){ return qf::exp(a); }},
      {"expm1",    [](double x){ return std::isfinite(x) && x < 88.0; },
                   uniform(-5.0, 5.0),     [](qf::QuadFloat a){ return qf::expm1(a); }},
      // exp2(a)=e^(a*ln2): a*ln2 < 88 -> |a| < ~127. exp10: |a| < ~38.
      {"exp2",     [](double x){ return std::isfinite(x) && std::fabs(x) < 126.0; },
                   uniform(-120.0, 120.0), [](qf::QuadFloat a){ return qf::exp2(a); }},
      {"exp10",    [](double x){ return std::isfinite(x) && std::fabs(x) < 38.0; },
                   uniform(-37.0, 37.0),   [](qf::QuadFloat a){ return qf::exp10(a); }},
      // log-family window [1e-34, 1e34] normal: keeps the internal exp Newton step
      // (arg ~ -log x) inside exp's 88-guard (log(1e34) ~ 78 < 88).
      {"log",      [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
                   uniform(1e-6, 1e6),   [](qf::QuadFloat a){ return qf::log(a); }},
      {"log2",     [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
                   uniform(1e-6, 1e6),   [](qf::QuadFloat a){ return qf::log2(a); }},
      {"log10",    [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
                   uniform(1e-6, 1e6),   [](qf::QuadFloat a){ return qf::log10(a); }},
      {"log1p",    [](double x){ return std::isfinite(x) && x > -1.0 && (1.0 + x) >= 1e-34 && x <= 1e34; },
                   uniform(-0.9, 1e6),   [](qf::QuadFloat a){ return qf::log1p(a); }},
      // trig: no lower bound (QF sincos converges for arbitrarily small args).
      {"sin",      dom_trig, uniform(-1000.0, 1000.0), [](qf::QuadFloat a){ return qf::sin(a); }},
      {"cos",      dom_trig, uniform(-1000.0, 1000.0), [](qf::QuadFloat a){ return qf::cos(a); }},
      {"tan",      dom_trig, uniform(-1000.0, 1000.0), [](qf::QuadFloat a){ return qf::tan(a); }},
      {"asin",     [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0; },
                   uniform(-1.0, 1.0),   [](qf::QuadFloat a){ return qf::asin(a); }},
      {"acos",     [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0; },
                   uniform(-1.0, 1.0),   [](qf::QuadFloat a){ return qf::acos(a); }},
      // atan(a)=angle(1,a): angle forms sqrt(1+a^2); a*a must stay finite -> |a|<1e18.
      {"atan",     [](double x){ return std::isfinite(x) && std::fabs(x) < 1e18; },
                   uniform(-1e6, 1e6),   [](qf::QuadFloat a){ return qf::atan(a); }},
      // sinh/cosh use exp(a) for |a|>=0.5; exp guards at 88 -> |a| < 88.
      {"sinh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 88.0; },
                   uniform(-80.0, 80.0), [](qf::QuadFloat a){ return qf::sinh(a); }},
      {"cosh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 88.0; },
                   uniform(-80.0, 80.0), [](qf::QuadFloat a){ return qf::cosh(a); }},
      // tanh via expm1(2a): exp(2a) must not overflow -> |a| < 44.
      {"tanh",     [](double x){ return std::isfinite(x) && std::fabs(x) < 44.0; },
                   uniform(-40.0, 40.0), [](qf::QuadFloat a){ return qf::tanh(a); }},
      // asinh/acosh reduce to log(x + sqrt(x^2 +/- 1)); cap |x| < 1e18.
      {"asinh",    [](double x){ return std::isfinite(x) && std::fabs(x) < 1e18; },
                   uniform(-1e6, 1e6),   [](qf::QuadFloat a){ return qf::asinh(a); }},
      {"acosh",    [](double x){ return std::isfinite(x) && x >= 1.0 && x < 1e18; },
                   uniform(1.0, 1e6),    [](qf::QuadFloat a){ return qf::acosh(a); }},
      {"atanh",    [](double x){ return std::isfinite(x) && std::fabs(x) < 1.0; },
                   uniform(-0.999, 0.999), [](qf::QuadFloat a){ return qf::atanh(a); }},
    };

    // Two-output ops, tested as one "op" per output component.
    // qf sincos(a, sin_a, cos_a)  and  sinhcosh(a, sinh_a, cosh_a)  — NOTE the
    // out-param order (sin first, sinh first), the reverse of ff_math.hpp's.
    const std::vector<UnaryOp> two_out_ops = {
      {"sincos.sin",    dom_trig, uniform(-1000.0, 1000.0),
                        [](qf::QuadFloat a){ qf::QuadFloat s, c; qf::sincos(a, s, c); return s; }},
      {"sincos.cos",    dom_trig, uniform(-1000.0, 1000.0),
                        [](qf::QuadFloat a){ qf::QuadFloat s, c; qf::sincos(a, s, c); return c; }},
      {"sinhcosh.sinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 88.0; },
                        uniform(-80.0, 80.0),
                        [](qf::QuadFloat a){ qf::QuadFloat s, c; qf::sinhcosh(a, s, c); return s; }},
      {"sinhcosh.cosh", [](double x){ return std::isfinite(x) && std::fabs(x) < 88.0; },
                        uniform(-80.0, 80.0),
                        [](qf::QuadFloat a){ qf::QuadFloat s, c; qf::sinhcosh(a, s, c); return c; }},
    };

    // ---- Binary op registry ------------------------------------------------
    const Dom2 dom2_any = [](double a, double b){ return std::isfinite(a) && std::isfinite(b); };
    const Dom2 dom2_bnz = [](double a, double b){ return std::isfinite(a) && std::isfinite(b) && b != 0.0; };
    // fmod/remainder: q=a/b then trunc/nint(q); keep |a| < |b|*1e13 so the quotient
    // and b*q stay in range (extreme-ratio corpus pairs are skipped, not fed).
    const Dom2 dom2_modbnz = [](double a, double b){
      return std::isfinite(a) && std::isfinite(b) && b != 0.0 &&
             std::fabs(a) < std::fabs(b) * 1e13;
    };
    // pow(a,b)=exp(b*log a): base positive in the normal log window; gate on the
    // predicted exponent |b*ln a| < 88 so only representable results are fed.
    const Dom2 dom2_powpos = [](double a, double b){
      return std::isnormal(a) && std::isfinite(b) && a >= 1e-34 && a <= 1e34 &&
             std::fabs(b * std::log(a)) < 88.0;
    };
    // atan2(y,x)/angle(x,y): sqrt(x^2+y^2) must stay finite -> cap max mag < 1e18.
    // NO subnormal-tiny floor is needed (QF sincos has no FFCSSNR stall) — a
    // relaxation vs T2.2's dom2_atan2. (0,0) is defined (returns 0).
    const Dom2 dom2_angle = [](double a, double b){
      if (!(std::isfinite(a) && std::isfinite(b))) return false;
      return std::fmax(std::fabs(a), std::fabs(b)) < 1e18;
    };

    const std::vector<BinaryOp> binary_ops = {
      {"add",       dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::add(a, b); }},
      {"subtract",  dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::subtract(a, b); }},
      {"ieee_add",  dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::ieee_add(a, b); }},
      {"sloppy_add",dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::sloppy_add(a, b); }},
      {"multiply",  dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::multiply(a, b); }},
      {"divide",    dom2_bnz, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::divide(a, b); }},
      {"divide_accurate", dom2_bnz, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::divide_accurate(a, b); }},
      {"pow",       dom2_powpos, uniform(0.1, 20.0), uniform(-6.0, 6.0),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::pow(a, b); }},
      {"angle",     dom2_angle, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::angle(a, b); }},
      {"atan2",     dom2_angle, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::atan2(a, b); }},
      {"hypot",     dom2_angle, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::hypot(a, b); }},
      {"fmod",      dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fmod(a, b); }},
      {"remainder", dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::remainder(a, b); }},
      {"copysign",  dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::copysign(a, b); }},
      {"fmax",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fmax(a, b); }},
      {"fmin",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fmin(a, b); }},
      {"fdim",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
                    [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fdim(a, b); }},
    };

    // -- Test A: every op ----------------------------------------------------
    std::printf("[Test A] per-op invariant (%d random + full corpus)\n", kRandomN);
    uint64_t seed = 12345ULL;
    for (const auto& op : unary_ops)   summary.push_back(run_unary(op,  seed++));
    for (const auto& op : two_out_ops) summary.push_back(run_unary(op,  seed++));
    for (const auto& op : binary_ops)  summary.push_back(run_binary(op, seed++));

    // multiply_scalar(a, float b): renorms, so any finite b is fine.
    {
      InvCount c; int samples_left = 3;
      InputDist gx = uniform(-1e6, 1e6), gb = uniform(-1e6, 1e6);
      std::mt19937_64 gen(seed++);
      for (int i = 0; i < kRandomN; ++i) {
        double x = gx(gen);
        qf::QuadFloat q = make_wide_input(x, gen);
        float b = (float)gb(gen);
        if (!std::isfinite(x) || !std::isfinite(b)) { ++c.skipped; continue; }
        qf::QuadFloat d = qf::multiply_scalar(q, b);
        if (!result_checkable(d)) { ++c.skipped; continue; }
        ++c.tested;
        double ratio = 0.0; int idx = -1;
        NovlClass cl = grade(d, c, &ratio, &idx);
        if (cl == NOVL_FAIL && samples_left > 0) { print_fail_unary("multiply_scalar", x, d); --samples_left; }
        else if (cl == NOVL_WEAK && samples_left > 0) { print_weak("multiply_scalar", x, d, ratio, idx); --samples_left; }
      }
      std::printf("  %-18s tested=%-9ld skipped=%-9ld failures=%ld weak=%ld worst=%.4f\n",
                  "multiply_scalar", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
      summary.push_back(InvSummary{"multiply_scalar", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio, c.worst_idx});
    }

    // mul_pwr2(a, float b): componentwise multiply with NO renorm — non-overlap
    // is preserved ONLY when b is a power of two (any other b breaks the ordering).
    // So we test it EXCLUSIVELY with b = +-2^k. This is the correct, tightest use.
    {
      InvCount c; int samples_left = 3;
      InputDist gx = uniform(-1e8, 1e8);
      std::mt19937_64 gen(seed++);
      std::uniform_int_distribution<int> dk(-40, 40);
      std::uniform_int_distribution<int> dsgn(0, 1);
      for (int i = 0; i < kRandomN; ++i) {
        double x = gx(gen);
        qf::QuadFloat q = make_wide_input(x, gen);
        float b = std::ldexp(1.0f, dk(gen)) * (dsgn(gen) ? 1.0f : -1.0f);  // +-2^k
        if (!std::isfinite(x)) { ++c.skipped; continue; }
        qf::QuadFloat d = qf::mul_pwr2(q, b);
        if (!result_checkable(d)) { ++c.skipped; continue; }
        ++c.tested;
        double ratio = 0.0; int idx = -1;
        NovlClass cl = grade(d, c, &ratio, &idx);
        if (cl == NOVL_FAIL && samples_left > 0) { print_fail_unary("mul_pwr2", x, d); --samples_left; }
        else if (cl == NOVL_WEAK && samples_left > 0) { print_weak("mul_pwr2", x, d, ratio, idx); --samples_left; }
      }
      std::printf("  %-18s tested=%-9ld skipped=%-9ld failures=%ld weak=%ld worst=%.4f  (b = +-2^k only)\n",
                  "mul_pwr2", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
      summary.push_back(InvSummary{"mul_pwr2", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio, c.worst_idx});
    }

    // Ternary fma(a, b, c) = add(multiply(a,b), c). Ranges kept modest so a*b+c
    // stays finite in FP32 (1e4*1e4 = 1e8 << FLT_MAX).
    {
      InvCount c; int samples_left = 3;
      InputDist g = uniform(-1e4, 1e4);
      std::mt19937_64 gen(seed++);
      for (int i = 0; i < kRandomN; ++i) {
        double a = g(gen);  qf::QuadFloat qa = make_wide_input(a, gen);
        double b = g(gen);  qf::QuadFloat qb = make_wide_input(b, gen);
        double cc = g(gen); qf::QuadFloat qc = make_wide_input(cc, gen);
        if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(cc))) { ++c.skipped; continue; }
        qf::QuadFloat d = qf::fma(qa, qb, qc);
        if (!result_checkable(d)) { ++c.skipped; continue; }
        ++c.tested;
        double ratio = 0.0; int idx = -1;
        NovlClass cl = grade(d, c, &ratio, &idx);
        if (cl != NOVL_OK && samples_left > 0) {
          std::printf("    %s fma          a=%.9g b=%.9g c=%.9g  ratio=%.6f @idx%d  f=[%.9g %.9g %.9g %.9g]\n",
                      cl == NOVL_FAIL ? "FAIL" : (kStrictPriestGate ? "FAIL(weak-norm)" : "WEAK-NORM"),
                      a, b, cc, ratio, idx, d.f0, d.f1, d.f2, d.f3);
          --samples_left;
        }
      }
      std::printf("  %-18s tested=%-9ld skipped=%-9ld failures=%ld weak=%ld worst=%.4f\n",
                  "fma", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
      summary.push_back(InvSummary{"fma", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio, c.worst_idx});
    }

    // pow_int(a, n): integer exponent. Domain: not 0^negative (guard), and keep
    // |a|^|n| finite in FP32 — base in [-5,5], n in [-20,20] gives <= 5^20 ~ 9.5e13.
    {
      InvCount c; int samples_left = 3;
      InputDist gx = uniform(-5.0, 5.0);
      std::mt19937_64 gen(seed++);
      std::uniform_int_distribution<int> dn(-20, 20);
      for (int i = 0; i < kRandomN; ++i) {
        double x = gx(gen);
        qf::QuadFloat q = make_wide_input(x, gen);
        int n = dn(gen);
        if (!std::isfinite(x) || (x == 0.0 && n < 0)) { ++c.skipped; continue; }
        qf::QuadFloat d = qf::pow_int(q, n);
        if (!result_checkable(d)) { ++c.skipped; continue; }
        ++c.tested;
        double ratio = 0.0; int idx = -1;
        NovlClass cl = grade(d, c, &ratio, &idx);
        if (cl != NOVL_OK && samples_left > 0) {
          std::printf("    %s pow_int      x=%.9g n=%d  ratio=%.6f @idx%d  f=[%.9g %.9g %.9g %.9g]\n",
                      cl == NOVL_FAIL ? "FAIL" : (kStrictPriestGate ? "FAIL(weak-norm)" : "WEAK-NORM"),
                      x, n, ratio, idx, d.f0, d.f1, d.f2, d.f3);
          --samples_left;
        }
      }
      std::printf("  %-18s tested=%-9ld skipped=%-9ld failures=%ld weak=%ld worst=%.4f\n",
                  "pow_int", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
      summary.push_back(InvSummary{"pow_int", c.tested, c.skipped, c.failures, c.weak, c.worst_ratio, c.worst_idx});
    }

    // -- Test B: device tripwire (5 representative ops) ----------------------
    std::printf("\n[Test B] device tripwire (5 ops, 10^5 random on %s)\n",
                Kokkos::DefaultExecutionSpace::name());
    const int nd = 100'000;
    summary.push_back(device_binary("add", nd, 55501ULL,
        uniform(-1e8, 1e8), uniform(-1e8, 1e8), dom2_any,
        KOKKOS_LAMBDA(qf::QuadFloat a, qf::QuadFloat b){ return qf::add(a, b); }));
    summary.push_back(device_binary("multiply", nd, 55502ULL,
        uniform(-1e6, 1e6), uniform(-1e6, 1e6), dom2_any,
        KOKKOS_LAMBDA(qf::QuadFloat a, qf::QuadFloat b){ return qf::multiply(a, b); }));
    summary.push_back(device_unary("sqrt", nd, 55503ULL,
        uniform(0.0, 1e8), dom_nonneg,
        KOKKOS_LAMBDA(qf::QuadFloat x){ return qf::sqrt(x); }));
    summary.push_back(device_unary("exp", nd, 55504ULL,
        uniform(-88.0, 87.5), [](double x){ return std::isfinite(x) && x < 88.0; },
        KOKKOS_LAMBDA(qf::QuadFloat x){ return qf::exp(x); }));
    summary.push_back(device_unary("sin", nd, 55505ULL,
        uniform(-1000.0, 1000.0), dom_trig,
        KOKKOS_LAMBDA(qf::QuadFloat x){ return qf::sin(x); }));

    // -- Test C: corner cases as named asserts -------------------------------
    // zero, +-ulp, subnormals, +-inf, NaN — the invariant's boundary inputs.
    std::printf("\n[Test C] corner cases (zero / +-ulp / subnormal / +-inf / NaN)\n");
    int c_pass = 0, c_total = 0;
    auto corner = [&](const char* label, qf::QuadFloat in,
                      const QFUnary& fn, bool expect_checkable) {
      ++c_total;
      qf::QuadFloat d = fn(in);
      bool checkable = result_checkable(d);
      bool inv = checkable ? invariant_holds(d) : true;   // uncheckable -> trivially OK
      bool ok  = expect_checkable ? (checkable && inv) : inv;
      std::printf("    %-26s f=[%.6g %.6g %.6g %.6g]  checkable=%s inv=%s  %s\n",
                  label, d.f0, d.f1, d.f2, d.f3,
                  checkable ? "yes" : "no", inv ? "ok" : "BAD",
                  ok ? "PASS" : "FAIL");
      if (ok) ++c_pass;
    };

    const float fmin_normal = std::numeric_limits<float>::min();       // smallest normal
    const float fsub        = std::numeric_limits<float>::denorm_min(); // smallest subnormal
    const float finf        = std::numeric_limits<float>::infinity();
    const float fnan        = std::numeric_limits<float>::quiet_NaN();
    const float ulp1        = std::ldexp(1.0f, -23);                   // ulp(1.0f) = 2^-23

    // zero: exact zero through a renorming op stays a packed zero.
    corner("add(0,0)",            qf::QuadFloat(0.0f),
           [](qf::QuadFloat a){ return qf::add(a, qf::QuadFloat(0.0f)); }, true);
    corner("multiply(0,7)",       qf::QuadFloat(0.0f),
           [](qf::QuadFloat a){ return qf::multiply(a, qf::QuadFloat(7.0f)); }, true);
    corner("sqrt(0)",             qf::QuadFloat(0.0f),
           [](qf::QuadFloat a){ return qf::sqrt(a); }, true);
    // +-ulp: 1 +/- ulp(1) — a minimal two-word expansion; add must keep it packed.
    corner("add(1, +ulp)",        qf::QuadFloat(1.0f, ulp1, 0.0f, 0.0f),
           [](qf::QuadFloat a){ return qf::add(a, qf::QuadFloat(0.0f)); }, true);
    corner("add(1, -ulp)",        qf::QuadFloat(1.0f, -ulp1, 0.0f, 0.0f),
           [](qf::QuadFloat a){ return qf::add(a, qf::QuadFloat(0.0f)); }, true);
    corner("multiply(1+ulp, 1+ulp)", qf::QuadFloat(1.0f, ulp1, 0.0f, 0.0f),
           [](qf::QuadFloat a){ return qf::multiply(a, a); }, true);
    // subnormals: smallest-normal and smallest-subnormal leading words. The
    // underflow-tail guard is EXPECTED to mark these uncheckable (not a defect).
    corner("negate(FLT_MIN)",     qf::QuadFloat(fmin_normal),
           [](qf::QuadFloat a){ return qf::negate(a); }, false);
    corner("abs(-denorm_min)",    qf::QuadFloat(-fsub),
           [](qf::QuadFloat a){ return qf::abs(a); }, false);
    corner("add(denorm_min,0)",   qf::QuadFloat(fsub),
           [](qf::QuadFloat a){ return qf::add(a, qf::QuadFloat(0.0f)); }, false);
    // +-inf: exp/add overflow to inf -> uncheckable (skipped, not a failure).
    corner("add(+inf,1)",         qf::QuadFloat(finf),
           [](qf::QuadFloat a){ return qf::add(a, qf::QuadFloat(1.0f)); }, false);
    corner("negate(+inf)",        qf::QuadFloat(finf),
           [](qf::QuadFloat a){ return qf::negate(a); }, false);
    // NaN: any NaN-leading result is uncheckable.
    corner("abs(NaN)",            qf::QuadFloat(fnan),
           [](qf::QuadFloat a){ return qf::abs(a); }, false);
    corner("add(NaN,1)",          qf::QuadFloat(fnan),
           [](qf::QuadFloat a){ return qf::add(a, qf::QuadFloat(1.0f)); }, false);
    std::printf("  Test C: %d/%d passed\n", c_pass, c_total);

    // -- Summary table -------------------------------------------------------
    // failures = NOVL_FAIL (>ulp / packing break): fatal under either gate.
    // weak     = NOVL_WEAK (in (1/2 ulp, ulp]): QD weak-normalization; fatal only
    //            under the strict Priest gate (kStrictPriestGate). worst = worst
    //            weak-overlap ratio (1.0 = exactly 1/2 ulp, 2.0 = exactly ulp).
    std::printf("\n=== Summary (op : tested / skipped / failures / weak / worst : status) ===\n");
    std::printf("    gate posture: %s (kStrictPriestGate=%s)\n",
                kStrictPriestGate ? "STRICT Priest 1/2-ulp -- WEAK counts as failure"
                                  : "WEAK Shewchuk <=ulp -- WEAK tolerated",
                kStrictPriestGate ? "true" : "false");
    long total_tested = 0, total_skipped = 0, total_failures = 0, total_weak = 0;
    double overall_worst = 0.0;
    for (const auto& s : summary) {
      total_tested += s.tested; total_skipped += s.skipped;
      total_failures += s.failures; total_weak += s.weak;
      if (s.worst_ratio > overall_worst) overall_worst = s.worst_ratio;
      long op_gate_fail = s.failures + (kStrictPriestGate ? s.weak : 0);
      std::printf("  %-24s %11ld %11ld %9ld %8ld %8.4f   %s\n",
                  s.name.c_str(), s.tested, s.skipped, s.failures, s.weak, s.worst_ratio,
                  op_gate_fail == 0 ? "OK" : "FAIL");
    }
    std::printf("  %-24s %11ld %11ld %9ld %8ld %8.4f\n",
                "TOTAL", total_tested, total_skipped, total_failures, total_weak, overall_worst);

    // Gate: NOVL_FAIL always fatal; NOVL_WEAK fatal only under the strict Priest
    // gate. To CHANGE the posture, flip kStrictPriestGate -- do not edit the check.
    const long gate_fail = total_failures + (kStrictPriestGate ? total_weak : 0);
    if (total_weak > 0)
      std::printf("\n  NOTE: %ld weak-normalization deviation(s) in (1/2 ulp, ulp] "
                  "(worst ratio %.4f). These are QD renorm's Shewchuk-weak non-overlap,\n"
                  "  not per-op bugs (see PORT_NOTES / T3.2 report). Under the strict "
                  "Priest gate they count as failures.\n", total_weak, overall_worst);
    KOKKOS_EP_ASSERT(gate_fail == 0,
                     "one or more QF ops violated the length-4 non-overlap gate");
    KOKKOS_EP_ASSERT(c_pass == c_total, "a corner-case invariant check failed");

    rc = ep_exit_code();
    std::printf("\n=== qf_nonoverlap_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();
  return rc;
}
