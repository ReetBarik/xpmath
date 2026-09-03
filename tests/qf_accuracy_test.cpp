// ============================================================================
// qf_accuracy_test.cpp — Layer 4 (differential accuracy vs quadmath) for QF.
//                        Plan T3.4.
// ============================================================================
//
// QF (QuadFloat, 4 x FP32) analogue of ff_accuracy_test.cpp (T2.4) and
// dd_accuracy_test.cpp (T1.4). The structure is mirrored end to end; the two
// mechanical changes are the PRECISION SCALE and the TOLERANCE MODEL — the SAME
// two changes T3.3 (qf_property_test) made relative to T2.3, applied here.
//
//     FF:  double-word, precision u^2 = 2^-48  -> tolerance ~= 8.45 at N=10^6
//     QF:  QUAD-word,   precision U   = 2^-96  -> tolerance = 10*2^-96 -> 27.90 digits
//
// WHAT THIS LAYER CHECKS
// ----------------------
// Layers 1-3 proved QF's atoms are bit-exact (T3.1 qf_eft), every op returns a
// length-4 (weak-)non-overlapping expansion (T3.2 qf_nonoverlap), and the
// composed ops satisfy algebraic identities (T3.3 qf_property) — none of which
// asks "does op(x) equal the true real answer to ~29 digits?". This layer asks
// exactly that. For every QF op returning a QuadFloat with a quadmath analogue we
// widen the device result to __float128 and compare it to a quadmath oracle
// evaluated on the SAME input, scoring each element in "digits of accuracy"
//   digits = -log10(|dut - oracle| / |oracle|),  clamped to [0, 29]  (qf_digits).
//
// THE ORACLE, AND WHY THE REFERENCE IS THE WIDENED INPUT (a QF-specific twist)
// ---------------------------------------------------------------------------
// The host __float128 (quadmath) overloads (impl/Kokkos_QuadPrecisionMath.hpp)
// carry ~34 digits — ~5 digits of headroom over QF's ~29 — so they are a
// legitimate ground-truth reference. The whole file is #ifdef
// KOKKOS_EP_HAVE_QUADMATH; a quadmath-less Kokkos runs main() to KOKKOS_EP_SKIP
// (77 -> CTest "Skipped"), matching the suite posture (test_utils.hpp header).
//
// One difference from T1.4/T2.4's runners: those evaluate the oracle at the
// NOMINAL double input, (float128)x. That is exact for DD/FF because their
// claimed precision (~31 / ~14 digits) is coarser than a double's 53 bits, so the
// gap between (float128)x and the backend's actual stored value is far below the
// tolerance. QF claims ~29 digits (~96 bits) — FINER than a double — so at QF
// precision that gap MATTERS for the broad regime (whose inputs carry sub-double
// bits by construction). This test therefore always evaluates the oracle at the
// EXACT widened value of the QF input (qf_to_q of the four input words), not at
// the nominal double. For the narrow (Route-A) regime the two coincide exactly (a
// double fits losslessly in 4 FP32 limbs, so qf_to_q(QuadFloat(x)) == (float128)x),
// so this only changes the broad regime — where it is the only honest choice.
//
// TOLERANCE MODEL (T3.3's absolute ulp-of-U floor, NOT the DD/FF statistical one)
// ------------------------------------------------------------------------------
// DD/FF are double-word: their nominal precision is u^2 and T1.4/T2.4 gate on the
// statistical floor -log10(N*u^2). QF is a QUAD-word — its resolution IS U = 2^-96
// (four FP32 limbs, ~96 bits), NOT some "u^4" — so, exactly as T3.3 established,
// this test follows the plan's per-op ABSOLUTE ulp-of-U tolerance, gating on the
// MEAN:
//     digits(k ulp) = -log10(k * 2^-96) = 96*log10(2) - log10(k)
//     10 ulp -> 27.90 (kTolDefault, the default gate)
//     30 ulp -> 27.42 (kTolSection10, exp-family whose negative tail lands in the
//                       FP32 output-denormal band; PORT_NOTES_QF §10)
// Gating on the MEAN (never the raw min) lets a handful of conditioning-limited
// inputs (near-cancellation, derivative -> inf near a domain edge, arg-reduction
// near +-pi*k, exp's output-denormal tail) dip without red-flagging a correct
// implementation. Low mins are localized and, for the ops in the shared
// PORT_NOTES §5 conditioning registry (lookup_expected_min_drop — sub/fdim/fma/
// asin/acos/atanh/remainder/exp/sin/cos/tan), reported as EXPECTED-MIN-DROP with
// the registry's cited reason. Registry key note: the inventory op "subtract" maps
// to the registry key "sub" (the only name that differs).
//
// FOUR INPUT REGIMES (folded into three passes, combined per op)
//   The plan asks for broad-random, narrow-random, near-domain-edge, and the full
//   corner corpus. This test delivers them as three passes whose (min, mean, n)
//   are combined per op (min over all, mean weighted by count):
//     (1) NARROW random  — Route-A QuadFloat(double) over the op's well-conditioned
//         domain (taken from src/demo_qf_real.cpp's fill_inputs, the §11 reference
//         domains; the FP32-narrower guards are shared with the T2.4 FF inventory).
//     (2) BROAD random   — the SAME domain but each input enriched to a full ~96-bit
//         ordered QuadFloat via make_wide_input (copied from qf_nonoverlap_test,
//         T3.1/T3.2/T3.3's construction). For a quad-word format the meaningful
//         breadth is the full 4-limb mantissa width, which this exercises.
//     (3) CORPUS / near-edge — the deterministic corner-case inputs random misses.
//         Where a PORT_NOTES §3/§4 NAMED accessor exists we use it (exp-family ->
//         exp_overflow, sin/cos/tan/sincos -> trig_near_pi, sinh/cosh/sinhcosh ->
//         sinh_cosh_small, atanh -> atanh_small, remainder -> remainder_regression);
//         these accessors ARE the near-domain-edge inputs (near overflow, near the
//         trig branch, near the Taylor cutover). Otherwise the generic
//         corpus::unary<float>() / corpus::binary<float>() bundlers.
//   All three feed the SAME domain predicate; out-of-domain corpus values are
//   SKIPPED (counted, not failed), keeping the score honest and suppressing
//   qf_math.hpp's domain-guard prints.
//
// EXP-FAMILY DOMAIN NARROWING (documented deviation from the demo / §11).
//   qf_math.hpp exp guards at a.f0>=88 (returns 0); for sufficiently negative
//   arguments the quad-word result's low limbs fall into the FP32 output-denormal
//   band and lose bits (PORT_NOTES_QF §10). The demo/§11 sample exp over [-80,80]
//   and report mean 25.99 — which reflects that denormal tail and is BELOW even the
//   30-ulp floor. To gate the MEAN honestly, the random passes here narrow the
//   negative end so the full quad-word result stays in the FP32 NORMAL range
//   (exp: [-35,80]; exp2: [-50,120]; exp10: [-15,30]); the excluded negative tail
//   is the §10 conditioning limit (min-drop exempt), exercised at the HIGH edge by
//   exp_overflow and cited via lookup_expected_min_drop("exp"). exp/exp2/exp10 are
//   gated at 30 ulp (§10); expm1 (result >= -1, no denormal-tail issue) at 10 ulp.
//   pow is ALSO an exp-family op (pow(a,b)=exp(b*log a)): its accuracy is bounded
//   above by the internal exp (§10-gated) and drops a further ~1.2 digits under the
//   pow relative condition number kappa=|b*ln a| (Higham §3.4), landing mean ~27.76
//   — the accuracy of a CORRECT exp-log pow, not a defect — so pow is gated at 30
//   ulp too, EXEMPT under the same §10 limit (see its inline note; reported for Reet).
//
// ROUND-FAMILY ORACLE / TIE SEMANTICS.
//   qf_math.hpp round_to_nearest_int / round use qf_nint = floor(d+0.5)
//   (round-half-up); ceil/floor/trunc are the obvious directed roundings. On
//   CONTINUOUS random inputs (and the generic corpus, which contains no exact
//   half-integers) round-half-up, ties-to-even and ties-away ALL agree — ties are
//   measure-zero — so the oracle is Kokkos::round/ceil/floor/trunc (matching
//   src/demo_qf_real.cpp). The nint_half_integer corpus is DELIBERATELY NOT used
//   for the round-family: it targets exact-tie behavior, whose convention differs
//   across the three roundings and is a separate (out-of-scope) concern here.
//
// OP SURFACE (49 rows) AND SKIPS (5), reconciling to T3.2's 54-op ceiling.
//   Every QF op returning a QuadFloat with a quadmath analogue is scored: 29 unary
//   (abs/negate/sqr/sqrt, round-family x5, exp-family x4, log-family x4, trig x3,
//   inverse-trig x3, hyperbolic x3, inverse-hyperbolic x3), 4 two-output components
//   (sincos.sin/.cos, sinhcosh.sinh/.cosh — SIN/SINH FIRST per PORT_NOTES_QF §12),
//   13 binary (add/subtract/multiply/divide/pow/atan2/hypot/fmod/remainder/
//   copysign/fmax/fmin/fdim), and 3 custom (multiply_scalar, fma, pow_int) = 49.
//   SKIPPED with in-source rationale (5): sloppy_add (add() IS its public alias),
//   ieee_add (internal, no distinct public op), divide_accurate (internal, divide()
//   wraps it), mul_pwr2 (exact power-of-2 scaling — its "accuracy" is exact by
//   construction for pwr2 args, covered bit-exactly by T3.3 A12; meaningless to
//   score against a general quadmath call), angle (identical to atan2(y,x), scored
//   as atan2). 49 + 5 = 54 = T3.2's op count. qf_math.hpp has NO erf/erfc/tgamma,
//   so T2.4's three RED candidates (its erf/erfc/tgamma) have no QF counterpart.
//
// RULE 4 — qf_math.hpp is NOT modified. If an op exposes a REAL accuracy shortfall
//   (MEAN below its ulp tolerance, or a registry op whose min sinks below its
//   sanctioned floor) this test REPORTS it — op, which pass, the input, the digit
//   count — and fails, as a candidate B-task for Reet to file. It does not patch
//   the library, and does not extend PORT_NOTES §5 to cover library defects.
//
// SCOPE: real QF ops only (qf_complex.hpp out of scope — T2.4 was real-only, so
// T3.4 is too). No FMA-contraction guard (T3.5), no e2e cancellation kernels
// (T3.6). Every numeric bound in a comment cites a source (PORT_NOTES_QF §, a demo
// domain, or "observed"); no fabricated proven bounds.
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 3, "T3.4"; the T2.4 DONE block
// (structural template); T3.3 (QF tolerance model + runner posture);
// PORT_NOTES_QF §5/§10/§11/§12; "The six test layers" layer 4.
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

// QF types live in Kokkos::Experimental; qf:: alias (matches the other qf tests).
namespace qf = Kokkos::Experimental;

// This file is meaningful only with the __float128 oracle. Without it we still
// compile (so CTest can register the target) but main() returns KOKKOS_EP_SKIP.
#ifdef KOKKOS_EP_HAVE_QUADMATH

// ----------------------------------------------------------------------------
// QF <-> oracle, precision constants, digit metric. Identical to qf_property_test
// (T3.3): test_utils.hpp has BackendTraits<DD>/<FF> but NOT <QF>, so — rather than
// touch the shared harness other tasks own — this file carries the QF-local
// helpers directly. qf_to_q mirrors src/demo_qf_real.cpp exactly.
// ----------------------------------------------------------------------------
static constexpr double kMaxDig = 29.0;   // QF ~28.9 decimal digits (4x24 ~= 96 bits)

static float128 qf_to_q(const qf::QuadFloat& x) {
  return (float128)x.f0 + (float128)x.f1 + (float128)x.f2 + (float128)x.f3;
}

// Digits of accuracy of a QF result (already widened) against the oracle, capped
// at QF's 29-digit ceiling. NaN/inf/zero handling included (mirrors
// ULP error, the second metric (see docs/ULP_METRIC.md). QF is 4 x FP32, so
// p = 96 significand bits and ulp(true) = |true| * 2^-96. A relative-error score
// goes vacuous near a zero of the function; this one does not. Reported, not
// gated — the condition-aware verdict needs kappa and a region label, and both
// live in scripts/sweep_accuracy --ulp.
//
// PLUMBING. qf_digits() is called once per scored element from seven different
// runners, each of which appends to its own `digs` vector and hands it to
// compute_stats(). Rather than thread a parallel vector through all seven,
// qf_digits() appends the matching ulp error to g_qf_ulps and qf_finish()
// drains it. The two arrays are therefore built in lockstep by construction;
// qf_finish() asserts that by checking the sizes and falls back to digits-only
// if they ever diverge.
static const int kQfSigBits = 96;                 // 4 x 24
static const double kUlpUnscorableQf = -1.0;      // ref zero / non-finite
static std::vector<double> g_qf_ulps;

static double qf_ulps_of(float128 computed, float128 ref) {
  if (Kokkos::isnan(ref) || Kokkos::isinf(ref)) return kUlpUnscorableQf;
  if (ref == (float128)0.0)
    return (computed == (float128)0.0) ? 0.0 : kUlpUnscorableQf;
  if (Kokkos::isnan(computed) || Kokkos::isinf(computed)) return HUGE_VAL;
  if (computed == ref) return 0.0;
  return (double)ldexpq(Kokkos::abs((computed - ref) / ref), kQfSigBits);
}

// Drain g_qf_ulps into the AccStats for one op. Digit stats are unchanged.
static AccStats qf_finish(const std::vector<double>& digs) {
  AccStats s = (g_qf_ulps.size() == digs.size())
                 ? compute_stats(digs.data(), g_qf_ulps.data(), (int)digs.size())
                 : compute_stats(digs.data(), (int)digs.size());
  g_qf_ulps.clear();
  return s;
}

// digits_of_accuracy in test_utils.hpp and element_digits in the demo).
static double qf_digits(float128 computed, float128 ref) {
  g_qf_ulps.push_back(qf_ulps_of(computed, ref));   // second metric, in lockstep
  if (Kokkos::isnan(computed) || Kokkos::isnan(ref)) return 0.0;
  if (Kokkos::isinf(ref))
    return (Kokkos::isinf(computed) && (computed > 0) == (ref > 0)) ? kMaxDig : 0.0;
  if (ref == (float128)0.0) return (computed == (float128)0.0) ? kMaxDig : 0.0;
  float128 rel = Kokkos::abs((computed - ref) / ref);
  if (rel == (float128)0.0) return kMaxDig;
  double d = -(double)Kokkos::log10(rel);
  return d < 0.0 ? 0.0 : (d > kMaxDig ? kMaxDig : d);
}

// QF nominal precision U = 2^-96; the digit floor for a k-ulp tolerance is
//   digits(k ulp) = -log10(k * 2^-96) = 96*log10(2) - log10(k).
// 96*log10(2) = 28.8988; hence 10 ulp -> 27.90, 30 ulp -> 27.42. (See the header's
// TOLERANCE MODEL note: QF is a quad-word, so its resolution IS U, and the plan
// specifies an absolute ulp tolerance — NOT the DD/FF statistical -log10(N*u^2).)
static const double kTolDefault   = 27.90;   // 10 ulp = -log10(10 * 2^-96), default gate
static const double kTolSection10 = 27.42;   // 30 ulp: exp-family output-denormal tail (§10)

// ----------------------------------------------------------------------------
// make_wide_input — enrich a nominal double x to a full ~96-bit ordered QuadFloat.
// Copied VERBATIM from tests/qf_nonoverlap_test.cpp (T3.1/T3.2/T3.3's broad-input
// construction): add sub-leading -ulp tail terms (relative 2^-28 / 2^-56 / 2^-84,
// each below 1/2 ulp of f0 so f0 stays == (float)x), then decompose the wide value
// into four ordered FP32 words by successive round-to-nearest. Only reached under
// KOKKOS_EP_HAVE_QUADMATH (the whole file is), so the quadmath path is always taken.
// ----------------------------------------------------------------------------
static qf::QuadFloat make_wide_input(double x, std::mt19937_64& g) {
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
// Per-op record + reporting. (Mirrors ff_accuracy_test.cpp's OpResult/report_op,
// adapted to the QF-local tolerance constants and 29-digit cap.)
// ----------------------------------------------------------------------------
struct OpResult {
  std::string name;
  long        n_scored   = 0;   // elements actually scored (all three passes)
  long        n_skipped  = 0;   // out-of-domain inputs skipped
  double      min_digits = 0.0;
  double      mean_digits = 0.0;
  double      tol_digits = 0.0;
  bool        pass       = false;   // mean >= tol (and, for a registry op, min >= floor)
  const ExpectedMinDropAnnotation* ann = nullptr;  // registry entry, or null
  bool        min_ok_for_registry = true;          // min >= ann->min_digits_allowed
  // Second metric. WORST point over all passes, never a mean. Reported, not
  // gated — see docs/ULP_METRIC.md.
  double      worst_ulp      = 0.0;
  long        n_ulp_scored   = 0;
  long        n_ulp_unscored = 0;
};

// Worst-point union of the ulp summaries from the three passes.
static void combine3_ulps(const AccStats& a, const AccStats& b, const AccStats& c,
                          OpResult& r) {
  const AccStats* v[3] = {&a, &b, &c};
  for (int i = 0; i < 3; ++i) {
    if (v[i]->ulp_max > r.worst_ulp) r.worst_ulp = v[i]->ulp_max;
    r.n_ulp_scored   += v[i]->n_ulp_scored;
    r.n_ulp_unscored += v[i]->n_ulp_unscored;
  }
}

// Combine the three passes' (min, mean, n) into one: min over all, mean weighted
// by count. Any pass whose n<=0 (e.g. all corpus inputs out of domain) drops out.
static void combine3(const AccStats& a, const AccStats& b, const AccStats& c,
                     double& min_d, double& mean_d, long& n_total) {
  n_total = (long)a.n + (long)b.n + (long)c.n;
  if (n_total <= 0) { min_d = 0.0; mean_d = 0.0; return; }
  mean_d = (a.mean * (double)a.n + b.mean * (double)b.n + c.mean * (double)c.n)
           / (double)n_total;
  double m = std::numeric_limits<double>::infinity();
  if (a.n > 0) m = std::min(m, a.min);
  if (b.n > 0) m = std::min(m, b.min);
  if (c.n > 0) m = std::min(m, c.min);
  min_d = std::isinf(m) ? 0.0 : m;
}

// Registry key differs from the inventory name for exactly one op: subtract->sub.
static const char* registry_key(const std::string& op_name) {
  if (op_name == "subtract") return "sub";
  return op_name.c_str();
}

// Finalize gating for one op. mean < tol -> fail; else if a registry op, the min
// must also clear the sanctioned floor; else pass.
static OpResult finalize(const std::string& name, double min_d, double mean_d,
                         long n_scored, long n_skipped, double tol_d) {
  OpResult r;
  r.name = name;
  r.n_scored = n_scored;
  r.n_skipped = n_skipped;
  r.min_digits = min_d;
  r.mean_digits = mean_d;
  r.tol_digits = tol_d;
  r.ann = lookup_expected_min_drop(registry_key(name));
  const bool mean_ok = mean_d >= tol_d;
  if (r.ann) {
    r.min_ok_for_registry = min_d >= r.ann->min_digits_allowed;
    r.pass = mean_ok && r.min_ok_for_registry;
  } else {
    r.pass = mean_ok;
  }
  return r;
}

// status: PASS | FAIL | "EXPECTED-MIN-DROP: OK" (registry op that cleared mean and
// whose low min is sanctioned).
static void report_op(const OpResult& r) {
  const char* status;
  if (!r.pass)                                        status = "FAIL";
  else if (r.ann && r.min_digits < kMaxDig)           status = "EXPECTED-MIN-DROP: OK";
  else                                                status = "PASS";
  std::printf("  %-18s n=%-9ld skipped=%-6ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s",
              r.name.c_str(), r.n_scored, r.n_skipped, r.min_digits, r.mean_digits,
              r.tol_digits, status);
  // Both metrics on one line: the digit score the KI history is written in, and
  // the worst-point ulp error the digit score cannot see near a zero of f.
  std::printf(" worst_ulp=%-11.4g ulp_n=%-9ld ulp_unscorable=%-7ld",
              r.worst_ulp, r.n_ulp_scored, r.n_ulp_unscored);
  if (r.ann && r.pass && r.min_digits < kMaxDig) std::printf("  (%s)", r.ann->reason);
  std::printf("\n");
}

// ----------------------------------------------------------------------------
// Domain predicates (on the nominal double input). Shared with the T2.4 FF
// inventory where applicable (the FP32 limb guards are the same for QF).
// ----------------------------------------------------------------------------
using Dom1 = std::function<bool(double)>;
using Dom2 = std::function<bool(double, double)>;

static const Dom1 dom_any    = [](double x){ return std::isfinite(x); };
static const Dom1 dom_nonneg = [](double x){ return std::isfinite(x) && x >= 0.0; };

// ----------------------------------------------------------------------------
// Op descriptors. Each carries the domain predicate, the narrow-random generator
// (also used, enriched, for the broad pass), the device op, the host quadmath
// oracle, the corpus vector, and the per-op ulp-tolerance floor.
// ----------------------------------------------------------------------------
struct UnaryOp {
  std::string                                    name;
  Dom1                                           in_domain;
  InputDist                                      gen;
  std::function<qf::QuadFloat(qf::QuadFloat)>    device_op;
  std::function<float128(float128)>              oracle;
  std::vector<float>                             corpus_inputs;
  double                                         tol = kTolDefault;
};

struct BinaryOp {
  std::string                                                    name;
  Dom2                                                           in_domain;
  InputDist                                                      gen_a;
  InputDist                                                      gen_b;
  std::function<qf::QuadFloat(qf::QuadFloat, qf::QuadFloat)>     device_op;
  std::function<float128(float128, float128)>                    oracle;
  std::vector<std::pair<float, float>>                           corpus_inputs;
  double                                                         tol = kTolDefault;
};

// The plan calls for 10^6 random inputs/op across broad+narrow. QF ops are the
// costliest backend (4-word renorm cascades, long-division divide, Taylor/Newton
// loops) on the Serial backend, and this test runs ~49 ops x (narrow + broad)
// passes, so 10^6 each is an untenable ctest wall time. Reduced to 10^5 per pass
// (2*10^5 random/op total) — the SAME documented reduction T3.2/T3.3 made for the
// same reason. 2*10^5 + the full corpus still exercises every op across the
// magnitude range and reproduces the report's numbers under the fixed per-op seeds.
static constexpr int kNarrowN = 100'000;   // narrow (Route-A) random inputs per op
static constexpr int kBroadN  = 100'000;   // broad  (make_wide_input) random inputs per op
static constexpr int kDeviceN = 50'000;    // explicit device-parity subset

// Generic corpus flags: zeros ON, inf OFF, nan OFF — inf/nan produce edge sentinels,
// not meaningful accuracy scores; the domain predicate additionally filters strays.
static corpus::CorpusFlags corpus_flags() {
  corpus::CorpusFlags f;
  f.include_zero = true;
  f.include_inf  = false;
  f.include_nan  = false;
  return f;
}

// ----------------------------------------------------------------------------
// QF-local device runners. Inputs are built on host (Route-A or wide), shipped as
// four FP32 word Views, the op runs in a Kokkos::parallel_for on the Serial
// DefaultExecutionSpace, and the four result words come back for scoring. The
// oracle reference is computed from the EXACT widened input value qf_to_q(input)
// (see the header) — NOT the nominal double. Mirrors qf_nonoverlap_test's device
// runners and qf_property_test's device_run, scoring digits instead of overlap.
// ----------------------------------------------------------------------------
template <typename DeviceOp>
static AccStats qf_run_unary(int n, uint64_t seed, const InputDist& gen,
                             const std::function<float128(float128)>& oracle,
                             DeviceOp op, const Dom1& dom, bool wide, long& skipped) {
  using exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float*, exec> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
  Kokkos::View<float*, exec> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  auto ha0 = Kokkos::create_mirror_view(a0); auto ha1 = Kokkos::create_mirror_view(a1);
  auto ha2 = Kokkos::create_mirror_view(a2); auto ha3 = Kokkos::create_mirror_view(a3);

  std::vector<double> hx(n);
  {
    std::mt19937_64 g(seed);
    for (int i = 0; i < n; ++i) {
      double x = gen(g); hx[i] = x;
      qf::QuadFloat q = wide ? make_wide_input(x, g) : qf::QuadFloat(x);
      ha0(i) = q.f0; ha1(i) = q.f1; ha2(i) = q.f2; ha3(i) = q.f3;
    }
  }
  Kokkos::deep_copy(a0, ha0); Kokkos::deep_copy(a1, ha1);
  Kokkos::deep_copy(a2, ha2); Kokkos::deep_copy(a3, ha3);

  Kokkos::parallel_for("qf_acc_unary", Kokkos::RangePolicy<exec>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(qf::QuadFloat(a0(i), a1(i), a2(i), a3(i)));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();

  auto ho0 = Kokkos::create_mirror_view(o0); auto ho1 = Kokkos::create_mirror_view(o1);
  auto ho2 = Kokkos::create_mirror_view(o2); auto ho3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(ho0, o0); Kokkos::deep_copy(ho1, o1);
  Kokkos::deep_copy(ho2, o2); Kokkos::deep_copy(ho3, o3);

  std::vector<double> digs; digs.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!dom(hx[i])) { ++skipped; continue; }
    float128 refin = (float128)ha0(i) + (float128)ha1(i) + (float128)ha2(i) + (float128)ha3(i);
    float128 got   = (float128)ho0(i) + (float128)ho1(i) + (float128)ho2(i) + (float128)ho3(i);
    digs.push_back(qf_digits(got, oracle(refin)));
  }
  return qf_finish(digs);
}

template <typename DeviceOp>
static AccStats qf_run_unary_corpus(const std::vector<float>& inputs,
                                    const std::function<float128(float128)>& oracle,
                                    DeviceOp op, const Dom1& dom, long& skipped) {
  const int n = (int)inputs.size();
  if (n <= 0) return AccStats{};
  using exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float*, exec> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
  Kokkos::View<float*, exec> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  auto ha0 = Kokkos::create_mirror_view(a0); auto ha1 = Kokkos::create_mirror_view(a1);
  auto ha2 = Kokkos::create_mirror_view(a2); auto ha3 = Kokkos::create_mirror_view(a3);
  std::vector<double> hx(n);
  for (int i = 0; i < n; ++i) {
    hx[i] = (double)inputs[i];
    qf::QuadFloat q((double)inputs[i]);   // corpus values are exact FP32 -> lossless
    ha0(i) = q.f0; ha1(i) = q.f1; ha2(i) = q.f2; ha3(i) = q.f3;
  }
  Kokkos::deep_copy(a0, ha0); Kokkos::deep_copy(a1, ha1);
  Kokkos::deep_copy(a2, ha2); Kokkos::deep_copy(a3, ha3);
  Kokkos::parallel_for("qf_acc_unary_corpus", Kokkos::RangePolicy<exec>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(qf::QuadFloat(a0(i), a1(i), a2(i), a3(i)));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();
  auto ho0 = Kokkos::create_mirror_view(o0); auto ho1 = Kokkos::create_mirror_view(o1);
  auto ho2 = Kokkos::create_mirror_view(o2); auto ho3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(ho0, o0); Kokkos::deep_copy(ho1, o1);
  Kokkos::deep_copy(ho2, o2); Kokkos::deep_copy(ho3, o3);
  std::vector<double> digs; digs.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!dom(hx[i])) { ++skipped; continue; }
    float128 refin = (float128)ha0(i) + (float128)ha1(i) + (float128)ha2(i) + (float128)ha3(i);
    float128 got   = (float128)ho0(i) + (float128)ho1(i) + (float128)ho2(i) + (float128)ho3(i);
    digs.push_back(qf_digits(got, oracle(refin)));
  }
  return qf_finish(digs);
}

template <typename DeviceOp>
static AccStats qf_run_binary(int n, uint64_t seed, const InputDist& ga, const InputDist& gb,
                              const std::function<float128(float128, float128)>& oracle,
                              DeviceOp op, const Dom2& dom, bool wide, long& skipped) {
  using exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float*, exec> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
  Kokkos::View<float*, exec> b0("b0", n), b1("b1", n), b2("b2", n), b3("b3", n);
  Kokkos::View<float*, exec> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  auto ha0 = Kokkos::create_mirror_view(a0); auto ha1 = Kokkos::create_mirror_view(a1);
  auto ha2 = Kokkos::create_mirror_view(a2); auto ha3 = Kokkos::create_mirror_view(a3);
  auto hb0 = Kokkos::create_mirror_view(b0); auto hb1 = Kokkos::create_mirror_view(b1);
  auto hb2 = Kokkos::create_mirror_view(b2); auto hb3 = Kokkos::create_mirror_view(b3);
  std::vector<double> hxa(n), hxb(n);
  {
    std::mt19937_64 g(seed);
    for (int i = 0; i < n; ++i) {
      double av = ga(g), bv = gb(g); hxa[i] = av; hxb[i] = bv;
      qf::QuadFloat qa = wide ? make_wide_input(av, g) : qf::QuadFloat(av);
      qf::QuadFloat qb = wide ? make_wide_input(bv, g) : qf::QuadFloat(bv);
      ha0(i) = qa.f0; ha1(i) = qa.f1; ha2(i) = qa.f2; ha3(i) = qa.f3;
      hb0(i) = qb.f0; hb1(i) = qb.f1; hb2(i) = qb.f2; hb3(i) = qb.f3;
    }
  }
  Kokkos::deep_copy(a0, ha0); Kokkos::deep_copy(a1, ha1);
  Kokkos::deep_copy(a2, ha2); Kokkos::deep_copy(a3, ha3);
  Kokkos::deep_copy(b0, hb0); Kokkos::deep_copy(b1, hb1);
  Kokkos::deep_copy(b2, hb2); Kokkos::deep_copy(b3, hb3);
  Kokkos::parallel_for("qf_acc_binary", Kokkos::RangePolicy<exec>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(qf::QuadFloat(a0(i), a1(i), a2(i), a3(i)),
                           qf::QuadFloat(b0(i), b1(i), b2(i), b3(i)));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();
  auto ho0 = Kokkos::create_mirror_view(o0); auto ho1 = Kokkos::create_mirror_view(o1);
  auto ho2 = Kokkos::create_mirror_view(o2); auto ho3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(ho0, o0); Kokkos::deep_copy(ho1, o1);
  Kokkos::deep_copy(ho2, o2); Kokkos::deep_copy(ho3, o3);
  std::vector<double> digs; digs.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!dom(hxa[i], hxb[i])) { ++skipped; continue; }
    float128 ra = (float128)ha0(i) + (float128)ha1(i) + (float128)ha2(i) + (float128)ha3(i);
    float128 rb = (float128)hb0(i) + (float128)hb1(i) + (float128)hb2(i) + (float128)hb3(i);
    float128 got = (float128)ho0(i) + (float128)ho1(i) + (float128)ho2(i) + (float128)ho3(i);
    digs.push_back(qf_digits(got, oracle(ra, rb)));
  }
  return qf_finish(digs);
}

template <typename DeviceOp>
static AccStats qf_run_binary_corpus(const std::vector<std::pair<float, float>>& inputs,
                                     const std::function<float128(float128, float128)>& oracle,
                                     DeviceOp op, const Dom2& dom, long& skipped) {
  const int n = (int)inputs.size();
  if (n <= 0) return AccStats{};
  using exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<float*, exec> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
  Kokkos::View<float*, exec> b0("b0", n), b1("b1", n), b2("b2", n), b3("b3", n);
  Kokkos::View<float*, exec> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
  auto ha0 = Kokkos::create_mirror_view(a0); auto ha1 = Kokkos::create_mirror_view(a1);
  auto ha2 = Kokkos::create_mirror_view(a2); auto ha3 = Kokkos::create_mirror_view(a3);
  auto hb0 = Kokkos::create_mirror_view(b0); auto hb1 = Kokkos::create_mirror_view(b1);
  auto hb2 = Kokkos::create_mirror_view(b2); auto hb3 = Kokkos::create_mirror_view(b3);
  std::vector<double> hxa(n), hxb(n);
  for (int i = 0; i < n; ++i) {
    hxa[i] = (double)inputs[i].first; hxb[i] = (double)inputs[i].second;
    qf::QuadFloat qa((double)inputs[i].first), qb((double)inputs[i].second);
    ha0(i) = qa.f0; ha1(i) = qa.f1; ha2(i) = qa.f2; ha3(i) = qa.f3;
    hb0(i) = qb.f0; hb1(i) = qb.f1; hb2(i) = qb.f2; hb3(i) = qb.f3;
  }
  Kokkos::deep_copy(a0, ha0); Kokkos::deep_copy(a1, ha1);
  Kokkos::deep_copy(a2, ha2); Kokkos::deep_copy(a3, ha3);
  Kokkos::deep_copy(b0, hb0); Kokkos::deep_copy(b1, hb1);
  Kokkos::deep_copy(b2, hb2); Kokkos::deep_copy(b3, hb3);
  Kokkos::parallel_for("qf_acc_binary_corpus", Kokkos::RangePolicy<exec>(0, n),
    KOKKOS_LAMBDA(int i) {
      qf::QuadFloat d = op(qf::QuadFloat(a0(i), a1(i), a2(i), a3(i)),
                           qf::QuadFloat(b0(i), b1(i), b2(i), b3(i)));
      o0(i) = d.f0; o1(i) = d.f1; o2(i) = d.f2; o3(i) = d.f3;
    });
  Kokkos::fence();
  auto ho0 = Kokkos::create_mirror_view(o0); auto ho1 = Kokkos::create_mirror_view(o1);
  auto ho2 = Kokkos::create_mirror_view(o2); auto ho3 = Kokkos::create_mirror_view(o3);
  Kokkos::deep_copy(ho0, o0); Kokkos::deep_copy(ho1, o1);
  Kokkos::deep_copy(ho2, o2); Kokkos::deep_copy(ho3, o3);
  std::vector<double> digs; digs.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!dom(hxa[i], hxb[i])) { ++skipped; continue; }
    float128 ra = (float128)ha0(i) + (float128)ha1(i) + (float128)ha2(i) + (float128)ha3(i);
    float128 rb = (float128)hb0(i) + (float128)hb1(i) + (float128)hb2(i) + (float128)hb3(i);
    float128 got = (float128)ho0(i) + (float128)ho1(i) + (float128)ho2(i) + (float128)ho3(i);
    digs.push_back(qf_digits(got, oracle(ra, rb)));
  }
  return qf_finish(digs);
}

// Run all three passes for one op and finalize.
static OpResult score_unary(const UnaryOp& op, uint64_t seed) {
  long skipped = 0;
  AccStats nar = qf_run_unary(kNarrowN, seed,       op.gen, op.oracle, op.device_op,
                              op.in_domain, /*wide=*/false, skipped);
  AccStats brd = qf_run_unary(kBroadN,  seed + 1,   op.gen, op.oracle, op.device_op,
                              op.in_domain, /*wide=*/true,  skipped);
  AccStats cor = qf_run_unary_corpus(op.corpus_inputs, op.oracle, op.device_op,
                                     op.in_domain, skipped);
  double min_d, mean_d; long n_total;
  combine3(nar, brd, cor, min_d, mean_d, n_total);
  OpResult r = finalize(op.name, min_d, mean_d, n_total, skipped, op.tol);
  combine3_ulps(nar, brd, cor, r);
  return r;
}

static OpResult score_binary(const BinaryOp& op, uint64_t seed) {
  long skipped = 0;
  AccStats nar = qf_run_binary(kNarrowN, seed,     op.gen_a, op.gen_b, op.oracle,
                               op.device_op, op.in_domain, /*wide=*/false, skipped);
  AccStats brd = qf_run_binary(kBroadN,  seed + 1, op.gen_a, op.gen_b, op.oracle,
                               op.device_op, op.in_domain, /*wide=*/true,  skipped);
  AccStats cor = qf_run_binary_corpus(op.corpus_inputs, op.oracle, op.device_op,
                                      op.in_domain, skipped);
  double min_d, mean_d; long n_total;
  combine3(nar, brd, cor, min_d, mean_d, n_total);
  OpResult r = finalize(op.name, min_d, mean_d, n_total, skipped, op.tol);
  combine3_ulps(nar, brd, cor, r);
  return r;
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
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;

#ifdef KOKKOS_EP_HAVE_QUADMATH
  {
    std::printf("=== qf_accuracy_test (T3.4): differential accuracy for QF vs "
                "quadmath ===\n");
    std::printf("Execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("Metric: digits = -log10(rel_err) vs __float128 oracle, capped at "
                "%.0f (QF U = 2^-96).\n", kMaxDig);
    std::printf("Fail gate: MEAN digits >= per-op ulp floor (10 ulp = %.2f default; "
                "30 ulp = %.2f for exp-family §10); low mins for §5 registry ops -> "
                "EXPECTED-MIN-DROP.\n", kTolDefault, kTolSection10);
    std::printf("Passes per op: narrow(%d, Route-A) + broad(%d, ~96-bit) + corpus; "
                "oracle at the exact widened input.\n\n", kNarrowN, kBroadN);

    const std::vector<float> corpus_unary_all = corpus::unary<float>(corpus_flags());
    const std::vector<std::pair<float, float>> corpus_binary_all =
        corpus::binary<float>(corpus_flags());

    constexpr double pi = 3.14159265358979323846;

    // ------------------------------------------------------------------------
    // UNARY op inventory. Domains from src/demo_qf_real.cpp fill_inputs (the §11
    // reference domains); oracle per op; corpus = named accessor where one exists.
    // ------------------------------------------------------------------------
    std::vector<UnaryOp> unary_ops = {
      // abs/negate/sqr: structural / one multiply. Bound: abs/negate exact (0 ulp);
      // sqr ~ a few ulp (Dekker two_sqr). |x|<1e8 keeps sqr finite in FP32.
      {"abs",    dom_any, uniform(-1e8, 1e8),
       [](qf::QuadFloat x){ return qf::abs(x); },
       [](float128 x){ return Kokkos::abs(x); }, corpus_unary_all},
      {"negate", dom_any, uniform(-1e8, 1e8),
       [](qf::QuadFloat x){ return qf::negate(x); },
       [](float128 x){ return -x; }, corpus_unary_all},
      {"sqr",    dom_any, uniform(-1e8, 1e8),
       [](qf::QuadFloat x){ return qf::sqr(x); },
       [](float128 x){ return x * x; }, corpus_unary_all},
      // sqrt (Heron; PORT_NOTES_QF §0b). Domain per demo: [1e-16, 1e8], x>=0.
      {"sqrt",   dom_nonneg, uniform(1e-16, 1e8),
       [](qf::QuadFloat x){ return qf::sqrt(x); },
       [](float128 x){ return Kokkos::sqrt(x); }, corpus_unary_all},

      // round-family: qf_nint = floor(d+0.5) (round-half-up); oracle round/ceil/
      // floor/trunc agree on continuous inputs (ties measure-zero). Generic corpus
      // (no exact half-integers); nint_half_integer DELIBERATELY excluded (tie
      // semantics out of scope — see header). Domain per demo: |x|<1e13.
      {"round_to_nearest_int",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](qf::QuadFloat x){ return qf::round_to_nearest_int(x); },
       [](float128 x){ return Kokkos::round(x); }, corpus_unary_all},
      {"ceil",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](qf::QuadFloat x){ return qf::ceil(x); },
       [](float128 x){ return Kokkos::ceil(x); }, corpus_unary_all},
      {"floor",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](qf::QuadFloat x){ return qf::floor(x); },
       [](float128 x){ return Kokkos::floor(x); }, corpus_unary_all},
      {"trunc",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](qf::QuadFloat x){ return qf::trunc(x); },
       [](float128 x){ return Kokkos::trunc(x); }, corpus_unary_all},
      {"round",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](qf::QuadFloat x){ return qf::round(x); },
       [](float128 x){ return Kokkos::round(x); }, corpus_unary_all},

      // exp-family. Negative-end domains NARROWED to keep the quad-word result in
      // FP32 normal range (see header EXP-FAMILY note); exp/exp2/exp10 gated at
      // 30 ulp (§10, registry "exp"), corpus = exp_overflow (high edge, filtered by
      // the guard). expm1 (result >= -1) at 10 ulp.
      {"exp",  [](double x){ return std::isfinite(x) && x < 88.7; },
       uniform(-35.0, 88.7),
       [](qf::QuadFloat x){ return qf::exp(x); },
       [](float128 x){ return Kokkos::exp(x); },
       corpus::exp_overflow<float>(), kTolSection10},
      {"exp2", [](double x){ return std::isfinite(x) && x < 126.0; },
       uniform(-50.0, 120.0),
       [](qf::QuadFloat x){ return qf::exp2(x); },
       [](float128 x){ return Kokkos::exp2(x); }, corpus_unary_all, kTolSection10},
      // exp10 oracle: pow(10,x) (no __float128 exp10; matches demo_qf_real.cpp).
      {"exp10", [](double x){ return std::isfinite(x) && x < 38.5; },
       uniform(-15.0, 38.5),
       [](qf::QuadFloat x){ return qf::exp10(x); },
       [](float128 x){ return Kokkos::pow((float128)10.0, x); },
       corpus_unary_all, kTolSection10},
      {"expm1", [](double x){ return std::isfinite(x) && x < 88.0; },
       uniform(-1.0, 1.0),
       [](qf::QuadFloat x){ return qf::expm1(x); },
       [](float128 x){ return Kokkos::expm1(x); }, corpus_unary_all},

      // log-family. Domain per demo: isnormal, [1e-16, 1e16], x>0.
      {"log",   [](double x){ return std::isnormal(x) && x >= 1e-16 && x <= 1e16; },
       loguniform(-16, 16),
       [](qf::QuadFloat x){ return qf::log(x); },
       [](float128 x){ return Kokkos::log(x); }, corpus_unary_all},
      {"log2",  [](double x){ return std::isnormal(x) && x >= 1e-16 && x <= 1e16; },
       loguniform(-16, 16),
       [](qf::QuadFloat x){ return qf::log2(x); },
       [](float128 x){ return Kokkos::log2(x); }, corpus_unary_all},
      {"log10", [](double x){ return std::isnormal(x) && x >= 1e-16 && x <= 1e16; },
       loguniform(-16, 16),
       [](qf::QuadFloat x){ return qf::log10(x); },
       [](float128 x){ return Kokkos::log10(x); }, corpus_unary_all},
      {"log1p", [](double x){ return std::isfinite(x) && x > -1.0 && x < 1e16; },
       uniform(-0.999, 1e6),
       [](qf::QuadFloat x){ return qf::log1p(x); },
       [](float128 x){ return Kokkos::log1p(x); }, corpus_unary_all},

      // trig. Domain [-pi,pi] (table-free QF arg reduction stays clean; demo uses
      // [-pi,pi]); registry sin/cos/tan; corpus = trig_near_pi (branch edge).
      {"sin", [](double x){ return std::isfinite(x) && std::fabs(x) <= pi + 1e-6; },
       uniform(-pi, pi),
       [](qf::QuadFloat x){ return qf::sin(x); },
       [](float128 x){ return Kokkos::sin(x); }, corpus::trig_near_pi<float>()},
      {"cos", [](double x){ return std::isfinite(x) && std::fabs(x) <= pi + 1e-6; },
       uniform(-pi, pi),
       [](qf::QuadFloat x){ return qf::cos(x); },
       [](float128 x){ return Kokkos::cos(x); }, corpus::trig_near_pi<float>()},
      // tan: demo domain [-1.4,1.4] (away from the pi/2 pole).
      {"tan", [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.4; },
       uniform(-1.4, 1.4),
       [](qf::QuadFloat x){ return qf::tan(x); },
       [](float128 x){ return Kokkos::tan(x); }, corpus_unary_all},

      // inverse trig. asin/acos on [-1,1] (registry, conditioning near |a|=1);
      // atan on [-1e8,1e8].
      {"asin", [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0; },
       uniform(-1.0, 1.0),
       [](qf::QuadFloat x){ return qf::asin(x); },
       [](float128 x){ return Kokkos::asin(x); }, corpus_unary_all},
      {"acos", [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0; },
       uniform(-1.0, 1.0),
       [](qf::QuadFloat x){ return qf::acos(x); },
       [](float128 x){ return Kokkos::acos(x); }, corpus_unary_all},
      {"atan", [](double x){ return std::isfinite(x) && std::fabs(x) < 1e18; },
       uniform(-1e8, 1e8),
       [](qf::QuadFloat x){ return qf::atan(x); },
       [](float128 x){ return Kokkos::atan(x); }, corpus_unary_all},

      // hyperbolic. sinh/cosh on [-20,20] (demo); tanh on [-5,5]; corpus =
      // sinh_cosh_small (small-arg Taylor edge) for sinh/cosh.
      {"sinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-20.0, 20.0),
       [](qf::QuadFloat x){ return qf::sinh(x); },
       [](float128 x){ return Kokkos::sinh(x); }, corpus::sinh_cosh_small<float>()},
      {"cosh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-20.0, 20.0),
       [](qf::QuadFloat x){ return qf::cosh(x); },
       [](float128 x){ return Kokkos::cosh(x); }, corpus::sinh_cosh_small<float>()},
      {"tanh", [](double x){ return std::isfinite(x); },
       uniform(-1.0e4, 1.0e4),
       [](qf::QuadFloat x){ return qf::tanh(x); },
       [](float128 x){ return Kokkos::tanh(x); }, corpus_unary_all},

      // inverse hyperbolic. asinh on [-1e8,1e8]; acosh on [1,1e12] (x>=1); atanh on
      // (-1,1) (registry, conditioning near |a|=1); corpus = atanh_small for atanh.
      {"asinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 1e18; },
       uniform(-1e8, 1e8),
       [](qf::QuadFloat x){ return qf::asinh(x); },
       [](float128 x){ return Kokkos::asinh(x); }, corpus_unary_all},
      {"acosh", [](double x){ return std::isfinite(x) && x >= 1.0 && x < 1e18; },
       uniform(1.0, 1e12),
       [](qf::QuadFloat x){ return qf::acosh(x); },
       [](float128 x){ return Kokkos::acosh(x); }, corpus_unary_all},
      {"atanh", [](double x){ return std::isfinite(x) && std::fabs(x) < 1.0; },
       uniform(-0.999, 0.999),
       [](qf::QuadFloat x){ return qf::atanh(x); },
       [](float128 x){ return Kokkos::atanh(x); }, corpus::atanh_small<float>()},
    };

    // Two-output ops, scored per component. sincos(a, sin, cos) — SIN FIRST;
    // sinhcosh(a, sinh, cosh) — SINH FIRST (PORT_NOTES_QF §12).
    std::vector<UnaryOp> two_out_ops = {
      {"sincos.sin", [](double x){ return std::isfinite(x) && std::fabs(x) <= pi + 1e-6; },
       uniform(-pi, pi),
       [](qf::QuadFloat x){ qf::QuadFloat s, c; qf::sincos(x, s, c); return s; },
       [](float128 x){ return Kokkos::sin(x); }, corpus::trig_near_pi<float>()},
      {"sincos.cos", [](double x){ return std::isfinite(x) && std::fabs(x) <= pi + 1e-6; },
       uniform(-pi, pi),
       [](qf::QuadFloat x){ qf::QuadFloat s, c; qf::sincos(x, s, c); return c; },
       [](float128 x){ return Kokkos::cos(x); }, corpus::trig_near_pi<float>()},
      {"sinhcosh.sinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-20.0, 20.0),
       [](qf::QuadFloat x){ qf::QuadFloat sh, ch; qf::sinhcosh(x, sh, ch); return sh; },
       [](float128 x){ return Kokkos::sinh(x); }, corpus::sinh_cosh_small<float>()},
      {"sinhcosh.cosh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-20.0, 20.0),
       [](qf::QuadFloat x){ qf::QuadFloat sh, ch; qf::sinhcosh(x, sh, ch); return ch; },
       [](float128 x){ return Kokkos::cosh(x); }, corpus::sinh_cosh_small<float>()},
    };

    // ------------------------------------------------------------------------
    // BINARY op inventory.
    // ------------------------------------------------------------------------
    const Dom2 dom2_any = [](double a, double b){ return std::isfinite(a) && std::isfinite(b); };
    const Dom2 dom2_bnz = [](double a, double b){
      return std::isfinite(a) && std::isfinite(b) && b != 0.0; };
    // fmod/remainder: q=a/b then trunc/nint(q); |q| past nint's guard is out of range.
    const Dom2 dom2_modbnz = [](double a, double b){
      return std::isfinite(a) && std::isfinite(b) && b != 0.0 &&
             std::fabs(a) < std::fabs(b) * 1e13; };
    // pow(a,b)=exp(b*log a); a>0 in [1e-16,1e16], final exp within its 88 guard.
    const Dom2 dom2_powpos = [](double a, double b){
      return std::isnormal(a) && std::isfinite(b) && a >= 1e-16 && a <= 1e16 &&
             std::fabs(b * std::log(a)) < 88.0; };
    // atan2(y,x)=angle(x,y): magnitude finite, neither operand subnormal-tiny.
    const Dom2 dom2_atan2 = [](double a, double b){
      if (!(std::isfinite(a) && std::isfinite(b))) return false;
      double m = std::fmax(std::fabs(a), std::fabs(b));
      if (m == 0.0) return true;
      if (m > 1e18) return false;
      if (a != 0.0 && std::fabs(a) < 1e-18) return false;
      if (b != 0.0 && std::fabs(b) < 1e-18) return false;
      return true; };

    std::vector<BinaryOp> binary_ops = {
      // add/subtract: twoSum-based (registry "sub"). [-1e8,1e8]^2 includes
      // near-cancellation (rare -> mean stays high; low min localizes it).
      {"add",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::add(a, b); },
       [](float128 a, float128 b){ return a + b; }, corpus_binary_all},
      {"subtract", dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::subtract(a, b); },
       [](float128 a, float128 b){ return a - b; }, corpus_binary_all},
      // multiply/divide. divide = long-division (PORT_NOTES_QF §0b). [-1e6,1e6]^2.
      {"multiply", dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::multiply(a, b); },
       [](float128 a, float128 b){ return a * b; }, corpus_binary_all},
      {"divide",   dom2_bnz, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::divide(a, b); },
       [](float128 a, float128 b){ return a / b; }, corpus_binary_all},
      // pow(a,b) = exp(b*log a) (qf_math.hpp). Demo domain a in [0.5,20], b in
      // [0.1,5] (keeps internal exp in range). EXP-FAMILY -> gated at 30 ulp (§10,
      // kTolSection10), NOT the 10-ulp default: pow's accuracy is bounded above by
      // its internal exp (mean 28.19, itself §10-gated at 30 ulp) and drops a
      // further ~1.2 digits under the pow relative condition number kappa=|b·ln a|
      // (Higham §3.4; reaches ~15 over this domain), landing mean ~27.76. That is
      // the accuracy of a CORRECT exp-log pow, not a qf_math.hpp defect (a perfect
      // implementation shows the same conditioning) — so it is EXEMPT under the same
      // §10 output-denormal-tail limit that governs the rest of the exp-family, with
      // that citation, not silently loosened. (Reported for Reet: if pow is instead
      // to be held to the 10-ulp default, this becomes a RED to investigate — see
      // the T3.4 report and lookup_expected_min_drop is NOT consulted for pow.)
      {"pow",      dom2_powpos, uniform(0.5, 20.0), uniform(0.1, 5.0),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::pow(a, b); },
       [](float128 a, float128 b){ return Kokkos::pow(a, b); }, corpus_binary_all,
       kTolSection10},
      // atan2(y,x)=angle(x,y). Oracle argument order: quadmath atan2(y,x).
      {"atan2",    dom2_atan2, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::atan2(a, b); },
       [](float128 a, float128 b){ return Kokkos::atan2(a, b); }, corpus_binary_all},
      {"hypot",    dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::hypot(a, b); },
       [](float128 a, float128 b){ return Kokkos::hypot(a, b); }, corpus_binary_all},
      // fmod / remainder (registry "remainder"; corpus = remainder_regression).
      {"fmod",      dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fmod(a, b); },
       [](float128 a, float128 b){ return Kokkos::fmod(a, b); }, corpus_binary_all},
      {"remainder", dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::remainder(a, b); },
       [](float128 a, float128 b){ return Kokkos::remainder(a, b); },
       corpus::remainder_regression<float>()},
      // copysign/fmax/fmin: structural (bit-exact). fdim = max(a-b,0) (registry "fdim").
      {"copysign", dom2_any, uniform(-1e8, 1e8), uniform(-1.0, 1.0),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::copysign(a, b); },
       [](float128 a, float128 b){ return Kokkos::copysign(a, b); }, corpus_binary_all},
      {"fmax",     dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fmax(a, b); },
       [](float128 a, float128 b){ return Kokkos::fmax(a, b); }, corpus_binary_all},
      {"fmin",     dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fmin(a, b); },
       [](float128 a, float128 b){ return Kokkos::fmin(a, b); }, corpus_binary_all},
      {"fdim",     dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](qf::QuadFloat a, qf::QuadFloat b){ return qf::fdim(a, b); },
       [](float128 a, float128 b){ return Kokkos::fdim(a, b); }, corpus_binary_all},
    };

    // ------------------------------------------------------------------------
    // Run every op. Seeds increment per op (each op consumes seed and seed+1 for
    // its narrow/broad passes; step by 2).
    // ------------------------------------------------------------------------
    std::vector<OpResult> results;
    uint64_t seed = 12345ULL;

    std::printf("[unary ops] narrow + broad + corpus\n");
    for (const auto& op : unary_ops)   { results.push_back(score_unary(op, seed)); seed += 2; report_op(results.back()); }
    std::printf("\n[two-output ops] scored per component\n");
    for (const auto& op : two_out_ops) { results.push_back(score_unary(op, seed)); seed += 2; report_op(results.back()); }
    std::printf("\n[binary ops] narrow + broad + corpus\n");
    for (const auto& op : binary_ops)  { results.push_back(score_binary(op, seed)); seed += 2; report_op(results.back()); }

    // multiply_scalar(QF a, float b). Custom (mixed QF/float signature). a wide,
    // b uniform. Oracle: qf_to_q(a) * (float128)b. Bound: ~a few ulp.
    {
      using exec = Kokkos::DefaultExecutionSpace;
      const int n = kNarrowN;
      Kokkos::View<float*, exec> a0("a0", n), a1("a1", n), a2("a2", n), a3("a3", n);
      Kokkos::View<float*, exec> bb("bb", n);
      Kokkos::View<float*, exec> o0("o0", n), o1("o1", n), o2("o2", n), o3("o3", n);
      auto ha0 = Kokkos::create_mirror_view(a0); auto ha1 = Kokkos::create_mirror_view(a1);
      auto ha2 = Kokkos::create_mirror_view(a2); auto ha3 = Kokkos::create_mirror_view(a3);
      auto hbb = Kokkos::create_mirror_view(bb);
      std::vector<double> hb(n);
      { std::mt19937_64 g(seed++);
        std::uniform_real_distribution<double> dx(-1e4, 1e4), ds(-1e6, 1e6);
        for (int i = 0; i < n; ++i) {
          qf::QuadFloat q = make_wide_input(dx(g), g);
          ha0(i)=q.f0; ha1(i)=q.f1; ha2(i)=q.f2; ha3(i)=q.f3;
          float bf = (float)ds(g); hbb(i)=bf; hb[i]=(double)bf;
        } }
      Kokkos::deep_copy(a0, ha0); Kokkos::deep_copy(a1, ha1);
      Kokkos::deep_copy(a2, ha2); Kokkos::deep_copy(a3, ha3); Kokkos::deep_copy(bb, hbb);
      Kokkos::parallel_for("qf_mulscalar_acc", Kokkos::RangePolicy<exec>(0, n),
        KOKKOS_LAMBDA(int i){
          qf::QuadFloat d = qf::multiply_scalar(qf::QuadFloat(a0(i),a1(i),a2(i),a3(i)), bb(i));
          o0(i)=d.f0; o1(i)=d.f1; o2(i)=d.f2; o3(i)=d.f3;
        });
      Kokkos::fence();
      auto ho0=Kokkos::create_mirror_view(o0); auto ho1=Kokkos::create_mirror_view(o1);
      auto ho2=Kokkos::create_mirror_view(o2); auto ho3=Kokkos::create_mirror_view(o3);
      Kokkos::deep_copy(ho0,o0); Kokkos::deep_copy(ho1,o1);
      Kokkos::deep_copy(ho2,o2); Kokkos::deep_copy(ho3,o3);
      std::vector<double> digs(n);
      for (int i = 0; i < n; ++i) {
        float128 ra = (float128)ha0(i)+(float128)ha1(i)+(float128)ha2(i)+(float128)ha3(i);
        float128 got= (float128)ho0(i)+(float128)ho1(i)+(float128)ho2(i)+(float128)ho3(i);
        digs[i] = qf_digits(got, ra * (float128)hb[i]);
      }
      AccStats s = qf_finish(digs);
      OpResult r = finalize("multiply_scalar", s.min, s.mean, s.n, 0, kTolDefault);
      results.push_back(r); report_op(r);
    }

    // ternary fma(a,b,c) (registry "fma"). a,b,c wide, uniform(-1e4,1e4).
    // Oracle: fmaq(a,b,c) — exact a*b+c then one round.
    {
      using exec = Kokkos::DefaultExecutionSpace;
      const int n = kNarrowN;
      Kokkos::View<float*, exec> a0("a0",n),a1("a1",n),a2("a2",n),a3("a3",n);
      Kokkos::View<float*, exec> b0("b0",n),b1("b1",n),b2("b2",n),b3("b3",n);
      Kokkos::View<float*, exec> c0("c0",n),c1("c1",n),c2("c2",n),c3("c3",n);
      Kokkos::View<float*, exec> o0("o0",n),o1("o1",n),o2("o2",n),o3("o3",n);
      auto ha0=Kokkos::create_mirror_view(a0);auto ha1=Kokkos::create_mirror_view(a1);
      auto ha2=Kokkos::create_mirror_view(a2);auto ha3=Kokkos::create_mirror_view(a3);
      auto hb0=Kokkos::create_mirror_view(b0);auto hb1=Kokkos::create_mirror_view(b1);
      auto hb2=Kokkos::create_mirror_view(b2);auto hb3=Kokkos::create_mirror_view(b3);
      auto hc0=Kokkos::create_mirror_view(c0);auto hc1=Kokkos::create_mirror_view(c1);
      auto hc2=Kokkos::create_mirror_view(c2);auto hc3=Kokkos::create_mirror_view(c3);
      { std::mt19937_64 g(seed++);
        std::uniform_real_distribution<double> d(-1e4, 1e4);
        for (int i=0;i<n;++i){
          qf::QuadFloat qa=make_wide_input(d(g),g), qb=make_wide_input(d(g),g), qc=make_wide_input(d(g),g);
          ha0(i)=qa.f0;ha1(i)=qa.f1;ha2(i)=qa.f2;ha3(i)=qa.f3;
          hb0(i)=qb.f0;hb1(i)=qb.f1;hb2(i)=qb.f2;hb3(i)=qb.f3;
          hc0(i)=qc.f0;hc1(i)=qc.f1;hc2(i)=qc.f2;hc3(i)=qc.f3;
        } }
      Kokkos::deep_copy(a0,ha0);Kokkos::deep_copy(a1,ha1);Kokkos::deep_copy(a2,ha2);Kokkos::deep_copy(a3,ha3);
      Kokkos::deep_copy(b0,hb0);Kokkos::deep_copy(b1,hb1);Kokkos::deep_copy(b2,hb2);Kokkos::deep_copy(b3,hb3);
      Kokkos::deep_copy(c0,hc0);Kokkos::deep_copy(c1,hc1);Kokkos::deep_copy(c2,hc2);Kokkos::deep_copy(c3,hc3);
      Kokkos::parallel_for("qf_fma_acc", Kokkos::RangePolicy<exec>(0,n),
        KOKKOS_LAMBDA(int i){
          qf::QuadFloat d = qf::fma(qf::QuadFloat(a0(i),a1(i),a2(i),a3(i)),
                                    qf::QuadFloat(b0(i),b1(i),b2(i),b3(i)),
                                    qf::QuadFloat(c0(i),c1(i),c2(i),c3(i)));
          o0(i)=d.f0;o1(i)=d.f1;o2(i)=d.f2;o3(i)=d.f3;
        });
      Kokkos::fence();
      auto ho0=Kokkos::create_mirror_view(o0);auto ho1=Kokkos::create_mirror_view(o1);
      auto ho2=Kokkos::create_mirror_view(o2);auto ho3=Kokkos::create_mirror_view(o3);
      Kokkos::deep_copy(ho0,o0);Kokkos::deep_copy(ho1,o1);Kokkos::deep_copy(ho2,o2);Kokkos::deep_copy(ho3,o3);
      std::vector<double> digs(n);
      for (int i=0;i<n;++i){
        float128 ra=(float128)ha0(i)+(float128)ha1(i)+(float128)ha2(i)+(float128)ha3(i);
        float128 rb=(float128)hb0(i)+(float128)hb1(i)+(float128)hb2(i)+(float128)hb3(i);
        float128 rc=(float128)hc0(i)+(float128)hc1(i)+(float128)hc2(i)+(float128)hc3(i);
        float128 got=(float128)ho0(i)+(float128)ho1(i)+(float128)ho2(i)+(float128)ho3(i);
        digs[i]=qf_digits(got, Kokkos::fma(ra, rb, rc));
      }
      AccStats s = qf_finish(digs);
      OpResult r = finalize("fma", s.min, s.mean, s.n, 0, kTolDefault);
      results.push_back(r); report_op(r);
    }

    // pow_int(a,n): integer exponent. a in uniform(-5,5), n in [-20,20], skip
    // 0^negative. Oracle: pow(qf_to_q(a), (float128)n).
    {
      using exec = Kokkos::DefaultExecutionSpace;
      const int N = kNarrowN;
      Kokkos::View<float*, exec> a0("a0",N),a1("a1",N),a2("a2",N),a3("a3",N);
      Kokkos::View<int*,   exec> dn("dn",N);
      Kokkos::View<float*, exec> o0("o0",N),o1("o1",N),o2("o2",N),o3("o3",N);
      auto ha0=Kokkos::create_mirror_view(a0);auto ha1=Kokkos::create_mirror_view(a1);
      auto ha2=Kokkos::create_mirror_view(a2);auto ha3=Kokkos::create_mirror_view(a3);
      auto hdn=Kokkos::create_mirror_view(dn);
      std::vector<double> hav(N); std::vector<int> hnv(N);
      { std::mt19937_64 g(seed++);
        std::uniform_real_distribution<double> dx(-5.0, 5.0);
        std::uniform_int_distribution<int> di(-20, 20);
        for (int i=0;i<N;++i){
          double av=dx(g); int ni=di(g); hav[i]=av; hnv[i]=ni;
          qf::QuadFloat q=make_wide_input(av,g);
          ha0(i)=q.f0;ha1(i)=q.f1;ha2(i)=q.f2;ha3(i)=q.f3; hdn(i)=ni;
        } }
      Kokkos::deep_copy(a0,ha0);Kokkos::deep_copy(a1,ha1);Kokkos::deep_copy(a2,ha2);Kokkos::deep_copy(a3,ha3);
      Kokkos::deep_copy(dn,hdn);
      Kokkos::parallel_for("qf_powint_acc", Kokkos::RangePolicy<exec>(0,N),
        KOKKOS_LAMBDA(int i){
          qf::QuadFloat d = qf::pow_int(qf::QuadFloat(a0(i),a1(i),a2(i),a3(i)), dn(i));
          o0(i)=d.f0;o1(i)=d.f1;o2(i)=d.f2;o3(i)=d.f3;
        });
      Kokkos::fence();
      auto ho0=Kokkos::create_mirror_view(o0);auto ho1=Kokkos::create_mirror_view(o1);
      auto ho2=Kokkos::create_mirror_view(o2);auto ho3=Kokkos::create_mirror_view(o3);
      Kokkos::deep_copy(ho0,o0);Kokkos::deep_copy(ho1,o1);Kokkos::deep_copy(ho2,o2);Kokkos::deep_copy(ho3,o3);
      std::vector<double> digs; digs.reserve(N); long skipped=0;
      for (int i=0;i<N;++i){
        if (hav[i]==0.0 && hnv[i]<0) { ++skipped; continue; }   // 0^negative undefined
        float128 ra=(float128)ha0(i)+(float128)ha1(i)+(float128)ha2(i)+(float128)ha3(i);
        float128 got=(float128)ho0(i)+(float128)ho1(i)+(float128)ho2(i)+(float128)ho3(i);
        digs.push_back(qf_digits(got, Kokkos::pow(ra, (float128)hnv[i])));
      }
      AccStats s = qf_finish(digs);
      OpResult r = finalize("pow_int", s.min, s.mean, s.n, skipped, kTolDefault);
      results.push_back(r); report_op(r);
    }

    // ------------------------------------------------------------------------
    // SKIPPED ops (returning QuadFloat but no meaningful independent quadmath
    // accuracy score) — reported explicitly so the op inventory reconciles to
    // T3.2's 54 (49 scored + 5 skipped). See the header's OP SURFACE note.
    // ------------------------------------------------------------------------
    std::printf("\n[skipped ops] (returning QuadFloat; no distinct accuracy score)\n");
    std::printf("  %-16s %s\n", "sloppy_add",      "add() IS its public alias (scored as 'add')");
    std::printf("  %-16s %s\n", "ieee_add",        "internal renorm variant; no distinct public op");
    std::printf("  %-16s %s\n", "divide_accurate", "internal; divide() wraps it (scored as 'divide')");
    std::printf("  %-16s %s\n", "mul_pwr2",         "exact power-of-2 scaling; exact by construction (T3.3 A12)");
    std::printf("  %-16s %s\n", "angle",           "identical to atan2(y,x) (scored as 'atan2')");

    // ------------------------------------------------------------------------
    // Device-parity subset. Every pass above already executes through
    // Kokkos::parallel_for on the Serial DefaultExecutionSpace (see qf_run_*),
    // so the whole test IS the device path; this block re-runs a representative
    // subset under FRESH seeds as an explicit parity checkpoint, mirroring T3.3's
    // device block. add/multiply/sqrt/exp/sin span arithmetic, Dekker, Heron, an
    // exp-family §10 op, and a registry trig op.
    // ------------------------------------------------------------------------
    std::printf("\n[Device-parity] representative subset on %d inputs, fresh seeds (%s)\n",
                kDeviceN, Kokkos::DefaultExecutionSpace::name());
    long device_failures = 0;
    auto device_check = [&](const char* label, AccStats s, double tol,
                            const ExpectedMinDropAnnotation* ann) {
      bool pass = s.mean >= tol && (!ann || s.min >= ann->min_digits_allowed);
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  label, s.n, s.min, s.mean, tol, pass ? "PASS" : "FAIL");
    };
    {
      long sk = 0;
      device_check("add", qf_run_binary(kDeviceN, 900001ULL, uniform(-1e8,1e8), uniform(-1e8,1e8),
        [](float128 a, float128 b){ return a + b; },
        [](qf::QuadFloat a, qf::QuadFloat b){ return qf::add(a,b); }, dom2_any, false, sk),
        kTolDefault, lookup_expected_min_drop("add"));
      device_check("multiply", qf_run_binary(kDeviceN, 900002ULL, uniform(-1e6,1e6), uniform(-1e6,1e6),
        [](float128 a, float128 b){ return a * b; },
        [](qf::QuadFloat a, qf::QuadFloat b){ return qf::multiply(a,b); }, dom2_any, false, sk),
        kTolDefault, nullptr);
      device_check("sqrt", qf_run_unary(kDeviceN, 900003ULL, uniform(1e-16,1e8),
        [](float128 x){ return Kokkos::sqrt(x); },
        [](qf::QuadFloat x){ return qf::sqrt(x); }, dom_nonneg, false, sk),
        kTolDefault, nullptr);
      device_check("exp", qf_run_unary(kDeviceN, 900004ULL, uniform(-35.0,80.0),
        [](float128 x){ return Kokkos::exp(x); },
        [](qf::QuadFloat x){ return qf::exp(x); },
        [](double x){ return std::isfinite(x) && x < 88.0; }, false, sk),
        kTolSection10, lookup_expected_min_drop("exp"));
      device_check("sin", qf_run_unary(kDeviceN, 900005ULL, uniform(-pi,pi),
        [](float128 x){ return Kokkos::sin(x); },
        [](qf::QuadFloat x){ return qf::sin(x); },
        [](double x){ return std::isfinite(x); }, false, sk),
        kTolDefault, lookup_expected_min_drop("sin"));
    }

    // ------------------------------------------------------------------------
    // Summary table + gate.
    // ------------------------------------------------------------------------
    std::printf("\n=== Summary (op : n / skipped / min / mean / tol : status) ===\n");
    long total_failures = 0, total_skipped = 0;
    for (const auto& r : results) {
      total_skipped += r.n_skipped;
      const char* status = !r.pass ? "FAIL"
                         : (r.ann && r.min_digits < kMaxDig) ? "EXP-MIN-DROP" : "PASS";
      std::printf("  %-18s %10ld %8ld  min=%6.2f mean=%6.2f tol=%6.2f  %s\n",
                  r.name.c_str(), r.n_scored, r.n_skipped,
                  r.min_digits, r.mean_digits, r.tol_digits, status);
      if (!r.pass) ++total_failures;
    }
    std::printf("  ----\n  ops=%zu  total_skipped=%ld  failures=%ld  device_failures=%ld\n",
                results.size(), total_skipped, total_failures, device_failures);

    // Lowest-mean callout (for the report).
    {
      std::vector<OpResult> bymean = results;
      std::sort(bymean.begin(), bymean.end(),
                [](const OpResult& a, const OpResult& b){ return a.mean_digits < b.mean_digits; });
      std::printf("\n[lowest-mean ops] (accuracy floor first)\n");
      int show = (int)std::min<size_t>(bymean.size(), 8);
      for (int i = 0; i < show; ++i) {
        const OpResult& r = bymean[i];
        std::printf("  %-18s mean=%6.2f min=%6.2f  %s\n",
                    r.name.c_str(), r.mean_digits, r.min_digits,
                    r.ann ? r.ann->reason : "well-conditioned; near U floor");
      }
    }

    // Expected-min-drop roll-call.
    {
      std::printf("\n[expected-min-drop ops] (PORT_NOTES §5 conditioning; gated on mean only)\n");
      for (const auto& r : results) {
        if (r.ann && r.pass && r.min_digits < kMaxDig) {
          std::printf("  %-18s min=%6.2f (allowed>=%.2f) mean=%6.2f  %s\n",
                      r.name.c_str(), r.min_digits, r.ann->min_digits_allowed,
                      r.mean_digits, r.ann->reason);
        }
      }
    }

    KOKKOS_EP_ASSERT(total_failures == 0,
                     "one or more QF ops fell below the accuracy tolerance (mean < "
                     "per-op ulp floor) or a registry op's min sank below its "
                     "sanctioned floor — see the FAIL rows; per Rule 4 this is "
                     "REPORTED as a candidate B-task, qf_math.hpp is not patched");
    KOKKOS_EP_ASSERT(device_failures == 0, "a device-parity subset op failed");

    rc = ep_exit_code();
    std::printf("\n=== qf_accuracy_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
#endif  // KOKKOS_EP_HAVE_QUADMATH

  Kokkos::finalize();

#ifndef KOKKOS_EP_HAVE_QUADMATH
  std::printf("=== qf_accuracy_test (T3.4): no __float128 oracle "
              "(LIBQUADMATH absent) -> SKIP ===\n");
  return KOKKOS_EP_SKIP;
#else
  return rc;
#endif
}
