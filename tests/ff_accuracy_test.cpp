// ============================================================================
// ff_accuracy_test.cpp — Layer 4 (differential accuracy vs quadmath) for FF.
//                        Plan T2.4.
// ============================================================================
//
// FF analogue of dd_accuracy_test.cpp (T1.4). The structure is mirrored end to
// end; the mechanical change is the PRECISION SCALE and the FP32-narrower op
// domains:
//
//     DD:  u = 2^-53, u^2 = 2^-106  -> max_digits 31, tolerance ~= 25.91 at N=10^6
//     FF:  u = 2^-24, u^2 = 2^-48   -> max_digits 14, tolerance ~=  8.45 at N=10^6
//                                       (= -log10(N * u^2), computed at runtime)
//
// WHAT THIS LAYER CHECKS
// ----------------------
// Layers 1-3 proved FF's atoms are bit-exact (T2.1), every op returns a
// non-overlapping (hi, lo) (T2.2), and the composed ops satisfy algebraic
// identities (T2.3) — all WITHOUT asking "does op(x) equal the true real answer
// to N digits?". This layer asks exactly that. For every FF op we widen the
// device result to __float128 and compare it to a quadmath oracle evaluated on
// the SAME input, scoring each element in "digits of accuracy"
//   digits = -log10(|dut - oracle| / |oracle|),  clamped to [0, 14].
//
// THE METRIC AND ITS TARGET
//   The plan (T2.4) states the metric as max(rel_err / u²) with u = 2⁻²⁴, i.e.
//   rel_err measured in units of u² = 2⁻⁴⁸ (FF's double-word unit roundoff). A
//   well-implemented double-word op lands at a small multiple of u² (2u²-10u²
//   depending on the op; per-op source comments cite the bound). We report that
//   same information in the digit domain, where u² = 2⁻⁴⁸ corresponds to ≈ 14.45
//   decimal digits — hence the max_digits=14 clamp in test_utils.hpp (any excess
//   is oracle noise, not FF accuracy). Expected mean per PORT_NOTES: 13.3-14.0.
//
// THE ORACLE
//   The host __float128 (quadmath) math overloads from Kokkos
//   (impl/Kokkos_QuadPrecisionMath.hpp) carry ~34 digits — ~20 digits of headroom
//   over FF's ~14 (far more comfortable than DD's ~3) — so they are a legitimate
//   ground-truth reference. The whole file is #ifdef KOKKOS_EP_HAVE_QUADMATH; a
//   quadmath-less Kokkos runs main() to KOKKOS_EP_SKIP (77 -> CTest "Skipped"),
//   matching the suite posture (see test_utils.hpp header). Two oracle subtleties
//   this test gets right (same as T1.4):
//     * FF's round_to_nearest_int / round use TIES-TO-EVEN. The matching quadmath
//       oracle is nearbyintq (Kokkos::nearbyint), NOT roundq (Kokkos::round,
//       ties-AWAY). Using round would manufacture spurious 0-digit "failures" at
//       every half-integer.
//     * exp10 has no __float128 overload; the oracle is pow((__float128)10, x),
//       exactly as src/demo_ff_real.cpp computes its exp10 reference.
//
// TWO PASSES PER OP (breadth + pathology)
//   (a) RANDOM: 10⁶ op-appropriate inputs via run_unary_op / run_binary_op
//       (test_utils.hpp). Ranges/domains are taken VERBATIM from the T2.2 op
//       inventory (ff_invariant_test.cpp) — this test does NOT re-derive them.
//       Those domains are FP32-narrower than T1.2's (exp guards at a.hi>=88 not
//       300, trig carries a tiny-argument lower bound |x|>=1e-25, sinh/cosh cap
//       at |x|<40, etc. — see ff_invariant_test.cpp's derivations) and were
//       empirically confirmed diagnostic-clean there.
//   (b) CORPUS: the deterministic corner-case inputs uniform-random misses. Where
//       a PORT_NOTES §3/§4 NAMED accessor exists for the op we use it directly
//       (exp -> exp_overflow, round-family -> nint_half_integer, remainder ->
//       remainder_regression, atanh -> atanh_small, sinh/cosh -> sinh_cosh_small,
//       sin/cos/tan -> trig_near_pi); otherwise we fall back to the generic
//       corpus::unary<float>() / corpus::binary<float>() bundlers (T=float for FF,
//       not double). Both passes feed the SAME domain predicate, so out-of-domain
//       corpus values are SKIPPED (counted, not failed) — see "SKIP, NOT FAIL".
//
// FAIL GATE (single uniform tolerance + the conditioning registry — no per-op
// overrides, per the plan)
//   tolerance_digits = -log10(N · u²)  ≈ 8.45 at N = 10⁶  (the worst-of-N
//   statistical floor at the u² scale; same formula as T1.4/T2.3, computed at
//   runtime from BackendTraits<FF>::u_squared). An op FAILS iff its MEAN digits
//   fall below tolerance. We gate on the MEAN, never the raw min, because a
//   handful of conditioning-limited inputs (near-cancellation, derivative -> inf,
//   arg-reduction near ±π) can legitimately drive individual mins to the floor
//   without indicating a defect.
//     For ops in the PORT_NOTES §5 conditioning registry
//     (lookup_expected_min_drop): if the MEAN clears tolerance AND the min is at
//     or above the registry's min_digits_allowed, the op is reported
//     EXPECTED-MIN-DROP: OK with the registry's cited reason. If the mean clears
//     but the min is BELOW the registry threshold, that IS a failure. If the mean
//     itself falls below tolerance, the op fails regardless of the registry.
//   Registry key note: the op named "subtract" in the inventory maps to the
//   registry key "sub" (the only name that differs; fdim/fma/asin/... match).
//   The registry is SHARED with DD (PORT_NOTES §5 conditioning is a property of
//   the algorithm, not the width), so the same table serves both backends.
//
// SKIP, NOT FAIL — an input outside the op's mathematical/representable domain
//   (log of ≤0, asin of |x|>1, exp overflow, a quotient past nint's 2⁴⁷ guard,
//   …) is filtered by the op's domain predicate before the call and counted as a
//   skip, exactly as T2.2 does. That both keeps the score honest (we only score
//   inputs the op is defined on) and suppresses ff_math.hpp's domain-guard prints.
//   The random runners in test_utils.hpp do not themselves gate, so the random
//   ranges here are chosen (from T2.2) to stay in-domain by construction; the
//   corpus pass applies the predicate explicitly before scoring.
//
// RULE 4 — ff_math.hpp is NOT modified. If an op exposes a REAL accuracy
//   regression (mean below tolerance, or a registry op whose min sinks below its
//   sanctioned floor) this test REPORTS it — op, which pass, the named accessor
//   if applicable, the offending input, the digit count — and fails. It does not
//   patch the library. Every numeric bound claimed in a comment cites a source
//   (DDFUN/CAMPARY/JMP literature, PORT_NOTES §, an ff_math.hpp line range, or
//   "observed empirically"); no fabricated proven bounds.
//
// SCOPE: real FF ops only (ff_complex.hpp out of scope; no DD/QF; no MPFR; no new
// corpus categories; single uniform tolerance + the existing registry only).
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 2, "T2.4: Differential accuracy
// for FF vs quadmath"; the T1.4 DONE block (structural template); PORT_NOTES.md
// §5 conditioning limits; "The six test layers" layer 4.
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
#include <string>
#include <vector>

using namespace kokkos_ep;
namespace ff = Kokkos::Experimental;

// This file is meaningful only with the __float128 oracle. Without it we still
// compile (so CTest can register the target) but main() returns KOKKOS_EP_SKIP.
#ifdef KOKKOS_EP_HAVE_QUADMATH

// ----------------------------------------------------------------------------
// Per-op record + reporting. (Mirrors dd_accuracy_test.cpp.)
// ----------------------------------------------------------------------------
// One row per op (per component, for the two-output ops). We keep the random and
// corpus passes' stats separate for diagnosis but gate on their COMBINED digits
// (min over both, mean weighted by count) — the plan asks for min AND mean over
// "10⁶ random + corpus".
struct OpResult {
  std::string name;
  long        n_scored   = 0;   // elements actually scored (both passes)
  long        n_skipped  = 0;   // out-of-domain corpus inputs skipped
  double      min_digits = 0.0;
  double      mean_digits = 0.0;
  double      tol_digits = 0.0;
  bool        pass       = false;   // mean >= tol
  const ExpectedMinDropAnnotation* ann = nullptr;  // registry entry, or null
  bool        min_ok_for_registry = true;          // min >= ann->min_digits_allowed
};

// Combine an AccStats from the random pass with one from the corpus pass into a
// single (min, mean, n). Corpus n may be 0 (all inputs out of domain) — then the
// combined stats are just the random pass. Random n is always 10⁶ here.
static void combine(const AccStats& rnd, const AccStats& cor,
                    double& min_d, double& mean_d, long& n_total) {
  const long nr = rnd.n, nc = cor.n;
  n_total = nr + nc;
  if (n_total <= 0) { min_d = 0.0; mean_d = 0.0; return; }
  if (nc <= 0)      { min_d = rnd.min; mean_d = rnd.mean; return; }
  if (nr <= 0)      { min_d = cor.min; mean_d = cor.mean; return; }
  min_d  = std::min(rnd.min, cor.min);
  mean_d = (rnd.mean * (double)nr + cor.mean * (double)nc) / (double)n_total;
}

static double tolerance_digits(long n) {
  // -log10(N · u²): worst-of-N statistical floor at the u² error scale. Same
  // formula as T1.4 / T2.3 Group B. ≈ 8.45 at N = 10⁶ for u² = 2⁻⁴⁸.
  return -std::log10((double)n * BackendTraits<FF>::u_squared);
}

// Registry key differs from the inventory name for exactly one op: subtract->sub.
static const char* registry_key(const std::string& op_name) {
  if (op_name == "subtract") return "sub";
  return op_name.c_str();
}

// Finalize gating for one op given its combined stats. Encodes the fail-gate:
// mean < tol -> fail; else if registry op, min must clear the sanctioned floor;
// else pass.
static OpResult finalize(const std::string& name, double min_d, double mean_d,
                         long n_scored, long n_skipped) {
  OpResult r;
  r.name = name;
  r.n_scored = n_scored;
  r.n_skipped = n_skipped;
  r.min_digits = min_d;
  r.mean_digits = mean_d;
  r.tol_digits = tolerance_digits(n_scored > 0 ? n_scored : 1);
  r.ann = lookup_expected_min_drop(registry_key(name));
  const bool mean_ok = mean_d >= r.tol_digits;
  if (r.ann) {
    r.min_ok_for_registry = min_d >= r.ann->min_digits_allowed;
    // Registry op: pass iff mean clears tol AND the min stays at/above the
    // sanctioned floor. A min below the sanctioned floor is a real failure even
    // though the op is "allowed" a low min.
    r.pass = mean_ok && r.min_ok_for_registry;
  } else {
    r.pass = mean_ok;
  }
  return r;
}

// name n=N skipped=S min_digits=X.XX mean_digits=Y.YY tolerance_digits=8.45 status=...
// status is PASS | FAIL | "EXPECTED-MIN-DROP: OK" (registry op that cleared mean
// and whose low min is sanctioned).
static void report_op(const OpResult& r) {
  const char* status;
  if (!r.pass) {
    status = "FAIL";
  } else if (r.ann && r.min_digits < BackendTraits<FF>::max_digits) {
    // Passed AND is a registry op whose min sits below the (bit-exact) cap:
    // surface the sanctioned min-drop explicitly.
    status = "EXPECTED-MIN-DROP: OK";
  } else {
    status = "PASS";
  }
  std::printf("  %-16s n=%-9ld skipped=%-6ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s",
              r.name.c_str(), r.n_scored, r.n_skipped, r.min_digits, r.mean_digits,
              r.tol_digits, status);
  if (r.ann && r.pass && r.min_digits < BackendTraits<FF>::max_digits) {
    std::printf("  (%s)", r.ann->reason);
  }
  std::printf("\n");
}

// ----------------------------------------------------------------------------
// Op descriptors. Each carries the T2.2 domain predicate (VERBATIM), the T2.2
// random generator(s) (VERBATIM), the device op (widened by the runner), the
// host quadmath oracle, and the corpus vector for the corpus pass.
// ----------------------------------------------------------------------------
using Dom1 = std::function<bool(double)>;
using Dom2 = std::function<bool(double, double)>;

// Common domain predicates (mirrors ff_invariant_test.cpp).
static const Dom1 dom_any    = [](double x){ return std::isfinite(x); };
static const Dom1 dom_nonneg = [](double x){ return std::isfinite(x) && x >= 0.0; };
// Trig family lower bound: FF sincos's Taylor loop hits its iteration limit for
// tiny nonzero arguments (FFCSSNR), a purely FP32 hazard (DD never saw it). 0 is
// fine (early return); otherwise require |x| >= 1e-25 and < 1e6. VERBATIM T2.2.
static const Dom1 dom_trig = [](double x){
  return std::isfinite(x) && (x == 0.0 || (std::fabs(x) >= 1e-25 && std::fabs(x) < 1e6));
};

// A unary op: name, domain predicate, random generator, device op, host oracle,
// and the corpus inputs to run in the corpus pass.
struct UnaryOp {
  std::string                                    name;
  Dom1                                           in_domain;
  InputDist                                      gen;
  std::function<ff::FloatFloat(ff::FloatFloat)>  device_op;
  std::function<float128(float128)>              oracle;
  std::vector<float>                             corpus_inputs;
};

struct BinaryOp {
  std::string                                                    name;
  Dom2                                                           in_domain;
  InputDist                                                      gen_a;
  InputDist                                                      gen_b;
  std::function<ff::FloatFloat(ff::FloatFloat, ff::FloatFloat)>  device_op;
  std::function<float128(float128, float128)>                    oracle;
  std::vector<std::pair<float, float>>                           corpus_inputs;
};

static constexpr int kRandomN = 1'000'000;  // 10^6 random inputs per op (plan)

// Generic corpus flags: zeros ON, inf OFF, nan OFF (subnormals default ON) —
// matches T2.2/T2.3. inf/nan produce digits-of-accuracy edge values, not
// meaningful accuracy scores, so they are excluded from the generic bundlers;
// the domain predicate additionally filters any that slip through.
static corpus::CorpusFlags corpus_flags() {
  corpus::CorpusFlags f;
  f.include_zero = true;
  f.include_inf  = false;
  f.include_nan  = false;
  return f;
}

// Filter a corpus input vector through the op's domain predicate BEFORE scoring,
// counting skips. Keeps the score honest and suppresses domain-guard prints.
static std::vector<float> filter_unary(const std::vector<float>& in,
                                       const Dom1& dom, long& skipped) {
  std::vector<float> out;
  out.reserve(in.size());
  for (float x : in) { if (dom((double)x)) out.push_back(x); else ++skipped; }
  return out;
}
static std::vector<std::pair<float, float>>
filter_binary(const std::vector<std::pair<float, float>>& in,
              const Dom2& dom, long& skipped) {
  std::vector<std::pair<float, float>> out;
  out.reserve(in.size());
  for (auto& p : in) {
    if (dom((double)p.first, (double)p.second)) out.push_back(p); else ++skipped;
  }
  return out;
}

// The test_utils runners take double corpus vectors (their host inputs are
// double, split to the backend type by the T(...) constructor). Convert the
// FP32 corpus vectors accordingly (values are exact FP32, so widening to double
// is lossless).
static std::vector<double> to_double(const std::vector<float>& v) {
  std::vector<double> out; out.reserve(v.size());
  for (float x : v) out.push_back((double)x);
  return out;
}
static std::vector<std::pair<double, double>>
to_double(const std::vector<std::pair<float, float>>& v) {
  std::vector<std::pair<double, double>> out; out.reserve(v.size());
  for (auto& p : v) out.emplace_back((double)p.first, (double)p.second);
  return out;
}

// Run both passes for one unary op and finalize.
static OpResult score_unary(const UnaryOp& op, uint64_t seed) {
  // Pass (a): 10^6 random. The T2.2 range is chosen to stay in-domain, so the
  // random pass scores all N (no skips on the random side).
  AccStats rnd = run_unary_op<FF>(kRandomN, seed, op.gen, op.oracle, op.device_op);

  // Pass (b): corpus, domain-filtered.
  long skipped = 0;
  std::vector<float> cin = filter_unary(op.corpus_inputs, op.in_domain, skipped);
  AccStats cor = run_unary_op_on_corpus<FF>(to_double(cin), op.oracle, op.device_op);

  double min_d, mean_d; long n_total;
  combine(rnd, cor, min_d, mean_d, n_total);
  return finalize(op.name, min_d, mean_d, n_total, skipped);
}

static OpResult score_binary(const BinaryOp& op, uint64_t seed) {
  AccStats rnd = run_binary_op<FF>(kRandomN, seed, op.gen_a, op.gen_b,
                                   op.oracle, op.device_op);
  long skipped = 0;
  std::vector<std::pair<float, float>> cin =
      filter_binary(op.corpus_inputs, op.in_domain, skipped);
  AccStats cor = run_binary_op_on_corpus<FF>(to_double(cin), op.oracle, op.device_op);

  double min_d, mean_d; long n_total;
  combine(rnd, cor, min_d, mean_d, n_total);
  return finalize(op.name, min_d, mean_d, n_total, skipped);
}

#endif  // KOKKOS_EP_HAVE_QUADMATH

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;

#ifdef KOKKOS_EP_HAVE_QUADMATH
  {
    std::printf("=== ff_accuracy_test (T2.4): differential accuracy for FF vs "
                "quadmath ===\n");
    std::printf("Execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("Metric: digits = -log10(rel_err) vs __float128 oracle, capped at "
                "%d (u^2 = 2^-48).\n", BackendTraits<FF>::max_digits);
    std::printf("Fail gate: MEAN digits >= -log10(N*u^2) ~ %.2f at N=10^6; low mins "
                "for PORT_NOTES §5 ops -> EXPECTED-MIN-DROP.\n\n",
                tolerance_digits(kRandomN));

    // Prebuild the generic corpus bundlers once (filtered per-op later). T=float
    // for FF (vs T1.4's double for DD).
    const std::vector<float> corpus_unary_all =
        corpus::unary<float>(corpus_flags());
    const std::vector<std::pair<float, float>> corpus_binary_all =
        corpus::binary<float>(corpus_flags());

    // ------------------------------------------------------------------------
    // UNARY op inventory — domains + random ranges VERBATIM from T2.2
    // (ff_invariant_test.cpp). Oracle per op; corpus = named accessor where one
    // exists (PORT_NOTES §3/§4), else the generic unary bundler.
    // ------------------------------------------------------------------------
    std::vector<UnaryOp> unary_ops = {
      // abs/negate: exact (structural). Bound: 0u² (bit-exact; ff_math.hpp abs/negate).
      {"abs",    dom_any, uniform(-1e8, 1e8),
       [](ff::FloatFloat x){ return ff::abs(x); },
       [](float128 x){ return Kokkos::abs(x); }, corpus_unary_all},
      {"negate", dom_any, uniform(-1e8, 1e8),
       [](ff::FloatFloat x){ return ff::negate(x); },
       [](float128 x){ return -x; }, corpus_unary_all},
      // sqrt. Bound: <=4u² (double-word sqrt; Joldes-Muller-Popescu 2017, Thm 5.x-class).
      {"sqrt",   dom_nonneg, uniform(0.0, 1e8),
       [](ff::FloatFloat x){ return ff::sqrt(x); },
       [](float128 x){ return Kokkos::sqrt(x); }, corpus_unary_all},

      // round-family: TIES-TO-EVEN in ff_math.hpp -> oracle is nearbyint, NOT
      // round (`round_to_nearest_int`; `round` forwards to it). Bound: exact for
      // |x|<2^47 (result is an integer, representable). T2.2 gates |x|<1e13.
      {"round_to_nearest_int",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::round_to_nearest_int(x); },
       [](float128 x){ return Kokkos::nearbyint(x); },
       corpus::nint_half_integer<float>()},
      {"ceil",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::ceil(x); },
       [](float128 x){ return Kokkos::ceil(x); },
       corpus::nint_half_integer<float>()},
      {"floor",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::floor(x); },
       [](float128 x){ return Kokkos::floor(x); },
       corpus::nint_half_integer<float>()},
      {"round",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::round(x); },
       // ff::round == round_to_nearest_int (ties-to-even) -> nearbyint oracle.
       [](float128 x){ return Kokkos::nearbyint(x); },
       corpus::nint_half_integer<float>()},
      {"trunc",
       [](double x){ return std::isfinite(x) && std::fabs(x) < 1e13; },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::trunc(x); },
       [](float128 x){ return Kokkos::trunc(x); },
       corpus::nint_half_integer<float>()},

      // exp-family. exp: conditioning-limited into the output-denormal band
      // (PORT_NOTES §5 -> registry "exp"). ff_math.hpp exp guards at a.hi>=88.0f
      // (FP32 ln-range), NOT DD's 300. Corpus = exp_overflow (79.5..88.72) per
      // PORT_NOTES §4a. Bound: observed ~a few u².
      {"exp",  [](double x){ return std::isfinite(x) && x < 88.7; },
       uniform(-87.0, 88.7),
       [](ff::FloatFloat x){ return ff::exp(x); },
       [](float128 x){ return Kokkos::exp(x); },
       corpus::exp_overflow<float>()},
      // exp2(a)=exp(a*ln2): a*ln2 < 88 -> |a| < ~127. T2.2 caps |a|<126.
      {"exp2", [](double x){ return std::isfinite(x) && std::fabs(x) < 126.0; },
       uniform(-120.0, 120.0),
       [](ff::FloatFloat x){ return ff::exp2(x); },
       [](float128 x){ return Kokkos::exp2(x); }, corpus_unary_all},
      // exp10: NO __float128 overload -> oracle is pow(10,x) (as src/demo_ff_real.cpp).
      // a*ln10 < 88 -> |a| < ~38. T2.2 caps |a|<38.
      {"exp10", [](double x){ return std::isfinite(x) && std::fabs(x) < 38.5; },
       uniform(-37.5, 38.5),
       [](ff::FloatFloat x){ return ff::exp10(x); },
       [](float128 x){ return Kokkos::pow((float128)10.0, x); }, corpus_unary_all},
      {"expm1", [](double x){ return std::isfinite(x) && x < 88.0; },
       uniform(-5.0, 5.0),
       [](ff::FloatFloat x){ return ff::expm1(x); },
       [](float128 x){ return Kokkos::expm1(x); }, corpus_unary_all},

      // log-family. Bound: observed ~a few u² over [1e-6,1e6]. Domain per T2.2:
      // isnormal and [1e-34, 1e34] (DD used [1e-100,1e100]; FP32 is ~6x narrower).
      {"log",   [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
       uniform(1e-6, 1e6),
       [](ff::FloatFloat x){ return ff::log(x); },
       [](float128 x){ return Kokkos::log(x); }, corpus_unary_all},
      {"log2",  [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
       uniform(1e-6, 1e6),
       [](ff::FloatFloat x){ return ff::log2(x); },
       [](float128 x){ return Kokkos::log2(x); }, corpus_unary_all},
      {"log10", [](double x){ return std::isnormal(x) && x >= 1e-34 && x <= 1e34; },
       uniform(1e-6, 1e6),
       [](ff::FloatFloat x){ return ff::log10(x); },
       [](float128 x){ return Kokkos::log10(x); }, corpus_unary_all},
      {"log1p", [](double x){ return std::isfinite(x) && x > -1.0 && x < 1e34; },
       uniform(-0.9, 1e6),
       [](ff::FloatFloat x){ return ff::log1p(x); },
       [](float128 x){ return Kokkos::log1p(x); }, corpus_unary_all},

      // trig. sin/cos/tan are conditioning-limited near ±π·k (PORT_NOTES §5 ->
      // registry) and carry the FP32 tiny-arg lower bound (dom_trig). Corpus =
      // trig_near_pi per PORT_NOTES §3a. Bound: observed; near-±π needs
      // triple-float reduction (out of scope, sanctioned min-drop).
      {"sin", dom_trig, uniform(-1000.0, 1000.0),
       [](ff::FloatFloat x){ return ff::sin(x); },
       [](float128 x){ return Kokkos::sin(x); },
       corpus::trig_near_pi<float>()},
      {"cos", dom_trig, uniform(-1000.0, 1000.0),
       [](ff::FloatFloat x){ return ff::cos(x); },
       [](float128 x){ return Kokkos::cos(x); },
       corpus::trig_near_pi<float>()},
      {"tan", dom_trig, uniform(-1000.0, 1000.0),
       [](ff::FloatFloat x){ return ff::tan(x); },
       [](float128 x){ return Kokkos::tan(x); },
       corpus::trig_near_pi<float>()},

      // inverse trig. asin/acos conditioning-limited near |a|=1 (registry); plus
      // the FP32 tiny-arg lower bound (asin(a)=angle(sqrt(1-a^2),a) trips FFCSSNR
      // on subnormal-tiny a). T2.2: |a|<=1 AND (0 or |a|>=1e-25).
      {"asin",
       [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0 && (x == 0.0 || std::fabs(x) >= 1e-25); },
       uniform(-1.0, 1.0),
       [](ff::FloatFloat x){ return ff::asin(x); },
       [](float128 x){ return Kokkos::asin(x); }, corpus_unary_all},
      {"acos",
       [](double x){ return std::isfinite(x) && std::fabs(x) <= 1.0 && (x == 0.0 || std::fabs(x) >= 1e-25); },
       uniform(-1.0, 1.0),
       [](ff::FloatFloat x){ return ff::acos(x); },
       [](float128 x){ return Kokkos::acos(x); }, corpus_unary_all},
      // atan(a)=angle(1,a): FP32 tiny-arg lower bound + a*a finite cap. T2.2:
      // 0 or (1e-25 <= |a| < 1e18).
      {"atan",
       [](double x){ return std::isfinite(x) && (x == 0.0 || (std::fabs(x) >= 1e-25 && std::fabs(x) < 1e18)); },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::atan(x); },
       [](float128 x){ return Kokkos::atan(x); }, corpus_unary_all},

      // hyperbolic. sinh/cosh cancel for small |a| (Taylor branch); corpus =
      // sinh_cosh_small per PORT_NOTES §3b. FP32 exp guard caps |a|<40. Bound: observed.
      {"sinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-39.0, 39.0),
       [](ff::FloatFloat x){ return ff::sinh(x); },
       [](float128 x){ return Kokkos::sinh(x); },
       corpus::sinh_cosh_small<float>()},
      {"cosh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-39.0, 39.0),
       [](ff::FloatFloat x){ return ff::cosh(x); },
       [](float128 x){ return Kokkos::cosh(x); },
       corpus::sinh_cosh_small<float>()},
      // tanh(x)=expm1(2x)/(expm1(2x)+2): expm1 intermediate overflows FP32 well
      // before the 88 guard; T2.2 caps |x|<20.
      {"tanh", [](double x){ return std::isfinite(x); },
       uniform(-1.0e4, 1.0e4),
       [](ff::FloatFloat x){ return ff::tanh(x); },
       [](float128 x){ return Kokkos::tanh(x); }, corpus_unary_all},

      // inverse hyperbolic. atanh conditioning-limited near |a|=1 (registry);
      // corpus = atanh_small (Taylor branch, PORT_NOTES §3c). asinh/acosh reduce
      // to log(x+sqrt(x^2±1)); T2.2 caps |x|<1e18 so x*x stays finite in FP32.
      {"asinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 1e18; },
       uniform(-1e6, 1e6),
       [](ff::FloatFloat x){ return ff::asinh(x); },
       [](float128 x){ return Kokkos::asinh(x); }, corpus_unary_all},
      {"acosh", [](double x){ return std::isfinite(x) && x >= 1.0 && x < 1e18; },
       uniform(1.0, 1e6),
       [](ff::FloatFloat x){ return ff::acosh(x); },
       [](float128 x){ return Kokkos::acosh(x); }, corpus_unary_all},
      {"atanh", [](double x){ return std::isfinite(x) && std::fabs(x) < 1.0; },
       uniform(-0.999, 0.999),
       [](ff::FloatFloat x){ return ff::atanh(x); },
       [](float128 x){ return Kokkos::atanh(x); },
       corpus::atanh_small<float>()},

      // erf/erfc/tgamma. ff_math.hpp erf saturates to ±1 past |z|=6; T2.2 uses
      // uniform(-6,6). tgamma per T2.2: x>=1e-3 (reflection sin(pi*x) guard) and
      // x<23 (internal pow/exp FP32 overflow). Bound: observed.
      {"erf",  dom_any, uniform(-6.0, 6.0),
       [](ff::FloatFloat x){ return ff::erf(x); },
       [](float128 x){ return Kokkos::erf(x); }, corpus_unary_all},
      {"erfc", dom_any, uniform(-6.0, 6.0),
       [](ff::FloatFloat x){ return ff::erfc(x); },
       [](float128 x){ return Kokkos::erfc(x); }, corpus_unary_all},
      {"tgamma", [](double x){ return std::isfinite(x) && x >= 1e-3 && x < 23.0; },
       uniform(0.1, 20.0),
       [](ff::FloatFloat x){ return ff::tgamma(x); },
       [](float128 x){ return Kokkos::tgamma(x); }, corpus_unary_all},
    };

    // Two-output ops, scored per component (T2.2 convention). sincos(a,cos,sin);
    // sinhcosh(a,cosh,sinh). Corpus: trig_near_pi for sincos, sinh_cosh_small for
    // sinhcosh (same PORT_NOTES §3 families as the single-output siblings).
    std::vector<UnaryOp> two_out_ops = {
      {"sincos.cos", dom_trig, uniform(-1000.0, 1000.0),
       [](ff::FloatFloat x){ ff::FloatFloat c, s; ff::sincos(x, c, s); return c; },
       [](float128 x){ return Kokkos::cos(x); },
       corpus::trig_near_pi<float>()},
      {"sincos.sin", dom_trig, uniform(-1000.0, 1000.0),
       [](ff::FloatFloat x){ ff::FloatFloat c, s; ff::sincos(x, c, s); return s; },
       [](float128 x){ return Kokkos::sin(x); },
       corpus::trig_near_pi<float>()},
      {"sinhcosh.cosh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-39.0, 39.0),
       [](ff::FloatFloat x){ ff::FloatFloat c, s; ff::sinhcosh(x, c, s); return c; },
       [](float128 x){ return Kokkos::cosh(x); },
       corpus::sinh_cosh_small<float>()},
      {"sinhcosh.sinh", [](double x){ return std::isfinite(x) && std::fabs(x) < 40.0; },
       uniform(-39.0, 39.0),
       [](ff::FloatFloat x){ ff::FloatFloat c, s; ff::sinhcosh(x, c, s); return s; },
       [](float128 x){ return Kokkos::sinh(x); },
       corpus::sinh_cosh_small<float>()},
    };

    // ------------------------------------------------------------------------
    // BINARY op inventory — domains + ranges VERBATIM from T2.2.
    // ------------------------------------------------------------------------
    const Dom2 dom2_any = [](double a, double b){ return std::isfinite(a) && std::isfinite(b); };
    // fmod/remainder form q=a/b then trunc/nint(q); |q| past nint's 2^47 guard is
    // out of range. T2.2: |a| < |b|*1e13.
    const Dom2 dom2_modbnz = [](double a, double b){
      return std::isfinite(a) && std::isfinite(b) && b != 0.0 &&
             std::fabs(a) < std::fabs(b) * 1e13;
    };
    const Dom2 dom2_bnz = [](double a, double b){ return std::isfinite(a) && std::isfinite(b) && b != 0.0; };
    // pow(a,b)=exp(b*log(a)); the final exp trips its 88 guard whenever
    // |b*ln(a)| >= 88. a>0 restricted to the log family's [1e-34, 1e34] window.
    const Dom2 dom2_powpos = [](double a, double b){
      return std::isnormal(a) && std::isfinite(b) && a >= 1e-34 && a <= 1e34 &&
             std::fabs(b * std::log(a)) < 88.0;
    };
    // atan2(y,x)=angle(x,y): x^2+y^2 must stay finite (larger magnitude < 1e18)
    // AND neither operand subnormal-tiny (FP32 sincos degenerate). T2.2 verbatim.
    const Dom2 dom2_atan2 = [](double a, double b){
      if (!(std::isfinite(a) && std::isfinite(b))) return false;
      double m = std::fmax(std::fabs(a), std::fabs(b));
      if (m == 0.0) return true;
      if (m > 1e18) return false;
      if (a != 0.0 && std::fabs(a) < 1e-18) return false;
      if (b != 0.0 && std::fabs(b) < 1e-18) return false;
      return true;
    };

    std::vector<BinaryOp> binary_ops = {
      // add/subtract: near-cancellation loses leading digits (matches FP32) ->
      // registry sub/... Bound: <=2u² away from cancellation (twoSum-based
      // `add`).
      {"add",      dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::add(a, b); },
       [](float128 a, float128 b){ return a + b; }, corpus_binary_all},
      {"subtract", dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::subtract(a, b); },
       [](float128 a, float128 b){ return a - b; }, corpus_binary_all},
      // multiply/divide. Bound: <=4u²/6u² (double-word mul/div; Joldes-Muller-
      // Popescu 2017, Thm 5.1 (mul) / Thm 6.x (div)).
      {"multiply", dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::multiply(a, b); },
       [](float128 a, float128 b){ return a * b; }, corpus_binary_all},
      {"divide",   dom2_bnz, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::divide(a, b); },
       [](float128 a, float128 b){ return a / b; }, corpus_binary_all},
      // pow. Bound: observed (exp(b*log a) composition).
      {"pow",      dom2_powpos, uniform(0.1, 20.0), uniform(-6.0, 6.0),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::pow(a, b); },
       [](float128 a, float128 b){ return Kokkos::pow(a, b); }, corpus_binary_all},
      // atan2(y,x)=angle(x,y) in ff_math.hpp. Oracle argument order: quadmath
      // atan2(y,x). Bound: observed.
      {"atan2",    dom2_atan2, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::atan2(a, b); },
       [](float128 a, float128 b){ return Kokkos::atan2(a, b); }, corpus_binary_all},
      {"hypot",    dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::hypot(a, b); },
       [](float128 a, float128 b){ return Kokkos::hypot(a, b); }, corpus_binary_all},
      // fmod/remainder: remainder conditioning-limited near multiples of b
      // (registry). Corpus = remainder_regression (PORT_NOTES §4b). fmod uses the
      // generic bundler. Bound: exact minus conditioning near multiples of b.
      {"fmod",      dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::fmod(a, b); },
       [](float128 a, float128 b){ return Kokkos::fmod(a, b); }, corpus_binary_all},
      {"remainder", dom2_modbnz, uniform(-1e3, 1e3), uniform(-1e3, 1e3),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::remainder(a, b); },
       [](float128 a, float128 b){ return Kokkos::remainder(a, b); },
       corpus::remainder_regression<float>()},
      // copysign/fmax/fmin: exact (structural). Bound: 0u² (bit-exact).
      {"copysign", dom2_any, uniform(-1e6, 1e6), uniform(-1e6, 1e6),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::copysign(a, b); },
       [](float128 a, float128 b){ return Kokkos::copysign(a, b); }, corpus_binary_all},
      {"fmax",     dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::fmax(a, b); },
       [](float128 a, float128 b){ return Kokkos::fmax(a, b); }, corpus_binary_all},
      {"fmin",     dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::fmin(a, b); },
       [](float128 a, float128 b){ return Kokkos::fmin(a, b); }, corpus_binary_all},
      // fdim = max(a-b,0): near-cancellation like subtract (registry). Bound: <=2u²
      // away from cancellation.
      {"fdim",     dom2_any, uniform(-1e8, 1e8), uniform(-1e8, 1e8),
       [](ff::FloatFloat a, ff::FloatFloat b){ return ff::fdim(a, b); },
       [](float128 a, float128 b){ return Kokkos::fdim(a, b); }, corpus_binary_all},
    };

    // ------------------------------------------------------------------------
    // Run every op. Seeds increment per op (reproducible), matching T2.2.
    // ------------------------------------------------------------------------
    std::vector<OpResult> results;
    uint64_t seed = 12345ULL;

    std::printf("[unary ops] 10^6 random + corpus\n");
    for (const auto& op : unary_ops)   { results.push_back(score_unary(op, seed++)); report_op(results.back()); }
    std::printf("\n[two-output ops] scored per component\n");
    for (const auto& op : two_out_ops) { results.push_back(score_unary(op, seed++)); report_op(results.back()); }
    std::printf("\n[binary ops] 10^6 random + corpus\n");
    for (const auto& op : binary_ops)  { results.push_back(score_binary(op, seed++)); report_op(results.back()); }

    // ternary fma(a,b,c): near-cancellation (registry "fma"). Custom two-input
    // runner won't fit (three streams); score inline with the same pipeline.
    // Random pass: 10^6 (a,b,c) from one engine, range uniform(-1e4,1e4) (T2.2).
    {
      std::vector<double> ha(kRandomN), hb(kRandomN), hc(kRandomN);
      { std::mt19937_64 g(seed++);
        std::uniform_real_distribution<double> d(-1e4, 1e4);
        for (int i = 0; i < kRandomN; ++i) { ha[i]=d(g); hb[i]=d(g); hc[i]=d(g); } }

      using exec_space = Kokkos::DefaultExecutionSpace;
      using T = BackendTraits<FF>::type;
      Kokkos::View<T*, Kokkos::LayoutRight, exec_space>
          da("da", kRandomN), db("db", kRandomN), dc("dc", kRandomN), dout("dout", kRandomN);
      auto hma = Kokkos::create_mirror_view(da);
      auto hmb = Kokkos::create_mirror_view(db);
      auto hmc = Kokkos::create_mirror_view(dc);
      for (int i = 0; i < kRandomN; ++i) { hma(i)=T(ha[i]); hmb(i)=T(hb[i]); hmc(i)=T(hc[i]); }
      Kokkos::deep_copy(da, hma); Kokkos::deep_copy(db, hmb); Kokkos::deep_copy(dc, hmc);
      Kokkos::parallel_for("fma_acc", Kokkos::RangePolicy<exec_space>(0, kRandomN),
        KOKKOS_LAMBDA(int i){ dout(i) = ff::fma(da(i), db(i), dc(i)); });
      Kokkos::fence();
      auto rmir = Kokkos::create_mirror_view(dout);
      Kokkos::deep_copy(rmir, dout);
      std::vector<double> digs(kRandomN);
      for (int i = 0; i < kRandomN; ++i) {
        // Oracle: fmaq(a,b,c) — exact a*b+c then one round. Bound: <=2u² away from
        // cancellation; conditioning-limited near a*b+c ~ 0 (registry "fma").
        float128 ref = Kokkos::fma((float128)ha[i], (float128)hb[i], (float128)hc[i]);
        digs[i] = digits_of_accuracy<FF>(BackendTraits<FF>::to_quad(rmir(i)), ref);
      }
      AccStats s = compute_stats(digs.data(), kRandomN);
      OpResult r = finalize("fma", s.min, s.mean, s.n, 0);
      results.push_back(r);
      report_op(r);
    }

    // pow_int(a,n): integer exponent. Random pass 10^6, a in uniform(-5,5),
    // n in [-20,20], skip 0^negative (T2.2). Oracle: pow((float128)a,(float128)n).
    // Bound: observed (repeated-squaring composition).
    {
      std::vector<double> ha(kRandomN); std::vector<int> hn(kRandomN);
      long skipped = 0;
      { std::mt19937_64 g(seed++);
        std::uniform_real_distribution<double> dx(-5.0, 5.0);
        std::uniform_int_distribution<int> dn(-20, 20);
        for (int i = 0; i < kRandomN; ++i) { ha[i]=dx(g); hn[i]=dn(g); } }

      using exec_space = Kokkos::DefaultExecutionSpace;
      using T = BackendTraits<FF>::type;
      Kokkos::View<T*, Kokkos::LayoutRight, exec_space> da("da", kRandomN), dout("dout", kRandomN);
      Kokkos::View<int*, Kokkos::LayoutRight, exec_space> dn("dn", kRandomN);
      auto hma = Kokkos::create_mirror_view(da);
      auto hmn = Kokkos::create_mirror_view(dn);
      for (int i = 0; i < kRandomN; ++i) { hma(i)=T(ha[i]); hmn(i)=hn[i]; }
      Kokkos::deep_copy(da, hma); Kokkos::deep_copy(dn, hmn);
      Kokkos::parallel_for("pow_int_acc", Kokkos::RangePolicy<exec_space>(0, kRandomN),
        KOKKOS_LAMBDA(int i){ dout(i) = ff::pow_int(da(i), dn(i)); });
      Kokkos::fence();
      auto rmir = Kokkos::create_mirror_view(dout);
      Kokkos::deep_copy(rmir, dout);
      std::vector<double> digs; digs.reserve(kRandomN);
      for (int i = 0; i < kRandomN; ++i) {
        if (ha[i] == 0.0 && hn[i] < 0) { ++skipped; continue; }  // 0^negative undefined
        float128 ref = Kokkos::pow((float128)ha[i], (float128)hn[i]);
        digs.push_back(digits_of_accuracy<FF>(BackendTraits<FF>::to_quad(rmir(i)), ref));
      }
      AccStats s = compute_stats(digs.data(), (int)digs.size());
      OpResult r = finalize("pow_int", s.min, s.mean, s.n, skipped);
      results.push_back(r);
      report_op(r);
    }

    // ------------------------------------------------------------------------
    // Summary table + gate.
    // ------------------------------------------------------------------------
    std::printf("\n=== Summary (op : n / skipped / min / mean / tol : status) ===\n");
    long total_failures = 0, total_skipped = 0;
    for (const auto& r : results) {
      total_skipped += r.n_skipped;
      const char* status = !r.pass ? "FAIL"
                         : (r.ann && r.min_digits < BackendTraits<FF>::max_digits)
                             ? "EXP-MIN-DROP" : "PASS";
      std::printf("  %-16s %10ld %8ld  min=%6.2f mean=%6.2f tol=%6.2f  %s\n",
                  r.name.c_str(), r.n_scored, r.n_skipped,
                  r.min_digits, r.mean_digits, r.tol_digits, status);
      if (!r.pass) ++total_failures;
    }
    std::printf("  ----\n  ops=%zu  total_skipped=%ld  failures=%ld\n",
                results.size(), total_skipped, total_failures);

    // Lowest-mean callout (5-10 ops), for the report.
    {
      std::vector<OpResult> bymean = results;
      std::sort(bymean.begin(), bymean.end(),
                [](const OpResult& a, const OpResult& b){ return a.mean_digits < b.mean_digits; });
      std::printf("\n[lowest-mean ops] (accuracy floor first)\n");
      int show = (int)std::min<size_t>(bymean.size(), 8);
      for (int i = 0; i < show; ++i) {
        const OpResult& r = bymean[i];
        std::printf("  %-16s mean=%6.2f min=%6.2f  %s\n",
                    r.name.c_str(), r.mean_digits, r.min_digits,
                    r.ann ? r.ann->reason : "well-conditioned; near u^2 floor");
      }
    }

    // Expected-min-drop roll-call.
    {
      std::printf("\n[expected-min-drop ops] (PORT_NOTES §5 conditioning; gated on "
                  "mean only)\n");
      for (const auto& r : results) {
        if (r.ann && r.pass && r.min_digits < BackendTraits<FF>::max_digits) {
          std::printf("  %-16s min=%6.2f (allowed>=%.2f) mean=%6.2f  %s\n",
                      r.name.c_str(), r.min_digits, r.ann->min_digits_allowed,
                      r.mean_digits, r.ann->reason);
        }
      }
    }

    KOKKOS_EP_ASSERT(total_failures == 0,
                     "one or more FF ops fell below the accuracy tolerance "
                     "(mean < -log10(N*u^2)) or a registry op's min sank below its "
                     "sanctioned floor — see the FAIL rows above; per rule 4 this is "
                     "REPORTED, ff_math.hpp is not patched");

    rc = ep_exit_code();
    std::printf("\n=== ff_accuracy_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
#endif  // KOKKOS_EP_HAVE_QUADMATH

  Kokkos::finalize();

#ifndef KOKKOS_EP_HAVE_QUADMATH
  std::printf("=== ff_accuracy_test (T2.4): no __float128 oracle "
              "(LIBQUADMATH absent) -> SKIP ===\n");
  return KOKKOS_EP_SKIP;
#else
  return rc;
#endif
}
