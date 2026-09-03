// ============================================================================
// ff_complex_accuracy_test.cpp — per-op accuracy gate for FloatFloat COMPLEX
// ============================================================================
//
// WHY THIS FILE EXISTS
//
// Until this test landed, every ctest target in the suite scored REAL arithmetic
// only. The complex headers were exercised solely by the demos, which print a
// table nobody diffs per-op against an oracle. That is why KI-1 (complex acosh
// wrong for Re(z) < 0, in all four backends) survived from the original DD port
// through FF, QF and TF without a single red light. See docs/KNOWN_ISSUES.md.
//
// DECOUPLED FROM libquadmath
//
// This translation unit does NOT include <quadmath.h>, does NOT link
// -lquadmath and does NOT need -fext-numeric-literals. The binary128 reference
// values come from tests/data/xp_corpus_complex.bin, produced once on an x86_64
// GCC host by scripts/gen_corpus.cpp (which does use libquadmath). Scoring is
// done in plain `double` via corpus_binary.hpp's exact 3-double decode of the
// binary128 significand. This is the whole point: hipcc and icpx are clang-based
// and have no GCC-flavoured quadmath, so an oracle-linked test can only
// runtime-SKIP there, reporting no accuracy signal at all. This one runs.
//
//   verify:
//     g++ -std=c++17 -O2 -Iinclude -Itests
//         -DKOKKOS_EP_COMPLEX_CORPUS='"tests/data/xp_corpus_complex.bin"'
//         tests/ff_complex_accuracy_test.cpp -o /tmp/t && /tmp/t
//
// WHAT IS SCORED
//
// The 24 complex operations shared by all four backends, re-derived from
// include/xp/ff_complex.hpp: the four operators (add sub mul div) plus the 20
// free functions abs conj sqrt exp log log10 sin cos tan asin acos atan sinh
// cosh tanh asinh acosh atanh pow polar. QF and TF additionally expose norm()
// and arg(); DD and FF do not, and the corpus does not carry them, so they are
// out of scope here.
//
// Each element scores the real and imaginary components SEPARATELY (the
// convention of src/demo_ff_complex.cpp) as
//
//     digits = -log10(|got - ref| / |ref|)   clamped to [0, kMaxDig]
//
// and the element's score is min(d_re, d_im) — the weaker component decides.
// The gate is on the MEAN over the surviving elements, matching the real-side
// tests (tf_accuracy_test.cpp:644, qf_accuracy_test.cpp:1114-1121).
//
// THE CAP RULE (a trap this repo has already fallen into)
//
// A tolerance at or above kMaxDig makes a BIT-EXACT op register as FAIL, because
// an exact result scores exactly kMaxDig and the comparison is `mean >= tol`
// evaluated against a number the scorer can never exceed. Nine rows of
// tf_accuracy_test shipped that way. Every tolerance below is strictly less than
// kMaxDig.
//
// THIS TEST CAN ACTUALLY FAIL
//
// tf_accuracy_test once shipped calling ep_exit_code() with nothing ever
// incrementing the failure counter: it printed "Failed: 14" and exited 0. Each
// FAIL row here increments g_failures directly, and main() asserts
// passed == scored before returning ep_exit_code(). Verified by deliberately
// tightening a tolerance past a measured mean and confirming ctest reports
// FAILED.
//
// ============================================================================

#include "corpus_binary.hpp"

// The numeric headers print a one-line diagnostic on a domain violation. The
// corpus deliberately probes overflow-adjacent arguments, and elements that are
// perfectly VALID to score still trip an internal guard (tanh of a large real
// has a finite reference of +/-1, but evaluates exp(2x) on the way). Tens of
// thousands of such lines would bury the result table. config.hpp section 3
// guarantees the return values are identical with the diagnostics compiled out.
#define XPMATH_ENABLE_DIAGNOSTICS 0

#include "xp/ff_complex.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef KOKKOS_EP_COMPLEX_CORPUS
#define KOKKOS_EP_COMPLEX_CORPUS "tests/data/xp_corpus_complex.bin"
#endif

namespace cb = kokkos_ep::corpus_binary;

// ---------------------------------------------------------------------------
// Backend binding. Only this block differs between the four *_complex_accuracy
// tests; everything below it is identical by construction.
// ---------------------------------------------------------------------------
using Scalar  = xp::FloatFloat;
using Complex = xp::FloatFloatComplex;

// 2 x FP32 -> ~14 decimal digits. Same cap as ff_accuracy_test / demo_ff_complex.
static constexpr double kMaxDig = 14.0;

static constexpr const char* kBackend = "FF (FloatFloat, 2 x FP32)";

// A corpus input is a `double`, and the oracle evaluated the op at exactly that
// double. FloatFloat(double) keeps only ~48 bits (two FP32 limbs), so FF is the
// one backend scored on a value that differs from the one the reference saw —
// by up to 2^-48 relative. For a well-conditioned op that costs roughly 0.3
// digits off the 14.0 cap, which the tolerances below absorb. It bites hardest
// on the corpus's deliberately near-cancelling operand pairs, where the input
// rounding is amplified past the result itself — that is why FF's add and sub
// gates sit at 13.13/13.14 while DD/QF/TF hold those two ops bit-exact at their
// caps. Those elements are kept, not excluded: they measure something real about
// using FF with double-precision inputs. DD/QF/TF hold a double exactly
// (106 / 96 / 72 bits >= 53) and score the cancel family at full accuracy.
static Scalar mk_scalar(double x) { return Scalar(x); }

struct Got {
  double re[4];
  int    nre;
  double im[4];
  int    nim;
};

static Got words(Complex z) {
  Got g;
  g.nre = 2; g.re[0] = double(z.re.hi); g.re[1] = double(z.re.lo);
  g.nim = 2; g.im[0] = double(z.im.hi); g.im[1] = double(z.im.lo);
  return g;
}

static Got words_real(Scalar x) {
  Got g;
  g.nre = 2; g.re[0] = double(x.hi); g.re[1] = double(x.lo);
  g.nim = 1; g.im[0] = 0.0;
  return g;
}

// ---------------------------------------------------------------------------
// Failure plumbing. Deliberately NOT test_utils.hpp: that header unconditionally
// includes <Kokkos_Core.hpp>, which would drag Kokkos (and, through the build's
// interface flags, the quadmath define) back into a TU whose entire purpose is
// standing alone. Semantics mirror KOKKOS_EP_ASSERT / ep_exit_code exactly.
// ---------------------------------------------------------------------------
static int g_failures = 0;

static void check(bool cond, const char* msg) {
  if (!cond) {
    std::printf("ASSERT FAILED: %s\n", msg);
    ++g_failures;
  }
}

static int ep_exit_code() { return g_failures == 0 ? 0 : 1; }

// ---------------------------------------------------------------------------
// SCORING A COMPONENT WHOSE REFERENCE IS EXACTLY ZERO
//
// -log10(|got - ref| / |ref|) is undefined at ref == 0, and demanding a bit-exact
// zero is the wrong test. Concrete case that forced this: corpus element 1087 of
// `div` has b exactly equal to a/2, so the true quotient is 2 + 0i. QF returns
// im = 5.17e-39 — a residue 2.6e-39 relative to the quotient's modulus, i.e. ten
// orders of magnitude below QF's own 2^-96 resolution, and therefore a perfect
// answer. Scored as a relative error against zero it registers as 0.00 digits.
//
// So an exactly-zero reference component is scored in ABSOLUTE terms against the
// magnitude of the OTHER component: digits = -log10(|got| / |ref_other|). That is
// the standard measure for complex error and it still catches a genuinely wrong
// component — a sign-flipped or misplaced value is O(1) relative to the modulus
// and scores 0. If BOTH reference components are zero, exactness is required.
// ---------------------------------------------------------------------------
static double ref_mag(const cb::QuadRef& r) {
  if (r.cls != cb::QuadRef::kFinite || r.is_zero) return 0.0;
  return std::ldexp(r.w[0] + r.w[1] + r.w[2], r.e);
}

static double component_digits(const double* got, int n, const cb::QuadRef& r,
                               double other_mag) {
  if (r.cls == cb::QuadRef::kFinite && r.is_zero && other_mag > 0.0) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += got[i];
    if (s == 0.0) return kMaxDig;              // exactly zero: the best possible
    if (!std::isfinite(s)) return 0.0;
    const double d = -std::log10(std::fabs(s) / other_mag);
    if (!(d > 0.0)) return 0.0;
    return d > kMaxDig ? kMaxDig : d;
  }
  return cb::ref_digits(got, n, r, kMaxDig);
}

// ---------------------------------------------------------------------------
// Op inventory, in corpus order.
// ---------------------------------------------------------------------------
enum Op {
  kAdd, kSub, kMul, kDiv,
  kAbs, kConj, kSqrt, kExp, kLog, kLog10,
  kSin, kCos, kTan, kAsin, kAcos, kAtan,
  kSinh, kCosh, kTanh, kAsinh, kAcosh, kAtanh,
  kPow, kPolar,
  kOpCount
};

static const char* const kOpNames[kOpCount] = {
    "add", "sub", "mul", "div", "abs", "conj", "sqrt", "exp",
    "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "pow", "polar"};

static Got eval(int op, double are, double aim, double bre, double bim) {
  const Complex a(mk_scalar(are), mk_scalar(aim));
  const Complex b(mk_scalar(bre), mk_scalar(bim));
  switch (op) {
    case kAdd:   return words(a + b);
    case kSub:   return words(a - b);
    case kMul:   return words(a * b);
    case kDiv:   return words(a / b);
    case kAbs:   return words_real(xp::abs(a));
    case kConj:  return words(xp::conj(a));
    case kSqrt:  return words(xp::sqrt(a));
    case kExp:   return words(xp::exp(a));
    case kLog:   return words(xp::log(a));
    case kLog10: return words(xp::log10(a));
    case kSin:   return words(xp::sin(a));
    case kCos:   return words(xp::cos(a));
    case kTan:   return words(xp::tan(a));
    case kAsin:  return words(xp::asin(a));
    case kAcos:  return words(xp::acos(a));
    case kAtan:  return words(xp::atan(a));
    case kSinh:  return words(xp::sinh(a));
    case kCosh:  return words(xp::cosh(a));
    case kTanh:  return words(xp::tanh(a));
    case kAsinh: return words(xp::asinh(a));
    case kAcosh: return words(xp::acosh(a));
    case kAtanh: return words(xp::atanh(a));
    case kPow:   return words(xp::pow(a, b));
    // polar(r, theta): the corpus stores r in the real slot, theta in the
    // imaginary slot of operand a, and references rad*cos(th), rad*sin(th).
    case kPolar: return words(xp::polar(mk_scalar(are), mk_scalar(aim)));
  }
  std::printf("internal error: op index %d out of range\n", op);
  std::exit(2);
}

// ---------------------------------------------------------------------------
// SHARED MAGNITUDE WINDOW  (an exponent-range limit, NOT an accuracy excuse)
//
// All four backends score the SAME corpus elements, so the window is set by the
// narrowest word layout, which is QF's: its fourth FP32 limb sits ~2^-72 below
// the leading one and must stay normal (>= 2^-126), i.e. |x| >= 2^-54. Rounded
// to 2^-50 and mirrored on the high side, which also keeps a squared
// intermediate (abs, sqrt, div, log all form re^2 + im^2) inside FP32's normal
// range: (2^50)^2 = 2^100 < 2^128.
//
// An element is skipped when any input component, or either reference
// component, is nonzero with |x| outside [2^-50, 2^50], or when a reference
// component is non-finite. Exactly zero always passes: it is representable in
// every backend, and ref_digits() scores zero-vs-zero at the cap.
// ---------------------------------------------------------------------------
static constexpr double kMagLo = 0x1p-50;
static constexpr double kMagHi = 0x1p50;

static bool in_window(double x) {
  if (x == 0.0) return true;
  if (!std::isfinite(x)) return false;
  const double m = std::fabs(x);
  return m >= kMagLo && m <= kMagHi;
}

static bool ref_in_window(const cb::QuadRef& q) {
  if (q.cls != cb::QuadRef::kFinite) return false;
  if (q.is_zero) return true;
  return q.e >= -50 && q.e <= 50;
}

// ---------------------------------------------------------------------------
// PER-OP DOMAIN EXCLUSIONS
//
// Precedent: tf_accuracy_test.cpp keeps sqr_in_domain / mul_scalar_in_domain /
// inv_trig_in_domain / round_in_domain rather than loosening a tolerance to
// swallow a known defect. A tolerance says "this op is this accurate"; a domain
// predicate says "this op is not evaluated here, and here is why". Only the
// second is honest about a wrong answer.
// ---------------------------------------------------------------------------

// KI-1 — complex acosh returns the wrong branch whenever Re(z) < 0.
//
// docs/KNOWN_ISSUES.md describes the defect as living on the negative real axis.
// Measuring it here shows the region is WIDER than that: the shared identity
//
//     acosh(z) = log(z + sqrt(z*z - 1))          ff_complex.hpp:291-293
//
// takes the principal square root, so for Re(z) < 0 the sum z + sqrt(z^2 - 1)
// lands on the wrong sheet and the result comes back sign-flipped (and, on the
// real axis where the true real part is a difference of near-equal quantities,
// as outright cancellation noise). Off-axis example: z = -5 + 3i returns
// (-2.456, -2.593) where the principal value is (+2.4529, +2.5943). The reason
// KI-1's -0.5+0i and -1+0i rows score at the cap is that for real z in [-1, 1]
// the true real part is exactly 0, so a sign flip is invisible.
//
// Excluded by domain, not by tolerance, and NOT fixed here: the fix changes
// shipped DD/FF/QF numerical output, which is deliberately deferred (KI-1
// "Why it was not fixed when found"). Delete this predicate — in all four
// *_complex_accuracy_test.cpp files — the day the branch selection in the four
// *_complex.hpp headers is corrected; that is the acceptance test for the fix.
static bool extra_in_domain(int /*op*/, double /*are*/, double /*aim*/) {
  return true;
}

// ---------------------------------------------------------------------------
// PER-OP TOLERANCES — set from measurement, with margin, all < kMaxDig.
//
// Column meanings in the table below: `tol` is the gate; `measured` is the mean
// this build produces over the surviving elements. Margin is roughly 0.3-1.0
// digit, wider where the op's accuracy is condition-number-bounded rather than
// backend-bounded (src/demo_ff_complex.cpp's is_conditioning_limited list: sub,
// div, tan, asin, acos, atan, atanh, pow, log, log10 — for those, a below-cap
// mean is the mathematics, not a defect).
// ---------------------------------------------------------------------------
struct OpTol {
  const char* name;
  double      tol;
};

// clang-format off
static const OpTol kOpTols[kOpCount] = {
    // Measured on the committed corpus (tests/data/xp_corpus_complex.bin, n=2000
    // elements/op, seed 12345) with GCC 13.3.0. The numbers are insensitive to
    // -O0/-O2/-O3 and to -ffp-contract=off/fast, so the 0.30-digit margin is
    // there for a change of libm or of toolchain, not for run-to-run noise.
    //
    // Reading the `min` column: a low minimum is expected and is NOT a defect.
    // Componentwise relative accuracy is bounded by the conditioning of the
    // COMPONENT, and the corpus deliberately includes corner inputs (multiples of
    // pi/2, near-cancelling operand pairs) where one component of the true result
    // is a cancellation residue. The gate is on the mean for exactly that reason.

    //  op         tol        measured   worst    n     why the gate sits where it does
    {"add",     13.13},  //  13.43    0.00  1519   cancelling operand pairs
    {"sub",     13.14},  //  13.45    0.00  1504   conditioning-limited
    {"mul",     13.90},  //  14.00   12.92  1375   bit-exact
    {"div",     13.90},  //  14.00   13.16  1275   bit-exact
    {"abs",     13.90},  //  14.00   13.84  1059   bit-exact
    {"conj",    13.90},  //  14.00   14.00  1057   bit-exact
    {"sqrt",    13.90},  //  14.00   13.71  1052   bit-exact
    {"exp",     12.70},  //  13.00    0.00  1469   zeros of sin/cos near k*pi
    {"log",     13.47},  //  13.78    6.92  1054   conditioning-limited
    {"log10",   13.47},  //  13.77    6.92  1014   conditioning-limited
    {"sin",     12.66},  //  12.96    0.00  1535   zeros of sin/cos near k*pi
    {"cos",     12.59},  //  12.89    0.00  1264   zeros of sin/cos near k*pi
    {"tan",    13.03},  //  13.33    0.00  1486   KI-18 asymptotic branch: ratcheted, was 12.26
    {"asin",    10.82},  //  11.12    0.00  1563   conditioning-limited; KI-5(d) fixed: on-cut sheet
    {"acos",    10.69},  //  11.03    0.00  1572   KI-5(c) fixed (Kahan Re + exact Im)
    {"atan",   13.61},  //  13.91   12.63  1540   KI-11/KI-18 component form: ratcheted, was 9.28
    {"sinh",    12.68},  //  12.98    0.00  1539   zeros of sin/cos near k*pi
    {"cosh",    12.61},  //  12.91    0.00  1279   zeros of sin/cos near k*pi
    {"tanh",   12.99},  //  13.29    0.00  1472   KI-18 asymptotic branch: ratcheted, was 12.22
    {"asinh",   11.51},  //  11.81    0.00  1580   KI-5(a) fixed: reflected for Re(z) << 0
    {"acosh",   11.48},  //  11.78    0.00  1571   KI-1 fixed (Kahan): whole plane, was 957 elems
    {"atanh",  13.52},  //  13.82    0.15  1538   KI-11/KI-18 component form: ratcheted, was 12.08
    {"pow",     12.58},  //  12.88    0.00  1256   conditioning-limited
    {"polar",   12.76},  //  13.06    0.00  1338   zeros of sin/cos near k*pi
};
// clang-format on

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const std::string path = (argc > 1) ? argv[1] : std::string(KOKKOS_EP_COMPLEX_CORPUS);

  std::printf("=====================================================================\n");
  std::printf(" complex accuracy — %s\n", kBackend);
  std::printf(" corpus: %s\n", path.c_str());
  std::printf(" cap: %.2f digits | window: |x| in [2^-50, 2^50] or exactly 0\n", kMaxDig);
  std::printf("=====================================================================\n");

  std::vector<std::string> want;
  for (int i = 0; i < kOpCount; ++i) want.push_back(kOpNames[i]);

  double sum_mean = 0.0;
  int    scored = 0, passed = 0;

  try {
    cb::Corpus corpus(path);
    corpus.require_ops({}, want);
    corpus.verify_checksum();
    std::printf(" elements/op: %llu   seed: %llu   checksum OK\n\n",
                (unsigned long long)corpus.header().n_elems,
                (unsigned long long)corpus.header().seed);

    std::printf("  %-7s %8s %8s %8s %7s %7s   %s\n",
                "op", "mean", "min", "n", "skip", "tol", "result");
    std::printf("  ------- -------- -------- -------- ------- -------   ------\n");

    for (int op = 0; op < kOpCount; ++op) {
      const cb::ComplexOpData d = corpus.load_complex(kOpNames[op]);

      double sum = 0.0, min_dig = kMaxDig;
      long   n = 0, skip = 0;

      for (uint64_t i = 0; i < d.n; ++i) {
        const double are = d.a_re[i], aim = d.a_im[i];
        const double bre = d.n_operands >= 2 ? d.b_re[i] : 0.0;
        const double bim = d.n_operands >= 2 ? d.b_im[i] : 0.0;

        const cb::QuadRef rr = cb::decode_ref128(d.ref_re[i]);
        const cb::QuadRef ri = cb::decode_ref128(d.ref_im[i]);

        if (!in_window(are) || !in_window(aim) || !in_window(bre) || !in_window(bim) ||
            !ref_in_window(rr) || !ref_in_window(ri) || !extra_in_domain(op, are, aim)) {
          ++skip;
          continue;
        }

        const Got    g    = eval(op, are, aim, bre, bim);
        const double d_re = component_digits(g.re, g.nre, rr, ref_mag(ri));
        const double d_im = component_digits(g.im, g.nim, ri, ref_mag(rr));
        const double dig  = d_re < d_im ? d_re : d_im;

        sum += dig;
        if (dig < min_dig) min_dig = dig;
        ++n;
      }

      const double mean = n ? sum / double(n) : 0.0;
      const double tol  = kOpTols[op].tol;
      check(std::strcmp(kOpTols[op].name, kOpNames[op]) == 0,
            "tolerance table is out of order with the op inventory");
      check(tol < kMaxDig, "a tolerance is at or above the scoring cap — an exact op "
                           "would register as FAIL (see THE CAP RULE)");
      check(n > 0, "an op scored zero elements — the domain window excluded everything");

      const bool ok = n > 0 && mean >= tol;
      std::printf("  %-7s %8.2f %8.2f %8ld %7ld %7.2f   %s\n",
                  kOpNames[op], mean, min_dig, n, skip, tol, ok ? "PASS" : "FAIL");

      if (!ok) ++g_failures;
      if (ok) ++passed;
      sum_mean += mean;
      ++scored;
    }
  } catch (const cb::CorpusError& e) {
    std::printf("\nFATAL: %s\n", e.what());
    std::printf("Generate it with:\n"
                "  g++ -std=c++17 -fext-numeric-literals -O2 scripts/gen_corpus.cpp "
                "-lquadmath -o gen_corpus\n"
                "  ./gen_corpus --complex-only --n 2000 --out %s\n", path.c_str());
    return 1;
  }

  std::printf("\n  ops scored : %d\n", scored);
  std::printf("  passed     : %d\n", passed);
  std::printf("  failed     : %d\n", scored - passed);
  std::printf("  mean of means: %.2f digits (cap %.2f)\n",
              scored ? sum_mean / double(scored) : 0.0, kMaxDig);

  check(scored == kOpCount, "not every complex op was scored");
  check(passed == scored,
        "one or more complex ops fell below their accuracy tolerance "
        "(mean < per-op gate) — see the FAIL rows above");

  std::printf("\n%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
  return ep_exit_code();
}
