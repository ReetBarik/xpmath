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
//   exp 21.35 | log 22.01 | sin 22.17 | cos 22.68 | atan 7.76
//
// TOLERANCE TABLE (per-op, derived from measurement with margin)
// ---------------------------------------------------------------
// The phase-1 multiply bug measured 7.87 digits where ~23 was expected, and
// self-reporting missed it. This table must make a regression of that size fail
// loudly. Set each tolerance from measurement with a sensible margin.
//
// KNOWN-SHORT OPS (documented in PORT_NOTES_TF.md — NOT defects)
// ---------------------------------------------------------------
//   * atan, asin, acos, atan2, angle: FP32 scalar PLACEHOLDERS at ~7.5 digits,
//     unimplemented, a later phase. EXCLUDED from the scored inventory with an
//     explicit comment pointing at the port notes.
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
  return tf::TripleFloat::from_bits(f0, f1, f2);
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

static const OpTolerance kOpTols[] = {
  // Arithmetic (measured means: add 24.61, sub 24.57, mul 23.45, div 22.79, sqrt 22.97)
  {"add", 22.0},         // 2.6 margin below 24.61
  {"subtract", 22.0},    // 2.6 margin
  {"multiply", 21.0},    // 2.4 margin below 23.45
  {"divide", 20.5},      // 2.3 margin below 22.79
  {"sqrt", 20.5},        // 2.5 margin below 22.97
  {"sqr", 21.0},         // same as multiply
  {"abs", 22.0},         // exact, use add's margin
  {"negate", 22.0},      // exact, use add's margin

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
  {"pow", 19.0},         // exp-family, ~21.0-21.4 measured
  {"pow_int", 20.0},     // repeated multiply
  {"hypot", 20.0},       // sqrt-based

  // Binary ops
  {"fmod", 20.0},
  {"remainder", 19.0},
  {"fdim", 20.0},
  {"fmax", 22.0},        // exact selection
  {"fmin", 22.0},        // exact selection
  {"fma", 19.0},         // triple composition
  {"multiply_scalar", 21.0},  // scalar multiply

  // Rounding family
  {"round", 22.0},       // exact integer rounding
  {"ceil", 22.0},
  {"floor", 22.0},
  {"trunc", 22.0},
  {"round_to_nearest_int", 22.0},
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
};

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
      std::printf("  %-20s: mean %6.2f, min %6.2f, n=%5d, skip=%4d [tol %.2f] %s\n",
                  name, mean, R.min_dig, R.n, R.skip, tol, ok ? "PASS" : "FAIL");
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
    report("sqr", test_unary("sqr", [](auto x) { return tf::sqr(x); }, [](auto x) { return x * x; }, UnaryDomain{-1e15, 1e15}));
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

    // EXCLUDED: atan, asin, acos, atan2, angle (FP32 placeholders ~7.5 digits, PORT_NOTES_TF.md §8d.1)
    std::printf("  [atan, asin, acos, atan2, angle EXCLUDED: FP32 placeholders, see PORT_NOTES_TF.md §8d.1]\n");

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
    report("hypot", test_binary("hypot", [](auto a, auto b) { return xp::sqrt(xp::add(xp::sqr(a), xp::sqr(b))); },
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
    // copysign not implemented in tf_math.hpp
    report("fma", test_binary("fma", [](auto a, auto b) { return xp::fma(a, b, tf::TripleFloat(1.0)); },
                               [](auto a, auto b) { return fmaq(a, b, 1.0Q); }, BinaryDomain{-10.0, 10.0, -10.0, 10.0}));
    report("multiply_scalar", test_unary("multiply_scalar", [](auto x) { return xp::multiply_scalar(x, 3.14159f); },
                                          [](auto x) { return x * 3.14159Q; }, UnaryDomain{-1e15, 1e15}));

    // Rounding family
    std::printf("\n[Rounding]\n");
    report("round", test_unary("round", [](auto x) { return tf::round(x); }, roundq, UnaryDomain{-1e6, 1e6}));
    report("ceil", test_unary("ceil", [](auto x) { return tf::ceil(x); }, ceilq, UnaryDomain{-1e6, 1e6}));
    report("floor", test_unary("floor", [](auto x) { return tf::floor(x); }, floorq, UnaryDomain{-1e6, 1e6}));
    report("trunc", test_unary("trunc", [](auto x) { return tf::trunc(x); }, truncq, UnaryDomain{-1e6, 1e6}));
    report("round_to_nearest_int", test_unary("round_to_nearest_int",
                                               [](auto x) { return tf::round_to_nearest_int(x); },
                                               roundq, UnaryDomain{-1e6, 1e6}));

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
      rc = ep_exit_code();
    }

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
