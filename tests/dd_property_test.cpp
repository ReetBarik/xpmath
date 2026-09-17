// ============================================================================
// dd_property_test.cpp — Layer 3 (property / identity tests) for DD.  Plan T1.3.
// ============================================================================
//
// WHAT THIS LAYER CHECKS
// ----------------------
// Layer 1 (dd_eft_test, T1.1) proved the two atoms (twoSum, Dekker twoProduct)
// are bit-exact; Layer 2 (dd_invariant_test, T1.2) proved every op returns a
// non-overlapping (hi, lo). This layer checks ALGEBRAIC identities the composed
// ops must satisfy. Two flavours:
//
//   GROUP A — bit-exact identities.  Pure sign/structure identities that MUST
//     hold to the last bit with NO tolerance and NO oracle: additive inverse,
//     self-subtraction, multiply by ±1, |a| sign branches, double negation, and
//     addition commutativity. These are gated by exact FP64 equality on both
//     components. Group A runs unconditionally — it needs no __float128.
//
//   GROUP B — approximate identities.  Round-trips (sqrt/square, exp/log),
//     Pythagorean sin²+cos², sin/cos symmetry, tan·cos, double-angle, and the
//     DEMOTED multiply-commutativity (see below). These carry a rounding
//     tolerance and are reported in "digits of accuracy" via the __float128
//     machinery in test_utils_host.hpp. Group B and Test C used to be #ifdef'd on
//     KOKKOS_EP_HAVE_QUADMATH and to SKIP (77) without it; they now run
//     unconditionally, because __float128 needs no TPL.
//
//   TEST C — named-constant regressions.  sin(π)≈0, log(e)≈1, exp(log2)≈2,
//     √2·√2≈2, log(10)≈ln(10) constant — each to ≥30 digits.
//
// CLASSIFICATION NOTES (decided by inspecting dd_math.hpp, then confirmed by the
// run) — why A7 is DEMOTED and A8 is NOT:
//
//   * add(a,b) is Knuth twoSum folding a.lo+b.lo. For
//     inputs built from a plain double (lo==0), the twoSum error term is EXACT
//     and independent of operand order, and the +a.lo+b.lo tail vanishes; hence
//     add(a,b)==add(b,a) BIT-EXACTLY on single-double inputs → Group A (A8). We
//     therefore feed A8 single-double operands (the whole suite's convention);
//     with a nonzero .lo the trailing (E+a.lo)+b.lo would reorder and lose
//     bit-exactness — that generalization is deliberately NOT claimed here.
//
//   * multiply(a,b) is Dekker, whose cross-term sum
//     c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2 adds a1*b2 and a2*b1 in a
//     FIXED a-first order. Swapping operands reorders those two addends, and FP
//     addition is not associative, so multiply is NOT bit-symmetric even on
//     single-double inputs. So the task's A7 (multiply commutativity) is DEMOTED
//     from Group A to Group B with a rounding tolerance (identity "mul_comm").
//
//   * sin(-a)==-sin(a) / cos(-a)==cos(a): `sincos` reduces
//     the argument mod 2π and recovers sin as is·√(1-cos²). The reduction and
//     recovery are not guaranteed sign-symmetric bit-for-bit, so B5/B6 are placed
//     in Group B with a tight (~2u²) tolerance rather than asserted bit-exact.
//     (The run reports how close to bit-exact they actually land.)
//
// DELIBERATELY NOT TESTED (anti-tests) — see the block near the bottom of main():
// associativity of add and distributivity across large-magnitude differences.
// Both are FALSE for any finite-precision format, DD included; asserting them
// would be testing IEEE arithmetic's known non-properties, not DD correctness.
//
// SCOPE: real DD ops only (dd_complex.hpp out of scope). dd_math.hpp is NOT
// modified (rule 4): if an identity failed unexpectedly this test would REPORT
// (bit patterns) and fail, not patch the library.
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 1, "T1.3: Property/identity
// tests for DD" and "The six test layers" layer 3.
// ============================================================================

#include "test_utils_host.hpp"
#include "corpus.hpp"
#include <dd_math.hpp>

#include <algorithm>
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
// Bit-pattern helpers (probe_op.cpp / dd_eft_test.cpp hex format).
// ----------------------------------------------------------------------------
static uint64_t dbits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(double));
  return b;
}

// Exact FP64 equality on both components. NOTE: uses value equality (==), so
// +0.0 and -0.0 compare equal — intended, since the zero-result identities are
// specified as "hi==0.0 && lo==0.0", i.e. the value zero regardless of sign bit.
static bool dd_eq(const dd::DoubleDouble& x, const dd::DoubleDouble& y) {
  return x.hi == y.hi && x.lo == y.lo;
}
static bool dd_is_zero(const dd::DoubleDouble& x) {
  return x.hi == 0.0 && x.lo == 0.0;
}

static void print_fail_unary(const char* id, double x, const dd::DoubleDouble& got,
                             const dd::DoubleDouble& want) {
  std::printf("    FAIL %-10s x=%.17g (0x%016llx)  got.hi=%.17g (0x%016llx) "
              "got.lo=%.17g (0x%016llx)  want.hi=%.17g (0x%016llx) "
              "want.lo=%.17g (0x%016llx)\n",
              id, x, (unsigned long long)dbits(x),
              got.hi, (unsigned long long)dbits(got.hi),
              got.lo, (unsigned long long)dbits(got.lo),
              want.hi, (unsigned long long)dbits(want.hi),
              want.lo, (unsigned long long)dbits(want.lo));
}
static void print_fail_binary(const char* id, double a, double b,
                              const dd::DoubleDouble& got, const dd::DoubleDouble& want) {
  std::printf("    FAIL %-10s a=%.17g (0x%016llx) b=%.17g (0x%016llx)  "
              "got.hi=%.17g (0x%016llx) got.lo=%.17g (0x%016llx)  "
              "want.hi=%.17g (0x%016llx) want.lo=%.17g (0x%016llx)\n",
              id, a, (unsigned long long)dbits(a), b, (unsigned long long)dbits(b),
              got.hi, (unsigned long long)dbits(got.hi),
              got.lo, (unsigned long long)dbits(got.lo),
              want.hi, (unsigned long long)dbits(want.hi),
              want.lo, (unsigned long long)dbits(want.lo));
}

// ----------------------------------------------------------------------------
// Uniform report line, shared by Group A and Group B.
//   name n=N min_digits=X.XX mean_digits=Y.YY tolerance_digits=Z.ZZ status=...
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
static constexpr double kMaxDig  = (double)BackendTraits<DD>::max_digits;  // 31

// ============================================================================
// GROUP A — bit-exact identities (no oracle, no tolerance)
// ============================================================================
// Each identity is a predicate on a DD value that must hold to the bit. A runner
// sweeps 10^6 random single-double inputs + the full finite corpus, counts
// failures, and dumps the first ≤3 with bit patterns. On zero failures the
// reported min/mean are the max_digits cap (bit-exact ⇒ "infinite" agreement,
// clamped to 31, per the plan).

struct GroupAResult { std::string name; long n; long skipped; long failures; };

// Domain predicates for Group A. Most sign/additive identities hold for every
// finite input, but the multiply-based ones (A3, A4) go through Dekker's Veltkamp
// split (a.hi*(2^27+1)); that product overflows to inf→nan for |x| >= 2^996, so
// such inputs are OUT OF multiply's domain and are SKIPPED, not failed. This is
// the same Dekker splitter-overflow limit documented in T1.1 (dd_eft_test), not
// a defect in multiply. 2^996 ≈ 6.7e299; random |x|<1e8 never approaches it, so
// this only gates the corpus's maxT / top-power-of-two / huge_tiny entries.
static const std::function<bool(double)> dom_all    = [](double x){ return std::isfinite(x); };
static const std::function<bool(double)> dom_dekker =
    [](double x){ return std::isfinite(x) && std::fabs(x) < std::ldexp(1.0, 996); };

// A unary identity: given a (as DD), return (got, want); dd_eq(got,want) must hold.
using UnaryIdentity = std::function<void(const dd::DoubleDouble& a,
                                         dd::DoubleDouble& got, dd::DoubleDouble& want)>;

static GroupAResult run_group_a_unary(const char* name, uint64_t seed,
                                      const InputDist& gen, const UnaryIdentity& id,
                                      const std::function<bool(double)>& in_domain = dom_all) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  auto one = [&](double x) {
    if (!in_domain(x)) { ++skipped; return; }
    dd::DoubleDouble a(x), got, want;
    id(a, got, want);
    ++n;
    if (!dd_eq(got, want)) {
      ++fails;
      if (samples_left > 0) { print_fail_unary(name, x, got, want); --samples_left; }
    }
  };
  { std::mt19937_64 g(seed); for (int i = 0; i < kRandomN; ++i) one(gen(g)); }
  for (double x : corpus::unary<double>(corpus_flags())) one(x);

  bool pass = (fails == 0);
  std::printf("  %-16s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

// A binary identity (only A8 add-commutativity here). Single-double operands.
using BinaryIdentity = std::function<void(const dd::DoubleDouble& a, const dd::DoubleDouble& b,
                                          dd::DoubleDouble& got, dd::DoubleDouble& want)>;

static GroupAResult run_group_a_binary(const char* name, uint64_t seed,
                                       const InputDist& ga, const InputDist& gb,
                                       const BinaryIdentity& id) {
  long n = 0, skipped = 0, fails = 0; int samples_left = 3;
  // add() does not Veltkamp-split, so it has no splitter-overflow limit; the only
  // out-of-domain case for A8 is a non-finite operand (skip, don't fail).
  auto one = [&](double av, double bv) {
    if (!(std::isfinite(av) && std::isfinite(bv))) { ++skipped; return; }
    dd::DoubleDouble a(av), b(bv), got, want;
    id(a, b, got, want);
    ++n;
    if (!dd_eq(got, want)) {
      ++fails;
      if (samples_left > 0) { print_fail_binary(name, av, bv, got, want); --samples_left; }
    }
  };
  { std::mt19937_64 g(seed);
    for (int i = 0; i < kRandomN; ++i) { double a = ga(g), b = gb(g); one(a, b); } }
  for (auto& p : corpus::binary<double>(corpus_flags())) one(p.first, p.second);

  bool pass = (fails == 0);
  std::printf("  %-16s n=%-9ld skipped=%-4ld min_digits=%6.2f mean_digits=%6.2f "
              "tolerance_digits=%6.2f status=%s\n",
              name, n, skipped, pass ? kMaxDig : 0.0, pass ? kMaxDig : 0.0,
              kMaxDig, pass ? "PASS" : "FAIL");
  return GroupAResult{name, n, skipped, fails};
}

// ============================================================================
// GROUP B — approximate identities (tolerance-based, needs the __float128 path)
// ============================================================================
// Each identity produces, per input, a (computed_quad, reference_quad) pair and
// scores digits_of_accuracy against it. We report min AND mean, and gate on the
// MEAN clearing the statistical threshold
//     tolerance_digits = -log10(N · u²),  u = 2^-53, u² = 2^-106,
// which is the worst-of-N floor for a per-sample error at the u² scale (the
// per-identity theoretical bound, 2u²/4u²/10u², is cited in a comment at each
// call site per plan rule 5). Gating on the mean — not the min — lets
// conditioning-limited samples (sin²+cos² far from a small argument, etc.) dip
// without red-flagging a correct implementation; low mins are annotated instead.

struct GroupBResult { std::string name; long n; double min_d, mean_d, tol_d; bool pass; };

static double threshold_digits(long n) {
  // -log10(N · u²): statistical floor over N samples at the u² error scale.
  return -std::log10((double)n * BackendTraits<DD>::u_squared);
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
    digs.push_back(digits_of_accuracy<DD>(c, r));
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
    digs.push_back(digits_of_accuracy<DD>(c, r));
  }
  AccStats s = compute_stats(digs.data(), (int)digs.size());
  double tol = threshold_digits((long)digs.size());
  bool pass = s.mean >= tol;
  report_line(name, s.n, s.min, s.mean, tol, pass);
  return GroupBResult{name, s.n, s.min, s.mean, tol, pass};
}

// Log-uniform generator: 10^u, u ~ Uniform[explo, exphi]. Spans magnitudes so
// the round-trip identities exercise the whole exponent range, not just [0,1].
static InputDist loguniform(double explo, double exphi) {
  return [explo, exphi](std::mt19937_64& g) {
    std::uniform_real_distribution<double> d(explo, exphi);
    return std::pow(10.0, d(g));
  };
}

// ============================================================================
// Device pass. 3 Group A (A1, A3, A5) + 2 Group B (B1, B4) on 10^5 inputs.
// A generic runner ships one DD result per input back to host; the caller does
// the (bit-exact for A / oracle for B) comparison. Catches a Serial→CUDA
// regression the host pass cannot see.
// ============================================================================
template <typename DeviceOp>
static void device_run(int n, uint64_t seed, const InputDist& gen, DeviceOp op,
                       std::vector<double>& x_out,
                       std::vector<double>& hi_out, std::vector<double>& lo_out) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  std::vector<double> hx(n);
  { std::mt19937_64 g(seed); for (int i = 0; i < n; ++i) hx[i] = gen(g); }

  Kokkos::View<double*, exec_space> dx("dx", n), dhi("dhi", n), dlo("dlo", n);
  auto hmx = Kokkos::create_mirror_view(dx);
  for (int i = 0; i < n; ++i) hmx(i) = hx[i];
  Kokkos::deep_copy(dx, hmx);

  Kokkos::parallel_for("dd_prop_dev", Kokkos::RangePolicy<exec_space>(0, n),
    KOKKOS_LAMBDA(int i) {
      dd::DoubleDouble d = op(dd::DoubleDouble(dx(i)));
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
    std::printf("=== dd_property_test (T1.3): algebraic identities for DD ===\n");
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
      [](const dd::DoubleDouble& a, dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::add(a, dd::negate(a)); want = dd::DoubleDouble(0.0, 0.0);
      }));
    // A2: a - a == 0, via operator-.
    ga.push_back(run_group_a_unary("A2_self_sub", seed++, wide,
      [](const dd::DoubleDouble& a, dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = a - a; want = dd::DoubleDouble(0.0, 0.0);
      }));
    // A3: a * 1 == a.  Dekker with b=(1,0) leaves both components unchanged.
    ga.push_back(run_group_a_unary("A3_mul_one", seed++, wide,
      [](const dd::DoubleDouble& a, dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::multiply(a, dd::DoubleDouble(1.0)); want = a;
      }, dom_dekker));
    // A4: a * (-1) == -a.  (Dekker split domain — see dom_dekker.)
    ga.push_back(run_group_a_unary("A4_mul_negone", seed++, wide,
      [](const dd::DoubleDouble& a, dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::multiply(a, dd::DoubleDouble(-1.0)); want = dd::negate(a);
      }, dom_dekker));
    // A5: |a| == a when a.hi>=0 else -a. Exercises BOTH sign branches (the wide
    //     symmetric range + corpus ±0/±powers hit the a.hi>=0 and a.hi<0 arms).
    ga.push_back(run_group_a_unary("A5_abs_branch", seed++, wide,
      [](const dd::DoubleDouble& a, dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::abs(a); want = (a.hi >= 0.0) ? a : dd::negate(a);
      }));
    // A6: -(-a) == a.
    ga.push_back(run_group_a_unary("A6_double_neg", seed++, wide,
      [](const dd::DoubleDouble& a, dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::negate(dd::negate(a)); want = a;
      }));
    // A8: add(a,b) == add(b,a).  Bit-exact for single-double operands — see the
    //     classification note in the file header (Knuth twoSum error is exact and
    //     order-independent; the +a.lo+b.lo tail is zero here).
    ga.push_back(run_group_a_binary("A8_add_comm", seed++, wide, uniform(-1e8, 1e8),
      [](const dd::DoubleDouble& a, const dd::DoubleDouble& b,
         dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::add(a, b); want = dd::add(b, a);
      }));

    for (const auto& r : ga) groupA_failures += r.failures;

    // ------------------------------------------------------------------------
    // GROUP B — approximate identities (tolerance, digits vs oracle).
    // ------------------------------------------------------------------------
    std::printf("\n[Group B] approximate identities (digits of accuracy vs __float128)\n");
    std::vector<GroupBResult> gb;

    // A7 (DEMOTED): multiply(a,b) ≈ multiply(b,a). Reference = one ordering,
    // computed = the other; digits measure how close to bit-symmetric the Dekker
    // cross-term sum is. Bound: ~2u² (a single reordered add in c21;
    // Joldes-Muller-Popescu 2017 double-word mul is 4u²-accurate, so the two
    // orderings differ by O(u²)).
    gb.push_back(run_group_b_binary("B0_mul_comm", kRandomN, seed++,
      uniform(-1e6, 1e6), uniform(-1e6, 1e6),
      [](double a, double b, float128& c, float128& r) {
        r = OracleTraits<DD>::to_quad(dd::multiply(dd::DoubleDouble(a), dd::DoubleDouble(b)));
        c = OracleTraits<DD>::to_quad(dd::multiply(dd::DoubleDouble(b), dd::DoubleDouble(a)));
      }));

    // B1: sqrt(a)² ≈ a, a>0 in [1e-30,1e30]. Bound: 4u² (sqrt double-word +
    // one double-word square; Joldes-Muller-Popescu 2017, Thm 5.1-class bound).
    gb.push_back(run_group_b_unary("B1_sqrt_sq", kRandomN, seed++, loguniform(-30, 30),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble s = dd::sqrt(dd::DoubleDouble(x));
        c = OracleTraits<DD>::to_quad(dd::multiply(s, s));
        r = (float128)x;
      }));
    // B2: exp(log(a)) ≈ a, a>0 in [1e-100,1e100]. Bound: 10u² (two transcendental
    // round-trips; observed scale, no tighter proven DDFUN bound available).
    gb.push_back(run_group_b_unary("B2_exp_log", kRandomN, seed++, loguniform(-100, 100),
      [](double x, float128& c, float128& r) {
        c = OracleTraits<DD>::to_quad(dd::exp(dd::log(dd::DoubleDouble(x))));
        r = (float128)x;
      }));
    // B3: log(exp(a)) ≈ a, a in [-290,290]. DEVIATION from the task's [-700,700]:
    // dd_math.hpp exp guards a.hi>=300 (returns 0 + prints), so ±290 keeps exp in
    // range and the log clean. Bound: 10u² (as B2).
    gb.push_back(run_group_b_unary("B3_log_exp", kRandomN, seed++, uniform(-290.0, 290.0),
      [](double x, float128& c, float128& r) {
        c = OracleTraits<DD>::to_quad(dd::log(dd::exp(dd::DoubleDouble(x))));
        r = (float128)x;
      }));
    // B4: sin²(a) + cos²(a) ≈ 1, a in [-100,100]. Bound: 10u². The min may dip
    // where the argument reduction is hardest (near a=±π·k); annotated below.
    gb.push_back(run_group_b_unary("B4_pythag", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble cc, ss; dd::sincos(dd::DoubleDouble(x), cc, ss);
        c = OracleTraits<DD>::to_quad(dd::add(dd::multiply(ss, ss), dd::multiply(cc, cc)));
        r = (float128)1.0;
      }));
    // B5: sin(-a) ≈ -sin(a), a in [-100,100]. Placed in Group B (not asserted
    // bit-exact) — see header. Bound: 2u² (one sign flip; expected near-exact).
    gb.push_back(run_group_b_unary("B5_sin_odd", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        c = OracleTraits<DD>::to_quad(dd::sin(dd::negate(dd::DoubleDouble(x))));
        r = OracleTraits<DD>::to_quad(dd::negate(dd::sin(dd::DoubleDouble(x))));
      }));
    // B6: cos(-a) ≈ cos(a). Bound: 2u².
    gb.push_back(run_group_b_unary("B6_cos_even", kRandomN, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        c = OracleTraits<DD>::to_quad(dd::cos(dd::negate(dd::DoubleDouble(x))));
        r = OracleTraits<DD>::to_quad(dd::cos(dd::DoubleDouble(x)));
      }));
    // B7: tan(a)·cos(a) ≈ sin(a), a in [-1.3,1.3] (avoids cos≈0 near ±π/2). Bound: 10u².
    gb.push_back(run_group_b_unary("B7_tan_cos", kRandomN, seed++, uniform(-1.3, 1.3),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble a(x);
        c = OracleTraits<DD>::to_quad(dd::multiply(dd::tan(a), dd::cos(a)));
        r = OracleTraits<DD>::to_quad(dd::sin(a));
      }));
    // B8: 2·sin(a)·cos(a) ≈ sin(2a), a in [-100,100], 10^5 samples. Bound: 10u².
    gb.push_back(run_group_b_unary("B8_double_angle", 100'000, seed++, uniform(-100.0, 100.0),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble a(x);
        c = OracleTraits<DD>::to_quad(dd::multiply_scalar(dd::multiply(dd::sin(a), dd::cos(a)), 2.0));
        r = OracleTraits<DD>::to_quad(dd::sin(dd::multiply_scalar(a, 2.0)));
      }));
    // B9: exp(a)·exp(-a) ≈ 1, a in [-290,290]. Bound: 4u².
    gb.push_back(run_group_b_unary("B9_exp_prod", kRandomN, seed++, uniform(-290.0, 290.0),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble a(x);
        c = OracleTraits<DD>::to_quad(dd::multiply(dd::exp(a), dd::exp(dd::negate(a))));
        r = (float128)1.0;
      }));
    // B10: hypot(a,b)² ≈ a²+b², a,b in [-1e3,1e3]. Bound: 10u².
    gb.push_back(run_group_b_binary("B10_hypot_sq", kRandomN, seed++,
      uniform(-1e3, 1e3), uniform(-1e3, 1e3),
      [](double a, double b, float128& c, float128& r) {
        dd::DoubleDouble da(a), db(b);
        dd::DoubleDouble h = dd::hypot(da, db);
        c = OracleTraits<DD>::to_quad(dd::multiply(h, h));
        r = OracleTraits<DD>::to_quad(dd::add(dd::multiply(da, da), dd::multiply(db, db)));
      }));
    // B11: pow(a,2) ≈ a·a, a>0 in [1e-15,1e15] (keeps 2·ln(a) well inside exp's
    // ±300 guard). Bound: 4u² (pow = exp(2·log a) vs one double-word square).
    gb.push_back(run_group_b_unary("B11_pow_two", kRandomN, seed++, loguniform(-15, 15),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble a(x);
        c = OracleTraits<DD>::to_quad(dd::pow(a, dd::DoubleDouble(2.0)));
        r = OracleTraits<DD>::to_quad(dd::multiply(a, a));
      }));
    // B12: atanh(a) ≈ ½·(log(1+a) - log(1-a)), |a|<0.5. Reformulated as an
    // equivalence (not "difference ≈ 0") so the digit metric is well-defined.
    // Bound: 10u².
    gb.push_back(run_group_b_unary("B12_atanh_log", kRandomN, seed++, uniform(-0.5, 0.5),
      [](double x, float128& c, float128& r) {
        dd::DoubleDouble a(x);
        c = OracleTraits<DD>::to_quad(dd::atanh(a));
        dd::DoubleDouble lhs = dd::log(dd::add(dd::DoubleDouble(1.0), a));
        dd::DoubleDouble rhs = dd::log(dd::subtract(dd::DoubleDouble(1.0), a));
        r = OracleTraits<DD>::to_quad(dd::multiply_scalar(dd::subtract(lhs, rhs), 0.5));
      }));

    // B4 min annotation (conditioning near ±π·k). lookup returns the sin
    // conditioning note; the identity itself is well-behaved so its mean stays
    // high, but we surface any low min honestly rather than hiding it.
    {
      const auto* ann = lookup_expected_min_drop("sin");
      const GroupBResult& b4 = *std::find_if(gb.begin(), gb.end(),
          [](const GroupBResult& g){ return g.name == "B4_pythag"; });
      std::printf("    note[B4_pythag]: min_digits=%.2f — dips are argument-reduction "
                  "conditioning near a=±pi*k (%s)\n", b4.min_d,
                  ann ? ann->reason : "sin near +/-pi; PORT_NOTES §5");
    }

    // ------------------------------------------------------------------------
    // Device pass: 3 Group A (bit-exact) + 2 Group B (tolerance) on 10^5 inputs.
    // ------------------------------------------------------------------------
    std::printf("\n[Device] 3 Group A (bit-exact) + 2 Group B on %d inputs (%s)\n",
                kDeviceN, Kokkos::DefaultExecutionSpace::name());
    long device_failures = 0;

    // Device A1: a + (-a) == 0.
    {
      std::vector<double> x, hi, lo;
      device_run(kDeviceN, 700001ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(dd::DoubleDouble a){ return dd::add(a, dd::negate(a)); }, x, hi, lo);
      long f = 0; for (int i = 0; i < kDeviceN; ++i) if (!(hi[i] == 0.0 && lo[i] == 0.0)) ++f;
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A1_add_neg", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }
    // Device A3: a * 1 == a (compare component-wise to the input DD).
    {
      std::vector<double> x, hi, lo;
      device_run(kDeviceN, 700002ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(dd::DoubleDouble a){ return dd::multiply(a, dd::DoubleDouble(1.0)); }, x, hi, lo);
      long f = 0; for (int i = 0; i < kDeviceN; ++i) if (!(hi[i] == x[i] && lo[i] == 0.0)) ++f;
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A3_mul_one", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }
    // Device A5: |a| == (a.hi>=0 ? a : -a).
    {
      std::vector<double> x, hi, lo;
      device_run(kDeviceN, 700003ULL, uniform(-1e8, 1e8),
        KOKKOS_LAMBDA(dd::DoubleDouble a){ return dd::abs(a); }, x, hi, lo);
      long f = 0;
      for (int i = 0; i < kDeviceN; ++i) {
        double wh = (x[i] >= 0.0) ? x[i] : -x[i];   // single-double: lo==0
        if (!(hi[i] == wh && lo[i] == 0.0)) ++f;
      }
      device_failures += f;
      std::printf("  [device] %-14s n=%d failures=%ld status=%s\n",
                  "A5_abs_branch", kDeviceN, f, f == 0 ? "PASS" : "FAIL");
    }

    // Device B1: sqrt(a)² ≈ a.
    {
      std::vector<double> x, hi, lo;
      device_run(kDeviceN, 700004ULL, loguniform(-30, 30),
        KOKKOS_LAMBDA(dd::DoubleDouble a){ dd::DoubleDouble s = dd::sqrt(a); return dd::multiply(s, s); },
        x, hi, lo);
      std::vector<double> digs(kDeviceN);
      for (int i = 0; i < kDeviceN; ++i)
        digs[i] = digits_of_accuracy<DD>(OracleTraits<DD>::to_quad(dd::DoubleDouble(hi[i], lo[i])),
                                         (float128)x[i]);
      AccStats s = compute_stats(digs.data(), kDeviceN);
      double tol = threshold_digits(kDeviceN); bool pass = s.mean >= tol;
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  "B1_sqrt_sq", kDeviceN, s.min, s.mean, tol, pass ? "PASS" : "FAIL");
    }
    // Device B4: sin²+cos² ≈ 1.
    {
      std::vector<double> x, hi, lo;
      device_run(kDeviceN, 700005ULL, uniform(-100.0, 100.0),
        KOKKOS_LAMBDA(dd::DoubleDouble a){
          dd::DoubleDouble cc, ss; dd::sincos(a, cc, ss);
          return dd::add(dd::multiply(ss, ss), dd::multiply(cc, cc));
        }, x, hi, lo);
      std::vector<double> digs(kDeviceN);
      for (int i = 0; i < kDeviceN; ++i)
        digs[i] = digits_of_accuracy<DD>(OracleTraits<DD>::to_quad(dd::DoubleDouble(hi[i], lo[i])),
                                         (float128)1.0);
      AccStats s = compute_stats(digs.data(), kDeviceN);
      double tol = threshold_digits(kDeviceN); bool pass = s.mean >= tol;
      if (!pass) ++device_failures;
      std::printf("  [device] %-14s n=%d min=%.2f mean=%.2f tol=%.2f status=%s\n",
                  "B4_pythag", kDeviceN, s.min, s.mean, tol, pass ? "PASS" : "FAIL");
    }

    // ------------------------------------------------------------------------
    // TEST C — named-constant regressions (each ≥30 digits).
    // ------------------------------------------------------------------------
    std::printf("\n[Test C] named-constant regressions (target ≥30 digits)\n");
    int c_pass = 0, c_total = 0;
    const double kNamedMin = 30.0;

    auto case_digits = [&](const char* name, float128 computed, float128 ref) {
      ++c_total;
      double d = digits_of_accuracy<DD>(computed, ref);
      bool ok = d >= kNamedMin;
      std::printf("    %-14s digits=%6.2f  %s\n", name, d, ok ? "PASS" : "FAIL");
      if (ok) ++c_pass;
    };

    // C1: sin(π) ≈ 0. digits-of-accuracy against a nonzero ref is ill-defined at
    // ref=0, so measure "digits of zero-ness" = -log10(|sin(π)|), softened by the
    // sin conditioning note near ±π (lookup_expected_min_drop("sin")).
    {
      ++c_total;
      dd::DoubleDouble sp = dd::sin(dd::DoubleDouble_pi());
      float128 v = OracleTraits<DD>::to_quad(sp);
      double zero_digits = (v == (float128)0.0) ? kMaxDig
                           : -(double)q_log10(q_abs(v));
      const auto* ann = lookup_expected_min_drop("sin");
      bool ok = zero_digits >= kNamedMin;
      std::printf("    %-14s |sin(pi)|=%.3e zero_digits=%6.2f  %s  (%s)\n",
                  "C1_sin_pi", (double)q_abs(v), zero_digits,
                  ok ? "PASS" : "FAIL", ann ? ann->reason : "sin near pi");
      if (ok) ++c_pass;
    }
    // C2: log(e) ≈ 1.
    case_digits("C2_log_e",
                OracleTraits<DD>::to_quad(dd::log(dd::DoubleDouble_e())), (float128)1.0);
    // C3: exp(log2) ≈ 2.
    case_digits("C3_exp_log2",
                OracleTraits<DD>::to_quad(dd::exp(dd::DoubleDouble_log2())), (float128)2.0);
    // C4: √2 · √2 ≈ 2.
    case_digits("C4_sqrt2_sq",
                OracleTraits<DD>::to_quad(dd::multiply(dd::DoubleDouble_sqrt2(),
                                                        dd::DoubleDouble_sqrt2())), (float128)2.0);
    // C5: log(10) ≈ the DoubleDouble_log10() constant (which stores ln(10)).
    case_digits("C5_log_ten",
                OracleTraits<DD>::to_quad(dd::log(dd::DoubleDouble(10.0))),
                OracleTraits<DD>::to_quad(dd::DoubleDouble_log10()));
    // C6: euler_gamma / digamma — SKIPPED: no digamma in dd_math.hpp, and the
    // stored euler_gamma constant has no independent DD op to check it against
    // (would be a tautology). Documented, not silently dropped.
    std::printf("    %-14s SKIPPED (no digamma op; constant has no independent "
                "DD oracle in dd_math.hpp)\n", "C6_gamma");

    std::printf("  Test C: %d/%d passed\n", c_pass, c_total);

    // ------------------------------------------------------------------------
    // ANTI-TESTS — identities DELIBERATELY NOT tested, and why.
    // ------------------------------------------------------------------------
    // These are FALSE for every finite-precision floating format, DD included;
    // asserting them would test IEEE arithmetic's known non-properties, not DD.
    //
    //  * Associativity of addition:  add(add(a,b),c) == add(a,add(b,c)).
    //    FP (and DD) addition is COMMUTATIVE but NOT ASSOCIATIVE — the two
    //    groupings round intermediate sums differently. A classic counterexample:
    //    a=1, b=2^-60, c=-1 (or the DD analogue) reassociates the tiny term across
    //    a cancellation and lands on different results. This is by design, not a
    //    bug; T1.4 already measures DD add's accuracy vs the oracle.
    //
    //  * Distributivity across large-magnitude differences:
    //    multiply(a, subtract(b,c)) == subtract(multiply(a,b), multiply(a,c)).
    //    When b and c are close and large, subtract(b,c) cancels to a small exact
    //    value while multiply(a,b) and multiply(a,c) are large and individually
    //    rounded; their difference carries the two roundings and cannot match the
    //    LHS bit-for-bit. Again a property of finite precision, not of DD.
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

    long groupB_failed = 0; for (const auto& r : gb) if (!r.pass) ++groupB_failed;
    std::printf("  Group B: %zu identities, mean below tolerance=%ld\n", gb.size(), groupB_failed);
    KOKKOS_EP_ASSERT(groupB_failed == 0,
                     "a Group B identity's MEAN digits fell below the -log10(N*u^2) tolerance");
    KOKKOS_EP_ASSERT(c_pass == c_total, "a Test C named-constant regression fell below 30 digits");

    std::printf("  Device: total failures=%ld\n", device_failures);
    KOKKOS_EP_ASSERT(device_failures == 0, "a device identity check failed");

    rc = ep_exit_code();
    std::printf("\n=== dd_property_test: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  }
  Kokkos::finalize();

  return rc;
}
