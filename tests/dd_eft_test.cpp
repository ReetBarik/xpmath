// ============================================================================
// dd_eft_test.cpp — Layer 1 (EFT unit tests) for the DD backend.  Plan T1.1.
// ============================================================================
//
// WHAT AN EFT IS AND WHY IT MATTERS
// ---------------------------------
// An "error-free transform" (EFT) rewrites a rounded floating-point operation
// as the rounded result PLUS an exactly-representable error term:
//
//   twoSum(a, b)  -> (s, e)   with  s = fl(a + b)  and  e = (a + b) - s exactly
//   twoProd(a, b) -> (p, e)   with  p = fl(a * b)  and  e = (a * b) - p exactly
//
// These two transforms are the atoms of double-double arithmetic. Every DD op
// in third_party/include/dd_math.hpp is built on the twoSum inside `add` and the
// Dekker twoProduct inside `multiply` (equivalently the standalone `two_prod`).
// If either EFT is not bit-exact, NOTHING downstream (sqrt/exp/log/sin/…)
// is trustworthy — the whole precision claim rests on these primitives. So this
// layer tests them in isolation, at the raw-double level, BEFORE any higher-
// level op is exercised (those are T1.2 invariants / T1.3 properties / T1.4
// accuracy).
//
// WHY GROUND TRUTH IS __float128 (PROVABLE, NOT MERELY HIGHER-PRECISION)
// ---------------------------------------------------------------------
// The exact sum of two FP64 values needs at most 54 significant bits; the exact
// product needs at most 2*53 = 106 bits. __float128 (IEEE binary128) has a
// 113-bit mantissa. Since 54 <= 113 and 106 <= 113, widening each FP64 operand
// to __float128 and computing the sum/product there is EXACT — no rounding.
// Therefore
//     (__float128)s + (__float128)e  ==  (__float128)a + (__float128)b
//     (__float128)p + (__float128)e  ==  (__float128)a * (__float128)b
// is a *provable* bit-equality, not an approximate "close enough" check. The
// oracle is __float128 arithmetic, which comes from libgcc and is available on
// every x86_64 build. (It used to come through Kokkos's quadmath overloads in
// impl/Kokkos_QuadPrecisionMath.hpp, so this test SKIPped when Kokkos was built
// without LIBQUADMATH; that gate is gone -- see test_utils.hpp.)
//
// WHY -ffp-contract=off IS REQUIRED (DEKKER SPLITTER CORRECTNESS)
// --------------------------------------------------------------
// DD's `multiply` uses Dekker splitting (splitter constant 134217729.0 = 2^27+1),
// NOT an FMA-based twoProd. The correctness of the error term hinges on
// `a1*b1 - c11` being TWO distinct rounded operations. If the compiler contracts
// that into a single fused multiply-add, the "error" collapses to zero and the
// EFT silently breaks — the test would then validate a transform the shipped
// binary does not actually perform. This translation unit is therefore compiled
// with -ffp-contract=off on host (and --fmad=false on CUDA), applied by the
// kokkos_ep_add_eft_test() helper in tests/CMakeLists.txt. (T1.5 later builds the
// full contraction-on/off regression matrix; T1.1 only needs the posture here so
// its own results are meaningful.)
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 1, "T1.1: EFT unit tests for
// DD" and "The six test layers" layer 1.
//
// TEST STRUCTURE
//   Test A — twoSum bit-exactness  (10^6 + 10^6 + 10^5 random, full corpus x-prod)
//   Test B — Dekker twoProd bit-exactness (same corpus shape; splitter-overflow
//            and under/overflow regimes skipped — see Dekker's precondition note)
//   Test C — named hard cases (regression corpus + hand-picked)
//   Test D — device parity (run the SAME helpers in a Kokkos parallel_for)
// ============================================================================

#include "test_utils.hpp"
#include "corpus.hpp"
#include <dd_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace kokkos_ep;


// ----------------------------------------------------------------------------
// The two EFT primitives under test, mirrored from dd_math.hpp for RAW doubles.
// ----------------------------------------------------------------------------
// DECISION (mirror-and-comment): dd_math.hpp's `add`/`multiply` embed the twoSum
// and Dekker twoProduct inside longer sequences that also fold in the input .lo
// components. For EFT testing we want the transform of two RAW doubles (lo == 0),
// so we duplicate just the primitive here. This is the cleanest way to isolate
// exactly the algorithm under test, and the duplication doubles as documentation
// of it. dd_math.hpp is NOT modified. When a.lo == b.lo == 0, these helpers are
// bit-identical to the transforms embedded in `add` / `multiply` (the trailing
// DoubleDouble renormalization there is a no-op on an already-nonoverlapping
// (s, e) / (p, e) pair).

struct TwoOut { double hi; double lo; };

// Mirrors the twoSum embedded in Kokkos::Experimental::add, with
// a.lo == b.lo == 0. Knuth's twoSum: unconditionally exact for all finite
// FP64 a, b when a + b does not overflow (subnormals included — addition has no
// underflow hazard).
KOKKOS_INLINE_FUNCTION TwoOut two_sum(double a, double b) {
    double s   = a + b;
    double e   = s - a;
    double err = (b - e) + (a - (s - e));   // == a.lo/b.lo-free t2 term in `add`
    return TwoOut{ s, err };                 // hi = s = fl(a+b), lo = err = exact error
}

// Mirrors the Dekker twoProduct embedded in Kokkos::Experimental::multiply,
// equivalently the standalone `two_prod`, with
// a.lo == b.lo == 0. Splitter 134217729.0 = 2^27 + 1. Exact provided no overflow
// occurs in the splitter (a*split), in a1*b1, or in the product, and no underflow
// occurs (Dekker 1971; Muller et al., "Handbook of Floating-Point Arithmetic",
// §4.4 — Veltkamp/Dekker require operands and result in the normal range).
KOKKOS_INLINE_FUNCTION TwoOut two_prod_dekker(double a, double b) {
    const double split = 134217729.0;        // 2^27 + 1
    double cona = a * split, conb = b * split;
    double a1 = cona - (cona - a), b1 = conb - (conb - b);
    double a2 = a - a1,            b2 = b - b1;
    double p  = a * b;                                             // fl(a*b)
    double e  = (((a1 * b1 - p) + a1 * b2) + a2 * b1) + a2 * b2;   // exact error
    return TwoOut{ p, e };
}

// ----------------------------------------------------------------------------
// Oracle comparisons (host only — __float128 is host-only).
// ----------------------------------------------------------------------------
inline bool sum_is_exact(double a, double b) {
    TwoOut r = two_sum(a, b);
    float128 lhs = (float128)r.hi + (float128)r.lo;
    float128 rhs = (float128)a   + (float128)b;
    return lhs == rhs;
}
inline bool prod_is_exact(double a, double b) {
    TwoOut r = two_prod_dekker(a, b);
    float128 lhs = (float128)r.hi + (float128)r.lo;
    float128 rhs = (float128)a   * (float128)b;
    return lhs == rhs;
}

// ----------------------------------------------------------------------------
// Skip predicates — define the "no overflow / no underflow" domain each EFT is
// proven over. A pair outside the domain is not a failure; it is simply out of
// scope for a bit-exactness claim, so it is skipped (and does not count as
// "tested"). See the Dekker precondition note above.
// ----------------------------------------------------------------------------

// twoSum domain: finite a, b whose sum does not overflow. (No underflow hazard.)
inline bool sum_in_domain(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return std::isfinite(a + b);
}

// 2^996 ~ 6.7e299. |a| < 2^996 keeps a*split (split < 2^28) below DBL_MAX, so the
// Dekker splitter does not overflow. a1*b1 overflow (possible when both operands
// approach 2^996) is caught separately by the product-finite check below.
inline double split_safe_max() { return std::ldexp(1.0, 996); }

// twoProd domain: finite normal (or zero) operands, with the TRUE product
// sitting comfortably inside the normal range (so both the rounded product p AND
// its error term e are representable without underflow). Dekker's twoProduct is
// exact only in the absence of underflow (Dekker 1971; Muller et al., "Handbook
// of Floating-Point Arithmetic", §4.4): the error term e ~ ulp(p) becomes
// subnormal — and thus loses bits — once |a*b| approaches DBL_MIN, so a genuine
// gradual-underflow product is OUT OF DOMAIN, not a defect in dd_math.hpp's
// multiply. We enforce this on the EXACT (float128) product, not the rounded FP64
// product: the rounded product flushes to 0 in the underflow regime, which would
// otherwise let underflowing pairs masquerade as in-domain (that was the trap the
// first cut of this predicate fell into). Subnormal OPERANDS are likewise
// excluded — splitting a subnormal is itself lossy.
//   Threshold DBL_MIN * 2^53 = 2^-969: the error term is at most ~ulp(p) below
//   the product, so requiring |p| >= 2^-969 keeps e >= ~2^-1022 (still normal).
inline bool prod_in_domain(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const double dmin = std::numeric_limits<double>::min();  // smallest normal
    auto normal_or_zero = [dmin](double x) {
        return x == 0.0 || std::fabs(x) >= dmin;             // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    // Splitter-overflow guard FIRST: the Dekker split of an operand overflows for
    // |x| >= split_safe_max REGARDLESS of the other operand — even x*0 splits x and
    // produces inf-inf = NaN. So this must gate the zero-product shortcut too.
    const double ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;  // splitter overflow
    if (a == 0.0 || b == 0.0) return true;                        // exact product 0
    // Exact product magnitude, computed in the 113-bit oracle (a*b is exact there).
    float128 tp = (float128)a * (float128)b;
    float128 mag = tp < (float128)0 ? -tp : tp;
    const float128 hi_lim = (float128)std::numeric_limits<double>::max();  // overflow guard
    const float128 lo_lim = (float128)std::ldexp(1.0, -969);              // underflow headroom
    if (mag > hi_lim) return false;   // product (and a1*b1) would overflow
    if (mag < lo_lim) return false;   // error term would fall into subnormals
    return true;
}

// ----------------------------------------------------------------------------
// Failure-sample printer (first few only), with input bit patterns.
// ----------------------------------------------------------------------------
inline void print_fail_sample(const char* which, double a, double b) {
    uint64_t ab, bb;
    std::memcpy(&ab, &a, sizeof(double));
    std::memcpy(&bb, &b, sizeof(double));
    TwoOut r = (std::strcmp(which, "twoSum") == 0) ? two_sum(a, b) : two_prod_dekker(a, b);
    std::printf("    FAIL %s  a=%.17g (0x%016llx)  b=%.17g (0x%016llx)  hi=%.17g lo=%.17g\n",
                which, a, (unsigned long long)ab, b, (unsigned long long)bb, r.hi, r.lo);
}

// ----------------------------------------------------------------------------
// Accumulator for one corpus/random batch.
// ----------------------------------------------------------------------------
struct EftCount { long tested = 0; long skipped = 0; long failures = 0; };

enum class Op { Sum, Prod };

inline void check_pair(Op op, double a, double b, EftCount& c, int& samples_left) {
    const bool in_domain = (op == Op::Sum) ? sum_in_domain(a, b) : prod_in_domain(a, b);
    if (!in_domain) { ++c.skipped; return; }
    ++c.tested;
    const bool ok = (op == Op::Sum) ? sum_is_exact(a, b) : prod_is_exact(a, b);
    if (!ok) {
        ++c.failures;
        if (samples_left > 0) {
            print_fail_sample(op == Op::Sum ? "twoSum" : "twoProd", a, b);
            --samples_left;
        }
    }
}

// Run the four standard corpora (2 random ranges, 1 |a|>>|b| range, corpus
// cross-product) for one op and return the aggregate counts. Prints per-corpus
// subtotals.
static EftCount run_host_batches(Op op, const char* op_label) {
    EftCount total;
    int samples_left = 5;  // cap failure samples across all corpora for this op

    auto run_uniform = [&](const char* tag, int n, double lo, double hi, uint64_t seed) {
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> d(lo, hi);
        EftCount c;
        for (int i = 0; i < n; ++i) {
            double a = d(gen), b = d(gen);
            check_pair(op, a, b, c, samples_left);
        }
        std::printf("    [%s] %s: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, tag, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    };

    // Corpus 1: 10^6 uniform in [-1e100, 1e100].
    run_uniform("uniform[-1e100,1e100] n=1e6", 1'000'000, -1e100, 1e100, 12345ULL);
    // Corpus 2: 10^6 uniform in [-1,1] (subnormal-adjacent magnitudes reachable).
    run_uniform("uniform[-1,1] n=1e6",         1'000'000, -1.0,   1.0,   23456ULL);

    // Corpus 3: 10^5 pairs with |a| >> |b|: b = a * 2^-k, k in [1,60].
    {
        std::mt19937_64 gen(34567ULL);
        std::uniform_real_distribution<double> da(-1e100, 1e100);
        std::uniform_int_distribution<int>     dk(1, 60);
        EftCount c;
        for (int i = 0; i < 100'000; ++i) {
            double a = da(gen);
            int    k = dk(gen);
            double b = a * std::ldexp(1.0, -k);
            check_pair(op, a, b, c, samples_left);
        }
        std::printf("    [%s] |a|>>|b| (b=a*2^-k) n=1e5: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    }

    // Corpus 4: full corner-case corpus cross-product with itself, i < j.
    // NaN opt-in stays OFF (twoSum/twoProd are undefined on NaN/inf); we skip any
    // non-finite pair via the domain predicates regardless. Include zero and
    // subnormals — they are in twoSum's domain (twoProd's domain check filters
    // subnormals itself). Cap pairs generously (spec: ~500*500 = 250k).
    {
        corpus::CorpusFlags flags;         // include_inf/zero/subnormals = true, nan = false
        std::vector<double> xs = corpus::unary<double>(flags);
        const size_t N = xs.size();
        const size_t kMaxPairs = 250'000;  // 500*500 guidance; corpus is far smaller
        EftCount c;
        size_t made = 0;
        for (size_t i = 0; i < N && made < kMaxPairs; ++i) {
            for (size_t j = i + 1; j < N && made < kMaxPairs; ++j) {
                check_pair(op, xs[i], xs[j], c, samples_left);
                ++made;
            }
        }
        std::printf("    [%s] corpus x-product (|corpus|=%zu, pairs=%zu): tested=%ld skipped=%ld failures=%ld\n",
                    op_label, N, made, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    }

    return total;
}

// ----------------------------------------------------------------------------
// Test C — named hard cases.
// ----------------------------------------------------------------------------
struct NamedResult { int passed = 0; int total = 0; };

static NamedResult run_named_cases() {
    NamedResult R;
    auto case_sum = [&](const char* name, double a, double b) {
        ++R.total;
        bool ok = sum_in_domain(a, b) && sum_is_exact(a, b);
        std::printf("    twoSum  %-28s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else print_fail_sample("twoSum", a, b);
    };
    auto case_prod = [&](const char* name, double a, double b) {
        ++R.total;
        bool ok = prod_in_domain(a, b) && prod_is_exact(a, b);
        std::printf("    twoProd %-28s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else print_fail_sample("twoProd", a, b);
    };

    // Exact cancellation.
    case_sum("a == b (=1.0)",            1.0,  1.0);
    case_sum("a == -b (=1.0,-1.0)",      1.0, -1.0);
    case_sum("a == b (=pi)",             3.141592653589793, 3.141592653589793);

    // Both subnormal (twoSum is defined on subnormals).
    {
        double s = std::numeric_limits<double>::denorm_min();  // smallest subnormal
        double s3 = std::ldexp(s, 5);                          // still subnormal
        case_sum("both subnormal",       s, s3);
    }

    // Zero with a nonzero partner, symmetric — defined for both ops.
    case_sum("a=0, b=nonzero",           0.0, 3.5);
    case_sum("a=nonzero, b=0",           3.5, 0.0);
    case_prod("a=0, b=nonzero",          0.0, 3.5);
    case_prod("a=nonzero, b=0",          3.5, 0.0);

    // +0 / -0, symmetric.
    case_sum("a=+0, b=-0",               0.0, -0.0);
    case_sum("a=-0, b=+0",              -0.0,  0.0);

    // Bailey's known hard twoSum case: 1.0 + 2^-53 rounds to 1.0 (round-to-even),
    // so the error term must be exactly 2^-53.
    {
        double a = 1.0, b = std::ldexp(1.0, -53);   // 2^-53 = half ulp(1.0)
        TwoOut r = two_sum(a, b);
        bool ok = sum_is_exact(a, b) && (r.hi == 1.0) && (r.lo == std::ldexp(1.0, -53));
        ++R.total;
        std::printf("    twoSum  %-28s : %s  (hi=%.17g lo=%.17g, want hi=1 lo=2^-53=%.17g)\n",
                    "Bailey 1.0 + 2^-53", ok ? "PASS" : "FAIL", r.hi, r.lo, std::ldexp(1.0, -53));
        if (ok) ++R.passed;
    }

    // twoProd spot-checks against known references (bit-exact vs float128 product).
    case_prod("pi * pi",           3.141592653589793, 3.141592653589793);
    case_prod("e * e",             2.718281828459045, 2.718281828459045);
    case_prod("sqrt(2) * sqrt(2)", 1.4142135623730951, 1.4142135623730951);

    return R;
}

// ----------------------------------------------------------------------------
// Test D — device parity. Run the SAME helpers inside a Kokkos parallel_for,
// copy results back, and compare bit-exactly against the host __float128 oracle.
// On a Serial-only Kokkos this reduces to host execution (still a valid run); on
// CUDA/HIP/SYCL it catches device-side FP differences (subnormal flush,
// contraction) the host pass cannot see. Inputs are drawn from the splitter- and
// underflow-safe range [-1e100, 1e100] so no element is skipped.
// ----------------------------------------------------------------------------
static NamedResult run_device_parity() {
    NamedResult R;
    using exec_space = Kokkos::DefaultExecutionSpace;
    const int nd = 100'000;

    std::vector<double> ha(nd), hb(nd);
    {
        std::mt19937_64 gen(99999ULL);
        std::uniform_real_distribution<double> d(-1e100, 1e100);
        for (int i = 0; i < nd; ++i) { ha[i] = d(gen); hb[i] = d(gen); }
    }

    Kokkos::View<double*, exec_space> va("va", nd), vb("vb", nd);
    Kokkos::View<double*, exec_space> s_hi("s_hi", nd), s_lo("s_lo", nd);
    Kokkos::View<double*, exec_space> p_hi("p_hi", nd), p_lo("p_lo", nd);
    auto hva = Kokkos::create_mirror_view(va);
    auto hvb = Kokkos::create_mirror_view(vb);
    for (int i = 0; i < nd; ++i) { hva(i) = ha[i]; hvb(i) = hb[i]; }
    Kokkos::deep_copy(va, hva);
    Kokkos::deep_copy(vb, hvb);

    Kokkos::parallel_for("dd_eft_device", Kokkos::RangePolicy<exec_space>(0, nd),
        KOKKOS_LAMBDA(int i) {
            TwoOut s = two_sum(va(i), vb(i));
            TwoOut p = two_prod_dekker(va(i), vb(i));
            s_hi(i) = s.hi; s_lo(i) = s.lo;
            p_hi(i) = p.hi; p_lo(i) = p.lo;
        });
    Kokkos::fence();

    auto hshi = Kokkos::create_mirror_view(s_hi);
    auto hslo = Kokkos::create_mirror_view(s_lo);
    auto hphi = Kokkos::create_mirror_view(p_hi);
    auto hplo = Kokkos::create_mirror_view(p_lo);
    Kokkos::deep_copy(hshi, s_hi);
    Kokkos::deep_copy(hslo, s_lo);
    Kokkos::deep_copy(hphi, p_hi);
    Kokkos::deep_copy(hplo, p_lo);

    long sum_fail = 0, prod_fail = 0;
    int samples_left = 5;
    for (int i = 0; i < nd; ++i) {
        double a = ha[i], b = hb[i];
        // twoSum parity
        float128 s_lhs = (float128)hshi(i) + (float128)hslo(i);
        float128 s_rhs = (float128)a + (float128)b;
        if (s_lhs != s_rhs) {
            ++sum_fail;
            if (samples_left > 0) { print_fail_sample("twoSum", a, b); --samples_left; }
        }
        // twoProd parity
        float128 p_lhs = (float128)hphi(i) + (float128)hplo(i);
        float128 p_rhs = (float128)a * (float128)b;
        if (p_lhs != p_rhs) {
            ++prod_fail;
            if (samples_left > 0) { print_fail_sample("twoProd", a, b); --samples_left; }
        }
    }
    std::printf("    device twoSum : %ld/%d passed\n",  (long)nd - sum_fail,  nd);
    std::printf("    device twoProd: %ld/%d passed\n",  (long)nd - prod_fail, nd);
    R.total  = 2 * nd;
    R.passed = (int)(2L * nd - sum_fail - prod_fail);
    return R;
}


// ============================================================================
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        std::printf("=== dd_eft_test (T1.1): EFT bit-exactness for DD twoSum + Dekker twoProd ===\n");
        std::printf("Oracle: __float128 (exact — 54-bit sum / 106-bit product both fit in 113-bit mantissa)\n\n");

        // -- Test A: twoSum --------------------------------------------------
        std::printf("[Test A] twoSum bit-exactness\n");
        EftCount A = run_host_batches(Op::Sum, "A");
        std::printf("  Test A twoSum: total tested=%ld, skipped=%ld, failures=%ld\n\n",
                    A.tested, A.skipped, A.failures);
        KOKKOS_EP_ASSERT(A.failures == 0, "twoSum was not bit-exact for some finite input pair");

        // -- Test B: Dekker twoProd -----------------------------------------
        std::printf("[Test B] Dekker twoProduct bit-exactness\n");
        EftCount B = run_host_batches(Op::Prod, "B");
        std::printf("  Test B Dekker twoProd: total tested=%ld, skipped=%ld, failures=%ld\n\n",
                    B.tested, B.skipped, B.failures);
        KOKKOS_EP_ASSERT(B.failures == 0, "Dekker twoProd was not bit-exact for some in-domain input pair");

        // -- Test C: named cases --------------------------------------------
        std::printf("[Test C] named hard cases\n");
        NamedResult C = run_named_cases();
        std::printf("  Test C named cases: %d/%d passed\n\n", C.passed, C.total);
        KOKKOS_EP_ASSERT(C.passed == C.total, "a named EFT case failed");

        // -- Test D: device parity ------------------------------------------
        std::printf("[Test D] device parity (%s)\n",
                    Kokkos::DefaultExecutionSpace::name());
        NamedResult D = run_device_parity();
        std::printf("  Test D device parity: %d/%d passed\n\n", D.passed, D.total);
        KOKKOS_EP_ASSERT(D.passed == D.total, "device EFT parity mismatch vs host binary128 oracle");

        rc = ep_exit_code();
        std::printf("=== dd_eft_test: %s ===\n", rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    }
    Kokkos::finalize();
    return rc;
}
