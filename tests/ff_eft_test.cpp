// ============================================================================
// ff_eft_test.cpp — Layer 1 (EFT unit tests) for the FF backend.  Plan T2.1.
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
// These two transforms are the atoms of float-float arithmetic. Every FF op in
// third_party/include/ff_math.hpp is built on the twoSum inside `add` and the
// Dekker twoProduct inside `multiply` (equivalently the standalone `two_prod`).
// If either EFT is not bit-exact, NOTHING downstream (sqrt/exp/log/sin/…)
// is trustworthy — the whole precision claim rests on these primitives. So this
// layer tests them in isolation, at the raw-float level, BEFORE any higher-level
// op is exercised (those are T2.2 invariants / T2.3 properties / T2.4 accuracy).
//
// This is the FF analogue of tests/dd_eft_test.cpp (T1.1). Every design decision
// T1.1 locked in carries over verbatim — mirror-and-comment for EFT primitive
// extraction, out-of-domain SKIP-not-fail, four test corpora (broad random +
// narrow random + |a|>>|b| + full corpus cross-product), host+device parity,
// oracle-independent domain predicate, per-target -ffp-contract=off. The ONE
// material difference is the oracle (see next block).
//
// WHY GROUND TRUTH IS PLAIN FP64 (PROVABLE, STRONGER THAN DD's QUADMATH ORACLE)
// ----------------------------------------------------------------------------
// The exact sum of two FP32 values needs at most 25 significant bits; the exact
// product needs at most 2*24 = 48 bits. IEEE double (FP64) has a 53-bit mantissa.
// Since 25 <= 53 and 48 <= 53, widening each FP32 operand to double and computing
// the sum/product THERE is EXACT — no rounding. Therefore
//     (double)s + (double)e  ==  (double)a + (double)b
//     (double)p + (double)e  ==  (double)a * (double)b
// is a *provable* bit-equality, not an approximate "close enough" check.
//
// This is a STRONGER oracle than DD's: DD's ground truth is __float128, which is
// exact for DD EFTs but only because binary128's 113-bit mantissa happens
// to exceed the 106-bit DD product — it is a higher-precision type doing the job.
// FF's FP64 oracle is exact by the same headroom argument but needs no type
// wider than a machine word: it is algebraically provable with the hardware
// double every compiler already has, and so runs on any target, not just x86_64.
// This is the single deliberate divergence from T1.1's shape.
//
// (This block used to add "does NOT need KOKKOS_EP_HAVE_QUADMATH / does NOT need
// a runtime SKIP-77 fallback / runs UNCONDITIONALLY". That is now true of every
// test in the suite — the gate is gone — so it no longer distinguishes this one.)
//
// WHY -ffp-contract=off IS REQUIRED (DEKKER SPLITTER CORRECTNESS)
// --------------------------------------------------------------
// FF's `multiply` uses Dekker splitting (splitter constant 8193.0f = 2^13+1),
// NOT an FMA-based twoProd. The correctness of the error term hinges on
// `a1*b1 - c11` being TWO distinct rounded operations. If the compiler contracts
// that into a single fused multiply-add, the "error" collapses to zero and the
// EFT silently breaks — the test would then validate a transform the shipped
// binary does not actually perform. This translation unit is therefore compiled
// with -ffp-contract=off on host (and --fmad=false on CUDA), applied by the
// kokkos_ep_add_eft_test() helper in tests/CMakeLists.txt. (T2.5 later builds the
// contraction-on reporter mirror; T2.1 only needs the posture here.)
//
// SPLITTER CONSTANT NOTE (deviation from the T2.1 prompt — see report)
// --------------------------------------------------------------------
// The T2.1 task prompt repeatedly names the FF splitter "2^12 + 1 = 8193.0f".
// That is arithmetically inconsistent: 2^12 + 1 = 4097, whereas 8193 = 2^13 + 1.
// The SHIPPED code uses 8193.0f (the `split` constant in `multiply`) and its own
// comment there correctly reads "Splitter = 2^13 + 1". This test mirrors the
// shipped constant 8193.0f and cites it correctly as 2^13 + 1. (Both 4097 and
// 8193 are in fact valid Veltkamp splitters that yield a bit-exact twoProduct on
// FP32 — verified empirically — but the transform under test is the one the
// binary runs, so 8193.0f is used.) A stale copy of the "2^12+1" typo also lives
// in ff_math.hpp's license header (the FP32-specific modifications list);
// flagged in the report, not fixed here (rule 4: this task does not modify
// ff_math.hpp).
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 2, "T2.1: EFT unit tests for
// FF" and "The six test layers" layer 1; PORT_NOTES.md §4a (splitter overflow).
//
// TEST STRUCTURE
//   Test A — twoSum bit-exactness  (10^6 + 10^6 + 10^5 random, full corpus x-prod)
//   Test B — Dekker twoProd bit-exactness (same corpus shape; splitter-overflow
//            and under/overflow regimes skipped — see Dekker's precondition note)
//   Test C — named hard cases (regression corpus + hand-picked)
//   Test D — device parity (run the SAME helpers in a Kokkos parallel_for)
// ============================================================================

// The device half of the test harness names only the xp core, so the Kokkos
// runtime this TU drives (View / parallel_for / initialize) is included here
// rather than arriving transitively through the harness header.
#include <Kokkos_Core.hpp>
#include "test_utils_device.hpp"
#include "corpus.hpp"
#include <ff_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace kokkos_ep;

// ----------------------------------------------------------------------------
// The two EFT primitives under test, mirrored from ff_math.hpp for RAW floats.
// ----------------------------------------------------------------------------
// DECISION (mirror-and-comment): ff_math.hpp's `add`/`multiply` embed the twoSum
// and Dekker twoProduct inside longer sequences that also fold in the input .lo
// components. For EFT testing we want the transform of two RAW floats (lo == 0),
// so we duplicate just the primitive here. This is the cleanest way to isolate
// exactly the algorithm under test, and the duplication doubles as documentation
// of it. ff_math.hpp is NOT modified (rule 4). When a.lo == b.lo == 0.0f, these
// helpers are bit-identical to the transforms embedded in `add` / `multiply` (the
// trailing FloatFloat renormalization there is a no-op on an already-non-
// overlapping (s, e) / (p, e) pair).

struct TwoOut { float hi; float lo; };

// Mirrors the twoSum embedded in Kokkos::Experimental::add, with
// a.lo == b.lo == 0. Knuth's twoSum: unconditionally exact for all finite
// FP32 a, b when a + b does not overflow (subnormals included — addition has no
// underflow hazard).
KOKKOS_INLINE_FUNCTION TwoOut two_sum(float a, float b) {
    float s   = a + b;
    float e   = s - a;
    float err = (b - e) + (a - (s - e));   // == the a.lo/b.lo-free t2 term in `add`
    return TwoOut{ s, err };               // hi = s = fl(a+b), lo = err = exact error
}

// Mirrors the Dekker twoProduct embedded in Kokkos::Experimental::multiply,
// equivalently the standalone `two_prod`, with
// a.lo == b.lo == 0. Splitter 8193.0f = 2^13 + 1. Exact provided no overflow
// occurs in the splitter (a*split), in a1*b1, or in the product, and no underflow
// occurs (Dekker 1971; Muller et al., "Handbook of Floating-Point Arithmetic",
// §4.4 — Veltkamp/Dekker require operands and result in the normal range).
KOKKOS_INLINE_FUNCTION TwoOut two_prod_dekker(float a, float b) {
    const float split = 8193.0f;             // 2^13 + 1  (matches `multiply`'s split)
    float cona = a * split, conb = b * split;
    float a1 = cona - (cona - a), b1 = conb - (conb - b);
    float a2 = a - a1,            b2 = b - b1;
    float p  = a * b;                                             // fl(a*b)
    float e  = (((a1 * b1 - p) + a1 * b2) + a2 * b1) + a2 * b2;   // exact error
    return TwoOut{ p, e };
}

// ----------------------------------------------------------------------------
// Oracle comparisons (host). Ground truth is plain FP64 — provably exact (25-bit
// sum / 48-bit product both fit in FP64's 53-bit mantissa). No __float128.
// ----------------------------------------------------------------------------
inline bool sum_is_exact(float a, float b) {
    TwoOut r = two_sum(a, b);
    double lhs = (double)r.hi + (double)r.lo;
    double rhs = (double)a    + (double)b;
    return lhs == rhs;
}
inline bool prod_is_exact(float a, float b) {
    TwoOut r = two_prod_dekker(a, b);
    double lhs = (double)r.hi + (double)r.lo;
    double rhs = (double)a    * (double)b;
    return lhs == rhs;
}

// ----------------------------------------------------------------------------
// Skip predicates — define the "no overflow / no underflow" domain each EFT is
// proven over. A pair outside the domain is not a failure; it is simply out of
// scope for a bit-exactness claim, so it is skipped (and does not count as
// "tested"). See the Dekker precondition note above.
// ----------------------------------------------------------------------------

// twoSum domain: finite a, b whose FP32 sum does not overflow. (No underflow
// hazard — Knuth twoSum is exact on subnormals too.)
inline bool sum_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return std::isfinite(a + b);
}

// Splitter-overflow bound, DERIVED from two_prod_dekker's body (cite in report):
// the first FP32 op the Dekker split performs is `cona = a * split` with
// split = 8193.0f. That product overflows to +inf once |a| * 8193 > FLT_MAX, i.e.
// |a| >= FLT_MAX / 8193. Then `cona - (cona - a)` becomes inf - inf = NaN and the
// error term is poisoned (this is exactly PORT_NOTES §4a's `exp` splitter-overflow
// mechanism, there triggered at b ~ 2^115). FLT_MAX = 2^128*(2-2^-23) ~ 3.4028e38,
// so FLT_MAX / 8193 ~ 4.1533e34 ~ 2^114.9998; empirically 2^114 * 8193 is finite
// and 2^115 * 8193 = inf, consistent with this bound. We reject |x| >= this for
// EITHER operand REGARDLESS of the other — even x*0 splits x first.
inline float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

// twoProd domain: finite normal (or zero) operands, with the TRUE product sitting
// comfortably inside the normal range (so both the rounded product p AND its error
// term e are representable without underflow). Dekker's twoProduct is exact only
// in the absence of underflow (Dekker 1971; Muller et al., "Handbook of Floating-
// Point Arithmetic", §4.4): the error term e ~ ulp(p) becomes subnormal — and thus
// loses bits — once |a*b| approaches FLT_MIN, so a genuine gradual-underflow
// product is OUT OF DOMAIN, not a defect in ff_math.hpp's multiply. We enforce the
// underflow/overflow test on the EXACT (double) product, not the rounded FP32
// product: the rounded product flushes to 0 in the underflow regime, which would
// otherwise let underflowing pairs masquerade as in-domain (the same trap the DD
// predicate documents). Subnormal OPERANDS are likewise excluded — splitting a
// subnormal is itself lossy.
//   Threshold FLT_MIN * 2^24 = 2^-102: the error term is at most ~ulp(p) below the
//   product, so requiring |p| >= 2^-102 keeps e >= ~2^-126 (still a normal FP32).
inline bool prod_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float fmin = std::numeric_limits<float>::min();   // smallest normal FP32
    auto normal_or_zero = [fmin](float x) {
        return x == 0.0f || std::fabs(x) >= fmin;           // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    // Splitter-overflow guard FIRST: the Dekker split of an operand overflows for
    // |x| >= split_safe_max REGARDLESS of the other operand — even x*0 splits x and
    // produces inf-inf = NaN. So this must gate the zero-product shortcut too.
    const float ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;  // splitter overflow
    if (a == 0.0f || b == 0.0f) return true;                       // exact product 0
    // Exact product magnitude, computed in FP64 (a*b is exact there: 48 <= 53).
    double tp  = (double)a * (double)b;
    double mag = tp < 0.0 ? -tp : tp;
    const double hi_lim = (double)std::numeric_limits<float>::max();  // FP32 overflow guard
    const double lo_lim = std::ldexp(1.0, -102);                      // underflow headroom
    if (mag > hi_lim) return false;   // product (and a1*b1) would overflow FP32
    if (mag < lo_lim) return false;   // error term would fall into FP32 subnormals
    return true;
}

// ----------------------------------------------------------------------------
// Failure-sample printer (first few only), with input bit patterns. FP32 hex is
// 0x%08x (probe_op.cpp convention), NOT the 0x%016llx DD uses for FP64.
// ----------------------------------------------------------------------------
inline void print_fail_sample(const char* which, float a, float b) {
    uint32_t ab, bb;
    std::memcpy(&ab, &a, sizeof(float));
    std::memcpy(&bb, &b, sizeof(float));
    TwoOut r = (std::strcmp(which, "twoSum") == 0) ? two_sum(a, b) : two_prod_dekker(a, b);
    std::printf("    FAIL %s  a=%.9g (0x%08x)  b=%.9g (0x%08x)  hi=%.9g lo=%.9g\n",
                which, (double)a, ab, (double)b, bb, (double)r.hi, (double)r.lo);
}

// ----------------------------------------------------------------------------
// Accumulator for one corpus/random batch.
// ----------------------------------------------------------------------------
struct EftCount { long tested = 0; long skipped = 0; long failures = 0; };

enum class Op { Sum, Prod };

inline void check_pair(Op op, float a, float b, EftCount& c, int& samples_left) {
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

// Broad-range bound, per op. FP32's exponent range is ~6x narrower than FP64's,
// so a single broad range cannot serve both transforms the way T1.1's [-1e100,
// 1e100] did for DD (there, 1e100 products = 1e200 still fit in DBL_MAX ~1.8e308):
//   * twoSum:  the SUM must stay finite. 1e30 (~2^99.6) is broad yet well below
//     FLT_MAX (~3.4e38), so sums never overflow — matches the T2.1 prompt's
//     [-1e30f,1e30f] broad range, justified there by the splitter-overflow bound.
//   * twoProduct: the PRODUCT must stay in FP32's NORMAL range. At 1e30, products
//     ~1e60 overflow FLT_MAX and are (correctly) domain-skipped — so a 1e30 broad
//     range leaves twoProduct with essentially ZERO tested pairs (empirically 1 of
//     1e6). FP32 forces a tighter broad range for products: 1e18 (~2^59.8) keeps
//     |a*b| <= 1e36 < FLT_MAX and each operand's split safe (1e18 << FLT_MAX/8193
//     ~4.15e34), so the broad-magnitude product path is actually exercised. This
//     is a deliberate, FP32-forced deviation from the prompt's single-range mirror
//     (the prompt reused one range for both ops, as T1.1 did; see report).
inline float broad_bound(Op op) { return op == Op::Sum ? 1e30f : 1e18f; }

// Run the four standard corpora (2 random ranges, 1 |a|>>|b| range, corpus
// cross-product) for one op and return the aggregate counts. Prints per-corpus
// subtotals. Ranges are FP32-appropriate (much tighter than DD's FP64 ranges).
static EftCount run_host_batches(Op op, const char* op_label) {
    EftCount total;
    int samples_left = 5;  // cap failure samples across all corpora for this op
    const float R = broad_bound(op);

    auto run_uniform = [&](const char* tag, int n, float lo, float hi, uint64_t seed) {
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<float> d(lo, hi);
        EftCount c;
        for (int i = 0; i < n; ++i) {
            float a = d(gen), b = d(gen);
            check_pair(op, a, b, c, samples_left);
        }
        std::printf("    [%s] %s: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, tag, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    };

    // Corpus 1: 10^6 uniform in [-R, R] (R = 1e30 for sum, 1e18 for product; see
    // broad_bound above for the FP32 range rationale).
    {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "uniform[-%.0e,%.0e] n=1e6", (double)R, (double)R);
        run_uniform(tag, 1'000'000, -R, R, 12345ULL);
    }
    // Corpus 2: 10^6 uniform in [-1,1] (subnormal-adjacent magnitudes reachable).
    run_uniform("uniform[-1,1] n=1e6",       1'000'000, -1.0f,  1.0f,  23456ULL);

    // Corpus 3: 10^5 pairs with |a| >> |b|: b = a * 2^-k, k in [1,20]. FP32's
    // mantissa is 24 bits (not FP64's 53), so k > 20 makes b subnormal for many a
    // and yields no useful bit-exactness signal — hence [1,20], not T1.1's [1,60].
    // a drawn from the same per-op broad range R so products stay in FP32 domain.
    {
        std::mt19937_64 gen(34567ULL);
        std::uniform_real_distribution<float> da(-R, R);
        std::uniform_int_distribution<int>    dk(1, 20);
        EftCount c;
        for (int i = 0; i < 100'000; ++i) {
            float a = da(gen);
            int   k = dk(gen);
            float b = a * std::ldexp(1.0f, -k);
            check_pair(op, a, b, c, samples_left);
        }
        std::printf("    [%s] |a|>>|b| (b=a*2^-k, k in [1,20]) n=1e5: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    }

    // Corpus 4: full corner-case corpus cross-product with itself, i < j.
    // corpus::unary<float> — the corpus is already parametric on float (T0.2). NaN
    // opt-in stays OFF (twoSum/twoProd are undefined on NaN/inf); non-finite pairs
    // are skipped by the domain predicates regardless. Include zero and subnormals
    // — they are in twoSum's domain (twoProd's domain check filters subnormals).
    {
        corpus::CorpusFlags flags;         // include_inf/zero/subnormals = true, nan = false
        std::vector<float> xs = corpus::unary<float>(flags);
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
// Test C — named hard cases. A named case reports PASS / SKIP / FAIL; the gate is
// on FAIL count only (SKIP is informational, e.g. both-subnormal for twoProduct,
// which is legitimately out of Dekker's domain).
// ----------------------------------------------------------------------------
struct NamedResult { int passed = 0; int skipped = 0; int failed = 0; int total = 0; };

static NamedResult run_named_cases() {
    NamedResult R;
    auto case_sum = [&](const char* name, float a, float b) {
        ++R.total;
        if (!sum_in_domain(a, b)) {
            ++R.skipped;
            std::printf("    twoSum  %-30s : SKIP (out of twoSum domain)\n", name);
            return;
        }
        bool ok = sum_is_exact(a, b);
        std::printf("    twoSum  %-30s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else { ++R.failed; print_fail_sample("twoSum", a, b); }
    };
    auto case_prod = [&](const char* name, float a, float b) {
        ++R.total;
        if (!prod_in_domain(a, b)) {
            ++R.skipped;
            std::printf("    twoProd %-30s : SKIP (out of Dekker domain)\n", name);
            return;
        }
        bool ok = prod_is_exact(a, b);
        std::printf("    twoProd %-30s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else { ++R.failed; print_fail_sample("twoProd", a, b); }
    };

    // Exact cancellation.
    case_sum("a == b (=1.0f)",            1.0f,  1.0f);
    case_sum("a == -b (=1.0f,-1.0f)",     1.0f, -1.0f);
    case_sum("a == b (=pi_f)",            3.14159265f, 3.14159265f);

    // Both subnormal. twoSum IS defined on subnormals (Knuth twoSum has no
    // underflow hazard), so it is tested and must pass; the SAME pair fed to the
    // Dekker twoProduct is SKIPPED (splitting a subnormal operand is lossy — out
    // of Dekker's domain, per the T2.1 prompt's "both-subnormal SKIPPED").
    {
        float s  = std::numeric_limits<float>::denorm_min();  // smallest FP32 subnormal
        float s3 = std::ldexp(s, 5);                          // still subnormal
        case_sum ("both subnormal",       s, s3);             // tested, must PASS
        case_prod("both subnormal",       s, s3);             // SKIPPED (out of domain)
    }

    // Zero with a nonzero partner, symmetric — defined for both ops.
    case_sum("a=0, b=nonzero",            0.0f, 3.5f);
    case_sum("a=nonzero, b=0",            3.5f, 0.0f);
    case_prod("a=0, b=nonzero",           0.0f, 3.5f);
    case_prod("a=nonzero, b=0",           3.5f, 0.0f);

    // +0 / -0, symmetric.
    case_sum("a=+0, b=-0",                0.0f, -0.0f);
    case_sum("a=-0, b=+0",               -0.0f,  0.0f);

    // Bailey's known hard twoSum case, FP32 form: 1.0f + 2^-24 rounds to 1.0f
    // (round-to-even, since 2^-24 is exactly half ulp(1.0f) = half of 2^-23), so
    // the error term must be exactly 2^-24.
    {
        float a = 1.0f, b = std::ldexp(1.0f, -24);   // 2^-24 = half ulp(1.0f)
        TwoOut r = two_sum(a, b);
        bool ok = sum_is_exact(a, b) && (r.hi == 1.0f) && (r.lo == std::ldexp(1.0f, -24));
        ++R.total;
        std::printf("    twoSum  %-30s : %s  (hi=%.9g lo=%.9g, want hi=1 lo=2^-24=%.9g)\n",
                    "Bailey 1.0f + 2^-24", ok ? "PASS" : "FAIL",
                    (double)r.hi, (double)r.lo, (double)std::ldexp(1.0f, -24));
        if (ok) ++R.passed; else ++R.failed;
    }

    // twoProd spot-checks against known references, evaluated at the FP32-nearest
    // neighbors of pi / e / sqrt(2) (bit-exact vs the FP64 product).
    case_prod("pi_f * pi_f",           3.14159265f, 3.14159265f);
    case_prod("e_f * e_f",             2.71828183f, 2.71828183f);
    case_prod("sqrt2_f * sqrt2_f",     1.41421356f, 1.41421356f);

    return R;
}

// ----------------------------------------------------------------------------
// Test D — device parity. Run the SAME helpers inside a Kokkos parallel_for,
// copy results back, and compare bit-exactly against the host FP64 oracle. On a
// Serial-only Kokkos this reduces to host execution (still a valid run); on
// CUDA/HIP/SYCL it catches device-side FP differences (subnormal flush,
// contraction) the host pass cannot see. Inputs are drawn from the splitter- and
// underflow-safe range so essentially no element is skipped. Both transforms run
// on the SAME input arrays, so the range must keep BOTH the sum finite AND the
// product inside FP32's normal range: [-1e18f, 1e18f] gives |a+b| <= 2e18 (finite)
// and |a*b| <= 1e36 < FLT_MAX with operands well under the splitter-overflow bound.
// (Using twoSum's wider 1e30 range here would domain-skip nearly every twoProduct
// pair — the vacuous-coverage trap FP32's narrow exponent range sets; see report.)
// ----------------------------------------------------------------------------
static NamedResult run_device_parity() {
    NamedResult R;
    using exec_space = Kokkos::DefaultExecutionSpace;
    const int nd = 200'000;

    std::vector<float> ha(nd), hb(nd);
    {
        std::mt19937_64 gen(99999ULL);
        std::uniform_real_distribution<float> d(-1e18f, 1e18f);
        for (int i = 0; i < nd; ++i) { ha[i] = d(gen); hb[i] = d(gen); }
    }

    Kokkos::View<float*, exec_space> va("va", nd), vb("vb", nd);
    Kokkos::View<float*, exec_space> s_hi("s_hi", nd), s_lo("s_lo", nd);
    Kokkos::View<float*, exec_space> p_hi("p_hi", nd), p_lo("p_lo", nd);
    auto hva = Kokkos::create_mirror_view(va);
    auto hvb = Kokkos::create_mirror_view(vb);
    for (int i = 0; i < nd; ++i) { hva(i) = ha[i]; hvb(i) = hb[i]; }
    Kokkos::deep_copy(va, hva);
    Kokkos::deep_copy(vb, hvb);

    Kokkos::parallel_for("ff_eft_device", Kokkos::RangePolicy<exec_space>(0, nd),
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

    long sum_fail = 0, prod_fail = 0, sum_skip = 0, prod_skip = 0;
    int samples_left = 5;
    for (int i = 0; i < nd; ++i) {
        float a = ha[i], b = hb[i];
        // twoSum parity (skip only genuinely out-of-domain pairs).
        if (sum_in_domain(a, b)) {
            double s_lhs = (double)hshi(i) + (double)hslo(i);
            double s_rhs = (double)a + (double)b;
            if (s_lhs != s_rhs) {
                ++sum_fail;
                if (samples_left > 0) { print_fail_sample("twoSum", a, b); --samples_left; }
            }
        } else { ++sum_skip; }
        // twoProd parity.
        if (prod_in_domain(a, b)) {
            double p_lhs = (double)hphi(i) + (double)hplo(i);
            double p_rhs = (double)a * (double)b;
            if (p_lhs != p_rhs) {
                ++prod_fail;
                if (samples_left > 0) { print_fail_sample("twoProd", a, b); --samples_left; }
            }
        } else { ++prod_skip; }
    }
    std::printf("    device twoSum : %ld tested (%ld skipped), %ld failures\n",
                (long)nd - sum_skip, sum_skip, sum_fail);
    std::printf("    device twoProd: %ld tested (%ld skipped), %ld failures\n",
                (long)nd - prod_skip, prod_skip, prod_fail);
    R.total   = 2 * nd;
    R.skipped = (int)(sum_skip + prod_skip);
    R.failed  = (int)(sum_fail + prod_fail);
    R.passed  = R.total - R.skipped - R.failed;
    return R;
}

// ============================================================================
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        std::printf("=== ff_eft_test (T2.1): EFT bit-exactness for FF twoSum + Dekker twoProd ===\n");
        std::printf("Oracle: FP64 (exact — 25-bit sum / 48-bit product both fit in 53-bit mantissa)\n");
        std::printf("Splitter: 8193.0f = 2^13 + 1 (mirrors ff_math.hpp multiply)\n\n");

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
        std::printf("  Test C named cases: %d passed, %d skipped, %d failed (of %d)\n\n",
                    C.passed, C.skipped, C.failed, C.total);
        KOKKOS_EP_ASSERT(C.failed == 0, "a named EFT case failed");

        // -- Test D: device parity ------------------------------------------
        std::printf("[Test D] device parity (%s)\n",
                    Kokkos::DefaultExecutionSpace::name());
        NamedResult D = run_device_parity();
        std::printf("  Test D device parity: %d passed, %d skipped, %d failed (of %d)\n\n",
                    D.passed, D.skipped, D.failed, D.total);
        KOKKOS_EP_ASSERT(D.failed == 0, "device EFT parity mismatch vs host FP64 oracle");

        rc = ep_exit_code();
        std::printf("=== ff_eft_test: %s ===\n", rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    }
    Kokkos::finalize();
    return rc;
}
