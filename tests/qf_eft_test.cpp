// ============================================================================
// qf_eft_test.cpp — Layer 1 (EFT unit tests) for the QF backend.  Plan T3.1.
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
// QuadFloat (4 x FP32) is a length-4 non-overlapping expansion. Every QF op in
// third_party/include/qf_math.hpp composes on these two FP32 EFTs plus a
// renormalization step (renorm / renorm_4) that collapses the wide unnormalized
// accumulator produced by add/multiply/divide back into a non-overlapping
// length-4 QuadFloat. If any of these primitives is not bit-exact / does not
// preserve value, NOTHING downstream (sqrt/exp/log/sin/…) is trustworthy — the
// whole ~29-digit precision claim rests on them. So this layer tests them in
// isolation, at the raw-float level, BEFORE any higher-level op is exercised
// (those are T3.2 invariants / T3.3 properties / T3.4 accuracy).
//
// This is the QF analogue of tests/ff_eft_test.cpp (T2.1) and tests/dd_eft_test
// .cpp (T1.1). Every design decision those locked in carries over: out-of-domain
// SKIP-not-fail, four test corpora (broad random + narrow random + |a|>>|b| +
// full corpus cross-product), host+device parity, oracle-independent domain
// predicate, per-target -ffp-contract=off.
//
// TWO STRUCTURAL DIFFERENCES FROM T2.1
// ------------------------------------
// (1) NO MIRROR-AND-COMMENT. T2.1 had to *duplicate* FF's twoSum / Dekker
//     twoProduct into the test file because ff_math.hpp embeds them inside the
//     longer add()/multiply() sequences — there was no standalone primitive to
//     call. qf_math.hpp is different: it EXPOSES the shipped primitives as free
//     functions in namespace Kokkos::Experimental —
//        qf_two_sum, qf_quick_two_sum, qf_two_prod, qf_two_sqr
//        and  renorm / renorm_4.
//     So this test calls the ACTUAL shipped code, not a mirror of it. That is a
//     strictly stronger test: a mirror can drift from the header; a direct call
//     cannot. (Rule 4 is trivially respected — we only #include, never edit.)
//
// (2) renorm_4 HAS NO FF ANALOGUE. FF's two-word type never renormalizes a wide
//     expansion — its "renorm" is the trailing quick_two_sum inside add/multiply,
//     already covered by the twoSum test. QF's renorm_4 (length-5 -> length-4)
//     and renorm (length-4 -> length-4) are genuinely QF-unique surface. Their
//     oracle strategy is therefore T3.1-original (see the renorm test block).
//
// FP32 PRIMITIVE COVERAGE CROSS-REFERENCE
// ---------------------------------------
// twoSum/twoProd FP32 primitive coverage also lives in tests/ff_eft_test.cpp
// (T2.1, commit 4e025b0): FF's twoSum is bit-identical to qf_two_sum and FF's
// Dekker twoProduct uses the SAME 8193.0f splitter as qf_two_prod. We do NOT
// skip them here — qf_math.hpp defines its own separately-compiled copies
// (QD by-reference form, inline.h:35-99), so re-exercising them against the
// shipped QF symbols guards against a QF-side copy drifting from FF's. We ADD
// qf_two_sqr (the squaring EFT, no FF-EFT-test analogue) and the renorm family.
//
// WHY GROUND TRUTH IS PLAIN FP64 FOR twoSum/twoProd (PROVABLE, NO QUADMATH)
// ------------------------------------------------------------------------
// The exact sum of two FP32 values needs at most 25 significant bits; the exact
// product needs at most 2*24 = 48 bits. IEEE double (FP64) has a 53-bit mantissa.
// Since 25 <= 53 and 48 <= 53, widening each FP32 operand to double and computing
// the sum/product THERE is EXACT — no rounding. So
//     (double)s + (double)e  ==  (double)a + (double)b
//     (double)p + (double)e  ==  (double)a * (double)b
// is a *provable* bit-equality. This half of the test needs no type wider than
// a machine word — same posture as T2.1. (It used to be contrasted with the
// wide-spread half as "needs no quadmath, no SKIP-77 fallback"; that gate is
// gone and both halves now run unconditionally.)
//
// renorm's INPUT CONTRACT (why the words must be ORDERED)
// -------------------------------------------------------
// renorm / renorm_4 are QD's renormalization step. Their quick_two_sum cascade
// (qf_quick_two_sum requires |a| >= |b|) ASSUMES the input is a magnitude-ordered
// "sloppy" expansion — exactly what add/multiply/divide/QuadFloat(double) produce
// as their unnormalized accumulator. Feeding renorm an ARBITRARY unordered set of
// FP32 words violates that precondition and is a caller error, NOT a renorm defect
// (an early version of this test made that mistake and saw spurious value/overlap
// failures — see the report). So this test generates every renorm input by
// SUCCESSIVE FP32 DECOMPOSITION of a wider value (float128 / double), which yields
// a properly-ordered, magnitude-decreasing expansion — the same shape callers feed.
//
// WHY renorm VALUE-PRESERVATION IS EXACT IN FP64 (53-BIT SOURCE REGIME)
// --------------------------------------------------------------------
// renorm rearranges its ordered words into non-overlapping form via quick_two_sum
// chains (each an EFT, individually lossless). The ONLY lossy step is dropping
// information that will not fit in the 4x24 = 96-bit output. If the input words are
// the ordered decomposition of a 53-bit DOUBLE, the exact real value is that double
// (<= 3 nonzero ordered FP32 words suffice), which fits EXACTLY in QF's 96-bit
// capacity, so renorm drops NOTHING and value-preservation is exact:
//     (double)(b0+b1+b2+b3)  ==  x   bit-for-bit  (x = the source double).
// This unconditional, provable FP64 check is the primary value-preservation gate
// (matches T2.1's "provable FP64, machine word only" philosophy). A SECOND,
// wide-spread value-preservation test (input = ordered decomposition of a
// ~113-bit float128, where renorm genuinely truncates the tail below word 4)
// checks relative agreement within the QF truncation threshold (see that
// block). Observed max rel there ~1.6e-30 (~2^-99). That second block used to
// run only under KOKKOS_EP_HAVE_QUADMATH; it runs unconditionally now.
//
// WHY -ffp-contract=off IS REQUIRED (DEKKER SPLITTER CORRECTNESS)
// --------------------------------------------------------------
// qf_two_prod / qf_two_sqr use Dekker splitting (splitter 8193.0f = 2^13+1),
// NOT an FMA-based twoProd. The error term hinges on `a1*b1 - p` being TWO
// distinct rounded operations. If the compiler contracts that into a single
// fused multiply-add, the "error" collapses to zero and the EFT silently breaks.
// This TU is compiled with -ffp-contract=off (host) / --fmad=false (CUDA) by the
// kokkos_ep_add_eft_test() helper in tests/CMakeLists.txt. (T3.5 later builds the
// contraction-on reporter mirror; T3.1 only needs the OFF posture.)
//
// Cross-reference: docs/TEST_SUITE_PLAN.md, Phase 3, "T3.1: EFT unit tests for
// QF"; QD 2.3.24 qd/include/qd/inline.h (twoSum/twoProd/two_sqr) and
// qd/include/qd/qd_inline.h (renorm 4-word:95-125, renorm 5-word:127-177);
// PORT_NOTES_QF.md (splitter-overflow branch not ported).
//
// TEST STRUCTURE
//   Test A — qf_two_sum + qf_quick_two_sum bit-exactness (FP64 oracle)
//   Test B — qf_two_prod + qf_two_sqr bit-exactness (FP64 oracle; splitter-
//            overflow and under/overflow regimes skipped)
//   Test C — renorm_4 (len 5->4) + renorm (len 4->4): Priest non-overlap
//            invariant |f_{i+1}| <= 1/2 ulp(f_i) (oracle-independent) + exact FP64
//            value-preservation (ordered 53-bit-source input) + binary128
//            wide-spread (~113-bit-source) truncation check
//   Test D — named hard cases (zero, +/-ulp, cancellation, subnormals, inf/nan)
//   Test E — device parity (run the SAME primitives in a Kokkos parallel_for)
// ============================================================================

#include "test_utils.hpp"
#include "corpus.hpp"
#include <qf_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace kokkos_ep;
namespace qf = Kokkos::Experimental;

// ----------------------------------------------------------------------------
// Oracle comparisons for the additive/multiplicative EFTs (host). Ground truth
// is plain FP64 — provably exact (25-bit sum / 48-bit product both fit in FP64's
// 53-bit mantissa). No __float128. Calls the SHIPPED qf_math.hpp primitives.
// ----------------------------------------------------------------------------
inline bool sum_is_exact(float a, float b) {
    float e;
    float s = qf::qf_two_sum(a, b, e);                 // shipped primitive, called directly
    return (double)s + (double)e == (double)a + (double)b;
}
// quick_two_sum is only exact under its precondition |a| >= |b|. Caller must
// order the operands; check_pair does so before invoking.
inline bool quick_sum_is_exact(float a, float b) {
    float e;
    float s = qf::qf_quick_two_sum(a, b, e);           // shipped primitive, called directly
    return (double)s + (double)e == (double)a + (double)b;
}
inline bool prod_is_exact(float a, float b) {
    float e;
    float p = qf::qf_two_prod(a, b, e);                // shipped primitive, called directly
    return (double)p + (double)e == (double)a * (double)b;
}
inline bool sqr_is_exact(float a) {
    float e;
    float q = qf::qf_two_sqr(a, e);                    // shipped primitive, called directly
    return (double)q + (double)e == (double)a * (double)a;
}

// ----------------------------------------------------------------------------
// Skip predicates — the "no overflow / no underflow" domain each EFT is proven
// over. A pair outside the domain is not a failure; it is simply out of scope
// for a bit-exactness claim, so it is skipped. Ported verbatim from T2.1's
// ff_eft_test (the FP32 arithmetic is identical). See that file for the full
// derivation of split_safe_max and the underflow floor.
// ----------------------------------------------------------------------------

// twoSum domain: finite a, b whose FP32 sum does not overflow. (No underflow
// hazard — Knuth twoSum is exact on subnormals too.)
inline bool sum_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return std::isfinite(a + b);
}

// Splitter-overflow bound, DERIVED from qf_two_prod's body:
// the first FP32 op is `cona = a * split` with split = 8193.0f. That overflows to
// +inf once |a| * 8193 > FLT_MAX, i.e. |a| >= FLT_MAX / 8193 (~2^114.9998); then
// `cona - (cona - a)` becomes inf - inf = NaN and the error term is poisoned. We
// reject |x| >= this for EITHER operand regardless of the other — even x*0 splits
// x first. (This is the QD _QD_SPLIT_THRESH branch, inline.h:66-83, deliberately
// NOT ported to qf_math.hpp — see PORT_NOTES_QF.md.)
inline float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

// twoProd/twoSqr domain: finite normal-or-zero operands with the TRUE product
// comfortably inside the normal range (so both p and its error term e are
// representable without underflow). Dekker's twoProduct is exact only absent
// underflow (Dekker 1971; Muller et al., "Handbook of Floating-Point
// Arithmetic", §4.4). The underflow/overflow predicate is evaluated on the EXACT
// (double) product, not the rounded FP32 product (which would flush to 0 in the
// underflow regime and let underflowing pairs masquerade as in-domain).
// Subnormal OPERANDS are excluded — splitting a subnormal is itself lossy.
//   Threshold FLT_MIN * 2^24 = 2^-102 keeps the error term e >= ~2^-126 (normal).
inline bool prod_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float fmin = std::numeric_limits<float>::min();   // smallest normal FP32
    auto normal_or_zero = [fmin](float x) {
        return x == 0.0f || std::fabs(x) >= fmin;           // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    const float ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;  // splitter overflow
    if (a == 0.0f || b == 0.0f) return true;                       // exact product 0
    double tp  = (double)a * (double)b;                            // exact (48 <= 53)
    double mag = tp < 0.0 ? -tp : tp;
    const double hi_lim = (double)std::numeric_limits<float>::max();
    const double lo_lim = std::ldexp(1.0, -102);
    if (mag > hi_lim) return false;   // product (and a1*b1) would overflow FP32
    if (mag < lo_lim) return false;   // error term would fall into FP32 subnormals
    return true;
}

// ----------------------------------------------------------------------------
// Failure-sample printers (first few only), with input bit patterns. FP32 hex is
// 0x%08x (probe_op.cpp convention).
// ----------------------------------------------------------------------------
inline uint32_t fbits(float x) { uint32_t b; std::memcpy(&b, &x, sizeof(float)); return b; }

inline void print_fail_sum(const char* which, float a, float b) {
    float e; float s = qf::qf_two_sum(a, b, e);
    std::printf("    FAIL %s  a=%.9g (0x%08x)  b=%.9g (0x%08x)  s=%.9g e=%.9g\n",
                which, (double)a, fbits(a), (double)b, fbits(b), (double)s, (double)e);
}
inline void print_fail_prod(const char* which, float a, float b) {
    float e; float p = qf::qf_two_prod(a, b, e);
    std::printf("    FAIL %s  a=%.9g (0x%08x)  b=%.9g (0x%08x)  p=%.9g e=%.9g\n",
                which, (double)a, fbits(a), (double)b, fbits(b), (double)p, (double)e);
}

// ----------------------------------------------------------------------------
// Accumulator for one corpus/random batch.
// ----------------------------------------------------------------------------
struct EftCount { long tested = 0; long skipped = 0; long failures = 0; };

enum class Op { Sum, QuickSum, Prod, Sqr };

inline void check_pair(Op op, float a, float b, EftCount& c, int& samples_left) {
    // Sqr only reads a; QuickSum requires |a|>=|b| (order the operands here).
    if (op == Op::QuickSum && std::fabs(b) > std::fabs(a)) { float t = a; a = b; b = t; }

    bool in_domain;
    switch (op) {
        case Op::Sum:      in_domain = sum_in_domain(a, b);  break;
        case Op::QuickSum: in_domain = sum_in_domain(a, b);  break;
        case Op::Prod:     in_domain = prod_in_domain(a, b); break;
        case Op::Sqr:      in_domain = prod_in_domain(a, a); break;
    }
    if (!in_domain) { ++c.skipped; return; }
    ++c.tested;

    bool ok;
    switch (op) {
        case Op::Sum:      ok = sum_is_exact(a, b);       break;
        case Op::QuickSum: ok = quick_sum_is_exact(a, b); break;
        case Op::Prod:     ok = prod_is_exact(a, b);      break;
        case Op::Sqr:      ok = sqr_is_exact(a);          break;
    }
    if (!ok) {
        ++c.failures;
        if (samples_left > 0) {
            if (op == Op::Prod || op == Op::Sqr) print_fail_prod("twoProd/sqr", a, b);
            else                                 print_fail_sum("twoSum", a, b);
            --samples_left;
        }
    }
}

// Broad-range bound, per op (FP32's exponent range is ~6x narrower than FP64's).
// twoSum: the SUM must stay finite -> 1e30 (~2^99.6, well below FLT_MAX ~3.4e38).
// twoProd/twoSqr: the PRODUCT must stay in FP32's NORMAL range -> 1e18 (~2^59.8)
// keeps |a*b| <= 1e36 < FLT_MAX and each operand below the splitter bound. Same
// FP32-forced split T2.1 documented (a single range gives vacuous product
// coverage: at 1e30, products ~1e60 overflow and are all domain-skipped).
inline float broad_bound(Op op) { return (op == Op::Sum || op == Op::QuickSum) ? 1e30f : 1e18f; }

// Run the four standard corpora (2 random ranges, 1 |a|>>|b| range, corpus
// cross-product) for one op and return the aggregate counts.
static EftCount run_host_batches(Op op, const char* op_label) {
    EftCount total;
    int samples_left = 5;
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

    // Corpus 1: 10^6 uniform in [-R, R].
    {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "uniform[-%.0e,%.0e] n=1e6", (double)R, (double)R);
        run_uniform(tag, 1'000'000, -R, R, 12345ULL);
    }
    // Corpus 2: 10^6 uniform in [-1,1] (subnormal-adjacent magnitudes reachable).
    run_uniform("uniform[-1,1] n=1e6", 1'000'000, -1.0f, 1.0f, 23456ULL);

    // Corpus 3: 10^5 pairs with |a| >> |b|: b = a * 2^-k, k in [1,20]. (k>20 makes
    // b subnormal for many a on FP32's 24-bit mantissa -> no useful signal.)
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
    {
        corpus::CorpusFlags flags;         // include_inf/zero/subnormals = true, nan = false
        std::vector<float> xs = corpus::unary<float>(flags);
        const size_t N = xs.size();
        const size_t kMaxPairs = 250'000;
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

// ============================================================================
// renorm test (Test C) — QF-UNIQUE. Priest non-overlap invariant on the output
// plus value-preservation. No FF/DD-EFT-test analogue; the oracle strategy here
// is T3.1-original.
// ============================================================================

// Mathematical 1/2 ulp of a normal float, computed from its binade exponent
// (NOT the bit-form fl(hi+lo)==hi, which has round-to-even false positives at
// exact ties — the plan text specifies the mathematical form |f_{i+1}| <= 1/2
// ulp(f_i), and that is what we use). frexp writes x = m * 2^e with m in
// [0.5, 1); for FP32's 24-bit significand ulp(x) = 2^(e-24), so 1/2 ulp = 2^(e-25).
inline double half_ulp(float x) {
    if (x == 0.0f) return 0.0;
    int e;
    std::frexp((double)x, &e);
    return std::ldexp(1.0, e - 25);
}

// UNDERFLOW-TAIL gate: the 1/2 ulp bound is only WELL-POSED when 1/2 ulp(f_i) is
// itself a normal float, i.e. |f_i| >= 2^-102; below that, subnormal quantization
// makes the comparison ill-posed (identical reasoning to T2.2's
// ff_invariant_test kUnderflowTail, gated a hair higher at 2^-100 for margin).
static constexpr float kUnderflowTail = 0x1p-100f;
inline bool pair_checkable(float hi) {
    if (!std::isfinite(hi)) return false;
    if (hi == 0.0f) return false;                                  // trailing zero
    if (std::fabs(hi) < std::numeric_limits<float>::min()) return false;  // subnormal
    if (std::fabs(hi) < kUnderflowTail) return false;              // underflow tail
    return true;
}

// Check the Priest length-4 non-overlap invariant |f_{i+1}| <= 1/2 ulp(f_i) on
// (b0,b1,b2,b3). Returns false on the first genuine overlap; increments *skips
// for pairs whose leading word is out of the well-posed domain. Also verifies
// renorm's packing guarantee: once a word is zero, all lower words must be zero
// too (no "hole" in the expansion).
inline bool nonoverlap_holds(float b0, float b1, float b2, float b3, int* skips) {
    const float b[4] = {b0, b1, b2, b3};
    // packing: no nonzero word may follow a zero word.
    bool seen_zero = false;
    for (int i = 0; i < 4; ++i) {
        if (b[i] == 0.0f) seen_zero = true;
        else if (seen_zero) return false;   // nonzero after zero -> not packed
    }
    for (int i = 0; i < 3; ++i) {
        if (b[i] == 0.0f) break;            // trailing zeros: invariant trivially holds
        if (!pair_checkable(b[i])) { if (skips) ++*skips; continue; }
        // Priest: |f_{i+1}| <= 1/2 ulp(f_i). A strict excess is a genuine overlap.
        if (std::fabs((double)b[i + 1]) > half_ulp(b[i])) return false;
    }
    return true;
}

// Draw a properly-ORDERED unnormalized length-5 expansion by successive FP32
// decomposition of a random 53-bit double (the same construction add/multiply/
// QuadFloat(double) produce — magnitude-decreasing words). renorm's quick_two_sum
// cascade ASSUMES this ordering; feeding arbitrary unordered words violates its
// precondition (that is a caller contract, not a renorm defect). A 53-bit value
// decomposes into <= 3 nonzero ordered FP32 words, so:
//   * the exact real sum of the words == the original double EXACTLY, and
//   * computing that sum in FP64 is itself exact (each partial sum is
//     representable), giving a PROVABLE FP64 value-preservation oracle.
// Returns the five words AND the exact double value they represent.
inline double draw_ordered_double(std::mt19937_64& g, float out[5]) {
    std::uniform_int_distribution<int>     de(-40, 40);      // exponent (sum stays finite)
    std::uniform_real_distribution<double> dm(-1.0, 1.0);
    double x = dm(g) * std::ldexp(1.0, de(g));
    double r = x;
    for (int k = 0; k < 5; ++k) { out[k] = (float)r; r -= (double)out[k]; }
    return x;   // == (double)(out[0]+..+out[4]) exactly (ordered, <=3 nonzero words)
}

struct RenormResult { long tested = 0; long skips = 0; long overlap_fail = 0; long value_fail = 0; };

// renorm_4 (length-5 -> length-4). Non-overlap + exact FP64 value-preservation.
// Input is an ORDERED 5-word decomposition of a 53-bit double (renorm's
// precondition); the exact real value is that double, so value-preservation is a
// provable FP64 bit-equality (double)(out sum) == x.
static RenormResult test_renorm_4_bounded(int n, uint64_t seed) {
    RenormResult R;
    std::mt19937_64 g(seed);
    int samples_left = 5;
    for (int i = 0; i < n; ++i) {
        float c[5];
        double x = draw_ordered_double(g, c);
        float b0 = c[0], b1 = c[1], b2 = c[2], b3 = c[3], b4 = c[4];
        qf::renorm_4(b0, b1, b2, b3, b4);              // shipped primitive, called directly
        ++R.tested;
        int local_skips = 0;
        if (!nonoverlap_holds(b0, b1, b2, b3, &local_skips)) {
            ++R.overlap_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm_4 non-overlap: in=[%.9g %.9g %.9g %.9g %.9g] "
                            "out=[%.9g %.9g %.9g %.9g]\n",
                            (double)c[0], (double)c[1], (double)c[2], (double)c[3], (double)c[4],
                            (double)b0, (double)b1, (double)b2, (double)b3);
                --samples_left;
            }
        }
        R.skips += local_skips;
        double out_sum = (double)b0 + b1 + b2 + b3;
        if (out_sum != x) {
            ++R.value_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm_4 value: x=%.17g out_sum=%.17g "
                            "in=[%.9g %.9g %.9g %.9g %.9g]\n",
                            x, out_sum,
                            (double)c[0], (double)c[1], (double)c[2], (double)c[3], (double)c[4]);
                --samples_left;
            }
        }
    }
    return R;
}

// renorm (length-4 -> length-4). Same two properties on an ordered 4-word
// decomposition of a 53-bit double.
static RenormResult test_renorm_bounded(int n, uint64_t seed) {
    RenormResult R;
    std::mt19937_64 g(seed);
    int samples_left = 5;
    for (int i = 0; i < n; ++i) {
        // Decompose x into 4 ordered FP32 words directly (renorm is the 4-word variant).
        std::uniform_int_distribution<int>     de(-40, 40);
        std::uniform_real_distribution<double> dm(-1.0, 1.0);
        double x = dm(g) * std::ldexp(1.0, de(g));
        double r = x; float c[4];
        for (int k = 0; k < 4; ++k) { c[k] = (float)r; r -= (double)c[k]; }
        float b0 = c[0], b1 = c[1], b2 = c[2], b3 = c[3];
        qf::renorm(b0, b1, b2, b3);                    // shipped primitive, called directly
        ++R.tested;
        int local_skips = 0;
        if (!nonoverlap_holds(b0, b1, b2, b3, &local_skips)) {
            ++R.overlap_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm non-overlap: in=[%.9g %.9g %.9g %.9g] "
                            "out=[%.9g %.9g %.9g %.9g]\n",
                            (double)c[0], (double)c[1], (double)c[2], (double)c[3],
                            (double)b0, (double)b1, (double)b2, (double)b3);
                --samples_left;
            }
        }
        R.skips += local_skips;
        double out_sum = (double)b0 + b1 + b2 + b3;
        if (out_sum != x) {
            ++R.value_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm value: x=%.17g out_sum=%.17g\n", x, out_sum);
                --samples_left;
            }
        }
    }
    return R;
}

// Wide-spread renorm_4: the input is an ORDERED 5-word decomposition of a random
// ~113-bit __float128 (not a 53-bit double), so the words span the full ~96-bit QF
// range and renorm genuinely truncates the tail below word 4. Value-preservation
// is then no longer exact; the residual must sit within QF's truncation threshold.
//   Bound: relative error <= 2^-88.
//   SOURCE OF BOUND: QF carries 4x24 = 96 significand bits, u = 2^-96 (qf_math
//   .hpp:11). renorm truncating a length-5 -> length-4 expansion loses at most a
//   few ulps at the 4th word, i.e. rel err ~ few * 2^-96. We gate at 2^-88 (a
//   256x margin above 2^-96) to allow the accumulated rounding of the
//   quick_two_sum cascade without masking a real defect (a broken renorm would
//   miss by O(1) relative, not 2^-88). Non-overlap still checked exactly.
//   Observed max rel ~1.6e-30 (~2^-99) — comfortably inside the bound.
static RenormResult test_renorm_4_wide(int n, uint64_t seed) {
    RenormResult R;
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<int>     de(-40, 40);
    std::uniform_real_distribution<double> dm(-1.0, 1.0);
    int samples_left = 5;
    const float128 kRelBound = q_ldexp((float128)1.0, -88);
    for (int i = 0; i < n; ++i) {
        // Build a random ~113-bit value: three FP64 mantissa chunks at 2^0/2^-40/2^-80.
        int e0 = de(g);
        float128 x = ((float128)dm(g)
                    + (float128)dm(g) * q_ldexp((float128)1.0, -40)
                    + (float128)dm(g) * q_ldexp((float128)1.0, -80))
                    * q_ldexp((float128)1.0, e0);
        // ORDERED decomposition into 5 FP32 words (renorm's precondition).
        float128 r = x; float c[5];
        for (int k = 0; k < 5; ++k) { c[k] = (float)r; r -= (float128)c[k]; }
        float128 in_sum = (float128)c[0] + (float128)c[1] + (float128)c[2]
                        + (float128)c[3] + (float128)c[4];  // == value the 5 words represent
        float b0 = c[0], b1 = c[1], b2 = c[2], b3 = c[3], b4 = c[4];
        qf::renorm_4(b0, b1, b2, b3, b4);
        ++R.tested;
        int local_skips = 0;
        if (!nonoverlap_holds(b0, b1, b2, b3, &local_skips)) ++R.overlap_fail;
        R.skips += local_skips;
        float128 out_sum = (float128)b0 + (float128)b1 + (float128)b2 + (float128)b3;
        if (in_sum != (float128)0.0) {
            float128 rel = q_abs((out_sum - in_sum) / in_sum);
            if (rel > kRelBound) {
                ++R.value_fail;
                if (samples_left > 0) {
                    std::printf("    FAIL renorm_4 wide value: rel=%.3e (bound 2^-88)\n",
                                (double)rel);
                    --samples_left;
                }
            }
        } else if (out_sum != (float128)0.0) {
            ++R.value_fail;
        }
    }
    return R;
}

// ----------------------------------------------------------------------------
// Test D — named hard cases. A named case reports PASS / SKIP / FAIL; the gate
// is on FAIL count only (SKIP is informational).
// ----------------------------------------------------------------------------
struct NamedResult { int passed = 0; int skipped = 0; int failed = 0; int total = 0; };

static NamedResult run_named_cases() {
    NamedResult R;
    auto case_sum = [&](const char* name, float a, float b) {
        ++R.total;
        if (!sum_in_domain(a, b)) { ++R.skipped;
            std::printf("    qf_two_sum  %-28s : SKIP (out of domain)\n", name); return; }
        bool ok = sum_is_exact(a, b);
        std::printf("    qf_two_sum  %-28s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else { ++R.failed; print_fail_sum("qf_two_sum", a, b); }
    };
    auto case_prod = [&](const char* name, float a, float b) {
        ++R.total;
        if (!prod_in_domain(a, b)) { ++R.skipped;
            std::printf("    qf_two_prod %-28s : SKIP (out of Dekker domain)\n", name); return; }
        bool ok = prod_is_exact(a, b);
        std::printf("    qf_two_prod %-28s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else { ++R.failed; print_fail_prod("qf_two_prod", a, b); }
    };

    // Exact cancellation.
    case_sum("a == b (=1.0f)",          1.0f,  1.0f);
    case_sum("a == -b (=1.0f,-1.0f)",   1.0f, -1.0f);
    case_sum("a == b (=pi_f)",          3.14159265f, 3.14159265f);

    // Both subnormal: qf_two_sum IS defined (Knuth twoSum, no underflow hazard);
    // the SAME pair fed to qf_two_prod is SKIPPED (splitting a subnormal is lossy).
    {
        float s  = std::numeric_limits<float>::denorm_min();
        float s3 = std::ldexp(s, 5);
        case_sum ("both subnormal", s, s3);   // tested, must PASS
        case_prod("both subnormal", s, s3);   // SKIPPED (out of domain)
    }

    // Zero partners, symmetric.
    case_sum("a=0, b=nonzero",  0.0f, 3.5f);
    case_sum("a=nonzero, b=0",  3.5f, 0.0f);
    case_prod("a=0, b=nonzero", 0.0f, 3.5f);
    case_prod("a=nonzero, b=0", 3.5f, 0.0f);

    // +0 / -0.
    case_sum("a=+0, b=-0", 0.0f, -0.0f);
    case_sum("a=-0, b=+0", -0.0f, 0.0f);

    // Bailey's hard twoSum case, FP32 form: 1.0f + 2^-24 rounds to 1.0f (2^-24 is
    // exactly half ulp(1.0f) = half of 2^-23), so the error term must be exactly 2^-24.
    {
        float a = 1.0f, b = std::ldexp(1.0f, -24);
        float e; float s = qf::qf_two_sum(a, b, e);
        bool ok = sum_is_exact(a, b) && (s == 1.0f) && (e == std::ldexp(1.0f, -24));
        ++R.total;
        std::printf("    qf_two_sum  %-28s : %s  (s=%.9g e=%.9g, want s=1 e=2^-24=%.9g)\n",
                    "Bailey 1.0f + 2^-24", ok ? "PASS" : "FAIL",
                    (double)s, (double)e, (double)std::ldexp(1.0f, -24));
        if (ok) ++R.passed; else ++R.failed;
    }

    // +/- ulp round-trip: a and a +/- ulp(a).
    {
        float a  = 1.0f;
        float ap = std::nextafter(a, 2.0f);   // a + ulp
        case_sum("a, a+ulp",  a, ap - a);      // b = ulp(a), exact
        case_sum("a, -(a)",   a, -a);          // exact cancellation to 0
    }

    // qf_two_prod / qf_two_sqr spot-checks at FP32-nearest pi/e/sqrt2.
    case_prod("pi_f * pi_f",       3.14159265f, 3.14159265f);
    case_prod("e_f * e_f",         2.71828183f, 2.71828183f);
    case_prod("sqrt2_f * sqrt2_f", 1.41421356f, 1.41421356f);
    {
        auto case_sqr = [&](const char* name, float a) {
            ++R.total;
            if (!prod_in_domain(a, a)) { ++R.skipped;
                std::printf("    qf_two_sqr  %-28s : SKIP (out of Dekker domain)\n", name); return; }
            bool ok = sqr_is_exact(a);
            std::printf("    qf_two_sqr  %-28s : %s\n", name, ok ? "PASS" : "FAIL");
            if (ok) ++R.passed; else { ++R.failed; print_fail_prod("qf_two_sqr", a, a); }
        };
        case_sqr("sqr(pi_f)",   3.14159265f);
        case_sqr("sqr(e_f)",    2.71828183f);
        case_sqr("sqr(sqrt2_f)",1.41421356f);
    }

    // renorm inf/nan propagation: qf renorm has `if (isinf(c0)) return;` (qf_math
    // .hpp:184,215) — it must NOT crash and must leave the leading word intact.
    {
        float b0 = std::numeric_limits<float>::infinity(), b1 = 1.0f, b2 = 0.0f, b3 = 0.0f, b4 = 0.0f;
        qf::renorm_4(b0, b1, b2, b3, b4);
        bool ok = std::isinf(b0);
        ++R.total;
        std::printf("    renorm_4    %-28s : %s (leading word stays inf, no crash)\n",
                    "inf leading word", ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else ++R.failed;
    }
    {
        float b0 = std::numeric_limits<float>::quiet_NaN(), b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
        qf::renorm(b0, b1, b2, b3);
        bool ok = std::isnan(b0);   // NaN propagates cleanly (may pass through the sum cascade)
        ++R.total;
        std::printf("    renorm      %-28s : %s (nan propagates, no crash)\n",
                    "nan leading word", ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else ++R.failed;
    }

    return R;
}

// ----------------------------------------------------------------------------
// Test E — device parity. Run the SAME shipped primitives inside a Kokkos
// parallel_for, copy results back, and compare bit-exactly against the host FP64
// oracle. On Serial-only Kokkos this reduces to host execution (still valid); on
// CUDA/HIP/SYCL it catches device-side FP differences (subnormal flush,
// contraction) the host pass cannot see. Inputs drawn from the splitter- and
// underflow-safe range [-1e18,1e18] so BOTH sum stays finite AND product stays in
// FP32's normal range (using twoSum's wider 1e30 would domain-skip nearly every
// product — the vacuous-coverage trap FP32's narrow exponent range sets).
// ----------------------------------------------------------------------------
static NamedResult run_device_parity() {
    NamedResult R;
    using exec_space = Kokkos::DefaultExecutionSpace;
    const int nd = 200'000;

    std::vector<float> ha(nd), hb(nd);
    // renorm_4 inputs: ORDERED 5-word decompositions of a 53-bit double, so device
    // value-preservation is checked bit-exactly against that double (see host Test C).
    std::vector<float> c0(nd), c1(nd), c2(nd), c3(nd), c4(nd);
    std::vector<double> cx(nd);   // the exact double each ordered expansion represents
    {
        std::mt19937_64 gen(99999ULL);
        std::uniform_real_distribution<float> d(-1e18f, 1e18f);
        for (int i = 0; i < nd; ++i) { ha[i] = d(gen); hb[i] = d(gen); }
        std::mt19937_64 gr(88888ULL);
        for (int i = 0; i < nd; ++i) {
            float e[5]; cx[i] = draw_ordered_double(gr, e);
            c0[i]=e[0]; c1[i]=e[1]; c2[i]=e[2]; c3[i]=e[3]; c4[i]=e[4];
        }
    }

    Kokkos::View<float*, exec_space> va("va", nd), vb("vb", nd);
    Kokkos::View<float*, exec_space> s_hi("s_hi", nd), s_lo("s_lo", nd);
    Kokkos::View<float*, exec_space> p_hi("p_hi", nd), p_lo("p_lo", nd);
    Kokkos::View<float*, exec_space> rc0("rc0", nd), rc1("rc1", nd), rc2("rc2", nd),
                                     rc3("rc3", nd), rc4("rc4", nd);
    Kokkos::View<float*, exec_space> rb0("rb0", nd), rb1("rb1", nd), rb2("rb2", nd), rb3("rb3", nd);

    auto hva = Kokkos::create_mirror_view(va);
    auto hvb = Kokkos::create_mirror_view(vb);
    auto hc0 = Kokkos::create_mirror_view(rc0);
    auto hc1 = Kokkos::create_mirror_view(rc1);
    auto hc2 = Kokkos::create_mirror_view(rc2);
    auto hc3 = Kokkos::create_mirror_view(rc3);
    auto hc4 = Kokkos::create_mirror_view(rc4);
    for (int i = 0; i < nd; ++i) {
        hva(i) = ha[i]; hvb(i) = hb[i];
        hc0(i) = c0[i]; hc1(i) = c1[i]; hc2(i) = c2[i]; hc3(i) = c3[i]; hc4(i) = c4[i];
    }
    Kokkos::deep_copy(va, hva);  Kokkos::deep_copy(vb, hvb);
    Kokkos::deep_copy(rc0, hc0); Kokkos::deep_copy(rc1, hc1); Kokkos::deep_copy(rc2, hc2);
    Kokkos::deep_copy(rc3, hc3); Kokkos::deep_copy(rc4, hc4);

    Kokkos::parallel_for("qf_eft_device", Kokkos::RangePolicy<exec_space>(0, nd),
        KOKKOS_LAMBDA(int i) {
            float es, ep;
            s_hi(i) = qf::qf_two_sum(va(i), vb(i), es);  s_lo(i) = es;
            p_hi(i) = qf::qf_two_prod(va(i), vb(i), ep); p_lo(i) = ep;
            float b0 = rc0(i), b1 = rc1(i), b2 = rc2(i), b3 = rc3(i), b4 = rc4(i);
            qf::renorm_4(b0, b1, b2, b3, b4);
            rb0(i) = b0; rb1(i) = b1; rb2(i) = b2; rb3(i) = b3;
        });
    Kokkos::fence();

    auto hshi = Kokkos::create_mirror_view(s_hi);
    auto hslo = Kokkos::create_mirror_view(s_lo);
    auto hphi = Kokkos::create_mirror_view(p_hi);
    auto hplo = Kokkos::create_mirror_view(p_lo);
    auto hb0 = Kokkos::create_mirror_view(rb0);
    auto hb1 = Kokkos::create_mirror_view(rb1);
    auto hb2 = Kokkos::create_mirror_view(rb2);
    auto hb3 = Kokkos::create_mirror_view(rb3);
    Kokkos::deep_copy(hshi, s_hi); Kokkos::deep_copy(hslo, s_lo);
    Kokkos::deep_copy(hphi, p_hi); Kokkos::deep_copy(hplo, p_lo);
    Kokkos::deep_copy(hb0, rb0); Kokkos::deep_copy(hb1, rb1);
    Kokkos::deep_copy(hb2, rb2); Kokkos::deep_copy(hb3, rb3);

    long sum_fail = 0, prod_fail = 0, ren_fail = 0, sum_skip = 0, prod_skip = 0, ren_over = 0;
    int samples_left = 5;
    for (int i = 0; i < nd; ++i) {
        float a = ha[i], b = hb[i];
        if (sum_in_domain(a, b)) {
            if ((double)hshi(i) + (double)hslo(i) != (double)a + (double)b) {
                ++sum_fail; if (samples_left > 0) { print_fail_sum("twoSum", a, b); --samples_left; }
            }
        } else ++sum_skip;
        if (prod_in_domain(a, b)) {
            if ((double)hphi(i) + (double)hplo(i) != (double)a * (double)b) {
                ++prod_fail; if (samples_left > 0) { print_fail_prod("twoProd", a, b); --samples_left; }
            }
        } else ++prod_skip;
        // renorm_4 parity: exact FP64 value-preservation (out sum == x) + non-overlap.
        double out_sum = (double)hb0(i) + hb1(i) + hb2(i) + hb3(i);
        int dummy = 0;
        bool value_ok   = (out_sum == cx[i]);
        bool overlap_ok = nonoverlap_holds(hb0(i), hb1(i), hb2(i), hb3(i), &dummy);
        if (!(value_ok && overlap_ok)) { ++ren_fail; if (!overlap_ok) ++ren_over; }
    }
    std::printf("    device qf_two_sum : %ld tested (%ld skipped), %ld failures\n",
                (long)nd - sum_skip, sum_skip, sum_fail);
    std::printf("    device qf_two_prod: %ld tested (%ld skipped), %ld failures\n",
                (long)nd - prod_skip, prod_skip, prod_fail);
    std::printf("    device renorm_4   : %ld tested, %ld failures (%ld non-overlap)\n",
                (long)nd, ren_fail, ren_over);
    R.total   = 3 * nd;
    R.skipped = (int)(sum_skip + prod_skip);
    R.failed  = (int)(sum_fail + prod_fail + ren_fail);
    R.passed  = R.total - R.skipped - R.failed;
    return R;
}

// ============================================================================
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        std::printf("=== qf_eft_test (T3.1): EFT bit-exactness for QF twoSum / twoProd / twoSqr / renorm ===\n");
        std::printf("Oracle: FP64 (exact for twoSum/twoProd/twoSqr and ordered-53-bit-source renorm value-preservation)\n");
        std::printf("Splitter: 8193.0f = 2^13 + 1 (qf_math.hpp `qf_two_prod`); primitives called directly from qf_math.hpp\n\n");

        // -- Test A: qf_two_sum + qf_quick_two_sum --------------------------
        std::printf("[Test A] qf_two_sum bit-exactness\n");
        EftCount A1 = run_host_batches(Op::Sum, "A/twoSum");
        std::printf("  qf_two_sum: total tested=%ld, skipped=%ld, failures=%ld\n", A1.tested, A1.skipped, A1.failures);
        KOKKOS_EP_ASSERT(A1.failures == 0, "qf_two_sum was not bit-exact for some finite input pair");
        std::printf("[Test A'] qf_quick_two_sum bit-exactness (operands ordered |a|>=|b|)\n");
        EftCount A2 = run_host_batches(Op::QuickSum, "A/quickSum");
        std::printf("  qf_quick_two_sum: total tested=%ld, skipped=%ld, failures=%ld\n\n", A2.tested, A2.skipped, A2.failures);
        KOKKOS_EP_ASSERT(A2.failures == 0, "qf_quick_two_sum was not bit-exact for some ordered pair");

        // -- Test B: qf_two_prod + qf_two_sqr -------------------------------
        std::printf("[Test B] qf_two_prod (Dekker twoProduct) bit-exactness\n");
        EftCount B1 = run_host_batches(Op::Prod, "B/twoProd");
        std::printf("  qf_two_prod: total tested=%ld, skipped=%ld, failures=%ld\n", B1.tested, B1.skipped, B1.failures);
        KOKKOS_EP_ASSERT(B1.failures == 0, "qf_two_prod was not bit-exact for some in-domain pair");
        std::printf("[Test B'] qf_two_sqr bit-exactness\n");
        EftCount B2 = run_host_batches(Op::Sqr, "B/twoSqr");
        std::printf("  qf_two_sqr: total tested=%ld, skipped=%ld, failures=%ld\n\n", B2.tested, B2.skipped, B2.failures);
        KOKKOS_EP_ASSERT(B2.failures == 0, "qf_two_sqr was not bit-exact for some in-domain input");

        // -- Test C: renorm_4 + renorm --------------------------------------
        std::printf("[Test C] renorm_4 (len 5->4) + renorm (len 4->4): non-overlap + value-preservation\n");
        RenormResult C1 = test_renorm_4_bounded(1'000'000, 45678ULL);
        std::printf("    renorm_4 bounded (exact FP64): tested=%ld  non-overlap-fail=%ld  value-fail=%ld  (pair-skips=%ld)\n",
                    C1.tested, C1.overlap_fail, C1.value_fail, C1.skips);
        KOKKOS_EP_ASSERT(C1.overlap_fail == 0, "renorm_4 produced an overlapping (non-Priest) length-4 result");
        KOKKOS_EP_ASSERT(C1.value_fail   == 0, "renorm_4 did not preserve value exactly on bounded-spread input");

        RenormResult C2 = test_renorm_bounded(1'000'000, 56789ULL);
        std::printf("    renorm   bounded (exact FP64): tested=%ld  non-overlap-fail=%ld  value-fail=%ld  (pair-skips=%ld)\n",
                    C2.tested, C2.overlap_fail, C2.value_fail, C2.skips);
        KOKKOS_EP_ASSERT(C2.overlap_fail == 0, "renorm produced an overlapping (non-Priest) length-4 result");
        KOKKOS_EP_ASSERT(C2.value_fail   == 0, "renorm did not preserve value exactly on bounded-spread input");

        RenormResult C3 = test_renorm_4_wide(1'000'000, 67890ULL);
        std::printf("    renorm_4 wide-spread (binary128, rel <= 2^-88): tested=%ld  non-overlap-fail=%ld  value-fail=%ld  (pair-skips=%ld)\n",
                    C3.tested, C3.overlap_fail, C3.value_fail, C3.skips);
        KOKKOS_EP_ASSERT(C3.overlap_fail == 0, "renorm_4 produced an overlapping result on wide-spread input");
        KOKKOS_EP_ASSERT(C3.value_fail   == 0, "renorm_4 exceeded the QF truncation threshold on wide-spread input");
        std::printf("\n");

        // -- Test D: named cases --------------------------------------------
        std::printf("[Test D] named hard cases\n");
        NamedResult D = run_named_cases();
        std::printf("  Test D named cases: %d passed, %d skipped, %d failed (of %d)\n\n",
                    D.passed, D.skipped, D.failed, D.total);
        KOKKOS_EP_ASSERT(D.failed == 0, "a named EFT case failed");

        // -- Test E: device parity ------------------------------------------
        std::printf("[Test E] device parity (%s)\n", Kokkos::DefaultExecutionSpace::name());
        NamedResult E = run_device_parity();
        std::printf("  Test E device parity: %d passed, %d skipped, %d failed (of %d)\n\n",
                    E.passed, E.skipped, E.failed, E.total);
        KOKKOS_EP_ASSERT(E.failed == 0, "device EFT parity mismatch vs host FP64 oracle");

        rc = ep_exit_code();
        std::printf("=== qf_eft_test: %s ===\n", rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    }
    Kokkos::finalize();
    return rc;
}
