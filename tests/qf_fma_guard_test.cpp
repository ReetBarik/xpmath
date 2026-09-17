// ============================================================================
// qf_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for QF.  Plan T3.5.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// QF's arithmetic is built on Dekker's error-free transforms. Two of them are
// TWO-ROUNDED-OPERATION sequences whose error term is meaningful only because a
// product and its subtraction are DISTINCT rounded steps:
//
//   qf_two_prod(a,b)  ->  p = fl(a*b),  e = ((a1*b1 - p) + a1*b2 + a2*b1) + a2*b2
//   qf_two_sqr (a)    ->  q = fl(a*a),  e = ((hi*hi  - q) + 2*hi*lo)      + lo*lo
//                                             ^^^^^^^^^^^ the contraction hazard
//   (in qf_two_prod / qf_two_sqr; Veltkamp splitter 8193.0f = 2^13 + 1.)
//
// If the compiler CONTRACTS `a1*b1 - p` (resp. `hi*hi - q`) into a single fused
// multiply-add, the rounding Dekker's algebra depends on never happens and the
// "error" term is silently wrong — a "correct" test binary that never exercises
// the shipped algorithm. T3.1's qf_eft_test builds with -ffp-contract=off /
// --fmad=false (via kokkos_ep_add_eft_test) so its OWN results are trustworthy;
// that is a *defensive* posture protecting one test. T3.5 is the *positive*
// test of that posture: build the very same shipped Dekker primitives under BOTH
// contraction settings and cross-check against a contraction-immune FP64 oracle.
//
//   * contraction-OFF build  -> the error terms must be EXACT. This is a stronger
//                               restatement of what T3.1 already asserts, and it
//                               FAIL-GATES (KOKKOS_EP_ASSERT).
//   * contraction-ON  build  -> the compiler is ALLOWED to contract. Each input is
//                               classified (see the three-way scheme below); the
//                               verdict is REPORTED, not gated on contraction —
//                               the guard's job is to make the compiler's behavior
//                               VISIBLE, not to forbid it. The ONE genuinely-bad
//                               outcome (a nonzero-but-wrong error term) is the
//                               only thing that fails the ON target.
//
// This is the QF analogue of tests/ff_fma_guard_test.cpp (T2.5, the FP32 twoProduct
// guard) and tests/dd_fma_guard_test.cpp (T1.5). Every design decision T2.5 locked
// in carries over — single-source/two-targets, a contraction-immune FP64 oracle
// (no __float128), a twoSum CONTROL, host + device passes, OFF gates / ON reports
// with a committed baseline.
//
// TWO DELIBERATE DIVERGENCES FROM T2.5 (both reported in the DONE block)
// ---------------------------------------------------------------------
// (1) NO MIRROR-AND-COMMENT. T2.5 had to *duplicate* FF's Dekker twoProduct into
//     the test file because ff_math.hpp embeds it inside multiply() — there is no
//     standalone primitive to call. qf_math.hpp is different: it EXPOSES the
//     shipped primitives as free functions (qf_two_prod / qf_two_sqr / qf_two_sum,
//     free functions). So this test calls the ACTUAL shipped code under the ON
//     flags — which is strictly stronger for a contraction guard: it characterizes
//     whether GCC contracts *qf_math.hpp's own source*, not a copy that could
//     drift. This mirrors T3.1's same divergence from T2.1. (Rule 4 is trivially
//     respected — we only #include, never edit.)
// (2) qf_two_sqr IS ALSO GUARDED. T2.5 was twoProduct-only (FF exposes no squaring
//     EFT). QF ships qf_two_sqr, a SECOND Dekker sequence with a `hi*hi - q`
//     contraction hazard, so T3.5 guards both — matching T3.1's op surface, which
//     added qf_two_sqr over T2.1.
//
// THREE-WAY CLASSIFICATION (the ON reporter's refinement over T2.5's binary F)
// ---------------------------------------------------------------------------
// T2.5 counted a single number F = "error-term mismatches"; under contraction ON
// it exits 0 regardless, because on GCC 13.3.0 F stays 0 (see that DONE block).
// QF refines this into three mutually-exclusive buckets so a genuinely-broken
// output is distinguishable from harmless contraction. For each in-domain input,
// with got = (shipped hi, shipped lo), exact = the contraction-immune FP64 product,
// and e_ref = the unique exact residual float(exact - got.hi):
//
//   id_ok := ((double)got.hi + (double)got.lo == exact)   // the Dekker EFT identity
//
//   * e_ref == 0 (true error is legitimately zero, e.g. an exact product):
//       -> TRIVIAL. Uninformative about contraction (both a correct and a
//          contracted implementation yield lo == 0). Reported separately, not gated.
//   * e_ref != 0 and  id_ok                 -> ERR_NONZERO_CORRECT
//       (the compiler either did not contract, or contracted harmlessly — Veltkamp
//        splitting makes each partial product exactly representable, so a fused
//        `partial +/- accumulator` introduces no rounding difference; see T1.5's
//        DONE block for the -mfma analysis.)
//   * e_ref != 0 and !id_ok and got.lo == 0 -> ERR_ZERO
//       (the error term COLLAPSED to zero — the classic contraction signature.)
//   * e_ref != 0 and !id_ok and got.lo != 0 -> ERR_NONZERO_WRONG
//       (the error term is nonzero yet violates the Dekker identity — the only
//        genuinely-broken outcome; would indicate a real miscompilation and open a
//        B-task candidate. Expected to NEVER fire on GCC 13.3.0.)
//
//   OFF gate: ERR_ZERO == 0 && ERR_NONZERO_WRONG == 0   (F := their sum).
//   ON  PASS: ERR_NONZERO_WRONG == 0                    (any ZERO/CORRECT mix is
//             acceptable — ERR_ZERO under ON is INFORMATIVE, not a fault of
//             qf_math.hpp). This is T2.5's ratified reporter policy, refined to the
//             three-way scheme: T2.5 could not tell "contracted-to-zero" from
//             "contracted-to-wrong", so it exited 0 on both; the ERR_NONZERO_WRONG
//             bucket lets T3.5 fail ONLY on the genuinely-broken case. On this
//             toolchain ERR_NONZERO_WRONG == 0, so the ON target passes either way.
//
// A SINGLE SOURCE, TWO TARGETS
// ----------------------------
// This one file is compiled TWICE by tests/CMakeLists.txt into two executables:
//   qf_fma_guard_test              (kokkos_ep_add_eft_test             -> OFF)
//   qf_fma_guard_test_contract_on  (kokkos_ep_add_eft_test_contract_on -> ON)
// Compiling the IDENTICAL bytes over the IDENTICAL inputs under opposite compile
// flags makes "identical test" a guarantee of the build system, not a claim a
// reviewer must verify across two files. The only per-variant knobs are compile
// DEFINITIONS the CMake helpers set: KOKKOS_EP_CONTRACTION_MODE (0 = OFF/gate,
// 1 = ON/report) selects the gate-vs-report behavior, and KOKKOS_EP_BASELINE_PATH
// (ON variant only) points at the recorded baseline count.
//
// WHY GROUND TRUTH IS PLAIN FP64 (PROVABLE, NO QUADMATH)
// -----------------------------------------------------
// Inherited from qf_eft_test (T3.1) / ff_fma_guard_test (T2.5). The exact product
// of two FP32 values needs at most 2*24 = 48 bits; the exact square at most 48.
// IEEE double (FP64) has a 53-bit mantissa. Since 48 <= 53, widening each FP32
// operand to double and multiplying THERE is EXACT — no rounding. So
//     p_ref = (float)((double)a * (double)b)               // == fl(a*b)
//     e_ref = (float)((double)a * (double)b - (double)p_ref)  // exact residual
// is a *provable* decomposition, not an approximate check. This oracle needs no
// type wider than a machine word, so it runs on any target rather than x86_64
// only. It is also contraction-immune: `(double)a *
// (double)b` is exactly representable in FP64 (48 <= 53), so the residual
// `exact - p_ref` is the same value whether computed as two ops or one fused FMA.
//
// SCOPE (docs/TEST_SUITE_PLAN.md T3.5, layer 5): the two shipped Dekker sequences
// qf_two_prod and qf_two_sqr ONLY — the QF primitives where contraction is a
// documented hazard. qf_two_sum is built from adds/subtracts with no mul-then-+/-
// adjacency, so it is contraction-immune; it is included below purely as a labeled
// CONTROL (it must stay exact under both postures). qf_math.hpp is NOT modified
// (rule 4). Higher-level ops (sqrt/exp/sin/...) and complex ops are out of scope.
// FMA-contraction posture is a COMPILER characterization, not an input-conditioning
// one, so it is NOT registered in PORT_NOTES_QF §5 (see the T3.5 scope-out).
//
// C4 CHUNK D: MIGRATED WHOLE, NOT SPLIT — AND THAT IS A DELIBERATE DEVIATION
// --------------------------------------------------------------------------
// CORE_PLAN section C4 lists this file among eight MIXED translation units, to be
// split into a host half carrying the oracle and a device half carrying the
// launch. That list was measured BEFORE C2 split tests/test_utils.hpp, and the
// classification no longer holds here for exactly the reason chunk C recorded in
// tests/ff_fma_guard_test.cpp: this TU's oracle is an exact FP64 product, not a
// binary128 one (see "WHY GROUND TRUTH IS PLAIN FP64" above, which predates C4
// and gives the algebra), and after C2 it includes test_utils_device.hpp rather
// than the host header. It is not mixed. MEASURED on this tree before it joined
// the purity gate: the only mentions of the 128-bit type's name in this file were
// prose, which the preprocessor strips — plus ONE STRING LITERAL, which it does
// not, and which was reworded rather than left to fail the gate.
//
// Splitting a TU that is already pure would duplicate ~260 lines of input
// construction, classification and oracle across two files, turn two targets into
// four, and contradict the "A SINGLE SOURCE, TWO TARGETS" rationale above — all to
// separate a host oracle from a device launch that can legitimately share a
// compilation. Chunk C reached the same conclusion for the FF twin, and a split
// suite with two conventions for the same situation is worse than one.
//
// What DID change here: the Kokkos View/parallel_for/initialize surface became
// xpt::buffer/parallel_for_n, the launch is checked for a vendor error code,
// <qf_math.hpp> became <xp/qf_math.hpp>, and rc became reachable from
// ep_exit_code() under BOTH postures (see the rc note in main). The test body,
// the inputs, the oracle, both postures and the committed baseline are untouched.
// ============================================================================

// C4 chunk D: the launch goes through tests/device_harness.hpp (C3) rather than
// Kokkos. This TU names only the xp core, so it links no Kokkos at all and is
// registered in the Kokkos-free set.
#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include "corpus.hpp"
#include <xp/qf_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace kokkos_ep;

// qf:: alias over the standalone core. This used to read
// `namespace qf = Kokkos::Experimental;`; third_party/include/qf_math.hpp is a
// true alias of the same type, so the C4 migration changed the spelling and not
// the code under test — these are still the SHIPPED primitives.
namespace qf = xp;

// Which contraction posture were we compiled under? Set by the CMake helpers
// (kokkos_ep_add_eft_test -> 0, kokkos_ep_add_eft_test_contract_on -> 1). Default
// to OFF/gate if somehow unset so a flagless build fails loud rather than silently
// skipping the gate.
#ifndef KOKKOS_EP_CONTRACTION_MODE
#  define KOKKOS_EP_CONTRACTION_MODE 0
#endif

#if KOKKOS_EP_CONTRACTION_MODE == 0
static const char* kPostureName = "OFF (-ffp-contract=off / --fmad=false)";
#else
static const char* kPostureName = "ON  (-ffp-contract=fast / --fmad=true)";
#endif

// ----------------------------------------------------------------------------
// Domain predicates — the "no overflow / no underflow" region each Dekker EFT is
// proven over. A pair outside the domain is SKIPPED, not counted (EFT skip-not-fail
// discipline). Ported verbatim from T3.1's qf_eft_test (the FP32 arithmetic is
// identical); see that file for the full derivation of split_safe_max and the
// underflow floor.
// ----------------------------------------------------------------------------

// Splitter-overflow bound, DERIVED from qf_two_prod's body:
// the first FP32 op is `cona = a * split` with split = 8193.0f. That overflows to
// +inf once |a| * 8193 > FLT_MAX, i.e. |a| >= FLT_MAX / 8193 (~2^114.9998); then
// `cona - (cona - a)` becomes inf - inf = NaN and the error term is poisoned. (This
// is the QD _QD_SPLIT_THRESH branch, inline.h:66-83, deliberately NOT ported to
// qf_math.hpp — see PORT_NOTES_QF.md.)
inline float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

// twoProd/twoSqr domain: finite normal-or-zero operands with the TRUE product
// comfortably inside the normal range (so both p and its error term e are
// representable without underflow). Dekker's twoProduct is exact only absent
// underflow (Dekker 1971; Muller et al., "Handbook of Floating-Point Arithmetic",
// §4.4). The predicate is evaluated on the EXACT (double) product, not the rounded
// FP32 product (which would flush to 0 in the underflow regime and let underflowing
// pairs masquerade as in-domain). Subnormal OPERANDS are excluded — splitting a
// subnormal is itself lossy. Threshold FLT_MIN * 2^24 = 2^-102 keeps e >= ~2^-126.
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

// twoSum domain: finite a, b whose FP32 sum does not overflow. (No underflow
// hazard — Knuth twoSum is exact on subnormals too.)
inline bool sum_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return std::isfinite(a + b);
}

// twoSum oracle-faithfulness predicate (CONTROL pass only). FP32-SPECIFIC — same
// guard T2.5's ff_fma_guard_test introduced (no DD counterpart), reproduced here
// for the same reason. A product-domain pair like (a = FLT_MIN = 2^-126, b = 2^24)
// has an in-range PRODUCT (2^-102) yet an exact SUM spanning 2^24 down to 2^-126 —
// far beyond the FP64 oracle's 53-bit mantissa. The FP32 twoSum is CORRECT there
// (hi = 2^24, lo = 2^-126, non-overlapping); it is the FP64 *decomposition oracle*
// that collapses the tiny tail, so the exact FP32 lo would look like a false
// "mismatch". A pair is in the control's domain only when the exact FP64 sum is
// faithful, i.e. the FP64 twoSum error term is zero (the double sum carries no
// rounding). This excludes ONLY wide-exponent sums, never a pair where FP32 twoSum
// is actually wrong. (qf_two_prod / qf_two_sqr need no such guard — their exact
// product is always <= 48 bits, so the FP64 oracle is unconditionally faithful.)
inline bool sum_oracle_faithful(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    double da = (double)a, db = (double)b;
    double s  = da + db;
    if (!std::isfinite(s)) return false;             // sum must not overflow (twoSum domain)
    double e   = s - da;                             // FP64 twoSum error term;
    double err = (db - e) + (da - (s - e));          // zero <=> the double sum is exact.
    return err == 0.0;
}

inline uint32_t fbits(float x) { uint32_t b; std::memcpy(&b, &x, sizeof(float)); return b; }

// ----------------------------------------------------------------------------
// The three-way classifier (see header). Accumulated per op, per pass.
// ----------------------------------------------------------------------------
struct GuardStat {
    long tested  = 0;   // informative inputs (e_ref != 0): err_zero+err_correct+err_wrong
    long trivial = 0;   // e_ref == 0 (true error legitimately zero — uninformative)
    long err_zero    = 0;   // error term collapsed to 0 (contraction signature)
    long err_correct = 0;   // nonzero error term, Dekker identity holds
    long err_wrong   = 0;   // nonzero error term, identity violated (genuine bug)

    void add(const GuardStat& o) {
        tested += o.tested; trivial += o.trivial;
        err_zero += o.err_zero; err_correct += o.err_correct; err_wrong += o.err_wrong;
    }
    // F: the OFF-gate quantity (== T2.5's "mismatch count") and the baseline number.
    long F() const { return err_zero + err_wrong; }
};

// Classify one (got_hi, got_lo) against the contraction-immune exact product.
inline void classify(GuardStat& c, float got_hi, float got_lo, double exact,
                     const char* which, const char* where, float a, float b,
                     int& samples_left) {
    const float  e_ref = (float)(exact - (double)got_hi);   // unique exact residual
    const bool   id_ok = ((double)got_hi + (double)got_lo == exact);
    if (e_ref == 0.0f) {                                     // true error is zero
        ++c.trivial;
        if (!id_ok) {                                        // exact but lo spurious -> bug
            ++c.err_wrong;
            if (samples_left > 0) {
                std::printf("    WRONG %s[%s] (e_ref==0 but id violated) a=%.9g (0x%08x) "
                            "b=%.9g (0x%08x) hi=%.9g lo=%.9g\n",
                            which, where, (double)a, fbits(a), (double)b, fbits(b),
                            (double)got_hi, (double)got_lo);
                --samples_left;
            }
        }
        return;
    }
    ++c.tested;
    if (id_ok)                 { ++c.err_correct; return; }  // correct nonzero error
    if (got_lo == 0.0f)        { ++c.err_zero;    return; }  // collapsed -> contracted
    ++c.err_wrong;                                           // nonzero but wrong -> bug
    if (samples_left > 0) {
        std::printf("    WRONG %s[%s] a=%.9g (0x%08x) b=%.9g (0x%08x) hi=%.9g lo=%.9g "
                    "e_ref=%.9g\n",
                    which, where, (double)a, fbits(a), (double)b, fbits(b),
                    (double)got_hi, (double)got_lo, (double)e_ref);
        --samples_left;
    }
}

// ----------------------------------------------------------------------------
// Input corpus. Reuses T3.1's four-corpus shape (broad random + narrow random +
// |a|>>|b| + full corner-case cross-product), sized so each op sees ~2.5x10^5
// in-domain pairs (per the T3.5 task's "~2x10^5 per op is fine"). Ranges are the
// FP32-forced ones T3.1 documents: broad [-1e18,1e18] keeps every PRODUCT splitter-
// and underflow-safe (|a*b| <= 1e36 < FLT_MAX), narrow [-1,1] reaches subnormal-
// adjacent magnitudes. Deterministic seeds. Every pair is prod_in_domain-filtered
// at build time so the product passes never re-check the domain.
// ----------------------------------------------------------------------------
static std::vector<std::pair<float,float>> build_prod_inputs() {
    std::vector<std::pair<float,float>> pairs;
    pairs.reserve(260'000);

    // Broad: 10^5 uniform in [-1e18, 1e18].
    {
        std::mt19937_64 gen(0x3F5A0C117ULL);
        std::uniform_real_distribution<float> d(-1e18f, 1e18f);
        long made = 0;
        while (made < 100'000) {
            float a = d(gen), b = d(gen);
            if (prod_in_domain(a, b)) { pairs.emplace_back(a, b); ++made; }
        }
    }
    // Narrow: 10^5 uniform in [-1, 1] (subnormal-adjacent magnitudes reachable).
    {
        std::mt19937_64 gen(0x2C1D9E337ULL);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        long made = 0;
        while (made < 100'000) {
            float a = d(gen), b = d(gen);
            if (prod_in_domain(a, b)) { pairs.emplace_back(a, b); ++made; }
        }
    }
    // |a| >> |b|: b = a * 2^-k, k in [1,20]. (k>20 makes b subnormal for many a on
    // FP32's 24-bit mantissa -> no useful signal.) 5x10^4 in-domain pairs.
    {
        std::mt19937_64 gen(0x11ABCDEF3ULL);
        std::uniform_real_distribution<float> da(-1e18f, 1e18f);
        std::uniform_int_distribution<int>    dk(1, 20);
        long made = 0, guard = 0;
        while (made < 50'000 && guard < 5'000'000) {
            ++guard;
            float a = da(gen);
            int   k = dk(gen);
            float b = a * std::ldexp(1.0f, -k);
            if (prod_in_domain(a, b)) { pairs.emplace_back(a, b); ++made; }
        }
    }
    // Full corner-case corpus cross-product (i < j), in-domain members only.
    {
        corpus::CorpusFlags flags;   // inf/zero/subnormals on, nan off
        std::vector<float> xs = corpus::unary<float>(flags);
        for (size_t i = 0; i < xs.size(); ++i)
            for (size_t j = i + 1; j < xs.size(); ++j)
                if (prod_in_domain(xs[i], xs[j])) pairs.emplace_back(xs[i], xs[j]);
    }
    return pairs;
}

// twoSqr inputs: in-domain single values (prod_in_domain(a,a)). Drawn from a broad
// and a narrow random range plus the corpus, matching the product corpora.
static std::vector<float> build_sqr_inputs() {
    std::vector<float> vals;
    vals.reserve(210'000);
    {
        std::mt19937_64 gen(0x77E1A2B33ULL);
        std::uniform_real_distribution<float> d(-1e18f, 1e18f);
        long made = 0;
        while (made < 100'000) { float a = d(gen); if (prod_in_domain(a, a)) { vals.push_back(a); ++made; } }
    }
    {
        std::mt19937_64 gen(0x55D0C4A99ULL);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        long made = 0;
        while (made < 100'000) { float a = d(gen); if (prod_in_domain(a, a)) { vals.push_back(a); ++made; } }
    }
    {
        corpus::CorpusFlags flags;
        std::vector<float> xs = corpus::unary<float>(flags);
        for (float x : xs) if (prod_in_domain(x, x)) vals.push_back(x);
    }
    return vals;
}

// ----------------------------------------------------------------------------
// Host passes: recompute the SHIPPED primitives directly on the host (governed by
// the CXX -ffp-contract flag), classify against the FP64 oracle.
// ----------------------------------------------------------------------------
static GuardStat host_prod_pass(const std::vector<std::pair<float,float>>& in, int& samples_left) {
    GuardStat c;
    for (const auto& pr : in) {
        float a = pr.first, b = pr.second;
        float e; float p = qf::qf_two_prod(a, b, e);        // shipped primitive, called directly
        classify(c, p, e, (double)a * (double)b, "twoProd", "host", a, b, samples_left);
    }
    return c;
}
static GuardStat host_sqr_pass(const std::vector<float>& in, int& samples_left) {
    GuardStat c;
    for (float a : in) {
        float e; float q = qf::qf_two_sqr(a, e);            // shipped primitive, called directly
        classify(c, q, e, (double)a * (double)a, "twoSqr", "host", a, a, samples_left);
    }
    return c;
}

// ----------------------------------------------------------------------------
// Device passes: recompute the SHIPPED primitives through the C3 harness. On a
// CUDA/HIP build this is a real kernel governed by the device compiler's --fmad
// flag; on a host build the harness runs the same functor in a serial loop, still
// through two genuinely separate allocations and a memcpy each way. Mirrors
// ff_fma_guard_test's device pass and qf_eft_test's Test E.
//
// Each functor is a trivially-copyable STRUCT holding raw device pointers, not a
// lambda: device_harness.hpp passes it by value into a __global__ and
// deliberately does not require nvcc --extended-lambda of its callers.
// ----------------------------------------------------------------------------
struct ProdKernel {
    const float* a;
    const float* b;
    float* p_hi;
    float* p_lo;

    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        float e; float p = qf::qf_two_prod(a[i], b[i], e);
        p_hi[i] = p; p_lo[i] = e;
    }
};
struct SqrKernel {
    const float* a;
    float* q_hi;
    float* q_lo;

    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        float e; float q = qf::qf_two_sqr(a[i], e);
        q_hi[i] = q; q_lo[i] = e;
    }
};

static GuardStat device_prod_pass(const std::vector<std::pair<float,float>>& in, int& samples_left) {
    const std::size_t n = in.size();
    xpt::buffer<float> va(n), vb(n), p_hi(n), p_lo(n);
    for (std::size_t i = 0; i < n; ++i) { va.host()[i] = in[i].first; vb.host()[i] = in[i].second; }
    va.to_device();
    vb.to_device();

    // parallel_for_n records the launch error and fences before it returns, so
    // from_device() below cannot race a live kernel.
    xpt::parallel_for_n(n, ProdKernel{va.device(), vb.device(),
                                      p_hi.device(), p_lo.device()});

    p_hi.from_device();
    p_lo.from_device();

    GuardStat c;
    for (std::size_t i = 0; i < n; ++i)
        classify(c, p_hi.host()[i], p_lo.host()[i],
                 (double)in[i].first * (double)in[i].second,
                 "twoProd", "device", in[i].first, in[i].second, samples_left);
    return c;
}
static GuardStat device_sqr_pass(const std::vector<float>& in, int& samples_left) {
    const std::size_t n = in.size();
    xpt::buffer<float> va(n), q_hi(n), q_lo(n);
    for (std::size_t i = 0; i < n; ++i) va.host()[i] = in[i];
    va.to_device();

    xpt::parallel_for_n(n, SqrKernel{va.device(), q_hi.device(), q_lo.device()});

    q_hi.from_device();
    q_lo.from_device();

    GuardStat c;
    for (std::size_t i = 0; i < n; ++i)
        classify(c, q_hi.host()[i], q_lo.host()[i], (double)in[i] * (double)in[i],
                 "twoSqr", "device", in[i], in[i], samples_left);
    return c;
}

// ----------------------------------------------------------------------------
// twoSum CONTROL (host). Contraction-immune (no mul-then-+/- adjacency), so it must
// stay bit-exact under BOTH postures. Reuses the product corpus with the FP32
// oracle-faithfulness guard (skips wide-exponent sums the FP64 oracle cannot
// witness). Reported as a control; folded into the OFF gate.
// ----------------------------------------------------------------------------
struct SumCount { long tested = 0; long skipped = 0; long mismatches = 0; };
static SumCount sum_control(const std::vector<std::pair<float,float>>& in, int& samples_left) {
    SumCount c;
    for (const auto& pr : in) {
        float a = pr.first, b = pr.second;
        if (!sum_in_domain(a, b) || !sum_oracle_faithful(a, b)) { ++c.skipped; continue; }
        float e; float s = qf::qf_two_sum(a, b, e);         // shipped primitive, called directly
        ++c.tested;
        if ((double)s + (double)e != (double)a + (double)b) {
            ++c.mismatches;
            if (samples_left > 0) {
                std::printf("    MISMATCH twoSum[host] a=%.9g (0x%08x) b=%.9g (0x%08x) "
                            "s=%.9g e=%.9g\n",
                            (double)a, fbits(a), (double)b, fbits(b), (double)s, (double)e);
                --samples_left;
            }
        }
    }
    return c;
}

// ----------------------------------------------------------------------------
// Corner cases as named asserts (T3.5 deliverable). Each reports the op, the
// classification bucket it landed in, or SKIP if out of the Dekker domain. Gated
// (OFF) / reported (ON) via the same F/err_wrong logic as the batch passes; here we
// simply require the identity to hold on the in-domain named cases under OFF.
// ----------------------------------------------------------------------------
struct NamedResult { int passed = 0; int skipped = 0; int failed = 0; int total = 0; };

static NamedResult run_named_cases() {
    NamedResult R;
    int samples_left = 8;
    auto bucket = [](const GuardStat& c) -> const char* {
        if (c.err_wrong)   return "ERR_NONZERO_WRONG";
        if (c.err_zero)    return "ERR_ZERO";
        if (c.err_correct) return "ERR_NONZERO_CORRECT";
        return "TRIVIAL (true error 0)";
    };
    auto case_prod = [&](const char* name, float a, float b) {
        ++R.total;
        if (!prod_in_domain(a, b)) { ++R.skipped;
            std::printf("    qf_two_prod %-30s : SKIP (out of Dekker domain)\n", name); return; }
        GuardStat c; float e; float p = qf::qf_two_prod(a, b, e);
        classify(c, p, e, (double)a * (double)b, "twoProd", "named", a, b, samples_left);
        bool ok = (c.err_wrong == 0);                       // ON: only WRONG fails
#if KOKKOS_EP_CONTRACTION_MODE == 0
        ok = ok && (c.err_zero == 0);                       // OFF: also gate collapse
#endif
        std::printf("    qf_two_prod %-30s : %s [%s]\n", name, ok ? "PASS" : "FAIL", bucket(c));
        if (ok) ++R.passed; else ++R.failed;
    };
    auto case_sqr = [&](const char* name, float a) {
        ++R.total;
        if (!prod_in_domain(a, a)) { ++R.skipped;
            std::printf("    qf_two_sqr  %-30s : SKIP (out of Dekker domain)\n", name); return; }
        GuardStat c; float e; float q = qf::qf_two_sqr(a, e);
        classify(c, q, e, (double)a * (double)a, "twoSqr", "named", a, a, samples_left);
        bool ok = (c.err_wrong == 0);
#if KOKKOS_EP_CONTRACTION_MODE == 0
        ok = ok && (c.err_zero == 0);
#endif
        std::printf("    qf_two_sqr  %-30s : %s [%s]\n", name, ok ? "PASS" : "FAIL", bucket(c));
        if (ok) ++R.passed; else ++R.failed;
    };

    // Zero factors — exact product 0, error term must be 0 (TRIVIAL, no collapse).
    case_prod("a=0, b=nonzero",        0.0f, 3.5f);
    case_prod("a=nonzero, b=0",        3.5f, 0.0f);
    case_sqr ("sqr(0)",                0.0f);

    // +/- ulp: product carries a genuine nonzero error term (informative).
    {
        float a  = 1.0f;
        float ap = std::nextafter(a, 2.0f);   // 1 + ulp(1) = 1 + 2^-23
        case_prod("a * (a+ulp)",       a, ap);
        case_prod("(a+ulp) * (a+ulp)", ap, ap);
        case_sqr ("sqr(a+ulp)",        ap);
    }

    // Near-cancellation of the PRODUCT error: pi_f * pi_f etc. exercise a full-width
    // residual (the classic twoProduct stress the EFT test also spot-checks).
    case_prod("pi_f * pi_f",           3.14159265f, 3.14159265f);
    case_prod("e_f * e_f",             2.71828183f, 2.71828183f);
    case_sqr ("sqr(sqrt2_f)",          1.41421356f);
    case_sqr ("sqr(pi_f)",             3.14159265f);

    // Near-overflow: operands just under the splitter bound, product still finite.
    {
        float big = split_safe_max() * 0.5f;              // ~2^113.9998, splitter-safe
        case_prod("near-splitter * tiny",  big, std::ldexp(1.0f, -60));
        case_sqr ("sqr(near-splitter)",    std::ldexp(1.0f, -6) * big);  // keep product finite
    }

    // Subnormal factor: SKIPPED (splitting a subnormal is lossy — out of domain).
    {
        float s = std::numeric_limits<float>::denorm_min();
        case_prod("subnormal * 2",     s, 2.0f);          // expect SKIP
        case_sqr ("sqr(subnormal)",    s);                // expect SKIP
    }

    return R;
}

// ----------------------------------------------------------------------------
// Baseline mechanism (ON variant only). Records the observed F (== err_zero +
// err_wrong summed over both ops, host + device) so a LATER build with a different
// compiler/ISA that changes contraction behavior is caught as drift. WARN-only
// (never fails) — mirrors ff_fma_guard_test / dd_fma_guard_test verbatim. File
// format is one integer on the first non-comment line.
// ----------------------------------------------------------------------------
#ifdef KOKKOS_EP_BASELINE_PATH
static void check_baseline(long observed) {
    const char* path = KOKKOS_EP_BASELINE_PATH;
    std::ifstream f(path);
    if (!f) {
        std::printf("  baseline: no file at %s\n", path);
        std::printf("            record this run by writing \"%ld\" as the first\n", observed);
        std::printf("            non-comment line of that file to arm drift detection.\n");
        return;
    }
    long baseline = -1;
    std::string line;
    bool got = false;
    while (std::getline(f, line)) {
        size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] == '#') continue;   // blank / comment
        try { baseline = std::stol(line.substr(p)); got = true; } catch (...) {}
        break;
    }
    if (!got) {
        std::printf("  baseline: %s present but unparseable; skipping drift check\n", path);
        return;
    }
    if (observed == baseline) {
        std::printf("  baseline: OK — observed F=%ld matches baseline\n", observed);
    } else {
        std::printf("  baseline: *** DRIFT *** observed=%ld baseline=%ld\n", observed, baseline);
        std::printf("            contraction behavior changed since the baseline was\n");
        std::printf("            recorded (compiler/ISA/flag change). This is a WARNING,\n");
        std::printf("            not a failure. If the new value is correct, update %s.\n", path);
    }
}
#endif

static void report_op(const char* op, const GuardStat& c) {
    std::printf("[%s] tested=%ld (trivial=%ld)  ERR_ZERO=%ld  ERR_NONZERO_CORRECT=%ld  "
                "ERR_NONZERO_WRONG=%ld\n",
                op, c.tested, c.trivial, c.err_zero, c.err_correct, c.err_wrong);
}

// ============================================================================
int main(int, char**) {
    int rc = 0;
    {
        std::printf("=== qf_fma_guard_test (T3.5): FMA-contraction guard for QF Dekker "
                    "qf_two_prod + qf_two_sqr ===\n");
        std::printf("contraction posture: %s\n", kPostureName);
        std::printf("execution space: %s\n", xpt::where_name());
        // Deliberately does NOT spell the 128-bit type's name here. A string
        // literal survives preprocessing, so it would fail
        // scripts/check_device_tu_purity.sh, which carries this file.
        std::printf("Oracle: FP64 exact product (contraction-immune; 48 <= 53 bits, "
                    "needs no type wider than a machine word)\n");
        std::printf("Primitives called DIRECTLY from qf_math.hpp (no mirror-and-comment; "
                    "cf. T3.1 qf_eft_test)\n\n");

        std::vector<std::pair<float,float>> prod_in = build_prod_inputs();
        std::vector<float>                  sqr_in  = build_sqr_inputs();
        std::printf("inputs: qf_two_prod %zu in-domain pairs, qf_two_sqr %zu in-domain values "
                    "(10^5 broad + 10^5 narrow + 5x10^4 |a|>>|b| + corpus)\n\n",
                    prod_in.size(), sqr_in.size());

        int samples_left = 8;

        // --- twoSum control (must be bit-exact under both postures) ----------
        SumCount S = sum_control(prod_in, samples_left);
        std::printf("[control] qf_two_sum (contraction-immune): tested=%ld skipped=%ld mismatches=%ld\n\n",
                    S.tested, S.skipped, S.mismatches);

        // --- qf_two_prod : host + device -------------------------------------
        GuardStat PH = host_prod_pass(prod_in, samples_left);
        GuardStat PD = device_prod_pass(prod_in, samples_left);
        GuardStat P;  P.add(PH); P.add(PD);
        report_op("qf_two_prod host  ", PH);
        report_op("qf_two_prod device", PD);

        // --- qf_two_sqr : host + device --------------------------------------
        GuardStat QH = host_sqr_pass(sqr_in, samples_left);
        GuardStat QD = device_sqr_pass(sqr_in, samples_left);
        GuardStat Q;  Q.add(QH); Q.add(QD);
        report_op("qf_two_sqr  host  ", QH);
        report_op("qf_two_sqr  device", QD);

        const long F = P.F() + Q.F();
        std::printf("\ncontraction posture: %s\n", kPostureName);
        std::printf("  qf_two_prod : ERR_ZERO=%ld  ERR_NONZERO_CORRECT=%ld  ERR_NONZERO_WRONG=%ld\n",
                    P.err_zero, P.err_correct, P.err_wrong);
        std::printf("  qf_two_sqr  : ERR_ZERO=%ld  ERR_NONZERO_CORRECT=%ld  ERR_NONZERO_WRONG=%ld\n",
                    Q.err_zero, Q.err_correct, Q.err_wrong);
        std::printf("  F (err_zero + err_wrong) = %ld\n", F);

        // --- named corner cases ----------------------------------------------
        std::printf("\n[named corner cases]\n");
        NamedResult N = run_named_cases();
        std::printf("  named cases: %d passed, %d skipped, %d failed (of %d)\n",
                    N.passed, N.skipped, N.failed, N.total);

#if KOKKOS_EP_CONTRACTION_MODE == 0
        // OFF variant: FAIL-GATE. Every shipped Dekker error term MUST be exact —
        // a stronger form of what T3.1 already asserts. No collapse, no wrong term,
        // control exact, no named-case failure.
        std::printf("\nmode=OFF: fail-gating on any collapsed/wrong error term.\n");
        KOKKOS_EP_ASSERT(S.mismatches == 0,
                         "qf_two_sum control not exact under contraction-off (unexpected)");
        KOKKOS_EP_ASSERT(P.err_wrong == 0 && Q.err_wrong == 0,
                         "a QF Dekker error term was nonzero-but-wrong under contraction-off");
        KOKKOS_EP_ASSERT(F == 0,
                         "a QF Dekker error term collapsed under contraction-off — "
                         "the -ffp-contract=off posture is not taking effect");
        KOKKOS_EP_ASSERT(N.failed == 0, "a named QF Dekker corner case failed under contraction-off");
        KOKKOS_EP_ASSERT(xpt::last_error() == 0,
                         "device harness reported a nonzero vendor error code");
        rc = ep_exit_code();
        std::printf("=== qf_fma_guard_test [OFF]: %s ===\n",
                    rc == 0 ? "ALL EXACT (posture holds)" : "FAILURES PRESENT");
#else
        // ON variant: REPORT. The compiler is allowed to contract; ERR_ZERO is
        // INFORMATIVE (contraction happened, harmlessly for the value — the tail was
        // legitimately droppable or the shipped algo still holds). Only a genuinely-
        // broken output (ERR_NONZERO_WRONG) fails. Matches T2.5's ratified reporter
        // policy, refined to the three-way scheme (T2.5 could not distinguish
        // collapse from a wrong nonzero term, so it exited 0 on both).
        std::printf("\nmode=ON: reporting; PASS iff ERR_NONZERO_WRONG == 0.\n");
        const long wrong = P.err_wrong + Q.err_wrong;
        if (F == 0) {
            std::printf("  result: F == 0 — this compiler did NOT contract the QF Dekker\n");
            std::printf("          sequences at -ffp-contract=fast on this ISA target.\n");
            std::printf("          The -ffp-contract=off safety posture is belt+suspenders here.\n");
        } else if (wrong == 0) {
            std::printf("  result: ERR_ZERO=%ld (contraction collapsed the error term), but\n", F);
            std::printf("          ERR_NONZERO_WRONG=0 — informative, not a fault of qf_math.hpp.\n");
            std::printf("          The -ffp-contract=off posture in the shipped build is REQUIRED.\n");
        } else {
            std::printf("  result: ERR_NONZERO_WRONG=%ld — a Dekker error term is nonzero yet\n", wrong);
            std::printf("          violates the identity. This is a GENUINE defect (compiler\n");
            std::printf("          miscompilation or algorithm bug), NOT mere contraction.\n");
        }
        if (S.mismatches != 0)
            std::printf("  WARNING: qf_two_sum control showed %ld mismatches under ON — unexpected "
                        "(twoSum has no contractible adjacency).\n", S.mismatches);
#  ifdef KOKKOS_EP_BASELINE_PATH
        check_baseline(F);
#  endif
        // Reporter: PASS unless a genuinely-broken (nonzero-wrong) term appeared.
        //
        // C4 chunk D — THE APPARATUS TERM IS NOT OPTIONAL. This line used to read
        // `rc = (wrong == 0 && N.failed == 0) ? 0 : 1;`, which never consults
        // ep_exit_code(). That was survivable while every KOKKOS_EP_ASSERT in this
        // file sat inside the OFF branch. The migration adds one to the path BOTH
        // postures take (the harness error check below), and a reporter that
        // printed ASSERT FAILED and still exited 0 would be worse than no check at
        // all — chunk B had to fix exactly that shape in tf_fma_guard_test. So the
        // assert count joins the verdict here rather than the assertion being kept
        // OFF-only: the reporter still exits 0 on the MEASUREMENT (ERR_ZERO under
        // ON is informative, not a fault), and nonzero on the APPARATUS. A launch
        // that failed did not report anything, so F and wrong would be noise.
        KOKKOS_EP_ASSERT(xpt::last_error() == 0,
                         "device harness reported a nonzero vendor error code");
        const int apparatus_rc = ep_exit_code();
        rc = (wrong == 0 && N.failed == 0 && apparatus_rc == 0) ? 0 : 1;
        std::printf("=== qf_fma_guard_test [ON]: REPORTED (F=%ld, ERR_NONZERO_WRONG=%ld, exit %d) ===\n",
                    F, wrong, rc);
#endif
    }
    return rc;
}
