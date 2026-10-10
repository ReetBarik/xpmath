// ============================================================================
// ff_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for FF.  Plan T2.5.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// FF's production `two_prod` (include/xp/ff_math.hpp, backing `multiply`,
// `multiply_scalar`, `ff_divide_core`, `divide_scalar`) computes the exact
// product error term via an FMA-based TwoProduct, not Dekker/Veltkamp
// splitting:
//
//     p = fl(a*b);  e = fma(a, b, -p);
//
// This is correct provided (a) `p` is the correctly-rounded product and
// (b) the single rounding inside `fma` is the ONLY rounding — i.e. the
// compiler is not allowed to further reassociate the already-explicit `fma`
// call, and the volatile guards inside `detail::eft_mul`/`detail::eft_fma`
// (include/xp/config.hpp) are not allowed to be optimized away. Those guards
// exist precisely because aggressive compilers (gfx90a under ROCm 7.0.2 even
// with -ffp-contract=off; g++ 13.3 under -ffp-contract=fast/-ffast-math) have
// been observed to fuse or fold adjacent rounded ops that look, to a
// real-number-algebra optimizer, like they cancel to zero. T2.1's ff_eft_test
// builds under -ffp-contract=off / --fmad=false so its OWN results are
// trustworthy; that is a *defensive* posture protecting one test. T2.5 is the
// *positive* test of that posture: build the SAME shipped `two_prod` under
// BOTH contraction settings and cross-check it against a reference built a
// structurally different way.
//
//   * contraction-OFF build  -> `two_prod` must be bit-exact against the
//                               FP64 oracle (F == 0). This is a stronger
//                               restatement of what T2.1 already asserts,
//                               and it FAIL-GATES (XPMATH_ASSERT).
//   * contraction-ON  build  -> the compiler is ALLOWED to contract/
//                               reassociate more aggressively. Two outcomes,
//                               both informative, neither a failure:
//                                 (a) F == 0  — the shipped guards held even
//                                     under the more permissive posture.
//                                 (b) F  > 0  — something upstream of the
//                                     volatile guards broke; the
//                                     -ffp-contract=off posture in
//                                     ff_math.hpp's build is REQUIRED, and
//                                     this count is the evidence.
//                               The ON variant therefore REPORTS, it does not
//                               gate (always exits 0). A *change* in F between
//                               builds is the regression signal (see baseline
//                               mechanism).
//
// This is the FF analogue of tests/dd_fma_guard_test.cpp (T1.5). Every design
// decision T1.5 locked in carries over verbatim — single-source/two-targets, a
// contraction-immune oracle, a twoSum CONTROL, host+device passes, OFF gates /
// ON reports with a committed baseline. The ONE material difference is the oracle
// (see next block), inherited from ff_eft_test's divergence from dd_eft_test.
//
// WHY THIS NO LONGER TESTS A LOCAL DEKKER MIRROR
// -----------------------------------------------
// Before the FMA-based TwoProduct migration, this file duplicated
// ff_math.hpp's Dekker splitting sequence locally (mirror-and-comment,
// matching T2.1's ff_eft_test.cpp) because the thing under contraction
// scrutiny — `a1*b1 - c11` (and the a1*b2/a2*b1/a2*b2 terms) — was buried
// inside `multiply`'s longer renormalization sequence and needed isolating.
// `two_prod` is now two lines with no hidden algebra to duplicate, so this
// file calls the PRODUCTION function directly (`xp::two_prod`, forward-
// declared and defined in ff_math.hpp, already included below) rather than
// re-deriving it, on BOTH the host pass and the device kernel. That is
// strictly stronger: it exercises the actual translation unit, actual
// inlining, and actual NOINLINE/volatile annotations the shipped binary
// uses, not a hand-copied stand-in that could silently drift from them.
//
// ff_math.hpp's historical Dekker twoProduct implementation is NOT deleted —
// it still exists as `two_prod_dekker` in tests/ff_eft_test.cpp (T2.1), kept
// there as an independent, second exact algorithm for the same mathematical
// transform, cross-checked against the same FP64 oracle. It has no further
// role here once it stopped being what `multiply`/`two_prod` ship.
//
// DOMAIN WIDENING vs. the pre-migration version of this file
// ------------------------------------------------------------
// The old Dekker-based domain predicate excluded |operand| >= split_safe_max
// (~4.1533e34 = FLT_MAX/8193, ~2^114.9998) because SPLITTING an operand that
// large overflows the splitter itself — a hazard specific to Veltkamp
// splitting that does not exist for FMA-based `two_prod` (it never splits).
// That exclusion is removed below. What remains — rejecting subnormal
// operands and bounding the TRUE product away from FP32's overflow/underflow
// edges — is NOT splitter-specific: it is the IEEE-754 "exact FMA residual"
// precondition (Boldo & Muller), the same requirement an FMA-based
// TwoProduct has that a Dekker one does. It stays. Removing the operand-
// level exclusion admits corner-case corpus entries between ~2^115 and
// FLT_MAX (e.g. `finite_specials<float>()`'s maxT, `powers_of_two<float>()`'s
// upper end) into the shared input list for the first time, which is exactly
// the band that exercises `sum_oracle_faithful` below — see its comment for
// why that guard, already present in this file since T2.5, needs no further
// extension to cover it (it is a general exactness check, not a threshold
// tied to the old splitter bound).
//
// A SINGLE SOURCE, TWO TARGETS
// ----------------------------
// This one file is compiled TWICE by tests/CMakeLists.txt into two executables:
//   ff_fma_guard_test              (xpm_add_device_eft_test             -> OFF)
//   ff_fma_guard_test_contract_on  (xpm_add_device_eft_test_contract_on -> ON)
// Single-source (not two-sources-with-shared-header) is deliberate: the whole
// point is to run the IDENTICAL test body over the IDENTICAL inputs under
// different compile flags. Compiling the same bytes twice makes "identical" a
// guarantee of the build system, not a claim a reviewer has to verify across two
// files that could drift. The only per-variant knobs are compile DEFINITIONS the
// CMake helpers set: XPMATH_CONTRACTION_MODE (0 = OFF, 1 = ON) selects the
// gate-vs-report behavior, and XPMATH_BASELINE_PATH (ON variant only) points
// at the recorded baseline count.
//
// WHY GROUND TRUTH IS PLAIN FP64 (PROVABLE, STRONGER THAN DD's QUADMATH ORACLE)
// ----------------------------------------------------------------------------
// This is the single deliberate divergence from T1.5's shape, inherited from
// ff_eft_test (T2.1). The exact product of two FP32 values needs at most
// 2*24 = 48 bits; the exact sum needs at most 25. IEEE double (FP64) has a 53-bit
// mantissa. Since 48 <= 53 and 25 <= 53, widening each FP32 operand to double and
// multiplying/summing THERE is EXACT — no rounding. So the reference
//     p_ref = (float)((double)a * (double)b)             // == fl(a*b), round-to-nearest
//     e_ref = (float)((double)a * (double)b - (double)p_ref)  // exact, fits in a float
// is a *provable* decomposition, not an approximate "close enough" check.
//
// This is a STRONGER oracle than DD's: DD's ground truth is __float128, exact
// for DD EFTs only because binary128's 113-bit mantissa exceeds the 106-bit
// DD product. FF's FP64 oracle is exact by the same headroom argument but needs
// no type wider than a machine word — it is algebraically provable with the
// hardware double every compiler already has, and so runs on any target rather
// than x86_64 only. Reported as the one shape divergence from T1.1 — see the
// T2.5 report / DONE block.
//
// (This block used to add "does NOT need KOKKOS_EP_HAVE_QUADMATH / does NOT
// runtime-SKIP-77 / runs UNCONDITIONALLY", contrasting with dd_fma_guard_test.
// That gate is gone from every test now, so the contrast has no subject.)
//
// WHY THE ORACLE CANNOT BE CORRUPTED BY CONTRACTION
// -------------------------------------------------
// The reference (p_ref, e_ref) is built from the EXACT FP64 product. In the ON
// build the WHOLE translation unit compiles under -ffp-contract=fast, so one might
// worry the oracle's own `exact - (double)p_ref` (a mul-then-sub adjacency) gets
// fused. It does not matter if it does: `(double)a * (double)b` is EXACTLY
// representable in FP64 (48 <= 53 bits), so the residual `exact - p_ref` is itself
// exact whether computed as a separate subtract or fused into one FMA — a fused
// `fma(a, b, -p_ref)` and the two-op form yield the identical value because there
// is no rounding to disagree about. The reference is therefore ground truth
// regardless of how aggressively the surrounding TU is built. (The DD oracle is
// immune for the same headroom reason: binary128 add/sub/mul are libgcc soft-float
// CALLS (__addtf3 and friends), which a compiler cannot fuse at all.)
//
// SCOPE (docs/TEST_SUITE_PLAN.md T2.5, layer 5): production `two_prod` only —
// the one FF primitive where (now FMA-based, formerly splitter-based)
// contraction/reassociation is a documented hazard. twoSum is built from
// adds/subtracts with no mul-then-± adjacency, so it is contraction-immune; it is
// included below purely as a labeled CONTROL (it must stay exact under both
// postures). ff_math.hpp is NOT modified (rule 4). Higher-level ops (log/sin/…)
// are out of scope, as are complex ops. See the "Scope-out" list in the T2.5 task.
//
// C4 CHUNK C: MIGRATED WHOLE, NOT SPLIT — AND THAT IS A DELIBERATE DEVIATION
// --------------------------------------------------------------------------
// CORE_PLAN section C4 lists this file among eight MIXED translation units, to be
// split into a host half carrying the oracle and a device half carrying the
// launch. That list was measured BEFORE C2 split tests/test_utils.hpp, and the
// classification no longer holds: this TU's oracle is FP64, not binary128 (see
// the block above, which predates C4 and gives the algebra), and after C2 it
// includes test_utils_device.hpp rather than the host header. It is not mixed.
// MEASURED on this tree: the only two occurrences of the 128-bit type's name in
// this file are in prose, and the preprocessor strips both.
//
// Splitting a TU that is already pure would duplicate ~250 lines of input
// construction and oracle across two files, turn two targets into four, and
// contradict the "single source, two targets" rationale three blocks up — all to
// separate a host oracle from a device launch that can legitimately share a
// compilation. So this file follows its own sibling instead: chunk B left
// tests/ff_eft_test.cpp, whose oracle is the same FP64 argument, as ONE whole TU
// on the C3 harness and added it to the purity gate's FILES list. This file is
// now in that list too, which is what turns "it is pure" from a claim into a
// checked property.
//
// What DID change here: the Kokkos View/parallel_for/initialize surface became
// xpt::buffer/parallel_for_n, the launch is checked for a vendor error code, and
// <ff_math.hpp> became <xp/ff_math.hpp>. The test body, the inputs, the oracle,
// both postures and the committed baseline are untouched.
// ============================================================================

// C4 chunk C: the launch goes through tests/device_harness.hpp (C3) rather than
// Kokkos. This TU names only the xp core, so it links no Kokkos at all and is
// registered in the Kokkos-free set.
#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include "corpus.hpp"
#include <xp/ff_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace xpmath;

// Which contraction posture were we compiled under? Set by the CMake helpers
// (xpm_add_device_eft_test -> 0, xpm_add_device_eft_test_contract_on -> 1). Default
// to OFF/gate if somehow unset so a flagless build fails loud rather than silently
// skipping the gate.
#ifndef XPMATH_CONTRACTION_MODE
#  define XPMATH_CONTRACTION_MODE 0
#endif

#if XPMATH_CONTRACTION_MODE == 0
static const char* kPostureName = "OFF (-ffp-contract=off / --fmad=false)";
#else
static const char* kPostureName = "ON  (-ffp-contract=fast / --fmad=true)";
#endif

// ----------------------------------------------------------------------------
// twoSum, COPIED VERBATIM from tests/ff_eft_test.cpp (T2.1). CONTROL: all
// +/-, no mul-then-± adjacency, so no compiler can contract it; it must stay
// exact under both postures regardless of what happens to two_prod.
// ----------------------------------------------------------------------------
struct TwoOut { float hi; float lo; };

XPMATH_INLINE_FUNCTION TwoOut two_sum(float a, float b) {
    float s   = a + b;
    float e   = s - a;
    float err = (b - e) + (a - (s - e));
    return TwoOut{ s, err };
}

// ----------------------------------------------------------------------------
// twoProduct domain predicate. See the "DOMAIN WIDENING" header note: the
// splitter-overflow exclusion is gone (two_prod never splits); what remains is
// the FMA-residual-exactness precondition, which applies regardless of how
// the residual is computed. FP32's exponent range is ~6x narrower than
// FP64's, so these bounds are much tighter than dd_fma_guard_test's FP64
// versions.
// ----------------------------------------------------------------------------

inline bool prod_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float fmin = std::numeric_limits<float>::min();  // smallest normal FP32
    auto normal_or_zero = [fmin](float x) {
        return x == 0.0f || std::fabs(x) >= fmin;          // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    if (a == 0.0f || b == 0.0f) return true;                       // exact product 0
    // Exact product magnitude in FP64 (a*b is exact there: 48 <= 53).
    double tp  = (double)a * (double)b;
    double mag = tp < 0.0 ? -tp : tp;
    const double hi_lim = (double)std::numeric_limits<float>::max();  // FP32 overflow
    const double lo_lim = std::ldexp(1.0, -102);                      // underflow headroom
    if (mag > hi_lim) return false;   // product (and a1*b1) would overflow FP32
    if (mag < lo_lim) return false;   // error term would fall into FP32 subnormals
    return true;
}

// ----------------------------------------------------------------------------
// twoSum oracle-faithfulness predicate (control pass only). FP32-SPECIFIC — has
// NO counterpart in dd_fma_guard_test, and this is a *reported* deviation from
// T1.5's shape (see the T2.5 report / DONE block).
//
// WHY IT IS NEEDED. The DD guard reuses its twoProduct input list for the twoSum
// control unchanged, and that is safe at FP64 scale: the twoProduct domain filter
// admits only products in the normal range, which for FP64 implies operands whose
// exact SUM also fits binary128's 113-bit oracle. At FP32 that implication breaks.
// A pair like (a = FLT_MIN = 2^-126, b = 2^24) has an in-range product (2^-102,
// admitted by prod_in_domain) yet an exact real sum spanning 2^24 down to 2^-126 —
// 150+24 = 174 significant bits, FAR beyond FP64's 53-bit mantissa. The FP32
// twoSum is CORRECT there (hi = 2^24, lo = 2^-126, both representable and
// non-overlapping); it is the FP64 *decomposition oracle* that collapses — `(double)
// a + (double)b` rounds the tiny addend away, so oracle_two_sum reports lo == 0 and
// the exact FP32 lo looks like a "mismatch". Verified: the FP64 twoSum error term
// on such a pair is nonzero, i.e. the double sum is itself inexact, so the oracle —
// not the transform — is out of domain. (ff_eft_test never hit this because it uses
// a SEPARATE, narrower sum domain and a reconstruction-style check; this file
// inherited DD's single-list shape, which is what surfaces the gap.)
//
// THE FIX is exactly the EFT tests' skip-not-fail discipline, applied to the
// control's oracle: a pair is in the twoSum control's domain only when the exact
// FP64 sum is faithful, i.e. the FP64 twoSum error term is zero (the double sum
// carries no rounding). That is a provable, tautological test of "does my 53-bit
// oracle represent this sum exactly," and it excludes ONLY the wide-exponent pairs
// the oracle cannot witness — never a pair where FP32 twoSum is actually wrong.
// twoProduct needs no such guard: its exact product is always <= 48 bits, so the
// FP64 oracle is unconditionally faithful (confirmed empirically — 0 mismatches).
//
// FMA MIGRATION NOTE: removing the splitter-overflow exclusion from
// prod_in_domain (see "DOMAIN WIDENING" in the file header) admits a wider
// band of operand magnitudes (up to FLT_MAX, not just up to ~2^115) into the
// shared input list, which is exactly the kind of wide-exponent-gap pair this
// predicate exists to catch. No change to this predicate was needed: it is
// derived purely from (a, b) themselves, not from any splitter-specific
// threshold, so it already generalizes to the widened domain. Re-verified
// empirically after the widening: of the 110675 pairs the widened domain now
// admits, this predicate excludes 524 (leaving 110151 for the twoSum
// control); twoProduct's 110675 are unaffected (no analogous exclusion
// applies to it) and all remain exact under contraction-off (0 mismatches).
// ----------------------------------------------------------------------------
inline bool sum_oracle_faithful(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    double da = (double)a, db = (double)b;
    double s  = da + db;
    if (!std::isfinite(s)) return false;             // sum must not overflow (twoSum domain)
    // FP64 twoSum error term; zero <=> the double sum is exact <=> oracle faithful.
    double e  = s - da;
    double err = (db - e) + (da - (s - e));
    return err == 0.0;
}

// ----------------------------------------------------------------------------
// Contraction-IMMUNE oracle for twoProduct. Built from the exact FP64 product;
// see the header note. Returns the TRUE (p, e) decomposition.
// ----------------------------------------------------------------------------
inline TwoOut oracle_two_prod(float a, float b) {
    double exact = (double)a * (double)b;         // exact: 48 <= 53 bits
    float  p = (float)exact;                       // fl(a*b), round-to-nearest
    float  e = (float)(exact - (double)p);         // exact residual, fits in a float
    return TwoOut{ p, e };
}
inline TwoOut oracle_two_sum(float a, float b) {
    double exact = (double)a + (double)b;          // exact: 25 <= 53 bits
    float  s = (float)exact;
    float  e = (float)(exact - (double)s);
    return TwoOut{ s, e };
}

// ----------------------------------------------------------------------------
// Failure-sample printer (first few only), with input bit patterns. FP32 hex is
// 0x%08x (mirrors ff_eft_test), NOT the 0x%016llx DD uses for FP64.
// ----------------------------------------------------------------------------
inline void print_mismatch(const char* which, const char* where,
                           float a, float b, TwoOut got, TwoOut ref) {
    uint32_t ab, bb;
    std::memcpy(&ab, &a, sizeof(float));
    std::memcpy(&bb, &b, sizeof(float));
    std::printf("    MISMATCH %s[%s]  a=%.9g (0x%08x)  b=%.9g (0x%08x)\n"
                "        got hi=%.9g lo=%.9g   ref hi=%.9g lo=%.9g\n",
                which, where, (double)a, ab, (double)b, bb,
                (double)got.hi, (double)got.lo, (double)ref.hi, (double)ref.lo);
}

// ----------------------------------------------------------------------------
// Input corpus: 10^5 random in-domain pairs + the corner-case corpus cross-
// product. Range [-1e18f, 1e18f] keeps every product comfortably underflow-safe
// and well inside FP32's normal range (|a*b| <= 1e36 < FLT_MAX); FMA-based
// two_prod has no splitter to overflow, so this range is chosen purely for
// product-magnitude safety now, not splitter safety. This is ff_eft_test's
// twoProduct broad range, NOT DD's 1e100: at 1e100 an FP32 product overflows and
// essentially every pair would be domain-skipped (the vacuous-coverage trap FP32's
// narrow exponent range sets). The corpus cross-product below (not this random
// range) is what reaches the wider operand magnitudes the domain widening now
// admits (see "DOMAIN WIDENING" in the file header). Deterministic seed.
// ----------------------------------------------------------------------------
static std::vector<std::pair<float,float>> build_inputs() {
    std::vector<std::pair<float,float>> pairs;
    pairs.reserve(120'000);

    std::mt19937_64 gen(0xF3A5C0117ULL);
    std::uniform_real_distribution<float> d(-1e18f, 1e18f);
    while (pairs.size() < 100'000) {
        float a = d(gen), b = d(gen);
        if (prod_in_domain(a, b)) pairs.emplace_back(a, b);
    }

    // Full corner-case corpus cross-product (i < j), in-domain members only.
    corpus::CorpusFlags flags;   // inf/zero/subnormals on, nan off
    std::vector<float> xs = corpus::unary<float>(flags);
    for (size_t i = 0; i < xs.size(); ++i)
        for (size_t j = i + 1; j < xs.size(); ++j)
            if (prod_in_domain(xs[i], xs[j])) pairs.emplace_back(xs[i], xs[j]);

    return pairs;
}

struct GuardCount { long tested = 0; long mismatches = 0; };

// Host pass: recompute production xp::two_prod directly on the host (governed
// by the CXX -ffp-contract flag), compare bit-exactly to the oracle reference.
static GuardCount run_host_pass(const std::vector<std::pair<float,float>>& in,
                                const std::vector<TwoOut>& ref, int& samples_left) {
    GuardCount c;
    for (size_t i = 0; i < in.size(); ++i) {
        xp::FloatFloat got_ff = xp::two_prod(in[i].first, in[i].second);
        TwoOut got{ got_ff.hi, got_ff.lo };
        ++c.tested;
        if (got.hi != ref[i].hi || got.lo != ref[i].lo) {
            ++c.mismatches;
            if (samples_left > 0) {
                print_mismatch("twoProd", "host", in[i].first, in[i].second, got, ref[i]);
                --samples_left;
            }
        }
    }
    return c;
}

// Device pass: recompute production xp::two_prod through the C3 harness. On a
// CUDA/HIP build this is a real kernel governed by the device compiler's
// --fmad flag; on a host build the harness runs the same functor in a serial
// loop, still through two genuinely separate allocations and a memcpy each
// way. Mirrors ff_eft_test's device parity pass.
//
// The functor is a trivially-copyable STRUCT holding raw device pointers, not a
// lambda: device_harness.hpp passes it by value into a __global__ and
// deliberately does not require nvcc --extended-lambda of its callers.
struct GuardKernel {
    const float* a;
    const float* b;
    float* p_hi;
    float* p_lo;

    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        xp::FloatFloat p = xp::two_prod(a[i], b[i]);
        p_hi[i] = p.hi;
        p_lo[i] = p.lo;
    }
};

static GuardCount run_device_pass(const std::vector<std::pair<float,float>>& in,
                                  const std::vector<TwoOut>& ref, int& samples_left) {
    const std::size_t n = in.size();

    xpt::buffer<float> va(n), vb(n), p_hi(n), p_lo(n);
    for (std::size_t i = 0; i < n; ++i) { va.host()[i] = in[i].first; vb.host()[i] = in[i].second; }
    va.to_device();
    vb.to_device();

    // parallel_for_n records the launch error and fences before it returns, so
    // from_device() below cannot race a live kernel.
    xpt::parallel_for_n(n, GuardKernel{va.device(), vb.device(),
                                       p_hi.device(), p_lo.device()});

    p_hi.from_device();
    p_lo.from_device();

    GuardCount c;
    for (std::size_t i = 0; i < n; ++i) {
        ++c.tested;
        if (p_hi.host()[i] != ref[i].hi || p_lo.host()[i] != ref[i].lo) {
            ++c.mismatches;
            if (samples_left > 0) {
                print_mismatch("twoProd", "device", in[i].first, in[i].second,
                               TwoOut{p_hi.host()[i], p_lo.host()[i]}, ref[i]);
                --samples_left;
            }
        }
    }
    return c;
}

// twoSum control pass (host): must stay exact under both postures.
static GuardCount run_sum_control(const std::vector<std::pair<float,float>>& in,
                                  int& samples_left) {
    GuardCount c;
    for (size_t i = 0; i < in.size(); ++i) {
        float a = in[i].first, b = in[i].second;
        // Skip pairs the FP64 oracle cannot witness (wide-exponent sums > 53 bits).
        // This is the FP32-specific control-domain guard — see sum_oracle_faithful.
        if (!sum_oracle_faithful(a, b)) continue;
        TwoOut got = two_sum(a, b);
        TwoOut ref = oracle_two_sum(a, b);
        ++c.tested;
        if (got.hi != ref.hi || got.lo != ref.lo) {
            ++c.mismatches;
            if (samples_left > 0) {
                print_mismatch("twoSum", "host", a, b, got, ref);
                --samples_left;
            }
        }
    }
    return c;
}

// ----------------------------------------------------------------------------
// Baseline mechanism (ON variant only). Records the observed mismatch count so a
// LATER build with a different compiler/ISA that changes contraction behavior is
// caught as drift. WARN-only (never fails) — the ON variant is a reporter.
//
// DECISION: implemented rather than deferred, mirroring dd_fma_guard_test. It is
// ~30 lines, and it is the only thing that turns the ON variant from a one-shot
// print into a real regression sentinel across compiler upgrades. File format is
// one integer on the first non-comment line; missing/unparseable file degrades
// gracefully to "print and hint", still exit 0.
// ----------------------------------------------------------------------------
#ifdef XPMATH_BASELINE_PATH
static void check_baseline(long observed) {
    const char* path = XPMATH_BASELINE_PATH;
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
        std::printf("  baseline: OK — observed mismatch count %ld matches baseline\n", observed);
    } else {
        std::printf("  baseline: *** DRIFT *** observed=%ld baseline=%ld\n", observed, baseline);
        std::printf("            contraction behavior changed since the baseline was\n");
        std::printf("            recorded (compiler/ISA/flag change). This is a WARNING,\n");
        std::printf("            not a failure. If the new value is correct, update %s.\n", path);
    }
}
#endif

// ============================================================================
int main(int, char**) {
    int rc = 0;
    {
        std::printf("=== ff_fma_guard_test (T2.5): FMA-contraction guard for FF production two_prod ===\n");
        std::printf("contraction posture: %s\n", kPostureName);
        std::printf("execution space: %s\n", xpt::where_name());
        // Deliberately does NOT spell the 128-bit type's name here. A string
        // literal survives preprocessing, so it would fail
        // scripts/check_device_tu_purity.sh, which carries this file.
        std::printf("Oracle: FP64 exact product (contraction-immune; 48 <= 53 bits, "
                    "needs no type wider than a machine word)\n\n");

        // Build inputs + contraction-immune reference once.
        std::vector<std::pair<float,float>> inputs = build_inputs();
        std::vector<TwoOut> ref(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i)
            ref[i] = oracle_two_prod(inputs[i].first, inputs[i].second);
        std::printf("inputs: %zu in-domain pairs (10^5 random + corpus cross-product)\n\n",
                    inputs.size());

        int samples_left = 8;

        // --- twoSum control (must be exact under both postures) --------------
        GuardCount S = run_sum_control(inputs, samples_left);
        std::printf("[control] twoSum (contraction-immune): tested=%ld mismatches=%ld\n",
                    S.tested, S.mismatches);

        // --- production two_prod: host + device passes ------------------------
        GuardCount H = run_host_pass(inputs, ref, samples_left);
        GuardCount D = run_device_pass(inputs, ref, samples_left);
        const long F = H.mismatches + D.mismatches;

        std::printf("[twoProd] host  : tested=%ld mismatches=%ld\n", H.tested, H.mismatches);
        std::printf("[twoProd] device: tested=%ld mismatches=%ld\n", D.tested, D.mismatches);
        std::printf("\ncontraction posture: %s. tested=%ld exact=%ld mismatches=%ld\n",
                    kPostureName, H.tested + D.tested, (H.tested + D.tested) - F, F);

#if XPMATH_CONTRACTION_MODE == 0
        // OFF variant: FAIL-GATE. The error terms MUST be exact — a stronger form
        // of what T2.1 already asserts. The twoSum control must also stay exact.
        std::printf("\nmode=OFF: fail-gating on any mismatch.\n");
        XPMATH_ASSERT(S.mismatches == 0,
                         "twoSum control not exact under contraction-off (unexpected)");
        XPMATH_ASSERT(F == 0,
                         "production two_prod not exact under contraction-off — "
                         "the -ffp-contract=off posture is not taking effect");
        // A nonzero vendor code is a TEST FAILURE, not a warning: a launch that
        // never ran leaves p_hi/p_lo at whatever the allocator returned, and the
        // comparison loop cannot tell that from a clean pass. Sticky since
        // process start, so a later good call cannot erase it.
        XPMATH_ASSERT(xpt::last_error() == 0,
                         "device harness reported a nonzero vendor error code");
        rc = ep_exit_code();
        std::printf("=== ff_fma_guard_test [OFF]: %s ===\n",
                    rc == 0 ? "ALL EXACT (posture holds)" : "FAILURES PRESENT");
#else
        // ON variant: REPORT ONLY. F may be nonzero if the compiler contracted the
        // production two_prod sequence; that is informative, not a failure. Exit 0
        // regardless.
        std::printf("\nmode=ON: reporting only (never fail-gates).\n");
        if (F == 0) {
            std::printf("  result: F == 0 — production two_prod stayed exact at\n");
            std::printf("          -ffp-contract=fast on this compiler/ISA target.\n");
            std::printf("          The -ffp-contract=off safety posture is belt+suspenders here.\n");
        } else {
            std::printf("  result: F == %ld — production two_prod was NOT exact under\n", F);
            std::printf("          contraction-on. The -ffp-contract=off posture in\n");
            std::printf("          ff_math.hpp's build is REQUIRED; this count is the evidence.\n");
        }
        // The twoSum control should still be exact even in ON mode (contraction-
        // immune). If it is not, something is very wrong — but still WARN, do not
        // gate (this variant is a reporter by contract).
        if (S.mismatches != 0)
            std::printf("  WARNING: twoSum control showed %ld mismatches under ON — unexpected "
                        "(twoSum has no contractible adjacency).\n", S.mismatches);
#  ifdef XPMATH_BASELINE_PATH
        check_baseline(F);
#  endif
        // The reporter exits 0 on the MEASUREMENT and nonzero on the APPARATUS.
        // A launch that failed did not report anything, so F would be noise.
        XPMATH_ASSERT(xpt::last_error() == 0,
                         "device harness reported a nonzero vendor error code");
        rc = ep_exit_code();
        std::printf("=== ff_fma_guard_test [ON]: REPORTED (mismatches=%ld, exit %d) ===\n",
                    F, rc);
#endif
    }
    return rc;
}
