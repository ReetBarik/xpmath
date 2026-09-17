// ============================================================================
// ff_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for FF.  Plan T2.5.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// FF's `multiply` computes the exact product error term via Dekker's twoProduct
// (splitter 8193.0f = 2^13 + 1, in `multiply` / the standalone `two_prod`). The
// error term
//
//     e = (((a1*b1 - p) + a1*b2) + a2*b1) + a2*b2
//
// is only correct if each `x*y - z` is TWO distinct rounded operations. If the
// compiler CONTRACTS `a1*b1 - p` into a single fused multiply-add, the rounding
// that Dekker's algebra depends on never happens and the "error" term is wrong —
// silently, exactly as at FP64 (the hazard is width-independent; only the
// splitter and the mantissa change). T2.1's ff_eft_test builds under
// -ffp-contract=off / --fmad=false so its OWN results are trustworthy; that is a
// *defensive* posture protecting one test. T2.5 is the *positive* test of that
// posture: build the very same Dekker twoProduct under BOTH contraction settings
// and cross-check against a contraction-immune FP64 oracle.
//
//   * contraction-OFF build  -> the error terms must be EXACT (F == 0). This is a
//                               stronger restatement of what T2.1 already asserts,
//                               and it FAIL-GATES (KOKKOS_EP_ASSERT).
//   * contraction-ON  build  -> the compiler is ALLOWED to contract. Two outcomes,
//                               both informative, neither a failure:
//                                 (a) F == 0  — the compiler did NOT contract this
//                                     sequence (e.g. GCC at -O3 -ffp-contract=fast
//                                     on a non-FMA ISA target). Our safety posture
//                                     is belt+suspenders on this toolchain.
//                                 (b) F  > 0  — the compiler DID contract. The
//                                     -ffp-contract=off posture in ff_math.hpp's
//                                     build is REQUIRED, and this count is the
//                                     evidence.
//                               The ON variant therefore REPORTS, it does not gate
//                               (always exits 0). A *change* in F between builds is
//                               the regression signal (see baseline mechanism).
//
// This is the FF analogue of tests/dd_fma_guard_test.cpp (T1.5). Every design
// decision T1.5 locked in carries over verbatim — single-source/two-targets, a
// contraction-immune oracle, a twoSum CONTROL, host+device passes, OFF gates /
// ON reports with a committed baseline. The ONE material difference is the oracle
// (see next block), inherited from ff_eft_test's divergence from dd_eft_test.
//
// A SINGLE SOURCE, TWO TARGETS
// ----------------------------
// This one file is compiled TWICE by tests/CMakeLists.txt into two executables:
//   ff_fma_guard_test              (kokkos_ep_add_eft_test          -> OFF)
//   ff_fma_guard_test_contract_on  (kokkos_ep_add_eft_test_contract_on -> ON)
// Single-source (not two-sources-with-shared-header) is deliberate: the whole
// point is to run the IDENTICAL test body over the IDENTICAL inputs under
// different compile flags. Compiling the same bytes twice makes "identical" a
// guarantee of the build system, not a claim a reviewer has to verify across two
// files that could drift. The only per-variant knobs are compile DEFINITIONS the
// CMake helpers set: KOKKOS_EP_CONTRACTION_MODE (0 = OFF, 1 = ON) selects the
// gate-vs-report behavior, and KOKKOS_EP_BASELINE_PATH (ON variant only) points
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
// SCOPE (docs/TEST_SUITE_PLAN.md T2.5, layer 5): the Dekker twoProduct ONLY — the
// one FF primitive where contraction is a documented hazard. twoSum is built from
// adds/subtracts with no mul-then-± adjacency, so it is contraction-immune; it is
// included below purely as a labeled CONTROL (it must stay exact under both
// postures). ff_math.hpp is NOT modified (rule 4). Higher-level ops (log/sin/…)
// are out of scope, as are complex ops. See the "Scope-out" list in the T2.5 task.
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
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace kokkos_ep;

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
// EFT primitives, COPIED VERBATIM from tests/ff_eft_test.cpp (T2.1).
// ----------------------------------------------------------------------------
// Tests are standalone: we duplicate rather than #include ff_eft_test.cpp. The
// duplication is intentional and self-documenting — it is the exact sequence from
// ff_math.hpp's `multiply` (lines 193-207) / `two_prod` (266-274), the thing
// under contraction scrutiny. If this ever drifts from ff_eft_test.cpp's copy,
// that is itself a bug worth catching.

struct TwoOut { float hi; float lo; };

// twoSum (Knuth) — mirrors the transform embedded in `add`,
// with a.lo == b.lo == 0. CONTROL: all +/-, no mul-then-± adjacency, so no
// compiler can contract it; it must stay exact under both postures.
KOKKOS_INLINE_FUNCTION TwoOut two_sum(float a, float b) {
    float s   = a + b;
    float e   = s - a;
    float err = (b - e) + (a - (s - e));
    return TwoOut{ s, err };
}

// Dekker twoProduct — mirrors `multiply` / the standalone `two_prod`,
// with a.lo == b.lo == 0. Splitter 8193.0f = 2^13 + 1.
// THE PRIMITIVE UNDER TEST: `a1*b1 - p` (and the a1*b2 / a2*b1 / a2*b2 terms) are
// the mul-then-± pairs a compiler may fuse into an FMA, which would break the EFT.
KOKKOS_INLINE_FUNCTION TwoOut two_prod_dekker(float a, float b) {
    const float split = 8193.0f;             // 2^13 + 1
    float cona = a * split, conb = b * split;
    float a1 = cona - (cona - a), b1 = conb - (conb - b);
    float a2 = a - a1,            b2 = b - b1;
    float p  = a * b;                                             // fl(a*b)
    float e  = (((a1 * b1 - p) + a1 * b2) + a2 * b1) + a2 * b2;   // exact error
    return TwoOut{ p, e };
}

// ----------------------------------------------------------------------------
// twoProduct domain predicate, COPIED from ff_eft_test.cpp (T2.1). A pair outside
// Dekker's proven domain (subnormal operands, splitter overflow, product over/
// underflow) is skipped, not counted — see the precondition note in T2.1. FP32's
// exponent range is ~6x narrower than FP64's, so these bounds are much tighter
// than dd_fma_guard_test's FP64 versions.
// ----------------------------------------------------------------------------

// Splitter-overflow bound: `cona = a * 8193.0f` overflows to +inf once
// |a| >= FLT_MAX / 8193 (~2^114.9998), poisoning `cona - (cona - a)` = inf-inf =
// NaN. This is PORT_NOTES §4a's `exp` splitter-overflow mechanism.
inline float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

inline bool prod_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float fmin = std::numeric_limits<float>::min();  // smallest normal FP32
    auto normal_or_zero = [fmin](float x) {
        return x == 0.0f || std::fabs(x) >= fmin;          // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    const float ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;  // splitter overflow
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
// product. Range [-1e18f, 1e18f] keeps every product splitter- and underflow-safe
// (|a| << FLT_MAX/8193 ~ 4.15e34; |a*b| <= 1e36 < FLT_MAX). This is ff_eft_test's
// twoProduct broad range, NOT DD's 1e100: at 1e100 an FP32 product overflows and
// essentially every pair would be domain-skipped (the vacuous-coverage trap FP32's
// narrow exponent range sets). Deterministic seed.
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

// Host pass: recompute two_prod_dekker directly on the host (governed by the CXX
// -ffp-contract flag), compare bit-exactly to the oracle reference.
static GuardCount run_host_pass(const std::vector<std::pair<float,float>>& in,
                                const std::vector<TwoOut>& ref, int& samples_left) {
    GuardCount c;
    for (size_t i = 0; i < in.size(); ++i) {
        TwoOut got = two_prod_dekker(in[i].first, in[i].second);
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

// Device pass: recompute two_prod_dekker inside a parallel_for on the default
// execution space (governed by the CUDA --fmad flag on a CUDA build; reduces to
// host on a Serial build). Mirrors ff_eft_test's Test D.
static GuardCount run_device_pass(const std::vector<std::pair<float,float>>& in,
                                  const std::vector<TwoOut>& ref, int& samples_left) {
    using exec_space = Kokkos::DefaultExecutionSpace;
    const int n = (int)in.size();

    Kokkos::View<float*, exec_space> va("va", n), vb("vb", n);
    Kokkos::View<float*, exec_space> p_hi("p_hi", n), p_lo("p_lo", n);
    auto hva = Kokkos::create_mirror_view(va);
    auto hvb = Kokkos::create_mirror_view(vb);
    for (int i = 0; i < n; ++i) { hva(i) = in[i].first; hvb(i) = in[i].second; }
    Kokkos::deep_copy(va, hva);
    Kokkos::deep_copy(vb, hvb);

    Kokkos::parallel_for("ff_fma_guard_device", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(int i) {
            TwoOut p = two_prod_dekker(va(i), vb(i));
            p_hi(i) = p.hi; p_lo(i) = p.lo;
        });
    Kokkos::fence();

    auto hphi = Kokkos::create_mirror_view(p_hi);
    auto hplo = Kokkos::create_mirror_view(p_lo);
    Kokkos::deep_copy(hphi, p_hi);
    Kokkos::deep_copy(hplo, p_lo);

    GuardCount c;
    for (int i = 0; i < n; ++i) {
        ++c.tested;
        if (hphi(i) != ref[i].hi || hplo(i) != ref[i].lo) {
            ++c.mismatches;
            if (samples_left > 0) {
                print_mismatch("twoProd", "device", in[i].first, in[i].second,
                               TwoOut{hphi(i), hplo(i)}, ref[i]);
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
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        std::printf("=== ff_fma_guard_test (T2.5): FMA-contraction guard for FF Dekker twoProduct ===\n");
        std::printf("contraction posture: %s\n", kPostureName);
        std::printf("execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
        std::printf("Oracle: FP64 exact product (contraction-immune; 48 <= 53 bits, no __float128)\n\n");

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

        // --- Dekker twoProduct: host + device passes -------------------------
        GuardCount H = run_host_pass(inputs, ref, samples_left);
        GuardCount D = run_device_pass(inputs, ref, samples_left);
        const long F = H.mismatches + D.mismatches;

        std::printf("[twoProd] host  : tested=%ld mismatches=%ld\n", H.tested, H.mismatches);
        std::printf("[twoProd] device: tested=%ld mismatches=%ld\n", D.tested, D.mismatches);
        std::printf("\ncontraction posture: %s. tested=%ld exact=%ld mismatches=%ld\n",
                    kPostureName, H.tested + D.tested, (H.tested + D.tested) - F, F);

#if KOKKOS_EP_CONTRACTION_MODE == 0
        // OFF variant: FAIL-GATE. The error terms MUST be exact — a stronger form
        // of what T2.1 already asserts. The twoSum control must also stay exact.
        std::printf("\nmode=OFF: fail-gating on any mismatch.\n");
        KOKKOS_EP_ASSERT(S.mismatches == 0,
                         "twoSum control not exact under contraction-off (unexpected)");
        KOKKOS_EP_ASSERT(F == 0,
                         "Dekker twoProduct not exact under contraction-off — "
                         "the -ffp-contract=off posture is not taking effect");
        rc = ep_exit_code();
        std::printf("=== ff_fma_guard_test [OFF]: %s ===\n",
                    rc == 0 ? "ALL EXACT (posture holds)" : "FAILURES PRESENT");
#else
        // ON variant: REPORT ONLY. F may be nonzero if the compiler contracted the
        // Dekker sequence; that is informative, not a failure. Exit 0 regardless.
        std::printf("\nmode=ON: reporting only (never fail-gates).\n");
        if (F == 0) {
            std::printf("  result: F == 0 — this compiler did NOT contract the Dekker\n");
            std::printf("          sequence at -ffp-contract=fast on this ISA target.\n");
            std::printf("          The -ffp-contract=off safety posture is belt+suspenders here.\n");
        } else {
            std::printf("  result: F == %ld — this compiler DID contract the Dekker\n", F);
            std::printf("          sequence. The -ffp-contract=off posture in ff_math.hpp's\n");
            std::printf("          build is REQUIRED; this count is the evidence.\n");
        }
        // The twoSum control should still be exact even in ON mode (contraction-
        // immune). If it is not, something is very wrong — but still WARN, do not
        // gate (this variant is a reporter by contract).
        if (S.mismatches != 0)
            std::printf("  WARNING: twoSum control showed %ld mismatches under ON — unexpected "
                        "(twoSum has no contractible adjacency).\n", S.mismatches);
#  ifdef KOKKOS_EP_BASELINE_PATH
        check_baseline(F);
#  endif
        rc = 0;  // reporter: always success
        std::printf("=== ff_fma_guard_test [ON]: REPORTED (mismatches=%ld, exit 0) ===\n", F);
#endif
    }
    Kokkos::finalize();
    return rc;
}
