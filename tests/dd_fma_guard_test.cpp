// ============================================================================
// dd_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for DD.  Plan T1.5.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// DD's `multiply` computes the exact product error term via Dekker's twoProduct
// (splitter 134217729.0 = 2^27 + 1, in `multiply`). The error term
//
//     e = (((a1*b1 - p) + a1*b2) + a2*b1) + a2*b2
//
// is only correct if each `x*y - z` is TWO distinct rounded operations. If the
// compiler CONTRACTS `a1*b1 - p` into a single fused multiply-add, the rounding
// that Dekker's algebra depends on never happens and the "error" term is wrong —
// silently. T1.1's dd_eft_test builds under -ffp-contract=off / --fmad=false so
// its OWN results are trustworthy; that is a *defensive* posture protecting one
// test. T1.5 is the *positive* test of that posture: build the very same Dekker
// twoProduct under BOTH contraction settings and cross-check against a
// contraction-immune __float128 oracle.
//
//   * contraction-OFF build  -> the error terms must be EXACT (F == 0). This is a
//                               stronger restatement of what T1.1 already asserts,
//                               and it FAIL-GATES (KOKKOS_EP_ASSERT).
//   * contraction-ON  build  -> the compiler is ALLOWED to contract. Two outcomes,
//                               both informative, neither a failure:
//                                 (a) F == 0  — the compiler did NOT contract this
//                                     sequence (e.g. GCC at -O3 -ffp-contract=fast
//                                     on a non-FMA ISA target). Our safety posture
//                                     is belt+suspenders on this toolchain.
//                                 (b) F  > 0  — the compiler DID contract. The
//                                     -ffp-contract=off posture in dd_math.hpp's
//                                     build is REQUIRED, and this count is the
//                                     evidence.
//                               The ON variant therefore REPORTS, it does not gate
//                               (always exits 0). A *change* in F between builds is
//                               the regression signal (see baseline mechanism).
//
// A SINGLE SOURCE, TWO TARGETS
// ----------------------------
// This one file is compiled TWICE by tests/CMakeLists.txt into two executables:
//   dd_fma_guard_test              (xpm_add_host_eft_test             -> OFF)
//   dd_fma_guard_test_contract_on  (xpm_add_host_eft_test_contract_on  -> ON)
// Single-source (not two-sources-with-shared-header) is deliberate: the whole
// point is to run the IDENTICAL test body over the IDENTICAL inputs under
// different compile flags. Compiling the same bytes twice makes "identical" a
// guarantee of the build system, not a claim a reviewer has to verify across two
// files that could drift. The only per-variant knobs are compile DEFINITIONS the
// CMake helpers set: KOKKOS_EP_CONTRACTION_MODE (0 = OFF, 1 = ON) selects the
// gate-vs-report behavior, and KOKKOS_EP_BASELINE_PATH (ON variant only) points
// at the recorded baseline count.
//
// WHY THE ORACLE CANNOT BE CORRUPTED BY CONTRACTION
// -------------------------------------------------
// The reference (p_ref, e_ref) is built from the EXACT __float128 product:
//     p_ref = (double)((f128)a * (f128)b)          // == fl(a*b), round-to-nearest
//     e_ref = (double)((f128)a*(f128)b - (f128)p_ref)  // exact, fits in a double
// binary128's 113-bit mantissa holds the exact 106-bit FP64 product, so both are
// exact and computed by a code path (a single f128 multiply) that has no
// mul-then-add adjacency for a compiler to contract. The reference is therefore
// the ground truth regardless of how aggressively the Dekker sequence is built.
//
// SCOPE (docs/TEST_SUITE_PLAN.md T1.5, layer 5): the Dekker twoProduct ONLY — the
// one DD primitive where contraction is a documented hazard. twoSum is built from
// adds/subtracts with no mul-then-±-adjacency, so it is contraction-immune; it is
// included below purely as a labeled CONTROL (it must stay exact under both
// postures). dd_math.hpp is NOT modified. Higher-level ops (log/sin/…) are out of
// scope. See the "Scope-out" list in the T1.5 task.
//
// THIS IS THE HOST HALF. CORE_PLAN section C4 step 2 split the TU.
//   There used to be a run_device_pass() here, running the same Dekker sequence
//   through a Kokkos parallel_for against the same binary128 reference. That
//   cannot be built for a GPU and the obstruction is structural, not a flag:
//   nvcc gives every TU a device pass and rejects std::vector<__float128> inside
//   it (S6), so the oracle and the launch cannot share a file. The whole target,
//   host pass included, was therefore unbuildable on A100 (S8b).
//
//   The device pass is now tests/dd_fma_guard_test_device.cpp, with the same two
//   postures and the same baseline mechanism. It links no Kokkos and carries no
//   128-bit type, so its reference is the FMA identity fma(a,b,-a*b) rather than
//   the binary128 product — exact for the same reason and immune to contraction
//   for a better one (an EXPLICIT fused operation is not what -ffp-contract
//   licenses the compiler to change).
//
//   Consequently NOTHING BELOW LAUNCHES A KERNEL. F is now the host mismatch
//   count alone.
// ============================================================================

#include "test_utils_host.hpp"
#include "corpus.hpp"
#include <xp/dd_math.hpp>

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
// (xpm_add_host_eft_test -> 0, xpm_add_host_eft_test_contract_on -> 1). Default
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
// EFT primitives, COPIED VERBATIM from tests/dd_eft_test.cpp (T1.1).
// ----------------------------------------------------------------------------
// Tests are standalone: we duplicate rather than #include dd_eft_test.cpp. The
// duplication is intentional and self-documenting — it is the exact sequence from
// dd_math.hpp's `multiply` (lines 197-211) / `two_prod` (270-278), the thing
// under contraction scrutiny. If this ever drifts from dd_eft_test.cpp's copy,
// that is itself a bug worth catching.

struct TwoOut { double hi; double lo; };

// twoSum (Knuth) — mirrors the transform embedded in `add`,
// with a.lo == b.lo == 0. CONTROL: all +/-, no mul-then-± adjacency, so no
// compiler can contract it; it must stay exact under both postures.
XPMATH_INLINE_FUNCTION TwoOut two_sum(double a, double b) {
    double s   = a + b;
    double e   = s - a;
    double err = (b - e) + (a - (s - e));
    return TwoOut{ s, err };
}

// Dekker twoProduct — mirrors `multiply` / the standalone `two_prod`,
// with a.lo == b.lo == 0. Splitter 134217729.0 = 2^27 + 1.
// THE PRIMITIVE UNDER TEST: `a1*b1 - p` (and the a1*b2 / a2*b1 / a2*b2 terms) are
// the mul-then-± pairs a compiler may fuse into an FMA, which would break the EFT.
XPMATH_INLINE_FUNCTION TwoOut two_prod_dekker(double a, double b) {
    const double split = 134217729.0;        // 2^27 + 1
    double cona = a * split, conb = b * split;
    double a1 = cona - (cona - a), b1 = conb - (conb - b);
    double a2 = a - a1,            b2 = b - b1;
    double p  = a * b;                                             // fl(a*b)
    double e  = (((a1 * b1 - p) + a1 * b2) + a2 * b1) + a2 * b2;   // exact error
    return TwoOut{ p, e };
}

// ----------------------------------------------------------------------------
// twoProduct domain predicate, COPIED from dd_eft_test.cpp (T1.1). A pair outside
// Dekker's proven domain (subnormal operands, splitter overflow, product over/
// underflow) is skipped, not counted — see the precondition note in T1.1.
// ----------------------------------------------------------------------------
inline double split_safe_max() { return std::ldexp(1.0, 996); }  // ~6.7e299

inline bool prod_in_domain(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const double dmin = std::numeric_limits<double>::min();  // smallest normal
    auto normal_or_zero = [dmin](double x) {
        return x == 0.0 || std::fabs(x) >= dmin;             // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    const double ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;  // splitter overflow
    if (a == 0.0 || b == 0.0) return true;                        // exact product 0
    float128 tp = (float128)a * (float128)b;
    float128 mag = tp < (float128)0 ? -tp : tp;
    const float128 hi_lim = (float128)std::numeric_limits<double>::max();
    const float128 lo_lim = (float128)std::ldexp(1.0, -969);
    if (mag > hi_lim) return false;   // product (and a1*b1) would overflow
    if (mag < lo_lim) return false;   // error term would fall into subnormals
    return true;
}

// ----------------------------------------------------------------------------
// Contraction-IMMUNE oracle for twoProduct. Built from the exact __float128
// product; see the header note. Returns the TRUE (p, e) decomposition.
// ----------------------------------------------------------------------------
inline TwoOut oracle_two_prod(double a, double b) {
    float128 exact = (float128)a * (float128)b;
    double p = (double)exact;                     // fl(a*b), round-to-nearest
    double e = (double)(exact - (float128)p);     // exact residual, fits in a double
    return TwoOut{ p, e };
}
inline TwoOut oracle_two_sum(double a, double b) {
    float128 exact = (float128)a + (float128)b;
    double s = (double)exact;
    double e = (double)(exact - (float128)s);
    return TwoOut{ s, e };
}

// ----------------------------------------------------------------------------
// Failure-sample printer (first few only), with input bit patterns.
// ----------------------------------------------------------------------------
inline void print_mismatch(const char* which, const char* where,
                           double a, double b, TwoOut got, TwoOut ref) {
    uint64_t ab, bb;
    std::memcpy(&ab, &a, sizeof(double));
    std::memcpy(&bb, &b, sizeof(double));
    std::printf("    MISMATCH %s[%s]  a=%.17g (0x%016llx)  b=%.17g (0x%016llx)\n"
                "        got hi=%.17g lo=%.17g   ref hi=%.17g lo=%.17g\n",
                which, where, a, (unsigned long long)ab, b, (unsigned long long)bb,
                got.hi, got.lo, ref.hi, ref.lo);
}

// ----------------------------------------------------------------------------
// Input corpus: 10^5 random in-domain pairs + the corner-case corpus cross-
// product. Range [-1e100, 1e100] keeps every product splitter- and underflow-
// safe (|a| < 2^996; |a*b| < 1e200 << DBL_MAX). Deterministic seed.
// ----------------------------------------------------------------------------
static std::vector<std::pair<double,double>> build_inputs() {
    std::vector<std::pair<double,double>> pairs;
    pairs.reserve(120'000);

    std::mt19937_64 gen(0xF3A5C0117ULL);
    std::uniform_real_distribution<double> d(-1e100, 1e100);
    while (pairs.size() < 100'000) {
        double a = d(gen), b = d(gen);
        if (prod_in_domain(a, b)) pairs.emplace_back(a, b);
    }

    // Full corner-case corpus cross-product (i < j), in-domain members only.
    corpus::CorpusFlags flags;   // inf/zero/subnormals on, nan off
    std::vector<double> xs = corpus::unary<double>(flags);
    for (size_t i = 0; i < xs.size(); ++i)
        for (size_t j = i + 1; j < xs.size(); ++j)
            if (prod_in_domain(xs[i], xs[j])) pairs.emplace_back(xs[i], xs[j]);

    return pairs;
}

struct GuardCount { long tested = 0; long mismatches = 0; };

// Host pass: recompute two_prod_dekker directly on the host (governed by the CXX
// -ffp-contract flag), compare bit-exactly to the oracle reference.
static GuardCount run_host_pass(const std::vector<std::pair<double,double>>& in,
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

// twoSum control pass (host): must stay exact under both postures.
static GuardCount run_sum_control(const std::vector<std::pair<double,double>>& in,
                                  int& samples_left) {
    GuardCount c;
    for (size_t i = 0; i < in.size(); ++i) {
        double a = in[i].first, b = in[i].second;
        if (!std::isfinite(a + b)) continue;   // twoSum domain: sum must not overflow
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
// DECISION: implemented rather than deferred. It is ~30 lines, and it is the only
// thing that turns the ON variant from a one-shot print into a real regression
// sentinel across compiler upgrades (the stated goal in the T1.5 task). File
// format is one integer on the first non-comment line; missing/unparseable file
// degrades gracefully to "print and hint", still exit 0.
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
int main(int, char**) {
    int rc = 0;
    {
        std::printf("=== dd_fma_guard_test (T1.5): FMA-contraction guard for DD Dekker twoProduct ===\n");
        std::printf("contraction posture: %s\n", kPostureName);
        std::printf("execution space: host (the device pass moved to "
                    "dd_fma_guard_test_device in C4)\n");
        std::printf("Oracle: binary128 exact product (contraction-immune)\n\n");

        // Build inputs + contraction-immune reference once.
        std::vector<std::pair<double,double>> inputs = build_inputs();
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

        // --- Dekker twoProduct: host pass ------------------------------------
        GuardCount H = run_host_pass(inputs, ref, samples_left);
        const long F = H.mismatches;

        std::printf("[twoProd] host  : tested=%ld mismatches=%ld\n", H.tested, H.mismatches);
        std::printf("\ncontraction posture: %s. tested=%ld exact=%ld mismatches=%ld\n",
                    kPostureName, H.tested, H.tested - F, F);

#if KOKKOS_EP_CONTRACTION_MODE == 0
        // OFF variant: FAIL-GATE. The error terms MUST be exact — a stronger form
        // of what T1.1 already asserts. The twoSum control must also stay exact.
        std::printf("\nmode=OFF: fail-gating on any mismatch.\n");
        KOKKOS_EP_ASSERT(S.mismatches == 0,
                         "twoSum control not exact under contraction-off (unexpected)");
        KOKKOS_EP_ASSERT(F == 0,
                         "Dekker twoProduct not exact under contraction-off — "
                         "the -ffp-contract=off posture is not taking effect");
        rc = ep_exit_code();
        std::printf("=== dd_fma_guard_test [OFF]: %s ===\n",
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
            std::printf("          sequence. The -ffp-contract=off posture in dd_math.hpp's\n");
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
        std::printf("=== dd_fma_guard_test [ON]: REPORTED (mismatches=%ld, exit 0) ===\n", F);
#endif
    }
    return rc;
}
