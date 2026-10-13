// ============================================================================
// dd_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for DD.  Plan T1.5.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// DD's production `two_prod` (include/xp/dd_math.hpp, backing `multiply`,
// `multiply_scalar`, `divide`, `divide_scalar`) computes the exact product
// error term via an FMA-based TwoProduct, not Dekker/Veltkamp splitting:
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
// real-number-algebra optimizer, like they cancel to zero. This test is the
// *positive*, ctest-gated check of that posture: build the SAME shipped
// `two_prod` under BOTH contraction settings and cross-check it against a
// reference built a structurally different way.
//
//   * contraction-OFF build  -> `two_prod` must be bit-exact against the
//                               binary128 oracle (F == 0). FAIL-GATES
//                               (XPMATH_ASSERT).
//   * contraction-ON  build  -> the compiler is ALLOWED to contract/reassociate
//                               more aggressively. Two outcomes, both
//                               informative, neither a failure:
//                                 (a) F == 0  — the shipped guards held even
//                                     under the more permissive posture.
//                                 (b) F  > 0  — something upstream of the
//                                     volatile guards broke; the -ffp-contract
//                                     =off posture in dd_math.hpp's own build
//                                     is REQUIRED, and this count is evidence.
//                               The ON variant REPORTS, it does not gate
//                               (always exits 0). A *change* in F between
//                               builds (this compiler vs. the last one) is the
//                               regression signal (see baseline mechanism).
//
// WHY THIS NO LONGER TESTS A LOCAL DEKKER MIRROR
// -----------------------------------------------
// Before the FMA-based TwoProduct migration, this file duplicated dd_math.hpp's
// Dekker splitting sequence locally (mirror-and-comment, matching T1.1's
// dd_eft_test.cpp) because the thing under contraction scrutiny —
// `a1*b1 - c11` — was buried inside `multiply`'s longer renormalization
// sequence and needed isolating. `two_prod` is now two lines with no hidden
// algebra to duplicate, so this file calls the PRODUCTION function directly
// (`xp::two_prod`, forward-declared and defined in dd_math.hpp, already
// included below) rather than re-deriving it. That is strictly stronger: it
// exercises the actual translation unit, actual inlining, and actual
// NOINLINE/volatile annotations the shipped binary uses, not a hand-copied
// stand-in that could silently drift from them.
//
// dd_math.hpp's historical Dekker twoProduct implementation is NOT deleted —
// it still exists as `two_prod_dekker` in tests/dd_eft_test.cpp (T1.1), kept
// there as an independent, second exact algorithm for the same mathematical
// transform, cross-checked against the same binary128 oracle. It has no
// further role here once it stopped being what `multiply`/`two_prod` ship.
//
// WHY THE ORACLE CANNOT BE CORRUPTED BY CONTRACTION
// -------------------------------------------------
// The reference (p_ref, e_ref) is built from the EXACT __float128 product:
//     p_ref = (double)((f128)a * (f128)b)          // == fl(a*b), round-to-nearest
//     e_ref = (double)((f128)a*(f128)b - (f128)p_ref)  // exact, fits in a double
// binary128's 113-bit mantissa holds the exact 106-bit FP64 product, so both are
// exact and computed by a code path (a single f128 multiply) that has no
// mul-then-add adjacency for a compiler to contract. The reference is therefore
// the ground truth regardless of how aggressively `two_prod`'s own sequence is
// built.
//
// SCOPE (docs/TEST_SUITE_PLAN.md T1.5, layer 5): production `two_prod` only —
// the one DD primitive where (now FMA-based, formerly splitter-based)
// contraction/reassociation is a documented hazard. twoSum is built from
// adds/subtracts with no mul-then-± adjacency, so it is contraction-immune; it
// is included below purely as a labeled CONTROL (it must stay exact under both
// postures). Higher-level ops (log/sin/…) are out of scope.
//
// DOMAIN WIDENING vs. the pre-migration version of this file
// ------------------------------------------------------------
// The old Dekker-based domain predicate excluded |operand| >= ~1.34e300
// (2^996) because SPLITTING an operand that large overflows the splitter
// itself — a hazard specific to Veltkamp splitting that does not exist for
// FMA-based `two_prod` (it never splits). That exclusion is removed below.
// What remains — rejecting subnormal operands and bounding the TRUE product
// away from double's overflow/underflow edges — is NOT splitter-specific: it
// is the IEEE-754 "exact FMA residual" precondition (Boldo & Muller), the same
// requirement an FMA-based TwoProduct has that a Dekker one does. It stays.
//
// THIS IS THE HOST HALF. CORE_PLAN section C4 step 2 split the TU.
//   There used to be a run_device_pass() here, running the same sequence
//   through a Kokkos parallel_for against the same binary128 reference. That
//   cannot be built for a GPU and the obstruction is structural, not a flag:
//   nvcc gives every TU a device pass and rejects std::vector<__float128> inside
//   it (S6), so the oracle and the launch cannot share a file. The whole target,
//   host pass included, was therefore unbuildable on A100 (S8b).
//
//   The device pass is now tests/dd_fma_guard_test_device.cpp, with the same two
//   postures and the same baseline mechanism. It links no Kokkos and carries no
//   128-bit type, so its reference is an independently-coded FMA identity
//   (fma(a,b,-a*b), built WITHOUT the production volatile guards) rather than
//   the binary128 product — exact for the same reason, and the comparison asks
//   whether the shipped guarded path still agrees with the naive unguarded one.
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

using namespace xpmath;

// Which contraction posture were we compiled under? Set by the CMake helpers
// (xpm_add_host_eft_test -> 0, xpm_add_host_eft_test_contract_on -> 1). Default
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
// twoSum, COPIED VERBATIM from tests/dd_eft_test.cpp (T1.1). CONTROL: all
// +/-, no mul-then-± adjacency, so no compiler can contract it; it must stay
// exact under both postures regardless of what happens to two_prod.
// ----------------------------------------------------------------------------
struct TwoOut { double hi; double lo; };

XPMATH_INLINE_FUNCTION TwoOut two_sum(double a, double b) {
    double s   = a + b;
    double e   = s - a;
    double err = (b - e) + (a - (s - e));
    return TwoOut{ s, err };
}

// ----------------------------------------------------------------------------
// twoProduct domain predicate. See the "DOMAIN WIDENING" header note: the
// splitter-overflow exclusion is gone (two_prod never splits); what remains is
// the FMA-residual-exactness precondition, which applies regardless of how
// the residual is computed.
// ----------------------------------------------------------------------------
inline bool prod_in_domain(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const double dmin = std::numeric_limits<double>::min();  // smallest normal
    auto normal_or_zero = [dmin](double x) {
        return x == 0.0 || std::fabs(x) >= dmin;             // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    if (a == 0.0 || b == 0.0) return true;                        // exact product 0
    // Exact product magnitude, computed in the 113-bit oracle (a*b is exact there).
    float128 tp = (float128)a * (float128)b;
    float128 mag = tp < (float128)0 ? -tp : tp;
    const float128 hi_lim = (float128)std::numeric_limits<double>::max();  // overflow guard
    const float128 lo_lim = (float128)std::ldexp(1.0, -969);              // underflow headroom
    if (mag > hi_lim) return false;   // product would overflow
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

// sum_oracle_faithful: is the __float128 oracle itself wide enough to witness
// this pair's exact sum?
//
// float128 has a 113-bit mantissa — ample for oracle_two_prod (a double product
// needs at most 106 significant bits, always well inside 113), but NOT always
// enough for a SUM: a+b can need up to ~2098 significant bits to represent
// exactly when a and b's exponents differ by more than float128's headroom
// (e.g. a=1.0, b=DBL_MAX differ by 1023 bits). When that happens
// `(float128)a + (float128)b` itself rounds, silently discarding whatever part
// of the smaller operand falls below the 113-bit window — the DD-analogue of
// the FP32 "sum_oracle_faithful" guard T2.5's ff_fma_guard_test needed for the
// same reason, one precision level up.
//
// Removing twoProduct's splitter-overflow domain exclusion (see the
// "DOMAIN WIDENING" header note) let exactly these wide-exponent-gap pairs
// into the shared input list for the first time — two_prod's own oracle is
// unaffected (still exact), but the twoSum CONTROL, which reuses that same
// list, started reporting false mismatches where the oracle, not the code
// under test, was unfaithful.
//
// Detected by redoing Knuth's twoSum decomposition IN float128 itself: if
// float128 addition rounded away part of the true sum, re-deriving the
// residual by the same exact algorithm, at float128's own precision, yields a
// nonzero "error of the error" — proof the representation lost bits.
inline bool sum_oracle_faithful(double a, double b) {
    float128 af = (float128)a, bf = (float128)b;
    float128 s  = af + bf;
    float128 v  = s - af;
    float128 e  = (af - (s - v)) + (bf - v);
    return e == (float128)0;
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
// product. Range [-1e100, 1e100] keeps every product underflow-safe.
// Deterministic seed.
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

// Host pass: recompute production xp::two_prod directly on the host (governed
// by the CXX -ffp-contract flag), compare bit-exactly to the oracle reference.
static GuardCount run_host_pass(const std::vector<std::pair<double,double>>& in,
                                const std::vector<TwoOut>& ref, int& samples_left) {
    GuardCount c;
    for (size_t i = 0; i < in.size(); ++i) {
        xp::DoubleDouble got_dd = xp::two_prod(in[i].first, in[i].second);
        TwoOut got{ got_dd.hi, got_dd.lo };
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
        if (!sum_oracle_faithful(a, b)) continue;  // oracle out of its own domain
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
        std::printf("=== dd_fma_guard_test (T1.5): FMA-contraction guard for DD production two_prod ===\n");
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

        // --- production two_prod: host pass -----------------------------------
        GuardCount H = run_host_pass(inputs, ref, samples_left);
        const long F = H.mismatches;

        std::printf("[twoProd] host  : tested=%ld mismatches=%ld\n", H.tested, H.mismatches);
        std::printf("\ncontraction posture: %s. tested=%ld exact=%ld mismatches=%ld\n",
                    kPostureName, H.tested, H.tested - F, F);

#if XPMATH_CONTRACTION_MODE == 0
        // OFF variant: FAIL-GATE. The error terms MUST be exact — a stronger form
        // of what T1.1 already asserts. The twoSum control must also stay exact.
        std::printf("\nmode=OFF: fail-gating on any mismatch.\n");
        XPMATH_ASSERT(S.mismatches == 0,
                         "twoSum control not exact under contraction-off (unexpected)");
        XPMATH_ASSERT(F == 0,
                         "production two_prod not exact under contraction-off — "
                         "the FMA-based TwoProduct's guards are not holding");
        rc = ep_exit_code();
        std::printf("=== dd_fma_guard_test [OFF]: %s ===\n",
                    rc == 0 ? "ALL EXACT (posture holds)" : "FAILURES PRESENT");
#else
        // ON variant: REPORT ONLY. F may be nonzero if the compiler reassociated
        // something upstream of two_prod's guards; that is informative, not a
        // failure. Exit 0 regardless.
        std::printf("\nmode=ON: reporting only (never fail-gates).\n");
        if (F == 0) {
            std::printf("  result: F == 0 — production two_prod stayed exact at\n");
            std::printf("          -ffp-contract=fast on this compiler/ISA target.\n");
            std::printf("          The -ffp-contract=off safety posture is belt+suspenders here.\n");
        } else {
            std::printf("  result: F == %ld — production two_prod was NOT exact under\n", F);
            std::printf("          contraction-on. The -ffp-contract=off posture in\n");
            std::printf("          dd_math.hpp's build is REQUIRED; this count is the evidence.\n");
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
        rc = 0;  // reporter: always success
        std::printf("=== dd_fma_guard_test [ON]: REPORTED (mismatches=%ld, exit 0) ===\n", F);
#endif
    }
    return rc;
}
