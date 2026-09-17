// ============================================================================
// dd_fma_guard_test_device.cpp — the DEVICE half of T1.5's FMA-contraction
//                                guard for DD.
// ============================================================================
// Added by CORE_PLAN section C4 step 2 (chunk C). tests/dd_fma_guard_test.cpp
// used to run a host pass AND a Kokkos device pass against one binary128
// reference; nvcc cannot give a device pass to a TU containing
// std::vector<__float128> (S6), so the whole target — host pass included — was
// unbuildable for a GPU. The host pass stays there; the device pass is here.
//
// WHAT IS BEING GUARDED
//   Dekker's twoProduct error term
//       e = (((a1*b1 - p) + a1*b2) + a2*b1) + a2*b2
//   is correct only if each `x*y - z` is TWO separately rounded operations. A
//   compiler that contracts `a1*b1 - p` into one fused multiply-add destroys the
//   algebra SILENTLY: the result still looks like a plausible error term. So the
//   guard needs a reference the contraction cannot touch.
//
// THE CONTRACTION-IMMUNE REFERENCE, WITHOUT A WIDER TYPE
//   The host half uses the exact binary128 product. That is unavailable here, and
//   its replacement is better suited to this particular question anyway:
//
//       p_ref = a * b ;  e_ref = fma(a, b, -p_ref)
//
//   IEEE 754-2019 requires fma to round a*b + c exactly once from the infinitely
//   precise value. Whenever a*b neither overflows nor underflows, a*b - p_ref is
//   exactly representable in a double, so that single rounding is the identity
//   and e_ref is the EXACT residual — the same number the binary128 oracle
//   produces, by a different route.
//
//   Crucially it is immune to the very thing under test. `-ffp-contract=fast`
//   licenses the compiler to FUSE separate operations; it does not change what an
//   EXPLICIT fma() call computes, in either posture. So the reference is stable
//   across the two builds of this file while the Dekker sequence is exactly what
//   moves — which is the measurement.
//
//   NOT A SECOND SCORER (docs/CORRECTNESS.md). No ulp, no digit count, no
//   tolerance: every comparison below is `==` on two doubles, and the number
//   reported is a count of bit mismatches.
//
// A SINGLE SOURCE, TWO TARGETS — unchanged discipline
//   dd_fma_guard_test_device              (xpm_add_device_eft_test          -> OFF)
//   dd_fma_guard_test_device_contract_on  (xpm_add_device_eft_test_contract_on -> ON)
//   OFF fail-gates on any mismatch; ON reports and always exits 0, with the same
//   WARN-only baseline drift check the host half has. Identical bytes compiled
//   twice, so "the two runs differ only in flags" is a property of the build
//   system rather than a claim about two files.
//
// NO 128-bit TYPE AND NO KOKKOS — enforced, not merely intended, by
// scripts/check_device_tu_purity.sh, which carries this file in its FILES list.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include "corpus.hpp"
#include <xp/dd_math.hpp>

#include <cmath>
#include <cstddef>
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

// Which contraction posture were we compiled under? Set by the CMake helpers
// (xpm_add_device_eft_test -> 0, ..._contract_on -> 1). Default to OFF/gate if
// somehow unset, so a flagless build fails loud rather than silently skipping.
#ifndef KOKKOS_EP_CONTRACTION_MODE
#  define KOKKOS_EP_CONTRACTION_MODE 0
#endif

#if KOKKOS_EP_CONTRACTION_MODE == 0
static const char* kPostureName = "OFF (-ffp-contract=off / --fmad=false)";
#else
static const char* kPostureName = "ON  (-ffp-contract=fast / --fmad=true)";
#endif

// ----------------------------------------------------------------------------
// Scalar fma, host+device. include/xp/config.hpp's detail:: dispatch covers the
// functions the library itself calls, and fma is not among them; this mirrors
// that header's two-branch pattern locally rather than widening a shipped header
// for a test's benefit.
// ----------------------------------------------------------------------------
XPMATH_INLINE_FUNCTION double dev_fma(double a, double b, double c) {
#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
    return ::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

// ----------------------------------------------------------------------------
// EFT primitives, COPIED VERBATIM from tests/dd_fma_guard_test.cpp, which copied
// them from tests/dd_eft_test.cpp, which mirrors dd_math.hpp's `multiply` /
// `two_prod`. The duplication is the same deliberate mirror-and-comment those
// files document: drift between the copies is itself a defect worth catching.
// ----------------------------------------------------------------------------

struct TwoOut { double hi; double lo; };

// twoSum (Knuth). CONTROL: all +/-, no mul-then-± adjacency, so no compiler can
// contract it; it must stay exact under both postures.
XPMATH_INLINE_FUNCTION TwoOut two_sum(double a, double b) {
    double s   = a + b;
    double e   = s - a;
    double err = (b - e) + (a - (s - e));
    return TwoOut{ s, err };
}

// Dekker twoProduct. THE PRIMITIVE UNDER TEST: `a1*b1 - p` and the three cross
// terms are the mul-then-± pairs a compiler may fuse.
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
// The two contraction-immune references, both computed ON THE DEVICE alongside
// the sequences they judge.
// ----------------------------------------------------------------------------

// twoProduct reference: the FMA identity. See the header block.
XPMATH_INLINE_FUNCTION TwoOut ref_fma_two_prod(double a, double b) {
    double p = a * b;
    double e = dev_fma(a, b, -p);
    return TwoOut{ p, e };
}

// twoSum reference: Dekker's fast2sum on magnitude-sorted operands. Exact for
// |x| >= |y| with x+y finite (Dekker 1971), 3 flops against Knuth's 6, and
// structurally a different expression — so agreement is evidence rather than a
// tautology. The sort is a compare and a swap; neither rounds, so the
// precondition holds by construction.
XPMATH_INLINE_FUNCTION TwoOut ref_fast2sum(double a, double b) {
    double x = a, y = b;
    if (xp::detail::fabs(y) > xp::detail::fabs(x)) { double t = x; x = y; y = t; }
    double s = x + y;
    double e = y - (s - x);
    return TwoOut{ s, e };
}

// ----------------------------------------------------------------------------
// Dekker's domain, by EXPONENT rather than by exact product magnitude.
// tests/dd_fma_guard_test.cpp forms the exact binary128 product and bounds it;
// that predicate cannot come here. frexp(x, &e) writes x = m * 2^e with
// |m| in [0.5, 1), so |x| is in [2^(e-1), 2^e) and |a*b| is in
// [2^(ea+eb-2), 2^(ea+eb)). Bounding THAT interval is a conservative reading of
// the same limits: it can refuse a pair binary128 would admit, and can never
// admit one binary128 would refuse — the only direction in which a loose
// predicate could turn a real defect into a pass.
// ----------------------------------------------------------------------------
static double split_safe_max() { return std::ldexp(1.0, 996); }  // ~6.7e299

static bool prod_in_domain(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const double dmin = std::numeric_limits<double>::min();  // smallest normal
    auto normal_or_zero = [dmin](double x) {
        return x == 0.0 || std::fabs(x) >= dmin;             // reject subnormals
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    const double ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;  // splitter overflow
    if (a == 0.0 || b == 0.0) return true;                         // exact product 0
    int ea = 0, eb = 0;
    std::frexp(a, &ea);
    std::frexp(b, &eb);
    if (ea + eb - 2 < -969) return false;   // error term could fall subnormal
    if (ea + eb > 1023)     return false;   // product could overflow
    return true;
}

// ----------------------------------------------------------------------------
// Failure-sample printer (first few only), with input bit patterns.
// ----------------------------------------------------------------------------
static void print_mismatch(const char* which, double a, double b,
                           TwoOut got, TwoOut ref) {
    uint64_t ab, bb;
    std::memcpy(&ab, &a, sizeof(double));
    std::memcpy(&bb, &b, sizeof(double));
    std::printf("    MISMATCH %s[device]  a=%.17g (0x%016llx)  b=%.17g (0x%016llx)\n"
                "        got hi=%.17g lo=%.17g   ref hi=%.17g lo=%.17g\n",
                which, a, (unsigned long long)ab, b, (unsigned long long)bb,
                got.hi, got.lo, ref.hi, ref.lo);
}

// ----------------------------------------------------------------------------
// Input corpus. 10^5 exponent-banded random pairs, |x| in [2^-400, 2^401), which
// puts |a*b| in [2^-800, 2^802) — inside Dekker's domain BY CONSTRUCTION, so the
// accept loop never spins and nothing is skipped. Plus the corner-case corpus
// cross-product, filtered by the exponent predicate above. Deterministic seed,
// the same 0xF3A5C0117 the host half uses.
// ----------------------------------------------------------------------------
static double gen_banded(std::mt19937_64& g) {
    std::uniform_real_distribution<double> dm(1.0, 2.0);
    std::uniform_int_distribution<int>     de(-400, 400);
    double v = std::ldexp(dm(g), de(g));
    return (g() & 1u) ? v : -v;
}

static std::vector<std::pair<double,double>> build_inputs() {
    std::vector<std::pair<double,double>> pairs;
    pairs.reserve(120'000);

    std::mt19937_64 gen(0xF3A5C0117ULL);
    while (pairs.size() < 100'000) {
        double a = gen_banded(gen), b = gen_banded(gen);
        if (prod_in_domain(a, b)) pairs.emplace_back(a, b);
    }

    corpus::CorpusFlags flags;   // inf/zero/subnormals on, nan off
    std::vector<double> xs = corpus::unary<double>(flags);
    for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = i + 1; j < xs.size(); ++j)
            if (prod_in_domain(xs[i], xs[j])) pairs.emplace_back(xs[i], xs[j]);

    return pairs;
}

// ----------------------------------------------------------------------------
// The kernel. A trivially-copyable struct holding raw device pointers, NOT a
// lambda: device_harness.hpp passes F by value into a __global__ and deliberately
// does not require nvcc --extended-lambda.
//
// BOTH the sequence under test AND its reference are computed on the device. A
// host-computed reference would be testing the GPU against an x86 CPU, which is
// a different question; the point here is whether the DEVICE compiler contracted
// the DEVICE Dekker sequence, and only a reference built by the same compiler on
// the same hardware answers it.
// ----------------------------------------------------------------------------
struct GuardKernel {
    const double* a;
    const double* b;
    double* p_hi;  double* p_lo;    // Dekker twoProduct (under test)
    double* r_hi;  double* r_lo;    // FMA reference
    double* s_hi;  double* s_lo;    // Knuth twoSum (control)
    double* q_hi;  double* q_lo;    // fast2sum reference

    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        TwoOut p = two_prod_dekker(a[i], b[i]);
        TwoOut r = ref_fma_two_prod(a[i], b[i]);
        TwoOut s = two_sum(a[i], b[i]);
        TwoOut q = ref_fast2sum(a[i], b[i]);
        p_hi[i] = p.hi; p_lo[i] = p.lo;
        r_hi[i] = r.hi; r_lo[i] = r.lo;
        s_hi[i] = s.hi; s_lo[i] = s.lo;
        q_hi[i] = q.hi; q_lo[i] = q.lo;
    }
};

struct GuardCount { long tested = 0; long mismatches = 0; };

// ============================================================================
int main(int, char**) {
    int rc = 0;

    std::printf("=== dd_fma_guard_test_device (T1.5, C4 device half): FMA-contraction "
                "guard for DD Dekker twoProduct ===\n");
    std::printf("contraction posture: %s\n", kPostureName);
    std::printf("execution space: %s\n", xpt::where_name());
    std::printf("Reference: fma(a,b,-a*b) — exact and contraction-immune, and it needs "
                "no type wider than a double\n\n");

    std::vector<std::pair<double,double>> in = build_inputs();
    const std::size_t n = in.size();
    std::printf("inputs: %zu in-domain pairs (10^5 exponent-banded random + corpus "
                "cross-product)\n\n", n);
    // A guard that tested nothing passes silently, so the input count is itself
    // asserted rather than printed and trusted.
    KOKKOS_EP_ASSERT(n > 0, "no in-domain input pairs were built — the guard would "
                            "have measured nothing");

    xpt::buffer<double> va(n), vb(n);
    xpt::buffer<double> p_hi(n), p_lo(n), r_hi(n), r_lo(n);
    xpt::buffer<double> s_hi(n), s_lo(n), q_hi(n), q_lo(n);
    for (std::size_t i = 0; i < n; ++i) {
        va.host()[i] = in[i].first;
        vb.host()[i] = in[i].second;
    }
    va.to_device();
    vb.to_device();

    xpt::parallel_for_n(n, GuardKernel{va.device(), vb.device(),
                                       p_hi.device(), p_lo.device(),
                                       r_hi.device(), r_lo.device(),
                                       s_hi.device(), s_lo.device(),
                                       q_hi.device(), q_lo.device()});

    p_hi.from_device(); p_lo.from_device();
    r_hi.from_device(); r_lo.from_device();
    s_hi.from_device(); s_lo.from_device();
    q_hi.from_device(); q_lo.from_device();

    int samples_left = 8;
    GuardCount P, S;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = in[i].first, b = in[i].second;

        // Dekker twoProduct vs the FMA reference. Every pair here is in-domain
        // by construction of build_inputs().
        ++P.tested;
        TwoOut pg{ p_hi.host()[i], p_lo.host()[i] };
        TwoOut pr{ r_hi.host()[i], r_lo.host()[i] };
        if (pg.hi != pr.hi || pg.lo != pr.lo) {
            ++P.mismatches;
            if (samples_left > 0) { print_mismatch("twoProd", a, b, pg, pr); --samples_left; }
        }

        // twoSum control vs fast2sum. Its only domain condition is that the sum
        // not overflow, which the corpus can violate where the product predicate
        // already let the pair through (a +/- 0 pair, for instance).
        if (std::isfinite(a + b)) {
            ++S.tested;
            TwoOut sg{ s_hi.host()[i], s_lo.host()[i] };
            TwoOut sr{ q_hi.host()[i], q_lo.host()[i] };
            if (sg.hi != sr.hi || sg.lo != sr.lo) {
                ++S.mismatches;
                if (samples_left > 0) { print_mismatch("twoSum", a, b, sg, sr); --samples_left; }
            }
        }
    }

    const long F = P.mismatches;

    std::printf("[control] twoSum (contraction-immune): tested=%ld mismatches=%ld\n",
                S.tested, S.mismatches);
    std::printf("[twoProd] device: tested=%ld mismatches=%ld\n", P.tested, P.mismatches);
    std::printf("\ncontraction posture: %s. tested=%ld exact=%ld mismatches=%ld\n",
                kPostureName, P.tested, P.tested - F, F);

    // A nonzero vendor code is a TEST FAILURE in BOTH postures, including the
    // reporting one. If the launch never ran, the output buffers hold whatever
    // the allocator returned and F is meaningless — reporting it as evidence
    // would be worse than reporting nothing. Sticky since process start.
    const int verr = xpt::last_error();

#if KOKKOS_EP_CONTRACTION_MODE == 0
    // OFF variant: FAIL-GATE. The error terms MUST be exact.
    std::printf("\nmode=OFF: fail-gating on any mismatch.\n");
    KOKKOS_EP_ASSERT(S.tested > 0, "the twoSum control tested nothing");
    KOKKOS_EP_ASSERT(S.mismatches == 0,
                     "twoSum control not exact under contraction-off (unexpected)");
    KOKKOS_EP_ASSERT(F == 0,
                     "device Dekker twoProduct not exact under contraction-off — "
                     "the -ffp-contract=off / --fmad=false posture is not taking effect");
    KOKKOS_EP_ASSERT(verr == 0, "device harness reported a nonzero vendor error code");
    rc = ep_exit_code();
    std::printf("=== dd_fma_guard_test_device [OFF]: %s ===\n",
                rc == 0 ? "ALL EXACT (posture holds)" : "FAILURES PRESENT");
#else
    // ON variant: REPORT ONLY. F may be nonzero if the compiler contracted the
    // Dekker sequence; that is informative, not a failure.
    std::printf("\nmode=ON: reporting only (never fail-gates on the mismatch count).\n");
    if (F == 0) {
        std::printf("  result: F == 0 — this compiler did NOT contract the Dekker\n");
        std::printf("          sequence at -ffp-contract=fast / --fmad=true on this target.\n");
        std::printf("          The contraction-off safety posture is belt+suspenders here.\n");
    } else {
        std::printf("  result: F == %ld — this compiler DID contract the Dekker\n", F);
        std::printf("          sequence. The contraction-off posture is REQUIRED;\n");
        std::printf("          this count is the evidence.\n");
    }
    if (S.mismatches != 0)
        std::printf("  WARNING: twoSum control showed %ld mismatches under ON — unexpected "
                    "(twoSum has no contractible adjacency).\n", S.mismatches);
#  ifdef KOKKOS_EP_BASELINE_PATH
    {
        // WARN-only drift check, same contract and same file format as the host
        // half: one integer on the first non-comment line, missing/unparseable
        // degrades to a hint, never a failure.
        const char* path = KOKKOS_EP_BASELINE_PATH;
        std::ifstream f(path);
        if (!f) {
            std::printf("  baseline: no file at %s\n", path);
            std::printf("            record this run by writing \"%ld\" as the first\n", F);
            std::printf("            non-comment line of that file to arm drift detection.\n");
        } else {
            long baseline = -1;
            std::string line;
            bool got = false;
            while (std::getline(f, line)) {
                std::size_t p = line.find_first_not_of(" \t");
                if (p == std::string::npos || line[p] == '#') continue;
                try { baseline = std::stol(line.substr(p)); got = true; } catch (...) {}
                break;
            }
            if (!got) {
                std::printf("  baseline: %s present but unparseable; skipping drift check\n", path);
            } else if (F == baseline) {
                std::printf("  baseline: OK — observed mismatch count %ld matches baseline\n", F);
            } else {
                std::printf("  baseline: *** DRIFT *** observed=%ld baseline=%ld\n", F, baseline);
                std::printf("            contraction behavior changed since the baseline was\n");
                std::printf("            recorded (compiler/ISA/flag change). This is a WARNING,\n");
                std::printf("            not a failure. If the new value is correct, update %s.\n", path);
            }
        }
    }
#  endif
    // The reporter exits 0 on the MEASUREMENT and nonzero on the APPARATUS: a
    // launch that failed did not report anything.
    KOKKOS_EP_ASSERT(verr == 0, "device harness reported a nonzero vendor error code");
    rc = ep_exit_code();
    std::printf("=== dd_fma_guard_test_device [ON]: REPORTED (mismatches=%ld, exit %d) ===\n",
                F, rc);
#endif
    return rc;
}
