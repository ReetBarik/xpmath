// ============================================================================
// dd_property_test_device.cpp — the DEVICE half of T1.3's property tests for DD.
// ============================================================================
// Added by CORE_PLAN section C4 step 2 (chunk C).
//
// WHY THE SPLIT
//   tests/dd_property_test.cpp ran Group A (bit-exact identities, no oracle),
//   Group B (tolerance identities scored in digits against binary128), Test C
//   (named constants, also binary128) and a five-check device pass, all in one
//   TU. nvcc gives every TU a device pass and rejects std::vector<__float128>
//   inside it (S6), so that file could not be built for a GPU at all — Group A
//   included, which needs no oracle whatsoever. This file is the device pass,
//   with no 128-bit type anywhere in it (enforced by
//   scripts/check_device_tu_purity.sh, which carries it in its FILES list).
//
// WHAT MOVED, WHAT GREW, AND WHAT WAS LOST — stated plainly
//
//   GREW. The old device pass ran three Group A identities: A1, A3, A5. All
//   SEVEN are here now — A1, A2, A3, A4, A5, A6 and the binary A8. Every one is
//   a pure sign/structure identity with no oracle and no tolerance, so there was
//   never a reason for the device side to carry only three; the reason it did is
//   that the mixed TU had a large host pass and the device pass was a spot check
//   bolted on. Each also now sweeps the corner-case corpus (subnormals, +/-0,
//   powers of two) in addition to 10^5 random inputs, which is what the host
//   Group A runner always did and the device pass never did.
//
//   LOST. The old device pass also ran two GROUP B checks — B1_sqrt_sq and
//   B4_pythag — scoring digits_of_accuracy against the binary128 oracle. Those
//   cannot come here and are NOT replaced:
//
//     * They are not structural. "sqrt(a)^2 recovers a to N digits" is a
//       statement about a REFERENCE VALUE, and the only reference is the oracle.
//       There is no oracle-free identity in the neighbourhood: unlike twoSum,
//       which has an independent exact transform to cross-check against, sqrt has
//       no second exact algorithm at this width.
//
//     * Host/device bit-parity is NOT a substitute, and shipping it would be
//       worse than shipping nothing. A real GPU's sqrt and sincos may legitimately
//       differ from the host libm in the last bits; a bit-parity check would go
//       red on correct hardware, and the only way to make it green again would be
//       to add a tolerance — which is a SECOND SCORER, and docs/CORRECTNESS.md
//       allows exactly one measurement and one verdict per point.
//
//   So B1_sqrt_sq and B4_pythag are HOST-ONLY from C4 onward. The host half still
//   runs both over 10^6 inputs, which is 10x the old device pass's 10^5; what is
//   genuinely gone is the ability to notice a GPU whose sqrt or sincos is
//   accurate on the host and inaccurate on the device. The apparatus that DOES
//   cover accuracy end to end is validation/sweep/ (docs/CORRECTNESS.md), and it
//   is host-measured for exactly the same reason — the oracle cannot share a
//   translation unit with device code.
//
// NOTHING HERE SCORES. Every check below is `==` on a pair of doubles.
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
#include <random>
#include <utility>
#include <vector>

using namespace kokkos_ep;

static constexpr int kDeviceN = 100'000;   // 10^5 random inputs per identity

// ----------------------------------------------------------------------------
// Bit-pattern helper (same hex format as dd_property_test.cpp / dd_eft_test.cpp).
// ----------------------------------------------------------------------------
static uint64_t dbits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(double));
    return b;
}

// Exact FP64 equality on both components. Value equality (==), so +0.0 and -0.0
// compare equal — intended, exactly as in the host half: the zero-result
// identities are specified as "hi == 0 && lo == 0", i.e. the value zero
// regardless of sign bit.
static bool dd_eq(double ghi, double glo, double whi, double wlo) {
    return ghi == whi && glo == wlo;
}

// ----------------------------------------------------------------------------
// Domain predicates, copied from the host half.
// A3/A4 multiply, and multiply goes through Dekker's Veltkamp split
// (a.hi * (2^27+1)), which overflows to inf -> nan for |x| >= 2^996. Such inputs
// are OUT OF multiply's domain and are SKIPPED, not failed — the same splitter
// limit T1.1 documents, not a defect. Random |x| < 1e8 never approaches it, so
// this only gates the corpus's largest entries.
// ----------------------------------------------------------------------------
static bool dom_all(double x)    { return std::isfinite(x); }
static bool dom_dekker(double x) { return std::isfinite(x) && std::fabs(x) < std::ldexp(1.0, 996); }

// Corpus flags: zeros ON, inf OFF, nan OFF (subnormals default ON) — matches the
// host half and the invariant test. inf/nan are excluded because e.g.
// inf + (-inf) = nan is not the zero identity.
static corpus::CorpusFlags corpus_flags() {
    corpus::CorpusFlags f;
    f.include_zero = true;
    f.include_inf  = false;
    f.include_nan  = false;
    return f;
}

// ----------------------------------------------------------------------------
// The identities, as tag structs with a device-callable apply().
//
// Each one computes BOTH sides on the device and hands back (got, want); the
// host only compares bits. That is deliberate: evaluating `want` host-side would
// silently turn several of these into host-vs-device comparisons, which is a
// different question. A5, for instance, is only a real test of dd::abs's sign
// BRANCH if the branch runs where the arithmetic does.
//
// Bodies are copied verbatim from the host half's Group A lambdas, so a
// divergence between the two is visible as a textual difference.
// ----------------------------------------------------------------------------

struct IdA1 {                                       // a + (-a) == 0
    static constexpr const char* name = "A1_add_neg";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::add(a, dd::negate(a)); want = dd::DoubleDouble(0.0, 0.0);
    }
};
struct IdA2 {                                       // a - a == 0
    static constexpr const char* name = "A2_self_sub";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = a - a; want = dd::DoubleDouble(0.0, 0.0);
    }
};
struct IdA3 {                                       // a * 1 == a
    static constexpr const char* name = "A3_mul_one";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::multiply(a, dd::DoubleDouble(1.0)); want = a;
    }
};
struct IdA4 {                                       // a * (-1) == -a
    static constexpr const char* name = "A4_mul_negone";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::multiply(a, dd::DoubleDouble(-1.0)); want = dd::negate(a);
    }
};
struct IdA5 {                                       // |a| == (a.hi >= 0 ? a : -a)
    static constexpr const char* name = "A5_abs_branch";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::abs(a); want = (a.hi >= 0.0) ? a : dd::negate(a);
    }
};
struct IdA6 {                                       // -(-a) == a
    static constexpr const char* name = "A6_double_neg";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::negate(dd::negate(a)); want = a;
    }
};
struct IdA8 {                                       // add(a,b) == add(b,a)
    static constexpr const char* name = "A8_add_comm";
    XPMATH_INLINE_FUNCTION static void apply(const dd::DoubleDouble& a, const dd::DoubleDouble& b,
                                             dd::DoubleDouble& got, dd::DoubleDouble& want) {
        got = dd::add(a, b); want = dd::add(b, a);
    }
};

// ----------------------------------------------------------------------------
// Kernels. Trivially-copyable structs holding raw device pointers, NOT lambdas:
// device_harness.hpp passes the functor by value into a __global__ and
// deliberately does not require nvcc --extended-lambda of its callers.
// ----------------------------------------------------------------------------
template <class Id>
struct UnaryIdKernel {
    const double* x;
    double* gh; double* gl; double* wh; double* wl;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        dd::DoubleDouble a(x[i]), got, want;
        Id::apply(a, got, want);
        gh[i] = got.hi; gl[i] = got.lo; wh[i] = want.hi; wl[i] = want.lo;
    }
};

template <class Id>
struct BinaryIdKernel {
    const double* a; const double* b;
    double* gh; double* gl; double* wh; double* wl;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        dd::DoubleDouble x(a[i]), y(b[i]), got, want;
        Id::apply(x, y, got, want);
        gh[i] = got.hi; gl[i] = got.lo; wh[i] = want.hi; wl[i] = want.lo;
    }
};

// ----------------------------------------------------------------------------
// Runners.
// ----------------------------------------------------------------------------
struct IdResult { long n = 0; long skipped = 0; long failures = 0; };

// Build the input list for a unary identity: kDeviceN random draws from
// [-1e8, 1e8] plus the finite corpus, keeping only in-domain values. Filtering
// here rather than inside the kernel keeps the kernel branch-free and keeps
// "skipped" an honest count rather than a silent pass.
static std::vector<double> unary_inputs(uint64_t seed, bool (*in_domain)(double),
                                        long& skipped) {
    std::vector<double> xs;
    xs.reserve(kDeviceN + 128);
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-1e8, 1e8);
    for (int i = 0; i < kDeviceN; ++i) {
        double v = d(g);
        if (in_domain(v)) xs.push_back(v); else ++skipped;
    }
    for (double v : corpus::unary<double>(corpus_flags())) {
        if (in_domain(v)) xs.push_back(v); else ++skipped;
    }
    return xs;
}

template <class Id>
static IdResult run_unary(uint64_t seed, bool (*in_domain)(double) = dom_all) {
    IdResult R;
    std::vector<double> xs = unary_inputs(seed, in_domain, R.skipped);
    const std::size_t n = xs.size();

    xpt::buffer<double> dx(n), gh(n), gl(n), wh(n), wl(n);
    for (std::size_t i = 0; i < n; ++i) dx.host()[i] = xs[i];
    dx.to_device();
    xpt::parallel_for_n(n, UnaryIdKernel<Id>{dx.device(), gh.device(), gl.device(),
                                             wh.device(), wl.device()});
    gh.from_device(); gl.from_device(); wh.from_device(); wl.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        ++R.n;
        if (!dd_eq(gh.host()[i], gl.host()[i], wh.host()[i], wl.host()[i])) {
            ++R.failures;
            if (samples_left > 0) {
                std::printf("    FAIL %-14s x=%.17g (0x%016llx)  got.hi=%.17g got.lo=%.17g  "
                            "want.hi=%.17g want.lo=%.17g\n",
                            Id::name, xs[i], (unsigned long long)dbits(xs[i]),
                            gh.host()[i], gl.host()[i], wh.host()[i], wl.host()[i]);
                --samples_left;
            }
        }
    }
    std::printf("  [device] %-14s n=%-8ld skipped=%-4ld failures=%ld status=%s\n",
                Id::name, R.n, R.skipped, R.failures, R.failures == 0 ? "PASS" : "FAIL");
    return R;
}

template <class Id>
static IdResult run_binary(uint64_t seed) {
    IdResult R;
    std::vector<std::pair<double,double>> ps;
    ps.reserve(kDeviceN + 4096);
    {
        // add() does not Veltkamp-split, so it has no splitter-overflow limit;
        // the only out-of-domain case is a non-finite operand.
        std::mt19937_64 g(seed);
        std::uniform_real_distribution<double> d(-1e8, 1e8);
        for (int i = 0; i < kDeviceN; ++i) {
            double a = d(g), b = d(g);
            if (std::isfinite(a) && std::isfinite(b)) ps.emplace_back(a, b); else ++R.skipped;
        }
        for (const auto& p : corpus::binary<double>(corpus_flags())) {
            if (std::isfinite(p.first) && std::isfinite(p.second)) ps.push_back(p);
            else ++R.skipped;
        }
    }
    const std::size_t n = ps.size();

    xpt::buffer<double> da(n), db(n), gh(n), gl(n), wh(n), wl(n);
    for (std::size_t i = 0; i < n; ++i) { da.host()[i] = ps[i].first; db.host()[i] = ps[i].second; }
    da.to_device(); db.to_device();
    xpt::parallel_for_n(n, BinaryIdKernel<Id>{da.device(), db.device(),
                                              gh.device(), gl.device(), wh.device(), wl.device()});
    gh.from_device(); gl.from_device(); wh.from_device(); wl.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        ++R.n;
        if (!dd_eq(gh.host()[i], gl.host()[i], wh.host()[i], wl.host()[i])) {
            ++R.failures;
            if (samples_left > 0) {
                std::printf("    FAIL %-14s a=%.17g (0x%016llx) b=%.17g (0x%016llx)  "
                            "got.hi=%.17g got.lo=%.17g  want.hi=%.17g want.lo=%.17g\n",
                            Id::name, ps[i].first, (unsigned long long)dbits(ps[i].first),
                            ps[i].second, (unsigned long long)dbits(ps[i].second),
                            gh.host()[i], gl.host()[i], wh.host()[i], wl.host()[i]);
                --samples_left;
            }
        }
    }
    std::printf("  [device] %-14s n=%-8ld skipped=%-4ld failures=%ld status=%s\n",
                Id::name, R.n, R.skipped, R.failures, R.failures == 0 ? "PASS" : "FAIL");
    return R;
}

// ============================================================================
int main(int, char**) {
    std::printf("=== dd_property_test_device (T1.3, C4 device half): Group A "
                "bit-exact identities for DD ===\n");
    std::printf("execution space: %s\n", xpt::where_name());
    std::printf("Group A only: pure sign/structure identities, no oracle and no "
                "tolerance. Group B and Test C stay in dd_property_test (they need "
                "the 128-bit oracle) — see this file's header for what that costs.\n\n");

    std::printf("[Group A, device] 10^5 random in [-1e8, 1e8] + the finite corpus, "
                "per identity\n");

    long failures = 0, total_n = 0;
    auto tally = [&](IdResult r) { failures += r.failures; total_n += r.n; };

    tally(run_unary<IdA1>(700001ULL));
    tally(run_unary<IdA2>(700006ULL));
    tally(run_unary<IdA3>(700002ULL, dom_dekker));
    tally(run_unary<IdA4>(700007ULL, dom_dekker));
    tally(run_unary<IdA5>(700003ULL));
    tally(run_unary<IdA6>(700008ULL));
    tally(run_binary<IdA8>(700009ULL));

    std::printf("\n=== Summary ===\n");
    std::printf("  7 identities, %ld total checks, failures=%ld\n", total_n, failures);

    // A run that checked nothing is not a pass. The domain filters could in
    // principle reject everything (they do not, but that is a measurement, not an
    // assumption), and every count above would then read 0 failures.
    KOKKOS_EP_ASSERT(total_n > 0, "no identity check ran — every input was filtered out");
    KOKKOS_EP_ASSERT(failures == 0,
                     "a Group A bit-exact identity did not hold to the last bit on the device");
    // A nonzero vendor code is a TEST FAILURE, not a warning: a launch that never
    // ran leaves the output buffers at whatever the allocator returned, and the
    // comparison loop cannot tell that from a clean pass. Sticky since process
    // start, so a later good call cannot erase it.
    KOKKOS_EP_ASSERT(xpt::last_error() == 0,
                     "device harness reported a nonzero vendor error code");

    int rc = ep_exit_code();
    std::printf("=== dd_property_test_device: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    return rc;
}
