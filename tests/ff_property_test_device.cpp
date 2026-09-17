// ============================================================================
// ff_property_test_device.cpp — the DEVICE half of T2.3's property tests for FF.
// ============================================================================
// Added by CORE_PLAN section C4 step 2 (chunk C). This is the FF analogue of
// dd_property_test_device.cpp; that file's header carries the full argument for
// the split and for what the device side gives up. The FF-specific points:
//
//   * ROUTE-A OPERANDS. The host half feeds unary identities ff::FloatFloat(x)
//     built from a DOUBLE, so the constructor splits it and .lo is generally
//     NONZERO — that is the whole point of Route A, and a single-float operand
//     would leave half of add()/multiply() untested. This TU therefore ships the
//     raw double to the device and constructs the FF operand INSIDE the kernel,
//     exactly as the old Kokkos device_run did.
//
//   * TWO domain limits, not one. FF's Veltkamp split multiplies by 8193, which
//     overflows for |x| >= FLT_MAX/8193 ~= 4.15e34; and FP32's subnormal tail has
//     a round-to-even hole below ~2^-100. Both are FORMAT limits documented in
//     T2.1/T2.3, so inputs hitting them are SKIPPED, not failed — the same
//     dom_dekker and denormal-tail audit the host half applies.
//
//   * GROUP S IS NOT HERE, and not because of the oracle. It is a table of IEEE
//     special-value requirements for divide (x / +/-inf = +/-0, nan propagation,
//     FLT_MAX divisors). Non-finite behaviour is precisely what a device
//     backend's fast-math posture may legitimately change, so asserting it on
//     the device would assert something no platform promises. It stays host-only
//     and that is a deliberate scope decision, not a limitation.
//
//   * LOST, exactly as in DD: the two Group B device checks, B1_sqrt_sq and
//     B4_pythag, scored digits against the binary128 oracle. They cannot come
//     here and are NOT replaced by a bit-parity or tolerance substitute — see
//     dd_property_test_device.cpp's header. The host half still runs both at
//     10^6 inputs.
//
// NOTHING HERE SCORES. Every check below is `==` on a pair of floats.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include "corpus.hpp"
#include <xp/ff_math.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using namespace kokkos_ep;
namespace ff = xp;

static constexpr int kDeviceN = 100'000;   // 10^5 random inputs per identity

// ----------------------------------------------------------------------------
// Bit-pattern helper (ff_property_test.cpp / ff_eft_test.cpp hex format).
// ----------------------------------------------------------------------------
static uint32_t fbits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(float));
    return b;
}

// Exact FP32 equality on both components. Value equality (==), so +0.0 and -0.0
// compare equal — intended, matching the host half.
static bool ff_eq(float ghi, float glo, float whi, float wlo) {
    return ghi == whi && glo == wlo;
}

// ----------------------------------------------------------------------------
// Domain limits, copied from the host half so the two consume identical bounds.
// ----------------------------------------------------------------------------
static constexpr double kUnderflowTail = 0x1p-100;   // ~7.9e-31, the FP32 round-to-even hole
static float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}
static bool in_underflow_tail(float hi) {
    return hi != 0.0f && std::fabs((double)hi) < kUnderflowTail;
}

static bool dom_all(double x) { return std::isfinite(x); }
static bool dom_dekker(double x) {
    if (!std::isfinite(x)) return false;
    double ax = std::fabs(x);
    if (ax == 0.0) return true;                        // exact: a*(+/-1) with a==0
    if (ax >= (double)split_safe_max()) return false;  // Dekker splitter overflow
    if (ax < kUnderflowTail) return false;             // denormal tail
    return true;
}

// Corpus flags: zeros ON, inf OFF, nan OFF (subnormals default ON) — matches the
// host half and the invariant test.
static corpus::CorpusFlags corpus_flags() {
    corpus::CorpusFlags f;
    f.include_zero = true;
    f.include_inf  = false;
    f.include_nan  = false;
    return f;
}

// ----------------------------------------------------------------------------
// The identities, as tag structs with a device-callable apply(). Both sides are
// evaluated ON THE DEVICE and the host compares bits only; see the DD file for
// why (evaluating `want` host-side silently changes the question being asked).
// Bodies are copied verbatim from the host half's Group A lambdas.
// ----------------------------------------------------------------------------

struct IdA1 {                                       // a + (-a) == 0
    static constexpr const char* name = "A1_add_neg";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::add(a, ff::negate(a)); want = ff::FloatFloat(0.0f, 0.0f);
    }
};
struct IdA2 {                                       // a - a == 0
    static constexpr const char* name = "A2_self_sub";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = a - a; want = ff::FloatFloat(0.0f, 0.0f);
    }
};
struct IdA3 {                                       // a * 1 == a
    static constexpr const char* name = "A3_mul_one";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::multiply(a, ff::FloatFloat(1.0f)); want = a;
    }
};
struct IdA4 {                                       // a * (-1) == -a
    static constexpr const char* name = "A4_mul_negone";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::multiply(a, ff::FloatFloat(-1.0f)); want = ff::negate(a);
    }
};
struct IdA5 {                                       // |a| == (a.hi >= 0 ? a : -a)
    static constexpr const char* name = "A5_abs_branch";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::abs(a); want = (a.hi >= 0.0f) ? a : ff::negate(a);
    }
};
struct IdA6 {                                       // -(-a) == a
    static constexpr const char* name = "A6_double_neg";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::negate(ff::negate(a)); want = a;
    }
};
struct IdA8 {                                       // add(a,b) == add(b,a)
    static constexpr const char* name = "A8_add_comm";
    XPMATH_INLINE_FUNCTION static void apply(const ff::FloatFloat& a, const ff::FloatFloat& b,
                                             ff::FloatFloat& got, ff::FloatFloat& want) {
        got = ff::add(a, b); want = ff::add(b, a);
    }
};

// ----------------------------------------------------------------------------
// Kernels. Trivially-copyable structs holding raw device pointers, NOT lambdas
// (device_harness.hpp passes the functor by value into a __global__ and does not
// require its callers to enable nvcc's extended lambdas).
//
// The unary kernel takes the input as a DOUBLE and builds the FF operand on the
// device: Route A, .lo generally nonzero. The binary kernel takes FLOATS, which
// is the host half's A8 convention (single-float operands, .lo == 0 — see the
// classification note in ff_property_test.cpp's header).
// ----------------------------------------------------------------------------
template <class Id>
struct UnaryIdKernel {
    const double* x;
    float* gh; float* gl; float* wh; float* wl;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        ff::FloatFloat a(x[i]), got, want;
        Id::apply(a, got, want);
        gh[i] = got.hi; gl[i] = got.lo; wh[i] = want.hi; wl[i] = want.lo;
    }
};

template <class Id>
struct BinaryIdKernel {
    const float* a; const float* b;
    float* gh; float* gl; float* wh; float* wl;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        ff::FloatFloat x(a[i]), y(b[i]), got, want;
        Id::apply(x, y, got, want);
        gh[i] = got.hi; gl[i] = got.lo; wh[i] = want.hi; wl[i] = want.lo;
    }
};

// ----------------------------------------------------------------------------
// Runners.
// ----------------------------------------------------------------------------
struct IdResult { long n = 0; long skipped = 0; long failures = 0; };

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
    for (float v : corpus::unary<float>(corpus_flags())) {
        if (in_domain((double)v)) xs.push_back((double)v); else ++skipped;
    }
    return xs;
}

template <class Id>
static IdResult run_unary(uint64_t seed, bool (*in_domain)(double) = dom_all) {
    IdResult R;
    std::vector<double> xs = unary_inputs(seed, in_domain, R.skipped);
    const std::size_t n = xs.size();

    xpt::buffer<double> dx(n);
    xpt::buffer<float>  gh(n), gl(n), wh(n), wl(n);
    for (std::size_t i = 0; i < n; ++i) dx.host()[i] = xs[i];
    dx.to_device();
    xpt::parallel_for_n(n, UnaryIdKernel<Id>{dx.device(), gh.device(), gl.device(),
                                             wh.device(), wl.device()});
    gh.from_device(); gl.from_device(); wh.from_device(); wl.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = gh.host()[i], b = gl.host()[i], c = wh.host()[i], d = wl.host()[i];
        // Denormal-tail audit, same as the host half: a mismatch whose limbs sit
        // in the FP32 subnormal tail is a round-to-even hole in the FORMAT, so it
        // is SKIPPED rather than counted as a failure of ff_math.hpp.
        if (!ff_eq(a, b, c, d) && (in_underflow_tail(a) || in_underflow_tail(c))) {
            ++R.skipped;
            continue;
        }
        ++R.n;
        if (!ff_eq(a, b, c, d)) {
            ++R.failures;
            if (samples_left > 0) {
                std::printf("    FAIL %-14s x=%.17g  got.hi=%.9g (0x%08x) got.lo=%.9g (0x%08x)  "
                            "want.hi=%.9g (0x%08x) want.lo=%.9g (0x%08x)\n",
                            Id::name, xs[i], a, fbits(a), b, fbits(b), c, fbits(c), d, fbits(d));
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
    std::vector<std::pair<float,float>> ps;
    ps.reserve(kDeviceN + 4096);
    {
        // add() does not Veltkamp-split, so it has no splitter-overflow limit;
        // the only out-of-domain case is a non-finite operand.
        std::mt19937_64 g(seed);
        std::uniform_real_distribution<double> d(-1e8, 1e8);
        for (int i = 0; i < kDeviceN; ++i) {
            double a = d(g), b = d(g);
            if (std::isfinite(a) && std::isfinite(b)) ps.emplace_back((float)a, (float)b);
            else ++R.skipped;
        }
        for (const auto& p : corpus::binary<float>(corpus_flags())) {
            if (std::isfinite(p.first) && std::isfinite(p.second)) ps.push_back(p);
            else ++R.skipped;
        }
    }
    const std::size_t n = ps.size();

    xpt::buffer<float> da(n), db(n), gh(n), gl(n), wh(n), wl(n);
    for (std::size_t i = 0; i < n; ++i) { da.host()[i] = ps[i].first; db.host()[i] = ps[i].second; }
    da.to_device(); db.to_device();
    xpt::parallel_for_n(n, BinaryIdKernel<Id>{da.device(), db.device(),
                                              gh.device(), gl.device(), wh.device(), wl.device()});
    gh.from_device(); gl.from_device(); wh.from_device(); wl.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = gh.host()[i], b = gl.host()[i], c = wh.host()[i], d = wl.host()[i];
        ++R.n;
        if (!ff_eq(a, b, c, d)) {
            ++R.failures;
            if (samples_left > 0) {
                std::printf("    FAIL %-14s a=%.9g b=%.9g  got.hi=%.9g (0x%08x) got.lo=%.9g (0x%08x)  "
                            "want.hi=%.9g (0x%08x) want.lo=%.9g (0x%08x)\n",
                            Id::name, ps[i].first, ps[i].second,
                            a, fbits(a), b, fbits(b), c, fbits(c), d, fbits(d));
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
    std::printf("=== ff_property_test_device (T2.3, C4 device half): Group A "
                "bit-exact identities for FF ===\n");
    std::printf("execution space: %s\n", xpt::where_name());
    std::printf("Group A only: pure sign/structure identities, no oracle and no "
                "tolerance. Group B, Group S and Test C stay in ff_property_test "
                "— see this file's header for why each one does.\n\n");

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

    // A run that checked nothing is not a pass: the domain filters and the
    // denormal-tail audit could in principle account for every input, and every
    // count above would then read 0 failures.
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
    std::printf("=== ff_property_test_device: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    return rc;
}
