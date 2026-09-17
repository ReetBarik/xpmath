// ============================================================================
// qf_property_test_device.cpp — the DEVICE half of T3.3's property tests for QF.
// ============================================================================
// Added by CORE_PLAN section C4 step 2 (chunk D), mirroring chunk C's
// dd_property_test_device.cpp / ff_property_test_device.cpp exactly.
//
// WHY THE SPLIT
//   tests/qf_property_test.cpp ran Group A (bit-exact identities, no oracle),
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
//   GREW. The old device pass ran three Group A identities: A1, A5, A9. All
//   TWELVE are here now — A1..A10 unary, the binary A11, and the A12 mul_pwr2
//   power-of-two round-trip. Every one is a pure sign/structure identity with no
//   oracle and no tolerance, so there was never a reason for the device side to
//   carry only three; the reason it did is that the mixed TU had a large host
//   pass and the device pass was a spot check bolted on. Each also now sweeps
//   the corner-case corpus (subnormals, +/-0, powers of two) in addition to 10^5
//   random inputs, which is what the host Group A runners always did and the
//   device pass never did.
//
//   LOST. The old device pass also ran two GROUP B checks — B1_sqrt_sq and
//   B4_pythag — scoring qf_digits against the binary128 oracle. Those cannot
//   come here and are NOT replaced, for the reasons chunk C wrote down for DD:
//
//     * They are not structural. "sqrt(a)^2 recovers a to N digits" is a
//       statement about a REFERENCE VALUE, and the only reference is the oracle.
//       There is no oracle-free identity in the neighbourhood: unlike qf_two_sum,
//       which has an independent exact transform to cross-check against, sqrt has
//       no second exact algorithm at this width. QF's sqrt is Heron on an FP32
//       seed and its sincos is a Taylor/argument-reduction pair; neither has an
//       exact companion to be checked against without a tolerance.
//
//     * Host/device bit-parity is NOT a substitute, and shipping it would be
//       worse than shipping nothing. A real GPU's sqrt and sincos may legitimately
//       differ from the host libm in the last bits; a bit-parity check would go
//       red on correct hardware, and the only way to make it green again would be
//       to add a tolerance — which is a SECOND SCORER, and docs/CORRECTNESS.md
//       allows exactly one measurement and one verdict per point.
//
//   So B1_sqrt_sq and B4_pythag are HOST-ONLY from C4 onward. The host half still
//   runs both over 2*10^5 inputs; what is genuinely gone is the ability to notice
//   a GPU whose sqrt or sincos is accurate on the host and inaccurate on the
//   device. The apparatus that DOES cover accuracy end to end is
//   validation/sweep/ (docs/CORRECTNESS.md), and it is host-measured for exactly
//   the same reason — the oracle cannot share a translation unit with device code.
//
// NOTHING HERE SCORES. Every check below is `==` on a pair of FP32 words.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include "corpus.hpp"
#include <xp/qf_math.hpp>

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

// test_utils_device.hpp declares `namespace dd = xp;` and tag structs for DD and
// FF, but has no QF tag (a TODO there). Rather than add a traits specialization
// — shared harness state other tasks own — this file declares the alias locally,
// the same posture qf_eft_test.cpp takes.
namespace qf = xp;

static constexpr int kDeviceN = 100'000;   // 10^5 random inputs per identity

// ----------------------------------------------------------------------------
// Bit-pattern helper (same hex format as qf_property_test.cpp / qf_eft_test.cpp).
// ----------------------------------------------------------------------------
static uint32_t fbits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(float));
    return b;
}

// ----------------------------------------------------------------------------
// Domain predicates and the denormal-tail guard, copied from the host half.
//
// split_safe_max: the Dekker (Veltkamp) split computes x*8193, which overflows
// to inf (-> NaN in the split) for |x| >= FLT_MAX/8193 ~= 2^114.9998, regardless
// of the other operand. Such inputs are OUT of multiply's domain and are
// SKIPPED, not failed.
//
// kUnderflowTail: the strict 4-word comparison can trip the FP32 round-to-even
// hole when a result's leading word falls into the subnormal tail. Below this
// magnitude a mismatch is counted SKIPPED, not FAILED (a domain limit of FP32,
// not a qf_math.hpp defect). 2^-100 matches T3.1/T3.2/T3.3.
// ----------------------------------------------------------------------------
static constexpr float kUnderflowTail = 0x1p-100f;

static float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

static bool dom_all(double x) { return std::isfinite(x); }
static bool dom_dekker(double x) {
    if (!std::isfinite(x)) return false;
    double ax = std::fabs(x);
    if (ax == 0.0) return true;                        // exact: a*{0,+-1} with a==0
    if (ax >= (double)split_safe_max()) return false;  // Dekker splitter overflow
    if (ax < (double)kUnderflowTail) return false;     // denormal-tail / subnormal
    return true;
}

// Corpus flags: zeros ON, inf OFF, nan OFF (subnormals default ON) — matches the
// host half and the invariant tests. inf/nan are excluded because e.g.
// inf + (-inf) = nan is not the zero identity.
static corpus::CorpusFlags corpus_flags() {
    corpus::CorpusFlags f;
    f.include_zero = true;
    f.include_inf  = false;
    f.include_nan  = false;
    return f;
}

// ----------------------------------------------------------------------------
// A QF value crossing the PCIe bus is four FP32 word arrays. Quad4 owns them,
// Quad4Ptr is the trivially-copyable device-side view a kernel captures.
// ----------------------------------------------------------------------------
struct Quad4Ptr {
    float* w0; float* w1; float* w2; float* w3;
    XPMATH_INLINE_FUNCTION void store(std::size_t i, const qf::QuadFloat& v) const {
        w0[i] = v.f0; w1[i] = v.f1; w2[i] = v.f2; w3[i] = v.f3;
    }
};

struct Quad4 {
    xpt::buffer<float> w0, w1, w2, w3;
    explicit Quad4(std::size_t n) : w0(n), w1(n), w2(n), w3(n) {}
    Quad4Ptr dev() { return Quad4Ptr{w0.device(), w1.device(), w2.device(), w3.device()}; }
    void from_device() { w0.from_device(); w1.from_device(); w2.from_device(); w3.from_device(); }
    // Not const: xpt::buffer::host() is a non-const accessor (the harness
    // hands out a writable staging pointer and has no const overload).
    qf::QuadFloat get(std::size_t i) {
        return qf::QuadFloat(w0.host()[i], w1.host()[i], w2.host()[i], w3.host()[i]);
    }
};

// Exact 4-word equality (value ==, so +0/-0 compare equal — intended, exactly as
// in the host half: the zero-result identities are specified as "the value zero",
// regardless of sign bit).
static bool qf_eq(const qf::QuadFloat& x, const qf::QuadFloat& y) {
    return x.f0 == y.f0 && x.f1 == y.f1 && x.f2 == y.f2 && x.f3 == y.f3;
}
static bool in_underflow_tail(const qf::QuadFloat& v) {
    return v.f0 != 0.0f && std::fabs((double)v.f0) < (double)kUnderflowTail;
}

// ----------------------------------------------------------------------------
// The identities, as tag structs with a device-callable apply().
//
// Each one computes BOTH sides on the device and hands back (got, want); the
// host only compares bits. That is deliberate: evaluating `want` host-side would
// silently turn several of these into host-vs-device comparisons, which is a
// different question. A9, for instance, is only a real test of qf::abs's sign
// BRANCH if the branch runs where the arithmetic does.
//
// Bodies are copied verbatim from the host half's Group A lambdas, so a
// divergence between the two is visible as a textual difference.
// ----------------------------------------------------------------------------

struct IdA1 {                                       // a + (-a) == 0
    static constexpr const char* name = "A1_add_neg";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::add(a, qf::negate(a)); want = qf::QuadFloat(0.0f);
    }
};
struct IdA2 {                                       // a - a == 0
    static constexpr const char* name = "A2_self_sub";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::subtract(a, a); want = qf::QuadFloat(0.0f);
    }
};
struct IdA3 {                                       // a + 0 == a
    static constexpr const char* name = "A3_add_zero";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::add(a, qf::QuadFloat(0.0f)); want = a;
    }
};
struct IdA4 {                                       // a - 0 == a
    static constexpr const char* name = "A4_sub_zero";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::subtract(a, qf::QuadFloat(0.0f)); want = a;
    }
};
struct IdA5 {                                       // a * 1 == a   (Dekker domain)
    static constexpr const char* name = "A5_mul_one";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::multiply(a, qf::QuadFloat(1.0f)); want = a;
    }
};
struct IdA6 {                                       // a * 0 == 0   (Dekker domain)
    static constexpr const char* name = "A6_mul_zero";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::multiply(a, qf::QuadFloat(0.0f)); want = qf::QuadFloat(0.0f);
    }
};
struct IdA7 {                                       // a * (-1) == -a  (Dekker domain)
    static constexpr const char* name = "A7_mul_negone";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::multiply(a, qf::QuadFloat(-1.0f)); want = qf::negate(a);
    }
};
struct IdA8 {                                       // -(-a) == a
    static constexpr const char* name = "A8_double_neg";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::negate(qf::negate(a)); want = a;
    }
};
struct IdA9 {                                       // |a| == (a.f0 >= 0 ? a : -a)
    static constexpr const char* name = "A9_abs_branch";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::abs(a); want = (a.f0 >= 0.0f) ? a : qf::negate(a);
    }
};
struct IdA10 {                                      // abs(-a) == abs(a)
    static constexpr const char* name = "A10_abs_neg";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::abs(qf::negate(a)); want = qf::abs(a);
    }
};
struct IdA11 {                                      // add(a,b) == add(b,a)
    static constexpr const char* name = "A11_add_comm";
    XPMATH_INLINE_FUNCTION static void apply(const qf::QuadFloat& a, const qf::QuadFloat& b,
                                             qf::QuadFloat& got, qf::QuadFloat& want) {
        got = qf::add(a, b); want = qf::add(b, a);
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
    Quad4Ptr got, want;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        qf::QuadFloat a(x[i]), g, w;
        Id::apply(a, g, w);
        got.store(i, g); want.store(i, w);
    }
};

template <class Id>
struct BinaryIdKernel {
    const double* a; const double* b;
    Quad4Ptr got, want;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        qf::QuadFloat x(a[i]), y(b[i]), g, w;
        Id::apply(x, y, g, w);
        got.store(i, g); want.store(i, w);
    }
};

// A12 needs its own kernel: the "identity" is a paired op with a per-input
// exponent, and the host has to see the INTERMEDIATE to apply the range
// exclusion below (scaling up by 2^k can overflow before the round-trip ever
// gets a chance to be wrong).
struct MulPwr2Kernel {
    const double* x; const float* up; const float* down;
    Quad4Ptr mid, got, want;
    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        qf::QuadFloat a(x[i]);
        qf::QuadFloat m = qf::mul_pwr2(a, up[i]);
        qf::QuadFloat g = qf::mul_pwr2(m, down[i]);
        mid.store(i, m); got.store(i, g); want.store(i, a);
    }
};

// ----------------------------------------------------------------------------
// Runners.
// ----------------------------------------------------------------------------
struct IdResult { long n = 0; long skipped = 0; long failures = 0; };

static void print_fail_unary(const char* id, double x, const qf::QuadFloat& got,
                             const qf::QuadFloat& want) {
    std::printf("    FAIL %-14s x=%.9g  got=[%.9g %.9g %.9g %.9g] "
                "(0x%08x 0x%08x 0x%08x 0x%08x)  want=[%.9g %.9g %.9g %.9g]\n",
                id, x, got.f0, got.f1, got.f2, got.f3,
                fbits(got.f0), fbits(got.f1), fbits(got.f2), fbits(got.f3),
                want.f0, want.f1, want.f2, want.f3);
}
static void print_fail_binary(const char* id, double a, double b,
                              const qf::QuadFloat& got, const qf::QuadFloat& want) {
    std::printf("    FAIL %-14s a=%.9g b=%.9g  got=[%.9g %.9g %.9g %.9g] "
                "(0x%08x 0x%08x 0x%08x 0x%08x)  want=[%.9g %.9g %.9g %.9g]\n",
                id, a, b, got.f0, got.f1, got.f2, got.f3,
                fbits(got.f0), fbits(got.f1), fbits(got.f2), fbits(got.f3),
                want.f0, want.f1, want.f2, want.f3);
}

static void report(const char* name, const IdResult& R) {
    std::printf("  [device] %-14s n=%-8ld skipped=%-4ld failures=%ld status=%s\n",
                name, R.n, R.skipped, R.failures, R.failures == 0 ? "PASS" : "FAIL");
}

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
    Quad4 got(n), want(n);
    for (std::size_t i = 0; i < n; ++i) dx.host()[i] = xs[i];
    dx.to_device();
    xpt::parallel_for_n(n, UnaryIdKernel<Id>{dx.device(), got.dev(), want.dev()});
    got.from_device(); want.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        qf::QuadFloat g = got.get(i), w = want.get(i);
        // Denormal-tail audit, same rule as the host half: a mismatch whose
        // leading word is in the FP32 subnormal tail is a domain limit
        // (round-to-even hole), SKIP not FAIL.
        if (!qf_eq(g, w) && (in_underflow_tail(g) || in_underflow_tail(w))) {
            ++R.skipped; continue;
        }
        ++R.n;
        if (!qf_eq(g, w)) {
            ++R.failures;
            if (samples_left > 0) { print_fail_unary(Id::name, xs[i], g, w); --samples_left; }
        }
    }
    report(Id::name, R);
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
        for (const auto& p : corpus::binary<float>(corpus_flags())) {
            if (std::isfinite(p.first) && std::isfinite(p.second))
                ps.emplace_back((double)p.first, (double)p.second);
            else ++R.skipped;
        }
    }
    const std::size_t n = ps.size();

    xpt::buffer<double> da(n), db(n);
    Quad4 got(n), want(n);
    for (std::size_t i = 0; i < n; ++i) { da.host()[i] = ps[i].first; db.host()[i] = ps[i].second; }
    da.to_device(); db.to_device();
    xpt::parallel_for_n(n, BinaryIdKernel<Id>{da.device(), db.device(), got.dev(), want.dev()});
    got.from_device(); want.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        qf::QuadFloat g = got.get(i), w = want.get(i);
        ++R.n;
        if (!qf_eq(g, w)) {
            ++R.failures;
            if (samples_left > 0) {
                print_fail_binary(Id::name, ps[i].first, ps[i].second, g, w);
                --samples_left;
            }
        }
    }
    report(Id::name, R);
    return R;
}

// A12: mul_pwr2(mul_pwr2(a, +-2^k), +-2^-k) == a, bit-exact. mul_pwr2 scales each
// component by an EXACT power of two (no rounding, no renorm), so the round-trip
// must return the input to the bit. The exponent is random in [-40,40] for the
// random pass and fixed at k=7 for the corpus pass, matching the host runner.
static IdResult run_mulpwr2(uint64_t seed) {
    IdResult R;
    std::vector<double> xs; std::vector<float> ups, downs;
    xs.reserve(kDeviceN + 128); ups.reserve(kDeviceN + 128); downs.reserve(kDeviceN + 128);
    {
        std::mt19937_64 g(seed);
        std::uniform_real_distribution<double> d(-1e8, 1e8);
        std::uniform_int_distribution<int> dk(-40, 40);
        std::uniform_int_distribution<int> dsgn(0, 1);
        auto push = [&](double x, int k, int sgn) {
            if (!std::isfinite(x)) { ++R.skipped; return; }
            xs.push_back(x);
            ups.push_back(std::ldexp(1.0f,  k) * (sgn ? 1.0f : -1.0f));
            downs.push_back(std::ldexp(1.0f, -k) * (sgn ? 1.0f : -1.0f));
        };
        for (int i = 0; i < kDeviceN; ++i) push(d(g), dk(g), dsgn(g));
        for (float x : corpus::unary<float>(corpus_flags())) push((double)x, 7, 1);
    }
    const std::size_t n = xs.size();

    xpt::buffer<double> dx(n);
    xpt::buffer<float> dup(n), ddn(n);
    Quad4 mid(n), got(n), want(n);
    for (std::size_t i = 0; i < n; ++i) {
        dx.host()[i] = xs[i]; dup.host()[i] = ups[i]; ddn.host()[i] = downs[i];
    }
    dx.to_device(); dup.to_device(); ddn.to_device();
    xpt::parallel_for_n(n, MulPwr2Kernel{dx.device(), dup.device(), ddn.device(),
                                         mid.dev(), got.dev(), want.dev()});
    mid.from_device(); got.from_device(); want.from_device();

    int samples_left = 3;
    for (std::size_t i = 0; i < n; ++i) {
        qf::QuadFloat m = mid.get(i), g = got.get(i), w = want.get(i);
        // Overflow/underflow domain exclusion: the round-trip scales UP by 2^k
        // first; for |x| near FLT_MAX that intermediate overflows to inf (the
        // corpus's FLT_MAX entry with a positive k), and for a tiny |x| with k<0
        // it can land in the denormal tail. Either is an FP32 RANGE limit, not a
        // mul_pwr2 defect, so SKIP.
        if (!std::isfinite(m.f0) || in_underflow_tail(m)) { ++R.skipped; continue; }
        if (!qf_eq(g, w) && (in_underflow_tail(g) || in_underflow_tail(w))) {
            ++R.skipped; continue;
        }
        ++R.n;
        if (!qf_eq(g, w)) {
            ++R.failures;
            if (samples_left > 0) { print_fail_unary("A12_mulpwr2_rt", xs[i], g, w); --samples_left; }
        }
    }
    report("A12_mulpwr2_rt", R);
    return R;
}

// ============================================================================
int main(int, char**) {
    std::printf("=== qf_property_test_device (T3.3, C4 device half): Group A "
                "bit-exact identities for QF ===\n");
    std::printf("execution space: %s\n", xpt::where_name());
    std::printf("Group A only: pure sign/structure identities, no oracle and no "
                "tolerance. Group B and Test C stay in qf_property_test (they need "
                "the 128-bit oracle) — see this file's header for what that costs.\n\n");

    std::printf("[Group A, device] 10^5 random in [-1e8, 1e8] + the finite corpus, "
                "per identity\n");

    long failures = 0, total_n = 0;
    auto tally = [&](IdResult r) { failures += r.failures; total_n += r.n; };

    tally(run_unary<IdA1>(700001ULL));
    tally(run_unary<IdA2>(700011ULL));
    tally(run_unary<IdA3>(700012ULL));
    tally(run_unary<IdA4>(700013ULL));
    tally(run_unary<IdA5>(700002ULL, dom_dekker));
    tally(run_unary<IdA6>(700014ULL, dom_dekker));
    tally(run_unary<IdA7>(700015ULL, dom_dekker));
    tally(run_unary<IdA8>(700016ULL));
    tally(run_unary<IdA9>(700003ULL));
    tally(run_unary<IdA10>(700017ULL));
    tally(run_binary<IdA11>(700018ULL));
    tally(run_mulpwr2(700019ULL));

    std::printf("\n=== Summary ===\n");
    std::printf("  12 identities, %ld total checks, failures=%ld\n", total_n, failures);

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
    std::printf("=== qf_property_test_device: %s ===\n",
                rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    return rc;
}
