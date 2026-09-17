// ============================================================================
// qf_eft_test_device.cpp — the DEVICE half of T3.1's QF EFT unit tests.
// ============================================================================
//
// WHY THE SPLIT
// -------------
// CORE_PLAN section C4 names eight MIXED translation units — files carrying both
// a host oracle and a device launch. tests/qf_eft_test.cpp was a NINTH, and C4
// step 3 said so out loud rather than quietly: chunk B migrated its launch off
// Kokkos onto tests/device_harness.hpp but left it OUT of the FILES array in
// scripts/check_device_tu_purity.sh, with the reason written into both that array
// and the test's own header. The reason was Test C's `test_renorm_4_wide`, which
// builds a ~113-bit-source value in binary128 to probe renorm_4's truncation
// threshold. That check needs __float128, __float128 cannot survive nvcc's device
// pass (the S6 blocker), and so the whole TU could not claim device purity no
// matter how clean its kernel was.
//
// Chunk D splits it exactly like the other three. qf_eft_test.cpp keeps the
// target name, keeps tests/test_utils_host.hpp, keeps Tests A–D including the
// binary128 wide-spread check, and no longer launches anything. This file carries
// Test E, joins the FILES array, and is now a device TU somebody is checking.
//
// WHAT MOVED, AND WHAT DID NOT
// ----------------------------
// MOVED, unchanged: EftKernel and run_device_parity, the seeds (99999 for the
// twoSum/twoProd operands, 88888 for the renorm_4 expansions), nd = 200'000, the
// [-1e18,1e18] draw range and the reason for it, and every comparison the parity
// loop makes. The oracle those comparisons use is plain FP64 and was ALREADY
// device-pure: the exact sum of two FP32 values needs 25 bits and the exact
// product 48, both inside FP64's 53-bit mantissa, so `(double)hi + (double)lo ==
// (double)a + (double)b` is a provable bit-equality, not a tolerance. Nothing
// here scores and nothing here needed weakening to cross the boundary.
//
// COPIED, because they are header-free helpers in a .cpp: sum_in_domain,
// split_safe_max, prod_in_domain, fbits, print_fail_sum, print_fail_prod,
// half_ulp, kUnderflowTail, pair_checkable, nonoverlap_holds, draw_ordered_double
// and NamedResult. The host half keeps its own copies. Copying rather than
// cross-including a .cpp is the convention the whole QF test family already uses
// (qf_nonoverlap_test copies the same non-overlap machinery out of this file's
// host half); a shared header for them would be a fifth place to look.
//
// NOT MOVED: Tests A–D. Test C in particular stays host-side by nature — the
// binary128 truncation check is the single thing that made this TU mixed, so
// carrying it here would reproduce the exact defect the split exists to remove.
// Test A/B/D are host-only batch and named-case work with no kernel in them.
//
// LOSES NO COVERAGE. Test E ran once before the split and runs once after, on the
// same inputs, in the same order, against the same oracle.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include <xp/qf_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace kokkos_ep;

// qf:: alias over the standalone core, matching the host half. tests/
// test_utils_device.hpp declares dd:: and ff:: but has no QF tag struct yet
// (its Phase-3 TODO), so this TU declares its own.
namespace qf = xp;

// ----------------------------------------------------------------------------
// Domain predicates — the "no overflow / no underflow" region each EFT is proven
// over. A pair outside the domain is not a failure; it is simply out of scope for
// a bit-exactness claim, so it is skipped. Copied verbatim from the host half.
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
// `cona - (cona - a)` becomes inf - inf = NaN and the error term is poisoned.
inline float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}

// twoProd/twoSqr domain: finite normal-or-zero operands with the TRUE product
// comfortably inside the normal range. Evaluated on the EXACT (double) product,
// not the rounded FP32 one (which would flush to 0 in the underflow regime and
// let underflowing pairs masquerade as in-domain). Threshold FLT_MIN * 2^24 =
// 2^-102 keeps the error term e >= ~2^-126 (normal).
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
// Failure-sample printers (first few only), with input bit patterns.
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
// The Priest length-4 non-overlap invariant, used on renorm_4's device output.
// Oracle-independent: it compares a float against a power of two derived from
// another float's binade. Copied verbatim from the host half.
// ----------------------------------------------------------------------------

// Mathematical 1/2 ulp of a normal float, from its binade exponent. frexp writes
// x = m * 2^e with m in [0.5, 1); for FP32's 24-bit significand ulp(x) = 2^(e-24),
// so 1/2 ulp = 2^(e-25).
inline double half_ulp(float x) {
    if (x == 0.0f) return 0.0;
    int e;
    std::frexp((double)x, &e);
    return std::ldexp(1.0, e - 25);
}

// UNDERFLOW-TAIL gate: the 1/2 ulp bound is only WELL-POSED when 1/2 ulp(f_i) is
// itself a normal float, i.e. |f_i| >= 2^-102; below that, subnormal quantization
// makes the comparison ill-posed. Gated a hair higher at 2^-100 for margin.
static constexpr float kUnderflowTail = 0x1p-100f;
inline bool pair_checkable(float hi) {
    if (!std::isfinite(hi)) return false;
    if (hi == 0.0f) return false;                                  // trailing zero
    if (std::fabs(hi) < std::numeric_limits<float>::min()) return false;  // subnormal
    if (std::fabs(hi) < kUnderflowTail) return false;              // underflow tail
    return true;
}

// Check |f_{i+1}| <= 1/2 ulp(f_i) on (b0,b1,b2,b3), plus renorm's packing
// guarantee: once a word is zero, all lower words must be zero too.
inline bool nonoverlap_holds(float b0, float b1, float b2, float b3, int* skips) {
    const float b[4] = {b0, b1, b2, b3};
    bool seen_zero = false;
    for (int i = 0; i < 4; ++i) {
        if (b[i] == 0.0f) seen_zero = true;
        else if (seen_zero) return false;   // nonzero after zero -> not packed
    }
    for (int i = 0; i < 3; ++i) {
        if (b[i] == 0.0f) break;            // trailing zeros: invariant trivially holds
        if (!pair_checkable(b[i])) { if (skips) ++*skips; continue; }
        if (std::fabs((double)b[i + 1]) > half_ulp(b[i])) return false;
    }
    return true;
}

// Draw a properly-ORDERED unnormalized length-5 expansion by successive FP32
// decomposition of a random 53-bit double (the same construction add/multiply/
// QuadFloat(double) produce — magnitude-decreasing words). renorm's quick_two_sum
// cascade ASSUMES this ordering. A 53-bit value decomposes into <= 3 nonzero
// ordered FP32 words, so the exact real sum of the words == the original double
// EXACTLY and computing that sum in FP64 is itself exact — a PROVABLE FP64
// value-preservation oracle that needs nothing wider than a machine word.
inline double draw_ordered_double(std::mt19937_64& g, float out[5]) {
    std::uniform_int_distribution<int>     de(-40, 40);      // exponent (sum stays finite)
    std::uniform_real_distribution<double> dm(-1.0, 1.0);
    double x = dm(g) * std::ldexp(1.0, de(g));
    double r = x;
    for (int k = 0; k < 5; ++k) { out[k] = (float)r; r -= (double)out[k]; }
    return x;   // == (double)(out[0]+..+out[4]) exactly (ordered, <=3 nonzero words)
}

struct NamedResult { int passed = 0; int skipped = 0; int failed = 0; int total = 0; };

// ----------------------------------------------------------------------------
// Test E — device parity. Run the SAME shipped primitives inside
// xpt::parallel_for_n, copy results back, and compare bit-exactly against the
// host FP64 oracle. With the harness's host backend this reduces to a host loop
// over two real allocations (still valid); under hipcc/nvcc it is a real kernel
// and catches device-side FP differences (subnormal flush,
// contraction) the host pass cannot see. Inputs drawn from the splitter- and
// underflow-safe range [-1e18,1e18] so BOTH sum stays finite AND product stays in
// FP32's normal range (using twoSum's wider 1e30 would domain-skip nearly every
// product — the vacuous-coverage trap FP32's narrow exponent range sets).
// ----------------------------------------------------------------------------
// Trivially-copyable kernel struct, not a lambda: the harness passes F by value
// into a __global__ and does not require nvcc --extended-lambda of its callers.
struct EftKernel {
    const float* a;
    const float* b;
    float* s_hi; float* s_lo;
    float* p_hi; float* p_lo;
    const float* c0; const float* c1; const float* c2; const float* c3; const float* c4;
    float* rb0; float* rb1; float* rb2; float* rb3;

    XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
        float es, ep;
        s_hi[i] = qf::qf_two_sum(a[i], b[i], es);  s_lo[i] = es;
        p_hi[i] = qf::qf_two_prod(a[i], b[i], ep); p_lo[i] = ep;
        float b0 = c0[i], b1 = c1[i], b2 = c2[i], b3 = c3[i], b4 = c4[i];
        qf::renorm_4(b0, b1, b2, b3, b4);
        rb0[i] = b0; rb1[i] = b1; rb2[i] = b2; rb3[i] = b3;
    }
};

static NamedResult run_device_parity() {
    NamedResult R;
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

    xpt::buffer<float> va(nd), vb(nd);
    xpt::buffer<float> s_hi(nd), s_lo(nd), p_hi(nd), p_lo(nd);
    xpt::buffer<float> rc0(nd), rc1(nd), rc2(nd), rc3(nd), rc4(nd);
    xpt::buffer<float> rb0(nd), rb1(nd), rb2(nd), rb3(nd);
    for (int i = 0; i < nd; ++i) {
        va.host()[i] = ha[i]; vb.host()[i] = hb[i];
        rc0.host()[i] = c0[i]; rc1.host()[i] = c1[i]; rc2.host()[i] = c2[i];
        rc3.host()[i] = c3[i]; rc4.host()[i] = c4[i];
    }
    va.to_device();  vb.to_device();
    rc0.to_device(); rc1.to_device(); rc2.to_device();
    rc3.to_device(); rc4.to_device();

    xpt::parallel_for_n(static_cast<std::size_t>(nd),
        EftKernel{va.device(), vb.device(),
                  s_hi.device(), s_lo.device(), p_hi.device(), p_lo.device(),
                  rc0.device(), rc1.device(), rc2.device(), rc3.device(), rc4.device(),
                  rb0.device(), rb1.device(), rb2.device(), rb3.device()});

    s_hi.from_device(); s_lo.from_device();
    p_hi.from_device(); p_lo.from_device();
    rb0.from_device(); rb1.from_device(); rb2.from_device(); rb3.from_device();

    const float* hshi = s_hi.host();
    const float* hslo = s_lo.host();
    const float* hphi = p_hi.host();
    const float* hplo = p_lo.host();
    const float* hb0 = rb0.host();
    const float* hb1 = rb1.host();
    const float* hb2 = rb2.host();
    const float* hb3 = rb3.host();

    long sum_fail = 0, prod_fail = 0, ren_fail = 0, sum_skip = 0, prod_skip = 0, ren_over = 0;
    int samples_left = 5;
    for (int i = 0; i < nd; ++i) {
        float a = ha[i], b = hb[i];
        if (sum_in_domain(a, b)) {
            if ((double)hshi[i] + (double)hslo[i] != (double)a + (double)b) {
                ++sum_fail; if (samples_left > 0) { print_fail_sum("twoSum", a, b); --samples_left; }
            }
        } else ++sum_skip;
        if (prod_in_domain(a, b)) {
            if ((double)hphi[i] + (double)hplo[i] != (double)a * (double)b) {
                ++prod_fail; if (samples_left > 0) { print_fail_prod("twoProd", a, b); --samples_left; }
            }
        } else ++prod_skip;
        // renorm_4 parity: exact FP64 value-preservation (out sum == x) + non-overlap.
        double out_sum = (double)hb0[i] + hb1[i] + hb2[i] + hb3[i];
        int dummy = 0;
        bool value_ok   = (out_sum == cx[i]);
        bool overlap_ok = nonoverlap_holds(hb0[i], hb1[i], hb2[i], hb3[i], &dummy);
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
int main(int, char**) {
    int rc = 0;
    {
        std::printf("=== qf_eft_test_device (T3.1 Test E, C4 device half): QF EFT parity "
                    "for qf_two_sum / qf_two_prod / renorm_4 ===\n");
        std::printf("Oracle: FP64 (exact for twoSum/twoProd and ordered-53-bit-source "
                    "renorm value-preservation; needs no type wider than a machine word)\n");
        std::printf("Tests A-D (host batches, named cases, the wide-spread renorm "
                    "truncation check) stayed in qf_eft_test.\n\n");

        std::printf("[Test E] device parity (%s)\n", xpt::where_name());
        NamedResult E = run_device_parity();
        std::printf("  Test E device parity: %d passed, %d skipped, %d failed (of %d)\n\n",
                    E.passed, E.skipped, E.failed, E.total);
        KOKKOS_EP_ASSERT(E.failed == 0, "device EFT parity mismatch vs host FP64 oracle");
        // A nonzero vendor code is a TEST FAILURE, not a warning: a launch that
        // never ran leaves the output buffers at whatever the allocator returned,
        // and the comparison loop cannot tell that from a clean pass. Sticky
        // since process start, so a later good call cannot erase it.
        KOKKOS_EP_ASSERT(xpt::last_error() == 0,
                         "device harness reported a nonzero vendor error code");

        rc = ep_exit_code();
        std::printf("=== qf_eft_test_device: %s ===\n", rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    }
    return rc;
}
