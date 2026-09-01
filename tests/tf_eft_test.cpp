// ============================================================================
// tf_eft_test.cpp — Layer 1 (EFT unit tests) for the TF backend (S10 Phase 2).
// ============================================================================
//
// TF (TripleFloat, 3×FP32) analogue of qf_eft_test.cpp (T3.1) and ff_eft_test.cpp
// (T2.1). Tests error-free transforms (twoSum / twoProd / twoSqr) and renormalization
// (renorm, renorm_3) in isolation before any higher-level ops are exercised. TF reuses
// FF's already-validated FP32 EFT primitives (tf_two_sum, tf_quick_two_sum, tf_two_prod,
// tf_two_sqr — bit-identical to FF's), so the structure mirrors qf_eft_test exactly.
//
// ORACLE: FP64 (provable for twoSum/twoProd/twoSqr: 25-bit sum / 48-bit product fit
// in FP64's 53-bit mantissa). renorm value-preservation uses ordered decomposition
// of 53-bit doubles so output sum == input exactly in FP64, plus a __float128 wide-
// spread check under KOKKOS_EP_HAVE_QUADMATH.
//
// TEST STRUCTURE
//   Test A — tf_two_sum + tf_quick_two_sum bit-exactness (FP64 oracle)
//   Test B — tf_two_prod + tf_two_sqr bit-exactness (FP64 oracle, splitter 8193.0f)
//   Test C — renorm_3 (len 4->3) + renorm (len 3->3): Priest non-overlap invariant
//            plus exact FP64 value-preservation on ordered 53-bit-source input
//   Test D — named hard cases (zero, cancellation, subnormals, inf/nan)
//   Test E — device parity (same primitives in parallel_for)
// ============================================================================

#include "test_utils.hpp"
#include "corpus.hpp"
#include <xp/tf_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace kokkos_ep;

// ----------------------------------------------------------------------------
// Oracle comparisons (FP64, provably exact). Calls the SHIPPED primitives.
// ----------------------------------------------------------------------------
inline bool sum_is_exact(float a, float b) {
    float e;
    float s = xp::tf_two_sum(a, b, e);
    return (double)s + (double)e == (double)a + (double)b;
}
inline bool quick_sum_is_exact(float a, float b) {
    float e;
    float s = xp::tf_quick_two_sum(a, b, e);
    return (double)s + (double)e == (double)a + (double)b;
}
inline bool prod_is_exact(float a, float b) {
    float e;
    float p = xp::tf_two_prod(a, b, e);
    return (double)p + (double)e == (double)a * (double)b;
}
inline bool sqr_is_exact(float a) {
    float e;
    float q = xp::tf_two_sqr(a, e);
    return (double)q + (double)e == (double)a * (double)a;
}

// ----------------------------------------------------------------------------
// Skip predicates (domain = no overflow/underflow for provable bit-exactness).
// ----------------------------------------------------------------------------
inline bool sum_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return std::isfinite(a + b);
}
inline float split_safe_max() {
    return std::numeric_limits<float>::max() / 8193.0f;   // ~2^114.9998
}
inline bool prod_in_domain(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float fmin = std::numeric_limits<float>::min();
    auto normal_or_zero = [fmin](float x) {
        return x == 0.0f || std::fabs(x) >= fmin;
    };
    if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
    const float ssm = split_safe_max();
    if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;
    if (a == 0.0f || b == 0.0f) return true;
    double tp = (double)a * (double)b;
    double mag = tp < 0.0 ? -tp : tp;
    const double hi_lim = (double)std::numeric_limits<float>::max();
    const double lo_lim = std::ldexp(1.0, -102);
    if (mag > hi_lim) return false;
    if (mag < lo_lim) return false;
    return true;
}

// ----------------------------------------------------------------------------
// Failure-sample printers
// ----------------------------------------------------------------------------
inline uint32_t fbits(float x) { uint32_t b; std::memcpy(&b, &x, sizeof(float)); return b; }
inline void print_fail_sum(const char* which, float a, float b) {
    float e; float s = xp::tf_two_sum(a, b, e);
    std::printf("    FAIL %s  a=%.9g (0x%08x)  b=%.9g (0x%08x)  s=%.9g e=%.9g\n",
                which, (double)a, fbits(a), (double)b, fbits(b), (double)s, (double)e);
}
inline void print_fail_prod(const char* which, float a, float b) {
    float e; float p = xp::tf_two_prod(a, b, e);
    std::printf("    FAIL %s  a=%.9g (0x%08x)  b=%.9g (0x%08x)  p=%.9g e=%.9g\n",
                which, (double)a, fbits(a), (double)b, fbits(b), (double)p, (double)e);
}

// ----------------------------------------------------------------------------
// Accumulator for one batch
// ----------------------------------------------------------------------------
struct EftCount { long tested = 0; long skipped = 0; long failures = 0; };
enum class Op { Sum, QuickSum, Prod, Sqr };

inline void check_pair(Op op, float a, float b, EftCount& c, int& samples_left) {
    if (op == Op::QuickSum && std::fabs(b) > std::fabs(a)) { float t = a; a = b; b = t; }
    bool in_domain;
    switch (op) {
        case Op::Sum:      in_domain = sum_in_domain(a, b);  break;
        case Op::QuickSum: in_domain = sum_in_domain(a, b);  break;
        case Op::Prod:     in_domain = prod_in_domain(a, b); break;
        case Op::Sqr:      in_domain = prod_in_domain(a, a); break;
    }
    if (!in_domain) { ++c.skipped; return; }
    ++c.tested;
    bool ok;
    switch (op) {
        case Op::Sum:      ok = sum_is_exact(a, b);       break;
        case Op::QuickSum: ok = quick_sum_is_exact(a, b); break;
        case Op::Prod:     ok = prod_is_exact(a, b);      break;
        case Op::Sqr:      ok = sqr_is_exact(a);          break;
    }
    if (!ok) {
        ++c.failures;
        if (samples_left > 0) {
            if (op == Op::Prod || op == Op::Sqr) print_fail_prod("twoProd/sqr", a, b);
            else                                 print_fail_sum("twoSum", a, b);
            --samples_left;
        }
    }
}

inline float broad_bound(Op op) { return (op == Op::Sum || op == Op::QuickSum) ? 1e30f : 1e18f; }

static EftCount run_host_batches(Op op, const char* op_label) {
    EftCount total;
    int samples_left = 5;
    const float R = broad_bound(op);
    auto run_uniform = [&](const char* tag, int n, float lo, float hi, uint64_t seed) {
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<float> d(lo, hi);
        EftCount c;
        for (int i = 0; i < n; ++i) {
            float a = d(gen), b = d(gen);
            check_pair(op, a, b, c, samples_left);
        }
        std::printf("    [%s] %s: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, tag, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    };
    char tag[64];
    std::snprintf(tag, sizeof(tag), "uniform[-%.0e,%.0e] n=1e6", (double)R, (double)R);
    run_uniform(tag, 1'000'000, -R, R, 12345ULL);
    run_uniform("uniform[-1,1] n=1e6", 1'000'000, -1.0f, 1.0f, 23456ULL);
    {
        std::mt19937_64 gen(34567ULL);
        std::uniform_real_distribution<float> da(-R, R);
        std::uniform_int_distribution<int>    dk(1, 20);
        EftCount c;
        for (int i = 0; i < 100'000; ++i) {
            float a = da(gen);
            int   k = dk(gen);
            float b = a * std::ldexp(1.0f, -k);
            check_pair(op, a, b, c, samples_left);
        }
        std::printf("    [%s] |a|>>|b| n=1e5: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    }
    {
        corpus::CorpusFlags flags;
        std::vector<float> xs = corpus::unary<float>(flags);
        const size_t N = xs.size();
        const size_t kMaxPairs = 250'000;
        EftCount c;
        size_t made = 0;
        for (size_t i = 0; i < N && made < kMaxPairs; ++i) {
            for (size_t j = i + 1; j < N && made < kMaxPairs; ++j) {
                check_pair(op, xs[i], xs[j], c, samples_left);
                ++made;
            }
        }
        std::printf("    [%s] corpus x-product pairs=%zu: tested=%ld skipped=%ld failures=%ld\n",
                    op_label, made, c.tested, c.skipped, c.failures);
        total.tested += c.tested; total.skipped += c.skipped; total.failures += c.failures;
    }
    return total;
}

// ============================================================================
// renorm test (Test C) — Priest non-overlap + value-preservation
// ============================================================================
inline double half_ulp(float x) {
    if (x == 0.0f) return 0.0;
    int e;
    std::frexp((double)x, &e);
    return std::ldexp(1.0, e - 25);
}
static constexpr float kUnderflowTail = 0x1p-100f;
inline bool pair_checkable(float hi) {
    if (!std::isfinite(hi)) return false;
    if (hi == 0.0f) return false;
    if (std::fabs(hi) < std::numeric_limits<float>::min()) return false;
    if (std::fabs(hi) < kUnderflowTail) return false;
    return true;
}
inline bool nonoverlap_holds(float b0, float b1, float b2, int* skips) {
    const float b[3] = {b0, b1, b2};
    bool seen_zero = false;
    for (int i = 0; i < 3; ++i) {
        if (b[i] == 0.0f) seen_zero = true;
        else if (seen_zero) return false;
    }
    for (int i = 0; i < 2; ++i) {
        if (b[i] == 0.0f) break;
        if (!pair_checkable(b[i])) { if (skips) ++*skips; continue; }
        if (std::fabs((double)b[i + 1]) > half_ulp(b[i])) return false;
    }
    return true;
}

inline double draw_ordered_double(std::mt19937_64& g, float out[4]) {
    std::uniform_int_distribution<int>     de(-40, 40);
    std::uniform_real_distribution<double> dm(-1.0, 1.0);
    double x = dm(g) * std::ldexp(1.0, de(g));
    double r = x;
    for (int k = 0; k < 4; ++k) { out[k] = (float)r; r -= (double)out[k]; }
    return x;
}

struct RenormResult { long tested = 0; long skips = 0; long overlap_fail = 0; long value_fail = 0; };

static RenormResult test_renorm_3_bounded(int n, uint64_t seed) {
    RenormResult R;
    std::mt19937_64 g(seed);
    int samples_left = 5;
    for (int i = 0; i < n; ++i) {
        float c[4];
        double x = draw_ordered_double(g, c);
        float b0 = c[0], b1 = c[1], b2 = c[2], b3 = c[3];
        xp::renorm_3(b0, b1, b2, b3);
        ++R.tested;
        int local_skips = 0;
        if (!nonoverlap_holds(b0, b1, b2, &local_skips)) {
            ++R.overlap_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm_3 non-overlap: in=[%.9g %.9g %.9g %.9g] out=[%.9g %.9g %.9g]\n",
                            (double)c[0], (double)c[1], (double)c[2], (double)c[3],
                            (double)b0, (double)b1, (double)b2);
                --samples_left;
            }
        }
        R.skips += local_skips;
        double out_sum = (double)b0 + b1 + b2;
        if (out_sum != x) {
            ++R.value_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm_3 value: x=%.17g out_sum=%.17g\n", x, out_sum);
                --samples_left;
            }
        }
    }
    return R;
}

static RenormResult test_renorm_bounded(int n, uint64_t seed) {
    RenormResult R;
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<int>     de(-40, 40);
    std::uniform_real_distribution<double> dm(-1.0, 1.0);
    int samples_left = 5;
    for (int i = 0; i < n; ++i) {
        double x = dm(g) * std::ldexp(1.0, de(g));
        double r = x; float c[3];
        for (int k = 0; k < 3; ++k) { c[k] = (float)r; r -= (double)c[k]; }
        float b0 = c[0], b1 = c[1], b2 = c[2];
        xp::renorm(b0, b1, b2);
        ++R.tested;
        int local_skips = 0;
        if (!nonoverlap_holds(b0, b1, b2, &local_skips)) {
            ++R.overlap_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm non-overlap: in=[%.9g %.9g %.9g] out=[%.9g %.9g %.9g]\n",
                            (double)c[0], (double)c[1], (double)c[2], (double)b0, (double)b1, (double)b2);
                --samples_left;
            }
        }
        R.skips += local_skips;
        double out_sum = (double)b0 + b1 + b2;
        if (out_sum != x) {
            ++R.value_fail;
            if (samples_left > 0) {
                std::printf("    FAIL renorm value: x=%.17g out_sum=%.17g\n", x, out_sum);
                --samples_left;
            }
        }
    }
    return R;
}

// ----------------------------------------------------------------------------
// Test D — named cases
// ----------------------------------------------------------------------------
struct NamedResult { int passed = 0; int skipped = 0; int failed = 0; int total = 0; };

static NamedResult run_named_cases() {
    NamedResult R;
    auto case_sum = [&](const char* name, float a, float b) {
        ++R.total;
        if (!sum_in_domain(a, b)) { ++R.skipped;
            std::printf("    tf_two_sum  %-28s : SKIP (out of domain)\n", name); return; }
        bool ok = sum_is_exact(a, b);
        std::printf("    tf_two_sum  %-28s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else { ++R.failed; print_fail_sum("tf_two_sum", a, b); }
    };
    auto case_prod = [&](const char* name, float a, float b) {
        ++R.total;
        if (!prod_in_domain(a, b)) { ++R.skipped;
            std::printf("    tf_two_prod %-28s : SKIP (out of domain)\n", name); return; }
        bool ok = prod_is_exact(a, b);
        std::printf("    tf_two_prod %-28s : %s\n", name, ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else { ++R.failed; print_fail_prod("tf_two_prod", a, b); }
    };
    case_sum("a == b (=1.0f)",          1.0f,  1.0f);
    case_sum("a == -b (=1.0f,-1.0f)",   1.0f, -1.0f);
    case_sum("a == b (=pi_f)",          3.14159265f, 3.14159265f);
    {
        float s  = std::numeric_limits<float>::denorm_min();
        float s3 = std::ldexp(s, 5);
        case_sum ("both subnormal", s, s3);
        case_prod("both subnormal", s, s3);
    }
    case_sum("a=0, b=nonzero",  0.0f, 3.5f);
    case_sum("a=nonzero, b=0",  3.5f, 0.0f);
    case_prod("a=0, b=nonzero", 0.0f, 3.5f);
    case_prod("a=nonzero, b=0", 3.5f, 0.0f);
    case_sum("a=+0, b=-0", 0.0f, -0.0f);
    case_sum("a=-0, b=+0", -0.0f, 0.0f);
    {
        float a = 1.0f, b = std::ldexp(1.0f, -24);
        float e; float s = xp::tf_two_sum(a, b, e);
        bool ok = sum_is_exact(a, b) && (s == 1.0f) && (e == std::ldexp(1.0f, -24));
        ++R.total;
        std::printf("    tf_two_sum  %-28s : %s  (s=%.9g e=%.9g)\n",
                    "Bailey 1.0f + 2^-24", ok ? "PASS" : "FAIL", (double)s, (double)e);
        if (ok) ++R.passed; else ++R.failed;
    }
    {
        float a  = 1.0f;
        float ap = std::nextafter(a, 2.0f);
        case_sum("a, a+ulp",  a, ap - a);
        case_sum("a, -(a)",   a, -a);
    }
    case_prod("pi_f * pi_f",       3.14159265f, 3.14159265f);
    case_prod("e_f * e_f",         2.71828183f, 2.71828183f);
    case_prod("sqrt2_f * sqrt2_f", 1.41421356f, 1.41421356f);
    {
        auto case_sqr = [&](const char* name, float a) {
            ++R.total;
            if (!prod_in_domain(a, a)) { ++R.skipped;
                std::printf("    tf_two_sqr  %-28s : SKIP\n", name); return; }
            bool ok = sqr_is_exact(a);
            std::printf("    tf_two_sqr  %-28s : %s\n", name, ok ? "PASS" : "FAIL");
            if (ok) ++R.passed; else { ++R.failed; print_fail_prod("tf_two_sqr", a, a); }
        };
        case_sqr("sqr(pi_f)",   3.14159265f);
        case_sqr("sqr(e_f)",    2.71828183f);
        case_sqr("sqr(sqrt2_f)",1.41421356f);
    }
    {
        float b0 = std::numeric_limits<float>::infinity(), b1 = 1.0f, b2 = 0.0f, b3 = 0.0f;
        xp::renorm_3(b0, b1, b2, b3);
        bool ok = std::isinf(b0);
        ++R.total;
        std::printf("    renorm_3    %-28s : %s\n", "inf leading word", ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else ++R.failed;
    }
    {
        float b0 = std::numeric_limits<float>::quiet_NaN(), b1 = 0.0f, b2 = 0.0f;
        xp::renorm(b0, b1, b2);
        bool ok = std::isnan(b0);
        ++R.total;
        std::printf("    renorm      %-28s : %s\n", "nan leading word", ok ? "PASS" : "FAIL");
        if (ok) ++R.passed; else ++R.failed;
    }
    return R;
}

// ----------------------------------------------------------------------------
// Test E — device parity
// ----------------------------------------------------------------------------
static NamedResult run_device_parity() {
    NamedResult R;
    using exec_space = Kokkos::DefaultExecutionSpace;
    const int nd = 200'000;
    std::vector<float> ha(nd), hb(nd);
    std::vector<float> c0(nd), c1(nd), c2(nd), c3(nd);
    std::vector<double> cx(nd);
    {
        std::mt19937_64 gen(99999ULL);
        std::uniform_real_distribution<float> d(-1e18f, 1e18f);
        for (int i = 0; i < nd; ++i) { ha[i] = d(gen); hb[i] = d(gen); }
        std::mt19937_64 gr(88888ULL);
        for (int i = 0; i < nd; ++i) {
            float e[4]; cx[i] = draw_ordered_double(gr, e);
            c0[i]=e[0]; c1[i]=e[1]; c2[i]=e[2]; c3[i]=e[3];
        }
    }
    Kokkos::View<float*, exec_space> va("va", nd), vb("vb", nd);
    Kokkos::View<float*, exec_space> s_hi("s_hi", nd), s_lo("s_lo", nd);
    Kokkos::View<float*, exec_space> p_hi("p_hi", nd), p_lo("p_lo", nd);
    Kokkos::View<float*, exec_space> rc0("rc0", nd), rc1("rc1", nd), rc2("rc2", nd), rc3("rc3", nd);
    Kokkos::View<float*, exec_space> rb0("rb0", nd), rb1("rb1", nd), rb2("rb2", nd);

    auto hva = Kokkos::create_mirror_view(va);
    auto hvb = Kokkos::create_mirror_view(vb);
    auto hc0 = Kokkos::create_mirror_view(rc0);
    auto hc1 = Kokkos::create_mirror_view(rc1);
    auto hc2 = Kokkos::create_mirror_view(rc2);
    auto hc3 = Kokkos::create_mirror_view(rc3);
    for (int i = 0; i < nd; ++i) {
        hva(i) = ha[i]; hvb(i) = hb[i];
        hc0(i) = c0[i]; hc1(i) = c1[i]; hc2(i) = c2[i]; hc3(i) = c3[i];
    }
    Kokkos::deep_copy(va, hva);  Kokkos::deep_copy(vb, hvb);
    Kokkos::deep_copy(rc0, hc0); Kokkos::deep_copy(rc1, hc1);
    Kokkos::deep_copy(rc2, hc2); Kokkos::deep_copy(rc3, hc3);

    Kokkos::parallel_for("tf_eft_device", Kokkos::RangePolicy<exec_space>(0, nd),
        KOKKOS_LAMBDA(int i) {
            float es, ep;
            s_hi(i) = xp::tf_two_sum(va(i), vb(i), es);  s_lo(i) = es;
            p_hi(i) = xp::tf_two_prod(va(i), vb(i), ep); p_lo(i) = ep;
            float b0 = rc0(i), b1 = rc1(i), b2 = rc2(i), b3 = rc3(i);
            xp::renorm_3(b0, b1, b2, b3);
            rb0(i) = b0; rb1(i) = b1; rb2(i) = b2;
        });
    Kokkos::fence();

    auto hshi = Kokkos::create_mirror_view(s_hi);
    auto hslo = Kokkos::create_mirror_view(s_lo);
    auto hphi = Kokkos::create_mirror_view(p_hi);
    auto hplo = Kokkos::create_mirror_view(p_lo);
    auto hb0 = Kokkos::create_mirror_view(rb0);
    auto hb1 = Kokkos::create_mirror_view(rb1);
    auto hb2 = Kokkos::create_mirror_view(rb2);
    Kokkos::deep_copy(hshi, s_hi); Kokkos::deep_copy(hslo, s_lo);
    Kokkos::deep_copy(hphi, p_hi); Kokkos::deep_copy(hplo, p_lo);
    Kokkos::deep_copy(hb0, rb0); Kokkos::deep_copy(hb1, rb1); Kokkos::deep_copy(hb2, rb2);

    long sum_fail = 0, prod_fail = 0, ren_fail = 0, sum_skip = 0, prod_skip = 0, ren_over = 0;
    int samples_left = 5;
    for (int i = 0; i < nd; ++i) {
        float a = ha[i], b = hb[i];
        if (sum_in_domain(a, b)) {
            if ((double)hshi(i) + (double)hslo(i) != (double)a + (double)b) {
                ++sum_fail; if (samples_left > 0) { print_fail_sum("twoSum", a, b); --samples_left; }
            }
        } else ++sum_skip;
        if (prod_in_domain(a, b)) {
            if ((double)hphi(i) + (double)hplo(i) != (double)a * (double)b) {
                ++prod_fail; if (samples_left > 0) { print_fail_prod("twoProd", a, b); --samples_left; }
            }
        } else ++prod_skip;
        double out_sum = (double)hb0(i) + hb1(i) + hb2(i);
        int dummy = 0;
        bool value_ok   = (out_sum == cx[i]);
        bool overlap_ok = nonoverlap_holds(hb0(i), hb1(i), hb2(i), &dummy);
        if (!(value_ok && overlap_ok)) { ++ren_fail; if (!overlap_ok) ++ren_over; }
    }
    std::printf("    device tf_two_sum : %ld tested (%ld skipped), %ld failures\n",
                (long)nd - sum_skip, sum_skip, sum_fail);
    std::printf("    device tf_two_prod: %ld tested (%ld skipped), %ld failures\n",
                (long)nd - prod_skip, prod_skip, prod_fail);
    std::printf("    device renorm_3   : %ld tested, %ld failures (%ld non-overlap)\n",
                (long)nd, ren_fail, ren_over);
    R.total   = 3 * nd;
    R.skipped = (int)(sum_skip + prod_skip);
    R.failed  = (int)(sum_fail + prod_fail + ren_fail);
    R.passed  = R.total - R.skipped - R.failed;
    return R;
}

// ============================================================================
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        std::printf("=== tf_eft_test: EFT bit-exactness for TF twoSum / twoProd / twoSqr / renorm ===\n");
        std::printf("Oracle: FP64 (exact, 25-bit sum / 48-bit product fit in 53-bit FP64)\n");
        std::printf("Splitter: 8193.0f = 2^13 + 1 (reused from ff_math.hpp)\n\n");

        std::printf("[Test A] tf_two_sum bit-exactness\n");
        EftCount A1 = run_host_batches(Op::Sum, "A/twoSum");
        std::printf("  tf_two_sum: total tested=%ld, skipped=%ld, failures=%ld\n", A1.tested, A1.skipped, A1.failures);
        KOKKOS_EP_ASSERT(A1.failures == 0, "tf_two_sum was not bit-exact");
        std::printf("[Test A'] tf_quick_two_sum bit-exactness (operands ordered |a|>=|b|)\n");
        EftCount A2 = run_host_batches(Op::QuickSum, "A/quickSum");
        std::printf("  tf_quick_two_sum: total tested=%ld, skipped=%ld, failures=%ld\n\n", A2.tested, A2.skipped, A2.failures);
        KOKKOS_EP_ASSERT(A2.failures == 0, "tf_quick_two_sum was not bit-exact");

        std::printf("[Test B] tf_two_prod (Dekker twoProduct) bit-exactness\n");
        EftCount B1 = run_host_batches(Op::Prod, "B/twoProd");
        std::printf("  tf_two_prod: total tested=%ld, skipped=%ld, failures=%ld\n", B1.tested, B1.skipped, B1.failures);
        KOKKOS_EP_ASSERT(B1.failures == 0, "tf_two_prod was not bit-exact");
        std::printf("[Test B'] tf_two_sqr bit-exactness\n");
        EftCount B2 = run_host_batches(Op::Sqr, "B/twoSqr");
        std::printf("  tf_two_sqr: total tested=%ld, skipped=%ld, failures=%ld\n\n", B2.tested, B2.skipped, B2.failures);
        KOKKOS_EP_ASSERT(B2.failures == 0, "tf_two_sqr was not bit-exact");

        std::printf("[Test C] renorm_3 (len 4->3) + renorm (len 3->3): non-overlap + value-preservation\n");
        RenormResult C1 = test_renorm_3_bounded(1'000'000, 45678ULL);
        std::printf("    renorm_3 bounded (exact FP64): tested=%ld  non-overlap-fail=%ld  value-fail=%ld  (pair-skips=%ld)\n",
                    C1.tested, C1.overlap_fail, C1.value_fail, C1.skips);
        KOKKOS_EP_ASSERT(C1.overlap_fail == 0, "renorm_3 produced an overlapping result");
        KOKKOS_EP_ASSERT(C1.value_fail   == 0, "renorm_3 did not preserve value exactly");

        RenormResult C2 = test_renorm_bounded(1'000'000, 56789ULL);
        std::printf("    renorm   bounded (exact FP64): tested=%ld  non-overlap-fail=%ld  value-fail=%ld  (pair-skips=%ld)\n\n",
                    C2.tested, C2.overlap_fail, C2.value_fail, C2.skips);
        KOKKOS_EP_ASSERT(C2.overlap_fail == 0, "renorm produced an overlapping result");
        KOKKOS_EP_ASSERT(C2.value_fail   == 0, "renorm did not preserve value exactly");

        std::printf("[Test D] named hard cases\n");
        NamedResult D = run_named_cases();
        std::printf("  Test D named cases: %d passed, %d skipped, %d failed (of %d)\n\n",
                    D.passed, D.skipped, D.failed, D.total);
        KOKKOS_EP_ASSERT(D.failed == 0, "a named EFT case failed");

        std::printf("[Test E] device parity (%s)\n", Kokkos::DefaultExecutionSpace::name());
        NamedResult E = run_device_parity();
        std::printf("  Test E device parity: %d passed, %d skipped, %d failed (of %d)\n\n",
                    E.passed, E.skipped, E.failed, E.total);
        KOKKOS_EP_ASSERT(E.failed == 0, "device EFT parity mismatch");

        rc = ep_exit_code();
        std::printf("=== tf_eft_test: %s ===\n", rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    }
    Kokkos::finalize();
    return rc;
}
