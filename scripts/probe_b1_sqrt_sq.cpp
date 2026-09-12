// ===========================================================================
// probe_b1_sqrt_sq.cpp -- why qf_property_test's B1_sqrt_sq mean lands at
// 27.88 against its 27.90 tolerance once the divide lift is in.
//
// B1 is  sqrt(a)^2 ~= a,  a = 10^u with u ~ Uniform[-30, 30], N = 200000,
// scored as the MEAN over samples of digits = -log10(|c-r|/|r|), capped at
// kMaxDig = 29.  (tests/qf_property_test.cpp:154, :447, :599.)  It is a REAL
// sqrt round-trip; nothing complex is involved.  qf::sqrt is Heron, and its
// iteration divides -- include/xp/qf_math.hpp:866, `divide(a, x)` -- so the
// lift reaches it.
//
// This probe runs the identical generator against BOTH divides, using the
// instrumented-header trick from probe_div_downstream.sh so the two can be
// switched at runtime from inside qf_math.hpp:
//
//   plain  -- qf_divide_core, the pre-lift quotient
//   lift   -- the shipped lifted quotient
//   floor  -- the exactly-rounded 400-bit quotient (the unbeatable answer)
//
// It reports, for each: the mean digits the gate reads, the same population in
// ULPS (the metric of record -- digits is the gate's unit, not the project's),
// a per-decade breakdown so "which samples drag the mean" is answered by
// location rather than by assertion, and the count of samples at each digit
// level.  The `floor` arm is the control that matters: if an exactly-rounded
// divide gives the same mean as the lift, the mean is a property of Heron and
// the cap, not of the lift.
//
// Build: scripts/probe_b1_sqrt_sq.sh
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <mpfr.h>

#include <xp/config.hpp>
#include <xp/qf_math.hpp>

using xp::QuadFloat;

static const mpfr_prec_t kPrec = 400;
static const int    kQfP      = 96;      // QF significand bits
static const double kMaxDig   = 29.0;    // tests/qf_property_test.cpp cap
static const long   kRandomN  = 200000;  // tests/qf_property_test.cpp:252
// B1 is the 15th seed++ consumer after `uint64_t seed = 12345ULL`
// (tests/qf_property_test.cpp:515); 14 uses precede line 599.
static const unsigned long long kB1Seed = 12345ULL + 14ULL;

enum Policy { P_PLAIN = 0, P_LIFT = 1, P_FLOOR = 2 };
static Policy g_policy = P_LIFT;
static long   g_calls = 0, g_fires = 0;

static void qf_to_mpfr(mpfr_t o, const QuadFloat& x) {
    mpfr_set_flt(o, x.f0, MPFR_RNDN);
    mpfr_t t; mpfr_init2(t, kPrec);
    mpfr_set_flt(t, x.f1, MPFR_RNDN); mpfr_add(o, o, t, MPFR_RNDN);
    mpfr_set_flt(t, x.f2, MPFR_RNDN); mpfr_add(o, o, t, MPFR_RNDN);
    mpfr_set_flt(t, x.f3, MPFR_RNDN); mpfr_add(o, o, t, MPFR_RNDN);
    mpfr_clear(t);
}

// greedy round of an exact value into the QF format: the format floor
static QuadFloat qf_from_mpfr(mpfr_t v) {
    mpfr_t r; mpfr_init2(r, kPrec); mpfr_set(r, v, MPFR_RNDN);
    float w[4];
    for (int i = 0; i < 4; ++i) {
        w[i] = mpfr_get_flt(r, MPFR_RNDN);
        if (!std::isfinite(w[i])) { for (int j = i + 1; j < 4; ++j) w[j] = 0.0f; break; }
        mpfr_t t; mpfr_init2(t, kPrec); mpfr_set_flt(t, w[i], MPFR_RNDN);
        mpfr_sub(r, r, t, MPFR_RNDN); mpfr_clear(t);
    }
    mpfr_clear(r);
    return QuadFloat(w[0], w[1], w[2], w[3]);
}

static QuadFloat exact_quotient(const QuadFloat& a, const QuadFloat& b) {
    mpfr_t ma, mb, q;
    mpfr_init2(ma, kPrec); mpfr_init2(mb, kPrec); mpfr_init2(q, kPrec);
    qf_to_mpfr(ma, a); qf_to_mpfr(mb, b);
    QuadFloat out;
    if (mpfr_zero_p(mb) || !mpfr_number_p(ma) || !mpfr_number_p(mb))
        out = qf_divide_shipped(a, b);           // leave the edge cases alone
    else { mpfr_div(q, ma, mb, MPFR_RNDN); out = qf_from_mpfr(q); }
    mpfr_clears(ma, mb, q, (mpfr_ptr)0);
    return out;
}

namespace xp {
XPMATH_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b) {
    ++g_calls;
    if (xp::detail::qf_div_lift_wanted(a, b)) ++g_fires;
    switch (g_policy) {
        case P_PLAIN: return xp::detail::qf_divide_core(a, b);
        case P_FLOOR: return exact_quotient(a, b);
        default:      return qf_divide_shipped(a, b);
    }
}
}  // namespace xp

// ---------------------------------------------------------------------------
struct Sample { double x; double digits; double ulps; };

// digits exactly as tests/qf_property_test.cpp:154 computes them, but at 400
// bits instead of __float128 -- a wider oracle can only make the score more
// honest, and it is the same oracle the rest of this diagnosis uses.
static void score(const QuadFloat& c, double x, double& digits, double& ulps) {
    mpfr_t mc, mr, d, rel;
    mpfr_init2(mc, kPrec); mpfr_init2(mr, kPrec);
    mpfr_init2(d, kPrec);  mpfr_init2(rel, kPrec);
    qf_to_mpfr(mc, c);
    mpfr_set_d(mr, x, MPFR_RNDN);
    if (!mpfr_number_p(mc)) { digits = 0.0; ulps = -1.0; goto done; }
    mpfr_sub(d, mc, mr, MPFR_RNDN); mpfr_abs(d, d, MPFR_RNDN);
    if (mpfr_zero_p(d)) { digits = kMaxDig; ulps = 0.0; goto done; }
    mpfr_div(rel, d, mr, MPFR_RNDN); mpfr_abs(rel, rel, MPFR_RNDN);
    digits = -std::log10(mpfr_get_d(rel, MPFR_RNDN));
    if (digits < 0.0) digits = 0.0;
    if (digits > kMaxDig) digits = kMaxDig;
    // ulps = |c - r| / (|r| * 2^-p), the project's metric of record
    mpfr_mul_2si(rel, rel, kQfP, MPFR_RNDN);
    ulps = mpfr_get_d(rel, MPFR_RNDN);
done:
    mpfr_clears(mc, mr, d, rel, (mpfr_ptr)0);
}

static void run(Policy p, const char* name, std::vector<Sample>& out) {
    g_policy = p; g_calls = g_fires = 0;
    out.clear(); out.reserve(kRandomN);
    std::mt19937_64 g(kB1Seed);
    std::uniform_real_distribution<double> u(-30.0, 30.0);
    double sum_d = 0.0, min_d = 1e300;
    for (long i = 0; i < kRandomN; ++i) {
        const double x = std::pow(10.0, u(g));
        QuadFloat s = xp::sqrt(QuadFloat(x));
        QuadFloat c = xp::multiply(s, s);
        Sample sm; sm.x = x;
        score(c, x, sm.digits, sm.ulps);
        sum_d += sm.digits;
        if (sm.digits < min_d) min_d = sm.digits;
        out.push_back(sm);
    }
    std::printf("%-6s n=%ld  min=%.2f  MEAN=%.4f  tol=27.90  %s   "
                "(divides %ld, lift-guard fires %ld)\n",
                name, kRandomN, min_d, sum_d / (double)kRandomN,
                (sum_d / (double)kRandomN) >= 27.90 ? "PASS" : "FAIL",
                g_calls, g_fires);
}

int main(int argc, char** argv) {
    const bool per_decade = (argc > 1 && std::strcmp(argv[1], "--decades") == 0);
    std::printf("# probe_b1_sqrt_sq   N=%ld  seed=%llu  oracle MPFR %ld bits\n",
                kRandomN, kB1Seed, (long)kPrec);
    std::printf("# B1_sqrt_sq is REAL sqrt(a)^2 vs a; qf::sqrt is Heron and its\n"
                "# iteration divides at qf_math.hpp:866.\n\n");

    std::vector<Sample> plain, lift, floor_;
    run(P_PLAIN, "plain", plain);
    run(P_LIFT,  "lift",  lift);
    run(P_FLOOR, "floor", floor_);

    // ---- where the mean lives -------------------------------------------
    std::printf("\n== digit histogram (samples per band) ==\n");
    const double edge[] = {0.0, 20.0, 25.0, 27.0, 27.9, 28.5, 28.9, 29.01};
    const int nb = (int)(sizeof(edge) / sizeof(edge[0])) - 1;
    std::printf("%-14s", "band");
    for (int b = 0; b < nb; ++b) std::printf(" %10.4g-%-6.4g", edge[b], edge[b + 1]);
    std::printf("\n");
    const std::vector<Sample>* arms[3] = {&plain, &lift, &floor_};
    const char* an[3] = {"plain", "lift", "floor"};
    for (int k = 0; k < 3; ++k) {
        long cnt[8] = {0};
        for (size_t i = 0; i < arms[k]->size(); ++i) {
            const double d = (*arms[k])[i].digits;
            for (int b = 0; b < nb; ++b)
                if (d >= edge[b] && d < edge[b + 1]) { ++cnt[b]; break; }
        }
        std::printf("%-14s", an[k]);
        for (int b = 0; b < nb; ++b) std::printf(" %17ld", cnt[b]);
        std::printf("\n");
    }

    // ---- which samples moved, and by how much ---------------------------
    std::printf("\n== lift vs plain, sample by sample ==\n");
    long moved = 0, better = 0, worse = 0;
    double worst_drop = 0.0, worst_x = 0.0, dsum = 0.0;
    for (size_t i = 0; i < plain.size(); ++i) {
        const double dd = lift[i].digits - plain[i].digits;
        if (dd == 0.0) continue;
        ++moved; dsum += dd;
        if (dd > 0) ++better; else ++worse;
        if (dd < worst_drop) { worst_drop = dd; worst_x = plain[i].x; }
    }
    std::printf("  moved %ld of %ld   better %ld   worse %ld\n",
                moved, kRandomN, better, worse);
    std::printf("  mean digits delta over the whole population: %+.6f\n",
                dsum / (double)kRandomN);
    std::printf("  largest single-sample digit drop: %.4f at x = %.6g\n",
                worst_drop, worst_x);

    // ---- ulps, the metric of record --------------------------------------
    std::printf("\n== the same population in ULPS (metric of record) ==\n");
    for (int k = 0; k < 3; ++k) {
        double su = 0.0, mx = 0.0; long n = 0;
        for (size_t i = 0; i < arms[k]->size(); ++i) {
            const double u = (*arms[k])[i].ulps;
            if (u < 0) continue;
            su += u; ++n; if (u > mx) mx = u;
        }
        std::printf("  %-6s mean ulps %.6f   max ulps %.6f   n %ld\n",
                    an[k], su / (double)n, mx, n);
    }
    long u_worse = 0, u_better = 0, u_same = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        if (lift[i].ulps < 0 || plain[i].ulps < 0) continue;
        if (lift[i].ulps > plain[i].ulps) ++u_worse;
        else if (lift[i].ulps < plain[i].ulps) ++u_better;
        else ++u_same;
    }
    std::printf("  lift vs plain in ulps: better %ld, worse %ld, identical %ld\n",
                u_better, u_worse, u_same);
    // The line that settles it: if the lift matches an EXACTLY-ROUNDED divide
    // over this population, then B1's mean is not something a better divide
    // can fix, and 27.90 was calibrated against the old, worse quotient.
    long f_worse = 0, f_better = 0, f_same = 0;
    for (size_t i = 0; i < lift.size(); ++i) {
        if (lift[i].ulps < 0 || floor_[i].ulps < 0) continue;
        if (lift[i].ulps > floor_[i].ulps) ++f_worse;
        else if (lift[i].ulps < floor_[i].ulps) ++f_better;
        else ++f_same;
    }
    std::printf("  lift vs FLOOR (exactly-rounded divide) in ulps: "
                "lift better %ld, lift worse %ld, identical %ld\n",
                f_better, f_worse, f_same);
    long fp_worse = 0, fp_better = 0, fp_same = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        if (plain[i].ulps < 0 || floor_[i].ulps < 0) continue;
        if (plain[i].ulps > floor_[i].ulps) ++fp_worse;
        else if (plain[i].ulps < floor_[i].ulps) ++fp_better;
        else ++fp_same;
    }
    std::printf("  plain vs FLOOR in ulps: plain better %ld, plain worse %ld, "
                "identical %ld\n", fp_better, fp_worse, fp_same);

    // ---- per-decade, so "which samples" is a location not a claim --------
    if (per_decade) {
        std::printf("\n== mean digits by decade of x ==\n");
        std::printf("%-10s %8s %12s %12s %12s\n",
                    "decade", "n", "plain", "lift", "floor");
        for (int e = -30; e < 30; e += 5) {
            double s[3] = {0, 0, 0}; long n = 0;
            for (size_t i = 0; i < plain.size(); ++i) {
                const double lx = std::log10(plain[i].x);
                if (lx < e || lx >= e + 5) continue;
                ++n;
                s[0] += plain[i].digits; s[1] += lift[i].digits; s[2] += floor_[i].digits;
            }
            if (!n) continue;
            std::printf("[%+4d,%+4d) %8ld %12.4f %12.4f %12.4f\n",
                        e, e + 5, n, s[0] / n, s[1] / n, s[2] / n);
        }
    }
    return 0;
}
