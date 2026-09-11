// probe_arith_floor.cpp — what is the BEST any algorithm can read on this
// metric, per backend, given the backend's own primitives?
//
// WHY THIS EXISTS.  The real sin/cos residual on DD and FF is 2-4 ulps while
// TF reads 0.53 and QF 0.98 on the same sweep, same metric, same grid.  Two
// stories fit that: (a) the DD/FF trig cores are worse, or (b) DD/FF ARITHMETIC
// is worse relative to the p each backend's ulp is defined at, so the same
// quality of algorithm reads a larger number.  Those have opposite consequences
// -- (a) is fixable in the trig core, (b) is inherent to the format -- and no
// amount of staring at sincos distinguishes them.
//
// So this probe leaves trig alone and scores the PRIMITIVES: add, subtract,
// multiply, divide, sqrt, each in the backend's own arithmetic, against MPFR at
// 400 bits, in ulps = |got-ref| / (|ref| * 2^-p) with the SAME p the sweep uses
// (106/48/72/96 for DD/FF/TF/QF, sweep_accuracy.cpp:ulps_scalar).  A correctly
// rounded primitive reads <= 0.5 ulps by definition.  Whatever a backend's
// primitives read is the floor its trig core cannot beat, because the core is
// built out of them.
//
// It also scores CHAINS, because a single primitive understates what a series
// does.  Every chain runs at |r| in (pi/8, pi/4), the worst part of the reduced
// range, with reduction, doubling and quadrant table all removed:
//   horner-14    a degree-14 polynomial with c_k = 1/(k+1) by Horner.  The
//                coefficients are FORMAT values and the reference uses their
//                exact limb sums; a first version used 1/(k+1) at 400 bits and
//                so scored the coefficients' representation error (2^72 ulps at
//                p = 96) rather than Horner's rounding.
//   taylor-sin   the shipped series shape: term *= r^2, term /= -(2k)(2k+1),
//                s += term, to convergence at the backend's own eps.
//   taylor-nodiv taylor-sin with the per-term divide replaced by a multiply by
//                a precomputed reciprocal.  Separates "the division" from "the
//                operation count" as explanations for any gap to horner-sin.
//   horner-sin   the SAME function as taylor-sin, as r*P(r^2) by Horner with
//                13 precomputed format coefficients.
//   double-x5    the five v-form doublings of dd_math.hpp:sincos, started from
//                an EXACTLY rounded (sin, v) at r/32.  What DD's nq = 5 pays
//                over FF/QF/TF's nq = 0, with every other stage removed.
//   repr-floor   sin(r) at 400 bits merely ROUNDED into the format.  No
//                algorithm can beat this; it is <= 0.5 by construction.
//
// ===========================================================================
// WHAT IT MEASURED, 2026-09-11, --n 20000
// ===========================================================================
//
//   arm            DD med / max / >1ulp     FF                TF               QF
//   multiply       0.283 / 2.976 / 5.15%    0.280/2.816/5.00% 0.022/0.243/0%   0.093/2.654/0.86%
//   taylor-sin     0.553 / 5.172 / 24.61%   0.367/4.197/13.1% 0.095/1.260/0.03% 0.038/0.487/0%
//   taylor-nodiv   0.554 / 5.172 / 24.66%   0.368/4.197/13.0% 0.095/1.260/0.03% 0.040/0.514/0%
//   horner-sin     0.237 / 2.227 / 2.90%    0.239/2.059/2.85% 0.039/0.337/0%   0.071/2.177/0.29%
//   double-x5      0.568 / 4.326 / 24.50%   0.565/5.027/24.3% 0.106/0.791/0%   0.215/3.108/2.48%
//   repr-floor     0.085 / 0.496 / 0%       0.086/0.494/0%    0.021/0.246/0%   0.005/0.123/0%
//
// 1. THE DD/FF TRIG RESIDUAL IS NOT A TRIG DEFECT.  A bare Taylor series, with
//    no reduction, no doubling and no quadrant table, reproduces the sweep's
//    whole DD/FF-vs-QF/TF split on its own: 24.6% and 13.1% of samples above
//    1 ulp against 0.03% and 0.00%.  Nothing in that arm is specific to trig,
//    so no trig-core change can be what separates the backends.
//
// 2. THE CAUSE IS THAT p IS CALIBRATED DIFFERENTLY AGAINST EACH FORMAT'S
//    ARITHMETIC.  A single DD `multiply` reads 0.283 median / 2.976 max at
//    p = 106; TF's reads 0.022 / 0.243 at p = 72.  That is ~3.6 bits, and it
//    is a property of the formats: DD's 2-limb multiply is the accurate variant
//    (it carries a.lo*b.lo, and the Dekker two-product is exact), and its
//    measured 2.976 max sits at the classical 4u^2 bound, u = 2^-53.  So DD
//    cannot round to its own p = 106 and TF has ~2 bits of slack under p = 72.
//    The shipped DD sin's worst SWEEP row is 3.326 ulps -- within 12% of what
//    ONE DD multiply costs.  This is NOT an argument for changing the metric;
//    it is the reason the same quality of algorithm reads a bigger number here.
//
// 3. THE DOUBLING CHAIN IS DD'S WHOLE BUDGET, AND IT TRACKS `multiply`.  Fed an
//    exactly rounded (sin, v), five doublings alone cost DD 0.568 / 4.326 and
//    TF 0.106 / 0.791 -- the same ~5x ratio as their multiplies, not a
//    trig-specific mechanism.
//
// 4. THERE IS ONE REAL LEVER, AND IT IS OPERATION COUNT, NOT DIVISION.
//    horner-sin computes the identical function 2.3x better in the median and
//    8.5x better in the >1 ulp rate on DD (24.61% -> 2.90%) and FF (13.02% ->
//    2.85%).  taylor-nodiv is what rules out the alternative explanation: it is
//    BIT-IDENTICAL to taylor-sin (DD max 5.1719 both), so the per-term divide
//    costs nothing and the gap is the forward sum's undamped roundings against
//    a running total of size |r|, where Horner's early roundings are multiplied
//    down by r^2 at every subsequent step.
//    THE LEVER IS DD/FF ONLY.  QF gets WORSE under Horner (0.487 -> 2.177 max),
//    so this is not a uniform improvement and must not be applied uniformly.
//
// OPERANDS.  Drawn as a random 400-bit MPFR value in [1, 2) times a random
// power of two, then ROUNDED INTO the backend format by repeated
// subtract-and-split, so the operand is a generic format value rather than a
// short decimal that happens to be exact.  The reference is the EXACT limb sum
// of the operands actually handed to the primitive -- never the MPFR value they
// were rounded from -- so input rounding cannot leak into the score.  That is
// the f(x_grid) vs f(x_stored) trap the sweep fixed in b7c4b64.
//
// Exponent ranges are kept modest (2^-20 .. 2^20) so that no limb goes
// subnormal: this probe is about the arithmetic, not about KI-33's
// representational floor, which is measured elsewhere.
//
// Build (needs MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_arith_floor.cpp -o /tmp/probe_arith_floor -lmpfr -lgmp
//   /tmp/probe_arith_floor            # 200000 samples per cell
//   /tmp/probe_arith_floor --n 50000
//   /tmp/probe_arith_floor --poison   # negative control, see below
//
// POISON.  --poison scores each primitive against a reference computed at 24
// bits instead of 400.  Every cell must blow up; if a cell stays small the
// scoring path is not actually comparing against the oracle it claims to.
// RUN 2026-09-11, --n 3000 --poison: every arm on every backend goes to 100%
// above 1 ulp -- DD add 1.9e24, DD multiply 1.58e24, DD taylor-sin/horner-sin/
// double-x5/repr-floor all ~1.6e24; FF ~5.6e6; TF ~9.4e13; QF ~1.5e21.  Note
// that repr-floor blowing up is the load-bearing one: it is the arm that reads
// <= 0.5 BY CONSTRUCTION, so if the oracle were not really being consulted it
// is the cell most likely to stay quiet.
//
// WHAT THIS PROBE DOES NOT SETTLE.  horner-sin is measured here in isolation,
// at the unreduced r.  Horner and nq are ORTHOGONAL -- nothing stops a Horner
// series being evaluated at the scaled-down u = r/2^nq and then doubled back --
// so row 4 is not, by itself, an argument for nq = 0.  The argument for nq = 0
// is row 5: double-x5 says the doublings ALONE cost DD 24.50% above 1 ulp, so
// Horner at nq = 5 would still read ~24% and the series would not be what is
// left to fix.  But nq = 0 on DD was measured ON THE SWEEP to cut DD+FF real
// trig only 24.9% (898 -> 674 rows above 1 ulp) while REGRESSING DD complex pow
// by up to +30.85 ulps (point 251: 6.136 -> 36.98).  Complex pow reads cos's low
// bits through exp(w*log z) -- probe_cpow.cpp later showed why that coupling is
// so tight: exp maps an ABSOLUTE perturbation of its argument to a RELATIVE one,
// so cos's low bits are amplified by 2^p, and with exact inputs the residual of
// complex pow on DD is 259 rows at max 3.41 ulps against a worst shipped sin of
// 3.326.  So row 4 is an upper bound on an isolated stage, not a shippable delta.

#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>
#include <xp/qf_math.hpp>
#include <xp/tf_math.hpp>

#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace xp;

static const mpfr_prec_t kPrec = 400;
static mpfr_prec_t       g_ref_prec = kPrec;   // --poison lowers this

// ---------------------------------------------------------------------------
// format traits: limb count, limb type, p, and limb access
// ---------------------------------------------------------------------------
template <class T> struct Fmt;

template <> struct Fmt<DoubleDouble> {
    using Limb = double;
    static const int  n = 2, p = 106;
    static const char* name() { return "DD"; }
    static double limb(const DoubleDouble& a, int i) { return i ? a.lo : a.hi; }
    static DoubleDouble make(const double* w) { return DoubleDouble(w[0], w[1]); }
};
template <> struct Fmt<FloatFloat> {
    using Limb = float;
    static const int  n = 2, p = 48;
    static const char* name() { return "FF"; }
    static float limb(const FloatFloat& a, int i) { return i ? a.lo : a.hi; }
    static FloatFloat make(const float* w) { return FloatFloat(w[0], w[1]); }
};
template <> struct Fmt<TripleFloat> {
    using Limb = float;
    static const int  n = 3, p = 72;
    static const char* name() { return "TF"; }
    static float limb(const TripleFloat& a, int i) { return i == 0 ? a.f0 : i == 1 ? a.f1 : a.f2; }
    static TripleFloat make(const float* w) { return TripleFloat(w[0], w[1], w[2]); }
};
template <> struct Fmt<QuadFloat> {
    using Limb = float;
    static const int  n = 4, p = 96;
    static const char* name() { return "QF"; }
    static float limb(const QuadFloat& a, int i) {
        return i == 0 ? a.f0 : i == 1 ? a.f1 : i == 2 ? a.f2 : a.f3;
    }
    static QuadFloat make(const float* w) { return QuadFloat(w[0], w[1], w[2], w[3]); }
};

// exact limb sum of a format value -> mpfr
template <class T> static void to_mpfr(mpfr_t out, const T& a) {
    mpfr_set_zero(out, 1);
    mpfr_t t; mpfr_init2(t, kPrec);
    for (int i = 0; i < Fmt<T>::n; ++i) {
        mpfr_set_d(t, (double)Fmt<T>::limb(a, i), MPFR_RNDN);   // each limb exact
        mpfr_add(out, out, t, MPFR_RNDN);
    }
    mpfr_clear(t);
}

// round an mpfr value into the format by repeated nearest-limb extraction
template <class T> static T from_mpfr(mpfr_srcptr v) {
    typename Fmt<T>::Limb w[4] = {0, 0, 0, 0};
    mpfr_t r; mpfr_init2(r, kPrec); mpfr_set(r, v, MPFR_RNDN);
    for (int i = 0; i < Fmt<T>::n; ++i) {
        w[i] = (typename Fmt<T>::Limb)mpfr_get_d(r, MPFR_RNDN);
        mpfr_t t; mpfr_init2(t, kPrec);
        mpfr_set_d(t, (double)w[i], MPFR_RNDN);
        mpfr_sub(r, r, t, MPFR_RNDN);
        mpfr_clear(t);
    }
    mpfr_clear(r);
    return Fmt<T>::make(w);
}

// ulps at the backend's p, sweep_accuracy.cpp:ulps_scalar convention
template <class T> static double ulps(const T& got, mpfr_srcptr ref) {
    if (mpfr_zero_p(ref) || !mpfr_number_p(ref)) return -1.0;
    mpfr_t g, d; mpfr_init2(g, kPrec); mpfr_init2(d, kPrec);
    to_mpfr(g, got);
    if (!mpfr_number_p(g)) { mpfr_clear(g); mpfr_clear(d); return -2.0; }
    mpfr_sub(d, g, ref, MPFR_RNDN);
    mpfr_abs(d, d, MPFR_RNDN);
    mpfr_t rr; mpfr_init2(rr, kPrec);
    mpfr_abs(rr, ref, MPFR_RNDN);
    mpfr_mul_2si(rr, rr, -Fmt<T>::p, MPFR_RNDN);      // |ref| * 2^-p
    mpfr_div(d, d, rr, MPFR_RNDN);
    double u = mpfr_get_d(d, MPFR_RNDN);
    mpfr_clear(g); mpfr_clear(d); mpfr_clear(rr);
    return u;
}

struct Acc {
    std::vector<double> v;
    void add(double u) { if (u >= 0) v.push_back(u); }
    double med() { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size()/2]; }
    double pct(double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end());
                           return v[(size_t)(q*(v.size()-1))]; }
    double max() { double m = 0; for (double x : v) m = std::max(m, x); return m; }
    size_t over(double t) { size_t c = 0; for (double x : v) if (x > t) ++c; return c; }
};

// ---------------------------------------------------------------------------
// per-backend run
// ---------------------------------------------------------------------------
template <class T> static void run(int N, unsigned seed) {
    std::mt19937_64 rng(seed);
    mpfr_t ma, mb, mr, tmp;
    mpfr_init2(ma, kPrec); mpfr_init2(mb, kPrec);
    mpfr_init2(mr, g_ref_prec); mpfr_init2(tmp, kPrec);

    Acc acc_add, acc_sub, acc_mul, acc_div, acc_sqrt, acc_horner, acc_taylor;
    Acc acc_double, acc_repr, acc_horsin, acc_nodiv;

    auto draw = [&](mpfr_t out) {
        // random 400-bit mantissa in [1,2), random exponent in [-20, 20]
        mpfr_set_ui(out, 1, MPFR_RNDN);
        mpfr_t frac; mpfr_init2(frac, kPrec);
        mpfr_set_ui(frac, 0, MPFR_RNDN);
        for (int k = 0; k < 7; ++k) {                 // 7*64 = 448 random bits
            mpfr_mul_2si(frac, frac, 64, MPFR_RNDN);
            mpfr_add_ui(frac, frac, (unsigned long)(rng() >> 1), MPFR_RNDN);
        }
        mpfr_div_2si(frac, frac, 7*64 - 1, MPFR_RNDN);   // into [0,2)
        mpfr_div_ui(frac, frac, 2, MPFR_RNDN);           // into [0,1)
        mpfr_add(out, out, frac, MPFR_RNDN);             // [1,2)
        int e = (int)(rng() % 41) - 20;
        mpfr_mul_2si(out, out, e, MPFR_RNDN);
        if (rng() & 1) mpfr_neg(out, out, MPFR_RNDN);
        mpfr_clear(frac);
    };

    for (int i = 0; i < N; ++i) {
        draw(ma); draw(mb);
        T a = from_mpfr<T>(ma), b = from_mpfr<T>(mb);
        // EXACT operands actually handed to the primitive
        mpfr_t ea, eb; mpfr_init2(ea, kPrec); mpfr_init2(eb, kPrec);
        to_mpfr(ea, a); to_mpfr(eb, b);

        mpfr_add(mr, ea, eb, MPFR_RNDN); acc_add.add(ulps<T>(add(a, b), mr));
        mpfr_sub(mr, ea, eb, MPFR_RNDN); acc_sub.add(ulps<T>(subtract(a, b), mr));
        mpfr_mul(mr, ea, eb, MPFR_RNDN); acc_mul.add(ulps<T>(multiply(a, b), mr));
        mpfr_div(mr, ea, eb, MPFR_RNDN); acc_div.add(ulps<T>(divide(a, b), mr));
        if (mpfr_sgn(ea) > 0) { mpfr_sqrt(mr, ea, MPFR_RNDN); acc_sqrt.add(ulps<T>(sqrt(a), mr)); }

        // horner: p(x) = sum_{k=0}^{K} c_k x^k with c_k = 1/(k+1), |x| < 1/2.
        // The coefficients are FORMAT values and the reference uses their EXACT
        // limb sums, not 1/(k+1) at 400 bits -- otherwise this arm scores the
        // coefficients' representation error (2^-24 on an FP32 limb, i.e. 2^72
        // ulps at p = 96) instead of Horner's rounding, which is what a first
        // version of it did.
        {
            const int K = 14;
            T x = multiply(a, T(typename Fmt<T>::Limb(0.25)));   // shrink into (-1/2,1/2) scale
            mpfr_t ex; mpfr_init2(ex, kPrec); to_mpfr(ex, x);
            auto coef = [&](int k, mpfr_t out) {                 // c_k as a format value
                mpfr_t c; mpfr_init2(c, kPrec);
                mpfr_set_ui(c, 1, MPFR_RNDN); mpfr_div_ui(c, c, k + 1, MPFR_RNDN);
                T cf = from_mpfr<T>(c);
                to_mpfr(out, cf);
                mpfr_clear(c);
                return cf;
            };
            mpfr_t eh, ec; mpfr_init2(eh, kPrec); mpfr_init2(ec, kPrec);
            T h = coef(K, eh);
            for (int k = K - 1; k >= 0; --k) {
                T ck = coef(k, ec);
                h = add(multiply(h, x), ck);
                mpfr_mul(eh, eh, ex, MPFR_RNDN);
                mpfr_add(eh, eh, ec, MPFR_RNDN);
            }
            mpfr_set(mr, eh, MPFR_RNDN);
            acc_horner.add(ulps<T>(h, mr));
            mpfr_clear(ex); mpfr_clear(eh); mpfr_clear(ec);
        }

        // taylor-sin at r drawn in (-pi/4, pi/4): the trig core's series, alone
        {
            mpfr_t er; mpfr_init2(er, kPrec);
            mpfr_set(er, ma, MPFR_RNDN);
            mpfr_set_exp(er, 0);                     // |er| in [1/2, 1)
            mpfr_mul_d(er, er, 0.7853981633974483, MPFR_RNDN);   // (pi/8, pi/4)
            T r = from_mpfr<T>(er);
            mpfr_t exr; mpfr_init2(exr, kPrec); to_mpfr(exr, r);

            T r2 = multiply(r, r), s = r, term = r;
            for (int k = 1; k <= 200; ++k) {
                term = divide_scalar(multiply(term, r2),
                                     -(typename Fmt<T>::Limb)((2*k) * (2*k + 1)));
                T ns = add(s, term);
                if (Fmt<T>::limb(ns, 0) == Fmt<T>::limb(s, 0) &&
                    Fmt<T>::limb(term, 0) == (typename Fmt<T>::Limb)0) { s = ns; break; }
                s = ns;
                if (std::fabs((double)Fmt<T>::limb(term, 0)) <
                    std::ldexp(std::fabs((double)Fmt<T>::limb(s, 0)), -Fmt<T>::p - 4)) break;
            }
            mpfr_sin(mr, exr, MPFR_RNDN);
            acc_taylor.add(ulps<T>(s, mr));

            // RECURRENCE WITH NO DIVISION.  Identical to taylor-sin except the
            // per-term divide_scalar by (2k)(2k+1) is replaced by a multiply by
            // its reciprocal, precomputed as a FORMAT value.  This separates the
            // two things that differ between taylor-sin and horner-sin -- the
            // division, and the operation count -- so the win can be attributed
            // to one of them instead of to both at once.
            {
                static bool init = false;
                static T    recip[64];
                if (!init) {
                    mpfr_t f; mpfr_init2(f, kPrec);
                    for (int k = 1; k < 64; ++k) {
                        mpfr_set_ui(f, (unsigned long)((2*k) * (2*k + 1)), MPFR_RNDN);
                        mpfr_ui_div(f, 1, f, MPFR_RNDN);
                        recip[k] = from_mpfr<T>(f);
                    }
                    mpfr_clear(f); init = true;
                }
                T s2 = r, tm = r;
                for (int k = 1; k <= 60; ++k) {
                    tm = negate(multiply(multiply(tm, r2), recip[k]));
                    T ns = add(s2, tm);
                    s2 = ns;
                    if (std::fabs((double)Fmt<T>::limb(tm, 0)) <
                        std::ldexp(std::fabs((double)Fmt<T>::limb(s2, 0)), -Fmt<T>::p - 4)) break;
                }
                mpfr_sin(mr, exr, MPFR_RNDN);
                acc_nodiv.add(ulps<T>(s2, mr));
            }

            // HORNER-SIN: the SAME function, evaluated with half the multiplies.
            // sin(r) = r * P(r^2), P = sum_k (-1)^k r^2k/(2k+1)!, by Horner with
            // the coefficients precomputed as format values.  The recurrence
            // form above spends 2 multiplies and 1 divide per term; Horner
            // spends 1 multiply and 1 add.  If operation count -- rather than
            // the format -- were what puts DD at 5 ulps, this arm would be
            // several times better.  Coefficients are format values and are
            // built once (static), exactly as a shipped implementation would.
            {
                static bool init = false;
                static T    c[14];
                if (!init) {
                    mpfr_t f; mpfr_init2(f, kPrec);
                    for (int k = 0; k < 14; ++k) {
                        mpfr_fac_ui(f, (unsigned long)(2*k + 1), MPFR_RNDN);
                        mpfr_ui_div(f, 1, f, MPFR_RNDN);
                        if (k & 1) mpfr_neg(f, f, MPFR_RNDN);
                        c[k] = from_mpfr<T>(f);
                    }
                    mpfr_clear(f); init = true;
                }
                T h = c[13];
                for (int k = 12; k >= 0; --k) h = add(multiply(h, r2), c[k]);
                h = multiply(h, r);
                mpfr_sin(mr, exr, MPFR_RNDN);
                acc_horsin.add(ulps<T>(h, mr));
            }

            // THE DOUBLING CHAIN, ALONE.  Start from the EXACT (sin, v) at
            // u = r/2^5, rounded into the format -- so the series contributes
            // nothing but its representation error -- and run the five v-form
            // doublings dd_math.hpp:sincos runs.  Score against sin(r).  This
            // is exactly what DD's nq = 5 pays over FF/QF/TF's nq = 0, with
            // every other stage removed.  A perfect chain reads <= 0.5.
            {
                mpfr_t eu, es, ev; mpfr_init2(eu, kPrec);
                mpfr_init2(es, kPrec); mpfr_init2(ev, kPrec);
                mpfr_div_2si(eu, exr, 5, MPFR_RNDN);          // u = r/32, exact
                mpfr_sin(es, eu, MPFR_RNDN);                  // sin u
                mpfr_cos(ev, eu, MPFR_RNDN);
                mpfr_ui_sub(ev, 1, ev, MPFR_RNDN);            // v = 1 - cos u
                T sr = from_mpfr<T>(es), vr = from_mpfr<T>(ev);
                for (int q = 0; q < 5; ++q) {
                    const T c_q = subtract(T(typename Fmt<T>::Limb(1)), vr);
                    const T ns  = multiply_scalar(multiply(sr, c_q),
                                                  (typename Fmt<T>::Limb)2);
                    vr = multiply_scalar(multiply(sr, sr), (typename Fmt<T>::Limb)2);
                    sr = ns;
                }
                mpfr_sin(mr, exr, MPFR_RNDN);
                acc_double.add(ulps<T>(sr, mr));
                mpfr_clear(eu); mpfr_clear(es); mpfr_clear(ev);
            }

            // REPRESENTATION FLOOR: sin(r) computed at 400 bits and merely
            // ROUNDED into the format.  No algorithm can beat this.
            {
                mpfr_sin(tmp, exr, MPFR_RNDN);
                T best = from_mpfr<T>(tmp);
                mpfr_set(mr, tmp, MPFR_RNDN);
                acc_repr.add(ulps<T>(best, mr));
            }
            mpfr_clear(er); mpfr_clear(exr);
        }
        mpfr_clear(ea); mpfr_clear(eb);
    }

    printf("=== %s (p = %d) ===\n", Fmt<T>::name(), Fmt<T>::p);
    printf("  %-11s %9s %9s %9s %9s %9s\n", "op", "median", "p95", "p99.9", "max", ">1ulp%");
    struct Row { const char* n; Acc* a; } rows[] = {
        {"add",        &acc_add},   {"subtract", &acc_sub},  {"multiply", &acc_mul},
        {"divide",     &acc_div},   {"sqrt",     &acc_sqrt},
        {"horner-14",  &acc_horner},{"taylor-sin", &acc_taylor},
        {"taylor-nodiv",&acc_nodiv}, {"horner-sin", &acc_horsin},
        {"double-x5",  &acc_double}, {"repr-floor", &acc_repr},
    };
    for (auto& r : rows) {
        printf("  %-11s %9.4f %9.4f %9.4f %9.4f %8.2f%%\n", r.n, r.a->med(), r.a->pct(0.95),
               r.a->pct(0.999), r.a->max(),
               100.0 * (double)r.a->over(1.0) / (double)std::max<size_t>(r.a->v.size(), 1));
    }
    printf("\n");
    mpfr_clear(ma); mpfr_clear(mb); mpfr_clear(mr); mpfr_clear(tmp);
}

int main(int argc, char** argv) {
    int N = 200000; unsigned seed = 20260911u;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc)      N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--poison"))            g_ref_prec = 24;
    }
    printf("probe_arith_floor: %d samples/cell, reference at %ld bits%s\n\n",
           N, (long)g_ref_prec, g_ref_prec == kPrec ? "" : "   *** POISONED ***");
    printf("ulps = |got-ref| / (|ref| * 2^-p); a correctly rounded op reads <= 0.5.\n");
    printf("Operands are generic format values; the reference is their EXACT limb sum.\n\n");
    run<DoubleDouble>(N, seed);
    run<FloatFloat>(N, seed);
    run<TripleFloat>(N, seed);
    run<QuadFloat>(N, seed);
    return 0;
}
