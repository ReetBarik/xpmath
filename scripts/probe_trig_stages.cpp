// probe_trig_stages.cpp — attribute DD trig error to a STAGE, not to a guess.
//
// WHY THIS EXISTS.  `sweep_accuracy` scores the finished answer.  When that
// answer moves, the sweep cannot say whether the argument reduction or the
// Taylor/doubling stage moved it, and a commit message that asserts one or the
// other without measuring is exactly the kind of unverifiable claim this repo
// has been burned by.  This probe measures each stage separately against
// MPFR at 400 bits.
//
// It reports three things per point:
//
//   1. REDUCTION ACCURACY, old vs new.  Bits of relative accuracy of r_mod =
//      a - j*(pi/2) against the exact value.  "old" is the two-stage
//      `a - 2pi*nint(a/2pi)` then mod pi/2 that shipped before Payne-Hanek;
//      "new" is xp_ph_reduce.  DoubleDouble carries p = 106, so ~106-107 bits
//      is the format's own cap and nothing above it is meaningful.
//
//   2. PER-STAGE BITS for the shipped path: f (the Payne-Hanek fraction, 3
//      doubles), fdd (that fraction summed into a DoubleDouble), r_mod, and
//      the finished sin/cos.
//
//   3. THE nq SWEEP.  dd_math.hpp:sincos evaluates the series at r_mod/2^nq
//      and then doubles nq times.  This arm replicates sincos EXACTLY with nq
//      lifted out, so the reduction is bit-identical across every arm and any
//      dependence of the error on nq cannot come from it.
//
// MEASURED, 2026-09-11, at the three points that carry the post-Payne-Hanek
// DD peaks (see the Phase 3 DD commit):
//
//   x            old bits   new bits    ulps at nq=0   ulps at nq=5 (shipped)
//   -10000          94.02     110.05      sin  0.620      sin  45.20
//   1.75           106.56     106.48      sin  0.269      sin  42.44
//   -7.65          104.43     106.48      cos  0.957      cos   2.44
//   182.2123739     42.42     107.49      sin  0.356      sin   0.356
//   1e15            56.96     107.88
//   1.20557e16      53.88     108.63
//
// i.e. the reduction is at the format's cap everywhere, and the residual is
// the doubling loop: same reduction, nq=0 vs nq=5 moves sin at x=-10000 by a
// factor of 73.  Holding nq at 5 is a series-stage question and is deliberately
// left to the series phase; it is recorded here so that phase starts from a
// measurement instead of a hypothesis.
//
// A second mode, `--range LO HI N`, answers a different question: what does a
// magnitude bail cost?  It draws N arguments log-uniformly from [LO, HI] and
// reports the worst ulps of the Payne-Hanek path against the worst ulps of the
// identity point (1, 0) that dd_math.hpp:sincos returns above 1e60.  MEASURED,
// 2026-09-11, `--range 1e60 1.7976931348623157e308 4000`:
//
//   Payne-Hanek    max sin    50.79 ulps   max cos    51.14 ulps
//   identity (1,0) max sin 8.113e+31 ulps   max cos 1.664e+36 ulps  (3995/3995 wrong)
//
// A third mode, `--widen`, is the NEGATIVE CONTROL for the whole phase: could
// the old `a - 2pi*nint(a/2pi)` have been rescued by a wider 2pi instead?  It
// hands that form an exact n and exact 4096-bit arithmetic, so the constant's
// width q is the only error left.  MEASURED, 2026-09-11:
//
//   x                    log2|x/r|   q=106    q=212    q=318    q=424   q=1239
//   182.21237390820801      66.00     40.8    149.2    252.7    359.1   1173.7
//   1e15                    48.75     58.1    166.5    269.9    376.3   1190.9
//   1e60                   200.06    -93.2     15.2    118.6    225.0   1039.6
//   FP64 worst case        848.85   -742.0   -633.6   -530.2   -423.8    390.8
//   DD table pin          1021.77   -914.9   -806.5   -703.1   -596.7    217.9
//
// Every row is bits = q - log2(|x|/|r|): widening buys bit for bit and never
// closes the gap, because the gap belongs to x.  Covering every DoubleDouble
// needs q >= p + 4 + D = 1239 with D = 1129.4 measured -- the same 2/pi string
// the shipped 52-chunk table already is.  And the second half of that mode
// shows the form cannot be run at any q: nint(a/2pi) is off by 2^909 at the
// table pin, tracking log2|n| - 106, because the QUOTIENT cannot hold n.
//
// A fourth mode, `--fp32`, does for QF and TF what the default mode does for
// DD.  It exists because Payne-Hanek left QF -- and only QF -- above its
// format floor at grid family (5): 22.8 ulps for cos and tan at the FP64 pi/2
// (sweep point 1652) and 405 ulps for sin and tan at 182.21 (point 1160),
// where TF scores 0.86 and DD 0.98 on the same inputs.  22.8 ulps passes its
// derived bound of 5.739e17 by seventeen orders of magnitude -- every one of
// these points is a near-zero of the function -- so none is a gate failure,
// but it is not the format's floor either, and "QF is worse than TF" is the
// kind of result that must be attributed rather than documented.
//
// It is NOT the reduction.  MEASURED, 2026-09-11:
//
//   point           log2|x/r|   pio2    fr    r_mod   r_best   deficit
//   1652  pi/2          54.51  103.87  96.59   94.55   96.17      1.62
//   1160  182.21        66.00  103.87  92.45   95.46   95.46      0.00
//    870  91.106        66.00  103.87  92.45   90.56   90.56      0.00
//
// r_best is the best a QuadFloat can represent of the exact r_mod, so the
// deficit column is what Payne-Hanek costs over a perfect reduction: zero at
// two of the three points and 1.62 bits at the third.  That cannot produce a
// 4.8-bit error.
//
// It IS the nq scale-down/double-back, and the nq sweep proves it by holding
// the reduction bit-identical across every arm:
//
//   point            nq=0    1     2     3     4     5(ship)  6     7
//   1652  pi/2      110.3 110.3 110.3  93.1  93.1   91.6    90.8  90.8
//   1160  182.21    119.6  90.5  90.5  90.5  87.7   87.3    86.0  85.2
//    870  91.106    121.6  89.5  89.5  87.9  87.2   86.1    85.1  85.1
//
// The loss grows as |r| shrinks, which is the FP32 subnormal floor being
// reached by the SCALED intermediates: u = r_mod/2^nq puts a QuadFloat's
// fourth limb at 2^(e-nq-72), under FLT_MIN = 2^-126 once e < -49, and u^2
// lower still.  Worst of 64 random full-width r per row, with the mean count
// of subnormal limbs the arm actually touched, and TF (nq=4) for contrast:
//
//   log2|r|   QF nq=0  sub    QF nq=5  sub     loss    TF nq=4
//     -32       99.91  3.7      97.99  3.5      1.92      72.31
//     -44       99.35  5.2      98.71  9.3      0.64      88.60
//     -48       99.54  4.1      97.06 12.4      2.49      96.59
//     -52      104.59  5.0      93.31 24.3     11.28     104.59
//     -56      112.59  5.0      89.24 201.9    23.35     112.59
//     -64      128.59 201.4     80.94 196.3    47.65     128.59
//
// The nq=0 and TF columns rise above p because sin(r) -> r as r shrinks, so
// the answer is very nearly the input and almost nothing is left to get wrong;
// that is the shape the shipped arm should have and does not.  TF matching the
// nq=0 column exactly from -52 down is the same effect, not TF beating QF at
// arithmetic -- but it is why TF scores 0.86 ulps at point 1652 where QF
// scores 22.8.
//
// Read that correlation carefully: subnormal limbs are NOT sufficient on their
// own.  The nq=0 arm touches just as many at the bottom of the range (201 at
// 2^-64) and loses nothing, because there they sit in the r^3/6 tail terms.
// What costs bits is subnormal limbs in u and u^2 THEMSELVES, which the
// doublings then scale back up by 2^nq.  So: proven at the nq level, and
// consistent with -- not proven to be -- the subnormal floor at the limb level.
//
// Holding nq at 5 is a series-stage question, the same one the DD nq sweep
// above raises, and it is deliberately left to the series phase.  Recorded
// here so that phase starts from a measurement instead of a hypothesis.
//
// Build (needs MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_trig_stages.cpp -o /tmp/probe_trig_stages -lmpfr -lgmp
//   /tmp/probe_trig_stages -10000 1.75 -7.65 182.21237390820801
//   /tmp/probe_trig_stages --range 1e60 1.7976931348623157e308 4000
//   /tmp/probe_trig_stages --widen
//   /tmp/probe_trig_stages --fp32

#include <xp/dd_math.hpp>
#include <xp/qf_math.hpp>
#include <xp/tf_math.hpp>

#include <mpfr.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

using namespace xp;

static const mpfr_prec_t kPrec = 400;

static double log2abs(mpfr_srcptr v) {
    if (mpfr_zero_p(v)) return -1e9;
    mpfr_t t;
    mpfr_init2(t, 64);
    mpfr_abs(t, v, MPFR_RNDN);
    mpfr_log2(t, t, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

// bits of relative accuracy of the pair (hi + lo) against `exact`
static double bits_of(mpfr_srcptr exact, const DoubleDouble& v) {
    mpfr_t e;
    mpfr_init2(e, kPrec);
    mpfr_set_d(e, v.hi, MPFR_RNDN);
    mpfr_add_d(e, e, v.lo, MPFR_RNDN);
    mpfr_sub(e, e, exact, MPFR_RNDN);
    const double r = log2abs(exact) - log2abs(e);
    mpfr_clear(e);
    return r;
}

static double ulps_of(mpfr_srcptr exact, const DoubleDouble& v) {
    if (mpfr_zero_p(exact)) return -1.0;   // unscorable, as in sweep_accuracy
    mpfr_t e, a;
    mpfr_inits2(kPrec, e, a, (mpfr_ptr)0);
    mpfr_set_d(e, v.hi, MPFR_RNDN);
    mpfr_add_d(e, e, v.lo, MPFR_RNDN);
    mpfr_sub(e, e, exact, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    mpfr_abs(a, exact, MPFR_RNDN);
    mpfr_div(e, e, a, MPFR_RNDN);
    mpfr_mul_2si(e, e, 106, MPFR_RNDN);
    const double r = mpfr_get_d(e, MPFR_RNDN);
    mpfr_clears(e, a, (mpfr_ptr)0);
    return r;
}

// The Payne-Hanek half of dd_math.hpp:sincos, lifted out verbatim.
static int reduce_new(DoubleDouble a, DoubleDouble& r_mod, double f[3]) {
    if (detail::fabs(a.hi) <= 0.75) { r_mod = a; f[0] = f[1] = f[2] = 0.0; return 0; }
    const double win[2] = { a.hi, a.lo };
    const int    j      = detail::xp_ph_reduce<double>(win, 2, detail::kPhGuardDD,
                                                       detail::kPhChunksDD, f, 3);
    const DoubleDouble pio2 =
        add(add(DoubleDouble(detail::xp_ph_pio2_d(0)),
                DoubleDouble(detail::xp_ph_pio2_d(1))),
            DoubleDouble(detail::xp_ph_pio2_d(2)));
    const DoubleDouble fdd =
        add(add(DoubleDouble(f[0]), DoubleDouble(f[1])), DoubleDouble(f[2]));
    r_mod = multiply(fdd, pio2);
    return j;
}

// The reduction that shipped BEFORE Payne-Hanek (git show 2226a3c:include/xp/dd_math.hpp).
static int reduce_old(DoubleDouble a, DoubleDouble& r_mod) {
    const DoubleDouble pi2 = multiply_scalar(DoubleDouble_pi(), 2.0);
    const DoubleDouble s3  =
        subtract(a, multiply(pi2, round_to_nearest_int(divide(a, pi2))));
    const DoubleDouble ph = multiply_scalar(DoubleDouble_pi(), 0.5);
    const DoubleDouble n  = round_to_nearest_int(divide(s3, ph));
    r_mod = subtract(s3, multiply(ph, n));
    return ((int)n.hi) & 3;
}

// dd_math.hpp:sincos with nq a parameter and the guards stripped (the probe
// only ever drives ordinary finite arguments).  cos into x, sin into y, as DD does.
static void sincos_nq(DoubleDouble a, int nq, DoubleDouble& x, DoubleDouble& y) {
    const int    itrmx = 1000;
    const double eps   = 1.0e-32;
    DoubleDouble r_mod;
    double       f[3];
    const int    j = reduce_new(a, r_mod, f);

    DoubleDouble r  = multiply_scalar(r_mod, 1.0 / (double)(1 << nq));
    DoubleDouble r2 = multiply(r, r);
    DoubleDouble sin_r = r, cos_r = DoubleDouble(1.0);
    DoubleDouble sterm = r, cterm = DoubleDouble(1.0);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(double)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(double)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (detail::fabs(sterm.hi) < eps * detail::fabs(sin_r.hi) &&
            detail::fabs(cterm.hi) < eps) break;
    }
    for (int d = 0; d < nq; ++d) {
        const DoubleDouble ns = multiply_scalar(multiply(sin_r, cos_r), 2.0);
        const DoubleDouble nc = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = ns;
        cos_r = nc;
    }
    if (j == 0)      { x = cos_r;         y = sin_r; }
    else if (j == 1) { x = negate(sin_r); y = cos_r; }
    else if (j == 2) { x = negate(cos_r); y = negate(sin_r); }
    else             { x = sin_r;         y = negate(cos_r); }
}

// --widen: THE NEGATIVE CONTROL.  Could the old `a - 2pi*nint(a/2pi)` have been
// saved by using a wider 2pi?  This mode answers with a measurement instead of
// an argument.  It gives that form EVERY advantage the format cannot: n is the
// exact nearest integer, and both the multiply and the subtract are carried out
// at 4096 bits, which is exact for every q below (log2|n| + q <= 2263 < 4096).
// The ONLY error left is the width q of the constant.  If widening worked, the
// residual would come out accurate at some affordable q.
static const mpfr_prec_t kWide = 4096;

// x = m * 2^e, exactly, then rounded to DoubleDouble (which is the argument the
// library would actually be handed).  Returns the DD; leaves the exact value of
// THAT DD in X, so nothing downstream is scored against a value x never had.
static DoubleDouble set_pt(mpfr_ptr X, const char* m, long e) {
    mpfr_set_str(X, m, 10, MPFR_RNDN);
    mpfr_mul_2si(X, X, e, MPFR_RNDN);
    DoubleDouble a(mpfr_get_d(X, MPFR_RNDN));
    mpfr_sub_d(X, X, a.hi, MPFR_RNDN);
    a.lo = mpfr_get_d(X, MPFR_RNDN);
    mpfr_set_d(X, a.hi, MPFR_RNDN);
    mpfr_add_d(X, X, a.lo, MPFR_RNDN);
    return a;
}

static int widen_mode() {
    static const int kQ[] = { 106, 212, 318, 424, 1239 };
    // Every DD value is m * 2^e.  The last two are the arguments that pin the
    // problem: the published IEEE-double worst case, and the argmax of
    // log2(|x|/|f|) over DoubleDouble that sized the shipped table
    // (include/xp/trig_reduction_data.hpp).
    struct Pt { const char* label; const char* m; long e; };
    static const Pt kPts[] = {
        { "182.21237390820801", "3205513981387887",                     -44 },
        { "1e15",               "1000000000000000",                       0 },
        { "1e60",               "1401298464324817",                     149 },
        { "FP64 worst case",    "6381956970095103",                     797 },
        { "DD table pin",       "54146676858324748940541860376576",     917 },
    };

    mpfr_t X, TP, N, T, RT, RQ, D;
    mpfr_inits2(kWide, X, TP, N, T, RT, RQ, D, (mpfr_ptr)0);
    mpfr_const_pi(TP, MPFR_RNDN);
    mpfr_mul_2ui(TP, TP, 1, MPFR_RNDN);          // 2pi to 4096 bits

    std::printf("negative control: n exact, arithmetic exact at %ld bits, only the\n"
                "2pi constant finite.  q is its width in bits.\n\n", (long)kWide);
    std::printf("  %-19s %10s %10s", "x", "log2|x|", "log2|x/r|");
    for (int qi = 0; qi < (int)(sizeof(kQ)/sizeof(*kQ)); ++qi)
        std::printf("   q=%-4d", kQ[qi]);
    std::printf("\n");
    std::printf("  %-19s %10s %10s", "", "", "");
    for (int qi = 0; qi < (int)(sizeof(kQ)/sizeof(*kQ)); ++qi) std::printf("  %7s", "bits");
    std::printf("\n");

    for (const Pt& pt : kPts) {
        set_pt(X, pt.m, pt.e);
        mpfr_div(T, X, TP, MPFR_RNDN);
        mpfr_rint(N, T, MPFR_RNDN);              // n, exact
        mpfr_mul(T, N, TP, MPFR_RNDN);
        mpfr_sub(RT, X, T, MPFR_RNDN);           // r_true

        std::printf("  %-19s %10.3f %10.3f", pt.label, log2abs(X),
                    log2abs(X) - log2abs(RT));
        for (int qi = 0; qi < (int)(sizeof(kQ)/sizeof(*kQ)); ++qi) {
            mpfr_t C;
            mpfr_init2(C, kQ[qi]);
            mpfr_const_pi(C, MPFR_RNDN);
            mpfr_mul_2ui(C, C, 1, MPFR_RNDN);    // 2pi rounded to q bits
            mpfr_mul(T, N, C, MPFR_RNDN);        // exact: log2|n| + q < 4096
            mpfr_sub(RQ, X, T, MPFR_RNDN);       // exact
            mpfr_sub(D, RQ, RT, MPFR_RNDN);
            std::printf("  %7.1f", mpfr_zero_p(D) ? 9999.0 : log2abs(RT) - log2abs(D));
            mpfr_clear(C);
        }
        std::printf("\n");
    }
    std::printf("\n  Each 106 bits of constant buys 106 bits of residual and no more:\n"
                "  bits = q - log2(|x|/|r|), and log2(|x|/|r|) belongs to x, not to q.\n"
                "  D = 1129.4 is the MEASURED max of log2(|x|/|f|) over DoubleDouble\n"
                "  (include/xp/trig_reduction_data.hpp), so reaching p+4 = 110 bits at\n"
                "  every DD argument needs q >= 1239.  The shipped Payne-Hanek table is\n"
                "  52 chunks x 24 = 1248 bits: THE SAME STRING.  What differs is that\n"
                "  Payne-Hanek reads a WINDOW of it, guard/24 = 10 chunks wide, while\n"
                "  `a - 2pi*n` must multiply all 1248 bits by n.\n\n");

    // ...and that multiply has nowhere to happen.  n needs log2|n| bits of
    // INTEGER; DoubleDouble carries 106 of significand.  This arm is the
    // library's own quotient, not a model of it.
    std::printf("  the quotient cannot carry n either (this is dd_math.hpp's own\n"
                "  round_to_nearest_int(divide(a, 2pi)), scored against exact n):\n");
    std::printf("  %-19s %10s %10s %10s\n", "x", "log2|n|",
                "log2 err", "predicted");
    const DoubleDouble pi2 = multiply_scalar(DoubleDouble_pi(), 2.0);
    for (const Pt& pt : kPts) {
        const DoubleDouble a  = set_pt(X, pt.m, pt.e);
        mpfr_div(T, X, TP, MPFR_RNDN);
        mpfr_rint(N, T, MPFR_RNDN);
        const DoubleDouble nd = round_to_nearest_int(divide(a, pi2));
        mpfr_set_d(D, nd.hi, MPFR_RNDN);
        mpfr_add_d(D, D, nd.lo, MPFR_RNDN);
        mpfr_sub(D, D, N, MPFR_RNDN);
        if (mpfr_zero_p(D))
            std::printf("  %-19s %10.3f %10s %10.3f\n", pt.label, log2abs(N),
                        "exact", log2abs(N) - 106.0);
        else
            std::printf("  %-19s %10.3f %10.3f %10.3f\n", pt.label, log2abs(N),
                        log2abs(D), log2abs(N) - 106.0);
    }
    std::printf("\n  `predicted` is log2|n| - 106: the absolute error a 106-bit relative\n"
                "  quotient must have at that magnitude.  The measured error tracks it,\n"
                "  so this is the DIVIDE running out of significand -- not the rounding.\n"
                "  (round_to_nearest_int is exact at EVERY magnitude since the KI-14\n"
                "  sibling audit, dd_math.hpp:492.  An earlier draft of\n"
                "  include/xp/trig_reduction.hpp claimed it saturated; it does not.)\n"
                "  An error of e in n is an error of e*2pi in a residual that is O(1),\n"
                "  so the old form dies at log2|n| = 106, i.e. |x| ~ 2^108.7, for ANY\n"
                "  width of constant.  Widening and Payne-Hanek are not two ways to buy\n"
                "  the same thing: only one of them avoids forming n.\n");
    mpfr_clears(X, TP, N, T, RT, RQ, D, (mpfr_ptr)0);
    return 0;
}

// --range: what does refusing to reduce above a magnitude cost?
static int range_mode(double lo, double hi, long n) {
    mpfr_t X, S, C;
    mpfr_inits2(kPrec, X, S, C, (mpfr_ptr)0);
    std::mt19937_64 g(20260911);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double elo = std::log2(lo), ehi = std::log2(hi);
    const DoubleDouble bail_cos(1.0), bail_sin(0.0);
    double phs = 0, phc = 0, bs = 0, bc = 0, worst = 0;
    long   used = 0, bailwrong = 0;
    for (long i = 0; i < n; ++i) {
        const double e = elo + U(g) * (ehi - elo);
        double       x = std::ldexp(1.0 + U(g), (int)e - 1);
        if (!std::isfinite(x) || x < lo || x > hi) continue;
        if (U(g) < 0.5) x = -x;
        mpfr_set_d(X, x, MPFR_RNDN);
        mpfr_sin(S, X, MPFR_RNDN);
        mpfr_cos(C, X, MPFR_RNDN);
        DoubleDouble cc, ss;
        sincos_nq(DoubleDouble(x), 5, cc, ss);      // 5 is the shipped nq
        const double us = ulps_of(S, ss), uc = ulps_of(C, cc);
        if (us > phs) { phs = us; worst = x; }
        if (uc > phc) phc = uc;
        const double zs = ulps_of(S, bail_sin), zc = ulps_of(C, bail_cos);
        if (zs > bs) bs = zs;
        if (zc > bc) bc = zc;
        if (zs > 1.0 || zc > 1.0) ++bailwrong;
        ++used;
    }
    std::printf("n = %ld drawn log-uniformly from [%g, %g]\n", used, lo, hi);
    std::printf("  Payne-Hanek  max sin %10.4g ulps   max cos %10.4g ulps"
                "   (worst x %.17g)\n", phs, phc, worst);
    std::printf("  identity (1,0) max sin %8.4g ulps   max cos %10.4g ulps"
                "   (%ld of %ld over 1 ulp)\n", bs, bc, bailwrong, used);
    mpfr_clears(X, S, C, (mpfr_ptr)0);
    return 0;
}

// ---------------------------------------------------------------------------
// --fp32 : the same stage attribution for QF and TF.
// ---------------------------------------------------------------------------

static double bits_of(mpfr_srcptr exact, const QuadFloat& v) {
    mpfr_t e;
    mpfr_init2(e, kPrec);
    mpfr_set_flt(e, v.f0, MPFR_RNDN);
    mpfr_add_d(e, e, (double)v.f1, MPFR_RNDN);
    mpfr_add_d(e, e, (double)v.f2, MPFR_RNDN);
    mpfr_add_d(e, e, (double)v.f3, MPFR_RNDN);
    mpfr_sub(e, e, exact, MPFR_RNDN);
    const double r = log2abs(exact) - log2abs(e);
    mpfr_clear(e);
    return r;
}

static double bits_of(mpfr_srcptr exact, const TripleFloat& v) {
    mpfr_t e;
    mpfr_init2(e, kPrec);
    mpfr_set_flt(e, v.f0, MPFR_RNDN);
    mpfr_add_d(e, e, (double)v.f1, MPFR_RNDN);
    mpfr_add_d(e, e, (double)v.f2, MPFR_RNDN);
    mpfr_sub(e, e, exact, MPFR_RNDN);
    const double r = log2abs(exact) - log2abs(e);
    mpfr_clear(e);
    return r;
}

// The best value the format can hold, by greedy round-to-nearest limb.  This
// is the reference the reduction is judged against: a PERFECT reduction that
// still has to land in a QuadFloat scores exactly this.
static QuadFloat qf_best(mpfr_srcptr v) {
    mpfr_t t;
    mpfr_init2(t, kPrec);
    mpfr_set(t, v, MPFR_RNDN);
    float c[4];
    for (int i = 0; i < 4; ++i) {
        c[i] = mpfr_get_flt(t, MPFR_RNDN);
        mpfr_sub_d(t, t, (double)c[i], MPFR_RNDN);
    }
    mpfr_clear(t);
    return QuadFloat(c[0], c[1], c[2], c[3]);
}

static TripleFloat tf_best(mpfr_srcptr v) {
    mpfr_t t;
    mpfr_init2(t, kPrec);
    mpfr_set(t, v, MPFR_RNDN);
    float c[3];
    for (int i = 0; i < 3; ++i) {
        c[i] = mpfr_get_flt(t, MPFR_RNDN);
        mpfr_sub_d(t, t, (double)c[i], MPFR_RNDN);
    }
    mpfr_clear(t);
    return TripleFloat(c[0], c[1], c[2]);
}

static int subnormal_limbs(const QuadFloat& v) {
    const float w[4] = { v.f0, v.f1, v.f2, v.f3 };
    int n = 0;
    for (int i = 0; i < 4; ++i)
        if (w[i] != 0.0f && std::fabs(w[i]) < 1.17549435e-38f) ++n;
    return n;
}

// qf_math.hpp:sincos lines 1262-1284, verbatim, with nq lifted to a parameter
// and a tally of the subnormal limbs the arm touches.  The reduction is NOT
// part of this: every arm is handed the same r_mod, so nothing the sweep shows
// can be blamed on Payne-Hanek.
static void qf_series_nq(QuadFloat r_mod, int nq, QuadFloat& sin_r,
                         QuadFloat& cos_r, int* sub) {
    const int   itrmx = 100;
    const float eps   = 1.0e-28f;
    if (sub) *sub = 0;
    QuadFloat r  = mul_pwr2(r_mod, ldexpf(1.0f, -nq));
    QuadFloat r2 = multiply(r, r);
    if (sub) *sub += subnormal_limbs(r) + subnormal_limbs(r2);
    sin_r = r;
    cos_r = QuadFloat(1.0f);
    QuadFloat sterm = r, cterm = QuadFloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (sub) *sub += subnormal_limbs(sterm) + subnormal_limbs(sin_r)
                       + subnormal_limbs(cterm) + subnormal_limbs(cos_r);
        if (detail::fabs(sterm.f0) < eps * detail::fabs(sin_r.f0) &&
            detail::fabs(cterm.f0) < eps) break;
    }
    for (int j = 0; j < nq; ++j) {
        const QuadFloat ns = mul_pwr2(multiply(sin_r, cos_r), 2.0f);
        const QuadFloat nc = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = ns;
        cos_r = nc;
        if (sub) *sub += subnormal_limbs(sin_r) + subnormal_limbs(cos_r);
    }
}

// tf_math.hpp:sincos lines 1045-1073, verbatim, with nq lifted.  Note TF
// divides by 2^nq and squares with sqr(), where QF uses mul_pwr2 and
// multiply(r,r) -- the arms are each backend's own code, not a shared model.
static void tf_series_nq(TripleFloat r_mod, int nq, TripleFloat& sin_r,
                         TripleFloat& cos_r) {
    TripleFloat r  = divide_scalar(r_mod, float(1 << nq));
    TripleFloat r2 = sqr(r);
    sin_r = r;
    cos_r = TripleFloat(1.0f);
    TripleFloat term_sin = r, term_cos = TripleFloat(1.0f);
    int k = 1;
    while (k < 64 && (abs(term_sin).f0 > 1.0e-21f * abs(sin_r).f0 ||
                      abs(term_cos).f0 > 1.0e-21f * abs(cos_r).f0)) {
        term_sin = divide_scalar(multiply(term_sin, r2), -float((2*k) * (2*k+1)));
        term_cos = divide_scalar(multiply(term_cos, r2), -float((2*k-1) * (2*k)));
        sin_r = add(sin_r, term_sin);
        cos_r = add(cos_r, term_cos);
        k++;
    }
    for (int i = 0; i < nq; i++) {
        TripleFloat s = multiply(sin_r, cos_r);
        s = add(s, s);
        const TripleFloat c = subtract(sqr(cos_r), sqr(sin_r));
        sin_r = s;
        cos_r = c;
    }
}

static int fp32_mode() {
    mpfr_t PIO2, X, F, N, R, HP, T;
    mpfr_inits2(kPrec, PIO2, X, F, N, R, HP, T, (mpfr_ptr)0);
    mpfr_const_pi(PIO2, MPFR_RNDN);
    mpfr_div_ui(PIO2, PIO2, 2, MPFR_RNDN);
    mpfr_set(HP, PIO2, MPFR_RNDN);

    // The three grid points that carry QF's post-Payne-Hanek family-(5) peak.
    static const struct { const char* label; double x; } kPts[] = {
        { "1652  pi/2",   1.5707963267948966 },
        { "1160  182.21", 182.21237390820801 },
        { " 870  91.106", 91.106186954104004 },
    };

    std::printf("STAGE SPLIT -- is it the reduction?  (r_best = the best the "
                "format can hold)\n");
    std::printf("%-14s %10s %8s %8s %8s %8s %8s\n",
                "point", "log2|x/r|", "pio2", "fr", "r_mod", "r_best", "deficit");
    for (const auto& p : kPts) {
        mpfr_set_d(X, p.x, MPFR_RNDN);
        mpfr_div(F, X, PIO2, MPFR_RNDN);
        mpfr_rint(N, F, MPFR_RNDN);
        mpfr_sub(F, F, N, MPFR_RNDN);
        mpfr_mul(R, F, PIO2, MPFR_RNDN);

        const QuadFloat a((double)p.x);
        const float     win[4] = { a.f0, a.f1, a.f2, a.f3 };
        float           f[5];
        detail::xp_ph_reduce<float>(win, 4, detail::kPhGuardQF,
                                    detail::kPhChunksQF, f, 5);
        QuadFloat pio2 = QuadFloat(detail::xp_ph_pio2_f(0));
        for (int k = 1; k < detail::kPhPio2WordsF; ++k)
            pio2 = add(pio2, QuadFloat(detail::xp_ph_pio2_f(k)));
        QuadFloat fr = QuadFloat(f[0]);
        for (int k = 1; k < 5; ++k) fr = add(fr, QuadFloat(f[k]));
        const QuadFloat rmod  = multiply(fr, pio2);
        const QuadFloat rbest = qf_best(R);

        std::printf("%-14s %10.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
                    p.label, log2abs(X) - log2abs(R), bits_of(HP, pio2),
                    bits_of(F, fr), bits_of(R, rmod), bits_of(R, rbest),
                    bits_of(R, rbest) - bits_of(R, rmod));
    }

    std::printf("\nnq SWEEP -- same r_mod in every arm, so this cannot be the "
                "reduction.  bits of sin.\n");
    std::printf("%-14s %8s", "point", "log2|r|");
    for (int nq = 0; nq <= 7; ++nq) std::printf(" %7s%d", "nq=", nq);
    std::printf("\n");
    for (const auto& p : kPts) {
        mpfr_set_d(X, p.x, MPFR_RNDN);
        mpfr_div(F, X, PIO2, MPFR_RNDN);
        mpfr_rint(N, F, MPFR_RNDN);
        mpfr_sub(F, F, N, MPFR_RNDN);
        mpfr_mul(R, F, PIO2, MPFR_RNDN);
        const QuadFloat rbest = qf_best(R);
        // Reference is sin of the value the series ACTUALLY receives, so the
        // rounding of r into the format is not charged to the series.
        mpfr_set_flt(T, rbest.f0, MPFR_RNDN);
        mpfr_add_d(T, T, (double)rbest.f1, MPFR_RNDN);
        mpfr_add_d(T, T, (double)rbest.f2, MPFR_RNDN);
        mpfr_add_d(T, T, (double)rbest.f3, MPFR_RNDN);
        mpfr_sin(T, T, MPFR_RNDN);
        std::printf("%-14s %8.2f", p.label, log2abs(R));
        for (int nq = 0; nq <= 7; ++nq) {
            QuadFloat s, c;
            qf_series_nq(rbest, nq, s, c, nullptr);
            std::printf(" %8.1f", bits_of(T, s));
        }
        std::printf("   (QF ships nq=5)\n");
    }

    // Magnitude scan.  A full-width value neither format can hold exactly:
    // six random limbs, so QF (4) and TF (3) both have to round.
    std::printf("\nMAGNITUDE SCAN -- worst of %d full-width r per row.  QF "
                "nq=0 vs the shipped nq=5,\n", 64);
    std::printf("with the mean count of subnormal FP32 limbs each arm touched, "
                "and TF for contrast.\n");
    std::printf("%-9s %10s %7s %10s %7s %9s %10s\n",
                "log2|r|", "QF nq=0", "sub", "QF nq=5", "sub", "loss", "TF nq=4");
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> um(1.0, 2.0);
    for (int e = -24; e >= -64; e -= 4) {
        double w0 = 1e9, w5 = 1e9, wt = 1e9, s0 = 0.0, s5 = 0.0;
        int    n = 0;
        for (int trial = 0; trial < 64; ++trial) {
            mpfr_set_zero(R, 1);
            for (int i = 0; i < 6; ++i) {
                mpfr_set_d(T, um(rng), MPFR_RNDN);
                mpfr_mul_2si(T, T, e - 24 * i, MPFR_RNDN);
                mpfr_add(R, R, T, MPFR_RNDN);
            }
            const QuadFloat   qa = qf_best(R);
            const TripleFloat ta = tf_best(R);

            mpfr_set_flt(T, qa.f0, MPFR_RNDN);
            mpfr_add_d(T, T, (double)qa.f1, MPFR_RNDN);
            mpfr_add_d(T, T, (double)qa.f2, MPFR_RNDN);
            mpfr_add_d(T, T, (double)qa.f3, MPFR_RNDN);
            mpfr_sin(T, T, MPFR_RNDN);
            QuadFloat s, c;
            int       sub0 = 0, sub5 = 0;
            qf_series_nq(qa, 0, s, c, &sub0);
            const double b0 = bits_of(T, s);
            qf_series_nq(qa, 5, s, c, &sub5);
            const double b5 = bits_of(T, s);

            mpfr_set_flt(X, ta.f0, MPFR_RNDN);
            mpfr_add_d(X, X, (double)ta.f1, MPFR_RNDN);
            mpfr_add_d(X, X, (double)ta.f2, MPFR_RNDN);
            mpfr_sin(X, X, MPFR_RNDN);
            TripleFloat ts, tc;
            tf_series_nq(ta, 4, ts, tc);
            const double bt = bits_of(X, ts);

            if (b0 < w0) w0 = b0;
            if (b5 < w5) w5 = b5;
            if (bt < wt) wt = bt;
            s0 += sub0;
            s5 += sub5;
            ++n;
        }
        std::printf("%-9d %10.2f %7.1f %10.2f %7.1f %9.2f %10.2f\n",
                    e, w0, s0 / n, w5, s5 / n, w0 - w5, wt);
    }
    std::printf("\nSubnormal limbs are NOT sufficient on their own: the nq=0 "
                "arm touches as many at\nthe bottom of the range and loses "
                "nothing, because there they sit in the r^3/6\ntail.  What "
                "costs bits is subnormal limbs in u and u^2 themselves, which "
                "the\ndoublings scale back up by 2^nq.\n");

    mpfr_clears(PIO2, X, F, N, R, HP, T, (mpfr_ptr)0);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: probe_trig_stages <x> [x ...]\n"
                     "       probe_trig_stages --range <lo> <hi> <n>\n"
                     "       probe_trig_stages --widen\n"
                     "       probe_trig_stages --fp32\n"
                     "  bits of each reduction stage, then the nq sweep\n");
        return 2;
    }
    mpfr_set_default_prec(kPrec);
    if (std::string(argv[1]) == "--widen") return widen_mode();
    if (std::string(argv[1]) == "--fp32")  return fp32_mode();
    if (std::string(argv[1]) == "--range") {
        if (argc != 5) { std::fprintf(stderr, "--range wants <lo> <hi> <n>\n"); return 2; }
        return range_mode(strtod(argv[2], nullptr), strtod(argv[3], nullptr),
                          strtol(argv[4], nullptr, 10));
    }
    mpfr_t X, S, C, PIO2, R, N, F, FE;
    mpfr_inits2(kPrec, X, S, C, PIO2, R, N, F, FE, (mpfr_ptr)0);
    mpfr_const_pi(PIO2, MPFR_RNDN);
    mpfr_div_ui(PIO2, PIO2, 2, MPFR_RNDN);

    for (int i = 1; i < argc; ++i) {
        const double x = strtod(argv[i], nullptr);
        const DoubleDouble a(x);

        mpfr_set_d(X, x, MPFR_RNDN);
        mpfr_sin(S, X, MPFR_RNDN);
        mpfr_cos(C, X, MPFR_RNDN);
        mpfr_div(F, X, PIO2, MPFR_RNDN);       // exact f = frac(x*2/pi), |f| <= 1/2
        mpfr_rint(N, F, MPFR_RNDN);
        mpfr_sub(F, F, N, MPFR_RNDN);
        mpfr_mul(R, F, PIO2, MPFR_RNDN);       // exact r_mod

        DoubleDouble rNew, rOld;
        double       f[3];
        const int    jNew = reduce_new(a, rNew, f);
        const int    jOld = reduce_old(a, rOld);

        mpfr_set_d(FE, f[0], MPFR_RNDN);
        mpfr_add_d(FE, FE, f[1], MPFR_RNDN);
        mpfr_add_d(FE, FE, f[2], MPFR_RNDN);
        mpfr_sub(FE, FE, F, MPFR_RNDN);
        const DoubleDouble fdd =
            add(add(DoubleDouble(f[0]), DoubleDouble(f[1])), DoubleDouble(f[2]));

        std::printf("x = %.17g   j old/new = %d/%d   log2|r_mod| = %.3f\n",
                    x, jOld, jNew, log2abs(R));
        std::printf("   reduction   old %7.2f bits    new %7.2f bits"
                    "   (p = 106 is the cap)\n",
                    bits_of(R, rOld), bits_of(R, rNew));
        std::printf("   stages      f %7.2f   fdd %7.2f   r_mod %7.2f bits\n",
                    log2abs(F) - log2abs(FE), bits_of(F, fdd), bits_of(R, rNew));
        std::printf("   nq sweep (same reduction in every arm):\n");
        for (int nq = 0; nq <= 8; ++nq) {
            DoubleDouble cc, ss;
            sincos_nq(a, nq, cc, ss);
            std::printf("      nq=%d   sin %12.5g   cos %12.5g ulps%s\n",
                        nq, ulps_of(S, ss), ulps_of(C, cc),
                        nq == 5 ? "   <- shipped" : "");
        }
    }
    mpfr_clears(X, S, C, PIO2, R, N, F, FE, (mpfr_ptr)0);
    return 0;
}
