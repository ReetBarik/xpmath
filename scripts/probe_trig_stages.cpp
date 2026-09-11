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
// Build (needs MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_trig_stages.cpp -o /tmp/probe_trig_stages -lmpfr -lgmp
//   /tmp/probe_trig_stages -10000 1.75 -7.65 182.21237390820801
//   /tmp/probe_trig_stages --range 1e60 1.7976931348623157e308 4000
//   /tmp/probe_trig_stages --widen

#include <xp/dd_math.hpp>

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: probe_trig_stages <x> [x ...]\n"
                     "       probe_trig_stages --range <lo> <hi> <n>\n"
                     "       probe_trig_stages --widen\n"
                     "  bits of each reduction stage, then the nq sweep\n");
        return 2;
    }
    mpfr_set_default_prec(kPrec);
    if (std::string(argv[1]) == "--widen") return widen_mode();
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
