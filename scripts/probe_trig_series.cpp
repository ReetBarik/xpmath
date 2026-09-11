// probe_trig_series.cpp — pick the trig SERIES CORE by measurement.
//
// WHY THIS EXISTS.  Phase 3 (scripts/probe_trig_stages.cpp) attributed the
// post-Payne-Hanek trig residual to the nq scale-down/double-back chain and NOT
// to the argument reduction: holding r_mod bit-identical and lifting nq out
// moves QF point 1652 from 110.3 bits at nq=0 to 91.6 at the shipped nq=5.
// That says WHERE the error is.  It does not say what to replace it with.
//
// Two replacements were proposed, and both were measured against a Decimal
// model rather than against the shipped headers:
//
//   v-form   carry v = 1 - cos instead of cos, with v' = 2s^2 and s' = 2s(1-v),
//            building v from its own series (v = r^2/2 - r^4/24 + ...) so the
//            leading 1 never enters.  Keeps nq and ~7 terms.  Modelled 1.97/1.09.
//   nq -> 0  evaluate the series directly at r_mod.  ~14 terms, no recurrence
//            at all.  Modelled 1.59/1.42.
//
// A Decimal model is not the shipped arithmetic — it has no sloppy add, no
// Dekker splitter, no FP32 subnormal floor.  This probe re-runs both candidates
// IN THE SHIPPED PRIMITIVES, on the real grid families, scored in ULPS against
// MPFR at 400 bits, so the choice is made on the same metric the sweep uses.
//
// WHAT IS HELD FIXED.  Every arm receives the SAME (j, r_mod) from the shipped
// Payne-Hanek reduction, and every arm applies the SAME quadrant table.  Only
// the series core varies.  Any difference between arms is therefore the core.
//
// THE ARMS
//   ship      the shipped code: cos carried with its leading 1, doubled by
//             c' = c^2 - s^2, at each backend's shipped nq.
//   nq=K      ship with nq lifted to K.  K=0 is the "nq -> 0" candidate.
//   vform     v = 1 - cos carried, v' = 2s^2, s' = 2s(1-v), at the shipped nq.
//   vform0    the v-form at nq=0 (degenerates to: build v, return 1-v).
//   fact      ship with c' = (c-s)(c+s) instead of c^2 - s^2.  dd_complex.hpp
//             :569-579 already uses that form and the core does not; the plan
//             asks for it to be tested.
//
// SCORING.  ulps = |got - ref| / (|ref| * 2^-p), p = 106/48/72/96 for
// DD/FF/TF/QF — ulps_scalar (sweep_accuracy.cpp:1328) verbatim, so the numbers
// here are comparable to the sweep's.  `ref` is sin/cos of the EXACT value of
// the stored argument (a.hi + a.lo, exactly), never of the decimal that was
// typed: that is the f(x_grid) vs f(x_stored) trap b7c4b64 fixed in the sweep
// and the plan's own author hit while modelling this phase.
//
// Points whose reference is zero are dropped, as unscorable, exactly as the
// sweep drops them; a near-zero of sin or cos has no meaningful relative error.
//
// THE ARGUMENT SET is the sweep's own real grid, rebuilt here, restricted to
// the families that carry the defect.  Measured at HEAD (Phase 3 merged):
//
//   backend   linear median / max      where the worst rows are
//   DD        4.237 / 42.4             ordinary arguments, 756 of 961 > 1 ulp
//   FF        2.138 / 14.5             same
//   TF        0.455 / 5.37             same
//   QF        1.021 /  9.10            QF's worst is NOT here: it is the
//                                      `ulp` family at tiny |r| (1029.9 ulps
//                                      at x = 91.106), the FP32 subnormal
//                                      floor probe_trig_stages --fp32 found.
//
// That split is the reason the two candidates are not interchangeable.  The
// v-form still evaluates at u = r_mod/2^nq, so it cannot touch a defect whose
// mechanism is u's limbs going subnormal; nq -> 0 removes the scale-down and
// should.  The probe prints both families so that prediction is checked rather
// than assumed.
//
// Build (needs MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_trig_series.cpp -o /tmp/probe_trig_series -lmpfr -lgmp
//   /tmp/probe_trig_series            # all backends, all arms, all families
//   /tmp/probe_trig_series --terms    # series length each arm actually runs
//   /tmp/probe_trig_series --win-nq 0 --win-ship    # point the third arm elsewhere
//
// ===========================================================================
// WHAT IT MEASURED, and what Phase 2 therefore shipped
// ===========================================================================
//
// DECISION METRIC: worst ulps over the grid at condition number <= 4, worst of
// sin and cos.  kappa = |x/tan x| (sin) and |x tan x| (cos) — the amplification
// of a one-ulp perturbation of the argument, the same quantity the sweep's
// derived bound is built from.  Points above kappa = 4 are near-zeros, where
// relative error diverges for reasons no series core can address; scoring the
// choice on them would pick whichever arm happened to land better on a cliff.
//
//   arm                    DD      FF      TF          QF         terms (DD)
//   ship (shipped nq)    42.44   14.50   2.6e7       1.3e15         7
//   fact (c-s)(c+s)      60.33   23.97   2.6e7       1.3e15         7
//   vform @ shipped nq    1.74    1.95   2.6e7       1.3e15         7
//   vform nq=1            1.82    2.34   2.09e6      3.5e13        12
//   vform nq=0            3.40    1.78    0.593       0.98         14
//
// 1. `fact` IS REJECTED.  It is worse than shipped in every family and on every
//    backend, not merely no better.  dd_complex.hpp:569-579 may still want it
//    for its own reasons; the core does not.
//
// 2. THE V-FORM IS THE FIX, on all four backends.  It is what removes the 2^nq
//    amplification, and at the shipped nq it costs nothing: the convergence test
//    still stops at 7 terms.
//
// 3. THE PLAN'S "nq -> 0" IS RIGHT FOR THE FP32 BACKENDS AND WRONG FOR DD.
//    Measured, not derived, and the two halves have different mechanisms:
//
//    DD: nq=5 beats nq=0 (1.74 vs 3.40).  Fourteen series terms accumulate more
//    rounding than seven terms plus five v-form doublings, because once the
//    doublings are done in v they are no longer the dominant error term.  nq=5
//    is also the cheaper of the two.  DD KEEPS nq = 5.
//
//    FP32: the scale-down itself is the defect, so no core can survive it.
//    u = r_mod/2^nq puts a QuadFloat's 4th limb at 2^(e-nq-72), under
//    FLT_MIN = 2^-126 once e < -49; the doublings then scale the shed bits back
//    up by 2^nq.  A falsifiable prediction was made and confirmed before any
//    header was touched: the v-form cannot fix a scale-down defect because it
//    still evaluates at u.  QF's two worst cells came out BIT-IDENTICAL to
//    shipped under `vform nq=5` — 1029.8804 and 405.0246 ulps, to the digit —
//    and fell to 43.3832 and 1.4575 only when nq went to 0.  FF/TF/QF USE nq = 0.
//
// 4. THE STOP CONDITION IS MET FOR DD, and the third arm is why.  `exact red`
//    re-runs the chosen core on the exact r_mod computed at 400 bits.  Over the
//    30 cells that remain above 1 ulp at kappa <= 4, it returns the IDENTICAL
//    value at all 30 — zero improve by even 0.1 ulp.  No reduction headroom
//    remains at those arguments; what is left is the core and the format.
//    (Run at `vform nq=0` the same arm DOES show headroom — 2.4552 -> 1.0298 at
//    x = 3.5 — so the check is not vacuous.)
//
// FIDELITY CHECK.  The `ship` arm is compared limb-for-limb against xp::sincos
// on every grid point, because an arm that is not actually the shipped code
// makes every row above meaningless.  It reports 8/1636 (FF), 6/1636 (TF),
// 8/1636 (QF) mismatches, ALL of them at the FP32 magnitude bails (first at
// x = 1e30 for FF/QF, x = 4.0156e151 for TF), which this probe does not
// replicate.  DD matches everywhere.  Those points are state U in the sweep and
// carry no verdict, so they cannot inform the choice either way.

#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>
#include <xp/qf_math.hpp>
#include <xp/tf_math.hpp>

#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace xp;

static const mpfr_prec_t kPrec = 400;

// ===========================================================================
// The grid, rebuilt from sweep_accuracy.cpp:build_real_grid
// ===========================================================================

struct Pt { double x; const char* family; };

static double pow10i(int e) { return std::pow(10.0, double(e)); }

static std::vector<Pt> build_grid() {
    std::vector<Pt> g;
    // (1) log sweep, half-decade steps, both signs.
    for (int e = -30; e <= 30; ++e) {
        const double v = pow10i(e);
        g.push_back({ v, "log"}); g.push_back({-v, "log"});
        if (e < 30) {
            const double h = 3.1622776601683795 * v;
            g.push_back({ h, "log"}); g.push_back({-h, "log"});
        }
    }
    // (2) dense linear sweep, x = i/20 over [-8, 8].
    for (int i = -160; i <= 160; ++i) g.push_back({double(i) / 20.0, "linear"});
    // (3) +-2 ulp around 0, +-1, +-pi/2, +-k*pi for k = 1..100.
    std::vector<double> anchors;
    anchors.push_back(0.0);
    anchors.push_back(1.0);    anchors.push_back(-1.0);
    anchors.push_back(M_PI_2); anchors.push_back(-M_PI_2);
    for (int k = 1; k <= 100; ++k) {
        anchors.push_back(double(k) * M_PI); anchors.push_back(-double(k) * M_PI);
    }
    for (size_t i = 0; i < anchors.size(); ++i)
        for (int n = -2; n <= 2; ++n) {
            double v = anchors[i];
            for (int s = 0; s < (n < 0 ? -n : n); ++s)
                v = std::nextafter(v, n < 0 ? -HUGE_VAL : HUGE_VAL);
            g.push_back({v, "ulp"});
        }
    // (5) hard reduction cases -- the continued-fraction convergents of 2/pi.
    static const double kHard[] = {
        0x1.921fb54442d18p+0,   0x1.921fb54442d18p+1,   0x1.5fdbbe9bba775p+3,
        0x1.5801201165294p+8,   0x1.94d600033aacep+15,  0x1.fcd1800015de1p+17,
        0x1.17e27fffff624p+19,  0x1.4ac55bfffffd7p+22,  0x1.4665d1ffffffep+25,
        0x1.d4ec654p+26,        0x1.fdb91f8p+28,        0x1.6fa37476p+31,
        0x1.39b821694p+34,      0x1.a41fc970f8p+39,     0x1.38a466d1e78p+41,
        0x1.04bd49b47d2p+43,    0x1.dbd58768f97p+45,    0x1.fc6d309f8914p+46,
        0x1.8577cec54ab8p+47,   0x1.5cba89af1f855p+52,  0x1.56a4aa740a5a7p+53,
        0x1.888ecd2edf9ccp+503, 0x1.88451b1af0ca3p+658, 0x1.6ac5b262ca1ffp+849,
    };
    for (size_t i = 0; i < sizeof(kHard)/sizeof(kHard[0]); ++i) {
        g.push_back({ kHard[i], "hardred"}); g.push_back({-kHard[i], "hardred"});
    }
    return g;
}

// ===========================================================================
// Scoring -- ulps_scalar (sweep_accuracy.cpp:1328) against MPFR at 400 bits
// ===========================================================================

// Sum an expansion into an MPFR at kPrec.  Every limb is a float or double and
// kPrec is far wider than the expansion, so each add is exact: this is the
// exact value of the stored number, not a rounding of it.
static void exact_of(mpfr_ptr out, const double* w, int n) {
    mpfr_set_d(out, w[0], MPFR_RNDN);
    for (int i = 1; i < n; ++i) mpfr_add_d(out, out, w[i], MPFR_RNDN);
}

// |got - ref| / (|ref| * 2^-p).  Returns -1.0 for an unscorable point, matching
// the sweep's kUnscorableUlps sentinel.
static double ulps_of(mpfr_srcptr ref, const double* w, int n, int p) {
    if (mpfr_zero_p(ref)) return -1.0;
    mpfr_t g, e;
    mpfr_inits2(kPrec, g, e, (mpfr_ptr)0);
    exact_of(g, w, n);
    for (int i = 0; i < n; ++i)
        if (std::isnan(w[i]) || std::isinf(w[i])) {
            mpfr_clears(g, e, (mpfr_ptr)0);
            return HUGE_VAL;
        }
    mpfr_sub(e, g, ref, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    mpfr_abs(g, ref, MPFR_RNDN);
    mpfr_div(e, e, g, MPFR_RNDN);
    mpfr_mul_2si(e, e, p, MPFR_RNDN);
    const double r = mpfr_get_d(e, MPFR_RNDN);
    mpfr_clears(g, e, (mpfr_ptr)0);
    return r;
}

// ===========================================================================
// DD arms
// ===========================================================================

// The Payne-Hanek half of dd_math.hpp:sincos, lifted verbatim, INCLUDING the
// 0.75 small-argument cut.  Every arm gets this same (j, r_mod).
static int dd_reduce(DoubleDouble a, DoubleDouble& r_mod) {
    if (detail::fabs(a.hi) <= 0.75) { r_mod = a; return 0; }
    const double win[2] = { a.hi, a.lo };
    double       f[3];
    const int    j = detail::xp_ph_reduce<double>(win, 2, detail::kPhGuardDD,
                                                  detail::kPhChunksDD, f, 3);
    const DoubleDouble pio2 =
        add(add(DoubleDouble(detail::xp_ph_pio2_d(0)),
                DoubleDouble(detail::xp_ph_pio2_d(1))),
            DoubleDouble(detail::xp_ph_pio2_d(2)));
    const DoubleDouble fdd =
        add(add(DoubleDouble(f[0]), DoubleDouble(f[1])), DoubleDouble(f[2]));
    return (r_mod = multiply(fdd, pio2)), j;
}

enum Arm { ARM_SHIP, ARM_VFORM, ARM_FACT };

// Shipped core (dd_math.hpp:960-990) with nq a parameter and, for ARM_FACT,
// the one line that differs.  Guards stripped: the probe drives finite
// ordinary arguments only, and every arm is stripped identically.
static void dd_core_ship(DoubleDouble r_mod, int nq, Arm arm,
                         DoubleDouble& sin_r, DoubleDouble& cos_r, int* terms) {
    const double eps = 1.0e-32;
    DoubleDouble r  = multiply_scalar(r_mod, 1.0 / (double)(1 << nq));
    DoubleDouble r2 = multiply(r, r);
    sin_r = r;             cos_r = DoubleDouble(1.0);
    DoubleDouble st = r,   ct    = DoubleDouble(1.0);
    int k = 1;
    for (; k <= 1000; ++k) {
        st = divide_scalar(multiply(st, r2), -(double)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, st);
        ct = divide_scalar(multiply(ct, r2), -(double)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, ct);
        if (detail::fabs(st.hi) < eps * detail::fabs(sin_r.hi) &&
            detail::fabs(ct.hi) < eps) break;
    }
    if (terms) *terms = k;
    for (int d = 0; d < nq; ++d) {
        const DoubleDouble ns = multiply_scalar(multiply(sin_r, cos_r), 2.0);
        const DoubleDouble nc =
            (arm == ARM_FACT)
                ? multiply(subtract(cos_r, sin_r), add(cos_r, sin_r))
                : subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = ns; cos_r = nc;
    }
}

// THE v-FORM.  v = 1 - cos is carried instead of cos, and built from its own
// series so the leading 1 is never present to be cancelled:
//
//     v(r) = r^2/2! - r^4/4! + r^6/6! - ...      (the cos series, minus its 1)
//     s(r) = r      - r^3/3! + r^5/5! - ...
//
// Doubling, from cos(2x) = 1 - 2 sin^2(x) and sin(2x) = 2 sin(x) cos(x):
//
//     v' = 2 s^2                 (no cancellation: v' is a square)
//     s' = 2 s (1 - v)           (1 - v is c, formed once per step)
//
// The error in v is RELATIVE to v, so the error it contributes to c = 1 - v is
// |v| times that -- bounded, because |r_mod| <= pi/4 caps the final v at
// 1 - cos(pi/4) = 0.293.  The shipped form instead carries c ~ 1 with a
// relative error that its own recurrence doubles at every step.
static void dd_core_vform(DoubleDouble r_mod, int nq,
                          DoubleDouble& sin_r, DoubleDouble& cos_r, int* terms) {
    const double eps = 1.0e-32;
    DoubleDouble r  = multiply_scalar(r_mod, 1.0 / (double)(1 << nq));
    DoubleDouble r2 = multiply(r, r);
    // v starts at its FIRST term, r^2/2, not at 1 - anything.
    DoubleDouble vt = divide_scalar(r2, 2.0);
    DoubleDouble v  = vt;
    DoubleDouble st = r;
    DoubleDouble s  = r;
    int k = 1;
    for (; k <= 1000; ++k) {
        st = divide_scalar(multiply(st, r2), -(double)((2*k) * (2*k + 1)));
        s  = add(s, st);
        // v_{k+1} = -v_k * r^2 / ((2k+1)(2k+2)):  r^2/2! -> -r^4/4! -> r^6/6!
        vt = divide_scalar(multiply(vt, r2), -(double)((2*k + 1) * (2*k + 2)));
        v  = add(v, vt);
        if (detail::fabs(st.hi) < eps * detail::fabs(s.hi) &&
            detail::fabs(vt.hi) < eps * detail::fabs(v.hi)) break;
    }
    if (terms) *terms = k;
    for (int d = 0; d < nq; ++d) {
        const DoubleDouble c  = subtract(DoubleDouble(1.0), v);
        const DoubleDouble ns = multiply_scalar(multiply(s, c), 2.0);
        const DoubleDouble nv = multiply_scalar(multiply(s, s), 2.0);
        s = ns; v = nv;
    }
    sin_r = s;
    cos_r = subtract(DoubleDouble(1.0), v);
}

// One finished (cos, sin) pair, quadrant table included -- identical in every
// arm, so the comparison is of the core and nothing else.
//
// THE SHIPPED GUARDS ARE KEPT, and keeping them is not cosmetic.  A first cut
// of this probe stripped them "because the probe only drives ordinary finite
// arguments", and the `ulp` family promptly read 8.1130e31 ulps -- which is
// exactly 2^106, i.e. `got == 0 against a nonzero ref` -- for every arm with
// nq >= 1 and for none at nq = 0.  That is the KI-12 band: the family carries
// +-2 ulp around zero, so a = 5e-324, and r = r_mod/2^nq underflows to zero
// before the first Taylor term.  It is what the guard exists to prevent, not a
// property of any candidate, and scoring it would have credited nq = 0 with
// fixing a defect the shipped code does not have.
static void dd_eval(DoubleDouble a, int nq, Arm arm, bool vform,
                    DoubleDouble& cosa, DoubleDouble& sina, int* terms) {
    if (terms) *terms = 0;
    if (a.hi == 0.0) { cosa = DoubleDouble(1.0); sina = DoubleDouble(0.0); return; }
    // KI-12 small-argument band, dd_math.hpp:857.  nq-dependent by construction.
    if (detail::fabs(a.hi) < (double)(1 << nq) * 2.2250738585072014e-308) {
        cosa = DoubleDouble(1.0); sina = a; return;
    }
    // The magnitude bail, dd_math.hpp:879.  Above it the shipped code returns
    // the identity point; the sweep reads those rows as unresolved.
    if (detail::fabs(a.hi) >= 1.0e60) {
        cosa = DoubleDouble(1.0); sina = DoubleDouble(0.0); return;
    }
    DoubleDouble r_mod;
    const int    j = dd_reduce(a, r_mod);
    DoubleDouble s, c;
    if (vform) dd_core_vform(r_mod, nq, s, c, terms);
    else       dd_core_ship (r_mod, nq, arm, s, c, terms);
    if (j == 0)      { cosa = c;         sina = s; }
    else if (j == 1) { cosa = negate(s); sina = c; }
    else if (j == 2) { cosa = negate(c); sina = negate(s); }
    else             { cosa = s;         sina = negate(c); }
}

// THE THIRD ARM, in the sense the exp commit used it: is what is left after a
// candidate the SERIES, or is it reduction headroom the candidate cannot see?
// This runs the same core on the EXACT r_mod -- computed at 400 bits from the
// exact stored argument and rounded once to DoubleDouble -- instead of on the
// reduction's r_mod.  If a candidate's residual is unchanged, the reduction has
// no headroom left at that point and the residual belongs to the core (or to
// the format).  Quadrant index comes from the exact reduction too.
static void dd_eval_exactred(DoubleDouble a, int nq, bool vform,
                             DoubleDouble& cosa, DoubleDouble& sina) {
    mpfr_t X, Q, T;
    mpfr_inits2(kPrec, X, Q, T, (mpfr_ptr)0);
    mpfr_set_d(X, a.hi, MPFR_RNDN);
    mpfr_add_d(X, X, a.lo, MPFR_RNDN);
    mpfr_const_pi(T, MPFR_RNDN);
    mpfr_div_2ui(T, T, 1, MPFR_RNDN);          // T = pi/2 at 400 bits
    mpfr_div(Q, X, T, MPFR_RNDN);
    mpfr_rint(Q, Q, MPFR_RNDN);                // n = nint(a / (pi/2))
    // n mod 4 via MPFR, not via a long: n reaches 6e59 at the magnitude bail
    // and mpfr_get_si would saturate silently.
    mpfr_t M, F;
    mpfr_inits2(kPrec, M, F, (mpfr_ptr)0);
    mpfr_set_ui(F, 4, MPFR_RNDN);
    mpfr_fmod(M, Q, F, MPFR_RNDN);
    long n = mpfr_get_si(M, MPFR_RNDN);
    mpfr_clears(M, F, (mpfr_ptr)0);
    mpfr_mul(Q, Q, T, MPFR_RNDN);
    mpfr_sub(X, X, Q, MPFR_RNDN);              // X = exact r_mod
    DoubleDouble r_mod(mpfr_get_d(X, MPFR_RNDN));
    mpfr_sub_d(X, X, r_mod.hi, MPFR_RNDN);
    r_mod.lo = mpfr_get_d(X, MPFR_RNDN);
    mpfr_clears(X, Q, T, (mpfr_ptr)0);
    const int j = (int)(((n % 4) + 4) % 4);
    DoubleDouble s, c;
    if (vform) dd_core_vform(r_mod, nq, s, c, nullptr);
    else       dd_core_ship (r_mod, nq, ARM_SHIP, s, c, nullptr);
    if (j == 0)      { cosa = c;         sina = s; }
    else if (j == 1) { cosa = negate(s); sina = c; }
    else if (j == 2) { cosa = negate(c); sina = negate(s); }
    else             { cosa = s;         sina = negate(c); }
}

// ===========================================================================
// FP32 arms -- FF (p=48, nq=4), TF (p=72, nq=4), QF (p=96, nq=5)
// ===========================================================================
//
// The three FP32 backends have the same series shape as DD but spell the
// scale-down and the square differently (FF multiply_scalar, TF divide_scalar
// + sqr, QF mul_pwr2), and each has its own convergence eps and limb count.
// Those spellings are preserved here rather than normalised, so an arm is the
// shipped code with one thing changed.

struct FFTraits {
    using T = FloatFloat;
    static const int p = 48, ship_nq = 4, words = 2, phwords = 3;
    static constexpr float eps = 1.0e-15f;
    static T scale_down(T v, int nq) { return multiply_scalar(v, 1.0f / (float)(1 << nq)); }
    static T square(T v)             { return multiply(v, v); }
    static T dbl(T v)                { return multiply_scalar(v, 2.0f); }
    static float lead(const T& v)    { return v.hi; }
    static void limbs(const T& v, double* w) { w[0] = v.hi; w[1] = v.lo; }
    static int reduce(T a, T& r_mod) {
        if (detail::fabs(a.hi) <= 0.75f) { r_mod = a; return 0; }
        const float win[2] = { a.hi, a.lo };
        float f[3];
        const int j = detail::xp_ph_reduce<float>(win, 2, detail::kPhGuardFF,
                                                  detail::kPhChunksFF, f, 3);
        T pio2 = T(detail::xp_ph_pio2_f(0));
        for (int k = 1; k < detail::kPhPio2WordsF; ++k)
            pio2 = add(pio2, T(detail::xp_ph_pio2_f(k)));
        T fr = T(f[0]);
        for (int k = 1; k < 3; ++k) fr = add(fr, T(f[k]));
        r_mod = multiply(fr, pio2);
        return j;
    }
};
struct TFTraits {
    using T = TripleFloat;
    static const int p = 72, ship_nq = 4, words = 3, phwords = 4;
    static constexpr float eps = 1.0e-21f;
    static T scale_down(T v, int nq) { return divide_scalar(v, float(1 << nq)); }
    static T square(T v)             { return sqr(v); }
    static T dbl(T v)                { return multiply_scalar(v, 2.0f); }
    static float lead(const T& v)    { return v.f0; }
    static void limbs(const T& v, double* w) { w[0]=v.f0; w[1]=v.f1; w[2]=v.f2; }
    static int reduce(T a, T& r_mod) {
        if (detail::fabs(a.f0) <= 0.75f) { r_mod = a; return 0; }
        const float win[3] = { a.f0, a.f1, a.f2 };
        float f[4];
        const int j = detail::xp_ph_reduce<float>(win, 3, detail::kPhGuardTF,
                                                  detail::kPhChunksTF, f, 4);
        T pio2 = T(detail::xp_ph_pio2_f(0));
        for (int k = 1; k < detail::kPhPio2WordsF; ++k)
            pio2 = add(pio2, T(detail::xp_ph_pio2_f(k)));
        T fr = T(f[0]);
        for (int k = 1; k < 4; ++k) fr = add(fr, T(f[k]));
        r_mod = multiply(fr, pio2);
        return j;
    }
};
struct QFTraits {
    using T = QuadFloat;
    static const int p = 96, ship_nq = 5, words = 4, phwords = 5;
    static constexpr float eps = 1.0e-28f;
    static T scale_down(T v, int nq) { return mul_pwr2(v, ldexpf(1.0f, -nq)); }
    static T square(T v)             { return multiply(v, v); }
    static T dbl(T v)                { return mul_pwr2(v, 2.0f); }
    static float lead(const T& v)    { return v.f0; }
    static void limbs(const T& v, double* w) { w[0]=v.f0; w[1]=v.f1; w[2]=v.f2; w[3]=v.f3; }
    static int reduce(T a, T& r_mod) {
        if (detail::fabs(a.f0) <= 0.75f) { r_mod = a; return 0; }
        const float win[4] = { a.f0, a.f1, a.f2, a.f3 };
        float f[5];
        const int j = detail::xp_ph_reduce<float>(win, 4, detail::kPhGuardQF,
                                                  detail::kPhChunksQF, f, 5);
        T pio2 = T(detail::xp_ph_pio2_f(0));
        for (int k = 1; k < detail::kPhPio2WordsF; ++k)
            pio2 = add(pio2, T(detail::xp_ph_pio2_f(k)));
        T fr = T(f[0]);
        for (int k = 1; k < 5; ++k) fr = add(fr, T(f[k]));
        r_mod = multiply(fr, pio2);
        return j;
    }
};

template <class Tr>
static void fp32_core_ship(typename Tr::T r_mod, int nq, bool fact,
                           typename Tr::T& s, typename Tr::T& c) {
    using T = typename Tr::T;
    T r  = Tr::scale_down(r_mod, nq);
    T r2 = Tr::square(r);
    s = r;              c  = T(1.0f);
    T st = r,           ct = T(1.0f);
    for (int k = 1; k <= 100; ++k) {
        st = divide_scalar(multiply(st, r2), -(float)((2*k) * (2*k + 1)));
        s  = add(s, st);
        ct = divide_scalar(multiply(ct, r2), -(float)((2*k - 1) * (2*k)));
        c  = add(c, ct);
        if (detail::fabs(Tr::lead(st)) <= Tr::eps * detail::fabs(Tr::lead(s)) &&
            detail::fabs(Tr::lead(ct)) <= Tr::eps) break;
    }
    for (int d = 0; d < nq; ++d) {
        const T ns = Tr::dbl(multiply(s, c));
        const T nc = fact ? multiply(subtract(c, s), add(c, s))
                          : subtract(Tr::square(c), Tr::square(s));
        s = ns; c = nc;
    }
}

template <class Tr>
static void fp32_core_vform(typename Tr::T r_mod, int nq,
                            typename Tr::T& s, typename Tr::T& c) {
    using T = typename Tr::T;
    T r  = Tr::scale_down(r_mod, nq);
    T r2 = Tr::square(r);
    T vt = divide_scalar(r2, 2.0f);
    T v  = vt;
    T st = r;
    s = r;
    for (int k = 1; k <= 100; ++k) {
        st = divide_scalar(multiply(st, r2), -(float)((2*k) * (2*k + 1)));
        s  = add(s, st);
        vt = divide_scalar(multiply(vt, r2), -(float)((2*k + 1) * (2*k + 2)));
        v  = add(v, vt);
        if (detail::fabs(Tr::lead(st)) <= Tr::eps * detail::fabs(Tr::lead(s)) &&
            detail::fabs(Tr::lead(vt)) <= Tr::eps * detail::fabs(Tr::lead(v))) break;
    }
    for (int d = 0; d < nq; ++d) {
        const T cc = subtract(T(1.0f), v);
        const T ns = Tr::dbl(multiply(s, cc));
        const T nv = Tr::dbl(Tr::square(s));
        s = ns; v = nv;
    }
    c = subtract(T(1.0f), v);
}

// The shipped guards, as for DD.  All three FP32 backends use the same KI-12
// band (2^nq * FLT_MIN) and none of them carries DD's 1e60 magnitude bail.
template <class Tr>
static void fp32_eval(double xd, int nq, bool vform, bool fact,
                      typename Tr::T& cosa, typename Tr::T& sina) {
    using T = typename Tr::T;
    // T(double), not T((float)xd): sweep_accuracy builds its argument with the
    // double constructor, which SPLITS across the limbs.  Truncating to one
    // float first would hand every FP32 backend a different argument from the
    // one it is scored on, and the reference below is taken from the stored
    // limbs for the same reason (the sweep's LAYER 0, f(x_stored)).
    const T a(xd);
    if (Tr::lead(a) == 0.0f) { cosa = T(1.0f); sina = T(0.0f); return; }
    if (detail::fabs(Tr::lead(a)) < (float)(1 << nq) * 1.17549435e-38f) {
        cosa = T(1.0f); sina = a; return;
    }
    T r_mod;
    const int j = Tr::reduce(a, r_mod);
    T s, c;
    if (vform) fp32_core_vform<Tr>(r_mod, nq, s, c);
    else       fp32_core_ship <Tr>(r_mod, nq, fact, s, c);
    if (j == 0)      { cosa = c;         sina = s; }
    else if (j == 1) { cosa = negate(s); sina = c; }
    else if (j == 2) { cosa = negate(c); sina = negate(s); }
    else             { cosa = s;         sina = negate(c); }
}

// ===========================================================================
// Reporting
// ===========================================================================

struct Stat {
    std::vector<double> v;
    void add(double u) { if (u >= 0.0) v.push_back(u); }
    double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
    double med() {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    int over(double t) const {
        int n = 0; for (double u : v) if (u > t) ++n; return n;
    }
    size_t n() const { return v.size(); }
};

struct ArmSpec { const char* name; int nq; Arm arm; bool vform; };

// The arm the measurement below selects, reported in detail at the end.
// Overridable with --win-nq N / --win-ship so the third arm can be pointed at
// any candidate without a rebuild.
static int  kWinNq    = 5;
static bool kWinVform = true;

int main(int argc, char** argv) {
    bool terms_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--terms") == 0) terms_mode = true;
        else if (std::strcmp(argv[i], "--win-ship") == 0) kWinVform = false;
        else if (std::strcmp(argv[i], "--win-nq") == 0 && i + 1 < argc)
            kWinNq = std::atoi(argv[++i]);
    }

    const std::vector<Pt> grid = build_grid();
    const char* fams[] = { "linear", "log", "ulp", "hardred" };

    // DD ships nq = 5.
    const ArmSpec dd_arms[] = {
        { "ship  nq=5", 5, ARM_SHIP,  false },
        { "fact  nq=5", 5, ARM_FACT,  false },
        { "vform nq=6", 6, ARM_VFORM, true  },
        { "vform nq=5", 5, ARM_VFORM, true  },
        { "vform nq=4", 4, ARM_VFORM, true  },
        { "vform nq=3", 3, ARM_VFORM, true  },
        { "vform nq=2", 2, ARM_VFORM, true  },
        { "vform nq=1", 1, ARM_VFORM, true  },
        { "vform nq=0", 0, ARM_VFORM, true  },
        { "ship  nq=3", 3, ARM_SHIP,  false },
        { "ship  nq=1", 1, ARM_SHIP,  false },
        { "ship  nq=0", 0, ARM_SHIP,  false },
    };
    const int nda = (int)(sizeof(dd_arms) / sizeof(dd_arms[0]));

    mpfr_t X, S, C;
    mpfr_inits2(kPrec, X, S, C, (mpfr_ptr)0);

    if (terms_mode) {
        std::printf("SERIES LENGTH -- iterations the convergence test actually runs.\n");
        std::printf("Worst over the whole grid; the doubling loop adds nq more steps.\n\n");
        std::printf("  %-12s %8s %8s\n", "arm", "max k", "+nq");
        for (int i = 0; i < nda; ++i) {
            int worst = 0;
            for (const Pt& p : grid) {
                if (!std::isfinite(p.x)) continue;
                DoubleDouble cc, ss; int k = 0;
                dd_eval(DoubleDouble(p.x), dd_arms[i].nq, dd_arms[i].arm,
                        dd_arms[i].vform, cc, ss, &k);
                worst = std::max(worst, k);
            }
            std::printf("  %-12s %8d %8d\n", dd_arms[i].name, worst, dd_arms[i].nq);
        }
        mpfr_clears(X, S, C, (mpfr_ptr)0);
        return 0;
    }

    std::printf("DD SERIES CORE -- ulps against MPFR at %ld bits, p = 106.\n",
                (long)kPrec);
    std::printf("Same Payne-Hanek (j, r_mod) and same quadrant table in every "
                "arm.\n\n");

    // kappa per point, computed once: |x/tan x| for sin, |x tan x| for cos.
    // This is the amplification of a one-ulp argument perturbation -- the same
    // quantity sweep_accuracy's derived bound is built from.  Where it is large
    // the point is a near-zero of the function and relative error diverges
    // there for reasons no series core can address; the plan's own modelling
    // excluded such points.  `max|k<=4` below is therefore the decision metric,
    // and the unfiltered max is kept beside it so nothing is hidden.
    std::vector<double> ks(grid.size()), kc(grid.size());
    std::vector<char>   bailed(grid.size(), 0);
    for (size_t i = 0; i < grid.size(); ++i) {
        const DoubleDouble a(grid[i].x);
        bailed[i] = (detail::fabs(a.hi) >= 1.0e60) ? 1 : 0;
        mpfr_t K, TN;
        mpfr_inits2(kPrec, K, TN, (mpfr_ptr)0);
        mpfr_set_d(X, a.hi, MPFR_RNDN);
        mpfr_add_d(X, X, a.lo, MPFR_RNDN);
        mpfr_tan(TN, X, MPFR_RNDN);
        mpfr_div(K, X, TN, MPFR_RNDN); mpfr_abs(K, K, MPFR_RNDN);
        ks[i] = mpfr_get_d(K, MPFR_RNDN);
        mpfr_mul(K, X, TN, MPFR_RNDN); mpfr_abs(K, K, MPFR_RNDN);
        kc[i] = mpfr_get_d(K, MPFR_RNDN);
        mpfr_clears(K, TN, (mpfr_ptr)0);
    }

    std::printf("POINTS EXCLUDED FROM THE TABLES BELOW\n");
    {
        int nb = 0;
        for (size_t i = 0; i < grid.size(); ++i) if (bailed[i]) ++nb;
        std::printf("  %d points at |x| >= 1e60, where dd_math.hpp:879 bails to\n"
                    "  the identity point (1, 0) BEFORE any series runs.  Their\n"
                    "  error is identical in every arm -- 8.1130e31 and 1.7309e50\n"
                    "  ulps -- so they cannot inform this choice, and the sweep\n"
                    "  reads all of them as state U.  Removing the bail is the\n"
                    "  separate commit dd_math.hpp:875 describes.\n\n", nb);
    }

    std::printf("  max|k<=4 = worst ulps over points that are NOT near-zeros.\n"
                "  THIS IS THE DECISION METRIC.\n\n");
    for (const char* fam : fams) {
        std::printf("--- family %s ---\n", fam);
        std::printf("  %-12s %8s %8s %9s | %8s %8s %9s | %6s\n",
                    "arm", "sin med", "sin max", "sin max|k<=4",
                    "cos med", "cos max", "cos max|k<=4", ">1");
        for (int i = 0; i < nda; ++i) {
            Stat ssin, scos, fsin, fcos;
            for (size_t gi = 0; gi < grid.size(); ++gi) {
                if (std::strcmp(grid[gi].family, fam) != 0) continue;
                if (bailed[gi]) continue;
                const DoubleDouble a(grid[gi].x);
                mpfr_set_d(X, a.hi, MPFR_RNDN);
                mpfr_add_d(X, X, a.lo, MPFR_RNDN);
                mpfr_sin(S, X, MPFR_RNDN);
                mpfr_cos(C, X, MPFR_RNDN);
                DoubleDouble cc, ss;
                dd_eval(a, dd_arms[i].nq, dd_arms[i].arm, dd_arms[i].vform,
                        cc, ss, nullptr);
                const double sw[2] = { ss.hi, ss.lo };
                const double cw[2] = { cc.hi, cc.lo };
                const double us = ulps_of(S, sw, 2, 106);
                const double uc = ulps_of(C, cw, 2, 106);
                ssin.add(us); scos.add(uc);
                if (ks[gi] <= 4.0) fsin.add(us);
                if (kc[gi] <= 4.0) fcos.add(uc);
            }
            std::printf("  %-12s %8.4f %8.4f %9.4f | %8.4f %8.4f %9.4f | %6d\n",
                        dd_arms[i].name, ssin.med(), ssin.max(), fsin.max(),
                        scos.med(), scos.max(), fcos.max(),
                        ssin.over(1.0) + scos.over(1.0));
        }
        std::printf("\n");
    }

    // DD FIDELITY, against the WINNER rather than against `ship`.
    //
    // Once dd_math.hpp carries the chosen core, the probe's DD arm and the
    // shipped code are supposed to be the same arithmetic -- and "supposed to
    // be" is how a probe ends up validating something other than what ships.
    // This asserts it limb-for-limb, which makes the DD tables above a
    // statement about dd_math.hpp and not merely about this file.  Before the
    // header change it reports MISMATCH on essentially every point, which is
    // the correct answer then and the reason the line prints the count.
    {
        int bad = 0; double first_x = 0.0;
        for (const Pt& p : grid) {
            const DoubleDouble a(p.x);
            DoubleDouble cc, ss, sc, ss2;
            dd_eval(a, kWinNq, kWinVform ? ARM_VFORM : ARM_SHIP, kWinVform,
                    cc, ss, nullptr);
            xp::sincos(a, sc, ss2);           // DD writes cos into the FIRST param
            if (ss.hi != ss2.hi || ss.lo != ss2.lo ||
                cc.hi != sc.hi  || cc.lo != sc.lo) {
                if (!bad) first_x = p.x;
                ++bad;
            }
        }
        std::printf("--- fidelity: WINNER (%s nq=%d) vs xp::sincos, limb for "
                    "limb ---\n", kWinVform ? "vform" : "ship", kWinNq);
        std::printf("  DD  %s  (%d of %zu points differ%s)\n\n",
                    bad ? "MISMATCH -- dd_math.hpp is NOT running this core"
                        : "identical -- dd_math.hpp runs exactly this core",
                    bad, grid.size(),
                    bad ? (std::string(", first at x=") +
                           std::to_string(first_x)).c_str() : "");
    }

    // ---------------------------------------------------------------------
    // FP32 backends.  Same arms, each at its own shipped nq and p.
    // ---------------------------------------------------------------------
    // QF is here for a specific reason.  Its worst real trig cells are NOT the
    // linear family the DD defect lives in -- they are `ulp` points whose
    // reduced |r| is tiny (1029.9 ulps at x = 91.106186954104004, log2|r| ~
    // -59.8), which probe_trig_stages --fp32 attributed to the FP32 subnormal
    // floor: u = r_mod/2^nq puts a QuadFloat's fourth limb at 2^(e-nq-72),
    // under FLT_MIN once e < -49, and the doublings then scale the shed bits
    // back up by 2^nq.
    //
    // That mechanism is a property of the SCALE-DOWN, not of the leading 1, so
    // the v-form -- which still evaluates at u = r_mod/2^nq -- is predicted NOT
    // to fix it, while nq = 0 is.  The table below is where that prediction is
    // checked instead of assumed.
    {
        struct FA { const char* name; int nq; bool vform; bool fact; };
        std::printf("FP32 BACKENDS -- same arms, each at its own p.\n\n");

        // FIDELITY CHECK, and it is not a formality.  Every number in the
        // tables below is a comparison against the `ship` arm, so if that arm
        // is not the shipped code the whole comparison is void.  This asserts
        // it limb-for-limb against xp::sincos itself over the entire grid.
        // The FP32 backends do not share one loop shape -- TF tests
        // convergence BEFORE the update in a while loop while FF and QF test
        // after, in a for loop -- so the generic arm above is a transcription
        // and transcriptions are exactly what this repo has been burned by.
        std::printf("--- fidelity: `ship nq=S` vs xp::sincos, limb for limb ---\n");
        auto fidelity = [&](auto tr_tag, const char* be) {
            using Tr = decltype(tr_tag);
            using T  = typename Tr::T;
            int bad = 0; double worst_x = 0.0;
            for (const Pt& p : grid) {
                T cc, ss;
                fp32_eval<Tr>(p.x, Tr::ship_nq, false, false, cc, ss);
                T sc, ss2;
                const T a(p.x);
                // DD/FF write cos into the FIRST out-param, TF/QF sin first.
                if (Tr::p == 48) xp::sincos(a, sc, ss2);
                else             xp::sincos(a, ss2, sc);
                double w1[4], w2[4], w3[4], w4[4];
                Tr::limbs(ss, w1); Tr::limbs(ss2, w2);
                Tr::limbs(cc, w3); Tr::limbs(sc,  w4);
                for (int i = 0; i < Tr::words; ++i)
                    if (w1[i] != w2[i] || w3[i] != w4[i]) {
                        if (!bad) worst_x = p.x;
                        ++bad; break;
                    }
            }
            std::printf("  %-3s %s  (%d of %zu points differ%s)\n", be,
                        bad ? "MISMATCH -- the arm is NOT the shipped code"
                            : "identical",
                        bad, grid.size(),
                        bad ? (std::string(", first at x=") +
                               std::to_string(worst_x)).c_str() : "");
        };
        fidelity(FFTraits{}, "FF");
        fidelity(TFTraits{}, "TF");
        fidelity(QFTraits{}, "QF");
        std::printf("\n");

        auto run = [&](auto tr_tag, const char* be) {
            using Tr = decltype(tr_tag);
            const FA arms[] = {
                { "ship  nq=S", Tr::ship_nq, false, false },
                { "fact  nq=S", Tr::ship_nq, false, true  },
                { "vform nq=S", Tr::ship_nq, true,  false },
                { "vform nq=2", 2,           true,  false },
                { "vform nq=1", 1,           true,  false },
                { "vform nq=0", 0,           true,  false },
                { "ship  nq=0", 0,           false, false },
            };
            // kappa PER BACKEND.  The DD kappa above cannot be reused: an FP32
            // backend stores a different number from the double the grid names
            // -- FF carries 48 bits of a 53-bit double -- so the near-zero it
            // actually sits on is a different point, and filtering on the
            // double's kappa would mislabel exactly the cells that matter.
            std::vector<double> bks(grid.size()), bkc(grid.size());
            for (size_t gi = 0; gi < grid.size(); ++gi) {
                const typename Tr::T a(grid[gi].x);
                double aw[4]; Tr::limbs(a, aw);
                exact_of(X, aw, Tr::words);
                mpfr_t K, TN;
                mpfr_inits2(kPrec, K, TN, (mpfr_ptr)0);
                mpfr_tan(TN, X, MPFR_RNDN);
                mpfr_div(K, X, TN, MPFR_RNDN); mpfr_abs(K, K, MPFR_RNDN);
                bks[gi] = mpfr_get_d(K, MPFR_RNDN);
                mpfr_mul(K, X, TN, MPFR_RNDN); mpfr_abs(K, K, MPFR_RNDN);
                bkc[gi] = mpfr_get_d(K, MPFR_RNDN);
                mpfr_clears(K, TN, (mpfr_ptr)0);
            }
            std::printf("=== %s (p=%d, shipped nq=%d) ===\n", be, Tr::p, Tr::ship_nq);
            std::printf("  %-11s %8s %8s %9s | %8s %8s %9s | %6s\n",
                        "arm", "sin med", "sin max", "sin mx|k<=4",
                        "cos med", "cos max", "cos mx|k<=4", ">1");
            for (const FA& A : arms) {
                Stat ssin, scos, fsin, fcos;
                double wx = 0.0, wu = -1.0;
                for (size_t gi = 0; gi < grid.size(); ++gi) {
                    if (std::strcmp(grid[gi].family, "hardred") == 0) continue;
                    typename Tr::T cc, ss;
                    fp32_eval<Tr>(grid[gi].x, A.nq, A.vform, A.fact, cc, ss);
                    // Reference at the STORED argument, as the sweep does.
                    const typename Tr::T a(grid[gi].x);
                    double aw[4]; Tr::limbs(a, aw);
                    exact_of(X, aw, Tr::words);
                    if (!mpfr_number_p(X)) continue;
                    mpfr_sin(S, X, MPFR_RNDN);
                    mpfr_cos(C, X, MPFR_RNDN);
                    double sw[4], cw[4];
                    Tr::limbs(ss, sw); Tr::limbs(cc, cw);
                    const double us = ulps_of(S, sw, Tr::words, Tr::p);
                    const double uc = ulps_of(C, cw, Tr::words, Tr::p);
                    ssin.add(us); scos.add(uc);
                    if (bks[gi] <= 4.0) { fsin.add(us);
                        if (us > wu) { wu = us; wx = grid[gi].x; } }
                    if (bkc[gi] <= 4.0) { fcos.add(uc);
                        if (uc > wu) { wu = uc; wx = grid[gi].x; } }
                }
                (void)wx;
                std::printf("  %-11s %8.4f %8.4f %9.4f | %8.4f %8.4f %9.4f | %6d"
                            "   worst k<=4 at x=%.17g\n",
                            A.name, ssin.med(), ssin.max(), fsin.max(),
                            scos.med(), scos.max(), fcos.max(),
                            ssin.over(1.0) + scos.over(1.0), wx);
            }
            std::printf("\n");
        };
        run(FFTraits{}, "FF");
        run(TFTraits{}, "TF");
        run(QFTraits{}, "QF");

        // The two QF cells that motivated this, called out by name so the
        // prediction is testable at a glance rather than buried in a max.
        std::printf("=== QF, the two cells probe_trig_stages --fp32 attributed "
                    "to the scale-down ===\n");
        std::printf("  %-11s %12s %12s %12s %12s\n", "arm",
                    "sin 91.106", "tan 91.106", "sin 182.21", "tan 182.21");
        const double qpts[2] = { 91.106186954104004, 182.21237390820801 };
        const FA qarms[] = {
            { "ship  nq=5", 5, false, false }, { "vform nq=5", 5, true, false },
            { "vform nq=3", 3, true,  false }, { "vform nq=1", 1, true, false },
            { "vform nq=0", 0, true,  false }, { "ship  nq=0", 0, false, false },
        };
        for (const FA& A : qarms) {
            double col[4];
            for (int i = 0; i < 2; ++i) {
                QuadFloat cc, ss;
                fp32_eval<QFTraits>(qpts[i], A.nq, A.vform, A.fact, cc, ss);
                const QuadFloat a(qpts[i]);
                double aw[4]; QFTraits::limbs(a, aw);
                exact_of(X, aw, 4);
                mpfr_sin(S, X, MPFR_RNDN);
                mpfr_cos(C, X, MPFR_RNDN);
                double sw[4], cw[4];
                QFTraits::limbs(ss, sw); QFTraits::limbs(cc, cw);
                col[i * 2] = ulps_of(S, sw, 4, 96);
                // tan = sin/cos, as qf_math.hpp:tan forms it.
                const QuadFloat t = divide(ss, cc);
                double tw[4]; QFTraits::limbs(t, tw);
                mpfr_tan(C, X, MPFR_RNDN);
                col[i * 2 + 1] = ulps_of(C, tw, 4, 96);
            }
            std::printf("  %-11s %12.4f %12.4f %12.4f %12.4f\n",
                        A.name, col[0], col[1], col[2], col[3]);
        }
        std::printf("\n");
    }

    // ---------------------------------------------------------------------
    // Where the winner's residual actually is, and whether it is the series.
    // ---------------------------------------------------------------------
    // kappa = |x / tan(x)| for sin and |x * tan(x)| for cos: the amplification
    // of a one-ulp perturbation of the argument, the same quantity
    // sweep_accuracy's derived bound is built from.  A point with a large kappa
    // is a near-zero of the function, where relative error legitimately
    // diverges and no series can help -- distinguishing those from real series
    // residue is the whole reason this column is here.
    struct Bad { double u, x, kappa, u_exact; const char* fam; int isc; };
    std::vector<Bad> bad;
    for (const Pt& p : grid) {
        const DoubleDouble a(p.x);
        mpfr_set_d(X, a.hi, MPFR_RNDN);
        mpfr_add_d(X, X, a.lo, MPFR_RNDN);
        mpfr_sin(S, X, MPFR_RNDN);
        mpfr_cos(C, X, MPFR_RNDN);
        DoubleDouble cc, ss, ce, se;
        dd_eval(a, kWinNq, kWinVform ? ARM_VFORM : ARM_SHIP, kWinVform,
                cc, ss, nullptr);
        const bool small = detail::fabs(a.hi) < 1.0e60;
        if (small) dd_eval_exactred(a, kWinNq, kWinVform, ce, se);
        const double sw[2] = { ss.hi, ss.lo }, cw[2] = { cc.hi, cc.lo };
        const double se_w[2] = { se.hi, se.lo }, ce_w[2] = { ce.hi, ce.lo };
        const double us = ulps_of(S, sw, 2, 106), uc = ulps_of(C, cw, 2, 106);
        mpfr_t K, TN;
        mpfr_inits2(kPrec, K, TN, (mpfr_ptr)0);
        mpfr_tan(TN, X, MPFR_RNDN);
        if (us > 1.0) {
            mpfr_div(K, X, TN, MPFR_RNDN); mpfr_abs(K, K, MPFR_RNDN);
            bad.push_back({us, p.x, mpfr_get_d(K, MPFR_RNDN),
                           small ? ulps_of(S, se_w, 2, 106) : -1.0, p.family, 0});
        }
        if (uc > 1.0) {
            mpfr_mul(K, X, TN, MPFR_RNDN); mpfr_abs(K, K, MPFR_RNDN);
            bad.push_back({uc, p.x, mpfr_get_d(K, MPFR_RNDN),
                           small ? ulps_of(C, ce_w, 2, 106) : -1.0, p.family, 1});
        }
        mpfr_clears(K, TN, (mpfr_ptr)0);
    }
    std::sort(bad.begin(), bad.end(),
              [](const Bad& a, const Bad& b) { return a.u > b.u; });
    std::printf("--- WINNER (%s nq=%d): every point above 1 ulp, worst 20 ---\n",
                kWinVform ? "vform" : "ship", kWinNq);
    std::printf("  THIRD ARM: `exact red` re-runs the same core on the exact\n"
                "  r_mod at %ld bits.  Unchanged => the reduction has no\n"
                "  headroom left there and the residue is the core or the format.\n",
                (long)kPrec);
    std::printf("  %-4s %9s %9s %12s %-9s %s\n",
                "f", "ulps", "exact red", "kappa", "family", "x");
    for (size_t i = 0; i < bad.size() && i < 20; ++i)
        std::printf("  %-4s %9.4f %9.4f %12.4g %-9s %.17g\n",
                    bad[i].isc ? "cos" : "sin", bad[i].u, bad[i].u_exact,
                    bad[i].kappa, bad[i].fam, bad[i].x);
    std::printf("  %zu of %zu (sin, cos) cells above 1 ulp over the whole grid\n",
                bad.size(), 2 * grid.size());
    int kap = 0;
    for (const Bad& b : bad) if (b.kappa > 4.0) ++kap;
    std::printf("  of those, %d have kappa > 4 (near-zeros, where relative "
                "error diverges)\n", kap);

    // The stop condition is stated over the non-near-zero set, so list that set
    // whole rather than truncating it -- the third-arm column is the evidence
    // for "no reduction headroom remains at those arguments", and a truncated
    // list could hide a point where headroom does remain.
    std::printf("\n--- WINNER, the kappa<=4 set above 1 ulp, IN FULL ---\n");
    std::printf("  %-4s %9s %9s %12s %-9s %s\n",
                "f", "ulps", "exact red", "kappa", "family", "x");
    int shown = 0, headroom = 0;
    for (const Bad& b : bad) {
        if (b.kappa > 4.0) continue;
        ++shown;
        // "headroom" = the exact-reduction arm is materially better, i.e. the
        // shipped Payne-Hanek is still leaving error on the table here.
        if (b.u_exact >= 0.0 && b.u - b.u_exact > 0.1) ++headroom;
        std::printf("  %-4s %9.4f %9.4f %12.4g %-9s %.17g\n",
                    b.isc ? "cos" : "sin", b.u, b.u_exact, b.kappa, b.fam, b.x);
    }
    std::printf("  %d cells; %d of them improve by >0.1 ulp under exact "
                "reduction\n", shown, headroom);

    mpfr_clears(X, S, C, (mpfr_ptr)0);
    return 0;
}
