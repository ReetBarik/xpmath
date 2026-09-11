// probe_acosh_imag.cpp -- ablate two ways to compute Im acosh(x+iy), in each
// backend's own arithmetic, before either of them is shipped.
//
// THE DEFECT.  QF acosh reads 1.53e14 ulps at z = (1, -1e-30) and TF reads
// 9.10e6 at the same point.  40 rows between them, points 878..895 of the
// cut-re family -- z = (1, +-eps) with eps walking from 1e-22 down to 1e-30 --
// and they are the largest cluster of real library error left above 8 ulps in
// complex trig.  Input rounding is not the cause: the classifier in
// probe_complex_input_rounding.cpp scores them against the STORED operand and
// they do not move.  The floor is not the cause either: the answer's modulus is
// ~sqrt(eps) ~ 1.4e-15, whose representational floor is 2^96 * 2^-149 / 1.4e-15
// = 0.079 ulps, so the reading sits 1.9e15 times ABOVE its own floor.
//
// WHICH COMPONENT.  Only the imaginary one.  Re acosh is already
// xp_asin_imag_mag(|x|,|y|), the well-conditioned form the acos work installed,
// and reads 0.084 ulps at z = (1, 1e-24).  Im acosh is the last survivor of the
// original Kahan chain:
//
//     FORM A   rp = sqrt((x+1)/2 + iy/2),  rm = sqrt((x-1)/2 + iy/2)
//              Im acosh = |2 * Im log(rp + rm)|, signed by sign(y)
//
// which is arg(rp+rm) doubled.
//
// THE MECHANISM, and the FIRST HYPOTHESIS WAS WRONG.  The chain forms y/2, so
// an inexact halving looked like the cause: a QuadFloat holding 1e-30 has words
// at 1e-30, 6e-38 and 1.8e-45, the last already subnormal, and halving it
// should push a bit off the end.  Measured, that is not what happens.  --trace
// 894 replays the chain one step at a time against MPC from the stored operand,
// and at the worst point on the grid y/2 is EXACT, 0 ulps, as is (x-1)/2.  The
// 2.56e14 ulps appear two steps later, inside sqrt(rm), and arg() then reports
// the destroyed operand faithfully -- arg of the COMPUTED sum is right to 0.249
// ulps.  Cross-tabulating a limb-exact `halving is lossy' predicate against
// which form wins confirmed it: the predicate is true at only 4 of the 42 rows
// above 8 ulps, and false at the two worst.
//
// What is actually wrong is MAGNITUDE.  A QuadFloat at 5e-31 wants its fourth
// word at 5e-31 * 2^-72 = 1.1e-52, far below the smallest subnormal 2^-149 =
// 1.4e-45, so the operand sqrt() receives holds about 49 bits rather than 96 no
// matter how it was formed.  sqrt halves the relative error to ~4e-16 and the
// measured relative error is 1.9e-15; the order matches.  It is KI-33's
// representational floor, reached through an INTERMEDIATE the formula did not
// have to form -- the ANSWER at that point is ~sqrt(y)(1+i), both components
// ~1e-15 and comfortably normal, with a floor of 0.079 ulps.
//
//     FORM B   Im acosh = |Re acos(z)| = |atan2(leg, x)|, signed by sign(y),
//              leg = xp_asin_real_leg(|x|, |y|)
//
// This is not a new formula, it is the identity acosh(z) = +-i acos(z) applied
// to the component that was left behind.  i(A + iB) = -B + iA, so Re acosh is
// -Im acos and Im acosh is Re acos.  The header already uses the first half of
// that identity for the real part; form B uses the second half for the
// imaginary part, and the whole function becomes a rotation of an acos that has
// already been reformulated, measured and gated.  It never forms y/2, never
// calls complex sqrt, and never calls complex log.
//
//     FORM G   form B where the chain's sqrt operand is too short to hold the
//              format's width, form A everywhere else.  This is what shipped.
//
// RESULT, and what shipped.  Form B UNGUARDED is better on average on every
// backend -- on the sweep's modulus-relative metric its medians are 0.6551 ->
// 0.2912 (DD), 0.3293 -> 0.3073 (FF), 0.1015 -> 0.05335 (QF) and 0.1067 ->
// 0.07696 (TF), and rows above 1 ulp go 1187 -> 407.  It is NOT better
// pointwise.  Shipped unguarded it cost 629 rows across the four backends and
// the repo's monotone gate rejected it: decreased 132, worst drop 2.03 digits
// equivalent, every listed regression a DD row.  Average improvement does not
// buy a pointwise loss, so the substitution was confined to where the chain
// provably cannot work.
//
// THE PREDICATE, and it took two attempts.  The first version tested whether
// y/2 alone was below the format's full-width range.  It covered the defect --
// 40 of 42 rows above 8 ulps -- but it OVER-FIRED, at z = (2, tiny), points
// 936..955, where the chain is perfectly healthy because the real half of the
// sqrt operand is 0.5 and dominates.  There the rotation made the imaginary
// component worse by up to 0.9 digits.  The sweep's modulus-relative metric
// could not see it -- Im is ~1e-30 against a modulus of ~1.3, so the whole
// component hides behind the real part and the ULPS column was BIT-IDENTICAL
// either way -- and it surfaced only because the monotone gate also carries the
// per-component digit count: 66 rows with unchanged ulps and a fallen digits
// column.  That is the criticism recorded at sweep_accuracy.cpp:2105 happening
// in practice, and it is why this probe scores BOTH metrics.
//
// The second version tests the scale of the WHOLE operand handed to each sqrt,
// max(|(x-+1)/2|, |y/2|), which is the thing that actually has to be
// representable.  It does not fire at z = (2, tiny), still covers 40 of the 42
// rows above 8 ulps (the other 2 are an FF pair at 8.408 that the rotation does
// not improve either), and makes NO row worse on EITHER metric on ANY backend.
//
//     bk    max A        max B(unguarded)   max G(shipped)   rows >8  A/B/G
//     DD    6.0127       5.4251             6.0127            0 / 0 / 0
//     FF    8.4082       8.4082             8.4082            2 / 2 / 2
//     QF    1.5266e+14   5.364              5.364            28 / 0 / 0
//     TF    9.0993e+06   7.061              7.061            12 / 0 / 0
//
// In the sweep itself the guarded form moves 22 scored rows, all c acosh: QF
// 1.2589e8 -> 5.364 with its 16 rows above 8 ulps going to zero, TF 7.5071 ->
// 7.061, DD and FF bit-identical, no row worse, ctest 38/38 and the monotone
// gate clean.
//
// WHAT THIS PROBE MEASURES.  Both forms, evaluated in each backend's own
// arithmetic, scored against MPC's Im acosh at the STORED operand, over the
// whole 1780-point complex grid on all four backends.  A form is only worth
// shipping if it is better where it is supposed to be better and no worse
// anywhere else, which is why the whole grid is run and not the 40 points the
// defect lives at.
//
// form_a() is a transcription of what the headers shipped BEFORE this change,
// kept so the ablation stays runnable.  form_g() is what they ship now, and the
// probe CHECKS that: on every grid point it compares form_g() limb by limb
// against xp::acosh(z).im and reports any mismatch.  That is what catches a
// porting slip between this file and the four headers -- without it, a probe
// that agrees with itself proves nothing about the shipped code.
//
// THE CUT IS CHECKED SEPARATELY.  Im acosh carries the sheet: acosh(-2+0i) is
// 1.317 + pi i and acosh(-2-0i) is 1.317 - pi i, and the sign of a zero decides
// which.  A form that is more accurate and lands on the wrong sheet is not an
// improvement, so eight signed-zero cases are checked by value and by
// std::signbit -- never by ==, which cannot tell +0 from -0 -- on all four
// backends, for every form.  Twelve cases, including z = -0.1 -+ 0i: the acosh
// header used to CLAIM Kahan's chain returned +1.670964i there on QF and TF,
// the wrong sheet.  Checked, that is false -- form A passes all twelve on all
// four backends -- so the claim was removed from the header rather than shipped.
// It mattered: if the chain really did lose the sheet, guarding it back in for
// the healthy majority would have reintroduced a correctness bug.
//
// TWO POISONS.
//
// --poison replaces form B's atan2(leg, x) with atan2(leg, |x|), dropping the
// sign of the real part.  That is a real slip of exactly this kind -- the
// header's own acos comment records that the sign of x is what selects the
// sheet -- and it must show up as a large regression for Re z < 0, or this
// probe cannot tell a right rotation from a wrong one.  It does: 3389 rows read
// above 1 ulp and more than 2x form A, and 12 of the 144 cut checks fail.
//
// --poison-guard forces the predicate false, which degrades form G to exactly
// form A.  This is the one that proves the GUARD is load-bearing rather than
// decorative: the 42 rows above 8 ulps come back, and 46 grid points change.
// A threshold that never fires would pass every other check in this file.
//
// NOT A GATE.  docs/CORRECTNESS.md allows one measurement and one verdict; this
// is neither.  The verdict stays with the sweep.
//
// BUILD.  -DXPMATH_ENABLE_DIAGNOSTICS=0 silences the domain printfs
// (config.hpp:121 -- return values are unaffected).
//
//   g++ -O2 -std=c++17 -fext-numeric-literals -DXPMATH_ENABLE_DIAGNOSTICS=0 \
//       -I include scripts/probe_acosh_imag.cpp -o /tmp/probe_acosh_imag \
//       -lmpc -lmpfr -lgmp
//
// RUN from the repo root:
//   /tmp/probe_acosh_imag
//   /tmp/probe_acosh_imag --poison
//   /tmp/probe_acosh_imag --poison-guard
//   /tmp/probe_acosh_imag --csv /tmp/acosh.csv
//   /tmp/probe_acosh_imag --trace 894

#include <mpc.h>
#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xp/dd_complex.hpp"
#include "xp/ff_complex.hpp"
#include "xp/qf_complex.hpp"
#include "xp/tf_complex.hpp"

namespace {

const char* const kGrid = "validation/sweep/sweep_grid.csv";
const mpfr_prec_t kPrec = 400;
bool g_poison = false;
bool g_poison_guard = false;

template <class T>
struct Tr;
template <>
struct Tr<xp::DoubleDouble> {
    using C = xp::DoubleDoubleComplex;
    static const char* name() { return "DD"; }
    static int sig_bits() { return 106; }
    static int denorm_exp() { return -1074; }
    static int nlimb() { return 2; }
    static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::FloatFloat> {
    using C = xp::FloatFloatComplex;
    static const char* name() { return "FF"; }
    static int sig_bits() { return 48; }
    static int denorm_exp() { return -149; }
    static int nlimb() { return 2; }
    static double limb(const xp::FloatFloat& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::QuadFloat> {
    using C = xp::QuadFloatComplex;
    static const char* name() { return "QF"; }
    static int sig_bits() { return 96; }
    static int denorm_exp() { return -149; }
    static int nlimb() { return 4; }
    static double limb(const xp::QuadFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : i == 2 ? v.f2 : v.f3;
    }
};
template <>
struct Tr<xp::TripleFloat> {
    using C = xp::TripleFloatComplex;
    static const char* name() { return "TF"; }
    static int sig_bits() { return 72; }
    static int denorm_exp() { return -149; }
    static int nlimb() { return 3; }
    static double limb(const xp::TripleFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : v.f2;
    }
};

// Sum the limbs.  This is the widening that destroys the sign of a zero --
// (-0.0) + (+0.0) is +0.0 -- so the sign is restored from limb 0, and the cut
// checks below read the sign off limb 0 directly rather than off the sum.
template <class T>
void widen(const T& v, mpfr_ptr out) {
    mpfr_set_d(out, Tr<T>::limb(v, 0), MPFR_RNDN);
    for (int i = 1; i < Tr<T>::nlimb(); ++i) mpfr_add_d(out, out, Tr<T>::limb(v, i), MPFR_RNDN);
    if (mpfr_zero_p(out) && std::signbit(Tr<T>::limb(v, 0)))
        mpfr_setsign(out, out, 1, MPFR_RNDN);
}

template <class T>
double lead(const T& v) {
    return Tr<T>::limb(v, 0);
}

double ulps_real(mpfr_srcptr got, mpfr_srcptr ref, int p) {
    static bool init = false;
    static mpfr_t e;
    if (!init) { mpfr_init2(e, kPrec); init = true; }
    if (!mpfr_number_p(ref) || !mpfr_number_p(got)) return -1.0;
    if (mpfr_zero_p(ref)) return mpfr_zero_p(got) ? 0.0 : -1.0;
    mpfr_sub(e, got, ref, MPFR_RNDN);
    mpfr_div(e, e, ref, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    mpfr_mul_2si(e, e, p, MPFR_RNDN);
    return mpfr_get_d(e, MPFR_RNDN);
}

// The SWEEP's metric, transcribed: |got - ref| / |ref| * 2^p, over the whole
// complex value.  It is modulus-relative on purpose (sweep_accuracy.cpp:2105),
// and it is the metric of record, so any decision to ship is made on it.  The
// component reading above is kept alongside precisely because this one lets one
// component hide behind the other -- which is the whole shape of this defect.
double ulps_mod(mpfr_srcptr gr, mpfr_srcptr gi, mpc_srcptr ref, int p) {
    static bool init = false;
    static mpfr_t a, b, m;
    if (!init) { mpfr_inits2(kPrec, a, b, m, (mpfr_ptr)0); init = true; }
    if (!mpfr_number_p(gr) || !mpfr_number_p(gi)) return -1.0;
    mpfr_hypot(m, mpc_realref(ref), mpc_imagref(ref), MPFR_RNDN);
    if (mpfr_zero_p(m)) return -1.0;
    mpfr_sub(a, gr, mpc_realref(ref), MPFR_RNDN);
    mpfr_sub(b, gi, mpc_imagref(ref), MPFR_RNDN);
    mpfr_hypot(a, a, b, MPFR_RNDN);
    mpfr_div(a, a, m, MPFR_RNDN);
    mpfr_mul_2si(a, a, p, MPFR_RNDN);
    return mpfr_get_d(a, MPFR_RNDN);
}

// 2^p * D / |c|, the smallest relative increment the format can add to c,
// expressed in the same ulps.  A difference between two forms that is below
// this is below what the format can represent, and is not a difference.
double floor_ulps(mpfr_srcptr c, int p, int de) {
    if (!mpfr_number_p(c) || mpfr_zero_p(c)) return 0.0;
    return std::ldexp(1.0, p + de) / std::fabs(mpfr_get_d(c, MPFR_RNDN));
}

// FORM A, transcribed from what dd_complex.hpp and its three siblings shipped
// before this change.  rp, rm = sqrt((x+-1)/2 + iy/2); Im acosh = |2 arg(rp+rm)|
// signed by y.  The y/2 is the term that underflows -- see the top of the file.
template <class T>
T form_a(const T& x, const T& y) {
    using C = typename Tr<T>::C;
    using S = decltype(lead(x));
    const T one(static_cast<S>(1.0));
    const T half_im = xp::multiply_scalar(y, static_cast<S>(0.5));
    const C rp = xp::sqrt(C(xp::multiply_scalar(xp::add(x, one), static_cast<S>(0.5)), half_im));
    const C rm =
        xp::sqrt(C(xp::multiply_scalar(xp::subtract(x, one), static_cast<S>(0.5)), half_im));
    const C lg = xp::log(rp + rm);
    T im_ = xp::multiply_scalar(lg.im, static_cast<S>(2.0));
    if (lead(im_) < 0.0) im_ = xp::negate(im_);
    if (std::signbit(lead(y))) im_ = xp::negate(im_);
    return im_;
}

// CAN THE FORMAT STILL CARRY ITS FULL WIDTH AT THIS MAGNITUDE?
//
// An expansion of p bits at magnitude |v| needs its trailing word down at
// |v| * 2^-(p-w).  Once that is below the smallest subnormal the word does not
// exist and the value silently carries fewer bits than the type advertises --
// KI-33's representational floor, but reached at an INTERMEDIATE rather than at
// the answer.  Bits actually available between |v| and 2^denorm_exp:
//
//     avail = ilogb(|v|) + 1 - denorm_exp
//
// and the value is NARROWED when avail < p.  This is a property of the format
// and the magnitude only; there is no tuned constant in it.
//
//   QF  p=96,  de=-149  -> narrowed below 2^-54  = 5.55e-17
//   TF  p=72,  de=-149  -> narrowed below 2^-78  = 3.31e-24
//   FF  p=48,  de=-149  -> narrowed below 2^-102 = 1.97e-31
//   DD  p=106, de=-1074 -> narrowed below 2^-969 = 4.2e-292
template <class T>
bool narrowed_mag(double h) {
    if (h == 0.0) return true;   // an all-zero operand carries nothing at all
    return std::ilogb(h) + 1 - Tr<T>::denorm_exp() < Tr<T>::sig_bits();
}
template <class T>
bool narrowed(const T& v) {
    const double h = std::fabs(lead(v));
    if (h == 0.0) return false;
    return narrowed_mag<T>(h);
}

// IS THE OPERAND HANDED TO sqrt() SHORT?  Form A's exposure is not y/2 on its
// own -- it is the COMPLEX operand ((x-+1)/2, y/2) that sqrt() receives.  A
// narrow y/2 is harmless when the real half dominates: at z = (2, 1e-30) the
// operand is (0.5, 5e-31), sqrt() sees a well-scaled number and the tiny
// imaginary part just rides along.  It is fatal only where the real half is
// gone too, as at the branch point z = (1, y) where (x-1)/2 is exactly zero and
// y/2 IS the operand.  So the test is on the operand's own scale, max of the
// two halves, for each of the two square roots.
template <class T>
bool chain_operand_short(const T& x, const T& y) {
    using S = decltype(lead(x) * 1.0f);
    const T one(static_cast<S>(1.0));
    const double hy = std::fabs(lead(xp::multiply_scalar(y, static_cast<S>(0.5))));
    const double am =
        std::fabs(lead(xp::multiply_scalar(xp::subtract(x, one), static_cast<S>(0.5))));
    const double ap =
        std::fabs(lead(xp::multiply_scalar(xp::add(x, one), static_cast<S>(0.5))));
    return narrowed_mag<T>(std::fmax(am, hy)) || narrowed_mag<T>(std::fmax(ap, hy));
}

// IS THE HALVING LOSSY?  The first hypothesis, kept because its DISPROOF is the
// reason the predicate above exists.  Form A halves y before it does anything
// else, so a halving that loses bits looked like the mechanism.  It is not: at
// the worst point on the grid (QF, point 894) y/2 is exact to 0 ulps and the
// 2.56e14 ulps appear two steps later, inside sqrt().  See --trace 894.
template <class T>
bool halving_lossy(const T& y) {
    const T h = xp::multiply_scalar(y, static_cast<decltype(lead(y))>(0.5));
    const T b = xp::multiply_scalar(h, static_cast<decltype(lead(y))>(2.0));
    for (int i = 0; i < Tr<T>::nlimb(); ++i)
        if (Tr<T>::limb(b, i) != Tr<T>::limb(y, i)) return true;
    return false;
}

// FORM B.  Im acosh = Re acos, signed by Im z.  atan2(leg, x) is already in
// [0, pi] because leg is a magnitude, so no absolute value is needed before the
// sign is attached; negate() on a zero yields a negative zero, which is what
// the cut on the real axis requires.
template <class T>
T form_b(const T& x, const T& y) {
    using S = decltype(lead(x) * 1.0f);
    T leg = xp::xp_asin_real_leg(xp::xp_abs_word(x), xp::xp_abs_word(y));
    // Never -0: atan2(-0, x < 0) is -pi where atan2(+0, x < 0) is +pi, so the
    // sheet would flip.  Same guard, and same reason, as acos().
    if (lead(leg) == 0.0) leg = T(static_cast<S>(0.0));
    const T den = g_poison ? xp::xp_abs_word(x) : x;
    T r = xp::atan2(leg, den);
    if (std::signbit(lead(y))) r = xp::negate(r);
    return r;
}

struct Pt {
    int idx;
    double re, im;
    std::string family;
};

// FORM G, the guarded composite the four headers now ship: form B only where
// form A's own y/2 has left the format's full-width range, form A everywhere
// else.  --poison-guard forces the predicate false, which is exactly the
// unguarded form A, and must bring the >8 ulp cohort back.
template <class T>
T form_g(const T& x, const T& y) {
    const T half_im = xp::multiply_scalar(y, static_cast<decltype(lead(y))>(0.5));
    (void)half_im;
    return (!g_poison_guard && chain_operand_short<T>(x, y)) ? form_b<T>(x, y) : form_a<T>(x, y);
}

// TRACE.  Replay form A one step at a time and score each intermediate against
// the same step computed exactly in MPC from the STORED operands.  This is how
// the mechanism is located rather than guessed at: the first step whose relative
// error is large is the step that loses the answer.  --trace <point>.
template <class T>
void trace(const Pt& pt) {
    using C = typename Tr<T>::C;
    const int p = Tr<T>::sig_bits();
    const T x = T(pt.re), y = T(pt.im);
    using S = decltype(lead(x) * 1.0f);

    mpfr_t X, Y, t;
    mpfr_inits2(kPrec, X, Y, t, (mpfr_ptr)0);
    widen<T>(x, X);
    widen<T>(y, Y);

    mpc_t e_hi, e_rp, e_rm, e_sum;   // exact counterparts
    mpc_init2(e_hi, kPrec); mpc_init2(e_rp, kPrec);
    mpc_init2(e_rm, kPrec); mpc_init2(e_sum, kPrec);

    const T one(static_cast<S>(1.0));
    const T half_im = xp::multiply_scalar(y, static_cast<S>(0.5));
    const T rp_re = xp::multiply_scalar(xp::add(x, one), static_cast<S>(0.5));
    const T rm_re = xp::multiply_scalar(xp::subtract(x, one), static_cast<S>(0.5));
    const C rp = xp::sqrt(C(rp_re, half_im));
    const C rm = xp::sqrt(C(rm_re, half_im));
    const C sum = rp + rm;

    // exact: half_im, then rp/rm from the EXACT halves so each step is scored
    // against a perfect predecessor -- error that appears at a step is made there.
    mpfr_div_2ui(t, Y, 1, MPFR_RNDN);                       // y/2 exact
    mpc_set_fr_fr(e_hi, X, t, MPC_RNDNN);
    mpfr_add_ui(mpc_realref(e_rp), X, 1, MPFR_RNDN);
    mpfr_div_2ui(mpc_realref(e_rp), mpc_realref(e_rp), 1, MPFR_RNDN);
    mpfr_set(mpc_imagref(e_rp), t, MPFR_RNDN);
    mpfr_sub_ui(mpc_realref(e_rm), X, 1, MPFR_RNDN);
    mpfr_div_2ui(mpc_realref(e_rm), mpc_realref(e_rm), 1, MPFR_RNDN);
    mpfr_set(mpc_imagref(e_rm), t, MPFR_RNDN);
    mpc_sqrt(e_rp, e_rp, MPC_RNDNN);
    mpc_sqrt(e_rm, e_rm, MPC_RNDNN);
    mpc_add(e_sum, e_rp, e_rm, MPC_RNDNN);

    mpfr_t G;
    mpfr_init2(G, kPrec);
    auto one_step = [&](const char* label, const T& got, mpfr_srcptr want) {
        widen<T>(got, G);
        std::printf("    %-14s %-24.16Re want %-24.16Re  rel %10.3g ulps\n", label, G,
                    (mpfr_srcptr)want, ulps_real(G, want, p));
    };

    std::printf("  %s  x=%.17g  y=%.17g\n", Tr<T>::name(), pt.re, pt.im);
    one_step("y/2", half_im, t);
    one_step("(x-1)/2", rm_re, mpc_realref(e_rm));   // scored vs Re of the pre-sqrt operand? no --
    // the line above compares against Re sqrt(...); recompute the pre-sqrt value instead:
    mpfr_sub_ui(t, X, 1, MPFR_RNDN);
    mpfr_div_2ui(t, t, 1, MPFR_RNDN);
    one_step("(x-1)/2  [re]", rm_re, t);
    one_step("Re rm", rm.re, mpc_realref(e_rm));
    one_step("Im rm", rm.im, mpc_imagref(e_rm));
    one_step("Re rp", rp.re, mpc_realref(e_rp));
    one_step("Im rp", rp.im, mpc_imagref(e_rp));
    one_step("Re rp+rm", sum.re, mpc_realref(e_sum));
    one_step("Im rp+rm", sum.im, mpc_imagref(e_sum));

    // arg(rp+rm) scored against arg of the EXACT sum, and against arg of the
    // COMPUTED sum -- the gap between the two says whether arg() is at fault or
    // whether it faithfully reported an operand that was already wrong.
    const C lg = xp::log(sum);
    mpfr_t a_exact, a_ofgot, SR, SI;
    mpfr_inits2(kPrec, a_exact, a_ofgot, SR, SI, (mpfr_ptr)0);
    mpfr_atan2(a_exact, mpc_imagref(e_sum), mpc_realref(e_sum), MPFR_RNDN);
    widen<T>(sum.re, SR);
    widen<T>(sum.im, SI);
    mpfr_atan2(a_ofgot, SI, SR, MPFR_RNDN);
    widen<T>(lg.im, G);
    std::printf("    %-14s %-24.16Re want %-24.16Re  rel %10.3g ulps   (vs arg of the\n",
                "arg(rp+rm)", G, (mpfr_srcptr)a_exact, ulps_real(G, a_exact, p));
    std::printf("    %-14s %49s  rel %10.3g ulps    computed sum)\n", "", "",
                ulps_real(G, a_ofgot, p));

    mpfr_clears(X, Y, t, G, a_exact, a_ofgot, SR, SI, (mpfr_ptr)0);
    mpc_clear(e_hi); mpc_clear(e_rp); mpc_clear(e_rm); mpc_clear(e_sum);
}


std::vector<Pt> load_grid() {
    std::vector<Pt> v;
    FILE* f = std::fopen(kGrid, "r");
    if (!f) {
        std::fprintf(stderr, "cannot open %s (run from the repo root)\n", kGrid);
        std::exit(2);
    }
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] != 'c' || line[1] != ',') continue;
        char* fields[5];
        int nf = 0;
        fields[nf++] = line;
        for (char* q = line; *q && nf < 5; ++q)
            if (*q == ',') { *q = 0; fields[nf++] = q + 1; }
        if (nf < 5) continue;
        Pt p;
        p.idx = std::atoi(fields[1]);
        p.family = fields[2];
        p.re = std::strtod(fields[3], nullptr);
        p.im = std::strtod(fields[4], nullptr);
        v.push_back(p);
    }
    std::fclose(f);
    return v;
}

struct Row {
    const char* bk;
    int pt;
    const char* family;
    double a, b, g;    // component-relative, on Im: forms A, B and G
    double ma, mb, mg; // modulus-relative, whole value -- the sweep's metric
    double fl;         // component floor for Im
    bool lossy;        // form A's y/2 lost bits at this point (disproven hypothesis)
    bool narrow;       // form A's y/2 is below the format's full-width range
};
std::vector<Row> g_rows;
long g_mismatch = 0;   // grid points where the shipped acosh is not form B

template <class T>
void run(const std::vector<Pt>& grid) {
    using C = typename Tr<T>::C;
    const int p = Tr<T>::sig_bits();
    mpc_t zs, r;
    mpfr_t wa, wb, wg, wr;
    mpc_init2(zs, kPrec);
    mpc_init2(r, kPrec);
    mpfr_inits2(kPrec, wa, wb, wg, wr, (mpfr_ptr)0);
    for (const Pt& pt : grid) {
        const T x(pt.re), y(pt.im);
        widen(x, mpc_realref(zs));
        widen(y, mpc_imagref(zs));
        mpc_acosh(r, zs, MPC_RNDNN);
        const C got = xp::acosh(C(x, y));
        const T a = form_a<T>(x, y);
        const T b = form_b<T>(x, y);
        const T g = form_g<T>(x, y);
        // Does the header actually compute form B?  Limb by limb, not by value:
        // a transcription slip that lands within a rounding of the right answer
        // is still a slip, and == on the widened sum would miss it.
        if (!g_poison) {
            bool same = true;
            for (int i = 0; i < Tr<T>::nlimb(); ++i)
                if (Tr<T>::limb(got.im, i) != Tr<T>::limb(g, i)) same = false;
            if (!same && g_mismatch++ < 6)
                std::printf("  SHIPPED != FORM G  %s pt %d\n", Tr<T>::name(), pt.idx);
        }
        widen(a, wa);
        widen(b, wb);
        widen(g, wg);
        const double ua = ulps_real(wa, mpc_imagref(r), p);
        const double ub = ulps_real(wb, mpc_imagref(r), p);
        const double ug = ulps_real(wg, mpc_imagref(r), p);
        // Re is identical under both forms, so the two modulus readings differ
        // only through Im -- exactly the substitution under test.
        widen(got.re, wr);
        const double ma = ulps_mod(wr, wa, r, p);
        const double mb = ulps_mod(wr, wb, r, p);
        const double mg = ulps_mod(wr, wg, r, p);
        const double fl = floor_ulps(mpc_imagref(r), p, Tr<T>::denorm_exp());
        if (ua < 0 || ub < 0 || ug < 0 || ma < 0 || mb < 0 || mg < 0) continue;
        g_rows.push_back(
            {Tr<T>::name(), pt.idx, pt.family.c_str(), ua, ub, ug, ma, mb, mg, fl,
             halving_lossy<T>(y),
             chain_operand_short<T>(x, y)});
    }
    mpfr_clears(wa, wb, wg, wr, (mpfr_ptr)0);
    mpc_clear(zs);
    mpc_clear(r);
}

// ------------------------------------------------------------------ the cut
//
// Im acosh carries the sheet.  Eight cases, value and sign of zero.  `wz` is
// 0 for "not a zero", +1 for "must be +0", -1 for "must be -0".

struct Cut {
    const char* label;
    double re, im;
    double want;   // exact expected Im acosh
    int wz;
};

int g_cut_fail = 0;

template <class T>
void check_cut(const char* form, const T& got, const Cut& c) {
    const double g = [&] {
        double s = 0;
        for (int i = 0; i < Tr<T>::nlimb(); ++i) s += Tr<T>::limb(got, i);
        return s;
    }();
    bool bad;
    if (c.wz != 0) {
        bad = !(g == 0.0) || (std::signbit(lead(got)) != (c.wz < 0));
    } else {
        // A branch check, not an accuracy measurement: this only has to catch a
        // wrong sheet, which is off by pi or by a sign, not by an ulp.
        bad = !(std::fabs(g - c.want) <= 1e-6 * std::fabs(c.want));
    }
    if (bad) {
        // The wanted sign comes from wz for a zero (where the value cannot carry
        // it) and from the value itself otherwise.  Reading it from wz in both
        // cases printed "+pi" for a case that wants -pi.
        const bool wneg = c.wz != 0 ? (c.wz < 0) : std::signbit(c.want);
        std::printf("  FAIL %-3s %-5s %-11s Im: got %s%.10g, want %s%.10g\n", Tr<T>::name(),
                    form, c.label, std::signbit(lead(got)) ? "-" : "+", std::fabs(g),
                    wneg ? "-" : "+", std::fabs(c.want));
        ++g_cut_fail;
    }
}

const double kPi = 3.14159265358979323846;

const Cut kCuts[] = {
    // |x| <= 1 on the real axis: acosh is purely imaginary, Im = +-acos(x).
    {"z = 0.5+0i", 0.5, 0.0, 1.0471975511965979, 0},
    {"z = 0.5-0i", 0.5, -0.0, -1.0471975511965979, 0},
    // |x| > 1, x > 0: acosh is real, so Im is a zero and only its SIGN is left.
    {"z = 2+0i", 2.0, 0.0, 0.0, +1},
    {"z = 2-0i", 2.0, -0.0, 0.0, -1},
    // |x| > 1, x < 0: the sheet is +-pi, and the sign of the zero picks it.
    {"z = -2+0i", -2.0, 0.0, kPi, 0},
    {"z = -2-0i", -2.0, -0.0, -kPi, 0},
    // the branch points themselves
    {"z = 1+0i", 1.0, 0.0, 0.0, +1},
    {"z = -1+0i", -1.0, 0.0, kPi, 0},
    // A NEGATIVE ZERO REAL PART.  C99 Annex G: cacos(+-0 + 0i) is pi/2 - 0i for
    // BOTH signs, so Im acosh is pi/2 either way.  Worth checking explicitly
    // because the rotation's atan2(leg, x) is handed that zero directly, and
    // atan2(positive, -0.0) is pi, not pi/2 -- if leg were positive here the
    // rotation would land a whole quadrant out.
    {"z = 0+0i", 0.0, 0.0, 1.5707963267948966, 0},
    {"z = -0+0i", -0.0, 0.0, 1.5707963267948966, 0},
    // A NEGATIVE REAL PART INSIDE THE SEGMENT, both signs of the zero.  The
    // acosh() header comment claimed Kahan's chain returned +1.670964i for the
    // -0i case on QF and TF -- the wrong sheet -- so the claim is checked here
    // rather than believed.  Im acosh(-0.1 -+ 0i) = -+acos(-0.1).
    {"z = -0.1+0i", -0.1, 0.0, 1.6709637479564565, 0},
    {"z = -0.1-0i", -0.1, -0.0, -1.6709637479564565, 0},
};

template <class T>
void run_cuts() {
    using C = typename Tr<T>::C;
    for (const Cut& c : kCuts) {
        const C z{T(c.re), T(c.im)};
        check_cut<T>("A", form_a<T>(T(c.re), T(c.im)), c);
        check_cut<T>("S", xp::acosh(z).im, c);
        check_cut<T>("B", form_b<T>(T(c.re), T(c.im)), c);
        check_cut<T>("G", form_g<T>(T(c.re), T(c.im)), c);
    }
}

double med(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int trace_pt = -1;
    const char* csv = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--poison")) g_poison = true;
        else if (!std::strcmp(argv[i], "--poison-guard")) g_poison_guard = true;
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!std::strcmp(argv[i], "--trace") && i + 1 < argc) trace_pt = std::atoi(argv[++i]);
    }

    const std::vector<Pt> grid = load_grid();

    if (trace_pt >= 0) {
        for (const Pt& q : grid)
            if (q.idx == trace_pt) {
                std::printf("TRACE point %d (%s)\n", q.idx, q.family.c_str());
                trace<xp::DoubleDouble>(q);
                trace<xp::FloatFloat>(q);
                trace<xp::QuadFloat>(q);
                trace<xp::TripleFloat>(q);
            }
        return 0;
    }
    std::printf("# grid %zu complex points, MPC %s at %ld bits\n", grid.size(), MPC_VERSION_STRING,
                (long)kPrec);
    if (g_poison || g_poison_guard)
        std::printf(g_poison ? "# POISON: form B uses atan2(leg, |x|), losing the sign of the real part\n"
                 : "# POISON: the guard predicate is forced false, so form G degrades to form A\n");

    run<xp::DoubleDouble>(grid);
    run<xp::FloatFloat>(grid);
    run<xp::QuadFloat>(grid);
    run<xp::TripleFloat>(grid);

    if (csv) {
        FILE* f = std::fopen(csv, "w");
        std::fprintf(f, "backend,point,family,im_a,im_b,im_g,mod_a,mod_b,mod_g,im_floor,halving_lossy,narrowed\n");
        for (const Row& r : g_rows)
            std::fprintf(f, "%s,%d,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d\n", r.bk, r.pt, r.family,
                         r.a, r.b, r.g, r.ma, r.mb, r.mg, r.fl, r.lossy ? 1 : 0, r.narrow ? 1 : 0);
        std::fclose(f);
        std::printf("wrote %zu rows to %s\n", g_rows.size(), csv);
    }

    std::printf("\nIm acosh(x+iy): FORM A (shipped, Kahan sqrt/log) vs FORM B (Re acos)\n");
    std::printf("%-3s %7s %9s %9s %12s %12s %9s %9s\n", "bk", "rows", "med A", "med B", "max A",
                "max B", "B worse", "B >2x A");
    const char* bks[] = {"DD", "FF", "QF", "TF"};
    for (const char* bk : bks) {
        std::vector<double> A, B;
        double mA = 0, mB = 0;
        int worse = 0, twice = 0;
        for (const Row& r : g_rows) {
            if (std::strcmp(r.bk, bk)) continue;
            A.push_back(r.a);
            B.push_back(r.b);
            mA = std::max(mA, r.a);
            mB = std::max(mB, r.b);
            if (r.b > r.a) ++worse;
            if (r.b > 1.0 && r.b > 2.0 * r.a) ++twice;
        }
        if (A.empty()) continue;
        std::printf("%-3s %7zu %9.4g %9.4g %12.5g %12.5g %9d %9d\n", bk, A.size(), med(A), med(B),
                    mA, mB, worse, twice);
    }

    // The ladder this was built for: z = (1, +-eps), points 878..895.
    std::printf("\nthe cut-re ladder at the branch point z = 1\n");
    for (const Row& r : g_rows)
        if (r.pt >= 878 && r.pt <= 895 && (r.pt % 2) == 0 && (r.a > 8.0 || r.b > 8.0))
            std::printf("  %s pt %-4d  A %12.5g   B %12.5g\n", r.bk, r.pt, r.a, r.b);

    std::printf("\nBRANCH CUT, %d cases x 4 backends x 4 (form A, form B, form G, shipped)\n",
                (int)(sizeof kCuts / sizeof kCuts[0]));
    run_cuts<xp::DoubleDouble>();
    run_cuts<xp::FloatFloat>();
    run_cuts<xp::QuadFloat>();
    run_cuts<xp::TripleFloat>();
    std::printf("  %d of %d component checks failed\n", g_cut_fail,
                (int)(sizeof kCuts / sizeof kCuts[0]) * 4 * 3);
    if (g_poison)
        std::printf("\nSHIPPED vs FORM G: not run -- a poison is active\n");
    else
        std::printf("\nSHIPPED vs FORM G: %ld of %zu grid points differ in any limb\n",
                    g_mismatch, g_rows.size());

    if (g_poison) {
        int bad = 0;
        for (const Row& r : g_rows)
            if (r.b > 1.0 && r.b > 2.0 * r.a) ++bad;
        std::printf("\nPOISON: form B's atan2 denominator stripped of its sign.  It must now\n"
                    "be materially worse than the shipped form somewhere, or the probe\n"
                    "cannot tell a right rotation from a wrong one.\n"
                    "  rows where B is above 1 ulp and more than 2x A: %d\n", bad);
        std::printf("  %s -- the comparison is live.\n", bad > 0 ? "PASS" : "FAIL");
        return bad > 0 ? 0 : 1;
    }
    return 0;
}
