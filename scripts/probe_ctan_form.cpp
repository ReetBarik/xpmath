// probe_ctan_form.cpp -- ablate two ways to compute complex tan and tanh on
// their DIRECT branches, in each backend's own arithmetic, against MPC, before
// either of them is shipped.  A composition floor is measured alongside so the
// residual can be called inherent or not on evidence rather than on assertion.
//
// THE POPULATION.  There is NO DEFECT here by the repo's verdict rule: of 14,224
// scored complex tan/tanh rows in the sweep, 2,395 read above 1 ulp and ZERO
// read above 8, the largest being 5.568.  Every row passes its bound.  What is
// under test is whether the sub-8-ulp floor is reducible, which is a different
// question and needs a different kind of answer.
//
// THE STRUCTURAL CLUE.  Both functions have two branches and they behave very
// differently on the same grid:
//
//     tan  |Im z| >= 2  (asymptotic, real denominator)     21 of 1376 above 1 ulp
//     tan  |Im z| <  2  (direct, sin(z)/cos(z))          1509 of 5736
//     tanh |Re z| >= 2  (asymptotic, real denominator)     26 of 2128
//     tanh |Re z| <  2  (direct)                          839 of 4984
//
// 1.3% against 21.9%.  That is suggestive and it is NOT proof -- the two
// branches sample different regions, and |Im z| >= 2 is where tan is nearly the
// constant i and is trivially well conditioned.  The ablation below is the
// proof, because it runs both forms at the SAME points.
//
// THE MECHANISM.  Write s = sin x, c = cos x, S = sinh y, C = cosh y.
//
// tan's direct branch is sin(z)/cos(z) with sin(z) = s*C + i*c*S and
// cos(z) = c*C - i*s*S.  The complex divide's real numerator is the determinant
// a*p + b*q with a = s*C, b = c*S, p = c*C, q = -s*S:
//
//     a*p + b*q = s*c*C^2 - s*c*S^2 = s*c*(C^2 - S^2) = s*c,
//
// a difference of two terms each C^2/(C^2-S^2) = cosh^2(y) times larger than
// their difference.  s*C and c*S are already rounded when the divide sees them,
// so their relative error is amplified into Re tan by cosh(2y): 27.3x, 4.8 bits,
// at the |Im z| = 2 seam.  A compensated determinant cannot recover this -- the
// error is in the OPERANDS, not in the summation.  (dd_cross is compensated and
// DD is still the worst backend here, which is consistent.)
//
// tanh's direct branch already has a real denominator, and loses its bits in a
// different place: it forms 1 - tanh^2(x) by cancellation.  That quantity is
// exactly sech^2(x), and the cancellation's amplification is again cosh(2x),
// 27.3x at the |Re z| = 2 seam.  Predicted consequence: tan should be worse than
// tanh, because tan pays the amplification AND the complex divide.  Measured
// 1509 against 839.
//
//     FORM A   tan:  sin(z) / cos(z)                       [shipped]
//              tanh: T = tanh x;  D = cos^2 y + T^2 sin^2 y
//                    Re = T/D,  Im = sin y cos y (1 - T^2)/D
//
//     FORM B   D = S^2 + c^2  for tan,   D = S^2 + c^2 with the roles of the
//              two arguments exchanged for tanh:
//              tan:  Re = s*c/D,        Im = S*C/D,   D = sinh^2 y + cos^2 x
//              tanh: Re = S*C/D,        Im = s*c/D,   D = sinh^2 x + cos^2 y
//
// Form B has no cancellation anywhere: D is a sum of two non-negative terms and
// vanishes only at the true poles.  The two functions are exact duals of one
// another under x <-> y and Re <-> Im, which is why one pair of formulas covers
// both and why a transcription slip in one shows up as an asymmetry.
//
// THE METRIC, AND A WARNING CARRIED OVER FROM THE acosh STEP.  The sweep scores
// complex rows MODULUS-RELATIVE (sweep_accuracy.cpp:2105).  That metric damps
// exactly this mechanism: where the amplification is largest the damaged
// component is the SMALL one.  At y = 2, Re tan ~ 0.038 against |tan| ~ 1, so a
// 27x amplified relative error on Re arrives as ~1x on the modulus.  Both
// metrics are therefore reported per row.  The modulus one is the metric of
// record and decides whether anything ships; the component one is what the
// acosh step proved can move by 0.9 digits while the modulus reading stays
// BIT-IDENTICAL, and it is what says whether the mechanism above is real.
//
// THE FLOOR.  Any implementation built on the format's own sin/cos/sinh/cosh
// inherits their rounding.  Two floors are measured:
//
//     FL_ARITH   correctly rounded s, c, S, C -> form B's algebra in the backend
//     FL_EXACT   correctly rounded s, c, S, C -> the same algebra at 400 bits
//
// FL_EXACT is a hard lower bound for ANY formula in these four blocks; FL_ARITH
// adds the cost of the backend's own multiplies and divides.  If form B sits at
// FL_ARITH the formula is finished and the rest is the real library's.  If
// FL_EXACT itself reads above 1 ulp the residual is inherent to composing
// rounded blocks in a fixed width, and no formula reaches it.
//
// POISONS, three of them, because three separate things are being claimed.
//
//   --poison-b   swaps cos^2 for sin^2 in form B's denominator, in both
//                functions and in both floors.  Algebraically wrong but
//                numerically close wherever |sin| ~ |cos|, so it is a real test
//                that the grid has discriminating points rather than a
//                tautology.  Form B's readings must explode and the cut table
//                must fail.
//   --poison-a   perturbs form A's transcription (conjugates cos(z)'s argument),
//                which is a no-op wherever Im z = 0 and wrong elsewhere.  The
//                limb-for-limb SHIPPED == FORM A check must fire.  Without this
//                the claim `form A is what the header does' is untested.
//   --poison-cut flips the expected sign of every zero in the special-value
//                table.  A correct build must then fail it.
//
// SEAM.  Replacing the direct branch changes the discontinuity where it meets
// the asymptotic branch at |Im z| = 2 / |Re z| = 2.  --seam measures that jump
// under form A and under form B; a formula that improves the interior while
// tearing the seam wider is not an improvement.
//
//   module load gcc/13.3.0
//   g++ -O2 -std=c++17 -fext-numeric-literals -DXPMATH_ENABLE_DIAGNOSTICS=0 \
//       -I include scripts/probe_ctan_form.cpp -o /tmp/probe_ctan_form \
//       -lmpc -lmpfr -lgmp
//
//   /tmp/probe_ctan_form                      # ablation + cut table
//   /tmp/probe_ctan_form --poison-a
//   /tmp/probe_ctan_form --poison-b
//   /tmp/probe_ctan_form --poison-cut
//   /tmp/probe_ctan_form --seam
//   /tmp/probe_ctan_form --blocks             # attribute to the real blocks
//   /tmp/probe_ctan_form --csv /tmp/ctan.csv
//
// Run from the repo root; the grid is read from validation/sweep/.
//
// ---------------------------------------------------------------------------
// THE OPERAND TRAP, and why the first run of this probe was wrong.
//
// FF, QF and TF each have BOTH a T(float) and a T(double) constructor, and the
// T(double) one SPLITS the double across words.  sweep_complex() builds its
// operands with the double form (sweep_accuracy.cpp:2511).  This probe first
// cast to the backend's scalar type, which on the three FP32 backends truncated
// the operand to float and measured a DIFFERENT POINT: TF c tan point 116 read
// 0.2306 ulps instead of the sweep's 5.568.  Caught by cross-checking form A
// against the sweep rather than by inspection.  Every FF/QF/TF number from that
// run was invalid.  Operands are now built as T(pt.re) / T(pt.im); after the
// fix QF and TF reproduce the sweep's counts and TF tan's 5.568 maximum exactly.
//
// ---------------------------------------------------------------------------
// RESULT: STOP AND REPORT.  NOTHING SHIPS FROM THIS PROBE.
//
// Stop condition, fixed BEFORE implementing: form B ships only if it cuts
// direct-branch rows above 1 ulp by >= 30% on the modulus-relative metric of
// record while making no row worse.  Measured, 10,736 direct-branch rows over
// four backends, scored against MPC:
//
//     form A (shipped)                        2332 rows > 1 ulp   max 5.568
//     form B (cancellation-free)              2094               max 5.310
//     FL_ARITH (exact blocks, backend algebra) 1021               max 4.062
//     FL_EXACT (exact blocks, 400-bit algebra)    0               max 0.6362
//
// Form B is worth 10.2%, a third of the bar, and it makes 3,886 rows WORSE (354
// of them by more than 1 ulp, worst +3.518); DD tanh and TF tan regress
// outright.  No implementable guard rescues it: bucketing by the predicted
// amplification cosh(2h) shows the mechanism exactly where predicted -- at
// amp >= 12, form A's 26 rows become form B's 7 -- but the amp >= 4 buckets hold
// only 256 of 10,736 rows, so a perfect guard buys ~30 rows (1.3%).  Even the
// unachievable oracle guard min(A, B) reaches only 1456 (62.4% of A).
//
// THE MECHANISM IS REAL AND THE METRIC OF RECORD CANNOT SEE IT.  Per COMPONENT,
// form B improves the worst cells by 4-9x: DD tan 33.52 -> 7.316, FF tan
// 50.25 -> 5.31, DD tanh 16.26 -> 5.812, FF tanh 20.31 -> 4.998.  At the seam
// (--seam, DD tan at x = 3) form A reads 26.42 component ulps against the
// predicted cosh(4) = 27.3 amplification and form B reads 0.3054 -- and the
// modulus metric reads that same pair as 0.436 against 0.3054, i.e. as nothing.
// This is the second independent confirmation, after the acosh step, that the
// modulus-relative metric is blind to a cancelling component.  It is NOT a
// reason to change the metric here: a per-component metric was tried and
// reverted on measurement (6,105 false defects), and QF/TF component maxima in
// this very run read ~1.78e14 under form A, form B AND FL_ARITH alike -- the
// same collapsing-scale artifact that caused the revert.
//
// WHERE THE RESIDUAL ACTUALLY LIVES.  FL_EXACT = 0 rows above 1 ulp on every one
// of the eight cells proves the residual is NOT inherent to the composition.
// FL_ARITH = 1021 splits form B's 2094 nearly in half: ~49% is the backend's own
// multiply/add/divide, ~51% is the four real blocks.  --blocks scores those
// blocks on the direct branch's own arguments against MPFR:
//
//     DD  2684 args | sin >1ulp 223 max 2.945 | cos 135 max 2.65  | sinh 88 | cosh 49
//     FF                | sin  48 max 1.641 | cos 379 max 3.52  | sinh 67 | cosh 39
//     QF                | sin   0 max 0.3037| cos   0 max 0.6191| sinh  4 | cosh  2
//     TF                | sin   0 max 0.5914| cos   0 max 0.5523| sinh 12 max 4.344
//
// The backends whose real blocks are inaccurate (DD, FF) are exactly the ones
// whose complex tan/tanh is inaccurate (DD tan 672 and FF tan 701 rows above
// 1 ulp, against QF 85 and TF 30).  Complex tan/tanh accuracy on this grid is
// therefore NOT a complex-trig formula problem; it is DD/FF real sin/cos plus
// extended-precision multiply/divide rounding.  Neither is in scope here and
// neither is reachable by reformulating the complex functions.
//
// ---------------------------------------------------------------------------
// VALIDATION.  Shipped == form A limb-for-limb at 0 of 10,736 point-backend
// pairs disagreeing.  --poison-a fires 8,819 mismatches (0 clean), so that
// equality is a real check.  --poison-b puts all 10,432 scorable rows above
// 1 ulp (max 8.113e+91) and takes cut failures 144 -> 224.  --poison-cut takes
// them 144 -> 192.
//
// THE 144 CUT FAILURES ARE PRE-EXISTING AND UPSTREAM.  They are IDENTICAL for
// ship, form A and form B (zero form-B-only, zero ship-only), and they are not
// a complex-layer defect at all: the real blocks lose the sign of zero, with
// sin(-0) and tanh(-0) returning +0 on all four backends.  No complex formula
// can restore a sign its inputs have already discarded.  Filed as an
// observation, not fixed: out of scope for an accuracy task.
//
// THE ORACLE.  MPC is used here, not libquadmath, and unlike the earlier
// complex-oracle attempt that produced 1,996 false defects this is safe FOR
// THESE TWO FUNCTIONS SPECIFICALLY: tan and tanh are meromorphic and have no
// branch cuts, so the (-0.0)+(+0.0) = +0.0 sign-of-zero loss in to_q()'s widening
// cannot select the wrong sheet.  Cross-checked rather than asserted: over the
// 10,720 rows matched to the sweep, mean 0.6278 (MPC) against 0.6307
// (libquadmath), the above-1-ulp verdict differs on 44 rows (0.41%), the largest
// single gap is 0.8256 ulps, and QF/TF counts reproduce exactly.

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
const double kAsymptote = 2.0;   // kXpTanAsymptote in the four headers

bool g_poison_a = false;
bool g_poison_b = false;
bool g_poison_cut = false;

// ---------------------------------------------------------------- backends
//
// sincos AND sinhcosh HAVE DIFFERENT OUT-PARAMETER ORDERS ON DIFFERENT
// BACKENDS.  dd/ff spell them (a, cos, sin) and (a, cosh, sinh); qf/tf spell
// them (a, sin, cos) and (a, sinh, cosh).  The shipped complex code gets this
// right by swapping at each call site, which makes the trap invisible when
// reading one header at a time.  It is wrapped here once, in the traits, so the
// forms below can be written the same way for all four and a swap cannot be
// introduced by copy-paste.

template <class T>
struct Tr;
template <>
struct Tr<xp::DoubleDouble> {
    using C = xp::DoubleDoubleComplex;
    using S = double;
    static const char* name() { return "DD"; }
    static int sig_bits() { return 106; }
    static int denorm_exp() { return -1074; }
    static int nlimb() { return 2; }
    static bool is_fp32() { return false; }
    static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
    static void sc(xp::DoubleDouble a, xp::DoubleDouble& s, xp::DoubleDouble& c) {
        xp::sincos(a, c, s);          // cos first
    }
    static void shch(xp::DoubleDouble a, xp::DoubleDouble& sh, xp::DoubleDouble& ch) {
        xp::sinhcosh(a, ch, sh);      // cosh first
    }
};
template <>
struct Tr<xp::FloatFloat> {
    using C = xp::FloatFloatComplex;
    using S = float;
    static const char* name() { return "FF"; }
    static int sig_bits() { return 48; }
    static int denorm_exp() { return -149; }
    static int nlimb() { return 2; }
    static bool is_fp32() { return true; }
    static double limb(const xp::FloatFloat& v, int i) { return i ? v.lo : v.hi; }
    static void sc(xp::FloatFloat a, xp::FloatFloat& s, xp::FloatFloat& c) {
        xp::sincos(a, c, s);          // cos first
    }
    static void shch(xp::FloatFloat a, xp::FloatFloat& sh, xp::FloatFloat& ch) {
        xp::sinhcosh(a, ch, sh);      // cosh first
    }
};
template <>
struct Tr<xp::QuadFloat> {
    using C = xp::QuadFloatComplex;
    using S = float;
    static const char* name() { return "QF"; }
    static int sig_bits() { return 96; }
    static int denorm_exp() { return -149; }
    static int nlimb() { return 4; }
    static bool is_fp32() { return true; }
    static double limb(const xp::QuadFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : i == 2 ? v.f2 : v.f3;
    }
    static void sc(xp::QuadFloat a, xp::QuadFloat& s, xp::QuadFloat& c) {
        xp::sincos(a, s, c);          // sin first
    }
    static void shch(xp::QuadFloat a, xp::QuadFloat& sh, xp::QuadFloat& ch) {
        xp::sinhcosh(a, sh, ch);      // sinh first
    }
};
template <>
struct Tr<xp::TripleFloat> {
    using C = xp::TripleFloatComplex;
    using S = float;
    static const char* name() { return "TF"; }
    static int sig_bits() { return 72; }
    static int denorm_exp() { return -149; }
    static int nlimb() { return 3; }
    static bool is_fp32() { return true; }
    static double limb(const xp::TripleFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : v.f2;
    }
    static void sc(xp::TripleFloat a, xp::TripleFloat& s, xp::TripleFloat& c) {
        xp::sincos(a, s, c);          // sin first
    }
    static void shch(xp::TripleFloat a, xp::TripleFloat& sh, xp::TripleFloat& ch) {
        xp::sinhcosh(a, sh, ch);      // sinh first
    }
};

template <class T>
T mk(double w) {
    return T(static_cast<typename Tr<T>::S>(w));
}
template <class T>
double lead(const T& v) {
    return Tr<T>::limb(v, 0);
}

// Sum the limbs.  This widening destroys the sign of a zero -- (-0.0) + (+0.0)
// is +0.0, the trap that made an earlier complex oracle answer for the
// conjugate -- so the sign is restored from limb 0 afterwards.
template <class T>
void widen(const T& v, mpfr_ptr out) {
    mpfr_set_d(out, Tr<T>::limb(v, 0), MPFR_RNDN);
    for (int i = 1; i < Tr<T>::nlimb(); ++i) mpfr_add_d(out, out, Tr<T>::limb(v, i), MPFR_RNDN);
    if (mpfr_zero_p(out) && std::signbit(Tr<T>::limb(v, 0)))
        mpfr_setsign(out, out, 1, MPFR_RNDN);
}

// ---------------------------------------------------------------- metrics

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

// The SWEEP's metric, transcribed: |got - ref| / |ref| * 2^p over the whole
// complex value (sweep_accuracy.cpp:2105-2140).  Metric of record.
double ulps_mod(mpfr_srcptr gr, mpfr_srcptr gi, mpc_srcptr ref, int p) {
    static bool init = false;
    static mpfr_t a, b, m;
    if (!init) { mpfr_inits2(kPrec, a, b, m, (mpfr_ptr)0); init = true; }
    if (!mpfr_number_p(gr) || !mpfr_number_p(gi)) return -1.0;
    mpfr_hypot(m, mpc_realref(ref), mpc_imagref(ref), MPFR_RNDN);
    if (!mpfr_number_p(m) || mpfr_zero_p(m)) return -1.0;
    mpfr_sub(a, gr, mpc_realref(ref), MPFR_RNDN);
    mpfr_sub(b, gi, mpc_imagref(ref), MPFR_RNDN);
    mpfr_hypot(a, a, b, MPFR_RNDN);
    mpfr_div(a, a, m, MPFR_RNDN);
    mpfr_mul_2si(a, a, p, MPFR_RNDN);
    return mpfr_get_d(a, MPFR_RNDN);
}

// The worse of the two components, each scored relative to itself.  This is the
// reading the modulus metric can hide.
double ulps_comp(mpfr_srcptr gr, mpfr_srcptr gi, mpc_srcptr ref, int p) {
    const double a = ulps_real(gr, mpc_realref(ref), p);
    const double b = ulps_real(gi, mpc_imagref(ref), p);
    if (a < 0.0 || b < 0.0) return -1.0;
    return a > b ? a : b;
}

// ---------------------------------------------------------------- rounding

// Round an exact 400-bit value to what the format can actually hold, by
// repeated leading-word extraction -- the same construction the expansion
// itself uses, so the result is a representable value of type T.
template <class T>
T from_mpfr(mpfr_srcptr x) {
    static bool init = false;
    static mpfr_t rem;
    if (!init) { mpfr_init2(rem, kPrec); init = true; }
    mpfr_set(rem, x, MPFR_RNDN);
    T out = mk<T>(0.0);
    for (int i = 0; i < Tr<T>::nlimb(); ++i) {
        const double w =
            Tr<T>::is_fp32() ? (double)mpfr_get_flt(rem, MPFR_RNDN) : mpfr_get_d(rem, MPFR_RNDN);
        if (!std::isfinite(w) || w == 0.0) break;
        out = xp::add(out, mk<T>(w));
        mpfr_sub_d(rem, rem, w, MPFR_RNDN);
    }
    return out;
}

// ---------------------------------------------------------------- forms

// FORM A, tan: exactly what the header does below the asymptote.  --poison-a
// conjugates cos(z)'s argument: a no-op on the real axis, wrong off it, so the
// limb-for-limb equality check against the shipped function must catch it.
template <class T>
typename Tr<T>::C form_a_tan(const T& x, const T& y) {
    using C = typename Tr<T>::C;
    return xp::sin(C(x, y)) / xp::cos(C(x, g_poison_a ? xp::negate(y) : y));
}

// FORM A, tanh: transcribed from the header's direct branch.  The 1 - T^2 is
// the cancellation named at the top of the file.
template <class T>
typename Tr<T>::C form_a_tanh(const T& x, const T& y) {
    using C = typename Tr<T>::C;
    const T T_ = xp::tanh(g_poison_a ? xp::negate(x) : x);
    T s, c;
    Tr<T>::sc(y, s, c);
    const T T2 = xp::multiply(T_, T_);
    const T den = xp::add(xp::multiply(c, c), xp::multiply(T2, xp::multiply(s, s)));
    return C(xp::divide(T_, den),
             xp::divide(xp::multiply(xp::multiply(s, c), xp::subtract(mk<T>(1.0), T2)), den));
}

// FORM B, both functions, from the four blocks.  `swap` selects tanh: the two
// are duals under x <-> y and Re <-> Im, so the same three lines serve both and
// an asymmetric slip cannot hide.
//
//   tan (x + iy)  = ( sin x cos x  + i sinh y cosh y ) / (sinh^2 y + cos^2 x)
//   tanh(x + iy)  = ( sinh x cosh x + i sin y cos y  ) / (sinh^2 x + cos^2 y)
//
// D is a sum of two non-negative terms and vanishes only at a true pole, so
// there is no cancellation anywhere in the form.
//
// Signed zeros survive: at y = -0 the tan case has sinh y = -0, cosh y = 1,
// D = cos^2 x, so Im = (-0 * 1)/D = -0 and Re = sin x cos x / cos^2 x = tan x.
template <class T>
typename Tr<T>::C form_b(const T& x, const T& y, bool swap) {
    using C = typename Tr<T>::C;
    const T& trig_arg = swap ? y : x;    // the argument that goes to sin/cos
    const T& hyp_arg  = swap ? x : y;    // the argument that goes to sinh/cosh
    T s, c, sh, ch;
    Tr<T>::sc(trig_arg, s, c);
    Tr<T>::shch(hyp_arg, sh, ch);
    // --poison-b: cos^2 -> sin^2 in the denominator.  Close wherever
    // |sin| ~ |cos|, so the grid has to contain points that tell them apart.
    const T guard = g_poison_b ? s : c;
    const T D = xp::add(xp::multiply(sh, sh), xp::multiply(guard, guard));
    const T trig_half = xp::divide(xp::multiply(s, c), D);
    const T hyp_half  = xp::divide(xp::multiply(sh, ch), D);
    return swap ? C(hyp_half, trig_half) : C(trig_half, hyp_half);
}

// FL_ARITH: form B's algebra, in the backend, on correctly rounded blocks.
template <class T>
typename Tr<T>::C floor_arith(mpfr_srcptr Xt, mpfr_srcptr Yt, bool swap) {
    using C = typename Tr<T>::C;
    static bool init = false;
    static mpfr_t s, c, sh, ch;
    if (!init) { mpfr_inits2(kPrec, s, c, sh, ch, (mpfr_ptr)0); init = true; }
    mpfr_srcptr ta = swap ? Yt : Xt;
    mpfr_srcptr ha = swap ? Xt : Yt;
    mpfr_sin(s, ta, MPFR_RNDN);
    mpfr_cos(c, ta, MPFR_RNDN);
    mpfr_sinh(sh, ha, MPFR_RNDN);
    mpfr_cosh(ch, ha, MPFR_RNDN);
    const T Ts = from_mpfr<T>(s), Tc = from_mpfr<T>(c);
    const T Tsh = from_mpfr<T>(sh), Tch = from_mpfr<T>(ch);
    const T guard = g_poison_b ? Ts : Tc;
    const T D = xp::add(xp::multiply(Tsh, Tsh), xp::multiply(guard, guard));
    const T trig_half = xp::divide(xp::multiply(Ts, Tc), D);
    const T hyp_half  = xp::divide(xp::multiply(Tsh, Tch), D);
    return swap ? C(hyp_half, trig_half) : C(trig_half, hyp_half);
}

// Round an exact value to the format's representable set and hand it back as
// an mpfr, so the exact floor below starts from the same blocks the arithmetic
// floor starts from.
template <class T>
void round_to_format(mpfr_srcptr x, mpfr_ptr out) {
    widen<T>(from_mpfr<T>(x), out);
}

// FL_EXACT: the same algebra at 400 bits on the same rounded blocks.  A hard
// lower bound for ANY formula built out of these four numbers -- it charges the
// rounding of sin/cos/sinh/cosh to the answer and nothing else.
template <class T>
void floor_exact(mpfr_srcptr Xt, mpfr_srcptr Yt, bool swap, mpfr_ptr out_re, mpfr_ptr out_im) {
    static bool init = false;
    static mpfr_t s, c, sh, ch, D, t, h;
    if (!init) { mpfr_inits2(kPrec, s, c, sh, ch, D, t, h, (mpfr_ptr)0); init = true; }
    mpfr_srcptr ta = swap ? Yt : Xt;
    mpfr_srcptr ha = swap ? Xt : Yt;
    mpfr_sin(t, ta, MPFR_RNDN);   round_to_format<T>(t, s);
    mpfr_cos(t, ta, MPFR_RNDN);   round_to_format<T>(t, c);
    mpfr_sinh(t, ha, MPFR_RNDN);  round_to_format<T>(t, sh);
    mpfr_cosh(t, ha, MPFR_RNDN);  round_to_format<T>(t, ch);

    mpfr_srcptr guard = g_poison_b ? (mpfr_srcptr)s : (mpfr_srcptr)c;
    mpfr_mul(D, sh, sh, MPFR_RNDN);
    mpfr_mul(t, guard, guard, MPFR_RNDN);
    mpfr_add(D, D, t, MPFR_RNDN);

    mpfr_mul(t, s, c, MPFR_RNDN);
    mpfr_div(t, t, D, MPFR_RNDN);        // sin*cos / D
    mpfr_mul(h, sh, ch, MPFR_RNDN);
    mpfr_div(h, h, D, MPFR_RNDN);        // sinh*cosh / D

    mpfr_set(out_re, swap ? h : t, MPFR_RNDN);
    mpfr_set(out_im, swap ? t : h, MPFR_RNDN);
}

// ---------------------------------------------------------------- grid

struct Pt {
    int idx;
    double re, im;
    std::string family;
};

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

// ---------------------------------------------------------------- ablation

struct Row {
    const char* bk;
    const char* op;
    int pt;
    std::string fam;
    double ma, mb, mfa, mfe;   // modulus-relative -- the metric of record
    double ca, cb, cfa, cfe;   // worse component, each relative to itself
};
std::vector<Row> g_rows;
long g_mismatch = 0;    // grid points where the shipped function is not form A

template <class T>
void run(const std::vector<Pt>& grid, bool swap) {
    using C = typename Tr<T>::C;
    const int p = Tr<T>::sig_bits();
    const char* op = swap ? "tanh" : "tan";
    mpc_t zs, ref;
    mpfr_t X, Y, ar, ai, br, bi, far, fai, fer, fei;
    mpc_init2(zs, kPrec);
    mpc_init2(ref, kPrec);
    mpfr_inits2(kPrec, X, Y, ar, ai, br, bi, far, fai, fer, fei, (mpfr_ptr)0);

    for (const Pt& pt : grid) {
        // T(double), NOT T(float): all three FP32 backends have a double
        // constructor that SPLITS the double across words, and sweep_complex()
        // builds its operands that way (sweep_accuracy.cpp:2511).  Casting to
        // float first truncates the operand and measures a different point --
        // it moved TF c tan point 116 from 5.568 ulps to 0.23.
        const T x(pt.re);
        const T y(pt.im);
        // The header's own branch test, on the leading word of the STORED
        // operand.  Only the direct branch is under test.
        const double gate = std::fabs(lead(swap ? x : y));
        if (!(gate < kAsymptote)) continue;

        widen<T>(x, X);
        widen<T>(y, Y);
        mpc_set_fr_fr(zs, X, Y, MPC_RNDNN);
        if (swap) mpc_tanh(ref, zs, MPC_RNDNN); else mpc_tan(ref, zs, MPC_RNDNN);
        if (!mpfr_number_p(mpc_realref(ref)) || !mpfr_number_p(mpc_imagref(ref))) continue;

        const C got = swap ? xp::tanh(C(x, y)) : xp::tan(C(x, y));
        const C a = swap ? form_a_tanh<T>(x, y) : form_a_tan<T>(x, y);
        const C b = form_b<T>(x, y, swap);
        const C fa = floor_arith<T>(X, Y, swap);
        floor_exact<T>(X, Y, swap, fer, fei);

        // Is the shipped function form A?  Limb by limb: a transcription slip
        // that lands within a rounding of the right answer is still a slip, and
        // == on the widened sum would miss it.
        bool same = true;
        for (int i = 0; i < Tr<T>::nlimb(); ++i) {
            if (Tr<T>::limb(got.re, i) != Tr<T>::limb(a.re, i)) same = false;
            if (Tr<T>::limb(got.im, i) != Tr<T>::limb(a.im, i)) same = false;
        }
        if (!same && g_mismatch++ < 8)
            std::printf("  SHIPPED != FORM A  %s c %s pt %d\n", Tr<T>::name(), op, pt.idx);

        widen<T>(a.re, ar);   widen<T>(a.im, ai);
        widen<T>(b.re, br);   widen<T>(b.im, bi);
        widen<T>(fa.re, far); widen<T>(fa.im, fai);

        Row r;
        r.bk = Tr<T>::name();
        r.op = op;
        r.pt = pt.idx;
        r.fam = pt.family;
        r.ma  = ulps_mod(ar, ai, ref, p);
        r.mb  = ulps_mod(br, bi, ref, p);
        r.mfa = ulps_mod(far, fai, ref, p);
        r.mfe = ulps_mod(fer, fei, ref, p);
        r.ca  = ulps_comp(ar, ai, ref, p);
        r.cb  = ulps_comp(br, bi, ref, p);
        r.cfa = ulps_comp(far, fai, ref, p);
        r.cfe = ulps_comp(fer, fei, ref, p);
        g_rows.push_back(r);
    }
    mpc_clear(zs);
    mpc_clear(ref);
    mpfr_clears(X, Y, ar, ai, br, bi, far, fai, fer, fei, (mpfr_ptr)0);
}

// ---------------------------------------------------------------- cut sheet

// Special values, checked by VALUE and by the SIGN OF EVERY ZERO.  Comparing
// +0.0 == -0.0 succeeds, which is exactly the hole this closes.  C99 Annex G
// asks tan and tanh to be odd in both arguments, so all four sign combinations
// of (+-0, +-0) must pass their signs straight through.
struct CutCase {
    const char* label;
    double re, im;
    int rekind;   // 0 = value in `val`, +1 = must be +0, -1 = must be -0
    int imkind;
    int reval;    // index into the value table: 0 none, 1 tan1, 2 tanh1, negated by sign
    int imval;
    int resign, imsign;
};

// tan(1) and tanh(1) at 400 bits; index 1 and 2 above.
mpfr_srcptr cut_value(int which) {
    static bool init = false;
    static mpfr_t t1, h1, zero;
    if (!init) {
        mpfr_inits2(kPrec, t1, h1, zero, (mpfr_ptr)0);
        mpfr_set_ui(t1, 1, MPFR_RNDN);
        mpfr_tan(t1, t1, MPFR_RNDN);
        mpfr_set_ui(h1, 1, MPFR_RNDN);
        mpfr_tanh(h1, h1, MPFR_RNDN);
        mpfr_set_zero(zero, 1);
        init = true;
    }
    return which == 1 ? t1 : which == 2 ? h1 : zero;
}

// tan: Re carries the circular part of x, Im the hyperbolic part of y.
const CutCase kTanCuts[] = {
    {"tan(+0+0i)",   0.0,  0.0, +1, +1, 0, 0, +1, +1},
    {"tan(-0+0i)",  -0.0,  0.0, -1, +1, 0, 0, -1, +1},
    {"tan(+0-0i)",   0.0, -0.0, +1, -1, 0, 0, +1, -1},
    {"tan(-0-0i)",  -0.0, -0.0, -1, -1, 0, 0, -1, -1},
    {"tan(1+0i)",    1.0,  0.0,  0, +1, 1, 0, +1, +1},
    {"tan(1-0i)",    1.0, -0.0,  0, -1, 1, 0, +1, -1},
    {"tan(-1+0i)",  -1.0,  0.0,  0, +1, 1, 0, -1, +1},
    {"tan(+0+1i)",   0.0,  1.0, +1,  0, 0, 2, +1, +1},
    {"tan(+0-1i)",   0.0, -1.0, +1,  0, 0, 2, +1, -1},
    {"tan(-0+1i)",  -0.0,  1.0, -1,  0, 0, 2, -1, +1},
};
// tanh is the dual: Re carries the hyperbolic part of x, Im the circular of y.
const CutCase kTanhCuts[] = {
    {"tanh(+0+0i)",  0.0,  0.0, +1, +1, 0, 0, +1, +1},
    {"tanh(-0+0i)", -0.0,  0.0, -1, +1, 0, 0, -1, +1},
    {"tanh(+0-0i)",  0.0, -0.0, +1, -1, 0, 0, +1, -1},
    {"tanh(-0-0i)", -0.0, -0.0, -1, -1, 0, 0, -1, -1},
    {"tanh(1+0i)",   1.0,  0.0,  0, +1, 2, 0, +1, +1},
    {"tanh(1-0i)",   1.0, -0.0,  0, -1, 2, 0, +1, -1},
    {"tanh(-1+0i)", -1.0,  0.0,  0, +1, 2, 0, -1, +1},
    {"tanh(+0+1i)",  0.0,  1.0, +1,  0, 0, 1, +1, +1},
    {"tanh(+0-1i)",  0.0, -1.0, +1,  0, 0, 1, +1, -1},
    {"tanh(-0+1i)", -0.0,  1.0, -1,  0, 0, 1, -1, +1},
};

int g_cut_fail = 0, g_cut_total = 0;

template <class T>
void check_cut_component(const char* bk, const char* label, const char* which, const char* part,
                         const T& got, int kind, int val, int sign) {
    ++g_cut_total;
    if (g_poison_cut && kind != 0) kind = -kind;
    static bool init = false;
    static mpfr_t g, want, e;
    if (!init) { mpfr_inits2(kPrec, g, want, e, (mpfr_ptr)0); init = true; }
    widen<T>(got, g);

    if (kind != 0) {
        const bool is_zero = mpfr_zero_p(g);
        const bool neg = std::signbit(lead(got));   // never off the widened sum
        if (!is_zero || (kind < 0) != neg) {
            std::printf("  CUT FAIL %s %-6s %-14s %s: got %s%.6g, want %c0\n", bk, which, label,
                        part, neg ? "-" : "+", std::fabs(mpfr_get_d(g, MPFR_RNDN)),
                        kind < 0 ? '-' : '+');
            ++g_cut_fail;
        }
        return;
    }
    mpfr_set(want, cut_value(val), MPFR_RNDN);
    if (sign < 0) mpfr_neg(want, want, MPFR_RNDN);
    // Loose on purpose: this is a BRANCH and SIGN check, not an accuracy
    // measurement.  A wrong sheet is off by a whole value, not by an ulp.
    const double tol = std::pow(10.0, -(Tr<T>::sig_bits() * 0.30103 - 2.0));
    mpfr_sub(e, g, want, MPFR_RNDN);
    mpfr_div(e, e, want, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    const double rel = mpfr_get_d(e, MPFR_RNDN);
    if (!(rel <= tol)) {
        std::printf("  CUT FAIL %s %-6s %-14s %s: rel %.3g > %.3g\n", bk, which, label, part, rel,
                    tol);
        ++g_cut_fail;
    }
}

template <class T>
void run_cuts() {
    using C = typename Tr<T>::C;
    const char* bk = Tr<T>::name();
    for (int fn = 0; fn < 2; ++fn) {
        const bool swap = (fn == 1);
        const CutCase* cases = swap ? kTanhCuts : kTanCuts;
        const int n = (int)(swap ? sizeof(kTanhCuts) : sizeof(kTanCuts)) / (int)sizeof(CutCase);
        for (int i = 0; i < n; ++i) {
            const CutCase& cc = cases[i];
            const T x(cc.re);   // T(double); see run() on why not T(float)
            const T y(cc.im);
            const C shipped = swap ? xp::tanh(C(x, y)) : xp::tan(C(x, y));
            const C a = swap ? form_a_tanh<T>(x, y) : form_a_tan<T>(x, y);
            const C b = form_b<T>(x, y, swap);
            struct { const char* tag; const C* v; } trio[] = {
                {"ship", &shipped}, {"formA", &a}, {"formB", &b}};
            for (auto& t : trio) {
                check_cut_component<T>(bk, cc.label, t.tag, "Re", t.v->re, cc.rekind, cc.reval,
                                       cc.resign);
                check_cut_component<T>(bk, cc.label, t.tag, "Im", t.v->im, cc.imkind, cc.imval,
                                       cc.imsign);
            }
        }
    }
}

// ---------------------------------------------------------------- seam

// The asymptotic branch, transcribed, so the seam can be measured.  Replacing
// the direct branch moves the discontinuity where the two meet; a formula that
// improves the interior while tearing the seam wider is not an improvement.
template <class T>
typename Tr<T>::C form_asym(const T& x, const T& y, bool swap) {
    using C = typename Tr<T>::C;
    using S = typename Tr<T>::S;
    const T& big = swap ? x : y;      // the argument that crossed the asymptote
    const T& sml = swap ? y : x;
    T s, c;
    Tr<T>::sc(sml, s, c);
    const T s2 = xp::multiply_scalar(xp::multiply(s, c), static_cast<S>(2.0));
    const T c2 = xp::multiply(xp::subtract(c, s), xp::add(c, s));
    const T t = xp::exp(xp::multiply_scalar(big, static_cast<S>(lead(big) < 0.0 ? 2.0 : -2.0)));
    const T t2 = xp::multiply(t, t);
    const T den = xp::add(xp::add(mk<T>(1.0), t2),
                          xp::multiply_scalar(xp::multiply(t, c2), static_cast<S>(2.0)));
    T along = xp::divide(xp::subtract(mk<T>(1.0), t2), den);          // -> +-1
    if (lead(big) < 0.0) along = xp::negate(along);
    const T across = xp::divide(xp::multiply_scalar(xp::multiply(t, s2), static_cast<S>(2.0)), den);
    return swap ? C(along, across) : C(across, along);
}

// ---------------------------------------------------------------- blocks

// THE REAL BUILDING BLOCKS, scored on their own.  The gap between FORM B and
// FL_ARITH is already an end-to-end attribution -- it is the same algebra with
// the library's sin/cos/sinh/cosh swapped for correctly rounded ones -- but the
// primitive numbers say WHICH block is carrying it, and that is what points at
// the next target.  Scored on every argument the direct branch actually feeds
// them: both components of every grid point, each in both roles.
template <class T>
void run_blocks(const std::vector<Pt>& grid) {
    const int p = Tr<T>::sig_bits();
    const char* names[4] = {"sin", "cos", "sinh", "cosh"};
    long n = 0, above[4] = {0, 0, 0, 0};
    double mx[4] = {0, 0, 0, 0};
    mpfr_t A, want, got;
    mpfr_inits2(kPrec, A, want, got, (mpfr_ptr)0);
    for (const Pt& pt : grid) {
        for (int half = 0; half < 2; ++half) {
            const double v = half ? pt.im : pt.re;
            if (!(std::fabs(v) < kAsymptote)) continue;   // direct branch only
            const T a(v);
            widen<T>(a, A);
            ++n;
            for (int k = 0; k < 4; ++k) {
                T g;
                switch (k) {
                    case 0: { T s, c; Tr<T>::sc(a, s, c);   g = s; mpfr_sin(want, A, MPFR_RNDN);  break; }
                    case 1: { T s, c; Tr<T>::sc(a, s, c);   g = c; mpfr_cos(want, A, MPFR_RNDN);  break; }
                    case 2: { T s, c; Tr<T>::shch(a, s, c); g = s; mpfr_sinh(want, A, MPFR_RNDN); break; }
                    default:{ T s, c; Tr<T>::shch(a, s, c); g = c; mpfr_cosh(want, A, MPFR_RNDN); break; }
                }
                widen<T>(g, got);
                const double u = ulps_real(got, want, p);
                if (u < 0.0) continue;
                if (u > 1.0) ++above[k];
                if (u > mx[k]) mx[k] = u;
            }
        }
    }
    std::printf("  %-3s %6ld args |", Tr<T>::name(), n);
    for (int k = 0; k < 4; ++k)
        std::printf("  %s >1ulp %5ld max %8.4g |", names[k], above[k], mx[k]);
    std::printf("\n");
    mpfr_clears(A, want, got, (mpfr_ptr)0);
}

template <class T>
void run_seam() {
    using C = typename Tr<T>::C;
    const int p = Tr<T>::sig_bits();
    const double xs[] = {0.1, 0.5, 1.0, 1.5, 3.0};
    mpc_t zs, ref;
    mpfr_t X, Y, r1, i1;
    mpc_init2(zs, kPrec);
    mpc_init2(ref, kPrec);
    mpfr_inits2(kPrec, X, Y, r1, i1, (mpfr_ptr)0);
    for (int fn = 0; fn < 2; ++fn) {
        const bool swap = (fn == 1);
        for (double xv : xs) {
            // The seam itself: the crossing argument sits exactly on 2, where
            // the header takes the asymptotic side.
            const T sml(xv);    // T(double); see run() on why not T(float)
            const T big(kAsymptote);
            const T x = swap ? big : sml;
            const T y = swap ? sml : big;
            widen<T>(x, X);
            widen<T>(y, Y);
            mpc_set_fr_fr(zs, X, Y, MPC_RNDNN);
            if (swap) mpc_tanh(ref, zs, MPC_RNDNN); else mpc_tanh(ref, zs, MPC_RNDNN);
            if (!swap) mpc_tan(ref, zs, MPC_RNDNN);
            const C as = form_asym<T>(x, y, swap);
            const C a = swap ? form_a_tanh<T>(x, y) : form_a_tan<T>(x, y);
            const C b = form_b<T>(x, y, swap);
            auto score = [&](const C& v) {
                widen<T>(v.re, r1);
                widen<T>(v.im, i1);
                return ulps_mod(r1, i1, ref, p);
            };
            auto scorec = [&](const C& v) {
                widen<T>(v.re, r1);
                widen<T>(v.im, i1);
                return ulps_comp(r1, i1, ref, p);
            };
            std::printf("  %s %-4s other=%4.1f  asym %9.4g /%9.4g   formA %9.4g /%9.4g   "
                        "formB %9.4g /%9.4g   (mod/comp ulps)\n",
                        Tr<T>::name(), swap ? "tanh" : "tan", xv, score(as), scorec(as), score(a),
                        scorec(a), score(b), scorec(b));
        }
    }
    mpc_clear(zs);
    mpc_clear(ref);
    mpfr_clears(X, Y, r1, i1, (mpfr_ptr)0);
}

// ---------------------------------------------------------------- report

struct Agg {
    long n = 0;
    long above[4] = {0, 0, 0, 0};       // A, B, FL_ARITH, FL_EXACT
    double mx[4] = {0, 0, 0, 0};
    long worse = 0, worse_1 = 0;        // B worse than A: strictly, and by >1 ulp
    double worst_reg = 0.0;
};

void tally(Agg& g, const double v[4]) {
    ++g.n;
    for (int i = 0; i < 4; ++i) {
        if (v[i] > 1.0) ++g.above[i];
        if (v[i] > g.mx[i]) g.mx[i] = v[i];
    }
    if (v[1] > v[0]) {
        ++g.worse;
        if (v[1] > v[0] + 1.0) ++g.worse_1;
        if (v[1] - v[0] > g.worst_reg) g.worst_reg = v[1] - v[0];
    }
}

void report(bool component) {
    std::printf("\n%s\n", component ? "PER-COMPONENT (worse of Re, Im -- what the modulus "
                                      "metric can hide)"
                                    : "MODULUS-RELATIVE (the sweep's metric, the metric of record)");
    std::printf("  %-3s %-5s %6s | %s\n", "bk", "op", "rows",
                "rows > 1 ulp: A / B / FLarith / FLexact        max: A / B / FLarith / FLexact"
                "     B worse: n (>1ulp)");
    const char* bks[] = {"DD", "FF", "QF", "TF"};
    const char* ops[] = {"tan", "tanh"};
    Agg all;
    for (const char* bk : bks) {
        for (const char* op : ops) {
            Agg g;
            for (const Row& r : g_rows) {
                if (std::strcmp(r.bk, bk) || std::strcmp(r.op, op)) continue;
                const double v[4] = {component ? r.ca : r.ma, component ? r.cb : r.mb,
                                     component ? r.cfa : r.mfa, component ? r.cfe : r.mfe};
                if (v[0] < 0 || v[1] < 0 || v[2] < 0 || v[3] < 0) continue;
                tally(g, v);
                tally(all, v);
            }
            if (!g.n) continue;
            std::printf("  %-3s %-5s %6ld | %6ld %6ld %6ld %6ld    %10.4g %10.4g %10.4g %10.4g"
                        "    %5ld (%ld)\n",
                        bk, op, g.n, g.above[0], g.above[1], g.above[2], g.above[3], g.mx[0],
                        g.mx[1], g.mx[2], g.mx[3], g.worse, g.worse_1);
        }
    }
    std::printf("  %-3s %-5s %6ld | %6ld %6ld %6ld %6ld    %10.4g %10.4g %10.4g %10.4g"
                "    %5ld (%ld)   worst regression %.4g\n",
                "ALL", "", all.n, all.above[0], all.above[1], all.above[2], all.above[3], all.mx[0],
                all.mx[1], all.mx[2], all.mx[3], all.worse, all.worse_1, all.worst_reg);
}

void write_csv(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return; }
    std::fprintf(f, "backend,op,point,family,mod_a,mod_b,mod_fla,mod_fle,"
                    "comp_a,comp_b,comp_fla,comp_fle\n");
    for (const Row& r : g_rows)
        std::fprintf(f, "%s,%s,%d,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", r.bk, r.op, r.pt,
                     r.fam.c_str(), r.ma, r.mb, r.mfa, r.mfe, r.ca, r.cb, r.cfa, r.cfe);
    std::fclose(f);
    std::printf("\nwrote %s (%zu rows)\n", path, g_rows.size());
}

}  // namespace

int main(int argc, char** argv) {
    const char* csv = nullptr;
    bool seam = false, blocks = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--blocks")) blocks = true;
        else if (!std::strcmp(argv[i], "--poison-a")) g_poison_a = true;
        else if (!std::strcmp(argv[i], "--poison-b")) g_poison_b = true;
        else if (!std::strcmp(argv[i], "--poison-cut")) g_poison_cut = true;
        else if (!std::strcmp(argv[i], "--seam")) seam = true;
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    mpfr_set_default_prec(kPrec);

    if (blocks) {
        std::printf("REAL BUILDING BLOCKS on the direct branch's own arguments, scored against\n"
                    "MPFR.  These are what FORM B and FL_ARITH differ by.\n");
        run_blocks<xp::DoubleDouble>(load_grid());
        run_blocks<xp::FloatFloat>(load_grid());
        run_blocks<xp::QuadFloat>(load_grid());
        run_blocks<xp::TripleFloat>(load_grid());
        return 0;
    }

    if (seam) {
        std::printf("SEAM: the crossing argument sits exactly on %.1f, where the header takes\n"
                    "the asymptotic side.  All three forms scored against MPC at that point.\n",
                    kAsymptote);
        run_seam<xp::DoubleDouble>();
        run_seam<xp::FloatFloat>();
        run_seam<xp::QuadFloat>();
        run_seam<xp::TripleFloat>();
        return 0;
    }

    if (g_poison_a)
        std::printf("POISON A: form A conjugates its second argument.  The limb-for-limb\n"
                    "SHIPPED == FORM A check MUST fire.\n");
    if (g_poison_b)
        std::printf("POISON B: form B's denominator uses sin^2 where cos^2 belongs, in the\n"
                    "form AND in both floors.  Form B MUST get much worse.\n");
    if (g_poison_cut)
        std::printf("POISON CUT: every expected zero-sign is flipped.  A correct build MUST\n"
                    "fail the special-value sheet.\n");

    const std::vector<Pt> grid = load_grid();
    std::printf("grid: %zu complex points\n", grid.size());

    for (int fn = 0; fn < 2; ++fn) {
        const bool swap = (fn == 1);
        run<xp::DoubleDouble>(grid, swap);
        run<xp::FloatFloat>(grid, swap);
        run<xp::QuadFloat>(grid, swap);
        run<xp::TripleFloat>(grid, swap);
    }
    std::printf("shipped != form A at %ld point-backend pairs\n", g_mismatch);

    report(false);
    report(true);

    run_cuts<xp::DoubleDouble>();
    run_cuts<xp::FloatFloat>();
    run_cuts<xp::QuadFloat>();
    run_cuts<xp::TripleFloat>();
    std::printf("\nspecial values: %d of %d component checks failed\n", g_cut_fail, g_cut_total);

    if (csv) write_csv(csv);

    if (g_poison_a) {
        std::printf("\npoisoned run: %ld mismatches (want > 0)\n", g_mismatch);
        return g_mismatch > 0 ? 0 : 1;
    }
    if (g_poison_cut) {
        std::printf("\npoisoned run: %d cut failures (want > 0)\n", g_cut_fail);
        return g_cut_fail > 0 ? 0 : 1;
    }
    if (g_poison_b) {
        std::printf("\npoisoned run: inspect the table above; form B must be far worse.\n");
        return 0;
    }
    return (g_mismatch == 0 && g_cut_fail == 0) ? 0 : 1;
}
