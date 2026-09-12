// probe_div_lift.cpp — is real `divide`'s subnormal-tail error the FORMAT, or
// an exact power-of-two rescale the four real divides never got?
//
// WHY THIS EXISTS.  In the committed baseline real div carries 405 rows above
// 1 ulp out of 6752 scored, and the distribution is bimodal: a last-bit bulk,
// and a separate structural cluster living entirely below |a| ~ 1e-10 that
// reaches 7.5e21 ulps.  Only 15 of the 405 are above their derived bound, and
// zero complex-div rows are — so on the verdict machinery div reads clean.
//
// It reads clean because the bound is self-fulfilling.  `in_floor`
// (sweep_accuracy.cpp:2400) charges each operand's own subnormal-limb floor,
// justified in-code by "the residuals an algorithm forms from it live at the
// operand's scale, so the floor enters through kappa".  At QF r div point 8
// that hands the row a bound of 2.22e12, which absorbs a measured 4.76e7
// without a murmur.  If the residuals can be moved OFF the operand's scale by
// an exact rescale, the premise is false and the bound is excusing a defect.
//
// THE CLAIMED MECHANISM.  QF `divide` (qf_math.hpp:707) forms
//
//     r = subtract(a, multiply_scalar(b, q0))
//
// and an L-word expansion spaces its words 2^-24 apart, so the lowest word of
// b*q0 sits at |b|*|q0|*2^-72 ~ |a|*2^-72.  Below FLT_MIN that word is
// subnormal and carries a couple of bits instead of 24, so the residual the
// next quotient digit is computed from is quietly truncated.  a/b is invariant
// under scaling BOTH operands by the same power of two, and a power-of-two
// scale of an expansion is exact — every word's exponent shifts and no bit
// moves.  This is bit-for-bit the KI-41 mechanism and the KI-41 remedy
// (qf_complex.hpp:325-352), which the four real divides never got: every guard
// in every one of them is a splitter-OVERFLOW guard at the other end of the
// range (dd_math.hpp:344, ff_math.hpp:290).
//
// WHAT THIS PROBE HAD TO SETTLE FIRST, AND DID NOT EXPECT TO.  The plan this
// came from rests the whole div section on four points — r div 8, 15, 36, 43 —
// described as "quotient exactly 1.0, exactly representable, no conditioning
// excuse at all".  --points reproduces the measured ulps EXACTLY (QF 4.76273e7
// / 2.05157e10 / 9617.04 / 760.475, against the committed CSV to every printed
// digit) and REFUTES the descriptor: --dump-operands gives the quotients as
// 0.99999999999994316, 1.0000152590218967, 0.99999999999818109 and
// 1.0322580645161289.  None is 1.0; the last is not within 3% of it.
//
// That descriptor was doing real work — it was the argument that no
// conditioning excuse applies — so `floor` replaces the argument with a
// measurement.  It rounds the true quotient into the format greedily, which no
// implementation can beat, and that is the honest version of "representable".
//
// ARMS:
//
//   lib     xp::divide(a, b) as shipped.  On the --points source this must
//           reproduce the committed CSV, and nothing else here means anything
//           if it does not.
//   lift    the guarded exact-pow2 lift: scale a and b by exact powers of two
//           until the low word of the residual is normal again, run the
//           UNCHANGED divide, unscale.  Modelled on KI-41 rather than written
//           fresh.
//   lift0   THE POISON ARM, and the one that decides whether the mechanism is
//           what this file says.  The predicate is evaluated and the branch is
//           taken exactly as in `lift`, but both scale factors are forced to
//           2^0.  Every row `lift` improved must come back to its `lib` value.
//           If numbers still improve with the scaling disabled, the improvement
//           is coming from the branch (or from re-association inside it) and
//           not from the rescale.
//   floor   the exact quotient at 400 bits, merely ROUNDED into the format by
//           greedy limb extraction.  No implementation can beat this; <= 0.5 by
//           construction.  This is the format's own answer to "could this
//           quotient have been represented", and it is what makes the
//           inherent/addressable split a measurement instead of a claim.
//
// OPERAND SOURCES:
//
//   points  the four load-bearing grid points, as the RAW GRID DOUBLES read off
//           `--dump-operands` (which prints (__float128)a_in[i], the grid
//           double, not the stored value — so one set of literals serves all
//           four backends).  Reference is the exact quotient of the STORED
//           operands, matching the sweep's f(x_stored) convention (b7c4b64,
//           sweep_accuracy.cpp:2378).  This arm's `lib` column is directly
//           comparable to the committed CSV.
//   bands   synthetic pairs, both operands drawn log-uniform in a common
//           exponent band, rounded into the format.  Three bands, which are the
//           three the regression question needs:
//             fully-precise  2^+-40   — no limb near subnormal on any backend.
//                                       The guard must fire ZERO times here and
//                                       every row must be BIT-IDENTICAL.
//             cliff          2^+-100  — straddles the FP32 backends' cliffs.
//             wide           2^+-300  — straddles DD's.
//           These are NOT the sweep's operands and are not claimed to be: the
//           sweep's real-div operand stream is not reconstructible
//           (sweep_accuracy.cpp:2470 — three variants of fill_real_operands
//           scored 0 of 8 against known rows).  They exist to answer the
//           regression question, which is about the guard's behaviour on
//           generic operands, not about any particular row.
//
// Build (needs MPFR; not part of the CMake build):
//   module use /soft/modulefiles && module load gcc/13.3.0
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_div_lift.cpp -o /tmp/probe_div_lift -lmpfr -lgmp
//   /tmp/probe_div_lift
//   /tmp/probe_div_lift --poison    # negative control
//
// POISON.  --poison scores every arm against a reference computed at 24 bits
// instead of 400.  Every arm on every backend must go to 100% above 1 ulp,
// `floor` included — `floor` is the load-bearing one, because it reads <= 0.5
// BY CONSTRUCTION and is therefore the cell most likely to stay quiet if the
// oracle is not really being consulted.
//
// The first version of this control DID stay quiet, and the reason is worth
// keeping: `floor` was rounding the POISONED reference into the format and
// then being scored against that same poisoned reference.  24 bits fits in all
// four formats, so it came back 0.0 on every row — a perfect score from an
// oracle that was not being consulted at all, which is precisely what the
// control is for.  `floor` is now built from the 400-bit quotient and scored
// against the poisoned one; see ref_of().  Read the fully-precise band for the
// verdict: the cliff/wide bands read ~1114/1200 rather than 1200/1200 because
// their non-finite rows are excluded and a handful of random quotients are
// genuinely exact in 24 bits (probe_sqrt_iter records the same effect and
// enumerates them).
//
// ===========================================================================
// WHAT IT MEASURED.  gcc 13.3.0, MPFR 400 bits, 2026-09-12.
//
// INSTRUMENT VALIDATED FIRST.  `lib` on the --points source reproduces the
// committed baseline CSV to every printed digit on FF, TF and QF — QF
// 4.7627e7 / 2.0516e10 / 9617 / 760.47 against the CSV's 4.76273e+07 /
// 2.05157e+10 / 9617.04 / 760.475, TF 2.8388 / 1222.8 / 0.00057322 / 0.42987.
// DD differs in the 3rd decimal (0.024441 vs 0.023438 etc.) because this probe
// references at 400 bits and the sweep references at __float128's 113; one
// oracle quantum at p=113 is 2^-7 = 0.0078 ulps in p=106 units, and every DD
// delta is inside that.  probe_sqrt_iter records the same offset.
//
// THE PLAN'S DESCRIPTOR OF THESE FOUR POINTS IS WRONG AND THEIR NUMBERS ARE
// RIGHT.  They are described as "quotient exactly 1.0, exactly representable,
// no conditioning excuse at all".  --dump-operands gives the quotients as
// 0.99999999999994316, 1.0000152590218967, 0.99999999999818109 and
// 1.0322580645161289 — none is 1.0 and the last is 3% away.  The `floor` arm
// is what replaces that argument, and it reaches the same conclusion by
// measurement: the format CAN hold these quotients, to 2.67e-8 / 4.25e-4 /
// 2.20e-6 / 6.22e-3 ulps respectively.  So there is no representability
// excuse; there just was not one to be had from "exactly 1.0".
//
// THE FOUR POINTS, QF (committed == lib for all four):
//
//   pt   lib          lift        lift2        lift0(poison)   floor
//    8   4.7627e+07   1.62e-03    2.6658e-08   4.7627e+07      2.6658e-08
//   15   2.0516e+10   4.25e-04    4.2543e-04   2.0516e+10      4.2543e-04
//   36   9.6170e+03   1.35e-03    5.4274e-06   9.6170e+03      2.2020e-06
//   43   7.6047e+02   6.22e-03    6.2241e-03   7.6047e+02      6.2241e-03
//
// lift2 lands ON the format floor at 8, 15 and 43 and within 2.5x of it at 36.
// The shipped code is 1.8e15x / 4.8e13x / 4.4e9x / 1.2e5x above a floor it
// could have reached by shifting exponents.  TF shows the same shape at pts 8
// and 15 (2.8388 -> 1.86e-07, 1222.8 -> 9.14e-04); at 36 and 43 TF's guard does
// not fire and TF is already fine, which is the predicate behaving.
//
// THE POISON ARM PASSES, WHICH IS WHAT MAKES THE MECHANISM THE RESCALE.  With
// the predicate firing and the branch taken but both factors pinned to 2^0,
// all four points return EXACTLY their shipped value, and `POISON moved` is 0
// on every backend in every band.  The improvement is the exponent shift and
// nothing else — not the branch, not re-association inside it.
//
// THE FULLY-PRECISE BAND IS AN EXACT NO-OP ON ALL FOUR BACKENDS: fires=0,
// 1200/1200 bit-identical, 0 better, 0 worse.  That is the property that makes
// this admissible on a primitive every transcendental in the library calls.
//
// BANDS (n=1200 each; `nonfin` rows carry no ordering and are excluded from
// better/worse; exponent ranges are per-backend, see Fmt<>):
//
//   backend band            fires  lib>1u  lift>1u  floor>1u  better  worse  maxgrowth
//   DD      fully-precise       0       0        0         0       0      0     —
//   DD      cliff  2^+-990     27     154      151       151       3      0     —
//   DD      wide   2^+-1020    50     167      158       158       9      0     —
//   FF      fully-precise       0     139      139         0       0      0     —
//   FF      cliff  2^+-100      0     247      247       134       0      0     —
//   FF      wide   2^+-126    204     339      295       196      54      0     —
//   TF      cliff  2^+-100    244     298      247       200      61      2   2.561
//   TF      wide   2^+-126    407     398      314       276     112      1   2.625
//   QF      cliff  2^+-100    483     432      331       305     141      3   1.247
//   QF      wide   2^+-126    577     492      373       353     177      1   2.625
//
// READ THE `floor>1u` COLUMN, NOT THE `lift>1u` ONE, TO SCORE THE FIX.  On QF
// cliff the lift takes 432 bad rows to 331 against a floor of 305: of the 127
// rows that were ADDRESSABLE it recovers 101, and the 305 that remain are the
// format declining to hold the quotient at all.  A residual count near the
// floor is the fix succeeding, not failing.
//
// DD's guard fires 27 and 50 times, and only in bands placed at DD's own cliff
// (2^-969); the plan predicted DD's committed real-div column would not move
// and that is consistent — the sweep's real-div grid bottoms out at 1e-30, 262
// decades above where DD's guard can fire.
//
// ONE THING THE LIFT DOES NOT FIX, REPORTED BECAUSE IT IS IN THIS PROBE'S
// OUTPUT AND IS NOT THIS PROBE'S SUBJECT.  In the FULLY-PRECISE band, where no
// limb is anywhere near subnormal, FF still reads 139 of 1200 rows above 1 ulp
// with max 5.406 against a format floor of 0.4818, TF 63 with max 9.066 against
// 0.2323, and QF 38 addressable rows (92 bad, 54 at floor).  DD reads 0.  That
// is the "last-bit bulk" the plan's §1.2e already describes (FF 3.939 / TF
// 5.404 / QF 6.744 maxima on the sweep) and it is a DIFFERENT mechanism from
// the one measured here — the lift is a provable no-op on exactly these rows.
// Not chased.
// ===========================================================================

#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>
#include <xp/qf_math.hpp>
#include <xp/tf_math.hpp>

#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace xp;

static const mpfr_prec_t kPrec = 400;
static mpfr_prec_t       g_ref_prec = kPrec;   // --poison lowers this

// ---------------------------------------------------------------------------
// format traits — verbatim from probe_arith_floor.cpp / probe_sqrt_iter.cpp
// ---------------------------------------------------------------------------
template <class T> struct Fmt;

// `cliff_exp` / `wide_exp` are per-backend on purpose.  A single 2^+-300 band
// is meaningless on two backends at once: for the FP32 formats it is entirely
// past FLT_MAX and measures overflow, and for DD it never reaches DD's own
// cliff at DBL_MIN*2^53 = 2.0042e-292 (~2^-969), so DD's guard would read
// "never fires" for want of an operand rather than by property.  Each band is
// placed against the backend's own limits.
template <> struct Fmt<DoubleDouble> {
    using Limb = double;
    static const int  n = 2, p = 106, limb_bits = 53;
    static const int  cliff_exp = 990, wide_exp = 1020;
    static const char* name() { return "DD"; }
    static double min_norm() { return 2.2250738585072014e-308; }   // DBL_MIN
    static double limb(const DoubleDouble& a, int i) { return i ? a.lo : a.hi; }
    static DoubleDouble make(const double* w) { return DoubleDouble(w[0], w[1]); }
};
template <> struct Fmt<FloatFloat> {
    using Limb = float;
    static const int  n = 2, p = 48, limb_bits = 24;
    static const int  cliff_exp = 100, wide_exp = 126;
    static const char* name() { return "FF"; }
    static float min_norm() { return 1.17549435e-38f; }            // FLT_MIN
    static float limb(const FloatFloat& a, int i) { return i ? a.lo : a.hi; }
    static FloatFloat make(const float* w) { return FloatFloat(w[0], w[1]); }
};
template <> struct Fmt<TripleFloat> {
    using Limb = float;
    static const int  n = 3, p = 72, limb_bits = 24;
    static const int  cliff_exp = 100, wide_exp = 126;
    static const char* name() { return "TF"; }
    static float min_norm() { return 1.17549435e-38f; }
    static float limb(const TripleFloat& a, int i) { return i == 0 ? a.f0 : i == 1 ? a.f1 : a.f2; }
    static TripleFloat make(const float* w) { return TripleFloat(w[0], w[1], w[2]); }
};
template <> struct Fmt<QuadFloat> {
    using Limb = float;
    static const int  n = 4, p = 96, limb_bits = 24;
    static const int  cliff_exp = 100, wide_exp = 126;
    static const char* name() { return "QF"; }
    static float min_norm() { return 1.17549435e-38f; }
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
    double max() { double m = 0; for (double x : v) m = std::max(m, x); return m; }
    size_t over(double t) { size_t c = 0; for (double x : v) if (x > t) ++c; return c; }
};

// ---------------------------------------------------------------------------
// the arms
// ---------------------------------------------------------------------------

// Shipped.
template <class T> static T arm_lib(const T& a, const T& b) { return divide(a, b); }

// The guard predicate, in one place so `lift` and the poison arm cannot drift.
// min(|a0|,|b0|) * 2^-(limb_bits*(L-1)) < min_norm * 4 — i.e. "the lowest word
// of a product at the operands' scale is at or under the subnormal boundary".
// The 4x margin keeps the lifted word clear of the boundary itself, as KI-41's
// does.
template <class T> static bool lift_wanted(const T& a, const T& b) {
    using L = typename Fmt<T>::Limb;
    const L a0 = detail::fabs(Fmt<T>::limb(a, 0));
    const L b0 = detail::fabs(Fmt<T>::limb(b, 0));
    if (a0 == L(0) || b0 == L(0)) return false;
    const L m = (a0 < b0) ? a0 : b0;
    const L spread = (L)std::ldexp(1.0, -Fmt<T>::limb_bits * (Fmt<T>::n - 1));
    return m * spread < Fmt<T>::min_norm() * L(4);
}

template <class T> static T pow2_scale(const T& a, typename Fmt<T>::Limb s);
template <> DoubleDouble pow2_scale<DoubleDouble>(const DoubleDouble& a, double s) {
    return detail::dd_pow2_scale(a, s);
}
template <> FloatFloat pow2_scale<FloatFloat>(const FloatFloat& a, float s) {
    return detail::ff_pow2_scale(a, s);
}
template <> TripleFloat pow2_scale<TripleFloat>(const TripleFloat& a, float s) {
    return detail::tf_pow2_scale(a, s);
}
template <> QuadFloat pow2_scale<QuadFloat>(const QuadFloat& a, float s) {
    return detail::qf_pow2_scale(a, s);
}

// arm lift / lift2 / lift0.
//
//   scale=false is THE POISON: the predicate is evaluated, the branch is
//   entered, and both factors are pinned to 2^0.
//   `extra_words` is how deep below the leading word the lift aims to keep
//   normal.  n-1 is the spacing of the operand's own expansion, which is what
//   the guard predicate tests (arm `lift`).  n goes one word deeper, because
//   the residual is formed from two_prod of a limb with a quotient digit and
//   THAT product has a low half another 2^-limb_bits down (arm `lift2`).
template <class T> static T arm_lift(const T& a, const T& b, bool scale, int extra_words) {
    using L = typename Fmt<T>::Limb;
    if (!lift_wanted(a, b)) return divide(a, b);
    L sa = L(1), sb = L(1);
    if (scale) {
        // Lift each operand until its own leading word clears the boundary the
        // residual words are measured from.  a/b is invariant under
        // (a*sa)/(b*sb) up to the factor sb/sa, which is undone exactly at the
        // end; every factor is a power of two, so no step rounds.
        //
        // The step cap is not decoration.  sa is a single limb, so an unbounded
        // loop overflows it to inf on a subnormal operand and the "exact"
        // rescale silently stops being exact.  2^(6*limb_bits) is 2^144 on FP32
        // -- past FLT_MAX -- so the cap has to bite before the loop does.
        const L step   = (L)std::ldexp(1.0, Fmt<T>::limb_bits);
        const L spread = (L)std::ldexp(1.0, -Fmt<T>::limb_bits * extra_words);
        const L target = Fmt<T>::min_norm() * L(4);
        const L cap    = (L)std::ldexp(1.0, Fmt<T>::limb_bits * (Fmt<T>::n + 1));
        L pa = detail::fabs(Fmt<T>::limb(a, 0)), pb = detail::fabs(Fmt<T>::limb(b, 0));
        for (int k = 0; k < 16 && pa * spread < target && sa < cap; ++k) { sa *= step; pa *= step; }
        for (int k = 0; k < 16 && pb * spread < target && sb < cap; ++k) { sb *= step; pb *= step; }
    }
    const T q = divide(pow2_scale<T>(a, sa), pow2_scale<T>(b, sb));
    return pow2_scale<T>(q, sb / sa);
}

// arm floor — the exact quotient rounded into the format.  Unbeatable.
template <class T> static T arm_floor(mpfr_srcptr ref) { return from_mpfr<T>(ref); }

// ---------------------------------------------------------------------------
// operands
// ---------------------------------------------------------------------------

// The four load-bearing grid points, as the RAW GRID DOUBLES printed by
//   sweep_accuracy --dump-operands div:N
// (which emits (__float128)a_in[i], i.e. the grid double, so one set of
// literals is correct for all four backends).  `csv_qf` is the committed
// baseline's QF ulps for that point — the instrument check.
struct Pt { int id; const char* a; const char* b; double csv_qf, csv_tf, csv_ff, csv_dd; };
static const Pt kPts[] = {
    {  8, "9.99999999999999971232543461600619677e-29",
          "1.00000000000005680789825647618101634e-28",
       4.76273e+07, 2.83881,   4.48957e-08, 0.0234375},
    { 15, "-3.16227766016837973709909897107285962e-27",
          "-3.1622294076406061404953553020998644e-27",
       2.05157e+10, 1222.83,   0.0241901,   0.062499},
    { 36, "9.99999999999999907537452227896371397e-22",
          "1.0000000000018188204761406026947019e-21",
       9617.04,     0.00057322, 1.00156e-05, 0.0390625},
    { 43, "-3.16227766016837921774640932359306045e-20",
          "-3.06345648328811761169465900326657897e-20",
       760.475,     0.429869,  0.27765,     0.0756836},
};
static const int kNPts = 4;

static double csv_for(const Pt& p, const char* be) {
    if (!std::strcmp(be, "QF")) return p.csv_qf;
    if (!std::strcmp(be, "TF")) return p.csv_tf;
    if (!std::strcmp(be, "FF")) return p.csv_ff;
    return p.csv_dd;
}

// ---------------------------------------------------------------------------
// runners
// ---------------------------------------------------------------------------

// The exact quotient of the STORED operands (the sweep's f(x_stored)
// convention, b7c4b64).  Two outputs, and they must stay two:
//
//   `truth` is always at kPrec.  It is what the `floor` arm rounds into the
//   format, i.e. the best answer that exists.
//   `out`   is the SCORING reference, at g_ref_prec, which --poison drops to 24.
//
// Rounding the poisoned reference into the format and then scoring it against
// that same poisoned reference would give 0 by construction on every backend,
// because 24 bits fits in all four formats — the `floor` arm would read 0.0
// under --poison and look quiet for a reason that has nothing to do with
// whether the oracle is consulted.  That is the exact failure mode --poison
// exists to catch, so `floor` is built from `truth` and scored against `out`.
template <class T>
static void ref_of(mpfr_t out, mpfr_t truth, const T& sa, const T& sb) {
    mpfr_t x, y;
    mpfr_init2(x, kPrec); mpfr_init2(y, kPrec);
    to_mpfr(x, sa); to_mpfr(y, sb);
    mpfr_div(truth, x, y, MPFR_RNDN);            // kPrec, always
    mpfr_t q; mpfr_init2(q, g_ref_prec);
    mpfr_div(q, x, y, MPFR_RNDN);
    mpfr_set(out, q, MPFR_RNDN);                 // g_ref_prec, poisonable
    mpfr_clear(x); mpfr_clear(y); mpfr_clear(q);
}

template <class T> static void run_points() {
    std::printf("\n  %s  --  the four load-bearing points (raw grid doubles)\n", Fmt<T>::name());
    std::printf("    pt   committed        lib         lift        lift2      lift0(poison)    floor      guard\n");
    for (int i = 0; i < kNPts; ++i) {
        const double ad = std::strtod(kPts[i].a, nullptr);
        const double bd = std::strtod(kPts[i].b, nullptr);
        const T sa(ad), sb(bd);
        mpfr_t ref, truth; mpfr_init2(ref, kPrec); mpfr_init2(truth, kPrec);
        ref_of<T>(ref, truth, sa, sb);
        const double u_lib   = ulps<T>(arm_lib<T>(sa, sb), ref);
        const double u_lift  = ulps<T>(arm_lift<T>(sa, sb, true,  Fmt<T>::n - 1), ref);
        const double u_lift2 = ulps<T>(arm_lift<T>(sa, sb, true,  Fmt<T>::n), ref);
        const double u_lift0 = ulps<T>(arm_lift<T>(sa, sb, false, Fmt<T>::n - 1), ref);
        const double u_floor = ulps<T>(arm_floor<T>(truth), ref);
        std::printf("    %3d  %11.5g %11.5g %11.5g %11.5g   %11.5g  %11.5g   %s\n",
                    kPts[i].id, csv_for(kPts[i], Fmt<T>::name()),
                    u_lib, u_lift, u_lift2, u_lift0, u_floor,
                    lift_wanted<T>(sa, sb) ? "fires" : "-");
        mpfr_clear(ref); mpfr_clear(truth);
    }
}

// One synthetic band.  Both operands drawn log-uniform over 2^+-`ex`, so the
// quotient stays O(1)-ish while the operands roam; that is the shape the
// structural cluster has.
template <class T>
static void run_band(const char* label, int ex, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> mant(1.0, 2.0);
    std::uniform_int_distribution<int>     expo(-ex, ex);

    Acc a_lib, a_lift, a_lift2, a_floor;
    size_t fires = 0, identical = 0, better = 0, worse = 0, p_moved = 0, nonfinite = 0;
    double worst_growth = 0.0;

    for (int i = 0; i < n; ++i) {
        mpfr_t va, vb; mpfr_init2(va, kPrec); mpfr_init2(vb, kPrec);
        mpfr_set_d(va, mant(rng), MPFR_RNDN); mpfr_mul_2si(va, va, expo(rng), MPFR_RNDN);
        mpfr_set_d(vb, mant(rng), MPFR_RNDN); mpfr_mul_2si(vb, vb, expo(rng), MPFR_RNDN);
        const T sa = from_mpfr<T>(va), sb = from_mpfr<T>(vb);
        mpfr_clear(va); mpfr_clear(vb);
        if (Fmt<T>::limb(sb, 0) == typename Fmt<T>::Limb(0)) continue;

        mpfr_t ref, truth; mpfr_init2(ref, kPrec); mpfr_init2(truth, kPrec);
        ref_of<T>(ref, truth, sa, sb);

        const T q_lib   = arm_lib<T>(sa, sb);
        const T q_lift  = arm_lift<T>(sa, sb, true,  Fmt<T>::n - 1);
        const T q_lift2 = arm_lift<T>(sa, sb, true,  Fmt<T>::n);
        const T q_lift0 = arm_lift<T>(sa, sb, false, Fmt<T>::n - 1);
        const double u_lib  = ulps<T>(q_lib, ref);
        const double u_lift = ulps<T>(q_lift, ref);
        a_lib.add(u_lib); a_lift.add(u_lift); a_lift2.add(ulps<T>(q_lift2, ref));
        a_floor.add(ulps<T>(arm_floor<T>(truth), ref));

        if (lift_wanted<T>(sa, sb)) ++fires;

        // Bit-identity, limb by limb, NOT an ulp comparison.  NaN == NaN has to
        // count as identical here: at the band edges the true quotient is
        // inf/inf and both arms return NaN, and `!=` on two NaNs is true, which
        // would report the poison arm as having MOVED every overflow row.  That
        // is the difference between a poison arm that passes and one that looks
        // like it failed on 105 rows.
        auto same_limbs = [](const T& x, const T& y) {
            for (int k = 0; k < Fmt<T>::n; ++k) {
                const double xk = (double)Fmt<T>::limb(x, k), yk = (double)Fmt<T>::limb(y, k);
                if (std::isnan(xk) && std::isnan(yk)) continue;
                if (xk != yk) return false;
            }
            return true;
        };
        const bool same  = same_limbs(q_lib, q_lift);
        if (!same_limbs(q_lib, q_lift0)) ++p_moved;

        // A row whose quotient is not a finite number carries no ordering, so it
        // is counted and excluded rather than filed as a regression.
        if (u_lib < 0 || u_lift < 0) {
            ++nonfinite;
        } else {
            if (same) ++identical; else if (u_lift < u_lib) ++better; else ++worse;
            if (u_lib > 0 && u_lift > u_lib && u_lift > 1.0)
                worst_growth = std::max(worst_growth, u_lift / u_lib);
        }
        mpfr_clear(ref); mpfr_clear(truth);
    }

    std::printf("    %-14s %-4s n=%-5d fires=%-5zu nonfin=%-5zu | lib med %-9.4g max %-10.4g >1u %-5zu"
                " | lift med %-9.4g max %-10.4g >1u %-5zu | lift2 >1u %-5zu | floor >1u %-5zu max %-10.4g"
                " | ident %-5zu better %-4zu worse %-4zu maxgrowth %-7.4g"
                " | POISON moved %zu\n",
                label, Fmt<T>::name(), n, fires, nonfinite,
                a_lib.med(), a_lib.max(), a_lib.over(1.0),
                a_lift.med(), a_lift.max(), a_lift.over(1.0),
                a_lift2.over(1.0),
                a_floor.over(1.0), a_floor.max(),
                identical, better, worse, worst_growth, p_moved);
}

template <class T> static void run_all() {
    run_points<T>();
    std::printf("\n  %s  --  synthetic bands (exponent ranges are per-backend; see Fmt<>)\n",
                Fmt<T>::name());
    run_band<T>("fully-precise", 40,                    1200, 0x5EEDu);
    run_band<T>("cliff",         Fmt<T>::cliff_exp,     1200, 0x5EEDu);
    run_band<T>("wide",          Fmt<T>::wide_exp,      1200, 0x5EEDu);
}

int main(int argc, char** argv) {
    bool poison = false;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--poison")) poison = true;
    if (poison) {
        g_ref_prec = 24;
        std::printf("*** --poison: reference at 24 bits.  EVERY arm on EVERY backend must\n"
                    "*** go to 100%% above 1 ulp, `floor` included.\n");
    }
    std::printf("probe_div_lift — real divide, exact power-of-two lift\n");
    std::printf("reference: exact quotient of the STORED operands at %ld bits\n",
                (long)g_ref_prec);

    run_all<DoubleDouble>();
    run_all<FloatFloat>();
    run_all<TripleFloat>();
    run_all<QuadFloat>();
    return 0;
}
