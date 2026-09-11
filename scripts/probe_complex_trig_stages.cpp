// probe_complex_trig_stages.cpp — attribute complex trig error to a STAGE.
//
// WHY THIS EXISTS.  scripts/probe_complex_oracle.cpp established that the
// complex trig residual is the library's and not the oracle's (worst oracle
// discrepancy 0.030 DD ulps against a 3-6 ulp residual).  That says the number
// is real; it does not say which line produces it.  The direct complex trig
// ops are two-stage,
//
//     sin(x+iy) = sin(x) cosh(y) + i cos(x) sinh(y)
//
// so the finished error is either INHERITED from the four real primitives or
// ADDED by the complex layer's own multiplies.  Those two call for completely
// different fixes and the sweep cannot tell them apart.  This probe can.
//
// WHAT IT MEASURES, per backend and per grid point:
//
//   total     the shipped complex op vs MPC at 400 bits, modulus-relative, in
//             ulps of the backend -- the same number the sweep's `ulps` column
//             carries, recomputed here so the two can be cross-checked.
//   composed  the library's OWN real primitives, widened exactly to MPFR and
//             recombined there in 400-bit arithmetic, vs the same MPC answer.
//             This is what the complex op would score if its own arithmetic
//             were free.  Anything left is inherited from the primitives.
//   added     total - composed, i.e. what the complex layer's arithmetic costs
//             on top of what it was handed.
//   the four primitive errors sin(x), cos(x), sinh(y), cosh(y) separately, each
//             relative to ITS OWN magnitude, in backend ulps.
//
// READING IT.  composed ~ total means the complex layer is innocent and the
// work belongs in the real primitives.  composed ~ 0 with total large means the
// recombination is where the digits go.  Neither is assumed.
//
// WIDENING, and the signed zero.  A backend expansion is widened to MPFR by
// summing its limbs in 400-bit arithmetic, which is exact -- no limb pair is
// more than ~290 exponents apart in a normalized expansion, so nothing rounds.
// The one thing a plain sum destroys is the sign of a zero: (-0.0) + (+0.0) is
// +0.0, which is what made the previous attempt at a wide complex oracle answer
// for the conjugate at 1,996 cut points.  to_mpfr() therefore reads the sign off
// the LEADING limb whenever the sum is zero.  This probe compares magnitudes of
// differences, so the correction changes nothing it prints today, but getting
// it wrong silently is exactly the failure this repo has already paid for once.
//
// MODES
//   (default)     every backend x every direct-family op x the complex grid;
//                 per-cell worst total / composed / added
//   --op NAME     restrict to one op
//   --backend B   restrict to DD | FF | QF | TF
//   --point OP N  dump one point in full: every stage, both components
//   --top N       the N worst points for the selected cell, with the split
//   --poison      NEGATIVE CONTROL.  Corrupts the library's cos by one backend
//                 ulp before the recombination, measures how far that moves the
//                 recombined POINT, and requires the movement to equal what the
//                 perturbation that actually landed predicts.  Exits nonzero on
//                 any miss.  MEASURED: 20,688 non-dilute cases, 0 misses, worst
//                 relative miss 0, 0 cases where the kick was absorbed.
//
//                 Two weaker versions of this control were written first and
//                 both were wrong, in ways worth recording because each looked
//                 fine until it was run.  (1) Thresholding the poisoned reading
//                 against a fixed 0.5 ulp charged 7,465 failures that were
//                 really the metric working: where the corrupted factor is
//                 DILUTE, a modulus-relative reading SHOULD not move.  (2)
//                 Comparing the two error MAGNITUDES charged 2,166 more,
//                 because the injected displacement lies along one axis and the
//                 clean error can lie along the other, so |e+d| - |e| is second
//                 order.  Only comparing displacements, against a prediction
//                 derived from the perturbation that actually landed rather
//                 than from the nominal ulp, is free of both.
//
// MEASURED, 2026-09-11.  Everything below is a real run, not a prediction.
//
// The probe agrees with the measurement of record.  Against sweep_accuracy's
// own `ulps` column for the same (backend, op) cells, FF, TF and QF reproduce
// BOTH the count above one ulp AND the worst point exactly; DD differs only in
// the fourth digit and by a handful of counts, which is the known <=0.030 DD
// ulp gap between MPC and the libquadmath oracle the sweep uses (established in
// probe_complex_oracle.cpp) landing either side of the 1.0 cut.
//
//   cell        sweep >1ulp / worst (pt)     this probe, total
//   DD c sin      391 / 3.249 (452)            386 / 3.247 (452)
//   DD c cos      289 / 3.059 (175)            283 / 3.066 (175)
//   DD c sinh     273 / 3.249 (1092)           275 / 3.247 (1092)
//   DD c cosh     270 / 2.796 (139)            266 / 2.788 (139)
//   FF c sin      227 / 2.981 (388)            227 / 2.981 (388)
//   FF c cos      400 / 3.677 (193)            400 / 3.677 (193)
//   FF c sinh     112 / 2.096 (201)            112 / 2.096 (201)
//   FF c cosh     342 / 3.680 (201)            342 / 3.680 (201)
//   TF c sin        4 / 3.821 (187)              4 / 3.821 (187)
//   TF c cos        4 / 3.234 (187)              4 / 3.234 (187)
//   TF c sinh       4 / 3.740 (179)              4 / 3.740 (179)
//   TF c cosh       5 / 3.224 (179)              5 / 3.224 (179)
//   QF c sin        3 / 1.579 (283)              3 / 1.579 (283)
//   QF c cos        4 / 2.035 (1096)             4 / 2.035 (1096)
//   QF c sinh       2 / 1.676 (275)              2 / 1.676 (275)
//   QF c cosh       3 / 1.702 (308)              3 / 1.702 (308)
//
// THE RESULT.  `composed` tracks `total` on every cell and every backend, and
// `added` never exceeds about one ulp.  The complex layer's own arithmetic --
// the two multiplies that assemble the answer -- is not where the digits go.
// The error is INHERITED from the four real primitives, and the per-primitive
// columns name which one at each point:
//
//   pt  452  z=(-10, 0.1)      DD total 3.247 = composed 2.239 + added 1.008,
//                              and composed is e(sin x) = 1.781.
//   pt  388  z=(-100, 0.1)     FF total 2.981 = composed 2.406 + added 0.575,
//                              and composed is e(sin x) = 2.467.
//   pt  187  z=(0.5556,-0.8315) TF total 3.821 = composed 3.810 + added 0.012,
//                              and composed is e(sinh y) = 4.344.
//
// This is corroborated across backends by the REAL sweep cells, which show the
// same magnitudes at the same primitives, and it explains the otherwise strange
// ordering in which TF and QF complex sin/cos are 100x cleaner than DD's:
//
//   DD r sin  192 rows >1ulp, worst 3.326  ->  DD c sin  391 rows, worst 3.249
//   FF r sin   30 rows >1ulp, worst 3.148  ->  FF c sin  227 rows, worst 2.981
//   TF r sin    0 rows >1ulp, worst 0.531  ->  TF c sin    4 rows, worst 3.821
//   QF r sin   18 rows >1ulp, worst 43.38  ->  QF c sin    3 rows, worst 1.579
//
// TF complex sin is nearly clean because TF REAL sin is nearly clean, and TF's
// four surviving points are exactly the ones the sinh/cosh column dominates --
// TF r sinh carries 54 rows above one ulp, worst 9.983.
//
// Build (needs MPC + MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_complex_trig_stages.cpp -o /tmp/probe_complex_trig_stages \
//       -lmpc -lmpfr -lgmp -lquadmath
//   /tmp/probe_complex_trig_stages --poison
//   /tmp/probe_complex_trig_stages
//   /tmp/probe_complex_trig_stages --backend DD --op sin --top 12

#define MPFR_WANT_FLOAT128
#include <mpc.h>
#include <mpfr.h>
#include <quadmath.h>

#include <xp/dd_complex.hpp>
#include <xp/ff_complex.hpp>
#include <xp/qf_complex.hpp>
#include <xp/tf_complex.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* const kGrid = "validation/sweep/sweep_grid.csv";
const mpfr_prec_t kPrec = 400;

struct Pt { int idx; double re, im; std::string family; };

std::vector<Pt> load_grid() {
  std::vector<Pt> v;
  FILE* f = std::fopen(kGrid, "r");
  if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", kGrid); std::exit(2); }
  char line[512];
  while (std::fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == 'k' || line[0] != 'c') continue;
    char* fields[5]; int nf = 0; fields[nf++] = line;
    for (char* q = line; *q && nf < 5; ++q) if (*q == ',') { *q = 0; fields[nf++] = q + 1; }
    if (nf < 5) continue;
    v.push_back({std::atoi(fields[1]), std::strtod(fields[3], nullptr),
                 std::strtod(fields[4], nullptr), fields[2]});
  }
  std::fclose(f);
  return v;
}

// ---------------------------------------------------------------------------
// Backend traits: limb count, significand width, and an EXACT widen to MPFR.
// ---------------------------------------------------------------------------
template <class T> struct Tr;

template <> struct Tr<xp::DoubleDouble> {
  using S = double;
  static const char* name() { return "DD"; }
  static int  sig_bits()    { return 106; }
  static int  nlimb()       { return 2; }
  using Cx = xp::DoubleDoubleComplex;
  // DD/FF declare sincos(a, cos_out, sin_out); QF/TF declare
  // sincos(a, sin_out, cos_out).  Both conventions ship, and calling one
  // through the other silently transposes the answer -- which is exactly what
  // the first cut of this probe did, reading 1e29 ulps on QF and TF.  The two
  // wrappers below are the only place the difference is allowed to exist.
  static void sc(const xp::DoubleDouble& a, xp::DoubleDouble& c, xp::DoubleDouble& s) { xp::sincos(a, c, s); }
  static void hc(const xp::DoubleDouble& a, xp::DoubleDouble& c, xp::DoubleDouble& s) { xp::sinhcosh(a, c, s); }
  static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
};
template <> struct Tr<xp::FloatFloat> {
  using S = float;
  static const char* name() { return "FF"; }
  static int  sig_bits()    { return 48; }
  static int  nlimb()       { return 2; }
  using Cx = xp::FloatFloatComplex;
  static void sc(const xp::FloatFloat& a, xp::FloatFloat& c, xp::FloatFloat& s) { xp::sincos(a, c, s); }
  static void hc(const xp::FloatFloat& a, xp::FloatFloat& c, xp::FloatFloat& s) { xp::sinhcosh(a, c, s); }
  static double limb(const xp::FloatFloat& v, int i) { return i ? (double)v.lo : (double)v.hi; }
};
template <> struct Tr<xp::QuadFloat> {
  using S = float;
  static const char* name() { return "QF"; }
  static int  sig_bits()    { return 96; }
  static int  nlimb()       { return 4; }
  using Cx = xp::QuadFloatComplex;
  static void sc(const xp::QuadFloat& a, xp::QuadFloat& c, xp::QuadFloat& s) { xp::sincos(a, s, c); }
  static void hc(const xp::QuadFloat& a, xp::QuadFloat& c, xp::QuadFloat& s) { xp::sinhcosh(a, s, c); }
  static double limb(const xp::QuadFloat& v, int i) {
    return i == 0 ? (double)v.f0 : i == 1 ? (double)v.f1 : i == 2 ? (double)v.f2 : (double)v.f3;
  }
};
template <> struct Tr<xp::TripleFloat> {
  using S = float;
  static const char* name() { return "TF"; }
  static int  sig_bits()    { return 72; }
  static int  nlimb()       { return 3; }
  using Cx = xp::TripleFloatComplex;
  static void sc(const xp::TripleFloat& a, xp::TripleFloat& c, xp::TripleFloat& s) { xp::sincos(a, s, c); }
  static void hc(const xp::TripleFloat& a, xp::TripleFloat& c, xp::TripleFloat& s) { xp::sinhcosh(a, s, c); }
  static double limb(const xp::TripleFloat& v, int i) {
    return i == 0 ? (double)v.f0 : i == 1 ? (double)v.f1 : (double)v.f2;
  }
};

// Exact widen.  Summing the limbs at 400 bits cannot round: the limbs of a
// normalized expansion span at most ~4 x 53 exponents.  The sign of a zero is
// taken from the leading limb, because the sum loses it -- see the header
// comment.
template <class T>
void to_mpfr(const T& v, mpfr_t out) {
  mpfr_set_d(out, Tr<T>::limb(v, 0), MPFR_RNDN);
  for (int i = 1; i < Tr<T>::nlimb(); ++i)
    mpfr_add_d(out, out, Tr<T>::limb(v, i), MPFR_RNDN);
  if (mpfr_zero_p(out) && std::signbit(Tr<T>::limb(v, 0)))
    mpfr_setsign(out, out, 1, MPFR_RNDN);
}

// modulus-relative error of (ar,ai) against (br,bi), in ulps of `sig_bits`.
double ulps_mod(mpfr_t ar, mpfr_t ai, mpfr_t br, mpfr_t bi, int sig_bits) {
  mpfr_t d1, d2, e, m;
  mpfr_inits2(kPrec, d1, d2, e, m, (mpfr_ptr)0);
  mpfr_sub(d1, ar, br, MPFR_RNDN);
  mpfr_sub(d2, ai, bi, MPFR_RNDN);
  mpfr_hypot(e, d1, d2, MPFR_RNDN);
  mpfr_hypot(m, br, bi, MPFR_RNDN);
  double r;
  if (mpfr_zero_p(m) || !mpfr_number_p(m) || !mpfr_number_p(e)) r = -1.0;
  else { mpfr_div(e, e, m, MPFR_RNDN); mpfr_mul_2si(e, e, sig_bits, MPFR_RNDN); r = mpfr_get_d(e, MPFR_RNDN); }
  mpfr_clears(d1, d2, e, m, (mpfr_ptr)0);
  return r;
}

// relative error of a single real value, in ulps of `sig_bits`.
double ulps_real(mpfr_t got, mpfr_t ref, int sig_bits) {
  if (mpfr_zero_p(ref)) return mpfr_zero_p(got) ? 0.0 : -1.0;
  mpfr_t d; mpfr_init2(d, kPrec);
  mpfr_sub(d, got, ref, MPFR_RNDN);
  mpfr_div(d, d, ref, MPFR_RNDN);
  mpfr_abs(d, d, MPFR_RNDN);
  mpfr_mul_2si(d, d, sig_bits, MPFR_RNDN);
  const double r = mpfr_get_d(d, MPFR_RNDN);
  mpfr_clear(d);
  return r;
}

// ---------------------------------------------------------------------------
// The four direct-family ops, as (primitive pair) x (recombination).
// Every one of them is  f(x+iy) = P(x)*Q(y) + i*R(x)*S(y)  with the four
// factors drawn from {sin x, cos x} and {sinh y, cosh y}.
// ---------------------------------------------------------------------------
enum Op { kSin, kCos, kSinh, kCosh, kNDirect };
const char* const kOpName[kNDirect] = {"sin", "cos", "sinh", "cosh"};

int mpc_direct(int op, mpc_ptr w, mpc_srcptr z) {
  switch (op) {
    case kSin:  return mpc_sin(w, z, MPC_RNDNN);
    case kCos:  return mpc_cos(w, z, MPC_RNDNN);
    case kSinh: return mpc_sinh(w, z, MPC_RNDNN);
    default:    return mpc_cosh(w, z, MPC_RNDNN);
  }
}

struct Stage {
  double total = -1, composed = -1;
  double e_sinx = -1, e_cosx = -1, e_sinhy = -1, e_coshy = -1;
  // Share of the modulus carried by the term the --poison mode corrupts.  Only
  // the poison control reads it; see run_backend().  For sin the corrupted
  // cos(x) sits in the imaginary term, for the other three in the real one.
  double share_poisoned = 0;
  // --poison only.  `poison_delta` is the distance the recombined point moved,
  // `poison_pred` the distance the perturbation that actually landed says it
  // should have moved; both modulus-relative, in backend ulps.  They must agree.
  double poison_delta = -1, poison_pred = -1;
  bool   poison_landed = false;
};

// Evaluate one (backend, op, point) and split the error.
//
// `poison_cos` corrupts the library's cos(x) by one backend ulp AFTER it is
// produced and BEFORE it reaches either the recombination or the reported
// primitive error, which is what makes it a control on the composed arm.
template <class T>
Stage measure(int op, double xre, double xim, bool poison_cos) {
  Stage st;

  // Construct from the DOUBLE, not from the word type.  sweep_accuracy.cpp
  // builds its operand as `Z a{S(grid[i].re), S(grid[i].im)}` where S is the
  // EXPANSION type, so FloatFloat(double)/QuadFloat(double)/TripleFloat(double)
  // split the double across limbs.  Casting to float first instead -- which the
  // first cut of this probe did -- throws away 29 bits before the library is
  // even entered, and reported 1e21 ulps of "primitive error" on QF that was
  // nothing but the discarded input.
  const T x = T(xre);
  const T y = T(xim);

  // --- the shipped complex op ---------------------------------------------
  T gr, gi;
  {
    typename Tr<T>::Cx z(x, y);
    auto r = (op == kSin) ? xp::sin(z) : (op == kCos) ? xp::cos(z)
           : (op == kSinh) ? xp::sinh(z) : xp::cosh(z);
    gr = r.re; gi = r.im;
  }

  // --- the library's own real primitives ----------------------------------
  T cx, sx, cy, sy;
  Tr<T>::sc(x, cx, sx);
  Tr<T>::hc(y, cy, sy);

  mpfr_t mcx, msx, mcy, msy, mgr, mgi, mrr, mri, mcr, mci, t1, t2;
  mpfr_inits2(kPrec, mcx, msx, mcy, msy, mgr, mgi, mrr, mri, mcr, mci, t1, t2, (mpfr_ptr)0);
  to_mpfr(cx, mcx); to_mpfr(sx, msx); to_mpfr(cy, mcy); to_mpfr(sy, msy);
  to_mpfr(gr, mgr); to_mpfr(gi, mgi);

  // --- MPC reference -------------------------------------------------------
  mpc_t z, w;
  mpc_init2(z, kPrec); mpc_init2(w, kPrec);
  mpc_set_d_d(z, xre, xim, MPC_RNDNN);
  mpc_direct(op, w, z);
  mpfr_set(mrr, mpc_realref(w), MPFR_RNDN);
  mpfr_set(mri, mpc_imagref(w), MPFR_RNDN);

  // --- recombination in 400-bit arithmetic ---------------------------------
  //   sin : ( sin x cosh y,  cos x sinh y)
  //   cos : ( cos x cosh y, -sin x sinh y)
  //   sinh: ( sinh x cos y,  cosh x sin y)   <- roles of the two axes swap
  //   cosh: ( cosh x cos y,  sinh x sin y)
  // For sinh/cosh the library calls sinhcosh on the REAL part and sincos on the
  // imaginary one, so the primitives are recomputed with the axes exchanged.
  if (op == kSinh || op == kCosh) {
    Tr<T>::hc(x, cx, sx);        // cosh x, sinh x
    Tr<T>::sc(y, cy, sy);        // cos y,  sin y
    to_mpfr(cx, mcx); to_mpfr(sx, msx); to_mpfr(cy, mcy); to_mpfr(sy, msy);
  }
  // In the four variables as they now stand -- (cx, sx) from the axis that gets
  // the circular pair and (cy, sy) from the axis that gets the hyperbolic one,
  // already exchanged above for sinh/cosh -- all four ops share one shape.
  auto recombine = [&](mpfr_srcptr Cx, mpfr_srcptr Sx, mpfr_srcptr Cy,
                       mpfr_srcptr Sy, mpfr_ptr rr, mpfr_ptr ri) {
    switch (op) {
      case kSin:
      case kSinh: mpfr_mul(rr, Sx, Cy, MPFR_RNDN); mpfr_mul(ri, Cx, Sy, MPFR_RNDN); break;
      case kCos:  mpfr_mul(rr, Cx, Cy, MPFR_RNDN); mpfr_mul(ri, Sx, Sy, MPFR_RNDN);
                  mpfr_neg(ri, ri, MPFR_RNDN); break;
      default:    mpfr_mul(rr, Cx, Cy, MPFR_RNDN); mpfr_mul(ri, Sx, Sy, MPFR_RNDN); break;
    }
  };
  recombine(mcx, msx, mcy, msy, mcr, mci);

  st.total    = ulps_mod(mgr, mgi, mrr, mri, Tr<T>::sig_bits());
  st.composed = ulps_mod(mcr, mci, mrr, mri, Tr<T>::sig_bits());
  {
    // Share of the modulus carried by the term holding the poisoned cos().
    mpfr_hypot(t1, mcr, mci, MPFR_RNDN);
    // sin puts the poisoned cos(x) in the IMAGINARY term; the other three --
    // including sinh, whose axes are exchanged so that the poisoned factor is
    // cos(y) sitting in Sx*Cy -- put it in the real one.
    mpfr_abs(t2, (op == kSin) ? mci : mcr, MPFR_RNDN);
    st.share_poisoned = mpfr_zero_p(t1) ? 0.0 : mpfr_get_d(t2, MPFR_RNDN) / mpfr_get_d(t1, MPFR_RNDN);
  }
  if (poison_cos) {
    // THE CONTROL.  Corrupt the library's cos by one backend ulp, recombine,
    // and measure how far the recombined POINT moved -- not how far its error
    // MAGNITUDE moved.  The distinction matters: the injected displacement is
    // purely along one axis, so if the clean error vector happens to lie along
    // the other, |e + d| - |e| is second order and a magnitude test reads a
    // live arm as dead.  That is exactly what produced 2,166 spurious dead
    // cases on the first attempt.  Displacement has no such escape: it must
    // come out equal to the poisoned term's share of the modulus.
    const bool swapped = (op == kSinh || op == kCosh);
    T pc = swapped ? cy : cx;
    const double c0 = Tr<T>::limb(pc, 0);
    pc = xp::add(pc, T(std::ldexp(std::fabs(c0), -Tr<T>::sig_bits())));
    mpfr_t mpc_, pr, pi, dd, pred;
    mpfr_inits2(kPrec, mpc_, pr, pi, dd, pred, (mpfr_ptr)0);
    to_mpfr(pc, mpc_);
    if (swapped) recombine(mcx, msx, mpc_, msy, pr, pi);
    else         recombine(mpc_, msx, mcy, msy, pr, pi);

    // How much perturbation ACTUALLY landed.  Deriving the prediction from the
    // nominal one-ulp kick instead was the second wrong cut: xp::add rounds the
    // kick into the expansion's last limb, and on QF that limb's own ulp can be
    // a subnormal float, so the kick that arrives is not the kick that was
    // asked for.  Measuring what landed makes the test exact and leaves it a
    // test of the RECOMBINATION, which is the thing under control here.
    mpfr_sub(dd, mpc_, swapped ? mcy : mcx, MPFR_RNDN);
    mpfr_abs(dd, dd, MPFR_RNDN);
    mpfr_abs(pred, (op == kSin) ? msy : (op == kCos) ? mcy
                 : (op == kSinh) ? msx : mcx, MPFR_RNDN);
    mpfr_mul(pred, pred, dd, MPFR_RNDN);

    mpfr_sub(pr, pr, mcr, MPFR_RNDN);
    mpfr_sub(pi, pi, mci, MPFR_RNDN);
    mpfr_hypot(pr, pr, pi, MPFR_RNDN);
    mpfr_hypot(t1, mrr, mri, MPFR_RNDN);
    const double scale = mpfr_zero_p(t1) ? 0.0
                       : std::ldexp(1.0, Tr<T>::sig_bits()) / mpfr_get_d(t1, MPFR_RNDN);
    st.poison_delta = mpfr_zero_p(t1) ? -1.0 : mpfr_get_d(pr, MPFR_RNDN) * scale;
    st.poison_pred  = mpfr_zero_p(t1) ? -1.0 : mpfr_get_d(pred, MPFR_RNDN) * scale;
    st.poison_landed = !mpfr_zero_p(dd);
    mpfr_clears(mpc_, pr, pi, dd, pred, (mpfr_ptr)0);
  }

  // --- the four primitives, each against its own exact value ---------------
  {
    // The primitive arms are judged against the function of the STORED
    // argument, not of the exact double.  On DD the two are the same -- a
    // DoubleDouble holds any double exactly -- but on the FP32-word backends
    // the stored x differs from xre by up to in_delta, and charging that
    // difference to sin() would report the FORMAT's input floor as an
    // algorithm defect.  `total` and `composed` above deliberately keep the
    // storage error, because that is what the sweep's own column measures.
    const bool swap = (op == kSinh || op == kCosh);
    mpfr_t ax, ay, ex;
    mpfr_inits2(kPrec, ax, ay, ex, (mpfr_ptr)0);
    to_mpfr(x, ax);
    to_mpfr(y, ay);
    if (!swap) {
      mpfr_sin(ex, ax, MPFR_RNDN);  st.e_sinx  = ulps_real(msx, ex, Tr<T>::sig_bits());
      mpfr_cos(ex, ax, MPFR_RNDN);  st.e_cosx  = ulps_real(mcx, ex, Tr<T>::sig_bits());
      mpfr_sinh(ex, ay, MPFR_RNDN); st.e_sinhy = ulps_real(msy, ex, Tr<T>::sig_bits());
      mpfr_cosh(ex, ay, MPFR_RNDN); st.e_coshy = ulps_real(mcy, ex, Tr<T>::sig_bits());
    } else {
      mpfr_sinh(ex, ax, MPFR_RNDN); st.e_sinx  = ulps_real(msx, ex, Tr<T>::sig_bits());
      mpfr_cosh(ex, ax, MPFR_RNDN); st.e_cosx  = ulps_real(mcx, ex, Tr<T>::sig_bits());
      mpfr_sin(ex, ay, MPFR_RNDN);  st.e_sinhy = ulps_real(msy, ex, Tr<T>::sig_bits());
      mpfr_cos(ex, ay, MPFR_RNDN);  st.e_coshy = ulps_real(mcy, ex, Tr<T>::sig_bits());
    }
    mpfr_clears(ax, ay, ex, (mpfr_ptr)0);
  }

  mpc_clear(z); mpc_clear(w);
  mpfr_clears(mcx, msx, mcy, msy, mgr, mgi, mrr, mri, mcr, mci, t1, t2, (mpfr_ptr)0);
  return st;
}

struct Row { double total, composed; int pt; };

template <class T>
void run_backend(const std::vector<Pt>& grid, const char* only_op, int top_n, bool poison,
                 int& poison_seen, int& poison_dead, double& poison_worst,
                 int& poison_absorbed) {
  for (int op = 0; op < kNDirect; ++op) {
    if (only_op && std::strcmp(kOpName[op], only_op)) continue;
    std::vector<Row> rows;
    double wt = -1, wc = -1; int wtp = -1, wcp = -1;
    int n_above1 = 0, n_composed_above1 = 0, n = 0;
    for (const Pt& p : grid) {
      const Stage s = measure<T>(op, p.re, p.im, poison);
      if (s.total < 0 || s.composed < 0) continue;
      ++n;
      if (s.total > 1.0) ++n_above1;
      if (s.composed > 1.0) ++n_composed_above1;
      if (s.total > wt) { wt = s.total; wtp = p.idx; }
      if (s.composed > wc) { wc = s.composed; wcp = p.idx; }
      rows.push_back({s.total, s.composed, p.idx});
      if (poison) {
        // The control, sharpened.  Comparing the poisoned `composed` against a
        // fixed 0.5 threshold is too weak: on a point where the corrupted
        // primitive is DILUTE -- its term contributes a negligible share of the
        // modulus -- a full one-ulp corruption legitimately moves the
        // modulus-relative reading by almost nothing, and the case reads as a
        // failure that is really the metric behaving correctly.  So compare
        // like with like: run the same point clean, and require the poisoned
        // reading to differ.  A case only counts as dead if the injected ulp
        // moved `composed` by less than a quarter ulp AND the corrupted term
        // was NOT dilute, which is the only configuration in which silence
        // would mean the arm is not wired to the library at all.
        if (s.share_poisoned <= 0.1 || s.poison_delta < 0) continue;
        if (!s.poison_landed) { ++poison_absorbed; continue; }
        ++poison_seen;
        const double rel = std::fabs(s.poison_delta - s.poison_pred) / s.poison_pred;
        if (rel > 1e-9) ++poison_dead;
        if (rel > poison_worst) poison_worst = rel;
      }
    }
    if (poison) continue;
    std::printf("%-3s c %-5s  n=%4d   >1ulp total %4d  composed %4d   worst total %10.4g (pt %d)"
                "   worst composed %10.4g (pt %d)\n",
                Tr<T>::name(), kOpName[op], n, n_above1, n_composed_above1, wt, wtp, wc, wcp);
    if (top_n > 0) {
      std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.total > b.total; });
      for (int i = 0; i < top_n && i < (int)rows.size(); ++i) {
        const Pt* q = nullptr;
        for (const Pt& g : grid) if (g.idx == rows[i].pt) { q = &g; break; }
        std::printf("      pt %4d  total %10.4g  composed %10.4g  added %10.4g   z=(%.6g, %.6g) %s\n",
                    rows[i].pt, rows[i].total, rows[i].composed,
                    rows[i].total - rows[i].composed,
                    q ? q->re : 0.0, q ? q->im : 0.0, q ? q->family.c_str() : "?");
      }
    }
  }
}

int run(int argc, char** argv) {
  const char* only_op = nullptr;
  const char* only_be = nullptr;
  const char* dump_op = nullptr;
  int dump_pt = -1, top_n = 0;
  bool poison = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--op") && i + 1 < argc)           only_op = argv[++i];
    else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) only_be = argv[++i];
    else if (!std::strcmp(argv[i], "--point") && i + 2 < argc)  { dump_op = argv[++i]; dump_pt = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "--top") && i + 1 < argc)     top_n = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--poison"))                  poison = true;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  const std::vector<Pt> grid = load_grid();
  std::printf("probe_complex_trig_stages: %zu complex grid points, MPC %s at %ld bits\n",
              grid.size(), mpc_get_version(), (long)kPrec);
  std::printf("total/composed/added are MODULUS-RELATIVE, in ulps OF THE BACKEND\n\n");

  if (dump_op) {
    for (int op = 0; op < kNDirect; ++op) {
      if (std::strcmp(kOpName[op], dump_op)) continue;
      for (const Pt& p : grid) {
        if (p.idx != dump_pt) continue;
        std::printf("c %s point %d  family=%s  z=(%.17g, %.17g)\n\n",
                    dump_op, dump_pt, p.family.c_str(), p.re, p.im);
        std::printf("%-3s %10s %10s %10s | %9s %9s %9s %9s\n",
                    "be", "total", "composed", "added", "e(sin x)", "e(cos x)", "e(sinh y)", "e(cosh y)");
#define DUMP(T)                                                                                  \
        { const Stage s = measure<T>(op, p.re, p.im, false);                                     \
          std::printf("%-3s %10.4g %10.4g %10.4g | %9.4g %9.4g %9.4g %9.4g\n", Tr<T>::name(),    \
                      s.total, s.composed, s.total - s.composed, s.e_sinx, s.e_cosx,             \
                      s.e_sinhy, s.e_coshy); }
        DUMP(xp::DoubleDouble) DUMP(xp::FloatFloat) DUMP(xp::TripleFloat) DUMP(xp::QuadFloat)
#undef DUMP
        return 0;
      }
    }
    std::fprintf(stderr, "no such op/point\n");
    return 2;
  }

  int ps = 0, pd = 0, pa = 0;
  double pw = 0;
  const bool wantDD = !only_be || !std::strcmp(only_be, "DD");
  const bool wantFF = !only_be || !std::strcmp(only_be, "FF");
  const bool wantTF = !only_be || !std::strcmp(only_be, "TF");
  const bool wantQF = !only_be || !std::strcmp(only_be, "QF");
  if (wantDD) run_backend<xp::DoubleDouble>(grid, only_op, top_n, poison, ps, pd, pw, pa);
  if (wantFF) run_backend<xp::FloatFloat>(grid, only_op, top_n, poison, ps, pd, pw, pa);
  if (wantTF) run_backend<xp::TripleFloat>(grid, only_op, top_n, poison, ps, pd, pw, pa);
  if (wantQF) run_backend<xp::QuadFloat>(grid, only_op, top_n, poison, ps, pd, pw, pa);

  if (poison) {
    // The control is a DIFFERENCE, not a threshold.  Each poisoned case is
    // re-run clean at the same point and the two `composed` readings compared,
    // and a case only counts as dead when the injected ulp moved the reading by
    // less than a quarter ulp WHILE the corrupted term carried more than a
    // tenth of the modulus.  A fixed threshold on the poisoned reading alone
    // was the first cut and it was wrong: it charged 7,465 cases in which the
    // corrupted primitive is legitimately dilute -- cos(x) multiplied into a
    // term that is negligible against the modulus, where a full ulp of
    // corruption SHOULD be invisible to a modulus-relative metric.
    std::printf("POISON  corrupted the library's cos() by one backend ulp before the\n");
    std::printf("        recombination and measured the displacement it caused.\n");
    std::printf("        non-dilute cases (poisoned term > 10%% of the modulus): %d\n", ps);
    std::printf("        cases where the displacement missed its prediction: %d\n", pd);
    std::printf("        worst relative miss: %.4g\n", pw);
    std::printf("        cases the expansion absorbed the kick entirely: %d\n", pa);
    if (ps == 0) { std::printf("        FAIL: nothing was poisoned.\n"); return 1; }
    if (pd != 0) { std::printf("        FAIL: an injected ulp did not arrive at the recombination.\n"); return 1; }
    std::printf("        PASS: every injected ulp arrived, at the predicted size.  The\n");
    std::printf("              composed arm reads the library's primitives, so a low\n");
    std::printf("              reading there is a real result and not a dead wire.\n");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
