// probe_cpow.cpp -- attribute the complex-pow residual, stage by stage.
//
// WHY THIS EXISTS.  c pow carries the worst numbers anywhere in the sweep:
// 3,263 rows above 1 ulp, DD max 141.59, FF max 2.3107e+11, QF 6.504e+25,
// TF 3.8767e+18.  pow is `exp(w * log(z))` (dd_complex.hpp:1418), so it sits on
// top of complex exp (real exp + sincos) and complex log (real log + atan2), and
// a raw ulp count cannot say which of those is responsible -- or whether ANY of
// them is, because exp(w log z) has condition number kappa = |w log z| and the
// sweep's own operand draw lets that reach ~165.
//
// Worse, the reference has the same problem.  The sweep scores complex rows
// against libquadmath, which is 113 bits, while DD's ulp is defined at p = 106.
// At kappa ~ 165 the oracle keeps only 113 - 7.4 bits, so it carries ~1.3 DD
// ulps of its own error -- immaterial against a 141-ulp max, but NOT against the
// 1-ulp line that 1,043 DD rows are counted by.  That has to be measured before
// any of those rows are called defects.  probe_complex_oracle.cpp audits the
// one-argument complex ops and does not cover pow.
//
// WHAT IT MEASURES.  It replicates the sweep's operand draw for C_Pow exactly
// (same splitmix64/fnv1a/Rng, same stream_seed(seed,"pow",1), same grid order,
// so w at point i is the same w the sweep used), then for each point computes
// MPC z^w at 400 bits as the truth, cpowq as the sweep's own oracle, and a
// LADDER of arms per backend.  Each arm hands one stage its exact value and
// leaves the rest as shipped, so the arm that recovers the accuracy names the
// stage that lost it:
//
//   repr       the true answer merely ROUNDED into the format.  The absolute
//              floor: no implementation can read better than this.
//   full       xp::pow(z, w), as shipped.
//   exactlog   exp(w * L), L = MPC log(z) rounded in.  Gives away complex log.
//   exactexp   MPC exp of the BACKEND's own P.  Gives away complex exp.
//   exactmul   backend's own log AND exp, product w*L done exactly.  This is
//              the direct prediction of a compensated complex multiply in the
//              shipped code -- not an idealised bracket.
//   exactP     P = w log z exact, rounded into the format, backend exp.
//   floorP     P exact, rounded in, then exponentiated EXACTLY by MPC.
//   widemul    the product done exactly and kept as an unevaluated hi+lo pair,
//              backend's own log, exp(hi) + exp(hi)*lo.
//   wideP      same wide form fed an EXACT log.  Ceiling of the wide trick.
//   logulp     complex log's own error, in its own ulps.
//   logcost    what that error is worth ON THE ANSWER: |w|*|dL|*2^p.
//
// WHY floorP IS NOT A FLOOR.  It is tempting to call kappa*2^-p inherent:
// storing P costs |P|*2^-p ABSOLUTE, d(exp)/exp = dP, so it lands as kappa ulps
// relative no matter how good exp and log are.  That reasoning is wrong, and
// the wideP arm is what disproves it.  An implementation need not MATERIALISE
// P.  Carried as hi+lo,
//     exp(hi+lo) = exp(hi)*exp(lo) = exp(hi)*(1 + lo + lo^2/2 + ...)
// and |lo| <= |P|*2^-p puts the quadratic term below the format's resolution,
// so exp(hi) + exp(hi)*lo suffices.  Measured, that takes floorP's 611/598/
// 326/166 rows to 259/200/61/101 -- and TF's 61 is EXACTLY its repr count, i.e.
// with exact inputs the wide form lands precisely on the representation floor.
// Any analysis that charges the formula for rounding P is charging it for a
// cost it does not have to pay.  The excess analysis below uses max(repr,wideP).
//
// WHAT IT FOUND (1,778 points/backend; totals over all four).
//
//   arm        DD    FF    TF    QF   total   cut     what it would take
//   full     1048  1012   555   653    3268    0 %    shipped
//   exactlog  995   970   520   616    3101  5.1 %    a perfect complex log
//   exactexp  934   955   555   639    3083  5.7 %    a perfect complex exp
//   exactmul  944   887   532   366    2729 16.5 %    a perfect complex multiply
//   widemul   910   873   474   324    2581 21.0 %    wide product into exp
//   exactP    767   720   323   183    1993 39.0 %    perfect log AND product
//   floorP    611   598   326   166    1701 47.9 %    (not a floor -- see above)
//   wideP     259   200    61   101     621 81.0 %    wide product + EXACT log
//   repr        0    36    61    90     187 94.3 %    unreachable by anyone
//
// Four conclusions follow, and they are the point of the file.
//
// 1. NEITHER COMPONENT IS THE DEFECT.  A perfect complex exp buys 5.7 % and a
//    perfect complex log buys 5.1 %.  Both are already essentially correctly
//    rounded -- median logulp is 0.213/0.198/0.052/0.019, all under the 0.5 ulp
//    a correctly-rounded result is allowed.  (logulp's huge maxima, up to
//    1.8e16, are the |log z| -> 0 artifact at |z| ~ 1: the RELATIVE metric
//    explodes as |L| -> 0 while the absolute error stays negligible, which is
//    exactly what logcost confirms.)  This re-derives, independently and after
//    Phases 1-3, the earlier finding that complex pow's sensitivity is to the
//    ARGUMENT BITS and not to exp being wrong.
//
// 2. THE ERROR IS CONDITIONING ON P, AND MOSTLY IT IS LOG'S ROUNDING.  exp maps
//    an ABSOLUTE perturbation of P to a RELATIVE one on the answer, so every
//    absolute error in P is multiplied by 2^p to become ulps.  A CORRECTLY
//    ROUNDED log already injects |L|*2^-p absolute, which costs
//    |w|*|L| = kappa ulps.  logcost measures exactly that: median 0.669/0.570/
//    0.145/0.053, with 786/773/466/272 rows over 1 ulp.  This is why giving
//    away log alone buys nothing (5.1 %) -- a format-precision log, however
//    perfect, still pays it.
//
// 3. THE CEILING FOR ANY FIX THAT KEEPS A FORMAT-PRECISION LOG IS 21 %.  That
//    is widemul: the best possible complex multiply, its result carried wide
//    into exp, with the shipped log.  A compensated multiply alone is 16.5 %.
//    Reaching 81 % requires log(z) to return MORE than p bits, so that P's
//    absolute error falls below 2^-p -- a change to real log and atan2 in four
//    backends, not a change to complex pow.  On DD and FF it would also run
//    straight into the real sin/cos block error (see probe_arith_floor.cpp):
//    wideP's residual is 259 DD rows at max 3.41 ulps, against DD's 223 sin /
//    135 cos rows and a worst shipped sin of 3.326.
//
// 4. EVERY CATASTROPHIC MAXIMUM IS THE REPRESENTATION FLOOR.  repr reproduces
//    FF/TF/QF's headline maxima EXACTLY -- 2.31069e+11, 3.87669e+18,
//    6.50401e+25 -- so on those rows the library's answer is as good as any
//    answer could be; the format cannot hold the result.  187 of the 3,268 rows
//    are of this kind (DD 0, FF 36, TF 61, QF 90).  Point 634 is the type case:
//    z = (-1, 1e-28), |z| = 1.0 EXACTLY, which makes the sweep's own operand
//    clamp inert (it is guarded on `mod != 1.0`), and the clamp only bounds
//    |w*log2|z|| anyway -- it ignores that |z^w| = exp(-w_im * arg z) with
//    arg z ~ pi.  FF/TF/QF also show a precision-INDEPENDENT relative error at
//    these points (bit-identical to 5 significant figures at p = 48/72/96),
//    which rounding error cannot produce and the FP32 exponent range can.
//
// ORACLE AUDIT.  libquadmath cpowq against MPC at 400 bits, in each backend's
// own ulps: DD med 0.0156 / max 1.07 / ONE row above 1 ulp out of 1,778; FF, TF
// and QF zero rows.  The sweep's reference is adequate for pow.  (DD's median
// of 0.0156 is 2^-6 -- the 113-bit oracle expressed in 106-bit ulps.)
//
// SCORING.  Identical to the sweep, sweep_accuracy.cpp:2596 --
//   ulps = ldexp( |got - ref| / |ref|, p )
// modulus-relative, hypot of the two component errors over the modulus of the
// reference, at each backend's own p (106/48/72/96).  The complex metric is
// deliberately modulus-relative; a per-component version was tried in the sweep
// and reverted (6,105 false defects from collapsing scales), and this probe does
// not second-guess it.
//
// REPLICATION CHECK (--check).  The probe's `full` arm scored against `quad`
// must reproduce validation/sweep/sweep_baseline.csv row for row.  If the w draw
// were replicated wrongly, every number here would be about different operands
// than the sweep's and the attribution would be meaningless.  This is not
// optional -- run it before believing any other output.  It reads 0 disagree on
// 7,106 rows, using two SEPARATELY DERIVED slacks and calling a row a
// disagreement only if BOTH are exceeded:
//   - the CSV writes ulps with %.6g, so 6 significant digits, up to 5e-6
//     relative (sweep 2.05842e+25 vs probe 2.058416e+25 is not a discrepancy);
//   - the sweep forms its error in __float128 via to_q(), and
//     to_q(DoubleDouble) = (__float128)hi + (__float128)lo ROUNDS to 113 bits,
//     dropping lo's tail when the result is O(1).  So the sweep's complex
//     metric has a measurement floor of 2^(p-112) ulps -- 0.0156 for DD, and
//     <= 2^-16 for FF/TF/QF.  That predicted the residual disagreements would
//     be DD-only and under 0.0156; all 20 were, worst 0.0063.
//
// POISON (--poison).  Displaces the exactly-rounded REFERENCE by
// kPoisonUlps = 2^20 and requires every `full` cell to read 2^20 back.  Three
// things about this design are load-bearing and were each arrived at by getting
// it wrong first:
//   - it perturbs the REFERENCE, not the library's answer.  Perturbing the
//     answer by one ulp reads |err +- delta|, not delta, because the answer
//     already carries error -- that version read 1.886/1.866/1.077/1.154 and
//     was measuring nothing.
//   - the displacement is sized with an INDEPENDENT literal p, not Fmt<T>::p,
//     so a wrong p in the scorer cannot cancel against a wrong p in the poison.
//   - it is gated on repr < 1.  Without the gate FF/TF/QF "fail" with
//     deviations exactly equal to their `full` maxima -- which is not a bug but
//     conclusion 4: at those points the format cannot hold the reference, so
//     there is nothing to displace.
// Reads: DD 0.9202, FF 1.7241, TF 0.3971, QF 0.9551 deviation out of 1048576,
// on the 1778/1740/1715/1686 points each format can hold.
//
// MODES
//   (default)      every complex grid point; per-backend, per-arm summary
//   --check        replication check against the committed baseline CSV
//   --point N      dump one point in full: z, w, kappa, every arm, both refs
//   --top N        the N worst points for the `full` arm, per backend
//   --poison       negative control, see above
//   --prec BITS    MPC working precision (default 400)
//
// Build (needs MPC + MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_cpow.cpp -o /tmp/probe_cpow -lmpc -lmpfr -lgmp -lquadmath
//   /tmp/probe_cpow --check
//   /tmp/probe_cpow --poison
//   /tmp/probe_cpow

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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static mpfr_prec_t g_prec = 400;

// --poison displaces a known-good answer by this many ulps of the modulus.
// Large enough that the ~1 ulp cost of rounding the reference into the format
// is a 1e-6 effect, so the check can be tight.
static const long kPoisonUlps = 1L << 20;

// ---------------------------------------------------------------------------
// The sweep's operand draw, copied verbatim from scripts/sweep_accuracy.cpp so
// that w at point i is bit-for-bit the w the sweep used.  splitmix64 / fnv1a /
// Rng / stream_seed are sweep_accuracy.cpp:270-304; the C_Pow branch is
// fill_complex_operands, sweep_accuracy.cpp:657.  Copied rather than included
// because that file is a program, not a header.
// ---------------------------------------------------------------------------
static uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static uint64_t fnv1a(const char* s) {
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
  return h;
}
struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t s) : g(s) {}
  double unit() { return double(g() >> 11) * (1.0 / 9007199254740992.0); }
  int    in(int lo, int hi) { return lo + int(g() % uint64_t(hi - lo + 1)); }
  bool   coin() { return (g() & 1ull) != 0ull; }
  double logunif(int lo, int hi) { return std::ldexp(1.0 + unit(), in(lo, hi)); }
  double slogunif(int lo, int hi) { const double v = logunif(lo, hi); return coin() ? v : -v; }
};
static uint64_t stream_seed(uint64_t base, const char* name, unsigned kind) {
  return splitmix64(base ^ fnv1a(name) ^ (uint64_t(kind) * 0x9E3779B97F4A7C15ull));
}

static void draw_pow_exponent(double are, double aim, Rng& rng, double& bre, double& bim) {
  bre = rng.slogunif(-10, 4);
  bim = rng.slogunif(-10, 4);
  const double mod = std::hypot(are, aim);
  if (std::isfinite(mod) && mod > 0.0 && mod != 1.0) {
    const double la  = std::log2(mod);
    const double mag = std::hypot(bre, bim) * std::fabs(la);
    if (mag > 120.0) { const double s = 120.0 / mag; bre *= s; bim *= s; }
  }
}

// ---------------------------------------------------------------------------
// Format traits.  Same shape as probe_arith_floor.cpp: limb count, p, limb
// accessor, and a constructor from limbs.
// ---------------------------------------------------------------------------
template <class T> struct Fmt;

template <> struct Fmt<xp::DoubleDouble> {
  typedef double Limb; typedef xp::DoubleDoubleComplex Cx;
  static const int  n = 2, p = 106;
  static const char* name() { return "DD"; }
  static Limb  get(const xp::DoubleDouble& x, int i) { return i == 0 ? x.hi : x.lo; }
  static xp::DoubleDouble make(const Limb* v) { return xp::DoubleDouble(v[0], v[1]); }
};
template <> struct Fmt<xp::FloatFloat> {
  typedef float Limb; typedef xp::FloatFloatComplex Cx;
  static const int  n = 2, p = 48;
  static const char* name() { return "FF"; }
  static Limb  get(const xp::FloatFloat& x, int i) { return i == 0 ? x.hi : x.lo; }
  static xp::FloatFloat make(const Limb* v) { return xp::FloatFloat(v[0], v[1]); }
};
template <> struct Fmt<xp::TripleFloat> {
  typedef float Limb; typedef xp::TripleFloatComplex Cx;
  static const int  n = 3, p = 72;
  static const char* name() { return "TF"; }
  static Limb  get(const xp::TripleFloat& x, int i) { return i == 0 ? x.f0 : (i == 1 ? x.f1 : x.f2); }
  static xp::TripleFloat make(const Limb* v) { return xp::TripleFloat(v[0], v[1], v[2]); }
};
template <> struct Fmt<xp::QuadFloat> {
  typedef float Limb; typedef xp::QuadFloatComplex Cx;
  static const int  n = 4, p = 96;
  static const char* name() { return "QF"; }
  static Limb  get(const xp::QuadFloat& x, int i) {
    return i == 0 ? x.f0 : (i == 1 ? x.f1 : (i == 2 ? x.f2 : x.f3));
  }
  static xp::QuadFloat make(const Limb* v) { return xp::QuadFloat(v[0], v[1], v[2], v[3]); }
};

// Exact limb sum of a stored value, into an MPFR at g_prec.  Never used to feed
// the oracle an INPUT (that is the signed-zero trap probe_complex_oracle.cpp
// documents); only to read a stored RESULT back out at full width.
template <class T> static void to_mpfr(mpfr_t out, const T& x) {
  mpfr_set_zero(out, 1);
  mpfr_t t; mpfr_init2(t, g_prec);
  for (int i = 0; i < Fmt<T>::n; ++i) {
    mpfr_set_d(t, (double)Fmt<T>::get(x, i), MPFR_RNDN);
    mpfr_add(out, out, t, MPFR_RNDN);
  }
  mpfr_clear(t);
}

// Round an MPFR value INTO the format, by repeated nearest-limb extraction.
// This is the "exactly rounded" input the ablation arms are built from.
template <class T> static T from_mpfr(mpfr_srcptr v) {
  typedef typename Fmt<T>::Limb L;
  L    limb[4] = {L(0), L(0), L(0), L(0)};
  mpfr_t r, t; mpfr_init2(r, g_prec); mpfr_init2(t, g_prec);
  mpfr_set(r, v, MPFR_RNDN);
  for (int i = 0; i < Fmt<T>::n; ++i) {
    limb[i] = (L)mpfr_get_d(r, MPFR_RNDN);       // nearest double, then narrow
    if (!std::isfinite((double)limb[i])) break;
    mpfr_set_d(t, (double)limb[i], MPFR_RNDN);
    mpfr_sub(r, r, t, MPFR_RNDN);
  }
  mpfr_clear(r); mpfr_clear(t);
  return Fmt<T>::make(limb);
}

// ---------------------------------------------------------------------------
// The sweep's complex ulp metric, sweep_accuracy.cpp:2596.
//     ulps = ldexp( |got - ref| / |ref| , p )
// with |.| the complex modulus.  Computed here in MPFR at g_prec so that the
// SCORING itself cannot be the thing that runs out of precision.
// ---------------------------------------------------------------------------
static double ulps_vs(mpfr_srcptr gre, mpfr_srcptr gim, mpc_srcptr ref, int p) {
  mpfr_t dr, di, num, den;
  mpfr_inits2(g_prec, dr, di, num, den, (mpfr_ptr)0);
  mpfr_sub(dr, gre, mpc_realref(ref), MPFR_RNDN);
  mpfr_sub(di, gim, mpc_imagref(ref), MPFR_RNDN);
  mpfr_hypot(num, dr, di, MPFR_RNDN);
  mpfr_hypot(den, mpc_realref(ref), mpc_imagref(ref), MPFR_RNDN);
  double out;
  if (mpfr_zero_p(den) || !mpfr_number_p(den) || !mpfr_number_p(num)) {
    out = std::nan("");
  } else {
    mpfr_div(num, num, den, MPFR_RNDN);
    mpfr_mul_2si(num, num, p, MPFR_RNDN);
    out = mpfr_get_d(num, MPFR_RNDN);
  }
  mpfr_clears(dr, di, num, den, (mpfr_ptr)0);
  return out;
}

template <class T> static double ulps_stored(const typename Fmt<T>::Cx& g, mpc_srcptr ref) {
  mpfr_t a, b; mpfr_inits2(g_prec, a, b, (mpfr_ptr)0);
  to_mpfr<T>(a, g.re); to_mpfr<T>(b, g.im);
  const double u = ulps_vs(a, b, ref, Fmt<T>::p);
  mpfr_clears(a, b, (mpfr_ptr)0);
  return u;
}

// ---------------------------------------------------------------------------
// Grid loading.  Reads the doubles straight out of the manifest with strtod, so
// -0.0 round-trips as -0.0 and both references see the identical input.
// ---------------------------------------------------------------------------
struct GP { double re, im; std::string family; };

static std::vector<GP> load_complex_grid(const char* path) {
  std::vector<GP> g;
  FILE* f = std::fopen(path, "r");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  char line[512];
  while (std::fgets(line, sizeof line, f)) {
    if (line[0] == '#') continue;
    char* p = line;
    if (*p != 'c') continue;                       // kind
    char* fields[5] = {0,0,0,0,0};
    int   nf = 0;
    fields[nf++] = p;
    for (char* q = p; *q && nf < 5; ++q) if (*q == ',') { *q = 0; fields[nf++] = q + 1; }
    if (nf < 5) continue;
    GP gp;
    gp.family = fields[2];
    gp.re = std::strtod(fields[3], 0);
    gp.im = std::strtod(fields[4], 0);
    g.push_back(gp);
  }
  std::fclose(f);
  return g;
}

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------
struct Acc {
  std::vector<double> v;
  int   nbad = 0;
  void add(double u) { if (std::isfinite(u)) v.push_back(u); else ++nbad; }
  double med() { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size()/2]; }
  double mx()  { double m = 0; for (double x : v) if (x > m) m = x; return m; }
  int    over(double t) { int c = 0; for (double x : v) if (x > t) ++c; return c; }
  size_t n() { return v.size(); }
};

// ---------------------------------------------------------------------------
// Per-point evaluation for one backend: the full call and the three ablations.
// ---------------------------------------------------------------------------
struct ArmOut { double full, exactlog, exactP, exactexp, exactmul,
                       wideP, widemul, repr, floorP, kappa,
                       logulp, logcost; };

template <class T>
static ArmOut eval_backend(double zre, double zim, double wre, double wim,
                           mpc_srcptr ref, mpc_srcptr Lex, mpc_srcptr Pex,
                           bool poison, int p_lit) {
  typedef typename Fmt<T>::Cx Cx;
  const Cx z{T(zre), T(zim)};
  const Cx w{T(wre), T(wim)};

  ArmOut o;

  // --- repr: the FLOOR.  The true answer merely ROUNDED into the format.
  // No implementation of pow can beat this, because it is what the format
  // costs to hold the answer at all.  On the FP32-word backends the result of
  // z^w routinely lands outside the FP32 exponent range, where the leading
  // limb is subnormal and the trailing limbs are dead -- so this arm is not a
  // formality here, it is the measurement that decides whether the c-pow
  // residual is an algorithm defect or the format running out of exponent.
  {
    const Cx q{from_mpfr<T>(mpc_realref(ref)), from_mpfr<T>(mpc_imagref(ref))};
    o.repr = ulps_stored<T>(q, ref);
  }

  // --- full: exactly what the sweep calls -------------------------------
  if (poison) {
    // NEGATIVE CONTROL.  Perturbing the LIBRARY's answer would be useless: it
    // already carries its own error, so the reading would be |err +- delta|,
    // which is what the first version of this did and why it read 1.89 instead
    // of 1.00.  Instead build a KNOWN answer -- the reference rounded into the
    // format -- and displace it by a known kPoisonUlps ulps of the modulus.
    //
    // The displacement is sized with p_lit, a literal passed in from main's own
    // {106,48,72,96} table, NOT with Fmt<T>::p.  If the scoring path used the
    // wrong p, sizing the poison with the same wrong p would cancel the error
    // out and the control would pass while measuring nothing.
    mpfr_t m; mpfr_init2(m, g_prec);
    mpfr_hypot(m, mpc_realref(ref), mpc_imagref(ref), MPFR_RNDN);
    mpfr_mul_2si(m, m, -p_lit, MPFR_RNDN);         // one ulp of the modulus
    mpfr_mul_d(m, m, (double)kPoisonUlps, MPFR_RNDN);
    Cx q{from_mpfr<T>(mpc_realref(ref)), from_mpfr<T>(mpc_imagref(ref))};
    q.re = xp::add(q.re, from_mpfr<T>(m));
    mpfr_clear(m);
    // Error vector is purely real and of size kPoisonUlps ulps of |ref|, so the
    // modulus-relative reading must be kPoisonUlps -- give or take the ~1 ulp
    // that rounding the reference into the format costs in the first place.
    o.full = ulps_stored<T>(q, ref);
  } else {
    o.full = ulps_stored<T>(xp::pow(z, w), ref);
  }

  // --- exactlog: exp(w * L), L exactly rounded --------------------------
  {
    const Cx L{from_mpfr<T>(mpc_realref(Lex)), from_mpfr<T>(mpc_imagref(Lex))};
    o.exactlog = ulps_stored<T>(xp::exp(w * L), ref);
  }

  // --- exactP: exp(P), P = w*log z exactly rounded -----------------------
  {
    const Cx P{from_mpfr<T>(mpc_realref(Pex)), from_mpfr<T>(mpc_imagref(Pex))};
    o.exactP = ulps_stored<T>(xp::exp(P), ref);
  }

  // --- exactmul: the SHIPPED chain with ONLY the complex multiply fixed ---
  // exactlog and exactP bracket the multiply, but both hand the backend a log
  // it does not actually have.  This arm keeps the backend's own log(z) and
  // its own exp, and does only the product w * L exactly.  It is therefore the
  // direct prediction of what a compensated complex multiply would buy in the
  // shipped code -- the number to justify writing one, rather than the
  // idealised bracket.
  {
    const Cx Lbk = xp::log(z);
    mpfr_t lr, li; mpfr_inits2(g_prec, lr, li, (mpfr_ptr)0);
    to_mpfr<T>(lr, Lbk.re); to_mpfr<T>(li, Lbk.im);
    mpc_t L2, W2, P3; mpc_init2(L2, g_prec); mpc_init2(W2, g_prec); mpc_init2(P3, g_prec);
    mpc_set_fr_fr(L2, lr, li, MPC_RNDNN);
    mpc_set_d_d(W2, wre, wim, MPC_RNDNN);
    mpc_mul(P3, W2, L2, MPC_RNDNN);
    const Cx P{from_mpfr<T>(mpc_realref(P3)), from_mpfr<T>(mpc_imagref(P3))};
    o.exactmul = ulps_stored<T>(xp::exp(P), ref);
    mpc_clear(L2); mpc_clear(W2); mpc_clear(P3); mpfr_clears(lr, li, (mpfr_ptr)0);
  }

  // --- wideP / widemul: IS THE CONDITIONING FLOOR ACTUALLY A FLOOR? ------
  // floorP charges the formula for rounding P into the format, which costs
  // |P|*2^-p ABSOLUTE and therefore kappa ulps relative.  But that cost is a
  // property of MATERIALISING P, not of the identity z^w = exp(w log z).  If
  // the product is carried as an unevaluated hi+lo pair, then
  //     exp(hi+lo) = exp(hi)*exp(lo) = exp(hi)*(1+lo+lo^2/2+...)
  // and since |lo| <= |P|*2^-p the quadratic term is below the format's own
  // resolution, so exp(hi) + exp(hi)*lo is enough.  These two arms measure
  // whether that escapes the floor:
  //   wideP   -- exact P split hi/lo.  Ceiling of the trick.
  //   widemul -- the BACKEND's own log(z), product done exactly and split
  //              hi/lo.  What a compensated complex multiply returning a
  //              double-length result would actually deliver in shipped code.
  {
    auto wide_from = [&](mpc_srcptr Psrc, double& out) {
      mpfr_t hr, hi_, tr, ti;
      mpfr_inits2(g_prec, hr, hi_, tr, ti, (mpfr_ptr)0);
      const T phr = from_mpfr<T>(mpc_realref(Psrc));
      const T phi = from_mpfr<T>(mpc_imagref(Psrc));
      to_mpfr<T>(hr, phr); to_mpfr<T>(hi_, phi);
      mpfr_sub(tr, mpc_realref(Psrc), hr,  MPFR_RNDN);   // residual, exact
      mpfr_sub(ti, mpc_imagref(Psrc), hi_, MPFR_RNDN);
      const Cx Phi{phr, phi};
      const Cx Plo{from_mpfr<T>(tr), from_mpfr<T>(ti)};
      const Cx E = xp::exp(Phi);
      out = ulps_stored<T>(E + E * Plo, ref);            // exp(hi)*(1+lo)
      mpfr_clears(hr, hi_, tr, ti, (mpfr_ptr)0);
    };
    wide_from(Pex, o.wideP);

    const Cx Lbk2 = xp::log(z);
    mpfr_t lr2, li2; mpfr_inits2(g_prec, lr2, li2, (mpfr_ptr)0);
    to_mpfr<T>(lr2, Lbk2.re); to_mpfr<T>(li2, Lbk2.im);
    mpc_t L3, W3, P4; mpc_init2(L3, g_prec); mpc_init2(W3, g_prec); mpc_init2(P4, g_prec);
    mpc_set_fr_fr(L3, lr2, li2, MPC_RNDNN);
    mpc_set_d_d(W3, wre, wim, MPC_RNDNN);
    mpc_mul(P4, W3, L3, MPC_RNDNN);
    wide_from(P4, o.widemul);
    mpc_clear(L3); mpc_clear(W3); mpc_clear(P4); mpfr_clears(lr2, li2, (mpfr_ptr)0);
  }

  // --- logulp / logcost: IS THE COMPLEX LOG AT ITS OWN FLOOR? ------------
  // widemul (wide product, shipped log) and wideP (wide product, exact log)
  // differ only by the error the backend's own log(z) injects, so the gap
  // between them is entirely log's.  Two readings decide whether that gap is
  // fixable inside the format:
  //   logulp  -- log(z)'s own error in log(z)'s own ulps.  ~0.5 means it is
  //              already correctly rounded and only a WIDER log can help.
  //   logcost -- what that error is worth on the ANSWER.  exp turns an
  //              ABSOLUTE perturbation of P into a relative one, so the price
  //              is |w| * |L_bk - L_exact| * 2^p ulps, independent of how
  //              small it looks relative to |L|.
  {
    const Cx Lbk3 = xp::log(z);
    mpfr_t lr3, li3, dr, di; mpfr_inits2(g_prec, lr3, li3, dr, di, (mpfr_ptr)0);
    to_mpfr<T>(lr3, Lbk3.re); to_mpfr<T>(li3, Lbk3.im);
    mpfr_sub(dr, lr3, mpc_realref(Lex), MPFR_RNDN);
    mpfr_sub(di, li3, mpc_imagref(Lex), MPFR_RNDN);
    mpfr_t eh, lh; mpfr_inits2(g_prec, eh, lh, (mpfr_ptr)0);
    mpfr_hypot(eh, dr, di, MPFR_RNDN);
    mpfr_hypot(lh, mpc_realref(Lex), mpc_imagref(Lex), MPFR_RNDN);
    const double eabs = mpfr_get_d(eh, MPFR_RNDN);
    const double labs = mpfr_get_d(lh, MPFR_RNDN);
    o.logulp  = (labs > 0.0) ? std::ldexp(eabs / labs, Fmt<T>::p) : 0.0;
    o.logcost = std::ldexp(std::hypot(wre, wim) * eabs, Fmt<T>::p);
    mpfr_clears(lr3, li3, dr, di, eh, lh, (mpfr_ptr)0);
  }

  // --- floorP: THE INHERENT FLOOR OF THE FORMULA -------------------------
  // z^w = exp(P) requires P = w log z to exist as an intermediate, and storing
  // P in the format costs |P| * 2^-p ABSOLUTE.  d(exp)/exp = dP, so that lands
  // as |P| * 2^-p RELATIVE on the answer -- i.e. exactly kappa = |P| ulps, no
  // matter how good exp and log are.  This arm measures that and nothing else:
  // P rounded into the format, then exponentiated EXACTLY by MPC and rounded
  // in.  No implementation of exp(w log z) in this format can read better.
  {
    const T pr_t = from_mpfr<T>(mpc_realref(Pex)), pi_t = from_mpfr<T>(mpc_imagref(Pex));
    mpfr_t pr, pi; mpfr_inits2(g_prec, pr, pi, (mpfr_ptr)0);
    to_mpfr<T>(pr, pr_t); to_mpfr<T>(pi, pi_t);
    mpc_t P2, E2; mpc_init2(P2, g_prec); mpc_init2(E2, g_prec);
    mpc_set_fr_fr(P2, pr, pi, MPC_RNDNN);
    mpc_exp(E2, P2, MPC_RNDNN);
    const Cx got{from_mpfr<T>(mpc_realref(E2)), from_mpfr<T>(mpc_imagref(E2))};
    o.floorP = ulps_stored<T>(got, ref);
    mpc_clear(P2); mpc_clear(E2); mpfr_clears(pr, pi, (mpfr_ptr)0);
  }

  // --- exactexp: exp given away, backend's own P kept --------------------
  {
    const Cx Pbk = w * xp::log(z);
    mpfr_t pr, pi; mpfr_inits2(g_prec, pr, pi, (mpfr_ptr)0);
    to_mpfr<T>(pr, Pbk.re); to_mpfr<T>(pi, Pbk.im);
    mpc_t P, E; mpc_init2(P, g_prec); mpc_init2(E, g_prec);
    mpc_set_fr_fr(P, pr, pi, MPC_RNDNN);
    mpc_exp(E, P, MPC_RNDNN);
    const Cx got{from_mpfr<T>(mpc_realref(E)), from_mpfr<T>(mpc_imagref(E))};
    o.exactexp = ulps_stored<T>(got, ref);
    mpc_clear(P); mpc_clear(E);
    mpfr_clears(pr, pi, (mpfr_ptr)0);
  }
  return o;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const char* grid_path = "validation/sweep/sweep_grid.csv";
  bool poison = false, do_check = false;
  int  top = 0, one_point = -1;
  const char* base_csv = "validation/sweep/sweep_baseline.csv";

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--poison")) poison = true;
    else if (!std::strcmp(argv[i], "--check")) do_check = true;
    else if (!std::strcmp(argv[i], "--top") && i + 1 < argc) top = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--point") && i + 1 < argc) one_point = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--prec") && i + 1 < argc) g_prec = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--grid") && i + 1 < argc) grid_path = argv[++i];
    else if (!std::strcmp(argv[i], "--baseline") && i + 1 < argc) base_csv = argv[++i];
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }

  const std::vector<GP> grid = load_complex_grid(grid_path);
  std::printf("complex grid: %zu points   MPC prec %ld bits%s\n",
              grid.size(), (long)g_prec, poison ? "   [POISONED]" : "");

  // Replicate the sweep's stream for op "pow", kind 1 (complex), seed 12345.
  std::vector<double> wre(grid.size()), wim(grid.size());
  {
    Rng rng(stream_seed(12345ull, "pow", 1u));
    for (size_t i = 0; i < grid.size(); ++i)
      draw_pow_exponent(grid[i].re, grid[i].im, rng, wre[i], wim[i]);
  }

  // Accumulators: [backend][arm]; arm 0=full 1=exactlog 2=exactP 3=exactexp,
  // plus the oracle audit (quad vs mpc) at each backend's p.
  Acc acc[4][11], oracle[4];
  // (repr, full) kept per point and side by side.  Acc::add DROPS non-finite
  // readings, so its vectors do not share an index space across arms and
  // cannot be zipped -- gating the poison check on acc[][].v[k] silently
  // compares different points.
  std::vector<std::pair<double,double>> rp[4];
  // Per point: repr, floorP, full -- for the excess analysis below.
  struct Pt { double repr, floorP, wideP, widemul, full, kappa; int i; };
  std::vector<Pt> pts[4];
  Acc kappa_acc;
  std::vector<std::pair<double,int>> worst[4];
  // full scored against QUAD, for the replication check against the baseline.
  std::vector<double> full_vs_quad[4];
  for (int b = 0; b < 4; ++b) full_vs_quad[b].assign(grid.size(), std::nan(""));

  mpc_t Z, W, R, L, P;
  mpc_init2(Z, g_prec); mpc_init2(W, g_prec); mpc_init2(R, g_prec);
  mpc_init2(L, g_prec); mpc_init2(P, g_prec);

  for (size_t i = 0; i < grid.size(); ++i) {
    if (one_point >= 0 && (int)i != one_point) continue;
    const double zre = grid[i].re, zim = grid[i].im;

    mpc_set_d_d(Z, zre, zim, MPC_RNDNN);
    mpc_set_d_d(W, wre[i], wim[i], MPC_RNDNN);
    mpc_log(L, Z, MPC_RNDNN);
    mpc_mul(P, W, L, MPC_RNDNN);
    mpc_exp(R, P, MPC_RNDNN);            // reference: z^w, computed as exp(w log z)

    // kappa = |w log z|, the condition number of exp(w log z) w.r.t. a
    // relative perturbation of log z.
    double kap;
    { mpfr_t k; mpfr_init2(k, g_prec);
      mpfr_hypot(k, mpc_realref(P), mpc_imagref(P), MPFR_RNDN);
      kap = mpfr_get_d(k, MPFR_RNDN); mpfr_clear(k); }
    if (std::isfinite(kap)) kappa_acc.add(kap);

    // --- the sweep's own oracle -------------------------------------------
    __complex128 zq, wq, rq;
    __real__ zq = (__float128)zre; __imag__ zq = (__float128)zim;
    __real__ wq = (__float128)wre[i]; __imag__ wq = (__float128)wim[i];
    rq = cpowq(zq, wq);
    mpfr_t qr, qi; mpfr_inits2(g_prec, qr, qi, (mpfr_ptr)0);
    mpfr_set_float128(qr, __real__ rq, MPFR_RNDN);
    mpfr_set_float128(qi, __imag__ rq, MPFR_RNDN);
    mpc_t Q; mpc_init2(Q, g_prec); mpc_set_fr_fr(Q, qr, qi, MPC_RNDNN);

    const int ps[4] = {106, 48, 72, 96};
    for (int b = 0; b < 4; ++b) oracle[b].add(ulps_vs(qr, qi, R, ps[b]));

    ArmOut a[4];
    a[0] = eval_backend<xp::DoubleDouble>(zre, zim, wre[i], wim[i], R, L, P, poison, 106);
    a[1] = eval_backend<xp::FloatFloat>  (zre, zim, wre[i], wim[i], R, L, P, poison,  48);
    a[2] = eval_backend<xp::TripleFloat> (zre, zim, wre[i], wim[i], R, L, P, poison,  72);
    a[3] = eval_backend<xp::QuadFloat>   (zre, zim, wre[i], wim[i], R, L, P, poison,  96);

    // full scored against the QUAD oracle -- this is the sweep's own number.
    {
      const xp::DoubleDoubleComplex rd = xp::pow(
          xp::DoubleDoubleComplex{xp::DoubleDouble(zre), xp::DoubleDouble(zim)},
          xp::DoubleDoubleComplex{xp::DoubleDouble(wre[i]), xp::DoubleDouble(wim[i])});
      full_vs_quad[0][i] = ulps_stored<xp::DoubleDouble>(rd, Q);
      const xp::FloatFloatComplex rf = xp::pow(
          xp::FloatFloatComplex{xp::FloatFloat(zre), xp::FloatFloat(zim)},
          xp::FloatFloatComplex{xp::FloatFloat(wre[i]), xp::FloatFloat(wim[i])});
      full_vs_quad[1][i] = ulps_stored<xp::FloatFloat>(rf, Q);
      const xp::TripleFloatComplex rt = xp::pow(
          xp::TripleFloatComplex{xp::TripleFloat(zre), xp::TripleFloat(zim)},
          xp::TripleFloatComplex{xp::TripleFloat(wre[i]), xp::TripleFloat(wim[i])});
      full_vs_quad[2][i] = ulps_stored<xp::TripleFloat>(rt, Q);
      const xp::QuadFloatComplex rqf = xp::pow(
          xp::QuadFloatComplex{xp::QuadFloat(zre), xp::QuadFloat(zim)},
          xp::QuadFloatComplex{xp::QuadFloat(wre[i]), xp::QuadFloat(wim[i])});
      full_vs_quad[3][i] = ulps_stored<xp::QuadFloat>(rqf, Q);
    }

    for (int b = 0; b < 4; ++b) {
      acc[b][0].add(a[b].repr);     acc[b][1].add(a[b].floorP);
      acc[b][2].add(a[b].full);     acc[b][3].add(a[b].exactlog);
      acc[b][4].add(a[b].exactmul); acc[b][5].add(a[b].exactP);
      acc[b][6].add(a[b].exactexp); acc[b][7].add(a[b].widemul);
      acc[b][8].add(a[b].wideP);    acc[b][9].add(a[b].logulp);
      acc[b][10].add(a[b].logcost);
      if (std::isfinite(a[b].full)) worst[b].push_back({a[b].full, (int)i});
      rp[b].push_back({a[b].repr, a[b].full});
      pts[b].push_back({a[b].repr, a[b].floorP, a[b].wideP,
                        a[b].widemul, a[b].full, kap, (int)i});
    }

    if (one_point >= 0) {
      static const char* bn[4] = {"DD", "FF", "TF", "QF"};
      std::printf("\npoint %zu  family %s\n", i, grid[i].family.c_str());
      std::printf("  z = (%.17g, %.17g)\n  w = (%.17g, %.17g)\n", zre, zim, wre[i], wim[i]);
      std::printf("  |z| = %.17g   kappa = |w log z| = %.6g\n", std::hypot(zre, zim), kap);
      mpfr_printf("  log z   = (%.30Rg, %.30Rg)\n", mpc_realref(L), mpc_imagref(L));
      mpfr_printf("  P=w*log z=(%.30Rg, %.30Rg)\n", mpc_realref(P), mpc_imagref(P));
      mpfr_printf("  mpc  z^w= (%.30Rg, %.30Rg)\n", mpc_realref(R), mpc_imagref(R));
      mpfr_printf("  quad z^w= (%.30Rg, %.30Rg)\n", qr, qi);
      std::printf("  %-4s %13s %13s %13s %13s %13s %13s\n",
                  "bk", "oracle", "repr", "full", "exactlog", "exactP", "exactexp");
      for (int b = 0; b < 4; ++b)
        std::printf("  %-4s %13.6g %13.6g %13.6g %13.6g %13.6g %13.6g\n", bn[b],
                    oracle[b].v.empty() ? 0.0 : oracle[b].v.back(), a[b].repr,
                    a[b].full, a[b].exactlog, a[b].exactP, a[b].exactexp);
    }

    mpc_clear(Q); mpfr_clears(qr, qi, (mpfr_ptr)0);
  }

  if (one_point >= 0) return 0;

  static const char* bn[4] = {"DD", "FF", "TF", "QF"};
  static const char* an[11] = {"repr", "floorP", "full", "exactlog", "exactmul",
                              "exactP", "exactexp", "widemul", "wideP",
                              "logulp", "logcost"};

  std::printf("\nkappa = |w log z| over the grid: med %.4g  max %.4g\n",
              kappa_acc.med(), kappa_acc.mx());

  std::printf("\nORACLE AUDIT -- libquadmath cpowq vs MPC at %ld bits,\n"
              "in each backend's own ulps.  A row above 1.0 means the sweep's\n"
              "reference is itself past the line it is judging by.\n", (long)g_prec);
  std::printf("  %-4s %8s %14s %14s %10s\n", "bk", "n", "med", "max", ">1ulp");
  for (int b = 0; b < 4; ++b)
    std::printf("  %-4s %8zu %14.6g %14.6g %10d\n", bn[b], oracle[b].n(),
                oracle[b].med(), oracle[b].mx(), oracle[b].over(1.0));

  std::printf("\nSTAGE ATTRIBUTION -- all arms scored against MPC, not against\n"
              "libquadmath, so the oracle's own error is out of these numbers.\n");
  std::printf("  %-4s %-10s %8s %14s %14s %10s\n", "bk", "arm", "n", "med", "max", ">1ulp");
  for (int b = 0; b < 4; ++b) {
    for (int k = 0; k < 11; ++k)
      std::printf("  %-4s %-10s %8zu %14.6g %14.6g %10d\n", bn[b], an[k],
                  acc[b][k].n(), acc[b][k].med(), acc[b][k].mx(), acc[b][k].over(1.0));
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // EXCESS OVER THE FLOOR.  A row above 1 ulp is only a DEFECT if the format
  // could have held the answer (repr < 1) and the formula's own conditioning
  // did not already spend the budget (floorP < 1).  Everything else is the
  // format or the problem, not the library.
  // -----------------------------------------------------------------------
  if (!poison) {
    std::printf("EXCESS ANALYSIS -- of the rows above 1 ulp, how many are\n"
                "beyond ANY implementation's reach.  The floor used here is\n"
                "max(repr, wideP), NOT floorP: floorP charges the formula for\n"
                "rounding P into the format, and the wideP arm proves an\n"
                "implementation does not have to pay that.  `addressable` is\n"
                "what a better complex pow could still take.\n");
    std::printf("  %-4s %8s %8s %8s %11s %10s %10s %10s\n",
                "bk", ">1ulp", "unrepr", "inherent", "addressable",
                "worst-add", "widemul", "med kappa");
    for (int b = 0; b < 4; ++b) {
      int over = 0, unrepr = 0, inherent = 0, addressable = 0, wm = 0;
      double worstadd = 0; int worstpt = -1;
      std::vector<double> kk;
      for (const Pt& q : pts[b]) {
        if (std::isfinite(q.kappa)) kk.push_back(q.kappa);
        if (q.widemul > 1.0) ++wm;
        if (!(q.full > 1.0)) continue;
        ++over;
        if (!std::isfinite(q.repr)  || q.repr  >= 1.0) { ++unrepr;   continue; }
        if (!std::isfinite(q.wideP) || q.wideP >= 1.0) { ++inherent; continue; }
        ++addressable;
        if (q.full > worstadd) { worstadd = q.full; worstpt = q.i; }
      }
      std::sort(kk.begin(), kk.end());
      std::printf("  %-4s %8d %8d %8d %11d %10.4g %10d %10.4g   (worst at pt %d)\n",
                  bn[b], over, unrepr, inherent, addressable, worstadd, wm,
                  kk.empty() ? 0.0 : kk[kk.size()/2], worstpt);
    }
    std::printf("\n");
  }

  if (poison) {
    std::printf("POISON CHECK: a known answer displaced by %ld ulps of the modulus.\n"
                "Every `full` cell must read %ld back, within the ~1 ulp that rounding\n"
                "the reference into the format costs.  A cell reading 0 is a dead\n"
                "comparison; a cell reading %ld * 2^k has the wrong p.\n",
                kPoisonUlps, kPoisonUlps, kPoisonUlps);
    int fail = 0;
    for (int b = 0; b < 4; ++b) {
      // Restricted to points the format can actually HOLD the answer at
      // (repr < 1 ulp).  Where the result underflows the FP32 exponent range
      // the reference cannot be rounded in at all, so a displacement of it
      // measures the format's floor, not the scoring path -- that is what the
      // repr arm is for, and it is reported in the main table.
      double worst = 0; size_t used = 0;
      for (size_t k = 0; k < rp[b].size(); ++k) {
        const double repr = rp[b][k].first, got = rp[b][k].second;
        if (!std::isfinite(repr) || repr >= 1.0) continue;   // format cannot hold it
        if (!std::isfinite(got)) { worst = 1e300; ++used; continue; }
        worst = std::max(worst, std::fabs(got - (double)kPoisonUlps));
        ++used;
      }
      const bool ok = used > 0 && worst <= 4.0;
      std::printf("  %-4s n %zu of %zu representable  worst deviation from %ld: %.4f  %s\n",
                  bn[b], used, rp[b].size(), kPoisonUlps, worst, ok ? "ok" : "FAIL");
      if (!ok) ++fail;
    }
    if (fail) { std::printf("POISON FAILED on %d backend(s)\n", fail); return 1; }
    std::printf("POISON OK -- the scoring path is live and correctly scaled.\n");
    return 0;
  }

  if (top > 0) {
    for (int b = 0; b < 4; ++b) {
      std::sort(worst[b].begin(), worst[b].end(), std::greater<std::pair<double,int>>());
      std::printf("\n%s worst %d (full, vs MPC):\n", bn[b], top);
      for (int k = 0; k < top && k < (int)worst[b].size(); ++k) {
        const int i = worst[b][k].second;
        std::printf("  pt %-5d %-10s ulps %-14.6g z=(%.6g,%.6g) w=(%.6g,%.6g)\n",
                    i, grid[i].family.c_str(), worst[b][k].first,
                    grid[i].re, grid[i].im, wre[i], wim[i]);
      }
    }
  }

  if (do_check) {
    // REPLICATION CHECK.  full-vs-quad must equal the committed baseline's
    // c pow rows.  Anything else means the operand draw was not replicated and
    // no other number in this probe is about the sweep's inputs.
    FILE* f = std::fopen(base_csv, "r");
    if (!f) { std::fprintf(stderr, "\n--check: cannot open %s\n", base_csv); return 2; }
    char line[512];
    int  n = 0, bad = 0, skipped = 0, floored = 0;
    // TOLERANCE, and why it is not just "equal".
    //
    // The sweep forms its complex error term in __float128:
    //     e_re = to_q(r.re) - sref_re            sweep_accuracy.cpp:2553
    // and to_q(DoubleDouble) is (__float128)hi + (__float128)lo, an addition
    // that ROUNDS TO 113 BITS.  When the result is O(1) and DD's lo sits more
    // than 113 bits below hi, that add silently drops lo's tail and the sweep
    // reads e_re = 0 for a difference this probe (at 400 bits) still sees.
    // So the sweep's complex metric has a MEASUREMENT FLOOR of about
    //     2^-112 relative  ->  2^(p-112) in the backend's own ulps
    // which is 0.0156 ulps for DD (p=106) and at most 2^-16 for FF/TF/QF.
    //
    // That is a falsifiable prediction, not an excuse: it says disagreements
    // may appear ONLY on DD and ONLY below 0.0156 ulps.  Measured: 20 rows
    // disagree, all 20 DD, worst 0.0063 ulps.  Both halves hold.
    //
    // The second term is the CSV's own quantization -- ulps is written %.6g
    // (sweep_accuracy.cpp:2763), so six significant digits is all there is.
    std::vector<std::pair<double,std::string>> diffs;
    while (std::fgets(line, sizeof line, f)) {
      if (line[0] == '#') continue;
      char bkn[8], kind[8], op[16]; int pt; double dg, ul, bd; char st;
      if (std::sscanf(line, "%7[^,],%7[^,],%15[^,],%d,%lf,%lf,%lf,%c",
                      bkn, kind, op, &pt, &dg, &ul, &bd, &st) != 8) continue;
      if (std::strcmp(kind, "c") || std::strcmp(op, "pow")) continue;
      if (st != 'S') { ++skipped; continue; }   // only scored rows carry a verdict
      int b = -1;
      for (int k = 0; k < 4; ++k) if (!std::strcmp(bkn, bn[k])) b = k;
      if (b < 0 || pt < 0 || pt >= (int)grid.size()) continue;
      const double mine = full_vs_quad[b][pt];
      if (!std::isfinite(mine) || !std::isfinite(ul)) continue;
      ++n;
      // Judge by the ABSOLUTE gap in ulps: that is what could move a row across
      // the 1-ulp line.  A relative criterion says nothing on rows where both
      // readings are ~1e-30.
      const double d = std::fabs(mine - ul);
      static const int pbits[4] = {106, 48, 72, 96};
      // Two independent, separately derived slacks.
      //   csv   : ulps is written %.6g, six significant decimal digits, so the
      //           worst-case representation error is half a unit in the sixth
      //           digit -- 5e-6 relative when the mantissa is just above 1.
      //   floor : the sweep's 113-bit measurement floor, derived above.
      const double tol_csv   = 1e-5 * std::fabs(ul);
      const double tol_floor = std::ldexp(1.0, pbits[b] - 112);
      if (d > tol_csv && d > tol_floor) {
        ++bad;
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s pt %-5d  sweep %-15.8g probe %-15.8g  gap %.3g ulps",
                      bn[b], pt, ul, mine, d);
        diffs.push_back({d, std::string(buf)});
      } else if (d > tol_csv) {
        ++floored;   // agrees ONLY because the sweep could not resolve it
      }
    }
    std::fclose(f);
    std::printf("\nREPLICATION CHECK vs %s\n", base_csv);
    std::printf("  %d scored c-pow rows compared (%d non-S skipped); %d disagree\n",
                n, skipped, bad);
    std::printf("  %d further rows agree only inside the sweep's 113-bit measurement\n"
                "  floor (2^(p-112) ulps) -- all far below the 1-ulp line.\n", floored);
    if (bad) {
      std::sort(diffs.begin(), diffs.end(), std::greater<std::pair<double,std::string>>());
      for (size_t k = 0; k < diffs.size() && k < 20; ++k)
        std::printf("    %s\n", diffs[k].second.c_str());
      std::printf("  FAIL -- this probe is not reproducing the sweep's own numbers.\n");
      return 1;
    }
    std::printf("  OK -- this probe is scoring the sweep's own operands.\n");
  }

  mpc_clear(Z); mpc_clear(W); mpc_clear(R); mpc_clear(L); mpc_clear(P);
  return 0;
}
