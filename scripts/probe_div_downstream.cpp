// ===========================================================================
// probe_div_downstream — WHY does a MORE accurate real divide make QF/TF
// complex asin / asinh / atan / atanh SCORE WORSE?
//
// Context.  scripts/probe_div_lift.cpp measured the exact power-of-two lift
// for real divide and d7ac5af shipped it.  The div ops themselves improved
// (QF r div 110 -> 88 rows above 1 ulp of 1686 scored; the four load-bearing
// points land on the format floor).  Sweep-wide: 438 rows improved, 85
// increased, 0 new defects, open_defects.txt still empty.  But 28 rows
// regressed by more than the 3.73x pre-accepted ratio, all of them DOWNSTREAM
// of divide rather than divide itself, and that killed P5.1 by its own stop
// condition.
//
// The precedent this has to be judged against is the sqrt seed (f81a98c),
// which merged on a superficially similar trade.  It merged because three
// things were shown, by measurement, not argument:
//     (1) every op that moved has sqrt in its call path;
//     (2) every op WITHOUT sqrt in its path did not move at all;
//     (3) nothing became a defect.
// This probe runs the equivalent test for divide, and then answers the
// question (1)-(3) cannot: whether the moved rows are error REDISTRIBUTION --
// a correct divide no longer cancelling against an incorrect caller -- or a
// real DEFECT in the lift.
//
// ---------------------------------------------------------------------------
// HOW IT SEES THE INTERNAL DIVIDES
// ---------------------------------------------------------------------------
// scripts/probe_div_downstream.sh copies include/xp/ to a temp tree and
// renames each `divide` DEFINITION to `<be>_divide_shipped`, leaving the
// forward declaration each header already carries (dd:84 ff:96 tf:84 qf:95)
// undefined.  This file supplies that definition.  Every `divide` call inside
// qf_complex.hpp and friends therefore lands in probe_div() below, which
//   * counts the call and whether the lift guard fired,
//   * optionally records (a, b, q_core, q_lifted) for later MPFR scoring,
//   * and returns whichever of FIVE divide policies is selected:
//
//       plain   detail::<be>_divide_core     -- the pre-lift divide, exactly
//       lift    <be>_divide_shipped          -- what d7ac5af ships
//       floor   exact quotient at 400 bits, greedily rounded into the format
//               -- the best any divide could possibly return
//       poison  plain with its last word zeroed -- a deliberately WORSE divide
//       wreck   plain's leading word scaled by (1 + 2^-20) -- also worse, but
//               unlike poison it cannot be a no-op
//
// `floor` is the arm that settles the question.  If the caller gets worse
// under `lift`, and worse again (or the same) under `floor`, then a PERFECT
// divide also makes the caller worse, and the lift is not the defect -- the
// caller's own error was being masked.  If instead `lift` is worse than
// `floor` by a wide margin, the lift is returning something a correct divide
// would not, and the mechanism is broken.
//
// `poison`/`wreck` are the negative controls.  Every arm of this probe must be
// able to fail: if degrading divide does NOT move the downstream score, the
// hook is not on the path being measured and nothing else here means anything.
// `poison` alone was NOT good enough -- zeroing the last word is a literal
// no-op when that word has already flushed below FLT_TRUE_MIN, which is
// precisely the regime the headline rows live in, so it sat inert on exactly
// the points it was supposed to police.  A control that can be silently inert
// is not a control; `wreck` was added because it always moves.
//
// The shipped headers are never touched.  git status stays clean.
//
// ---------------------------------------------------------------------------
// HOW IT REPRODUCES THE SWEEP'S POINTS
// ---------------------------------------------------------------------------
// All 28 regressed rows are COMPLEX rows, and 26 of them are UNARY complex
// ops.  sweep_accuracy.cpp:659 fill_complex_operands() returns immediately
// with bre = bim = 0 when kComplex[id].nops < 2, and build_complex_grid()
// (:525-580) uses NO RNG at all, so a unary complex op's operand is literally
// cgrid[i].re, cgrid[i].im and is exactly reproducible.  This is the opposite
// of the REAL realm, where operand reconstruction is a known hazard
// (sweep_accuracy.cpp:2470; three variants of fill_real_operands scored 0 of
// 8 points).  build_complex_grid(), Rng, stream_seed and
// fill_complex_operands are transcribed below so that the binary ops are
// reproducible too, and --verify checks the reproduction against the
// committed CSVs rather than assuming it.
//
// ---------------------------------------------------------------------------
// ORACLE
// ---------------------------------------------------------------------------
// MPC at 400 bits, not the sweep's __complex128.  400 bits is 4.2x QF's p=96,
// so the oracle is not a party to any of the differences measured here.  The
// sweep's ulps convention is reproduced exactly: modulus-relative,
// |got - ref| / |ref| * 2^p, against the reference at the GRID doubles, with
// layer 0 (score-at-stored-operand) deliberately NOT applied on the complex
// side -- see sweep_accuracy.cpp:2551-2575 for why (to_q destroys the sign of
// a zero and the complex grid carries both signed zeros on purpose).
//
// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------
//     bash scripts/probe_div_downstream.sh
//     /tmp/probe_div_downstream --census        per op x backend: divide calls,
//                                               guard fires, points changed
//     /tmp/probe_div_downstream --verify        reproduce the CSVs' 28 rows
//     /tmp/probe_div_downstream --points        the five families, five arms
//     /tmp/probe_div_downstream --trace QF asin 738     one point's divides
//     /tmp/probe_div_downstream --traceall      EVERY divide in ALL 28 rows,
//                                               each scored against the exact
//                                               400-bit quotient
//     /tmp/probe_div_downstream --traceall-control      the same with the two
//                                               quotients swapped; must report
//                                               the mirror-image verdict, which
//                                               is what makes --traceall able
//                                               to fail
//
//     bash scripts/probe_div_sites.sh           which call site each divide is,
//                                               via a separate -DPROBE_SITES
//                                               build and addr2line -i
// ===========================================================================

#include <xp/dd_complex.hpp>
#include <xp/ff_complex.hpp>
#include <xp/qf_complex.hpp>
#include <xp/tf_complex.hpp>

#include <mpc.h>
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

// ---------------------------------------------------------------------------
// format traits — verbatim from probe_div_lift.cpp, plus the complex type
// ---------------------------------------------------------------------------
template <class T> struct Fmt;

template <> struct Fmt<DoubleDouble> {
    using Limb = double;  using Z = DoubleDoubleComplex;
    static const int n = 2, p = 106, limb_bits = 53;
    static const char* name() { return "DD"; }
    static double limb(const DoubleDouble& a, int i) { return i ? a.lo : a.hi; }
    static DoubleDouble make(const double* w) { return DoubleDouble(w[0], w[1]); }
};
template <> struct Fmt<FloatFloat> {
    using Limb = float;   using Z = FloatFloatComplex;
    static const int n = 2, p = 48, limb_bits = 24;
    static const char* name() { return "FF"; }
    static float limb(const FloatFloat& a, int i) { return i ? a.lo : a.hi; }
    static FloatFloat make(const float* w) { return FloatFloat(w[0], w[1]); }
};
template <> struct Fmt<TripleFloat> {
    using Limb = float;   using Z = TripleFloatComplex;
    static const int n = 3, p = 72, limb_bits = 24;
    static const char* name() { return "TF"; }
    static float limb(const TripleFloat& a, int i) { return i == 0 ? a.f0 : i == 1 ? a.f1 : a.f2; }
    static TripleFloat make(const float* w) { return TripleFloat(w[0], w[1], w[2]); }
};
template <> struct Fmt<QuadFloat> {
    using Limb = float;   using Z = QuadFloatComplex;
    static const int n = 4, p = 96, limb_bits = 24;
    static const char* name() { return "QF"; }
    static float limb(const QuadFloat& a, int i) {
        return i == 0 ? a.f0 : i == 1 ? a.f1 : i == 2 ? a.f2 : a.f3;
    }
    static QuadFloat make(const float* w) { return QuadFloat(w[0], w[1], w[2], w[3]); }
};

// exact limb sum -> mpfr
template <class T> static void to_mpfr(mpfr_t out, const T& a) {
    mpfr_set_zero(out, 1);
    mpfr_t t; mpfr_init2(t, kPrec);
    for (int i = 0; i < Fmt<T>::n; ++i) {
        mpfr_set_d(t, (double)Fmt<T>::limb(a, i), MPFR_RNDN);
        mpfr_add(out, out, t, MPFR_RNDN);
    }
    mpfr_clear(t);
}

// greedy nearest-limb extraction: the format floor for a real value
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

// scalar ulps, sweep_accuracy.cpp:ulps_scalar convention
template <class T> static double ulps(const T& got, mpfr_srcptr ref) {
    if (mpfr_zero_p(ref) || !mpfr_number_p(ref)) return -1.0;
    mpfr_t g, d, rr;
    mpfr_init2(g, kPrec); mpfr_init2(d, kPrec); mpfr_init2(rr, kPrec);
    to_mpfr(g, got);
    if (!mpfr_number_p(g)) { mpfr_clears(g, d, rr, (mpfr_ptr)0); return -2.0; }
    mpfr_sub(d, g, ref, MPFR_RNDN);
    mpfr_abs(d, d, MPFR_RNDN);
    mpfr_abs(rr, ref, MPFR_RNDN);
    mpfr_mul_2si(rr, rr, -Fmt<T>::p, MPFR_RNDN);
    mpfr_div(d, d, rr, MPFR_RNDN);
    double u = mpfr_get_d(d, MPFR_RNDN);
    mpfr_clears(g, d, rr, (mpfr_ptr)0);
    return u;
}

// ===========================================================================
// THE DIVIDE HOOK
// ===========================================================================
// P_POISON drops the quotient's last word.  That is a real degradation
// wherever the last word carries anything -- but at operands small enough for
// the lift guard to fire, the last word is frequently ALREADY zero (flushed
// below FLT_TRUE_MIN), and then the poison is a literal no-op.  P_WRECK exists
// because a negative control that can be silently inert is not a control: it
// perturbs the LEADING word by 2^-20 relative, which no divide can survive, so
// a point where P_WRECK does not move the score is a point where the hook is
// not on the path and nothing else measured there means anything.
enum Policy { P_PLAIN = 0, P_LIFT = 1, P_FLOOR = 2, P_POISON = 3, P_WRECK = 4 };
static const char* kPolicyName[5] = {"plain", "lift", "floor", "poison", "wreck"};

static int  g_policy = P_LIFT;
static bool g_trace  = false;
static long g_calls  = 0;
static long g_fires  = 0;

struct DivRec {
    const char* be;
    int    n, p, fired;
    double a[4], b[4], core[4], lift[4];
    void*  site;    // caller address; only meaningful under PROBE_SITES
};
static std::vector<DivRec> g_recs;

// ---------------------------------------------------------------------------
// Call-site attribution (PROBE_SITES build only).
//
// Every real divide in the complex headers resolves to the hook below, so the
// hook by itself cannot say WHICH call site a divide came from.  Under
// PROBE_SITES the four hooks are made noinline and record
// __builtin_return_address(0); `addr2line -i` then unwinds the whole inline
// chain back to the qf_complex.hpp line.
//
// This is deliberately a SEPARATE binary, not a flag on the measurement one.
// Forbidding the inline of `divide` also forbids GCC from contracting a
// multiply inside divide with an add in the caller, which could move the last
// bit.  Attribution must not be able to perturb the numbers it explains, so
// the sites binary is built apart and re-checked with --verify: if it does not
// reproduce the same 28 ulps, its attribution is not about the same code.
// ---------------------------------------------------------------------------
#ifdef PROBE_SITES
#  define PROBE_DIV_ATTR __attribute__((noinline))
#  define PROBE_DIV_SITE __builtin_return_address(0)
#else
#  define PROBE_DIV_ATTR XPMATH_INLINE_FUNCTION
#  define PROBE_DIV_SITE nullptr
#endif

template <class T> static T exact_quotient(const T& a, const T& b) {
    mpfr_t ma, mb, q;
    mpfr_init2(ma, kPrec); mpfr_init2(mb, kPrec); mpfr_init2(q, kPrec);
    to_mpfr(ma, a); to_mpfr(mb, b);
    T out;
    if (mpfr_zero_p(mb) || !mpfr_number_p(ma) || !mpfr_number_p(mb)) {
        out = a;                       // leave degenerate cases to the caller
    } else {
        mpfr_div(q, ma, mb, MPFR_RNDN);
        out = from_mpfr<T>(q);
    }
    mpfr_clears(ma, mb, q, (mpfr_ptr)0);
    return out;
}

// negative control: a divide that is strictly worse than `plain` by one word
template <class T> static T drop_last_word(const T& v) {
    typename Fmt<T>::Limb w[4] = {0, 0, 0, 0};
    for (int i = 0; i < Fmt<T>::n - 1; ++i) w[i] = Fmt<T>::limb(v, i);
    return Fmt<T>::make(w);
}

// harder negative control: 2^-20 relative on the leading word
template <class T> static T wreck(const T& v) {
    typename Fmt<T>::Limb w[4] = {0, 0, 0, 0};
    for (int i = 0; i < Fmt<T>::n; ++i) w[i] = Fmt<T>::limb(v, i);
    w[0] = (typename Fmt<T>::Limb)((double)w[0] * (1.0 + 0x1p-20));
    return Fmt<T>::make(w);
}

// numeric word equality: +0 and -0 are the SAME NUMBER, and a trailing word
// flipping its zero sign is not a change in the value.  memcmp would call it
// one, and did -- an earlier revision of the census reported 280 "changed"
// points for QF c tanh against 0 moved rows in the CSV, all signed zeros.
template <class T> static bool same_value(const T& x, const T& y) {
    for (int i = 0; i < Fmt<T>::n; ++i) {
        const double a = (double)Fmt<T>::limb(x, i), b = (double)Fmt<T>::limb(y, i);
        if (std::isnan(a) && std::isnan(b)) continue;
        if (a != b) return false;
    }
    return true;
}

template <class T>
static T probe_div(const T& a, const T& b, const T& core, const T& lifted, bool fired,
                   void* site) {
    ++g_calls;
    if (fired) ++g_fires;
    if (g_trace) {
        DivRec r;
        r.be = Fmt<T>::name(); r.n = Fmt<T>::n; r.p = Fmt<T>::p; r.fired = fired ? 1 : 0;
        r.site = site;
        for (int i = 0; i < 4; ++i) { r.a[i] = r.b[i] = r.core[i] = r.lift[i] = 0.0; }
        for (int i = 0; i < Fmt<T>::n; ++i) {
            r.a[i]    = (double)Fmt<T>::limb(a, i);
            r.b[i]    = (double)Fmt<T>::limb(b, i);
            r.core[i] = (double)Fmt<T>::limb(core, i);
            r.lift[i] = (double)Fmt<T>::limb(lifted, i);
        }
        g_recs.push_back(r);
    }
    switch (g_policy) {
        case P_PLAIN:  return core;
        case P_FLOOR:  return exact_quotient<T>(a, b);
        case P_POISON: return drop_last_word<T>(core);
        case P_WRECK:  return wreck<T>(core);
        default:       return lifted;
    }
}

namespace xp {
PROBE_DIV_ATTR DoubleDouble divide(DoubleDouble a, DoubleDouble b) {
    return probe_div<DoubleDouble>(a, b, detail::dd_divide_core(a, b),
                                   dd_divide_shipped(a, b), detail::dd_div_lift_wanted(a, b),
                                   PROBE_DIV_SITE);
}
PROBE_DIV_ATTR FloatFloat divide(FloatFloat a, FloatFloat b) {
    return probe_div<FloatFloat>(a, b, detail::ff_divide_core(a, b),
                                 ff_divide_shipped(a, b), detail::ff_div_lift_wanted(a, b),
                                 PROBE_DIV_SITE);
}
PROBE_DIV_ATTR TripleFloat divide(TripleFloat a, TripleFloat b) {
    return probe_div<TripleFloat>(a, b, detail::tf_divide_core(a, b),
                                  tf_divide_shipped(a, b), detail::tf_div_lift_wanted(a, b),
                                  PROBE_DIV_SITE);
}
PROBE_DIV_ATTR QuadFloat divide(QuadFloat a, QuadFloat b) {
    return probe_div<QuadFloat>(a, b, detail::qf_divide_core(a, b),
                                qf_divide_shipped(a, b), detail::qf_div_lift_wanted(a, b),
                                PROBE_DIV_SITE);
}
}  // namespace xp

// ===========================================================================
// THE SWEEP'S COMPLEX GRID AND OPERANDS — transcribed from
// scripts/sweep_accuracy.cpp:270-330 (rng), :405 (op table), :525-580 (grid),
// :659-682 (operands), :2495 (eval).  --verify checks the transcription.
// ===========================================================================
struct GridPoint { double re, im; };

static const int kCutDecades = 30;
static double pow10i(int e) { return std::pow(10.0, double(e)); }

static std::vector<GridPoint> build_complex_grid() {
    std::vector<GridPoint> g;
    const double kMod[] = {1e-8, 1e-4, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 10.0, 1e4, 1e8};
    const int    kArgs  = 32;
    for (size_t m = 0; m < sizeof(kMod) / sizeof(kMod[0]); ++m)
        for (int j = 0; j < kArgs; ++j) {
            const double th = 2.0 * M_PI * double(j) / double(kArgs);
            g.push_back({kMod[m] * std::cos(th), kMod[m] * std::sin(th)});
        }
    const double kRealAnchor[] = {-100.0, -10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0};
    const double kImagAnchor[] = {-10.0, -2.0, -1.0, 1.0, 2.0, 10.0};
    for (size_t i = 0; i < sizeof(kRealAnchor) / sizeof(kRealAnchor[0]); ++i) {
        const double x = kRealAnchor[i];
        g.push_back({x, 0.0});
        g.push_back({x, -0.0});
        for (int p = 0; p <= kCutDecades; ++p) {
            const double d = pow10i(-p);
            g.push_back({x,  d});
            g.push_back({x, -d});
        }
    }
    for (size_t i = 0; i < sizeof(kImagAnchor) / sizeof(kImagAnchor[0]); ++i) {
        const double y = kImagAnchor[i];
        g.push_back({0.0,  y});
        g.push_back({-0.0, y});
        for (int p = 0; p <= kCutDecades; ++p) {
            const double d = pow10i(-p);
            g.push_back({ d, y});
            g.push_back({-d, y});
        }
    }
    {
        std::vector<double> x;
        for (int k = 1; k <= 30; ++k) { const double d = pow10i(-k); x.push_back(d); x.push_back(-d); }
        for (int k = 1; k <= 15; ++k) { const double d = pow10i( k); x.push_back(d); x.push_back(-d); }
        for (int k = 1; k <= 16; ++k) {
            const double d = pow10i(-k);
            x.push_back(1.0 - d); x.push_back(-1.0 + d);
            x.push_back(1.0 + d); x.push_back(-1.0 - d);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            g.push_back({x[i],  0.0});
            g.push_back({x[i], -0.0});
        }
    }
    return g;
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t fnv1a(const char* s) {
    uint64_t h = 1469598103934665603ull;
    for (; *s; ++s) { h ^= uint64_t((unsigned char)*s); h *= 1099511628211ull; }
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
static const uint64_t kDefaultSeed = 12345ull;

enum C {
    C_Add, C_Sub, C_Mul, C_Div,
    C_Abs, C_Conj, C_Sqrt, C_Exp, C_Log, C_Log10,
    C_Sin, C_Cos, C_Tan, C_Asin, C_Acos, C_Atan,
    C_Sinh, C_Cosh, C_Tanh, C_Asinh, C_Acosh, C_Atanh,
    C_Pow, C_Polar,
    C_COUNT
};
struct ComplexSpec { const char* name; int nops; };
static const ComplexSpec kComplex[C_COUNT] = {
    {"add", 2}, {"sub", 2}, {"mul", 2}, {"div", 2},
    {"abs", 1}, {"conj", 1}, {"sqrt", 1}, {"exp", 1}, {"log", 1}, {"log10", 1},
    {"sin", 1}, {"cos", 1}, {"tan", 1}, {"asin", 1}, {"acos", 1}, {"atan", 1},
    {"sinh", 1}, {"cosh", 1}, {"tanh", 1}, {"asinh", 1}, {"acosh", 1}, {"atanh", 1},
    {"pow", 2}, {"polar", 1},
};

static void fill_complex_operands(int id, size_t i, const std::vector<GridPoint>& grid,
                                  double are, double aim, Rng& rng, double& bre, double& bim) {
    bre = 0.0; bim = 0.0;
    if (kComplex[id].nops < 2) return;
    if (id == C_Pow) {
        bre = rng.slogunif(-10, 4);
        bim = rng.slogunif(-10, 4);
        const double mod = std::hypot(are, aim);
        if (std::isfinite(mod) && mod > 0.0 && mod != 1.0) {
            const double la  = std::log2(mod);
            const double mag = std::hypot(bre, bim) * std::fabs(la);
            if (mag > 120.0) { const double s = 120.0 / mag; bre *= s; bim *= s; }
        }
        return;
    }
    if (i % 7 == 1) {
        const double eps = std::ldexp(1.0, -rng.in(1, 53)) * (rng.coin() ? 1.0 : -1.0);
        const double s   = (id == C_Add) ? -(1.0 + eps) : (1.0 + eps);
        bre = are * s; bim = aim * s;
        return;
    }
    const GridPoint& p = grid[(i * 7 + 3) % grid.size()];
    bre = p.re; bim = p.im;
}

template <class T>
static typename Fmt<T>::Z eval_c(int id, const typename Fmt<T>::Z& a,
                                 const typename Fmt<T>::Z& b, bool& is_real) {
    using Z = typename Fmt<T>::Z;
    is_real = false;
    switch (id) {
        case C_Add:   return a + b;
        case C_Sub:   return a - b;
        case C_Mul:   return a * b;
        case C_Div:   return a / b;
        case C_Abs:   is_real = true; return Z(xp::abs(a), T(0.0));
        case C_Conj:  return xp::conj(a);
        case C_Sqrt:  return xp::sqrt(a);
        case C_Exp:   return xp::exp(a);
        case C_Log:   return xp::log(a);
        case C_Log10: return xp::log10(a);
        case C_Sin:   return xp::sin(a);
        case C_Cos:   return xp::cos(a);
        case C_Tan:   return xp::tan(a);
        case C_Asin:  return xp::asin(a);
        case C_Acos:  return xp::acos(a);
        case C_Atan:  return xp::atan(a);
        case C_Sinh:  return xp::sinh(a);
        case C_Cosh:  return xp::cosh(a);
        case C_Tanh:  return xp::tanh(a);
        case C_Asinh: return xp::asinh(a);
        case C_Acosh: return xp::acosh(a);
        case C_Atanh: return xp::atanh(a);
        case C_Pow:   return xp::pow(a, b);
        case C_Polar: return xp::polar(a.re, a.im);
    }
    return a;
}

// ===========================================================================
// MPC oracle at 400 bits.  Same op semantics as reference_complex().
// ===========================================================================
static bool mpc_reference(int id, mpc_t out, double are, double aim, double bre, double bim) {
    mpc_t a, b;
    mpc_init2(a, kPrec); mpc_init2(b, kPrec);
    mpc_set_d_d(a, are, aim, MPC_RNDNN);
    mpc_set_d_d(b, bre, bim, MPC_RNDNN);
    bool ok = true;
    switch (id) {
        case C_Add:   mpc_add(out, a, b, MPC_RNDNN); break;
        case C_Sub:   mpc_sub(out, a, b, MPC_RNDNN); break;
        case C_Mul:   mpc_mul(out, a, b, MPC_RNDNN); break;
        case C_Div:   mpc_div(out, a, b, MPC_RNDNN); break;
        case C_Abs: { mpfr_t m; mpfr_init2(m, kPrec); mpc_abs(m, a, MPFR_RNDN);
                      mpc_set_fr(out, m, MPC_RNDNN); mpfr_clear(m); break; }
        case C_Conj:  mpc_conj(out, a, MPC_RNDNN); break;
        case C_Sqrt:  mpc_sqrt(out, a, MPC_RNDNN); break;
        case C_Exp:   mpc_exp(out, a, MPC_RNDNN); break;
        case C_Log:   mpc_log(out, a, MPC_RNDNN); break;
        case C_Log10: mpc_log10(out, a, MPC_RNDNN); break;
        case C_Sin:   mpc_sin(out, a, MPC_RNDNN); break;
        case C_Cos:   mpc_cos(out, a, MPC_RNDNN); break;
        case C_Tan:   mpc_tan(out, a, MPC_RNDNN); break;
        case C_Asin:  mpc_asin(out, a, MPC_RNDNN); break;
        case C_Acos:  mpc_acos(out, a, MPC_RNDNN); break;
        case C_Atan:  mpc_atan(out, a, MPC_RNDNN); break;
        case C_Sinh:  mpc_sinh(out, a, MPC_RNDNN); break;
        case C_Cosh:  mpc_cosh(out, a, MPC_RNDNN); break;
        case C_Tanh:  mpc_tanh(out, a, MPC_RNDNN); break;
        case C_Asinh: mpc_asinh(out, a, MPC_RNDNN); break;
        case C_Acosh: mpc_acosh(out, a, MPC_RNDNN); break;
        case C_Atanh: mpc_atanh(out, a, MPC_RNDNN); break;
        case C_Pow:   mpc_pow(out, a, b, MPC_RNDNN); break;
        case C_Polar: { // polar(r, theta) = r * (cos t + i sin t)
                      mpfr_t r, t, c, s;
                      mpfr_init2(r, kPrec); mpfr_init2(t, kPrec);
                      mpfr_init2(c, kPrec); mpfr_init2(s, kPrec);
                      mpfr_set_d(r, are, MPFR_RNDN); mpfr_set_d(t, aim, MPFR_RNDN);
                      mpfr_sin_cos(s, c, t, MPFR_RNDN);
                      mpfr_mul(c, c, r, MPFR_RNDN); mpfr_mul(s, s, r, MPFR_RNDN);
                      mpc_set_fr_fr(out, c, s, MPC_RNDNN);
                      mpfr_clears(r, t, c, s, (mpfr_ptr)0); break; }
        default: ok = false;
    }
    mpc_clear(a); mpc_clear(b);
    return ok;
}

// sweep_accuracy.cpp:2578 -- modulus-relative, |got-ref|/|ref| * 2^p
template <class T>
static double cplx_ulps(const typename Fmt<T>::Z& got, mpc_srcptr ref) {
    mpfr_t gre, gim, ere, eim, num, den;
    mpfr_init2(gre, kPrec); mpfr_init2(gim, kPrec);
    mpfr_init2(ere, kPrec); mpfr_init2(eim, kPrec);
    mpfr_init2(num, kPrec); mpfr_init2(den, kPrec);
    to_mpfr(gre, got.re); to_mpfr(gim, got.im);
    double u;
    mpc_abs(den, ref, MPFR_RNDN);
    if (!mpfr_number_p(gre) || !mpfr_number_p(gim) || mpfr_zero_p(den) || !mpfr_number_p(den)) {
        u = -1.0;
    } else {
        mpfr_sub(ere, gre, mpc_realref(ref), MPFR_RNDN);
        mpfr_sub(eim, gim, mpc_imagref(ref), MPFR_RNDN);
        mpfr_hypot(num, ere, eim, MPFR_RNDN);
        mpfr_div(num, num, den, MPFR_RNDN);
        mpfr_mul_2si(num, num, Fmt<T>::p, MPFR_RNDN);
        u = mpfr_get_d(num, MPFR_RNDN);
    }
    mpfr_clears(gre, gim, ere, eim, num, den, (mpfr_ptr)0);
    return u;
}

// |got - ref| in absolute terms, as a double
template <class T>
static double cplx_abserr(const typename Fmt<T>::Z& got, mpc_srcptr ref) {
    mpfr_t gre, gim, ere, eim, num;
    mpfr_init2(gre, kPrec); mpfr_init2(gim, kPrec);
    mpfr_init2(ere, kPrec); mpfr_init2(eim, kPrec); mpfr_init2(num, kPrec);
    to_mpfr(gre, got.re); to_mpfr(gim, got.im);
    mpfr_sub(ere, gre, mpc_realref(ref), MPFR_RNDN);
    mpfr_sub(eim, gim, mpc_imagref(ref), MPFR_RNDN);
    mpfr_hypot(num, ere, eim, MPFR_RNDN);
    double v = mpfr_get_d(num, MPFR_RNDN);
    mpfr_clears(gre, gim, ere, eim, num, (mpfr_ptr)0);
    return v;
}

// the FORMAT FLOOR for a complex result: each component greedily rounded
template <class T> static typename Fmt<T>::Z round_into(mpc_srcptr ref) {
    return typename Fmt<T>::Z(from_mpfr<T>(mpc_realref(ref)),
                              from_mpfr<T>(mpc_imagref(ref)));
}

// ===========================================================================
// evaluation helper: run one (op, point) under one policy
// ===========================================================================
template <class T>
struct Run { typename Fmt<T>::Z got; long calls, fires; };

template <class T>
static Run<T> run_point(int id, const std::vector<GridPoint>& grid, size_t i,
                        double bre, double bim, int policy) {
    using Z = typename Fmt<T>::Z;
    const T are(grid[i].re), aim(grid[i].im), bre_(bre), bim_(bim);
    const Z a(are, aim);
    const Z b(bre_, bim_);
    const int save = g_policy;
    g_policy = policy;
    g_calls = 0; g_fires = 0;
    bool is_real = false;
    Run<T> r;
    r.got = eval_c<T>(id, a, b, is_real);
    r.calls = g_calls; r.fires = g_fires;
    g_policy = save;
    return r;
}

// ===========================================================================
// --census: does the lift guard fire, per op x backend, over the whole grid?
//
// This is the sqrt precedent's decisive test, transposed.  For sqrt the
// evidence was "ops WITHOUT sqrt in their path did not move at all".  The
// equivalent here is stronger than a call-path question, because divide is on
// far more paths than the lift guard ever fires on: what matters is not
// whether an op CALLS divide but whether any of its divides is subnormal
// enough for the guard to fire.  An op whose guard never fires is
// bit-identical before and after by construction, so it MUST not move.
// ===========================================================================
template <class T>
static void census_backend(const std::vector<GridPoint>& grid) {
    for (int id = 0; id < C_COUNT; ++id) {
        Rng rng(stream_seed(kDefaultSeed, kComplex[id].name, 1u));
        long tot_calls = 0, tot_fires = 0, pts_fired = 0, pts_changed = 0;
        for (size_t i = 0; i < grid.size(); ++i) {
            double bre, bim;
            fill_complex_operands(id, i, grid, grid[i].re, grid[i].im, rng, bre, bim);
            Run<T> lo = run_point<T>(id, grid, i, bre, bim, P_PLAIN);
            Run<T> hi = run_point<T>(id, grid, i, bre, bim, P_LIFT);
            tot_calls += lo.calls;
            tot_fires += lo.fires;
            if (lo.fires) ++pts_fired;
            if (!same_value<T>(lo.got.re, hi.got.re) || !same_value<T>(lo.got.im, hi.got.im))
                ++pts_changed;
        }
        std::printf("CENSUS %-2s c %-9s calls %8ld  fires %7ld  pts_fired %5ld  pts_changed %5ld\n",
                    Fmt<T>::name(), kComplex[id].name, tot_calls, tot_fires, pts_fired, pts_changed);
    }
}

// ===========================================================================
// --points: the 28 regressed rows
// ===========================================================================
struct Target { const char* be; int id; int point; double csv_before, csv_after; };
static const Target kTargets[] = {
    // op, point, baseline ulps, post-lift ulps  (from validation/sweep/
    // sweep_baseline.csv.gz and /tmp/div_lift_k.csv, field 6)
    {"QF", C_Asin,   752, 0.0,          2.22045e+07},
    {"QF", C_Asin,   753, 0.0,          2.22045e+07},
    {"TF", C_Asin,   752, 0.0,          1.32349},
    {"TF", C_Asin,   753, 0.0,          1.32349},
    {"QF", C_Asinh, 1560, 0.0,          2.22045e+07},
    {"QF", C_Asinh, 1561, 0.0,          2.22045e+07},
    {"QF", C_Asinh, 1562, 0.0,          2.22045e+07},
    {"QF", C_Asinh, 1563, 0.0,          2.22045e+07},
    {"TF", C_Asinh, 1560, 0.0,          1.32349},
    {"TF", C_Asinh, 1561, 0.0,          1.32349},
    {"TF", C_Asinh, 1562, 0.0,          1.32349},
    {"TF", C_Asinh, 1563, 0.0,          1.32349},
    {"QF", C_Asin,   738, 1.35526e-04,  4.44103},
    {"QF", C_Asin,   739, 1.35526e-04,  4.44103},
    {"QF", C_Asinh, 1532, 1.35526e-04,  4.44103},
    {"QF", C_Asinh, 1533, 1.35526e-04,  4.44103},
    {"QF", C_Asinh, 1534, 1.35526e-04,  4.44103},
    {"QF", C_Asinh, 1535, 1.35526e-04,  4.44103},
    {"QF", C_Atan,  1254, 0.123369,     1.71082},
    {"QF", C_Atan,  1255, 0.123369,     1.71082},
    {"QF", C_Atan,  1318, 0.123369,     1.71082},
    {"QF", C_Atan,  1319, 0.123369,     1.71082},
    {"QF", C_Atanh,  614, 0.123369,     1.71082},
    {"QF", C_Atanh,  615, 0.123369,     1.71082},
    {"QF", C_Atanh,  870, 0.123369,     1.71082},
    {"QF", C_Atanh,  871, 0.123369,     1.71082},
    {"QF", C_Abs,     27, 0.0480213,    0.382715},
    {"QF", C_Sqrt,    11, 0.0309759,    0.192617},
    // The four rows above 1 ulp that a >3.73x filter misses but the monotone
    // gate does NOT: its factor is kNoiseFactor = 1.2589 (a tenth of a digit),
    // not 3.73.  Adding them here so --points/--traceall cover every complex
    // row the gate would actually call a regression.  (The fifth, QF r hypot
    // 36, is a REAL row: 16916.8 -> 60371 ulps against bound 188310.  It is
    // out of this probe's complex reach and is reported from the CSVs only.)
    {"QF", C_Sqrt,   764, 3.79651e+12,  5.55112e+12},
    {"QF", C_Sqrt,   765, 3.79651e+12,  5.55112e+12},
    {"TF", C_Sqrt,   764, 226290.0,     330873.0},
    {"TF", C_Sqrt,   765, 226290.0,     330873.0},
};
static const int kNTargets = (int)(sizeof(kTargets) / sizeof(kTargets[0]));

template <class T>
static void report_point(const Target& t, const std::vector<GridPoint>& grid) {
    if (std::strcmp(t.be, Fmt<T>::name()) != 0) return;
    const size_t i = (size_t)t.point;

    Rng rng(stream_seed(kDefaultSeed, kComplex[t.id].name, 1u));
    double bre = 0, bim = 0;
    for (size_t k = 0; k <= i; ++k)
        fill_complex_operands(t.id, k, grid, grid[k].re, grid[k].im, rng, bre, bim);

    mpc_t ref; mpc_init2(ref, kPrec);
    if (!mpc_reference(t.id, ref, grid[i].re, grid[i].im, bre, bim)) {
        std::printf("POINT %s c %s %d  NO ORACLE\n", t.be, kComplex[t.id].name, t.point);
        mpc_clear(ref); return;
    }

    std::printf("\n=== %s c %-6s point %-5d  z = (%.17g, %.17g)   csv %.6g -> %.6g\n",
                t.be, kComplex[t.id].name, t.point, grid[i].re, grid[i].im,
                t.csv_before, t.csv_after);
    {
        char* s = mpc_get_str(10, 25, ref, MPC_RNDNN);
        std::printf("    ref(400b) = %s\n", s);
        mpc_free_str(s);
    }

    // format floor: the best any implementation could return
    typename Fmt<T>::Z fl = round_into<T>(ref);
    const double fl_u = cplx_ulps<T>(fl, ref);
    const double fl_e = cplx_abserr<T>(fl, ref);

    const int pol[5] = {P_PLAIN, P_LIFT, P_FLOOR, P_POISON, P_WRECK};
    typename Fmt<T>::Z got[5];
    double uu[5], ee[5];
    for (int k = 0; k < 5; ++k) {
        Run<T> r = run_point<T>(t.id, grid, i, bre, bim, pol[k]);
        got[k] = r.got;
        const double u = cplx_ulps<T>(r.got, ref);
        const double e = cplx_abserr<T>(r.got, ref);
        uu[k] = u; ee[k] = e;
        std::printf("    %-6s ulps %-14.6g abserr %-13.6g  e/e_floor %-10.4g  divides %ld fires %ld\n",
                    kPolicyName[pol[k]], u, e, (fl_e > 0 ? e / fl_e : (e == 0 ? 1.0 : 1e300)),
                    r.calls, r.fires);
        if (pol[k] == P_PLAIN || pol[k] == P_LIFT) {
            std::printf("           re words:");
            for (int w = 0; w < Fmt<T>::n; ++w) std::printf(" %.9g", (double)Fmt<T>::limb(r.got.re, w));
            std::printf("\n           im words:");
            for (int w = 0; w < Fmt<T>::n; ++w) std::printf(" %.9g", (double)Fmt<T>::limb(r.got.im, w));
            std::printf("\n");
        }
    }
    std::printf("    %-6s ulps %-14.6g abserr %-13.6g  (unbeatable in this format)\n",
                "FLOOR*", fl_u, fl_e);
    // THE DECIDING COMPARISON.  If the shipped lift and an exactly-rounded
    // 400-bit divide drive the caller to the SAME value, then no divide could
    // have avoided the move and the lift is not what broke it.
    const bool lf = same_value<T>(got[1].re, got[2].re) && same_value<T>(got[1].im, got[2].im);
    std::printf("    VERDICT lift %s exact-divide   |   wreck moved: %s\n",
                lf ? "== " : "!= ", (uu[4] != uu[0]) ? "yes" : "NO (hook not on this path!)");
    // "!=" alone is not a finding.  What matters is HOW MUCH of the caller's
    // move an exactly-rounded divide takes back: 1.0 would mean the lift is
    // simply not accurate enough and a better divide fixes the row, 0.0 means
    // a perfect divide lands exactly where the lift did.  Printed at more
    // digits than the table above, because on the sqrt rows the two arms agree
    // to six significant figures and the difference is only visible past them.
    if (!lf) {
        const double move = uu[1] - uu[0];
        std::printf("           exact-divide ulps %.10g vs lift %.10g "
                    "(recovers %.4g%% of the %.10g-ulp move)\n",
                    uu[2], uu[1],
                    (move != 0.0) ? 100.0 * (uu[1] - uu[2]) / move : 0.0, move);
    }
    // the quantum of the LAST word of the result -- the granularity below
    // which no expansion in this format can move at all
    {
        mpfr_t m; mpfr_init2(m, kPrec); mpc_abs(m, ref, MPFR_RNDN);
        const double mod = mpfr_get_d(m, MPFR_RNDN);
        mpfr_clear(m);
        const int lb = Fmt<T>::limb_bits;
        double lastword = std::ldexp(mod, -(Fmt<T>::n - 1) * lb);
        double quantum  = std::ldexp(lastword, -lb);
        const double tiny = (Fmt<T>::limb_bits == 24) ? 1.4012984643248171e-45   // FLT_TRUE_MIN
                                                      : 4.9406564584124654e-324; // DBL_TRUE_MIN
        if (quantum < tiny) quantum = tiny;
        std::printf("    quantum of the last word = %.6g  (>= subnormal floor %.6g)\n",
                    quantum, tiny);
        // RESOLUTION OF THE METRIC vs RESOLUTION OF THE FORMAT.
        // One ulp at this modulus is |ref| * 2^-p.  One representable step is
        // `quantum`.  Where quantum > 1 ulp the metric is asking for a
        // precision the format cannot express: the row can only ever read 0 or
        // some multiple of (quantum / ulp), with nothing available in between.
        // Printing the measured move in QUANTA rather than in ulps says how
        // big the move actually was in the only unit the format has.
        const double ulp_abs = std::ldexp(mod, -Fmt<T>::p);
        const double per_q   = (ulp_abs > 0.0) ? quantum / ulp_abs : 0.0;
        std::printf("    1 ulp = %.6g abs;  1 representable step = %.6g ulps;  "
                    "the lift's move = %.4g steps\n",
                    ulp_abs, per_q, (quantum > 0.0) ? (ee[1] / quantum) : 0.0);
    }
    mpc_clear(ref);
}

// ===========================================================================
// --verify: does the reproduction agree with the committed CSVs?
// ===========================================================================
template <class T>
static void verify_point(const Target& t, const std::vector<GridPoint>& grid,
                         int& n_ok, int& n_bad) {
    if (std::strcmp(t.be, Fmt<T>::name()) != 0) return;
    const size_t i = (size_t)t.point;
    Rng rng(stream_seed(kDefaultSeed, kComplex[t.id].name, 1u));
    double bre = 0, bim = 0;
    for (size_t k = 0; k <= i; ++k)
        fill_complex_operands(t.id, k, grid, grid[k].re, grid[k].im, rng, bre, bim);
    mpc_t ref; mpc_init2(ref, kPrec);
    if (!mpc_reference(t.id, ref, grid[i].re, grid[i].im, bre, bim)) { mpc_clear(ref); return; }
    Run<T> lo = run_point<T>(t.id, grid, i, bre, bim, P_PLAIN);
    Run<T> hi = run_point<T>(t.id, grid, i, bre, bim, P_LIFT);
    const double ulo = cplx_ulps<T>(lo.got, ref), uhi = cplx_ulps<T>(hi.got, ref);
    // "agree" = within 1% relative, or both below 1e-3 ulps (where the CSV's
    // __complex128 oracle and this 400-bit one are no longer comparable)
    auto agree = [](double got, double csv) {
        if (csv == 0.0) return got < 1e-3;
        if (got < 1e-3 && csv < 1e-3) return true;
        return std::fabs(got - csv) <= 0.01 * csv;
    };
    const bool ok = agree(ulo, t.csv_before) && agree(uhi, t.csv_after);
    std::printf("VERIFY %-2s c %-6s %-5d  plain %-13.6g (csv %-13.6g)  lift %-13.6g (csv %-13.6g)  %s\n",
                t.be, kComplex[t.id].name, t.point, ulo, t.csv_before, uhi, t.csv_after,
                ok ? "OK" : "MISMATCH");
    if (ok) ++n_ok; else ++n_bad;
    mpc_clear(ref);
}

// ===========================================================================
// --trace: every internal divide at one (backend, op, point)
// ===========================================================================
static int g_agg_better = 0, g_agg_worse = 0, g_agg_same = 0, g_agg_total = 0;

// Negative control for the trace scorer itself.  With this set, the plain and
// lifted quotients are scored in the OPPOSITE slots, so every LIFT BETTER must
// come back LIFT WORSE.  A scorer that can only ever say "better" is not
// evidence of anything; this makes it possible for --traceall to fail.
static bool g_trace_swap = false;

static void score_trace(bool quiet = false) {
    if (!quiet)
        std::printf("  %-3s %-5s %-14s %-14s %-14s %-14s %s\n",
                    "be", "fired", "|a|", "|b|", "ulps(plain)", "ulps(lift)", "verdict");
    int n_lift_worse = 0, n_lift_better = 0, n_same = 0;
    for (size_t k = 0; k < g_recs.size(); ++k) {
        const DivRec& r = g_recs[k];
        mpfr_t ma, mb, q, g1, g2, d, rr;
        mpfr_init2(ma, kPrec); mpfr_init2(mb, kPrec); mpfr_init2(q, kPrec);
        mpfr_init2(g1, kPrec); mpfr_init2(g2, kPrec);
        mpfr_init2(d, kPrec);  mpfr_init2(rr, kPrec);
        mpfr_set_zero(ma, 1); mpfr_set_zero(mb, 1);
        mpfr_set_zero(g1, 1); mpfr_set_zero(g2, 1);
        mpfr_t t; mpfr_init2(t, kPrec);
        for (int w = 0; w < r.n; ++w) {
            mpfr_set_d(t, r.a[w],    MPFR_RNDN); mpfr_add(ma, ma, t, MPFR_RNDN);
            mpfr_set_d(t, r.b[w],    MPFR_RNDN); mpfr_add(mb, mb, t, MPFR_RNDN);
            mpfr_set_d(t, r.core[w], MPFR_RNDN); mpfr_add(g1, g1, t, MPFR_RNDN);
            mpfr_set_d(t, r.lift[w], MPFR_RNDN); mpfr_add(g2, g2, t, MPFR_RNDN);
        }
        mpfr_clear(t);
        if (g_trace_swap) mpfr_swap(g1, g2);
        double u1 = -1, u2 = -1;
        if (!mpfr_zero_p(mb) && mpfr_number_p(ma) && mpfr_number_p(mb)) {
            mpfr_div(q, ma, mb, MPFR_RNDN);
            if (!mpfr_zero_p(q)) {
                mpfr_abs(rr, q, MPFR_RNDN);
                mpfr_mul_2si(rr, rr, -r.p, MPFR_RNDN);
                mpfr_sub(d, g1, q, MPFR_RNDN); mpfr_abs(d, d, MPFR_RNDN);
                mpfr_div(d, d, rr, MPFR_RNDN); u1 = mpfr_get_d(d, MPFR_RNDN);
                mpfr_sub(d, g2, q, MPFR_RNDN); mpfr_abs(d, d, MPFR_RNDN);
                mpfr_div(d, d, rr, MPFR_RNDN); u2 = mpfr_get_d(d, MPFR_RNDN);
            }
        }
        const char* verdict = "-";
        if (u1 >= 0 && u2 >= 0) {
            if (u2 < u1 * 0.999)      { verdict = "LIFT BETTER"; ++n_lift_better; }
            else if (u2 > u1 * 1.001) { verdict = "LIFT WORSE";  ++n_lift_worse; }
            else                      { verdict = "same";        ++n_same; }
        }
        const bool worse = (u1 >= 0 && u2 >= 0 && u2 > u1 * 1.001);
        if ((!quiet && (r.fired || (u1 >= 0 && u2 >= 0 && std::fabs(u1 - u2) > 1e-12))) ||
            (quiet && worse))
            std::printf("  %-3s %-5d %-14.6g %-14.6g %-14.6g %-14.6g %s\n",
                        r.be, r.fired, mpfr_get_d(ma, MPFR_RNDN), mpfr_get_d(mb, MPFR_RNDN),
                        u1, u2, verdict);
        mpfr_clears(ma, mb, q, g1, g2, d, rr, (mpfr_ptr)0);
    }
    g_agg_better += n_lift_better; g_agg_worse += n_lift_worse;
    g_agg_same   += n_same;        g_agg_total += (int)g_recs.size();
    if (!quiet)
        std::printf("  TRACE SUMMARY: %zu divides, lift better %d, lift worse %d, identical %d\n",
                    g_recs.size(), n_lift_better, n_lift_worse, n_same);
}

template <class T>
static void trace_point(int id, int point, const std::vector<GridPoint>& grid,
                        bool quiet = false) {
    Rng rng(stream_seed(kDefaultSeed, kComplex[id].name, 1u));
    double bre = 0, bim = 0;
    for (int k = 0; k <= point; ++k)
        fill_complex_operands(id, (size_t)k, grid, grid[k].re, grid[k].im, rng, bre, bim);
    g_recs.clear();
    g_trace = true;
    run_point<T>(id, grid, (size_t)point, bre, bim, P_LIFT);
    g_trace = false;
    if (!quiet)
        std::printf("TRACE %s c %s point %d  z = (%.17g, %.17g)\n",
                    Fmt<T>::name(), kComplex[id].name, point, grid[point].re, grid[point].im);
    else
        std::printf("%-3s c %-6s %-5d  divides %zu\n",
                    Fmt<T>::name(), kComplex[id].name, point, g_recs.size());
    score_trace(quiet);
}

// --traceall: the same divide-by-divide scoring, over EVERY one of the 28 rows.
// The five-point sample answered "is the lifted quotient closer to the true
// quotient at these operands"; this answers it exhaustively, so a single
// LIFT WORSE anywhere in the 28 would surface.  Rows that ARE worse print;
// everything else is counted only.
template <class T>
static void traceall_point(const Target& t, const std::vector<GridPoint>& grid) {
    if (std::strcmp(t.be, Fmt<T>::name()) != 0) return;
    trace_point<T>(t.id, t.point, grid, true);
}

// ---------------------------------------------------------------------------
// --sites: which call site does each of the 28 rows divide at?
//
// Prints one SITE line per (row, address, fired) with a count.  Addresses are
// resolved by scripts/probe_div_sites.sh via `addr2line -i`, which unwinds the
// inline chain and so names the qf_complex.hpp line rather than the hook.
// ---------------------------------------------------------------------------
template <class T>
static void sites_point(const Target& t, const std::vector<GridPoint>& grid) {
    if (std::strcmp(t.be, Fmt<T>::name()) != 0) return;
    Rng rng(stream_seed(kDefaultSeed, kComplex[t.id].name, 1u));
    double bre = 0, bim = 0;
    for (size_t k = 0; k <= (size_t)t.point; ++k)
        fill_complex_operands(t.id, k, grid, grid[k].re, grid[k].im, rng, bre, bim);
    g_recs.clear();
    g_trace = true;
    run_point<T>(t.id, grid, (size_t)t.point, bre, bim, P_LIFT);
    g_trace = false;
    // (site, fired) -> count, in first-seen order
    std::vector<std::pair<std::pair<void*, int>, int> > tally;
    for (size_t k = 0; k < g_recs.size(); ++k) {
        std::pair<void*, int> key(g_recs[k].site, g_recs[k].fired);
        size_t j = 0;
        for (; j < tally.size(); ++j) if (tally[j].first == key) break;
        if (j == tally.size()) tally.push_back(std::make_pair(key, 0));
        ++tally[j].second;
    }
    for (size_t j = 0; j < tally.size(); ++j)
        std::printf("SITE %s c %s %d  %p fired=%d count=%d\n",
                    Fmt<T>::name(), kComplex[t.id].name, t.point,
                    tally[j].first.first, tally[j].first.second, tally[j].second);
}

// ===========================================================================
int main(int argc, char** argv) {
    const std::vector<GridPoint> grid = build_complex_grid();
    std::string mode = (argc > 1) ? argv[1] : "--points";

    std::printf("# probe_div_downstream   grid %zu points   oracle MPFR/MPC %ld bits\n",
                grid.size(), (long)kPrec);
    if (grid.size() != 1780) {
        std::printf("!! grid size %zu != 1780 (max complex point index in the baseline + 1)\n",
                    grid.size());
        return 2;
    }

    if (mode == "--census") {
        census_backend<QuadFloat>(grid);
        census_backend<TripleFloat>(grid);
        census_backend<FloatFloat>(grid);
        census_backend<DoubleDouble>(grid);
        return 0;
    }
    if (mode == "--verify") {
        int ok = 0, bad = 0;
        for (int i = 0; i < kNTargets; ++i) {
            verify_point<QuadFloat>(kTargets[i], grid, ok, bad);
            verify_point<TripleFloat>(kTargets[i], grid, ok, bad);
        }
        std::printf("VERIFY: %d agree, %d mismatch\n", ok, bad);
        return bad ? 1 : 0;
    }
    if (mode == "--sites") {
#ifndef PROBE_SITES
        std::printf("!! --sites needs the PROBE_SITES build; "
                    "run scripts/probe_div_sites.sh\n");
        return 2;
#else
        for (int i = 0; i < kNTargets; ++i) {
            sites_point<QuadFloat>(kTargets[i], grid);
            sites_point<TripleFloat>(kTargets[i], grid);
        }
        return 0;
#endif
    }
    if (mode == "--traceall" || mode == "--traceall-control") {
        g_trace_swap = (mode == "--traceall-control");
        std::printf("# every internal divide at all %d target rows, scored against the\n"
                    "# exact 400-bit quotient at the same (a,b).  LIFT WORSE rows print.%s\n",
                    kNTargets, g_trace_swap ? "  [CONTROL: slots swapped]" : "");
        for (int i = 0; i < kNTargets; ++i) {
            traceall_point<QuadFloat>(kTargets[i], grid);
            traceall_point<TripleFloat>(kTargets[i], grid);
        }
        // "unscored" = the divisor or the true quotient was zero, so there is no
        // relative-ulps question to ask.  Reported, never silently folded in.
        const int unscored = g_agg_total - g_agg_better - g_agg_worse - g_agg_same;
        std::printf("TRACEALL%s AGGREGATE over %d rows: %d divides, "
                    "lift better %d, LIFT WORSE %d, identical %d, unscored %d\n",
                    g_trace_swap ? "-CONTROL" : "", kNTargets, g_agg_total,
                    g_agg_better, g_agg_worse, g_agg_same, unscored);
        if (g_trace_swap)
            return g_agg_worse > 0 ? 0 : 1;   // control must produce WORSE rows
        return g_agg_worse ? 1 : 0;
    }
    if (mode == "--trace") {
        if (argc < 5) { std::printf("usage: --trace <QF|TF> <op> <point>\n"); return 2; }
        int id = -1;
        for (int k = 0; k < C_COUNT; ++k)
            if (std::strcmp(kComplex[k].name, argv[3]) == 0) id = k;
        if (id < 0) { std::printf("unknown op %s\n", argv[3]); return 2; }
        const int pt = std::atoi(argv[4]);
        if (std::strcmp(argv[2], "QF") == 0)      trace_point<QuadFloat>(id, pt, grid);
        else if (std::strcmp(argv[2], "TF") == 0) trace_point<TripleFloat>(id, pt, grid);
        else { std::printf("unknown backend %s\n", argv[2]); return 2; }
        return 0;
    }
    // --points
    for (int i = 0; i < kNTargets; ++i) {
        report_point<QuadFloat>(kTargets[i], grid);
        report_point<TripleFloat>(kTargets[i], grid);
    }
    return 0;
}
