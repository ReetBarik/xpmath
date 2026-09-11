// probe_sqrt_iter.cpp — is DD/FF `sqrt` at the two-word floor, or above it?
//
// WHY THIS EXISTS.  In the committed baseline the four backends' real sqrt read
// DD 414 / FF 436 / TF 40 / QF 60 rows above 1 ulp out of ~1690 scored each,
// against the tightest bound in the library (1.5, because R_Sqrt is charged no
// intermediate at all).  Two stories fit that: (a) DD and FF are at the floor a
// two-word expansion can reach and TF/QF have slack under their own p, or (b)
// DD/FF run a WORSE ALGORITHM than TF/QF and the split is an implementation
// defect.  Those have opposite consequences and no amount of staring at the
// bound distinguishes them.
//
// The two implementations are not the same algorithm:
//
//   DD (dd_math.hpp:499) and FF (ff_math.hpp:603) — one Karp correction:
//       t1 = 1 / sqrt(a.hi);        // HARDWARE reciprocal root, one word
//       t2 = a.hi * t1;             // ~1 word of root
//       s1 = a - two_prod(t2, t2);  // exact residual, full format
//       t3 = 0.5 * s1.hi * t1;      // <-- correction, ONE word, seeded by t1
//       return t2 + t3;
//
//   TF (tf_math.hpp:714) and QF (qf_math.hpp:795) — Heron in FULL format
//   arithmetic, iterated until the correction falls under the format's own eps.
//
// The claimed mechanism is the SEED.  `t3` is standing in for s1/(2*t2), but the
// code forms it as 0.5*s1.hi*t1, and t1 = fl(1/fl(sqrt(a.hi))) is ~4u away from
// the 1/t2 it substitutes for (u = 2^-53 on DD).  That 4u multiplies a
// correction of relative size 2^-53, so it lands ~4*2^-106 into the answer —
// i.e. a handful of ulps at p = 106, on every call, with no second iteration
// downstream to absorb it.  If that is right, replacing the multiply by t1 with
// a single correctly-rounded divide by 2*t2 should remove almost all of it.
//
// This probe tests that by ablation, in C++, against MPFR at 400 bits, under
// the toolchain of record.  It deliberately reuses probe_arith_floor.cpp's
// Fmt<T> / to_mpfr / from_mpfr / ulps<T> / Acc / --poison verbatim, so the two
// probes are scoring on the same instrument.
//
// ARMS (all four backends where the shape applies; the Karp arms are 2-limb
// only, since TF/QF have no Karp step to ablate):
//
//   lib          xp::sqrt() as shipped.  On DD/FF this must come out BIT-EQUAL
//                to `karp0`; the probe checks that and prints the mismatch
//                count, because a silent divergence would invalidate every
//                other arm.  On TF/QF it is the Heron control.
//   karp0        the shipped DD/FF body, retyped here so it can be perturbed.
//   karp1        t3 = s1.hi / (2*t2).  ONE bare FP divide, one fewer multiply.
//   karp2        seed refined once, t1 <- t1*(1.5 - 0.5*a.hi*t1*t1), rest as
//                shipped.  Ceiling: t1 is a single word, so refinement can
//                never bring it closer than 1u to 1/t2.
//   karp12       karp2's refined seed AND karp1's divide.  Both changes at once.
//   karp6        DROP the reciprocal-root seed entirely: t2 = sqrt(a.hi)
//                directly (correctly rounded, one hardware sqrt in place of
//                sqrt+divide+multiply), t3 = s1.hi / (2*t2).  Added AFTER the
//                first pass, because karp1 == karp5 said the seed was not the
//                mechanism and t2's own accuracy was.
//   karp8        karp6, but the correction is kept as a full two-word value:
//                divide_scalar(s1, 2*t2) instead of s1.hi/(2*t2).  One hardware
//                divide, no format-level divide.  Also added after the fact.
//   karp3        shipped, then ONE Heron step in full format arithmetic.
//                Costs a format-level divide.
//   heron4       Heron to convergence at the backend's own width — the TF/QF
//                algorithm, run at DD/FF width.  On TF/QF this is the shipped
//                algorithm and must track `lib`.
//   karp5        t3 formed from the EXACT 1/sqrt(a) in MPFR, rounded once into
//                a single word.  Diagnostic, unreachable in a header: it prices
//                the seed error and nothing else.
//   repr         sqrt(a) at 400 bits merely ROUNDED into the format.  No
//                algorithm can beat this; <= 0.5 by construction.
//
// OPERANDS.  Two sources, reported separately:
//
//   grid     THE SWEEP'S OWN OPERANDS.  R_Sqrt is unary, so the sweep's operand
//            is exactly fabs(re) of each real grid point (sweep_accuracy.cpp:604
//            `case R_Sqrt: a = fabs(a)`), and the operand handed to the
//            primitive is S(a) (`:2352`).  That is reconstructible from
//            validation/sweep/sweep_grid.csv without touching the generator's
//            RNG, so none of the operand-reconstruction hazard at
//            sweep_accuracy.cpp:2470 applies here.  The reference is
//            sqrt(exact limb sum of the STORED operand), matching the sweep's
//            f(x_stored) convention (b7c4b64, `:2378`).  This arm's `lib`
//            column is therefore directly comparable to the committed CSV, and
//            reproducing DD 414 / FF 436 / TF 40 / QF 60 is what licenses the
//            other arms.
//   generic  probe_arith_floor's own operand draw: a random 400-bit value in
//            [1,2) times a random power of two in 2^-20..2^20, rounded into the
//            format.  Bulk-range only, no subnormal limbs — so it speaks to the
//            ordinary range and says nothing about either tail.
//
// Build (needs MPFR; not part of the CMake build):
//   module use /soft/modulefiles && module load gcc/13.3.0
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_sqrt_iter.cpp -o /tmp/probe_sqrt_iter -lmpfr -lgmp
//   /tmp/probe_sqrt_iter
//   /tmp/probe_sqrt_iter --poison    # negative control
//
// POISON.  --poison scores every arm against a reference computed at 24 bits
// instead of 400.  Every arm on every backend must go to 100% above 1 ulp.
// `repr` is the load-bearing one: it reads <= 0.5 BY CONSTRUCTION, so it is the
// cell most likely to stay quiet if the oracle is not really being consulted.
//
// ===========================================================================
// WHAT IT MEASURED.  gcc 13.3.0, MPFR 400 bits, 2026-09-11.
//
// INSTRUMENT VALIDATED FIRST, three ways, because nothing below means anything
// otherwise:
//   * `lib` on the grid source reproduces the committed baseline CSV row for
//     row: DD 414 bad / median 0.2389, FF 436 / 0.3658 (p95 2.5297, max 4.5108),
//     TF 40 / 0.0645, QF 60 / 0.0259.
//   * `karp0` (the retyped body every perturbation is built from) is BIT-EQUAL
//     to shipped xp::sqrt on DD and FF, 0 mismatches, both operand sources.
//   * `heron4` at TF/QF width tracks `lib` there, as it must — it is the
//     shipped algorithm.
//   --poison: 100.00% above 1 ulp on every arm and every backend on the generic
//   source, `repr` included.  On the grid source it reads 97.76% (DD) with
//   exactly 38 of 1698 rows quiet; those 38 are precisely the grid points whose
//   true root is representable in 24 bits (0.25, 1.0, 2.25, 4.0, 100.0, 1e4,
//   ... — enumerated independently from sweep_grid.csv), where the poisoned
//   reference is genuinely exact.  Control is sound.
//
// THE PLAN'S HYPOTHESIS ABOVE ("the claimed mechanism is the SEED") IS REFUTED.
// karp1 (plain divide, no exact seed) and karp5 (the EXACT 1/sqrt(a) from MPFR)
// are indistinguishable — DD generic 13.63% / max 4.793 vs 13.61% / max 4.546.
// An infinitely good seed buys nothing over one divide, so the seed is not what
// is left.  The Python replica that motivated this plan claimed the exact-seed
// arm reached 0.01% / max 1.075; measured here it is 13.61% / max 4.546, a
// >1000x discrepancy in rate.  The replica's "+1 Newton takes DD 414 -> 4,
// FF 436 -> 2" is likewise wrong: karp3 measures DD 414 -> 20, FF 436 -> 20.
//
// THE ACTUAL CEILING is Newton's QUADRATIC TRUNCATION term.  One correction
// step applied at t2 leaves delta^2/(2*sqrt(a)) with delta = t2 - sqrt(a).  The
// shipped t2 = a.hi * fl(1/fl(sqrt(a.hi))) carries ~3u of error (1.5u from the
// reciprocal root, 1u from the multiply, 0.5u from a.hi standing in for a), so
// the residue is ~4.5u^2 — about 4.5 ulps at p = 106, which is exactly the max
// observed.  Only a better t2, or a second iteration, can go below it.  That is
// what karp6/karp8 do, and why they were added: t2 = sqrt(a.hi) is correctly
// rounded (delta <= 0.5u), which cuts the quadratic term ~36x.
//
// ULPS, grid = the sweep's own operands, generic = 20000 random draws:
//
//   source            arm      DD >1ulp / max        FF >1ulp / max
//   grid full         lib      414 / 7.192e15        436 / 4.511
//   grid full         karp8      2 / 5.546e15          2 / 1.064
//   grid full         karp3     20 / 8.232e14         20 / 1.332
//   grid full         heron4    28 / 1.304e15         12 / 1.289
//   grid bulk 1e-10..1e10, n = 1510:
//   grid bulk         lib      370 / 4.294           400 / 4.511
//   grid bulk         karp8      0 / 0.7955            0 / 0.9799
//   grid bulk         karp6      0 / 0.7955           18 / 2.155
//   grid bulk         karp3     16 / 1.231           16 / 1.332
//   grid bulk         heron4    24 / 1.547           10 / 1.289
//   grid bulk         repr       0 / 0.5               0 / 0.4288
//   generic n = 20000:
//   generic           lib      30.30% / 6.702        29.84% / 6.173
//   generic           karp8     0.42% / 1.665         0.53% / 1.46
//   generic           heron4    1.80% / 2.104         1.56% / 2.887
//   generic           karp3     3.19% / 2.662         3.04% / 2.99
//
// The grid-full maxima ~1e15 are the two subnormal-tail rows, where the root of
// a value with one significant bit left is representationally unresolvable;
// `repr` shows the same thing, so they are inherent and not the algorithm's.
//
// TF/QF grid bulk for scale: TF lib 12 bad / max 1.188, QF lib 4 / 1.226.  So
// karp8 on DD/FF (0 bad) lands BELOW both, not merely level with them.
//
// karp6 vs karp8 on DD is a tie on the grid and not on FF (0 vs 18 bulk rows)
// because the grid's operands are doubles: DoubleDouble holds them exactly
// (a.lo == 0, so sqrt(a.hi) is already the correctly rounded root of the WHOLE
// value) while FloatFloat must split them.  On generic full-width operands the
// two agree (DD 3.29% vs FF 2.94%).  karp8 keeps s1.lo and so is width-neutral.
//
// CONCLUSION.  Phase 1's stop condition — "sqrt has headroom iff some arm drops
// the DD/FF >1 ulp rate to TF/QF's level (<= 3%) with max <= 2 ulps" — is MET,
// by karp8, karp6, karp3 and heron4.  On the tie-break ("cheapest arm wins")
// karp8 is the most accurate arm on every operand source, and the cheapest of
// the arms that meet the condition (it needs no full format-level divide, only
// divide_scalar).  It is NOT cheaper than what ships, which an earlier revision
// of this comment claimed on an operation count alone: measured afterwards with
// kokkos_ep_bench_cost it is about 2x SLOWER (DD 0.06x -> 0.13x f128, FF 7.0x ->
// 14.3x FP64), because divide_scalar is a two-word division with a renormalize
// where the shipped correction was one multiply.  Accuracy per unit cost is a
// separate question from accuracy, and this probe only measures accuracy.
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
// format traits — verbatim from probe_arith_floor.cpp
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
// the arms
// ---------------------------------------------------------------------------

// The stage every Karp variant shares: seed, root, exact residual.  `refine`
// runs one Newton step on the seed itself before the root is formed.
template <class T> struct KarpStage {
    using L = typename Fmt<T>::Limb;
    L t1, t2;
    T s1;
    KarpStage(const T& a, bool refine) {
        const L a0 = Fmt<T>::limb(a, 0);
        t1 = L(1) / detail::sqrt(a0);
        if (refine) t1 = t1 * (L(1.5) - L(0.5) * a0 * t1 * t1);
        t2 = a0 * t1;
        const T s0 = two_prod(t2, t2);       // exact
        s1 = subtract(a, s0);
    }
};

// arm karp0 — the shipped DD/FF body, retyped.
template <class T> static T arm_karp0(const T& a) {
    using L = typename Fmt<T>::Limb;
    KarpStage<T> k(a, false);
    const L t3 = L(0.5) * Fmt<T>::limb(k.s1, 0) * k.t1;
    return add(T(k.t2), T(t3));
}

// arm karp1 — the same, with the seed removed from the correction.
template <class T> static T arm_karp1(const T& a) {
    using L = typename Fmt<T>::Limb;
    KarpStage<T> k(a, false);
    const L t3 = Fmt<T>::limb(k.s1, 0) / (L(2) * k.t2);
    return add(T(k.t2), T(t3));
}

// arm karp2 — shipped, with one Newton refinement of the seed.
template <class T> static T arm_karp2(const T& a) {
    using L = typename Fmt<T>::Limb;
    KarpStage<T> k(a, true);
    const L t3 = L(0.5) * Fmt<T>::limb(k.s1, 0) * k.t1;
    return add(T(k.t2), T(t3));
}

// arm karp3 — shipped, then one Heron step in full format arithmetic.
template <class T> static T arm_karp3(const T& a) {
    using L = typename Fmt<T>::Limb;
    const T r0 = arm_karp0<T>(a);
    if (Fmt<T>::limb(r0, 0) == L(0)) return r0;
    return multiply_scalar(add(r0, divide(a, r0)), L(0.5));
}

// arm heron4 — the TF/QF algorithm at this backend's width.
template <class T> static T arm_heron4(const T& a) {
    using L = typename Fmt<T>::Limb;
    const L eps = (L)std::ldexp(1.0, -Fmt<T>::p);
    T r = T((L)detail::sqrt(Fmt<T>::limb(a, 0)));
    for (int i = 0; i < 10; ++i) {
        if (Fmt<T>::limb(r, 0) == L(0)) return r;
        const T ax = multiply_scalar(add(divide(a, r), r), L(0.5));
        const L d  = detail::fabs(Fmt<T>::limb(subtract(r, ax), 0));
        r = ax;
        if (d < detail::fabs(Fmt<T>::limb(r, 0)) * eps) return r;
    }
    return r;
}

// arm karp5 — shipped, with the seed replaced by the EXACT 1/sqrt(a),
// rounded once into a single word.  Prices the seed error alone.
template <class T> static T arm_karp5(const T& a, mpfr_srcptr ea) {
    using L = typename Fmt<T>::Limb;
    KarpStage<T> k(a, false);
    mpfr_t inv; mpfr_init2(inv, kPrec);
    mpfr_rec_sqrt(inv, ea, MPFR_RNDN);                       // exact 1/sqrt(a)
    mpfr_mul_d(inv, inv, 0.5 * (double)Fmt<T>::limb(k.s1, 0), MPFR_RNDN);
    const L t3 = (L)mpfr_get_d(inv, MPFR_RNDN);
    mpfr_clear(inv);
    return add(T(k.t2), T(t3));
}

// arm karp12 — BOTH single-word repairs at once: the seed refined, and the
// correction formed by division rather than by multiplying by the seed.
// Separates "the two fixes are the same fix" from "they compose".
template <class T> static T arm_karp12(const T& a) {
    using L = typename Fmt<T>::Limb;
    KarpStage<T> k(a, true);
    const L t3 = Fmt<T>::limb(k.s1, 0) / (L(2) * k.t2);
    return add(T(k.t2), T(t3));
}

// arm karp6 — the arm the karp1/karp5 result points at.  With the correction
// formed by division, what is left is Newton's own QUADRATIC truncation term,
// delta^2 / (2*sqrt(a)) with delta = t2 - sqrt(a).  The shipped t2 = a.hi * t1
// is ~3u off (1.5u from the reciprocal root, 1u from the multiply, 0.5u from
// a.hi standing in for a), so that term alone is ~4.5u^2 -- about 4.5 ulps at
// p = 106, which is what karp1 and karp5 both stall at.  Replacing the
// reciprocal-root-and-multiply with the hardware sqrt makes t2 correctly
// rounded, so delta drops to ~1u and the quadratic term to ~0.5 ulps.
//
// Strictly CHEAPER than shipped: one hardware sqrt and one hardware divide,
// against shipped's hardware sqrt, hardware divide (for the reciprocal) and
// two multiplies.  No format-level divide anywhere.
template <class T> static T arm_karp6(const T& a) {
    using L = typename Fmt<T>::Limb;
    const L t2 = detail::sqrt(Fmt<T>::limb(a, 0));   // correctly rounded
    const T s0 = two_prod(t2, t2);                   // exact
    const T s1 = subtract(a, s0);
    const L t3 = Fmt<T>::limb(s1, 0) / (L(2) * t2);
    return add(T(t2), T(t3));
}

// arm karp8 — karp6, with the correction kept as a FULL TWO-WORD value.
// karp6 still throws away s1.lo and still rounds t3 into one word; both cost
// about one ulp at p.  divide_scalar keeps the whole residual and returns two
// words for roughly the cost of one hardware divide plus a two_prod -- far
// cheaper than the format-level divide that karp3 and heron4 need.  What is
// left is Newton's quadratic term alone.
template <class T> static T arm_karp8(const T& a) {
    using L = typename Fmt<T>::Limb;
    const L t2 = detail::sqrt(Fmt<T>::limb(a, 0));   // correctly rounded
    const T s0 = two_prod(t2, t2);                   // exact
    const T s1 = subtract(a, s0);
    return add(T(t2), divide_scalar(s1, L(2) * t2));
}

// ---------------------------------------------------------------------------
// per-backend run over one operand set
// ---------------------------------------------------------------------------
// `bulk_only` restricts to |a| in [1e-10, 1e10].  Every backend's sqrt has a
// tail below its own subnormal-limb cliff where the answer degrades to bare
// hardware precision and NO algorithm helps (DD grid pts 563/567 at
// x = 2*2^-1074 carry one significant bit).  Those rows dominate the max on
// every arm including the ones that fix the bulk, so a max stated over the
// whole grid says nothing about the algorithm.  The bulk table is where the
// "max <= 2 ulps" half of the stop condition is legible.
template <class T>
static void run(const std::vector<T>& ops, const char* source, bool bulk_only) {
    mpfr_t ea, mr, hi;
    mpfr_init2(ea, kPrec); mpfr_init2(mr, g_ref_prec); mpfr_init2(hi, kPrec);

    Acc a_lib, a_k0, a_k1, a_k2, a_k12, a_k6, a_k8, a_k3, a_h4, a_k5, a_repr;
    size_t n_used = 0, lib_vs_k0_mismatch = 0;

    for (const T& a : ops) {
        if (Fmt<T>::limb(a, 0) <= (typename Fmt<T>::Limb)0) continue;
        const double mag = (double)Fmt<T>::limb(a, 0);
        if (!std::isfinite(mag))                             continue;
        if (bulk_only && (mag < 1e-10 || mag > 1e10))        continue;
        to_mpfr(ea, a);
        if (!mpfr_number_p(ea) || mpfr_sgn(ea) <= 0) continue;
        mpfr_sqrt(mr, ea, MPFR_RNDN);                 // the reference (poisonable)
        mpfr_sqrt(hi, ea, MPFR_RNDN);                 // always 400-bit, for repr
        ++n_used;

        const T lib = sqrt(a);
        a_lib.add(ulps<T>(lib, mr));

        if constexpr (Fmt<T>::n == 2) {
            const T k0 = arm_karp0<T>(a);
            // A silent divergence between the retyped body and the shipped one
            // would invalidate every other arm, so it is counted, not assumed.
            for (int i = 0; i < Fmt<T>::n; ++i)
                if (Fmt<T>::limb(k0, i) != Fmt<T>::limb(lib, i)) { ++lib_vs_k0_mismatch; break; }
            a_k0.add(ulps<T>(k0, mr));
            a_k1.add(ulps<T>(arm_karp1<T>(a), mr));
            a_k2.add(ulps<T>(arm_karp2<T>(a), mr));
            a_k12.add(ulps<T>(arm_karp12<T>(a), mr));
            a_k6.add(ulps<T>(arm_karp6<T>(a), mr));
            a_k8.add(ulps<T>(arm_karp8<T>(a), mr));
            a_k3.add(ulps<T>(arm_karp3<T>(a), mr));
            a_k5.add(ulps<T>(arm_karp5<T>(a, ea), mr));
        }
        a_h4.add(ulps<T>(arm_heron4<T>(a), mr));
        a_repr.add(ulps<T>(from_mpfr<T>(hi), mr));
    }

    printf("=== %s (p = %d)   source: %s%s   n = %zu ===\n",
           Fmt<T>::name(), Fmt<T>::p, source, bulk_only ? " [bulk 1e-10..1e10]" : "", n_used);
    if constexpr (Fmt<T>::n == 2)
        printf("  karp0 vs shipped xp::sqrt: %zu bit mismatches (must be 0)\n",
               lib_vs_k0_mismatch);
    printf("  %-8s %9s %9s %9s %9s %9s %8s\n",
           "arm", "median", "p95", "p99.9", "max", ">1ulp%", ">1ulp n");
    struct Row { const char* n; Acc* a; } rows[] = {
        {"lib",    &a_lib},  {"karp0",  &a_k0},  {"karp1", &a_k1}, {"karp2", &a_k2},
        {"karp12", &a_k12},  {"karp6",  &a_k6},  {"karp8", &a_k8}, {"karp3",  &a_k3},
        {"heron4", &a_h4},
        {"karp5",  &a_k5},   {"repr",   &a_repr},
    };
    for (auto& r : rows) {
        if (r.a->v.empty()) continue;
        printf("  %-8s %9.4f %9.4f %9.4f %9.4g %8.2f%% %8zu\n", r.n, r.a->med(),
               r.a->pct(0.95), r.a->pct(0.999), r.a->max(),
               100.0 * (double)r.a->over(1.0) / (double)r.a->v.size(), r.a->over(1.0));
    }
    printf("\n");
    mpfr_clear(ea); mpfr_clear(mr); mpfr_clear(hi);
}

// ---------------------------------------------------------------------------
// operand sources
// ---------------------------------------------------------------------------

// The sweep's own real-sqrt operands: fabs(re) of every real grid point.
static std::vector<double> load_grid(const char* path) {
    std::vector<double> v;
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return v; }
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == 'k') continue;      // comments, header
        if (line[0] != 'r') continue;                        // real grid only
        // kind,point,family,re,im
        char* p = line;
        int commas = 0;
        for (; *p; ++p) if (*p == ',' && ++commas == 3) { ++p; break; }
        if (commas != 3) continue;
        v.push_back(std::fabs(strtod(p, nullptr)));
    }
    fclose(f);
    return v;
}

// probe_arith_floor's own draw, verbatim in shape: a random 400-bit value in
// [1,2) times a random power of two in 2^-20..2^20, rounded into the format.
template <class T>
static std::vector<T> draw_generic(int N, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<T> out;
    out.reserve(N);
    mpfr_t x, frac; mpfr_init2(x, kPrec); mpfr_init2(frac, kPrec);
    for (int i = 0; i < N; ++i) {
        mpfr_set_ui(x, 1, MPFR_RNDN);
        mpfr_set_ui(frac, 0, MPFR_RNDN);
        for (int k = 0; k < 7; ++k) {
            mpfr_mul_2si(frac, frac, 64, MPFR_RNDN);
            mpfr_add_ui(frac, frac, (unsigned long)(rng() >> 1), MPFR_RNDN);
        }
        mpfr_div_2si(frac, frac, 7*64 - 1, MPFR_RNDN);
        mpfr_div_ui(frac, frac, 2, MPFR_RNDN);
        mpfr_add(x, x, frac, MPFR_RNDN);
        const int e = (int)(rng() % 41) - 20;
        mpfr_mul_2si(x, x, e, MPFR_RNDN);
        out.push_back(from_mpfr<T>(x));
    }
    mpfr_clear(x); mpfr_clear(frac);
    return out;
}

template <class T>
static std::vector<T> to_format(const std::vector<double>& raw) {
    std::vector<T> out;
    out.reserve(raw.size());
    // S(a_in[i]) — exactly the conversion sweep_accuracy.cpp:2352 performs.
    for (double d : raw) out.push_back(T(d));
    return out;
}

int main(int argc, char** argv) {
    int         N    = 20000;
    unsigned    seed = 20260911u;
    std::string grid = "validation/sweep/sweep_grid.csv";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc)         N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc) grid = argv[++i];
        else if (!strcmp(argv[i], "--poison"))               g_ref_prec = 24;
    }
    printf("probe_sqrt_iter: reference at %ld bits%s\n",
           (long)g_ref_prec, g_ref_prec == kPrec ? "" : "   *** POISONED ***");
    printf("ulps = |got-ref| / (|ref| * 2^-p); a correctly rounded sqrt reads <= 0.5.\n\n");

    const std::vector<double> graw = load_grid(grid.c_str());
    if (!graw.empty()) {
        for (int bulk = 0; bulk < 2; ++bulk) {
            printf("---- source: the sweep's own real-sqrt operands (%s)%s ----\n\n",
                   grid.c_str(), bulk ? "   BULK ONLY" : "");
            run<DoubleDouble>(to_format<DoubleDouble>(graw), "grid", bulk);
            run<FloatFloat>  (to_format<FloatFloat>  (graw), "grid", bulk);
            run<TripleFloat> (to_format<TripleFloat> (graw), "grid", bulk);
            run<QuadFloat>   (to_format<QuadFloat>   (graw), "grid", bulk);
        }
    }

    printf("---- source: generic format values, 2^-20 .. 2^20, %d samples ----\n\n", N);
    run<DoubleDouble>(draw_generic<DoubleDouble>(N, seed), "generic", false);
    run<FloatFloat>  (draw_generic<FloatFloat>  (N, seed), "generic", false);
    run<TripleFloat> (draw_generic<TripleFloat> (N, seed), "generic", false);
    run<QuadFloat>   (draw_generic<QuadFloat>   (N, seed), "generic", false);
    return 0;
}
