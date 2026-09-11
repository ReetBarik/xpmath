// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// ============================================================================
// trig_reduction.hpp — Payne-Hanek argument reduction for the xp backends
// ============================================================================
//
// Hand-written. The CONSTANTS it consumes are generated: see
// include/xp/trig_reduction_data.hpp and scripts/gen_trig_reduction_constants.cpp.
//
// WHAT THIS REPLACES AND WHY
// --------------------------
// The four sincos implementations reduce with
//
//     s3 = a - 2pi * round(a / 2pi)
//
// against a single p-bit 2pi. That subtraction is exact (both operands are
// representable), but 2pi is not: the constant carries an absolute error of
// about 2^-p * 2pi, and multiplying it by round(a/2pi) ~ |a| scales that error
// to |a| * 2^-p ABSOLUTE in the residual. The residual is then small, so the
// same absolute error is an enormous RELATIVE error, and sin(r) ~ r passes it
// straight through to the answer. No amount of widening the 2pi constant fixes
// this: widening to q bits merely moves the wall to |a| * 2^-q, and the wall
// must be beaten for every |a| the format can express, not for one of them.
//
// That is the negative control for this whole file, and it is MEASURED, not
// argued: `scripts/probe_trig_stages.cpp --widen` gives the old form an exact
// n and exact 4096-bit arithmetic, leaving q as the only error source, and
// reports bits = q - log2(|x|/|r|) at every point it probes. Widening buys bit
// for bit and never closes the gap, because the gap belongs to x. Covering
// every DoubleDouble needs q >= p + 4 + D = 1239 bits, D = 1129.4 measured
// (include/xp/trig_reduction_data.hpp) -- i.e. the same 2/pi string the table
// below already is, 52 chunks x 24 = 1248 bits. The difference is not the
// string. It is that Payne-Hanek reads a WINDOW of it, guard/24 = 10 chunks
// wide, while `a - 2pi*n` has to multiply the whole thing by n.
//
// Payne-Hanek removes the multiply entirely. Instead of forming n = round(x *
// 2/pi) as a number and subtracting n * (pi/2), it computes the FRACTIONAL
// part of x * (2/pi) directly, from a bit-string of 2/pi long enough that the
// bits which survive the cancellation are still exact. `n` is never formed —
// which matters here for a second, independent reason: n does not FIT. It
// needs log2|n| ~ log2|x| - 2.65 bits of integer, and the quotient carries p
// of significand, so `nint(a / 2pi)` is simply the wrong integer above
// |x| ~ 2^108.7 whatever constant it is fed -- the same probe measures it off
// by 2^909 at the argument that pins the table. An earlier draft of this
// comment blamed round_to_nearest_int for saturating instead; that was wrong
// and unmeasured. round_to_nearest_int is exact at every magnitude (its 2^105
// cap returns a.hi + rint(a.lo), dd_math.hpp:492; FF's 2^47 cap likewise,
// ff_math.hpp) since the KI-14 sibling audit. The defect is in the DIVIDE.
//
// THE ALGORITHM
// -------------
// Write x = sum_i w_i (a non-overlapping expansion) and 2/pi = sum_k T[k] *
// 2^(-cb*(k+1)) with cb-bit integer chunks.
//
//   1. Pick ONE reference exponent ex, the largest word's, and one fixed-point
//      accumulator A[a] on the grid of weights 2^(ex - cb*(jmin+a)). jmin is
//      chosen so A[0]'s weight is at most 2, i.e. the grid runs from just
//      below 4 down to 2^-guard, independent of how large |x| is.
//
//   2. Split each word into cb-bit SIGNED digits, relative to the grid-aligned
//      exponent just above its own:
//        w_i = 2^(ex - cb*s_i) * sum_t d[t] * 2^(-cb*(t+1)),  |d[t]| <= 2^cb
//      Exact, by repeated `u - rint(u)` (which is exact for any u). s_i is a
//      whole number of chunks, so every word's digits land ON the shared grid
//      no matter how far apart the words are.
//
//   3. Convolve d against T into A. Each digit product d[t]*T[k] fits exactly
//      in the working word by construction of cb (2*24 <= 53 for the FP64
//      path, 2*12 <= 24 for the FP32 path), and is split into a high and a low
//      half so no accumulator entry ever leaves the exact-integer range.
//
//      Positions above the grid are simply never generated, and the one high
//      half that would land above it is dropped: those are exact integer
//      multiples of 4 and cannot change n mod 4 or the fraction. THIS IS THE
//      STEP THAT MAKES THE COST FINITE — the window of table chunks actually
//      touched is O(guard/cb) wide regardless of how large |x| is.
//
//   4. n0 = rint(V), f = V - n0, both from the same rint so they stay
//      consistent even at a tie.
//
// WHY ONE SHARED ACCUMULATOR AND NOT ONE PER WORD. Reducing each word against
// its own exponent and summing the resulting fractions is also exact -- (a+b)
// mod m = ((a mod m) + (b mod m)) mod m -- but it is exact only if each word's
// fraction is carried to ABSOLUTE 2^-guard rather than to its own relative
// precision. The words' fractions are O(1) and cancel against each other down
// to 2^-C, so a per-word fraction rounded at p bits loses the whole guard. It
// was measured: on the DD worst case that route gives f to 2^-44 where the
// shared accumulator gives 2^-144. Summing in the digit domain, before any
// rounding, is what makes the cancellation free.
//
// SIZING
// ------
// Two sizes, measured separately, because they are maximised by different
// inputs and combining them is what silently over-sizes the table:
//
//   guard (kPhGuard*)   fractional bits of x*(2/pi) carried below the binary
//                       point = p + C + 4, C the MEASURED worst-case
//                       cancellation of the format. Not 2p, not a rule of
//                       thumb.
//   ntab  (kPhChunks*)  depth of the 2/pi table = max over x of
//                       log2|x| + min(p + C(x), the format's subnormal floor),
//                       rounded up to whole chunks and with NO slack added --
//                       an unreachable chunk would make the N-1 poison in
//                       tests/trig_reduction_test.cpp vacuous.
//
// scripts/gen_trig_reduction_constants.cpp --measure prints both derivations,
// validates its search against the published IEEE-double worst case, and
// validates its candidate enumeration against brute force.
//
// WHAT IS EXACT AND WHAT IS NOT
// -----------------------------
// Steps 1-4 are exact integer arithmetic in floating-point words: every
// intermediate is an integer below 2^24 (FP32 path) or 2^48 (FP64 path). The
// only approximation is the TRUNCATION of the 2/pi table at `guard` fractional
// bits, which leaves |f| in error by at most a few 2^-guard. Converting f to
// the backend's own type and multiplying by pi/2 then rounds normally.
//
// The one place the output can be worse than that: when |f| is so small that
// f * pi/2 needs words below the format's smallest subnormal, the low words
// are lost. For the FP32 backends that floor is 2^-149, and it is the format's,
// not the reduction's — the same wall KI-33 documents for QF complex inverse.
// ============================================================================
#ifndef XP_TRIG_REDUCTION_HPP
#define XP_TRIG_REDUCTION_HPP

#include <xp/config.hpp>
#include <xp/trig_reduction_data.hpp>

#include <cstdint>
#if !defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
#include <cstring>
#endif

namespace xp {
namespace detail {

// ---------------------------------------------------------------------------
// exponent
// ---------------------------------------------------------------------------
// Smallest integer e with |v| < 2^e. Zero returns 0. Subnormals are handled
// because the FP32 backends really do reach them (KI-12's band is below the
// reduction, but an unnormalized low word can be subnormal at any magnitude).

XPMATH_INLINE_FUNCTION int xp_ph_ceil_exp(double v) {
    uint64_t b;
#if !defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
    std::memcpy(&b, &v, sizeof(double));
#else
    b = static_cast<uint64_t>(__double_as_longlong(v));
#endif
    const int      be = static_cast<int>((b >> 52) & 0x7FFU);
    uint64_t       m  = b & 0xFFFFFFFFFFFFFULL;
    if (be != 0) return be - 1022;               // |v| < 2^(be-1022), tight
    if (m == 0) return 0;                        // +-0
    int e = -1074;                               // |v| = m * 2^-1074
    while (m) { m >>= 1; ++e; }
    return e;
}

XPMATH_INLINE_FUNCTION int xp_ph_ceil_exp(float v) {
    uint32_t b;
#if !defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
    std::memcpy(&b, &v, sizeof(float));
#else
    b = static_cast<uint32_t>(__float_as_int(v));
#endif
    const int be = static_cast<int>((b >> 23) & 0xFFU);
    uint32_t  m  = b & 0x7FFFFFU;
    if (be != 0) return be - 126;                // |v| < 2^(be-126), tight
    if (m == 0) return 0;
    int e = -149;
    while (m) { m >>= 1; ++e; }
    return e;
}

// ---------------------------------------------------------------------------
// per-scalar-type policy
// ---------------------------------------------------------------------------
// One template body, two instantiations. The FP64 and FP32 paths differ only
// in the chunk width, the table, and three array capacities; keeping them one
// body is what stops the four backends from drifting apart the way four
// hand-ported copies would.

template <typename S>
struct PhTraits;

// kInDigits  ceil((mantissa bits + cb - 1) / cb) — splits a word EXACTLY even
//            when its exponent is up to cb-1 below the shared grid
// kAcc       >= (guard + 2*cb + 2)/cb + 2 for the deepest backend on this path
// kMant      >= (p + cb)/cb + 1                  — words of f emitted
// kExp       >= kMant + 2                        — Shewchuk scratch
template <>
struct PhTraits<double> {
    static constexpr int kChunk    = kPhChunkBits64;  // 24
    static constexpr int kInDigits = 4;               // ceil((53+23)/24) = 4
    static constexpr int kAcc      = 16;              // DD needs 13
    static constexpr int kMant     = 7;               // (106+24)/24 + 1 = 6.4
    static constexpr int kExp      = 12;
    static XPMATH_INLINE_FUNCTION double table(int k) { return xp_ph_ipio2_d(k); }
};

template <>
struct PhTraits<float> {
    static constexpr int kChunk    = kPhChunkBits32;  // 12
    static constexpr int kInDigits = 3;               // ceil((24+11)/12) = 3
    static constexpr int kAcc      = 24;              // QF needs 21
    static constexpr int kMant     = 10;              // (96+12)/12 + 1 = 10
    static constexpr int kExp      = 16;
    static XPMATH_INLINE_FUNCTION float table(int k) { return xp_ph_ipio2_f(k); }
};

// ---------------------------------------------------------------------------
// expansion scratch
// ---------------------------------------------------------------------------
// The same Shewchuk grow/compress pair dd_math.hpp uses for pow (KI-44), typed
// over the working scalar. Duplicated rather than shared because dd_math.hpp
// sits above this header in the include order and the FP32 backends must not
// have to pull it in.

template <typename S>
XPMATH_INLINE_FUNCTION S xp_ph_two_sum(S a, S b, S& err) {
    const S s  = a + b;
    const S bb = s - a;
    err = (a - (s - bb)) + (b - bb);
    return s;
}

template <typename S>
XPMATH_INLINE_FUNCTION S xp_ph_quick_two_sum(S a, S b, S& err) {
    const S s = a + b;
    err = b - (s - a);
    return s;
}

template <typename S>
XPMATH_INLINE_FUNCTION void xp_ph_push(S* e, int& m, S t, int cap) {
    if (t == S(0)) return;
    S q = t;
    for (int i = 0; i < m; ++i) {
        S err;
        const S s = xp_ph_two_sum(q, e[i], err);
        e[i] = err;
        q    = s;
    }
    // The capacities above are sized so this never fires; folding rather than
    // dropping means a future caller that does overflow loses precision
    // instead of losing the expansion's largest term outright.
    if (m < cap) e[m++] = q;
    else         e[cap - 1] += q;
}

// Repack so every component is full-width, then emit the leading `n` in
// DESCENDING order. Returns the number of nonzero words written.
template <typename S>
XPMATH_INLINE_FUNCTION int xp_ph_compress(const S* e, int m, S* out, int n, S* g, S* h) {
    for (int k = 0; k < n; ++k) out[k] = S(0);
    if (m <= 0) return 0;
    int bottom = m - 1;
    S   q      = e[m - 1];
    for (int i = m - 2; i >= 0; --i) {
        S r;
        q = xp_ph_quick_two_sum(q, e[i], r);
        if (r != S(0)) { g[bottom--] = q; q = r; }
    }
    g[bottom] = q;
    int top = 0;
    for (int i = bottom + 1; i < m; ++i) {
        S r;
        q = xp_ph_quick_two_sum(g[i], q, r);
        if (r != S(0)) h[top++] = r;
    }
    h[top++] = q;
    for (int k = 0; k < n; ++k) out[k] = (top - 1 - k >= 0) ? h[top - 1 - k] : S(0);
    return top < n ? top : n;
}

// ---------------------------------------------------------------------------
// the reduction
// ---------------------------------------------------------------------------
// in:  w[0..nw-1]   the argument as an expansion; words may be unnormalized,
//                   may be zero, and may span arbitrarily many binades
//      guard        fractional bits of x*(2/pi) to carry (kPhGuard*)
//      ntab         table chunks available (kPhChunks*)
//      nfrac        words of f to produce
// out: frac[0..nfrac-1]   f = x*(2/pi) - n as a descending expansion,
//                         |f| <= 1/2 + O(2^-guard)
// ret: n mod 4, in 0..3
//
// The caller multiplies f by pi/2 (xp_ph_pio2_*) to get the reduced argument
// and uses the return value to pick the quadrant.

template <typename S>
XPMATH_INLINE_FUNCTION int xp_ph_reduce(const S* w, int nw, int guard, int ntab,
                                        S* frac, int nfrac) {
    typedef PhTraits<S> P;
    const int cb = P::kChunk;

    for (int k = 0; k < nfrac; ++k) frac[k] = S(0);

    // Accumulator depth: the lowest weight must reach 2^-(guard + 2*cb) so
    // that every product dropped below the grid is itself well below 2^-guard.
    int nacc = (guard + 2 * cb + 2) / cb + 2;
    if (nacc > P::kAcc) nacc = P::kAcc;
    if (nacc < 4)       nacc = 4;

    // ---- 1. the shared grid ------------------------------------------------
    // ex is the largest word's exponent. jmin = ceil((ex-1)/cb) is the first j
    // whose weight 2^(ex-cb*j) is at most 2, so A[0] is the highest position
    // that can still matter and the grid spans [4, 2^-guard) whatever |x| is.
    int  ex  = 0;
    bool any = false;
    for (int i = 0; i < nw; ++i) {
        if (w[i] == S(0)) continue;
        const int e = xp_ph_ceil_exp(w[i]);
        if (!any || e > ex) { ex = e; any = true; }
    }
    if (!any) return 0;

    const int jmin = (ex - 1 >= 0) ? (ex - 1 + cb - 1) / cb : -((1 - ex) / cb);
    const int w0   = ex - cb * jmin;             // in [2-cb, 1]

    S A[P::kAcc];
    for (int a = 0; a < nacc; ++a) A[a] = S(0);

    // ---- 2/3. split each word onto the grid and convolve against 2/pi ------
    for (int i = 0; i < nw; ++i) {
        if (w[i] == S(0)) continue;
        const int ei = xp_ph_ceil_exp(w[i]);
        const int s  = (ex - ei) / cb;           // >= 0, whole chunks
        // Every table index this word could reach is negative -- i.e. it would
        // need chunks of 2/pi above the binary point, of which there are none.
        if (s > jmin + nacc - 3) continue;
        const int eia = ex - cb * s;             // grid-aligned, >= ei

        S d[P::kInDigits];
        S u = detail::ldexp(w[i], -eia);         // |u| < 1, exact
        for (int t = 0; t < P::kInDigits; ++t) {
            u    = detail::ldexp(u, cb);         // exact
            d[t] = detail::rint(u);
            u    = u - d[t];                     // exact
        }

        for (int t = 0; t < P::kInDigits; ++t) {
            if (d[t] == S(0)) continue;
            const int tt = s + t;
            for (int a = 0; a < nacc; ++a) {
                const int k = jmin + a - tt - 2;
                if (k < 0) continue;
                if (k >= ntab) break;
                const S pr = d[t] * P::table(k);            // exact
                const S ph = detail::rint(detail::ldexp(pr, -cb));
                const S pl = pr - detail::ldexp(ph, cb);    // exact
                if (a > 0) A[a - 1] += ph;   // a==0: weight >= 4, an exact
                A[a] += pl;                  //   multiple of 4; drop it
            }
        }
    }

    // ---- carry, so every entry is a cb-bit signed digit --------------------
    for (int a = nacc - 1; a > 0; --a) {
        const S c = detail::rint(detail::ldexp(A[a], -cb));
        A[a] -= detail::ldexp(c, cb);
        A[a - 1] += c;
    }

    // ---- reduce the top position mod 4 ------------------------------------
    // A[0] has weight 2^w0; keeping it mod 2^(2-w0) discards exactly the
    // integer multiples of 4, which is all n mod 4 can afford to lose.
    {
        const int mb = 2 - w0;                   // in [1, cb]
        const S   q  = detail::rint(detail::ldexp(A[0], -mb));
        A[0] -= detail::ldexp(q, mb);
    }

    // ---- 4. n0 = rint(V), then f = V - n0 ---------------------------------
    // Three terms determine V to within 2^-cb, and the only inputs closer than
    // that to a tie are the ones where either choice is equally good: (n, f)
    // and (n+1, f-1) denote the same angle, and because n0 is subtracted back
    // out of the digits below, whichever rint picks stays self-consistent.
    S v = detail::ldexp(A[0], w0)
        + detail::ldexp(A[1], w0 - cb)
        + detail::ldexp(A[2], w0 - 2 * cb);
    const S n0 = detail::rint(v);
    if (n0 != S(0)) {
        // Subtract at position 1, where n0 * 2^(cb-w0) is still an exact
        // integer (position 0 would need n0/2 when w0 == 1).
        A[1] -= detail::ldexp(n0, cb - w0);
        const S c = detail::rint(detail::ldexp(A[1], -cb));
        A[1] -= detail::ldexp(c, cb);
        A[0] += c;
    }

    // ---- canonicalize the digits to one sign ------------------------------
    // The digits are SIGNED, so under extreme cancellation the leading ones
    // are nonzero and cancel against each other: A[0] can be ~2^cb while the
    // value is 2^-119. A leading-nonzero scan in that state stops at position
    // 0, and the emission below then truncates f kMant*cb bits under 1 rather
    // than kMant*cb bits under |f| -- throwing away exactly the bits the guard
    // exists to keep. Borrowing until every low digit carries the value's sign
    // turns the cancellation into leading ZEROS, which is what the scan needs.
    const S base = detail::ldexp(S(1), cb);
    for (int a = nacc - 1; a > 0; --a)
        if (A[a] < S(0)) { A[a] += base; A[a - 1] -= S(1); }
    S sgn = S(1);
    if (A[0] < S(0)) {
        for (int a = 0; a < nacc; ++a) A[a] = -A[a];
        for (int a = nacc - 1; a > 0; --a)
            if (A[a] < S(0)) { A[a] += base; A[a - 1] -= S(1); }
        sgn = S(-1);
    }

    // ---- emit f -----------------------------------------------------------
    int jl = -1;
    for (int a = 0; a < nacc; ++a) if (A[a] != S(0)) { jl = a; break; }
    if (jl < 0) return ((static_cast<int>(n0) % 4) + 4) & 3;

    const int se = w0 - cb * jl;
    S   e[P::kExp];
    int m = 0;
    for (int i = 0; i < P::kMant && jl + i < nacc; ++i) {
        // ldexp, not a multiply: the scale underflows for the FP32 backends at
        // extreme cancellation, and ldexp gives the correctly-rounded subnormal
        // (or zero) instead of a spurious result. That loss is the format's
        // subnormal floor, not the reduction's -- see the header comment.
        const S t = detail::ldexp(A[jl + i], se - cb * i);
        xp_ph_push(e, m, sgn * t, P::kExp);
    }
    S gbuf[P::kExp], hbuf[P::kExp];
    xp_ph_compress(e, m, frac, nfrac, gbuf, hbuf);

    return ((static_cast<int>(n0) % 4) + 4) & 3;
}

}  // namespace detail
}  // namespace xp

#endif  // XP_TRIG_REDUCTION_HPP
