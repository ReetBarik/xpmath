// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// Ported from DDFUN v04 (double-double → float-float):
//   https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz
//   Original author: David H. Bailey (LBNL retired / UC Davis)
//   Original license: DHB-License (modified BSD-3-Clause with §3
//     grant-back clause). Full text: LICENSES/LicenseRef-DHB-License.txt
//     or https://www.davidhbailey.com/dhbsoftware/DHB-License.txt.
//
// This C++ port is a derivative work distributed under the same
// DHB-License. See §3 of that license regarding upstream
// contribution rights.
//
// Modifications from the original DDFUN v04 sources:
//   * Translated from Fortran-90 via dd_math.hpp (2×FP64) to
//     ff_math.hpp (2×FP32). Function inventory, algorithm choices,
//     and coefficient tables descend from DDFUN v04.
//   * FP32-specific modifications (input narrowing, splitter constant
//     8193.0f = 2^13+1, joint sin/cos doublings, Taylor branches for
//     |a|<0.5 in sinh/cosh/atanh, direct exp scaling and scaled-splitter
//     guards at the multiply/divide primitives to avoid splitter
//     overflow, nint magic-constant replacement) are documented in
//     PORT_NOTES.md. These modifications fall under DHB-License §3
//     (grant-back) and are governed by the same terms as the original.
//   * Every function XPMATH_INLINE_FUNCTION for host + device
//     portability across CUDA/HIP/SYCL/OpenMP-target.
//   * Namespaced as xp::FloatFloat with STL-style
//     free-function names.
//   * See docs/TEST_SUITE_PLAN.md "Upstreaming considerations" for
//     naming and API conventions.

#pragma once

// Float-float real arithmetic — xp::FloatFloat. ~14 decimal digits
// from an unevaluated sum of two FP32 components (hi + lo, |lo| <= ulp(hi)/2).
//
// Mechanically ported from dd_math.hpp (this repo's port of DDFUN v04 from
// David H. Bailey, Lawrence Berkeley National Lab) by swapping 2×FP64 for
// 2×FP32. See PORT_NOTES.md for the FP32-specific fixes.
//
// Precision: ~14.4 decimal digits (24-bit FP32 mantissa × 2 = 48 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.
//
// DEPENDENCIES: none beyond the C++17 standard library. In particular this
// header does NOT include or require Kokkos — see xp/config.hpp for how the
// four portability facilities it needs (inline annotation, on-device
// detection, scalar math dispatch, diagnostic printf) are supplied. Kokkos
// users get today's `Kokkos::Experimental::FloatFloat` API unchanged through
// the compat wrapper at third_party/include/ff_math.hpp, which is the only
// place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// Naming conventions (T0.4/T2.0):
//   * Type + math live in one flat namespace so an upstream move is
//     mechanical rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions FloatFloat_pi(), FloatFloat_e(), ...
//     Chosen over a constants::pi<FloatFloat>() template because it mirrors
//     the repository's existing M_PI-style accessors and reads shorter at the
//     call site; they cannot be constexpr template variables because each is
//     built at runtime from IEEE-754 bit patterns, not a literal.
//   * The former bit-pattern constructor became the static factory
//     FloatFloat::from_bits(hi, lo): it is namespaced to the type,
//     discoverable, and needs no free-function symbol.
//   * Math functions are ADL-findable via the argument's namespace. The
//     `Kokkos::`-namespace forwarding overloads that used to sit at the bottom
//     of the original Kokkos-native header (so Kokkos::exp(ff) works like
//     Kokkos::exp(double)) now live in the compat wrapper, since they are
//     Kokkos-specific API surface. add/subtract/multiply/divide are not
//     forwarded — they are for operators and explicit ADL only.

#include <xp/config.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>

#if !defined(XPMATH_ON_DEVICE)
#  include <iomanip>
#  include <ostream>
#endif

namespace xp {
// ============================================================
// Forward declarations (struct uses them in operator bodies)
// ============================================================
struct FloatFloat;
XPMATH_INLINE_FUNCTION FloatFloat add(FloatFloat a, FloatFloat b);
XPMATH_INLINE_FUNCTION FloatFloat subtract(FloatFloat a, FloatFloat b);
XPMATH_INLINE_FUNCTION FloatFloat multiply(FloatFloat a, FloatFloat b);
XPMATH_INLINE_FUNCTION FloatFloat divide(FloatFloat a, FloatFloat b);
XPMATH_INLINE_FUNCTION FloatFloat multiply_scalar(FloatFloat a, float b);
XPMATH_INLINE_FUNCTION FloatFloat divide_scalar(FloatFloat a, float b);
XPMATH_INLINE_FUNCTION FloatFloat negate(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat abs(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat sqrt(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat round_to_nearest_int(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat pow_int(FloatFloat a, int n);
XPMATH_INLINE_FUNCTION FloatFloat exp(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat log(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat pow(FloatFloat a, FloatFloat b);
XPMATH_INLINE_FUNCTION void   sinhcosh(FloatFloat a, FloatFloat& x, FloatFloat& y);
XPMATH_INLINE_FUNCTION void   sincos(FloatFloat a, FloatFloat& x, FloatFloat& y);
XPMATH_INLINE_FUNCTION FloatFloat angle(FloatFloat x, FloatFloat y);

// ============================================================
// FloatFloat struct
// ============================================================
struct FloatFloat {
    float hi;
    float lo;

    XPMATH_INLINE_FUNCTION FloatFloat() : hi(0.0f), lo(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloat(float h) : hi(h), lo(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloat(float h, float l) : hi(h), lo(l) {}
    XPMATH_INLINE_FUNCTION FloatFloat(double h) : hi((float)h), lo((float)(h - (double)(float)h)) {}
    XPMATH_INLINE_FUNCTION FloatFloat(const FloatFloat& o) : hi(o.hi), lo(o.lo) {}
    XPMATH_INLINE_FUNCTION FloatFloat& operator=(const FloatFloat& o) { hi=o.hi; lo=o.lo; return *this; }

    // Factory: build a FloatFloat from the IEEE-754 bit patterns of its two
    // components. Safe on host (memcpy) and device (__int_as_float). Replaces
    // the former free bit-pattern constructor function make_ff().
    static XPMATH_INLINE_FUNCTION FloatFloat from_bits(uint32_t hi_bits, uint32_t lo_bits) {
        float h, l;
#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
        h = __int_as_float(static_cast<int>(hi_bits));
        l = __int_as_float(static_cast<int>(lo_bits));
#else
        std::memcpy(&h, &hi_bits, sizeof(float));
        std::memcpy(&l, &lo_bits, sizeof(float));
#endif
        return FloatFloat(h, l);
    }

    XPMATH_INLINE_FUNCTION FloatFloat operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION FloatFloat operator+(FloatFloat b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloat operator-(FloatFloat b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloat operator*(FloatFloat b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloat operator/(FloatFloat b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloat operator*(float b)  const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloat operator/(float b)  const { return divide_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloat operator+(float b)  const { return add(*this, FloatFloat(b)); }
    XPMATH_INLINE_FUNCTION FloatFloat operator-(float b)  const { return subtract(*this, FloatFloat(b)); }

    XPMATH_INLINE_FUNCTION FloatFloat& operator+=(FloatFloat b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator-=(FloatFloat b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator*=(FloatFloat b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator/=(FloatFloat b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator+=(float b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator-=(float b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator*=(float b) { *this = multiply_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION FloatFloat& operator/=(float b) { *this = divide_scalar(*this, b); return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(FloatFloat b) const { return hi==b.hi && lo==b.lo; }
    XPMATH_INLINE_FUNCTION bool operator!=(FloatFloat b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(FloatFloat b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    XPMATH_INLINE_FUNCTION bool operator>(FloatFloat b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    XPMATH_INLINE_FUNCTION bool operator<=(FloatFloat b) const { return !(b < *this); }
    XPMATH_INLINE_FUNCTION bool operator>=(FloatFloat b) const { return !(*this < b); }
};

XPMATH_INLINE_FUNCTION FloatFloat operator+(float a, FloatFloat b) { return add(FloatFloat(a), b); }
XPMATH_INLINE_FUNCTION FloatFloat operator-(float a, FloatFloat b) { return subtract(FloatFloat(a), b); }
XPMATH_INLINE_FUNCTION FloatFloat operator*(float a, FloatFloat b) { return multiply_scalar(b, a); }
XPMATH_INLINE_FUNCTION FloatFloat operator/(float a, FloatFloat b) { return divide(FloatFloat(a), b); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const FloatFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
// Auto-generated by scripts/gen_ff_constants.cpp -- do not edit by hand.
// Route A: round_to_nearest_FF(Bailey FP64 hi+lo pair).
XPMATH_INLINE_FUNCTION FloatFloat FloatFloat_pi          () { return FloatFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU); } // pi
XPMATH_INLINE_FUNCTION FloatFloat FloatFloat_e           () { return FloatFloat::from_bits(0x402df854U, 0x33b14577U); } // e
XPMATH_INLINE_FUNCTION FloatFloat FloatFloat_log2        () { return FloatFloat::from_bits(0x3f317218U, 0xb102e308U); } // ln(2)
XPMATH_INLINE_FUNCTION FloatFloat FloatFloat_log10       () { return FloatFloat::from_bits(0x40135d8eU, 0xb309555dU); } // ln(10)
XPMATH_INLINE_FUNCTION FloatFloat FloatFloat_sqrt2       () { return FloatFloat::from_bits(0x3fb504f3U, 0x32cfe77aU); } // sqrt(2)
XPMATH_INLINE_FUNCTION FloatFloat FloatFloat_euler_gamma () { return FloatFloat::from_bits(0x3f13c468U, 0xb1e4127aU); } // Euler gamma

// ============================================================
// Primitive arithmetic
// ============================================================

XPMATH_INLINE_FUNCTION FloatFloat negate(FloatFloat a) {
    return FloatFloat(-a.hi, -a.lo);
}

// TwoSum (Knuth)
XPMATH_INLINE_FUNCTION FloatFloat add(FloatFloat a, FloatFloat b) {
    float t1 = a.hi + b.hi;
    float e  = t1 - a.hi;
    float t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

XPMATH_INLINE_FUNCTION FloatFloat subtract(FloatFloat a, FloatFloat b) {
    float t1 = a.hi - b.hi;
    float e  = t1 - a.hi;
    float t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

// TwoProduct (Dekker splitting). Splitter = 2^13 + 1 for FP32 (24-bit mantissa).
XPMATH_INLINE_FUNCTION FloatFloat multiply(FloatFloat a, FloatFloat b) {
    // KI-27.  Non-finite / degenerate signalling on the PRODUCT — the
    // multiply-side counterpart of the guard KI-19 put on divide(), and the
    // exact rule qf_two_prod/tf_two_prod already carry.  Full derivation at
    // dd_math.hpp:multiply.  This supersedes the "Scope" paragraph below, which
    // recorded genuine product overflow as deliberately out of B9's scope: the
    // hardware product IS the answer when it is inf, NaN or zero, and the
    // error-free transform must not run on it.  Placed before the B9 scaling so
    // that it sees the TRUE product rather than a rescaled stand-in.
    //   was: multiply(1e30, 1e30) -> NaN;  now +inf.
    const float p = a.hi * b.hi;
    if (!detail::isfinite(p) || p == 0.0f) return FloatFloat(p, 0.0f);
    const float split = 8193.0f;
    // B9: scaled splitter — the same overflow B8 fixed in divide() below, at a
    // third site (see divide()'s comment for the full derivation). BOTH operands
    // are split here, so both need the guard: for |x.hi| > FLT_MAX / (split + 1)
    // ≈ 4.15e34 the product x.hi * split overflows to ±inf, and then
    // x1 = conx - (conx - x.hi) = inf - inf = NaN poisons the whole result.
    // Surfaced by B6: erfc's multiply(sqrt(pi), exp(z^2)) walks into it at
    // z ≈ 8.93 (PORT_NOTES §4i). B6 dodged the exposure by reformulating; this
    // closes it at the primitive.
    //
    // Unscale by MULTIPLYING by 2^64 per scaled operand, applied SEPARATELY —
    // never by 1/(sa*sb). When both operands are scaled, sa*sb = 2^-128 is
    // subnormal and 1/(sa*sb) overflows to +inf, which would replace one garbage
    // answer with another. Two exact power-of-two multiplies have no such
    // intermediate.
    //
    // Scope: this fixes products that are REPRESENTABLE but were reached through
    // a hazard-band operand. It does not change what happens when the product
    // itself overflows FP32 — multiply() already returned NaN there long before
    // any splitter scaling (c11 = inf makes a1*b1 - c11 = -inf and then
    // e = t1 - c11 = NaN), as multiply(1e30, 1e30) shows with both operands well
    // below the band. Both operands in the band always implies such an overflow
    // (|a.hi*b.hi| > 1.7e69), so that corner stays non-finite; it lands on
    // (inf, -inf) rather than NaN, which is neither better nor worse. Fixing
    // overflow semantics is a separate question from fixing the splitter.
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const bool  ha = detail::fabs(a.hi) > kSplitOverflowThresh;
    const bool  hb = detail::fabs(b.hi) > kSplitOverflowThresh;
    const float sa = ha ? ldexpf(1.0f, -64) : 1.0f;
    const float sb = hb ? ldexpf(1.0f, -64) : 1.0f;
    const float ua = ha ? ldexpf(1.0f,  64) : 1.0f;
    const float ub = hb ? ldexpf(1.0f,  64) : 1.0f;
    a = FloatFloat(a.hi * sa, a.lo * sa);
    b = FloatFloat(b.hi * sb, b.lo * sb);
    float cona = a.hi * split, conb = b.hi * split;
    float a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    float a2 = a.hi - a1,            b2 = b.hi - b1;
    float c11 = a.hi * b.hi;
    float c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2  = a.hi * b.lo + a.lo * b.hi;
    float t1  = c11 + c2;
    float e   = t1 - c11;
    float t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    float hi  = t1 + t2;
    float lo  = t2 - (hi - t1);
    // B9: unscale. ua/ub are exact powers of two (or 1.0f on the non-hazard path,
    // where `hi * 1.0f * 1.0f == hi` bit-for-bit).
    return FloatFloat(hi * ua * ub, lo * ua * ub);
}

XPMATH_INLINE_FUNCTION FloatFloat divide(FloatFloat a, FloatFloat b) {
    // §B8 extension: a NON-FINITE DIVISOR must never reach the Dekker splitter.
    // B8's divisor guard below classifies on |b.hi| > kSplitOverflowThresh, which
    // is TRUE for b.hi = ±inf, so it scales by 2^-64 — and inf * 2^-64 is still
    // inf. The splitter then computes conb = inf * split = inf and
    // b1 = conb - (conb - b.hi) = inf - inf = NaN, poisoning a quotient that IEEE
    // defines perfectly well: x / ±inf = ±0 for finite x. divide(1.0f, inf)
    // returned NaN where IEEE 754-2019 §6.1 requires +0.
    //
    // The whole answer is the bare float quotient a.hi / b.hi, which is exact for
    // every non-finite divisor and needs no FF arithmetic at all:
    //   finite / ±inf  ->  ±0, correctly signed (including ±0 / ±inf, where IEEE
    //                      preserves the sign of the zero numerator)
    //   ±inf   / ±inf  ->  NaN (invalid), unchanged
    //   anything/ NaN  ->  NaN, unchanged
    // Returning the quotient itself, rather than a copysign(0, ...) form, is what
    // keeps the inf/inf case NaN — copysign would flatten it to a signed zero.
    // lo = 0.0f because the result is exact, matching B10's Zone C convention.
    //
    // Early return, so this short-circuits ahead of every splitter below by
    // construction. Divide-by-ZERO is deliberately NOT touched here: b.hi = 0 is
    // finite, falls through, and keeps B10's Zone C ±inf semantics.
    if (!detail::isfinite(b.hi)) return FloatFloat(a.hi / b.hi, 0.0f);
    const float split = 8193.0f;
    // B8: the Dekker splitter below computes conb = b.hi * split (line ~"conb =")
    // to extract b.hi's high half. For |b.hi| > FLT_MAX / (split + 1) ≈ 4.15e34
    // that product overflows to ±inf, then b1 = conb - (conb - b.hi) = inf - inf
    // = NaN, which poisons the whole quotient. This bites log()'s Newton
    // iteration (divisor exp(b) ≈ a) for x ≳ e^79.7 ≈ 1.5e34, feeding NaN back
    // into exp() and hanging its Taylor loop at the 60-iteration cap (surfaced by
    // the B4 investigation — see TEST_SUITE_PLAN.md §B4/§B8).
    //
    // Fix: when b.hi is in the overflow-hazard band, pre-scale the divisor down
    // by an exact power of two (2^-64: no FP rounding, so b's full FF precision is
    // preserved), run the unchanged Dekker split, then unscale the quotient. Since
    // q = a / (b·s) = (a/b) / s, the true quotient a/b is recovered by MULTIPLYING
    // q by the same down-scale factor s. 2^-64 gives ample headroom: the largest
    // |b.hi| ≈ FLT_MAX = 2^128 maps to 2^64, and 2^64 * split ≈ 2^77 ≪ FLT_MAX.
    // Mirrors PORT_NOTES §4a's power-of-2 scaling pattern for exp's final scaling
    // (same bug class, different site — the splitter, not exp's scaling).
    //
    // B10: the SECOND splitter run in this function splits the QUOTIENT ESTIMATE
    // s1 = a.hi / b.hi via `cona = s1 * split`, and |s1| can land in the same
    // 4.15e34 band with BOTH operands far below it — divide(1e35f, 1.0f) and
    // divide(1e30f, 1e-5f) returned NaN with no large operand anywhere. Three
    // zones, classified from the actual value of s1:
    //
    //   Zone A  |s1| <= kSplitOverflowThresh. Safe; untouched, bit-identical.
    //   Zone B  kSplitOverflowThresh < |s1| < inf. The splitter overflows but the
    //           true quotient IS representable. Pre-scale the NUMERATOR down by
    //           the exact power of two 2^-32, run the unchanged body, unscale the
    //           quotient by 2^32. Was NaN; now correct.
    //   Zone C  |s1| == inf. The true quotient overflows FP32, so NO scaling
    //           recovers a finite answer. Return the correctly-signed ±inf
    //           instead of NaN.
    //
    // Zone C is a deliberate SEMANTIC upgrade (an honest overflow signal instead
    // of a NaN poison), not a bit-identical fix. Unlike B8 and B9, B10 cannot
    // promise "correct everywhere in the band", because part of the band has no
    // representable answer at all. That is inherent to division, not a shortcut.
    //
    // Classifying on s1 itself is overflow-safe by construction, which is why no
    // `|a.hi|` vs `|b.hi| * thresh` product is needed: IEEE division never traps,
    // delivers a correctly-signed ±inf exactly when the quotient exceeds FLT_MAX,
    // and yields NaN only for 0/0 and inf/inf — both of which fail
    // `fabs(s1) > thresh` and fall through to the unchanged body, where they
    // produced NaN before and still do.
    //
    // Why scale the NUMERATOR: the divisor is B8's to scale, and s1 is what has to
    // shrink. Multiplying `a` by an exact power of two scales s1, every Dekker
    // intermediate, and the final (hi, lo) by exactly that factor, so the result
    // is recovered exactly. Watch the direction, as in B8/B9:
    // q = (a*sn) / (b*s) = (a/b) * (sn/s), so recovering a/b means MULTIPLYING q
    // by s/sn — applied as two separate exact factors, never as one reciprocal.
    //
    // Why 2^-32 rather than B8/B9's 2^-64: this scale lands on the NUMERATOR,
    // which in Zone B can be as small as |b.hi| * 4.15e34 ~ 2^-34 (b.hi subnormal),
    // so an over-aggressive scale risks flushing a.lo into the subnormal range.
    // 2^-32 keeps ~19 binades of splitter headroom (|s1| <= 2^128 maps to 2^96 and
    // 2^96 * split ~ 2^109 << FLT_MAX) while leaving a.lo >= 2^-90 in that worst
    // corner, ~36 binades clear of 2^-126. 2^-64 would leave under 4.
    //
    // Interaction with B8's divisor scale: the two guards are mutually exclusive.
    // B8 fires only for |b.hi| > T, which already forces |a.hi/b.hi| <= FLT_MAX/T
    // = split+1 = 8194; after B8's 2^-64 the estimate is |s1| <= 8194 * 2^64
    // ~ 1.5e23, eleven orders below T. So B10 can never fire on top of B8. The
    // unscale is still written as two independent factors so it would compose
    // correctly if it ever could.
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const float s = (detail::fabs(b.hi) > kSplitOverflowThresh)
                        ? ldexpf(1.0f, -64) : 1.0f;
    b = FloatFloat(b.hi * s, b.lo * s);
    float s1 = a.hi / b.hi;
    float un = 1.0f;                                     // B10 numerator unscale
    if (detail::fabs(s1) > kSplitOverflowThresh) {       // not Zone A
        if (detail::isinf(s1)) return FloatFloat(s1, 0.0f);        // Zone C
        const float sn = ldexpf(1.0f, -32);                        // Zone B
        un = ldexpf(1.0f, 32);
        a  = FloatFloat(a.hi * sn, a.lo * sn);
        s1 = s1 * sn;   // exact, and identical to recomputing (a.hi*sn) / b.hi
    }
    float cona = s1 * split, conb = b.hi * split;
    float a1   = cona - (cona - s1), b1 = conb - (conb - b.hi);
    float a2   = s1 - a1,            b2 = b.hi - b1;
    float c11  = s1 * b.hi;
    float c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2   = s1 * b.lo;
    float t1   = c11 + c2;
    float e    = t1 - c11;
    float t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    float t12  = t1 + t2;
    float t22  = t2 - (t12 - t1);
    float t11  = a.hi - t12;
    e = t11 - a.hi;
    float t21  = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    float s2   = (t11 + t21) / b.hi;
    float hi   = s1 + s2;
    float lo   = s2 - (hi - s1);
    // B8/B10: unscale — recover a/b from (a·un⁻¹)/(b·s) by multiplying by s and by
    // un, as two separate exact powers of two. Both are 1.0f on the non-hazard
    // path, where `hi * 1.0f * 1.0f == hi` bit-for-bit.
    return FloatFloat(hi * s * un, lo * s * un);
}

XPMATH_INLINE_FUNCTION FloatFloat multiply_scalar(FloatFloat a, float b) {
    const float p = a.hi * b;                                    // KI-27, see multiply()
    if (!detail::isfinite(p) || p == 0.0f) return FloatFloat(p, 0.0f);
    const float split = 8193.0f;
    // B9: scaled splitter, identical in shape to multiply() above — both a.hi and
    // the scalar b are split, so both are guarded. Reachable from the public
    // operator*(FloatFloat, float) / operator*(float, FloatFloat) with either side
    // in the hazard band; the in-header callers all pass small scalars, but `a`
    // is unconstrained. Same separate-unscale rule as multiply().
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const bool  ha = detail::fabs(a.hi) > kSplitOverflowThresh;
    const bool  hb = detail::fabs(b)    > kSplitOverflowThresh;
    const float sa = ha ? ldexpf(1.0f, -64) : 1.0f;
    const float sb = hb ? ldexpf(1.0f, -64) : 1.0f;
    const float ua = ha ? ldexpf(1.0f,  64) : 1.0f;
    const float ub = hb ? ldexpf(1.0f,  64) : 1.0f;
    a = FloatFloat(a.hi * sa, a.lo * sa);
    b = b * sb;
    float cona = a.hi * split, conb = b * split;
    float a1   = cona - (cona - a.hi), b1 = conb - (conb - b);
    float a2   = a.hi - a1,            b2 = b - b1;
    float c11  = a.hi * b;
    float c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2   = a.lo * b;
    float t1   = c11 + c2;
    float e    = t1 - c11;
    float t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    float hi   = t1 + t2;
    float lo   = t2 - (hi - t1);
    return FloatFloat(hi * ua * ub, lo * ua * ub);   // B9 unscale, see multiply()
}

XPMATH_INLINE_FUNCTION FloatFloat divide_scalar(FloatFloat a, float b) {
    // §B8 extension: non-finite divisor, divide()'s guard at the scalar site. The
    // divisor guard below scales by 2^-64 for |b| > kSplitOverflowThresh, which
    // b = ±inf satisfies while staying inf, so conb = inf * split = inf and
    // b1 = inf - inf = NaN. Same bare-quotient answer, same reasoning; the full
    // derivation lives at divide() and is not repeated here.
    if (!detail::isfinite(b)) return FloatFloat(a.hi / b, 0.0f);
    const float split = 8193.0f;
    // B9: scaled splitter on the DIVISOR — B8's divide() fix at the scalar site.
    // For |b| > FLT_MAX / (split + 1) ≈ 4.15e34, conb = b * split overflows and
    // b1 = inf - inf = NaN. Pre-scale b down by the exact power of two 2^-64, run
    // the unchanged split, then recover a/b from a/(b·s) by MULTIPLYING the
    // quotient by s (B8's sign-error warning applies verbatim). Reachable from
    // the public operator/(FloatFloat, float).
    //
    // Scaling b down inflates the quotient estimate t1 = a.hi / b by 2^64, which
    // cannot itself reach the band: the guard only fires for |b| > 4.15e34, so
    // |t1| < FLT_MAX / 4.15e34 = 8193 beforehand and < 8193·2^64 ≈ 1.5e23 after —
    // eleven orders below the 4.15e34 threshold.
    //
    // B10: the quotient-estimate guard, divide()'s fix at the scalar site. t1 is
    // this function's `s1` — the estimate that `cona = t1 * split` splits — and it
    // reaches the hazard band on its own with a perfectly ordinary scalar
    // (divide_scalar(1e35f, 2.0f) returned NaN). Same three zones, same
    // classify-on-the-quotient-itself rule, same 2^-32 numerator scale, same
    // Zone C ±inf-instead-of-NaN semantic upgrade, same mutual exclusivity with
    // the divisor guard above. The full derivation lives at divide(); it is not
    // repeated here.
    //
    // No shared helper for the two sites: the operand types differ (FloatFloat vs
    // float divisor) and each guard is five lines inline at a hot arithmetic
    // primitive — the same trade B9 made across its four splitter sites.
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const float s = (detail::fabs(b) > kSplitOverflowThresh) ? ldexpf(1.0f, -64) : 1.0f;
    b = b * s;
    float t1 = a.hi / b;
    float un = 1.0f;                                     // B10 numerator unscale
    if (detail::fabs(t1) > kSplitOverflowThresh) {       // not Zone A
        if (detail::isinf(t1)) return FloatFloat(t1, 0.0f);        // Zone C
        const float sn = ldexpf(1.0f, -32);                        // Zone B
        un = ldexpf(1.0f, 32);
        a  = FloatFloat(a.hi * sn, a.lo * sn);
        t1 = t1 * sn;   // exact, and identical to recomputing (a.hi*sn) / b
    }
    float cona = t1 * split, conb = b * split;
    float a1   = cona - (cona - t1), b1 = conb - (conb - b);
    float a2   = t1 - a1,            b2 = b - b1;
    float t12  = t1 * b;
    float t22  = (((a1*b1 - t12) + a1*b2) + a2*b1) + a2*b2;
    float t11  = a.hi - t12;
    float e    = t11 - a.hi;
    float t21  = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    float t2   = (t11 + t21) / b;
    float hi   = t1 + t2;
    float lo   = t2 - (hi - t1);
    // B9/B10: unscale — two separate exact powers of two, both 1.0f off-hazard.
    return FloatFloat(hi * s * un, lo * s * un);
}

// Exact product of two floats
XPMATH_INLINE_FUNCTION FloatFloat two_prod(float fa, float fb) {
    const float p = fa * fb;                                     // KI-27, see multiply()
    if (!detail::isfinite(p) || p == 0.0f) return FloatFloat(p, 0.0f);
    const float split = 8193.0f;
    // B9: scaled splitter, multiply()'s guard at the bare-float site. sqrt() —
    // the only in-header caller — passes t2 ≈ sqrt(a.hi) ≤ 1.9e19, far below the
    // band, so this is unreachable from inside the header today. Guarded anyway:
    // two_prod is a public free function in Kokkos::Experimental, the fix is the
    // same four lines, and an exactness primitive silently returning NaN is a bad
    // trap to leave armed. Same separate-unscale rule as multiply().
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const bool  ha = detail::fabs(fa) > kSplitOverflowThresh;
    const bool  hb = detail::fabs(fb) > kSplitOverflowThresh;
    const float sa = ha ? ldexpf(1.0f, -64) : 1.0f;
    const float sb = hb ? ldexpf(1.0f, -64) : 1.0f;
    const float ua = ha ? ldexpf(1.0f,  64) : 1.0f;
    const float ub = hb ? ldexpf(1.0f,  64) : 1.0f;
    fa = fa * sa;
    fb = fb * sb;
    float cona = fa * split, conb = fb * split;
    float a1   = cona - (cona - fa), b1 = conb - (conb - fb);
    float a2   = fa - a1,            b2 = fb - b1;
    float s1   = fa * fb;
    float s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return FloatFloat(s1 * ua * ub, s2 * ua * ub);   // B9 unscale, see multiply()
}

// ============================================================
// Basic math
// ============================================================

XPMATH_INLINE_FUNCTION FloatFloat abs(FloatFloat a) {
    return (a.hi >= 0.0f) ? a : FloatFloat(-a.hi, -a.lo);
}

// Nearest integer, TIES TO EVEN. The DD-style magic-constant trick (using a
// 2^47 FF constant) is fragile in FP32: ULP at 2^47 is 2^24, much larger than
// typical integer inputs, so the FF lo component must rescue the precision and
// ties land on the wrong side. Instead, do the rounding in FP64 — FF values are
// bounded by 2^48 and fit exactly in FP64's 53-bit mantissa. `(double)a.hi +
// (double)a.lo` is exact for a NORMALIZED pair, where lo sits just under
// ulp(hi)/2 — 24 + 24 adjacent significant bits against FP64's 53. It is NOT
// exact for a pair whose limbs are far apart (lo below 2^-53*|hi|, which the
// non-overlap invariant permits); that case is KI-6, filed, not fixed here.
//
// KI-2 (2026-09-02): FF is NOT one of the two backends KI-2 affects. That defect
// is QD's `floor(d + 0.5)` double-rounding, and this routine never had that
// formulation. It previously used the FP64 magic constant
// `(total + 2^52) - 2^52`, which for |total| < 2^47 is exact and ties-to-even —
// i.e. already right. `detail::rint` is the same function in one instruction,
// with two advantages worth the swap: it needs no magic constant, and it stays
// correct under a non-default rounding mode, where the magic-constant trick
// silently is not. Verified bit-identical to the old form over the whole
// half-integer and near-half-integer class; see docs/KNOWN_ISSUES.md, KI-2 resolution.
//
// Ties-to-even is the SHIPPED semantics of ff::round and is what
// ff_accuracy_test.cpp's `nearbyintq` oracle expects. It is deliberately
// unchanged. (QF and TF are ties-toward-+infinity and are likewise unchanged;
// the two backend families disagree on ties and always have.)
XPMATH_INLINE_FUNCTION FloatFloat round_to_nearest_int(FloatFloat a) {
    if (a.hi == 0.0f) return FloatFloat(0.0f);
    double total = (double)a.hi + (double)a.lo;
    // KI-14.  The 2^47 cap is NOT a cast to a 32/48-bit integer, and it is not a
    // limit of the FP64 reduction either: it is a vestige of this routine's
    // PREVIOUS body, the magic-constant form `(total + 2^52) - 2^52`, which
    // stops rounding once |total| reaches 2^51 and was capped at 2^47 for
    // margin.  The KI-2 rewrite to `detail::rint` — exact at EVERY magnitude —
    // left the cap and its `return FloatFloat(0.0f)` in place, so every
    // |x| >= 2^47 came back as 0 from floor/ceil/trunc/round even though the
    // value was already an exact integer.
    //
    // Past the cap the answer is available for free, without the FP64 detour.
    // |total| >= 2^47 forces |a.hi| >= 2^46 (|a.lo| <= ulp(a.hi)/2), so
    // ulp(a.hi) >= 2^22 and a.hi is an exact EVEN integer; all fractional
    // content lives in a.lo.  Hence nint(a) = a.hi + nint(a.lo), and because
    // a.hi is even, ties-to-even on the low word is ties-to-even on the whole
    // value — the shipped semantics are preserved.  `add` of two floats is
    // two_sum-exact, so the reassembly loses nothing.  rint(a.lo) is
    // representable as a float: either |a.lo| >= 2^23 and it is already an
    // integer, or it is an integer below 2^23.
    if (detail::fabs(total) >= 1.40737488355328e14 /* 2^47 */) {
        return add(FloatFloat(a.hi), FloatFloat((float)detail::rint((double)a.lo)));
    }
    double rounded = detail::rint(total);
    // Zero-sign restore, exactly as qf_nint/tf_nint do. rint(total) is -0.0 for
    // every total in (-0.5, 0), where the magic-constant form returned +0.0
    // ((total - 2^52) + 2^52 is a cancellation, and x + (-x) is +0 under
    // round-to-nearest). Without this line the swap is NOT bit-identical to the
    // old form and the sign of a zero reduction quotient flips — observable,
    // because sincos multiplies that quotient back in. With it, the swap is
    // provably a no-op on every input.
    if (rounded == 0.0) rounded = 0.0;
    float hi = (float)rounded;
    float lo = (float)(rounded - (double)hi);
    return FloatFloat(hi, lo);
}

XPMATH_INLINE_FUNCTION FloatFloat sqrt(FloatFloat a) {
    if (a.hi == 0.0f) return FloatFloat(0.0f);
    if (a.hi < 0.0f) {
        XPMATH_PRINTF("FFSQRT: negative argument\n");
        return FloatFloat(0.0f);
    }
    float t1 = 1.0f / detail::sqrt(a.hi);
    float t2 = a.hi * t1;
    FloatFloat s0 = two_prod(t2, t2);
    FloatFloat s1 = subtract(a, s0);
    float t3  = 0.5f * s1.hi * t1;
    return add(FloatFloat(t2), FloatFloat(t3));
}

// Integer power
XPMATH_INLINE_FUNCTION FloatFloat pow_int(FloatFloat a, int n) {
    const float cl2 = 1.4426950408889633f;
    if (a.hi == 0.0f) {
        if (n >= 0) return FloatFloat(0.0f);
        XPMATH_PRINTF("FFNPWR: zero base with negative exponent\n");
        return FloatFloat(0.0f);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return FloatFloat(1.0f);
    if (nn == 1) return (n > 0) ? a : divide(FloatFloat(1.0f), a);
    if (nn == 2) { FloatFloat r = multiply(a,a); return (n>0) ? r : divide(FloatFloat(1.0f),r); }
    int mn = (int)(cl2 * detail::log((float)nn) + 1.0f + 1.0e-6f);
    FloatFloat s0 = a, s2 = FloatFloat(1.0f);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = multiply(s2, s0);
        kn = kk;
        if (j < mn) s0 = multiply(s0, s0);
    }
    if (n < 0) s2 = divide(FloatFloat(1.0f), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

XPMATH_INLINE_FUNCTION FloatFloat exp(FloatFloat a) {
    const int nq = 4;
    const float eps = 1.0e-15f;
    FloatFloat al2 = FloatFloat_log2();
    // KI-6: word-range-derived guard. e^x exceeds FLT_MAX (3.4028235e38) above
    // ln(FLT_MAX) = 88.722839, and falls below the smallest FP32 subnormal
    // (1.401e-45) below ln(2^-149) = -103.28. The old ±88 guard flushed the top
    // 0.72 and the whole subnormal tail to zero although both are representable.
    // What remains bounds `nz` before the (int) cast below.
    if (a.hi > 88.722839f) {
        XPMATH_PRINTF("FFEXP: overflow\n");
        return FloatFloat(HUGE_VALF);           // e^x > FLT_MAX: +inf is the answer
    }
    if (a.hi < -104.0f) return FloatFloat(0.0f); // e^x < 2^-149: 0 is the answer

    FloatFloat s0 = divide(a, al2);
    FloatFloat s1 = round_to_nearest_int(s0);
    float t1  = s1.hi;
    int nz    = (int)(t1 + detail::copysign(1.0e-6f, t1));
    s0 = subtract(a, multiply(al2, s1));

    if (s0.hi == 0.0f) {
        return FloatFloat(ldexpf(1.0f, nz));
    }
    // Scale down by 2^nq, Taylor, then square nq times.
    // KI-34: the series and the squarings track e^r - 1, not e^r, so the
    // squaring step is (1+s)^2 - 1 = s*(s+2), which PRESERVES relative error
    // instead of doubling it. See dd_math.hpp's exp for the derivation. Here
    // nq = 4, so the shipped form multiplied the series error by 16; measured
    // FF exp 5.3 -> and log's absolute floor 7.6-10 units of 2^-48.
    // Convergence is now relative to a sum of size |r| <= 0.0217, ~46x
    // stricter, and reached in 7 terms; the terms fall by r/k each step so the
    // FF exp-eps stall (eps = 1e-15 finer than FF's 3.55e-15 resolution) is not
    // reachable by tightening this way.
    s1 = multiply_scalar(s0, ldexpf(1.0f, -nq));
    FloatFloat s2 = s1, s3 = s1;                  // term = r, sum = e^r - 1
    for (int l1 = 2; l1 <= 60; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (float)l1);
        s0 = add(s3, s2);
        s3 = s0;
        if (detail::fabs(s2.hi) <= eps * detail::fabs(s3.hi)) break;
        // KI-6: no return-0 on the iteration limit. Falling through with the
        // partial sum is a precision shortfall; returning 0 is a wrong answer.
        // (Matches qf_math.hpp, which dropped the same return when eps was
        // retuned there.)
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, add(s3, FloatFloat(2.0f)));
    s3 = add(FloatFloat(1.0f), s3);

    // KI-6: scale by 2^nz through the EXPONENT, component-wise. Forming the
    // factor as `ldexpf(1.0f, nz)` first is +inf for nz >= 128 and 0 for
    // nz <= -150, so the top and bottom of the FP32 range were unreachable
    // whatever the guard said. (Going through multiply_scalar is worse still:
    // it computes b*8193 inside Dekker splitting, which overflows for
    // nz >= 115 — the cause of the older NaN outputs at the high end.)
    if (nz >= -125 && nz <= 127) {          // 2^nz exact and normal: cheap path
        const float pow2 = ldexpf(1.0f, nz);
        return FloatFloat(s3.hi * pow2, s3.lo * pow2);
    }
    return FloatFloat(ldexpf(s3.hi, nz), ldexpf(s3.lo, nz));
}

XPMATH_INLINE_FUNCTION FloatFloat log(FloatFloat a) {
    if (detail::isinf(a.hi)) return a;   // KI-6, see dd_math.hpp
    if (a.hi <= 0.0f) {
        XPMATH_PRINTF("FFLOG: non-positive argument\n");
        return FloatFloat(0.0f);
    }
    // Initial approximation then 2 Newton steps (FP32 base gives ~6 digits, doubles per iter -> 24 -> 48 bits)
    FloatFloat b = FloatFloat(detail::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        // KI-23, small-argument mirror image.  See qf_math.hpp's log for the
        // derivation.  The residual form evaluates e^{b}, which for a << 1 is
        // tiny and loses its lo word to FP32 underflow -- FF measured 9.54 of
        // 14 digits at 1e-38.  Switch to QD's form only where that bites: lo
        // sits at 2^(E-24), leaving the FP32 normal range 2^-126 once E < -102,
        // i.e. b < -102*ln2 = -70.7.  Above that the residual form is strictly
        // better (relative rather than absolute error on the correction), and
        // below -ln(FLT_MAX) the switch is unavailable because 1/a overflows.
        if (b.hi < -70.7f && -b.hi <= 88.722839f) {
            FloatFloat s0 = exp(negate(b));                     // e^{|b|} >= 1
            b = subtract(add(b, multiply(a, s0)), FloatFloat(1.0f));
        } else {
            FloatFloat s0 = exp(b);                             // e^{b} >= 1
            FloatFloat s1 = subtract(a, s0);
            FloatFloat s2 = divide(s1, s0);
            b = add(b, s2);
        }
    }
    return b;
}

XPMATH_INLINE_FUNCTION FloatFloat log2(FloatFloat a) {
    return divide(log(a), FloatFloat_log2());
}

XPMATH_INLINE_FUNCTION FloatFloat log10(FloatFloat a) {
    return divide(log(a), FloatFloat_log10());
}

// log1p(a) = log(1+a). Series 2*atanh(a/(2+a)) for |a| < 1/4, plain log(1+a)
// outside; see the derivation on dd_math.hpp's log1p (KI-5(b)), including why
// Goldberg's correction was measured and rejected. The old body
// `log(add(1, a))` kept no more than log10(1/|a|) digits for small a.
XPMATH_INLINE_FUNCTION FloatFloat log1p(FloatFloat a) {
    if (detail::fabs(a.hi) < 0.25f) {
        FloatFloat t   = divide(a, add(FloatFloat(2.0f), a));
        FloatFloat t2  = multiply(t, t);
        FloatFloat sum = t;
        FloatFloat trm = t;
        for (int k = 3; k < 80; k += 2) {
            trm = multiply(trm, t2);
            FloatFloat incr = divide(trm, FloatFloat((float)k));
            sum = add(sum, incr);
            if (detail::fabs(incr.hi) <= 1.0e-17f * detail::fabs(sum.hi)) break;
        }
        return multiply_scalar(sum, 2.0f);
    }
    return log(add(FloatFloat(1.0f), a));
}

XPMATH_INLINE_FUNCTION FloatFloat exp2(FloatFloat a) {
    return exp(multiply(a, FloatFloat_log2()));
}

XPMATH_INLINE_FUNCTION FloatFloat exp10(FloatFloat a) {
    return exp(multiply(a, FloatFloat_log10()));
}

XPMATH_INLINE_FUNCTION FloatFloat expm1(FloatFloat a) {
    if (detail::fabs(a.hi) > 0.5f) {
        return subtract(exp(a), FloatFloat(1.0f));
    }
    // Taylor: a + a^2/2! + a^3/3! + ...
    FloatFloat sum = a, term = a;
    for (int k = 2; k <= 30; ++k) {
        term = divide_scalar(multiply(term, a), (float)k);
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < 1.0e-15f * detail::fabs(sum.hi)) break;
    }
    return sum;
}

// ============================================================
// Trig — internal combined cos+sin, then derived
// ============================================================

// Track sin and cos jointly through nq doublings — avoids the sqrt(1-cos^2)
// recovery step, which loses relative precision when sin is near zero
// (i.e. when the answer most needs precision).
XPMATH_INLINE_FUNCTION void sincos(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    const int itrmx = 100, nq = 4;
    const float eps = 1.0e-15f;
    if (a.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    // KI-12 residual.  SMALL-ARGUMENT SHORT CIRCUIT.  The threshold is the
    // exact width of the DEGENERATE REDUCTION BAND, and nothing wider.
    //
    // The defect is not the series, it is the scaling in front of it: the
    // reduction forms r = s3/2^nq, and once |s3| < 2^nq * FLT_MIN the leading
    // word of r is SUBNORMAL, so r loses mantissa bits before the first Taylor
    // term is ever computed; for subnormal |a| it underflows to 0 outright and
    // the answer collapses.  Measured at HEAD~ : FF sin(1e-40) = 9.99967e-41
    // for an argument of 9.99995e-41 (5 digits lost), FF sin(1e-44) = 0
    // outright, and FF atan(1e-44) = 1.68e-44 against a true 9.81e-45.
    // dd_math.hpp:sincos guards the r == 0 half of this corner already; this
    // covers the shed-bits half too.
    //
    // ABOVE the band the series needs no help and must not get any.  Two wider
    // cuts were tried and both were measured to COST accuracy:
    //   * a cut at the unit roundoff u = 2^-48 -- wrong because a double-word
    //     near 1 is not a 48-bit format.  FloatFloat(1.0f, -e) is a legal
    //     normalized pair for any e down to FLT_TRUE_MIN, so cos(a) = 1 - a^2/2
    //     is EXACT while a^2/2 >= 2^-149 and the series does return it.  A
    //     2^-25 cut turned FF cos(1e-8) from 1 - 5e-17 into 1.
    //   * a cut at 2^-75, where a^2/2 first falls under FLT_TRUE_MIN.  (1, a)
    //     is indeed the correctly-rounded pair there, and for a REAL argument
    //     the series returns exactly that anyway -- r^2 underflows to zero, so
    //     sin_r = r, cos_r = 1, and the nq exact doublings reproduce (a, 1)
    //     bit for bit.  It is not a no-op for the COMPLEX callers, though:
    //     scored against the sweep grid it moved 132 complex points down by up
    //     to 1.87 digits (worst QF c asin 1398/1399, 20.07 -> 18.20) and the
    //     rest of this commit's gains stayed put.  A cut that can only ever
    //     tie on its own domain, and loses money downstream, is not worth
    //     taking; the band below is where the series genuinely cannot do
    //     better, so that is where the cut goes.
    if (detail::fabs(a.hi) < (float)(1 << nq) * 1.17549435e-38f /* 2^nq*FLT_MIN */) {
        x = FloatFloat(1.0f); y = a; return;
    }
    // KI-12 audit: |a.hi|, not a.hi.  The bare `a.hi >= 1.0e30f` made this fast
    // path fire for +1e30 and not for -1e30, which then took the full reduction
    // and fell out at the codomain guard below with the same (1, 0) — same
    // answer, two code paths, and nothing making them agree.
    if (detail::fabs(a.hi) >= 1.0e30f) {
        XPMATH_PRINTF("FFCSSNR: argument too large\n");
        x = FloatFloat(1.0f); y = FloatFloat(0.0f); return;      // KI-26
    }
    FloatFloat pi2 = multiply_scalar(FloatFloat_pi(), 2.0f);
    FloatFloat s1  = divide(a, pi2);
    FloatFloat s2  = round_to_nearest_int(s1);
    FloatFloat s3  = subtract(a, multiply(pi2, s2));
    if (s3.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    float scale = 1.0f / (float)(1 << nq);
    FloatFloat r  = multiply_scalar(s3, scale);   // r = s3 / 2^nq, |r| < pi/2^nq
    FloatFloat r2 = multiply(r, r);

    // sin(r) = r - r^3/3! + r^5/5! - ...
    // cos(r) = 1 - r^2/2! + r^4/4! - ...
    FloatFloat sin_r = r,             cos_r  = FloatFloat(1.0f);
    FloatFloat sterm = r,             cterm  = FloatFloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        // KI-25 (FF exposure, low end).  Two defects in these three lines.
        //
        // (1) `<` made the test vacuous once the series had already converged to
        //     the last bit.  For |a| <~ 1e-19 the scaled residual r is small
        //     enough that r^2 UNDERFLOWS FP32 to zero, so sterm is exactly 0 —
        //     and eps * |sin_r.hi| underflows to zero as well, making `0 < 0`
        //     false forever.  The loop then ran to itrmx on a series that had
        //     nothing left to add.  `<=` breaks on the first such iteration and
        //     changes no other outcome: the only newly-accepted case has
        //     sterm == 0 == threshold, where every remaining term is also zero.
        // (2) the itrmx arm `return`ed with x and y NEVER WRITTEN, so the caller
        //     read uninitialised storage.  That is what made FF atan(1e-30)
        //     come back NaN — angle()'s Newton step calls sincos on a tiny
        //     iterate — a codomain violation for a function bounded by pi/2.
        //     `break` falls through to the doublings and the assignments, which
        //     is what dd_math.hpp already does and what qf_math.hpp's
        //     "no return on itrmx" comment describes.
        if (detail::fabs(sterm.hi) <= eps * detail::fabs(sin_r.hi) &&
            detail::fabs(cterm.hi) <= eps) break;
        if (k == itrmx) { XPMATH_PRINTF("FFCSSNR: iteration limit\n"); break; }
    }

    // Doubling: sin(2x) = 2 sin x cos x, cos(2x) = cos^2 x - sin^2 x
    for (int j = 0; j < nq; ++j) {
        FloatFloat new_sin = multiply_scalar(multiply(sin_r, cos_r), 2.0f);
        FloatFloat new_cos = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }

    // KI-26 codomain guard: outside the slack band -> identity point, inside it
    // -> clamp, so |sin| <= 1 and |cos| <= 1 hold exactly for every finite
    // input.  Full rationale at dd_math.hpp:sincos.
    const float kSlack = 1.0009765625f;   // 1 + 2^-10
    if (!(detail::fabs(sin_r.hi) <= kSlack) || !(detail::fabs(cos_r.hi) <= kSlack)) {
        XPMATH_PRINTF("FFCSSNR: argument reduction under-determined\n");
        x = FloatFloat(1.0f); y = FloatFloat(0.0f); return;
    }
    if (sin_r.hi >  1.0f || (sin_r.hi ==  1.0f && sin_r.lo > 0.0f)) sin_r = FloatFloat( 1.0f);
    if (sin_r.hi < -1.0f || (sin_r.hi == -1.0f && sin_r.lo < 0.0f)) sin_r = FloatFloat(-1.0f);
    if (cos_r.hi >  1.0f || (cos_r.hi ==  1.0f && cos_r.lo > 0.0f)) cos_r = FloatFloat( 1.0f);
    if (cos_r.hi < -1.0f || (cos_r.hi == -1.0f && cos_r.lo < 0.0f)) cos_r = FloatFloat(-1.0f);

    x = cos_r; y = sin_r;
}

XPMATH_INLINE_FUNCTION FloatFloat sin(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return s;
}
XPMATH_INLINE_FUNCTION FloatFloat cos(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return c;
}
XPMATH_INLINE_FUNCTION FloatFloat tan(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return divide(s, c);
}

// Angle of point (x, y) = atan2(y, x)
namespace detail {
// KI-8.  Exact power-of-two scale factor: returns s = 2^-e with |m|*s in [1,2),
// so that s and 1/s are both exactly representable and scaling every word of an
// expansion by either loses no bit.
//
// e is clamped to [-125, 125] so both s and 1/s stay NORMAL floats.  At the
// clamp the operand is not brought all the way to [1,2), but it does not need
// to be: |m| <= 2^-125 scaled up by 2^125 lands at >= 2^-24, whose square's
// fourth FP32 word is 2^-120 -- still normal, which is all the caller needs;
// and |m| >= 2^125 scaled down by 2^-125 lands in [1,8), whose square is at
// most 64.
//
// No frexp: config.hpp's scalar dispatch does not carry one, and a
// dependency-free loop is portable to every device backend.  It runs only
// outside the direct band, never on the hot path.
XPMATH_INLINE_FUNCTION float ff_pow2_unit_scale(float m) {
    float t = (m < 0.0f) ? -m : m;
    if (!(t > 0.0f) || detail::isinf(t)) return 1.0f;   // 0/inf/nan: loop would not terminate
    int e = 0;
    while (t >= 16777216.0f)            { t *= 5.9604644775390625e-8f; e += 24; }
    while (t <  5.9604644775390625e-8f) { t *= 16777216.0f;            e -= 24; }
    while (t >= 2.0f) { t *= 0.5f; ++e; }
    while (t <  1.0f) { t *= 2.0f; --e; }
    if (e >  125) e =  125;
    if (e < -125) e = -125;
    return ldexpf(1.0f, -e);
}
XPMATH_INLINE_FUNCTION FloatFloat ff_pow2_scale(FloatFloat a, float s) {
    return FloatFloat(a.hi * s, a.lo * s);
}
}  // namespace detail

// Band in which a sum of squares may be formed DIRECTLY, shared by every FF
// site that forms one (hypot here; complex abs and complex operator/ in
// ff_complex.hpp).  Low edge is 2^-49, one binade above the derived 2^-51
// word-underflow limit of KI-8 note (1); the high edge is unchanged from the
// original fix and sits below sqrt(FLT_MAX) = 1.84e19.
namespace detail {
inline constexpr float kFFSqLo = 4.4408921e-16f;
inline constexpr float kFFSqHi = 1.0e18f;
}

XPMATH_INLINE_FUNCTION FloatFloat angle(FloatFloat x, FloatFloat y) {
    FloatFloat pi = FloatFloat_pi();
    if (x.hi == 0.0f && y.hi == 0.0f) return FloatFloat(0.0f);
    if (x.hi == 0.0f) return (y.hi > 0.0f) ? multiply_scalar(pi, 0.5f) : multiply_scalar(pi, -0.5f);
    if (y.hi == 0.0f) return (x.hi > 0.0f) ? FloatFloat(0.0f) : pi;
    // KI-13.  The r below is a sum of squares and has exactly hypot's exposure:
    // it overflowed the FP32 word above |x| ~ 1.8e19 (atan(3.16e19) returned
    // NaN, and atan2/asin/acos/complex arg inherited it) and shed low words
    // below the derived 2^-51 (see the KI-8 note at hypot).  angle is EXACTLY
    // scale-invariant -- atan2(ys, xs) = atan2(y, x) for any s > 0 -- so the
    // remedy here is one power-of-two rescale of both operands, which changes
    // no bit of the answer and puts the square back inside the word range.
    {
        const float mx = detail::fabs(x.hi), my = detail::fabs(y.hi);
        const float mm = (mx > my) ? mx : my;
        // HIGH side only.  The low side was tried and reverted: r is used only
        // to normalise (x/r, y/r) before a 3-step Newton refinement on
        // atan2, and that refinement re-evaluates sin/cos of the iterate, so
        // low words shed from r do not survive into the answer -- rescaling
        // there only perturbs the seed.  Measured on the 428,592-point sweep:
        // a two-sided gate cost 45 regressions (worst 1.27 digits, QF c atan
        // points 1254/1318) to buy 35 improvements.  The high side is a real
        // fix -- without it the square OVERFLOWS and atan/atan2/asin/acos/arg
        // return 0 or NaN outright.
        if (mm > detail::kFFSqHi) {
            const float s = detail::ff_pow2_unit_scale(mm);
            x = detail::ff_pow2_scale(x, s);
            y = detail::ff_pow2_scale(y, s);
        }
    }
    FloatFloat r = sqrt(add(multiply(x,x), multiply(y,y)));
    FloatFloat nx = divide(x, r), ny = divide(y, r);
    FloatFloat a = FloatFloat(detail::atan2(ny.hi, nx.hi));
    bool use_x = (detail::fabs(nx.hi) <= detail::fabs(ny.hi));
    FloatFloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        FloatFloat sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        FloatFloat corr;
        if (use_x) {
            corr = divide(subtract(target, cos_a), sin_a);
            a = subtract(a, corr);
        } else {
            corr = divide(subtract(target, sin_a), cos_a);
            a = add(a, corr);
        }
    }
    return a;
}

// KI-16.  Compare the VALUE, not the leading word.  Full derivation at
// dd_math.hpp's dd_cmp_one: `a.hi` is the value rounded to one FP32, and near a
// domain edge at +-1 that rounding crosses the edge in both directions -- it
// rejects legal arguments as far inside as 1 - 2.2e-16 (KI-16's reported FF and
// QF `atanh` symptom) and accepts illegal ones just outside.  `subtract`
// renormalises and, for |a| in [1/2, 2], cancels the leading words EXACTLY by
// Sterbenz, so the residual word decides and decides correctly.
// Returns -1, 0, +1 for a < 1, a == 1, a > 1.
XPMATH_INLINE_FUNCTION int ff_cmp_one(FloatFloat a) {
    const FloatFloat d = subtract(a, FloatFloat(1.0f));
    if (d.hi > 0.0f) return  1;
    if (d.hi < 0.0f) return -1;
    return 0;
}
XPMATH_INLINE_FUNCTION int ff_cmp_abs_one(FloatFloat a) { return ff_cmp_one(abs(a)); }

XPMATH_INLINE_FUNCTION FloatFloat asin(FloatFloat a) {
    if (ff_cmp_abs_one(a) > 0) {                                        // KI-16
        XPMATH_PRINTF("FFASIN: argument out of range\n");
        return FloatFloat(0.0f);
    }
    FloatFloat t = sqrt(subtract(FloatFloat(1.0f), multiply(a, a)));
    return angle(t, a);
}
XPMATH_INLINE_FUNCTION FloatFloat acos(FloatFloat a) {
    if (ff_cmp_abs_one(a) > 0) {                                        // KI-16
        XPMATH_PRINTF("FFACOS: argument out of range\n");
        return FloatFloat(0.0f);
    }
    FloatFloat t = sqrt(subtract(FloatFloat(1.0f), multiply(a, a)));
    return angle(a, t);
}
// KI-25 codomain clamp -- rationale in dd_math.hpp atan().  atan2 is not
// clamped; its range is (-pi, pi].
XPMATH_INLINE_FUNCTION FloatFloat atan(FloatFloat a) {
    FloatFloat r = angle(FloatFloat(1.0f), a);
    const FloatFloat p = FloatFloat_pi();
    const FloatFloat h(p.hi * 0.5f, p.lo * 0.5f);
    const FloatFloat ar = (r.hi < 0.0f) ? FloatFloat(-r.hi, -r.lo) : r;
    if (subtract(ar, h).hi > 0.0f)
        return (r.hi < 0.0f) ? FloatFloat(-h.hi, -h.lo) : h;
    return r;
}
XPMATH_INLINE_FUNCTION FloatFloat atan2(FloatFloat y, FloatFloat x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic
// ============================================================

// KI-6/KI-7: past this |a| the smaller exponential e^{-2|a|} is below FF's
// resolution u = 2^-48, so cosh == sinh == e^{|a|}/2 and tanh == ±1 exactly.
// ln(1/u)/2 = 16.6, rounded up for margin; must stay well under 44.36, where
// tanh's 2a argument would leave FP32's exp range.
constexpr float kFFHyperbolicSaturate = 19.0f;

XPMATH_INLINE_FUNCTION void sinhcosh(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    // Taylor series for |a| < 0.5 — avoids the (e^a - e^-a)/2 cancellation
    // when a is small (both exponentials approach 1, leading bits cancel).
    if (detail::fabs(a.hi) < 0.5f) {
        FloatFloat a2 = multiply(a, a);
        FloatFloat sinh_sum = a,             sinh_term = a;
        FloatFloat cosh_sum = FloatFloat(1.0f),  cosh_term = FloatFloat(1.0f);
        for (int k = 1; k <= 30; ++k) {
            sinh_term = divide_scalar(multiply(sinh_term, a2), (float)((2*k) * (2*k + 1)));
            sinh_sum  = add(sinh_sum, sinh_term);
            cosh_term = divide_scalar(multiply(cosh_term, a2), (float)((2*k - 1) * (2*k)));
            cosh_sum  = add(cosh_sum, cosh_term);
            if (detail::fabs(sinh_term.hi) < 1.0e-15f * detail::fabs(sinh_sum.hi) &&
                detail::fabs(cosh_term.hi) < 1.0e-15f) break;
        }
        x = cosh_sum; y = sinh_sum;
        return;
    }
    if (detail::fabs(a.hi) > kFFHyperbolicSaturate) {
        // See dd_math.hpp: e^{-2|a|} is below FF's resolution here, so
        // cosh == sinh == e^{|a|}/2. Halving is exact; the shifted-argument
        // form is used only when e^{|a|} overflows FP32 while cosh does not.
        //
        // KI-24.  Evaluate e^{|a|} DIRECTLY.  This branch used to compute
        // e = exp(a) on the SIGNED argument and then, for a < 0, recover
        // e^{|a|} as 1/e.  cosh is even and sinh is odd, so the sign belongs on
        // the RESULT, not on the argument of exp — and routing a < 0 through
        // exp(a) walks straight into FP32's subnormal floor.  For a = -88,
        // e^{a} = 6.06e-39 is itself subnormal (FLT_MIN = 1.18e-38) and for
        // a < -70.7 the lo word alone is subnormal, so exp(a) keeps only ~7 of
        // FF's 14 digits.  Reciprocating cannot recover what the format already
        // threw away, and the loss then lands on a result — cosh(-88) =
        // 8.26e37 — that is a perfectly healthy normal number with a normal lo
        // word.  Measured, sinh and cosh over x in [-88.7, -80]:
        //   before  11.10 (-80)  9.22 (-85)  7.86 (-87)  7.17 (-88)  7.01 (-88.7)
        //   after   14.23 (-80) 14.91 (-85) 12.77 (-87) 13.10 (-88) 13.44 (-88.7)
        // i.e. the full FP32 word KI-24 reported lost.  That the loss was an
        // artefact and not the format was already visible in this very
        // function: at a = -89, e^{a} is small enough that the isinf fallback
        // below fires and takes the exp(|a| - ln2) path instead, which scored
        // 13.79 digits while a = -88.7 next door scored 7.01.
        //
        // Crossover rather than an unconditional switch, matching the KI-9
        // floors qf_math.hpp (-40) and tf_math.hpp (-55) already carry, and
        // dd_math.hpp's (-672).  Above the floor exp(a) still occupies both
        // words and the reciprocal is healthy; the two routes then differ only
        // in the last ulp or two, which the monotone gate scores as pure
        // regression for no real gain.  Keeping the reciprocal there leaves
        // those points BIT-IDENTICAL.  FF's lo word sits at hi * 2^-24, so it
        // stays normal while exp(a) > FLT_MIN * 2^24 = 1.97e-31, i.e.
        // a > -70.7 -- the same floor ff log() derives for the same reason.
        // -75 rather than -70.7: right at the derived floor the lo word has
        // only just begun to lose bits, so the subnormal loss (0.06 digits at
        // a = -71) is still smaller than the two routes' ordinary last-ulp
        // disagreement, and switching there costs 10 sub-digit regressions.
        // Swept over the monotone gate, -75 is the value that takes every
        // available gain (50 points improved) at zero regression; -78 and -80
        // also pass but leave 10 and 20 of those gains on the table.
        const float kFFReciprocalFloor = -75.0f;
        FloatFloat aa = (a.hi < 0.0f) ? negate(a) : a;
        FloatFloat e;
        if (a.hi < kFFReciprocalFloor) {
            e = exp(aa);                        // KI-24: direct, at full width
        } else {
            e = exp(a);                         // a > 0 makes this exp(aa) too
            if (a.hi < 0.0f) e = divide(FloatFloat(1.0f), e);
        }
        FloatFloat h  = (detail::isinf(e.hi) || e.hi != e.hi) ? exp(subtract(aa, FloatFloat_log2()))
                                            : FloatFloat(e.hi * 0.5f, e.lo * 0.5f);
        x = h;
        y = (a.hi < 0.0f) ? negate(h) : h;
        return;
    }
    FloatFloat s0 = exp(a);
    FloatFloat s1 = divide(FloatFloat(1.0f), s0);
    x = multiply_scalar(add(s0, s1), 0.5f);
    y = multiply_scalar(subtract(s0, s1), 0.5f);
}

XPMATH_INLINE_FUNCTION FloatFloat sinh(FloatFloat a) {
    FloatFloat c, s; sinhcosh(a, c, s); return s;
}
XPMATH_INLINE_FUNCTION FloatFloat cosh(FloatFloat a) {
    FloatFloat c, s; sinhcosh(a, c, s); return c;
}
XPMATH_INLINE_FUNCTION FloatFloat tanh(FloatFloat a) {
    if (a.hi < 0.0f) return negate(tanh(negate(a)));
    if (a.hi > kFFHyperbolicSaturate) return FloatFloat(1.0f);  // KI-7, see dd_math.hpp
    FloatFloat e = expm1(multiply_scalar(a, 2.0f));
    return divide(e, add(e, FloatFloat(2.0f)));
}

// KI-13.  asinh and acosh both form a^2 +- 1, which leaves the FP32 word at
// |a| = sqrt(FLT_MAX) = 1.844e19 -- five decades short of the 3.4e38 the format
// reaches -- and sqrt(inf)/log(inf) then returned NaN.  Above the band the
// square is factored out instead:
//
//     asinh(a) = log(|a|) + log(1 + sqrt(1 + 1/a^2)),  sign-reflected
//     acosh(a) = log(a)   + log(1 + sqrt(1 - 1/a^2))
//
// u = 1/a^2 is formed as (1/a)^2 so nothing squares a; it is in (0,1], and for
// |a| past 1/sqrt(FLT_MIN) it simply flushes to zero, which is the correct
// limit (asinh(a) -> log(2a)).  No cancellation: log(|a|) >= 41 dominates the
// second term's log(2) ~ 0.69, and both are computed to full relative
// precision.  Below the band the original expression is kept bit-for-bit.
XPMATH_INLINE_FUNCTION FloatFloat asinh(FloatFloat a) {
    if (a.hi < 0.0f) return negate(asinh(negate(a)));
    if (a.hi > detail::kFFSqHi) {
        FloatFloat u = divide(FloatFloat(1.0f), a);
        u = multiply(u, u);
        return add(log(a), log(add(FloatFloat(1.0f), sqrt(add(FloatFloat(1.0f), u)))));
    }
    // KI-22: log1p(a + a^2/(1+sqrt(a^2+1))).  Derivation at dd_math.hpp's asinh.
    // KI-22 named only DD, but the old `1 + x` inside the log lost the same
    // u/|x| here -- one FP32 word of two at x = 1e-7.
    const FloatFloat a2 = multiply(a, a);
    const FloatFloat s  = sqrt(add(a2, FloatFloat(1.0f)));
    if (a.hi < 0.5f) {
        return log1p(add(a, divide(a2, add(FloatFloat(1.0f), s))));
    }
    // KI-29: 1/2 log1p(2a(a+s)) rather than log(a+s).  Halves the share of
    // log's constant absolute error that asinh inherits; the mid-band residual
    // KI-29 recorded is that constant, not Sterbenz.  Derived at dd_math.hpp.
    const FloatFloat t = add(a, s);
    const FloatFloat z = multiply(a, t);
    return multiply_scalar(log1p(add(z, z)), 0.5f);
}
XPMATH_INLINE_FUNCTION FloatFloat acosh(FloatFloat a) {
    if (ff_cmp_one(a) < 0) {                                            // KI-16
        XPMATH_PRINTF("FFACOSH: argument < 1\n"); return FloatFloat(0.0f); }
    if (a.hi > detail::kFFSqHi) {
        FloatFloat u = divide(FloatFloat(1.0f), a);
        u = multiply(u, u);
        return add(log(a), log(add(FloatFloat(1.0f), sqrt(subtract(FloatFloat(1.0f), u)))));
    }
    FloatFloat t1 = subtract(multiply(a, a), FloatFloat(1.0f));
    return log(add(a, sqrt(t1)));
}
XPMATH_INLINE_FUNCTION FloatFloat atanh(FloatFloat a) {
    // |a| == 1 is the C99 pole, not a domain error: atanh(+-1) = +-inf.
    const int c_atanh = ff_cmp_abs_one(a);                              // KI-16
    if (c_atanh > 0) {
        XPMATH_PRINTF("FFATANH: |argument| > 1\n"); return FloatFloat(0.0f); }
    if (c_atanh == 0) return FloatFloat(a.hi > 0.0f ? HUGE_VALF : -HUGE_VALF);
    // Taylor for |a|<0.5 avoids calling log (which loses precision when its
    // argument is close to 1). All terms positive — no cancellation.
    if (detail::fabs(a.hi) < 0.5f) {
        FloatFloat a2 = multiply(a, a);
        FloatFloat sum = a, pwr = a;
        for (int k = 1; k <= 60; ++k) {
            pwr  = multiply(pwr, a2);
            FloatFloat term = divide_scalar(pwr, (float)(2*k + 1));
            sum  = add(sum, term);
            if (detail::fabs(term.hi) < 1.0e-15f * detail::fabs(sum.hi)) break;
        }
        return sum;
    }
    // For 0.5 <= |a| < 1, log((1+a)/(1-a)) is well-conditioned (ratio >= 3).
    FloatFloat t1 = add(FloatFloat(1.0f), a);
    FloatFloat t2 = subtract(FloatFloat(1.0f), a);
    return multiply_scalar(log(divide(t1, t2)), 0.5f);
}

// ============================================================
// Multi-argument operations
// ============================================================

XPMATH_INLINE_FUNCTION FloatFloat pow(FloatFloat a, FloatFloat b) {
    if (a.hi <= 0.0f) {
        if (a.hi == 0.0f && b.hi > 0.0f) return FloatFloat(0.0f);
        XPMATH_PRINTF("FFPOW: non-positive base\n");
        return FloatFloat(0.0f);
    }
    return exp(multiply(log(a), b));
}

// hypot(a, b) = sqrt(a^2 + b^2), SCALED.  KI-8.
//
// QD 2.3.24 has no hypot (and no complex header at all), so this composition is
// original to this port: there is no upstream scaling that was dropped, and
// nothing to diverge from.
//
// The direct form squares its operands, so the intermediate a^2 leaves the
// FP32 word range at |a| ~ 1.8e19 and flushes to zero at |a| ~ 1.1e-19 -- in both
// cases while the ANSWER is perfectly representable.  hypot is precisely the
// call a caller reaches for BECAUSE it wants an overflow-safe magnitude, so it
// failing exactly there is worse than an ordinary accuracy defect.
//
// Remedy: factor the larger operand out.  t = min/max lies in [0,1], so t*t
// cannot overflow whatever the operands are, and the only scaling is the final
// multiply by m -- which overflows if and only if the true result does, so a
// genuinely unrepresentable answer still reports inf rather than a wrong finite
// value.
//
// The scaled form is NOT used unconditionally: it is gated to the range where
// the direct form actually breaks, and inside the gate the original expression
// is evaluated bit-for-bit.  Same reasoning as the atanh threshold in
// ff_complex.hpp -- fix the interval that is broken, do not churn the one that
// is not.
//
// KI-8, REOPENED 2026-09-03 -- TWO CHANGES.
//
// (1) THE GATE WAS ONE-SIDED IN PRACTICE.  Its low edge was 1.0e-18f, chosen
//     against the point where the SQUARE stops being a normal FLOAT.  That is
//     the wrong cliff for an expansion.  A W-word FP32 expansion of magnitude
//     v carries its last word at ~v*2^-24(W-1), so the square m^2 keeps all its
//     words only while m^2*2^-24(W-1) >= FLT_MIN = 2^-126, i.e.
//
//         m >= 2^-(126 - 24(W-1))/2  ->  FF (W=2): 2^-51 = 4.44e-16
//                                        TF (W=3): 2^-39 = 1.82e-12
//                                        QF (W=4): 2^-27 = 7.45e-9
//                                        DD (W=2, FP64): 2^-484.5 = 1.5e-146
//
//     Between that limit and the old 1.0e-18f edge the direct form ran and
//     silently shed the low words: measured QF hypot(1e-16, 1e-16) held 13.88
//     of 29 digits -- 51 bits, two whole FP32 words -- while the ANSWER
//     1.41e-16 sits comfortably above QF's own representation cliff of 2^-54.
//     The 2026-09-02 verification only probed 1e19..1e38 and 1e-25/1e-38, i.e.
//     above the old gate and far below it, and stepped straight over the band
//     the gate itself created.  The low edge is now the derived limit itself,
//     to the bit.  A binade of margin was tried first and cost 0.13-0.74 digits
//     at six sweep points whose magnitude landed inside the margin (QF c abs
//     7/17, c pow 13/25, c div 1/510) for no measured gain, so the margin came
//     back out.
//
// (2) THE TAIL IS UNCHANGED -- deliberately.  Scaling every word by an exact
//     power of two s = 2^-e loses no bit, and since s^2 = 2^-2e is also a power
//     of two, sqrt((as)^2 + (bs)^2) = s*sqrt(a^2 + b^2) is scale-equivariant on
//     paper, so replacing the divide-based tail with a pow2-scaled direct square
//     looks free.  It is not.  sqrt() of a square does not round-trip: when the
//     SMALLER operand is exactly zero the min/max tail returns m bit-for-bit
//     (t = 0, sqrt(1) = 1, m*1 = m) while the squared form returns sqrt(m^2),
//     which is a couple of ulps off.  Complex asin amplifies that by |z|^2.
//     Measured on the 428,592-point sweep: the pow2-squared tail cost 12.54
//     digits at QF c asin points 1628/1630 (z = +-1e10 + 0i) and 14.04 at
//     1648/1650 (z = +-1e15 + 0i), because asin routes through complex sqrt,
//     whose abs() sees (-1e20, +-0) -- exactly the zero-component case.  So the
//     tail keeps the min/max form and only the GATE moves, which is also the
//     smaller change.  Complex abs() likewise keeps its OWN squaring primitive
//     in band (multiply(); TF's hypot uses sqr()) and defers to this tail out
//     of band, exactly as before.
//
// inf/nan convention (C99 F.9.4.3): hypot(+-inf, y) is +inf for ANY y, NaN
// included, so the inf test comes first.  Otherwise a NaN operand propagates to
// NaN through the arithmetic.  Both operands are taken through abs() first, so
// the returned infinity is always +inf.

XPMATH_INLINE_FUNCTION FloatFloat hypot(FloatFloat a, FloatFloat b) {
    FloatFloat x = abs(a);
    FloatFloat y = abs(b);
    if (detail::isinf(x.hi)) return x;
    if (detail::isinf(y.hi)) return y;
    FloatFloat m = (x.hi < y.hi) ? y : x;
    FloatFloat n = (x.hi < y.hi) ? x : y;
    if (m.hi == 0.0f) return FloatFloat(0.0f);
    if (m.hi <= detail::kFFSqHi && m.hi >= detail::kFFSqLo)
        return sqrt(add(multiply(a, a), multiply(b, b)));
    FloatFloat t = divide(n, m);
    return multiply(m, sqrt(add(FloatFloat(1.0f), multiply(t, t))));
}

XPMATH_INLINE_FUNCTION FloatFloat ceil(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat floor(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat trunc(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloat round(FloatFloat a);

// fmod / remainder — exact iterative scale-and-subtract (KI-10, KI-15).
// Algorithm, loop bound, termination and exactness argument: dd_math.hpp,
// which carries the full derivation. FF is the backend where KI-15(a) bit
// hardest — `trunc(q)` returns 0 for |q| >= 2^47 (KI-14), so the old body
// returned the unreduced dividend. No quotient is formed here at all.
//
// FF DIFFERS FROM THE OTHER THREE BACKENDS IN ONE RESPECT, and it matters.
// The dd_math.hpp argument shows the running remainder never needs more than
// (word mantissa) + 2 bits of span. For DD that is 55 of the 106 DD carries,
// for QF 26 of 96, for TF 26 of 72 — all comfortable. For FF it is ~50 and FF
// carries 48, so the exact remainder is 1-2 bits too wide to hold, and a
// two-word loop rounds at every step. Measured on a 1e15 ratio: 5 of the 50
// steps were inexact and fmod came out at 10.93 of 14 digits. The loop
// therefore runs the remainder in THREE floats (72 bits) and rounds to a
// FloatFloat once, at the end. That is the only place FF pays for its narrow
// pair; the cost is ~30 extra flops per step and it buys back 3 digits.
//
// Conventions: C99 fmod (sign of a) and C99/IEEE-754 remainder (round-half-
// to-EVEN quotient, |r| <= |b|/2). fmod(a,0), remainder(a,0), fmod(+-inf,b),
// remainder(+-inf,b) -> NaN; f(a,+-inf) = a for finite a; f(+-0,b) = +-0.
namespace detail {

XPMATH_INLINE_FUNCTION FloatFloat ff_scale2(FloatFloat a, float s) {
    return FloatFloat(a.hi * s, a.lo * s);
}

// r[0..2] (nonoverlapping, decreasing) -= (Bs.hi + Bs.lo), exactly.
// The five terms are already in near-decreasing order, so four carry-
// propagating two-sum sweeps leave the expansion nonoverlapping; the two
// tail terms are zero whenever the true difference fits in three floats,
// which the span argument above guarantees at every step of the loop.
XPMATH_INLINE_FUNCTION void ff_sub3(float* r, FloatFloat Bs) {
    float t[5] = {r[0], -Bs.hi, r[1], -Bs.lo, r[2]};
    for (int pass = 0; pass < 4; ++pass) {
        for (int i = 4; i >= 1; --i) {
            float u = t[i - 1] + t[i];
            float v = u - t[i - 1];
            t[i]     = (t[i - 1] - (u - v)) + (t[i] - v);
            t[i - 1] = u;
        }
    }
    r[0] = t[0]; r[1] = t[1]; r[2] = (t[2] + t[3]) + t[4];
}

// r[0..2] >= Bs ?
XPMATH_INLINE_FUNCTION bool ff_ge3(const float* r, FloatFloat Bs) {
    if (r[0] != Bs.hi) return r[0] > Bs.hi;
    if (r[1] != Bs.lo) return r[1] > Bs.lo;
    return r[2] >= 0.0f;
}

// |A| mod |B|, reduced exactly. half_even = false gives the C99 fmod result in
// [0,B); true gives the IEEE-754 remainder result in [-B/2,+B/2].
// Precondition: A >= 0, B > 0, both finite.
XPMATH_INLINE_FUNCTION FloatFloat ff_reduce_abs(FloatFloat A, FloatFloat B,
                                                bool half_even) {
    float r[3] = {A.hi, A.lo, 0.0f};
    bool q_odd = false;

    if (A >= B) {
        // Ladder: smallest k with B*2^k <= A < B*2^(k+1). Coarse 2^30 rungs
        // first; an overflowed rung compares greater than A and is not taken.
        FloatFloat Bs = B;
        int k = 0;
        for (;;) {
            FloatFloat t = ff_scale2(Bs, 1073741824.0f);  // 2^30
            if (t > A) break;
            Bs = t; k += 30;
        }
        for (;;) {
            FloatFloat t = ff_scale2(Bs, 2.0f);
            if (t > A) break;
            Bs = t; ++k;
        }
        for (int i = k; i >= 0; --i) {
            if (ff_ge3(r, Bs)) {
                ff_sub3(r, Bs);               // Sterbenz-exact in 3 floats
                if (i == 0) q_odd = true;
            }
            Bs = ff_scale2(Bs, 0.5f);
        }
    }

    if (half_even) {
        // r is in [0,B). Step to r-B (quotient n+1) when that is strictly
        // closer, and on the exact tie 2r == B when n is odd.
        float two_r[3] = {r[0] * 2.0f, r[1] * 2.0f, r[2] * 2.0f};
        bool ge = ff_ge3(two_r, B);
        bool eq = (two_r[0] == B.hi && two_r[1] == B.lo && two_r[2] == 0.0f);
        if ((ge && !eq) || (eq && q_odd)) ff_sub3(r, B);
    }

    // Round the three-word remainder into a FloatFloat, once.
    float s = r[1] + r[2];
    float hi = r[0] + s;
    return FloatFloat(hi, s - (hi - r[0]));
}

}  // namespace detail

XPMATH_INLINE_FUNCTION FloatFloat fmod(FloatFloat a, FloatFloat b) {
    if (a.hi != a.hi || b.hi != b.hi) return FloatFloat(a.hi + b.hi);
    if (b.hi == 0.0f) { XPMATH_PRINTF("FFFMOD: zero modulus\n");
                        return FloatFloat(0.0f / 0.0f); }
    if (!detail::isfinite(a.hi)) { XPMATH_PRINTF("FFFMOD: infinite dividend\n");
                                   return FloatFloat(0.0f / 0.0f); }
    if (!detail::isfinite(b.hi)) return a;
    if (a.hi == 0.0f) return a;

    FloatFloat r = detail::ff_reduce_abs(abs(a), abs(b), false);
    return (a.hi < 0.0f) ? negate(r) : r;
}

XPMATH_INLINE_FUNCTION FloatFloat remainder(FloatFloat a, FloatFloat b) {
    if (a.hi != a.hi || b.hi != b.hi) return FloatFloat(a.hi + b.hi);
    if (b.hi == 0.0f) { XPMATH_PRINTF("FFREMAINDER: zero modulus\n");
                        return FloatFloat(0.0f / 0.0f); }
    if (!detail::isfinite(a.hi)) { XPMATH_PRINTF("FFREMAINDER: infinite dividend\n");
                                   return FloatFloat(0.0f / 0.0f); }
    if (!detail::isfinite(b.hi)) return a;
    if (a.hi == 0.0f) return a;

    FloatFloat r = detail::ff_reduce_abs(abs(a), abs(b), true);
    return (a.hi < 0.0f) ? negate(r) : r;
}

XPMATH_INLINE_FUNCTION FloatFloat copysign(FloatFloat a, FloatFloat b) {
    FloatFloat r = abs(a);
    if (b.hi < 0.0f || (b.hi == 0.0f && b.lo < 0.0f)) return negate(r);
    return r;
}

XPMATH_INLINE_FUNCTION FloatFloat fmax(FloatFloat a, FloatFloat b) {
    return (a > b) ? a : b;
}
XPMATH_INLINE_FUNCTION FloatFloat fmin(FloatFloat a, FloatFloat b) {
    return (a < b) ? a : b;
}
XPMATH_INLINE_FUNCTION FloatFloat fdim(FloatFloat a, FloatFloat b) {
    return (a > b) ? subtract(a, b) : FloatFloat(0.0f);
}
XPMATH_INLINE_FUNCTION FloatFloat fma(FloatFloat a, FloatFloat b, FloatFloat c) {
    return add(multiply(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

XPMATH_INLINE_FUNCTION FloatFloat floor(FloatFloat a) {
    FloatFloat n = round_to_nearest_int(a);
    if (n > a) return subtract(n, FloatFloat(1.0f));
    return n;
}
XPMATH_INLINE_FUNCTION FloatFloat ceil(FloatFloat a) {
    FloatFloat n = round_to_nearest_int(a);
    if (n < a) return add(n, FloatFloat(1.0f));
    return n;
}
XPMATH_INLINE_FUNCTION FloatFloat trunc(FloatFloat a) {
    return (a.hi >= 0.0f) ? floor(a) : ceil(a);
}
// round(a) — nearest integer, TIES TO EVEN (IEEE 754 roundToIntegralTiesToEven).
// KI-20 (2026-09-04); the convention and its divergence from C99 `round` and
// QD's `nint` are written out at dd_math.hpp's `round`. FF's
// round_to_nearest_int is `rint` on an exact FP64 reassembly and is already
// half-even, so no tie correction is needed here either.
XPMATH_INLINE_FUNCTION FloatFloat round(FloatFloat a) {
    return round_to_nearest_int(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

// Internal helper (not part of the FF API surface): the SUM of the asymptotic
// erfc expansion, A&S 7.1.23,
//     erfc(z) = e^{-z^2}/sqrt(pi) * sum_k term_k,
//     term_0 = 1/|z|,  term_k = term_{k-1} * (-(2k-1))/(2 z^2),
// evaluated with optimal truncation. The series is DIVERGENT: its terms shrink
// to ~e^{-z^2} and then grow again, so summing past the smallest term strictly
// loses accuracy and a relative-eps exit can never be the primary one. Both the
// eps exit and the k cap are secondary; the smallest-term test below is primary.
// At FP32 the k cap is also a hard overflow guard — see B5's note in erf().
//
// Only the SUM is returned, not erfc: the two callers need different scalings.
// erf() (B5) sees only |z| in [3.5, 6], where e^{z^2} <= 4.3e15, and divides by
// it; erfc() (B6) runs out to the FP32 underflow floor |z| ~ 10.05, where e^{z^2}
// ~ 7.3e43 would first overflow the Dekker splitter (|z| ~ 8.93) and then trip
// exp()'s hard |arg| >= 88 guard (|z| ~ 9.38) — so erfc() multiplies by e^{-z^2}
// instead. Keeping the scaling at the call sites lets erf()'s arithmetic stay
// bit-for-bit what B5 shipped while erfc() gets the overflow-safe form.
//
// Preconditions: az = |z| > 0 and z2 = z*z, both finite.
XPMATH_INLINE_FUNCTION FloatFloat erfc_asymptotic_sum(FloatFloat az, FloatFloat z2) {
    const float eps = 1.0e-14f;   // ~ FF relative resolution (2^-46); finer never fires
    FloatFloat two_z2 = multiply_scalar(z2, 2.0f);
    FloatFloat term = divide(FloatFloat(1.0f), az), sum = term;
    float prev_mag = detail::fabs(term.hi);
    for (int k = 1; k <= 60; ++k) {
        FloatFloat next = divide(multiply_scalar(term, -(2.0f*k - 1.0f)), two_z2);
        float mag = detail::fabs(next.hi);
        if (mag > prev_mag) break;                 // smallest term reached -> stop
        sum = add(sum, next);
        term = next; prev_mag = mag;
        if (mag <= eps * detail::fabs(sum.hi)) break;
    }
    return sum;
}

XPMATH_INLINE_FUNCTION FloatFloat erf(FloatFloat z) {
    // B5 (FF): repair both erf branches for FP32's narrow exponent range. The DD
    // port (dd_math.hpp:670) computes each Taylor/asymptotic term from a separately
    // accumulated numerator (t2) and denominator (t3). At FP64 those intermediates
    // stay finite; at FP32 (max ~3.4e38) they overflow long before the loop's fixed
    // 60-iteration cap, so erf returned NaN across the whole smooth range ~[1.9, 6]
    // (probe: [2,6) was 100% NaN). Two independent failures were compounding:
    //   (a) Taylor branch: t2 = 2^k*z^{2k+1} and t3 = (2k+1)!! each overflow FP32
    //       around k~26-31 for |z| in [2,4); the loop never converged first.
    //   (b) Asymptotic branch: the erfc asymptotic series is *divergent*, so the
    //       relative-eps test never fired; the loop ran to k=60 where (2k-1)!!
    //       overflows FP32 -> inf/inf = NaN. (This is the FF sibling of DD's B3.)
    // Fix: (1) accumulate each term from the previous via its running ratio, which
    // keeps every intermediate O(term) and thus finite; (2) truncate the divergent
    // asymptotic series at its smallest term (standard optimal truncation); and
    // (3) lower the Taylor->asymptotic switchover to 3.5 (see kTaylorMax below) so
    // each branch is used only where it is both convergent and overflow-safe.
    // Series identities: A&S 7.1.6 (Taylor) and A&S 7.1.23 (asymptotic erfc).
    const float eps = 1.0e-14f; // ~ FF relative resolution (2^-46); finer never fires
    if (z.hi == 0.0f) return FloatFloat(0.0f);
    const float large = 6.0f; // erfc(6) ~= 2e-17 << FF resolution -> erf saturates
    if (z.hi >  large) return FloatFloat( 1.0f);
    if (z.hi < -large) return FloatFloat(-1.0f);

    FloatFloat z2 = multiply(z, z);
    int sign = (z.hi >= 0.0f) ? 1 : -1;
    FloatFloat az = abs(z);

    // Switchover derivation (probe over |z| in [1,6], incremental recurrence):
    // Taylor stays >= FF precision and converges within the iteration cap up to
    // |z| ~ 4; the asymptotic erf (erf = 1 - tiny erfc) clears the 8.45-digit gate
    // for |z| >~ 3. 3.5 sits inside both windows with margin.
    const float kTaylorMax = 3.5f;

    if (detail::fabs(z.hi) < kTaylorMax) {
        // Taylor (A&S 7.1.6): erf(z) = (2/sqrt(pi)) e^{-z^2} sum_k term_k,
        //   term_0 = |z|,  term_k = term_{k-1} * (2 z^2) / (2k+1).
        // All terms positive; the running ratio keeps each O(<= sum) so no FP32
        // overflow (sum <= (sqrt(pi)/2) e^{z^2}, ~3.8e15 at |z|=6).
        FloatFloat two_z2 = multiply_scalar(z2, 2.0f);
        FloatFloat term = az, sum = az;
        for (int k = 1; k <= 100; ++k) {
            term = divide_scalar(multiply(term, two_z2), 2.0f*k + 1.0f);
            FloatFloat sumnew = add(sum, term);
            if (detail::fabs(term.hi) <= eps * detail::fabs(sumnew.hi)) { sum = sumnew; break; }
            sum = sumnew;
        }
        FloatFloat result = divide(multiply_scalar(sum, 2.0f),
                                   multiply(sqrt(FloatFloat_pi()), exp(z2)));
        return (sign > 0) ? result : negate(result);
    } else {
        // Asymptotic (A&S 7.1.23), optimal truncation — see erfc_asymptotic_sum()
        // above, which B6 factored out of this branch so erfc() can invoke the
        // identical series directly. The scaling stays here: |z| <= 6 on this
        // path, so e^{z^2} <= 4.3e15 and the divide is safe (erfc()'s
        // multiply-by-e^{-z^2} form is the one that has to reach |z| ~ 10). Then
        // erf = 1 - erfc; erfc is ~7.4e-7 at the |z| = 3.5 seam and smaller above,
        // so that subtraction is benign for erf (it is NOT for erfc itself — see
        // erfc() below).
        FloatFloat sum = erfc_asymptotic_sum(az, z2);
        FloatFloat erfc_val = divide(sum, multiply(sqrt(FloatFloat_pi()), exp(z2)));
        FloatFloat erf_val  = subtract(FloatFloat(1.0f), erfc_val);
        return (sign > 0) ? erf_val : negate(erf_val);
    }
}

// erfc — direct asymptotic expansion for z >= 5.75, 1 - erf(z) below that.
//
// B6: `erfc(z) = subtract(FloatFloat(1.0f), erf(z))` for ALL z is catastrophic
// cancellation as erf(z) -> 1. erf is accurate to u^2 RELATIVE to 1, so the
// difference carries an absolute error ~u^2 and erfc's relative error is
// u^2/erfc(z) — a loss of exactly log10(1/erfc(z)) digits, by construction, at
// every precision. The FP32 flavour of the DD sibling (B2, dd_math.hpp:erfc).
//
// What FF does differently from DD is the TOP of the range. DD's erf saturates
// at |z| = 8.5, well past where its asymptotic branch (live from 6.0) has pushed
// erfc's answer into erf's lo word, so DD's shipped erfc degraded to a flat
// ~16-digit shelf. FF's erf saturates to exactly ±1 at |z| > 6.0, only 2.5 above
// its own 3.5 switchover, so FF's shipped erfc does not get a shelf — it falls
// off a CLIFF: measured 7.60 digits at z = 6.00 and exactly 0.00 (erfc returns
// +0, relative error 1) at every z >= 6.02. That cliff, not a slow ramp, is what
// pinned the ff_accuracy_test erfc row's min at -0.00.
//
// Fix: for z >= kDirectMin, evaluate erfc directly from the SAME asymptotic
// series erf() uses (erfc_asymptotic_sum, A&S 7.1.23) and never form 1 - erf.
// Negative z needs no such path: erfc(-|z|) = 2 - erfc(|z|) is ~2, so
// subtract(1, erf) is benign there and already scores the 14-digit report cap.
//
// Threshold derivation. Same rule as B2: the cut goes where the direct series
// stops being worse than the fallback it replaces at EVERY measured point, not
// on average. Below the cut the fallback wins outright — the series' optimal-
// truncation floor is only 4.04 digits at z = 3.0 and 5.46 at z = 3.5, against
// the fallback's ~8 there. The two curves cross around z ~ 4.0, but the
// fallback's value is pure roundoff LUCK (its digit count is set by where
// erfc(z) happens to land in erf's 24-bit lo word), so it scatters: over
// [5.4, 6.0] its mean is 7.80 while individual points spike to 13.40. The direct
// series is smooth and climbs ~4 digits per unit z through that scatter band
// (7.09 at z = 4.0, 11.00 at 5.0, 13.18 at 5.5, then the 14-digit report cap).
// Enumerating EVERY representable float (not a grid — the fallback's scatter is
// per-float) over [5.4, 6.001], 1,260,389 inputs, exactly 2 regress, the highest
// at z = 5.6338043 by 0.106 digit. kDirectMin = 5.75 sits 0.116 above that, and
// over all 7,235,176 representable floats in [5.7, 10.3] NO input scores worse
// than the shipped 1 - erf code did.
//
// Rejected alternative — kDirectMin = 4.9, which a 0.00005-spaced grid says is
// clean (1 regressing point in 35,001 over [4.85, 6.6], worth 0.020 digit). It
// is not: exhaustive float enumeration finds the fallback's lucky spikes that a
// grid steps over. Grid sampling is the wrong instrument for a function whose
// error is roundoff-scattered; B2 used a 0.0005 grid because DD's fallback had a
// genuine 53-bit shelf with a narrow envelope, which FF's 24-bit lo word does
// not give.
//
// The pointwise rule is NOT free here, unlike at DD. B2 could apply it at no
// cost because DD's row mean was flat to +/-0.03 digit across its whole
// candidate window. FF's is not: the predicted ff_accuracy_test random-domain
// mean rises monotonically with a lower cut — 11.84 with no direct path, 11.96
// at 5.75, 12.26 at 5.0, 12.36 at 4.5-4.0 (the plateau), 12.21 at 3.0. So 5.75
// leaves ~0.40 digit of mean on the table versus the mean-optimal ~4.0. That is
// the deliberate trade: 4.0 regresses 814 of 6201 measured points by up to 1.93
// digits, and a measured regression is not worth 0.40 of a mean that already
// clears its gate by 3.5.
//
// KNOWN LIMITATION, not closed here (inherited verbatim from B2 DEV1). Below
// the cut the 1 - erf cancellation is a ramp, not a cliff, so it costs digits
// well under any usable threshold for THIS series. Exhaustive per-float window
// means against a quadmath oracle: [2, 3] 10.32, [3, 3.2] 8.77, [3.2, 3.5] 8.03,
// [3.5, 4] 6.26, [4, 4.5] 7.69, [4.5, 5] 7.81, [5, 5.5] 7.81, [5.5, 5.75] 7.81 —
// i.e. under the 8.45 test gate at scattered points from z ~ 2.94 and at
// substantially every point from z ~ 3.2 up to kDirectMin. A direct asymptotic
// path cannot help there: its optimal-truncation floor is 4.04 digits at z = 3.0
// and 5.46 at z = 3.5, at or below the subtract it would replace. (The global
// trough, 5.459 digits at z = 3.5, is not cancellation at all — it is erf()'s own
// Taylor->asymptotic seam, where the two paths return the identical value.)
// Closing that band needs a different algorithm (a Lentz continued fraction,
// A&S 7.1.14, or a triple-float erf). The ff_accuracy_test row gates on the MEAN
// and passes at 11.97 vs 8.45; the pointwise band is a separate, open concern.
XPMATH_INLINE_FUNCTION FloatFloat erfc(FloatFloat z) {
    // See derivation above. Sits below erf()'s own |z| > 6.0 saturation, so the
    // cliff that saturation creates is never reached through erfc.
    const float kDirectMin = 5.75f;
    // Above this, erfc(z) rounds to +0 in FP32 and +0 is the only representable
    // answer, so honest underflow beats evaluating the series. Derivation:
    // erfc(z) ~ e^{-z^2}/(z*sqrt(pi)) crosses FLT_TRUE_MIN/2 = 7.0e-46 at
    // z ~ 10.05 (oracle: erfc(10.04) = 9.3e-46, erfc(10.06) = 6.2e-46), and the
    // series below independently first returns exactly 0 at z = 10.04. The clamp
    // also keeps z*z away from the range where the corpus (which feeds erfc
    // FLT_MAX) would make z2 = +inf and poison the series into NaN, and where
    // exp() would print its "argument too large" diagnostic.
    const float kUnderflowMax = 10.05f;
    if (z.hi >= kDirectMin) {
        if (!(z.hi <= kUnderflowMax)) return FloatFloat(0.0f);  // also catches +inf/NaN
        FloatFloat z2 = multiply(z, z);
        FloatFloat sum = erfc_asymptotic_sum(z, z2);
        // e^{-z^2}, not 1/e^{z^2}: forming sqrt(pi)*e^{z^2} and dividing by it
        // failed twice over when this was written. (a) The Dekker splitter
        // overflowed: multiply() computes conb = b.hi*8193 with b.hi = e^{z^2},
        // which passes FLT_MAX/(8193+1) = 4.15e34 at z = 8.93, giving
        // inf - inf = NaN. That was B8 / PORT_NOTES §4d's bug at multiply()'s
        // splitter, which B8 did not reach. It is FIXED as of B9 / §4j, which
        // scaled the splitter at multiply() and its siblings — so (a) no longer
        // bites, and this sighting is in fact what motivated B9. (b) still
        // stands on its own, and is sufficient by itself to keep this form: ff
        // exp() hits its hard |arg| >= 88 guard (FP32's finite range) at
        // z = 9.38 and returns 0 AND prints "FFEXP: argument too large", so the
        // quotient is 1/0. Measured pre-B9: the divide form returned NaN for
        // every z >= 8.95 and printed from z >= 9.38; post-B9 only the printing
        // and the 1/0 remain. The multiply form has neither failure and reaches
        // the underflow floor, so it stays.
        FloatFloat emz2;
        if (z2.hi < 88.0f) {
            emz2 = exp(negate(z2));
        } else {
            // Same |arg| >= 88 guard: quarter the argument (max |z^2/4| = 25.25
            // at kUnderflowMax) and square twice. Costs ~2 bits of exp's relative
            // error, which is moot here — every z past 9.38 has e^{-z^2} below
            // FLT_MIN, so the result is subnormal and losing bits anyway.
            emz2 = exp(divide_scalar(negate(z2), 4.0f));
            emz2 = multiply(emz2, emz2);
            emz2 = multiply(emz2, emz2);
        }
        return divide(multiply(sum, emz2), sqrt(FloatFloat_pi()));
    }
    return subtract(FloatFloat(1.0f), erf(z));
}

// gamma — Lanczos approximation
XPMATH_INLINE_FUNCTION FloatFloat tgamma(FloatFloat a) {
    if (a.hi < 0.5f) {
        FloatFloat pi = FloatFloat_pi();
        FloatFloat sin_pi_a = sin(multiply(pi, a));
        return divide(pi, multiply(sin_pi_a, tgamma(subtract(FloatFloat(1.0f), a))));
    }
    // B7: Lanczos g=7 coefficients promoted from `float` to `double`. Stored as
    // `float` literals, each coefficient was truncated to FP32's ~7-digit ceiling,
    // capping tgamma at ~6 digits regardless of the enclosing FF arithmetic (a
    // mechanical DD->FF port artifact: the `double` literals of dd_math.hpp:730-738
    // were erroneously given `f` suffixes). As `double`, each FloatFloat(cN) call
    // below invokes the FloatFloat(double) constructor (ff_math.hpp:94), which splits
    // to a full FF pair (hi=(float)d, lo=(float)(d-(double)hi)) — ~14 digits, the FF
    // resolution floor. double (53-bit) exceeds FF (48-bit), so the split is
    // FF-exact. Coefficient set: Godfrey g=7 (P. Godfrey 2001; same values as
    // dd_math.hpp's DD tgamma / Boost.Math / Wikipedia "Lanczos approximation").
    const double c0 =  0.99999999999980993;
    const double c1 =  676.5203681218851;
    const double c2 = -1259.1392167224028;
    const double c3 =  771.32342877765313;
    const double c4 = -176.61502916214059;
    const double c5 =  12.507343278686905;
    const double c6 = -0.13857109526572012;
    const double c7 =  9.9843695780195716e-6;
    const double c8 =  1.5056327351493116e-7;
    FloatFloat x = subtract(a, FloatFloat(1.0f));
    FloatFloat t = add(x, FloatFloat(7.5f));
    FloatFloat s = FloatFloat(c0);
    s = add(s, divide(FloatFloat(c1), add(x, FloatFloat(1.0f))));
    s = add(s, divide(FloatFloat(c2), add(x, FloatFloat(2.0f))));
    s = add(s, divide(FloatFloat(c3), add(x, FloatFloat(3.0f))));
    s = add(s, divide(FloatFloat(c4), add(x, FloatFloat(4.0f))));
    s = add(s, divide(FloatFloat(c5), add(x, FloatFloat(5.0f))));
    s = add(s, divide(FloatFloat(c6), add(x, FloatFloat(6.0f))));
    s = add(s, divide(FloatFloat(c7), add(x, FloatFloat(7.0f))));
    s = add(s, divide(FloatFloat(c8), add(x, FloatFloat(8.0f))));
    // B7: sqrt(2*pi) leading factor — likewise `double` (was `2.5...f`), so it
    // splits to a full FF pair instead of capping the whole product at ~7 digits.
    FloatFloat two_pi_sqrt = FloatFloat(2.5066282746310002);
    return multiply(multiply(two_pi_sqrt, s),
                 multiply(pow(t, add(x, FloatFloat(0.5f))), exp(negate(t))));
}

// Bessel J0 via series
XPMATH_INLINE_FUNCTION FloatFloat bessel_j0(FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat x2 = multiply_scalar(multiply(x, x), -0.25f);
    FloatFloat term = FloatFloat(1.0f), sum = FloatFloat(1.0f);
    for (int k = 1; k <= 60; ++k) {
        term = divide_scalar(multiply(term, x2), (float)(k*k));
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < eps * detail::fabs(sum.hi)) break;
    }
    return sum;
}

XPMATH_INLINE_FUNCTION FloatFloat bessel_j1(FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat x2 = multiply_scalar(multiply(x, x), -0.25f);
    FloatFloat term = multiply_scalar(x, 0.5f), sum = term;
    for (int k = 1; k <= 60; ++k) {
        term = divide_scalar(multiply(term, x2), (float)(k * (k+1)));
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < eps * detail::fabs(sum.hi)) break;
    }
    return sum;
}

XPMATH_INLINE_FUNCTION FloatFloat bessel_jn(int n, FloatFloat x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    FloatFloat j0 = bessel_j0(x), j1 = bessel_j1(x);
    FloatFloat jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        FloatFloat jp1 = subtract(multiply_scalar(divide(j_cur, x), 2.0f*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

XPMATH_INLINE_FUNCTION FloatFloat bessel_y0(FloatFloat x) {
    FloatFloat two_over_pi = divide_scalar(FloatFloat(2.0f), FloatFloat_pi().hi);
    FloatFloat j0 = bessel_j0(x);
    return multiply(two_over_pi, multiply(j0, log(multiply_scalar(x, 0.5f))));
}
XPMATH_INLINE_FUNCTION FloatFloat bessel_y1(FloatFloat x) {
    FloatFloat two_over_pi = divide_scalar(FloatFloat(2.0f), FloatFloat_pi().hi);
    FloatFloat j1 = bessel_j1(x);
    return multiply(two_over_pi, multiply(j1, log(multiply_scalar(x, 0.5f))));
}
XPMATH_INLINE_FUNCTION FloatFloat bessel_yn(int n, FloatFloat x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    FloatFloat y0 = bessel_y0(x), y1 = bessel_y1(x);
    FloatFloat ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        FloatFloat yp1 = subtract(multiply_scalar(divide(y_cur, x), 2.0f*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

XPMATH_INLINE_FUNCTION FloatFloat zeta(FloatFloat s) {
    if (ff_cmp_one(s) <= 0) {                                           // KI-16
        XPMATH_PRINTF("FFZETA: s <= 1\n"); return FloatFloat(0.0f); }
    const int N = 30;
    FloatFloat sum = FloatFloat(0.0f);
    for (int k = 1; k <= N; ++k)
        sum = add(sum, exp(multiply(negate(s), log(FloatFloat((float)k)))));
    FloatFloat tail = divide(exp(multiply(subtract(FloatFloat(1.0f), s), log(FloatFloat((float)N)))),
                         subtract(s, FloatFloat(1.0f)));
    return add(sum, tail);
}

XPMATH_INLINE_FUNCTION FloatFloat expint(FloatFloat x) {
    FloatFloat eg = FloatFloat_euler_gamma();
    FloatFloat sum = add(eg, log(abs(x)));
    FloatFloat term = x;
    for (int k = 1; k <= 60; ++k) {
        sum = add(sum, divide_scalar(term, (float)(k * k)));
        term = multiply(term, x);
        if (detail::fabs(term.hi) * 1e-15f < detail::fabs(sum.hi)) break;
    }
    return sum;
}

XPMATH_INLINE_FUNCTION FloatFloat incgamma(FloatFloat a, FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat term = divide(exp(negate(x)), a);
    FloatFloat sum  = term;
    for (int k = 1; k <= 60; ++k) {
        term = multiply(term, divide(x, add(a, FloatFloat((float)k))));
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < eps * detail::fabs(sum.hi)) break;
    }
    return multiply(sum, exp(multiply(a, log(x))));
}


} // namespace xp
