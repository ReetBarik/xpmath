// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// Ported from DDFUN v04:
//   https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz
//   Original author: David H. Bailey (LBNL retired / UC Davis)
//   Original license: DHB-License (modified BSD-3-Clause with §3
//     grant-back clause). Full text: LICENSES/LicenseRef-DHB-License.txt
//     or https://www.davidhbailey.com/dhbsoftware/DHB-License.txt.
//
// This C++/Kokkos port is a derivative work distributed under the
// same DHB-License. See §3 of that license regarding upstream
// contribution rights. This is the complex layer; it builds on the
// float-float real arithmetic in ff_math.hpp (also DHB-License).
//
// Modifications from the original DDFUN v04 sources:
//   * Translated the complex float-float routines from Fortran-90
//     (ffc* entry points, mechanical port of ddc* code) to header-only C++17.
//   * Every function XPMATH_INLINE_FUNCTION for host + device
//     portability across CUDA/HIP/SYCL/OpenMP-target.
//   * Namespaced as xp::FloatFloatComplex (a bespoke struct,
//     not yet Kokkos::complex<FloatFloat>) with STL-style
//     free functions and ADL-friendly re-exposure under the
//     Kokkos-compat wrapper for potential upstreaming to Kokkos.
//   * See docs/TEST_SUITE_PLAN.md "Upstreaming considerations" for
//     naming and API conventions.

#pragma once

// Float-float complex arithmetic — xp::FloatFloatComplex.
// All functions XPMATH_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on ff_math.hpp.
//
// Ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab).
//
// DEPENDENCIES: none beyond the C++17 standard library and ff_math.hpp.
// In particular this header does NOT include or require Kokkos — see
// xp/config.hpp for how the portability facilities are supplied. Kokkos
// users get today's `Kokkos::Experimental::FloatFloatComplex` API
// unchanged through the compat wrapper at third_party/include/ff_complex.hpp,
// which is the only place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// Naming follows ff_math.hpp (T0.4): type + math live under
// xp:: for eventual upstreaming. This remains a bespoke struct
// rather than Kokkos::complex<FloatFloat> — that integration is a separate
// future task.

#include <xp/ff_math.hpp>

#if !defined(XPMATH_ON_DEVICE)
#  include <ostream>
#endif

namespace xp {

// ============================================================
// FloatFloatComplex struct
// ============================================================
struct FloatFloatComplex {
    FloatFloat re;
    FloatFloat im;

    XPMATH_INLINE_FUNCTION FloatFloatComplex() : re(0.0f), im(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplex(float r)               : re(r),    im(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplex(FloatFloat r)              : re(r),    im(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplex(float r, float i)      : re(r),    im(i)    {}
    XPMATH_INLINE_FUNCTION FloatFloatComplex(FloatFloat r, FloatFloat i)    : re(r),    im(i)    {}
    XPMATH_INLINE_FUNCTION FloatFloatComplex(const FloatFloatComplex& o)    : re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplex& operator=(const FloatFloatComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    XPMATH_INLINE_FUNCTION FloatFloatComplex& operator=(FloatFloat r) {
        re = r; im = FloatFloat(0.0f); return *this;
    }

    XPMATH_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloatComplex b) const {
        return FloatFloatComplex(add(re, b.re), add(im, b.im));
    }
    XPMATH_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloatComplex b) const {
        return FloatFloatComplex(subtract(re, b.re), subtract(im, b.im));
    }
    // KI-28.  Annex G.5.1 recovery plus the finite-operand overflow case Annex G
    // does not cover.  Full derivation at dd_complex.hpp's mul_recover.  The FP32
    // scale is 2^-65: |a|,|b| <= 2^128 gives a scaled product <= 2^126 and a sum
    // of two of those <= 2^127.
    static XPMATH_INLINE_FUNCTION FloatFloat scale2(FloatFloat v, float s) {
        FloatFloat r(v.hi * s, v.lo * s);
        if (r.hi != r.hi || detail::isinf(r.hi)) return FloatFloat(r.hi);
        return r;
    }
    static XPMATH_INLINE_FUNCTION FloatFloatComplex
    mul_recover(FloatFloatComplex a, FloatFloatComplex b,
                FloatFloat rr, FloatFloat ri) {
        const float ar = a.re.hi, ai = a.im.hi, br = b.re.hi, bi = b.im.hi;
        if (ar != ar || ai != ai || br != br || bi != bi)
            return FloatFloatComplex(rr, ri);        // NaN in, NaN out

        float nar = ar, nai = ai, nbr = br, nbi = bi;
        bool recalc = false;
        if (detail::isinf(ar) || detail::isinf(ai)) {          // Annex G.5.1
            nar = detail::copysign(detail::isinf(ar) ? 1.0f : 0.0f, ar);
            nai = detail::copysign(detail::isinf(ai) ? 1.0f : 0.0f, ai);
            recalc = true;
        }
        if (detail::isinf(br) || detail::isinf(bi)) {
            nbr = detail::copysign(detail::isinf(br) ? 1.0f : 0.0f, br);
            nbi = detail::copysign(detail::isinf(bi) ? 1.0f : 0.0f, bi);
            recalc = true;
        }
        if (recalc) {
            const float inf = HUGE_VALF;
            return FloatFloatComplex(
                FloatFloat(inf * (nar * nbr - nai * nbi)),
                FloatFloat(inf * (nar * nbi + nai * nbr)));
        }

        const float S = 0x1p-65f, U = 0x1p65f;
        FloatFloat sar = scale2(a.re, S), sai = scale2(a.im, S);
        FloatFloat sbr = scale2(b.re, S), sbi = scale2(b.im, S);
        FloatFloat qr = subtract(multiply(sar, sbr), multiply(sai, sbi));
        FloatFloat qi = add(multiply(sar, sbi), multiply(sai, sbr));
        return FloatFloatComplex(scale2(scale2(qr, U), U),
                                 scale2(scale2(qi, U), U));
    }

    XPMATH_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex b) const {
        FloatFloat rr = subtract(multiply(re, b.re), multiply(im, b.im));
        FloatFloat ri = add(multiply(re, b.im), multiply(im, b.re));
        if (rr.hi != rr.hi || ri.hi != ri.hi)                  // KI-28
            return mul_recover(*this, b, rr, ri);
        return FloatFloatComplex(rr, ri);
    }
    XPMATH_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex b) const {
        if (b.re.hi == 0.0f && b.im.hi == 0.0f) {
            XPMATH_PRINTF("FFCOMPLEX: division by zero\n");
            return FloatFloatComplex();
        }
        // KI-8.  The |denominator|^2 formulation squares b's components, so it
        // overflows and underflows for denominators whose quotient is perfectly
        // representable -- the same exposure as the unscaled hypot.  Past the
        // gate, use Smith's algorithm (1962), which divides through by the
        // larger component first so no intermediate exceeds the operands.
        // Inside the gate the original expression is kept bit-for-bit: Smith
        // costs two divides instead of one reciprocal and is slightly less
        // accurate, and there is nothing to win where the direct form works.
        {
            float mre = detail::fabs(b.re.hi);
            float mim = detail::fabs(b.im.hi);
            float mb  = (mre > mim) ? mre : mim;
            // KI-8 REOPENED: low edge widened from 1e-18f to the derived
            // word-underflow limit kFFSqLo -- the denominator's square shed low
            // words all the way up to 4.4e-16, not just below 1e-18f.  Smith's
            // algorithm forms no square at all, so it is correct across the
            // whole widened band.
            if (!(mb <= detail::kFFSqHi && mb >= detail::kFFSqLo)) {
                if (mre >= mim) {
                    FloatFloat rr = divide(b.im, b.re);
                    FloatFloat dd = add(b.re, multiply(b.im, rr));
                    return FloatFloatComplex(divide(add(re, multiply(im, rr)), dd),
                               divide(subtract(im, multiply(re, rr)), dd));
                } else {
                    FloatFloat rr = divide(b.re, b.im);
                    FloatFloat dd = add(multiply(b.re, rr), b.im);
                    return FloatFloatComplex(divide(add(multiply(re, rr), im), dd),
                               divide(subtract(multiply(im, rr), re), dd));
                }
            }
        }
        FloatFloat denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        FloatFloat inv   = divide(FloatFloat(1.0f), denom);
        return FloatFloatComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                         multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    XPMATH_INLINE_FUNCTION FloatFloatComplex operator-() const {
        return FloatFloatComplex(negate(re), negate(im));
    }

    XPMATH_INLINE_FUNCTION FloatFloatComplex& operator+=(FloatFloatComplex b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplex& operator-=(FloatFloatComplex b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplex& operator*=(FloatFloatComplex b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplex& operator/=(FloatFloatComplex b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(FloatFloatComplex b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(FloatFloatComplex b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION FloatFloat real() const { return re; }
    XPMATH_INLINE_FUNCTION FloatFloat imag() const { return im; }
};

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const FloatFloatComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed FloatFloat × FloatFloatComplex arithmetic
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(add(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(add(r, z.re), z.im); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(subtract(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(subtract(r, z.re), negate(z.im)); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(multiply(z.re, r), multiply(z.im, r)); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(multiply(r, z.re), multiply(r, z.im)); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(divide(z.re, r), divide(z.im, r)); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(r) / z; }

// ============================================================
// Mixed float × FloatFloatComplex arithmetic
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloatComplex z, float b) { return z + FloatFloat(b); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator+(float b, FloatFloatComplex z) { return FloatFloat(b) + z; }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloatComplex z, float b) { return z - FloatFloat(b); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator-(float b, FloatFloatComplex z) { return FloatFloat(b) - z; }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex z, float b) { return z * FloatFloat(b); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator*(float b, FloatFloatComplex z) { return FloatFloat(b) * z; }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex z, float b) { return z / FloatFloat(b); }
XPMATH_INLINE_FUNCTION FloatFloatComplex operator/(float b, FloatFloatComplex z) { return FloatFloat(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

// KI-8.  abs(z) is a magnitude and the unscaled sqrt(re^2 + im^2) below returns
// nan above |z| ~ 1.8e19 and 0 below |z| ~ 1.1e-19 on the FP32-word backends, in
// both cases while the answer is representable.  Past that range, defer to the
// scaled hypot.
//
// Inside the range the ORIGINAL expression is kept verbatim rather than routed
// through hypot as well.  That is not conservatism for its own sake: hypot's
// own fast path is written with the primitive each backend's hypot already
// used, and on TF that is sqr() where this one is multiply().  Swapping them
// costs up to 3.04 digits at four grid points (measured on the 428,592-point
// sweep, TF c abs points 736/737/1528..1531), so the two call sites keep their
// own primitives.
//
// KI-8 REOPENED: the band's low edge is now the derived word-underflow limit
// (ff_math.hpp's kFFSqLo, four decades above the old 1e-18f), and the out-of-
// band path scales by an EXACT power of two and then runs THIS site's own
// primitive rather than deferring to hypot.  Both changes are argued at
// ff_math.hpp's hypot; the second is what lets the band widen for free, since
// power-of-two scaling makes the direct expression exactly scale-equivariant.
XPMATH_INLINE_FUNCTION FloatFloat abs(FloatFloatComplex z) {
    float mr = detail::fabs(z.re.hi);
    float mi = detail::fabs(z.im.hi);
    float m  = (mr > mi) ? mr : mi;
    if (m == 0.0f) return FloatFloat(0.0f);
    if (m <= detail::kFFSqHi && m >= detail::kFFSqLo)
        return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
    // Out of band -- and that includes inf/nan, whose C99 F.9.4.3 convention
    // hypot owns -- defer to hypot's min/max tail, which forms no square at
    // all.  KI-8 REOPENED: an earlier revision of this fix scaled by an exact
    // power of two and squared anyway; that is exact for the SCALING but the
    // square still round-trips through sqrt, and complex asin amplifies the
    // resulting few ulps by |z|^2 (measured: QF c asin lost up to 14.04 digits
    // at sweep points 1628..1650).  The min/max tail is exact when one
    // component is zero, which is precisely those points.
    return hypot(z.re, z.im);
}
XPMATH_INLINE_FUNCTION FloatFloatComplex conj(FloatFloatComplex z) {
    return FloatFloatComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex sqrt(FloatFloatComplex z) {
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return FloatFloatComplex();
    FloatFloat r  = abs(z);   // KI-8: scaled magnitude, was sqrt(re^2+im^2) inline
    FloatFloat a1 = abs(z.re);
    FloatFloat s2 = multiply_scalar(add(r, a1), 0.5f);
    FloatFloat s0 = sqrt(s2);
    FloatFloat s1 = multiply_scalar(s0, 2.0f);
    FloatFloatComplex b;
    if (z.re.hi >= 0.0f) {
        b.re = s0;
        b.im = divide(z.im, s1);
    } else {
        b.re = divide(z.im, s1);
        if (b.re.hi < 0.0f) b.re = negate(b.re);
        b.im = s0;
        // KI-11. `z.im.hi < 0` is FALSE for -0.0, so both zero conventions landed
        // on the +i sheet and sqrt(-a - 0i) came back as +i*sqrt(a) -- the sign of
        // the whole answer wrong, 0.00 digits at every negative-real-axis grid
        // point (C99 Annex G: csqrt(-a -+ 0i) = -+ i*sqrt(a)). Reading the sign off
        // copysign instead costs nothing on the two conventions that were already
        // right. This also settles acosh(-a - 0i), whose imaginary part inherited
        // the sheet, and asinh(+-0 + yi) for |y| > 1, where sqrt(1 - y^2 -+ 0i) is
        // the term that decides which side of the cut the answer lands on.
        if (detail::copysign(1.0f, z.im.hi) < 0.0f) b.im = negate(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex exp(FloatFloatComplex z) {
    FloatFloat er = exp(z.re);
    FloatFloat c, s;
    sincos(z.im, c, s);
    return FloatFloatComplex(multiply(er, c), multiply(er, s));
}

XPMATH_INLINE_FUNCTION FloatFloatComplex log(FloatFloatComplex z) {
    FloatFloat modulus = abs(z);
    FloatFloat arg     = atan2(z.im, z.re);
    return FloatFloatComplex(log(modulus), arg);
}

// KI-5(b). Complex log1p(w) = log(1+w), accurate for small |w| -- the library
// had no complex log1p before this. Writing log(1 + w) directly is what made
// complex `atanh` collapse near the origin: 1 + w rounds w's information away
// before the log ever runs.
//
//     |1+w|^2 = 1 + (2*Re(w) + |w|^2)
//     Re log1p(w) = 0.5f * log1p( 2*Re(w) + |w|^2 )        <- REAL log1p
//     Im log1p(w) = atan2( Im(w), 1 + Re(w) )
//
// The whole point of the real part is the argument `2*Re(w) + |w|^2`: it is the
// small quantity by which |1+w|^2 differs from 1, formed WITHOUT ever adding 1,
// and handed to the real log1p (dd_math.hpp), which was rebuilt on the
// 2*atanh(a/(2+a)) series in the same change so that it can actually keep it.
//
// The imaginary part needs no such care. atan2(y, 1+x) for small w returns
// ~Im(w); its sensitivity to the rounding of 1 + Re(w) is d/dx atan2 = -y/|1+w|^2
// ~ -Im(w), so the absolute error eps in the second argument arrives as a
// RELATIVE error of eps in the answer. Nothing is lost.
//
// Divergence from the sources, recorded deliberately. QD 2.3f.24 (/tmp/qdsrc/QD)
// has no complex layer at all, so it offers no complex log1p to copy. Kahan 1987
// gives the formulation above (his `logp1`/`clogp1` discussion, and the same
// expression underlies his catanh); the residual weakness is his too -- when
// 2*Re(w) + |w|^2 itself cancels, i.e. on the circle |1+w| = 1, the real part
// loses relative accuracy. That locus is measure-zero, the answer there is ~0,
// and every hypot-based alternative loses the same digits on the same circle.
// Accepted rather than worked around.
XPMATH_INLINE_FUNCTION FloatFloatComplex log1p(FloatFloatComplex w) {
    FloatFloat t = add(multiply_scalar(w.re, 2.0f),
                         add(multiply(w.re, w.re), multiply(w.im, w.im)));
    return FloatFloatComplex(multiply_scalar(log1p(t), 0.5f),
                               atan2(w.im, add(FloatFloat(1.0f), w.re)));
}

XPMATH_INLINE_FUNCTION FloatFloatComplex log10(FloatFloatComplex z) {
    FloatFloatComplex lg = log(z);
    FloatFloat ln10 = FloatFloat_log10();
    return FloatFloatComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex sin(FloatFloatComplex z) {
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    FloatFloat ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return FloatFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex cos(FloatFloatComplex z) {
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    FloatFloat ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return FloatFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
// KI-18 fix. ASYMPTOTIC BRANCH for large |Im z|.
//
// `sin(z)/cos(z)` forms cosh(Im z) and sinh(Im z) explicitly. Both overflow the
// word type once |Im z| passes its exp ceiling (~709.8 for the FP64-word
// backend, ~88.7 for the FP32-word ones), and the quotient then evaluates
// inf/inf = NaN even though tan(z) -> +-i is perfectly bounded there:
// `DD tan(9807.8528 + 1950.90322i)` returned (NaN, NaN) for a true
// (-2.2769e-1695, 1). That is a wrong answer, not lost precision.
//
// Well before the overflow the REAL part is already gone. Written out, the
// complex quotient forms Re = (sa*ca*(cosh^2 b - sinh^2 b)) / |cos z|^2, and
// cosh^2 - sinh^2 = 1 is a difference of two quantities of size e^{2|b|}/4:
// it sheds 0.868*|Im z| decimal digits. Measured on DD, direct form:
// Im z = 5 -> 28.28 digits, 10 -> 24.02, 20 -> 15.64, 50 -> 0.00, then NaN.
//
// The remedy is the standard doubled-angle form with the exponential factored
// out. With t = exp(-2|Im z|) and s = sign(Im z),
//
//     tan(x + iy) = ( 2t*sin 2x + i*s*(1 - t^2) ) / ( 1 + t^2 + 2t*cos 2x )
//
// which is the usual [sin 2x + i sinh 2y] / [cos 2x + cosh 2y] with numerator
// and denominator both multiplied by 2t. Nothing overflows: t <= 1, the
// denominator is (1 - t)^2 + 2t(1 + cos 2x) >= 0, and the real part is formed
// as a product rather than as a difference, so it keeps full relative accuracy
// all the way down to the point where t itself underflows -- at which point
// 2t*sin 2x = 0 IS the correctly rounded answer, and the imaginary part is
// exactly +-1. No NaN is reachable for finite z.
//
// THRESHOLD kXpTanAsymptote = 1, on |Im z| (|Re z| for tanh), leading limb.
// Chosen from the denominator, which is the only thing the new form can lose
// to: it cancels only when t -> 1 AND cos 2x -> -1, i.e. only near Im z = 0.
// At |Im z| = 1, t = e^-2 = 0.1353 and the denominator is bounded below by
// (1-t)^2 = 0.747, so at most 0.13 digits are at risk; the direct form has
// already given up 0.87 by then. Below 1 the direct form is the better of the
// two and is kept unchanged, which also keeps every near-pole point (the poles
// of tan are on the real axis) bit-for-bit what it was. 1 is exactly
// representable, so the compare is exact. The test is `>=` so that the
// asymptotic branch owns the boundary.
//
// sin 2x and cos 2x are built from ONE sincos(x) as 2*sa*ca and
// (ca-sa)(ca+sa), not from sincos(2x): doubling before the argument reduction
// would spend a bit of x, and reusing the same reduction the direct branch
// uses keeps the two branches consistent across the threshold.
//
// tanh gets the identical treatment with the roles of the components swapped
// (tanh(z) = -i*tan(iz)), threshold on |Re z|. It also removes a second, worse
// symptom there: the old body divided by cos^2(b) + T^2 sin^2(b), and when FF's
// `sincos` handed back (0, 0) -- KI-12 -- that denominator was 0, so
// `FF tanh(-100 + 1e-29i)` returned (-inf, NaN). The new denominator is
// 1 + t^2 + 2t cos 2y, which is >= (1-t)^2 > 0 whatever sincos returns.
XPMATH_INLINE_FUNCTION FloatFloatComplex tan(FloatFloatComplex z) {
    const float kXpTanAsymptote = 4.0f;
    if (detail::fabs(z.im.hi) >= kXpTanAsymptote) {
        FloatFloat ca, sa;
        sincos(z.re, ca, sa);
        const FloatFloat s2 = multiply_scalar(multiply(sa, ca), 2.0f);      // sin 2x
        const FloatFloat c2 = multiply(subtract(ca, sa), add(ca, sa));         // cos 2x
        // t = exp(-2|Im z|); flushes to 0 far below the format's floor, which is
        // where +-i is the correctly rounded answer anyway.
        const FloatFloat t  = exp(multiply_scalar(z.im, z.im.hi < 0.0f ? 2.0f : -2.0f));
        const FloatFloat t2 = multiply(t, t);
        const FloatFloat den = add(add(FloatFloat(1.0f), t2), multiply_scalar(multiply(t, c2), 2.0f));
        FloatFloat im = divide(subtract(FloatFloat(1.0f), t2), den);
        if (z.im.hi < 0.0f) im = negate(im);
        return FloatFloatComplex(divide(multiply_scalar(multiply(t, s2), 2.0f), den), im);
    }
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
// KI-5(d) fix; see dd_complex.hpp:231-241 for the full rationale. On the real
// cut (Im(z) == +-0, |Re(z)| > 1) the sheet of sqrt(1 - z^2) is fixed by the
// sign of Im(z)'s zero, which the subtraction destroys (0 - (+-0) == +0 in
// round-to-nearest). Read it off Im(z) instead: Im(1 - z^2) = -2*Re*Im, so the
// root is negative-imaginary exactly when Re and Im share a sign.
// KI-11. |Im asin(z)| -- the one component the log form destroys, and the piece
// that also carries Re acosh and Re asinh (both are this same quantity under an
// exact identity; see their bodies).
//
// asin(z) = -i*log(w) with w = iz + sqrt(1 - z^2), so Im asin(z) = -log|w|, and
// |w| -> 1 for EVERY z near the real segment [-1,1] -- not just on it. Taking
// log of a number within eps of 1 rounds the answer away before log() is
// entered: at z = 0.5 + 1e-30i the true |w| is 1 - 1.15e-30, so forming it
// destroys log10(1/1.15e-30) = 29.9 digits and leaves ~1.1 of a 14-digit
// budget. Measured on this backend before the fix: 0.00 digits of 14.00.
//
// THAT IS NOT CONDITIONING. Probed against the binary128 oracle, the component
// condition number |(d Im f / d in_j)*in_j / Im f| is exactly 1.000 at these
// points (a relative eps on Im z moves Im asin by the same relative eps), and
// |z f'(z)/f(z)| = 1.10 -- so log10(kappa) = 0.04 and the format permits
// 14 - 0.04 digits here. The gap is entirely the formulation's.
//
// The cure is Hull, Fairgrove & Tang (1997), "Implementing the complex arcsine
// and arccosine functions using exception handling", ACM TOMS 23(3):299-335.
// With x = |Re z|, y = |Im z| and
//     r = hypot(x+1, y),   s = hypot(x-1, y),   a = (r + s)/2
// one has |Im asin z| = acosh(a) exactly, and a -> 1 is precisely the lossy
// region -- so carry a-1, never a. Because r and s are exact distances,
//     r = (x+1) + y^2/(r + (x+1)),        s = |x-1| + y^2/(s + |x-1|)
// hold identically, and substituting gives
//     a - 1 = (max(x,1) - 1) + (1/2)*( y^2/(r+(x+1)) + y^2/(s+|x-1|) )
// -- a sum of NON-NEGATIVE terms at every x, so there is no cancellation left
// to lose anything to. acosh(1+m) = log1p( m + sqrt(m*(m+2)) ) then keeps it,
// through the same real log1p that KI-5(b) rebuilt.
//
// y^2 is written y*(y/d), never (y*y)/d. Both r >= y and s >= y, so each
// quotient is <= 1 and the product cannot overflow at any |z|; the bare y*y
// overflows a 2xFP32 word above |y| ~ 1.8e19 and would hand the whole upper
// half of the range back as inf.
//
// One branch covers the whole plane: sqrt(m*(m+2)) is evaluated as the product
// of two separate roots, so nothing overflows however large |z| is, and log1p
// degrades gracefully into log for large arguments. An earlier revision split
// at a >= 2 into log(a) + log1p(sqrt(1 - (1/a)^2)); it measured WORSE (FF asin
// 14.00 -> 13.81, QF asinh 28.83 -> 27.75 at z = 2, pure rounding churn from
// the extra log), so the split does not ship.
XPMATH_INLINE_FUNCTION FloatFloat xp_asin_imag_mag(FloatFloat x, FloatFloat y) {
    const FloatFloat one(1.0f);
    const FloatFloat xp1  = add(x, one);
    const FloatFloat xm1s = subtract(x, one);            // signed, for the max(x,1) term
    FloatFloat xm1 = xm1s;
    if (xm1.hi < 0.0f) xm1 = negate(xm1);      // |x - 1|
    const FloatFloat r  = hypot(xp1, y);
    const FloatFloat s_ = hypot(xm1, y);
    const FloatFloat d1 = add(r,  xp1);
    const FloatFloat d2 = add(s_, xm1);
    // v = (1/d1 + 1/d2)/2, so the y-dependent half of a-1 is exactly y^2*v.
    // Carrying v rather than the two quotients is what lets sqrt(a-1) be formed
    // as y*sqrt(v) below, with y never squared.
    FloatFloat v(FloatFloat(0.0f));
    if (d1.hi != 0.0f) v = add(v, divide(one, d1));
    if (d2.hi != 0.0f) v = add(v, divide(one, d2));
    v = multiply_scalar(v, 0.5f);
    FloatFloat m = multiply(y, multiply(y, v));
    // + (max(x,1) - 1), tested on the SIGNED difference rather than on x's
    // leading word, so an x whose leading word is exactly 1 but whose tail is
    // positive still takes the term (the KI-16 value-based-guard rule).
    if (xm1s.hi > 0.0f) m = add(m, xm1s);
    // sqrt(a-1). Where the max(x,1) term is absent, a-1 is exactly y^2*v and
    // the root is y*sqrt(v) -- formed WITHOUT ever squaring y. That is not a
    // micro-optimisation: y^2 goes subnormal in an FP32 word below |y| ~ 1e-19
    // and zero below ~1e-22, and the first cut of this fix (which did square)
    // took QF asin at 0.5 + 1e-30i to -0.00 for exactly that reason. m itself
    // may still underflow there, and that is harmless -- next to sqrt(2*m) it
    // is a correction of relative size sqrt(m/2), i.e. already below the
    // format's own resolution wherever it underflows.
    const FloatFloat sm = (xm1s.hi > 0.0f) ? sqrt(m) : multiply(y, sqrt(v));
    // acosh(1+m) = log1p( m + sqrt(m)*sqrt(m+2) ). Split into two roots rather
    // than sqrt(m*(m+2)) so the product never overflows: each factor is O(|z|)
    // at worst and their product is the ~2|z| that log1p wants anyway. The
    // algebraically equivalent sm*(sm + sqrt(m+2)) was measured too and is
    // very slightly worse overall (5290 sweep cells down vs 5126), so this
    // form ships.
    return log1p(add(m, multiply(sm, sqrt(add(m, FloatFloat(2.0f))))));
}
// |Re z| and |Im z|, the two arguments xp_asin_imag_mag() wants. Split out so
// asin/acosh/asinh cannot disagree about them.
XPMATH_INLINE_FUNCTION FloatFloat xp_abs_word(FloatFloat v) {
    return (v.hi < 0.0f) ? negate(v) : v;
}
XPMATH_INLINE_FUNCTION FloatFloatComplex asin(FloatFloatComplex z) {
    // asin(z) = -i * log(iz + sqrt(1 - z^2))
    FloatFloatComplex iz  = FloatFloatComplex(negate(z.im), z.re);
    FloatFloatComplex z2  = z * z;
    FloatFloatComplex one_minus_z2 = FloatFloatComplex(FloatFloat(1.0f)) - z2;
    FloatFloatComplex root = sqrt(one_minus_z2);
    if (z.im.hi == 0.0f && detail::fabs(z.re.hi) > 1.0f) {
        const bool want_neg = (detail::copysign(1.0f, z.re.hi) ==
                               detail::copysign(1.0f, z.im.hi));
        const bool have_neg = (detail::copysign(1.0f, root.im.hi) < 0.0f);
        if (want_neg != have_neg) root.im = negate(root.im);
    }
    FloatFloatComplex sum = iz + root;
    // Re asin(z) = arg(sum) -- this component was never the problem, and
    // atan2(sum.im, sum.re) is exactly what log(sum).im was returning.
    const FloatFloat re = atan2(sum.im, sum.re);
    // Im asin(z) = sign(Im z) * acosh(a); see xp_asin_imag_mag above. The sign is
    // read off copysign so a signed zero on the real cuts picks the C99 Annex G
    // side -- the same convention the KI-5(d) block above enforces for the root
    // that feeds the real part.
    FloatFloat im = xp_asin_imag_mag(xp_abs_word(z.re), xp_abs_word(z.im));
    if (detail::copysign(1.0f, z.im.hi) < 0.0f) im = negate(im);
    return FloatFloatComplex(re, im);
}
// Principal sqrt of (u, v) with the sign of a ZERO v respected. The header's
// complex sqrt above tests `z.im.hi < 0.0f`, which is false for -0.0f, so for u < 0
// it puts BOTH zero conventions on the +i sheet. That is the same class of defect
// as KI-5(d) and it is corrected locally here rather than inside sqrt itself,
// which would move every other caller (acosh included) in one undocumented step.
// The general sqrt defect was recorded in docs/KNOWN_ISSUES.md as part of KI-11
// and HAS since been fixed at source (sqrt now reads the sheet off copysign), so
// this local helper is now belt-and-braces rather than the only correct path.
// `vsign` is +1/-1, the intended sign of v when v is a zero: multiply_scalar does
// NOT carry a signed zero through (it renormalizes, and quick_two_sum(-0,+0) is
// +0), so the caller passes the sign it read off the original Im(z) rather than
// trusting the halved copy.
XPMATH_INLINE_FUNCTION FloatFloatComplex sqrt_signed_cut(FloatFloat u, FloatFloat v, float vsign) {
    FloatFloatComplex r = sqrt(FloatFloatComplex(u, v));
    if (v.hi == 0.0f && u.hi < 0.0f && vsign != detail::copysign(1.0f, r.im.hi))
        r.im = negate(r.im);
    return r;
}
// KI-5(c) fix. acos(z) = -2i * log( sqrt((1+z)/2) + i*sqrt((1-z)/2) ) -- Kahan
// 1987, the exact companion of the acosh form adopted for KI-1 below.
//
// The old body was `pi/2 - asin(z)`, which is unconditionally stable nowhere near
// z = 1: acos(1) = 0, so as z -> 1 the difference cancels two quantities that
// both tend to pi/2 and the ANSWER's own magnitude tends to 0. Every digit of
// the result is a digit the subtraction destroyed. It is worse off the real axis
// -- acos(2 + 1e-20i) scored 11.31f on DD, 0.00f on FF -- because asin there is
// itself computed through 1 - z^2, so the loss compounds.
//
// Why Kahan's log form and not his other one, acos(z) = 2*atan(sqrt((1-z)/(1+z))).
// Both are well conditioned at z -> 1. The log form was chosen because (a) it is
// structurally identical to the acosh already in this header, so the two share a
// verified branch layout and one reader's understanding covers both, (b) it needs
// only sqrt/log, which are the two best-tested primitives here, where the atan
// form adds a complex atan on top of a complex divide by (1+z) -- a divide that
// is singular at z = -1, the OTHER end of the principal interval, so that form
// simply moves the bad point rather than removing it, and (c) it never forms z^2
// or 1/(1+z), hence no overflow at large |z| and no singularity at either end.
//
// BUT THE PURE LOG FORM IS ONLY HALF RIGHT, and the monotone gate is what said
// so. Writing w for the bracket, Im(acos z) = -2*ln|w|, and |w| -> 1 exactly
// where Im(acos z) -> 0, i.e. for z near the real segment [-1,1]. There the log
// is taken of a number whose information sits below the leading 1 -- the same
// disease as KI-5(b), relocated. Measured: 4016 sweep points down across the
// four backends, worst 31.00 -> 1.23 on DD at z = 0 + 1e-30i. Strictly worse
// than the defect being fixed.
//
// SO THIS IS PER-COMPONENT, and the split is exact rather than a compromise.
// Im(acos z) = -Im(asin z) is an IDENTITY (pi/2 is real), so the pi/2 - asin
// cancellation was only ever in the real part:
//
//     Re(acos z) = 2*arg( sqrt((1+z)/2) + i*sqrt((1-z)/2) )   <- Kahan, stable
//     Im(acos z) = -Im(asin z)                                <- exact identity
//
// Each form's good component, neither's bad one. This also inherits KI-5(d)'s
// cut fix on the imaginary part for free. Decreases fell from 4016 to 949.
//
// ACCEPTED LOSS, recorded in docs/KNOWN_ISSUES.md KI-5(c): for |z| >> 1 with
// arg(z) < 0 the two roots satisfy rm ~ -i*rp, so w = rp + i*rm is a difference
// of near-equal roots and arg(w) loses about log10|z|/2 digits -- up to 7.85 on
// the |z| = 1e8 polar ring. Reflecting by acos(conj z) = conj(acos z) should
// remove it and was tried both ways; both measured WORSE (1915 and 1648
// decreases against 949), so no reflection ships. See the KNOWN_ISSUES entry
// before retrying it.
//
// This is NOT routed through acosh even though acos(z) = +-i*acosh(z) holds. The
// sign of that relation flips with the half-plane AND with the side of each cut,
// so sharing the body would mean reintroducing exactly the case analysis Kahan's
// form exists to avoid; the two functions are three lines each and stay separate.
//
// BRANCH CHECK (each verified against the __complex128 oracle, both half-planes
// and both sides of both cuts): z=0 -> pi/2; z=1 -> 0; z=-1 -> pi;
// z=2+0i -> -1.3170i; z=2-0i -> +1.3170i; z=-2+0i -> pi-1.3170i;
// z=-2-0i -> pi+1.3170i. The last four are the cut points, and they are why
// sqrt_signed_cut exists: at z = 2+-0i the argument of the SECOND root is
// negative-real with a signed zero imaginary part, and at z = -2+-0i it is the
// FIRST root, so both calls need the corrected sheet.
XPMATH_INLINE_FUNCTION FloatFloatComplex acos(FloatFloatComplex z) {
    const FloatFloat one(1.0f);
    const FloatFloat half_im = multiply_scalar(z.im, 0.5f);
    const float s_im = detail::copysign(1.0f, z.im.hi);
    FloatFloatComplex rp = sqrt_signed_cut(
        multiply_scalar(add(one, z.re), 0.5f), half_im, s_im);
    FloatFloatComplex rm = sqrt_signed_cut(
        multiply_scalar(subtract(one, z.re), 0.5f), negate(half_im), -s_im);
    // w = rp + i*rm, with i*(a+bi) = -b + ai
    FloatFloatComplex w(subtract(rp.re, rm.im), add(rp.im, rm.re));
    return FloatFloatComplex(multiply_scalar(atan2(w.im, w.re), 2.0f), negate(asin(z).im));
}
// log(a^2 + b^2), formed without ever squaring the larger operand -- used by the
// two-log branches of atan()/atanh() below. Writing it as log(a*a + b*b) is what
// a first cut did, and it costs everything at the branch points: at
// z = -1 + 1e-19i the atanh numerator is (1+x)^2 + y^2 = 1e-38, which is
// SUBNORMAL in an FP32 word, so the FP32-word backends scored ~9 digits where
// the form they replaced scored 14 (FF) / 26 (QF) / 21 (TF). Factoring the
// larger operand out --
//     log(a^2 + b^2) = 2*log(s) + log1p((t/s)^2),  s = max(|a|,|b|), t = min
// -- never forms a product that can underflow or overflow, and returns the full
// width at those points. s == 0 (both operands zero) gives log(0) = -inf, which
// is the right answer for the branch point itself.
XPMATH_INLINE_FUNCTION FloatFloat xp_log_hypot2(FloatFloat a, FloatFloat b) {
    FloatFloat s = a, t = b;
    if (s.hi < 0.0f) s = negate(s);
    if (t.hi < 0.0f) t = negate(t);
    if (s.hi < t.hi) { FloatFloat tmp = s; s = t; t = tmp; }
    // Both operands zero: the pole itself. Returning the extended log(0) here
    // does NOT work -- feeding -inf into the caller's subtract() makes the
    // error limb inf - inf = NaN and destroys the whole result -- so the two
    // poles are intercepted at the top of atan()/atanh() instead.
    if (s.hi == 0.0f) return log(s);
    const FloatFloat r = divide(t, s);
    return add(multiply_scalar(log(s), 2.0f), log1p(multiply(r, r)));
}
// atan2()/angle() forms hypot(a, b) internally, so an operand pair whose
// SQUARES overflow the word format comes back NaN. That is what turned
// atan(1e10 + 0i) into NaN in the three FP32-word backends once the component
// form below started handing atan2 the raw 1 - x^2 - y^2 (monotone gate sweep
// point 1628, axis family: 14.00 -> 0.00). arg() is scale-invariant, so both
// operands are scaled down by a common EXACT power of two until the squares
// fit; being exact, the ratio -- and hence the answer -- is untouched. The
// loop runs at most once for every input the callers admit.
XPMATH_INLINE_FUNCTION FloatFloat xp_atan2_safe(FloatFloat a, FloatFloat b) {
    const float kXpAtan2Safe = 1.0e18f;
    const float kXpAtan2Down = 5.4210108624275222e-20f;   // exact power of two
    float m = detail::fabs(a.hi) > detail::fabs(b.hi) ? detail::fabs(a.hi)
                                                     : detail::fabs(b.hi);
    while (m > kXpAtan2Safe) {
        if (a.hi != 0.0f) a = multiply_scalar(a, kXpAtan2Down);
        if (b.hi != 0.0f) b = multiply_scalar(b, kXpAtan2Down);
        m *= kXpAtan2Down;
    }
    return atan2(a, b);
}
// KI-11 + KI-18 fix. The old body was
//     atan(z) = (i/2) * log( (1 - iz) / (1 + iz) )
// and it has two independent defects, both repaired here by moving to the
// component form -- which is nothing more than atan(z) = -i*atanh(iz) with the
// atanh block below (the KI-5(b) form) written out and the i folded in:
//
//     Re atan(z) = 0.5  * atan2( 2x, 1 - x^2 - y^2 )
//     Im atan(z) = 0.25 * log1p( 4y / (x^2 + (1-y)^2) )
//
// (1) KI-11 -- THE SMALL COMPONENT WAS BEING LOST. When |y| << |x| the ratio
// (1-iz)/(1+iz) has modulus 1 to within |y|/|x|, so log() is handed a number
// whose entire imaginary information sits below its own leading digit and the
// complex divide has already rounded it away. Measured, DD, direct form:
//     atan(-100 + 1e-29i)  0.00 digits -- returned Im = -1.6235e-33 for a true
//                          +9.9990e-34: WRONG SIGN, and bit-identical to what
//                          it returned at 1e-30i, i.e. a noise floor, not a
//                          function of the input any more
//     atan(0.5 + 1e-25i)   7.65      atan(2 + 1e-20i)   11.80
//     atan(100 + 1e-20i)   8.55      atan(1e8 + 1e-8i)  8.52
// In the component form the small component is never added to the large one:
// 4y/(x^2 + (1-y)^2) is ~4y/|1-iz|^2, small, formed without cancellation, and
// the real log1p keeps it. All five points above go to the type's cap.
//
// (2) KI-18 -- NaN AT THE BRANCH POINTS. As z -> +-i the divisor 1 + iz -> 0,
// so the complex divide overflowed or divided by a computed zero and BOTH
// components came back NaN even though only the imaginary one is genuinely
// infinite: `QF atan(1e-19 + 1i)` returned (NaN, 22.221132) for a true
// (0.785398163, 22.221132). In the component form the real part is an atan2,
// which is bounded everywhere, so the divergence stays in the component that
// actually diverges.
//
// THE log1p ARGUMENT CAN STILL OVERFLOW, and that is what the second branch is
// for. At z = 1e-19 + 1i the denominator is x^2 = 1e-38 while the numerator is
// 4, so 4y/D = 4e38 -- finite in an FP64 word, but past FLT_MAX in an FP32 one,
// where it would become inf and hand log1p an infinity. Since
//     0.25*log1p(4y/D) = 0.25*( log(x^2 + (1+y)^2) - log(x^2 + (1-y)^2) )
// identically, the two-log form is used once the ratio gets large. It is the
// wrong form for SMALL ratios -- that is the cancellation log1p exists to avoid
// -- and the right one for large, where the two logs differ by a wide margin.
// kXpAtanBigRatio is 1e30 for the FP32-word backends (well inside FLT_MAX =
// 3.4e38, with room for the 4y numerator) and 1e150 for the FP64-word one.
//
// kXpAtanBigL, L-infinity on the leading limbs, guards the OTHER end: x^2 and
// y^2 overflow the word type above sqrt of its range (~1.3e154 FP64-word,
// ~1.8e19 FP32-word), and there the old ratio form is finite and accurate --
// (1-iz)/(1+iz) -> -1 with no cancellation once |z| is huge -- so it is kept
// for that regime rather than replaced by something that overflows. The
// constants are set one decade inside the true limit.
//
// SIGNED ZERO ON THE CUTS. atan's cuts are on the imaginary axis, |Im z| > 1,
// where the sign of a zero REAL part picks the sheet: atan(+0 + 2i) = +pi/2 +
// 0.5493i and atan(-0 + 2i) = -pi/2 + 0.5493i. atan2 delivers that for free,
// but multiply_scalar renormalizes and does not carry a signed zero through, so
// the doubling of x is skipped when x is a zero (2*+-0 = +-0 exactly, so this
// is not an approximation) and the original limb is handed to atan2 instead.
XPMATH_INLINE_FUNCTION FloatFloatComplex atan(FloatFloatComplex z) {
    // C99 Annex G poles: catan(+-0 +- 1i) = +-0 +- inf*i. No formulation
    // built out of the extended log() can produce the infinity, because that
    // log() reports "non-positive argument" and returns 0 -- at HEAD these two
    // points came back (0, 0), a finite wrong answer. Intercept them.
    {
        const FloatFloat ay_ = z.im.hi < 0.0f ? negate(z.im) : z.im;
        if (z.re.hi == 0.0f && subtract(FloatFloat(1.0f), ay_).hi == 0.0f) {
            float inf_ = -detail::log(float(0));
            if (z.im.hi < 0.0f) inf_ = -inf_;
            return FloatFloatComplex(z.re, FloatFloat(inf_));
        }
    }
    const float kXpAtanBigL     = 1.0e18f;
    const float kXpAtanBigRatio = 1.0e4f;
    const FloatFloat one = FloatFloat(1.0f);
    if (detail::fabs(z.re.hi) < kXpAtanBigL && detail::fabs(z.im.hi) < kXpAtanBigL) {
        const FloatFloat x2 = multiply(z.re, z.re);
        const FloatFloat y2 = multiply(z.im, z.im);
        FloatFloat twox = multiply_scalar(z.re, 2.0f);
        if (z.re.hi == 0.0f) twox = z.re;          // keep the signed zero
        // 1 - x^2 - y^2 as (1-y)(1+y) - x^2. Forming `1 - (x^2 + y^2)` instead
        // rounds x^2 + y^2 to ONE word before the cancellation, so at |y| ~ 1
        // -- exactly the atan branch cut -- the tiny x^2 that survives is left
        // with only word-0 precision. The factored form is exact there
        // (Sterbenz on both factors) and costs one extra multiply.
        const FloatFloat d2 =
            subtract(multiply(subtract(one, z.im), add(one, z.im)), x2);
        FloatFloat re = multiply_scalar(xp_atan2_safe(twox, d2), 0.5f);
        // ON THE CUT (Re(z) a zero, |Im z| > 1) the sheet is chosen by the SIGN
        // of that zero -- atan(+0 + 2i) = +pi/2 + 0.5493i, atan(-0 + 2i) =
        // -pi/2 + 0.5493i. atan2() is handed the zero verbatim above but does
        // not carry its sign through, so +-pi/2 is installed directly, which is
        // the same correction atanh() below already makes on its own cut. The
        // monotone gate is what caught this: 30.74 -> 0.00 on DD at z = -0 + 2i.
        if (z.re.hi == 0.0f && d2.hi < 0.0f) {
            re = multiply_scalar(FloatFloat_pi(), 0.5f);
            if (detail::copysign(1.0f, z.re.hi) < 0.0f) re = negate(re);
        }
        const FloatFloat omy = subtract(one, z.im);
        const FloatFloat den = add(x2, multiply(omy, omy));
        const FloatFloat num = multiply_scalar(z.im, 4.0f);
        FloatFloat im;
        if (num.hi < den.hi * kXpAtanBigRatio &&
            num.hi > -0.875f * den.hi) {
            im = multiply_scalar(log1p(divide(num, den)), 0.25f);
        } else {
            im = multiply_scalar(
                subtract(xp_log_hypot2(z.re, add(one, z.im)),
                         xp_log_hypot2(z.re, omy)), 0.25f);
        }
        return FloatFloatComplex(re, im);
    }
    // |z| past sqrt(word range): squaring would overflow. The ratio form is
    // well behaved out here and is kept.
    FloatFloatComplex iz    = FloatFloatComplex(negate(z.im), z.re);
    FloatFloatComplex num   = FloatFloatComplex(one) - iz;
    FloatFloatComplex den   = FloatFloatComplex(one) + iz;
    FloatFloatComplex ratio = num / den;
    FloatFloatComplex lg    = log(ratio);
    // multiply by i/2: (a+bi)*(i/2) = (-b/2) + (a/2)*i
    return FloatFloatComplex(multiply_scalar(negate(lg.im), 0.5f), multiply_scalar(lg.re, 0.5f));
}

// ============================================================
// Complex hyperbolic
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex sinh(FloatFloatComplex z) {
    // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
    FloatFloat ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return FloatFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex cosh(FloatFloatComplex z) {
    // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
    FloatFloat ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return FloatFloatComplex(multiply(ca, cb), multiply(sa, sb));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex tanh(FloatFloatComplex z) {
    // KI-18: asymptotic branch, the tan() block above documents it in full.
    const float kXpTanAsymptote = 4.0f;
    if (detail::fabs(z.re.hi) >= kXpTanAsymptote) {
        FloatFloat cb, sb;
        sincos(z.im, cb, sb);
        const FloatFloat s2 = multiply_scalar(multiply(sb, cb), 2.0f);      // sin 2y
        const FloatFloat c2 = multiply(subtract(cb, sb), add(cb, sb));         // cos 2y
        const FloatFloat t  = exp(multiply_scalar(z.re, z.re.hi < 0.0f ? 2.0f : -2.0f));
        const FloatFloat t2 = multiply(t, t);
        const FloatFloat den = add(add(FloatFloat(1.0f), t2), multiply_scalar(multiply(t, c2), 2.0f));
        FloatFloat re = divide(subtract(FloatFloat(1.0f), t2), den);
        if (z.re.hi < 0.0f) re = negate(re);
        return FloatFloatComplex(re, divide(multiply_scalar(multiply(t, s2), 2.0f), den));
    }
    // |Re z| < 1: the direct form is the more accurate of the two here and is
    // kept verbatim.
    // tanh(a+bi): re = tanh(a) / (cos^2(b) + tanh^2(a)*sin^2(b))
    //             im = sin(b)*cos(b)*(1 - tanh^2(a)) / (same denominator)
    FloatFloat T_ = tanh(z.re);
    FloatFloat cb, sb;
    sincos(z.im, cb, sb);
    FloatFloat T2    = multiply(T_, T_);
    FloatFloat denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return FloatFloatComplex(divide(T_, denom),
               divide(multiply(multiply(sb, cb), subtract(FloatFloat(1.0f), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
// asinh(z) = log(z + sqrt(z^2 + 1)), reflected into the right half-plane via the
// oddness identity -asinh(-z) when the direct form would cancel. KI-5(a); see
// dd_complex.hpp's asinh for the full rationale, the magnitude threshold and its
// measured justification, and the Re(z) == +-0 boundary decision — all identical
// here.
// Re asinh(z), the KI-11 form. Kept as its own function so the reflected and
// unreflected branches below cannot drift apart.
XPMATH_INLINE_FUNCTION FloatFloat xp_ki11_asinh_re(FloatFloatComplex z) {
    FloatFloat v = xp_asin_imag_mag(xp_abs_word(z.im), xp_abs_word(z.re));
    if (detail::copysign(1.0f, z.re.hi) < 0.0f) v = negate(v);
    return v;
}
// KI-11. Re asinh(z) = sign(Re z) * |Im asin(iz)| -- exact, from
// asinh(z) = -i*asin(iz): the real part of asinh is the imaginary part of asin
// with the arguments transposed, since Re(iz) = -Im z and Im(iz) = Re z. So the
// same xp_asin_imag_mag() serves, called as (|Im z|, |Re z|) rather than
// (|Re z|, |Im z|). log(z + sqrt(z^2+1)) loses it for the same reason asin's
// log did -- |z + sqrt(z^2+1)| -> 1 all along the imaginary segment [-i,i] --
// measured 0.00 of 14.00 at z = 1e-25 + 0.5i. The imaginary part keeps the
// existing body, reflection branch and all; it was never the losing one.
XPMATH_INLINE_FUNCTION FloatFloatComplex asinh(FloatFloatComplex z) {
    const float t = 4.0f;   // kXpAsinhReflect
    if (z.re.hi < 0.0f && (-z.re.hi > t || detail::fabs(z.im.hi) > t)) {
        FloatFloatComplex w = -z;
        return FloatFloatComplex(xp_ki11_asinh_re(z), negate(log(w + sqrt(w*w + FloatFloatComplex(FloatFloat(1.0f)))).im));
    }
    return FloatFloatComplex(xp_ki11_asinh_re(z), log(z + sqrt(z*z + FloatFloatComplex(FloatFloat(1.0f)))).im);
}
// KI-1 fix: Kahan 1987's branch-correct form, acosh(z) = 2*log(sqrt((z+1)/2) +
// sqrt((z-1)/2)). See dd_complex.hpp:346-357 for the full rationale. The old
// log(z + sqrt(z*z - 1)) form was on the wrong sqrt sheet throughout
// Re(z) < 0, and overflowed above |z| ~ 1.8e19 where z*z leaves FP32 range.
XPMATH_INLINE_FUNCTION FloatFloatComplex acosh(FloatFloatComplex z) {
    const FloatFloat one(1.0f);
    const FloatFloat half_im = multiply_scalar(z.im, 0.5f);
    FloatFloatComplex rp = sqrt(FloatFloatComplex(
        multiply_scalar(add(z.re, one), 0.5f), half_im));
    FloatFloatComplex rm = sqrt(FloatFloatComplex(
        multiply_scalar(subtract(z.re, one), 0.5f), half_im));
    FloatFloatComplex lg = log(rp + rm);
    // KI-11. Re acosh(z) = |Im asin(z)| is an identity (acosh(z) = +-i*acos(z)
    // and Im acos = -Im asin), and 2*log|rp+rm| has exactly the disease
    // xp_asin_imag_mag() exists to cure: |rp+rm| -> 1 all along the real segment
    // [-1,1], so at z = 0.5 + 1e-30i it scored 0.00 of 14.00. The imaginary part
    // is Kahan's and stays -- it is the well-conditioned one here.
        // KI-11. acosh(conj z) = conj acosh(z) and the principal strip is
    // Im acosh in (-pi, pi], so sign(Im acosh z) = sign(Im z) EVERYWHERE,
    // signed zeros on the cut included (C99 Annex G). Kahan's form got that
    // sign from a chain of sqrt/log that drops -0.0 on the FP32-word backends:
    // QF and TF returned acosh(-0.1 - 0i) = +1.670964i, the wrong sheet and
    // 0.00 digits, while DD and FF happened to keep it. Taking the magnitude
    // and re-attaching the sign from copysign(Im z) is exact and cannot drift.
    FloatFloat im_ = multiply_scalar(lg.im, 2.0f);
    if (im_.hi < 0.0f) im_ = negate(im_);
    if (detail::copysign(1.0f, z.im.hi) < 0.0f) im_ = negate(im_);
    return FloatFloatComplex(xp_asin_imag_mag(xp_abs_word(z.re), xp_abs_word(z.im)), im_);
}
// KI-5(b) fix. Two independent defects lived in the old one-line body
// `0.5f*log((1+z)/(1-z))`, and both are repaired here.
//
// (1) CONDITIONING AS z -> 0. The ratio (1+z)/(1-z) -> 1, so the log is taken of
// a number whose entire information content sits below the leading 1. The
// argument reduction throws away log10(1/|z|) digits before log() is even
// entered, and the measured score falls off one digit per decade of |z|. The
// remedy is the classical one: 0.5f*log1p(2z/(1-z)), expanded here into its real
// and imaginary components so no complex divide is needed either --
//
//     Re atanh(z) = 0.25f * log1p( 4x / ((1-x)^2 + y^2) )
//     Im atanh(z) = 0.5f  * atan2( 2y, 1 - x^2 - y^2 )
//
// which is 0.5f*log1p(w) with w = 2z/(1-z) written out: 2*Re(w) + |w|^2 collapses
// to 4x/|1-z|^2 and arg(1+w) to atan2(2y, 1-x^2-y^2). For small z the log1p
// argument is ~4x, small and formed without cancellation, and the real log1p
// (dd_math.hpp, rebuilt on the 2*atanh(a/(2+a)) series in this same change)
// keeps it.
//
// THRESHOLD (kXpAtanhSmall = 0.0625, L-infinity on the leading limbs). The new form
// is used only where the old one is actually losing. Two reasons to gate rather
// than switch unconditionally. First, ((1-x)^2 + y^2) squares its operands, so
// it overflows to Inf for |z| above sqrt of the word type's range (~1.3e154 for
// the FP64-word backend, ~1.8e19 for the FP32-word ones) where the old ratio
// form is perfectly finite -- a switch would trade a conditioning defect for an
// overflow defect. Second, at |z| >= 0.5f the old form already scores at the
// type's cap (measured: DD 30.74f/31.00f at z = 0.5f), so there is nothing to win
// and only rounding churn to lose -- the first cut of this fix used 0.5 and the
// gate caught 43 points across the four backends losing up to 1.09 digits to
// exactly that churn in 0.0625 < |z| < 0.5, so the threshold is 0.0625 (2^-4,
// exactly representable, so the compare is exact) and one atanh decrease
// remains in the whole 428,592-point sweep. L-infinity, not hypot(), for the same reasons
// given on asinh above: no overflow, two compares, and the leading limb settles
// it except within an ulp of the boundary.
//
// (2) THE SIGNED ZERO ON THE CUTS -- the atanh analogue of KI-5(d), found by
// measurement while fixing (1), and fixed here because it is in the same body.
// The cuts are (-inf,-1] and [1,+inf). Approaching x > 1 from Im = +0 gives
// Im atanh = +pi/2, and from Im = -0 gives -pi/2; by oddness the SAME +pi/2
// holds for x < -1 with Im = +0. The old form computed (1+z)/(1-z) with a
// complex divide, whose multiplies destroy the sign of Im(z)'s zero, and so
// returned +pi/2 for both conventions: every `x -0i` cut point scored 0.00f on
// the imaginary component, in all four backends. The sign is read off Im(z)
// directly with detail::copysign -- these types do carry a signed zero through
// construction, copy and negate, they only lose it in arithmetic -- and pi/2 is
// installed with it. The two conventions that were already right are unchanged.
//
// KI-11 EXTENSION (this change). The gate above was `L-inf < 0.0625` only, so
// everything outside a small disc still went through the ratio form and still
// lost its small component -- the same defect KI-11 records for atan, one
// function over. Measured, DD, ratio form: atanh(1e-20 + 100i) 8.55 digits,
// atanh(1e8 + 1e-8i) 23.30, atanh(1e-6 - 2i) 24.36. The component form is now
// used for L-inf >= 0.5 as well, and the two-log fallback described on atan()
// above is used where 4x/D would overflow the word type.
//
// THE BAND 0.0625 <= L-inf < 0.5 IS DELIBERATELY LEFT ON THE RATIO FORM. That
// is the band KI-5(b) measured the component form to be WORSE in -- 43 points
// across the four backends losing up to 1.09 digits to rounding churn, which is
// why 0.0625 rather than 0.5 was chosen then. Nothing here contradicts that
// measurement, so nothing there moves.
//
// kXpAtanhBigL is the same overflow guard as atan's: above sqrt of the word
// type's range (1e18 for FP32 words, 1e150 for FP64 words) (1-x)^2 + y^2
// overflows and the ratio form is kept.
XPMATH_INLINE_FUNCTION FloatFloatComplex atanh(FloatFloatComplex z) {
    // C99 Annex G poles: catanh(+-1 +- 0i) = +-inf +- 0i. Same reason as atan
    // above -- at HEAD these came back (0, 0).
    {
        const FloatFloat ax_ = z.re.hi < 0.0f ? negate(z.re) : z.re;
        if (z.im.hi == 0.0f && subtract(FloatFloat(1.0f), ax_).hi == 0.0f) {
            float inf_ = -detail::log(float(0));
            if (z.re.hi < 0.0f) inf_ = -inf_;
            return FloatFloatComplex(FloatFloat(inf_), z.im);
        }
    }
    const FloatFloat one = FloatFloat(1.0f);
    const float kXpAtanhSmall    = 0.0625f;
    const float kXpAtanhWide     = 0.5f;
    const float kXpAtanhBigL     = 1.0e18f;
    const float kXpAtanBigRatio  = 1.0e4f;
    const float ax = detail::fabs(z.re.hi), ay = detail::fabs(z.im.hi);
    const float linf = ax > ay ? ax : ay;
    FloatFloatComplex r;
    if (linf < kXpAtanhSmall || (linf >= kXpAtanhWide && linf < kXpAtanhBigL)) {
        const FloatFloat omx = subtract(one, z.re);
        const FloatFloat y2  = multiply(z.im, z.im);
        const FloatFloat den = add(multiply(omx, omx), y2);
        const FloatFloat num = multiply_scalar(z.re, 4.0f);
        if (num.hi < den.hi * kXpAtanBigRatio &&
            num.hi > -0.875f * den.hi) {
            r.re = multiply_scalar(log1p(divide(num, den)), 0.25f);
        } else {
            r.re = multiply_scalar(
                subtract(xp_log_hypot2(add(one, z.re), z.im),
                         xp_log_hypot2(omx, z.im)), 0.25f);
        }
        FloatFloat twoy = multiply_scalar(z.im, 2.0f);
        if (z.im.hi == 0.0f) twoy = z.im;          // keep the signed zero
        r.im = multiply_scalar(
            xp_atan2_safe(twoy, subtract(multiply(subtract(one, z.re), add(one, z.re)), y2)), 0.5f);
    } else {
        FloatFloatComplex lg = log((FloatFloatComplex(one) + z) / (FloatFloatComplex(one) - z));
        r.re = multiply_scalar(lg.re, 0.5f);
        r.im = multiply_scalar(lg.im, 0.5f);
    }
    if (z.im.hi == 0.0f && detail::fabs(z.re.hi) > 1.0f) {
        FloatFloat half_pi = multiply_scalar(FloatFloat_pi(), 0.5f);
        if (detail::copysign(1.0f, z.im.hi) < 0.0f) half_pi = negate(half_pi);
        r.im = half_pi;
    }
    return r;
}

// ============================================================
// Complex power and polar
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex pow(FloatFloatComplex z, FloatFloatComplex w) {
    // z^w = exp(w * log(z))
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return FloatFloatComplex();
    return exp(w * log(z));
}

// polar(r, theta) = r * (cos(theta) + i*sin(theta))
XPMATH_INLINE_FUNCTION FloatFloatComplex polar(FloatFloat r, FloatFloat theta) {
    FloatFloat c, s;
    sincos(theta, c, s);
    return FloatFloatComplex(multiply(r, c), multiply(r, s));
}

} // namespace xp
