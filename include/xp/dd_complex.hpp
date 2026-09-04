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
// double-double real arithmetic in dd_math.hpp (also DHB-License).
//
// Modifications from the original DDFUN v04 sources:
//   * Translated the complex double-double routines from Fortran-90
//     (ddfunc.f90 / the ddc* entry points) to header-only C++17.
//   * Every function XPMATH_INLINE_FUNCTION for host + device
//     portability across CUDA/HIP/SYCL/OpenMP-target.
//   * Namespaced as xp::DoubleDoubleComplex (a bespoke struct,
//     not yet Kokkos::complex<DoubleDouble>) with STL-style
//     free functions and ADL-friendly re-exposure under the
//     Kokkos-compat wrapper for potential upstreaming to Kokkos.
//   * See docs/TEST_SUITE_PLAN.md "Upstreaming considerations" for
//     naming and API conventions.

#pragma once

// Double-double complex arithmetic — xp::DoubleDoubleComplex.
// All functions XPMATH_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on dd_math.hpp.
//
// Ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab).
//
// DEPENDENCIES: none beyond the C++17 standard library and dd_math.hpp.
// In particular this header does NOT include or require Kokkos — see
// xp/config.hpp for how the portability facilities are supplied. Kokkos
// users get today's `Kokkos::Experimental::DoubleDoubleComplex` API
// unchanged through the compat wrapper at third_party/include/dd_complex.hpp,
// which is the only place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// Naming follows dd_math.hpp (T0.4): type + math live under
// xp:: for eventual upstreaming. This remains a bespoke struct
// rather than Kokkos::complex<DoubleDouble> — that integration is a separate
// future task.

#include <xp/dd_math.hpp>

#if !defined(XPMATH_ON_DEVICE)
#  include <ostream>
#endif

namespace xp {

// ============================================================
// DoubleDoubleComplex struct
// ============================================================
struct DoubleDoubleComplex {
    DoubleDouble re;
    DoubleDouble im;

    XPMATH_INLINE_FUNCTION DoubleDoubleComplex() : re(0.0), im(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex(double r)                     : re(r),    im(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex(DoubleDouble r)               : re(r),    im(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex(double r, double i)           : re(r),    im(i)   {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex(DoubleDouble r, DoubleDouble i) : re(r),  im(i)   {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex(const DoubleDoubleComplex& o) : re(o.re), im(o.im){}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex& operator=(const DoubleDoubleComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex& operator=(DoubleDouble r) {
        re = r; im = DoubleDouble(0.0); return *this;
    }

    // Arithmetic
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDoubleComplex b) const {
        return DoubleDoubleComplex(add(re, b.re), add(im, b.im));
    }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDoubleComplex b) const {
        return DoubleDoubleComplex(subtract(re, b.re), subtract(im, b.im));
    }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDoubleComplex b) const {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return DoubleDoubleComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                         add(multiply(re, b.im), multiply(im, b.re)));
    }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDoubleComplex b) const {
        if (b.re.hi == 0.0 && b.im.hi == 0.0) {
            XPMATH_PRINTF("DDCOMPLEX: division by zero\n");
            return DoubleDoubleComplex();
        }
        // (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)
        // KI-8.  The |denominator|^2 formulation squares b's components, so it
        // overflows and underflows for denominators whose quotient is perfectly
        // representable -- the same exposure as the unscaled hypot.  Past the
        // gate, use Smith's algorithm (1962), which divides through by the
        // larger component first so no intermediate exceeds the operands.
        // Inside the gate the original expression is kept bit-for-bit: Smith
        // costs two divides instead of one reciprocal and is slightly less
        // accurate, and there is nothing to win where the direct form works.
        {
            double mre = detail::fabs(b.re.hi);
            double mim = detail::fabs(b.im.hi);
            double mb  = (mre > mim) ? mre : mim;
            // KI-8 REOPENED: low edge widened from 1.0e-150 to the derived
            // word-underflow limit kDDSqLo -- the denominator's square shed low
            // words well above it, not just below 1.0e-150.  Smith's algorithm forms
            // no square at all, so it is correct across the whole widened band.
            if (!(mb <= detail::kDDSqHi && mb >= detail::kDDSqLo)) {
                if (mre >= mim) {
                    DoubleDouble rr = divide(b.im, b.re);
                    DoubleDouble dd = add(b.re, multiply(b.im, rr));
                    return DoubleDoubleComplex(divide(add(re, multiply(im, rr)), dd),
                               divide(subtract(im, multiply(re, rr)), dd));
                } else {
                    DoubleDouble rr = divide(b.re, b.im);
                    DoubleDouble dd = add(multiply(b.re, rr), b.im);
                    return DoubleDoubleComplex(divide(add(multiply(re, rr), im), dd),
                               divide(subtract(multiply(im, rr), re), dd));
                }
            }
        }
        DoubleDouble denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        DoubleDouble inv   = divide(DoubleDouble(1.0), denom);
        return DoubleDoubleComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                         multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator-() const {
        return DoubleDoubleComplex(negate(re), negate(im));
    }

    XPMATH_INLINE_FUNCTION DoubleDoubleComplex& operator+=(DoubleDoubleComplex b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex& operator-=(DoubleDoubleComplex b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex& operator*=(DoubleDoubleComplex b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplex& operator/=(DoubleDoubleComplex b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(DoubleDoubleComplex b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(DoubleDoubleComplex b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION DoubleDouble real() const { return re; }
    XPMATH_INLINE_FUNCTION DoubleDouble imag() const { return im; }
};

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const DoubleDoubleComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed DoubleDouble × DoubleDoubleComplex arithmetic
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(add(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(add(r, z.re), z.im); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(subtract(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(subtract(r, z.re), negate(z.im)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(multiply(z.re, r), multiply(z.im, r)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(multiply(r, z.re), multiply(r, z.im)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(divide(z.re, r), divide(z.im, r)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(r) / z; }

// ============================================================
// Mixed double × DoubleDoubleComplex arithmetic
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDoubleComplex z, double b) { return z + DoubleDouble(b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator+(double b, DoubleDoubleComplex z) { return DoubleDouble(b) + z; }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDoubleComplex z, double b) { return z - DoubleDouble(b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator-(double b, DoubleDoubleComplex z) { return DoubleDouble(b) - z; }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDoubleComplex z, double b) { return z * DoubleDouble(b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator*(double b, DoubleDoubleComplex z) { return DoubleDouble(b) * z; }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDoubleComplex z, double b) { return z / DoubleDouble(b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex operator/(double b, DoubleDoubleComplex z) { return DoubleDouble(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

// KI-8.  abs(z) is a magnitude and the unscaled sqrt(re^2 + im^2) below returns
// nan above |z| ~ 1.3e154 and 0 below |z| ~ 1.5e-154 on the FP64-word backends, in
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
// (dd_math.hpp's kDDSqLo), and the out-of-band path scales by an EXACT power
// of two and then runs THIS site's own primitive rather than deferring to
// hypot.  Both changes are argued at ff_math.hpp's hypot; the second is what
// lets the band widen for free, since power-of-two scaling makes the direct
// expression exactly scale-equivariant.
XPMATH_INLINE_FUNCTION DoubleDouble abs(DoubleDoubleComplex z) {
    double mr = detail::fabs(z.re.hi);
    double mi = detail::fabs(z.im.hi);
    double m  = (mr > mi) ? mr : mi;
    if (m == 0.0) return DoubleDouble(0.0);
    if (m <= detail::kDDSqHi && m >= detail::kDDSqLo)
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex conj(DoubleDoubleComplex z) {
    return DoubleDoubleComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex sqrt(DoubleDoubleComplex z) {
    if (z.re.hi == 0.0 && z.im.hi == 0.0) return DoubleDoubleComplex();
    // B = sqrt((R+A1)/2) + i*sign(A2)*sqrt((R-A1)/2)  where R = |z|
    DoubleDouble r  = abs(z);   // KI-8: scaled magnitude, was sqrt(re^2+im^2) inline
    DoubleDouble a1 = abs(z.re);
    DoubleDouble s2 = multiply_scalar(add(r, a1), 0.5);
    DoubleDouble s0 = sqrt(s2);
    DoubleDouble s1 = multiply_scalar(s0, 2.0);
    DoubleDoubleComplex b;
    if (z.re.hi >= 0.0) {
        b.re = s0;
        b.im = divide(z.im, s1);
    } else {
        b.re = divide(z.im, s1);
        if (b.re.hi < 0.0) b.re = negate(b.re);
        b.im = s0;
        if (z.im.hi < 0.0) b.im = negate(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex exp(DoubleDoubleComplex z) {
    DoubleDouble er = exp(z.re);
    DoubleDouble c, s;
    sincos(z.im, c, s);
    return DoubleDoubleComplex(multiply(er, c), multiply(er, s));
}

XPMATH_INLINE_FUNCTION DoubleDoubleComplex log(DoubleDoubleComplex z) {
    DoubleDouble modulus = abs(z);
    DoubleDouble arg     = atan2(z.im, z.re); // atan2(im, re)
    return DoubleDoubleComplex(log(modulus), arg);
}

// KI-5(b). Complex log1p(w) = log(1+w), accurate for small |w| -- the library
// had no complex log1p before this. Writing log(1 + w) directly is what made
// complex `atanh` collapse near the origin: 1 + w rounds w's information away
// before the log ever runs.
//
//     |1+w|^2 = 1 + (2*Re(w) + |w|^2)
//     Re log1p(w) = 0.5 * log1p( 2*Re(w) + |w|^2 )        <- REAL log1p
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
// Divergence from the sources, recorded deliberately. QD 2.3.24 (/tmp/qdsrc/QD)
// has no complex layer at all, so it offers no complex log1p to copy. Kahan 1987
// gives the formulation above (his `logp1`/`clogp1` discussion, and the same
// expression underlies his catanh); the residual weakness is his too -- when
// 2*Re(w) + |w|^2 itself cancels, i.e. on the circle |1+w| = 1, the real part
// loses relative accuracy. That locus is measure-zero, the answer there is ~0,
// and every hypot-based alternative loses the same digits on the same circle.
// Accepted rather than worked around.
XPMATH_INLINE_FUNCTION DoubleDoubleComplex log1p(DoubleDoubleComplex w) {
    DoubleDouble t = add(multiply_scalar(w.re, 2.0),
                         add(multiply(w.re, w.re), multiply(w.im, w.im)));
    return DoubleDoubleComplex(multiply_scalar(log1p(t), 0.5),
                               atan2(w.im, add(DoubleDouble(1.0), w.re)));
}

XPMATH_INLINE_FUNCTION DoubleDoubleComplex log10(DoubleDoubleComplex z) {
    DoubleDoubleComplex lg = log(z);
    DoubleDouble ln10 = DoubleDouble_log10();
    return DoubleDoubleComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex sin(DoubleDoubleComplex z) {
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    DoubleDouble ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION DoubleDoubleComplex cos(DoubleDoubleComplex z) {
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    DoubleDouble ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(ca, cb), negate(multiply(sa, sb)));
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex tan(DoubleDoubleComplex z) {
    const double kXpTanAsymptote = 2.0;
    if (detail::fabs(z.im.hi) >= kXpTanAsymptote) {
        DoubleDouble ca, sa;
        sincos(z.re, ca, sa);
        const DoubleDouble s2 = multiply_scalar(multiply(sa, ca), 2.0);      // sin 2x
        const DoubleDouble c2 = multiply(subtract(ca, sa), add(ca, sa));         // cos 2x
        // t = exp(-2|Im z|); flushes to 0 far below the format's floor, which is
        // where +-i is the correctly rounded answer anyway.
        const DoubleDouble t  = exp(multiply_scalar(z.im, z.im.hi < 0.0 ? 2.0 : -2.0));
        const DoubleDouble t2 = multiply(t, t);
        const DoubleDouble den = add(add(DoubleDouble(1.0), t2), multiply_scalar(multiply(t, c2), 2.0));
        DoubleDouble im = divide(subtract(DoubleDouble(1.0), t2), den);
        if (z.im.hi < 0.0) im = negate(im);
        return DoubleDoubleComplex(divide(multiply_scalar(multiply(t, s2), 2.0), den), im);
    }
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
// KI-5(d) fix. On the real cut (Im(z) == +-0 and |Re(z)| > 1) the sheet of
// sqrt(1 - z^2) is fixed by the SIGN of Im(z)'s zero, and the subtraction that
// forms 1 - z^2 destroys it: in round-to-nearest both 0 - (+0) and 0 - (-0)
// give +0, so both approaches land on the same sheet and exactly one of the two
// C99 Annex G conventions comes out wrong at every cut point. The sign is read
// off Im(z) directly instead -- the expansion types do preserve a signed zero
// through construction and copy, they only lose it in arithmetic -- and the
// root is placed on the sheet it selects. Since Im(1 - z^2) = -2*Re(z)*Im(z),
// the root is negative-imaginary exactly when Re and Im share a sign. The
// correction is a no-op on the two conventions that were already right, so it
// cannot move any other point. All four complex headers carry this same block.
XPMATH_INLINE_FUNCTION DoubleDoubleComplex asin(DoubleDoubleComplex z) {
    // asin(z) = -i * log(iz + sqrt(1 - z^2))
    DoubleDoubleComplex iz  = DoubleDoubleComplex(negate(z.im), z.re);
    DoubleDoubleComplex z2  = z * z;
    DoubleDoubleComplex one_minus_z2 = DoubleDoubleComplex(DoubleDouble(1.0)) - z2;
    DoubleDoubleComplex root = sqrt(one_minus_z2);
    if (z.im.hi == 0.0 && detail::fabs(z.re.hi) > 1.0) {
        const bool want_neg = (detail::copysign(1.0, z.re.hi) ==
                               detail::copysign(1.0, z.im.hi));
        const bool have_neg = (detail::copysign(1.0, root.im.hi) < 0.0);
        if (want_neg != have_neg) root.im = negate(root.im);
    }
    DoubleDoubleComplex sum = iz + root;
    DoubleDoubleComplex lg  = log(sum);
    // multiply by -i: (a+bi)*(-i) = b - ai
    return DoubleDoubleComplex(lg.im, negate(lg.re));
}
// Principal sqrt of (u, v) with the sign of a ZERO v respected. The header's
// complex sqrt above tests `z.im.hi < 0.0`, which is false for -0.0, so for u < 0
// it puts BOTH zero conventions on the +i sheet. That is the same class of defect
// as KI-5(d) and it is corrected locally here rather than inside sqrt itself,
// which would move every other caller (acosh included) in one undocumented step.
// The general sqrt defect is recorded in docs/KNOWN_ISSUES.md, not fixed here.
// `vsign` is +1/-1, the intended sign of v when v is a zero: multiply_scalar does
// NOT carry a signed zero through (it renormalizes, and quick_two_sum(-0,+0) is
// +0), so the caller passes the sign it read off the original Im(z) rather than
// trusting the halved copy.
XPMATH_INLINE_FUNCTION DoubleDoubleComplex sqrt_signed_cut(DoubleDouble u, DoubleDouble v,
                                                           double vsign) {
    DoubleDoubleComplex r = sqrt(DoubleDoubleComplex(u, v));
    if (v.hi == 0.0 && u.hi < 0.0 && vsign != detail::copysign(1.0, r.im.hi))
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
// -- acos(2 + 1e-20i) scored 11.31 on DD, 0.00 on FF -- because asin there is
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex acos(DoubleDoubleComplex z) {
    const DoubleDouble one(1.0);
    const DoubleDouble half_im = multiply_scalar(z.im, 0.5);
    const double s_im = detail::copysign(1.0, z.im.hi);
    DoubleDoubleComplex rp = sqrt_signed_cut(
        multiply_scalar(add(one, z.re), 0.5), half_im, s_im);
    DoubleDoubleComplex rm = sqrt_signed_cut(
        multiply_scalar(subtract(one, z.re), 0.5), negate(half_im), -s_im);
    // w = rp + i*rm, with i*(a+bi) = -b + ai
    DoubleDoubleComplex w(subtract(rp.re, rm.im), add(rp.im, rm.re));
    return DoubleDoubleComplex(multiply_scalar(atan2(w.im, w.re), 2.0), negate(asin(z).im));
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
XPMATH_INLINE_FUNCTION DoubleDouble xp_log_hypot2(DoubleDouble a, DoubleDouble b) {
    DoubleDouble s = a, t = b;
    if (s.hi < 0.0) s = negate(s);
    if (t.hi < 0.0) t = negate(t);
    if (s.hi < t.hi) { DoubleDouble tmp = s; s = t; t = tmp; }
    // Both operands zero: the pole itself. Returning the extended log(0) here
    // does NOT work -- feeding -inf into the caller's subtract() makes the
    // error limb inf - inf = NaN and destroys the whole result -- so the two
    // poles are intercepted at the top of atan()/atanh() instead.
    if (s.hi == 0.0) return log(s);
    const DoubleDouble r = divide(t, s);
    return add(multiply_scalar(log(s), 2.0), log1p(multiply(r, r)));
}
// atan2()/angle() forms hypot(a, b) internally, so an operand pair whose
// SQUARES overflow the word format comes back NaN. That is what turned
// atan(1e10 + 0i) into NaN in the three FP32-word backends once the component
// form below started handing atan2 the raw 1 - x^2 - y^2 (monotone gate sweep
// point 1628, axis family: 14.00 -> 0.00). arg() is scale-invariant, so both
// operands are scaled down by a common EXACT power of two until the squares
// fit; being exact, the ratio -- and hence the answer -- is untouched. The
// loop runs at most once for every input the callers admit.
XPMATH_INLINE_FUNCTION DoubleDouble xp_atan2_safe(DoubleDouble a, DoubleDouble b) {
    const double kXpAtan2Safe = 1.0e150;
    const double kXpAtan2Down = 7.4583407312002070e-155;   // exact power of two
    double m = detail::fabs(a.hi) > detail::fabs(b.hi) ? detail::fabs(a.hi)
                                                     : detail::fabs(b.hi);
    while (m > kXpAtan2Safe) {
        if (a.hi != 0.0) a = multiply_scalar(a, kXpAtan2Down);
        if (b.hi != 0.0) b = multiply_scalar(b, kXpAtan2Down);
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex atan(DoubleDoubleComplex z) {
    // C99 Annex G poles: catan(+-0 +- 1i) = +-0 +- inf*i. No formulation
    // built out of the extended log() can produce the infinity, because that
    // log() reports "non-positive argument" and returns 0 -- at HEAD these two
    // points came back (0, 0), a finite wrong answer. Intercept them.
    {
        const DoubleDouble ay_ = z.im.hi < 0.0 ? negate(z.im) : z.im;
        if (z.re.hi == 0.0 && subtract(DoubleDouble(1.0), ay_).hi == 0.0) {
            double inf_ = -detail::log(double(0));
            if (z.im.hi < 0.0) inf_ = -inf_;
            return DoubleDoubleComplex(z.re, DoubleDouble(inf_));
        }
    }
    const double kXpAtanBigL     = 1.0e150;
    const double kXpAtanBigRatio = 1.0e4;
    const DoubleDouble one = DoubleDouble(1.0);
    if (detail::fabs(z.re.hi) < kXpAtanBigL && detail::fabs(z.im.hi) < kXpAtanBigL) {
        const DoubleDouble x2 = multiply(z.re, z.re);
        const DoubleDouble y2 = multiply(z.im, z.im);
        DoubleDouble twox = multiply_scalar(z.re, 2.0);
        if (z.re.hi == 0.0) twox = z.re;          // keep the signed zero
        // 1 - x^2 - y^2 as (1-y)(1+y) - x^2. Forming `1 - (x^2 + y^2)` instead
        // rounds x^2 + y^2 to ONE word before the cancellation, so at |y| ~ 1
        // -- exactly the atan branch cut -- the tiny x^2 that survives is left
        // with only word-0 precision. The factored form is exact there
        // (Sterbenz on both factors) and costs one extra multiply.
        const DoubleDouble d2 =
            subtract(multiply(subtract(one, z.im), add(one, z.im)), x2);
        DoubleDouble re = multiply_scalar(xp_atan2_safe(twox, d2), 0.5);
        // ON THE CUT (Re(z) a zero, |Im z| > 1) the sheet is chosen by the SIGN
        // of that zero -- atan(+0 + 2i) = +pi/2 + 0.5493i, atan(-0 + 2i) =
        // -pi/2 + 0.5493i. atan2() is handed the zero verbatim above but does
        // not carry its sign through, so +-pi/2 is installed directly, which is
        // the same correction atanh() below already makes on its own cut. The
        // monotone gate is what caught this: 30.74 -> 0.00 on DD at z = -0 + 2i.
        if (z.re.hi == 0.0 && d2.hi < 0.0) {
            re = multiply_scalar(DoubleDouble_pi(), 0.5);
            if (detail::copysign(1.0, z.re.hi) < 0.0) re = negate(re);
        }
        const DoubleDouble omy = subtract(one, z.im);
        const DoubleDouble den = add(x2, multiply(omy, omy));
        const DoubleDouble num = multiply_scalar(z.im, 4.0);
        DoubleDouble im;
        if (num.hi < den.hi * kXpAtanBigRatio &&
            num.hi > -0.875 * den.hi) {
            im = multiply_scalar(log1p(divide(num, den)), 0.25);
        } else {
            im = multiply_scalar(
                subtract(xp_log_hypot2(z.re, add(one, z.im)),
                         xp_log_hypot2(z.re, omy)), 0.25);
        }
        return DoubleDoubleComplex(re, im);
    }
    // |z| past sqrt(word range): squaring would overflow. The ratio form is
    // well behaved out here and is kept.
    DoubleDoubleComplex iz    = DoubleDoubleComplex(negate(z.im), z.re);
    DoubleDoubleComplex num   = DoubleDoubleComplex(one) - iz;
    DoubleDoubleComplex den   = DoubleDoubleComplex(one) + iz;
    DoubleDoubleComplex ratio = num / den;
    DoubleDoubleComplex lg    = log(ratio);
    // multiply by i/2: (a+bi)*(i/2) = (-b/2) + (a/2)*i
    return DoubleDoubleComplex(multiply_scalar(negate(lg.im), 0.5), multiply_scalar(lg.re, 0.5));
}

// ============================================================
// Complex hyperbolic
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex sinh(DoubleDoubleComplex z) {
    // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
    DoubleDouble ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION DoubleDoubleComplex cosh(DoubleDoubleComplex z) {
    // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
    DoubleDouble ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(ca, cb), multiply(sa, sb));
}
XPMATH_INLINE_FUNCTION DoubleDoubleComplex tanh(DoubleDoubleComplex z) {
    // KI-18: asymptotic branch, the tan() block above documents it in full.
    const double kXpTanAsymptote = 2.0;
    if (detail::fabs(z.re.hi) >= kXpTanAsymptote) {
        DoubleDouble cb, sb;
        sincos(z.im, cb, sb);
        const DoubleDouble s2 = multiply_scalar(multiply(sb, cb), 2.0);      // sin 2y
        const DoubleDouble c2 = multiply(subtract(cb, sb), add(cb, sb));         // cos 2y
        const DoubleDouble t  = exp(multiply_scalar(z.re, z.re.hi < 0.0 ? 2.0 : -2.0));
        const DoubleDouble t2 = multiply(t, t);
        const DoubleDouble den = add(add(DoubleDouble(1.0), t2), multiply_scalar(multiply(t, c2), 2.0));
        DoubleDouble re = divide(subtract(DoubleDouble(1.0), t2), den);
        if (z.re.hi < 0.0) re = negate(re);
        return DoubleDoubleComplex(re, divide(multiply_scalar(multiply(t, s2), 2.0), den));
    }
    // |Re z| < 1: the direct form is the more accurate of the two here and is
    // kept verbatim.
    // tanh(a+bi): re = tanh(a) / (cos^2(b) + tanh^2(a)*sin^2(b))
    //             im = sin(b)*cos(b)*(1 - tanh^2(a)) / (same denominator)
    DoubleDouble T_ = tanh(z.re);
    DoubleDouble cb, sb;
    sincos(z.im, cb, sb);
    DoubleDouble T2    = multiply(T_, T_);
    DoubleDouble denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return DoubleDoubleComplex(divide(T_, denom),
               divide(multiply(multiply(sb, cb), subtract(DoubleDouble(1.0), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
// asinh(z) = log(z + sqrt(z^2 + 1)), reflected into the right half-plane when
// the direct form would cancel.
//
// KI-5(a) (fixed 2026-09-02, see docs/KNOWN_ISSUES.md). For Re(z) < 0 the
// identity is ill-conditioned: sqrt(z^2 + 1) -> -z, so the sum is a difference
// of near-equal quantities. sqrt carries absolute error ~eps*|z|, which the sum
// (magnitude ~1/(2|z|)) inflates to a RELATIVE error of ~2|z|^2 * eps — two
// digits lost per decade of |z|, reaching total loss once the cancellation
// consumes the whole word. asinh is an ODD function, so -asinh(-z) moves the
// evaluation into the well-conditioned half-plane at the cost of two sign
// flips, which are exact. No new series and no new constants.
//
// THRESHOLD (kXpAsinhReflect = 4). The reflection is gated on magnitude as well
// as sign, which is a deliberate departure from the plain `Re(z) < 0` rule.
// The loss the reflection removes is log10(2|z|^2) digits; the reflection
// substitutes a different rounding path, worth a few tenths of a digit in
// either direction. Below |z| ~ 4 the loss it removes is smaller than the churn
// it introduces, so applying it there is a net harm — measured, not assumed. On
// the 1780-point complex sweep grid, reflecting unconditionally moved 535
// points DOWN and 396 up in the |z| <= 1 bin (worst -6.77 digits, QF at
// z = -1e-16, where asinh(z) ~ z and there was never any cancellation to fix),
// while |z| > 4 was 859 up against 25 down (best +28.22, DD at z = -1e15).
// Gating at 4 keeps every one of those 859 improvements and removes 944 of the
// 969 regressions. 4 also sits comfortably inside the region where the
// asymptotic argument holds (log10(2*16) = 1.5 digits already at stake) and is
// exactly representable, so the comparison itself is exact.
//
// The test is L-infinity on the LEADING limbs — max(-Re, |Im|) > 4 — not
// hypot(). That is deliberate: it cannot overflow for any finite z (hypot on
// z ~ 1e200 would), it costs two compares instead of two multiplies and a
// sqrt, and the leading limb settles the comparison outright unless |z| is
// within one ulp of the threshold, where either branch is equally good. It
// selects the L-inf ball of radius 4 rather than the L2 ball, i.e. it also
// reflects part of the annulus 4 < |z| <= 4*sqrt(2); the binning above shows
// that band behaves like the |z| > 4 side.
//
// BOUNDARY: the sign predicate is `Re(z) < 0`, so Re(z) == +0 AND Re(z) == -0
// both take the direct branch, exactly as before this change. Two reasons.
// (1) There is no cancellation to avoid on the imaginary axis — for z = iy the
// sum is i*(y + sqrt(y^2 - 1)), like signs — so the reflection would buy
// nothing. (2) asinh's branch cuts LIVE on the imaginary axis (|Im| > 1), where
// the sign of a zero real part selects the sheet; routing -0 through a negation
// would rewrite that selection. Leaving Re == -0 on the direct branch keeps the
// on-cut behaviour bit-for-bit what it was. (The headers' on-cut handling for
// Re(z) == -0 is separately wrong — it is the asinh analogue of KI-5(d) and is
// NOT addressed here.) All four complex headers use this same predicate and the
// same threshold.
XPMATH_INLINE_FUNCTION DoubleDoubleComplex asinh(DoubleDoubleComplex z) {
    const double t = 4.0;   // kXpAsinhReflect
    if (z.re.hi < 0.0 && (-z.re.hi > t || detail::fabs(z.im.hi) > t)) {
        DoubleDoubleComplex w = -z;
        return -log(w + sqrt(w*w + DoubleDoubleComplex(DoubleDouble(1.0))));
    }
    return log(z + sqrt(z*z + DoubleDoubleComplex(DoubleDouble(1.0))));
}
// KI-1 fix. acosh(z) = 2*log( sqrt((z+1)/2) + sqrt((z-1)/2) ) -- Kahan 1987,
// "Branch Cuts for Complex Elementary Functions". The older form
// log(z + sqrt(z*z - 1)) takes the WRONG sqrt sheet throughout Re(z) < 0 and
// returns essentially -acosh(z) there, an O(1) wrong value rather than lost
// digits. Kahan's form is branch-correct with no case analysis: for Re(z) < 0
// both roots are near-purely-imaginary with the same sign, for Re(z) > 0 both
// are near-real positive, so the addition never subtracts. It also never forms
// z*z, which is what made the old form overflow to Inf/NaN above sqrt of the
// word type's range (~1.3e154 for the FP64-word backend, ~1.8e19 for the
// FP32-word ones). Halving is per component via multiply_scalar, exact because
// 0.5 is a power of two; a complex multiply by (0.5, 0) would round.
XPMATH_INLINE_FUNCTION DoubleDoubleComplex acosh(DoubleDoubleComplex z) {
    const DoubleDouble one(1.0);
    const DoubleDouble half_im = multiply_scalar(z.im, 0.5);
    DoubleDoubleComplex rp = sqrt(DoubleDoubleComplex(
        multiply_scalar(add(z.re, one), 0.5), half_im));
    DoubleDoubleComplex rm = sqrt(DoubleDoubleComplex(
        multiply_scalar(subtract(z.re, one), 0.5), half_im));
    DoubleDoubleComplex lg = log(rp + rm);
    return DoubleDoubleComplex(multiply_scalar(lg.re, 2.0),
                               multiply_scalar(lg.im, 2.0));
}
// KI-5(b) fix. Two independent defects lived in the old one-line body
// `0.5*log((1+z)/(1-z))`, and both are repaired here.
//
// (1) CONDITIONING AS z -> 0. The ratio (1+z)/(1-z) -> 1, so the log is taken of
// a number whose entire information content sits below the leading 1. The
// argument reduction throws away log10(1/|z|) digits before log() is even
// entered, and the measured score falls off one digit per decade of |z|. The
// remedy is the classical one: 0.5*log1p(2z/(1-z)), expanded here into its real
// and imaginary components so no complex divide is needed either --
//
//     Re atanh(z) = 0.25 * log1p( 4x / ((1-x)^2 + y^2) )
//     Im atanh(z) = 0.5  * atan2( 2y, 1 - x^2 - y^2 )
//
// which is 0.5*log1p(w) with w = 2z/(1-z) written out: 2*Re(w) + |w|^2 collapses
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
// overflow defect. Second, at |z| >= 0.5 the old form already scores at the
// type's cap (measured: DD 30.74 at z = 0.5), so there is nothing to win
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
// returned +pi/2 for both conventions: every `x -0i` cut point scored 0.00 on
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex atanh(DoubleDoubleComplex z) {
    // C99 Annex G poles: catanh(+-1 +- 0i) = +-inf +- 0i. Same reason as atan
    // above -- at HEAD these came back (0, 0).
    {
        const DoubleDouble ax_ = z.re.hi < 0.0 ? negate(z.re) : z.re;
        if (z.im.hi == 0.0 && subtract(DoubleDouble(1.0), ax_).hi == 0.0) {
            double inf_ = -detail::log(double(0));
            if (z.re.hi < 0.0) inf_ = -inf_;
            return DoubleDoubleComplex(DoubleDouble(inf_), z.im);
        }
    }
    const DoubleDouble one = DoubleDouble(1.0);
    const double kXpAtanhSmall    = 0.0625;
    const double kXpAtanhWide     = 0.5;
    const double kXpAtanhBigL     = 1.0e150;
    const double kXpAtanBigRatio  = 1.0e4;
    const double ax = detail::fabs(z.re.hi), ay = detail::fabs(z.im.hi);
    const double linf = ax > ay ? ax : ay;
    DoubleDoubleComplex r;
    if (linf < kXpAtanhSmall || (linf >= kXpAtanhWide && linf < kXpAtanhBigL)) {
        const DoubleDouble omx = subtract(one, z.re);
        const DoubleDouble y2  = multiply(z.im, z.im);
        const DoubleDouble den = add(multiply(omx, omx), y2);
        const DoubleDouble num = multiply_scalar(z.re, 4.0);
        if (num.hi < den.hi * kXpAtanBigRatio &&
            num.hi > -0.875 * den.hi) {
            r.re = multiply_scalar(log1p(divide(num, den)), 0.25);
        } else {
            r.re = multiply_scalar(
                subtract(xp_log_hypot2(add(one, z.re), z.im),
                         xp_log_hypot2(omx, z.im)), 0.25);
        }
        DoubleDouble twoy = multiply_scalar(z.im, 2.0);
        if (z.im.hi == 0.0) twoy = z.im;          // keep the signed zero
        r.im = multiply_scalar(
            xp_atan2_safe(twoy, subtract(multiply(subtract(one, z.re), add(one, z.re)), y2)), 0.5);
    } else {
        DoubleDoubleComplex lg = log((DoubleDoubleComplex(one) + z) / (DoubleDoubleComplex(one) - z));
        r.re = multiply_scalar(lg.re, 0.5);
        r.im = multiply_scalar(lg.im, 0.5);
    }
    if (z.im.hi == 0.0 && detail::fabs(z.re.hi) > 1.0) {
        DoubleDouble half_pi = multiply_scalar(DoubleDouble_pi(), 0.5);
        if (detail::copysign(1.0, z.im.hi) < 0.0) half_pi = negate(half_pi);
        r.im = half_pi;
    }
    return r;
}

// ============================================================
// Complex power and polar
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDoubleComplex pow(DoubleDoubleComplex z, DoubleDoubleComplex w) {
    // z^w = exp(w * log(z))
    if (z.re.hi == 0.0 && z.im.hi == 0.0) return DoubleDoubleComplex();
    return exp(w * log(z));
}

// polar(r, theta) = r * (cos(theta) + i*sin(theta))
XPMATH_INLINE_FUNCTION DoubleDoubleComplex polar(DoubleDouble r, DoubleDouble theta) {
    DoubleDouble c, s;
    sincos(theta, c, s);
    return DoubleDoubleComplex(multiply(r, c), multiply(r, s));
}

} // namespace xp
