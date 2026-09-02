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
    XPMATH_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex b) const {
        return FloatFloatComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                         add(multiply(re, b.im), multiply(im, b.re)));
    }
    XPMATH_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex b) const {
        if (b.re.hi == 0.0f && b.im.hi == 0.0f) {
            XPMATH_PRINTF("FFCOMPLEX: division by zero\n");
            return FloatFloatComplex();
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

XPMATH_INLINE_FUNCTION FloatFloat abs(FloatFloatComplex z) {
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex conj(FloatFloatComplex z) {
    return FloatFloatComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root
// ============================================================
XPMATH_INLINE_FUNCTION FloatFloatComplex sqrt(FloatFloatComplex z) {
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return FloatFloatComplex();
    FloatFloat r  = sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
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
        if (z.im.hi < 0.0f) b.im = negate(b.im);
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
XPMATH_INLINE_FUNCTION FloatFloatComplex tan(FloatFloatComplex z) {
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
    FloatFloatComplex lg  = log(sum);
    // multiply by -i: (a+bi)*(-i) = b - ai
    return FloatFloatComplex(lg.im, negate(lg.re));
}
// Principal sqrt of (u, v) with the sign of a ZERO v respected. The header's
// complex sqrt above tests `z.im.hi < 0.0f`, which is false for -0.0f, so for u < 0
// it puts BOTH zero conventions on the +i sheet. That is the same class of defect
// as KI-5(d) and it is corrected locally here rather than inside sqrt itself,
// which would move every other caller (acosh included) in one undocumented step.
// The general sqrt defect is recorded in docs/KNOWN_ISSUES.md, not fixed here.
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
XPMATH_INLINE_FUNCTION FloatFloatComplex atan(FloatFloatComplex z) {
    // atan(z) = (i/2) * log((1-iz)/(1+iz))
    FloatFloatComplex iz    = FloatFloatComplex(negate(z.im), z.re);
    FloatFloatComplex num   = FloatFloatComplex(FloatFloat(1.0f)) - iz;
    FloatFloatComplex den   = FloatFloatComplex(FloatFloat(1.0f)) + iz;
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
    // tanh(a+bi): re = tanh(a) / (cos²(b) + tanh²(a)·sin²(b))
    //             im = sin(b)·cos(b)·(1 - tanh²(a)) / (cos²(b) + tanh²(a)·sin²(b))
    // Denominator ≥ 0 always; uses improved real tanh to avoid cancellation.
    FloatFloat T = tanh(z.re);
    FloatFloat cb, sb;
    sincos(z.im, cb, sb);
    FloatFloat T2    = multiply(T, T);
    FloatFloat denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return FloatFloatComplex(divide(T, denom),
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
XPMATH_INLINE_FUNCTION FloatFloatComplex asinh(FloatFloatComplex z) {
    const float t = 4.0f;   // kXpAsinhReflect
    if (z.re.hi < 0.0f && (-z.re.hi > t || detail::fabs(z.im.hi) > t)) {
        FloatFloatComplex w = -z;
        return -log(w + sqrt(w*w + FloatFloatComplex(FloatFloat(1.0f))));
    }
    return log(z + sqrt(z*z + FloatFloatComplex(FloatFloat(1.0f))));
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
    return FloatFloatComplex(multiply_scalar(lg.re, 2.0f),
                             multiply_scalar(lg.im, 2.0f));
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
XPMATH_INLINE_FUNCTION FloatFloatComplex atanh(FloatFloatComplex z) {
    const FloatFloat one(1.0f);
    const float t = 0.0625f;   // kXpAtanhSmall
    FloatFloatComplex r;
    if (detail::fabs(z.re.hi) < t && detail::fabs(z.im.hi) < t) {
        FloatFloat omx = subtract(one, z.re);
        FloatFloat den = add(multiply(omx, omx), multiply(z.im, z.im));
        r.re = multiply_scalar(
            log1p(divide(multiply_scalar(z.re, 4.0f), den)), 0.25f);
        r.im = multiply_scalar(
            atan2(multiply_scalar(z.im, 2.0f),
                  subtract(one, add(multiply(z.re, z.re), multiply(z.im, z.im)))),
            0.5f);
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
