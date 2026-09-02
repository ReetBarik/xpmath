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

XPMATH_INLINE_FUNCTION DoubleDouble abs(DoubleDoubleComplex z) {
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
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
    DoubleDouble r  = sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex tan(DoubleDoubleComplex z) {
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex atan(DoubleDoubleComplex z) {
    // atan(z) = (i/2) * log((1-iz)/(1+iz))
    DoubleDoubleComplex iz    = DoubleDoubleComplex(negate(z.im), z.re);
    DoubleDoubleComplex num   = DoubleDoubleComplex(DoubleDouble(1.0)) - iz;
    DoubleDoubleComplex den   = DoubleDoubleComplex(DoubleDouble(1.0)) + iz;
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
    // tanh(a+bi): re = tanh(a) / (cos²(b) + tanh²(a)·sin²(b))
    //             im = sin(b)·cos(b)·(1 - tanh²(a)) / (cos²(b) + tanh²(a)·sin²(b))
    // Denominator ≥ 0 always; uses improved real tanh to avoid cancellation.
    DoubleDouble T = tanh(z.re);
    DoubleDouble cb, sb;
    sincos(z.im, cb, sb);
    DoubleDouble T2    = multiply(T, T);
    DoubleDouble denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return DoubleDoubleComplex(divide(T, denom),
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex atanh(DoubleDoubleComplex z) {
    const DoubleDouble one(1.0);
    const double t = 0.0625;   // kXpAtanhSmall
    DoubleDoubleComplex r;
    if (detail::fabs(z.re.hi) < t && detail::fabs(z.im.hi) < t) {
        DoubleDouble omx = subtract(one, z.re);
        DoubleDouble den = add(multiply(omx, omx), multiply(z.im, z.im));
        r.re = multiply_scalar(
            log1p(divide(multiply_scalar(z.re, 4.0), den)), 0.25);
        r.im = multiply_scalar(
            atan2(multiply_scalar(z.im, 2.0),
                  subtract(one, add(multiply(z.re, z.re), multiply(z.im, z.im)))),
            0.5);
    } else {
        DoubleDoubleComplex lg = log((DoubleDoubleComplex(one) + z) /
                                     (DoubleDoubleComplex(one) - z));
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
