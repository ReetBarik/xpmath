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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex acos(DoubleDoubleComplex z) {
    // acos(z) = pi/2 - asin(z)
    DoubleDouble pi_over_2 = multiply_scalar(DoubleDouble_pi(), 0.5);
    DoubleDoubleComplex asin_z  = asin(z);
    return DoubleDoubleComplex(subtract(pi_over_2, asin_z.re), negate(asin_z.im));
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
XPMATH_INLINE_FUNCTION DoubleDoubleComplex atanh(DoubleDoubleComplex z) {
    // atanh(z) = (1/2)*log((1+z)/(1-z))
    DoubleDoubleComplex one = DoubleDoubleComplex(DoubleDouble(1.0));
    DoubleDoubleComplex lg  = log((one + z) / (one - z));
    return DoubleDoubleComplex(multiply_scalar(lg.re, 0.5), multiply_scalar(lg.im, 0.5));
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
