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
XPMATH_INLINE_FUNCTION FloatFloatComplex acos(FloatFloatComplex z) {
    // acos(z) = pi/2 - asin(z)
    FloatFloat pi_over_2 = multiply_scalar(FloatFloat_pi(), 0.5f);
    FloatFloatComplex asin_z  = asin(z);
    return FloatFloatComplex(subtract(pi_over_2, asin_z.re), negate(asin_z.im));
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
XPMATH_INLINE_FUNCTION FloatFloatComplex atanh(FloatFloatComplex z) {
    // atanh(z) = (1/2)*log((1+z)/(1-z))
    FloatFloatComplex one = FloatFloatComplex(FloatFloat(1.0f));
    FloatFloatComplex lg  = log((one + z) / (one - z));
    return FloatFloatComplex(multiply_scalar(lg.re, 0.5f), multiply_scalar(lg.im, 0.5f));
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
