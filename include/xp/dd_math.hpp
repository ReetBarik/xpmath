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
// This C++ port is a derivative work distributed under the same
// DHB-License. See §3 of that license regarding upstream
// contribution rights.
//
// Modifications from the original DDFUN v04 sources:
//   * Translated from Fortran-90 (ddfuna.f90, ddfune.f90) to
//     header-only C++17.
//   * Every function XPMATH_INLINE_FUNCTION for host + device
//     portability across CUDA/HIP/SYCL/OpenMP-target.
//   * Namespaced as xp::DoubleDouble with STL-style
//     free-function names.
//   * See docs/TEST_SUITE_PLAN.md "Upstreaming considerations" for
//     naming and API conventions.

#pragma once

// Double-double real arithmetic — xp::DoubleDouble. ~31 decimal digits
// from an unevaluated sum of two FP64 components (hi + lo, |lo| <= ulp(hi)/2).
//
// Ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab) Fortran
// sources (ddfuna.f90, ddfune.f90).
//
// DEPENDENCIES: none beyond the C++17 standard library. In particular this
// header does NOT include or require Kokkos — see xp/config.hpp for how the
// four portability facilities it needs (inline annotation, on-device
// detection, scalar math dispatch, diagnostic printf) are supplied. Kokkos
// users get today's `Kokkos::Experimental::DoubleDouble` API unchanged through
// the compat wrapper at third_party/include/dd_math.hpp, which is the only
// place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// Naming conventions (T0.4):
//   * Type + math live in one flat namespace so an upstream move is
//     mechanical rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions DoubleDouble_pi(), DoubleDouble_e(), ...
//     Chosen over a constants::pi<DoubleDouble>() template because it mirrors
//     Kokkos's existing M_PI-style accessors and reads shorter at the call site;
//     they cannot be constexpr template variables (Kokkos::numbers style) because
//     each is built at runtime from IEEE-754 bit patterns, not a literal.
//   * The former bit-pattern constructor became the static factory
//     DoubleDouble::from_bits(hi, lo): it is namespaced to the type,
//     discoverable, and needs no free-function symbol.
//   * Math functions are ADL-findable via the argument's namespace. The
//     `Kokkos::`-namespace forwarding overloads that used to sit at the bottom
//     of this header (so Kokkos::exp(dd) works like Kokkos::exp(double)) now
//     live in the compat wrapper, since they are Kokkos-specific API surface.
//     add/subtract/multiply/divide are not forwarded — they are for operators
//     and explicit ADL only.

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
struct DoubleDouble;
XPMATH_INLINE_FUNCTION DoubleDouble add(DoubleDouble a, DoubleDouble b);
XPMATH_INLINE_FUNCTION DoubleDouble subtract(DoubleDouble a, DoubleDouble b);
XPMATH_INLINE_FUNCTION DoubleDouble multiply(DoubleDouble a, DoubleDouble b);
XPMATH_INLINE_FUNCTION DoubleDouble divide(DoubleDouble a, DoubleDouble b);
XPMATH_INLINE_FUNCTION DoubleDouble multiply_scalar(DoubleDouble a, double b);
XPMATH_INLINE_FUNCTION DoubleDouble divide_scalar(DoubleDouble a, double b);
XPMATH_INLINE_FUNCTION DoubleDouble negate(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble abs(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble sqrt(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble round_to_nearest_int(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble pow_int(DoubleDouble a, int n);
XPMATH_INLINE_FUNCTION DoubleDouble exp(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble log(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble pow(DoubleDouble a, DoubleDouble b);
XPMATH_INLINE_FUNCTION void   sinhcosh(DoubleDouble a, DoubleDouble& x, DoubleDouble& y);
XPMATH_INLINE_FUNCTION void   sincos(DoubleDouble a, DoubleDouble& x, DoubleDouble& y);
XPMATH_INLINE_FUNCTION DoubleDouble angle(DoubleDouble x, DoubleDouble y);

// ============================================================
// DoubleDouble struct
// ============================================================
struct DoubleDouble {
    double hi;
    double lo;

    XPMATH_INLINE_FUNCTION DoubleDouble() : hi(0.0), lo(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDouble(double h) : hi(h), lo(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDouble(double h, double l) : hi(h), lo(l) {}
    XPMATH_INLINE_FUNCTION DoubleDouble(const DoubleDouble& o) : hi(o.hi), lo(o.lo) {}
    XPMATH_INLINE_FUNCTION DoubleDouble& operator=(const DoubleDouble& o) { hi=o.hi; lo=o.lo; return *this; }

    // Factory: build a DoubleDouble from the IEEE-754 bit patterns of its two
    // components. Safe on host (memcpy) and device (__longlong_as_double).
    // Replaces the former free bit-pattern constructor function.
    //
    // The guard is XPMATH_ON_DEVICE_CUDA_OR_HIP, deliberately NOT the general
    // XPMATH_ON_DEVICE: __longlong_as_double is a CUDA/HIP global-namespace
    // intrinsic with no SYCL equivalent, so a SYCL device build must take the
    // std::memcpy path. Both paths are the identical bit reinterpretation, so
    // this widening is a portability fix with no numerical content.
    static XPMATH_INLINE_FUNCTION DoubleDouble from_bits(uint64_t hi_bits, uint64_t lo_bits) {
        double h, l;
#if !defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
        std::memcpy(&h, &hi_bits, sizeof(double));
        std::memcpy(&l, &lo_bits, sizeof(double));
#else
        h = __longlong_as_double(static_cast<long long>(hi_bits));
        l = __longlong_as_double(static_cast<long long>(lo_bits));
#endif
        return DoubleDouble(h, l);
    }

    XPMATH_INLINE_FUNCTION DoubleDouble operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator+(DoubleDouble b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator-(DoubleDouble b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator*(DoubleDouble b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator/(DoubleDouble b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator*(double b)  const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator/(double b)  const { return divide_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator+(double b)  const { return add(*this, DoubleDouble(b)); }
    XPMATH_INLINE_FUNCTION DoubleDouble operator-(double b)  const { return subtract(*this, DoubleDouble(b)); }

    XPMATH_INLINE_FUNCTION DoubleDouble& operator+=(DoubleDouble b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator-=(DoubleDouble b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator*=(DoubleDouble b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator/=(DoubleDouble b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator+=(double b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator-=(double b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator*=(double b) { *this = multiply_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION DoubleDouble& operator/=(double b) { *this = divide_scalar(*this, b); return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(DoubleDouble b) const { return hi==b.hi && lo==b.lo; }
    XPMATH_INLINE_FUNCTION bool operator!=(DoubleDouble b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(DoubleDouble b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    XPMATH_INLINE_FUNCTION bool operator>(DoubleDouble b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    XPMATH_INLINE_FUNCTION bool operator<=(DoubleDouble b) const { return !(b < *this); }
    XPMATH_INLINE_FUNCTION bool operator>=(DoubleDouble b) const { return !(*this < b); }
};

XPMATH_INLINE_FUNCTION DoubleDouble operator+(double a, DoubleDouble b) { return add(DoubleDouble(a), b); }
XPMATH_INLINE_FUNCTION DoubleDouble operator-(double a, DoubleDouble b) { return subtract(DoubleDouble(a), b); }
XPMATH_INLINE_FUNCTION DoubleDouble operator*(double a, DoubleDouble b) { return multiply_scalar(b, a); }
XPMATH_INLINE_FUNCTION DoubleDouble operator/(double a, DoubleDouble b) { return divide(DoubleDouble(a), b); }

// Host-only: <ostream> is not usable in a device compilation pass. Guarded by
// XPMATH_ON_DEVICE (all three vendors), not the CUDA-only spelling.
#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const DoubleDouble& d) {
    os << "[" << std::setprecision(16) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
XPMATH_INLINE_FUNCTION DoubleDouble DoubleDouble_pi()          { return DoubleDouble::from_bits(0x400921fb54442d18ULL, 0x3ca1a62633145c07ULL); }
XPMATH_INLINE_FUNCTION DoubleDouble DoubleDouble_e()           { return DoubleDouble::from_bits(0x4005bf0a8b145769ULL, 0x3ca4d57ee2b1013aULL); }
XPMATH_INLINE_FUNCTION DoubleDouble DoubleDouble_log2()        { return DoubleDouble::from_bits(0x3fe62e42fefa39efULL, 0x3c7abc9e3b39803fULL); }
XPMATH_INLINE_FUNCTION DoubleDouble DoubleDouble_log10()       { return DoubleDouble::from_bits(0x40026bb1bbb55516ULL, 0xbcaf48ad494ea3eaULL); } // ln(10)
XPMATH_INLINE_FUNCTION DoubleDouble DoubleDouble_sqrt2()       { return DoubleDouble::from_bits(0x3ff6a09e667f3bcdULL, 0xbc9bdd3413b26456ULL); }
XPMATH_INLINE_FUNCTION DoubleDouble DoubleDouble_euler_gamma() { return DoubleDouble::from_bits(0x3fe2788cfc6fb619ULL, 0xbc56cb90701fbfabULL); }

// ============================================================
// Primitive arithmetic
// ============================================================

XPMATH_INLINE_FUNCTION DoubleDouble negate(DoubleDouble a) {
    return DoubleDouble(-a.hi, -a.lo);
}

// TwoSum (Knuth)
XPMATH_INLINE_FUNCTION DoubleDouble add(DoubleDouble a, DoubleDouble b) {
    double t1 = a.hi + b.hi;
    double e  = t1 - a.hi;
    double t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    double hi = t1 + t2;
    double lo = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

XPMATH_INLINE_FUNCTION DoubleDouble subtract(DoubleDouble a, DoubleDouble b) {
    double t1 = a.hi - b.hi;
    double e  = t1 - a.hi;
    double t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    double hi = t1 + t2;
    double lo = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

// TwoProduct (Dekker splitting)
XPMATH_INLINE_FUNCTION DoubleDouble multiply(DoubleDouble a, DoubleDouble b) {
    const double split = 134217729.0;
    double cona = a.hi * split, conb = b.hi * split;
    double a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    double a2 = a.hi - a1,           b2 = b.hi - b1;
    double c11 = a.hi * b.hi;
    double c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2  = a.hi * b.lo + a.lo * b.hi;
    double t1  = c11 + c2;
    double e   = t1 - c11;
    double t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    double hi  = t1 + t2;
    double lo  = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

XPMATH_INLINE_FUNCTION DoubleDouble divide(DoubleDouble a, DoubleDouble b) {
    const double split = 134217729.0;
    double s1  = a.hi / b.hi;
    double cona = s1 * split, conb = b.hi * split;
    double a1  = cona - (cona - s1), b1 = conb - (conb - b.hi);
    double a2  = s1 - a1,            b2 = b.hi - b1;
    double c11 = s1 * b.hi;
    double c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2  = s1 * b.lo;
    double t1  = c11 + c2;
    double e   = t1 - c11;
    double t2  = ((c2 - e) + (c11 - (t1 - e))) + c21;
    double t12 = t1 + t2;
    double t22 = t2 - (t12 - t1);
    double t11 = a.hi - t12;
    e = t11 - a.hi;
    double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    double s2  = (t11 + t21) / b.hi;
    double hi  = s1 + s2;
    double lo  = s2 - (hi - s1);
    return DoubleDouble(hi, lo);
}

XPMATH_INLINE_FUNCTION DoubleDouble multiply_scalar(DoubleDouble a, double b) {
    const double split = 134217729.0;
    double cona = a.hi * split, conb = b * split;
    double a1   = cona - (cona - a.hi), b1 = conb - (conb - b);
    double a2   = a.hi - a1,            b2 = b - b1;
    double c11  = a.hi * b;
    double c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2   = a.lo * b;
    double t1   = c11 + c2;
    double e    = t1 - c11;
    double t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    double hi   = t1 + t2;
    double lo   = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

XPMATH_INLINE_FUNCTION DoubleDouble divide_scalar(DoubleDouble a, double b) {
    const double split = 134217729.0;
    double t1  = a.hi / b;
    double cona = t1 * split, conb = b * split;
    double a1  = cona - (cona - t1), b1 = conb - (conb - b);
    double a2  = t1 - a1,            b2 = b - b1;
    double t12 = t1 * b;
    double t22 = (((a1*b1 - t12) + a1*b2) + a2*b1) + a2*b2;
    double t11 = a.hi - t12;
    double e   = t11 - a.hi;
    double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    double t2  = (t11 + t21) / b;
    double hi  = t1 + t2;
    double lo  = t2 - (hi - t1);
    return DoubleDouble(hi, lo);
}

// Exact product of two doubles
XPMATH_INLINE_FUNCTION DoubleDouble two_prod(double da, double db) {
    const double split = 134217729.0;
    double cona = da * split, conb = db * split;
    double a1   = cona - (cona - da), b1 = conb - (conb - db);
    double a2   = da - a1,            b2 = db - b1;
    double s1   = da * db;
    double s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return DoubleDouble(s1, s2);
}

// ============================================================
// Basic math
// ============================================================

XPMATH_INLINE_FUNCTION DoubleDouble abs(DoubleDouble a) {
    return (a.hi >= 0.0) ? a : DoubleDouble(-a.hi, -a.lo);
}

// Nearest integer, TIES TO EVEN (DDFUN's dnint: a DD-level magic constant,
// 2^105 + 2^52, added and subtracted so the DD add's own rounding does the
// work).
//
// KI-2 (2026-09-02): DD is NOT one of the two backends KI-2 affects, and this
// routine is deliberately left alone. KI-2 is QD's `floor(d + 0.5)`
// double-rounding, which QF and TF inherited and DD never used. Measured, not
// assumed: nint(0.49999999999999994) = 0 here, where the floor form gives 1.
// The magic-constant form is exact for every |a| < 2^105 — the fraction is
// rounded once, in the low word, ties to even — so there is nothing for `rint`
// to improve, and no scalar `rint` formulation reaches 106 bits anyway.
// Ties-to-even is also the shipped semantics of dd::round and what
// dd_accuracy_test.cpp's `nearbyintq` oracle expects.
// See docs/KNOWN_ISSUES.md, KI-2 resolution.
XPMATH_INLINE_FUNCTION DoubleDouble round_to_nearest_int(DoubleDouble a) {
    if (a.hi == 0.0) return DoubleDouble(0.0);
    const double T105 = detail::ldexp(1.0, 105); // 2^105
    const double T52  = detail::ldexp(1.0, 52);  // 2^52
    DoubleDouble CON = DoubleDouble(T105, T52);
    if (a.hi >= T105) {
        XPMATH_PRINTF("DDNINT: argument too large\n");
        return DoubleDouble(0.0);
    }
    if (a.hi > 0.0) return subtract(add(a, CON), CON);
    else            return add(subtract(a, CON), CON);
}

XPMATH_INLINE_FUNCTION DoubleDouble sqrt(DoubleDouble a) {
    if (a.hi == 0.0) return DoubleDouble(0.0);
    if (a.hi < 0.0) {
        XPMATH_PRINTF("DDSQRT: negative argument\n");
        return DoubleDouble(0.0);
    }
    double t1 = 1.0 / detail::sqrt(a.hi);
    double t2 = a.hi * t1;
    DoubleDouble s0 = two_prod(t2, t2);
    DoubleDouble s1 = subtract(a, s0);
    double t3  = 0.5 * s1.hi * t1;
    return add(DoubleDouble(t2), DoubleDouble(t3));
}

// Integer power
XPMATH_INLINE_FUNCTION DoubleDouble pow_int(DoubleDouble a, int n) {
    const double cl2 = 1.4426950408889633;
    if (a.hi == 0.0) {
        if (n >= 0) return DoubleDouble(0.0);
        XPMATH_PRINTF("DDNPWR: zero base with negative exponent\n");
        return DoubleDouble(0.0);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return DoubleDouble(1.0);
    if (nn == 1) return (n > 0) ? a : divide(DoubleDouble(1.0), a);
    if (nn == 2) { DoubleDouble r = multiply(a,a); return (n>0) ? r : divide(DoubleDouble(1.0),r); }
    int mn = (int)(cl2 * detail::log((double)nn) + 1.0 + 1.0e-14);
    DoubleDouble s0 = a, s2 = DoubleDouble(1.0);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = multiply(s2, s0);
        kn = kk;
        if (j < mn) s0 = multiply(s0, s0);
    }
    if (n < 0) s2 = divide(DoubleDouble(1.0), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

XPMATH_INLINE_FUNCTION DoubleDouble exp(DoubleDouble a) {
    const int nq = 6;
    const double eps = 1.0e-32;
    DoubleDouble al2 = DoubleDouble_log2();
    // KI-6: the guard is derived from the WORD range, not from a shared ±300
    // constant. e^x exceeds DBL_MAX (1.7977e308) above ln(DBL_MAX) =
    // 709.78271289338397, and falls below the smallest FP64 subnormal
    // (4.94e-324) below ln(2^-1074) = -745.13. Between those the result is
    // representable and must be returned, not flushed to zero — the old ±300
    // guard threw away ~170 decades that DD holds perfectly well. The guards
    // that remain exist only to bound `nz` before the (int) cast below.
    if (a.hi > 709.78271289338397) {
        XPMATH_PRINTF("DDEXP: overflow\n");
        return DoubleDouble(HUGE_VAL);          // e^x > DBL_MAX: +inf is the answer
    }
    if (a.hi < -745.2) return DoubleDouble(0.0); // e^x < 2^-1074: 0 is the answer

    DoubleDouble s0 = divide(a, al2);
    DoubleDouble s1 = round_to_nearest_int(s0);
    double t1  = s1.hi;
    int nz     = (int)(t1 + detail::copysign(1.0e-14, t1));
    s0 = subtract(a, multiply(al2, s1));

    if (s0.hi == 0.0) {
        return DoubleDouble(detail::ldexp(1.0, nz)); // result = 2^nz exactly
    }
    // Scale down by 2^nq then square nq times
    s1 = multiply_scalar(s0, detail::ldexp(1.0, -nq));
    DoubleDouble s2 = DoubleDouble(1.0), s3 = DoubleDouble(1.0);
    for (int l1 = 1; l1 <= 100; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (double)l1);
        s0 = add(s3, s2);
        s3 = s0;
        if (detail::fabs(s2.hi) <= eps * detail::fabs(s3.hi)) break;
        if (l1 == 100) { XPMATH_PRINTF("DDEXP: iteration limit\n"); return DoubleDouble(0.0); }
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, s3);

    // KI-6: scale by 2^nz through the EXPONENT, component-wise, not by forming
    // the factor 2^nz as a double and multiplying. ldexp(1.0, nz) is +inf for
    // nz >= 1024 and 0 for nz <= -1075, so the old multiply_scalar form could
    // not reach either end of the range even with the guard widened; and
    // multiply_scalar runs a Dekker split, which itself overflows above
    // ~1.3e300. Scaling each word by a power of two is exact (bar gradual
    // underflow at the very bottom, where the lo word is unrepresentable
    // anyway) and spans the full FP64 exponent range.
    // Inside the normal FP64 exponent band the factor 2^nz is itself exact and
    // representable, so a plain multiply is equivalent AND cheaper — the
    // compiler builds `ldexp(1.0, nz)` inline, whereas ldexp() on a general
    // mantissa is a libm call. Outside the band (only reachable at the extreme
    // ends of the newly-opened range) take the two calls.
    if (nz >= -1021 && nz <= 1023) {
        const double pow2 = detail::ldexp(1.0, nz);
        return DoubleDouble(s3.hi * pow2, s3.lo * pow2);
    }
    return DoubleDouble(detail::ldexp(s3.hi, nz), detail::ldexp(s3.lo, nz));
}

XPMATH_INLINE_FUNCTION DoubleDouble log(DoubleDouble a) {
    // KI-6: log(+inf) = +inf, returned directly. Since exp() now returns +inf
    // on genuine overflow instead of 0, the Newton step below would evaluate
    // (a - e^x)/e^x = (inf - inf)/inf = NaN on an infinite argument. atanh(±1)
    // reaches here through (1+a)/(1-a).
    if (detail::isinf(a.hi)) return a;
    if (a.hi <= 0.0) {
        XPMATH_PRINTF("DDLOG: non-positive argument\n");
        return DoubleDouble(0.0);
    }
    // Initial approximation then 3 Newton steps: b <- b + (a - exp(b)) / exp(b)
    DoubleDouble b = DoubleDouble(detail::log(a.hi));
    for (int k = 0; k < 3; ++k) {
        DoubleDouble s0 = exp(b);
        DoubleDouble s1 = subtract(a, s0);
        DoubleDouble s2 = divide(s1, s0);
        b = add(b, s2);
    }
    return b;
}

XPMATH_INLINE_FUNCTION DoubleDouble log2(DoubleDouble a) {
    return divide(log(a), DoubleDouble_log2());
}

XPMATH_INLINE_FUNCTION DoubleDouble log10(DoubleDouble a) {
    return divide(log(a), DoubleDouble_log10());
}

// log1p(a) = log(1+a), accurate for small |a|.
//
// KI-5(b). The old body was `log(add(1, a))`, which is not a log1p at all: for
// |a| << 1 the sum discards everything below the leading 1, so the result keeps
// only log10(1/|a|) fewer digits than the type has. That is exactly the loss the
// caller asked to avoid, and it is what made complex `atanh` collapse as z -> 0.
//
// The obvious repair does NOT work here, and was measured before being dropped.
//
// (i) REJECTED: Goldberg's correction (Higham, "Accuracy and Stability of
// Numerical Algorithms", 2nd ed., problem 1.5): with u = fl(1+a), log1p(a) =
// log(u)*a/(u-1). `u - 1` is EXACT whenever 0.5 <= u.hi <= 2, so a/(u-1) is
// exactly the factor by which rounding 1+a perturbed the argument. But that
// repairs the ARGUMENT and assumes log(u) is accurate for the u it is given. It
// is not, here: `log` seeds from the FP64 log of the leading limb and takes ONE
// Newton step x <- x + a*exp(-x) - 1; for u = 1 + e the seed is 0 and the step
// returns e itself, whose relative error against log(1+e) = e - e^2/2 is e/2.
// Measured: with Goldberg alone, log1p(4e-20) on DD scored 19.70 digits, not 31
// -- exactly the 2*log10(1/e) that one Newton step buys. Applying it for larger
// |a| as well, where there is no cancellation to undo, cost 583 sweep points up
// to 0.87 digits to its two extra roundings. So it is not used at all.
//
// (ii) WHAT SHIPS: a series for small |a|, which never calls log:
//
//     log1p(a) = 2*atanh(t),  t = a/(2+a),  atanh(t) = t + t^3/3 + t^5/5 + ...
//
// The atanh form rather than the plain alternating log(1+a) series because its
// terms are all positive (no cancellation) and it converges in t^2, so |a| < 1/4
// gives |t| < 1/7 and t^2 < 1/48 -- about 19 terms for DD's 32 digits. Forming
// t costs one divide, which is the whole price. Outside |a| < 1/4 the body is
// the ORIGINAL log(1+a), bit-identical to before this change.
//
// Edge cases: a == 0 gives t == 0 and returns 0 with its sign; a == -1 falls
// through to log(0) = -inf; large a falls through unchanged. Same body in all
// four backends, with the threshold fixed at 1/4 and only the convergence
// epsilon retyped.
XPMATH_INLINE_FUNCTION DoubleDouble log1p(DoubleDouble a) {
    if (detail::fabs(a.hi) < 0.25) {
        DoubleDouble t   = divide(a, add(DoubleDouble(2.0), a));
        DoubleDouble t2  = multiply(t, t);
        DoubleDouble sum = t;
        DoubleDouble trm = t;
        for (int k = 3; k < 80; k += 2) {
            trm = multiply(trm, t2);
            DoubleDouble incr = divide(trm, DoubleDouble((double)k));
            sum = add(sum, incr);
            if (detail::fabs(incr.hi) <= 1.0e-34 * detail::fabs(sum.hi)) break;
        }
        return multiply_scalar(sum, 2.0);
    }
    return log(add(DoubleDouble(1.0), a));
}

XPMATH_INLINE_FUNCTION DoubleDouble exp2(DoubleDouble a) {
    return exp(multiply(a, DoubleDouble_log2()));
}

XPMATH_INLINE_FUNCTION DoubleDouble exp10(DoubleDouble a) {
    return exp(multiply(a, DoubleDouble_log10()));
}

XPMATH_INLINE_FUNCTION DoubleDouble expm1(DoubleDouble a) {
    if (detail::fabs(a.hi) > 0.5) {
        // |exp(a)-1| > e^0.5-1 ~ 0.65: subtraction of 1 causes no significant cancellation
        return subtract(exp(a), DoubleDouble(1.0));
    }
    // Taylor series: a + a²/2! + a³/3! + ...
    // Avoids catastrophic cancellation of exp(a)-1 near a=0
    DoubleDouble sum = a, term = a;
    for (int k = 2; k <= 50; ++k) {
        term = divide_scalar(multiply(term, a), (double)k);
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < 1.0e-32 * detail::fabs(sum.hi)) break;
    }
    return sum;
}

// ============================================================
// Trig — internal combined cos+sin, then derived
// ============================================================

// sincos: compute (cos(a), sin(a)) via argument reduction + Taylor series
// x = cos(a), y = sin(a)
XPMATH_INLINE_FUNCTION void sincos(DoubleDouble a, DoubleDouble& x, DoubleDouble& y) {
    const int itrmx = 1000, nq = 5;
    const double eps = 1.0e-32;
    if (a.hi == 0.0) { x = DoubleDouble(1.0); y = DoubleDouble(0.0); return; }
    if (a.hi >= 1.0e60) {
        XPMATH_PRINTF("DDCSSNR: argument too large\n");
        x = DoubleDouble(0.0); y = DoubleDouble(0.0); return;
    }
    DoubleDouble pi2 = multiply_scalar(DoubleDouble_pi(), 2.0);
    DoubleDouble s1  = divide(a, pi2);
    DoubleDouble s2  = round_to_nearest_int(s1);
    DoubleDouble s3  = subtract(a, multiply(pi2, s2));
    if (s3.hi == 0.0) { x = DoubleDouble(1.0); y = DoubleDouble(0.0); return; }
    double scale = 1.0 / (double)(1 << nq);
    DoubleDouble r  = multiply_scalar(s3, scale);   // r = s3 / 2^nq, |r| < pi/2^nq
    // For subnormal |a| the scaling underflows r to zero, and then the relative
    // convergence test below is vacuous (0 < eps*0 is false) and the series runs
    // to itrmx. Answer it directly: sin(a) = a and cos(a) = 1 to far beyond DD
    // precision for any |a| this small.
    if (r.hi == 0.0) { x = DoubleDouble(1.0); y = s3; return; }
    DoubleDouble r2 = multiply(r, r);

    // sin(r) = r - r^3/3! + r^5/5! - ...
    // cos(r) = 1 - r^2/2! + r^4/4! - ...
    DoubleDouble sin_r = r,               cos_r = DoubleDouble(1.0);
    DoubleDouble sterm = r,               cterm = DoubleDouble(1.0);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(double)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(double)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (detail::fabs(sterm.hi) < eps * detail::fabs(sin_r.hi) &&
            detail::fabs(cterm.hi) < eps) break;
        // break, not return: returning here would leave x/y unassigned.
        if (k == itrmx) { XPMATH_PRINTF("DDCSSNR: iteration limit\n"); break; }
    }

    // Joint doubling nq times: sin(2x) = 2*sin(x)*cos(x), cos(2x) = cos^2(x) - sin^2(x).
    // Both series are carried through; the sine is never reconstructed from the
    // cosine via +/-sqrt(1 - cos^2). That reconstruction (a) is only sign-correct
    // for |s3| < pi, which round_to_nearest_int does not guarantee at half-integer
    // near-ties, and (b) amplifies the relative error of cos by cot^2(s3), which
    // diverges as s3 -> +/-pi. Matches ff/qf/tf_math.hpp. See KI-4.
    for (int j = 0; j < nq; ++j) {
        DoubleDouble new_sin = multiply_scalar(multiply(sin_r, cos_r), 2.0);
        DoubleDouble new_cos = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }

    x = cos_r; y = sin_r;
}

XPMATH_INLINE_FUNCTION DoubleDouble sin(DoubleDouble a) {
    DoubleDouble c, s; sincos(a, c, s); return s;
}
XPMATH_INLINE_FUNCTION DoubleDouble cos(DoubleDouble a) {
    DoubleDouble c, s; sincos(a, c, s); return c;
}
XPMATH_INLINE_FUNCTION DoubleDouble tan(DoubleDouble a) {
    DoubleDouble c, s; sincos(a, c, s); return divide(s, c);
}

// Angle of point (x, y) = atan2(y, x). Internal DDFUN primitive (DDANG); the
// public STL-ordered wrapper is atan2(y, x) below.
XPMATH_INLINE_FUNCTION DoubleDouble angle(DoubleDouble x, DoubleDouble y) {
    DoubleDouble pi = DoubleDouble_pi();
    if (x.hi == 0.0 && y.hi == 0.0) return DoubleDouble(0.0);
    if (x.hi == 0.0) return (y.hi > 0.0) ? multiply_scalar(pi, 0.5) : multiply_scalar(pi, -0.5);
    if (y.hi == 0.0) return (x.hi > 0.0) ? DoubleDouble(0.0) : pi;
    // Normalize
    DoubleDouble r = sqrt(add(multiply(x,x), multiply(y,y)));
    DoubleDouble nx = divide(x, r), ny = divide(y, r);
    // Initial approximation
    DoubleDouble a = DoubleDouble(detail::atan2(ny.hi, nx.hi));
    bool use_x = (detail::fabs(nx.hi) <= detail::fabs(ny.hi));
    DoubleDouble target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        DoubleDouble sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        DoubleDouble corr;
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

XPMATH_INLINE_FUNCTION DoubleDouble asin(DoubleDouble a) {
    if (detail::fabs(a.hi) > 1.0) {
        XPMATH_PRINTF("DDASIN: argument out of range\n");
        return DoubleDouble(0.0);
    }
    DoubleDouble t = sqrt(subtract(DoubleDouble(1.0), multiply(a, a)));
    return angle(t, a); // atan2(a, sqrt(1-a^2))
}
XPMATH_INLINE_FUNCTION DoubleDouble acos(DoubleDouble a) {
    if (detail::fabs(a.hi) > 1.0) {
        XPMATH_PRINTF("DDACOS: argument out of range\n");
        return DoubleDouble(0.0);
    }
    DoubleDouble t = sqrt(subtract(DoubleDouble(1.0), multiply(a, a)));
    return angle(a, t); // atan2(sqrt(1-a^2), a)
}
XPMATH_INLINE_FUNCTION DoubleDouble atan(DoubleDouble a) {
    return angle(DoubleDouble(1.0), a); // atan2(a, 1)
}
XPMATH_INLINE_FUNCTION DoubleDouble atan2(DoubleDouble y, DoubleDouble x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic — internal combined cosh+sinh, then derived
// ============================================================

// KI-6/KI-7: past this |a| the smaller exponential e^{-2|a|} is below DD's
// resolution u = 2^-106, so cosh(a) == sinh(a) == e^{|a|}/2 to the last bit and
// tanh(a) == ±1 exactly. Derived as ln(1/u)/2 = 36.7, rounded up for margin.
// Short-circuiting there is not just a guard: it is the only way to reach the
// top of the range, because e^{|a|} itself overflows FP64 above 709.78 while
// cosh/sinh do not overflow until 710.48, and tanh never does.
constexpr double kDDHyperbolicSaturate = 40.0;

// x = cosh(a), y = sinh(a)
XPMATH_INLINE_FUNCTION void sinhcosh(DoubleDouble a, DoubleDouble& x, DoubleDouble& y) {
    if (detail::fabs(a.hi) > kDDHyperbolicSaturate) {
        // e^{|a|}/2, reached by the SAME route the two-exponential form took, so
        // that this branch is bit-identical to it wherever it worked: for a > 0
        // that is exp(a) itself, for a < 0 the reciprocal 1/exp(a). Halving is
        // exact (power of two), and the dropped term is below the last bit here
        // by construction. Only when that route runs out of exponent — e^{|a|}
        // overflowing FP64 for a > 709.78, or 1/e^{a} doing so for a < -709.78,
        // in both cases while cosh is still finite up to |a| = 710.48 — is the
        // argument shifted instead — as it is when the reciprocal NaNs out
        // instead, which it does above ~1.3e300 where divide()'s Dekker
        // splitter overflows. That form costs up to ~1 digit (subtracting
        // ln2 perturbs the argument, and exp turns an absolute argument error
        // into a relative output error), which is why it is the fallback.
        DoubleDouble aa = (a.hi < 0.0) ? negate(a) : a;
        DoubleDouble e  = exp(a);
        if (a.hi < 0.0) e = divide(DoubleDouble(1.0), e);
        DoubleDouble h = (detail::isinf(e.hi) || e.hi != e.hi) ? exp(subtract(aa, DoubleDouble_log2()))
                                             : DoubleDouble(e.hi * 0.5, e.lo * 0.5);
        x = h;
        y = (a.hi < 0.0) ? negate(h) : h;
        return;
    }
    DoubleDouble s0 = exp(a);
    DoubleDouble s1 = divide(DoubleDouble(1.0), s0);
    x = multiply_scalar(add(s0, s1), 0.5);
    y = multiply_scalar(subtract(s0, s1), 0.5);
}

XPMATH_INLINE_FUNCTION DoubleDouble sinh(DoubleDouble a) {
    DoubleDouble c, s; sinhcosh(a, c, s); return s;
}
XPMATH_INLINE_FUNCTION DoubleDouble cosh(DoubleDouble a) {
    DoubleDouble c, s; sinhcosh(a, c, s); return c;
}
XPMATH_INLINE_FUNCTION DoubleDouble tanh(DoubleDouble a) {
    // tanh(x) = expm1(2x) / (expm1(2x) + 2), reflected for negative x
    // Avoids dividing two nearly-equal large numbers from sinhcosh
    if (a.hi < 0.0) return negate(tanh(negate(a)));
    // KI-7: saturate before evaluating anything. 1 - tanh(x) = 2e^{-2x} is below
    // DD's half-ulp at 1 (2^-107) for x > 37.4, so ±1 is the correctly rounded
    // answer here; and it is the only correct answer, because 2x overflows the
    // exp argument range above x = 354.9, where expm1 returns +inf and
    // inf/(inf+2) is NaN. Before this short-circuit the ±300 exp guard made
    // expm1(2x) = 0-1 = -1 and the expression collapsed to (-1)/(1) = -1 for
    // every large POSITIVE x — the wrong sign, KI-7's reported symptom.
    if (a.hi > kDDHyperbolicSaturate) return DoubleDouble(1.0);
    DoubleDouble e = expm1(multiply_scalar(a, 2.0));
    return divide(e, add(e, DoubleDouble(2.0)));
}

XPMATH_INLINE_FUNCTION DoubleDouble asinh(DoubleDouble a) {
    // Reflect: asinh(-a) = -asinh(a). For positive a, a + sqrt(a²+1) >= 1 always,
    // so log argument never causes cancellation.
    if (a.hi < 0.0) return negate(asinh(negate(a)));
    return log(add(a, sqrt(add(multiply(a, a), DoubleDouble(1.0)))));
}
XPMATH_INLINE_FUNCTION DoubleDouble acosh(DoubleDouble a) {
    if (a.hi < 1.0) { XPMATH_PRINTF("DDACOSH: argument < 1\n"); return DoubleDouble(0.0); }
    DoubleDouble t1 = subtract(multiply(a, a), DoubleDouble(1.0));
    return log(add(a, sqrt(t1)));
}
XPMATH_INLINE_FUNCTION DoubleDouble atanh(DoubleDouble a) {
    if (detail::fabs(a.hi) >= 1.0) { XPMATH_PRINTF("DDATANH: |argument| >= 1\n"); return DoubleDouble(0.0); }
    DoubleDouble t1 = add(DoubleDouble(1.0), a);
    DoubleDouble t2 = subtract(DoubleDouble(1.0), a);
    return multiply_scalar(log(divide(t1, t2)), 0.5);
}

// ============================================================
// Multi-argument operations
// ============================================================

XPMATH_INLINE_FUNCTION DoubleDouble pow(DoubleDouble a, DoubleDouble b) {
    if (a.hi <= 0.0) {
        if (a.hi == 0.0 && b.hi > 0.0) return DoubleDouble(0.0);
        XPMATH_PRINTF("DDPOW: non-positive base\n");
        return DoubleDouble(0.0);
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
// FP64 word range at |a| ~ 1.3e154 and flushes to zero at |a| ~ 1.5e-154 -- in both
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
// The scaled form is NOT used unconditionally.  It costs a divide and is a
// touch less accurate than the direct one (divide + square + sqrt + multiply
// versus square + add + sqrt), so it is gated to the range where the direct
// form actually breaks: for m inside [1.0e-150, 1.0e150] the old expression is
// evaluated exactly as before and the added cost is two compares, not a divide.
// Same reasoning as the atanh threshold in dd_complex.hpp -- fix the interval
// that is broken, do not churn the one that is not.
//
// inf/nan convention (C99 F.9.4.3): hypot(+-inf, y) is +inf for ANY y, NaN
// included, so the inf test comes first.  Otherwise a NaN operand propagates to
// NaN through the arithmetic.  Both operands are taken through abs() first, so
// the returned infinity is always +inf.
XPMATH_INLINE_FUNCTION DoubleDouble hypot(DoubleDouble a, DoubleDouble b) {
    DoubleDouble x = abs(a);
    DoubleDouble y = abs(b);
    if (detail::isinf(x.hi)) return x;
    if (detail::isinf(y.hi)) return y;
    DoubleDouble m = (x.hi < y.hi) ? y : x;
    DoubleDouble n = (x.hi < y.hi) ? x : y;
    if (m.hi == 0.0) return DoubleDouble(0.0);
    if (m.hi <= 1.0e150 && m.hi >= 1.0e-150)
        return sqrt(add(multiply(a, a), multiply(b, b)));
    DoubleDouble t = divide(n, m);
    return multiply(m, sqrt(add(DoubleDouble(1.0), multiply(t, t))));
}

XPMATH_INLINE_FUNCTION DoubleDouble ceil(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble floor(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble trunc(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDouble round(DoubleDouble a);

XPMATH_INLINE_FUNCTION DoubleDouble fmod(DoubleDouble a, DoubleDouble b) {
    DoubleDouble q = divide(a, b);
    DoubleDouble qt = trunc(q);
    return subtract(a, multiply(b, qt));
}

XPMATH_INLINE_FUNCTION DoubleDouble remainder(DoubleDouble a, DoubleDouble b) {
    DoubleDouble q = divide(a, b);
    DoubleDouble qn = round_to_nearest_int(q);
    return subtract(a, multiply(b, qn));
}

XPMATH_INLINE_FUNCTION DoubleDouble copysign(DoubleDouble a, DoubleDouble b) {
    DoubleDouble r = abs(a);
    if (b.hi < 0.0 || (b.hi == 0.0 && b.lo < 0.0)) return negate(r);
    return r;
}

XPMATH_INLINE_FUNCTION DoubleDouble fmax(DoubleDouble a, DoubleDouble b) {
    return (a > b) ? a : b;
}
XPMATH_INLINE_FUNCTION DoubleDouble fmin(DoubleDouble a, DoubleDouble b) {
    return (a < b) ? a : b;
}
XPMATH_INLINE_FUNCTION DoubleDouble fdim(DoubleDouble a, DoubleDouble b) {
    return (a > b) ? subtract(a, b) : DoubleDouble(0.0);
}
XPMATH_INLINE_FUNCTION DoubleDouble fma(DoubleDouble a, DoubleDouble b, DoubleDouble c) {
    return add(multiply(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

XPMATH_INLINE_FUNCTION DoubleDouble floor(DoubleDouble a) {
    DoubleDouble n = round_to_nearest_int(a);
    if (n > a) return subtract(n, DoubleDouble(1.0));
    return n;
}
XPMATH_INLINE_FUNCTION DoubleDouble ceil(DoubleDouble a) {
    DoubleDouble n = round_to_nearest_int(a);
    if (n < a) return add(n, DoubleDouble(1.0));
    return n;
}
XPMATH_INLINE_FUNCTION DoubleDouble trunc(DoubleDouble a) {
    return (a.hi >= 0.0) ? floor(a) : ceil(a);
}
XPMATH_INLINE_FUNCTION DoubleDouble round(DoubleDouble a) {
    return round_to_nearest_int(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

// Internal helper (not part of the DD API surface): the SUM of the asymptotic
// erfc expansion, A&S 7.1.23,
//     erfc(z) = e^{-z²}/sqrt(pi) · sum_k term_k,
//     term_0 = 1/|z|,  term_k = term_{k-1} · (-(2k-1))/(2z²),
// evaluated with optimal truncation. The series is DIVERGENT: its terms shrink
// to ~e^{-z²} and then grow again, so summing past the smallest term strictly
// loses accuracy and a relative-eps exit can never be the primary one. Both the
// eps exit and the k cap are secondary; the smallest-term test below is
// primary. Measured worst case 72 terms at |z| = 8.5.
//
// Only the SUM is returned, not erfc: the two callers need different scalings.
// erf() (B3) sees only |z| <= 8.5, where e^{z²} <= 2.4e31, and divides by it;
// erfc() (B2) runs out to the DD underflow floor |z| ~ 27.25, where e^{z²} would
// exceed the Dekker splitter's headroom (DBL_MAX/2^27+1 ~ 1.3e300, reached at
// |z| ~ 26.29) and turn divide() into NaN, so it multiplies by e^{-z²} instead.
// Keeping the scaling at the call sites lets erf()'s arithmetic stay
// bit-for-bit what B3 shipped while erfc() gets the overflow-safe form.
//
// Preconditions: az = |z| > 0 and z2 = z·z, both finite or +inf.
XPMATH_INLINE_FUNCTION DoubleDouble erfc_asymptotic_sum(DoubleDouble az, DoubleDouble z2) {
    const double eps = 1.0e-32;   // just under DD's u² = 2⁻¹⁰⁶ ~ 1.23e-32
    DoubleDouble two_z2 = multiply_scalar(z2, 2.0);
    DoubleDouble term = divide(DoubleDouble(1.0), az), sum = term;
    double prev_mag = detail::fabs(term.hi);
    for (int k = 1; k <= 100; ++k) {
        DoubleDouble next = divide(multiply_scalar(term, -(2.0*k - 1.0)), two_z2);
        double mag = detail::fabs(next.hi);
        if (mag > prev_mag) break;                 // smallest term reached -> stop
        sum = add(sum, next);
        term = next; prev_mag = mag;
        if (mag <= eps * detail::fabs(sum.hi)) break;
    }
    return sum;
}

// erf — Taylor series for |z| < 6, asymptotic expansion for 6 <= |z| <= 8.5,
// saturated to ±1 beyond that.
//
// B3: the shipped version degraded smoothly from ~30 digits at |z| = 5 to 3.17
// digits at |z| = 8.5, pulling the uniform(-10, 10) mean to 24.64 against a
// 25.91 gate. Two defects, both reachable only through the Taylor path:
//   (a) The Taylor series needs k ≈ z² + 50 terms to reach DD resolution
//       (measured against an mpmath reference: 106 terms at |z| = 5, 130 at 6,
//       197 at 8.5) but the loop capped at k = 100. From |z| ≈ 4.9 upward it
//       therefore returned a *truncated* sum whose error grows smoothly with
//       |z| — which is precisely the observed ramp, with no cliff.
//   (b) The asymptotic erfc series below is DIVERGENT, so its relative-eps
//       exit could never fire; it would have run to its iteration cap and
//       summed far past the smallest term. It never actually did, because the
//       branch was unreachable: its `|z| < 9.0` guard cannot be false once the
//       `|z| > 8.5` saturation above has returned. (The B3 stub attributed the
//       digit loss to this branch; see DEVIATION 1 in the commit body.)
// Fix: raise the Taylor cap, switch over to the asymptotic branch at |z| = 6
// so it is live, and truncate that divergent series at its smallest term.
// Series identities: A&S 7.1.6 (Taylor) and A&S 7.1.23 (asymptotic erfc);
// same pair as the FF sibling fix (B5, ff_math.hpp:erf).
//
// Term recurrence. Each term is grown from its predecessor by the running
// ratio rather than from separately accumulated numerator and denominator (the
// DDFUN-port form). Unlike FF, where the old form overflowed FP32 outright and
// erf returned NaN, FP64 has the headroom — at this switchover the old form's
// intermediates peak at 3.3e238 (numerator) and 1.7e255 ((2k+1)!!), both inside
// DBL_MAX, and it scores digit-for-digit the same. The recurrence is adopted
// because it DECOUPLES the iteration cap from overflow: with separate
// accumulators (2k+1)!! reaches DBL_MAX near k ≈ 150, so the cap and the
// switchover would be pinned within ~10% of each other (at kTaylorMax = 7 the
// old form returns NaN). It is also cheaper — one DD/double divide per term
// instead of a full DD/DD divide. This is the DD-vs-B5 boundary: B5's overflow
// safety was load-bearing at FP32; here it buys cap independence, not
// correctness.
XPMATH_INLINE_FUNCTION DoubleDouble erf(DoubleDouble z) {
    // DD relative resolution is u² = 2⁻¹⁰⁶ ≈ 1.23e-32; a finer eps could not
    // fire. This is the convergent (Taylor) branch's primary exit; the
    // divergent asymptotic branch's exits live in erfc_asymptotic_sum().
    const double eps = 1.0e-32;
    if (z.hi == 0.0) return DoubleDouble(0.0);
    // erfc(8.5) < 2⁻¹⁰⁴, i.e. below DD's last bit relative to 1, so erf
    // saturates exactly. threshold: sqrt(104 * ln2) ≈ 8.48
    const double large = 8.5;
    if (z.hi >  large) return DoubleDouble( 1.0);
    if (z.hi < -large) return DoubleDouble(-1.0);

    DoubleDouble z2 = multiply(z, z);
    int sign = (z.hi >= 0.0) ? 1 : -1;
    DoubleDouble az = abs(z);

    // Switchover derivation. The asymptotic expansion's optimal-truncation
    // floor is its smallest term, ~ e^{-z²}; carried through erf = 1 - erfc
    // that is an absolute error ≈ e^{-2z²}/(z·sqrt(pi)), which first drops
    // below u² at |z| ≈ 5.97. Below 6 the Taylor series still converges well
    // inside the cap (k ≤ 130). Measured over (0, 8.5] on a 0.005 grid the
    // minimum is 29.49 digits at z = 0.435 — a generic-roundoff point, not a
    // branch artifact — and there is no dip at the seam (erf(5.95) = 29.70,
    // erf(6.05) = 31.00, the report clamp).
    const double kTaylorMax = 6.0;

    if (detail::fabs(z.hi) < kTaylorMax) {
        // Taylor (A&S 7.1.6): erf(z) = (2/sqrt(pi)) e^{-z²} sum_k term_k,
        //   term_0 = |z|,  term_k = term_{k-1} · (2z²)/(2k+1).
        // All terms positive — no cancellation; the sum is bounded by
        // (sqrt(pi)/2)e^{z²} (≈ 4.3e15 at |z| = 6). Cap 200 against a measured
        // worst case of 130 terms as |z| → 6⁻; the recurrence makes the excess
        // free, since every intermediate stays O(sum).
        DoubleDouble two_z2 = multiply_scalar(z2, 2.0);
        DoubleDouble term = az, sum = az;
        for (int k = 1; k <= 200; ++k) {
            term = divide_scalar(multiply(term, two_z2), 2.0*k + 1.0);
            DoubleDouble sumnew = add(sum, term);
            if (detail::fabs(term.hi) <= eps * detail::fabs(sumnew.hi)) { sum = sumnew; break; }
            sum = sumnew;
        }
        DoubleDouble result = divide(multiply_scalar(sum, 2.0),
                                     multiply(sqrt(DoubleDouble_pi()), exp(z2)));
        return (sign > 0) ? result : negate(result);
    } else {
        // Asymptotic (A&S 7.1.23), optimal truncation — see
        // erfc_asymptotic_sum() above, which B2 factored out of this branch so
        // erfc() can invoke the identical series directly. The scaling stays
        // here: |z| <= 8.5 on this path, so e^{z²} <= 2.4e31 and the divide is
        // safe (erfc()'s multiply-by-e^{-z²} form is the one that has to reach
        // |z| ~ 26). Then erf = 1 - erfc; erfc is ≈ 2.2e-17 at the |z| = 6 seam
        // and smaller above, so that subtraction is benign for erf (it is NOT
        // for erfc itself — see erfc() below).
        DoubleDouble sum = erfc_asymptotic_sum(az, z2);
        DoubleDouble erfc_val = divide(sum, multiply(sqrt(DoubleDouble_pi()), exp(z2)));
        DoubleDouble erf_val  = subtract(DoubleDouble(1.0), erfc_val);
        return (sign > 0) ? erf_val : negate(erf_val);
    }
}

// erfc — direct asymptotic expansion for z >= 6.5, 1 - erf(z) below that.
//
// B2: `erfc(z) = subtract(DoubleDouble(1.0), erf(z))` for ALL z is catastrophic
// cancellation as erf(z) -> 1. erf is accurate to u² RELATIVE to 1, so the
// difference carries an absolute error ~u² and erfc's relative error is
// u²/erfc(z) — a loss of exactly log10(1/erfc(z)) digits, which is the smooth
// ramp measured off the shipped code: 28.3 digits at z = 2, 26.6 at 3, 23.0 at
// 4, 18.4 at 5, and 0 above z = 8.5, where erf saturates to exactly 1 and erfc
// returns exactly 0. Mean over uniform(-10, 10) was 24.87 against a 25.91 gate.
// (B3 had already lifted that from 19.50 by making erf's asymptotic branch
// live: over 6 <= z <= 8.5 erf returns 1 - erfc_val with erfc_val small enough
// to land in the lo word, so the outer subtract recovered ~16 digits. A lo-word
// round trip is a 53-bit channel, so ~16 digits is all it can ever recover —
// hence the flat 16-digit shelf there, and hence B2.)
//
// Fix: for z >= kDirectMin, evaluate erfc directly from the SAME asymptotic
// series erf() uses (erfc_asymptotic_sum, A&S 7.1.23) and never form 1 - erf.
// Negative z needs no such path: erfc(-|z|) = 2 - erfc(|z|) is ~2, so
// subtract(1, erf) is benign there and already scores 31 digits.
//
// Threshold derivation. The cut is placed where the direct series stops being
// worse than the fallback it replaces — at EVERY measured point, not on
// average, because the uniform(-10, 10) mean is flat to ±0.03 digit across the
// whole [5.6, 6.6] candidate window (27.99 down to 27.96) and so decides
// nothing. What the fallback delivers above z = 6.0 is the lo-word shelf B3
// created: a 53-bit channel, measured mean 16.53 digits over [6.3, 8.5] with
// roundoff scatter reaching 19.20. The direct series rises ~5.3 digits per unit
// z through that shelf and clears its full scatter envelope at z ~ 6.44
// (highest regressing point 6.4230 on a 0.0005 grid). kDirectMin = 6.5 sits
// just above, and over 6001 grid points on [6, 9] at 0.0005 spacing NO point
// scores worse than the shipped 1 - erf code did.
//
// Rejected alternative — kDirectMin = 5.75, the balanced-error (minimax) cut.
// It lifts the global trough of the composite curve from 13.55 to 14.50 digits
// (the trough is the last fallback point before the seam, where 1 - erf's
// cancellation is deepest), and it is the mean-optimal point (27.99 vs 27.97).
// But it regresses 34 of 721 grid points over (4, 10] by up to 1.86 digits, all
// inside the 16-18 digit scatter band. Trading measured regressions for a
// 0.95-digit gain on a trough that is 12 digits under gate either way is not
// worth it; the 13.55-digit trough at z = 5.915 is pre-existing 1 - erf
// behaviour and is left exactly as it was.
//
// KNOWN LIMITATION, not closed here. The 1 - erf cancellation is a ramp, not a
// cliff, so it also costs digits well below any usable threshold for THIS
// series — on a 0.02 grid, erfc scores under the 25.91 test gate at scattered
// points from z = 2.90 and at every point from z = 3.68 to 7.70. A direct
// asymptotic path cannot help there: its optimal-truncation floor is worth only
// ~11 digits at z = 5, i.e. worse than the subtract it would replace. Closing
// that band needs a different algorithm (a Lentz continued fraction, A&S
// 7.1.14, or a triple-double erf). The dd_accuracy_test row gates on the MEAN
// and passes at 27.97 vs 25.91; the pointwise band is a separate, open concern.
XPMATH_INLINE_FUNCTION DoubleDouble erfc(DoubleDouble z) {
    // See derivation above. Sits above erf()'s own kTaylorMax = 6.0 seam, so
    // [6.0, 6.5) of the lo-word shelf is still served by the fallback.
    const double kDirectMin = 6.5;
    // Above this, erfc(z) < 2⁻¹⁰⁷⁴ (the smallest IEEE double subnormal) and +0
    // is the only representable answer, so honest underflow beats evaluating
    // the series. Derivation: erfc(z) ~ e^{-z²}/(z·sqrt(pi)) crosses 4.94e-324
    // at z ~ 27.28 (measured: erfc(27.2) = 1.0e-323, erfc(27.3) = 4.2e-326).
    // This also keeps z² away from the range where 2z² would overflow the
    // Dekker splitter inside divide() (b.hi · 2²⁷+1 > DBL_MAX above 1.3e300)
    // and turn the series into NaN — reachable from the corpus, which feeds
    // erfc DBL_MAX.
    const double kUnderflowMax = 27.25;
    if (z.hi >= kDirectMin) {
        if (!(z.hi <= kUnderflowMax)) return DoubleDouble(0.0);  // also catches +inf
        DoubleDouble z2 = multiply(z, z);
        DoubleDouble sum = erfc_asymptotic_sum(z, z2);
        // e^{-z²}, not 1/e^{z²}: dividing by e^{z²} would (a) hit dd exp()'s
        // hard ±300 argument guard at z = 17.32 and return 0, and (b) overflow
        // the Dekker splitter in divide() at z = 26.29. The multiply form has
        // neither failure and reaches the underflow floor above.
        DoubleDouble emz2;
        if (z2.hi < 300.0) {
            emz2 = exp(negate(z2));
        } else {
            // Same ±300 guard: quarter the argument (max |z²/4| = 185.7 at
            // kUnderflowMax) and square twice. Costs ~2 bits of the relative
            // error of exp; measured 30.45 digits at z = 20, 29.38 at z = 26.
            emz2 = exp(divide_scalar(negate(z2), 4.0));
            emz2 = multiply(emz2, emz2);
            emz2 = multiply(emz2, emz2);
        }
        return divide(multiply(sum, emz2), sqrt(DoubleDouble_pi()));
    }
    return subtract(DoubleDouble(1.0), erf(z));
}

// gamma — Lanczos approximation at DD precision
//
// B1: promoted from Lanczos g=7 / n=9 with `double` coefficients (~14.6 digits)
// to g=14 / N=17 with DD-precision coefficients (~28.3 digits). The g=7 form
// could not reach DD precision at ANY coefficient precision: its intrinsic
// truncation-order ceiling is ~13 digits at large a (verified by recomputing the
// g=7 set to 25 exact digits and re-measuring), so the order had to rise too.
//
// Choice of g: this is the PARTIAL-FRACTION form, c_0 + sum_k c_k/(x+k). Raising
// g shrinks the truncation error but grows the coefficients as max|c_k| ~
// 10^(g/2) against an O(1) sum, so cancellation eats the gain — the two effects
// cross at an interior optimum near g=14. (Boost's lanczos24m113 uses g=20.32,
// but for the RATIONAL evaluation form, which is immune to that cancellation;
// its g does not transfer here. At g=20.32/N=24 this form measures 26.17 digits
// versus 28.30 at g=14/N=17 — both clear the 25.91 gate, g=14 by 8x the margin
// and with 7 fewer DD divisions per call.)
//
// Coefficients are derived for this form — not transcribed from a published
// table — by scripts/gen_dd_lanczos_coeffs.py: an exact N-node Cauchy solve at
// 150-digit precision (mpmath), then split into DD pairs via hi=(double)c,
// lo=(double)(c-hi). Regenerate with:
//     python3.12 scripts/gen_dd_lanczos_coeffs.py --g 14 --n 17
// Reference for the method: C. Lanczos, "A Precision Approximation of the Gamma
// Function", J. SIAM Numer. Anal. B 1 (1964) 86-96; P. Godfrey (2001), "A note
// on the computation of the convergent Lanczos complex Gamma approximation".
XPMATH_INLINE_FUNCTION DoubleDouble tgamma(DoubleDouble a) {
    if (a.hi < 0.5) {
        // Reflection. DoubleDouble_pi() is already a full two-word DD constant.
        DoubleDouble pi = DoubleDouble_pi();
        DoubleDouble sin_pi_a = sin(multiply(pi, a));
        return divide(pi, multiply(sin_pi_a, tgamma(subtract(DoubleDouble(1.0), a))));
    }
    // Lanczos g=14, N=17 partial-fraction terms; g+1/2 = 14.5 is exact in binary.
    // Stored as from_bits pairs (no static — not device-safe).
    const DoubleDouble c0  = DoubleDouble::from_bits(0x3ff0000000000000ULL, 0xbae5ccd249ecc19bULL); // 0.99999999999999999999999943648104
    const DoubleDouble c1  = DoubleDouble::from_bits(0x4130508002f7b2f9ULL, 0xbdde6b1d3115d868ULL); // 1069184.0115920883893762432104389
    const DoubleDouble c2  = DoubleDouble::from_bits(0xc1520c269bdd76d1ULL, 0x3de92b2f192ec69eULL); // -4731034.4353920973868757761804379
    const DoubleDouble c3  = DoubleDouble::from_bits(0x4160d8066039eda2ULL, 0xbdf87be0f302eef3ULL); // 8831027.007071319611220661269011
    const DoubleDouble c4  = DoubleDouble::from_bits(0xc16146a8bdf4bdfaULL, 0xbe0be918cc144d3eULL); // -9057605.9361257449464911660473997
    const DoubleDouble c5  = DoubleDouble::from_bits(0x4155446ee2ac5166ULL, 0x3dff11da6c1fbcf7ULL); // 5575099.5417674542269340470477891
    const DoubleDouble c6  = DoubleDouble::from_bits(0xc140204023677251ULL, 0x3def145fa217e389ULL); // -2113664.2765944378983143233455817
    const DoubleDouble c7  = DoubleDouble::from_bits(0x411dcda58708ca7dULL, 0xbdbc5517b2320728ULL); // 488297.38186947236287561024853562
    const DoubleDouble c8  = DoubleDouble::from_bits(0xc0f010ab7f5d5dd2ULL, 0x3d9caf781e0d5b98ULL); // -65802.718594900587803425288271468
    const DoubleDouble c9  = DoubleDouble::from_bits(0x40b2925c21e49b29ULL, 0x3d4703761d822039ULL); // 4754.3598921660242738590870170964
    const DoubleDouble c10 = DoubleDouble::from_bits(0xc063db3ec68d4616ULL, 0xbcf6895939019f5eULL); // -158.85141303627523757502741476457
    const DoubleDouble c11 = DoubleDouble::from_bits(0x3ffe0fc55cf4679aULL, 0x3c9c4101b51c3344ULL); // 1.8788503294985959705049465832809
    const DoubleDouble c12 = DoubleDouble::from_bits(0xbf72ccd49a96fda3ULL, 0x3bf8020a237ad597ULL); // -0.0045898728216679255070647986148431
    const DoubleDouble c13 = DoubleDouble::from_bits(0x3ea3e6fd3f82125aULL, 0x3b2ef6f032dc7da6ULL); // 0.00000059313481327921474287496854411048
    const DoubleDouble c14 = DoubleDouble::from_bits(0x3d8651737a83433fULL, 0x3a2e2e9b5d81e4f9ULL); // 2.5372819792517959806144197536557e-12
    const DoubleDouble c15 = DoubleDouble::from_bits(0xbd722f28a915bb2fULL, 0x3a09bbbac7a1a093ULL); // -1.0336529032774224293847426604539e-12
    const DoubleDouble c16 = DoubleDouble::from_bits(0x3d4253da47c4e9eaULL, 0xb9d23f10f048fc60ULL); // 1.3022507121571147552407939906776e-13
    const DoubleDouble sqrt_2pi = DoubleDouble::from_bits(0x40040d931ff62706ULL, 0xbcaa6a0d6f814637ULL); // sqrt(2*pi)

    DoubleDouble x = subtract(a, DoubleDouble(1.0));
    DoubleDouble t = add(x, DoubleDouble(14.5));  // x + g + 1/2
    DoubleDouble s = c0;
    s = add(s, divide(c1, add(x, DoubleDouble(1.0))));
    s = add(s, divide(c2, add(x, DoubleDouble(2.0))));
    s = add(s, divide(c3, add(x, DoubleDouble(3.0))));
    s = add(s, divide(c4, add(x, DoubleDouble(4.0))));
    s = add(s, divide(c5, add(x, DoubleDouble(5.0))));
    s = add(s, divide(c6, add(x, DoubleDouble(6.0))));
    s = add(s, divide(c7, add(x, DoubleDouble(7.0))));
    s = add(s, divide(c8, add(x, DoubleDouble(8.0))));
    s = add(s, divide(c9, add(x, DoubleDouble(9.0))));
    s = add(s, divide(c10, add(x, DoubleDouble(10.0))));
    s = add(s, divide(c11, add(x, DoubleDouble(11.0))));
    s = add(s, divide(c12, add(x, DoubleDouble(12.0))));
    s = add(s, divide(c13, add(x, DoubleDouble(13.0))));
    s = add(s, divide(c14, add(x, DoubleDouble(14.0))));
    s = add(s, divide(c15, add(x, DoubleDouble(15.0))));
    s = add(s, divide(c16, add(x, DoubleDouble(16.0))));

    return multiply(multiply(sqrt_2pi, s),
                 multiply(pow(t, add(x, DoubleDouble(0.5))), exp(negate(t))));
}

// Bessel J0 via series
XPMATH_INLINE_FUNCTION DoubleDouble bessel_j0(DoubleDouble x) {
    const double eps = 1.0e-32;
    DoubleDouble x2 = multiply_scalar(multiply(x, x), -0.25);
    DoubleDouble term = DoubleDouble(1.0), sum = DoubleDouble(1.0);
    for (int k = 1; k <= 100; ++k) {
        term = divide_scalar(multiply(term, x2), (double)(k*k));
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < eps * detail::fabs(sum.hi)) break;
    }
    return sum;
}

XPMATH_INLINE_FUNCTION DoubleDouble bessel_j1(DoubleDouble x) {
    const double eps = 1.0e-32;
    DoubleDouble x2 = multiply_scalar(multiply(x, x), -0.25);
    DoubleDouble term = multiply_scalar(x, 0.5), sum = term;
    for (int k = 1; k <= 100; ++k) {
        term = divide_scalar(multiply(term, x2), (double)(k * (k+1)));
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < eps * detail::fabs(sum.hi)) break;
    }
    return sum;
}

XPMATH_INLINE_FUNCTION DoubleDouble bessel_jn(int n, DoubleDouble x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    // Downward recurrence
    DoubleDouble j0 = bessel_j0(x), j1 = bessel_j1(x);
    DoubleDouble jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        DoubleDouble jp1 = subtract(multiply_scalar(divide(j_cur, x), 2.0*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

XPMATH_INLINE_FUNCTION DoubleDouble bessel_y0(DoubleDouble x) {
    // Y0(x) = (2/pi)*(J0(x)*log(x/2) + sum...)  — simplified
    DoubleDouble two_over_pi = divide_scalar(DoubleDouble(2.0), DoubleDouble_pi().hi);
    DoubleDouble j0 = bessel_j0(x);
    return multiply(two_over_pi, multiply(j0, log(multiply_scalar(x, 0.5))));
}
XPMATH_INLINE_FUNCTION DoubleDouble bessel_y1(DoubleDouble x) {
    DoubleDouble two_over_pi = divide_scalar(DoubleDouble(2.0), DoubleDouble_pi().hi);
    DoubleDouble j1 = bessel_j1(x);
    return multiply(two_over_pi, multiply(j1, log(multiply_scalar(x, 0.5))));
}
XPMATH_INLINE_FUNCTION DoubleDouble bessel_yn(int n, DoubleDouble x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    DoubleDouble y0 = bessel_y0(x), y1 = bessel_y1(x);
    DoubleDouble ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        DoubleDouble yp1 = subtract(multiply_scalar(divide(y_cur, x), 2.0*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

// Zeta function — Euler-Maclaurin for s > 1
XPMATH_INLINE_FUNCTION DoubleDouble zeta(DoubleDouble s) {
    if (s.hi <= 1.0) { XPMATH_PRINTF("DDZETA: s <= 1\n"); return DoubleDouble(0.0); }
    const int N = 50;
    DoubleDouble sum = DoubleDouble(0.0);
    for (int k = 1; k <= N; ++k)
        sum = add(sum, exp(multiply(negate(s), log(DoubleDouble((double)k)))));
    // tail correction integral: N^{1-s}/(s-1)
    DoubleDouble tail = divide(exp(multiply(subtract(DoubleDouble(1.0), s), log(DoubleDouble((double)N)))),
                         subtract(s, DoubleDouble(1.0)));
    return add(sum, tail);
}

// Exponential integral Ei(x) via series (x > 0)
XPMATH_INLINE_FUNCTION DoubleDouble expint(DoubleDouble x) {
    DoubleDouble eg = DoubleDouble_euler_gamma();
    DoubleDouble sum = add(eg, log(abs(x)));
    DoubleDouble term = x;
    for (int k = 1; k <= 100; ++k) {
        sum = add(sum, divide_scalar(term, (double)(k * k)));
        term = multiply(term, x);
        if (detail::fabs(term.hi) * 1e-32 < detail::fabs(sum.hi)) break;
    }
    return sum;
}

// Incomplete gamma P(a,x) via series
XPMATH_INLINE_FUNCTION DoubleDouble incgamma(DoubleDouble a, DoubleDouble x) {
    const double eps = 1.0e-32;
    DoubleDouble term = divide(exp(negate(x)), a);
    DoubleDouble sum  = term;
    for (int k = 1; k <= 100; ++k) {
        term = multiply(term, divide(x, add(a, DoubleDouble((double)k))));
        sum  = add(sum, term);
        if (detail::fabs(term.hi) < eps * detail::fabs(sum.hi)) break;
    }
    return multiply(sum, exp(multiply(a, log(x))));
}

}  // namespace xp
