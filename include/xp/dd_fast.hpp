// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// dd_fast.hpp -- the FAST tier of the DD backend (P3.0b).
//
// DoubleDoubleFast and DoubleDoubleComplexFast are shadow types of xp::DoubleDouble and xp::DoubleDoubleComplex: same limbs and
// alignment, but a distinct type, so tiers cannot be mixed by accident. Every
// public function and operator below FORWARDS to the strict implementation, so
// until a P3 task overrides one, fast is bit-identical to strict. Strict code is
// not edited by this header. See xp/tier.hpp for the override and chaining
// patterns and for what a P3 task must do to change one function.
//
// One forwarding function per line, so a P3 task changes one line. Conversions
// are explicit only: xp::to_strict / xp::to_fast.
#pragma once

#include <xp/tier.hpp>
#include <xp/dd_complex.hpp>

namespace xp {

struct DoubleDoubleFast;
struct DoubleDoubleComplexFast;

// Tier boundary conversions (defined below).
XPMATH_INLINE_FUNCTION DoubleDouble to_strict(DoubleDoubleFast a);
XPMATH_INLINE_FUNCTION DoubleDoubleFast to_fast(DoubleDouble a);
XPMATH_INLINE_FUNCTION DoubleDoubleComplex to_strict(DoubleDoubleComplexFast z);
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast to_fast(DoubleDoubleComplex z);

// Forwarders the struct bodies below call.
XPMATH_INLINE_FUNCTION DoubleDoubleFast add(DoubleDoubleFast a, DoubleDoubleFast b);
XPMATH_INLINE_FUNCTION DoubleDoubleFast subtract(DoubleDoubleFast a, DoubleDoubleFast b);
XPMATH_INLINE_FUNCTION DoubleDoubleFast multiply(DoubleDoubleFast a, DoubleDoubleFast b);
XPMATH_INLINE_FUNCTION DoubleDoubleFast divide(DoubleDoubleFast a, DoubleDoubleFast b);
XPMATH_INLINE_FUNCTION DoubleDoubleFast multiply_scalar(DoubleDoubleFast a, double b);
XPMATH_INLINE_FUNCTION DoubleDoubleFast divide_scalar(DoubleDoubleFast a, double b);
XPMATH_INLINE_FUNCTION DoubleDoubleFast negate(DoubleDoubleFast a);

struct alignas(16) DoubleDoubleFast {
    double hi;
    double lo;
    XPMATH_INLINE_FUNCTION DoubleDoubleFast() : hi(0.0), lo(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleFast(double h) : hi(h), lo(0.0) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleFast(double h, double l) : hi(h), lo(l) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleFast(const DoubleDoubleFast& o) : hi(o.hi), lo(o.lo) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator=(const DoubleDoubleFast& o) { hi=o.hi; lo=o.lo; return *this; }
    static XPMATH_INLINE_FUNCTION DoubleDoubleFast from_bits(uint64_t hi_bits, uint64_t lo_bits) {
        double h, l;
#if !defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
        std::memcpy(&h, &hi_bits, sizeof(double));
        std::memcpy(&l, &lo_bits, sizeof(double));
#else
        h = __longlong_as_double(static_cast<long long>(hi_bits));
        l = __longlong_as_double(static_cast<long long>(lo_bits));
#endif
        return DoubleDoubleFast(h, l);
    }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator+(DoubleDoubleFast b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator-(DoubleDoubleFast b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator*(DoubleDoubleFast b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator/(DoubleDoubleFast b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator*(double b)  const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator/(double b)  const { return divide_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator+(double b)  const { return add(*this, DoubleDoubleFast(b)); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast operator-(double b)  const { return subtract(*this, DoubleDoubleFast(b)); }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator+=(DoubleDoubleFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator-=(DoubleDoubleFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator*=(DoubleDoubleFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator/=(DoubleDoubleFast b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator+=(double b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator-=(double b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator*=(double b) { *this = multiply_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast& operator/=(double b) { *this = divide_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION bool operator==(DoubleDoubleFast b) const { return hi==b.hi && lo==b.lo; }
    XPMATH_INLINE_FUNCTION bool operator!=(DoubleDoubleFast b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(DoubleDoubleFast b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    XPMATH_INLINE_FUNCTION bool operator>(DoubleDoubleFast b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    XPMATH_INLINE_FUNCTION bool operator<=(DoubleDoubleFast b) const { return !(b < *this); }
    XPMATH_INLINE_FUNCTION bool operator>=(DoubleDoubleFast b) const { return !(*this < b); }

    // Tier boundary: explicit only, so mixing tiers is a compile error.
    explicit XPMATH_INLINE_FUNCTION DoubleDoubleFast(DoubleDouble s) : hi(s.hi), lo(s.lo) {}
    explicit XPMATH_INLINE_FUNCTION operator DoubleDouble() const { return DoubleDouble(hi, lo); }
};

struct DoubleDoubleComplexFast {
    DoubleDoubleFast re;
    DoubleDoubleFast im;

    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast() : re(), im() {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast(double r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast(DoubleDoubleFast r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast(double r, double i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast(DoubleDoubleFast r, DoubleDoubleFast i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast(const DoubleDoubleComplexFast& o) : re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast& operator=(const DoubleDoubleComplexFast& o) { re = o.re; im = o.im; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast& operator=(DoubleDoubleFast r) { re = r; im = DoubleDoubleFast(); return *this; }

    // Tier boundary: explicit only.
    explicit XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast(DoubleDoubleComplex z) : re(to_fast(z.re)), im(to_fast(z.im)) {}
    explicit XPMATH_INLINE_FUNCTION operator DoubleDoubleComplex() const { return DoubleDoubleComplex(to_strict(re), to_strict(im)); }

    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator+(DoubleDoubleComplexFast b) const { return DoubleDoubleComplexFast(add(re, b.re), add(im, b.im)); }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator-(DoubleDoubleComplexFast b) const { return DoubleDoubleComplexFast(subtract(re, b.re), subtract(im, b.im)); }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator*(DoubleDoubleComplexFast b) const { return to_fast(to_strict(*this) * to_strict(b)); }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator/(DoubleDoubleComplexFast b) const { return to_fast(to_strict(*this) / to_strict(b)); }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator-() const { return DoubleDoubleComplexFast(negate(re), negate(im)); }

    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast& operator+=(DoubleDoubleComplexFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast& operator-=(DoubleDoubleComplexFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast& operator*=(DoubleDoubleComplexFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast& operator/=(DoubleDoubleComplexFast b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(DoubleDoubleComplexFast b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(DoubleDoubleComplexFast b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION DoubleDoubleFast real() const { return re; }
    XPMATH_INLINE_FUNCTION DoubleDoubleFast imag() const { return im; }
};

XPMATH_INLINE_FUNCTION DoubleDouble to_strict(DoubleDoubleFast a) { return DoubleDouble(a); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast to_fast(DoubleDouble a) { return DoubleDoubleFast(a); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplex to_strict(DoubleDoubleComplexFast z) { return DoubleDoubleComplex(z); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast to_fast(DoubleDoubleComplex z) { return DoubleDoubleComplexFast(z); }

// ---- forwarders: one line per public strict function / operator ------------
XPMATH_INLINE_FUNCTION DoubleDoubleFast add(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast subtract(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(subtract(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast multiply(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(multiply(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast divide(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(divide(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast multiply_scalar(DoubleDoubleFast a, double b) { return to_fast(multiply_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast divide_scalar(DoubleDoubleFast a, double b) { return to_fast(divide_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast negate(DoubleDoubleFast a) { return to_fast(negate(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast abs(DoubleDoubleFast a) { return to_fast(abs(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast sqrt(DoubleDoubleFast a) { return to_fast(sqrt(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast round_to_nearest_int(DoubleDoubleFast a) { return to_fast(round_to_nearest_int(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast pow_int(DoubleDoubleFast a, int n) { return to_fast(pow_int(to_strict(a), n)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast exp(DoubleDoubleFast a) { return to_fast(exp(to_strict(a))); }
namespace detail {
XPMATH_INLINE_FUNCTION DoubleDouble dd_log_fast_impl(DoubleDouble a) {
    if (detail::isinf(a.hi)) return a;
    if (a.hi <= 0.0) {
        XPMATH_PRINTF("DDLOG: non-positive argument\n");
        return DoubleDouble(0.0);
    }
    DoubleDouble b = DoubleDouble(detail::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        if (b.hi < -671.7 && -b.hi <= 709.78271289338397) {
            DoubleDouble s0 = exp(negate(b));
            b = subtract(add(b, multiply(a, s0)), DoubleDouble(1.0));
        } else {
            DoubleDouble s0 = exp(b);
            DoubleDouble s1 = subtract(a, s0);
            DoubleDouble s2 = divide(s1, s0);
            b = add(b, s2);
        }
    }
    return b;
}
} // namespace detail
XPMATH_INLINE_FUNCTION DoubleDoubleFast log(DoubleDoubleFast a) { return to_fast(detail::dd_log_fast_impl(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast pow(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION void sinhcosh(DoubleDoubleFast a, DoubleDoubleFast& x, DoubleDoubleFast& y) { DoubleDouble t_x; DoubleDouble t_y; sinhcosh(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
XPMATH_INLINE_FUNCTION void sincos(DoubleDoubleFast a, DoubleDoubleFast& x, DoubleDoubleFast& y) { DoubleDouble t_x; DoubleDouble t_y; sincos(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast angle(DoubleDoubleFast x, DoubleDoubleFast y) { return to_fast(angle(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast operator+(double a, DoubleDoubleFast b) { return to_fast(a + to_strict(b)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast operator-(double a, DoubleDoubleFast b) { return to_fast(a - to_strict(b)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast operator*(double a, DoubleDoubleFast b) { return to_fast(a * to_strict(b)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast operator/(double a, DoubleDoubleFast b) { return to_fast(a / to_strict(b)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast DoubleDoubleFast_pi() { return to_fast(DoubleDouble_pi()); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast DoubleDoubleFast_e() { return to_fast(DoubleDouble_e()); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast DoubleDoubleFast_log2() { return to_fast(DoubleDouble_log2()); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast DoubleDoubleFast_log10() { return to_fast(DoubleDouble_log10()); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast DoubleDoubleFast_sqrt2() { return to_fast(DoubleDouble_sqrt2()); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast DoubleDoubleFast_euler_gamma() { return to_fast(DoubleDouble_euler_gamma()); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast log2(DoubleDoubleFast a) { return to_fast(log2(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast log10(DoubleDoubleFast a) { return to_fast(log10(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast log1p(DoubleDoubleFast a) { return to_fast(log1p(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast exp2(DoubleDoubleFast a) { return to_fast(exp2(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast exp10(DoubleDoubleFast a) { return to_fast(exp10(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast expm1(DoubleDoubleFast a) { return to_fast(expm1(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast sin(DoubleDoubleFast a) { return to_fast(sin(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast cos(DoubleDoubleFast a) { return to_fast(cos(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast tan(DoubleDoubleFast a) { return to_fast(tan(to_strict(a))); }
XPMATH_INLINE_FUNCTION int dd_cmp_one(DoubleDoubleFast a) { return dd_cmp_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION int dd_cmp_abs_one(DoubleDoubleFast a) { return dd_cmp_abs_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast asin(DoubleDoubleFast a) { return to_fast(asin(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast acos(DoubleDoubleFast a) { return to_fast(acos(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast atan(DoubleDoubleFast a) { return to_fast(atan(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast atan2(DoubleDoubleFast y, DoubleDoubleFast x) { return to_fast(atan2(to_strict(y), to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast sinh(DoubleDoubleFast a) { return to_fast(sinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast cosh(DoubleDoubleFast a) { return to_fast(cosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast tanh(DoubleDoubleFast a) { return to_fast(tanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast asinh(DoubleDoubleFast a) { return to_fast(asinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast acosh(DoubleDoubleFast a) { return to_fast(acosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast atanh(DoubleDoubleFast a) { return to_fast(atanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast hypot(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(hypot(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast ceil(DoubleDoubleFast a) { return to_fast(ceil(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast floor(DoubleDoubleFast a) { return to_fast(floor(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast trunc(DoubleDoubleFast a) { return to_fast(trunc(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast round(DoubleDoubleFast a) { return to_fast(round(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast fmod(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(fmod(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast remainder(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(remainder(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast copysign(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(copysign(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast fmax(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(fmax(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast fmin(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(fmin(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast fdim(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(fdim(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast fma(DoubleDoubleFast a, DoubleDoubleFast b, DoubleDoubleFast c) { return to_fast(fma(to_strict(a), to_strict(b), to_strict(c))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast erfc_asymptotic_sum(DoubleDoubleFast az, DoubleDoubleFast z2) { return to_fast(erfc_asymptotic_sum(to_strict(az), to_strict(z2))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast erf(DoubleDoubleFast z) { return to_fast(erf(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast erfc(DoubleDoubleFast z) { return to_fast(erfc(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast dd_tgamma_lanczos(DoubleDoubleFast a) { return to_fast(dd_tgamma_lanczos(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast tgamma(DoubleDoubleFast a) { return to_fast(tgamma(to_strict(a))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast bessel_j0(DoubleDoubleFast x) { return to_fast(bessel_j0(to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast bessel_j1(DoubleDoubleFast x) { return to_fast(bessel_j1(to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast bessel_jn(int n, DoubleDoubleFast x) { return to_fast(bessel_jn(n, to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast bessel_y0(DoubleDoubleFast x) { return to_fast(bessel_y0(to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast bessel_y1(DoubleDoubleFast x) { return to_fast(bessel_y1(to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast bessel_yn(int n, DoubleDoubleFast x) { return to_fast(bessel_yn(n, to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast zeta(DoubleDoubleFast s) { return to_fast(zeta(to_strict(s))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast expint(DoubleDoubleFast x) { return to_fast(expint(to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast incgamma(DoubleDoubleFast a, DoubleDoubleFast x) { return to_fast(incgamma(to_strict(a), to_strict(x))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator+(DoubleDoubleComplexFast z, DoubleDoubleFast r) { return to_fast(to_strict(z) + to_strict(r)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator+(DoubleDoubleFast r, DoubleDoubleComplexFast z) { return to_fast(to_strict(r) + to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator-(DoubleDoubleComplexFast z, DoubleDoubleFast r) { return to_fast(to_strict(z) - to_strict(r)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator-(DoubleDoubleFast r, DoubleDoubleComplexFast z) { return to_fast(to_strict(r) - to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator*(DoubleDoubleComplexFast z, DoubleDoubleFast r) { return to_fast(to_strict(z) * to_strict(r)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator*(DoubleDoubleFast r, DoubleDoubleComplexFast z) { return to_fast(to_strict(r) * to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator/(DoubleDoubleComplexFast z, DoubleDoubleFast r) { return to_fast(to_strict(z) / to_strict(r)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator/(DoubleDoubleFast r, DoubleDoubleComplexFast z) { return to_fast(to_strict(r) / to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator+(DoubleDoubleComplexFast z, double b) { return to_fast(to_strict(z) + b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator+(double b, DoubleDoubleComplexFast z) { return to_fast(b + to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator-(DoubleDoubleComplexFast z, double b) { return to_fast(to_strict(z) - b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator-(double b, DoubleDoubleComplexFast z) { return to_fast(b - to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator*(DoubleDoubleComplexFast z, double b) { return to_fast(to_strict(z) * b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator*(double b, DoubleDoubleComplexFast z) { return to_fast(b * to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator/(DoubleDoubleComplexFast z, double b) { return to_fast(to_strict(z) / b); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast operator/(double b, DoubleDoubleComplexFast z) { return to_fast(b / to_strict(z)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast abs(DoubleDoubleComplexFast z) { return to_fast(abs(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast conj(DoubleDoubleComplexFast z) { return to_fast(conj(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast sqrt(DoubleDoubleComplexFast z) { return to_fast(sqrt(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast exp(DoubleDoubleComplexFast z) { return to_fast(exp(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast log(DoubleDoubleComplexFast z) { return to_fast(log(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast log1p(DoubleDoubleComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast log10(DoubleDoubleComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast sin(DoubleDoubleComplexFast z) { return to_fast(sin(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast cos(DoubleDoubleComplexFast z) { return to_fast(cos(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast tan(DoubleDoubleComplexFast z) { return to_fast(tan(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_asin_imag_mag(DoubleDoubleFast x, DoubleDoubleFast y) { return to_fast(xp_asin_imag_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_asin_real_leg(DoubleDoubleFast x, DoubleDoubleFast y) { return to_fast(xp_asin_real_leg(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_asin_real_mag(DoubleDoubleFast x, DoubleDoubleFast y) { return to_fast(xp_asin_real_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_abs_word(DoubleDoubleFast v) { return to_fast(xp_abs_word(to_strict(v))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast asin(DoubleDoubleComplexFast z) { return to_fast(asin(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast acos(DoubleDoubleComplexFast z) { return to_fast(acos(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_log_hypot2(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(xp_log_hypot2(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_atan2_safe(DoubleDoubleFast a, DoubleDoubleFast b) { return to_fast(xp_atan2_safe(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION bool xp_sq_underflowed(DoubleDoubleFast v) { return xp_sq_underflowed(to_strict(v)); }
XPMATH_INLINE_FUNCTION bool xp_acosh_chain_short(DoubleDoubleFast x, DoubleDoubleFast y) { return xp_acosh_chain_short(to_strict(x), to_strict(y)); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_atan_re_split(DoubleDoubleFast x, DoubleDoubleFast y) { return to_fast(xp_atan_re_split(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast atan(DoubleDoubleComplexFast z) { return to_fast(atan(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast sinh(DoubleDoubleComplexFast z) { return to_fast(sinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast cosh(DoubleDoubleComplexFast z) { return to_fast(cosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast tanh(DoubleDoubleComplexFast z) { return to_fast(tanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast xp_ki11_asinh_re(DoubleDoubleComplexFast z) { return to_fast(xp_ki11_asinh_re(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast asinh(DoubleDoubleComplexFast z) { return to_fast(asinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast acosh(DoubleDoubleComplexFast z) { return to_fast(acosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast atanh(DoubleDoubleComplexFast z) { return to_fast(atanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast pow(DoubleDoubleComplexFast z, DoubleDoubleComplexFast w) { return to_fast(pow(to_strict(z), to_strict(w))); }
XPMATH_INLINE_FUNCTION DoubleDoubleComplexFast polar(DoubleDoubleFast r, DoubleDoubleFast theta) { return to_fast(polar(to_strict(r), to_strict(theta))); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const DoubleDoubleFast& d) { return os << to_strict(d); }
inline std::ostream& operator<<(std::ostream& os, const DoubleDoubleComplexFast& z) { return os << to_strict(z); }
#endif

} // namespace xp
