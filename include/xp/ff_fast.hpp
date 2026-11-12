// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// ff_fast.hpp -- the FAST tier of the FF backend (P3.0b).
//
// FloatFloatFast and FloatFloatComplexFast are shadow types of xp::FloatFloat and xp::FloatFloatComplex: same limbs and
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
#include <xp/ff_complex.hpp>

namespace xp {

struct FloatFloatFast;
struct FloatFloatComplexFast;

// Tier boundary conversions (defined below).
XPMATH_INLINE_FUNCTION FloatFloat to_strict(FloatFloatFast a);
XPMATH_INLINE_FUNCTION FloatFloatFast to_fast(FloatFloat a);
XPMATH_INLINE_FUNCTION FloatFloatComplex to_strict(FloatFloatComplexFast z);
XPMATH_INLINE_FUNCTION FloatFloatComplexFast to_fast(FloatFloatComplex z);

// Forwarders the struct bodies below call.
XPMATH_INLINE_FUNCTION FloatFloatFast add(FloatFloatFast a, FloatFloatFast b);
XPMATH_INLINE_FUNCTION FloatFloatFast subtract(FloatFloatFast a, FloatFloatFast b);
XPMATH_INLINE_FUNCTION FloatFloatFast multiply(FloatFloatFast a, FloatFloatFast b);
XPMATH_INLINE_FUNCTION FloatFloatFast divide(FloatFloatFast a, FloatFloatFast b);
XPMATH_INLINE_FUNCTION FloatFloatFast multiply_scalar(FloatFloatFast a, float b);
XPMATH_INLINE_FUNCTION FloatFloatFast divide_scalar(FloatFloatFast a, float b);
XPMATH_INLINE_FUNCTION FloatFloatFast negate(FloatFloatFast a);

struct alignas(8) FloatFloatFast {
    float hi;
    float lo;
    XPMATH_INLINE_FUNCTION FloatFloatFast() : hi(0.0f), lo(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloatFast(float h) : hi(h), lo(0.0f) {}
    XPMATH_INLINE_FUNCTION FloatFloatFast(float h, float l) : hi(h), lo(l) {}
    XPMATH_INLINE_FUNCTION FloatFloatFast(double h) : hi((float)h), lo((float)detail::eft_sub(h, (double)(float)h)) {}
    XPMATH_INLINE_FUNCTION FloatFloatFast(const FloatFloatFast& o) : hi(o.hi), lo(o.lo) {}
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator=(const FloatFloatFast& o) { hi=o.hi; lo=o.lo; return *this; }
    static XPMATH_INLINE_FUNCTION FloatFloatFast from_bits(uint32_t hi_bits, uint32_t lo_bits) {
        float h, l;
#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
        h = __int_as_float(static_cast<int>(hi_bits));
        l = __int_as_float(static_cast<int>(lo_bits));
#else
        std::memcpy(&h, &hi_bits, sizeof(float));
        std::memcpy(&l, &lo_bits, sizeof(float));
#endif
        return FloatFloatFast(h, l);
    }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator+(FloatFloatFast b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator-(FloatFloatFast b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator*(FloatFloatFast b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator/(FloatFloatFast b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator*(float b)  const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator/(float b)  const { return divide_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator+(float b)  const { return add(*this, FloatFloatFast(b)); }
    XPMATH_INLINE_FUNCTION FloatFloatFast operator-(float b)  const { return subtract(*this, FloatFloatFast(b)); }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator+=(FloatFloatFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator-=(FloatFloatFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator*=(FloatFloatFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator/=(FloatFloatFast b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator+=(float b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator-=(float b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator*=(float b) { *this = multiply_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatFast& operator/=(float b) { *this = divide_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION bool operator==(FloatFloatFast b) const { return hi==b.hi && lo==b.lo; }
    XPMATH_INLINE_FUNCTION bool operator!=(FloatFloatFast b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(FloatFloatFast b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    XPMATH_INLINE_FUNCTION bool operator>(FloatFloatFast b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    XPMATH_INLINE_FUNCTION bool operator<=(FloatFloatFast b) const { return !(b < *this); }
    XPMATH_INLINE_FUNCTION bool operator>=(FloatFloatFast b) const { return !(*this < b); }

    // Tier boundary: explicit only, so mixing tiers is a compile error.
    explicit XPMATH_INLINE_FUNCTION FloatFloatFast(FloatFloat s) : hi(s.hi), lo(s.lo) {}
    explicit XPMATH_INLINE_FUNCTION operator FloatFloat() const { return FloatFloat(hi, lo); }
};

struct FloatFloatComplexFast {
    FloatFloatFast re;
    FloatFloatFast im;

    XPMATH_INLINE_FUNCTION FloatFloatComplexFast() : re(), im() {}
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast(float r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast(FloatFloatFast r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast(float r, float i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast(FloatFloatFast r, FloatFloatFast i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast(const FloatFloatComplexFast& o) : re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast& operator=(const FloatFloatComplexFast& o) { re = o.re; im = o.im; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast& operator=(FloatFloatFast r) { re = r; im = FloatFloatFast(); return *this; }

    // Tier boundary: explicit only.
    explicit XPMATH_INLINE_FUNCTION FloatFloatComplexFast(FloatFloatComplex z) : re(to_fast(z.re)), im(to_fast(z.im)) {}
    explicit XPMATH_INLINE_FUNCTION operator FloatFloatComplex() const { return FloatFloatComplex(to_strict(re), to_strict(im)); }

    XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator+(FloatFloatComplexFast b) const { return FloatFloatComplexFast(add(re, b.re), add(im, b.im)); }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator-(FloatFloatComplexFast b) const { return FloatFloatComplexFast(subtract(re, b.re), subtract(im, b.im)); }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator*(FloatFloatComplexFast b) const { return to_fast(to_strict(*this) * to_strict(b)); }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator/(FloatFloatComplexFast b) const { return to_fast(to_strict(*this) / to_strict(b)); }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator-() const { return FloatFloatComplexFast(negate(re), negate(im)); }

    XPMATH_INLINE_FUNCTION FloatFloatComplexFast& operator+=(FloatFloatComplexFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast& operator-=(FloatFloatComplexFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast& operator*=(FloatFloatComplexFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION FloatFloatComplexFast& operator/=(FloatFloatComplexFast b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(FloatFloatComplexFast b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(FloatFloatComplexFast b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION FloatFloatFast real() const { return re; }
    XPMATH_INLINE_FUNCTION FloatFloatFast imag() const { return im; }
};

XPMATH_INLINE_FUNCTION FloatFloat to_strict(FloatFloatFast a) { return FloatFloat(a); }
XPMATH_INLINE_FUNCTION FloatFloatFast to_fast(FloatFloat a) { return FloatFloatFast(a); }
XPMATH_INLINE_FUNCTION FloatFloatComplex to_strict(FloatFloatComplexFast z) { return FloatFloatComplex(z); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast to_fast(FloatFloatComplex z) { return FloatFloatComplexFast(z); }

// ---- forwarders: one line per public strict function / operator ------------
XPMATH_INLINE_FUNCTION FloatFloatFast add(FloatFloatFast a, FloatFloatFast b) { return to_fast(add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast subtract(FloatFloatFast a, FloatFloatFast b) { return to_fast(subtract(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast multiply(FloatFloatFast a, FloatFloatFast b) { return to_fast(multiply(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast divide(FloatFloatFast a, FloatFloatFast b) { return to_fast(divide(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast multiply_scalar(FloatFloatFast a, float b) { return to_fast(multiply_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION FloatFloatFast divide_scalar(FloatFloatFast a, float b) { return to_fast(divide_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION FloatFloatFast negate(FloatFloatFast a) { return to_fast(negate(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast abs(FloatFloatFast a) { return to_fast(abs(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast sqrt(FloatFloatFast a) { return to_fast(sqrt(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast round_to_nearest_int(FloatFloatFast a) { return to_fast(round_to_nearest_int(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast pow_int(FloatFloatFast a, int n) { return to_fast(pow_int(to_strict(a), n)); }
XPMATH_INLINE_FUNCTION FloatFloatFast exp(FloatFloatFast a) { return to_fast(exp(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast log(FloatFloatFast a) { return to_fast(log(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast pow(FloatFloatFast a, FloatFloatFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_mul_ext(FloatFloatFast x, FloatFloatFast y, float& err) { return to_fast(ff_mul_ext(to_strict(x), to_strict(y), err)); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_log_ext(FloatFloatFast a, float& err) { return to_fast(ff_log_ext(to_strict(a), err)); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_exp_ext(FloatFloatFast a, float resid) { return to_fast(ff_exp_ext(to_strict(a), resid)); }
XPMATH_INLINE_FUNCTION void sinhcosh(FloatFloatFast a, FloatFloatFast& x, FloatFloatFast& y) { FloatFloat t_x; FloatFloat t_y; sinhcosh(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
XPMATH_INLINE_FUNCTION void sincos(FloatFloatFast a, FloatFloatFast& x, FloatFloatFast& y) { FloatFloat t_x; FloatFloat t_y; sincos(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
XPMATH_INLINE_FUNCTION FloatFloatFast angle(FloatFloatFast x, FloatFloatFast y) { return to_fast(angle(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION FloatFloatFast operator+(float a, FloatFloatFast b) { return to_fast(a + to_strict(b)); }
XPMATH_INLINE_FUNCTION FloatFloatFast operator-(float a, FloatFloatFast b) { return to_fast(a - to_strict(b)); }
XPMATH_INLINE_FUNCTION FloatFloatFast operator*(float a, FloatFloatFast b) { return to_fast(a * to_strict(b)); }
XPMATH_INLINE_FUNCTION FloatFloatFast operator/(float a, FloatFloatFast b) { return to_fast(a / to_strict(b)); }
XPMATH_INLINE_FUNCTION FloatFloatFast FloatFloatFast_pi() { return to_fast(FloatFloat_pi()); }
XPMATH_INLINE_FUNCTION FloatFloatFast FloatFloatFast_e() { return to_fast(FloatFloat_e()); }
XPMATH_INLINE_FUNCTION FloatFloatFast FloatFloatFast_log2() { return to_fast(FloatFloat_log2()); }
XPMATH_INLINE_FUNCTION FloatFloatFast FloatFloatFast_log10() { return to_fast(FloatFloat_log10()); }
XPMATH_INLINE_FUNCTION FloatFloatFast FloatFloatFast_sqrt2() { return to_fast(FloatFloat_sqrt2()); }
XPMATH_INLINE_FUNCTION FloatFloatFast FloatFloatFast_euler_gamma() { return to_fast(FloatFloat_euler_gamma()); }
XPMATH_INLINE_FUNCTION FloatFloatFast log2(FloatFloatFast a) { return to_fast(log2(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast log10(FloatFloatFast a) { return to_fast(log10(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast log1p(FloatFloatFast a) { return to_fast(log1p(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast exp2(FloatFloatFast a) { return to_fast(exp2(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast exp10(FloatFloatFast a) { return to_fast(exp10(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast expm1(FloatFloatFast a) { return to_fast(expm1(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast sin(FloatFloatFast a) { return to_fast(sin(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast cos(FloatFloatFast a) { return to_fast(cos(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast tan(FloatFloatFast a) { return to_fast(tan(to_strict(a))); }
XPMATH_INLINE_FUNCTION int ff_cmp_one(FloatFloatFast a) { return ff_cmp_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION int ff_cmp_abs_one(FloatFloatFast a) { return ff_cmp_abs_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION FloatFloatFast asin(FloatFloatFast a) { return to_fast(asin(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast acos(FloatFloatFast a) { return to_fast(acos(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast atan(FloatFloatFast a) { return to_fast(atan(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast atan2(FloatFloatFast y, FloatFloatFast x) { return to_fast(atan2(to_strict(y), to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast sinh(FloatFloatFast a) { return to_fast(sinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast cosh(FloatFloatFast a) { return to_fast(cosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast tanh(FloatFloatFast a) { return to_fast(tanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast asinh(FloatFloatFast a) { return to_fast(asinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast acosh(FloatFloatFast a) { return to_fast(acosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast atanh(FloatFloatFast a) { return to_fast(atanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast hypot(FloatFloatFast a, FloatFloatFast b) { return to_fast(hypot(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast ceil(FloatFloatFast a) { return to_fast(ceil(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast floor(FloatFloatFast a) { return to_fast(floor(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast trunc(FloatFloatFast a) { return to_fast(trunc(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast round(FloatFloatFast a) { return to_fast(round(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast fmod(FloatFloatFast a, FloatFloatFast b) { return to_fast(fmod(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast remainder(FloatFloatFast a, FloatFloatFast b) { return to_fast(remainder(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast copysign(FloatFloatFast a, FloatFloatFast b) { return to_fast(copysign(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast fmax(FloatFloatFast a, FloatFloatFast b) { return to_fast(fmax(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast fmin(FloatFloatFast a, FloatFloatFast b) { return to_fast(fmin(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast fdim(FloatFloatFast a, FloatFloatFast b) { return to_fast(fdim(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast fma(FloatFloatFast a, FloatFloatFast b, FloatFloatFast c) { return to_fast(fma(to_strict(a), to_strict(b), to_strict(c))); }
XPMATH_INLINE_FUNCTION FloatFloatFast erfc_asymptotic_sum(FloatFloatFast az, FloatFloatFast z2) { return to_fast(erfc_asymptotic_sum(to_strict(az), to_strict(z2))); }
XPMATH_INLINE_FUNCTION FloatFloatFast erf(FloatFloatFast z) { return to_fast(erf(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatFast erfc(FloatFloatFast z) { return to_fast(erfc(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_tgamma_lanczos(FloatFloatFast a) { return to_fast(ff_tgamma_lanczos(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast tgamma(FloatFloatFast a) { return to_fast(tgamma(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast bessel_j0(FloatFloatFast x) { return to_fast(bessel_j0(to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast bessel_j1(FloatFloatFast x) { return to_fast(bessel_j1(to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast bessel_jn(int n, FloatFloatFast x) { return to_fast(bessel_jn(n, to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast bessel_y0(FloatFloatFast x) { return to_fast(bessel_y0(to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast bessel_y1(FloatFloatFast x) { return to_fast(bessel_y1(to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast bessel_yn(int n, FloatFloatFast x) { return to_fast(bessel_yn(n, to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast zeta(FloatFloatFast s) { return to_fast(zeta(to_strict(s))); }
XPMATH_INLINE_FUNCTION FloatFloatFast expint(FloatFloatFast x) { return to_fast(expint(to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatFast incgamma(FloatFloatFast a, FloatFloatFast x) { return to_fast(incgamma(to_strict(a), to_strict(x))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator+(FloatFloatComplexFast z, FloatFloatFast r) { return to_fast(to_strict(z) + to_strict(r)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator+(FloatFloatFast r, FloatFloatComplexFast z) { return to_fast(to_strict(r) + to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator-(FloatFloatComplexFast z, FloatFloatFast r) { return to_fast(to_strict(z) - to_strict(r)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator-(FloatFloatFast r, FloatFloatComplexFast z) { return to_fast(to_strict(r) - to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator*(FloatFloatComplexFast z, FloatFloatFast r) { return to_fast(to_strict(z) * to_strict(r)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator*(FloatFloatFast r, FloatFloatComplexFast z) { return to_fast(to_strict(r) * to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator/(FloatFloatComplexFast z, FloatFloatFast r) { return to_fast(to_strict(z) / to_strict(r)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator/(FloatFloatFast r, FloatFloatComplexFast z) { return to_fast(to_strict(r) / to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator+(FloatFloatComplexFast z, float b) { return to_fast(to_strict(z) + b); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator+(float b, FloatFloatComplexFast z) { return to_fast(b + to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator-(FloatFloatComplexFast z, float b) { return to_fast(to_strict(z) - b); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator-(float b, FloatFloatComplexFast z) { return to_fast(b - to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator*(FloatFloatComplexFast z, float b) { return to_fast(to_strict(z) * b); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator*(float b, FloatFloatComplexFast z) { return to_fast(b * to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator/(FloatFloatComplexFast z, float b) { return to_fast(to_strict(z) / b); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast operator/(float b, FloatFloatComplexFast z) { return to_fast(b / to_strict(z)); }
XPMATH_INLINE_FUNCTION FloatFloatFast abs(FloatFloatComplexFast z) { return to_fast(abs(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast conj(FloatFloatComplexFast z) { return to_fast(conj(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast sqrt(FloatFloatComplexFast z) { return to_fast(sqrt(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast exp(FloatFloatComplexFast z) { return to_fast(exp(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast log(FloatFloatComplexFast z) { return to_fast(log(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast log1p(FloatFloatComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast log10(FloatFloatComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast sin(FloatFloatComplexFast z) { return to_fast(sin(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast cos(FloatFloatComplexFast z) { return to_fast(cos(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast tan(FloatFloatComplexFast z) { return to_fast(tan(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_asin_imag_mag(FloatFloatFast x, FloatFloatFast y) { return to_fast(xp_asin_imag_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_asin_real_leg(FloatFloatFast x, FloatFloatFast y) { return to_fast(xp_asin_real_leg(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_asin_real_mag(FloatFloatFast x, FloatFloatFast y) { return to_fast(xp_asin_real_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_abs_word(FloatFloatFast v) { return to_fast(xp_abs_word(to_strict(v))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast asin(FloatFloatComplexFast z) { return to_fast(asin(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast acos(FloatFloatComplexFast z) { return to_fast(acos(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_log_hypot2(FloatFloatFast a, FloatFloatFast b) { return to_fast(xp_log_hypot2(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_atan2_safe(FloatFloatFast a, FloatFloatFast b) { return to_fast(xp_atan2_safe(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION bool xp_sq_underflowed(FloatFloatFast v) { return xp_sq_underflowed(to_strict(v)); }
XPMATH_INLINE_FUNCTION bool xp_acosh_chain_short(FloatFloatFast x, FloatFloatFast y) { return xp_acosh_chain_short(to_strict(x), to_strict(y)); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_atan_re_split(FloatFloatFast x, FloatFloatFast y) { return to_fast(xp_atan_re_split(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast atan(FloatFloatComplexFast z) { return to_fast(atan(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast sinh(FloatFloatComplexFast z) { return to_fast(sinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast cosh(FloatFloatComplexFast z) { return to_fast(cosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast tanh(FloatFloatComplexFast z) { return to_fast(tanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatFast xp_ki11_asinh_re(FloatFloatComplexFast z) { return to_fast(xp_ki11_asinh_re(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast asinh(FloatFloatComplexFast z) { return to_fast(asinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast acosh(FloatFloatComplexFast z) { return to_fast(acosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast atanh(FloatFloatComplexFast z) { return to_fast(atanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast pow(FloatFloatComplexFast z, FloatFloatComplexFast w) { return to_fast(pow(to_strict(z), to_strict(w))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast polar(FloatFloatFast r, FloatFloatFast theta) { return to_fast(polar(to_strict(r), to_strict(theta))); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const FloatFloatFast& d) { return os << to_strict(d); }
inline std::ostream& operator<<(std::ostream& os, const FloatFloatComplexFast& z) { return os << to_strict(z); }
#endif

} // namespace xp
