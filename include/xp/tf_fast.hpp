// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// tf_fast.hpp -- the FAST tier of the TF backend (P3.0b).
//
// TripleFloatFast and TripleFloatComplexFast are shadow types of xp::TripleFloat and xp::TripleFloatComplex: same limbs and
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
#include <xp/tf_complex.hpp>

namespace xp {

struct TripleFloatFast;
struct TripleFloatComplexFast;

// Tier boundary conversions (defined below).
XPMATH_INLINE_FUNCTION TripleFloat to_strict(TripleFloatFast a);
XPMATH_INLINE_FUNCTION TripleFloatFast to_fast(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloatComplex to_strict(TripleFloatComplexFast z);
XPMATH_INLINE_FUNCTION TripleFloatComplexFast to_fast(TripleFloatComplex z);

// Forwarders the struct bodies below call.
XPMATH_INLINE_FUNCTION TripleFloatFast add(TripleFloatFast a, TripleFloatFast b);
XPMATH_INLINE_FUNCTION TripleFloatFast subtract(TripleFloatFast a, TripleFloatFast b);
XPMATH_INLINE_FUNCTION TripleFloatFast multiply(TripleFloatFast a, TripleFloatFast b);
XPMATH_INLINE_FUNCTION TripleFloatFast divide(TripleFloatFast a, TripleFloatFast b);
XPMATH_INLINE_FUNCTION TripleFloatFast multiply_scalar(TripleFloatFast a, float b);
XPMATH_INLINE_FUNCTION TripleFloatFast divide_scalar(TripleFloatFast a, float b);
XPMATH_INLINE_FUNCTION TripleFloatFast negate(TripleFloatFast a);

struct TripleFloatFast {
    float f0, f1, f2;
    XPMATH_INLINE_FUNCTION TripleFloatFast() : f0(0.0f), f1(0.0f), f2(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloatFast(float x) : f0(x), f1(0.0f), f2(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloatFast(float a0, float a1, float a2)
        : f0(a0), f1(a1), f2(a2) {}
    XPMATH_INLINE_FUNCTION TripleFloatFast(double x) {
        double r = x;
        float  c0 = (float)r; r = detail::eft_sub(r, (double)c0);
        float  c1 = (float)r; r = detail::eft_sub(r, (double)c1);
        float  c2 = (float)r;
        f0 = c0; f1 = c1; f2 = c2;
    }
    XPMATH_INLINE_FUNCTION TripleFloatFast(const TripleFloatFast& o)
        : f0(o.f0), f1(o.f1), f2(o.f2) {}
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator=(const TripleFloatFast& o) {
        f0=o.f0; f1=o.f1; f2=o.f2; return *this;
    }
    XPMATH_INLINE_FUNCTION float operator[](int i) const {
        return (i==0)?f0:(i==1)?f1:f2;
    }
    static XPMATH_INLINE_FUNCTION TripleFloatFast from_bits(uint32_t b0, uint32_t b1, uint32_t b2) {
        float f0, f1, f2;
#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
        f0 = __int_as_float(static_cast<int>(b0));
        f1 = __int_as_float(static_cast<int>(b1));
        f2 = __int_as_float(static_cast<int>(b2));
#else
        std::memcpy(&f0, &b0, sizeof(float));
        std::memcpy(&f1, &b1, sizeof(float));
        std::memcpy(&f2, &b2, sizeof(float));
#endif
        return TripleFloatFast(f0, f1, f2);
    }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator+(TripleFloatFast b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator-(TripleFloatFast b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator*(TripleFloatFast b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator/(TripleFloatFast b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator*(float b)  const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator/(float b)  const { return divide_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator+(float b)  const { return add(*this, TripleFloatFast(b)); }
    XPMATH_INLINE_FUNCTION TripleFloatFast operator-(float b)  const { return subtract(*this, TripleFloatFast(b)); }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator+=(TripleFloatFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator-=(TripleFloatFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator*=(TripleFloatFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator/=(TripleFloatFast b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator+=(float b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator-=(float b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator*=(float b) { *this = multiply_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatFast& operator/=(float b) { *this = divide_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION bool operator==(TripleFloatFast b) const { return f0==b.f0 && f1==b.f1 && f2==b.f2; }
    XPMATH_INLINE_FUNCTION bool operator!=(TripleFloatFast b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(TripleFloatFast b)  const {
        return f0<b.f0 || (f0==b.f0 && (f1<b.f1 || (f1==b.f1 && f2<b.f2)));
    }
    XPMATH_INLINE_FUNCTION bool operator>(TripleFloatFast b)  const {
        return f0>b.f0 || (f0==b.f0 && (f1>b.f1 || (f1==b.f1 && f2>b.f2)));
    }
    XPMATH_INLINE_FUNCTION bool operator<=(TripleFloatFast b) const { return !(b < *this); }
    XPMATH_INLINE_FUNCTION bool operator>=(TripleFloatFast b) const { return !(*this < b); }

    // Tier boundary: explicit only, so mixing tiers is a compile error.
    explicit XPMATH_INLINE_FUNCTION TripleFloatFast(TripleFloat s) : f0(s.f0), f1(s.f1), f2(s.f2) {}
    explicit XPMATH_INLINE_FUNCTION operator TripleFloat() const { return TripleFloat(f0, f1, f2); }
};

struct TripleFloatComplexFast {
    TripleFloatFast re;
    TripleFloatFast im;

    XPMATH_INLINE_FUNCTION TripleFloatComplexFast() : re(), im() {}
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast(float r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast(TripleFloatFast r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast(float r, float i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast(TripleFloatFast r, TripleFloatFast i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast(const TripleFloatComplexFast& o) : re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast& operator=(const TripleFloatComplexFast& o) { re = o.re; im = o.im; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast& operator=(TripleFloatFast r) { re = r; im = TripleFloatFast(); return *this; }

    // Tier boundary: explicit only.
    explicit XPMATH_INLINE_FUNCTION TripleFloatComplexFast(TripleFloatComplex z) : re(to_fast(z.re)), im(to_fast(z.im)) {}
    explicit XPMATH_INLINE_FUNCTION operator TripleFloatComplex() const { return TripleFloatComplex(to_strict(re), to_strict(im)); }

    XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator+(TripleFloatComplexFast b) const { return TripleFloatComplexFast(add(re, b.re), add(im, b.im)); }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator-(TripleFloatComplexFast b) const { return TripleFloatComplexFast(subtract(re, b.re), subtract(im, b.im)); }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator*(TripleFloatComplexFast b) const { return to_fast(to_strict(*this) * to_strict(b)); }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator/(TripleFloatComplexFast b) const { return to_fast(to_strict(*this) / to_strict(b)); }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator-() const { return TripleFloatComplexFast(negate(re), negate(im)); }

    XPMATH_INLINE_FUNCTION TripleFloatComplexFast& operator+=(TripleFloatComplexFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast& operator-=(TripleFloatComplexFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast& operator*=(TripleFloatComplexFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplexFast& operator/=(TripleFloatComplexFast b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(TripleFloatComplexFast b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(TripleFloatComplexFast b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION TripleFloatFast real() const { return re; }
    XPMATH_INLINE_FUNCTION TripleFloatFast imag() const { return im; }
};

XPMATH_INLINE_FUNCTION TripleFloat to_strict(TripleFloatFast a) { return TripleFloat(a); }
XPMATH_INLINE_FUNCTION TripleFloatFast to_fast(TripleFloat a) { return TripleFloatFast(a); }
XPMATH_INLINE_FUNCTION TripleFloatComplex to_strict(TripleFloatComplexFast z) { return TripleFloatComplex(z); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast to_fast(TripleFloatComplex z) { return TripleFloatComplexFast(z); }

// ---- forwarders: one line per public strict function / operator ------------
XPMATH_INLINE_FUNCTION TripleFloatFast add(TripleFloatFast a, TripleFloatFast b) { return to_fast(add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast subtract(TripleFloatFast a, TripleFloatFast b) { return to_fast(subtract(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast multiply(TripleFloatFast a, TripleFloatFast b) { return to_fast(multiply(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast divide(TripleFloatFast a, TripleFloatFast b) { return to_fast(divide(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast multiply_scalar(TripleFloatFast a, float b) { return to_fast(multiply_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast divide_scalar(TripleFloatFast a, float b) { return to_fast(divide_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast mul_pwr2(TripleFloatFast a, float b) { return to_fast(mul_pwr2(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast negate(TripleFloatFast a) { return to_fast(negate(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast abs(TripleFloatFast a) { return to_fast(abs(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast sqr(TripleFloatFast a) { return to_fast(sqr(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast sqrt(TripleFloatFast a) { return to_fast(sqrt(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast round_to_nearest_int(TripleFloatFast a) { return to_fast(round_to_nearest_int(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast pow_int(TripleFloatFast a, int n) { return to_fast(pow_int(to_strict(a), n)); }
XPMATH_INLINE_FUNCTION TripleFloatFast exp(TripleFloatFast a) { return to_fast(exp(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast log(TripleFloatFast a) { return to_fast(log(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast log1p(TripleFloatFast a) { return to_fast(log1p(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast pow(TripleFloatFast a, TripleFloatFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION void sincos(TripleFloatFast a, TripleFloatFast& sin_a, TripleFloatFast& cos_a) { TripleFloat t_sin_a; TripleFloat t_cos_a; sincos(to_strict(a), t_sin_a, t_cos_a); sin_a = to_fast(t_sin_a); cos_a = to_fast(t_cos_a); }
XPMATH_INLINE_FUNCTION void sinhcosh(TripleFloatFast a, TripleFloatFast& sinh_a, TripleFloatFast& cosh_a) { TripleFloat t_sinh_a; TripleFloat t_cosh_a; sinhcosh(to_strict(a), t_sinh_a, t_cosh_a); sinh_a = to_fast(t_sinh_a); cosh_a = to_fast(t_cosh_a); }
XPMATH_INLINE_FUNCTION TripleFloatFast angle(TripleFloatFast x, TripleFloatFast y) { return to_fast(angle(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION TripleFloatFast ceil(TripleFloatFast a) { return to_fast(ceil(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast floor(TripleFloatFast a) { return to_fast(floor(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast trunc(TripleFloatFast a) { return to_fast(trunc(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast round(TripleFloatFast a) { return to_fast(round(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast operator+(float a, TripleFloatFast b) { return to_fast(a + to_strict(b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast operator-(float a, TripleFloatFast b) { return to_fast(a - to_strict(b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast operator*(float a, TripleFloatFast b) { return to_fast(a * to_strict(b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast operator/(float a, TripleFloatFast b) { return to_fast(a / to_strict(b)); }
XPMATH_INLINE_FUNCTION TripleFloatFast TripleFloatFast_pi() { return to_fast(TripleFloat_pi()); }
XPMATH_INLINE_FUNCTION TripleFloatFast TripleFloatFast_e() { return to_fast(TripleFloat_e()); }
XPMATH_INLINE_FUNCTION TripleFloatFast TripleFloatFast_log2() { return to_fast(TripleFloat_log2()); }
XPMATH_INLINE_FUNCTION TripleFloatFast TripleFloatFast_log10() { return to_fast(TripleFloat_log10()); }
XPMATH_INLINE_FUNCTION TripleFloatFast TripleFloatFast_sqrt2() { return to_fast(TripleFloat_sqrt2()); }
XPMATH_INLINE_FUNCTION TripleFloatFast TripleFloatFast_euler_gamma() { return to_fast(TripleFloat_euler_gamma()); }
XPMATH_INLINE_FUNCTION TripleFloatFast sloppy_add(TripleFloatFast a, TripleFloatFast b) { return to_fast(sloppy_add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast ieee_add(TripleFloatFast a, TripleFloatFast b) { return to_fast(ieee_add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast sin(TripleFloatFast a) { return to_fast(sin(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast cos(TripleFloatFast a) { return to_fast(cos(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast tan(TripleFloatFast a) { return to_fast(tan(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast sinh(TripleFloatFast a) { return to_fast(sinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast cosh(TripleFloatFast a) { return to_fast(cosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast tanh(TripleFloatFast a) { return to_fast(tanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast atan2(TripleFloatFast y, TripleFloatFast x) { return to_fast(atan2(to_strict(y), to_strict(x))); }
XPMATH_INLINE_FUNCTION TripleFloatFast atan(TripleFloatFast a) { return to_fast(atan(to_strict(a))); }
XPMATH_INLINE_FUNCTION int tf_cmp_one(TripleFloatFast a) { return tf_cmp_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION int tf_cmp_abs_one(TripleFloatFast a) { return tf_cmp_abs_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION TripleFloatFast asin(TripleFloatFast a) { return to_fast(asin(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast acos(TripleFloatFast a) { return to_fast(acos(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast asinh(TripleFloatFast a) { return to_fast(asinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast acosh(TripleFloatFast a) { return to_fast(acosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast atanh(TripleFloatFast a) { return to_fast(atanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast exp2(TripleFloatFast a) { return to_fast(exp2(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast exp10(TripleFloatFast a) { return to_fast(exp10(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast expm1(TripleFloatFast a) { return to_fast(expm1(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast log10(TripleFloatFast a) { return to_fast(log10(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast log2(TripleFloatFast a) { return to_fast(log2(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast fmod(TripleFloatFast a, TripleFloatFast b) { return to_fast(fmod(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast remainder(TripleFloatFast a, TripleFloatFast b) { return to_fast(remainder(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast fdim(TripleFloatFast a, TripleFloatFast b) { return to_fast(fdim(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast fmax(TripleFloatFast a, TripleFloatFast b) { return to_fast(fmax(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast fmin(TripleFloatFast a, TripleFloatFast b) { return to_fast(fmin(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast fma(TripleFloatFast a, TripleFloatFast b, TripleFloatFast c) { return to_fast(fma(to_strict(a), to_strict(b), to_strict(c))); }
XPMATH_INLINE_FUNCTION TripleFloatFast hypot(TripleFloatFast a, TripleFloatFast b) { return to_fast(hypot(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast copysign(TripleFloatFast a, TripleFloatFast b) { return to_fast(copysign(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator+(TripleFloatComplexFast z, TripleFloatFast r) { return to_fast(to_strict(z) + to_strict(r)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator+(TripleFloatFast r, TripleFloatComplexFast z) { return to_fast(to_strict(r) + to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator-(TripleFloatComplexFast z, TripleFloatFast r) { return to_fast(to_strict(z) - to_strict(r)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator-(TripleFloatFast r, TripleFloatComplexFast z) { return to_fast(to_strict(r) - to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator*(TripleFloatComplexFast z, TripleFloatFast r) { return to_fast(to_strict(z) * to_strict(r)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator*(TripleFloatFast r, TripleFloatComplexFast z) { return to_fast(to_strict(r) * to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator/(TripleFloatComplexFast z, TripleFloatFast r) { return to_fast(to_strict(z) / to_strict(r)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator/(TripleFloatFast r, TripleFloatComplexFast z) { return to_fast(to_strict(r) / to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator+(TripleFloatComplexFast z, float b) { return to_fast(to_strict(z) + b); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator+(float b, TripleFloatComplexFast z) { return to_fast(b + to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator-(TripleFloatComplexFast z, float b) { return to_fast(to_strict(z) - b); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator-(float b, TripleFloatComplexFast z) { return to_fast(b - to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator*(TripleFloatComplexFast z, float b) { return to_fast(to_strict(z) * b); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator*(float b, TripleFloatComplexFast z) { return to_fast(b * to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator/(TripleFloatComplexFast z, float b) { return to_fast(to_strict(z) / b); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast operator/(float b, TripleFloatComplexFast z) { return to_fast(b / to_strict(z)); }
XPMATH_INLINE_FUNCTION TripleFloatFast abs(TripleFloatComplexFast z) { return to_fast(abs(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatFast norm(TripleFloatComplexFast z) { return to_fast(norm(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatFast arg(TripleFloatComplexFast z) { return to_fast(arg(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast conj(TripleFloatComplexFast z) { return to_fast(conj(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast sqrt(TripleFloatComplexFast z) { return to_fast(sqrt(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast exp(TripleFloatComplexFast z) { return to_fast(exp(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast log(TripleFloatComplexFast z) { return to_fast(log(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast log1p(TripleFloatComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast log10(TripleFloatComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast sin(TripleFloatComplexFast z) { return to_fast(sin(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast cos(TripleFloatComplexFast z) { return to_fast(cos(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast tan(TripleFloatComplexFast z) { return to_fast(tan(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_asin_imag_mag(TripleFloatFast x, TripleFloatFast y) { return to_fast(xp_asin_imag_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_asin_real_leg(TripleFloatFast x, TripleFloatFast y) { return to_fast(xp_asin_real_leg(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_abs_word(TripleFloatFast v) { return to_fast(xp_abs_word(to_strict(v))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast asin(TripleFloatComplexFast z) { return to_fast(asin(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast acos(TripleFloatComplexFast z) { return to_fast(acos(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_log_hypot2(TripleFloatFast a, TripleFloatFast b) { return to_fast(xp_log_hypot2(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_atan2_safe(TripleFloatFast a, TripleFloatFast b) { return to_fast(xp_atan2_safe(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION bool xp_sq_underflowed(TripleFloatFast v) { return xp_sq_underflowed(to_strict(v)); }
XPMATH_INLINE_FUNCTION bool xp_acosh_chain_short(TripleFloatFast x, TripleFloatFast y) { return xp_acosh_chain_short(to_strict(x), to_strict(y)); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_atan_re_split(TripleFloatFast x, TripleFloatFast y) { return to_fast(xp_atan_re_split(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast atan(TripleFloatComplexFast z) { return to_fast(atan(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast sinh(TripleFloatComplexFast z) { return to_fast(sinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast cosh(TripleFloatComplexFast z) { return to_fast(cosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast tanh(TripleFloatComplexFast z) { return to_fast(tanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatFast xp_ki11_asinh_re(TripleFloatComplexFast z) { return to_fast(xp_ki11_asinh_re(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast asinh(TripleFloatComplexFast z) { return to_fast(asinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast acosh(TripleFloatComplexFast z) { return to_fast(acosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast atanh(TripleFloatComplexFast z) { return to_fast(atanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast pow(TripleFloatComplexFast z, TripleFloatComplexFast w) { return to_fast(pow(to_strict(z), to_strict(w))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast polar(TripleFloatFast r, TripleFloatFast theta) { return to_fast(polar(to_strict(r), to_strict(theta))); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const TripleFloatFast& d) { return os << to_strict(d); }
inline std::ostream& operator<<(std::ostream& os, const TripleFloatComplexFast& z) { return os << to_strict(z); }
#endif

} // namespace xp
