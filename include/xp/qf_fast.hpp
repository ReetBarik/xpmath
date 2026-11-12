// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// qf_fast.hpp -- the FAST tier of the QF backend (P3.0b).
//
// QuadFloatFast and QuadFloatComplexFast are shadow types of xp::QuadFloat and xp::QuadFloatComplex: same limbs and
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
#include <xp/qf_complex.hpp>

namespace xp {

struct QuadFloatFast;
struct QuadFloatComplexFast;

// Tier boundary conversions (defined below).
XPMATH_INLINE_FUNCTION QuadFloat to_strict(QuadFloatFast a);
XPMATH_INLINE_FUNCTION QuadFloatFast to_fast(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloatComplex to_strict(QuadFloatComplexFast z);
XPMATH_INLINE_FUNCTION QuadFloatComplexFast to_fast(QuadFloatComplex z);

// Forwarders the struct bodies below call.
XPMATH_INLINE_FUNCTION QuadFloatFast add(QuadFloatFast a, QuadFloatFast b);
XPMATH_INLINE_FUNCTION QuadFloatFast subtract(QuadFloatFast a, QuadFloatFast b);
XPMATH_INLINE_FUNCTION QuadFloatFast multiply(QuadFloatFast a, QuadFloatFast b);
XPMATH_INLINE_FUNCTION QuadFloatFast divide(QuadFloatFast a, QuadFloatFast b);
XPMATH_INLINE_FUNCTION QuadFloatFast multiply_scalar(QuadFloatFast a, float b);
XPMATH_INLINE_FUNCTION QuadFloatFast divide_scalar(QuadFloatFast a, float b);
XPMATH_INLINE_FUNCTION QuadFloatFast negate(QuadFloatFast a);

struct alignas(16) QuadFloatFast {
    float f0, f1, f2, f3;
    XPMATH_INLINE_FUNCTION QuadFloatFast() : f0(0.0f), f1(0.0f), f2(0.0f), f3(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloatFast(float x) : f0(x), f1(0.0f), f2(0.0f), f3(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloatFast(float a0, float a1, float a2, float a3)
        : f0(a0), f1(a1), f2(a2), f3(a3) {}
    XPMATH_INLINE_FUNCTION QuadFloatFast(double x) {
        double r = x;
        float  c0 = (float)r; r = detail::eft_sub(r, (double)c0);
        float  c1 = (float)r; r = detail::eft_sub(r, (double)c1);
        float  c2 = (float)r; r = detail::eft_sub(r, (double)c2);
        float  c3 = (float)r;
        f0 = c0; f1 = c1; f2 = c2; f3 = c3;
    }
    XPMATH_INLINE_FUNCTION QuadFloatFast(const QuadFloatFast& o)
        : f0(o.f0), f1(o.f1), f2(o.f2), f3(o.f3) {}
    XPMATH_INLINE_FUNCTION QuadFloatFast& operator=(const QuadFloatFast& o) {
        f0=o.f0; f1=o.f1; f2=o.f2; f3=o.f3; return *this;
    }
    XPMATH_INLINE_FUNCTION float operator[](int i) const {
        return (i==0)?f0:(i==1)?f1:(i==2)?f2:f3;
    }
    static XPMATH_INLINE_FUNCTION QuadFloatFast from_bits(uint32_t b0, uint32_t b1,
                                                      uint32_t b2, uint32_t b3) {
        float a0, a1, a2, a3;
#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
        a0 = __int_as_float(static_cast<int>(b0));
        a1 = __int_as_float(static_cast<int>(b1));
        a2 = __int_as_float(static_cast<int>(b2));
        a3 = __int_as_float(static_cast<int>(b3));
#else
        std::memcpy(&a0, &b0, sizeof(float));
        std::memcpy(&a1, &b1, sizeof(float));
        std::memcpy(&a2, &b2, sizeof(float));
        std::memcpy(&a3, &b3, sizeof(float));
#endif
        return QuadFloatFast(a0, a1, a2, a3);
    }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator+(QuadFloatFast b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator-(QuadFloatFast b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator*(QuadFloatFast b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator/(QuadFloatFast b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator*(float b) const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator+(float b) const { return add(*this, QuadFloatFast(b)); }
    XPMATH_INLINE_FUNCTION QuadFloatFast operator-(float b) const { return subtract(*this, QuadFloatFast(b)); }
    XPMATH_INLINE_FUNCTION QuadFloatFast& operator+=(QuadFloatFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatFast& operator-=(QuadFloatFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatFast& operator*=(QuadFloatFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatFast& operator/=(QuadFloatFast b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION bool operator==(QuadFloatFast b) const {
        return f0==b.f0 && f1==b.f1 && f2==b.f2 && f3==b.f3;
    }
    XPMATH_INLINE_FUNCTION bool operator!=(QuadFloatFast b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(QuadFloatFast b) const {
        if (f0 != b.f0) return f0 < b.f0;
        if (f1 != b.f1) return f1 < b.f1;
        if (f2 != b.f2) return f2 < b.f2;
        return f3 < b.f3;
    }
    XPMATH_INLINE_FUNCTION bool operator>(QuadFloatFast b) const {
        if (f0 != b.f0) return f0 > b.f0;
        if (f1 != b.f1) return f1 > b.f1;
        if (f2 != b.f2) return f2 > b.f2;
        return f3 > b.f3;
    }
    XPMATH_INLINE_FUNCTION bool operator<=(QuadFloatFast b) const { return !(*this > b); }
    XPMATH_INLINE_FUNCTION bool operator>=(QuadFloatFast b) const { return !(*this < b); }

    // Tier boundary: explicit only, so mixing tiers is a compile error.
    explicit XPMATH_INLINE_FUNCTION QuadFloatFast(QuadFloat s) : f0(s.f0), f1(s.f1), f2(s.f2), f3(s.f3) {}
    explicit XPMATH_INLINE_FUNCTION operator QuadFloat() const { return QuadFloat(f0, f1, f2, f3); }
};

struct QuadFloatComplexFast {
    QuadFloatFast re;
    QuadFloatFast im;

    XPMATH_INLINE_FUNCTION QuadFloatComplexFast() : re(), im() {}
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast(float r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast(QuadFloatFast r) : re(r), im() {}
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast(float r, float i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast(QuadFloatFast r, QuadFloatFast i) : re(r), im(i) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast(const QuadFloatComplexFast& o) : re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast& operator=(const QuadFloatComplexFast& o) { re = o.re; im = o.im; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast& operator=(QuadFloatFast r) { re = r; im = QuadFloatFast(); return *this; }

    // Tier boundary: explicit only.
    explicit XPMATH_INLINE_FUNCTION QuadFloatComplexFast(QuadFloatComplex z) : re(to_fast(z.re)), im(to_fast(z.im)) {}
    explicit XPMATH_INLINE_FUNCTION operator QuadFloatComplex() const { return QuadFloatComplex(to_strict(re), to_strict(im)); }

    XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator+(QuadFloatComplexFast b) const { return QuadFloatComplexFast(add(re, b.re), add(im, b.im)); }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator-(QuadFloatComplexFast b) const { return QuadFloatComplexFast(subtract(re, b.re), subtract(im, b.im)); }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator*(QuadFloatComplexFast b) const { return to_fast(to_strict(*this) * to_strict(b)); }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator/(QuadFloatComplexFast b) const { return to_fast(to_strict(*this) / to_strict(b)); }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator-() const { return QuadFloatComplexFast(negate(re), negate(im)); }

    XPMATH_INLINE_FUNCTION QuadFloatComplexFast& operator+=(QuadFloatComplexFast b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast& operator-=(QuadFloatComplexFast b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast& operator*=(QuadFloatComplexFast b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplexFast& operator/=(QuadFloatComplexFast b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(QuadFloatComplexFast b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(QuadFloatComplexFast b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION QuadFloatFast real() const { return re; }
    XPMATH_INLINE_FUNCTION QuadFloatFast imag() const { return im; }
};

XPMATH_INLINE_FUNCTION QuadFloat to_strict(QuadFloatFast a) { return QuadFloat(a); }
XPMATH_INLINE_FUNCTION QuadFloatFast to_fast(QuadFloat a) { return QuadFloatFast(a); }
XPMATH_INLINE_FUNCTION QuadFloatComplex to_strict(QuadFloatComplexFast z) { return QuadFloatComplex(z); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast to_fast(QuadFloatComplex z) { return QuadFloatComplexFast(z); }

// ---- forwarders: one line per public strict function / operator ------------
XPMATH_INLINE_FUNCTION QuadFloatFast add(QuadFloatFast a, QuadFloatFast b) { return to_fast(add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast subtract(QuadFloatFast a, QuadFloatFast b) { return to_fast(subtract(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast multiply(QuadFloatFast a, QuadFloatFast b) { return to_fast(multiply(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast divide(QuadFloatFast a, QuadFloatFast b) { return to_fast(divide(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast multiply_scalar(QuadFloatFast a, float b) { return to_fast(multiply_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION QuadFloatFast divide_scalar(QuadFloatFast a, float b) { return to_fast(divide_scalar(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION QuadFloatFast mul_pwr2(QuadFloatFast a, float b) { return to_fast(mul_pwr2(to_strict(a), b)); }
XPMATH_INLINE_FUNCTION QuadFloatFast negate(QuadFloatFast a) { return to_fast(negate(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast abs(QuadFloatFast a) { return to_fast(abs(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast sqr(QuadFloatFast a) { return to_fast(sqr(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast sqrt(QuadFloatFast a) { return to_fast(sqrt(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast round_to_nearest_int(QuadFloatFast a) { return to_fast(round_to_nearest_int(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast pow_int(QuadFloatFast a, int n) { return to_fast(pow_int(to_strict(a), n)); }
XPMATH_INLINE_FUNCTION QuadFloatFast exp(QuadFloatFast a) { return to_fast(exp(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast log(QuadFloatFast a) { return to_fast(log(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast pow(QuadFloatFast a, QuadFloatFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION void sincos(QuadFloatFast a, QuadFloatFast& sin_a, QuadFloatFast& cos_a) { QuadFloat t_sin_a; QuadFloat t_cos_a; sincos(to_strict(a), t_sin_a, t_cos_a); sin_a = to_fast(t_sin_a); cos_a = to_fast(t_cos_a); }
XPMATH_INLINE_FUNCTION void sinhcosh(QuadFloatFast a, QuadFloatFast& sinh_a, QuadFloatFast& cosh_a) { QuadFloat t_sinh_a; QuadFloat t_cosh_a; sinhcosh(to_strict(a), t_sinh_a, t_cosh_a); sinh_a = to_fast(t_sinh_a); cosh_a = to_fast(t_cosh_a); }
XPMATH_INLINE_FUNCTION QuadFloatFast angle(QuadFloatFast x, QuadFloatFast y) { return to_fast(angle(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatFast ceil(QuadFloatFast a) { return to_fast(ceil(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast floor(QuadFloatFast a) { return to_fast(floor(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast trunc(QuadFloatFast a) { return to_fast(trunc(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast round(QuadFloatFast a) { return to_fast(round(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast operator+(float a, QuadFloatFast b) { return to_fast(a + to_strict(b)); }
XPMATH_INLINE_FUNCTION QuadFloatFast operator-(float a, QuadFloatFast b) { return to_fast(a - to_strict(b)); }
XPMATH_INLINE_FUNCTION QuadFloatFast operator*(float a, QuadFloatFast b) { return to_fast(a * to_strict(b)); }
XPMATH_INLINE_FUNCTION QuadFloatFast QuadFloatFast_pi() { return to_fast(QuadFloat_pi()); }
XPMATH_INLINE_FUNCTION QuadFloatFast QuadFloatFast_e() { return to_fast(QuadFloat_e()); }
XPMATH_INLINE_FUNCTION QuadFloatFast QuadFloatFast_log2() { return to_fast(QuadFloat_log2()); }
XPMATH_INLINE_FUNCTION QuadFloatFast QuadFloatFast_log10() { return to_fast(QuadFloat_log10()); }
XPMATH_INLINE_FUNCTION QuadFloatFast QuadFloatFast_sqrt2() { return to_fast(QuadFloat_sqrt2()); }
XPMATH_INLINE_FUNCTION QuadFloatFast QuadFloatFast_euler_gamma() { return to_fast(QuadFloat_euler_gamma()); }
XPMATH_INLINE_FUNCTION QuadFloatFast ieee_add(QuadFloatFast a, QuadFloatFast b) { return to_fast(ieee_add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast sloppy_add(QuadFloatFast a, QuadFloatFast b) { return to_fast(sloppy_add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast divide_accurate(QuadFloatFast a, QuadFloatFast b) { return to_fast(divide_accurate(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast log10(QuadFloatFast a) { return to_fast(log10(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast log2(QuadFloatFast a) { return to_fast(log2(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast log1p(QuadFloatFast a) { return to_fast(log1p(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast exp2(QuadFloatFast a) { return to_fast(exp2(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast exp10(QuadFloatFast a) { return to_fast(exp10(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast expm1(QuadFloatFast a) { return to_fast(expm1(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast sin(QuadFloatFast a) { return to_fast(sin(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast cos(QuadFloatFast a) { return to_fast(cos(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast tan(QuadFloatFast a) { return to_fast(tan(to_strict(a))); }
XPMATH_INLINE_FUNCTION int qf_cmp_one(QuadFloatFast a) { return qf_cmp_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION int qf_cmp_abs_one(QuadFloatFast a) { return qf_cmp_abs_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION QuadFloatFast asin(QuadFloatFast a) { return to_fast(asin(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast acos(QuadFloatFast a) { return to_fast(acos(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast atan(QuadFloatFast a) { return to_fast(atan(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast atan2(QuadFloatFast y, QuadFloatFast x) { return to_fast(atan2(to_strict(y), to_strict(x))); }
XPMATH_INLINE_FUNCTION QuadFloatFast sinh(QuadFloatFast a) { return to_fast(sinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast cosh(QuadFloatFast a) { return to_fast(cosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast tanh(QuadFloatFast a) { return to_fast(tanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast asinh(QuadFloatFast a) { return to_fast(asinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast acosh(QuadFloatFast a) { return to_fast(acosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast atanh(QuadFloatFast a) { return to_fast(atanh(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast hypot(QuadFloatFast a, QuadFloatFast b) { return to_fast(hypot(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast fmod(QuadFloatFast a, QuadFloatFast b) { return to_fast(fmod(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast remainder(QuadFloatFast a, QuadFloatFast b) { return to_fast(remainder(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast copysign(QuadFloatFast a, QuadFloatFast b) { return to_fast(copysign(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast fmax(QuadFloatFast a, QuadFloatFast b) { return to_fast(fmax(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast fmin(QuadFloatFast a, QuadFloatFast b) { return to_fast(fmin(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast fdim(QuadFloatFast a, QuadFloatFast b) { return to_fast(fdim(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast fma(QuadFloatFast a, QuadFloatFast b, QuadFloatFast c) { return to_fast(fma(to_strict(a), to_strict(b), to_strict(c))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator+(QuadFloatComplexFast z, QuadFloatFast r) { return to_fast(to_strict(z) + to_strict(r)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator+(QuadFloatFast r, QuadFloatComplexFast z) { return to_fast(to_strict(r) + to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator-(QuadFloatComplexFast z, QuadFloatFast r) { return to_fast(to_strict(z) - to_strict(r)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator-(QuadFloatFast r, QuadFloatComplexFast z) { return to_fast(to_strict(r) - to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator*(QuadFloatComplexFast z, QuadFloatFast r) { return to_fast(to_strict(z) * to_strict(r)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator*(QuadFloatFast r, QuadFloatComplexFast z) { return to_fast(to_strict(r) * to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator/(QuadFloatComplexFast z, QuadFloatFast r) { return to_fast(to_strict(z) / to_strict(r)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator/(QuadFloatFast r, QuadFloatComplexFast z) { return to_fast(to_strict(r) / to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator+(QuadFloatComplexFast z, float b) { return to_fast(to_strict(z) + b); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator+(float b, QuadFloatComplexFast z) { return to_fast(b + to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator-(QuadFloatComplexFast z, float b) { return to_fast(to_strict(z) - b); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator-(float b, QuadFloatComplexFast z) { return to_fast(b - to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator*(QuadFloatComplexFast z, float b) { return to_fast(to_strict(z) * b); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator*(float b, QuadFloatComplexFast z) { return to_fast(b * to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator/(QuadFloatComplexFast z, float b) { return to_fast(to_strict(z) / b); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast operator/(float b, QuadFloatComplexFast z) { return to_fast(b / to_strict(z)); }
XPMATH_INLINE_FUNCTION QuadFloatFast abs(QuadFloatComplexFast z) { return to_fast(abs(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatFast norm(QuadFloatComplexFast z) { return to_fast(norm(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatFast arg(QuadFloatComplexFast z) { return to_fast(arg(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast conj(QuadFloatComplexFast z) { return to_fast(conj(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast sqrt(QuadFloatComplexFast z) { return to_fast(sqrt(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast exp(QuadFloatComplexFast z) { return to_fast(exp(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast log(QuadFloatComplexFast z) { return to_fast(log(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast log1p(QuadFloatComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast log10(QuadFloatComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast sin(QuadFloatComplexFast z) { return to_fast(sin(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast cos(QuadFloatComplexFast z) { return to_fast(cos(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast tan(QuadFloatComplexFast z) { return to_fast(tan(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_asin_imag_mag(QuadFloatFast x, QuadFloatFast y) { return to_fast(xp_asin_imag_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_asin_real_leg(QuadFloatFast x, QuadFloatFast y) { return to_fast(xp_asin_real_leg(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_asin_real_mag(QuadFloatFast x, QuadFloatFast y) { return to_fast(xp_asin_real_mag(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_abs_word(QuadFloatFast v) { return to_fast(xp_abs_word(to_strict(v))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast asin(QuadFloatComplexFast z) { return to_fast(asin(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast acos(QuadFloatComplexFast z) { return to_fast(acos(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_log_hypot2(QuadFloatFast a, QuadFloatFast b) { return to_fast(xp_log_hypot2(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_atan2_safe(QuadFloatFast a, QuadFloatFast b) { return to_fast(xp_atan2_safe(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION bool xp_sq_underflowed(QuadFloatFast v) { return xp_sq_underflowed(to_strict(v)); }
XPMATH_INLINE_FUNCTION bool xp_acosh_chain_short(QuadFloatFast x, QuadFloatFast y) { return xp_acosh_chain_short(to_strict(x), to_strict(y)); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_atan_re_split(QuadFloatFast x, QuadFloatFast y) { return to_fast(xp_atan_re_split(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_atan_im(QuadFloatFast x, QuadFloatFast y) { return to_fast(xp_atan_im(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast atan(QuadFloatComplexFast z) { return to_fast(atan(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast sinh(QuadFloatComplexFast z) { return to_fast(sinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast cosh(QuadFloatComplexFast z) { return to_fast(cosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast tanh(QuadFloatComplexFast z) { return to_fast(tanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatFast xp_ki11_asinh_re(QuadFloatComplexFast z) { return to_fast(xp_ki11_asinh_re(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast asinh(QuadFloatComplexFast z) { return to_fast(asinh(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast acosh(QuadFloatComplexFast z) { return to_fast(acosh(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast atanh(QuadFloatComplexFast z) { return to_fast(atanh(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast pow(QuadFloatComplexFast z, QuadFloatComplexFast w) { return to_fast(pow(to_strict(z), to_strict(w))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast polar(QuadFloatFast r, QuadFloatFast theta) { return to_fast(polar(to_strict(r), to_strict(theta))); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const QuadFloatFast& d) { return os << to_strict(d); }
inline std::ostream& operator<<(std::ostream& os, const QuadFloatComplexFast& z) { return os << to_strict(z); }
#endif

} // namespace xp
