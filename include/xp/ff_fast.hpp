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
    FloatFloatFast(const FloatFloatFast&) = default;
    FloatFloatFast& operator=(const FloatFloatFast&) = default;
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
    FloatFloatComplexFast(const FloatFloatComplexFast&) = default;
    FloatFloatComplexFast& operator=(const FloatFloatComplexFast&) = default;
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
namespace detail {
// P3.2[ff] fast exp. Same guards, four Cody-Waite pieces, nq = 4
// squarings and 2^nz scaling as strict exp; two changes: the
// reduction integer is rint(a.hi * 1/ln2) (one native float multiply,
// not divide(a, ln2)), and e^x - 1 is a fixed 8-term Horner series
// with 1/n! constants (no divide_scalar, no data-dependent exit).
// Terms 5..8 weigh < 2e-9 of the sum at |x| <= ln2/32, so they run
// in float; terms 1..4 in FF.
XPMATH_INLINE_FUNCTION FloatFloat ff_exp_fast_impl(FloatFloat a) {
    if (a.hi != a.hi) return a;
    if (a.hi > 88.722839f) {
        XPMATH_PRINTF("FFEXP: overflow\n");
        return FloatFloat(HUGE_VALF);
    }
    if (a.hi < -104.0f) return FloatFloat(0.0f);
    const float kf = detail::rint(detail::eft_mul(a.hi, 0x1.715476p+0f));
    const int nz = (int)kf;
    const float kLn2_1 =  0x1.62e4p-1f;
    const float kLn2_2 =  0x1.7f7ep-20f;
    const float kLn2_3 = -0x1.c61p-37f;
    const float kLn2_4 = -0x1.950ep-54f;
    FloatFloat r = subtract(a, FloatFloat(kf * kLn2_1));
    r = subtract(r, FloatFloat(kf * kLn2_2));
    r = subtract(r, FloatFloat(kf * kLn2_3));
    r = subtract(r, FloatFloat(kf * kLn2_4));
    if (r.hi == 0.0f) return FloatFloat(ldexpf(1.0f, nz));
    const FloatFloat x = multiply_scalar(r, 0x1p-4f);
    float q = 0x1.a01a02p-16f;                                   // 1/8!
    q = detail::eft_add(0x1.a01a02p-13f, detail::eft_mul(q, x.hi)); // 1/7!
    q = detail::eft_add(0x1.6c16c2p-10f, detail::eft_mul(q, x.hi)); // 1/6!
    q = detail::eft_add(0x1.111112p-7f, detail::eft_mul(q, x.hi));  // 1/5!
    FloatFloat t = add(FloatFloat(0x1.555556p-5f, -0x1.555556p-30f), multiply(FloatFloat(q), x)); // 1/4!
    t = add(FloatFloat(0x1.555556p-3f, -0x1.555556p-28f), multiply(t, x)); // 1/3!
    t = add(FloatFloat(0.5f), multiply(t, x));
    t = add(FloatFloat(1.0f), multiply(t, x));
    FloatFloat s = multiply(t, x);                               // e^x - 1
    for (int i = 0; i < 4; ++i) s = multiply(s, add(s, FloatFloat(2.0f)));
    s = add(FloatFloat(1.0f), s);
    if (nz >= -125 && nz <= 127) {
        const float pow2 = ldexpf(1.0f, nz);
        return FloatFloat(s.hi * pow2, s.lo * pow2);
    }
    return FloatFloat(ldexpf(s.hi, nz), ldexpf(s.lo, nz));
}
} // namespace detail
XPMATH_INLINE_FUNCTION FloatFloatFast exp(FloatFloatFast a) { return to_fast(detail::ff_exp_fast_impl(to_strict(a))); }
namespace detail {
// P3.2[ff]-chain: strict FF log (2 Newton steps from a float
// seed, KI-23 small-argument branch); the Newton steps call the
// fast exp.
XPMATH_INLINE_FUNCTION FloatFloat ff_log_fast_impl(FloatFloat a) {
    if (detail::isinf(a.hi)) return a;
    if (a.hi <= 0.0f) {
        XPMATH_PRINTF("FFLOG: non-positive argument\n");
        return FloatFloat(0.0f);
    }
    FloatFloat b = FloatFloat(detail::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        if (b.hi < -70.7f && -b.hi <= 88.722839f) {
            FloatFloat s0 = ff_exp_fast_impl(negate(b));
            b = subtract(add(b, multiply(a, s0)), FloatFloat(1.0f));
        } else {
            FloatFloat s0 = ff_exp_fast_impl(b);
            FloatFloat s1 = subtract(a, s0);
            FloatFloat s2 = divide(s1, s0);
            b = add(b, s2);
        }
    }
    return b;
}
} // namespace detail
XPMATH_INLINE_FUNCTION FloatFloatFast log(FloatFloatFast a) { return to_fast(detail::ff_log_fast_impl(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast pow(FloatFloatFast a, FloatFloatFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_mul_ext(FloatFloatFast x, FloatFloatFast y, float& err) { return to_fast(ff_mul_ext(to_strict(x), to_strict(y), err)); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_log_ext(FloatFloatFast a, float& err) { return to_fast(ff_log_ext(to_strict(a), err)); }
XPMATH_INLINE_FUNCTION FloatFloatFast ff_exp_ext(FloatFloatFast a, float resid) { return to_fast(ff_exp_ext(to_strict(a), resid)); }
XPMATH_INLINE_FUNCTION void sinhcosh(FloatFloatFast a, FloatFloatFast& x, FloatFloatFast& y) { FloatFloat t_x; FloatFloat t_y; sinhcosh(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
namespace detail {
// P3.3[ff] fast sincos. Rare inputs (|a| >= 800, NaN/inf, huge
// cancellation after the cheap reduction) go to the strict sincos
// unchanged, out of line.
XPMATH_NOINLINE_FUNCTION void ff_sincos_slow(FloatFloat a, FloatFloat& x, FloatFloat& y) { sincos(a, x, y); }
// Same guards, nq = 0 and (sin, 1-cos) form as strict. Changes:
// (1) for 0.75 < |a| < 800 the reduction is k = rint(a.hi * 2/pi)
// and seven 15-bit pieces of pi/2 plus a float tail, so k*piece is
// one exact float for k <= 512; beyond |a| = 800, or when |r| < 2^-40 (the pieces no longer
// cover it), the strict Payne-Hanek path runs; (2) the series is a fixed Horner polynomial (sin to
// r^15, v to r^16) with 1/n! constants and no data-dependent exit;
// the last three terms of each weigh < 2^-26 of the sum and run in
// float.
XPMATH_INLINE_FUNCTION void ff_sincos_fast_impl(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    if (a.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    const float aa = detail::fabs(a.hi);
    if (aa < 1.17549435e-38f) { x = FloatFloat(1.0f); y = a; return; }
    if (!(aa < 800.0f)) { ff_sincos_slow(a, x, y); return; }
    int j = 0;
    FloatFloat r = a;
    if (aa > 0.75f) {
        const float kf = detail::rint(a.hi * 0x1.45f306p-1f);   // 2/pi
        r = subtract(r, FloatFloat(kf * 0x1.922p+0f));
        r = subtract(r, FloatFloat(kf * -0x1.2afp-18f));
        r = subtract(r, FloatFloat(kf * 0x1.0b48p-34f));
        r = subtract(r, FloatFloat(kf * -0x1.ee58p-50f));
        r = subtract(r, FloatFloat(kf * -0x1.d9ccp-66f));
        r = subtract(r, FloatFloat(kf * -0x1.d748p-83f));
        r = subtract(r, FloatFloat(kf * 0x1.b838p-104f));
        r = subtract(r, FloatFloat(kf * 0x1.a25204p-120f));
        if (!(detail::fabs(r.hi) >= 0x1p-40f)) { ff_sincos_slow(a, x, y); return; }
        j = (int)((long long)kf & 3);
    }
    const FloatFloat yy = multiply(r, r);
    float qs = -0x1.ae7f3ep-41f;                                           // -1/15!
    qs = detail::eft_add(0x1.612462p-33f, detail::eft_mul(qs, yy.hi));    // 1/13!
    qs = detail::eft_add(-0x1.ae6456p-26f, detail::eft_mul(qs, yy.hi));    // -1/11!
    FloatFloat ts = add(FloatFloat(0x1.71de3ap-19f, 0x1.55b1ccp-45f), multiply(FloatFloat(qs), yy));   // 1/9!
    ts = add(FloatFloat(-0x1.a01a02p-13f, 0x1.7f97fap-39f), multiply(ts, yy));   // -1/7!
    ts = add(FloatFloat(0x1.111112p-7f, -0x1.dddddep-32f), multiply(ts, yy));   // 1/5!
    ts = add(FloatFloat(-0x1.555556p-3f, 0x1.555556p-28f), multiply(ts, yy));   // -1/3!
    FloatFloat sin_r = add(r, multiply(r, multiply(ts, yy)));
    float qv = -0x1.ae7f3ep-45f;                                           // -1/16!
    qv = detail::eft_add(0x1.93974ap-37f, detail::eft_mul(qv, yy.hi));    // 1/14!
    qv = detail::eft_add(-0x1.1eed8ep-29f, detail::eft_mul(qv, yy.hi));    // -1/12!
    FloatFloat tv = add(FloatFloat(0x1.27e4fcp-22f, -0x1.10ec14p-47f), multiply(FloatFloat(qv), yy));   // 1/10!
    tv = add(FloatFloat(-0x1.a01a02p-16f, 0x1.7f97fap-42f), multiply(tv, yy));   // -1/8!
    tv = add(FloatFloat(0x1.6c16c2p-10f, -0x1.27d27ep-35f), multiply(tv, yy));   // 1/6!
    tv = add(FloatFloat(-0x1.555556p-5f, 0x1.555556p-30f), multiply(tv, yy));   // -1/4!
    tv = add(FloatFloat(0.5f), multiply(tv, yy));
    const FloatFloat v_r = multiply(tv, yy);
    FloatFloat cos_r = subtract(FloatFloat(1.0f), v_r);
    // NaN/inf in a word: strict answers with its identity-point
    // fallback; let strict do it.
    if (!(detail::fabs(sin_r.hi) <= 1.0009765625f) || !(detail::fabs(cos_r.hi) <= 1.0009765625f)) { ff_sincos_slow(a, x, y); return; }
    if (sin_r.hi >  1.0f || (sin_r.hi ==  1.0f && sin_r.lo > 0.0f)) sin_r = FloatFloat( 1.0f);
    if (sin_r.hi < -1.0f || (sin_r.hi == -1.0f && sin_r.lo < 0.0f)) sin_r = FloatFloat(-1.0f);
    if (cos_r.hi >  1.0f || (cos_r.hi ==  1.0f && cos_r.lo > 0.0f)) cos_r = FloatFloat( 1.0f);
    if (cos_r.hi < -1.0f || (cos_r.hi == -1.0f && cos_r.lo < 0.0f)) cos_r = FloatFloat(-1.0f);
    if (j == 0)      { x = cos_r;         y = sin_r; }
    else if (j == 1) { x = negate(sin_r); y = cos_r; }
    else if (j == 2) { x = negate(cos_r); y = negate(sin_r); }
    else             { x = sin_r;         y = negate(cos_r); }
}
} // namespace detail
XPMATH_INLINE_FUNCTION void sincos(FloatFloatFast a, FloatFloatFast& x, FloatFloatFast& y) { FloatFloat t_x; FloatFloat t_y; detail::ff_sincos_fast_impl(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
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
XPMATH_NOINLINE_FUNCTION FloatFloatFast sin(FloatFloatFast a) { FloatFloat c, s; detail::ff_sincos_fast_impl(to_strict(a), c, s); return to_fast(s); }
XPMATH_NOINLINE_FUNCTION FloatFloatFast cos(FloatFloatFast a) { FloatFloat c, s; detail::ff_sincos_fast_impl(to_strict(a), c, s); return to_fast(c); }
XPMATH_INLINE_FUNCTION FloatFloatFast tan(FloatFloatFast a) { FloatFloat c, s; detail::ff_sincos_fast_impl(to_strict(a), c, s); return to_fast(divide(s, c)); }
XPMATH_INLINE_FUNCTION int ff_cmp_one(FloatFloatFast a) { return ff_cmp_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION int ff_cmp_abs_one(FloatFloatFast a) { return ff_cmp_abs_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION FloatFloatFast asin(FloatFloatFast a) { return to_fast(asin(to_strict(a))); }
XPMATH_INLINE_FUNCTION FloatFloatFast acos(FloatFloatFast a) { return to_fast(acos(to_strict(a))); }
namespace detail {
// P3.3[ff] callers of sincos wired to the fast sincos (atan, atan2,
// real tan, complex exp/sin/cos/tan). Each body is the strict body
// (ff_math.hpp / ff_complex.hpp) with sincos -> ff_sincos_fast_impl;
// complex exp and tan also use the fast real exp. Strict is unchanged.
// angle: strict angle() body (comments dropped), 3 Newton steps.
XPMATH_INLINE_FUNCTION FloatFloat ff_angle_fast_body(FloatFloat x, FloatFloat y) {
    FloatFloat pi = FloatFloat_pi();
    if (y.hi == 0.0f) {
        const FloatFloat r =
            (detail::copysign(1.0f, x.hi) < 0.0f) ? pi : FloatFloat(0.0f);
        return (detail::copysign(1.0f, y.hi) < 0.0f) ? negate(r) : r;
    }
    if (x.hi == 0.0f) return (y.hi > 0.0f) ? multiply_scalar(pi, 0.5f) : multiply_scalar(pi, -0.5f);
    {
        const float mx = detail::fabs(x.hi), my = detail::fabs(y.hi);
        const float mm = (mx > my) ? mx : my;
        if (mm > detail::kFFSqHi) {
            const float s = detail::ff_pow2_unit_scale(mm);
            x = detail::ff_pow2_scale(x, s);
            y = detail::ff_pow2_scale(y, s);
        }
    }
    FloatFloat r = sqrt(add(multiply(x,x), multiply(y,y)));
    FloatFloat nx = divide(x, r), ny = divide(y, r);
    FloatFloat a = FloatFloat(detail::atan2(ny.hi, nx.hi));
    bool use_x = (detail::fabs(nx.hi) <= detail::fabs(ny.hi));
    FloatFloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        FloatFloat sin_a, cos_a;
        ff_sincos_fast_impl(a, cos_a, sin_a);
        FloatFloat corr;
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
// Out of line, like strict atan2.
XPMATH_NOINLINE_FUNCTION FloatFloat ff_angle_fast(FloatFloat x, FloatFloat y) { return ff_angle_fast_body(x, y); }
XPMATH_INLINE_FUNCTION FloatFloat ff_atan_fast_impl(FloatFloat a) {
    FloatFloat r = ff_angle_fast(FloatFloat(1.0f), a);
    const FloatFloat p = FloatFloat_pi();
    const FloatFloat h(p.hi * 0.5f, p.lo * 0.5f);
    const FloatFloat ar = (r.hi < 0.0f) ? FloatFloat(-r.hi, -r.lo) : r;
    if (subtract(ar, h).hi > 0.0f)
        return (r.hi < 0.0f) ? FloatFloat(-h.hi, -h.lo) : h;
    return r;
}
XPMATH_INLINE_FUNCTION FloatFloatComplex ff_cexp_fast_impl(FloatFloatComplex z) {
    FloatFloat er = ff_exp_fast_impl(z.re);
    FloatFloat c, s;
    ff_sincos_fast_impl(z.im, c, s);
    return FloatFloatComplex(multiply(er, c), multiply(er, s));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex ff_csin_fast_impl(FloatFloatComplex z) {
    FloatFloat ca, sa, cb, sb;
    ff_sincos_fast_impl(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return FloatFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex ff_ccos_fast_impl(FloatFloatComplex z) {
    FloatFloat ca, sa, cb, sb;
    ff_sincos_fast_impl(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return FloatFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
XPMATH_INLINE_FUNCTION FloatFloatComplex ff_ctan_fast_impl(FloatFloatComplex z) {
    const float kXpTanAsymptote = 4.0f;
    if (detail::fabs(z.im.hi) >= kXpTanAsymptote) {
        FloatFloat ca, sa;
        ff_sincos_fast_impl(z.re, ca, sa);
        const FloatFloat s2 = multiply_scalar(multiply(sa, ca), 2.0f);
        const FloatFloat c2 = multiply(subtract(ca, sa), add(ca, sa));
        const FloatFloat t  = ff_exp_fast_impl(multiply_scalar(z.im, z.im.hi < 0.0f ? 2.0f : -2.0f));
        const FloatFloat t2 = multiply(t, t);
        const FloatFloat den = add(add(FloatFloat(1.0f), t2), multiply_scalar(multiply(t, c2), 2.0f));
        FloatFloat im = divide(subtract(FloatFloat(1.0f), t2), den);
        if (z.im.hi < 0.0f) im = negate(im);
        return FloatFloatComplex(divide(multiply_scalar(multiply(t, s2), 2.0f), den), im);
    }
    return ff_csin_fast_impl(z) / ff_ccos_fast_impl(z);
}
} // namespace detail
XPMATH_INLINE_FUNCTION FloatFloatFast atan(FloatFloatFast a) { return to_fast(detail::ff_atan_fast_impl(to_strict(a))); }
XPMATH_NOINLINE_FUNCTION FloatFloatFast atan2(FloatFloatFast y, FloatFloatFast x) { return to_fast(detail::ff_angle_fast(to_strict(x), to_strict(y))); }
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
XPMATH_NOINLINE_FUNCTION FloatFloatComplexFast exp(FloatFloatComplexFast z) { return to_fast(detail::ff_cexp_fast_impl(to_strict(z))); }
// P3.2[ff]-chain: strict complex log body, fast real log.
// Not inline (strict is inline, but its real log is a call;
// this body inlines the fast real log and its fast exps). TD-4.
XPMATH_NOINLINE_FUNCTION FloatFloatComplexFast log(FloatFloatComplexFast z) {
    const FloatFloatComplex w = to_strict(z);
    const FloatFloat modulus = abs(w);
    const FloatFloat arg = atan2(w.im, w.re);
    return to_fast(FloatFloatComplex(detail::ff_log_fast_impl(modulus), arg));
}
XPMATH_INLINE_FUNCTION FloatFloatComplexFast log1p(FloatFloatComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast log10(FloatFloatComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast sin(FloatFloatComplexFast z) { return to_fast(detail::ff_csin_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast cos(FloatFloatComplexFast z) { return to_fast(detail::ff_ccos_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION FloatFloatComplexFast tan(FloatFloatComplexFast z) { return to_fast(detail::ff_ctan_fast_impl(to_strict(z))); }
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
