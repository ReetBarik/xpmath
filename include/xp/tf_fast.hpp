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
    TripleFloatFast(const TripleFloatFast&) = default;
    TripleFloatFast& operator=(const TripleFloatFast&) = default;
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
    TripleFloatComplexFast(const TripleFloatComplexFast&) = default;
    TripleFloatComplexFast& operator=(const TripleFloatComplexFast&) = default;
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

namespace detail {
// P3.4[tf] fast multiply, variant B. Same partial
// products and leading accumulation as strict; the tail is
// folded with fma. NOINLINE like strict multiply.
XPMATH_NOINLINE_FUNCTION TripleFloat tf_multiply_fast_impl(TripleFloat a, TripleFloat b) {
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3, q4, q5;
    float t0, t1;
    float s0, s1, s2;

    p0 = tf_two_prod(a.f0, b.f0, q0);

    p1 = tf_two_prod(a.f0, b.f1, q1);
    p2 = tf_two_prod(a.f1, b.f0, q2);

    p3 = tf_two_prod(a.f0, b.f2, q3);
    p4 = tf_two_prod(a.f1, b.f1, q4);
    p5 = tf_two_prod(a.f2, b.f0, q5);

    tf_three_sum(p1, p2, q0);
    float e0, e1, e2, e3;
    s0 = tf_two_sum(p2, q1, e0);
    t0 = tf_two_sum(p3, p4, e1);
    s0 = tf_two_sum(s0, t0, e2);
    t1 = detail::eft_add(q2, p5);
    s0 = tf_two_sum(s0, t1, e3);
    const float g = detail::eft_add(detail::eft_add(e0, e1),
                                    detail::eft_add(e2, e3));
    const float r = detail::eft_add(detail::eft_add(q0, q3),
                                    detail::eft_add(q4, q5));
    const float tail = detail::eft_fma(a.f1, b.f2,
                           detail::eft_fma(a.f2, b.f1, r));
    s1 = detail::eft_add(g, tail);
    renorm_3(p0, p1, s0, s1);
    return TripleFloat(p0, p1, s0);
}
} // namespace detail

// ---- forwarders: one line per public strict function / operator ------------
XPMATH_INLINE_FUNCTION TripleFloatFast add(TripleFloatFast a, TripleFloatFast b) { return to_fast(add(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast subtract(TripleFloatFast a, TripleFloatFast b) { return to_fast(subtract(to_strict(a), to_strict(b))); }
XPMATH_INLINE_FUNCTION TripleFloatFast multiply(TripleFloatFast a, TripleFloatFast b) { return to_fast(detail::tf_multiply_fast_impl(to_strict(a), to_strict(b))); }
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
namespace detail {
// P3.2[tf] fast exp. Strict exp's guards, reduction integer (one
// float multiply; TF already had it), five Cody-Waite pieces, nq = 5
// squarings and 2^m scaling are kept. Changes: r/32 by an exact
// power-of-two scaling (mul_pwr2, not divide_scalar), and e^x - 1
// as a fixed 9-term Horner series with 1/n! constants (strict's
// loop also stops after term 9 at |x| = ln2/64): no divide_scalar,
// no data-dependent exit. Terms 7..9 weigh < 4e-16 of the sum, so
// they run in float; terms 1..6 in TF (each constant three words).
// NaN returns NaN in all three words (strict reaches (int)NaN).
// NOINLINE like strict TF exp (TD-4; P2.2[tf] code-size lesson).
XPMATH_NOINLINE_FUNCTION TripleFloat tf_exp_fast_impl(TripleFloat a) {
    if (a.f0 != a.f0) return TripleFloat(a.f0, a.f0, a.f0);
    if (a.f0 < -104.0f) return TripleFloat(0.0f);
    if (a.f0 >  88.722839f) {
        XPMATH_PRINTF("TFEXP: overflow\n");
        return TripleFloat(HUGE_VALF);
    }
    const float k_inv_log2 = 1.44269504088896341f;
    const float m = detail::floor(detail::eft_add(
        detail::eft_mul(a.f0, k_inv_log2), 0.5f));
    const float kLn2_1 =  0x1.62e4p-1f;
    const float kLn2_2 =  0x1.7f7ep-20f;
    const float kLn2_3 = -0x1.c61p-37f;
    const float kLn2_4 = -0x1.950ep-54f;
    const float kLn2_5 =  0x1.e3b4p-72f;
    TripleFloat r = subtract(a, TripleFloat(m * kLn2_1));
    r = subtract(r, TripleFloat(m * kLn2_2));
    r = subtract(r, TripleFloat(m * kLn2_3));
    r = subtract(r, TripleFloat(m * kLn2_4));
    r = subtract(r, TripleFloat(m * kLn2_5));
    const TripleFloat x = mul_pwr2(r, 0x1p-5f);
    float q = 0x1.71de3ap-19f;                                 // 1/9!
    q = detail::eft_add(0x1.a01a02p-16f, detail::eft_mul(q, x.f0)); // 1/8!
    q = detail::eft_add(0x1.a01a02p-13f, detail::eft_mul(q, x.f0)); // 1/7!
    TripleFloat t = add(TripleFloat(0x1.6c16c2p-10f, -0x1.27d27ep-35f, 0x1.b05b06p-60f),
                        multiply_scalar(x, q));               // 1/6!
    t = add(TripleFloat(0x1.111112p-7f, -0x1.dddddep-32f, 0x1.111112p-59f),
            multiply(t, x));                                  // 1/5!
    t = add(TripleFloat(0x1.555556p-5f, -0x1.555556p-30f, 0x1.555556p-55f),
            multiply(t, x));                                  // 1/4!
    t = add(TripleFloat(0x1.555556p-3f, -0x1.555556p-28f, 0x1.555556p-53f),
            multiply(t, x));                                  // 1/3!
    t = add(TripleFloat(0.5f), multiply(t, x));
    t = add(TripleFloat(1.0f), multiply(t, x));
    TripleFloat s = multiply(t, x);                           // e^x - 1
    for (int i = 0; i < 5; ++i) s = multiply(s, add(s, TripleFloat(2.0f)));
    s = add(TripleFloat(1.0f), s);
    const int mi = (int)m;
    if (mi >= -125 && mi <= 127) return mul_pwr2(s, ldexpf(1.0f, mi));
    return TripleFloat(ldexpf(s.f0, mi), ldexpf(s.f1, mi),
                       ldexpf(s.f2, mi));
}
} // namespace detail
XPMATH_INLINE_FUNCTION TripleFloatFast exp(TripleFloatFast a) { return to_fast(detail::tf_exp_fast_impl(to_strict(a))); }
namespace detail {
// P3.2[tf]-chain: strict TF log (3 Newton steps from a float
// seed, KI-23 small-argument branch); the Newton steps call the
// fast exp. NOINLINE like strict TF log (TD-4; P2.2[tf]
// code-size lesson).
XPMATH_NOINLINE_FUNCTION TripleFloat tf_log_fast_impl(TripleFloat a) {
    if (detail::isinf(a.f0)) return a;
    if (a.f0 <= 0.0f) {
        XPMATH_PRINTF("TFLOG: non-positive argument\n");
        return TripleFloat(0.0f);
    }
    TripleFloat x = TripleFloat(detail::log(a.f0));
    for (int i = 0; i < 3; i++) {
        if (x.f0 < -54.1f && -x.f0 <= 88.722839f) {
            TripleFloat e = tf_exp_fast_impl(negate(x));
            x = subtract(add(x, multiply(a, e)), TripleFloat(1.0f));
        } else {
            TripleFloat e = tf_exp_fast_impl(x);
            x = add(x, divide(subtract(a, e), e));
        }
    }
    return x;
}
} // namespace detail
XPMATH_INLINE_FUNCTION TripleFloatFast log(TripleFloatFast a) { return to_fast(detail::tf_log_fast_impl(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast log1p(TripleFloatFast a) { return to_fast(log1p(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast pow(TripleFloatFast a, TripleFloatFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
namespace detail {
// P3.3[tf] fast sincos. Rare inputs (|a| >= 800, NaN/inf, huge
// cancellation after the cheap reduction) go to the strict sincos
// unchanged, out of line. Out-params are (sin, cos) as in strict TF.
XPMATH_NOINLINE_FUNCTION void tf_sincos_slow(TripleFloat a, TripleFloat& s, TripleFloat& c) { sincos(a, s, c); }
// Same guards, nq = 0 and (sin, 1-cos) form as strict. Changes:
// (1) for 0.75 < |a| < 800 the reduction is k = rint(a.f0 * 2/pi)
// and 7 15-bit pieces of pi/2 plus a float tail, so k*piece is
// one exact float for k <= 512; beyond |a| = 800, or when |r| < 2^-40
// (the pieces no longer cover it), the strict Payne-Hanek path runs;
// (2) the series is a fixed Horner polynomial (sin to r^21, v to r^20)
// with 1/n! constants and no data-dependent exit; the last three terms
// of each weigh < 2^-48 of the sum and run in float.
XPMATH_INLINE_FUNCTION void tf_sincos_fast_impl(TripleFloat a, TripleFloat& sin_a, TripleFloat& cos_a) {
    if (a.f0 == 0.0f) { sin_a = TripleFloat(0.0f); cos_a = TripleFloat(1.0f); return; }
    const float aa = detail::fabs(a.f0);
    if (aa < 1.17549435e-38f) { sin_a = a; cos_a = TripleFloat(1.0f); return; }
    if (!(aa < 800.0f)) { tf_sincos_slow(a, sin_a, cos_a); return; }
    int j = 0;
    TripleFloat r = a;
    if (aa > 0.75f) {
        const float kf = detail::rint(a.f0 * 0x1.45f306p-1f);   // 2/pi
        r = subtract(r, TripleFloat(kf * 0x1.922p+0f));
        r = subtract(r, TripleFloat(kf * -0x1.2afp-18f));
        r = subtract(r, TripleFloat(kf * 0x1.0b48p-34f));
        r = subtract(r, TripleFloat(kf * -0x1.ee58p-50f));
        r = subtract(r, TripleFloat(kf * -0x1.d9ccp-66f));
        r = subtract(r, TripleFloat(kf * -0x1.d748p-83f));
        r = subtract(r, TripleFloat(kf * 0x1.b838p-104f));
        r = subtract(r, TripleFloat(kf * 0x1.a25204p-120f));
        if (!(detail::fabs(r.f0) >= 0x1p-40f)) { tf_sincos_slow(a, sin_a, cos_a); return; }
        j = (int)((long long)kf & 3);
    }
    const TripleFloat yy = multiply(r, r);
    float qs = 0x1.71b8fp-66f;                                           // 1/21!
    qs = detail::eft_add(-0x1.2f49b4p-57f, detail::eft_mul(qs, yy.f0));    // -1/19!
    qs = detail::eft_add(0x1.952c78p-49f, detail::eft_mul(qs, yy.f0));    // 1/17!
    TripleFloat ts = add(TripleFloat(-0x1.ae7f3ep-41f, -0x1.ccee08p-67f, 0x1.dc4f36p-94f), multiply_scalar(yy, qs));   // -1/15!
    ts = add(TripleFloat(0x1.612462p-33f, -0x1.8af25ep-58f, -0x1.a0d72p-83f), multiply(ts, yy));   // 1/13!
    ts = add(TripleFloat(-0x1.ae6456p-26f, -0x1.fd5138p-52f, -0x1.c7f3a4p-77f), multiply(ts, yy));   // -1/11!
    ts = add(TripleFloat(0x1.71de3ap-19f, 0x1.55b1ccp-45f, 0x1.c7d56p-70f), multiply(ts, yy));   // 1/9!
    ts = add(TripleFloat(-0x1.a01a02p-13f, 0x1.7f97fap-39f, -0x1.00d00ep-64f), multiply(ts, yy));   // -1/7!
    ts = add(TripleFloat(0x1.111112p-7f, -0x1.dddddep-32f, 0x1.111112p-59f), multiply(ts, yy));   // 1/5!
    ts = add(TripleFloat(-0x1.555556p-3f, 0x1.555556p-28f, -0x1.555556p-53f), multiply(ts, yy));   // -1/3!
    TripleFloat sin_r = add(r, multiply(r, multiply(ts, yy)));
    float qv = -0x1.e542bap-62f;                                           // -1/20!
    qv = detail::eft_add(0x1.682786p-53f, detail::eft_mul(qv, yy.f0));    // 1/18!
    qv = detail::eft_add(-0x1.ae7f3ep-45f, detail::eft_mul(qv, yy.f0));    // -1/16!
    TripleFloat tv = add(TripleFloat(0x1.93974ap-37f, 0x1.180f94p-62f, -0x1.6fa29p-88f), multiply_scalar(yy, qv));   // 1/14!
    tv = add(TripleFloat(-0x1.1eed8ep-29f, -0x1.ff1b12p-54f, -0x1.ed5136p-79f), multiply(tv, yy));   // -1/12!
    tv = add(TripleFloat(0x1.27e4fcp-22f, -0x1.10ec14p-47f, -0x1.c6888p-73f), multiply(tv, yy));   // 1/10!
    tv = add(TripleFloat(-0x1.a01a02p-16f, 0x1.7f97fap-42f, -0x1.00d00ep-67f), multiply(tv, yy));   // -1/8!
    tv = add(TripleFloat(0x1.6c16c2p-10f, -0x1.27d27ep-35f, 0x1.b05b06p-60f), multiply(tv, yy));   // 1/6!
    tv = add(TripleFloat(-0x1.555556p-5f, 0x1.555556p-30f, -0x1.555556p-55f), multiply(tv, yy));   // -1/4!
    tv = add(TripleFloat(0.5f), multiply(tv, yy));
    const TripleFloat v_r = multiply(tv, yy);
    TripleFloat cos_r = subtract(TripleFloat(1.0f), v_r);
    // NaN/inf in a word: strict answers with its identity-point
    // fallback; let strict do it.
    if (!(detail::fabs(sin_r.f0) <= 1.0009765625f) || !(detail::fabs(cos_r.f0) <= 1.0009765625f)) { tf_sincos_slow(a, sin_a, cos_a); return; }
    if (sin_r.f0 >  1.0f || (sin_r.f0 ==  1.0f && sin_r.f1 > 0.0f)) sin_r = TripleFloat( 1.0f);
    if (sin_r.f0 < -1.0f || (sin_r.f0 == -1.0f && sin_r.f1 < 0.0f)) sin_r = TripleFloat(-1.0f);
    if (cos_r.f0 >  1.0f || (cos_r.f0 ==  1.0f && cos_r.f1 > 0.0f)) cos_r = TripleFloat( 1.0f);
    if (cos_r.f0 < -1.0f || (cos_r.f0 == -1.0f && cos_r.f1 < 0.0f)) cos_r = TripleFloat(-1.0f);
    if (j == 0)      { sin_a = sin_r;         cos_a = cos_r; }
    else if (j == 1) { sin_a = cos_r;         cos_a = negate(sin_r); }
    else if (j == 2) { sin_a = negate(sin_r); cos_a = negate(cos_r); }
    else             { sin_a = negate(cos_r); cos_a = sin_r; }
}
} // namespace detail
XPMATH_INLINE_FUNCTION void sincos(TripleFloatFast a, TripleFloatFast& sin_a, TripleFloatFast& cos_a) { TripleFloat t_sin_a; TripleFloat t_cos_a; detail::tf_sincos_fast_impl(to_strict(a), t_sin_a, t_cos_a); sin_a = to_fast(t_sin_a); cos_a = to_fast(t_cos_a); }
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
XPMATH_NOINLINE_FUNCTION TripleFloatFast sin(TripleFloatFast a) { TripleFloat s, c; detail::tf_sincos_fast_impl(to_strict(a), s, c); return to_fast(s); }
XPMATH_NOINLINE_FUNCTION TripleFloatFast cos(TripleFloatFast a) { TripleFloat s, c; detail::tf_sincos_fast_impl(to_strict(a), s, c); return to_fast(c); }
XPMATH_INLINE_FUNCTION TripleFloatFast tan(TripleFloatFast a) { TripleFloat s, c; detail::tf_sincos_fast_impl(to_strict(a), s, c); return to_fast(divide(s, c)); }
XPMATH_INLINE_FUNCTION TripleFloatFast sinh(TripleFloatFast a) { return to_fast(sinh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast cosh(TripleFloatFast a) { return to_fast(cosh(to_strict(a))); }
XPMATH_INLINE_FUNCTION TripleFloatFast tanh(TripleFloatFast a) { return to_fast(tanh(to_strict(a))); }
namespace detail {
// P3.3[tf] callers of sincos wired to the fast sincos (atan, atan2,
// real tan, complex exp/sin/cos/tan). Each body is the strict body
// (tf_math.hpp / tf_complex.hpp) with sincos -> tf_sincos_fast_impl;
// complex exp and tan also use the fast real exp. Strict is unchanged.
XPMATH_INLINE_FUNCTION TripleFloat tf_angle_fast_body(TripleFloat x, TripleFloat y) {
    const TripleFloat pi   = TripleFloat_pi();
    const TripleFloat pi_2 = mul_pwr2(pi, 0.5f);
    const TripleFloat pi_4 = mul_pwr2(pi, 0.25f);
    if (y.f0 == 0.0f) {
        if (x.f0 == 0.0f) {
            XPMATH_PRINTF("TFATAN2: both arguments zero\n");
        }
        const TripleFloat r =
            (detail::copysign(1.0f, x.f0) < 0.0f) ? pi : TripleFloat(0.0f);
        return (detail::copysign(1.0f, y.f0) < 0.0f) ? negate(r) : r;
    }
    if (x.f0 == 0.0f) {
        return (y.f0 > 0.0f) ? pi_2 : negate(pi_2);
    }
    if (x == y || x == negate(y)) {
        const TripleFloat pi_34 = multiply_scalar(pi, 0.75f);
        if (x == y) return (y.f0 > 0.0f) ? pi_4 : negate(pi_34);
        return (y.f0 > 0.0f) ? pi_34 : negate(pi_4);
    }
    {
        const float mx = detail::fabs(x.f0), my = detail::fabs(y.f0);
        const float mm = (mx > my) ? mx : my;
        if (mm > detail::kTFSqHi) {
            const float s = detail::tf_pow2_unit_scale(mm);
            x = detail::tf_pow2_scale(x, s);
            y = detail::tf_pow2_scale(y, s);
        }
    }
    TripleFloat r  = sqrt(add(sqr(x), sqr(y)));
    TripleFloat xx = divide(x, r);
    TripleFloat yy = divide(y, r);
    TripleFloat z = TripleFloat(detail::atan2(y.f0, x.f0));
    TripleFloat sin_z, cos_z;
    if (detail::fabs(xx.f0) > detail::fabs(yy.f0)) {
        for (int k = 0; k < 2; ++k) {
            tf_sincos_fast_impl(z, sin_z, cos_z);
            z = add(z, divide(subtract(yy, sin_z), cos_z));
        }
    } else {
        for (int k = 0; k < 2; ++k) {
            tf_sincos_fast_impl(z, sin_z, cos_z);
            z = subtract(z, divide(subtract(xx, cos_z), sin_z));
        }
    }
    return z;
}
// Out of line, like strict atan2.
XPMATH_NOINLINE_FUNCTION TripleFloat tf_angle_fast(TripleFloat x, TripleFloat y) { return tf_angle_fast_body(x, y); }
XPMATH_INLINE_FUNCTION TripleFloat tf_atan_fast_impl(TripleFloat a) {
    TripleFloat r = tf_angle_fast(TripleFloat(1.0f), a);
    const TripleFloat p = TripleFloat_pi();
    const TripleFloat h(p.f0 * 0.5f, p.f1 * 0.5f, p.f2 * 0.5f);
    const TripleFloat ar = (r.f0 < 0.0f) ? TripleFloat(-r.f0, -r.f1, -r.f2) : r;
    if (subtract(ar, h).f0 > 0.0f)
        return (r.f0 < 0.0f) ? TripleFloat(-h.f0, -h.f1, -h.f2) : h;
    return r;
}
XPMATH_INLINE_FUNCTION TripleFloatComplex tf_cexp_fast_impl(TripleFloatComplex z) {
    TripleFloat er = tf_exp_fast_impl(z.re);
    TripleFloat s, c;
    tf_sincos_fast_impl(z.im, s, c);
    return TripleFloatComplex(multiply(er, c), multiply(er, s));
}
XPMATH_INLINE_FUNCTION TripleFloatComplex tf_csin_fast_impl(TripleFloatComplex z) {
    TripleFloat ca, sa, cb, sb;
    tf_sincos_fast_impl(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return TripleFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION TripleFloatComplex tf_ccos_fast_impl(TripleFloatComplex z) {
    TripleFloat ca, sa, cb, sb;
    tf_sincos_fast_impl(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return TripleFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
XPMATH_INLINE_FUNCTION TripleFloatComplex tf_ctan_fast_impl(TripleFloatComplex z) {
    const float kXpTanAsymptote = 2.0f;
    if (detail::fabs(z.im.f0) >= kXpTanAsymptote) {
        TripleFloat ca, sa;
        tf_sincos_fast_impl(z.re, sa, ca);
        const TripleFloat s2 = multiply_scalar(multiply(sa, ca), 2.0f);
        const TripleFloat c2 = multiply(subtract(ca, sa), add(ca, sa));
        const TripleFloat t  = tf_exp_fast_impl(multiply_scalar(z.im, z.im.f0 < 0.0f ? 2.0f : -2.0f));
        const TripleFloat t2 = multiply(t, t);
        const TripleFloat den = add(add(TripleFloat(1.0f), t2), multiply_scalar(multiply(t, c2), 2.0f));
        TripleFloat im = divide(subtract(TripleFloat(1.0f), t2), den);
        if (z.im.f0 < 0.0f) im = negate(im);
        return TripleFloatComplex(divide(multiply_scalar(multiply(t, s2), 2.0f), den), im);
    }
    return tf_csin_fast_impl(z) / tf_ccos_fast_impl(z);
}
} // namespace detail
XPMATH_NOINLINE_FUNCTION TripleFloatFast atan2(TripleFloatFast y, TripleFloatFast x) { return to_fast(detail::tf_angle_fast(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION TripleFloatFast atan(TripleFloatFast a) { return to_fast(detail::tf_atan_fast_impl(to_strict(a))); }
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
XPMATH_NOINLINE_FUNCTION TripleFloatComplexFast exp(TripleFloatComplexFast z) { return to_fast(detail::tf_cexp_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast log(TripleFloatComplexFast z) { return to_fast(log(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast log1p(TripleFloatComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast log10(TripleFloatComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast sin(TripleFloatComplexFast z) { return to_fast(detail::tf_csin_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast cos(TripleFloatComplexFast z) { return to_fast(detail::tf_ccos_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION TripleFloatComplexFast tan(TripleFloatComplexFast z) { return to_fast(detail::tf_ctan_fast_impl(to_strict(z))); }
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
