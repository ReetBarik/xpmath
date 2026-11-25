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
    QuadFloatFast(const QuadFloatFast&) = default;
    QuadFloatFast& operator=(const QuadFloatFast&) = default;
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
    QuadFloatComplexFast(const QuadFloatComplexFast&) = default;
    QuadFloatComplexFast& operator=(const QuadFloatComplexFast&) = default;
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
namespace detail {
// P3.2[qf] fast exp series: e^x - 1 for |x| <= ln2/128 as a fixed
// 11-term Horner series with 1/n! constants (strict's loop with
// divide_scalar and a data-dependent exit is replaced). Terms 9..11
// weigh < 3e-24 of the sum, so they run in float; terms 1..8 in QF
// (each constant four words). 10 terms was tested and fails (up to
// 5.3 ulps). NOINLINE like strict's detail::qf_exp_taylor (frame).
XPMATH_NOINLINE_FUNCTION QuadFloat qf_exp_fast_series(QuadFloat x) {
    float q = 0x1.ae6456p-26f;                          // 1/11!
    q = detail::eft_add(0x1.27e4fcp-22f,
                        detail::eft_mul(q, x.f0));      // 1/10!
    q = detail::eft_add(0x1.71de3ap-19f,
                        detail::eft_mul(q, x.f0));      // 1/9!
    QuadFloat t = add(QuadFloat(0x1.a01a02p-16f, -0x1.7f97fap-42f,
                  0x1.00d00ep-67f, -0x1.fe5fe6p-92f),
              multiply_scalar(x, q));  // 1/8!
    t = add(QuadFloat(0x1.a01a02p-13f, -0x1.7f97fap-39f,
                  0x1.00d00ep-64f, -0x1.fe5fe6p-89f),
              multiply(t, x));  // 1/7!
    t = add(QuadFloat(0x1.6c16c2p-10f, -0x1.27d27ep-35f,
                  0x1.b05b06p-60f, -0x1.3e93eap-86f),
              multiply(t, x));  // 1/6!
    t = add(QuadFloat(0x1.111112p-7f, -0x1.dddddep-32f,
                  0x1.111112p-59f, -0x1.dddddep-84f),
              multiply(t, x));  // 1/5!
    t = add(QuadFloat(0x1.555556p-5f, -0x1.555556p-30f,
                  0x1.555556p-55f, -0x1.555556p-80f),
              multiply(t, x));  // 1/4!
    t = add(QuadFloat(0x1.555556p-3f, -0x1.555556p-28f,
                  0x1.555556p-53f, -0x1.555556p-78f),
              multiply(t, x));  // 1/3!
    t = add(QuadFloat(0.5f), multiply(t, x));
    t = add(QuadFloat(1.0f), multiply(t, x));
    return multiply(t, x);                              // e^x - 1
}
// P3.2[qf] fast exp. Strict exp's guards (KI-6), reduction integer
// (native multiply: floor(a.f0 / ln2 + 0.5), no QF divide), six-piece
// Cody-Waite helper, nq = 6
// squarings helper and 2^nz scaling are kept; only the e^x - 1
// series changes (qf_exp_fast_series). NaN returns NaN in all four
// words (strict reaches (int)NaN). NOINLINE like strict QF exp.
XPMATH_NOINLINE_FUNCTION QuadFloat qf_exp_fast_impl(QuadFloat a) {
    if (a.f0 != a.f0) return QuadFloat(a.f0, a.f0, a.f0, a.f0);
    if (a.f0 > 88.722839f) {
        XPMATH_PRINTF("QFEXP: overflow\n");
        return QuadFloat(HUGE_VALF);
    }
    if (a.f0 < -104.0f) return QuadFloat(0.0f);
    const float k_inv_log2 = 1.44269504088896341f;
    const int nz = (int)detail::floor(detail::eft_add(
        detail::eft_mul(a.f0, k_inv_log2), 0.5f));
    const float kf = (float)nz;
    const QuadFloat s0 = detail::qf_exp_cw_reduce(a, kf);
    if (s0.f0 == 0.0f && s0.f1 == 0.0f) {
        return QuadFloat(ldexpf(1.0f, nz));
    }
    const QuadFloat r = mul_pwr2(s0, ldexpf(1.0f, -6));
    QuadFloat s3 = detail::qf_exp_fast_series(r);   // e^r - 1
    s3 = detail::qf_exp_squarings(s3);
    s3 = add(QuadFloat(1.0f), s3);
    if (nz >= -125 && nz <= 127) {
        const float pow2 = ldexpf(1.0f, nz);
        return QuadFloat(s3.f0 * pow2, s3.f1 * pow2,
                         s3.f2 * pow2, s3.f3 * pow2);
    }
    return QuadFloat(ldexpf(s3.f0, nz), ldexpf(s3.f1, nz),
                     ldexpf(s3.f2, nz), ldexpf(s3.f3, nz));
}
}  // namespace detail
XPMATH_INLINE_FUNCTION QuadFloatFast exp(QuadFloatFast a) {
    return to_fast(detail::qf_exp_fast_impl(to_strict(a)));
}
// P3.2[qf] chain: exp2 and exp10 are strict's bodies with the
// inner exp replaced by detail::qf_exp_fast_impl. pow stays strict.
namespace detail {
XPMATH_INLINE_FUNCTION QuadFloat qf_exp2_fast_impl(QuadFloat a) {
    QuadFloat k = round_to_nearest_int(a);
    const int ki = (int)k.f0;
    QuadFloat r = subtract(a, k);                       // EXACT
    QuadFloat s = (r.f0 == 0.0f && r.f1 == 0.0f && r.f2 == 0.0f && r.f3 == 0.0f)
                ? QuadFloat(1.0f)
                : qf_exp_fast_impl(multiply(r, QuadFloat_log2()));
    if (ki >= -125 && ki <= 127) {
        const float p2 = ldexpf(1.0f, ki);
        return QuadFloat(s.f0 * p2, s.f1 * p2, s.f2 * p2, s.f3 * p2);
    }
    return QuadFloat(ldexpf(s.f0, ki), ldexpf(s.f1, ki),
                     ldexpf(s.f2, ki), ldexpf(s.f3, ki));
}
XPMATH_INLINE_FUNCTION QuadFloat qf_exp10_fast_impl(QuadFloat a) {
    const float kLog2_10 = 3.32192809f;
    const float kf = detail::rint(a.f0 * kLog2_10);
    if (!(detail::fabs(kf) < 1.0e5f))
        return qf_exp_fast_impl(multiply(a, QuadFloat_log10()));
    const int ki = (int)kf;
    const float kLog10_2_1 =  0x1.3442p-2f;
    const float kLog10_2_2 = -0x1.95ecp-19f;
    const float kLog10_2_3 = -0x1.0c02p-39f;
    const float kLog10_2_4 = -0x1.9dc2p-59f;
    const float kLog10_2_5 =  0x1.2b36p-78f;
    const float kLog10_2_6 = -0x1.fa42p-96f;
    QuadFloat r = subtract(a,  QuadFloat(kf * kLog10_2_1));
    r = subtract(r, QuadFloat(kf * kLog10_2_2));
    r = subtract(r, QuadFloat(kf * kLog10_2_3));
    r = subtract(r, QuadFloat(kf * kLog10_2_4));
    r = subtract(r, QuadFloat(kf * kLog10_2_5));
    r = subtract(r, QuadFloat(kf * kLog10_2_6));
    QuadFloat s = qf_exp_fast_impl(multiply(r, QuadFloat_log10()));
    if (ki >= -125 && ki <= 127) {
        const float p2 = ldexpf(1.0f, ki);
        return QuadFloat(s.f0 * p2, s.f1 * p2, s.f2 * p2, s.f3 * p2);
    }
    return QuadFloat(ldexpf(s.f0, ki), ldexpf(s.f1, ki),
                     ldexpf(s.f2, ki), ldexpf(s.f3, ki));
}
// P3.2[qf] e3: expm1, sinhcosh (-> sinh, cosh) and tanh are strict's
// bodies with the exp calls replaced by detail::qf_exp_fast_impl. The
// small-argument Taylor branches are strict's own code, called as is.
XPMATH_INLINE_FUNCTION QuadFloat qf_expm1_fast_impl(QuadFloat a) {
    if (detail::fabs(a.f0) > 0.5f)
        return subtract(qf_exp_fast_impl(a), QuadFloat(1.0f));
    return expm1(a);
}
XPMATH_INLINE_FUNCTION void qf_sinhcosh_fast_impl(
    QuadFloat a, QuadFloat& sinh_a, QuadFloat& cosh_a) {
    if (detail::fabs(a.f0) < 0.5f) { sinhcosh(a, sinh_a, cosh_a); return; }
    if (detail::fabs(a.f0) > kQFHyperbolicSaturate) {
        QuadFloat aa = (a.f0 < 0.0f) ? negate(a) : a;
        QuadFloat h;
        const float kQFReciprocalFloor = -40.0f;
        if (a.f0 < kQFReciprocalFloor) {
            h = qf_exp_fast_impl(subtract(aa, QuadFloat_log2()));
        } else {
            QuadFloat e = qf_exp_fast_impl(a);
            if (a.f0 < 0.0f) e = divide(QuadFloat(1.0f), e);
            h = (detail::isinf(e.f0) || e.f0 != e.f0)
                    ? qf_exp_fast_impl(subtract(aa, QuadFloat_log2()))
                    : mul_pwr2(e, 0.5f);
        }
        cosh_a = h;
        sinh_a = (a.f0 < 0.0f) ? negate(h) : h;
        return;
    }
    QuadFloat s0 = qf_exp_fast_impl(a);
    QuadFloat s1 = divide(QuadFloat(1.0f), s0);
    cosh_a = mul_pwr2(add(s0, s1), 0.5f);
    sinh_a = mul_pwr2(subtract(s0, s1), 0.5f);
}
XPMATH_NOINLINE_FUNCTION QuadFloat qf_tanh_fast_impl(QuadFloat a) {
    const bool neg = (a.f0 < 0.0f);
    if (neg) a = negate(a);
    QuadFloat r;
    if (a.f0 > kQFHyperbolicSaturate) {
        r = QuadFloat(1.0f);
    } else {
        QuadFloat e = qf_expm1_fast_impl(mul_pwr2(a, 2.0f));
        r = divide(e, add(e, QuadFloat(2.0f)));
    }
    return neg ? negate(r) : r;
}
}  // namespace detail
XPMATH_INLINE_FUNCTION QuadFloatFast log(QuadFloatFast a) { return to_fast(log(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast pow(QuadFloatFast a, QuadFloatFast b) { return to_fast(pow(to_strict(a), to_strict(b))); }
namespace detail {
// P3.3[qf] fast sincos. Rare inputs (|a| >= 800, NaN/inf, huge
// cancellation after the cheap reduction) go to the strict sincos
// unchanged, out of line. Out-params are (sin, cos) as in strict QF.
XPMATH_NOINLINE_FUNCTION void qf_sincos_slow(QuadFloat a, QuadFloat& s, QuadFloat& c) { sincos(a, s, c); }
// Same guards, nq = 0 and (sin, 1-cos) form as strict. Changes:
// (1) for 0.75 < |a| < 800 the reduction is k = rint(a.f0 * 2/pi)
// and 8 15-bit pieces of pi/2 plus a float tail, so k*piece is
// one exact float for k <= 512; beyond |a| = 800, or when |r| < 2^-30
// (the pieces no longer cover it), the strict Payne-Hanek path runs;
// (2) the series is a fixed Horner polynomial (sin to r^27, v to r^28)
// with 1/n! constants and no data-dependent exit; the last three terms
// of each weigh < 2^-80 of the sum and run in float.
XPMATH_INLINE_FUNCTION void qf_sincos_fast_impl(QuadFloat a, QuadFloat& sin_a, QuadFloat& cos_a) {
    if (a.f0 == 0.0f) { sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return; }
    const float aa = detail::fabs(a.f0);
    if (aa < 1.17549435e-38f) { sin_a = a; cos_a = QuadFloat(1.0f); return; }
    if (!(aa < 800.0f)) { qf_sincos_slow(a, sin_a, cos_a); return; }
    int j = 0;
    QuadFloat r = a;
    if (aa > 0.75f) {
        const float kf = detail::rint(a.f0 * 0x1.45f306p-1f);   // 2/pi
        r = subtract(r, QuadFloat(kf * 0x1.922p+0f));
        r = subtract(r, QuadFloat(kf * -0x1.2afp-18f));
        r = subtract(r, QuadFloat(kf * 0x1.0b48p-34f));
        r = subtract(r, QuadFloat(kf * -0x1.ee58p-50f));
        r = subtract(r, QuadFloat(kf * -0x1.d9ccp-66f));
        r = subtract(r, QuadFloat(kf * -0x1.d748p-83f));
        r = subtract(r, QuadFloat(kf * 0x1.b838p-104f));
        r = subtract(r, QuadFloat(kf * 0x1.a254p-120f));
        r = subtract(r, QuadFloat(kf * -0x1.fb6p-136f));
        if (!(detail::fabs(r.f0) >= 0x1p-30f)) { qf_sincos_slow(a, sin_a, cos_a); return; }
        j = (int)((long long)kf & 3);
    }
    const QuadFloat yy = multiply(r, r);
    float qs = -0x1.d1ab1cp-94f;                                           // -1/27!
    qs = detail::eft_add(0x1.3f3ccep-84f, detail::eft_mul(qs, yy.f0));    // 1/25!
    qs = detail::eft_add(-0x1.761b42p-75f, detail::eft_mul(qs, yy.f0));    // -1/23!
    QuadFloat ts = add(QuadFloat(0x1.71b8fp-66f, -0x1.246152p-91f, 0x1.8bef14p-118f, 0x1.cp-144f), multiply_scalar(yy, qs));   // 1/21!
    ts = add(QuadFloat(-0x1.2f49b4p-57f, -0x1.a05056p-83f, 0x1.b66bc2p-110f, 0x1.e24p-136f), multiply(ts, yy));   // -1/19!
    ts = add(QuadFloat(0x1.952c78p-49f, -0x1.f9ea56p-74f, -0x1.65367ep-99f, -0x1.734448p-124f), multiply(ts, yy));   // 1/17!
    ts = add(QuadFloat(-0x1.ae7f3ep-41f, -0x1.ccee08p-67f, 0x1.dc4f36p-94f, -0x1.ac3ba4p-119f), multiply(ts, yy));   // -1/15!
    ts = add(QuadFloat(0x1.612462p-33f, -0x1.8af25ep-58f, -0x1.a0d72p-83f, 0x1.98e91ep-108f), multiply(ts, yy));   // 1/13!
    ts = add(QuadFloat(-0x1.ae6456p-26f, -0x1.fd5138p-52f, -0x1.c7f3a4p-77f, 0x1.b47c82p-106f), multiply(ts, yy));   // -1/11!
    ts = add(QuadFloat(0x1.71de3ap-19f, 0x1.55b1ccp-45f, 0x1.c7d56p-70f, 0x1.c88e5p-95f), multiply(ts, yy));   // 1/9!
    ts = add(QuadFloat(-0x1.a01a02p-13f, 0x1.7f97fap-39f, -0x1.00d00ep-64f, 0x1.fe5fe6p-89f), multiply(ts, yy));   // -1/7!
    ts = add(QuadFloat(0x1.111112p-7f, -0x1.dddddep-32f, 0x1.111112p-59f, -0x1.dddddep-84f), multiply(ts, yy));   // 1/5!
    ts = add(QuadFloat(-0x1.555556p-3f, 0x1.555556p-28f, -0x1.555556p-53f, 0x1.555556p-78f), multiply(ts, yy));   // -1/3!
    QuadFloat sin_r = add(r, multiply(r, multiply(ts, yy)));
    float qv = -0x1.0a18a2p-98f;                                           // -1/28!
    qv = detail::eft_add(0x1.88e86p-89f, detail::eft_mul(qv, yy.f0));    // 1/26!
    qv = detail::eft_add(-0x1.f2cf02p-80f, detail::eft_mul(qv, yy.f0));    // -1/24!
    QuadFloat tv = add(QuadFloat(0x1.0ce396p-70f, 0x1.b6ff0ap-95f, 0x1.4a2864p-121f, 0x1p-146f), multiply_scalar(yy, qv));   // 1/22!
    tv = add(QuadFloat(-0x1.e542bap-62f, -0x1.00808ap-88f, 0x1.7c2b1ap-113f, 0x1.2d4p-138f), multiply(tv, yy));   // -1/20!
    tv = add(QuadFloat(0x1.682786p-53f, 0x1.dcbeccp-80f, -0x1.089ff6p-106f, -0x1.de574p-131f), multiply(tv, yy));   // 1/18!
    tv = add(QuadFloat(-0x1.ae7f3ep-45f, -0x1.ccee08p-71f, 0x1.dc4f36p-98f, -0x1.ac3ba4p-123f), multiply(tv, yy));   // -1/16!
    tv = add(QuadFloat(0x1.93974ap-37f, 0x1.180f94p-62f, -0x1.6fa29p-88f, -0x1.d7440cp-114f), multiply(tv, yy));   // 1/14!
    tv = add(QuadFloat(-0x1.1eed8ep-29f, -0x1.ff1b12p-54f, -0x1.ed5136p-79f, -0x1.4c3d68p-104f), multiply(tv, yy));   // -1/12!
    tv = add(QuadFloat(0x1.27e4fcp-22f, -0x1.10ec14p-47f, -0x1.c6888p-73f, 0x1.6d3ea6p-98f), multiply(tv, yy));   // 1/10!
    tv = add(QuadFloat(-0x1.a01a02p-16f, 0x1.7f97fap-42f, -0x1.00d00ep-67f, 0x1.fe5fe6p-92f), multiply(tv, yy));   // -1/8!
    tv = add(QuadFloat(0x1.6c16c2p-10f, -0x1.27d27ep-35f, 0x1.b05b06p-60f, -0x1.3e93eap-86f), multiply(tv, yy));   // 1/6!
    tv = add(QuadFloat(-0x1.555556p-5f, 0x1.555556p-30f, -0x1.555556p-55f, 0x1.555556p-80f), multiply(tv, yy));   // -1/4!
    tv = add(QuadFloat(0.5f), multiply(tv, yy));
    const QuadFloat v_r = multiply(tv, yy);
    QuadFloat cos_r = subtract(QuadFloat(1.0f), v_r);
    // NaN/inf in a word: strict answers with its identity-point
    // fallback; let strict do it.
    if (!(detail::fabs(sin_r.f0) <= 1.0009765625f) || !(detail::fabs(cos_r.f0) <= 1.0009765625f)) { qf_sincos_slow(a, sin_a, cos_a); return; }
    if (sin_r.f0 >  1.0f || (sin_r.f0 ==  1.0f && sin_r.f1 > 0.0f)) sin_r = QuadFloat( 1.0f);
    if (sin_r.f0 < -1.0f || (sin_r.f0 == -1.0f && sin_r.f1 < 0.0f)) sin_r = QuadFloat(-1.0f);
    if (cos_r.f0 >  1.0f || (cos_r.f0 ==  1.0f && cos_r.f1 > 0.0f)) cos_r = QuadFloat( 1.0f);
    if (cos_r.f0 < -1.0f || (cos_r.f0 == -1.0f && cos_r.f1 < 0.0f)) cos_r = QuadFloat(-1.0f);
    if (j == 0)      { sin_a = sin_r;         cos_a = cos_r; }
    else if (j == 1) { sin_a = cos_r;         cos_a = negate(sin_r); }
    else if (j == 2) { sin_a = negate(sin_r); cos_a = negate(cos_r); }
    else             { sin_a = negate(cos_r); cos_a = sin_r; }
}
} // namespace detail
XPMATH_INLINE_FUNCTION void sincos(QuadFloatFast a, QuadFloatFast& sin_a, QuadFloatFast& cos_a) { QuadFloat t_sin_a; QuadFloat t_cos_a; detail::qf_sincos_fast_impl(to_strict(a), t_sin_a, t_cos_a); sin_a = to_fast(t_sin_a); cos_a = to_fast(t_cos_a); }
XPMATH_INLINE_FUNCTION void sinhcosh(QuadFloatFast a, QuadFloatFast& sinh_a, QuadFloatFast& cosh_a) { QuadFloat t_sinh_a; QuadFloat t_cosh_a; detail::qf_sinhcosh_fast_impl(to_strict(a), t_sinh_a, t_cosh_a); sinh_a = to_fast(t_sinh_a); cosh_a = to_fast(t_cosh_a); }
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
XPMATH_INLINE_FUNCTION QuadFloatFast exp2(QuadFloatFast a) { return to_fast(detail::qf_exp2_fast_impl(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast exp10(QuadFloatFast a) { return to_fast(detail::qf_exp10_fast_impl(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast expm1(QuadFloatFast a) { return to_fast(detail::qf_expm1_fast_impl(to_strict(a))); }
XPMATH_NOINLINE_FUNCTION QuadFloatFast sin(QuadFloatFast a) { QuadFloat s, c; detail::qf_sincos_fast_impl(to_strict(a), s, c); return to_fast(s); }
XPMATH_NOINLINE_FUNCTION QuadFloatFast cos(QuadFloatFast a) { QuadFloat s, c; detail::qf_sincos_fast_impl(to_strict(a), s, c); return to_fast(c); }
XPMATH_INLINE_FUNCTION QuadFloatFast tan(QuadFloatFast a) { QuadFloat s, c; detail::qf_sincos_fast_impl(to_strict(a), s, c); return to_fast(divide(s, c)); }
XPMATH_INLINE_FUNCTION int qf_cmp_one(QuadFloatFast a) { return qf_cmp_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION int qf_cmp_abs_one(QuadFloatFast a) { return qf_cmp_abs_one(to_strict(a)); }
XPMATH_INLINE_FUNCTION QuadFloatFast asin(QuadFloatFast a) { return to_fast(asin(to_strict(a))); }
XPMATH_INLINE_FUNCTION QuadFloatFast acos(QuadFloatFast a) { return to_fast(acos(to_strict(a))); }
namespace detail {
// P3.3[qf] callers of sincos wired to the fast sincos (atan, atan2,
// real tan, complex exp/sin/cos/tan). Each body is the strict body
// (qf_math.hpp / qf_complex.hpp) with sincos -> qf_sincos_fast_impl;
// complex exp and tan also use the fast real exp, complex sin/cos the
// fast sinhcosh. Strict is unchanged.
XPMATH_INLINE_FUNCTION QuadFloat qf_angle_fast_body(QuadFloat x, QuadFloat y) {
    QuadFloat pi = QuadFloat_pi();
    if (y.f0 == 0.0f) {
        const QuadFloat r =
            (detail::copysign(1.0f, x.f0) < 0.0f) ? pi : QuadFloat(0.0f);
        return (detail::copysign(1.0f, y.f0) < 0.0f) ? negate(r) : r;
    }
    if (x.f0 == 0.0f) return (y.f0 > 0.0f) ? mul_pwr2(pi, 0.5f) : mul_pwr2(pi, -0.5f);
    {
        const float mx = detail::fabs(x.f0), my = detail::fabs(y.f0);
        const float mm = (mx > my) ? mx : my;
        if (mm > detail::kQFSqHi) {
            const float s = detail::qf_pow2_unit_scale(mm);
            x = detail::qf_pow2_scale(x, s);
            y = detail::qf_pow2_scale(y, s);
        }
    }
    QuadFloat r  = sqrt(add(multiply(x, x), multiply(y, y)));
    QuadFloat nx = divide(x, r), ny = divide(y, r);
    QuadFloat a  = QuadFloat(detail::atan2(ny.f0, nx.f0));   // FP32 seed
    bool use_x = (detail::fabs(nx.f0) <= detail::fabs(ny.f0));
    QuadFloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        QuadFloat sin_a, cos_a;
        qf_sincos_fast_impl(a, sin_a, cos_a);
        if (use_x) {
            a = subtract(a, divide(subtract(target, cos_a), sin_a));
        } else {
            a = add(a, divide(subtract(target, sin_a), cos_a));
        }
    }
    return a;
}
// Out of line, like strict atan2.
XPMATH_NOINLINE_FUNCTION QuadFloat qf_angle_fast(QuadFloat x, QuadFloat y) { return qf_angle_fast_body(x, y); }
XPMATH_INLINE_FUNCTION QuadFloat qf_atan_fast_impl(QuadFloat a) {
    QuadFloat r = qf_angle_fast(QuadFloat(1.0f), a);
    const QuadFloat p = QuadFloat_pi();
    const QuadFloat h(p.f0 * 0.5f, p.f1 * 0.5f, p.f2 * 0.5f, p.f3 * 0.5f);
    const QuadFloat ar = (r.f0 < 0.0f) ? QuadFloat(-r.f0, -r.f1, -r.f2, -r.f3) : r;
    if (subtract(ar, h).f0 > 0.0f)
        return (r.f0 < 0.0f) ? QuadFloat(-h.f0, -h.f1, -h.f2, -h.f3) : h;
    return r;
}
XPMATH_INLINE_FUNCTION QuadFloatComplex qf_cexp_fast_impl(QuadFloatComplex z) {
    QuadFloat er = qf_exp_fast_impl(z.re);
    QuadFloat s, c;
    qf_sincos_fast_impl(z.im, s, c);
    return QuadFloatComplex(multiply(er, c), multiply(er, s));
}
XPMATH_INLINE_FUNCTION QuadFloatComplex qf_csin_fast_impl(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    qf_sincos_fast_impl(z.re, sa, ca);
    qf_sinhcosh_fast_impl(z.im, sb, cb);
    return QuadFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
XPMATH_INLINE_FUNCTION QuadFloatComplex qf_ccos_fast_impl(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    qf_sincos_fast_impl(z.re, sa, ca);
    qf_sinhcosh_fast_impl(z.im, sb, cb);
    return QuadFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
XPMATH_INLINE_FUNCTION QuadFloatComplex qf_ctan_fast_impl(QuadFloatComplex z) {
    const float kXpTanAsymptote = 2.0f;
    if (detail::fabs(z.im.f0) >= kXpTanAsymptote) {
        QuadFloat ca, sa;
        qf_sincos_fast_impl(z.re, sa, ca);
        const QuadFloat s2 = multiply_scalar(multiply(sa, ca), 2.0f);
        const QuadFloat c2 = multiply(subtract(ca, sa), add(ca, sa));
        const QuadFloat t  = qf_exp_fast_impl(multiply_scalar(z.im, z.im.f0 < 0.0f ? 2.0f : -2.0f));
        const QuadFloat t2 = multiply(t, t);
        const QuadFloat den = add(add(QuadFloat(1.0f), t2), multiply_scalar(multiply(t, c2), 2.0f));
        QuadFloat im = divide(subtract(QuadFloat(1.0f), t2), den);
        if (z.im.f0 < 0.0f) im = negate(im);
        return QuadFloatComplex(divide(multiply_scalar(multiply(t, s2), 2.0f), den), im);
    }
    return qf_csin_fast_impl(z) / qf_ccos_fast_impl(z);
}
} // namespace detail
XPMATH_INLINE_FUNCTION QuadFloatFast atan(QuadFloatFast a) { return to_fast(detail::qf_atan_fast_impl(to_strict(a))); }
XPMATH_NOINLINE_FUNCTION QuadFloatFast atan2(QuadFloatFast y, QuadFloatFast x) { return to_fast(detail::qf_angle_fast(to_strict(x), to_strict(y))); }
XPMATH_INLINE_FUNCTION QuadFloatFast sinh(QuadFloatFast a) { QuadFloat s, c; detail::qf_sinhcosh_fast_impl(to_strict(a), s, c); return to_fast(s); }
XPMATH_INLINE_FUNCTION QuadFloatFast cosh(QuadFloatFast a) { QuadFloat s, c; detail::qf_sinhcosh_fast_impl(to_strict(a), s, c); return to_fast(c); }
XPMATH_INLINE_FUNCTION QuadFloatFast tanh(QuadFloatFast a) { return to_fast(detail::qf_tanh_fast_impl(to_strict(a))); }
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
XPMATH_NOINLINE_FUNCTION QuadFloatComplexFast exp(QuadFloatComplexFast z) { return to_fast(detail::qf_cexp_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast log(QuadFloatComplexFast z) { return to_fast(log(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast log1p(QuadFloatComplexFast w) { return to_fast(log1p(to_strict(w))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast log10(QuadFloatComplexFast z) { return to_fast(log10(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast sin(QuadFloatComplexFast z) { return to_fast(detail::qf_csin_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast cos(QuadFloatComplexFast z) { return to_fast(detail::qf_ccos_fast_impl(to_strict(z))); }
XPMATH_INLINE_FUNCTION QuadFloatComplexFast tan(QuadFloatComplexFast z) { return to_fast(detail::qf_ctan_fast_impl(to_strict(z))); }
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
