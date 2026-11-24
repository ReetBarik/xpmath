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
    DoubleDoubleFast(const DoubleDoubleFast&) = default;
    DoubleDoubleFast& operator=(const DoubleDoubleFast&) = default;
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
    DoubleDoubleComplexFast(const DoubleDoubleComplexFast&) = default;
    DoubleDoubleComplexFast& operator=(const DoubleDoubleComplexFast&) = default;
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
namespace detail {
// P3.2[dd] fast exp. Same guards, Cody-Waite pieces, nq = 6 squarings
// and 2^nz scaling as strict exp; two changes: the reduction integer
// is rint(a.hi * 1/ln2) (one native multiply, not divide(a, ln2)),
// and e^x - 1 is a fixed 12-term Horner series with 1/n! constants
// (no divide_scalar, no data-dependent exit). Terms 7..12 weigh
// < 1e-17 of the sum at |x| <= ln2/128, so they run in double.
XPMATH_INLINE_FUNCTION DoubleDouble dd_exp_fast_impl(DoubleDouble a) {
    if (a.hi != a.hi) return a;
    if (a.hi > 709.78271289338397) {
        XPMATH_PRINTF("DDEXP: overflow\n");
        return DoubleDouble(HUGE_VAL);
    }
    if (a.hi < -745.2) return DoubleDouble(0.0);
    const double kd = detail::rint(a.hi * 0x1.71547652b82fep+0);
    const int nz = (int)kd;
    const double kLn2_1 =  0x1.62e42fefa38p-1;
    const double kLn2_2 =  0x1.ef35793c768p-45;
    const double kLn2_3 = -0x1.9ff0342543p-90;
    DoubleDouble r = subtract(a, DoubleDouble(kd * kLn2_1));
    r = subtract(r, DoubleDouble(kd * kLn2_2));
    r = subtract(r, DoubleDouble(kd * kLn2_3));
    if (r.hi == 0.0) return DoubleDouble(detail::ldexp(1.0, nz));
    const DoubleDouble x = multiply_scalar(r, 0x1p-6);
    double q = 0x1.1eed8eff8d898p-29;                                  // 1/12!
    q = detail::eft_add(0x1.ae64567f544e4p-26, detail::eft_mul(q, x.hi)); // 1/11!
    q = detail::eft_add(0x1.27e4fb7789f5cp-22, detail::eft_mul(q, x.hi)); // 1/10!
    q = detail::eft_add(0x1.71de3a556c734p-19, detail::eft_mul(q, x.hi)); // 1/9!
    q = detail::eft_add(0x1.a01a01a01a01ap-16, detail::eft_mul(q, x.hi)); // 1/8!
    q = detail::eft_add(0x1.a01a01a01a01ap-13, detail::eft_mul(q, x.hi)); // 1/7!
    DoubleDouble t = add(DoubleDouble(0x1.6c16c16c16c17p-10, -0x1.f49f49f49f49fp-65), multiply(DoubleDouble(q), x)); // 1/6!
    t = add(DoubleDouble(0x1.1111111111111p-7, 0x1.1111111111111p-63), multiply(t, x)); // 1/5!
    t = add(DoubleDouble(0x1.5555555555555p-5, 0x1.5555555555555p-59), multiply(t, x)); // 1/4!
    t = add(DoubleDouble(0x1.5555555555555p-3, 0x1.5555555555555p-57), multiply(t, x)); // 1/3!
    t = add(DoubleDouble(0.5), multiply(t, x));
    t = add(DoubleDouble(1.0), multiply(t, x));
    DoubleDouble s = multiply(t, x);                                   // e^x - 1
    for (int i = 0; i < 6; ++i) s = multiply(s, add(s, DoubleDouble(2.0)));
    s = add(DoubleDouble(1.0), s);
    if (nz >= -1021 && nz <= 1023) {
        const double pow2 = detail::ldexp(1.0, nz);
        return DoubleDouble(s.hi * pow2, s.lo * pow2);
    }
    return DoubleDouble(detail::ldexp(s.hi, nz), detail::ldexp(s.lo, nz));
}
} // namespace detail
XPMATH_INLINE_FUNCTION DoubleDoubleFast exp(DoubleDoubleFast a) { return to_fast(detail::dd_exp_fast_impl(to_strict(a))); }
namespace detail {
// P3.2[dd]-chain: the Newton steps call the fast exp.
XPMATH_INLINE_FUNCTION DoubleDouble dd_log_fast_impl(DoubleDouble a) {
    if (detail::isinf(a.hi)) return a;
    if (a.hi <= 0.0) {
        XPMATH_PRINTF("DDLOG: non-positive argument\n");
        return DoubleDouble(0.0);
    }
    DoubleDouble b = DoubleDouble(detail::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        if (b.hi < -671.7 && -b.hi <= 709.78271289338397) {
            DoubleDouble s0 = dd_exp_fast_impl(negate(b));
            b = subtract(add(b, multiply(a, s0)), DoubleDouble(1.0));
        } else {
            DoubleDouble s0 = dd_exp_fast_impl(b);
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
namespace detail {
// P3.3[dd] fast sincos. Rare inputs (|a| >= 1e6, NaN/inf, huge
// cancellation after the cheap reduction) go to the strict sincos
// unchanged, out of line.
XPMATH_NOINLINE_FUNCTION void dd_sincos_slow(DoubleDouble a, DoubleDouble& x, DoubleDouble& y) { sincos(a, x, y); }
// Same guards, nq = 5 halvings, (sin, 1-cos) form and 5 doublings as
// strict. Changes: (1) for 0.75 < |a| < 1e6 the reduction is
// k = rint(a.hi * 2/pi) and four 33-bit Cody-Waite pieces of pi/2 plus
// one full double (k*piece exact for |k| < 2^20), Payne-Hanek only
// above that or when |r| < 2^-40 (the pieces no longer cover it);
// (2) the series is a fixed Horner polynomial (6 sin terms, 7 v terms)
// with 1/n! constants and no data-dependent exit; terms past the third
// weigh < 1e-17 of the sum and run in double.
XPMATH_INLINE_FUNCTION void dd_sincos_fast_impl(DoubleDouble a, DoubleDouble& x, DoubleDouble& y) {
    if (a.hi == 0.0) { x = DoubleDouble(1.0); y = DoubleDouble(0.0); return; }
    const double aa = detail::fabs(a.hi);
    if (aa < 32.0 * 2.2250738585072014e-308) { x = DoubleDouble(1.0); y = a; return; }
    if (!(aa < 1.0e6)) { dd_sincos_slow(a, x, y); return; }
    int j = 0;
    DoubleDouble r_mod = a;
    if (aa > 0.75) {
        const double kd = detail::rint(a.hi * 0x1.45f306dc9c883p-1);   // 2/pi
        r_mod = subtract(a, DoubleDouble(kd * 0x1.921fb54400000p+0));
        r_mod = subtract(r_mod, DoubleDouble(kd * 0x1.0b4611a600000p-34));
        r_mod = subtract(r_mod, DoubleDouble(kd * 0x1.3198a2e000000p-69));
        r_mod = subtract(r_mod, DoubleDouble(kd * 0x1.b839a25200000p-104));
        r_mod = subtract(r_mod, DoubleDouble(kd * 0x1.27044533e63a0p-142));
        if (!(detail::fabs(r_mod.hi) >= 0x1p-40)) { dd_sincos_slow(a, x, y); return; }
        j = (int)((long long)kd & 3);
    }
    const DoubleDouble r  = multiply_scalar(r_mod, 0x1p-5);
    const DoubleDouble yy = multiply(r, r);
    double qs = 0x1.6124613a86d09p-33;                                   // 1/13!
    qs = detail::eft_add(-0x1.ae64567f544e4p-26, detail::eft_mul(qs, yy.hi)); // 1/11!
    qs = detail::eft_add(0x1.71de3a556c734p-19, detail::eft_mul(qs, yy.hi));  // 1/9!
    DoubleDouble ts = add(DoubleDouble(-0x1.a01a01a01a01ap-13, -0x1.a01a01a01a01ap-73), multiply(DoubleDouble(qs), yy)); // 1/7!
    ts = add(DoubleDouble(0x1.1111111111111p-7, 0x1.1111111111111p-63), multiply(ts, yy));   // 1/5!
    ts = add(DoubleDouble(-0x1.5555555555555p-3, -0x1.5555555555555p-57), multiply(ts, yy)); // 1/3!
    DoubleDouble sin_r = add(r, multiply(r, multiply(ts, yy)));
    double qv = 0x1.93974a8c07c9dp-37;                                   // 1/14!
    qv = detail::eft_add(-0x1.1eed8eff8d898p-29, detail::eft_mul(qv, yy.hi)); // 1/12!
    qv = detail::eft_add(0x1.27e4fb7789f5cp-22, detail::eft_mul(qv, yy.hi));  // 1/10!
    DoubleDouble tv = add(DoubleDouble(-0x1.a01a01a01a01ap-16, -0x1.a01a01a01a01ap-76), multiply(DoubleDouble(qv), yy)); // 1/8!
    tv = add(DoubleDouble(0x1.6c16c16c16c17p-10, -0x1.f49f49f49f49fp-65), multiply(tv, yy)); // 1/6!
    tv = add(DoubleDouble(-0x1.5555555555555p-5, -0x1.5555555555555p-59), multiply(tv, yy)); // 1/4!
    tv = add(DoubleDouble(0.5), multiply(tv, yy));
    DoubleDouble v_r = multiply(tv, yy);
    for (int q = 0; q < 5; ++q) {
        const DoubleDouble c_q = subtract(DoubleDouble(1.0), v_r);
        const DoubleDouble new_sin = multiply_scalar(multiply(sin_r, c_q), 2.0);
        v_r   = multiply_scalar(multiply(sin_r, sin_r), 2.0);
        sin_r = new_sin;
    }
    DoubleDouble cos_r = subtract(DoubleDouble(1.0), v_r);
    // NaN/inf in a word (e.g. a finite hi with a NaN lo): strict answers
    // with its identity-point fallback; let strict do it.
    if (!(detail::fabs(sin_r.hi) <= 1.0009765625) || !(detail::fabs(cos_r.hi) <= 1.0009765625)) { dd_sincos_slow(a, x, y); return; }
    if (sin_r.hi >  1.0 || (sin_r.hi ==  1.0 && sin_r.lo > 0.0)) sin_r = DoubleDouble( 1.0);
    if (sin_r.hi < -1.0 || (sin_r.hi == -1.0 && sin_r.lo < 0.0)) sin_r = DoubleDouble(-1.0);
    if (cos_r.hi >  1.0 || (cos_r.hi ==  1.0 && cos_r.lo > 0.0)) cos_r = DoubleDouble( 1.0);
    if (cos_r.hi < -1.0 || (cos_r.hi == -1.0 && cos_r.lo < 0.0)) cos_r = DoubleDouble(-1.0);
    if (j == 0)      { x = cos_r;         y = sin_r; }
    else if (j == 1) { x = negate(sin_r); y = cos_r; }
    else if (j == 2) { x = negate(cos_r); y = negate(sin_r); }
    else             { x = sin_r;         y = negate(cos_r); }
}
} // namespace detail
XPMATH_INLINE_FUNCTION void sincos(DoubleDoubleFast a, DoubleDoubleFast& x, DoubleDoubleFast& y) { DoubleDouble t_x; DoubleDouble t_y; detail::dd_sincos_fast_impl(to_strict(a), t_x, t_y); x = to_fast(t_x); y = to_fast(t_y); }
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
XPMATH_INLINE_FUNCTION DoubleDoubleFast sin(DoubleDoubleFast a) { DoubleDouble c, s; detail::dd_sincos_fast_impl(to_strict(a), c, s); return to_fast(s); }
XPMATH_INLINE_FUNCTION DoubleDoubleFast cos(DoubleDoubleFast a) { DoubleDouble c, s; detail::dd_sincos_fast_impl(to_strict(a), c, s); return to_fast(c); }
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
// P3.2[dd]-chain: strict complex log body, fast real log.
// Not inline, same as strict complex log (TD-4, sm_100).
XPMATH_NOINLINE_FUNCTION DoubleDoubleComplexFast log(DoubleDoubleComplexFast z) {
    const DoubleDoubleComplex w = to_strict(z);
    const DoubleDouble modulus = abs(w);
    const DoubleDouble arg = atan2(w.im, w.re);
    return to_fast(DoubleDoubleComplex(detail::dd_log_fast_impl(modulus), arg));
}
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
