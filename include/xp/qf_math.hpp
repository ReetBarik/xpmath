// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (original algorithms;
//   Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical port of the QD 2.3.24 quad-double package
// (qd/src/qd_real.cpp and qd/include/qd/qd_inline.h) from four-word FP64
// (quad-double, ~212-bit significand) to four-word FP32 (quad-float,
// "QuadFloat", ~96-bit significand, ~29 decimal digits, u = 2^-96). The
// algorithm structure — Priest renormalization (renorm_4), Hida-Li-Bailey
// sloppy/ieee addition, sloppy multiplication, long-division, and Heron
// square-root — descends directly from QD 2.3.24. Every non-trivial routine
// cites its QD source location.
//
// LICENSE LINEAGE (per docs/TEST_SUITE_PLAN.md §"Phase 2/3 open question — FF
// and QF port lineage", T3.0a kickoff): QF is modeled on QD, NOT on DDFUN, so
// it inherits QD's LBNL-BSD-License (triple-authored Hida/Li/Bailey, LBNL
// *institutional* copyright, commercial contact ipo@lbl.gov / TTD@lbl.gov) —
// which is a DIFFERENT license than the DHB-License that governs dd_math.hpp /
// ff_math.hpp (Bailey personal copyright, DDFUN provenance). The header here
// therefore carries LicenseRef-LBNL-BSD-License, not LicenseRef-DHB-License.
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.
//
// FP32-specific porting notes (splitter reuse, Newton/Heron iteration counts,
// sloppy_add safety at the narrower FP32 exponent, constant generation) are
// documented in docs/PORT_NOTES_QF.md.

#pragma once

// Quad-float real arithmetic — xp::QuadFloat. ~29 decimal digits
// from an unevaluated sum of four FP32 components (f0 + f1 + f2 + f3,
// |f1| <= ulp(f0)/2, |f2| <= ulp(f1)/2, |f3| <= ulp(f2)/2).
//
// Mechanically ported from QD 2.3.24 (quad-double at 4×FP64, Hida-Li-Bailey)
// by swapping 4×FP64 for 4×FP32. See docs/PORT_NOTES_QF.md for FP32 specifics.
//
// Precision: ~28.9 decimal digits (24-bit FP32 mantissa × 4 = ~96 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.
//
// DEPENDENCIES: none beyond the C++17 standard library. In particular this
// header does NOT include or require Kokkos — see xp/config.hpp for how the
// four portability facilities it needs (inline annotation, on-device
// detection, scalar math dispatch, diagnostic printf) are supplied. Kokkos
// users get today's `Kokkos::Experimental::QuadFloat` API unchanged through
// the compat wrapper at third_party/include/qf_math.hpp, which is the only
// place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// Naming conventions (T0.4/T2.0):
//   * Type + math live in one flat namespace so an upstream move is
//     mechanical rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions QuadFloat_pi(), QuadFloat_e(), ...
//     Chosen over a constants::pi<QuadFloat>() template because it mirrors
//     the repository's existing M_PI-style accessors and reads shorter at the
//     call site; they cannot be constexpr template variables because each is
//     built at runtime from IEEE-754 bit patterns, not a literal.
//   * The former bit-pattern constructor became the static factory
//     QuadFloat::from_bits(f0, f1, f2, f3): it is namespaced to the type,
//     discoverable, and needs no free-function symbol.
//   * Math functions are ADL-findable via the argument's namespace. The
//     `Kokkos::`-namespace forwarding overloads that used to sit at the bottom
//     of the original Kokkos-native header (so Kokkos::exp(qf) works like
//     Kokkos::exp(double)) now live in the compat wrapper, since they are
//     Kokkos-specific API surface. add/subtract/multiply/divide are not
//     forwarded — they are for operators and explicit ADL only.

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
// Forward declarations
// ============================================================
struct QuadFloat;
XPMATH_INLINE_FUNCTION QuadFloat add(QuadFloat a, QuadFloat b);
XPMATH_INLINE_FUNCTION QuadFloat subtract(QuadFloat a, QuadFloat b);
XPMATH_INLINE_FUNCTION QuadFloat multiply(QuadFloat a, QuadFloat b);
XPMATH_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b);
XPMATH_INLINE_FUNCTION QuadFloat multiply_scalar(QuadFloat a, float b);
XPMATH_INLINE_FUNCTION QuadFloat divide_scalar(QuadFloat a, float b);
XPMATH_INLINE_FUNCTION QuadFloat mul_pwr2(QuadFloat a, float b);
XPMATH_INLINE_FUNCTION QuadFloat negate(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat abs(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat sqr(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat sqrt(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat round_to_nearest_int(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat pow_int(QuadFloat a, int n);
// T3.0b transcendentals (forward decls — struct-independent, but several call
// each other, so declare the whole family up front).
XPMATH_INLINE_FUNCTION QuadFloat exp(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat log(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat pow(QuadFloat a, QuadFloat b);
XPMATH_INLINE_FUNCTION void      sincos(QuadFloat a, QuadFloat& sin_a, QuadFloat& cos_a);
XPMATH_INLINE_FUNCTION void      sinhcosh(QuadFloat a, QuadFloat& sinh_a, QuadFloat& cosh_a);
XPMATH_INLINE_FUNCTION QuadFloat angle(QuadFloat x, QuadFloat y);
XPMATH_INLINE_FUNCTION QuadFloat ceil(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat floor(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat trunc(QuadFloat a);
XPMATH_INLINE_FUNCTION QuadFloat round(QuadFloat a);

// ============================================================
// Error-free transforms (FP32).
// Bit-identical to the primitives validated for FF in ff_math.hpp (T2.1);
// expressed here in QD's by-reference form (QD 2.3.24 qd/include/qd/inline.h).
// ============================================================

// fl(a+b) and err, assuming |a| >= |b|.  QD inline.h:35-39 (quick_two_sum).
XPMATH_INLINE_FUNCTION float qf_quick_two_sum(float a, float b, float& err) {
    float s = a + b;
    err = b - (s - a);
    return s;
}

// fl(a+b) and err (Knuth TwoSum, no ordering assumption).  QD inline.h:49-55.
// Mirror of the twoSum inside ff_math.hpp add() (ff_math.hpp:174-181).
XPMATH_INLINE_FUNCTION float qf_two_sum(float a, float b, float& err) {
    float s  = a + b;
    float bb = s - a;
    err = (a - (s - bb)) + (b - bb);
    return s;
}

// Veltkamp split with QD's large-magnitude guard.  Port of qd::split,
// QD 2.3.24 qd/include/qd/inline.h:66-83, at FP32.
//
// Splitter 8193.0f = 2^13+1 for the 24-bit FP32 mantissa — the same constant
// used and validated for FF (ff_math.hpp:194, two_prod ff_math.hpp:266-274).
//
// KI-9 (fixed 2026-09-02, see docs/KNOWN_ISSUES.md). The original port
// deliberately dropped QD's `_QD_SPLIT_THRESH` branch, recorded in
// PORT_NOTES_QF §2 as an accepted simplification. It is not one: for
// |a| > FLT_MAX / (split + 1) ≈ 4.15e34 the product `a * split` overflows to
// ±inf, and then `hi = con - (con - a)` is inf - inf = NaN, which poisons every
// consumer — multiply, sqr, multiply_scalar, and through multiply_scalar the
// long-division quotient digits of divide(). KI-8's scaled hypot divides by
// max(|a|,|b|), so hypot/complex-abs went NaN for any operand above 4.15e34
// even though the answer (1.41e37 for hypot(1e37,1e37)) is comfortably inside
// FP32's 3.4e38 range. This is a PORTING DEFECT: QD guards it, the port did
// not; FF had already re-derived the same guard independently (B8/B9,
// ff_math.hpp:245).
//
// The guard scales by an exact power of two, so it introduces no rounding:
// pre-scale by 2^-14 (bringing FLT_MAX down to 2.08e34, inside the safe band),
// split, then unscale hi and lo by 2^14. Both unscalings are exact — |hi| <= |a|
// and |lo| < ulp(a), so neither can overflow or reach the subnormal range. This
// mirrors QD's 2^-28 / 2^28 pair, which is 2^(1024-996); the FP32 analogue is
// 2^(128-114).
//
// Unlike FF's B8/B9 guard, which scales the whole operand and unscales the
// RESULT, scaling only inside the split leaves `p = a * b` untouched, so the
// non-hazard path is bit-identical and the hazard path costs no extra rounding.
XPMATH_INLINE_FUNCTION void qf_split(float a, float& hi, float& lo) {
    const float split  = 8193.0f;
    const float thresh = 4.1528233e34f;      // FLT_MAX / (split + 1)
    float temp;
    if (a > thresh || a < -thresh) {
        a *= 6.103515625e-05f;               // 2^-14, exact
        temp = split * a;
        hi   = temp - (temp - a);
        lo   = a - hi;
        hi  *= 16384.0f;                     // 2^14, exact
        lo  *= 16384.0f;
    } else {
        temp = split * a;
        hi   = temp - (temp - a);
        lo   = a - hi;
    }
}

// fl(a*b) and err (Dekker TwoProduct via Veltkamp split).  QD inline.h:85-99.
// Empirically exact over |operands| <= 1e6 (scripts/gen_qf_constants harness /
// scripts/test_qfmul.cpp); KI-9 extends that to the full FP32 range.
//
// KI-9 also fixes the GENUINE-overflow tail. When the true product exceeds
// FLT_MAX, p is ±inf and the residual expression evaluates inf - inf = NaN, so
// an overflow that IEEE 754 defines as ±inf came back as a NaN-tailed
// expansion — hypot(3e38, 3e38) returned NaN where 4.24e38 correctly overflows
// to +inf. The error term of an overflowed product carries no information, so
// the only defensible value is 0. The early return also skips two splits on
// that path. Non-finite operands (±inf, NaN) take the same branch and
// propagate whatever `a * b` gives, which is already the IEEE answer.
XPMATH_INLINE_FUNCTION float qf_two_prod(float a, float b, float& err) {
    float p = a * b;
    if (!detail::isfinite(p)) { err = 0.0f; return p; }
    float a1, a2, b1, b2;
    qf_split(a, a1, a2);
    qf_split(b, b1, b2);
    err = ((a1 * b1 - p) + a1 * b2 + a2 * b1) + a2 * b2;
    return p;
}

// fl(a*a) and err.  QD inline.h:101-113 (two_sqr).
XPMATH_INLINE_FUNCTION float qf_two_sqr(float a, float& err) {
    float q = a * a;
    if (!detail::isfinite(q)) { err = 0.0f; return q; }
    float hi, lo;
    qf_split(a, hi, lo);
    err = ((hi * hi - q) + 2.0f * hi * lo) + lo * lo;
    return q;
}

// three_sum / three_sum2.  QD inline.h:192-204.
XPMATH_INLINE_FUNCTION void qf_three_sum(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = qf_two_sum(a, b, t2);
    a  = qf_two_sum(c, t1, t3);
    b  = qf_two_sum(t2, t3, c);
}
XPMATH_INLINE_FUNCTION void qf_three_sum2(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = qf_two_sum(a, b, t2);
    a  = qf_two_sum(c, t1, t3);
    b  = t2 + t3;
}

// ============================================================
// Renormalization (Priest normalization — Hida-Li-Bailey Algorithm 3)
// ============================================================

// Length-4 renormalization: collapse a 4-word unnormalized expansion to a
// non-overlapping length-4 QuadFloat.  Port of qd::renorm(c0,c1,c2,c3),
// QD 2.3.24 qd/include/qd/qd_inline.h:95-125.  Used by divide() and
// round_to_nearest_int().
XPMATH_INLINE_FUNCTION void renorm(float& c0, float& c1, float& c2, float& c3) {
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    if (detail::isinf(c0)) return;

    s0 = qf_quick_two_sum(c2, c3, c3);
    s0 = qf_quick_two_sum(c1, s0, c2);
    c0 = qf_quick_two_sum(c0, s0, c1);

    s0 = c0;
    s1 = c1;
    if (s1 != 0.0f) {
        s1 = qf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f)
            s2 = qf_quick_two_sum(s2, c3, s3);
        else
            s1 = qf_quick_two_sum(s1, c3, s2);
    } else {
        s0 = qf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f)
            s1 = qf_quick_two_sum(s1, c3, s2);
        else
            s0 = qf_quick_two_sum(s0, c3, s1);
    }
    c0 = s0; c1 = s1; c2 = s2; c3 = s3;
}

// Length-5 -> length-4 renormalization (the task's "renorm_4"): collapse a
// 5-word unnormalized accumulator (the natural output width of add/multiply/
// multiply_scalar/divide) to a non-overlapping length-4 QuadFloat.  Port of
// qd::renorm(c0,c1,c2,c3,c4), QD 2.3.24 qd_inline.h:127-177.
XPMATH_INLINE_FUNCTION void renorm_4(float& c0, float& c1, float& c2,
                                     float& c3, float& c4) {
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    if (detail::isinf(c0)) return;

    s0 = qf_quick_two_sum(c3, c4, c4);
    s0 = qf_quick_two_sum(c2, s0, c3);
    s0 = qf_quick_two_sum(c1, s0, c2);
    c0 = qf_quick_two_sum(c0, s0, c1);

    s0 = c0;
    s1 = c1;

    if (s1 != 0.0f) {
        s1 = qf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f) {
            s2 = qf_quick_two_sum(s2, c3, s3);
            if (s3 != 0.0f)
                s3 += c4;
            else
                s2 = qf_quick_two_sum(s2, c4, s3);
        } else {
            s1 = qf_quick_two_sum(s1, c3, s2);
            if (s2 != 0.0f)
                s2 = qf_quick_two_sum(s2, c4, s3);
            else
                s1 = qf_quick_two_sum(s1, c4, s2);
        }
    } else {
        s0 = qf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f) {
            s1 = qf_quick_two_sum(s1, c3, s2);
            if (s2 != 0.0f)
                s2 = qf_quick_two_sum(s2, c4, s3);
            else
                s1 = qf_quick_two_sum(s1, c4, s2);
        } else {
            s0 = qf_quick_two_sum(s0, c3, s1);
            if (s1 != 0.0f)
                s1 = qf_quick_two_sum(s1, c4, s2);
            else
                s0 = qf_quick_two_sum(s0, c4, s1);
        }
    }
    c0 = s0; c1 = s1; c2 = s2; c3 = s3;
}

// ============================================================
// QuadFloat struct
// ============================================================
struct QuadFloat {
    float f0, f1, f2, f3;

    XPMATH_INLINE_FUNCTION QuadFloat() : f0(0.0f), f1(0.0f), f2(0.0f), f3(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloat(float x) : f0(x), f1(0.0f), f2(0.0f), f3(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloat(float a0, float a1, float a2, float a3)
        : f0(a0), f1(a1), f2(a2), f3(a3) {}

    // Faithfully encode an FP64 value by successive FP32 splitting (Route-A,
    // length-4 analogue of ff_math.hpp's ffloat(double)). A double carries 53
    // bits, so two words suffice; the remaining words fall to 0 after the split.
    XPMATH_INLINE_FUNCTION QuadFloat(double x) {
        double r = x;
        float  c0 = (float)r; r -= (double)c0;
        float  c1 = (float)r; r -= (double)c1;
        float  c2 = (float)r; r -= (double)c2;
        float  c3 = (float)r;
        f0 = c0; f1 = c1; f2 = c2; f3 = c3;
    }

    XPMATH_INLINE_FUNCTION QuadFloat(const QuadFloat& o)
        : f0(o.f0), f1(o.f1), f2(o.f2), f3(o.f3) {}
    XPMATH_INLINE_FUNCTION QuadFloat& operator=(const QuadFloat& o) {
        f0=o.f0; f1=o.f1; f2=o.f2; f3=o.f3; return *this;
    }

    XPMATH_INLINE_FUNCTION float operator[](int i) const {
        return (i==0)?f0:(i==1)?f1:(i==2)?f2:f3;
    }

    // Factory: build a QuadFloat from the IEEE-754 bit patterns of its four
    // FP32 components. Safe on host (memcpy) and device (__int_as_float).
    static XPMATH_INLINE_FUNCTION QuadFloat from_bits(uint32_t b0, uint32_t b1,
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
        return QuadFloat(a0, a1, a2, a3);
    }

    XPMATH_INLINE_FUNCTION QuadFloat operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION QuadFloat operator+(QuadFloat b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloat operator-(QuadFloat b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloat operator*(QuadFloat b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloat operator/(QuadFloat b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloat operator*(float b) const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION QuadFloat operator+(float b) const { return add(*this, QuadFloat(b)); }
    XPMATH_INLINE_FUNCTION QuadFloat operator-(float b) const { return subtract(*this, QuadFloat(b)); }

    XPMATH_INLINE_FUNCTION QuadFloat& operator+=(QuadFloat b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloat& operator-=(QuadFloat b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloat& operator*=(QuadFloat b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloat& operator/=(QuadFloat b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(QuadFloat b) const {
        return f0==b.f0 && f1==b.f1 && f2==b.f2 && f3==b.f3;
    }
    XPMATH_INLINE_FUNCTION bool operator!=(QuadFloat b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(QuadFloat b) const {
        if (f0 != b.f0) return f0 < b.f0;
        if (f1 != b.f1) return f1 < b.f1;
        if (f2 != b.f2) return f2 < b.f2;
        return f3 < b.f3;
    }
    XPMATH_INLINE_FUNCTION bool operator>(QuadFloat b) const {
        if (f0 != b.f0) return f0 > b.f0;
        if (f1 != b.f1) return f1 > b.f1;
        if (f2 != b.f2) return f2 > b.f2;
        return f3 > b.f3;
    }
    XPMATH_INLINE_FUNCTION bool operator<=(QuadFloat b) const { return !(*this > b); }
    XPMATH_INLINE_FUNCTION bool operator>=(QuadFloat b) const { return !(*this < b); }
};

XPMATH_INLINE_FUNCTION QuadFloat operator+(float a, QuadFloat b) { return add(QuadFloat(a), b); }
XPMATH_INLINE_FUNCTION QuadFloat operator-(float a, QuadFloat b) { return subtract(QuadFloat(a), b); }
XPMATH_INLINE_FUNCTION QuadFloat operator*(float a, QuadFloat b) { return multiply_scalar(b, a); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const QuadFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific
       << d.f0 << ", " << d.f1 << ", " << d.f2 << ", " << d.f3 << "]";
    return os;
}
#endif

// ============================================================
// Constants (4×FP32 bit patterns).
// Auto-generated by scripts/gen_qf_constants.cpp — do not edit by hand.
// Successive-splitting of a 113-bit __float128 constant into 4 FP32 words;
// reconstruction rel_err < 6e-31 for every entry (well below u = 2^-96).
// ============================================================
XPMATH_INLINE_FUNCTION QuadFloat QuadFloat_pi          () { return QuadFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU, 0xa7772cedU, 0x19cc5170U); } // pi
XPMATH_INLINE_FUNCTION QuadFloat QuadFloat_e           () { return QuadFloat::from_bits(0x402df854U, 0x33b14577U, 0xa7559541U, 0x1ae2b101U); } // e
XPMATH_INLINE_FUNCTION QuadFloat QuadFloat_log2        () { return QuadFloat::from_bits(0x3f317218U, 0xb102e308U, 0xa4ca86c4U, 0x186ce601U); } // ln(2)
XPMATH_INLINE_FUNCTION QuadFloat QuadFloat_log10       () { return QuadFloat::from_bits(0x40135d8eU, 0xb309555dU, 0xa69f48adU, 0x9a129d48U); } // ln(10)
XPMATH_INLINE_FUNCTION QuadFloat QuadFloat_sqrt2       () { return QuadFloat::from_bits(0x3fb504f3U, 0x32cfe77aU, 0xa65bdd34U, 0x989d9323U); } // sqrt(2)
XPMATH_INLINE_FUNCTION QuadFloat QuadFloat_euler_gamma () { return QuadFloat::from_bits(0x3f13c468U, 0xb1e4127aU, 0x24f49a38U, 0x97e03f7fU); } // Euler gamma

// ============================================================
// Negation / absolute value
// ============================================================

// QD qd_inline.h:438-440 (operator-).
XPMATH_INLINE_FUNCTION QuadFloat negate(QuadFloat a) {
    return QuadFloat(-a.f0, -a.f1, -a.f2, -a.f3);
}

// QD qd_inline.h:788-790 (abs).
XPMATH_INLINE_FUNCTION QuadFloat abs(QuadFloat a) {
    return (a.f0 < 0.0f) ? negate(a) : a;
}

// ============================================================
// Addition
// ============================================================

// IEEE-style addition (satisfies the IEEE error bound).  Port of
// qd_real::ieee_add, QD 2.3.24 qd_inline.h:286-336, plus the quick_three_accum
// helper it calls (qd_inline.h:261-282).  NOT the default (QD builds with
// QD_IEEE_ADD off by default); provided for parity with the DD/FF story and for
// callers that need the tighter bound.
XPMATH_INLINE_FUNCTION float qf_quick_three_accum(float& a, float& b, float c) {
    float s;
    bool za, zb;
    s = qf_two_sum(b, c, b);
    s = qf_two_sum(a, s, a);
    za = (a != 0.0f);
    zb = (b != 0.0f);
    if (za && zb) return s;
    if (!zb) { b = a; a = s; }
    else     { a = s; }
    return 0.0f;
}

XPMATH_INLINE_FUNCTION QuadFloat ieee_add(QuadFloat a, QuadFloat b) {
    int i, j, k;
    float s, t;
    float u, v;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    i = j = k = 0;
    if (detail::fabs(a[i]) > detail::fabs(b[j])) u = a[i++]; else u = b[j++];
    if (detail::fabs(a[i]) > detail::fabs(b[j])) v = a[i++]; else v = b[j++];

    u = qf_quick_two_sum(u, v, v);

    while (k < 4) {
        if (i >= 4 && j >= 4) {
            x[k] = u;
            if (k < 3) x[++k] = v;
            break;
        }
        if (i >= 4)                                    t = b[j++];
        else if (j >= 4)                               t = a[i++];
        else if (detail::fabs(a[i]) > detail::fabs(b[j])) t = a[i++];
        else                                           t = b[j++];

        s = qf_quick_three_accum(u, v, t);
        if (s != 0.0f) x[k++] = s;
    }
    for (k = i; k < 4; k++) x[3] += a[k];
    for (k = j; k < 4; k++) x[3] += b[k];

    renorm(x[0], x[1], x[2], x[3]);
    return QuadFloat(x[0], x[1], x[2], x[3]);
}

// Cray-style ("sloppy") addition — the QD DEFAULT (QD_IEEE_ADD undefined).
// Port of qd_real::sloppy_add, QD 2.3.24 qd_inline.h:338-405 (the active,
// data-dependency-minimized form; QD documents it as identical to the
// commented two_sum version at qd_inline.h:339-354).  This is what
// add()/operator+ dispatch to, matching QD's out-of-the-box configuration.
// Safe at FP32: correctness rests on the inputs being non-overlapping
// expansions, not on the exponent range (see PORT_NOTES_QF).
XPMATH_INLINE_FUNCTION QuadFloat sloppy_add(QuadFloat a, QuadFloat b) {
    float s0, s1, s2, s3;
    float t0, t1, t2, t3;
    float v0, v1, v2, v3;
    float u0, u1, u2, u3;
    float w0, w1, w2, w3;

    s0 = a.f0 + b.f0;  s1 = a.f1 + b.f1;  s2 = a.f2 + b.f2;  s3 = a.f3 + b.f3;

    v0 = s0 - a.f0;    v1 = s1 - a.f1;    v2 = s2 - a.f2;    v3 = s3 - a.f3;
    u0 = s0 - v0;      u1 = s1 - v1;      u2 = s2 - v2;      u3 = s3 - v3;
    w0 = a.f0 - u0;    w1 = a.f1 - u1;    w2 = a.f2 - u2;    w3 = a.f3 - u3;

    u0 = b.f0 - v0;    u1 = b.f1 - v1;    u2 = b.f2 - v2;    u3 = b.f3 - v3;

    t0 = w0 + u0;      t1 = w1 + u1;      t2 = w2 + u2;      t3 = w3 + u3;

    s1 = qf_two_sum(s1, t0, t0);
    qf_three_sum(s2, t0, t1);
    qf_three_sum2(s3, t0, t2);
    t0 = t0 + t1 + t3;

    renorm_4(s0, s1, s2, s3, t0);
    return QuadFloat(s0, s1, s2, s3);
}

// QD default dispatch (qd_inline.h:408-414): operator+ -> sloppy_add.
XPMATH_INLINE_FUNCTION QuadFloat add(QuadFloat a, QuadFloat b) {
    return sloppy_add(a, b);
}

// QD qd_inline.h:459-461 (operator-): a - b == a + (-b).
XPMATH_INLINE_FUNCTION QuadFloat subtract(QuadFloat a, QuadFloat b) {
    return add(a, negate(b));
}

// ============================================================
// Multiplication
// ============================================================

// quad-float * float.  Port of operator*(qd_real, double),
// QD 2.3.24 qd_inline.h:490-514.
XPMATH_INLINE_FUNCTION QuadFloat multiply_scalar(QuadFloat a, float b) {
    float p0, p1, p2, p3;
    float q0, q1, q2;
    float s0, s1, s2, s3, s4;

    p0 = qf_two_prod(a.f0, b, q0);
    p1 = qf_two_prod(a.f1, b, q1);
    p2 = qf_two_prod(a.f2, b, q2);
    p3 = a.f3 * b;

    s0 = p0;
    s1 = qf_two_sum(q0, p1, s2);
    qf_three_sum(s2, q1, p2);
    qf_three_sum2(q1, q2, p3);
    s3 = q1;
    s4 = q2 + p2;

    renorm_4(s0, s1, s2, s3, s4);
    return QuadFloat(s0, s1, s2, s3);
}

// Exact multiplication by a power of two (no rounding).  QD qd_inline.h:485-487.
XPMATH_INLINE_FUNCTION QuadFloat mul_pwr2(QuadFloat a, float b) {
    return QuadFloat(a.f0 * b, a.f1 * b, a.f2 * b, a.f3 * b);
}

// quad-float * quad-float — "sloppy" multiplication, the QD DEFAULT
// (QD builds with QD_SLOPPY_MUL on).  16 partial products a_i*b_j; the leading
// 6 (weights u^0..u^2) are formed with exact two_prod, the O(u^3) terms are
// folded in scalar, and terms of weight u^4 and higher are dropped before
// renorm_4 collapses the length-5 accumulator to length-4.  Port of
// qd_real::sloppy_mul, QD 2.3.24 qd_inline.h:567-599.
XPMATH_INLINE_FUNCTION QuadFloat multiply(QuadFloat a, QuadFloat b) {
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3, q4, q5;
    float t0, t1;
    float s0, s1, s2;

    p0 = qf_two_prod(a.f0, b.f0, q0);

    p1 = qf_two_prod(a.f0, b.f1, q1);
    p2 = qf_two_prod(a.f1, b.f0, q2);

    p3 = qf_two_prod(a.f0, b.f2, q3);
    p4 = qf_two_prod(a.f1, b.f1, q4);
    p5 = qf_two_prod(a.f2, b.f0, q5);

    // Start accumulation.
    qf_three_sum(p1, p2, q0);

    // Six-three sum of (p2, q1, q2) and (p3, p4, p5).
    qf_three_sum(p2, q1, q2);
    qf_three_sum(p3, p4, p5);
    s0 = qf_two_sum(p2, p3, t0);
    s1 = qf_two_sum(q1, p4, t1);
    s2 = q2 + p5;
    s1 = qf_two_sum(s1, t0, t0);
    s2 += (t0 + t1);

    // O(u^3) terms: the nine remaining cross-products, folded in scalar.
    s1 += a.f0*b.f3 + a.f1*b.f2 + a.f2*b.f1 + a.f3*b.f0 + q0 + q3 + q4 + q5;
    renorm_4(p0, p1, s0, s1, s2);
    return QuadFloat(p0, p1, s0, s1);
}

// quad-float ^ 2.  Port of sqr(qd_real), QD 2.3.24 qd_inline.h:674-715.
XPMATH_INLINE_FUNCTION QuadFloat sqr(QuadFloat a) {
    // KI-9 genuine-overflow guard. The body forms `2.0f * a.f0 * a.f1` and
    // `2.0f * a.f0 * a.f3` left-to-right, so the DOUBLING overflows on its own
    // for |a.f0| > FLT_MAX/2 = 1.7e38, and inf * 0 (the usual value of a trailing
    // limb) is NaN — sqr(3e38) came back (inf, NaN, NaN, NaN). Whenever those
    // doublings can overflow, a.f0 * a.f0 has already overflowed, so the whole
    // square is out of range and ±inf is the complete answer. One test covers
    // every internal overflow site; NaN input propagates through it unchanged.
    float leading = a.f0 * a.f0;
    if (!detail::isfinite(leading)) return QuadFloat(leading, 0.0f, 0.0f, 0.0f);

    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3;
    float s0, s1;
    float t0, t1;

    p0 = qf_two_sqr(a.f0, q0);
    p1 = qf_two_prod(2.0f * a.f0, a.f1, q1);
    p2 = qf_two_prod(2.0f * a.f0, a.f2, q2);
    p3 = qf_two_sqr(a.f1, q3);

    p1 = qf_two_sum(q0, p1, q0);

    q0 = qf_two_sum(q0, q1, q1);
    p2 = qf_two_sum(p2, p3, p3);

    s0 = qf_two_sum(q0, p2, t0);
    s1 = qf_two_sum(q1, p3, t1);

    s1 = qf_two_sum(s1, t0, t0);
    t0 += t1;

    s1 = qf_quick_two_sum(s1, t0, t0);
    p2 = qf_quick_two_sum(s0, s1, t1);
    p3 = qf_quick_two_sum(t1, t0, q0);

    p4 = 2.0f * a.f0 * a.f3;
    p5 = 2.0f * a.f1 * a.f2;

    p4 = qf_two_sum(p4, p5, p5);
    q2 = qf_two_sum(q2, q3, q3);

    t0 = qf_two_sum(p4, q2, t1);
    t1 = t1 + p5 + q3;

    p3 = qf_two_sum(p3, t0, p4);
    p4 = p4 + q0 + t1;

    renorm_4(p0, p1, p2, p3, p4);
    return QuadFloat(p0, p1, p2, p3);
}

// ============================================================
// Division
// ============================================================

// quad-float / quad-float — "sloppy" long division, the QD DEFAULT
// (QD builds with QD_SLOPPY_DIV on).  Port of qd_real::sloppy_div,
// QD 2.3.24 qd/src/qd_real.cpp:693-712.
//
// NOTE ON ALGORITHM (source-fidelity, rule 6): the T3.0a task text describes
// divide as "Newton iteration, initial reciprocal from FP32 division, 3
// iterations". QD 2.3.24's qd_real::div is NOT Newton — it is classical long
// division: each quotient digit q_k = r[0]/b[0] contributes ~24 fresh bits
// (q0~24, q1~48, q2~72, q3~96), and the residual r is refined by a full
// QuadFloat multiply-subtract between digits. Four digits reach the ~96-bit
// QuadFloat width; the accurate variant adds a fifth digit + length-5 renorm.
// This header ports QD's actual routine and cites it; the discrepancy with the
// task text is recorded in PORT_NOTES_QF and the T3.0a report.
// KI-19.  Non-finite signalling.  The long division below is only meaningful
// when the QUOTIENT is a representable QuadFloat.  q0 = a.f0 / b.f0 is the
// leading quotient digit AND, to within one FP32 rounding, the whole answer, so
// classifying on q0 is both overflow-safe (IEEE division never traps) and
// exact about which of the three non-finite outcomes applies:
//
//   q0 = ±inf   the true quotient exceeds FLT_MAX (genuine overflow), or b = 0.
//               No finite QuadFloat exists — |value| of a QuadFloat is bounded
//               by |f0|(1+2^-24) — so ±inf IS the answer.  The old code carried
//               on: multiply_scalar(b, inf) = inf, subtract(a, inf) = -inf, and
//               renorm(inf, -inf, …) collapsed to NaN.  That is the whole of
//               KI-19: an isinf() branch downstream took the wrong path and the
//               caller never learned the magnitude had overflowed.
//   q0 = ±0     the true quotient is below the smallest FP32 subnormal (~7e-46;
//               anything above it divides to a subnormal, not to zero).  ±0 is
//               the answer, and IEEE's division sign rules give the right zero
//               — the long division used to lose it in renorm, which folds
//               -0.0 + 0.0 to +0.0.
//   q0 = NaN    0/0, inf/inf, or a NaN operand — genuinely undefined.  Only
//               these three reach NaN now.
//
// The one boundary case: a true quotient inside the top rounding interval
// [FLT_MAX, (2-2^-24)·2^127) rounds q0 up to inf and is reported as overflow.
// That band is 2^-24 relative wide at the very top of the format; a QuadFloat
// whose leading word is FLT_MAX cannot carry a meaningful tail anyway.
XPMATH_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b) {
    float q0, q1, q2, q3;
    QuadFloat r;

    q0 = a.f0 / b.f0;
    if (!detail::isfinite(q0) || q0 == 0.0f) return QuadFloat(q0);
    r = subtract(a, multiply_scalar(b, q0));

    q1 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q1));

    q2 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q2));

    q3 = r.f0 / b.f0;

    renorm(q0, q1, q2, q3);
    return QuadFloat(q0, q1, q2, q3);
}

// Long division by a plain FP32 scalar.  Same four-quotient-digit structure as
// divide() above and the same closing renorm; the only change is the residual
// correction. When the divisor is a single float, q_i * b is ONE exact product
// (qf_two_prod gives it as a non-overlapping (p, e) pair, so (p, e, 0, 0) is a
// valid QuadFloat), where divide() has to run a full four-word multiply_scalar.
//
// Motivation: the Taylor loops in exp/expm1/sincos/sinhcosh/atanh divide by a
// small loop integer on every iteration. Routing that through divide()
// promotes the integer to a QuadFloat and pays the full four-word division —
// the dominant cost in every QF transcendental. DD and FF have carried a
// divide_scalar since the DDFUN port; QF did not, and this closes that gap.
//
// Accuracy: identical algorithm to divide(), and the residual correction here
// is exact rather than merely faithful, so this is not a precision/speed
// trade. Validated against qf_accuracy_test / qf_property_test.
XPMATH_INLINE_FUNCTION QuadFloat divide_scalar(QuadFloat a, float b) {
    float q0, q1, q2, q3, p, e;
    QuadFloat r;

    q0 = a.f0 / b;
    if (!detail::isfinite(q0) || q0 == 0.0f) return QuadFloat(q0);   // KI-19; see divide()
    p = qf_two_prod(q0, b, e);  r = subtract(a, QuadFloat(p, e, 0.0f, 0.0f));
    q1 = r.f0 / b;  p = qf_two_prod(q1, b, e);  r = subtract(r, QuadFloat(p, e, 0.0f, 0.0f));
    q2 = r.f0 / b;  p = qf_two_prod(q2, b, e);  r = subtract(r, QuadFloat(p, e, 0.0f, 0.0f));

    q3 = r.f0 / b;

    renorm(q0, q1, q2, q3);
    return QuadFloat(q0, q1, q2, q3);
}

// Accurate long division (five quotient digits + length-5 renorm).  Port of
// qd_real::accurate_div, QD 2.3.24 qd_real.cpp:714-736. Not the default;
// provided for parity with QD and for tight-bound callers (T3.4).
XPMATH_INLINE_FUNCTION QuadFloat divide_accurate(QuadFloat a, QuadFloat b) {
    float q0, q1, q2, q3, q4;
    QuadFloat r;

    q0 = a.f0 / b.f0;
    if (!detail::isfinite(q0) || q0 == 0.0f) return QuadFloat(q0);   // KI-19; see divide()
    r = subtract(a, multiply_scalar(b, q0));
    q1 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q1));
    q2 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q2));
    q3 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q3));
    q4 = r.f0 / b.f0;

    renorm_4(q0, q1, q2, q3, q4);
    return QuadFloat(q0, q1, q2, q3);
}

// ============================================================
// Square root
// ============================================================

// QuadFloat square root — Heron's method (a Newton iteration on x^2 - a),
// each step doubling the number of correct digits: y = (1/2)(x + a/x).
// Port of fsqrt / sqrt(qd_real), QD 2.3.24 qd/src/qd_real.cpp:738-785.
//
// NOTE ON ALGORITHM (source-fidelity, rule 6): the T3.0a task text describes
// sqrt as "Newton iteration, initial reciprocal from FP32 division, same
// posture as divide" — that is the Karp reciprocal-Newton variant used by
// dd_real::sqrt (QD dd_real.cpp:47-72). QD 2.3.24's qd_real::sqrt is instead
// Heron's method (fsqrt), which needs a full QuadFloat divide per step. This
// header ports QD's actual qd_real::sqrt. Iteration count: the FP32 seed
// sqrt(a.f0) is accurate to ~24 bits; Heron doubles precision per step
// (24 -> 48 -> 96, saturating at the ~96-bit QuadFloat width), so 3 iterations
// suffice. QD's loop runs up to 10 with an early-out convergence test; the port
// keeps that structure with eps = 2^-96 so it stops after ~3 on real inputs.
XPMATH_INLINE_FUNCTION QuadFloat sqrt(QuadFloat a) {
    if (a.f0 == 0.0f && a.f1 == 0.0f && a.f2 == 0.0f && a.f3 == 0.0f)
        return QuadFloat(0.0f);
    if (a.f0 < 0.0f) {
        XPMATH_PRINTF("QFSQRT: negative argument\n");
        return QuadFloat(0.0f);
    }

    const float eps = 1.2621774e-29f; // ~= 2^-96, QuadFloat unit roundoff
    const QuadFloat half(0.5f);

    QuadFloat x = QuadFloat(detail::sqrt(a.f0));  // ~24-bit FP32 seed
    for (int i = 0; i < 10; ++i) {
        QuadFloat y    = multiply(half, add(x, divide(a, x)));
        QuadFloat diff = subtract(x, y);
        x = y;
        float e = detail::fabs(((diff.f3 + diff.f2) + diff.f1) + diff.f0);
        if (e < detail::fabs(x.f0) * eps)
            return x;
    }
    return x;
}

// ============================================================
// Nearest integer
// ============================================================

// Nearest FP32-int of a single float, ties toward +infinity.
// QD 2.3.24 qd/include/qd/inline.h:116-120 (nint(double)), at FP32.
// NOTE (PORT_NOTES_QF): QD's nint does NOT use the 2^(2p-1) magic-constant
// trick that broke FF's ffnint at FP32 (PORT_NOTES §4b).
//
// KI-2 (fixed 2026-09-02, see docs/KNOWN_ISSUES.md). QD's literal formulation is
// `floor(d + 0.5)`, and the ADD is not exact: for d = 0.49999997f the FP32 sum
// d + 0.5f rounds up to exactly 1.0f, so floor returns 1 where the nearest
// integer is 0. A wrong answer, not a precision shortfall, and FP64 QD has it
// too. `rint` performs the same rounding in one instruction with no intermediate
// to double-round, so the hazard cannot arise.
//
// The `d - r == 0.5f` line restores QD's TIE DIRECTION. rint is ties-to-EVEN;
// QD's floor(d + 0.5) is ties-toward-+INFINITY (note: toward +inf, not away from
// zero — floor(-2.5 + 0.5) = -2). The tie rule is deliberately left alone: it is
// user-visible through `qf::round`, whose oracle in qf_accuracy_test.cpp is
// `roundq` with exact ties excluded from the corpus, and the multi-word tie
// correction below (`x0 - a.f0 == 0.5f && a.f1 < 0`) assumes the leading word
// rounded UP. KI-2 is the near-tie wrong answer; the tie direction is a
// convention, and changing it is not part of closing KI-2.
//
// KI-20 (2026-09-04) settled that convention, and it is NOT this one.
// `qf::round` is IEEE 754 half-even; this routine and its multi-word caller
// `round_to_nearest_int` stay ties-toward-+infinity, DELIBERATELY, and are
// internal. Two reasons, both about correctness rather than taste: the
// correction below is only valid for a fixed lean (half-even on a limb needs the
// parity of the accumulated integer, which no single limb carries), and the only
// other callers are the sincos/exp argument reductions, where a tie is
// measure-zero and either neighbour reduces equally well. `round` supplies the
// half-even step on top; see its comment. Do not merge the two names.
//
// Exactness of `d - r`: r is an integer and |d - r| <= 0.5, so for |d| < 2^23
// the difference is exact (both operands are multiples of ulp(d) <= 0.5), and
// for |d| >= 2^23 there are no non-integers at all, so r == d and the test is
// false. No spurious tie is possible.
//
// The zero-sign line keeps the change MINIMAL. rint(d) for d in (-0.5, 0)
// returns -0.0f, where floor(d + 0.5f) returned +0.0f; that is 2^30 floats
// whose sign bit would flip, and the sign of a zero reduction quotient is
// observable downstream (the sincos argument reduction feeds it back through a
// multiply). Restoring +0.0 there — while leaving nint(-0.0) = -0.0, which QD's
// `d == floor(d)` short-circuit also gave — makes this fix provably minimal:
// verified exhaustively over all 2^32 floats, qf_nint differs from QD's
// floor(d + 0.5f) on EXACTLY ONE input, 0.49999997f, where the old value was
// wrong and the new one is right, and on no input is the new value wrong.
XPMATH_INLINE_FUNCTION float qf_nint(float d) {
    float r = detail::rint(d);
    if (d - r == 0.5f) r += 1.0f;
    if (r == 0.0f && d != 0.0f) r = 0.0f;   // (-0.5, 0) -> +0, as QD gave
    return r;
}

// Nearest integer of a QuadFloat, component-wise with half-integer tie
// corrections and a final renorm.  Port of nint(qd_real),
// QD 2.3.24 qd/src/qd_real.cpp:48-86.
XPMATH_INLINE_FUNCTION QuadFloat round_to_nearest_int(QuadFloat a) {
    float x0, x1, x2, x3;
    x0 = qf_nint(a.f0);
    x1 = x2 = x3 = 0.0f;

    if (x0 == a.f0) {
        x1 = qf_nint(a.f1);
        if (x1 == a.f1) {
            x2 = qf_nint(a.f2);
            if (x2 == a.f2) {
                x3 = qf_nint(a.f3);
            } else {
                if (detail::fabs(x2 - a.f2) == 0.5f && a.f3 < 0.0f) x2 -= 1.0f;
            }
        } else {
            if (detail::fabs(x1 - a.f1) == 0.5f && a.f2 < 0.0f) x1 -= 1.0f;
        }
    } else {
        if (detail::fabs(x0 - a.f0) == 0.5f && a.f1 < 0.0f) x0 -= 1.0f;
    }

    renorm(x0, x1, x2, x3);
    return QuadFloat(x0, x1, x2, x3);
}

// ============================================================
// Integer power
// ============================================================

// a^n by binary exponentiation using sqr.  Port of pow(qd_real, int),
// QD 2.3.24 qd/src/qd_real.cpp:568-598.
XPMATH_INLINE_FUNCTION QuadFloat pow_int(QuadFloat a, int n) {
    if (n == 0) return QuadFloat(1.0f);

    QuadFloat r = a;              // odd-case multiplier
    QuadFloat s = QuadFloat(1.0f); // running answer
    int N = (n < 0) ? -n : n;

    if (N > 1) {
        while (N > 0) {
            if (N % 2 == 1) s = multiply(s, r);
            N /= 2;
            if (N > 0) r = sqr(r);
        }
    } else {
        s = r;
    }

    if (n < 0) return divide(QuadFloat(1.0f), s);
    return s;
}

// ============================================================
// Exp / Log family
// ============================================================
//
// SOURCE-FIDELITY NOTE (transcendental block, rule 6). QD 2.3.24's actual
// transcendentals in qd/src/qd_real.cpp are TABLE-BASED: exp (qd_real.cpp:925)
// uses a 15-entry inv_fact[] Taylor table plus 16 squarings; sin/cos/sincos
// (qd_real.cpp:2136-2360) reduce mod pi/2 then mod pi/1024 and look up 256-entry
// sin_table/cos_table of sin/cos(k*pi/1024). The QF port instead uses the
// TABLE-FREE structure of the sibling dd_math.hpp / ff_math.hpp headers
// (divide-by-k Taylor, joint sin/cos doublings), for three reasons: (1) the
// T3.0b task text + PORT_NOTES.md §3a explicitly direct "joint sin/cos
// doublings" and a divide-by-k Taylor with "more terms than QD's FP64 version";
// (2) 256-entry 4xFP32 tables are device-hostile (constant memory / register
// pressure under CUDA) whereas dd/ff are the designated portable multi-word
// references; (3) it keeps QF byte-consistent in *style* with its DD/FF
// siblings. Each function below still cites the QD 2.3.24 routine it mirrors
// mathematically, and flags where it follows dd/ff structure instead. The
// argument-reduction and Newton skeletons (log-Newton, atan2-Newton) ARE ported
// faithfully from QD and cited. sin_table/cos_table/inv_fact are therefore NOT
// generated — see docs/PORT_NOTES_QF.md.

// e^a.  Mathematical mirror of exp(qd_real), QD 2.3.24 qd_real.cpp:925-983
// (same reduce-by-m*log2 / scale-by-2^-nq / Taylor / square-nq-times skeleton),
// but with the table-free divide-by-k Taylor of dd_math.hpp:345 / ff_math.hpp:347
// rather than QD's inv_fact[] table.
//
// EXP TERM-COUNT DERIVATION (T3.0b deliverable). After reduction |s0| <= log2/2
// = 0.347; scaling by 2^-nq gives |r| <= 0.347/2^nq. The divide-by-k Taylor
// e^r = sum_k r^k/k! must reach the QF unit roundoff u = 2^-96 ~= 1.3e-29:
// need |r|^N / N! < u.  With nq = 6 (|r| <= 5.4e-3): N = 11 terms suffice
// (5.4e-3^11 / 11! ~= 1e-30 < u).  QD's FP64 exp uses nq = 16 squarings + a
// 15-entry inv_fact table for its ~64-digit target; QF needs FAR fewer terms
// (11 vs QD's effective ~9 at a 4096x finer reduction) and NO factorial table,
// because dividing by k each step accumulates 1/k! directly.  nq = 6 (matching
// dd_math.hpp) balances squaring cost against Taylor length; the loop caps at 60
// and, unlike ff_math.hpp:376, does NOT return 0 on the cap (that FF behavior
// is a latent stall bug — see PORT_NOTES_QF; here we proceed with the best sum).
XPMATH_INLINE_FUNCTION QuadFloat exp(QuadFloat a) {
    const int   nq  = 6;
    // eps is deliberately COARSER than QF's resolution u = 2^-96 ~= 1.3e-29.
    // ff_math.hpp used eps = 1e-15f finer than FloatFloat resolution 3.55e-15,
    // which made the term never fall below eps*sum -> spurious stalls / 0-returns
    // (the FF exp-eps bug). 1e-28f > u keeps convergence reachable at QF width.
    const float eps = 1.0e-28f;
    QuadFloat al2 = QuadFloat_log2();
    // KI-6: word-range-derived guard, mirroring ff_math.hpp. e^a exceeds FLT_MAX
    // above ln(FLT_MAX) = 88.722839 and drops below the smallest FP32 subnormal
    // below -103.28; in between the answer is representable and is returned.
    // The old symmetric ±88 pair flushed both tails to zero.
    if (a.f0 > 88.722839f) {
        XPMATH_PRINTF("QFEXP: overflow\n");
        return QuadFloat(HUGE_VALF);            // e^a > FLT_MAX: +inf is the answer
    }
    if (a.f0 < -104.0f) return QuadFloat(0.0f); // e^a < 2^-149: 0 is the answer

    QuadFloat s0 = divide(a, al2);
    QuadFloat s1 = round_to_nearest_int(s0);
    float t1  = s1.f0;
    int   nz  = (int)(t1 + detail::copysign(1.0e-6f, t1));
    s0 = subtract(a, multiply(al2, s1));            // |s0| <= log2/2

    if (s0.f0 == 0.0f && s0.f1 == 0.0f) {
        return QuadFloat(ldexpf(1.0f, nz));         // result = 2^nz exactly
    }
    // Scale down by 2^nq (exact via mul_pwr2, no Dekker splitter), Taylor, then
    // square nq times: e^r squared nq times = e^(2^nq r) = e^s0.
    // KI-34: the series and the squarings track e^r - 1, not e^r, so the
    // squaring step is (1+s)^2 - 1 = s*(s+2), which PRESERVES relative error
    // instead of doubling it. See dd_math.hpp's exp for the derivation. nq = 6
    // here, so the shipped form multiplied the series error by 64 and left
    // `log` an absolute floor of ~9.3 units of 2^-96, flat in |ln v|.
    s1 = mul_pwr2(s0, ldexpf(1.0f, -nq));           // r = s0 / 2^nq
    QuadFloat s2 = s1, s3 = s1;                     // term = r, sum = e^r - 1
    for (int l1 = 2; l1 <= 60; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (float)l1);      // term = r^l1 / l1!
        s3 = add(s3, s2);
        if (detail::fabs(s2.f0) <= eps * detail::fabs(s3.f0)) break;
        // NOTE: no return-0 on l1 == 60 (see header comment); fall through with s3.
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, add(s3, QuadFloat(2.0f)));
    s3 = add(QuadFloat(1.0f), s3);

    // Final scaling by 2^nz.  PORT_NOTES §4a: power-of-2 multiplication is exact
    // in FP32 and must NOT go through multiply_scalar (which would compute
    // 8193*2^nz inside Dekker splitting and overflow FP32 for nz >= 116).  Scale
    // each component directly.  This is also QD's approach (ldexp(s, m),
    // qd_real.cpp:982, which is component-wise std::ldexp) — and KI-6 is why the
    // component-wise form matters rather than materialising `2^nz` as a float
    // first: that factor is +inf for nz >= 128 and 0 for nz <= -150, which put
    // both ends of the FP32 range out of reach.
    if (nz >= -125 && nz <= 127) {          // 2^nz exact and normal: cheap path
        const float pow2 = ldexpf(1.0f, nz);
        return QuadFloat(s3.f0 * pow2, s3.f1 * pow2, s3.f2 * pow2, s3.f3 * pow2);
    }
    return QuadFloat(ldexpf(s3.f0, nz), ldexpf(s3.f1, nz),
                     ldexpf(s3.f2, nz), ldexpf(s3.f3, nz));
}

// log(a) via Newton's iteration on f(x) = exp(x) - a.  Faithful port of
// log(qd_real), QD 2.3.24 qd_real.cpp:986-1011: seed x = log(a.f0), then
// x <- x + a*exp(-x) - 1 three times (Newton ~doubles correct digits per step;
// FP32 seed ~24 bits -> 48 -> 96, saturating at QF width on the 3rd).  Same
// three-step structure as dd_math.hpp:380.
XPMATH_INLINE_FUNCTION QuadFloat log(QuadFloat a) {
    if (detail::isinf(a.f0)) return a;   // KI-6, see dd_math.hpp
    if (a.f0 <= 0.0f) {
        XPMATH_PRINTF("QFLOG: non-positive argument\n");
        return QuadFloat(0.0f);
    }
    if (a.f0 == 1.0f && a.f1 == 0.0f && a.f2 == 0.0f && a.f3 == 0.0f)
        return QuadFloat(0.0f);
    QuadFloat x = QuadFloat(detail::log(a.f0));     // ~24-bit FP32 seed
    for (int k = 0; k < 3; ++k) {
        // KI-23: evaluate exp at a NON-NEGATIVE argument wherever the format
        // allows.  e^{|x|} >= 1 keeps all four words normal; e^{-|x|} pushes the
        // low words below the smallest FP32 subnormal (2^-149) and truncates
        // them, losing one bit of the result per bit of the argument's exponent.
        // QD's single form  x += a*e^{-x} - 1  (qd_real.cpp:1007-1009) is the
        // e^{-|x|} branch, and is the reason QF alone shed ~11 digits above
        // 1e29 while DD/FF/TF -- which all use the residual form -- were clean.
        // QD's form stays the DEFAULT for QF -- measured, it beats the residual
        // form across the whole mid-range (the sweep's QF log/asinh/acosh/pow
        // cells lose up to 6.77 digits if the residual form is used there),
        // which is presumably why QD chose it: QF's divide is a long division,
        // and the residual form needs one per Newton step.  DD/FF/TF have the
        // opposite preference and keep the residual form as their default.
        //
        // Switch to the residual form only where e^{-x} underflows: f3 sits at
        // 2^(E-72) for a value of magnitude 2^E, so it leaves the FP32 normal
        // range 2^-126 once E < -54, i.e. x > 54*ln2 = 37.4 (a > 1.7e16).  That
        // is exactly the KI-23 band.  Also take it below -ln(FLT_MAX), where
        // e^{-x} = 1/a overflows and the residual form is the only finite one.
        if (x.f0 > 37.4f || x.f0 < -88.722839f) {
            QuadFloat e = exp(x);                               // e^{x} normal
            x = add(x, divide(subtract(a, e), e));
        } else {
            QuadFloat e = exp(negate(x));                       // e^{-x} normal
            x = subtract(add(x, multiply(a, e)), QuadFloat(1.0f));
        }
    }
    return x;
}

// log10(a) = log(a) / log(10).  QD qd_real.cpp:1025 (log10 = log(a)/_log10).
XPMATH_INLINE_FUNCTION QuadFloat log10(QuadFloat a) {
    return divide(log(a), QuadFloat_log10());
}

// log2(a) = log(a) / log(2).  QD has no log2; composition (cf. dd_math.hpp:396).
XPMATH_INLINE_FUNCTION QuadFloat log2(QuadFloat a) {
    return divide(log(a), QuadFloat_log2());
}

// log1p(a) = log(1 + a).  QD has no log1p.  Series 2*atanh(a/(2+a)) for
// |a| < 1/4, plain log(1+a) outside; see dd_math.hpp's log1p for the derivation
// (KI-5(b)), including why Goldberg's correction was measured and rejected.
// The old body was the plain composition log(1+a), which loses log10(1/|a|)
// digits for small a.
XPMATH_INLINE_FUNCTION QuadFloat log1p(QuadFloat a) {
    if (detail::fabs(a.f0) < 0.25f) {
        QuadFloat t   = divide(a, add(QuadFloat(2.0f), a));
        QuadFloat t2  = multiply(t, t);
        QuadFloat sum = t;
        QuadFloat trm = t;
        for (int k = 3; k < 80; k += 2) {
            trm = multiply(trm, t2);
            QuadFloat incr = divide(trm, QuadFloat((float)k));
            sum = add(sum, incr);
            if (detail::fabs(incr.f0) <= 1.0e-32f * detail::fabs(sum.f0)) break;
        }
        return multiply_scalar(sum, 2.0f);
    }
    return log(add(QuadFloat(1.0f), a));
}

// exp2(a) = e^(a*ln2).  QD has no exp2; composition (cf. dd_math.hpp:409).
XPMATH_INLINE_FUNCTION QuadFloat exp2(QuadFloat a) {
    return exp(multiply(a, QuadFloat_log2()));
}

// exp10(a) = e^(a*ln10).  QD has no exp10; composition (cf. dd_math.hpp:413).
XPMATH_INLINE_FUNCTION QuadFloat exp10(QuadFloat a) {
    return exp(multiply(a, QuadFloat_log10()));
}

// expm1(a) = e^a - 1.  QD has no expm1; Taylor for |a| <= 0.5 to avoid the
// e^a - 1 cancellation near 0, else exp(a) - 1 (cf. dd_math.hpp:417 / ff:424).
XPMATH_INLINE_FUNCTION QuadFloat expm1(QuadFloat a) {
    const float eps = 1.0e-28f;
    if (detail::fabs(a.f0) > 0.5f) {
        return subtract(exp(a), QuadFloat(1.0f));
    }
    QuadFloat sum = a, term = a;
    for (int k = 2; k <= 60; ++k) {
        term = divide_scalar(multiply(term, a), (float)k);
        sum  = add(sum, term);
        if (detail::fabs(term.f0) < eps * detail::fabs(sum.f0)) break;
    }
    return sum;
}

// ============================================================
// Trig — joint sin/cos through the doublings (PORT_NOTES §3a)
// ============================================================

// sincos(a): writes sin_a = sin(a), cos_a = cos(a).  Mathematical mirror of
// sincos(qd_real), QD 2.3.24 qd_real.cpp:2298-2360 (same reduce-mod-2pi
// skeleton), BUT structured like ff_math.hpp:445 / dd_math.hpp:439 — a
// divide-by-k Taylor on r = s3/2^nq followed by nq angle-doublings — instead of
// QD's pi/1024 table lookup (see block header).  PORT_NOTES §3a: sin and cos are
// tracked JOINTLY through the doublings (sin(2x)=2 sin x cos x,
// cos(2x)=cos^2 x - sin^2 x) so no sqrt(1-cos^2) recovery loses relative
// precision near multiples of pi.  QF's 4-word _2pi (accurate to ~2^-96) makes
// the mod-2pi reduction good enough that near-pi sin/cos are distinguishable
// from noise (the T3.6 goal FF §5 could not reach with 2-word pi).
XPMATH_INLINE_FUNCTION void sincos(QuadFloat a, QuadFloat& sin_a, QuadFloat& cos_a) {
    const int   itrmx = 100, nq = 5;
    const float eps = 1.0e-28f;
    if (a.f0 == 0.0f) { sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return; }
    // KI-12 residual.  Small-argument short circuit over the degenerate
    // reduction band only; full derivation at ff_math.hpp:sincos.  QF's limbs
    // are FP32 too, so the band is the same shape: below 2^nq * FLT_MIN the
    // leading word of r = s3/2^nq is subnormal and sheds bits before the first
    // Taylor term (and is 0 outright for subnormal |a|).  Measured before:
    // QF sin(1e-40) = 9.99967e-41 for an argument of 9.99995e-41, and
    // QF sin(1e-44) = 0.  Above the band the series is already correct and a
    // wider cut was measured to cost complex-op digits -- see ff_math.hpp.
    if (detail::fabs(a.f0) < (float)(1 << nq) * 1.17549435e-38f /* 2^nq*FLT_MIN */) {
        sin_a = a; cos_a = QuadFloat(1.0f); return;
    }
    if (detail::fabs(a.f0) >= 1.0e30f) {
        XPMATH_PRINTF("QFCSSNR: argument too large\n");
        sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return;   // KI-26
    }
    // Reduce mod 2pi (QD qd_real.cpp:2306-2308: z = nint(a/2pi); t = a - 2pi*z).
    QuadFloat pi2 = mul_pwr2(QuadFloat_pi(), 2.0f);   // 2pi, exact from 4-word pi
    QuadFloat s1  = divide(a, pi2);
    QuadFloat s2  = round_to_nearest_int(s1);
    QuadFloat s3  = subtract(a, multiply(pi2, s2));   // |s3| <= pi
    if (s3.f0 == 0.0f) { sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return; }

    QuadFloat r  = mul_pwr2(s3, ldexpf(1.0f, -nq));   // r = s3 / 2^nq, |r| < pi/2^nq
    QuadFloat r2 = multiply(r, r);

    // sin(r) = r - r^3/3! + ... ; cos(r) = 1 - r^2/2! + ...
    QuadFloat sin_r = r,             cos_r = QuadFloat(1.0f);
    QuadFloat sterm = r,             cterm = QuadFloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (detail::fabs(sterm.f0) < eps * detail::fabs(sin_r.f0) &&
            detail::fabs(cterm.f0) < eps) break;
        // No return on itrmx (converges in ~9 terms at nq=5); fall through.
    }

    // Doublings: sin(2x)=2 sin x cos x, cos(2x)=cos^2 x - sin^2 x (PORT_NOTES §3a).
    for (int j = 0; j < nq; ++j) {
        QuadFloat new_sin = mul_pwr2(multiply(sin_r, cos_r), 2.0f);
        QuadFloat new_cos = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }

    // KI-26 codomain guard: outside the slack band -> identity point
    // (sin, cos) = (0, 1), inside it -> clamp, so |sin| <= 1 and |cos| <= 1 hold
    // exactly for every finite input.  Full rationale at dd_math.hpp:sincos.
    // Note the out-param order here is (sin, cos), the reverse of dd/ff.
    const float kSlack = 1.0009765625f;   // 1 + 2^-10
    if (!(detail::fabs(sin_r.f0) <= kSlack) || !(detail::fabs(cos_r.f0) <= kSlack)) {
        XPMATH_PRINTF("QFCSSNR: argument reduction under-determined\n");
        sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return;
    }
    if (sin_r.f0 >  1.0f || (sin_r.f0 ==  1.0f && sin_r.f1 > 0.0f)) sin_r = QuadFloat( 1.0f);
    if (sin_r.f0 < -1.0f || (sin_r.f0 == -1.0f && sin_r.f1 < 0.0f)) sin_r = QuadFloat(-1.0f);
    if (cos_r.f0 >  1.0f || (cos_r.f0 ==  1.0f && cos_r.f1 > 0.0f)) cos_r = QuadFloat( 1.0f);
    if (cos_r.f0 < -1.0f || (cos_r.f0 == -1.0f && cos_r.f1 < 0.0f)) cos_r = QuadFloat(-1.0f);

    sin_a = sin_r; cos_a = cos_r;
}

// tan(a) = sin(a)/cos(a).  QD qd_real.cpp:2473 (sincos then s/c).
XPMATH_INLINE_FUNCTION QuadFloat sin(QuadFloat a) {
    QuadFloat s, c; sincos(a, s, c); return s;
}
XPMATH_INLINE_FUNCTION QuadFloat cos(QuadFloat a) {
    QuadFloat s, c; sincos(a, s, c); return c;
}
XPMATH_INLINE_FUNCTION QuadFloat tan(QuadFloat a) {
    QuadFloat s, c; sincos(a, s, c); return divide(s, c);
}

// angle(x, y) = atan2(y, x).  Mathematical mirror of atan2(qd_real,qd_real),
// QD 2.3.24 qd_real.cpp:2393-2460: normalize (x,y) onto the unit circle, seed
// with the FP32 std::atan2, then Newton-refine z += (y - sin z)/cos z (or the
// cos variant when |x|>|y|), 3 iterations.  dd_math.hpp:497 uses the same
// structure; the joint sincos above supplies (sin z, cos z) per iteration.
namespace detail {
// KI-8.  Exact power-of-two scale factor: returns s = 2^-e with |m|*s in [1,2),
// so that s and 1/s are both exactly representable and scaling every word of an
// expansion by either loses no bit.  Mirrors detail::ff_pow2_unit_scale in
// ff_math.hpp, which carries the full argument.
//
// e is clamped to [-125, 125] so both s and 1/s stay NORMAL floats; at the
// clamp the operand is not brought all the way to [1,2), but the square is
// still inside the word range, which is all the caller needs.
//
// No frexp: config.hpp's scalar dispatch does not carry one, and a
// dependency-free loop is portable to every device backend.  It runs only
// outside the direct band, never on the hot path.
XPMATH_INLINE_FUNCTION float qf_pow2_unit_scale(float m) {
    float t = (m < 0.0f) ? -m : m;
    if (!(t > 0.0f) || detail::isinf(t)) return 1.0f;   // 0/inf/nan: loop would not terminate
    int e = 0;
    while (t >= 16777216.0f)   { t *= 5.9604644775390625e-8f; e += 24; }
    while (t <  5.9604644775390625e-8f) { t *= 16777216.0f;   e -= 24; }
    while (t >= 2.0f) { t *= 0.5f; ++e; }
    while (t <  1.0f) { t *= 2.0f;  --e; }
    if (e > 125) e = 125;
    if (e < -125) e = -125;
    return ldexpf(1.0f, -e);
}
XPMATH_INLINE_FUNCTION QuadFloat qf_pow2_scale(QuadFloat a, float s) {
    return QuadFloat(a.f0 * s, a.f1 * s, a.f2 * s, a.f3 * s);
}
}  // namespace detail

// Band in which a sum of squares may be formed DIRECTLY, shared by every QF
// site that forms one (hypot here; complex abs and complex operator/ in
// qf_complex.hpp).  Low edge is 2^-25, one binade above the derived
// 2^-27 word-underflow limit of KI-8 note (1) at ff_math.hpp's hypot; the
// high edge is unchanged from the original fix.
namespace detail {
inline constexpr float kQFSqLo = 7.4505806e-9f;
inline constexpr float kQFSqHi = 1.0e18f;
}

XPMATH_INLINE_FUNCTION QuadFloat angle(QuadFloat x, QuadFloat y) {
    QuadFloat pi = QuadFloat_pi();
    if (x.f0 == 0.0f && y.f0 == 0.0f) return QuadFloat(0.0f);
    if (x.f0 == 0.0f) return (y.f0 > 0.0f) ? mul_pwr2(pi, 0.5f) : mul_pwr2(pi, -0.5f);
    if (y.f0 == 0.0f) return (x.f0 > 0.0f) ? QuadFloat(0.0f) : pi;
    // KI-13.  The r below is a sum of squares and has exactly hypot's exposure:
    // it overflowed the word above |x| ~ 1.8e19 (atan(3.16e19) returned NaN, and
    // atan2/asin/acos/complex arg inherited it) and shed low words below the
    // derived 2^-27 (see the KI-8 note at ff_math.hpp's hypot).  angle is
    // EXACTLY scale-invariant -- atan2(ys, xs) = atan2(y, x) for any s > 0 --
    // so the remedy here is one power-of-two rescale of both operands, which
    // changes no bit of the answer and puts the square back inside the word
    // range.
    {
        const float mx = detail::fabs(x.f0), my = detail::fabs(y.f0);
        const float mm = (mx > my) ? mx : my;
        // HIGH side only.  The low side was tried and reverted: r is used only
        // to normalise (x/r, y/r) before a 3-step Newton refinement on
        // atan2, and that refinement re-evaluates sin/cos of the iterate, so
        // low words shed from r do not survive into the answer -- rescaling
        // there only perturbs the seed.  Measured on the 428,592-point sweep:
        // a two-sided gate cost 45 regressions (worst 1.27 digits, QF c atan
        // points 1254/1318) to buy 35 improvements.  The high side is a real
        // fix -- without it the square OVERFLOWS and atan/atan2/asin/acos/arg
        // return 0 or NaN outright.
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
        sincos(a, sin_a, cos_a);
        if (use_x) {
            // Newton on cos: z' = z - (x - cos z)/(-sin z) -> a -= (target-cos)/sin
            a = subtract(a, divide(subtract(target, cos_a), sin_a));
        } else {
            a = add(a, divide(subtract(target, sin_a), cos_a));
        }
    }
    return a;
}

// asin(a) = atan2(a, sqrt(1-a^2)).  QD qd_real.cpp:2479.
// KI-16.  Compare the VALUE, not the leading word.  Full derivation at
// dd_math.hpp's dd_cmp_one: `a.f0` is the value rounded to one FP32, and near a
// domain edge at +-1 that rounding crosses the edge in both directions -- it
// rejects legal arguments as far inside as 1 - 2.2e-16 (KI-16's reported QF
// `atanh` symptom, 42 of its 46 points) and accepts illegal ones just outside.
// `subtract` renormalises and, for |a| in [1/2, 2], cancels the leading words
// EXACTLY by Sterbenz, so the residual words decide and decide correctly.
// Returns -1, 0, +1 for a < 1, a == 1, a > 1.
XPMATH_INLINE_FUNCTION int qf_cmp_one(QuadFloat a) {
    const QuadFloat d = subtract(a, QuadFloat(1.0f));
    if (d.f0 > 0.0f) return  1;
    if (d.f0 < 0.0f) return -1;
    return 0;
}
XPMATH_INLINE_FUNCTION int qf_cmp_abs_one(QuadFloat a) { return qf_cmp_one(abs(a)); }

XPMATH_INLINE_FUNCTION QuadFloat asin(QuadFloat a) {
    if (qf_cmp_abs_one(a) > 0) {                                        // KI-16
        XPMATH_PRINTF("QFASIN: argument out of range\n");
        return QuadFloat(0.0f);
    }
    QuadFloat t = sqrt(subtract(QuadFloat(1.0f), multiply(a, a)));
    return angle(t, a);
}
// acos(a) = atan2(sqrt(1-a^2), a).  QD qd_real.cpp:2494.
XPMATH_INLINE_FUNCTION QuadFloat acos(QuadFloat a) {
    if (qf_cmp_abs_one(a) > 0) {                                        // KI-16
        XPMATH_PRINTF("QFACOS: argument out of range\n");
        return QuadFloat(0.0f);
    }
    QuadFloat t = sqrt(subtract(QuadFloat(1.0f), multiply(a, a)));
    return angle(a, t);
}
// atan(a) = atan2(a, 1).  QD qd_real.cpp:2389.
// KI-25 codomain clamp -- rationale in dd_math.hpp atan().  QF is the backend
// where the overshoot was actually measured (+4.73e-30 at a = 3.4e38).  atan2
// is not clamped; its range is (-pi, pi].
XPMATH_INLINE_FUNCTION QuadFloat atan(QuadFloat a) {
    QuadFloat r = angle(QuadFloat(1.0f), a);
    const QuadFloat p = QuadFloat_pi();
    const QuadFloat h(p.f0 * 0.5f, p.f1 * 0.5f, p.f2 * 0.5f, p.f3 * 0.5f);
    const QuadFloat ar = (r.f0 < 0.0f) ? QuadFloat(-r.f0, -r.f1, -r.f2, -r.f3) : r;
    if (subtract(ar, h).f0 > 0.0f)
        return (r.f0 < 0.0f) ? QuadFloat(-h.f0, -h.f1, -h.f2, -h.f3) : h;
    return r;
}
// atan2(y, x) = angle(x, y).  QD qd_real.cpp:2393 (STL argument order).
XPMATH_INLINE_FUNCTION QuadFloat atan2(QuadFloat y, QuadFloat x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic
// ============================================================

// sinhcosh(a): writes sinh_a, cosh_a.  Mathematical mirror of sinh/cosh(qd_real),
// QD 2.3.24 qd_real.cpp:2509-2545.  For small |a|, (e^a - e^-a)/2 cancels
// (both exponentials -> 1), so use a direct Taylor for sinh (PORT_NOTES §3b);
// cosh is well-conditioned and taken from the exponentials.  QD's Taylor
// threshold is 0.05; this port keeps ff_math.hpp:553's 0.5 instead — the
// exp-method relative error is ~u/|a|, i.e. digits_lost ~= log10(1/|a|), which
// is ~0.3 digits at |a|=0.5 and ~1.3 digits at QD's 0.05.  At QF's 29-digit
// budget the larger 0.5 threshold (wider Taylor coverage) is the safer choice;
// see docs/PORT_NOTES_QF.md §"sinh/cosh threshold".
// KI-6/KI-7: past this |a| the smaller exponential e^{-2|a|} is below QF's
// resolution u = 2^-96, so cosh == sinh == e^{|a|}/2 and tanh == ±1 exactly.
// ln(1/u)/2 = 33.3, rounded up; must stay under 44.36, where tanh's 2a argument
// would leave FP32's exp range.
constexpr float kQFHyperbolicSaturate = 36.0f;

XPMATH_INLINE_FUNCTION void sinhcosh(QuadFloat a, QuadFloat& sinh_a, QuadFloat& cosh_a) {
    const float eps = 1.0e-28f;
    if (detail::fabs(a.f0) < 0.5f) {
        QuadFloat a2 = multiply(a, a);
        QuadFloat sinh_sum = a,             sinh_term = a;
        QuadFloat cosh_sum = QuadFloat(1.0f), cosh_term = QuadFloat(1.0f);
        for (int k = 1; k <= 60; ++k) {
            sinh_term = divide_scalar(multiply(sinh_term, a2), (float)((2*k) * (2*k + 1)));
            sinh_sum  = add(sinh_sum, sinh_term);
            cosh_term = divide_scalar(multiply(cosh_term, a2), (float)((2*k - 1) * (2*k)));
            cosh_sum  = add(cosh_sum, cosh_term);
            if (detail::fabs(sinh_term.f0) < eps * detail::fabs(sinh_sum.f0) &&
                detail::fabs(cosh_term.f0) < eps) break;
        }
        sinh_a = sinh_sum; cosh_a = cosh_sum;
        return;
    }
    if (detail::fabs(a.f0) > kQFHyperbolicSaturate) {
        // See dd_math.hpp. Also removes the reciprocal, which for |a| ~ 80 built
        // 1/e^{80} ~ 1.8e-35 through QF's divide and returned NaN.
        QuadFloat aa = (a.f0 < 0.0f) ? negate(a) : a;
        QuadFloat h;
        // KI-9. Crossover between the two ways of forming e^{|a|}/2 for a < 0.
        // The reciprocal route (exp(a), then 1/e) is the more accurate of the
        // two while exp(a) still occupies all four words: the last limb sits at
        // f0 * 2^-72, so it stays normal for f0 > FLT_MIN * 2^72, i.e.
        // a > ln(FLT_MIN) + 72 ln2 = -37.4. Below that the trailing limbs go
        // subnormal and then zero, exp(a) carries only a fraction of its
        // nominal width, and no reciprocal can put the digits back — at
        // a = -81.7 it holds barely 10 of 29. The direct route,
        // exp(|a| - ln2), forms the large magnitude at full width and wins by
        // up to 21.9 digits there, but costs up to 1.0 digit above the
        // crossover. -40 is the derived -37.4 with a margin: the limbs are not
        // exactly 24 bits apart, and the sweep still shows the reciprocal ahead
        // at a = -37.70 (points 703/705/706).
        const float kQFReciprocalFloor = -40.0f;
        if (a.f0 < kQFReciprocalFloor) {
            // KI-9. This used to compute e = exp(a) — a value near 1e-35, whose
            // f1 limb is already subnormal and whose f2/f3 flush to zero, so it
            // holds barely 10 of QF's 29 digits — and then reciprocate it. The
            // reciprocal cannot recover what the tiny operand never had.
            //
            // It scored 28 digits anyway because divide() overflowed the Dekker
            // splitter for quotients above 4.15e34 and returned NaN, so the
            // `e.f0 != e.f0` test below fired and routed the point to
            // exp(|a| - ln2), which forms the large magnitude directly at full
            // precision. Fixing the splitter removed the NaN, the fallback
            // stopped firing, and 60 sweep points lost up to 21.9 digits.
            // Take the accurate route deliberately instead of by accident.
            h = exp(subtract(aa, QuadFloat_log2()));
        } else {
            QuadFloat e = exp(a);
            if (a.f0 < 0.0f) e = divide(QuadFloat(1.0f), e);
            h = (detail::isinf(e.f0) || e.f0 != e.f0) ? exp(subtract(aa, QuadFloat_log2()))
                                                      : mul_pwr2(e, 0.5f);
        }
        cosh_a = h;
        sinh_a = (a.f0 < 0.0f) ? negate(h) : h;
        return;
    }
    QuadFloat s0 = exp(a);
    QuadFloat s1 = divide(QuadFloat(1.0f), s0);
    cosh_a = mul_pwr2(add(s0, s1), 0.5f);
    sinh_a = mul_pwr2(subtract(s0, s1), 0.5f);
}

XPMATH_INLINE_FUNCTION QuadFloat sinh(QuadFloat a) {
    QuadFloat s, c; sinhcosh(a, s, c); return s;
}
XPMATH_INLINE_FUNCTION QuadFloat cosh(QuadFloat a) {
    QuadFloat s, c; sinhcosh(a, s, c); return c;
}
// tanh(a) via expm1(2a)/(expm1(2a)+2) with odd reflection — avoids dividing two
// nearly-equal large exponentials.  ff_math.hpp:580 (QD qd_real.cpp:2547 divides
// the exponentials directly; the expm1 form is better-conditioned near 0).
XPMATH_INLINE_FUNCTION QuadFloat tanh(QuadFloat a) {
    if (a.f0 < 0.0f) return negate(tanh(negate(a)));
    if (a.f0 > kQFHyperbolicSaturate) return QuadFloat(1.0f);  // KI-7, see dd_math.hpp
    QuadFloat e = expm1(mul_pwr2(a, 2.0f));
    return divide(e, add(e, QuadFloat(2.0f)));
}

// asinh(a) = log(a + sqrt(a^2 + 1)).  QD qd_real.cpp:2576.  Odd reflection
// (ff_math.hpp:586) keeps the log argument >= 1 for negative a.
// KI-13.  asinh and acosh both form a^2 +- 1, which leaves the word range at
// |a| = sqrt(FLT_MAX) = 1.844e19 -- far short of what the format reaches -- and
// sqrt(inf)/log(inf) then returned NaN.  Above the band the square is factored
// out instead:
//
//     asinh(a) = log(|a|) + log(1 + sqrt(1 + 1/a^2)),  sign-reflected
//     acosh(a) = log(a)   + log(1 + sqrt(1 - 1/a^2))
//
// u = 1/a^2 is formed as (1/a)^2 so nothing squares a; it is in (0,1], and for
// |a| past 1/sqrt(smallest normal) it simply flushes to zero, which is the
// correct limit (asinh(a) -> log(2a)).  No cancellation: log(|a|) dominates the
// second term's log(2) ~ 0.69 by orders of magnitude, and both are computed to
// full relative precision.  Below the band the original expression is kept
// bit-for-bit.  Full argument at ff_math.hpp's asinh.
XPMATH_INLINE_FUNCTION QuadFloat asinh(QuadFloat a) {
    if (a.f0 < 0.0f) return negate(asinh(negate(a)));
    if (a.f0 > detail::kQFSqHi) {
        QuadFloat u = divide(QuadFloat(1.0f), a);
        u = multiply(u, u);
        return add(log(a), log(add(QuadFloat(1.0f), sqrt(add(QuadFloat(1.0f), u)))));
    }
    // KI-22: log1p(a + a^2/(1+sqrt(a^2+1))).  Derivation at dd_math.hpp's asinh.
    const QuadFloat a2 = sqr(a);
    const QuadFloat s  = sqrt(add(a2, QuadFloat(1.0f)));
    if (a.f0 < 0.5f) {
        return log1p(add(a, divide(a2, add(QuadFloat(1.0f), s))));
    }
    // KI-29: 1/2 log1p(2a(a+s)) rather than log(a+s).  Halves the share of
    // log's constant absolute error that asinh inherits.  Derived at
    // dd_math.hpp's asinh.
    const QuadFloat t = add(a, s);
    return mul_pwr2(log1p(mul_pwr2(multiply(a, t), 2.0f)), 0.5f);
}
// acosh(a) = log(a + sqrt(a^2 - 1)).  QD qd_real.cpp:2580.
XPMATH_INLINE_FUNCTION QuadFloat acosh(QuadFloat a) {
    if (qf_cmp_one(a) < 0) {                                            // KI-16
        XPMATH_PRINTF("QFACOSH: argument < 1\n"); return QuadFloat(0.0f); }
    if (a.f0 > detail::kQFSqHi) {                                       // KI-13
        QuadFloat u = divide(QuadFloat(1.0f), a);
        u = multiply(u, u);
        return add(log(a), log(add(QuadFloat(1.0f), sqrt(subtract(QuadFloat(1.0f), u)))));
    }
    return log(add(a, sqrt(subtract(multiply(a, a), QuadFloat(1.0f)))));
}
// atanh(a).  QD qd_real.cpp:2589 is 0.5*log((1+a)/(1-a)) only; this port adds a
// Taylor branch for |a| < 0.5 (PORT_NOTES §3c, ff_math.hpp:595) — all-positive
// terms, no cancellation, and avoids log() evaluated near 1.
XPMATH_INLINE_FUNCTION QuadFloat atanh(QuadFloat a) {
    // |a| == 1 is the C99 pole, not a domain error: atanh(+-1) = +-inf.
    const int c_atanh = qf_cmp_abs_one(a);                              // KI-16
    if (c_atanh > 0) {
        XPMATH_PRINTF("QFATANH: |argument| > 1\n"); return QuadFloat(0.0f); }
    if (c_atanh == 0) return QuadFloat(a.f0 > 0.0f ? HUGE_VALF : -HUGE_VALF);
    const float eps = 1.0e-28f;
    if (detail::fabs(a.f0) < 0.5f) {
        QuadFloat a2 = multiply(a, a);
        QuadFloat sum = a, pwr = a;
        for (int k = 1; k <= 60; ++k) {
            pwr = multiply(pwr, a2);
            QuadFloat term = divide_scalar(pwr, (float)(2*k + 1));
            sum = add(sum, term);
            if (detail::fabs(term.f0) < eps * detail::fabs(sum.f0)) break;
        }
        return sum;
    }
    QuadFloat t1 = add(QuadFloat(1.0f), a);
    QuadFloat t2 = subtract(QuadFloat(1.0f), a);
    return mul_pwr2(log(divide(t1, t2)), 0.5f);
}

// ============================================================
// Power / multi-argument
// ============================================================

// pow(a, b) = e^(b log a).  QD qd_real.cpp:655 (pow(qd,qd) = exp(b*log(a))).
XPMATH_INLINE_FUNCTION QuadFloat pow(QuadFloat a, QuadFloat b) {
    if (a.f0 <= 0.0f) {
        if (a.f0 == 0.0f && b.f0 > 0.0f) return QuadFloat(0.0f);
        XPMATH_PRINTF("QFPOW: non-positive base\n");
        return QuadFloat(0.0f);
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
// FP32 word range at |a| ~ 1.8e19 and flushes to zero at |a| ~ 1.1e-19 -- in both
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
// form actually breaks: for m inside [1.0e-18f, 1.0e18f] the old expression is
// evaluated exactly as before and the added cost is two compares, not a divide.
// Same reasoning as the atanh threshold in qf_complex.hpp -- fix the interval
// that is broken, do not churn the one that is not.
//
// inf/nan convention (C99 F.9.4.3): hypot(+-inf, y) is +inf for ANY y, NaN
// included, so the inf test comes first.  Otherwise a NaN operand propagates to
// NaN through the arithmetic.  Both operands are taken through abs() first, so
// the returned infinity is always +inf.
XPMATH_INLINE_FUNCTION QuadFloat hypot(QuadFloat a, QuadFloat b) {
    QuadFloat x = abs(a);
    QuadFloat y = abs(b);
    if (detail::isinf(x.f0)) return x;
    if (detail::isinf(y.f0)) return y;
    QuadFloat m = (x.f0 < y.f0) ? y : x;
    QuadFloat n = (x.f0 < y.f0) ? x : y;
    if (m.f0 == 0.0f) return QuadFloat(0.0f);
    if (m.f0 <= detail::kQFSqHi && m.f0 >= detail::kQFSqLo)
        return sqrt(add(multiply(a, a), multiply(b, b)));
    QuadFloat t = divide(n, m);
    return multiply(m, sqrt(add(QuadFloat(1.0f), multiply(t, t))));
}

// fmod / remainder — exact iterative scale-and-subtract (KI-10, KI-15).
// These no longer follow QD 2.3.24 (`fmod` qd_real.cpp:2597 = a - b*aint(a/b),
// `drem` :2462 = a - b*nint(a/b)): QD has the same defect and diverging from it
// is deliberate. The full derivation — loop bound, termination, and why every
// subtraction is exact by Sterbenz — is in dd_math.hpp. On QF the one-shot form
// additionally inherited KI-19, `divide` returning NaN at large quotients; no
// quotient is formed here at all, so that path is gone too.
// Conventions: C99 fmod (sign of a) and C99/IEEE-754 remainder (round-half-to-
// EVEN quotient, |r| <= |b|/2 — NOT QD's half-away nint, see KI-20).
// fmod(a,0), remainder(a,0), fmod(+-inf,b), remainder(+-inf,b) -> NaN;
// f(a,+-inf) = a for finite a; f(+-0,b) = +-0.
namespace detail {

// |A| mod |B| exactly, plus the parity of the integral quotient.
// Precondition: A >= 0, B > 0, both finite.
XPMATH_INLINE_FUNCTION QuadFloat qf_fmod_abs(QuadFloat A, QuadFloat B,
                                             bool& q_odd) {
    q_odd = false;
    if (A < B) return A;

    QuadFloat Bs = B;
    int k = 0;
    for (;;) {
        QuadFloat t = mul_pwr2(Bs, 1073741824.0f);  // 2^30
        if (t > A) break;
        Bs = t; k += 30;
    }
    for (;;) {
        QuadFloat t = mul_pwr2(Bs, 2.0f);
        if (t > A) break;
        Bs = t; ++k;
    }

    QuadFloat r = A;
    for (int i = k; i >= 0; --i) {
        if (r >= Bs) {
            r = subtract(r, Bs);          // Sterbenz-exact
            if (i == 0) q_odd = true;
        }
        Bs = mul_pwr2(Bs, 0.5f);
    }
    return r;
}

}  // namespace detail

XPMATH_INLINE_FUNCTION QuadFloat fmod(QuadFloat a, QuadFloat b) {
    if (a.f0 != a.f0 || b.f0 != b.f0) return QuadFloat(a.f0 + b.f0);
    if (b.f0 == 0.0f) { XPMATH_PRINTF("QFFMOD: zero modulus\n");
                        return QuadFloat(0.0f / 0.0f); }
    if (!detail::isfinite(a.f0)) { XPMATH_PRINTF("QFFMOD: infinite dividend\n");
                                   return QuadFloat(0.0f / 0.0f); }
    if (!detail::isfinite(b.f0)) return a;
    if (a.f0 == 0.0f) return a;

    bool q_odd = false;
    QuadFloat r = detail::qf_fmod_abs(abs(a), abs(b), q_odd);
    return (a.f0 < 0.0f) ? negate(r) : r;
}

XPMATH_INLINE_FUNCTION QuadFloat remainder(QuadFloat a, QuadFloat b) {
    if (a.f0 != a.f0 || b.f0 != b.f0) return QuadFloat(a.f0 + b.f0);
    if (b.f0 == 0.0f) { XPMATH_PRINTF("QFREMAINDER: zero modulus\n");
                        return QuadFloat(0.0f / 0.0f); }
    if (!detail::isfinite(a.f0)) { XPMATH_PRINTF("QFREMAINDER: infinite dividend\n");
                                   return QuadFloat(0.0f / 0.0f); }
    if (!detail::isfinite(b.f0)) return a;
    if (a.f0 == 0.0f) return a;

    bool q_odd = false;
    QuadFloat B = abs(b);
    QuadFloat r = detail::qf_fmod_abs(abs(a), B, q_odd);
    QuadFloat two_r = mul_pwr2(r, 2.0f);
    if (two_r > B || (two_r == B && q_odd)) r = subtract(r, B);  // Sterbenz-exact
    return (a.f0 < 0.0f) ? negate(r) : r;
}

// copysign / fmax / fmin / fdim / fma — no QD analogue; componentwise (cf. dd).
XPMATH_INLINE_FUNCTION QuadFloat copysign(QuadFloat a, QuadFloat b) {
    QuadFloat r = abs(a);
    if (b.f0 < 0.0f || (b.f0 == 0.0f && b.f1 < 0.0f)) return negate(r);
    return r;
}
XPMATH_INLINE_FUNCTION QuadFloat fmax(QuadFloat a, QuadFloat b) { return (a > b) ? a : b; }
XPMATH_INLINE_FUNCTION QuadFloat fmin(QuadFloat a, QuadFloat b) { return (a < b) ? a : b; }
XPMATH_INLINE_FUNCTION QuadFloat fdim(QuadFloat a, QuadFloat b) {
    return (a > b) ? subtract(a, b) : QuadFloat(0.0f);
}
// ---- KI-38: fma with an EXACT product ----------------------------------
// See dd_math.hpp:fma for the derivation. QF carried 12 of the 60 failing
// sweep points, scoring 12.93-14.39 digits against a 29.00 cap -- almost
// exactly half, which is the signature of a product rounded to 96 bits and
// then cancelled against c to ~2^-53 of its own magnitude. All 12 reach the
// cap with the exact product; the operands are held exactly, so there was no
// conditioning floor.
XPMATH_INLINE_FUNCTION void qf_expansion_push(float* e, int& m, float t) {
    if (t == 0.0f) return;
    float q = t;
    for (int i = 0; i < m; ++i) {
        float err;
        const float s = qf_two_sum(q, e[i], err);
        e[i] = err;
        q    = s;
    }
    e[m++] = q;
}
XPMATH_INLINE_FUNCTION void qf_expansion_compress(const float* e, int m,
                                                  float* out, int n) {
    float g[36], h[36];
    int   bottom = m - 1;
    float q = e[m - 1];
    for (int i = m - 2; i >= 0; --i) {
        float r;
        q = qf_quick_two_sum(q, e[i], r);
        if (r != 0.0f) { g[bottom--] = q; q = r; }
    }
    g[bottom] = q;
    int top = 0;
    for (int i = bottom + 1; i < m; ++i) {
        float r;
        q = qf_quick_two_sum(g[i], q, r);
        if (r != 0.0f) h[top++] = r;
    }
    h[top++] = q;
    for (int k = 0; k < n; ++k) out[k] = (top - 1 - k >= 0) ? h[top - 1 - k] : 0.0f;
}
XPMATH_INLINE_FUNCTION QuadFloat fma(QuadFloat a, QuadFloat b, QuadFloat c) {
    const float p0 = a.f0 * b.f0;
    if (!detail::isfinite(p0) || !detail::isfinite(c.f0))
        return add(multiply(a, b), c);

    const float aw[4] = {a.f0, a.f1, a.f2, a.f3};
    const float bw[4] = {b.f0, b.f1, b.f2, b.f3};
    float e[36];                       // 16 two_prods (32 words) + c's 4 words
    int   m = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            if (aw[i] == 0.0f || bw[j] == 0.0f) continue;
            float err;
            const float hi = qf_two_prod(aw[i], bw[j], err);
            qf_expansion_push(e, m, hi);
            qf_expansion_push(e, m, err);
        }
    const float cw[4] = {c.f0, c.f1, c.f2, c.f3};
    for (int i = 0; i < 4; ++i) qf_expansion_push(e, m, cw[i]);
    if (m == 0) return add(multiply(a, b), c);

    float d[5];
    qf_expansion_compress(e, m, d, 5);
    renorm_4(d[0], d[1], d[2], d[3], d[4]);
    return QuadFloat(d[0], d[1], d[2], d[3]);
}

// ============================================================
// Rounding — component-wise floor/ceil with renorm (QD's, not FF's nint form)
// ============================================================

// floor(a).  Faithful port of floor(qd_real), QD 2.3.24 qd_real.cpp:136-157:
// floor each component while the previous is integral; renorm.
XPMATH_INLINE_FUNCTION QuadFloat floor(QuadFloat a) {
    float x0, x1, x2, x3;
    x1 = x2 = x3 = 0.0f;
    x0 = detail::floor(a.f0);
    if (x0 == a.f0) {
        x1 = detail::floor(a.f1);
        if (x1 == a.f1) {
            x2 = detail::floor(a.f2);
            if (x2 == a.f2) x3 = detail::floor(a.f3);
        }
        renorm(x0, x1, x2, x3);
        return QuadFloat(x0, x1, x2, x3);
    }
    return QuadFloat(x0, x1, x2, x3);
}
// ceil(a).  Faithful port of ceil(qd_real), QD 2.3.24 qd_real.cpp:159-180.
XPMATH_INLINE_FUNCTION QuadFloat ceil(QuadFloat a) {
    float x0, x1, x2, x3;
    x1 = x2 = x3 = 0.0f;
    x0 = detail::ceil(a.f0);
    if (x0 == a.f0) {
        x1 = detail::ceil(a.f1);
        if (x1 == a.f1) {
            x2 = detail::ceil(a.f2);
            if (x2 == a.f2) x3 = detail::ceil(a.f3);
        }
        renorm(x0, x1, x2, x3);
        return QuadFloat(x0, x1, x2, x3);
    }
    return QuadFloat(x0, x1, x2, x3);
}
// trunc(a) = (a >= 0) ? floor(a) : ceil(a).  QD aint, qd_inline.h:975.
XPMATH_INLINE_FUNCTION QuadFloat trunc(QuadFloat a) {
    return (a.f0 >= 0.0f) ? floor(a) : ceil(a);
}
// round(a) — nearest integer, TIES TO EVEN (IEEE 754 roundToIntegralTiesToEven).
//
// KI-20 (2026-09-04). This is the library's ONE tie convention, chosen
// deliberately and applied to `round` on all four backends. It DIVERGES FROM QD
// 2.3.24, whose `nint` (qd_real.cpp:48-86, reached here through
// round_to_nearest_int) breaks ties toward +infinity, and from C99 `round`,
// which breaks them away from zero. See docs/KNOWN_ISSUES.md, KI-20.
//
// round_to_nearest_int is deliberately NOT changed. It is the internal argument
// reducer for sincos/exp, where the tie direction is unobservable (ties are
// measure-zero and either neighbour is an equally valid reduction), and its
// multi-word correction `|x0 - a.f0| == 0.5 && a.f1 < 0 -> x0 -= 1` is only
// valid because qf_nint leans one fixed way — making it half-even would require
// the parity of the accumulated integer, which no single limb carries. The two
// names therefore mean two different things and say so in both places.
//
// The correction below costs nothing off a tie and needs no parity test.
// A tie is exactly `2a is an odd integer`; at a tie, HALVING first destroys the
// tie (k + 1/2 -> k/2 + 1/4 for even k, m + 3/4 for odd k = 2m+1), its nearest
// integer is the even neighbour halved either way, and doubling back is exact
// because mul_pwr2 only touches exponents.
XPMATH_INLINE_FUNCTION QuadFloat round(QuadFloat a) {
    const QuadFloat n = round_to_nearest_int(a);
    if (n.f0 == a.f0 && n.f1 == a.f1 && n.f2 == a.f2 && n.f3 == a.f3)
        return n;                                   // already integral: no tie
    const QuadFloat t  = mul_pwr2(a, 2.0f);         // exact
    const QuadFloat nt = round_to_nearest_int(t);
    if (!(nt.f0 == t.f0 && nt.f1 == t.f1 && nt.f2 == t.f2 && nt.f3 == t.f3))
        return n;                                   // 2a not integral: no tie
    return mul_pwr2(round_to_nearest_int(mul_pwr2(a, 0.5f)), 2.0f);
}


} // namespace xp
