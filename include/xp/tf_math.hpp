// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (original algorithms;
//   Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical port of the QD 2.3.24 quad-double package
// (qd/src/qd_real.cpp and qd/include/qd/qd_inline.h) from four-word FP64
// (quad-double, ~212-bit significand) to three-word FP32 (triple-float,
// "TripleFloat", ~72-bit significand, ~21.7 decimal digits, u = 2^-72). The
// algorithm structure — Priest renormalization specialized to k=3, Hida-Li-Bailey
// sloppy/ieee addition (specialized to k=3), sloppy multiplication (k=3),
// long-division (k=3), and Heron square-root — descends directly from QD 2.3.24.
// Every non-trivial routine cites its QD source location.
//
// LICENSE LINEAGE (per docs/PORT_NOTES_QF.md §"License lineage"): TF is the
// k=3 FP32 instantiation of the QD lineage, sharing QF's LBNL-BSD-License
// provenance (triple-authored Hida/Li/Bailey, LBNL *institutional* copyright,
// commercial contact ipo@lbl.gov / TTD@lbl.gov) — the same license as QF,
// distinct from the DHB-License that governs dd_math.hpp / ff_math.hpp.
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text.
//
// FP32-specific porting notes (splitter reuse from ff_math.hpp, k=3
// renormalization derivation from QD's k=4 cascade, Newton/Heron iteration
// counts, constant generation, term counts) are documented in
// docs/PORT_NOTES_TF.md.

#pragma once

// Triple-float real arithmetic — xp::TripleFloat. ~21.7 decimal digits
// from an unevaluated sum of three FP32 components (f0 + f1 + f2,
// |f1| <= ulp(f0)/2, |f2| <= ulp(f1)/2).
//
// Mechanically ported from QD 2.3.24 (quad-double at 4×FP64, Hida-Li-Bailey)
// by specializing k=4 to k=3 at 3×FP32. See docs/PORT_NOTES_TF.md for k=3
// derivations.
//
// Precision: ~21.68 decimal digits (24-bit FP32 mantissa × 3 = 72 bits).
// Range: bounded by FP32 (~[3.9e-31, 8.3e34]), identical to FloatFloat —
//        TF adds PRECISION, not range.
//
// DEPENDENCIES: none beyond the C++17 standard library. In particular this
// header does NOT include or require Kokkos — see xp/config.hpp for how the
// four portability facilities it needs (inline annotation, on-device
// detection, scalar math dispatch, diagnostic printf) are supplied. Kokkos
// users get `Kokkos::Experimental::TripleFloat` API through the compat wrapper
// at third_party/include/tf_math.hpp, which is the only place
// `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// Naming conventions (T0.4/T2.0/T3.0):
//   * Type + math live in one flat namespace so an upstream move is
//     mechanical rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions TripleFloat_pi(), TripleFloat_e(), ...
//   * The bit-pattern constructor is the static factory
//     TripleFloat::from_bits(f0, f1, f2): namespaced to the type.
//   * Math functions are ADL-findable via the argument's namespace.

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
struct TripleFloat;
XPMATH_INLINE_FUNCTION TripleFloat add(TripleFloat a, TripleFloat b);
XPMATH_INLINE_FUNCTION TripleFloat subtract(TripleFloat a, TripleFloat b);
XPMATH_INLINE_FUNCTION TripleFloat multiply(TripleFloat a, TripleFloat b);
XPMATH_INLINE_FUNCTION TripleFloat divide(TripleFloat a, TripleFloat b);
XPMATH_INLINE_FUNCTION TripleFloat multiply_scalar(TripleFloat a, float b);
XPMATH_INLINE_FUNCTION TripleFloat divide_scalar(TripleFloat a, float b);
XPMATH_INLINE_FUNCTION TripleFloat mul_pwr2(TripleFloat a, float b);
XPMATH_INLINE_FUNCTION TripleFloat negate(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat abs(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat sqr(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat sqrt(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat round_to_nearest_int(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat pow_int(TripleFloat a, int n);
XPMATH_INLINE_FUNCTION TripleFloat exp(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat log(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat log1p(TripleFloat a);   // asinh() is defined above it
XPMATH_INLINE_FUNCTION TripleFloat pow(TripleFloat a, TripleFloat b);
XPMATH_INLINE_FUNCTION void   sincos(TripleFloat a, TripleFloat& sin_a, TripleFloat& cos_a);
XPMATH_INLINE_FUNCTION void   sinhcosh(TripleFloat a, TripleFloat& sinh_a, TripleFloat& cosh_a);
XPMATH_INLINE_FUNCTION TripleFloat angle(TripleFloat x, TripleFloat y);
XPMATH_INLINE_FUNCTION TripleFloat ceil(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat floor(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat trunc(TripleFloat a);
XPMATH_INLINE_FUNCTION TripleFloat round(TripleFloat a);

// ============================================================
// Error-free transforms (FP32) — REUSED from ff_math.hpp
// Bit-identical to the primitives validated for FF (T2.1); expressed here
// in QD's by-reference form for consistency with the QD port pattern.
// These are properties of the WORD TYPE (FP32), not the word count.
// ============================================================

// fl(a+b) and err, assuming |a| >= |b|.  QD inline.h:35-39 (quick_two_sum).
// Mirrors qf_quick_two_sum (qf_math.hpp:124-128).
XPMATH_INLINE_FUNCTION float tf_quick_two_sum(float a, float b, float& err) {
    float s = a + b;
    err = b - (s - a);
    return s;
}

// fl(a+b) and err (Knuth TwoSum, no ordering assumption).  QD inline.h:49-55.
// Mirror of the twoSum inside ff_math.hpp add() (ff_math.hpp:200-207).
XPMATH_INLINE_FUNCTION float tf_two_sum(float a, float b, float& err) {
    float s  = a + b;
    float bb = s - a;
    err = (a - (s - bb)) + (b - bb);
    return s;
}

// Veltkamp split with QD's large-magnitude guard.  Port of qd::split,
// QD 2.3.24 qd/include/qd/inline.h:66-83, at FP32.
//
// Splitter 8193.0f = 2^13+1 for the 24-bit FP32 mantissa — reused from
// ff_math.hpp (ff_math.hpp:220, validated for FF at T2.1).
//
// KI-9 (fixed 2026-09-02). Identical defect and identical fix to
// qf_math.hpp:qf_split — see the derivation there. In short: without the
// threshold branch, `a * split` overflows for |a| > FLT_MAX / (split + 1) ≈
// 4.15e34 and `hi = con - (con - a)` becomes inf - inf = NaN, taking multiply,
// sqr, multiply_scalar and divide's quotient digits with it. The pre-scale by
// 2^-14 and unscale by 2^14 are exact powers of two, so the hazard path adds no
// rounding and the non-hazard path is bit-identical to before.
XPMATH_INLINE_FUNCTION void tf_split(float a, float& hi, float& lo) {
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
// Empirically exact over |operands| <= 1e6; KI-9 extends that to the full FP32
// range.
//
// The non-finite early return is the genuine-overflow half of KI-9; see
// qf_math.hpp:qf_two_prod for the rationale. Briefly: for an overflowed product
// the residual is inf - inf = NaN, so a correctly-±inf result came back
// NaN-tailed; the error term of an overflowed product carries no information
// and 0 is the only defensible value.
XPMATH_INLINE_FUNCTION float tf_two_prod(float a, float b, float& err) {
    float p = a * b;
    if (!detail::isfinite(p)) { err = 0.0f; return p; }
    float a1, a2, b1, b2;
    tf_split(a, a1, a2);
    tf_split(b, b1, b2);
    err = ((a1 * b1 - p) + a1 * b2 + a2 * b1) + a2 * b2;
    return p;
}

// fl(a*a) and err.  QD inline.h:101-113 (two_sqr).
XPMATH_INLINE_FUNCTION float tf_two_sqr(float a, float& err) {
    float q = a * a;
    if (!detail::isfinite(q)) { err = 0.0f; return q; }
    float hi, lo;
    tf_split(a, hi, lo);
    err = ((hi * hi - q) + 2.0f * hi * lo) + lo * lo;
    return q;
}

// three_sum / three_sum2.  QD inline.h:192-204 (used in add).
XPMATH_INLINE_FUNCTION void tf_three_sum(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = tf_two_sum(a, b, t2);
    a  = tf_two_sum(c, t1, t3);
    b  = tf_two_sum(t2, t3, c);
}
XPMATH_INLINE_FUNCTION void tf_three_sum2(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = tf_two_sum(a, b, t2);
    a  = tf_two_sum(c, t1, t3);
    b  = t2 + t3;
}

// quick_three_accum: add c to the two-word pair (a, b). If the sum does not
// fit in two words the overflow is returned and (a, b) holds the remainder;
// otherwise 0 is returned and (a, b) holds the sum.  QD inline.h — actually
// qd_inline.h:261-282 (qd::quick_three_accum), used only by ieee_add.
XPMATH_INLINE_FUNCTION float tf_quick_three_accum(float& a, float& b, float c) {
    float s;
    bool za, zb;

    s = tf_two_sum(b, c, b);
    s = tf_two_sum(a, s, a);

    za = (a != 0.0f);
    zb = (b != 0.0f);

    if (za && zb) return s;

    if (!zb) {
        b = a;
        a = s;
    } else {
        a = s;
    }
    return 0.0f;
}

// ============================================================
// Renormalization (Priest normalization — Hida-Li-Bailey, k=3 specialization)
// Derived from QD's k=4 renormalization by removing the c3/s3 logic.
// See docs/PORT_NOTES_TF.md for the full k=4 -> k=3 derivation.
// ============================================================

// Length-3 renormalization: collapse a 3-word unnormalized expansion to a
// non-overlapping length-3 TripleFloat.  Derived from qd::renorm(c0,c1,c2,c3),
// QD 2.3.24 qd_inline.h:95-125, by eliminating the c3 component.
XPMATH_INLINE_FUNCTION void renorm(float& c0, float& c1, float& c2) {
    float s0, s1, s2 = 0.0f;
    if (detail::isinf(c0)) return;

    // Initial cascade (k=3: only two quick_two_sum calls needed)
    s0 = tf_quick_two_sum(c1, c2, c2);
    c0 = tf_quick_two_sum(c0, s0, c1);

    // Refinement pass
    s0 = c0;
    s1 = c1;
    if (s1 != 0.0f) {
        s1 = tf_quick_two_sum(s1, c2, s2);
    } else {
        s0 = tf_quick_two_sum(s0, c2, s1);
    }
    c0 = s0; c1 = s1; c2 = s2;
}

// Length-4 -> length-3 renormalization: collapse a 4-word unnormalized
// accumulator (the natural output width of add/multiply/divide) to a
// non-overlapping length-3 TripleFloat.  Derived from
// qd::renorm(c0,c1,c2,c3,c4), QD 2.3.24 qd_inline.h:127-177, by eliminating
// the c4 component and adjusting the cascade depth.
XPMATH_INLINE_FUNCTION void renorm_3(float& c0, float& c1, float& c2, float& c3) {
    float s0, s1, s2 = 0.0f;
    if (detail::isinf(c0)) return;

    // Initial cascade: collapse c0..c3 down (k=3 target: 3 quick_two_sum)
    s0 = tf_quick_two_sum(c2, c3, c3);
    s0 = tf_quick_two_sum(c1, s0, c2);
    c0 = tf_quick_two_sum(c0, s0, c1);

    // Refinement pass
    s0 = c0;
    s1 = c1;

    if (s1 != 0.0f) {
        s1 = tf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f) {
            s2 += c3;  // Absorb the last component
        } else {
            s1 = tf_quick_two_sum(s1, c3, s2);
        }
    } else {
        s0 = tf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f) {
            s1 = tf_quick_two_sum(s1, c3, s2);
        } else {
            s0 = tf_quick_two_sum(s0, c3, s1);
        }
    }
    c0 = s0; c1 = s1; c2 = s2;
}

// ============================================================
// TripleFloat struct
// ============================================================
struct TripleFloat {
    float f0, f1, f2;

    XPMATH_INLINE_FUNCTION TripleFloat() : f0(0.0f), f1(0.0f), f2(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloat(float x) : f0(x), f1(0.0f), f2(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloat(float a0, float a1, float a2)
        : f0(a0), f1(a1), f2(a2) {}

    // Faithfully encode an FP64 value by successive FP32 splitting (Route-A,
    // length-3 analogue of qf_math.hpp's QuadFloat(double)). A double carries
    // 53 bits, so two words suffice; f2 falls to 0 after the split.
    XPMATH_INLINE_FUNCTION TripleFloat(double x) {
        double r = x;
        float  c0 = (float)r; r -= (double)c0;
        float  c1 = (float)r; r -= (double)c1;
        float  c2 = (float)r;
        f0 = c0; f1 = c1; f2 = c2;
    }

    XPMATH_INLINE_FUNCTION TripleFloat(const TripleFloat& o)
        : f0(o.f0), f1(o.f1), f2(o.f2) {}
    XPMATH_INLINE_FUNCTION TripleFloat& operator=(const TripleFloat& o) {
        f0=o.f0; f1=o.f1; f2=o.f2; return *this;
    }

    XPMATH_INLINE_FUNCTION float operator[](int i) const {
        return (i==0)?f0:(i==1)?f1:f2;
    }

    // Factory: build a TripleFloat from the IEEE-754 bit patterns of its three
    // FP32 components. Safe on host (memcpy) and device (__int_as_float).
    static XPMATH_INLINE_FUNCTION TripleFloat from_bits(uint32_t b0, uint32_t b1, uint32_t b2) {
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
        return TripleFloat(f0, f1, f2);
    }

    XPMATH_INLINE_FUNCTION TripleFloat operator-() const { return negate(*this); }
    XPMATH_INLINE_FUNCTION TripleFloat operator+(TripleFloat b) const { return add(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloat operator-(TripleFloat b) const { return subtract(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloat operator*(TripleFloat b) const { return multiply(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloat operator/(TripleFloat b) const { return divide(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloat operator*(float b)  const { return multiply_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloat operator/(float b)  const { return divide_scalar(*this, b); }
    XPMATH_INLINE_FUNCTION TripleFloat operator+(float b)  const { return add(*this, TripleFloat(b)); }
    XPMATH_INLINE_FUNCTION TripleFloat operator-(float b)  const { return subtract(*this, TripleFloat(b)); }

    XPMATH_INLINE_FUNCTION TripleFloat& operator+=(TripleFloat b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator-=(TripleFloat b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator*=(TripleFloat b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator/=(TripleFloat b) { *this = *this / b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator+=(float b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator-=(float b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator*=(float b) { *this = multiply_scalar(*this, b); return *this; }
    XPMATH_INLINE_FUNCTION TripleFloat& operator/=(float b) { *this = divide_scalar(*this, b); return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(TripleFloat b) const { return f0==b.f0 && f1==b.f1 && f2==b.f2; }
    XPMATH_INLINE_FUNCTION bool operator!=(TripleFloat b) const { return !(*this == b); }
    XPMATH_INLINE_FUNCTION bool operator<(TripleFloat b)  const {
        return f0<b.f0 || (f0==b.f0 && (f1<b.f1 || (f1==b.f1 && f2<b.f2)));
    }
    XPMATH_INLINE_FUNCTION bool operator>(TripleFloat b)  const {
        return f0>b.f0 || (f0==b.f0 && (f1>b.f1 || (f1==b.f1 && f2>b.f2)));
    }
    XPMATH_INLINE_FUNCTION bool operator<=(TripleFloat b) const { return !(b < *this); }
    XPMATH_INLINE_FUNCTION bool operator>=(TripleFloat b) const { return !(*this < b); }
};

XPMATH_INLINE_FUNCTION TripleFloat operator+(float a, TripleFloat b) { return add(TripleFloat(a), b); }
XPMATH_INLINE_FUNCTION TripleFloat operator-(float a, TripleFloat b) { return subtract(TripleFloat(a), b); }
XPMATH_INLINE_FUNCTION TripleFloat operator*(float a, TripleFloat b) { return multiply_scalar(b, a); }
XPMATH_INLINE_FUNCTION TripleFloat operator/(float a, TripleFloat b) { return divide(TripleFloat(a), b); }

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const TripleFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific << d.f0
       << ", " << d.f1 << ", " << d.f2 << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// Generated by splitting 113-bit __float128 literals three ways (Route-A,
// length-3): c0 = (float)x; r = x - c0; c1 = (float)r; r -= c1; c2 = (float)r.
// See docs/PORT_NOTES_TF.md §2 for the procedure.
//
// S10 Phase 1.5: the words below REPLACE Phase 1's placeholders, whose third
// word was fabricated (FF's two words plus a made-up tail) and overlapped the
// second word, capping every constant at ~8 digits and, through the exp/sincos
// argument reductions, every transcendental with it. Each constant now
// reconstructs to 22.5-23.6 digits against __float128. See PORT_NOTES_TF.md §8b.
// ============================================================
XPMATH_INLINE_FUNCTION TripleFloat TripleFloat_pi() {
    return TripleFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU, 0xa7772cedU);
}
XPMATH_INLINE_FUNCTION TripleFloat TripleFloat_e() {
    return TripleFloat::from_bits(0x402df854U, 0x33b14577U, 0xa7559541U);
}
XPMATH_INLINE_FUNCTION TripleFloat TripleFloat_log2() {
    return TripleFloat::from_bits(0x3f317218U, 0xb102e308U, 0xa4ca86c4U);
}
XPMATH_INLINE_FUNCTION TripleFloat TripleFloat_log10() {
    return TripleFloat::from_bits(0x40135d8eU, 0xb309555dU, 0xa69f48adU);
}
XPMATH_INLINE_FUNCTION TripleFloat TripleFloat_sqrt2() {
    return TripleFloat::from_bits(0x3fb504f3U, 0x32cfe77aU, 0xa65bdd34U);
}
XPMATH_INLINE_FUNCTION TripleFloat TripleFloat_euler_gamma() {
    return TripleFloat::from_bits(0x3f13c468U, 0xb1e4127aU, 0x24f49a38U);
}

// ============================================================
// Primitive arithmetic
// ============================================================

XPMATH_INLINE_FUNCTION TripleFloat negate(TripleFloat a) {
    return TripleFloat(-a.f0, -a.f1, -a.f2);
}

// Sloppy addition (QD's default, QD_IEEE_ADD off). Port of qd::sloppy_add
// (qd_inline.h:338-405, the k=4 version), specialized to k=3.
// QD rationale (applies identically to TF): non-overlapping expansions remain
// non-overlapping under component-wise addition, so the fixed-width
// three_sum/three_sum2 merge is safe and faster than ieee_add's digit-by-digit
// accumulation. See PORT_NOTES_QF.md §3 for the FP32 exponent safety analysis
// (which holds for k=3 as it does for k=4).
// k=3 term map (QD's k=4 accumulator with the a[3]/b[3] column removed):
//   s_i = two_sum(a_i, b_i, t_i)   — s_i has weight u^i, its error t_i has u^(i+1)
// QD merges the errors with two_sum / three_sum for the interior words and
// three_sum2 for the LAST word (QD applies three_sum2 to s3; at k=3 that is s2),
// then folds the remaining u^(k) terms into the extra renorm word.
XPMATH_INLINE_FUNCTION TripleFloat sloppy_add(TripleFloat a, TripleFloat b) {
    float s0, s1, s2;
    float t0, t1, t2;

    s0 = tf_two_sum(a.f0, b.f0, t0);
    s1 = tf_two_sum(a.f1, b.f1, t1);
    s2 = tf_two_sum(a.f2, b.f2, t2);

    s1 = tf_two_sum(s1, t0, t0);   // s1: u^1, t0: u^2
    tf_three_sum2(s2, t0, t1);     // s2: u^2, t0: u^3   (QD: three_sum2(s3,t0,t2))
    t0 = t0 + t2;                  // u^3 carry (QD: t0 = t0 + t1 + t3)

    renorm_3(s0, s1, s2, t0);
    return TripleFloat(s0, s1, s2);
}

// IEEE addition (digit-by-digit accumulation). Port of qd::ieee_add
// (qd_inline.h:270-336), specialized to k=3.
XPMATH_INLINE_FUNCTION TripleFloat ieee_add(TripleFloat a, TripleFloat b) {
    int i, j, k;
    float s, t;
    float u, v;
    float x[3] = {0.0f, 0.0f, 0.0f};

    i = j = k = 0;
    if (detail::fabs(a.f0) > detail::fabs(b.f0))
        u = a[i++];
    else
        u = b[j++];
    if (detail::fabs(a[i]) > detail::fabs(b[j]))
        v = a[i++];
    else
        v = b[j++];

    u = tf_quick_two_sum(u, v, v);

    while (k < 3) {
        if (i >= 3 && j >= 3) {
            x[k] = u;
            if (k < 2) x[++k] = v;
            break;
        }

        if (i >= 3)
            t = b[j++];
        else if (j >= 3)
            t = a[i++];
        else if (detail::fabs(a[i]) > detail::fabs(b[j])) {
            t = a[i++];
        } else
            t = b[j++];

        s = tf_quick_three_accum(u, v, t);

        if (s != 0.0f) {
            x[k++] = s;
        }
    }

    // Add the rest.
    for (k = i; k < 3; k++) x[2] += a[k];
    for (k = j; k < 3; k++) x[2] += b[k];

    renorm(x[0], x[1], x[2]);
    return TripleFloat(x[0], x[1], x[2]);
}

XPMATH_INLINE_FUNCTION TripleFloat add(TripleFloat a, TripleFloat b) {
    return sloppy_add(a, b);
}

XPMATH_INLINE_FUNCTION TripleFloat subtract(TripleFloat a, TripleFloat b) {
    return sloppy_add(a, negate(b));
}

XPMATH_INLINE_FUNCTION TripleFloat abs(TripleFloat a) {
    return (a.f0 < 0.0f) ? negate(a) : a;
}

// Sloppy multiplication (QD's default, QD_SLOPPY_MUL on). Port of
// qd_real::sloppy_mul, QD 2.3.24 qd_inline.h:567-599, specialized to k=3.
//
// Partial-product map — QD's own comment (qd_inline.h:556-566) minus the
// a[3]/b[3] column:
//     a0*b0                 u^0
//          a0*b1  a1*b0     u^1
//          a0*b2  a1*b1  a2*b0   u^2
//               a1*b2  a2*b1     u^3
// The six u^0..u^2 products are formed with exact two_prod (QD's p0..p5 use
// only a[0..2] and b[0..2], so ALL SIX survive the k=3 reduction unchanged);
// the u^3 cross-products are folded in scalar. Only two things change from
// QD's k=4 body: the O(u^3) fold loses the a[0]*b[3] and a[3]*b[0] terms
// (those words do not exist at k=3), and the closing renormalization is
// renorm_3 over four words (p0,p1,s0,s1) instead of renorm over five —
// so s2, QD's u^4 word, is folded into s1 rather than passed separately.
XPMATH_INLINE_FUNCTION TripleFloat multiply(TripleFloat a, TripleFloat b) {
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

    // Start accumulation.
    tf_three_sum(p1, p2, q0);

    // Six-three sum of (p2, q1, q2) and (p3, p4, p5).
    tf_three_sum(p2, q1, q2);
    tf_three_sum(p3, p4, p5);
    s0 = tf_two_sum(p2, p3, t0);
    s1 = tf_two_sum(q1, p4, t1);
    s2 = q2 + p5;
    s1 = tf_two_sum(s1, t0, t0);
    s2 += (t0 + t1);

    // O(u^3) terms, plus the u^4 remainder s2 folded in (renorm_3 takes four
    // words, so there is no fifth slot for it as there is at k=4).
    s1 += a.f1*b.f2 + a.f2*b.f1 + q0 + q3 + q4 + q5 + s2;
    renorm_3(p0, p1, s0, s1);
    return TripleFloat(p0, p1, s0);
}

// Multiplication by an exact power of 2 (no rounding).  QD qd_inline.h:544-548.
XPMATH_INLINE_FUNCTION TripleFloat mul_pwr2(TripleFloat a, float b) {
    return TripleFloat(a.f0 * b, a.f1 * b, a.f2 * b);
}

// Triple-float * float.  Port of operator*(qd_real, double), QD 2.3.24
// qd_inline.h:490-514, specialized to k=3. QD forms an exact two_prod for
// every word except the LAST, which it takes as a plain product (QD:
// p3 = a[3]*b); at k=3 that last word is a.f2, so a.f2*b is the plain one and
// the a[2] two_prod of the k=4 body disappears. The three u^2 words that QD
// merges with three_sum(s2, q1, p2) are the same three here, merged with
// three_sum2 because k=3 needs only one more word out of them.
XPMATH_INLINE_FUNCTION TripleFloat multiply_scalar(TripleFloat a, float b) {
    float p0, p1, p2;
    float q0, q1;
    float s0, s1, s2, s3;

    p0 = tf_two_prod(a.f0, b, q0);   // p0: u^0, q0: u^1
    p1 = tf_two_prod(a.f1, b, q1);   // p1: u^1, q1: u^2
    p2 = a.f2 * b;                   // p2: u^2  (QD's plain last-word product)

    s0 = p0;
    s1 = tf_two_sum(q0, p1, s2);     // s1: u^1, s2: u^2
    tf_three_sum2(s2, q1, p2);       // s2: u^2, q1: u^3
    s3 = q1;

    renorm_3(s0, s1, s2, s3);
    return TripleFloat(s0, s1, s2);
}

// Squaring. Port of sqr(qd_real), QD 2.3.24 qd_inline.h:674-715, specialized
// to k=3.  (x0+x1+x2)^2 = x0^2 + 2x0x1 + (2x0x2 + x1^2) + 2x1x2 + x2^2, so the
// u^0..u^2 structure is byte-for-byte QD's; the only k=3 change is in the u^3
// block, where QD's pair (p4 = 2a0a3, p5 = 2a1a2) collapses to the single term
// 2a1a2, and the closing renorm is renorm_3 over four words with QD's u^4 word
// p4 folded into p3.
XPMATH_INLINE_FUNCTION TripleFloat sqr(TripleFloat a) {
    // KI-9 genuine-overflow guard; see qf_math.hpp:sqr for the derivation. The
    // `2.0f * a.f0 * a.fk` products overflow at the doubling for
    // |a.f0| > FLT_MAX/2, and inf * 0 is NaN. If those can overflow then
    // a.f0 * a.f0 already has, so ±inf is the whole answer.
    float leading = a.f0 * a.f0;
    if (!detail::isfinite(leading)) return TripleFloat(leading, 0.0f, 0.0f);

    float p0, p1, p2, p3, p4;
    float q0, q1, q2, q3;
    float s0, s1;
    float t0, t1;

    p0 = tf_two_sqr(a.f0, q0);
    p1 = tf_two_prod(2.0f * a.f0, a.f1, q1);
    p2 = tf_two_prod(2.0f * a.f0, a.f2, q2);
    p3 = tf_two_sqr(a.f1, q3);

    p1 = tf_two_sum(q0, p1, q0);

    q0 = tf_two_sum(q0, q1, q1);
    p2 = tf_two_sum(p2, p3, p3);

    s0 = tf_two_sum(q0, p2, t0);
    s1 = tf_two_sum(q1, p3, t1);

    s1 = tf_two_sum(s1, t0, t0);
    t0 += t1;

    s1 = tf_quick_two_sum(s1, t0, t0);
    p2 = tf_quick_two_sum(s0, s1, t1);
    p3 = tf_quick_two_sum(t1, t0, q0);

    p4 = 2.0f * a.f1 * a.f2;   // k=3: QD's p4 = 2*a0*a3 has no counterpart
    q2 = tf_two_sum(q2, q3, q3);

    t0 = tf_two_sum(p4, q2, t1);
    t1 = t1 + q3;              // QD: t1 + p5 + q3; p5 = 2*a1*a2 is now p4

    p3 = tf_two_sum(p3, t0, p4);
    p4 = p4 + q0 + t1;

    p3 += p4;                  // fold QD's u^4 renorm word (no fifth slot at k=3)
    renorm_3(p0, p1, p2, p3);
    return TripleFloat(p0, p1, p2);
}

// Long division (QD's default, sloppy_div). Port of qd_real::sloppy_div
// (qd_real.cpp:693-736, the k=4 version with 4 quotient digits), specialized
// to k=3 (3 quotient digits reach ~72 bits).
// Each digit q_k = r[0]/b.f0 contributes ~24 fresh bits. Three digits
// (q0 ~24b, q1 ~48b, q2 ~72b) reach the TF width. Natural output is 4 words;
// collapsed via renorm_3.
// QD's k=4 sloppy_div produces four quotient digits and closes with the
// LENGTH-4 renorm (qd_real.cpp:775, `::renorm(q0,q1,q2,q3)`) — not the length-5
// one; the digits themselves are the expansion. The k=3 reduction is therefore
// three digits closed by the length-3 renorm.
// KI-19.  Non-finite signalling — identical defect and identical fix to
// qf_math.hpp:divide; the full derivation is there.  In short, q0 = a.f0 / b.f0
// is the leading quotient digit and, to within one FP32 rounding, the whole
// answer, so it classifies the three non-finite outcomes exactly: ±inf when the
// true quotient leaves FP32's range (or b = 0), ±0 when it falls below the
// smallest subnormal, NaN only for 0/0, inf/inf and NaN operands.  Without the
// branch, q0 = inf propagated into subtract(a, inf) = -inf and renorm collapsed
// the pair to NaN.
XPMATH_INLINE_FUNCTION TripleFloat divide(TripleFloat a, TripleFloat b) {
    float q0, q1, q2;
    TripleFloat r;

    q0 = a.f0 / b.f0;
    if (!detail::isfinite(q0) || q0 == 0.0f) return TripleFloat(q0);
    r = subtract(a, multiply_scalar(b, q0));

    q1 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q1));

    q2 = r.f0 / b.f0;

    renorm(q0, q1, q2);
    return TripleFloat(q0, q1, q2);
}

// Division by scalar. Derived from divide() by replacing b.f1, b.f2 with zeros
// and noting that all b*qN products become scalar multiplies (exact for small
// quotients, or handled via two_prod). Three digits reach ~72 bits.
XPMATH_INLINE_FUNCTION TripleFloat divide_scalar(TripleFloat a, float b) {
    float q0, q1, q2, p, e;
    TripleFloat r;

    q0 = a.f0 / b;
    if (!detail::isfinite(q0) || q0 == 0.0f) return TripleFloat(q0);   // KI-19; see divide()
    p = tf_two_prod(q0, b, e);  r = subtract(a, TripleFloat(p, e, 0.0f));
    q1 = r.f0 / b;  p = tf_two_prod(q1, b, e);  r = subtract(r, TripleFloat(p, e, 0.0f));

    q2 = r.f0 / b;

    renorm(q0, q1, q2);
    return TripleFloat(q0, q1, q2);
}

// Square root (Heron's method). Port of qd_real::sqrt (qd_real.cpp:738-785,
// the k=4 "fsqrt"), specialized to k=3.
// Iteration: x ← ½(x + a/x) doubles correct bits each step. FP32 seed ~24b →
// 48b → 72b, saturating at TF width, so 2 iterations reach full precision
// (confirmed by early-out on iteration 2). QD uses eps = 2^-212; TF uses
// eps = 2^-72, the unit roundoff.
XPMATH_INLINE_FUNCTION TripleFloat sqrt(TripleFloat a) {
    if (a.f0 == 0.0f)
        return TripleFloat(0.0f);

    if (a.f0 < 0.0f) {
        XPMATH_PRINTF("TFSQRT: negative argument\n");
        return TripleFloat(0.0f);
    }

    TripleFloat r = TripleFloat(detail::sqrt(a.f0));
    TripleFloat ax;

    // Heron iteration: r ← ½(r + a/r).  Up to 10 iterations with early-out.
    // TF reaches full precision in 2 iterations (24→48→72 bits).
    for (int i = 0; i < 10; i++) {
        ax = divide(a, r);
        ax = add(ax, r);
        ax = mul_pwr2(ax, 0.5f);

        TripleFloat d = subtract(r, ax);
        if (abs(d).f0 < abs(r).f0 * 1.0e-22f)  // 2^-72 ≈ 2.1e-22
            return ax;
        r = ax;
    }
    return r;
}

// Nearest FP32-int of a single float, ties toward +infinity. Mirrors
// qf_math.hpp's qf_nint exactly, including the KI-2 fix — see the long comment
// there for why this is `rint` plus a tie-direction restore and not QD's
// literal `floor(d + 0.5)`. In one line: 0.49999997f + 0.5f rounds up to 1.0f
// in FP32, so the floor form returns 1 where the nearest integer is 0. TF is
// where KI-2 was originally found (S10 Phase 3.5, PORT_NOTES_TF.md §12f).
//
// TF's exposure was MUCH wider than QF's, and the two are worth separating
// because only the new code is common. qf_nint short-circuited integers
// (`if (d == floor(d)) return d;`) before reaching the floor form, so QF's only
// wrong input was 0.49999997f. round_to_nearest_int below called
// `floor(a.fN + 0.5f)` bare, with no such guard — and for an ODD integer d in
// [2^23, 2^24), where ulp is exactly 1, `d + 0.5f` is a perfect tie that
// round-half-to-EVEN resolves UPWARD to d + 1, so floor returned d + 1 as the
// "nearest integer" of an integer. That is 2^22 wrong inputs per limb, not one,
// and it reached every limb of every reduction TF performs. The monotone gate
// caught it at the third limb of a `remainder` quotient (grid point r/893); see
// docs/KNOWN_ISSUES.md, KI-2 resolution.
XPMATH_INLINE_FUNCTION float tf_nint(float d) {
    float r = detail::rint(d);
    if (d - r == 0.5f) r += 1.0f;
    if (r == 0.0f && d != 0.0f) r = 0.0f;   // (-0.5, 0) -> +0, as QD gave
    return r;
}

// Round to nearest integer. Port of qd_real::nint (qd_real.cpp:48-86, the k=4
// floor(d+0.5) form), specialized to k=3, with tf_nint standing in for the
// floor form (KI-2). Half-integer tie corrections are keyed on the sign of the
// next component, and remain valid because tf_nint keeps QD's tie direction.
XPMATH_INLINE_FUNCTION TripleFloat round_to_nearest_int(TripleFloat a) {
    float f0, f1, f2;
    f0 = tf_nint(a.f0);
    f1 = 0.0f;
    f2 = 0.0f;

    if (f0 == a.f0) {
        f1 = tf_nint(a.f1);
        if (f1 == a.f1) {
            f2 = tf_nint(a.f2);
        } else {
            if (detail::fabs(f1 - a.f1) == 0.5f && a.f2 < 0.0f)
                f1 -= 1.0f;
        }
    } else {
        if (detail::fabs(f0 - a.f0) == 0.5f && a.f1 < 0.0f)
            f0 -= 1.0f;
    }

    renorm(f0, f1, f2);
    return TripleFloat(f0, f1, f2);
}

// Integer power. Port of qd_real::npwr (qd_real.cpp:862-890), specialized to k=3.
XPMATH_INLINE_FUNCTION TripleFloat pow_int(TripleFloat a, int n) {
    if (n == 0)
        return TripleFloat(1.0f);
    TripleFloat r = a;
    TripleFloat s = TripleFloat(1.0f);
    int N = detail::fabs((float)n);

    if (N > 1) {
        while (N > 0) {
            if (N % 2 == 1) {
                s = multiply(s, r);
            }
            N /= 2;
            if (N > 0)
                r = sqr(r);
        }
    } else {
        s = r;
    }

    if (n < 0)
        return divide(TripleFloat(1.0f), s);

    return s;
}

// ============================================================
// Transcendentals (table-free, matching dd/ff/qf pattern)
// QD 2.3.24's transcendentals are table-based (PORT_NOTES_QF.md §6), but TF
// follows the dd/ff/qf table-free structure: divide-by-k Taylor + joint
// sin/cos doublings. Term counts are derived for the k=3 target (~72 bits).
// See docs/PORT_NOTES_TF.md for iteration/term-count derivations.
// ============================================================

// exp: divide-by-k Taylor + nq squarings. After reduction,
// |s0| ≤ log2/2 ≈ 0.347; scaling by 2^-nq gives |r| ≤ 0.347/2^nq. Taylor
// e^r = Σ r^k/k! must reach TF unit roundoff u = 2^-72 ≈ 2.1e-22.
// With nq = 5, |r| ≤ 0.347/32 ≈ 0.0108:
//   |r|^8 / 8! ≈ 4.1e-21   (still above u)
//   |r|^9 / 9! ≈ 4.9e-23   (< u, converged)
// So N = 9 terms suffice. Convergence eps set to 1e-21f (coarser than u, per
// PORT_NOTES_QF.md §7 to avoid FF's exp-eps stall bug).
XPMATH_INLINE_FUNCTION TripleFloat exp(TripleFloat a) {
    const float k_inv_log2 = 1.44269504088896341f;  // 1/ln(2)
    const TripleFloat k_log2 = TripleFloat_log2();

    // KI-6: word-range-derived guard. The previous pair was ±80 and, worse than
    // flushing, the upper branch returned the sentinel 1.0e30f — a plausible-
    // looking number that is not an approximation of anything. e^a exceeds
    // FLT_MAX above ln(FLT_MAX) = 88.722839 and drops below the smallest FP32
    // subnormal below -103.28; between those the answer is representable.
    if (a.f0 < -104.0f) return TripleFloat(0.0f);   // e^a < 2^-149
    if (a.f0 >  88.722839f) {
        XPMATH_PRINTF("TFEXP: overflow\n");
        return TripleFloat(HUGE_VALF);              // e^a > FLT_MAX
    }

    // KI-2 audit (2026-09-02): this is the ONE surviving `floor(x + 0.5f)` in
    // the library, and it is deliberately not converted to tf_nint. Both KI-2
    // failure modes are unreachable here. The guards above bound a.f0 to
    // (-104, 88.73), so the argument is under 150 in magnitude — nowhere near the
    // [2^23, 2^24) band where the odd-integer tie bites. And the near-tie case
    // (0.49999997f-class) is harmless for a range reduction rather than wrong:
    // an off-by-one m shifts r by ln2, which then gets divided by 2^nq and
    // stays deep inside the series' convergence radius, and the same m is used
    // for the scale-back, so the result is unchanged. Converting it would
    // perturb the reduction on a large set of inputs to buy nothing.
    float m = detail::floor(a.f0 * k_inv_log2 + 0.5f);
    TripleFloat r = subtract(a, multiply_scalar(k_log2, m));

    const int nq = 5;
    r = divide_scalar(r, float(1 << nq));

    TripleFloat s = add(TripleFloat(1.0f), r);
    TripleFloat t = sqr(r);
    TripleFloat term = t;
    int k = 2;

    while (k < 64 && abs(term).f0 > 1.0e-21f * abs(s).f0) {
        term = divide_scalar(term, float(k));
        s = add(s, term);
        term = multiply(term, r);
        k++;
    }

    // nq squarings: e^r → e^(2r) → e^(4r) → ... → e^(2^nq·r)
    for (int i = 0; i < nq; i++) {
        s = multiply(s, s);
    }

    // Final scaling by 2^m through the EXPONENT, component-wise
    // (PORT_NOTES_QF.md §10). KI-6: materialising `2^m` as a float first is
    // +inf for m >= 128 and 0 for m <= -150, which put both ends of the FP32
    // range out of reach regardless of the guard.
    const int mi = (int)m;
    if (mi >= -125 && mi <= 127)            // 2^mi exact and normal: cheap path
        return mul_pwr2(s, ldexpf(1.0f, mi));
    return TripleFloat(ldexpf(s.f0, mi), ldexpf(s.f1, mi), ldexpf(s.f2, mi));
}

// log: Newton iteration x ← x + (a - e^x)/e^x. Port of qd_real::log
// (qd_real.cpp:998-1041), specialized to k=3. Initial estimate from
// FP32 log(a.f0). Three iterations double precision 24→48→72 bits.
XPMATH_INLINE_FUNCTION TripleFloat log(TripleFloat a) {
    if (detail::isinf(a.f0)) return a;   // KI-6, see dd_math.hpp
    if (a.f0 <= 0.0f) {
        XPMATH_PRINTF("TFLOG: non-positive argument\n");
        return TripleFloat(0.0f);
    }

    TripleFloat x = TripleFloat(detail::log(a.f0));

    for (int i = 0; i < 3; i++) {
        // KI-23, small-argument mirror image.  See qf_math.hpp's log for the
        // derivation.  The residual form evaluates e^{x}, which for a << 1 is
        // tiny and loses its low words to FP32 underflow -- TF measured 9.54 of
        // 22 digits at 1e-38.  Switch to QD's form only where that bites: f2
        // sits at 2^(E-48), leaving the FP32 normal range 2^-126 once E < -78,
        // i.e. x < -78*ln2 = -54.1.  Above that the residual form is strictly
        // better (relative rather than absolute error on the correction), and
        // below -ln(FLT_MAX) the switch is unavailable because 1/a overflows.
        if (x.f0 < -54.1f && -x.f0 <= 88.722839f) {
            TripleFloat e = exp(negate(x));                     // e^{|x|} >= 1
            x = subtract(add(x, multiply(a, e)), TripleFloat(1.0f));
        } else {
            TripleFloat e = exp(x);                             // e^{x} >= 1
            x = add(x, divide(subtract(a, e), e));
        }
    }

    return x;
}

// pow: a^b = exp(b·log a). PORT_NOTES_QF.md §10 conditioning caveat applies.
XPMATH_INLINE_FUNCTION TripleFloat pow(TripleFloat a, TripleFloat b) {
    return exp(multiply(b, log(a)));
}

// sin/cos: joint computation via argument reduction mod 2π, divide-by-k Taylor
// on the residual, and joint angle-doubling formulas (PORT_NOTES.md §3a).
// With nq = 4, r = s3/2^4, and sin(r)/cos(r) Taylor converges in ~7 terms to
// reach TF width. Four joint doublings recover sin(s3)/cos(s3).
XPMATH_INLINE_FUNCTION void sincos(TripleFloat a, TripleFloat& sin_a, TripleFloat& cos_a) {
    const TripleFloat k_2pi = mul_pwr2(TripleFloat_pi(), 2.0f);

    if (a.f0 == 0.0f) {
        sin_a = TripleFloat(0.0f);
        cos_a = TripleFloat(1.0f);
        return;
    }
    // KI-12 residual.  Small-argument short circuit over the degenerate
    // reduction band only; full derivation at ff_math.hpp:sincos.  TF's limbs
    // are FP32 too, so the band is the same shape: below 2^nq * FLT_MIN the
    // leading word of r = s3/2^nq is subnormal and sheds bits before the first
    // Taylor term (and is 0 outright for subnormal |a|).  Measured before:
    // TF sin(1e-40) = 9.99967e-41 for an argument of 9.99995e-41, and
    // TF sin(1e-44) = 0.  Above the band the series is already correct and a
    // wider cut was measured to cost complex-op digits -- see ff_math.hpp.
    // nq is declared below (== 4); the constant is 2^4 * FLT_MIN spelled out.
    if (detail::fabs(a.f0) < 1.8807842e-37f /* 2^4 * FLT_MIN */) {
        sin_a = a;
        cos_a = TripleFloat(1.0f);
        return;
    }

    // Reduce mod 2π
    TripleFloat z = round_to_nearest_int(divide(a, k_2pi));
    TripleFloat r = subtract(a, multiply(k_2pi, z));

    // Reduce by 2^nq
    const int nq = 4;
    r = divide_scalar(r, float(1 << nq));

    // Taylor: sin(r) = r - r^3/3! + ..., cos(r) = 1 - r^2/2! + ...
    TripleFloat r2 = sqr(r);
    TripleFloat sin_r = r;
    TripleFloat cos_r = TripleFloat(1.0f);
    TripleFloat term_sin = r;
    TripleFloat term_cos = TripleFloat(1.0f);
    int k = 1;

    while (k < 64 && (abs(term_sin).f0 > 1.0e-21f * abs(sin_r).f0 ||
                      abs(term_cos).f0 > 1.0e-21f * abs(cos_r).f0)) {
        term_sin = divide_scalar(multiply(term_sin, r2), -float((2*k) * (2*k+1)));
        term_cos = divide_scalar(multiply(term_cos, r2), -float((2*k-1) * (2*k)));
        sin_r = add(sin_r, term_sin);
        cos_r = add(cos_r, term_cos);
        k++;
    }

    // Joint angle-doubling: sin(2θ) = 2·sin(θ)·cos(θ), cos(2θ) = cos²(θ) - sin²(θ)
    for (int i = 0; i < nq; i++) {
        TripleFloat s = multiply(sin_r, cos_r);
        s = add(s, s);
        TripleFloat c = subtract(sqr(cos_r), sqr(sin_r));
        sin_r = s;
        cos_r = c;
    }

    // KI-26 codomain guard: outside the slack band -> identity point
    // (sin, cos) = (0, 1), inside it -> clamp, so |sin| <= 1 and |cos| <= 1 hold
    // exactly for every finite input.  Full rationale at dd_math.hpp:sincos.
    // TF had no guard of any kind and returned NaN from 3.16e25 upward; its
    // exactly-representable-integer ceiling is 2^72, so a/2pi stops resolving
    // around |a| ~ 3e22.  Note the out-param order here is (sin, cos).
    const float kSlack = 1.0009765625f;   // 1 + 2^-10
    if (!(detail::fabs(sin_r.f0) <= kSlack) || !(detail::fabs(cos_r.f0) <= kSlack)) {
        XPMATH_PRINTF("TFCSSNR: argument reduction under-determined\n");
        sin_a = TripleFloat(0.0f); cos_a = TripleFloat(1.0f);
        return;
    }
    if (sin_r.f0 >  1.0f || (sin_r.f0 ==  1.0f && sin_r.f1 > 0.0f)) sin_r = TripleFloat( 1.0f);
    if (sin_r.f0 < -1.0f || (sin_r.f0 == -1.0f && sin_r.f1 < 0.0f)) sin_r = TripleFloat(-1.0f);
    if (cos_r.f0 >  1.0f || (cos_r.f0 ==  1.0f && cos_r.f1 > 0.0f)) cos_r = TripleFloat( 1.0f);
    if (cos_r.f0 < -1.0f || (cos_r.f0 == -1.0f && cos_r.f1 < 0.0f)) cos_r = TripleFloat(-1.0f);

    sin_a = sin_r;
    cos_a = cos_r;
}

XPMATH_INLINE_FUNCTION TripleFloat sin(TripleFloat a) {
    TripleFloat s, c;
    sincos(a, s, c);
    return s;
}

XPMATH_INLINE_FUNCTION TripleFloat cos(TripleFloat a) {
    TripleFloat s, c;
    sincos(a, s, c);
    return c;
}

XPMATH_INLINE_FUNCTION TripleFloat tan(TripleFloat a) {
    TripleFloat s, c;
    sincos(a, s, c);
    return divide(s, c);
}

// sinh/cosh: for small |a|, Taylor; otherwise (e^a ± e^-a)/2.
// Taylor threshold 0.5 per PORT_NOTES_QF.md §8 rationale (applies to TF).
// KI-6/KI-7: past this |a| the smaller exponential e^{-2|a|} is below TF's
// resolution u = 2^-72, so cosh == sinh == e^{|a|}/2 and tanh == ±1 exactly.
// ln(1/u)/2 = 25.0, rounded up; must stay under 44.36, where a doubled argument
// would leave FP32's exp range.
constexpr float kTFHyperbolicSaturate = 27.0f;

XPMATH_INLINE_FUNCTION void sinhcosh(TripleFloat a, TripleFloat& sinh_a, TripleFloat& cosh_a) {
    if (abs(a).f0 > kTFHyperbolicSaturate) {
        // See dd_math.hpp. Also removes the reciprocal 1/e^{|a|}, which
        // underflowed FP32 and produced NaN for |a| >= 80.
        TripleFloat aa = (a.f0 < 0.0f) ? negate(a) : a;
        TripleFloat h;
        // KI-9 crossover; see qf_math.hpp:sinhcosh. TF's last limb sits at
        // f0 * 2^-48, so exp(a) keeps all three words while
        // a > ln(FLT_MIN) + 48 ln2 = -54.1; -55 is that with a small margin
        // (the sweep still favours the reciprocal at a = -53.41).
        const float kTFReciprocalFloor = -55.0f;
        if (a.f0 < kTFReciprocalFloor) {
            // KI-9; see qf_math.hpp:sinhcosh for the full account. exp(a) for
            // large negative a lands near 1e-35, where TF's trailing limbs are
            // subnormal or zero and only ~10 digits survive; reciprocating that
            // cannot restore them. The NaN this branch used to get from the
            // unfixed divide() was accidentally routing it to the accurate
            // exp(|a| - ln2) form. Do that on purpose.
            h = exp(subtract(aa, TripleFloat_log2()));
        } else {
            TripleFloat e = exp(a);
            if (a.f0 < 0.0f) e = divide(TripleFloat(1.0f), e);
            h = (detail::isinf(e.f0) || e.f0 != e.f0) ? exp(subtract(aa, TripleFloat_log2()))
                                                      : mul_pwr2(e, 0.5f);
        }
        cosh_a = h;
        sinh_a = (a.f0 < 0.0f) ? negate(h) : h;
        return;
    }
    if (abs(a).f0 < 0.5f) {
        // Taylor for sinh: a + a^3/3! + a^5/5! + ...
        TripleFloat a2 = sqr(a);
        TripleFloat sinh_r = a;
        TripleFloat cosh_r = TripleFloat(1.0f);
        TripleFloat term_sinh = a;
        TripleFloat term_cosh = TripleFloat(1.0f);
        int k = 1;

        while (k < 64 && (abs(term_sinh).f0 > 1.0e-21f * abs(sinh_r).f0 ||
                          abs(term_cosh).f0 > 1.0e-21f * abs(cosh_r).f0)) {
            term_sinh = divide_scalar(multiply(term_sinh, a2), float((2*k) * (2*k+1)));
            term_cosh = divide_scalar(multiply(term_cosh, a2), float((2*k-1) * (2*k)));
            sinh_r = add(sinh_r, term_sinh);
            cosh_r = add(cosh_r, term_cosh);
            k++;
        }

        sinh_a = sinh_r;
        cosh_a = cosh_r;
    } else {
        TripleFloat ea = exp(a);
        TripleFloat einv = divide(TripleFloat(1.0f), ea);
        sinh_a = mul_pwr2(subtract(ea, einv), 0.5f);
        cosh_a = mul_pwr2(add(ea, einv), 0.5f);
    }
}

XPMATH_INLINE_FUNCTION TripleFloat sinh(TripleFloat a) {
    TripleFloat s, c;
    sinhcosh(a, s, c);
    return s;
}

XPMATH_INLINE_FUNCTION TripleFloat cosh(TripleFloat a) {
    TripleFloat s, c;
    sinhcosh(a, s, c);
    return c;
}

XPMATH_INLINE_FUNCTION TripleFloat tanh(TripleFloat a) {
    // KI-7: saturate first. Past kTFHyperbolicSaturate the sinh/cosh branch
    // returns the SAME TripleFloat for both (they agree to the last bit there),
    // and dividing +inf by +inf at the top of the range gave NaN — TF's variant
    // of the wrong-sign defect the other three backends showed.
    if (abs(a).f0 > kTFHyperbolicSaturate)
        return TripleFloat(a.f0 < 0.0f ? -1.0f : 1.0f);
    TripleFloat s, c;
    sinhcosh(a, s, c);
    return divide(s, c);
}

// ============================================================
// Inverse trigonometric functions — Newton iteration on sin/cos
//
// Port of qd_real::atan2, QD 2.3.24 qd_real.cpp:2393-2458, specialized to k=3;
// atan / asin / acos are QD's own wrappers around it (qd_real.cpp:2389-2391,
// 2479-2491, 2494-2506). QD's strategy, quoted from its comment block
// (qd_real.cpp:2394-2409): rather than a Taylor series for arctan, solve
//
//     sin(z) = y/r   or   cos(z) = x/r,      r = sqrt(x^2 + y^2)
//
// by Newton's iteration
//
//     z' = z + (y - sin z) / cos z      (equation 1)
//     z' = z - (x - cos z) / sin z      (equation 2)
//
// with x, y normalized so x^2 + y^2 = 1. QD picks equation 1 when |x| > |y|
// (larger denominator), equation 2 otherwise.
//
// k=3 ITERATION COUNT: 2 (QD uses 3 at k=4/FP64; see PORT_NOTES_TF.md §11a).
// Newton on a simple root doubles the correct-bit count per step. The seed is
// the FP32 std::atan2 of the LEADING WORDS only, so it carries ~24 bits (one
// FP32 word); the iterates then run
//     24 -> 48 -> 96 bits,
// and 96 >= 72 = the TripleFloat width, so the SECOND iteration saturates.
// QD's k=4 target is 212 bits from a 53-bit FP64 seed (53 -> 106 -> 212), which
// is why it spends three. A third step here is pure cost: it cannot add bits
// past the width, and each step costs a full sincos + divide. Measured at k=3,
// 2 and 3 iterations agree to the last measured digit (§11c).
//
// There is NO separate convergence threshold: the iteration count is fixed, as
// it is in QD (three straight-line repetitions, no residual test). The eps used
// by exp/sincos (1e-21f, §3a) is not involved.
// ============================================================
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
XPMATH_INLINE_FUNCTION float tf_pow2_unit_scale(float m) {
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
XPMATH_INLINE_FUNCTION TripleFloat tf_pow2_scale(TripleFloat a, float s) {
    return TripleFloat(a.f0 * s, a.f1 * s, a.f2 * s);
}
}  // namespace detail

// Band in which a sum of squares may be formed DIRECTLY, shared by every TF
// site that forms one (hypot here; complex abs and complex operator/ in
// tf_complex.hpp).  Low edge is 2^-37, one binade above the derived
// 2^-39 word-underflow limit of KI-8 note (1) at ff_math.hpp's hypot; the
// high edge is unchanged from the original fix.
namespace detail {
inline constexpr float kTFSqLo = 1.8189894e-12f;
inline constexpr float kTFSqHi = 1.0e18f;
}

XPMATH_INLINE_FUNCTION TripleFloat angle(TripleFloat x, TripleFloat y) {
    const TripleFloat pi   = TripleFloat_pi();
    const TripleFloat pi_2 = mul_pwr2(pi, 0.5f);    // exact: power-of-2 scaling
    const TripleFloat pi_4 = mul_pwr2(pi, 0.25f);   // exact: power-of-2 scaling

    // Degenerate axes.  QD qd_real.cpp:2411-2422.
    if (x.f0 == 0.0f) {
        if (y.f0 == 0.0f) {
            // QD raises an error and returns NaN here (qd_real.cpp:2415-2417).
            // TF follows qf_math.hpp:1014 and returns 0 with a diagnostic —
            // see PORT_NOTES_TF.md §11b for this deliberate divergence.
            XPMATH_PRINTF("TFATAN2: both arguments zero\n");
            return TripleFloat(0.0f);
        }
        return (y.f0 > 0.0f) ? pi_2 : negate(pi_2);
    }
    if (y.f0 == 0.0f) {
        return (x.f0 > 0.0f) ? TripleFloat(0.0f) : pi;
    }

    // Exact octant cases.  QD qd_real.cpp:2424-2430. (qf_math.hpp omits these;
    // they are in the source, so TF carries them — PORT_NOTES_TF.md §11b.)
    // 3pi/4 is one rounding off pi (multiply_scalar), unlike the exact pi/2 and
    // pi/4 scalings; QD stores it as its own 4-word constant qd_real::_3pi4.
    if (x == y || x == negate(y)) {
        const TripleFloat pi_34 = multiply_scalar(pi, 0.75f);
        if (x == y) return (y.f0 > 0.0f) ? pi_4 : negate(pi_34);
        return (y.f0 > 0.0f) ? pi_34 : negate(pi_4);
    }

    // Normalize onto the unit circle.  QD qd_real.cpp:2432-2434.
    // KI-13.  The r below is a sum of squares and has exactly hypot's exposure:
    // it overflowed the word above |x| ~ 1.8e19 (atan(3.16e19) returned NaN, and
    // atan2/asin/acos/complex arg inherited it) and shed low words below the
    // derived 2^-39 (see the KI-8 note at ff_math.hpp's hypot).  angle is
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
        if (mm > detail::kTFSqHi) {
            const float s = detail::tf_pow2_unit_scale(mm);
            x = detail::tf_pow2_scale(x, s);
            y = detail::tf_pow2_scale(y, s);
        }
    }
    TripleFloat r  = sqrt(add(sqr(x), sqr(y)));
    TripleFloat xx = divide(x, r);
    TripleFloat yy = divide(y, r);

    // FP32 seed from the leading words.  QD seeds from to_double(y)/to_double(x),
    // i.e. the UNNORMALIZED pair (qd_real.cpp:2437); kept here.
    TripleFloat z = TripleFloat(detail::atan2(y.f0, x.f0));
    TripleFloat sin_z, cos_z;

    if (detail::fabs(xx.f0) > detail::fabs(yy.f0)) {
        // Equation 1: z' = z + (yy - sin z)/cos z.  QD qd_real.cpp:2441-2447.
        for (int k = 0; k < 2; ++k) {
            sincos(z, sin_z, cos_z);
            z = add(z, divide(subtract(yy, sin_z), cos_z));
        }
    } else {
        // Equation 2: z' = z - (xx - cos z)/sin z.  QD qd_real.cpp:2449-2455.
        for (int k = 0; k < 2; ++k) {
            sincos(z, sin_z, cos_z);
            z = subtract(z, divide(subtract(xx, cos_z), sin_z));
        }
    }

    return z;
}

// atan2(y, x) = angle(x, y).  QD qd_real.cpp:2393 (STL argument order).
XPMATH_INLINE_FUNCTION TripleFloat atan2(TripleFloat y, TripleFloat x) {
    return angle(x, y);
}

// atan(a) = atan2(a, 1).  QD qd_real.cpp:2389-2391.
// KI-25 codomain clamp -- rationale in dd_math.hpp atan().  atan2 is not
// clamped; its range is (-pi, pi].
XPMATH_INLINE_FUNCTION TripleFloat atan(TripleFloat a) {
    TripleFloat r = angle(TripleFloat(1.0f), a);
    const TripleFloat p = TripleFloat_pi();
    const TripleFloat h(p.f0 * 0.5f, p.f1 * 0.5f, p.f2 * 0.5f);
    const TripleFloat ar = (r.f0 < 0.0f) ? TripleFloat(-r.f0, -r.f1, -r.f2) : r;
    if (subtract(ar, h).f0 > 0.0f)
        return (r.f0 < 0.0f) ? TripleFloat(-h.f0, -h.f1, -h.f2) : h;
    return r;
}

// KI-16.  Compare the VALUE, not the leading word.  Full derivation at
// dd_math.hpp's dd_cmp_one: `a.f0` is the value rounded to one FP32, and near a
// domain edge at +-1 that rounding crosses the edge in both directions -- it
// rejects legal arguments strictly inside the domain and accepts illegal ones
// just outside.  `subtract` renormalises and, for |a| in [1/2, 2], cancels the
// leading words EXACTLY by Sterbenz, so the residual words decide and decide
// correctly.  Returns -1, 0, +1 for a < 1, a == 1, a > 1.
XPMATH_INLINE_FUNCTION int tf_cmp_one(TripleFloat a) {
    const TripleFloat d = subtract(a, TripleFloat(1.0f));
    if (d.f0 > 0.0f) return  1;
    if (d.f0 < 0.0f) return -1;
    return 0;
}
XPMATH_INLINE_FUNCTION int tf_cmp_abs_one(TripleFloat a) { return tf_cmp_one(abs(a)); }

// asin(a) = atan2(a, sqrt(1 - a^2)).  QD qd_real.cpp:2479-2491.
XPMATH_INLINE_FUNCTION TripleFloat asin(TripleFloat a) {
    TripleFloat abs_a = abs(a);
    const int c = tf_cmp_one(abs_a);                                    // KI-16
    if (c > 0) {
        XPMATH_PRINTF("TFASIN: argument out of domain\n");
        return TripleFloat(0.0f);
    }
    // |a| == 1 exactly: sqrt(1-a^2) is 0 and the Newton denominator degenerates.
    // QD short-circuits to +-pi/2 (qd_real.cpp:2487-2489).  The old test read
    // the three words directly; tf_cmp_one is the same test, spelled once.
    if (c == 0) {
        TripleFloat pi_2 = mul_pwr2(TripleFloat_pi(), 0.5f);
        return (a.f0 > 0.0f) ? pi_2 : negate(pi_2);
    }
    return angle(sqrt(subtract(TripleFloat(1.0f), sqr(a))), a);
}

// acos(a) = atan2(sqrt(1 - a^2), a).  QD qd_real.cpp:2494-2506.
// NOTE this is QD's form, not the pi/2 - asin(a) the Phase-1 placeholder used:
// the subtraction loses digits to cancellation as a -> 1.
XPMATH_INLINE_FUNCTION TripleFloat acos(TripleFloat a) {
    TripleFloat abs_a = abs(a);
    const int c = tf_cmp_one(abs_a);                                    // KI-16
    if (c > 0) {
        XPMATH_PRINTF("TFACOS: argument out of domain\n");
        return TripleFloat(0.0f);
    }
    if (c == 0) {
        return (a.f0 > 0.0f) ? TripleFloat(0.0f) : TripleFloat_pi();
    }
    return angle(a, sqrt(subtract(TripleFloat(1.0f), sqr(a))));
}

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
// KI-17.  asinh is ODD, and TF was the one backend that did not say so.  For
// a < 0 the closed form below computes sqrt(a^2+1) ~ |a| and then adds a to it,
// which is a catastrophic cancellation that lands on or below zero -- log then
// sees a non-positive argument and bails to 0.  TF asinh(-1e12) returned 0
// while asinh(+1e12) was correct; the asymmetry was the whole diagnosis.
// Reflecting first is exact (a sign flip on every word) and free, and it is
// what DD, FF and QF have always done.  Verified while fixing this: the REAL
// asinh in DD (dd_math.hpp), FF (ff_math.hpp) and QF (qf_math.hpp) each already
// open with the same `if (leading word < 0) return negate(asinh(negate(a)));`,
// so TF was the sole omission.  (KI-5(a) fixed the COMPLEX case separately.)
// The kTFSqHi branch below used to re-apply the sign itself; with the reflection
// in front of it, it only ever sees a >= 0, so that bookkeeping is gone.
XPMATH_INLINE_FUNCTION TripleFloat asinh(TripleFloat a) {
    if (a.f0 < 0.0f) return negate(asinh(negate(a)));                   // KI-17
    if (a.f0 > detail::kTFSqHi) {
        TripleFloat u  = divide(TripleFloat(1.0f), a);
        u = multiply(u, u);
        return add(log(a), log(add(TripleFloat(1.0f), sqrt(add(TripleFloat(1.0f), u)))));
    }
    // KI-22: log1p(a + a^2/(1+sqrt(a^2+1))).  Derivation at dd_math.hpp's asinh.
    // TF is the backend that MEASURED the series alternative as a regression --
    // 36 terms of rounding at the crossover to save 1 bit of cancellation.
    const TripleFloat a2 = sqr(a);
    const TripleFloat s  = sqrt(add(a2, TripleFloat(1.0f)));
    if (a.f0 < 0.5f) {
        return log1p(add(a, divide(a2, add(TripleFloat(1.0f), s))));
    }
    return log(add(a, s));
}

XPMATH_INLINE_FUNCTION TripleFloat acosh(TripleFloat a) {
    // KI-16 audit.  TF was the one backend with NO acosh domain guard at all:
    // a < 1 fell through to sqrt of a negative, whose own guard printed and
    // returned 0, and log(a + 0) then returned a silently wrong negative number
    // instead of the diagnostic.  Value-based from the start, per dd_cmp_one.
    if (tf_cmp_one(a) < 0) {
        XPMATH_PRINTF("TFACOSH: argument < 1\n"); return TripleFloat(0.0f); }
    if (a.f0 > detail::kTFSqHi) {
        TripleFloat u = divide(TripleFloat(1.0f), a);
        u = multiply(u, u);
        return add(log(a), log(add(TripleFloat(1.0f), sqrt(subtract(TripleFloat(1.0f), u)))));
    }
    return log(add(a, sqrt(subtract(sqr(a), TripleFloat(1.0f)))));
}

XPMATH_INLINE_FUNCTION TripleFloat atanh(TripleFloat a) {
    // KI-16 audit.  TF was the one backend with NO atanh domain guard: |a| >= 1
    // fell through to log of a non-positive quotient, which printed TFLOG's
    // diagnostic rather than TFATANH's and returned 0 by accident rather than
    // by decision.  Value-based from the start, per dd_cmp_one.
    // |a| == 1 is the C99 pole, not a domain error: atanh(+-1) = +-inf.  TF's
    // guardless code reached that by accident; the guard must preserve it.
    const int c_atanh = tf_cmp_abs_one(a);
    if (c_atanh > 0) {
        XPMATH_PRINTF("TFATANH: |argument| > 1\n"); return TripleFloat(0.0f); }
    if (c_atanh == 0) return TripleFloat(a.f0 > 0.0f ? HUGE_VALF : -HUGE_VALF);
    if (abs(a).f0 < 0.5f) {
        // Taylor: a + a^3/3 + a^5/5 + ...
        TripleFloat a2 = sqr(a);
        TripleFloat sum = a;
        TripleFloat term = a;
        int k = 1;
        while (k < 64 && abs(term).f0 > 1.0e-21f * abs(sum).f0) {
            term = multiply(term, a2);
            sum = add(sum, divide_scalar(term, float(2*k + 1)));
            k++;
        }
        return sum;
    } else {
        TripleFloat one_plus = add(TripleFloat(1.0f), a);
        TripleFloat one_minus = subtract(TripleFloat(1.0f), a);
        return mul_pwr2(log(divide(one_plus, one_minus)), 0.5f);
    }
}

// exp2, exp10, expm1, log1p, log10 (derived from exp/log)
XPMATH_INLINE_FUNCTION TripleFloat exp2(TripleFloat a) {
    return exp(multiply(a, TripleFloat_log2()));
}

XPMATH_INLINE_FUNCTION TripleFloat exp10(TripleFloat a) {
    return exp(multiply(a, TripleFloat_log10()));
}

XPMATH_INLINE_FUNCTION TripleFloat expm1(TripleFloat a) {
    if (detail::fabs(a.f0) > 0.5f) {
        // |e^a - 1| > e^0.5 - 1 ~ 0.65: subtracting 1 costs no significant
        // cancellation.  Crossover derived at dd_math.hpp's asinh (Sterbenz).
        return subtract(exp(a), TripleFloat(1.0f));
    }
    // KI-22.  a + a^2/2! + a^3/3! + ...   Below the crossover `e^a - 1` throws
    // away every bit of a below 2^-72 relative to 1: at a = 1e-9 that left
    // 10.81 of TF's 21.7 digits.  TF's u = 2^-72 needs a relative tail
    // a^(N-1)/N! <= u, which at |a| = 1/2 is N = 19 terms.  DD, FF and QF all
    // already had this branch; TF was the omission.
    TripleFloat sum = a, term = a;
    for (int k = 2; k <= 40; ++k) {
        term = divide_scalar(multiply(term, a), (float)k);
        sum  = add(sum, term);
        if (detail::fabs(term.f0) < 1.0e-21f * detail::fabs(sum.f0)) break;
    }
    return sum;
}

// Series 2*atanh(a/(2+a)) for |a| < 1/4, plain log(1+a) outside; see
// dd_math.hpp's log1p for the derivation (KI-5(b)), including why Goldberg's
// correction was measured and rejected. The old body log(a+1) lost
// log10(1/|a|) digits for small a.
XPMATH_INLINE_FUNCTION TripleFloat log1p(TripleFloat a) {
    if (detail::fabs(a.f0) < 0.25f) {
        TripleFloat t   = divide(a, add(TripleFloat(2.0f), a));
        TripleFloat t2  = multiply(t, t);
        TripleFloat sum = t;
        TripleFloat trm = t;
        for (int k = 3; k < 80; k += 2) {
            trm = multiply(trm, t2);
            TripleFloat incr = divide(trm, TripleFloat((float)k));
            sum = add(sum, incr);
            if (detail::fabs(incr.f0) <= 1.0e-24f * detail::fabs(sum.f0)) break;
        }
        return multiply_scalar(sum, 2.0f);
    }
    return log(add(a, TripleFloat(1.0f)));
}

XPMATH_INLINE_FUNCTION TripleFloat log10(TripleFloat a) {
    return divide(log(a), TripleFloat_log10());
}

XPMATH_INLINE_FUNCTION TripleFloat log2(TripleFloat a) {
    return divide(log(a), TripleFloat_log2());
}

// Rounding/data ops
XPMATH_INLINE_FUNCTION TripleFloat ceil(TripleFloat a) {
    float f0 = detail::ceil(a.f0);
    float f1 = 0.0f, f2 = 0.0f;

    if (f0 == a.f0) {
        f1 = detail::ceil(a.f1);
        if (f1 == a.f1) {
            f2 = detail::ceil(a.f2);
        }
    }
    renorm(f0, f1, f2);
    return TripleFloat(f0, f1, f2);
}

XPMATH_INLINE_FUNCTION TripleFloat floor(TripleFloat a) {
    float f0 = detail::floor(a.f0);
    float f1 = 0.0f, f2 = 0.0f;

    if (f0 == a.f0) {
        f1 = detail::floor(a.f1);
        if (f1 == a.f1) {
            f2 = detail::floor(a.f2);
        }
    }
    renorm(f0, f1, f2);
    return TripleFloat(f0, f1, f2);
}

XPMATH_INLINE_FUNCTION TripleFloat trunc(TripleFloat a) {
    return (a.f0 >= 0.0f) ? floor(a) : ceil(a);
}

XPMATH_INLINE_FUNCTION TripleFloat round(TripleFloat a) {
    return round_to_nearest_int(a);
}

// fmod(a, b) = a - b*aint(a/b), where aint TRUNCATES toward zero. Port of
// QD 2.3.24 qd_real.cpp:2597-2600 (`qd_real n = aint(a / b); return (a - b*n);`)
// with aint = qd_inline.h:975-977 (`(a[0] >= 0) ? floor(a) : ceil(a)`), which is
// exactly xp::trunc. Mirrors qf_math.hpp:1167-1171.
//
// S10 Phase 3.5: this called round_to_nearest_int (nint) rather than trunc —
// that is QD's `drem`, not its `fmod`. The two differ by a whole b whenever the
// fractional part of a/b exceeds 1/2, i.e. on about half of all inputs, so half
// the samples scored 0 digits and the row measured 11.26 against fmodq.
// See PORT_NOTES_TF.md §12c.
// S10 KI-10/KI-15: the QD shape above is replaced by an exact iterative
// scale-and-subtract that never forms a quotient. Full derivation — loop
// bound, termination, and why every subtraction is exact by Sterbenz — is in
// dd_math.hpp. QD 2.3.24 shares the defect; diverging from it is deliberate.
// Conventions: C99 fmod (sign of a) and C99/IEEE-754 remainder (round-half-to-
// EVEN quotient, |r| <= |b|/2 — NOT QD's half-away nint, see KI-20).
// fmod(a,0), remainder(a,0), fmod(+-inf,b), remainder(+-inf,b) -> NaN;
// f(a,+-inf) = a for finite a; f(+-0,b) = +-0.
namespace detail {

// |A| mod |B| exactly, plus the parity of the integral quotient.
// Precondition: A >= 0, B > 0, both finite.
XPMATH_INLINE_FUNCTION TripleFloat tf_fmod_abs(TripleFloat A, TripleFloat B,
                                               bool& q_odd) {
    q_odd = false;
    if (A < B) return A;

    TripleFloat Bs = B;
    int k = 0;
    for (;;) {
        TripleFloat t = mul_pwr2(Bs, 1073741824.0f);  // 2^30
        if (t > A) break;
        Bs = t; k += 30;
    }
    for (;;) {
        TripleFloat t = mul_pwr2(Bs, 2.0f);
        if (t > A) break;
        Bs = t; ++k;
    }

    TripleFloat r = A;
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

XPMATH_INLINE_FUNCTION TripleFloat fmod(TripleFloat a, TripleFloat b) {
    if (a.f0 != a.f0 || b.f0 != b.f0) return TripleFloat(a.f0 + b.f0);
    if (b.f0 == 0.0f) { XPMATH_PRINTF("TFFMOD: zero modulus\n");
                        return TripleFloat(0.0f / 0.0f); }
    if (!detail::isfinite(a.f0)) { XPMATH_PRINTF("TFFMOD: infinite dividend\n");
                                   return TripleFloat(0.0f / 0.0f); }
    if (!detail::isfinite(b.f0)) return a;
    if (a.f0 == 0.0f) return a;

    bool q_odd = false;
    TripleFloat r = detail::tf_fmod_abs(abs(a), abs(b), q_odd);
    return (a.f0 < 0.0f) ? negate(r) : r;
}

// remainder(a, b) = a - b*nint(a/b). Port of QD's `drem`, qd_real.cpp:2462-2465.
// Mirrors qf_math.hpp:1175-1179. Through S10 Phase 3 this was `return fmod(a,b);`
// and scored correctly only because fmod itself carried drem's nint; each now
// has its own QD body.
XPMATH_INLINE_FUNCTION TripleFloat remainder(TripleFloat a, TripleFloat b) {
    if (a.f0 != a.f0 || b.f0 != b.f0) return TripleFloat(a.f0 + b.f0);
    if (b.f0 == 0.0f) { XPMATH_PRINTF("TFREMAINDER: zero modulus\n");
                        return TripleFloat(0.0f / 0.0f); }
    if (!detail::isfinite(a.f0)) { XPMATH_PRINTF("TFREMAINDER: infinite dividend\n");
                                   return TripleFloat(0.0f / 0.0f); }
    if (!detail::isfinite(b.f0)) return a;
    if (a.f0 == 0.0f) return a;

    bool q_odd = false;
    TripleFloat B = abs(b);
    TripleFloat r = detail::tf_fmod_abs(abs(a), B, q_odd);
    TripleFloat two_r = mul_pwr2(r, 2.0f);
    if (two_r > B || (two_r == B && q_odd)) r = subtract(r, B);  // Sterbenz-exact
    return (a.f0 < 0.0f) ? negate(r) : r;
}

XPMATH_INLINE_FUNCTION TripleFloat fdim(TripleFloat a, TripleFloat b) {
    return (a > b) ? subtract(a, b) : TripleFloat(0.0f);
}

XPMATH_INLINE_FUNCTION TripleFloat fmax(TripleFloat a, TripleFloat b) {
    return (a > b) ? a : b;
}

XPMATH_INLINE_FUNCTION TripleFloat fmin(TripleFloat a, TripleFloat b) {
    return (a < b) ? a : b;
}

XPMATH_INLINE_FUNCTION TripleFloat fma(TripleFloat a, TripleFloat b, TripleFloat c) {
    return add(multiply(a, b), c);
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
// Same reasoning as the atanh threshold in tf_complex.hpp -- fix the interval
// that is broken, do not churn the one that is not.
//
// inf/nan convention (C99 F.9.4.3): hypot(+-inf, y) is +inf for ANY y, NaN
// included, so the inf test comes first.  Otherwise a NaN operand propagates to
// NaN through the arithmetic.  Both operands are taken through abs() first, so
// the returned infinity is always +inf.
XPMATH_INLINE_FUNCTION TripleFloat hypot(TripleFloat a, TripleFloat b) {
    TripleFloat x = abs(a);
    TripleFloat y = abs(b);
    if (detail::isinf(x.f0)) return x;
    if (detail::isinf(y.f0)) return y;
    TripleFloat m = (x.f0 < y.f0) ? y : x;
    TripleFloat n = (x.f0 < y.f0) ? x : y;
    if (m.f0 == 0.0f) return TripleFloat(0.0f);
    if (m.f0 <= detail::kTFSqHi && m.f0 >= detail::kTFSqLo)
        return sqrt(add(sqr(a), sqr(b)));
    TripleFloat t = divide(n, m);
    return multiply(m, sqrt(add(TripleFloat(1.0f), multiply(t, t))));
}

// copysign(a, b) = magnitude of a, sign of b. Exact sign manipulation; no QD
// analogue (cf. qf:1182-1186). Check the sign of the LOW words too: an
// expansion's words can have mixed signs and a naive f0-only check gets this
// wrong. The sign of a TF expansion is the sign of its FIRST NONZERO word.
XPMATH_INLINE_FUNCTION TripleFloat copysign(TripleFloat a, TripleFloat b) {
    TripleFloat r = abs(a);
    if (b.f0 < 0.0f || (b.f0 == 0.0f && (b.f1 < 0.0f || (b.f1 == 0.0f && b.f2 < 0.0f))))
        return negate(r);
    return r;
}

}  // namespace xp
