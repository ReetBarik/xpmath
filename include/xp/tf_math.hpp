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
#include <xp/trig_reduction.hpp>
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
// KI-44: the unevaluated-pair trio behind pow, defined after the expansion
// helpers they use (tf_expansion_push / _compress, ~line 1732).
namespace detail {
XPMATH_INLINE_FUNCTION TripleFloat tf_mul_ext(TripleFloat x, TripleFloat y, float& err);
XPMATH_INLINE_FUNCTION TripleFloat tf_log_ext(TripleFloat a, float& err);
XPMATH_INLINE_FUNCTION TripleFloat tf_exp_ext(TripleFloat a, float resid);
}  // namespace detail
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
namespace detail {
XPMATH_INLINE_FUNCTION TripleFloat tf_pow2_scale(TripleFloat a, float s);   // defined below

// KI-41 at the UNDERFLOW end.  Same defect, same remedy and same derivation as
// qf_math.hpp:divide — read it there.  TF's expansion is three words, so the
// residual's lowest word sits at |a|·2^-48 and its two_prod tail one word
// further down at |a|·2^-72; the predicate tests 2^-48 and the lift targets
// 2^-72.  Cap 2^96, for the reason qf_math.hpp gives.
XPMATH_INLINE_FUNCTION bool tf_div_lift_wanted(TripleFloat a, TripleFloat b) {
    const float a0 = detail::fabs(a.f0), b0 = detail::fabs(b.f0);
    if (a0 == 0.0f || b0 == 0.0f) return false;
    const float m = (a0 < b0) ? a0 : b0;
    return m * 0x1p-48f < 1.17549435e-38f * 4.0f;
}
XPMATH_INLINE_FUNCTION TripleFloat tf_divide_core(TripleFloat a, TripleFloat b) {
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
}  // namespace detail

XPMATH_INLINE_FUNCTION TripleFloat divide(TripleFloat a, TripleFloat b) {
    if (!detail::tf_div_lift_wanted(a, b)) return detail::tf_divide_core(a, b);
    const float step = 0x1p24f, target = 1.17549435e-38f * 4.0f, cap = 0x1p96f;
    float sa = 1.0f, sb = 1.0f;
    float pa = detail::fabs(a.f0), pb = detail::fabs(b.f0);
    // KI-41's own loop shape (qf_complex.hpp:325-352), not a paraphrase of it:
    // target the PRODUCT and lift whichever operand is currently smaller.  That
    // keeps sa/sb minimal, so the closing unscale moves the quotient as little
    // as the mechanism allows.
    for (int k = 0; k < 16 && (pa * pb) * 0x1p-72f < target && sa < cap && sb < cap; ++k) {
        if (pa < pb) { sa *= step; pa *= step; } else { sb *= step; pb *= step; }
    }
    const TripleFloat q = detail::tf_divide_core(detail::tf_pow2_scale(a, sa),
                                                 detail::tf_pow2_scale(b, sb));
    return detail::tf_pow2_scale(q, sb / sa);   // exact: both are powers of two
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

// Nearest FP32-int of a single float, ties toward +infinity. INTERNAL: KI-20
// made `tf::round` IEEE 754 half-even, and this routine deliberately did not
// follow — see qf_math.hpp's qf_nint for both reasons (the multi-word correction
// needs a fixed lean; the argument reductions do not care which). Mirrors
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

    // KI-42: Cody-Waite range reduction; see dd_math.hpp's exp for the
    // derivation and ff_math.hpp's for the FP32 width. `multiply_scalar(k_log2,
    // m)` was a rounded TF-by-scalar product with the same defect. Five 16-bit
    // pieces put the tail at 2^-89.3 (9.4e-04 ulps of 2^-72 at kmax); 0 of 1400
    // products inexact over k in [-151, 128]. Measured end-to-end over
    // a >= -54.1 (TF's subnormal wall, tf_math.hpp:917):
    //     shipped   702 rows > 1 ulp, worst 14.44
    //     4 pieces 1394 rows > 1 ulp, worst 242      <- one short is FATAL
    //     5 pieces    0 rows > 1 ulp, worst 0.8241
    // and 5 pieces equals a 400-bit oracle reduction, so nothing is left.
    // This is also why exp no longer calls multiply_scalar at all.
    const float kLn2_1 =  0x1.62e4p-1f;    // 15 significant bits
    const float kLn2_2 =  0x1.7f7ep-20f;   // 16
    const float kLn2_3 = -0x1.c61p-37f;    // 13
    const float kLn2_4 = -0x1.950ep-54f;   // 16
    const float kLn2_5 =  0x1.e3b4p-72f;   // 15
    TripleFloat r = subtract(a, TripleFloat(m * kLn2_1));
    r = subtract(r, TripleFloat(m * kLn2_2));
    r = subtract(r, TripleFloat(m * kLn2_3));
    r = subtract(r, TripleFloat(m * kLn2_4));
    r = subtract(r, TripleFloat(m * kLn2_5));

    const int nq = 5;
    r = divide_scalar(r, float(1 << nq));

    // KI-34: track s = e^r - 1, not e^r. The leading 1 is added back only after
    // the squarings, because squaring DOUBLES relative error while the
    // equivalent step on s, (1+s)^2 - 1 = s*(s+2), PRESERVES it. See
    // dd_math.hpp's exp for the derivation. nq = 5 here, so the shipped form
    // multiplied the series error by 32 and left `log` an absolute floor of
    // ~5.4 units of 2^-72, flat in |ln v|.
    TripleFloat s = r;
    TripleFloat t = sqr(r);
    TripleFloat term = t;
    int k = 2;

    while (k < 64 && abs(term).f0 > 1.0e-21f * abs(s).f0) {
        term = divide_scalar(term, float(k));
        s = add(s, term);
        term = multiply(term, r);
        k++;
    }

    // nq doublings on e^x - 1: s ← s*(s+2), then restore the leading 1.
    for (int i = 0; i < nq; i++) {
        s = multiply(s, add(s, TripleFloat(2.0f)));
    }
    s = add(TripleFloat(1.0f), s);

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
//
// The non-positive-base guard mirrors dd_math.hpp / ff_math.hpp / qf_math.hpp,
// which have carried it since their KI-19-era domain audit; TF was missed. Left
// bare, pow(0, b) fell through to log(0) -- which returns 0 after printing
// TFLOG's diagnostic -- and then exp(b*0) = 1, so TF answered 1 where the other
// three answer 0, with only the misleading TFLOG line to show for it.
//
// The sweep cannot catch this: repair_real's R_Pow case
// (scripts/sweep_accuracy.cpp:529-537) sets a = fabs(a) and maps a == 0 to 1,
// so no grid point ever presents pow with a non-positive base. Covered by a
// targeted test instead.
XPMATH_INLINE_FUNCTION TripleFloat pow(TripleFloat a, TripleFloat b) {
    if (a.f0 <= 0.0f) {
        if (a.f0 == 0.0f && b.f0 > 0.0f) return TripleFloat(0.0f);
        XPMATH_PRINTF("TFPOW: non-positive base\n");
        return TripleFloat(0.0f);
    }
    // KI-44: exponent kept as an unevaluated pair; see dd_math.hpp's pow.
    float le;
    const TripleFloat lp = detail::tf_log_ext(a, le);
    float e1;
    const TripleFloat p = detail::tf_mul_ext(lp, b, e1);
    return detail::tf_exp_ext(p, e1 + le * b.f0);
}

// sin/cos: joint computation via Payne-Hanek reduction mod π/2, divide-by-k
// Taylor on the residual, and joint angle-doubling formulas (PORT_NOTES.md
// §3a).  With nq = 4, r = r_mod/2^4, and sin(r)/cos(r) Taylor converges in ~7
// terms to reach TF width.  Four joint doublings recover sin/cos of r_mod, and
// the quadrant tables map that back onto a.
XPMATH_INLINE_FUNCTION void sincos(TripleFloat a, TripleFloat& sin_a, TripleFloat& cos_a) {
    if (a.f0 == 0.0f) {
        sin_a = TripleFloat(0.0f);
        cos_a = TripleFloat(1.0f);
        return;
    }
    // KI-12 residual.  Small-argument short circuit over the degenerate
    // reduction band only; full derivation at ff_math.hpp:sincos.  TF's limbs
    // are FP32 too, so the band is the same shape: below 2^nq * FLT_MIN the
    // leading word of r = r_mod/2^nq is subnormal and sheds bits before the first
    // Taylor term (and is 0 outright for subnormal |a|).  Measured before:
    // TF sin(1e-40) = 9.99967e-41 for an argument of 9.99995e-41, and
    // TF sin(1e-44) = 0.  Above the band the series is already correct and a
    // wider cut was measured to cost complex-op digits -- see ff_math.hpp.
    // nq is declared below and is now 0, so the band is FLT_MIN and not
    // 2^4 * FLT_MIN: with no scale-down the only way the leading word of r can
    // be subnormal is for a itself to be.  This is a NARROWING of the guard,
    // not a removal, and it is a no-op on the vacated strip rather than a
    // behaviour change -- for FLT_MIN <= |a| < 2^4*FLT_MIN the series now runs
    // and returns the same pair, because r^2 underflows to zero there, leaving
    // sin_r = r = a and v_r = 0 so cos_r = 1.
    //
    // The constant was spelled out literally when it was 2^4 * FLT_MIN and a
    // reader had to trust that 1.8807842e-37f was that product.  It is written
    // against nq now so it cannot drift from it again.
    if (detail::fabs(a.f0) < float(1 << 0) * 1.17549435e-38f /* 2^nq * FLT_MIN */) {
        sin_a = a;
        cos_a = TripleFloat(1.0f);
        return;
    }

    // ARGUMENT REDUCTION — Payne-Hanek.  a = j*(pi/2) + r_mod with |r_mod| <=
    // pi/4 and j = nint(a*2/pi) mod 4, with n never formed.  Mirrors
    // dd_math.hpp:sincos; why the old `a - 2pi*nint(a/2pi)` could not be
    // rescued by a wider constant is measured in
    // scripts/probe_trig_stages.cpp --widen, and kPhGuardTF / kPhChunksTF come
    // from scripts/gen_trig_reduction_constants.cpp.
    //
    // r = r_mod/2^nq below can no longer underflow: it is zero only for
    // |r_mod| < 2^nq * FLT_MIN = 2^-122, and |f| >= 2^-C with C = 78.649
    // MEASURED over every TripleFloat (include/xp/trig_reduction_data.hpp),
    // giving |r| >= 2^-83.  The KI-12 band above covers the only arguments that
    // can, which is what it was sized for.
    // nq = 0: no scale-down.  The FP32 subnormal mechanism and the measurement
    // behind it are at ff_math.hpp:sincos; TF's own worst cell moved from
    // 2.6e7 ulps to 0.59 on it, and its series length went 6 -> 10 terms.
    const int nq = 0;
    int         j;
    TripleFloat r_mod;
    if (detail::fabs(a.f0) <= 0.75f) {
        // Nothing to reduce: r_mod is a itself, exactly.  See dd_math.hpp for
        // the measurement behind both the branch and the 0.75.
        j = 0; r_mod = a;
    } else {
        const float win[3] = { a.f0, a.f1, a.f2 };
        float       f[4];
        j = detail::xp_ph_reduce<float>(win, 3, detail::kPhGuardTF,
                                        detail::kPhChunksTF, f, 4);
        // f is exact to 2^-kPhGuardTF ABSOLUTE, i.e. 2^-(p+4) RELATIVE at the
        // worst cancellation the format admits, so r_mod inherits ~2^-72
        // relative -- proportional to |r_mod|, where the old form's error was
        // proportional to |a|.
        TripleFloat pio2 = TripleFloat(detail::xp_ph_pio2_f(0));
        for (int k = 1; k < detail::kPhPio2WordsF; ++k)
            pio2 = add(pio2, TripleFloat(detail::xp_ph_pio2_f(k)));
        TripleFloat fr = TripleFloat(f[0]);
        for (int k = 1; k < 4; ++k) fr = add(fr, TripleFloat(f[k]));
        r_mod = multiply(fr, pio2);
    }

    // Reduce by 2^nq
    TripleFloat r = divide_scalar(r_mod, float(1 << nq));

    // Taylor: sin(r) = r - r^3/3! + ..., v(r) = r^2/2! - r^4/4! + ..., v = 1-cos.
    // Carried in (sin, v) rather than (sin, cos); derivation at dd_math.hpp:sincos.
    //
    // v STARTS ONE TERM IN.  The cos series began at term_cos = 1, so its loop
    // body produced -r^2/2 on the first pass; v_r is initialised TO r^2/2, so
    // the first pass must produce -r^4/4! instead.  That is why the ratio below
    // is -(2k+1)(2k+2) where the cos ratio was -(2k-1)(2k) -- same terms, index
    // shifted by one, not a different series.
    TripleFloat r2 = sqr(r);
    TripleFloat sin_r = r;
    TripleFloat term_sin = r;
    TripleFloat v_r = divide_scalar(r2, 2.0f);
    TripleFloat term_v = v_r;
    int k = 1;

    // TF alone tests convergence BEFORE the update, in a while loop, where FF
    // and QF test after in a for loop.  That is preserved verbatim: the v test
    // simply replaces the cos test in the same position, and it is relative
    // (against v_r) exactly as the cos test was relative (against cos_r).
    while (k < 64 && (abs(term_sin).f0 > 1.0e-21f * abs(sin_r).f0 ||
                      abs(term_v).f0   > 1.0e-21f * abs(v_r).f0)) {
        term_sin = divide_scalar(multiply(term_sin, r2), -float((2*k) * (2*k+1)));
        term_v   = divide_scalar(multiply(term_v,   r2), -float((2*k+1) * (2*k+2)));
        sin_r = add(sin_r, term_sin);
        v_r   = add(v_r, term_v);
        k++;
    }

    // Joint angle-doubling in (sin, v): sin(2θ) = 2·sin(θ)·(1-v), v(2θ) = 2·sin²(θ).
    // At nq = 0 this loop does not execute; kept because nq is what was measured.
    for (int i = 0; i < nq; i++) {
        const TripleFloat c_i = subtract(TripleFloat(1.0f), v_r);
        TripleFloat s = multiply(sin_r, c_i);
        s = add(s, s);
        v_r   = add(sqr(sin_r), sqr(sin_r));    // 2 sin^2, from the OLD sin_r
        sin_r = s;
    }
    TripleFloat cos_r = subtract(TripleFloat(1.0f), v_r);

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

    // Quadrant selection (TF has sin first, cos second)
    if (j == 0) { sin_a = sin_r;  cos_a = cos_r; }
    else if (j == 1) { sin_a = cos_r;  cos_a = negate(sin_r); }
    else if (j == 2) { sin_a = negate(sin_r); cos_a = negate(cos_r); }
    else { sin_a = negate(cos_r); cos_a = sin_r; }
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
    // `== 0.0f` is true of negative zero too, so the sign of a zero operand
    // cannot be recovered from the comparison and has to be read off copysign
    // -- IEEE-754 atan2 takes the sign of the result from y, and x = -0 belongs
    // on the pi side exactly like any negative x.  The y test has to run FIRST:
    // it subsumes the both-zero case, which the x test would otherwise answer
    // with +-pi/2 instead of the required +-pi / +-0.
    if (y.f0 == 0.0f) {
        if (x.f0 == 0.0f) {
            // QD raises an error and returns NaN here (qd_real.cpp:2415-2417).
            // TF follows qf_math.hpp:1014 and returns a value with a diagnostic
            // — see PORT_NOTES_TF.md §11b for this deliberate divergence.  The
            // value is now the IEEE-754 one (+-0 for x = +0, +-pi for x = -0)
            // rather than an unconditional 0; the diagnostic is unchanged.
            XPMATH_PRINTF("TFATAN2: both arguments zero\n");
        }
        const TripleFloat r =
            (detail::copysign(1.0f, x.f0) < 0.0f) ? pi : TripleFloat(0.0f);
        return (detail::copysign(1.0f, y.f0) < 0.0f) ? negate(r) : r;
    }
    if (x.f0 == 0.0f) {
        return (y.f0 > 0.0f) ? pi_2 : negate(pi_2);
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
    // KI-29: 1/2 log1p(2a(a+s)) rather than log(a+s).  TF is the backend that
    // FILED KI-29 (92 of its 140 decreases); the residual is log's constant
    // absolute error, not Sterbenz.  Derived at dd_math.hpp's asinh.
    const TripleFloat t = add(a, s);
    return mul_pwr2(log1p(mul_pwr2(multiply(a, t), 2.0f)), 0.5f);
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
// KI-43: see dd_math.hpp's exp2 for the derivation.
XPMATH_INLINE_FUNCTION TripleFloat exp2(TripleFloat a) {
    TripleFloat k = round_to_nearest_int(a);
    const int ki = (int)k.f0;
    TripleFloat r = subtract(a, k);                     // EXACT
    TripleFloat s = (r.f0 == 0.0f && r.f1 == 0.0f && r.f2 == 0.0f)
                  ? TripleFloat(1.0f)
                  : exp(multiply(r, TripleFloat_log2()));
    if (ki >= -125 && ki <= 127) return mul_pwr2(s, ldexpf(1.0f, ki));
    return TripleFloat(ldexpf(s.f0, ki), ldexpf(s.f1, ki), ldexpf(s.f2, ki));
}

// KI-43: see dd_math.hpp's exp10. Five 16-bit log10(2) pieces, tail 2^-95.0.
XPMATH_INLINE_FUNCTION TripleFloat exp10(TripleFloat a) {
    const float kLog2_10 = 3.32192809f;
    const float kf = detail::rint(a.f0 * kLog2_10);
    if (!(detail::fabs(kf) < 1.0e5f))
        return exp(multiply(a, TripleFloat_log10()));
    const int ki = (int)kf;
    const float kLog10_2_1 =  0x1.3442p-2f;
    const float kLog10_2_2 = -0x1.95ecp-19f;
    const float kLog10_2_3 = -0x1.0c02p-39f;
    const float kLog10_2_4 = -0x1.9dc2p-59f;
    const float kLog10_2_5 =  0x1.2b36p-78f;
    TripleFloat r = subtract(a,  TripleFloat(kf * kLog10_2_1));
    r = subtract(r, TripleFloat(kf * kLog10_2_2));
    r = subtract(r, TripleFloat(kf * kLog10_2_3));
    r = subtract(r, TripleFloat(kf * kLog10_2_4));
    r = subtract(r, TripleFloat(kf * kLog10_2_5));
    TripleFloat s = exp(multiply(r, TripleFloat_log10()));
    if (ki >= -125 && ki <= 127) return mul_pwr2(s, ldexpf(1.0f, ki));
    return TripleFloat(ldexpf(s.f0, ki), ldexpf(s.f1, ki), ldexpf(s.f2, ki));
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

// round(a) — nearest integer, TIES TO EVEN (IEEE 754 roundToIntegralTiesToEven).
// KI-20 (2026-09-04); the reasoning, and why round_to_nearest_int keeps QD's
// ties-toward-+infinity, is written out once in qf_math.hpp's `round`. In one
// line: this is the library's single user-visible tie convention, it diverges
// from both QD's `nint` and C99's `round`, and halving turns a tie into a
// non-tie whose nearest integer is the even neighbour.
XPMATH_INLINE_FUNCTION TripleFloat round(TripleFloat a) {
    const TripleFloat n = round_to_nearest_int(a);
    if (n.f0 == a.f0 && n.f1 == a.f1 && n.f2 == a.f2)
        return n;                                   // already integral: no tie
    const TripleFloat t  = mul_pwr2(a, 2.0f);       // exact
    const TripleFloat nt = round_to_nearest_int(t);
    if (!(nt.f0 == t.f0 && nt.f1 == t.f1 && nt.f2 == t.f2))
        return n;                                   // 2a not integral: no tie
    return mul_pwr2(round_to_nearest_int(mul_pwr2(a, 0.5f)), 2.0f);
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

// ---- KI-38: fma with an EXACT product ----------------------------------
// See dd_math.hpp:fma for the derivation. TF carried 44 of the 60 failing
// sweep points, scoring 6.07-10.81 digits against a 21.70 cap; all 44 reach
// the cap with the exact product. The operands at every one of those points
// are held EXACTLY by TripleFloat, so there was no conditioning floor to
// appeal to -- cap - log10(kappa) is the bound on PERTURBED inputs and the
// inputs here carry no perturbation.
//
// TF is also where the truncation subtlety showed up: taking the leading four
// components of the UNcompressed expansion stalled points 736/806/841/974/1583
// at 17.3-18.4 digits, because a nonoverlapping expansion's components need
// not be full-width and four of them can span fewer than 72 bits. COMPRESS
// repacks them first; then four components always suffice.
XPMATH_INLINE_FUNCTION void tf_expansion_push(float* e, int& m, float t) {
    if (t == 0.0f) return;
    float q = t;
    for (int i = 0; i < m; ++i) {
        float err;
        const float s = tf_two_sum(q, e[i], err);
        e[i] = err;
        q    = s;
    }
    e[m++] = q;
}
XPMATH_INLINE_FUNCTION void tf_expansion_compress(const float* e, int m,
                                                  float* out, int n) {
    float g[21], h[21];
    int   bottom = m - 1;
    float q = e[m - 1];
    for (int i = m - 2; i >= 0; --i) {
        float r;
        q = tf_quick_two_sum(q, e[i], r);
        if (r != 0.0f) { g[bottom--] = q; q = r; }
    }
    g[bottom] = q;
    int top = 0;
    for (int i = bottom + 1; i < m; ++i) {
        float r;
        q = tf_quick_two_sum(g[i], q, r);
        if (r != 0.0f) h[top++] = r;
    }
    h[top++] = q;
    for (int k = 0; k < n; ++k) out[k] = (top - 1 - k >= 0) ? h[top - 1 - k] : 0.0f;
}

namespace detail {
// KI-44, TF. See dd_math.hpp's dd_mul_ext / dd_log_ext / dd_exp_ext.
XPMATH_INLINE_FUNCTION TripleFloat tf_mul_ext(TripleFloat x, TripleFloat y, float& err) {
    const float aw[3] = {x.f0, x.f1, x.f2};
    const float bw[3] = {y.f0, y.f1, y.f2};
    float e[20];
    int   m = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (aw[i] == 0.0f || bw[j] == 0.0f) continue;
            float q;
            const float p = tf_two_prod(aw[i], bw[j], q);
            tf_expansion_push(e, m, p);
            tf_expansion_push(e, m, q);
        }
    if (m == 0) { err = 0.0f; return TripleFloat(0.0f); }
    float d[4];
    tf_expansion_compress(e, m, d, 4);
    err = d[3];
    return TripleFloat(d[0], d[1], d[2]);
}
XPMATH_INLINE_FUNCTION int tf_expo_of(float x) {
    int k = 0;
    double t = (double)x;
    while (t >= 18014398509481984.0)    { t *= 5.5511151231257827e-17; k += 54; }
    while (t <  5.5511151231257827e-17) { t *= 18014398509481984.0;    k -= 54; }
    while (t >= 2.0) { t *= 0.5; ++k; }
    while (t <  1.0) { t *= 2.0;  --k; }
    return k;
}
XPMATH_INLINE_FUNCTION TripleFloat tf_log_ext(TripleFloat a, float& err) {
    err = 0.0f;
    if (detail::isinf(a.f0)) return a;
    if (a.f0 <= 0.0f) { XPMATH_PRINTF("TFLOGEXT: non-positive argument\n"); return TripleFloat(0.0f); }
    const int k = tf_expo_of(a.f0);
    const TripleFloat m(ldexpf(a.f0, -k), ldexpf(a.f1, -k), ldexpf(a.f2, -k));
    const TripleFloat lm = log(m);
    const float kLn2_1 =  0x1.62e4p-1f;    // KI-42 FP32 pieces, 16 bits each
    const float kLn2_2 =  0x1.7f7ep-20f;
    const float kLn2_3 = -0x1.c61p-37f;
    const float kLn2_4 = -0x1.950ep-54f;
    const float kLn2_5 =  0x1.e3b4p-72f;
    const float kd = (float)k;
    float e[20];
    int   n = 0;
    tf_expansion_push(e, n, kd * kLn2_1);
    tf_expansion_push(e, n, kd * kLn2_2);
    tf_expansion_push(e, n, kd * kLn2_3);
    tf_expansion_push(e, n, kd * kLn2_4);
    tf_expansion_push(e, n, kd * kLn2_5);
    tf_expansion_push(e, n, lm.f0);
    tf_expansion_push(e, n, lm.f1);
    tf_expansion_push(e, n, lm.f2);
    if (n == 0) return TripleFloat(0.0f);
    float d[4];
    tf_expansion_compress(e, n, d, 4);
    err = d[3];
    return TripleFloat(d[0], d[1], d[2]);
}
XPMATH_INLINE_FUNCTION TripleFloat tf_exp_ext(TripleFloat a, float resid) {
    const float k_inv_log2 = 1.44269504088896341f;
    if (a.f0 < -104.0f) return TripleFloat(0.0f);
    if (a.f0 >  88.722839f) { XPMATH_PRINTF("TFEXP: overflow\n"); return TripleFloat(HUGE_VALF); }
    const float m = detail::floor(a.f0 * k_inv_log2 + 0.5f);
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
    if (resid != 0.0f) r = add(r, TripleFloat(resid));
    const int nq = 5;
    r = divide_scalar(r, float(1 << nq));
    TripleFloat s = r;
    TripleFloat t = sqr(r);
    TripleFloat term = t;
    int kk = 2;
    while (kk < 64 && abs(term).f0 > 1.0e-21f * abs(s).f0) {
        term = divide_scalar(term, float(kk));
        s = add(s, term);
        term = multiply(term, r);
        kk++;
    }
    for (int i = 0; i < nq; i++) s = multiply(s, add(s, TripleFloat(2.0f)));
    s = add(TripleFloat(1.0f), s);
    const int mi = (int)m;
    if (mi >= -125 && mi <= 127) return mul_pwr2(s, ldexpf(1.0f, mi));
    return TripleFloat(ldexpf(s.f0, mi), ldexpf(s.f1, mi), ldexpf(s.f2, mi));
}
}  // namespace detail
XPMATH_INLINE_FUNCTION TripleFloat fma(TripleFloat a, TripleFloat b, TripleFloat c) {
    const float p0 = a.f0 * b.f0;
    if (!detail::isfinite(p0) || !detail::isfinite(c.f0))
        return add(multiply(a, b), c);

    const float aw[3] = {a.f0, a.f1, a.f2};
    const float bw[3] = {b.f0, b.f1, b.f2};
    float e[21];                        // 9 two_prods (18 words) + c's 3 words
    int   m = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (aw[i] == 0.0f || bw[j] == 0.0f) continue;
            float err;
            const float hi = tf_two_prod(aw[i], bw[j], err);
            tf_expansion_push(e, m, hi);
            tf_expansion_push(e, m, err);
        }
    tf_expansion_push(e, m, c.f0);
    tf_expansion_push(e, m, c.f1);
    tf_expansion_push(e, m, c.f2);
    if (m == 0) return add(multiply(a, b), c);

    float d[4];
    tf_expansion_compress(e, m, d, 4);
    renorm_3(d[0], d[1], d[2], d[3]);
    return TripleFloat(d[0], d[1], d[2]);
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
