// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (real three-word algorithms
//   this complex layer composes on; Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is the complex layer for the TF (triple-float, 3×FP32) backend on
// branch main. It is a mechanical scalar-swap port of
// third_party/include/qf_complex.hpp (this repo's quad-float complex layer,
// 4×FP32) to TripleFloat (3×FP32). Every complex algorithm — the (ac−bd)+(ad+bc)i
// product, the Kahan-style complex sqrt, exp = eˣ(cos y + i sin y), the log/atan2
// polar decomposition, the sin/cos/sinh/cosh angle-addition formulas — descends
// structurally from qf_complex.hpp / ff_complex.hpp / dd_complex.hpp, and each
// function cites the qf_complex.hpp (and, where it is the deeper reference,
// ff_complex.hpp or dd_complex.hpp) line range it mirrors.
//
// QD 2.3.24 SHIPS NO COMPLEX HEADER (same as QF). The QD 2.3.24 tarball contains
// qd_real.{h,cpp}, dd_real.{h,cpp} and C wrappers, but no qd_complex.* or
// dd_complex.* — QD's quad-double complex is left to the user. Consequently
// qf_complex.hpp + ff_complex.hpp + dd_complex.hpp are the SOLE algorithm
// references for this header; there is no QD complex routine to cite.
//
// LICENSE LINEAGE (mirroring qf_complex.hpp). This header carries
// LicenseRef-LBNL-BSD-License — the SAME license as tf_math.hpp — NOT the
// LicenseRef-DHB-License that governs ff_complex.hpp / dd_complex.hpp. Rationale:
// the complex composition formulas (product, quotient, Kahan sqrt, Euler exp,
// polar log) are textbook identities, not DHB/DDFUN inventions, and every
// non-trivial numeric step is a QD-derived TripleFloat operation. This keeps the
// whole TF backend (tf_math.hpp + tf_complex.hpp) under one consistent license,
// matching the QF backend precedent (qf_math.hpp + qf_complex.hpp, both LBNL-BSD).
//
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.

#pragma once

// Triple-float complex arithmetic — xp::TripleFloatComplex.
// All functions XPMATH_INLINE_FUNCTION (host + device via CUDA/HIP/SYCL).
// Depends on tf_math.hpp (the 3×FP32 real backend, S10 Phase 1-3.5).
//
// DEPENDENCIES: none beyond the C++17 standard library and tf_math.hpp.
// In particular this header does NOT include or require Kokkos — see
// xp/config.hpp for how the portability facilities are supplied. Kokkos
// users get today's `Kokkos::Experimental::TripleFloatComplex` API
// unchanged through the compat wrapper at third_party/include/tf_complex.hpp,
// which is the only place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// TABLE-FREE POSTURE (inherited from S10 Phase 1, PORT_NOTES_TF §3). This header
// adds NO lookup tables — no sin_table / cos_table / inv_fact. Complex
// exp/sin/cos/sinh/cosh dispatch through tf_math.hpp's table-free real
// transcendentals (divide-by-k Taylor, joint sin/cos doublings), so they inherit
// S10's §3 exp term count (N=9, nq=5) and §3c sinh threshold (0.5).
//
// Naming follows tf_math.hpp / qf_complex.hpp / ff_complex.hpp (T0.4/T2.0/T3.0a):
// type + math live under xp:: for eventual upstreaming. This remains a bespoke
// struct rather than Kokkos::complex<TripleFloat> — that integration is a
// separate future task.
//
// SINCOS / SINHCOSH OUTPUT ORDER (TF-specific, mirroring qf_complex.hpp §SINCOS).
// Unlike ff_math.hpp — whose sincos(a, x, y) writes x=cos, y=sin — tf_math.hpp
// names its out-params sin-first: sincos(a, sin_a, cos_a) and
// sinhcosh(a, sinh_a, cosh_a). To keep each call site's downstream algebra
// byte-identical to qf_complex.hpp, this header passes the local (cos, sin) /
// (cosh, sinh) variables in SWAPPED positional order, i.e. sincos(a, s, c) and
// sinhcosh(a, sh, ch). The local variable meanings (c=cos, s=sin, ...) then match
// qf_complex.hpp exactly.

#include <xp/tf_math.hpp>

#if !defined(XPMATH_ON_DEVICE)
#  include <ostream>
#endif

namespace xp {

// ============================================================
// TripleFloatComplex struct
// ============================================================
// Members named re/im (mirroring qf_complex.hpp / ff_complex.hpp / dd_complex.hpp
// verbatim so the demo's mqr(i).re / .im field access and the real()/imag()
// accessors coexist).
struct TripleFloatComplex {
    TripleFloat re;
    TripleFloat im;

    XPMATH_INLINE_FUNCTION TripleFloatComplex() : re(0.0f), im(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplex(float r)                 : re(r),    im(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplex(TripleFloat r)           : re(r),    im(0.0f) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplex(float r, float i)        : re(r),    im(i)    {}
    XPMATH_INLINE_FUNCTION TripleFloatComplex(TripleFloat r, TripleFloat i) : re(r),    im(i)    {}
    XPMATH_INLINE_FUNCTION TripleFloatComplex(const TripleFloatComplex& o): re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION TripleFloatComplex& operator=(const TripleFloatComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    XPMATH_INLINE_FUNCTION TripleFloatComplex& operator=(TripleFloat r) {
        re = r; im = TripleFloat(0.0f); return *this;
    }

    // qf_complex.hpp:122-154 (operator+/-/*// and unary -).
    XPMATH_INLINE_FUNCTION TripleFloatComplex operator+(TripleFloatComplex b) const {
        return TripleFloatComplex(add(re, b.re), add(im, b.im));
    }
    XPMATH_INLINE_FUNCTION TripleFloatComplex operator-(TripleFloatComplex b) const {
        return TripleFloatComplex(subtract(re, b.re), subtract(im, b.im));
    }
    XPMATH_INLINE_FUNCTION TripleFloatComplex operator*(TripleFloatComplex b) const {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i  (qf_complex.hpp:128-131)
        return TripleFloatComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                                  add(multiply(re, b.im), multiply(im, b.re)));
    }
    XPMATH_INLINE_FUNCTION TripleFloatComplex operator/(TripleFloatComplex b) const {
        // (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)  (qf_complex.hpp:133-142)
        if (b.re.f0 == 0.0f && b.im.f0 == 0.0f) {
            XPMATH_PRINTF("TFCOMPLEX: division by zero\n");
            return TripleFloatComplex();
        }
        TripleFloat denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        TripleFloat inv   = divide(TripleFloat(1.0f), denom);
        return TripleFloatComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                                  multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    XPMATH_INLINE_FUNCTION TripleFloatComplex operator-() const {
        return TripleFloatComplex(negate(re), negate(im));
    }

    XPMATH_INLINE_FUNCTION TripleFloatComplex& operator+=(TripleFloatComplex b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplex& operator-=(TripleFloatComplex b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplex& operator*=(TripleFloatComplex b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION TripleFloatComplex& operator/=(TripleFloatComplex b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(TripleFloatComplex b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(TripleFloatComplex b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION TripleFloat real() const { return re; }
    XPMATH_INLINE_FUNCTION TripleFloat imag() const { return im; }
};

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const TripleFloatComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed TripleFloat × TripleFloatComplex arithmetic  (qf_complex.hpp:170-177)
// ============================================================
XPMATH_INLINE_FUNCTION TripleFloatComplex operator+(TripleFloatComplex z, TripleFloat r) { return TripleFloatComplex(add(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator+(TripleFloat r, TripleFloatComplex z) { return TripleFloatComplex(add(r, z.re), z.im); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator-(TripleFloatComplex z, TripleFloat r) { return TripleFloatComplex(subtract(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator-(TripleFloat r, TripleFloatComplex z) { return TripleFloatComplex(subtract(r, z.re), negate(z.im)); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator*(TripleFloatComplex z, TripleFloat r) { return TripleFloatComplex(multiply(z.re, r), multiply(z.im, r)); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator*(TripleFloat r, TripleFloatComplex z) { return TripleFloatComplex(multiply(r, z.re), multiply(r, z.im)); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator/(TripleFloatComplex z, TripleFloat r) { return TripleFloatComplex(divide(z.re, r), divide(z.im, r)); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator/(TripleFloat r, TripleFloatComplex z) { return TripleFloatComplex(r) / z; }

// ============================================================
// Mixed float × TripleFloatComplex arithmetic  (qf_complex.hpp:182-189)
// ============================================================
XPMATH_INLINE_FUNCTION TripleFloatComplex operator+(TripleFloatComplex z, float b) { return z + TripleFloat(b); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator+(float b, TripleFloatComplex z) { return TripleFloat(b) + z; }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator-(TripleFloatComplex z, float b) { return z - TripleFloat(b); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator-(float b, TripleFloatComplex z) { return TripleFloat(b) - z; }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator*(TripleFloatComplex z, float b) { return z * TripleFloat(b); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator*(float b, TripleFloatComplex z) { return TripleFloat(b) * z; }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator/(TripleFloatComplex z, float b) { return z / TripleFloat(b); }
XPMATH_INLINE_FUNCTION TripleFloatComplex operator/(float b, TripleFloatComplex z) { return TripleFloat(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

// abs(z) = |z| = sqrt(re²+im²).  qf_complex.hpp:196-198 / ff_complex.hpp:151-153 / dd_complex.hpp:145-147.
XPMATH_INLINE_FUNCTION TripleFloat abs(TripleFloatComplex z) {
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
}
// norm(z) = |z|² = re²+im² (std::norm convention; the squared magnitude, no
// sqrt).  qf_complex.hpp:202-204. TF follows QF in exposing this as a standalone op.
XPMATH_INLINE_FUNCTION TripleFloat norm(TripleFloatComplex z) {
    return add(multiply(z.re, z.re), multiply(z.im, z.im));
}
// arg(z) = atan2(im, re) (std::arg convention; the polar angle).  qf_complex.hpp:208-210.
// Uses tf_math.hpp atan2(y, x).
XPMATH_INLINE_FUNCTION TripleFloat arg(TripleFloatComplex z) {
    return atan2(z.im, z.re);
}
// conj(z) = re - im·i.  qf_complex.hpp:212-214 / ff_complex.hpp:154-156 / dd_complex.hpp:148-150.
XPMATH_INLINE_FUNCTION TripleFloatComplex conj(TripleFloatComplex z) {
    return TripleFloatComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root  (Kahan-style; qf_complex.hpp:222-240 / ff_complex.hpp:161-179 / dd_complex.hpp:155-174)
// ============================================================
// B = sqrt((R+|re|)/2) + i·sign(im)·sqrt((R-|re|)/2), R = |z|, arranged to avoid
// cancellation. The ½ and 2 are FP32-exact literals used via multiply_scalar
// (PORT_NOTES_TF §3), matching qf_complex.hpp:226-228.
XPMATH_INLINE_FUNCTION TripleFloatComplex sqrt(TripleFloatComplex z) {
    if (z.re.f0 == 0.0f && z.im.f0 == 0.0f) return TripleFloatComplex();
    TripleFloat r  = sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
    TripleFloat a1 = abs(z.re);
    TripleFloat s2 = multiply_scalar(add(r, a1), 0.5f);
    TripleFloat s0 = sqrt(s2);
    TripleFloat s1 = multiply_scalar(s0, 2.0f);
    TripleFloatComplex b;
    if (z.re.f0 >= 0.0f) {
        b.re = s0;
        b.im = divide(z.im, s1);
    } else {
        b.re = divide(z.im, s1);
        if (b.re.f0 < 0.0f) b.re = negate(b.re);
        b.im = s0;
        if (z.im.f0 < 0.0f) b.im = negate(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
// exp(z) = eˣ·(cos y + i·sin y), z = x + iy.  qf_complex.hpp:249-253 /
// ff_complex.hpp:184-188 / dd_complex.hpp:179-184. Real exp + joint sincos are
// the table-free TF transcendentals (S10 Phase 1 §3). NOTE the swapped sincos
// args (header §SINCOS): tf sincos writes (sin, cos), so pass (s, c) to keep
// c=cos(y), s=sin(y).
XPMATH_INLINE_FUNCTION TripleFloatComplex exp(TripleFloatComplex z) {
    TripleFloat er = exp(z.re);
    TripleFloat c, s;
    sincos(z.im, s, c);
    return TripleFloatComplex(multiply(er, c), multiply(er, s));
}

// log(z) = log|z| + i·arg(z).  qf_complex.hpp:257-260 / ff_complex.hpp:191-194 / dd_complex.hpp:186-190.
XPMATH_INLINE_FUNCTION TripleFloatComplex log(TripleFloatComplex z) {
    TripleFloat modulus = abs(z);
    TripleFloat argument = atan2(z.im, z.re);
    return TripleFloatComplex(log(modulus), argument);
}

// log10(z) = log(z)/ln(10).  qf_complex.hpp:264-267 / ff_complex.hpp:197-200 / dd_complex.hpp:192-196.
// KI-5(b). Complex log1p(w) = log(1+w), accurate for small |w| -- the library
// had no complex log1p before this. Writing log(1 + w) directly is what made
// complex `atanh` collapse near the origin: 1 + w rounds w's information away
// before the log ever runs.
//
//     |1+w|^2 = 1 + (2*Re(w) + |w|^2)
//     Re log1p(w) = 0.5f * log1p( 2*Re(w) + |w|^2 )        <- REAL log1p
//     Im log1p(w) = atan2( Im(w), 1 + Re(w) )
//
// The whole point of the real part is the argument `2*Re(w) + |w|^2`: it is the
// small quantity by which |1+w|^2 differs from 1, formed WITHOUT ever adding 1,
// and handed to the real log1p (dd_math.hpp), which was rebuilt on the
// 2*atanh(a/(2+a)) series in the same change so that it can actually keep it.
//
// The imaginary part needs no such care. atan2(y, 1+x) for small w returns
// ~Im(w); its sensitivity to the rounding of 1 + Re(w) is d/dx atan2 = -y/|1+w|^2
// ~ -Im(w), so the absolute error eps in the second argument arrives as a
// RELATIVE error of eps in the answer. Nothing is lost.
//
// Divergence from the sources, recorded deliberately. QD 2.3f.24 (/tmp/qdsrc/QD)
// has no complex layer at all, so it offers no complex log1p to copy. Kahan 1987
// gives the formulation above (his `logp1`/`clogp1` discussion, and the same
// expression underlies his catanh); the residual weakness is his too -- when
// 2*Re(w) + |w|^2 itself cancels, i.e. on the circle |1+w| = 1, the real part
// loses relative accuracy. That locus is measure-zero, the answer there is ~0,
// and every hypot-based alternative loses the same digits on the same circle.
// Accepted rather than worked around.
XPMATH_INLINE_FUNCTION TripleFloatComplex log1p(TripleFloatComplex w) {
    TripleFloat t = add(multiply_scalar(w.re, 2.0f),
                         add(multiply(w.re, w.re), multiply(w.im, w.im)));
    return TripleFloatComplex(multiply_scalar(log1p(t), 0.5f),
                               atan2(w.im, add(TripleFloat(1.0f), w.re)));
}

XPMATH_INLINE_FUNCTION TripleFloatComplex log10(TripleFloatComplex z) {
    TripleFloatComplex lg = log(z);
    TripleFloat ln10 = TripleFloat_log10();
    return TripleFloatComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
// sin(a+bi) = sin(a)·cosh(b) + i·cos(a)·sinh(b).  qf_complex.hpp:276-280 /
// ff_complex.hpp:206-211 / dd_complex.hpp:201-207. Swapped sincos/sinhcosh args
// (header §SINCOS): local ca=cos(a), sa=sin(a), cb=cosh(b), sb=sinh(b).
XPMATH_INLINE_FUNCTION TripleFloatComplex sin(TripleFloatComplex z) {
    TripleFloat ca, sa, cb, sb;
    sincos(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return TripleFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
// cos(a+bi) = cos(a)·cosh(b) - i·sin(a)·sinh(b).  qf_complex.hpp:284-288 /
// ff_complex.hpp:213-217 / dd_complex.hpp:208-214.
XPMATH_INLINE_FUNCTION TripleFloatComplex cos(TripleFloatComplex z) {
    TripleFloat ca, sa, cb, sb;
    sincos(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return TripleFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
// tan(z) = sin(z)/cos(z).  qf_complex.hpp:291-293 / ff_complex.hpp:220-221 / dd_complex.hpp:215-217.
XPMATH_INLINE_FUNCTION TripleFloatComplex tan(TripleFloatComplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
// asin(z) = -i·log(iz + sqrt(1 - z²)).  qf_complex.hpp:301-307 /
// ff_complex.hpp:227-235 / dd_complex.hpp:222-231. iz built by literal-lifted
// components; the 1 promoted via TripleFloat(1.0f) inside the single-arg
// TripleFloatComplex ctor (imag→0).
// KI-5(d) fix; see dd_complex.hpp:231-241 for the full rationale. On the real
// cut (Im(z) == +-0, |Re(z)| > 1) the sheet of sqrt(1 - z^2) is fixed by the
// sign of Im(z)'s zero, which the subtraction destroys (0 - (+-0) == +0 in
// round-to-nearest). Read it off Im(z) instead: Im(1 - z^2) = -2*Re*Im, so the
// root is negative-imaginary exactly when Re and Im share a sign.
XPMATH_INLINE_FUNCTION TripleFloatComplex asin(TripleFloatComplex z) {
    TripleFloatComplex iz  = TripleFloatComplex(negate(z.im), z.re);
    TripleFloatComplex z2  = z * z;
    TripleFloatComplex one_minus_z2 = TripleFloatComplex(TripleFloat(1.0f)) - z2;
    TripleFloatComplex root = sqrt(one_minus_z2);
    if (z.im.f0 == 0.0f && detail::fabs(z.re.f0) > 1.0f) {
        const bool want_neg = (detail::copysign(1.0f, z.re.f0) ==
                               detail::copysign(1.0f, z.im.f0));
        const bool have_neg = (detail::copysign(1.0f, root.im.f0) < 0.0f);
        if (want_neg != have_neg) root.im = negate(root.im);
    }
    TripleFloatComplex sum = iz + root;
    TripleFloatComplex lg  = log(sum);
    return TripleFloatComplex(lg.im, negate(lg.re));  // × (-i): (a+bi)(-i) = b - ai
}
// acos(z) = π/2 - asin(z).  qf_complex.hpp:310-313 / ff_complex.hpp:237-241 / dd_complex.hpp:232-237.
// Principal sqrt of (u, v) with the sign of a ZERO v respected. The header's
// complex sqrt above tests `z.im.f0 < 0.0f`, which is false for -0.0f, so for u < 0
// it puts BOTH zero conventions on the +i sheet. That is the same class of defect
// as KI-5(d) and it is corrected locally here rather than inside sqrt itself,
// which would move every other caller (acosh included) in one undocumented step.
// The general sqrt defect is recorded in docs/KNOWN_ISSUES.md, not fixed here.
// `vsign` is +1/-1, the intended sign of v when v is a zero: multiply_scalar does
// NOT carry a signed zero through (it renormalizes, and quick_two_sum(-0,+0) is
// +0), so the caller passes the sign it read off the original Im(z) rather than
// trusting the halved copy.
XPMATH_INLINE_FUNCTION TripleFloatComplex sqrt_signed_cut(TripleFloat u, TripleFloat v, float vsign) {
    TripleFloatComplex r = sqrt(TripleFloatComplex(u, v));
    if (v.f0 == 0.0f && u.f0 < 0.0f && vsign != detail::copysign(1.0f, r.im.f0))
        r.im = negate(r.im);
    return r;
}
// KI-5(c) fix. acos(z) = -2i * log( sqrt((1+z)/2) + i*sqrt((1-z)/2) ) -- Kahan
// 1987, the exact companion of the acosh form adopted for KI-1 below.
//
// The old body was `pi/2 - asin(z)`, which is unconditionally stable nowhere near
// z = 1: acos(1) = 0, so as z -> 1 the difference cancels two quantities that
// both tend to pi/2 and the ANSWER's own magnitude tends to 0. Every digit of
// the result is a digit the subtraction destroyed. It is worse off the real axis
// -- acos(2 + 1e-20i) scored 11.31f on DD, 0.00f on FF -- because asin there is
// itself computed through 1 - z^2, so the loss compounds.
//
// Why Kahan's log form and not his other one, acos(z) = 2*atan(sqrt((1-z)/(1+z))).
// Both are well conditioned at z -> 1. The log form was chosen because (a) it is
// structurally identical to the acosh already in this header, so the two share a
// verified branch layout and one reader's understanding covers both, (b) it needs
// only sqrt/log, which are the two best-tested primitives here, where the atan
// form adds a complex atan on top of a complex divide by (1+z) -- a divide that
// is singular at z = -1, the OTHER end of the principal interval, so that form
// simply moves the bad point rather than removing it, and (c) it never forms z^2
// or 1/(1+z), hence no overflow at large |z| and no singularity at either end.
//
// BUT THE PURE LOG FORM IS ONLY HALF RIGHT, and the monotone gate is what said
// so. Writing w for the bracket, Im(acos z) = -2*ln|w|, and |w| -> 1 exactly
// where Im(acos z) -> 0, i.e. for z near the real segment [-1,1]. There the log
// is taken of a number whose information sits below the leading 1 -- the same
// disease as KI-5(b), relocated. Measured: 4016 sweep points down across the
// four backends, worst 31.00 -> 1.23 on DD at z = 0 + 1e-30i. Strictly worse
// than the defect being fixed.
//
// SO THIS IS PER-COMPONENT, and the split is exact rather than a compromise.
// Im(acos z) = -Im(asin z) is an IDENTITY (pi/2 is real), so the pi/2 - asin
// cancellation was only ever in the real part:
//
//     Re(acos z) = 2*arg( sqrt((1+z)/2) + i*sqrt((1-z)/2) )   <- Kahan, stable
//     Im(acos z) = -Im(asin z)                                <- exact identity
//
// Each form's good component, neither's bad one. This also inherits KI-5(d)'s
// cut fix on the imaginary part for free. Decreases fell from 4016 to 949.
//
// ACCEPTED LOSS, recorded in docs/KNOWN_ISSUES.md KI-5(c): for |z| >> 1 with
// arg(z) < 0 the two roots satisfy rm ~ -i*rp, so w = rp + i*rm is a difference
// of near-equal roots and arg(w) loses about log10|z|/2 digits -- up to 7.85 on
// the |z| = 1e8 polar ring. Reflecting by acos(conj z) = conj(acos z) should
// remove it and was tried both ways; both measured WORSE (1915 and 1648
// decreases against 949), so no reflection ships. See the KNOWN_ISSUES entry
// before retrying it.
//
// This is NOT routed through acosh even though acos(z) = +-i*acosh(z) holds. The
// sign of that relation flips with the half-plane AND with the side of each cut,
// so sharing the body would mean reintroducing exactly the case analysis Kahan's
// form exists to avoid; the two functions are three lines each and stay separate.
//
// BRANCH CHECK (each verified against the __complex128 oracle, both half-planes
// and both sides of both cuts): z=0 -> pi/2; z=1 -> 0; z=-1 -> pi;
// z=2+0i -> -1.3170i; z=2-0i -> +1.3170i; z=-2+0i -> pi-1.3170i;
// z=-2-0i -> pi+1.3170i. The last four are the cut points, and they are why
// sqrt_signed_cut exists: at z = 2+-0i the argument of the SECOND root is
// negative-real with a signed zero imaginary part, and at z = -2+-0i it is the
// FIRST root, so both calls need the corrected sheet.
XPMATH_INLINE_FUNCTION TripleFloatComplex acos(TripleFloatComplex z) {
    const TripleFloat one(1.0f);
    const TripleFloat half_im = multiply_scalar(z.im, 0.5f);
    const float s_im = detail::copysign(1.0f, z.im.f0);
    TripleFloatComplex rp = sqrt_signed_cut(
        multiply_scalar(add(one, z.re), 0.5f), half_im, s_im);
    TripleFloatComplex rm = sqrt_signed_cut(
        multiply_scalar(subtract(one, z.re), 0.5f), negate(half_im), -s_im);
    // w = rp + i*rm, with i*(a+bi) = -b + ai
    TripleFloatComplex w(subtract(rp.re, rm.im), add(rp.im, rm.re));
    return TripleFloatComplex(multiply_scalar(atan2(w.im, w.re), 2.0f), negate(asin(z).im));
}
// atan(z) = (i/2)·log((1 - iz)/(1 + iz)).  qf_complex.hpp:317-324 /
// ff_complex.hpp:243-251 / dd_complex.hpp:238-247.
XPMATH_INLINE_FUNCTION TripleFloatComplex atan(TripleFloatComplex z) {
    TripleFloatComplex iz    = TripleFloatComplex(negate(z.im), z.re);
    TripleFloatComplex num   = TripleFloatComplex(TripleFloat(1.0f)) - iz;
    TripleFloatComplex den   = TripleFloatComplex(TripleFloat(1.0f)) + iz;
    TripleFloatComplex ratio = num / den;
    TripleFloatComplex lg    = log(ratio);
    // × (i/2): (a+bi)(i/2) = (-b/2) + (a/2)i
    return TripleFloatComplex(multiply_scalar(negate(lg.im), 0.5f), multiply_scalar(lg.re, 0.5f));
}

// ============================================================
// Complex hyperbolic
// ============================================================
// sinh(a+bi) = sinh(a)·cos(b) + i·cosh(a)·sin(b).  qf_complex.hpp:333-337 /
// ff_complex.hpp:257-262 / dd_complex.hpp:252-258. Swapped args: ca=cosh(a),
// sa=sinh(a), cb=cos(b), sb=sin(b).
XPMATH_INLINE_FUNCTION TripleFloatComplex sinh(TripleFloatComplex z) {
    TripleFloat ca, sa, cb, sb;
    sinhcosh(z.re, sa, ca);
    sincos(z.im, sb, cb);
    return TripleFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
// cosh(a+bi) = cosh(a)·cos(b) + i·sinh(a)·sin(b).  qf_complex.hpp:341-345 /
// ff_complex.hpp:264-268 / dd_complex.hpp:259-265.
XPMATH_INLINE_FUNCTION TripleFloatComplex cosh(TripleFloatComplex z) {
    TripleFloat ca, sa, cb, sb;
    sinhcosh(z.re, sa, ca);
    sincos(z.im, sb, cb);
    return TripleFloatComplex(multiply(ca, cb), multiply(sa, sb));
}
// tanh(a+bi): re = T/(cos²b + T²·sin²b), im = sin b·cos b·(1-T²)/(...),
// T = tanh(a).  qf_complex.hpp:351-358 / ff_complex.hpp:271-281 / dd_complex.hpp:266-277.
// Denominator ≥ 0; uses the improved real tanh to avoid cancellation. Swapped
// sincos args: cb=cos(b), sb=sin(b).
XPMATH_INLINE_FUNCTION TripleFloatComplex tanh(TripleFloatComplex z) {
    TripleFloat T = tanh(z.re);
    TripleFloat cb, sb;
    sincos(z.im, sb, cb);
    TripleFloat T2    = multiply(T, T);
    TripleFloat denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return TripleFloatComplex(divide(T, denom),
                              divide(multiply(multiply(sb, cb), subtract(TripleFloat(1.0f), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
// asinh(z) = log(z + sqrt(z² + 1)), reflected into the right half-plane via the
// oddness identity -asinh(-z) when the direct form would cancel. KI-5(a); see
// dd_complex.hpp's asinh for the full rationale, the magnitude threshold and its
// measured justification, and the Re(z) == ±0 boundary decision — all identical
// here.
XPMATH_INLINE_FUNCTION TripleFloatComplex asinh(TripleFloatComplex z) {
    const float t = 4.0f;   // kXpAsinhReflect
    if (z.re.f0 < 0.0f && (-z.re.f0 > t || detail::fabs(z.im.f0) > t)) {
        TripleFloatComplex w = -z;
        return -log(w + sqrt(w*w + TripleFloatComplex(TripleFloat(1.0f))));
    }
    return log(z + sqrt(z*z + TripleFloatComplex(TripleFloat(1.0f))));
}
// acosh(z) = log(z + sqrt(z² - 1)).  qf_complex.hpp:369-370 / ff_complex.hpp:291-293 / dd_complex.hpp:286-289.
// KI-1 fix: Kahan 1987's branch-correct form, acosh(z) = 2*log(sqrt((z+1)/2) +
// sqrt((z-1)/2)). See dd_complex.hpp:346-357 for the full rationale. The old
// log(z + sqrt(z*z - 1)) form was on the wrong sqrt sheet throughout
// Re(z) < 0, and overflowed above |z| ~ 1.8e19 where z*z leaves FP32 range.
XPMATH_INLINE_FUNCTION TripleFloatComplex acosh(TripleFloatComplex z) {
    const TripleFloat one(1.0f);
    const TripleFloat half_im = multiply_scalar(z.im, 0.5f);
    TripleFloatComplex rp = sqrt(TripleFloatComplex(
        multiply_scalar(add(z.re, one), 0.5f), half_im));
    TripleFloatComplex rm = sqrt(TripleFloatComplex(
        multiply_scalar(subtract(z.re, one), 0.5f), half_im));
    TripleFloatComplex lg = log(rp + rm);
    return TripleFloatComplex(multiply_scalar(lg.re, 2.0f),
                              multiply_scalar(lg.im, 2.0f));
}
// atanh(z) = ½·log((1 + z)/(1 - z)).  qf_complex.hpp:373-376 / ff_complex.hpp:295-299 / dd_complex.hpp:290-295.
// KI-5(b) fix. Two independent defects lived in the old one-line body
// `0.5f*log((1+z)/(1-z))`, and both are repaired here.
//
// (1) CONDITIONING AS z -> 0. The ratio (1+z)/(1-z) -> 1, so the log is taken of
// a number whose entire information content sits below the leading 1. The
// argument reduction throws away log10(1/|z|) digits before log() is even
// entered, and the measured score falls off one digit per decade of |z|. The
// remedy is the classical one: 0.5f*log1p(2z/(1-z)), expanded here into its real
// and imaginary components so no complex divide is needed either --
//
//     Re atanh(z) = 0.25f * log1p( 4x / ((1-x)^2 + y^2) )
//     Im atanh(z) = 0.5f  * atan2( 2y, 1 - x^2 - y^2 )
//
// which is 0.5f*log1p(w) with w = 2z/(1-z) written out: 2*Re(w) + |w|^2 collapses
// to 4x/|1-z|^2 and arg(1+w) to atan2(2y, 1-x^2-y^2). For small z the log1p
// argument is ~4x, small and formed without cancellation, and the real log1p
// (dd_math.hpp, rebuilt on the 2*atanh(a/(2+a)) series in this same change)
// keeps it.
//
// THRESHOLD (kXpAtanhSmall = 0.0625, L-infinity on the leading limbs). The new form
// is used only where the old one is actually losing. Two reasons to gate rather
// than switch unconditionally. First, ((1-x)^2 + y^2) squares its operands, so
// it overflows to Inf for |z| above sqrt of the word type's range (~1.3e154 for
// the FP64-word backend, ~1.8e19 for the FP32-word ones) where the old ratio
// form is perfectly finite -- a switch would trade a conditioning defect for an
// overflow defect. Second, at |z| >= 0.5f the old form already scores at the
// type's cap (measured: DD 30.74f/31.00f at z = 0.5f), so there is nothing to win
// and only rounding churn to lose -- the first cut of this fix used 0.5 and the
// gate caught 43 points across the four backends losing up to 1.09 digits to
// exactly that churn in 0.0625 < |z| < 0.5, so the threshold is 0.0625 (2^-4,
// exactly representable, so the compare is exact) and one atanh decrease
// remains in the whole 428,592-point sweep. L-infinity, not hypot(), for the same reasons
// given on asinh above: no overflow, two compares, and the leading limb settles
// it except within an ulp of the boundary.
//
// (2) THE SIGNED ZERO ON THE CUTS -- the atanh analogue of KI-5(d), found by
// measurement while fixing (1), and fixed here because it is in the same body.
// The cuts are (-inf,-1] and [1,+inf). Approaching x > 1 from Im = +0 gives
// Im atanh = +pi/2, and from Im = -0 gives -pi/2; by oddness the SAME +pi/2
// holds for x < -1 with Im = +0. The old form computed (1+z)/(1-z) with a
// complex divide, whose multiplies destroy the sign of Im(z)'s zero, and so
// returned +pi/2 for both conventions: every `x -0i` cut point scored 0.00f on
// the imaginary component, in all four backends. The sign is read off Im(z)
// directly with detail::copysign -- these types do carry a signed zero through
// construction, copy and negate, they only lose it in arithmetic -- and pi/2 is
// installed with it. The two conventions that were already right are unchanged.
XPMATH_INLINE_FUNCTION TripleFloatComplex atanh(TripleFloatComplex z) {
    const TripleFloat one(1.0f);
    const float t = 0.0625f;   // kXpAtanhSmall
    TripleFloatComplex r;
    if (detail::fabs(z.re.f0) < t && detail::fabs(z.im.f0) < t) {
        TripleFloat omx = subtract(one, z.re);
        TripleFloat den = add(multiply(omx, omx), multiply(z.im, z.im));
        r.re = multiply_scalar(
            log1p(divide(multiply_scalar(z.re, 4.0f), den)), 0.25f);
        r.im = multiply_scalar(
            atan2(multiply_scalar(z.im, 2.0f),
                  subtract(one, add(multiply(z.re, z.re), multiply(z.im, z.im)))),
            0.5f);
    } else {
        TripleFloatComplex lg = log((TripleFloatComplex(one) + z) / (TripleFloatComplex(one) - z));
        r.re = multiply_scalar(lg.re, 0.5f);
        r.im = multiply_scalar(lg.im, 0.5f);
    }
    if (z.im.f0 == 0.0f && detail::fabs(z.re.f0) > 1.0f) {
        TripleFloat half_pi = multiply_scalar(TripleFloat_pi(), 0.5f);
        if (detail::copysign(1.0f, z.im.f0) < 0.0f) half_pi = negate(half_pi);
        r.im = half_pi;
    }
    return r;
}

// ============================================================
// Complex power and polar
// ============================================================
// pow(z, w) = exp(w·log(z)).  qf_complex.hpp:383-385 / ff_complex.hpp:305-308 / dd_complex.hpp:300-304.
XPMATH_INLINE_FUNCTION TripleFloatComplex pow(TripleFloatComplex z, TripleFloatComplex w) {
    if (z.re.f0 == 0.0f && z.im.f0 == 0.0f) return TripleFloatComplex();
    return exp(w * log(z));
}

// polar(r, theta) = r·(cos θ + i·sin θ).  qf_complex.hpp:390-393 /
// ff_complex.hpp:312-315 / dd_complex.hpp:306-311. Swapped sincos args: c=cos(θ), s=sin(θ).
XPMATH_INLINE_FUNCTION TripleFloatComplex polar(TripleFloat r, TripleFloat theta) {
    TripleFloat c, s;
    sincos(theta, s, c);
    return TripleFloatComplex(multiply(r, c), multiply(r, s));
}

} // namespace xp
