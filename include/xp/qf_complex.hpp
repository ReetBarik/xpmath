// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (real four-word algorithms
//   this complex layer composes on; Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is the complex layer for the QF (quad-float, 4×FP32) backend on
// branch qffunKokkos. It is a mechanical scalar-swap port of
// third_party/include/ff_complex.hpp (this repo's float-float complex layer,
// itself a DD→FF translation of dd_complex.hpp) from FloatFloat (2×FP32) to
// QuadFloat (4×FP32). Every complex algorithm — the (ac−bd)+(ad+bc)i product,
// the Kahan-style complex sqrt, exp = eˣ(cos y + i sin y), the log/atan2 polar
// decomposition, the sin/cos/sinh/cosh angle-addition formulas — descends
// structurally from ff_complex.hpp / dd_complex.hpp, and each function cites the
// ff_complex.hpp (and, where it is the deeper reference, dd_complex.hpp) line
// range it mirrors.
//
// QD 2.3.24 SHIPS NO COMPLEX HEADER. The QD 2.3.24 tarball
// (qd/include/qd/, qd/src/) contains qd_real.{h,cpp}, dd_real.{h,cpp},
// c_dd.{h,cpp}, c_qd.{h,cpp} (a C-linkage wrapper of the *real* types) and
// fpu/bits/inline headers — but no qd_complex.* or dd_complex.* anywhere (only
// NEWS/README/TODO/Fortran mention the word "complex"). QD's quad-double complex
// is left to the user to layer on. Consequently ff_complex.hpp + dd_complex.hpp
// are the SOLE algorithm references for this header; there is no QD complex
// routine to cite. (Verified by grep -ril complex over the 2.3.24 tree during
// T3.0c; recorded in docs/PORT_NOTES_QF.md.)
//
// LICENSE LINEAGE (reported for review, per T3.0c task). This header carries
// LicenseRef-LBNL-BSD-License — the SAME license as qf_math.hpp — NOT the
// LicenseRef-DHB-License that governs ff_complex.hpp / dd_complex.hpp. Two
// lineage precedents were weighed:
//   (a) follow scalar dispatch (LBNL-BSD): every arithmetic call inside this
//       header dispatches to qf_math.hpp's LBNL-BSD QuadFloat routines, so the
//       observable numerics are QD-derived, exactly the reasoning T3.0a used to
//       put qf_math.hpp under LBNL-BSD.
//   (b) follow the structural template (DHB): qf_complex.hpp is a scalar-swap of
//       ff_complex.hpp, which is DHB-licensed for its DDFUN heritage.
// Decision: (a) LBNL-BSD. Rationale: the file contains NO DDFUN/DHB-original
// arithmetic — the complex composition formulas (product, quotient, Kahan sqrt,
// Euler exp, polar log) are textbook identities, not DDFUN inventions, and every
// non-trivial numeric step is a QD-derived QuadFloat operation. Applying the DHB
// license would attribute copyright to Bailey personally and cite the wrong
// commercial contact (dhbailey@lbl.gov) for a file whose substance is the LBNL
// institutional QD package. This keeps the whole QF backend (qf_math.hpp +
// qf_complex.hpp) under one consistent license, matching T3.0a/T3.0b. Reet to
// confirm in review; see docs/PORT_NOTES_QF.md "License lineage (complex layer)".
//
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.

#pragma once

// Quad-float complex arithmetic — xp::QuadFloatComplex.
// All functions XPMATH_INLINE_FUNCTION (host + device via CUDA/HIP/SYCL).
// Depends on qf_math.hpp (the 4×FP32 real backend, T3.0a/T3.0b).
//
// DEPENDENCIES: none beyond the C++17 standard library and qf_math.hpp.
// In particular this header does NOT include or require Kokkos — see
// xp/config.hpp for how the portability facilities are supplied. Kokkos
// users get today's `Kokkos::Experimental::QuadFloatComplex` API
// unchanged through the compat wrapper at third_party/include/qf_complex.hpp,
// which is the only place `namespace Kokkos` is mentioned.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
//
// TABLE-FREE POSTURE (inherited from T3.0b, PORT_NOTES_QF §6). This header adds
// NO lookup tables — no sin_table / cos_table / inv_fact. Complex exp/sin/cos/
// sinh/cosh dispatch through qf_math.hpp's table-free real transcendentals
// (divide-by-k Taylor, joint sin/cos doublings), so they inherit T3.0b's §7
// exp term count and §8 sinh threshold. Complex-only compositions
// (exp(z) = exp(x)·(cos y + i sin y), etc.) call the real exp/cos/sin directly.
//
// Naming follows qf_math.hpp / ff_complex.hpp (T0.4/T2.0/T3.0a): type + math
// live under xp:: for eventual upstreaming. This remains a bespoke struct
// rather than Kokkos::complex<QuadFloat> — that integration is a separate
// future task.
//
// SINCOS / SINHCOSH OUTPUT ORDER (QF-specific gotcha, PORT_NOTES_QF §12). Unlike
// ff_math.hpp — whose sincos(a, x, y) writes x=cos, y=sin and sinhcosh(a, x, y)
// writes x=cosh, y=sinh — qf_math.hpp names its out-params sin-first:
// sincos(a, sin_a, cos_a) and sinhcosh(a, sinh_a, cosh_a). To keep each call
// site's downstream algebra byte-identical to ff_complex.hpp, this header passes
// the local (cos, sin) / (cosh, sinh) variables in SWAPPED positional order,
// i.e. sincos(a, s, c) and sinhcosh(a, sh, ch). The local variable meanings
// (c=cos, s=sin, ...) then match ff_complex.hpp exactly.

#include <xp/qf_math.hpp>

#if !defined(XPMATH_ON_DEVICE)
#  include <ostream>
#endif

namespace xp {

// ============================================================
// QuadFloatComplex struct
// ============================================================
// Members named re/im (mirroring ff_complex.hpp / dd_complex.hpp verbatim so the
// demo's mqr(i).re / .im field access and the real()/imag() accessors coexist).
// The task's shorthand "{ QuadFloat real, imag; }" denotes the two QuadFloat
// components; re/im is the faithful structural match.
struct QuadFloatComplex {
    QuadFloat re;
    QuadFloat im;

    XPMATH_INLINE_FUNCTION QuadFloatComplex() : re(0.0f), im(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplex(float r)                  : re(r),    im(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplex(QuadFloat r)             : re(r),    im(0.0f) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplex(float r, float i)         : re(r),    im(i)    {}
    XPMATH_INLINE_FUNCTION QuadFloatComplex(QuadFloat r, QuadFloat i) : re(r),    im(i)    {}
    XPMATH_INLINE_FUNCTION QuadFloatComplex(const QuadFloatComplex& o): re(o.re), im(o.im) {}
    XPMATH_INLINE_FUNCTION QuadFloatComplex& operator=(const QuadFloatComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    XPMATH_INLINE_FUNCTION QuadFloatComplex& operator=(QuadFloat r) {
        re = r; im = QuadFloat(0.0f); return *this;
    }

    // ff_complex.hpp:61-83 (operator+/-/*// and unary -).
    XPMATH_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloatComplex b) const {
        return QuadFloatComplex(add(re, b.re), add(im, b.im));
    }
    XPMATH_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloatComplex b) const {
        return QuadFloatComplex(subtract(re, b.re), subtract(im, b.im));
    }
    XPMATH_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloatComplex b) const {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i  (ff_complex.hpp:67-70)
        return QuadFloatComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                                add(multiply(re, b.im), multiply(im, b.re)));
    }
    XPMATH_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloatComplex b) const {
        // (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)  (ff_complex.hpp:71-80)
        if (b.re.f0 == 0.0f && b.im.f0 == 0.0f) {
            XPMATH_PRINTF("QFCOMPLEX: division by zero\n");
            return QuadFloatComplex();
        }
        // KI-8.  The |denominator|^2 formulation squares b's components, so it
        // overflows and underflows for denominators whose quotient is perfectly
        // representable -- the same exposure as the unscaled hypot.  Past the
        // gate, use Smith's algorithm (1962), which divides through by the
        // larger component first so no intermediate exceeds the operands.
        // Inside the gate the original expression is kept bit-for-bit: Smith
        // costs two divides instead of one reciprocal and is slightly less
        // accurate, and there is nothing to win where the direct form works.
        {
            float mre = detail::fabs(b.re.f0);
            float mim = detail::fabs(b.im.f0);
            float mb  = (mre > mim) ? mre : mim;
            if (!(mb <= 1.0e18f && mb >= 1.0e-18f)) {
                if (mre >= mim) {
                    QuadFloat rr = divide(b.im, b.re);
                    QuadFloat dd = add(b.re, multiply(b.im, rr));
                    return QuadFloatComplex(divide(add(re, multiply(im, rr)), dd),
                               divide(subtract(im, multiply(re, rr)), dd));
                } else {
                    QuadFloat rr = divide(b.re, b.im);
                    QuadFloat dd = add(multiply(b.re, rr), b.im);
                    return QuadFloatComplex(divide(add(multiply(re, rr), im), dd),
                               divide(subtract(multiply(im, rr), re), dd));
                }
            }
        }
        QuadFloat denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        QuadFloat inv   = divide(QuadFloat(1.0f), denom);
        return QuadFloatComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                                multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    XPMATH_INLINE_FUNCTION QuadFloatComplex operator-() const {
        return QuadFloatComplex(negate(re), negate(im));
    }

    XPMATH_INLINE_FUNCTION QuadFloatComplex& operator+=(QuadFloatComplex b) { *this = *this + b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplex& operator-=(QuadFloatComplex b) { *this = *this - b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplex& operator*=(QuadFloatComplex b) { *this = *this * b; return *this; }
    XPMATH_INLINE_FUNCTION QuadFloatComplex& operator/=(QuadFloatComplex b) { *this = *this / b; return *this; }

    XPMATH_INLINE_FUNCTION bool operator==(QuadFloatComplex b) const { return re==b.re && im==b.im; }
    XPMATH_INLINE_FUNCTION bool operator!=(QuadFloatComplex b) const { return !(*this == b); }

    XPMATH_INLINE_FUNCTION QuadFloat real() const { return re; }
    XPMATH_INLINE_FUNCTION QuadFloat imag() const { return im; }
};

#if !defined(XPMATH_ON_DEVICE)
inline std::ostream& operator<<(std::ostream& os, const QuadFloatComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed QuadFloat × QuadFloatComplex arithmetic  (ff_complex.hpp:104-114)
// ============================================================
XPMATH_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(add(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(add(r, z.re), z.im); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(subtract(z.re, r), z.im); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(subtract(r, z.re), negate(z.im)); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(multiply(z.re, r), multiply(z.im, r)); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(multiply(r, z.re), multiply(r, z.im)); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(divide(z.re, r), divide(z.im, r)); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(r) / z; }

// ============================================================
// Mixed float × QuadFloatComplex arithmetic  (ff_complex.hpp:116-126)
// ============================================================
XPMATH_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloatComplex z, float b) { return z + QuadFloat(b); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator+(float b, QuadFloatComplex z) { return QuadFloat(b) + z; }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloatComplex z, float b) { return z - QuadFloat(b); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator-(float b, QuadFloatComplex z) { return QuadFloat(b) - z; }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloatComplex z, float b) { return z * QuadFloat(b); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator*(float b, QuadFloatComplex z) { return QuadFloat(b) * z; }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloatComplex z, float b) { return z / QuadFloat(b); }
XPMATH_INLINE_FUNCTION QuadFloatComplex operator/(float b, QuadFloatComplex z) { return QuadFloat(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

// abs(z) = |z| = sqrt(re²+im²).  ff_complex.hpp:132-134 / dd_complex.hpp:145-147.
// KI-8.  abs(z) is a magnitude and the unscaled sqrt(re^2 + im^2) below returns
// nan above |z| ~ 1.8e19 and 0 below |z| ~ 1.1e-19 on the FP32-word backends, in
// both cases while the answer is representable.  Past that range, defer to the
// scaled hypot.
//
// Inside the range the ORIGINAL expression is kept verbatim rather than routed
// through hypot as well.  That is not conservatism for its own sake: hypot's
// own fast path is written with the primitive each backend's hypot already
// used, and on TF that is sqr() where this one is multiply().  Swapping them
// costs up to 3.04 digits at four grid points (measured on the 428,592-point
// sweep, TF c abs points 736/737/1528..1531), so the two call sites keep their
// own primitives and share only the scaled tail.
XPMATH_INLINE_FUNCTION QuadFloat abs(QuadFloatComplex z) {
    float mr = detail::fabs(z.re.f0);
    float mi = detail::fabs(z.im.f0);
    float m  = (mr > mi) ? mr : mi;
    if (!(m <= 1.0e18f && m >= 1.0e-18f)) return hypot(z.re, z.im);
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
}
// norm(z) = |z|² = re²+im² (std::norm convention; the squared magnitude, no
// sqrt).  ff_complex.hpp / dd_complex.hpp ship no norm(); this is the textbook
// definition (cf. abs above without the sqrt), added per the T3.0c op inventory.
XPMATH_INLINE_FUNCTION QuadFloat norm(QuadFloatComplex z) {
    return add(multiply(z.re, z.re), multiply(z.im, z.im));
}
// arg(z) = atan2(im, re) (std::arg convention; the polar angle).  ff_complex.hpp
// / dd_complex.hpp expose this only inside log() (ff_complex.hpp:174); surfaced
// as a standalone op per the T3.0c inventory. Uses qf_math.hpp atan2(y, x).
XPMATH_INLINE_FUNCTION QuadFloat arg(QuadFloatComplex z) {
    return atan2(z.im, z.re);
}
// conj(z) = re - im·i.  ff_complex.hpp:135-137 / dd_complex.hpp:148-150.
XPMATH_INLINE_FUNCTION QuadFloatComplex conj(QuadFloatComplex z) {
    return QuadFloatComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root  (Kahan-style; ff_complex.hpp:142-160 / dd_complex.hpp:155-174)
// ============================================================
// B = sqrt((R+|re|)/2) + i·sign(im)·sqrt((R-|re|)/2), R = |z|, arranged to avoid
// cancellation. The ½ and 2 are FP32-exact literals used via multiply_scalar
// (PORT_NOTES §3-lift), matching ff_complex.hpp:146-148.
XPMATH_INLINE_FUNCTION QuadFloatComplex sqrt(QuadFloatComplex z) {
    if (z.re.f0 == 0.0f && z.im.f0 == 0.0f) return QuadFloatComplex();
    QuadFloat r  = abs(z);   // KI-8: scaled magnitude, was sqrt(re^2+im^2) inline
    QuadFloat a1 = abs(z.re);
    QuadFloat s2 = multiply_scalar(add(r, a1), 0.5f);
    QuadFloat s0 = sqrt(s2);
    QuadFloat s1 = multiply_scalar(s0, 2.0f);
    QuadFloatComplex b;
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
// exp(z) = eˣ·(cos y + i·sin y), z = x + iy.  ff_complex.hpp:165-170 /
// dd_complex.hpp:179-184. Real exp + joint sincos are the table-free QF
// transcendentals (T3.0b §6/§7). NOTE the swapped sincos args (header §sincos):
// qf sincos writes (sin, cos), so pass (s, c) to keep c=cos(y), s=sin(y).
XPMATH_INLINE_FUNCTION QuadFloatComplex exp(QuadFloatComplex z) {
    QuadFloat er = exp(z.re);
    QuadFloat c, s;
    sincos(z.im, s, c);
    return QuadFloatComplex(multiply(er, c), multiply(er, s));
}

// log(z) = log|z| + i·arg(z).  ff_complex.hpp:172-176 / dd_complex.hpp:186-190.
XPMATH_INLINE_FUNCTION QuadFloatComplex log(QuadFloatComplex z) {
    QuadFloat modulus = abs(z);
    QuadFloat argument = atan2(z.im, z.re);
    return QuadFloatComplex(log(modulus), argument);
}

// log10(z) = log(z)/ln(10).  ff_complex.hpp:178-182 / dd_complex.hpp:192-196.
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
XPMATH_INLINE_FUNCTION QuadFloatComplex log1p(QuadFloatComplex w) {
    QuadFloat t = add(multiply_scalar(w.re, 2.0f),
                         add(multiply(w.re, w.re), multiply(w.im, w.im)));
    return QuadFloatComplex(multiply_scalar(log1p(t), 0.5f),
                               atan2(w.im, add(QuadFloat(1.0f), w.re)));
}

XPMATH_INLINE_FUNCTION QuadFloatComplex log10(QuadFloatComplex z) {
    QuadFloatComplex lg = log(z);
    QuadFloat ln10 = QuadFloat_log10();
    return QuadFloatComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
// sin(a+bi) = sin(a)·cosh(b) + i·cos(a)·sinh(b).  ff_complex.hpp:187-192 /
// dd_complex.hpp:201-207. Swapped sincos/sinhcosh args (header §sincos): local
// ca=cos(a), sa=sin(a), cb=cosh(b), sb=sinh(b).
XPMATH_INLINE_FUNCTION QuadFloatComplex sin(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sincos(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return QuadFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
// cos(a+bi) = cos(a)·cosh(b) - i·sin(a)·sinh(b).  ff_complex.hpp:193-198 /
// dd_complex.hpp:208-214.
XPMATH_INLINE_FUNCTION QuadFloatComplex cos(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sincos(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return QuadFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
// tan(z) = sin(z)/cos(z).  ff_complex.hpp:199-201 / dd_complex.hpp:215-217.
XPMATH_INLINE_FUNCTION QuadFloatComplex tan(QuadFloatComplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
// asin(z) = -i·log(iz + sqrt(1 - z²)).  ff_complex.hpp:206-213 /
// dd_complex.hpp:222-231. iz built by literal-lifted components; the 1 promoted
// via QuadFloat(1.0f) inside the single-arg QuadFloatComplex ctor (imag→0).
// KI-5(d) fix; see dd_complex.hpp:231-241 for the full rationale. On the real
// cut (Im(z) == +-0, |Re(z)| > 1) the sheet of sqrt(1 - z^2) is fixed by the
// sign of Im(z)'s zero, which the subtraction destroys (0 - (+-0) == +0 in
// round-to-nearest). Read it off Im(z) instead: Im(1 - z^2) = -2*Re*Im, so the
// root is negative-imaginary exactly when Re and Im share a sign.
XPMATH_INLINE_FUNCTION QuadFloatComplex asin(QuadFloatComplex z) {
    QuadFloatComplex iz  = QuadFloatComplex(negate(z.im), z.re);
    QuadFloatComplex z2  = z * z;
    QuadFloatComplex one_minus_z2 = QuadFloatComplex(QuadFloat(1.0f)) - z2;
    QuadFloatComplex root = sqrt(one_minus_z2);
    if (z.im.f0 == 0.0f && detail::fabs(z.re.f0) > 1.0f) {
        const bool want_neg = (detail::copysign(1.0f, z.re.f0) ==
                               detail::copysign(1.0f, z.im.f0));
        const bool have_neg = (detail::copysign(1.0f, root.im.f0) < 0.0f);
        if (want_neg != have_neg) root.im = negate(root.im);
    }
    QuadFloatComplex sum = iz + root;
    QuadFloatComplex lg  = log(sum);
    return QuadFloatComplex(lg.im, negate(lg.re));  // × (-i): (a+bi)(-i) = b - ai
}
// acos(z) = π/2 - asin(z).  ff_complex.hpp:214-218 / dd_complex.hpp:232-237.
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
XPMATH_INLINE_FUNCTION QuadFloatComplex sqrt_signed_cut(QuadFloat u, QuadFloat v, float vsign) {
    QuadFloatComplex r = sqrt(QuadFloatComplex(u, v));
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
XPMATH_INLINE_FUNCTION QuadFloatComplex acos(QuadFloatComplex z) {
    const QuadFloat one(1.0f);
    const QuadFloat half_im = multiply_scalar(z.im, 0.5f);
    const float s_im = detail::copysign(1.0f, z.im.f0);
    QuadFloatComplex rp = sqrt_signed_cut(
        multiply_scalar(add(one, z.re), 0.5f), half_im, s_im);
    QuadFloatComplex rm = sqrt_signed_cut(
        multiply_scalar(subtract(one, z.re), 0.5f), negate(half_im), -s_im);
    // w = rp + i*rm, with i*(a+bi) = -b + ai
    QuadFloatComplex w(subtract(rp.re, rm.im), add(rp.im, rm.re));
    return QuadFloatComplex(multiply_scalar(atan2(w.im, w.re), 2.0f), negate(asin(z).im));
}
// atan(z) = (i/2)·log((1 - iz)/(1 + iz)).  ff_complex.hpp:219-226 /
// dd_complex.hpp:238-247.
XPMATH_INLINE_FUNCTION QuadFloatComplex atan(QuadFloatComplex z) {
    QuadFloatComplex iz    = QuadFloatComplex(negate(z.im), z.re);
    QuadFloatComplex num   = QuadFloatComplex(QuadFloat(1.0f)) - iz;
    QuadFloatComplex den   = QuadFloatComplex(QuadFloat(1.0f)) + iz;
    QuadFloatComplex ratio = num / den;
    QuadFloatComplex lg    = log(ratio);
    // × (i/2): (a+bi)(i/2) = (-b/2) + (a/2)i
    return QuadFloatComplex(multiply_scalar(negate(lg.im), 0.5f), multiply_scalar(lg.re, 0.5f));
}

// ============================================================
// Complex hyperbolic
// ============================================================
// sinh(a+bi) = sinh(a)·cos(b) + i·cosh(a)·sin(b).  ff_complex.hpp:231-236 /
// dd_complex.hpp:252-258. Swapped args: ca=cosh(a), sa=sinh(a), cb=cos(b),
// sb=sin(b).
XPMATH_INLINE_FUNCTION QuadFloatComplex sinh(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sinhcosh(z.re, sa, ca);
    sincos(z.im, sb, cb);
    return QuadFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
// cosh(a+bi) = cosh(a)·cos(b) + i·sinh(a)·sin(b).  ff_complex.hpp:237-242 /
// dd_complex.hpp:259-265.
XPMATH_INLINE_FUNCTION QuadFloatComplex cosh(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sinhcosh(z.re, sa, ca);
    sincos(z.im, sb, cb);
    return QuadFloatComplex(multiply(ca, cb), multiply(sa, sb));
}
// tanh(a+bi): re = T/(cos²b + T²·sin²b), im = sin b·cos b·(1-T²)/(...),
// T = tanh(a).  ff_complex.hpp:243-251 / dd_complex.hpp:266-277. Denominator ≥ 0;
// uses the improved real tanh to avoid cancellation. Swapped sincos args:
// cb=cos(b), sb=sin(b).
XPMATH_INLINE_FUNCTION QuadFloatComplex tanh(QuadFloatComplex z) {
    QuadFloat T = tanh(z.re);
    QuadFloat cb, sb;
    sincos(z.im, sb, cb);
    QuadFloat T2    = multiply(T, T);
    QuadFloat denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return QuadFloatComplex(divide(T, denom),
                            divide(multiply(multiply(sb, cb), subtract(QuadFloat(1.0f), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
// asinh(z) = log(z + sqrt(z² + 1)), reflected into the right half-plane via the
// oddness identity -asinh(-z) when the direct form would cancel. KI-5(a); see
// dd_complex.hpp's asinh for the full rationale, the magnitude threshold and its
// measured justification, and the Re(z) == ±0 boundary decision — all identical
// here.
XPMATH_INLINE_FUNCTION QuadFloatComplex asinh(QuadFloatComplex z) {
    const float t = 4.0f;   // kXpAsinhReflect
    if (z.re.f0 < 0.0f && (-z.re.f0 > t || detail::fabs(z.im.f0) > t)) {
        QuadFloatComplex w = -z;
        return -log(w + sqrt(w*w + QuadFloatComplex(QuadFloat(1.0f))));
    }
    return log(z + sqrt(z*z + QuadFloatComplex(QuadFloat(1.0f))));
}
// acosh(z) = log(z + sqrt(z² - 1)).  ff_complex.hpp:259-261 / dd_complex.hpp:286-289.
// KI-1 fix: Kahan 1987's branch-correct form, acosh(z) = 2*log(sqrt((z+1)/2) +
// sqrt((z-1)/2)). See dd_complex.hpp:346-357 for the full rationale. The old
// log(z + sqrt(z*z - 1)) form was on the wrong sqrt sheet throughout
// Re(z) < 0, and overflowed above |z| ~ 1.8e19 where z*z leaves FP32 range.
XPMATH_INLINE_FUNCTION QuadFloatComplex acosh(QuadFloatComplex z) {
    const QuadFloat one(1.0f);
    const QuadFloat half_im = multiply_scalar(z.im, 0.5f);
    QuadFloatComplex rp = sqrt(QuadFloatComplex(
        multiply_scalar(add(z.re, one), 0.5f), half_im));
    QuadFloatComplex rm = sqrt(QuadFloatComplex(
        multiply_scalar(subtract(z.re, one), 0.5f), half_im));
    QuadFloatComplex lg = log(rp + rm);
    return QuadFloatComplex(multiply_scalar(lg.re, 2.0f),
                            multiply_scalar(lg.im, 2.0f));
}
// atanh(z) = ½·log((1 + z)/(1 - z)).  ff_complex.hpp:262-266 / dd_complex.hpp:290-295.
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
XPMATH_INLINE_FUNCTION QuadFloatComplex atanh(QuadFloatComplex z) {
    const QuadFloat one(1.0f);
    const float t = 0.0625f;   // kXpAtanhSmall
    QuadFloatComplex r;
    if (detail::fabs(z.re.f0) < t && detail::fabs(z.im.f0) < t) {
        QuadFloat omx = subtract(one, z.re);
        QuadFloat den = add(multiply(omx, omx), multiply(z.im, z.im));
        r.re = multiply_scalar(
            log1p(divide(multiply_scalar(z.re, 4.0f), den)), 0.25f);
        r.im = multiply_scalar(
            atan2(multiply_scalar(z.im, 2.0f),
                  subtract(one, add(multiply(z.re, z.re), multiply(z.im, z.im)))),
            0.5f);
    } else {
        QuadFloatComplex lg = log((QuadFloatComplex(one) + z) / (QuadFloatComplex(one) - z));
        r.re = multiply_scalar(lg.re, 0.5f);
        r.im = multiply_scalar(lg.im, 0.5f);
    }
    if (z.im.f0 == 0.0f && detail::fabs(z.re.f0) > 1.0f) {
        QuadFloat half_pi = multiply_scalar(QuadFloat_pi(), 0.5f);
        if (detail::copysign(1.0f, z.im.f0) < 0.0f) half_pi = negate(half_pi);
        r.im = half_pi;
    }
    return r;
}

// ============================================================
// Complex power and polar
// ============================================================
// pow(z, w) = exp(w·log(z)).  ff_complex.hpp:271-274 / dd_complex.hpp:300-304.
XPMATH_INLINE_FUNCTION QuadFloatComplex pow(QuadFloatComplex z, QuadFloatComplex w) {
    if (z.re.f0 == 0.0f && z.im.f0 == 0.0f) return QuadFloatComplex();
    return exp(w * log(z));
}

// polar(r, theta) = r·(cos θ + i·sin θ).  ff_complex.hpp:276-280 /
// dd_complex.hpp:306-311. Swapped sincos args: c=cos(θ), s=sin(θ).
XPMATH_INLINE_FUNCTION QuadFloatComplex polar(QuadFloat r, QuadFloat theta) {
    QuadFloat c, s;
    sincos(theta, s, c);
    return QuadFloatComplex(multiply(r, c), multiply(r, s));
}

} // namespace xp
