// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (original algorithms;
//   Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// KOKKOS COMPATIBILITY WRAPPER for the standalone quad-float core.
//
// The implementation moved to include/xp/qf_math.hpp (namespace xp,
// zero Kokkos dependency). This file is what `#include <qf_math.hpp>` has
// always resolved to, and it keeps that API byte-for-byte: after including
// it, `Kokkos::Experimental::QuadFloat`, every free function that used to
// live in `Kokkos::Experimental`, and every `Kokkos::`-namespace math
// forwarder resolve exactly as before. No consumer — qf_complex.hpp, the QF
// tests, the QF demos — needs a single edit.
//
// Licensing is unchanged: the algorithms descend from the QD 2.3.24 port
// (Hida-Li-Bailey, LBNL-BSD-License); see include/xp/qf_math.hpp for the full
// notice and docs/PORT_NOTES_QF.md for the FP32-specific modifications.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
// Every Kokkos-facing name below is an alias, so the rename touched only this
// file's right-hand sides.

#pragma once

#include <Kokkos_Core.hpp>

#include <xp/qf_math.hpp>

// ============================================================
// Re-exposure under namespace Kokkos::Experimental
// ============================================================
// Explicit using-declarations rather than `using namespace xp;`. Three
// reasons: (1) the Kokkos-visible API surface stays an auditable list rather
// than "whatever xp happens to declare"; (2) a using-DIRECTIVE participates
// in qualified lookup in a way that is easy to get subtly wrong when Kokkos
// itself declares same-named overloads in Kokkos::Experimental (sqrt, exp,
// ... for half_t), whereas a using-DECLARATION simply merges into that
// overload set; (3) it documents, for the S4 RFC, exactly what an upstream
// Kokkos would be adopting.
//
// The type alias is what makes `Kokkos::Experimental::QuadFloat` name the
// same type as `xp::QuadFloat` — not a distinct wrapper — so the two
// spellings are interchangeable in every signature, including across the
// qf_complex.hpp boundary.
//
// Operators are deliberately absent: `qf + qf`, `1.0f * qf` and
// `os << qf` are found by ADL through the argument's real namespace (xp),
// so re-declaring them here would be redundant.
namespace Kokkos {
namespace Experimental {

// --- the type -----------------------------------------------------------
using QuadFloat = xp::QuadFloat;

// --- constants ----------------------------------------------------------
using xp::QuadFloat_e;
using xp::QuadFloat_euler_gamma;
using xp::QuadFloat_log10;
using xp::QuadFloat_log2;
using xp::QuadFloat_pi;
using xp::QuadFloat_sqrt2;

// --- primitive arithmetic -----------------------------------------------
using xp::add;
using xp::divide;
using xp::divide_accurate;
using xp::divide_scalar;
using xp::ieee_add;
using xp::mul_pwr2;
using xp::multiply;
using xp::multiply_scalar;
using xp::negate;
using xp::qf_quick_two_sum;  // needed by qf_eft_test
using xp::qf_two_prod;       // needed by qf_eft_test, qf_fma_guard_test
using xp::qf_two_sqr;        // needed by qf_fma_guard_test
using xp::qf_two_sum;        // needed by qf_eft_test, qf_fma_guard_test
using xp::renorm;            // needed by qf_eft_test
using xp::renorm_4;          // needed by qf_eft_test
using xp::sloppy_add;
using xp::subtract;

// --- basic math ---------------------------------------------------------
using xp::abs;
using xp::pow_int;
using xp::round_to_nearest_int;
using xp::sqr;
using xp::sqrt;

// --- exp / log family ---------------------------------------------------
using xp::exp;
using xp::exp10;
using xp::exp2;
using xp::expm1;
using xp::log;
using xp::log10;
using xp::log1p;
using xp::log2;

// --- trigonometric ------------------------------------------------------
using xp::acos;
using xp::angle;
using xp::asin;
using xp::atan;
using xp::atan2;
using xp::cos;
using xp::sin;
using xp::sincos;
using xp::tan;

// --- hyperbolic ---------------------------------------------------------
using xp::acosh;
using xp::asinh;
using xp::atanh;
using xp::cosh;
using xp::sinh;
using xp::sinhcosh;
using xp::tanh;

// --- multi-argument -----------------------------------------------------
using xp::copysign;
using xp::fdim;
using xp::fma;
using xp::fmax;
using xp::fmin;
using xp::fmod;
using xp::hypot;
using xp::pow;
using xp::remainder;

// --- rounding -----------------------------------------------------------
using xp::ceil;
using xp::floor;
using xp::round;
using xp::trunc;

}  // namespace Experimental
}  // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp's __float128 overloads so user code
// can call Kokkos::exp(qf) identically to Kokkos::exp(double)/Kokkos::exp(
// __float128). One-line forwards to the Kokkos::Experimental implementations.
// NOTE: add/subtract/multiply/divide are deliberately NOT forwarded here — those
// are reached via operators and explicit ADL, not as Kokkos::add etc.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat abs(Experimental::QuadFloat x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sqrt(Experimental::QuadFloat x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat exp(Experimental::QuadFloat x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat exp2(Experimental::QuadFloat x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat exp10(Experimental::QuadFloat x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat expm1(Experimental::QuadFloat x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log(Experimental::QuadFloat x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log2(Experimental::QuadFloat x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log10(Experimental::QuadFloat x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log1p(Experimental::QuadFloat x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sin(Experimental::QuadFloat x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat cos(Experimental::QuadFloat x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat tan(Experimental::QuadFloat x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat asin(Experimental::QuadFloat x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat acos(Experimental::QuadFloat x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat atan(Experimental::QuadFloat x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat atan2(Experimental::QuadFloat y, Experimental::QuadFloat x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sinh(Experimental::QuadFloat x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat cosh(Experimental::QuadFloat x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat tanh(Experimental::QuadFloat x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat asinh(Experimental::QuadFloat x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat acosh(Experimental::QuadFloat x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat atanh(Experimental::QuadFloat x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat pow(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat hypot(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fmod(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat remainder(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat copysign(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fmax(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fmin(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fdim(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fma(Experimental::QuadFloat a, Experimental::QuadFloat b, Experimental::QuadFloat c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat ceil(Experimental::QuadFloat x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat floor(Experimental::QuadFloat x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat round(Experimental::QuadFloat x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat trunc(Experimental::QuadFloat x) { return Experimental::trunc(x); }
// clang-format on
}  // namespace Kokkos
