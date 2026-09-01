// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// KOKKOS COMPATIBILITY WRAPPER for the standalone float-float core.
//
// The implementation moved to include/xp/ff_math.hpp (namespace xp,
// zero Kokkos dependency). This file is what `#include <ff_math.hpp>` has
// always resolved to, and it keeps that API byte-for-byte: after including
// it, `Kokkos::Experimental::FloatFloat`, every free function that used to
// live in `Kokkos::Experimental`, and every `Kokkos::`-namespace math
// forwarder resolve exactly as before. No consumer — ff_complex.hpp, the FF
// tests, the FF demos — needs a single edit.
//
// Licensing is unchanged: the algorithms descend from the DDFUN v04 port
// (David H. Bailey, DHB-License); see include/xp/ff_math.hpp for the full
// notice and PORT_NOTES.md for the FP32-specific modifications.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
// Every Kokkos-facing name below is an alias, so the rename touched only this
// file's right-hand sides.

#pragma once

#include <Kokkos_Core.hpp>

#include <xp/ff_math.hpp>

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
// The type alias is what makes `Kokkos::Experimental::FloatFloat` name the
// same type as `xp::FloatFloat` — not a distinct wrapper — so the two
// spellings are interchangeable in every signature, including across the
// ff_complex.hpp boundary.
//
// Operators are deliberately absent: `ff + ff`, `1.0f * ff` and
// `os << ff` are found by ADL through the argument's real namespace (xp),
// so re-declaring them here would be redundant.
namespace Kokkos {
namespace Experimental {

// --- the type -----------------------------------------------------------
using FloatFloat = xp::FloatFloat;

// --- constants ----------------------------------------------------------
using xp::FloatFloat_e;
using xp::FloatFloat_euler_gamma;
using xp::FloatFloat_log10;
using xp::FloatFloat_log2;
using xp::FloatFloat_pi;
using xp::FloatFloat_sqrt2;

// --- primitive arithmetic -----------------------------------------------
using xp::add;
using xp::divide;
using xp::divide_scalar;
using xp::multiply;
using xp::multiply_scalar;
using xp::negate;
using xp::subtract;
using xp::two_prod;

// --- basic math ---------------------------------------------------------
using xp::abs;
using xp::pow_int;
using xp::round_to_nearest_int;
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

// --- special functions --------------------------------------------------
using xp::erf;
using xp::erfc;
using xp::erfc_asymptotic_sum;
using xp::incgamma;
using xp::tgamma;

}  // namespace Experimental
}  // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp's __float128 overloads so user code
// can call Kokkos::exp(ff) identically to Kokkos::exp(double)/Kokkos::exp(
// __float128). One-line forwards to the Kokkos::Experimental implementations.
// NOTE: add/subtract/multiply/divide are deliberately NOT forwarded here — those
// are reached via operators and explicit ADL, not as Kokkos::add etc.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat abs(Experimental::FloatFloat x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sqrt(Experimental::FloatFloat x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp(Experimental::FloatFloat x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp2(Experimental::FloatFloat x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp10(Experimental::FloatFloat x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat expm1(Experimental::FloatFloat x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log(Experimental::FloatFloat x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log2(Experimental::FloatFloat x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log10(Experimental::FloatFloat x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log1p(Experimental::FloatFloat x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sin(Experimental::FloatFloat x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat cos(Experimental::FloatFloat x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tan(Experimental::FloatFloat x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat asin(Experimental::FloatFloat x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat acos(Experimental::FloatFloat x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atan(Experimental::FloatFloat x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atan2(Experimental::FloatFloat y, Experimental::FloatFloat x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sinh(Experimental::FloatFloat x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat cosh(Experimental::FloatFloat x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tanh(Experimental::FloatFloat x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat asinh(Experimental::FloatFloat x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat acosh(Experimental::FloatFloat x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atanh(Experimental::FloatFloat x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat pow(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat hypot(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmod(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat remainder(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat copysign(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmax(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmin(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fdim(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fma(Experimental::FloatFloat a, Experimental::FloatFloat b, Experimental::FloatFloat c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat ceil(Experimental::FloatFloat x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat floor(Experimental::FloatFloat x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat round(Experimental::FloatFloat x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat trunc(Experimental::FloatFloat x) { return Experimental::trunc(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat erf(Experimental::FloatFloat x)   { return Experimental::erf(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat erfc(Experimental::FloatFloat x)  { return Experimental::erfc(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tgamma(Experimental::FloatFloat x){ return Experimental::tgamma(x); }
// clang-format on
}  // namespace Kokkos
