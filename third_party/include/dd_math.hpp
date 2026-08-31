// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// KOKKOS COMPATIBILITY WRAPPER for the standalone double-double core.
//
// The implementation moved to include/EPLIB/dd_math.hpp (namespace eplib,
// zero Kokkos dependency). This file is what `#include <dd_math.hpp>` has
// always resolved to, and it keeps that API byte-for-byte: after including
// it, `Kokkos::Experimental::DoubleDouble`, every free function that used to
// live in `Kokkos::Experimental`, and every `Kokkos::`-namespace math
// forwarder resolve exactly as before. No consumer — dd_complex.hpp, the DD
// tests, the DD demos — needs a single edit.
//
// Licensing is unchanged: the algorithms are the DDFUN v04 port (David H.
// Bailey, DHB-License); see include/EPLIB/dd_math.hpp for the full notice.
//
// NAME IS A PLACEHOLDER: `eplib::` is provisional pending the S2 naming memo
// (docs/UPSTREAM_PLAN_STATUS.md); S3 applies the ratified name. Because every
// Kokkos-facing name below is an alias, the rename touches only this file's
// right-hand sides.

#pragma once

#include <Kokkos_Core.hpp>

#include <EPLIB/dd_math.hpp>

// ============================================================
// Re-exposure under namespace Kokkos::Experimental
// ============================================================
// Explicit using-declarations rather than `using namespace eplib;`. Three
// reasons: (1) the Kokkos-visible API surface stays an auditable list rather
// than "whatever eplib happens to declare"; (2) a using-DIRECTIVE participates
// in qualified lookup in a way that is easy to get subtly wrong when Kokkos
// itself declares same-named overloads in Kokkos::Experimental (sqrt, exp,
// ... for half_t), whereas a using-DECLARATION simply merges into that
// overload set; (3) it documents, for the S4 RFC, exactly what an upstream
// Kokkos would be adopting.
//
// The type alias is what makes `Kokkos::Experimental::DoubleDouble` name the
// same type as `eplib::DoubleDouble` — not a distinct wrapper — so the two
// spellings are interchangeable in every signature, including across the
// dd_complex.hpp boundary.
//
// Operators are deliberately absent: `dd + dd`, `1.0 * dd` and
// `os << dd` are found by ADL through the argument's real namespace (eplib),
// so re-declaring them here would be redundant.
namespace Kokkos {
namespace Experimental {

// --- the type -----------------------------------------------------------
using DoubleDouble = eplib::DoubleDouble;

// --- constants ----------------------------------------------------------
using eplib::DoubleDouble_e;
using eplib::DoubleDouble_euler_gamma;
using eplib::DoubleDouble_log10;
using eplib::DoubleDouble_log2;
using eplib::DoubleDouble_pi;
using eplib::DoubleDouble_sqrt2;

// --- primitive arithmetic -----------------------------------------------
using eplib::add;
using eplib::divide;
using eplib::divide_scalar;
using eplib::multiply;
using eplib::multiply_scalar;
using eplib::negate;
using eplib::subtract;
using eplib::two_prod;

// --- basic math ---------------------------------------------------------
using eplib::abs;
using eplib::pow_int;
using eplib::round_to_nearest_int;
using eplib::sqrt;

// --- exp / log family ---------------------------------------------------
using eplib::exp;
using eplib::exp10;
using eplib::exp2;
using eplib::expm1;
using eplib::log;
using eplib::log10;
using eplib::log1p;
using eplib::log2;

// --- trigonometric ------------------------------------------------------
using eplib::acos;
using eplib::angle;
using eplib::asin;
using eplib::atan;
using eplib::atan2;
using eplib::cos;
using eplib::sin;
using eplib::sincos;
using eplib::tan;

// --- hyperbolic ---------------------------------------------------------
using eplib::acosh;
using eplib::asinh;
using eplib::atanh;
using eplib::cosh;
using eplib::sinh;
using eplib::sinhcosh;
using eplib::tanh;

// --- multi-argument -----------------------------------------------------
using eplib::copysign;
using eplib::fdim;
using eplib::fma;
using eplib::fmax;
using eplib::fmin;
using eplib::fmod;
using eplib::hypot;
using eplib::pow;
using eplib::remainder;

// --- rounding -----------------------------------------------------------
using eplib::ceil;
using eplib::floor;
using eplib::round;
using eplib::trunc;

// --- special functions --------------------------------------------------
using eplib::bessel_j0;
using eplib::bessel_j1;
using eplib::bessel_jn;
using eplib::bessel_y0;
using eplib::bessel_y1;
using eplib::bessel_yn;
using eplib::erf;
using eplib::erfc;
using eplib::erfc_asymptotic_sum;
using eplib::expint;
using eplib::incgamma;
using eplib::tgamma;
using eplib::zeta;

}  // namespace Experimental
}  // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp's __float128 overloads so user code
// can call Kokkos::exp(dd) identically to Kokkos::exp(double)/Kokkos::exp(
// __float128). One-line forwards to the Kokkos::Experimental implementations.
// NOTE: add/subtract/multiply/divide are deliberately NOT forwarded here — those
// are reached via operators and explicit ADL, not as Kokkos::add etc.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble abs(Experimental::DoubleDouble x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble sqrt(Experimental::DoubleDouble x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble exp(Experimental::DoubleDouble x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble exp2(Experimental::DoubleDouble x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble exp10(Experimental::DoubleDouble x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble expm1(Experimental::DoubleDouble x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log(Experimental::DoubleDouble x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log2(Experimental::DoubleDouble x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log10(Experimental::DoubleDouble x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble log1p(Experimental::DoubleDouble x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble sin(Experimental::DoubleDouble x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble cos(Experimental::DoubleDouble x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble tan(Experimental::DoubleDouble x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble asin(Experimental::DoubleDouble x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble acos(Experimental::DoubleDouble x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble atan(Experimental::DoubleDouble x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble atan2(Experimental::DoubleDouble y, Experimental::DoubleDouble x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble sinh(Experimental::DoubleDouble x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble cosh(Experimental::DoubleDouble x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble tanh(Experimental::DoubleDouble x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble asinh(Experimental::DoubleDouble x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble acosh(Experimental::DoubleDouble x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble atanh(Experimental::DoubleDouble x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble pow(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble hypot(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fmod(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble remainder(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble copysign(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fmax(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fmin(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fdim(Experimental::DoubleDouble a, Experimental::DoubleDouble b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble fma(Experimental::DoubleDouble a, Experimental::DoubleDouble b, Experimental::DoubleDouble c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble ceil(Experimental::DoubleDouble x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble floor(Experimental::DoubleDouble x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble round(Experimental::DoubleDouble x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble trunc(Experimental::DoubleDouble x) { return Experimental::trunc(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble erf(Experimental::DoubleDouble x)   { return Experimental::erf(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble erfc(Experimental::DoubleDouble x)  { return Experimental::erfc(x); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble tgamma(Experimental::DoubleDouble x){ return Experimental::tgamma(x); }
// clang-format on
}  // namespace Kokkos
