// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// KOKKOS COMPATIBILITY WRAPPER for the standalone float-float complex core.
//
// The implementation moved to include/xp/ff_complex.hpp (namespace xp,
// zero Kokkos dependency). This file is what `#include <ff_complex.hpp>` has
// always resolved to, and it keeps that API byte-for-byte: after including
// it, `Kokkos::Experimental::FloatFloatComplex`, every free function that
// used to live in `Kokkos::Experimental`, and every `Kokkos::`-namespace
// math forwarder resolve exactly as before. No consumer — the FF complex
// tests, the FF demo — needs a single edit.
//
// Ported from DDFUN v04:
//   https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz
//   Original author: David H. Bailey (LBNL retired / UC Davis)
//   Original license: DHB-License (modified BSD-3-Clause with §3
//     grant-back clause). Full text: LICENSES/LicenseRef-DHB-License.txt
//     or https://www.davidhbailey.com/dhbsoftware/DHB-License.txt.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
// Every Kokkos-facing name below is an alias, so the rename touched only this
// file's right-hand sides.

#pragma once

#include <Kokkos_Core.hpp>

#include <xp/ff_complex.hpp>

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
// The type alias is what makes `Kokkos::Experimental::FloatFloatComplex`
// name the same type as `xp::FloatFloatComplex` — not a distinct wrapper —
// so the two spellings are interchangeable in every signature.
//
// Operators are deliberately absent: `ffc + ffc`, `ff * ffc` and
// `os << ffc` are found by ADL through the argument's real namespace (xp),
// so re-declaring them here would be redundant.
namespace Kokkos {
namespace Experimental {

// --- the type -----------------------------------------------------------
using FloatFloatComplex = xp::FloatFloatComplex;

// --- complex operations -------------------------------------------------
using xp::abs;
using xp::acos;
using xp::acosh;
using xp::asin;
using xp::asinh;
using xp::atan;
using xp::atanh;
using xp::conj;
using xp::cos;
using xp::cosh;
using xp::exp;
using xp::log;
using xp::log10;
using xp::polar;
using xp::pow;
using xp::sin;
using xp::sinh;
using xp::sqrt;
using xp::tan;
using xp::tanh;

}  // namespace Experimental
}  // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0)
// ============================================================
// Mirror of ff_math.hpp: so Kokkos::exp(ffc) works identically to
// Kokkos::exp(Kokkos::complex<double>). One-line forwards. Arithmetic operators
// are reached directly / via ADL and are not re-exposed here.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat        abs(Experimental::FloatFloatComplex z)   { return Experimental::abs(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex conj(Experimental::FloatFloatComplex z)  { return Experimental::conj(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex sqrt(Experimental::FloatFloatComplex z)  { return Experimental::sqrt(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex exp(Experimental::FloatFloatComplex z)   { return Experimental::exp(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex log(Experimental::FloatFloatComplex z)   { return Experimental::log(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex log10(Experimental::FloatFloatComplex z) { return Experimental::log10(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex sin(Experimental::FloatFloatComplex z)   { return Experimental::sin(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex cos(Experimental::FloatFloatComplex z)   { return Experimental::cos(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex tan(Experimental::FloatFloatComplex z)   { return Experimental::tan(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex asin(Experimental::FloatFloatComplex z)  { return Experimental::asin(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex acos(Experimental::FloatFloatComplex z)  { return Experimental::acos(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex atan(Experimental::FloatFloatComplex z)  { return Experimental::atan(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex sinh(Experimental::FloatFloatComplex z)  { return Experimental::sinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex cosh(Experimental::FloatFloatComplex z)  { return Experimental::cosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex tanh(Experimental::FloatFloatComplex z)  { return Experimental::tanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex asinh(Experimental::FloatFloatComplex z) { return Experimental::asinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex acosh(Experimental::FloatFloatComplex z) { return Experimental::acosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex atanh(Experimental::FloatFloatComplex z) { return Experimental::atanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex pow(Experimental::FloatFloatComplex z, Experimental::FloatFloatComplex w) { return Experimental::pow(z, w); }
// clang-format on
}  // namespace Kokkos
