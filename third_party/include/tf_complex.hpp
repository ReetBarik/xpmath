// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (real three-word algorithms
//   this complex layer composes on; Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// KOKKOS COMPATIBILITY WRAPPER for the standalone triple-float complex core.
//
// The implementation moved to include/xp/tf_complex.hpp (namespace xp,
// zero Kokkos dependency). This file is what `#include <tf_complex.hpp>` has
// always resolved to, and it keeps that API byte-for-byte: after including
// it, `Kokkos::Experimental::TripleFloatComplex`, every free function that
// used to live in `Kokkos::Experimental`, and every `Kokkos::`-namespace
// math forwarder resolve exactly as before. No consumer — the TF complex
// tests, the TF demo — needs a single edit.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
// Every Kokkos-facing name below is an alias, so the rename touched only this
// file's right-hand sides.
//
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.

#pragma once

#include <Kokkos_Core.hpp>

#include <xp/tf_complex.hpp>

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
// The type alias is what makes `Kokkos::Experimental::TripleFloatComplex`
// name the same type as `xp::TripleFloatComplex` — not a distinct wrapper —
// so the two spellings are interchangeable in every signature.
//
// Operators are deliberately absent: `tfc + tfc`, `tf * tfc` and
// `os << tfc` are found by ADL through the argument's real namespace (xp),
// so re-declaring them here would be redundant.
namespace Kokkos {
namespace Experimental {

// --- the type -----------------------------------------------------------
using TripleFloatComplex = xp::TripleFloatComplex;

// --- complex operations -------------------------------------------------
using xp::abs;
using xp::acos;
using xp::acosh;
using xp::arg;
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
using xp::norm;
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
// Re-exposure under namespace Kokkos (T0.4/T2.0/T3.0a convention)
// ============================================================
// Extends the tf_math.hpp Kokkos re-exposure block for the complex ops, so
// Kokkos::exp(tfc) works identically to Kokkos::exp(Kokkos::complex<double>).
// One-line forwards; this does NOT duplicate the real forwards in tf_math.hpp
// (different argument type → distinct overloads). Arithmetic operators are
// reached directly / via ADL and are not re-exposed here.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::TripleFloat        abs(Experimental::TripleFloatComplex z)   { return Experimental::abs(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloat        norm(Experimental::TripleFloatComplex z)  { return Experimental::norm(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloat        arg(Experimental::TripleFloatComplex z)   { return Experimental::arg(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex conj(Experimental::TripleFloatComplex z)  { return Experimental::conj(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex sqrt(Experimental::TripleFloatComplex z)  { return Experimental::sqrt(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex exp(Experimental::TripleFloatComplex z)   { return Experimental::exp(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex log(Experimental::TripleFloatComplex z)   { return Experimental::log(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex log10(Experimental::TripleFloatComplex z) { return Experimental::log10(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex sin(Experimental::TripleFloatComplex z)   { return Experimental::sin(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex cos(Experimental::TripleFloatComplex z)   { return Experimental::cos(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex tan(Experimental::TripleFloatComplex z)   { return Experimental::tan(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex asin(Experimental::TripleFloatComplex z)  { return Experimental::asin(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex acos(Experimental::TripleFloatComplex z)  { return Experimental::acos(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex atan(Experimental::TripleFloatComplex z)  { return Experimental::atan(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex sinh(Experimental::TripleFloatComplex z)  { return Experimental::sinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex cosh(Experimental::TripleFloatComplex z)  { return Experimental::cosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex tanh(Experimental::TripleFloatComplex z)  { return Experimental::tanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex asinh(Experimental::TripleFloatComplex z) { return Experimental::asinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex acosh(Experimental::TripleFloatComplex z) { return Experimental::acosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex atanh(Experimental::TripleFloatComplex z) { return Experimental::atanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::TripleFloatComplex pow(Experimental::TripleFloatComplex z, Experimental::TripleFloatComplex w) { return Experimental::pow(z, w); }
// clang-format on
}  // namespace Kokkos
