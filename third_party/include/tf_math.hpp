// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (original algorithms;
//   Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// KOKKOS COMPATIBILITY WRAPPER for the standalone triple-float core.
//
// The implementation moved to include/xp/tf_math.hpp (namespace xp,
// zero Kokkos dependency). This file is what `#include <tf_math.hpp>` resolves
// to, and it keeps that API byte-for-byte: after including it,
// `Kokkos::Experimental::TripleFloat`, every free function that used to live
// in `Kokkos::Experimental`, and every `Kokkos::`-namespace math forwarder
// resolve exactly as before. No consumer — tf_complex.hpp (future), the TF
// tests (future), the TF demos (future) — needs a single edit.
//
// Licensing is unchanged: the algorithms descend from the QD 2.3.24 port
// (Hida-Li-Bailey, LBNL-BSD-License); see include/xp/tf_math.hpp for the full
// notice and docs/PORT_NOTES_TF.md for the k=3-specific modifications.
//
// NAMING (ratified via S2 naming memo + S3): xp:: = extended precision,
// companion to MxP (mixed precision). See include/xp/config.hpp for rationale.
// Every Kokkos-facing name below is an alias, so the rename touched only this
// file's right-hand sides.

#pragma once

#include <Kokkos_Core.hpp>

#include <xp/tf_math.hpp>

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
// The type alias is what makes `Kokkos::Experimental::TripleFloat` name the
// same type as `xp::TripleFloat` — not a distinct wrapper — so the two
// spellings are interchangeable in every signature, including across the
// tf_complex.hpp (future) boundary.
//
// Operators are deliberately absent: `tf + tf`, `1.0f * tf` and
// `os << tf` are found by ADL through the argument's real namespace (xp),
// so re-declaring them here would be redundant.
namespace Kokkos {
namespace Experimental {

// --- the type -----------------------------------------------------------
using TripleFloat = xp::TripleFloat;

// --- constants ----------------------------------------------------------
using xp::TripleFloat_e;
using xp::TripleFloat_euler_gamma;
using xp::TripleFloat_log10;
using xp::TripleFloat_log2;
using xp::TripleFloat_pi;
using xp::TripleFloat_sqrt2;

// --- primitive arithmetic -----------------------------------------------
using xp::add;
using xp::divide;
using xp::divide_scalar;
using xp::ieee_add;
using xp::mul_pwr2;
using xp::multiply;
using xp::multiply_scalar;
using xp::negate;
using xp::renorm;       // needed by tf_eft_test (future)
using xp::renorm_3;     // needed by tf_eft_test (future)
using xp::sloppy_add;
using xp::subtract;
using xp::tf_quick_two_sum;  // needed by tf_eft_test (future)
using xp::tf_two_prod;       // needed by tf_eft_test, tf_fma_guard_test (future)
using xp::tf_two_sqr;        // needed by tf_fma_guard_test (future)
using xp::tf_two_sum;        // needed by tf_eft_test, tf_fma_guard_test (future)

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
using xp::pow;

// --- trig ---------------------------------------------------------------
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

// --- rounding / data ops ------------------------------------------------
using xp::ceil;
using xp::fdim;
using xp::floor;
using xp::fma;
using xp::fmax;
using xp::fmin;
using xp::fmod;
using xp::remainder;
using xp::round;
using xp::trunc;

}  // namespace Experimental

// ============================================================
// Kokkos:: namespace math forwarders (mirrors dd/ff/qf pattern)
// ============================================================
// Free functions in `namespace Kokkos` so that ADL-unqualified calls like
// `exp(tf)` work inside a Kokkos:: context. The standalone header declares
// everything in `namespace xp`, which ADL finds through the TripleFloat
// argument, but legacy Kokkos code may expect `Kokkos::exp(tf)` to resolve
// like `Kokkos::exp(double)` does. These overloads provide that spelling.
// The compat wrappers for dd/ff/qf all carry this block; TF mirrors it.
using xp::abs;
using xp::acos;
using xp::acosh;
using xp::asin;
using xp::asinh;
using xp::atan;
using xp::atan2;
using xp::atanh;
using xp::ceil;
using xp::cos;
using xp::cosh;
using xp::exp;
using xp::exp10;
using xp::exp2;
using xp::expm1;
using xp::fdim;
using xp::floor;
using xp::fma;
using xp::fmax;
using xp::fmin;
using xp::fmod;
using xp::log;
using xp::log10;
using xp::log1p;
using xp::log2;
using xp::pow;
using xp::remainder;
using xp::round;
using xp::sin;
using xp::sinh;
using xp::sqrt;
using xp::tan;
using xp::tanh;
using xp::trunc;

}  // namespace Kokkos
