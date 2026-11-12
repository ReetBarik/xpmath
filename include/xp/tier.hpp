// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// ============================================================================
// xp/tier.hpp -- two tiers per backend: strict (default) and fast  (P3.0b)
// ============================================================================
// Policy: docs/PERF_ACCURACY_POLICY.md. Gate: scripts/p3_tier_gate.py.
//
// WHAT EXISTS
//   xp::strict, xp::fast            tag types
//   basic_dd<T> ... basic_qf<T>     alias templates, T = strict | fast
//   basic_dd_complex<T> ...         likewise for the complex types
//   dd_real        = basic_dd<strict>   (IS xp::DoubleDouble)
//   dd_real_fast   = basic_dd<fast>     (xp::DoubleDoubleFast, dd_fast.hpp)
//   dd_complex / dd_complex_fast        (and ff_, tf_, qf_)
//   xp::to_strict(x) / xp::to_fast(x)   the ONLY ways across the boundary.
//   XPMATH_HAS_TIERS                feature test; consumers that must still
//                                   build against an older xpmath guard their
//                                   fast code with #if defined(XPMATH_HAS_TIERS).
//
// basic_<b> are alias templates, so a tier cannot be DEDUCED from basic_dd<T>
// (write the tier explicitly). P4.0 may replace the aliases with real class
// templates; it must keep both tiers' bits.
//
// HOW FAST RELATES TO STRICT
//   The fast types have the same limbs, alignment and members as the strict
//   ones, but are distinct types, so mixing tiers (dd_real + dd_real_fast, passing
//   one where the other is expected) does not compile; ctest tier_mix_compile_fail
//   asserts that. Every public function and operator of the fast type is a
//   one-line forwarder to the strict function, so FAST == STRICT, bit for bit,
//   until a P3 task overrides something. Strict code is never edited to add a
//   tier.
//
// PATTERN 1 -- override ONE function in the fast tier only
//   In <b>_fast.hpp, add an implementation with a detail:: name that takes and
//   returns the STRICT type (so it can reuse strict helpers), then change the one
//   forwarding line of that function:
//
//       namespace detail {
//       XPMATH_INLINE_FUNCTION DoubleDouble dd_exp_fast_impl(DoubleDouble a) { ... }
//       }
//       // was: { return to_fast(exp(to_strict(a))); }
//       XPMATH_INLINE_FUNCTION DoubleDoubleFast exp(DoubleDoubleFast a) {
//           return to_fast(detail::dd_exp_fast_impl(to_strict(a)));
//       }
//
//   Strict does not change. "Promote to strict" (gate exit 0) means moving the
//   body into the strict function and restoring the forwarding line.
//
// PATTERN 2 -- chaining: let a fast override reach its callers
//   A function that was not overridden forwards entirely to strict, so a fast
//   multiply is NOT seen by strict exp/log. When the override should reach a
//   caller, convert that caller (and only the callers the task chooses) into ONE
//   shared body, a template on the number type, called by both public functions:
//
//       namespace detail {
//       template <class S> XPMATH_INLINE_FUNCTION S exp_impl(S a) {
//           ... multiply(a, b) ... add(...) ...   // resolved per type by overload
//       }
//       }
//       XPMATH_NOINLINE_FUNCTION DoubleDouble     exp(DoubleDouble a)     { return detail::exp_impl(a); }
//       XPMATH_INLINE_FUNCTION   DoubleDoubleFast exp(DoubleDoubleFast a) { return detail::exp_impl(a); }
//
//   Strict exp then calls strict multiply and fast exp calls fast multiply, with
//   no copied code and no public API change. Chaining is transitive (log calls
//   exp) and is decided per task; callers that are not converted keep using
//   strict. The strict instantiation must stay bit-identical: this is exactly
//   what CHECK-DEVICE-IDENTICAL proves. Keep every inlining, noinline and
//   td4_workaround decision of the strict function on the strict wrapper.
// ============================================================================
#pragma once

#define XPMATH_HAS_TIERS 1

namespace xp {

struct strict {};
struct fast {};

struct DoubleDouble;
struct DoubleDoubleComplex;
struct DoubleDoubleFast;
struct DoubleDoubleComplexFast;
struct FloatFloat;
struct FloatFloatComplex;
struct FloatFloatFast;
struct FloatFloatComplexFast;
struct TripleFloat;
struct TripleFloatComplex;
struct TripleFloatFast;
struct TripleFloatComplexFast;
struct QuadFloat;
struct QuadFloatComplex;
struct QuadFloatFast;
struct QuadFloatComplexFast;

namespace detail {
template <class Tier> struct dd_tier;
template <class Tier> struct ff_tier;
template <class Tier> struct tf_tier;
template <class Tier> struct qf_tier;

template <> struct dd_tier<strict> { using real = DoubleDouble; using complex = DoubleDoubleComplex; };
template <> struct dd_tier<fast>   { using real = DoubleDoubleFast; using complex = DoubleDoubleComplexFast; };
template <> struct ff_tier<strict> { using real = FloatFloat; using complex = FloatFloatComplex; };
template <> struct ff_tier<fast>   { using real = FloatFloatFast; using complex = FloatFloatComplexFast; };
template <> struct tf_tier<strict> { using real = TripleFloat; using complex = TripleFloatComplex; };
template <> struct tf_tier<fast>   { using real = TripleFloatFast; using complex = TripleFloatComplexFast; };
template <> struct qf_tier<strict> { using real = QuadFloat; using complex = QuadFloatComplex; };
template <> struct qf_tier<fast>   { using real = QuadFloatFast; using complex = QuadFloatComplexFast; };
} // namespace detail

template <class Tier> using basic_dd = typename detail::dd_tier<Tier>::real;
template <class Tier> using basic_ff = typename detail::ff_tier<Tier>::real;
template <class Tier> using basic_tf = typename detail::tf_tier<Tier>::real;
template <class Tier> using basic_qf = typename detail::qf_tier<Tier>::real;
template <class Tier> using basic_dd_complex = typename detail::dd_tier<Tier>::complex;
template <class Tier> using basic_ff_complex = typename detail::ff_tier<Tier>::complex;
template <class Tier> using basic_tf_complex = typename detail::tf_tier<Tier>::complex;
template <class Tier> using basic_qf_complex = typename detail::qf_tier<Tier>::complex;

using dd_real = basic_dd<strict>;
using ff_real = basic_ff<strict>;
using tf_real = basic_tf<strict>;
using qf_real = basic_qf<strict>;
using dd_real_fast = basic_dd<fast>;
using ff_real_fast = basic_ff<fast>;
using tf_real_fast = basic_tf<fast>;
using qf_real_fast = basic_qf<fast>;

using dd_complex = basic_dd_complex<strict>;
using ff_complex = basic_ff_complex<strict>;
using tf_complex = basic_tf_complex<strict>;
using qf_complex = basic_qf_complex<strict>;
using dd_complex_fast = basic_dd_complex<fast>;
using ff_complex_fast = basic_ff_complex<fast>;
using tf_complex_fast = basic_tf_complex<fast>;
using qf_complex_fast = basic_qf_complex<fast>;

} // namespace xp
