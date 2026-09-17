// ============================================================================
// scripts/sweep_ops.hpp — the op inventory and the two evaluators, DEVICE-SAFE
// ============================================================================
// Added by CORE_PLAN section C6 (chunk A).
//
// WHAT THIS IS
// The 39 real and 24 complex operations the sweep measures, and the two
// switches that evaluate them on an xp:: backend. Nothing else. It exists so
// that the DEVICE producer (scripts/sweep_device.cpp) and the HOST scorer
// (scripts/sweep_accuracy.cpp) can be made to agree by construction instead of
// by two people keeping two switch statements in step.
//
// WHAT IT IS NOT
// It is NOT a scorer. There is no ulp arithmetic here, no tolerance, no bound
// and no oracle — docs/CORRECTNESS.md permits exactly one verdict per point and
// scripts/sweep_accuracy.cpp issues it. This header only says what to compute.
//
// DEVICE SAFETY IS THE WHOLE POINT, so this header must never grow:
//   * __float128 or any MPFR/MPC type   (nvcc rejects the first in a device
//                                        pass; the second is the oracle's)
//   * <vector>, <string> or <random>    (host input derivation lives in
//                                        scripts/sweep_inputs.hpp)
//   * anything that is not reachable from an XPMATH_INLINE_FUNCTION
// The name tables below ARE host-side data — they are read when a row is
// formatted and when an operand stream is derived, never inside a functor.
//
// PROVENANCE, AND THE DUPLICATE THAT IS STILL OPEN
// Every declaration below is transcribed VERBATIM from
// scripts/sweep_accuracy.cpp as of `1056341`:
//     enum R / kReal[]          — its "Op inventory" block
//     enum C / kComplex[]       — the same block
//     eval_real<S>              — its `template <class S> S eval_real(...)`
//     eval_complex<S, Z>        — its `template <class S, class Z> Z eval_complex(...)`
// The only edits are the namespace, the XPMATH_INLINE_FUNCTION annotations that
// make the two evaluators callable from a device functor, and `inline` on the
// two tables so they can live in a header.
//
// scripts/sweep_accuracy.cpp STILL CARRIES ITS OWN COPIES. C6 chunk A is not
// permitted to modify that file, so the de-duplication — deleting those four
// blocks and including this header instead — belongs to chunk B, which owns it.
// Until that happens the two are a MEASURED match, not a structural one: chunk
// A cross-checked the derived operand streams point for point against
// `sweep_accuracy --dump-operands`. If you are chunk B, do the deletion; if you
// are editing either copy for any other reason, edit both and re-run that
// cross-check.
// ============================================================================

#pragma once

// All four backends, real and complex. The *_complex.hpp headers pull in their
// *_math.hpp siblings, which is the same set scripts/sweep_accuracy.cpp
// includes. They are needed at DEFINITION time, not merely at instantiation:
// the evaluators below name xp::sqrt and friends through a qualified-id in a
// non-dependent namespace, so the names must already exist when the template is
// parsed. (An earlier draft included only <xp/config.hpp> and got 63 "'sqrt' is
// not a member of 'xp'" errors for exactly that reason.)
#include <xp/dd_complex.hpp>
#include <xp/ff_complex.hpp>
#include <xp/qf_complex.hpp>
#include <xp/tf_complex.hpp>

namespace xpsweep {

// ---------------------------------------------------------------------------
// Op inventory. Identical order and naming to scripts/sweep_accuracy.cpp — a
// device result row and a baseline row must refer to the same operation, and
// `point` is only an index into a shared grid, so the ORDER is load-bearing.
// ---------------------------------------------------------------------------
enum R {
  R_Add, R_Sub, R_Mul, R_Div,
  R_Sqrt, R_Abs, R_Exp, R_Log, R_Exp2, R_Exp10, R_Expm1, R_Log2, R_Log10, R_Log1p,
  R_Sin, R_Cos, R_Tan, R_Asin, R_Acos, R_Atan,
  R_Sinh, R_Cosh, R_Tanh, R_Acosh, R_Asinh, R_Atanh,
  R_Pow, R_Hypot, R_Fmod, R_Remainder, R_Copysign, R_Fmax, R_Fmin, R_Fdim,
  R_Fma,
  R_Ceil, R_Floor, R_Round, R_Trunc,
  R_COUNT
};

struct RealSpec { const char* name; int nops; int b_lo, b_hi; };

// clang-format off
inline const RealSpec kReal[R_COUNT] = {
  { "add",       2, -100, 100 }, { "sub",       2, -100, 100 },
  { "mul",       2,  -50,  50 }, { "div",       2,  -50,  50 },
  { "sqrt",      1,    0,   0 }, { "abs",       1,    0,   0 },
  { "exp",       1,    0,   0 }, { "log",       1,    0,   0 },
  { "exp2",      1,    0,   0 }, { "exp10",     1,    0,   0 },
  { "expm1",     1,    0,   0 }, { "log2",      1,    0,   0 },
  { "log10",     1,    0,   0 }, { "log1p",     1,    0,   0 },
  { "sin",       1,    0,   0 }, { "cos",       1,    0,   0 },
  { "tan",       1,    0,   0 }, { "asin",      1,    0,   0 },
  { "acos",      1,    0,   0 }, { "atan",      1,    0,   0 },
  { "sinh",      1,    0,   0 }, { "cosh",      1,    0,   0 },
  { "tanh",      1,    0,   0 }, { "acosh",     1,    0,   0 },
  { "asinh",     1,    0,   0 }, { "atanh",     1,    0,   0 },
  { "pow",       2,  -10,   6 }, { "hypot",     2, -100, 100 },
  { "fmod",      2, -100, 100 }, { "remainder", 2, -100, 100 },
  { "copysign",  2, -100, 100 }, { "fmax",      2, -100, 100 },
  { "fmin",      2, -100, 100 }, { "fdim",      2, -100, 100 },
  { "fma",       3,  -50,  50 },
  { "ceil",      1,    0,   0 }, { "floor",     1,    0,   0 },
  { "round",     1,    0,   0 }, { "trunc",     1,    0,   0 },
};
// clang-format on

enum C {
  C_Add, C_Sub, C_Mul, C_Div,
  C_Abs, C_Conj, C_Sqrt, C_Exp, C_Log, C_Log10,
  C_Sin, C_Cos, C_Tan, C_Asin, C_Acos, C_Atan,
  C_Sinh, C_Cosh, C_Tanh, C_Asinh, C_Acosh, C_Atanh,
  C_Pow, C_Polar,
  C_COUNT
};

struct ComplexSpec { const char* name; int nops; };

// clang-format off
inline const ComplexSpec kComplex[C_COUNT] = {
  {"add", 2}, {"sub", 2}, {"mul", 2}, {"div", 2},
  {"abs", 1}, {"conj", 1}, {"sqrt", 1}, {"exp", 1}, {"log", 1}, {"log10", 1},
  {"sin", 1}, {"cos", 1}, {"tan", 1}, {"asin", 1}, {"acos", 1}, {"atan", 1},
  {"sinh", 1}, {"cosh", 1}, {"tanh", 1}, {"asinh", 1}, {"acosh", 1}, {"atanh", 1},
  {"pow", 2}, {"polar", 1},
};
// clang-format on

// ---------------------------------------------------------------------------
// The evaluators.
// ---------------------------------------------------------------------------
// `id` is a RUNTIME parameter here, exactly as it is in sweep_accuracy.cpp, so
// that the expression evaluated for a given op is textually the same in both
// producers. scripts/sweep_device.cpp nevertheless calls these with a
// COMPILE-TIME constant, which folds the switch to its single live case and
// keeps each emitted device function the size of one operation rather than of
// all thirty-nine. That matters on gfx90a: TD-1 (docs/ROCM_BRANCH_RELAXATION_BUG.md)
// is a defect in an OVERSIZED device callee, and a 39-case switch inlined into
// one kernel is precisely that shape.

template <class S>
XPMATH_INLINE_FUNCTION S eval_real(int id, const S& a, const S& b, const S& c) {
  switch (id) {
    case R_Add:       return a + b;
    case R_Sub:       return a - b;
    case R_Mul:       return a * b;
    case R_Div:       return a / b;
    case R_Sqrt:      return xp::sqrt(a);
    case R_Abs:       return xp::abs(a);
    case R_Exp:       return xp::exp(a);
    case R_Log:       return xp::log(a);
    case R_Exp2:      return xp::exp2(a);
    case R_Exp10:     return xp::exp10(a);
    case R_Expm1:     return xp::expm1(a);
    case R_Log2:      return xp::log2(a);
    case R_Log10:     return xp::log10(a);
    case R_Log1p:     return xp::log1p(a);
    case R_Sin:       return xp::sin(a);
    case R_Cos:       return xp::cos(a);
    case R_Tan:       return xp::tan(a);
    case R_Asin:      return xp::asin(a);
    case R_Acos:      return xp::acos(a);
    case R_Atan:      return xp::atan(a);
    case R_Sinh:      return xp::sinh(a);
    case R_Cosh:      return xp::cosh(a);
    case R_Tanh:      return xp::tanh(a);
    case R_Acosh:     return xp::acosh(a);
    case R_Asinh:     return xp::asinh(a);
    case R_Atanh:     return xp::atanh(a);
    case R_Pow:       return xp::pow(a, b);
    case R_Hypot:     return xp::hypot(a, b);
    case R_Fmod:      return xp::fmod(a, b);
    case R_Remainder: return xp::remainder(a, b);
    case R_Copysign:  return xp::copysign(a, b);
    case R_Fmax:      return xp::fmax(a, b);
    case R_Fmin:      return xp::fmin(a, b);
    case R_Fdim:      return xp::fdim(a, b);
    case R_Fma:       return xp::fma(a, b, c);
    case R_Ceil:      return xp::ceil(a);
    case R_Floor:     return xp::floor(a);
    case R_Round:     return xp::round(a);
    case R_Trunc:     return xp::trunc(a);
  }
  return a;
}

// `is_real` reports that the op returns a real scalar placed in the real slot
// (complex abs), so the caller knows the imaginary component is a true zero and
// not a discarded one. The device producer emits it regardless — a raw result
// file records what the library returned, and interpreting it is the scorer's
// job — but the flag is kept so the two evaluators stay textually identical.
template <class S, class Z>
XPMATH_INLINE_FUNCTION Z eval_complex(int id, const Z& a, const Z& b, bool& is_real) {
  is_real = false;
  switch (id) {
    case C_Add:   return a + b;
    case C_Sub:   return a - b;
    case C_Mul:   return a * b;
    case C_Div:   return a / b;
    case C_Abs:   is_real = true; return Z(xp::abs(a), S(0.0));
    case C_Conj:  return xp::conj(a);
    case C_Sqrt:  return xp::sqrt(a);
    case C_Exp:   return xp::exp(a);
    case C_Log:   return xp::log(a);
    case C_Log10: return xp::log10(a);
    case C_Sin:   return xp::sin(a);
    case C_Cos:   return xp::cos(a);
    case C_Tan:   return xp::tan(a);
    case C_Asin:  return xp::asin(a);
    case C_Acos:  return xp::acos(a);
    case C_Atan:  return xp::atan(a);
    case C_Sinh:  return xp::sinh(a);
    case C_Cosh:  return xp::cosh(a);
    case C_Tanh:  return xp::tanh(a);
    case C_Asinh: return xp::asinh(a);
    case C_Acosh: return xp::acosh(a);
    case C_Atanh: return xp::atanh(a);
    case C_Pow:   return xp::pow(a, b);
    // polar(r, theta): the real slot of operand a carries r, the imaginary slot
    // theta — the convention sweep_accuracy.cpp and the complex demos both use.
    case C_Polar: return xp::polar(a.re, a.im);
  }
  return a;
}

}  // namespace xpsweep
