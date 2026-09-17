#pragma once
// ============================================================================
// corpus.hpp — corner-case input corpus for the extended-precision test suite
//              (T0.2)
// ============================================================================
//
// NO SPDX header: unlike third_party/include/dd_math.hpp (which is written for
// eventual upstreaming to Kokkos), this corpus is DOWNSTREAM-ONLY test
// scaffolding. It never leaves this repo, so it carries no Kokkos SPDX/copyright
// block — the same posture the plan mandates for demos, benchmarks, and the rest
// of the test harness (docs/TEST_SUITE_PLAN.md, "Upstreaming considerations").
//
// WHAT THIS IS: pure DATA plus a tiny API to iterate it. It is NOT tests. Every
//   test layer from Phase 1 onward runs its random-input pass AND a corpus pass;
//   this header supplies the corpus so the pathological inputs that uniform
//   random rarely hits (subnormals, ±inf, half-integer boundaries, the FF
//   splitter-overflow inputs) are always exercised. PORT_NOTES.md §4 (branch
//   fffunKokkos) documents two FF bugs that slipped through the demo's own
//   accuracy table precisely because uniform random almost never lands on them;
//   the named regression accessors below make those inputs impossible to miss.
//
// TWO API STYLES (ship both; pick per test):
//   (a) Bundlers  — unary<T>(flags) / binary<T>(flags): "throw the whole corpus
//       at this op". Used by the T*.2 invariant tests, which just need broad
//       coverage and don't care which category a failure came from.
//   (b) Named accessors — exp_overflow<T>(), nint_half_integer<T>(), etc.: grab
//       exactly the regression category an op needs. Used by the T*.4 accuracy
//       tests so a failure cites a specific PORT_NOTES bug ("exp_overflow item 3")
//       rather than "corpus item 47".
//
// RETURN SHAPE: std::vector<T> (unary) / std::vector<std::pair<T,T>> (binary),
//   NOT the InputDist generator-functor shape test_utils_host.hpp uses for random
//   sampling. Rationale: corpus entries are DETERMINISTIC CONSTANTS (a specific
//   subnormal, a specific ±inf, the literal 88.72 that broke FF exp), so a
//   materialized vector the caller iterates is the natural representation — one
//   element == one deterministic test input. Generator functors model random
//   draws; there is nothing random to draw here.
//
// PRECISION-PARAMETRIC: templated on the underlying scalar (double for DD today,
//   float for FF/QF-primitive later). Only double and float are supported; FF/QF
//   composite-type corpora are Phase 2/3 and out of scope here.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace kokkos_ep {
namespace corpus {

// ----------------------------------------------------------------------------
// Consumer-facing knobs. A test opts categories in/out; the bundlers honor these,
// the named accessors ignore them (a caller asking for exp_overflow wants it).
// ----------------------------------------------------------------------------
struct CorpusFlags {
  bool include_nan        = false;  // opt-in: only for ops whose semantics define NaN
  bool include_inf        = true;
  bool include_zero       = true;
  bool include_subnormals = true;
};

// ----------------------------------------------------------------------------
// Small precision-parametric helpers (double / float via <cmath> overloads).
// ----------------------------------------------------------------------------
namespace detail {

template <typename T> inline T next_up(T x)   { return std::nextafter(x, std::numeric_limits<T>::infinity()); }
template <typename T> inline T next_down(T x) { return std::nextafter(x, -std::numeric_limits<T>::infinity()); }

// High-precision math constants, narrowed to T at instantiation.
template <typename T> inline T pi_v()   { return static_cast<T>(3.141592653589793238462643383279502884L); }
template <typename T> inline T e_v()    { return static_cast<T>(2.718281828459045235360287471352662498L); }
template <typename T> inline T twopi()  { return static_cast<T>(2) * pi_v<T>(); }
template <typename T> inline T halfpi() { return pi_v<T>() / static_cast<T>(2); }

// Append b onto a (small helper; keeps accessors terse).
template <typename T> inline void append(std::vector<T>& a, std::initializer_list<T> b) {
  a.insert(a.end(), b.begin(), b.end());
}

}  // namespace detail

// ============================================================================
// Category accessors — each returns one corner-case family as a vector<T>.
// ============================================================================

// ±0. Numerically -0 == 0 but the bit patterns differ; both are included so an
// op's sign-of-zero handling gets exercised.
template <typename T>
std::vector<T> zeros() {
  return { static_cast<T>(0.0), -static_cast<T>(0.0) };
}

// ±inf.
template <typename T>
std::vector<T> infinities() {
  const T inf = std::numeric_limits<T>::infinity();
  return { inf, -inf };
}

// Quiet NaN. Opt-in only (see CorpusFlags::include_nan).
template <typename T>
std::vector<T> nans() {
  return { std::numeric_limits<T>::quiet_NaN() };
}

// Subnormals: smallest denormal, a mid-range denormal, and the largest
// subnormal (just below the smallest normal). For FP32 this spans ~1.4e-45 to
// ~1.18e-38; for FP64 ~4.9e-324 to ~2.2e-308.
template <typename T>
std::vector<T> subnormals() {
  const T dmin = std::numeric_limits<T>::denorm_min();     // smallest positive subnormal
  const T minN = std::numeric_limits<T>::min();            // smallest positive normal
  const T largest_sub = detail::next_down(minN);           // largest subnormal
  // Mid subnormal: push denorm_min up by half the mantissa width so it sits well
  // inside the subnormal range without reaching the normal boundary.
  const int half_mant = std::numeric_limits<T>::digits / 2;
  const T mid_sub = std::ldexp(dmin, half_mant);
  return { dmin, -dmin, mid_sub, -mid_sub, largest_sub, -largest_sub };
}

// Powers of two across the exponent range (exact in binary FP, no rounding).
// FP32: ~2^-126 .. 2^127; FP64: much wider. We sample a spread, not every power.
template <typename T>
std::vector<T> powers_of_two() {
  std::vector<T> v;
  const int lo = std::numeric_limits<T>::min_exponent + 1;   // stay in normal range
  const int hi = std::numeric_limits<T>::max_exponent - 1;
  // Sample endpoints, a stride through the middle, and the unit-scale neighbors.
  for (int e : { lo, lo + 3, -24, -10, -1, 0, 1, 10, 24, hi - 3, hi }) {
    if (e >= lo && e <= hi) {
      const T p = std::ldexp(static_cast<T>(1), e);
      v.push_back(p);
      v.push_back(-p);
    }
  }
  return v;
}

// nextafter neighbors: for a set of anchors, include x itself, one and two ulps
// toward +inf, and one and two ulps toward -inf.
template <typename T>
std::vector<T> nextafter_neighbors() {
  std::vector<T> v;
  const T anchors[] = {
    static_cast<T>(0), static_cast<T>(1), detail::pi_v<T>(), detail::e_v<T>(),
    static_cast<T>(1e6), static_cast<T>(1e-6)
  };
  for (T x : anchors) {
    const T up1 = detail::next_up(x);
    const T up2 = detail::next_up(up1);
    const T dn1 = detail::next_down(x);
    const T dn2 = detail::next_down(dn1);
    detail::append(v, { x, up1, up2, dn1, dn2 });
  }
  return v;
}

// A grab-bag of "interesting" finite values other categories don't cover:
// ±1, small integers, the max/lowest finite magnitudes, and the two constants.
template <typename T>
std::vector<T> finite_specials() {
  const T maxT = std::numeric_limits<T>::max();
  std::vector<T> v = {
    static_cast<T>(1), static_cast<T>(-1),
    static_cast<T>(2), static_cast<T>(-2),
    static_cast<T>(0.5), static_cast<T>(-0.5),
    detail::pi_v<T>(), -detail::pi_v<T>(),
    detail::e_v<T>(),  -detail::e_v<T>(),
    maxT, -maxT,
    std::numeric_limits<T>::min(), -std::numeric_limits<T>::min(),
  };
  return v;
}

// ============================================================================
// Binary category accessors — vector<pair<T,T>>.
// ============================================================================

// Near-cancellation pairs (a, a*(1+eps)) for a spread of eps from a few ulps
// down toward the precision floor. Subtracting these loses leading digits — the
// PORT_NOTES §5 "conditioning limit" family (sub/fdim/fma).
template <typename T>
std::vector<std::pair<T, T>> near_cancellation() {
  std::vector<std::pair<T, T>> v;
  const T bases[] = { static_cast<T>(1), detail::pi_v<T>(), static_cast<T>(1e6),
                      static_cast<T>(-3.5), static_cast<T>(1e-3) };
  // eps spread: 2 ulp, 8 ulp, and a descent of decade-ish magnitudes bottoming
  // out near the type's epsilon (float ~1e-7, double ~2e-16).
  const T u   = std::numeric_limits<T>::epsilon();
  const T eps[] = { static_cast<T>(2) * u, static_cast<T>(8) * u,
                    static_cast<T>(1e-4), static_cast<T>(1e-8),
                    static_cast<T>(1e-12), static_cast<T>(1e-15) };
  for (T a : bases)
    for (T e : eps)
      v.emplace_back(a, a * (static_cast<T>(1) + e));
  return v;
}

// Huge/tiny magnitude mixes: (a, b) with |a| >> |b| and the swap, spanning
// ratios 1e6 .. 1e30. Stresses splitter overflow and alignment-loss paths.
template <typename T>
std::vector<std::pair<T, T>> huge_tiny() {
  std::vector<std::pair<T, T>> v;
  const T base = static_cast<T>(1.5);
  const double ratios[] = { 1e6, 1e12, 1e18, 1e24, 1e30 };
  for (double r : ratios) {
    const T big   = static_cast<T>(base * static_cast<T>(r));
    const T small = static_cast<T>(base / static_cast<T>(r));
    // Guard against overflow/underflow to inf/0 at FP32 for the wide ratios.
    if (std::isfinite(big) && small != static_cast<T>(0)) {
      v.emplace_back(big, small);
      v.emplace_back(small, big);
    }
  }
  return v;
}

// ============================================================================
// Explicit PORT_NOTES §4 / §3 regression accessors. These appear verbatim as
// named entries so a test can cite the exact bug. Source of truth:
// PORT_NOTES.md on branch fffunKokkos (§3 precision fixes, §4 outright bugs).
// ============================================================================

// PORT_NOTES §4a: FF exp returned NaN for a > 79.4 because the Dekker splitter
// overflowed at nz >= 116 (b * 8193.0f -> inf, inf - inf -> NaN). These inputs
// must NOT produce NaN.
template <typename T> std::vector<T> exp_overflow() {
  return { static_cast<T>(79.5), static_cast<T>(80.0), static_cast<T>(85.0),
           static_cast<T>(88.7), static_cast<T>(88.72) };
}

// PORT_NOTES §4b: FF ffnint(19.4999993...) returned 20 instead of 19 because the
// 2^47 magic-constant trick fails at FP32's 24-bit mantissa. Cover k+0.5 minus
// a hair (should round DOWN to k) and the nextafter neighbors of k+0.5 for a
// range of k. This family also feeds floor/ceil/round/trunc/fmod and sincos arg
// reduction, all of which call nint.
template <typename T> std::vector<T> nint_half_integer() {
  std::vector<T> v;
  // The literal offender from PORT_NOTES, and its float-ulp-below-19.5 form.
  detail::append(v, { static_cast<T>(19.4999993),
                      detail::next_down(static_cast<T>(19.5)) });
  const long k_vals[] = { 0, 1, 2, 10, 100, 1000, 19 };
  for (long k : k_vals) {
    const T half = static_cast<T>(k) + static_cast<T>(0.5);
    detail::append(v, {
        half,
        detail::next_down(half),   // just below k+0.5 -> should round to k
        detail::next_up(half),     // just above k+0.5 -> should round to k+1
        -half,
        detail::next_down(-half),
        detail::next_up(-half),
    });
  }
  return v;
}

// PORT_NOTES §4b: remainder(68.379..., 3.5066...) returned -1.7533 instead of
// +1.7533 (downstream of the ffnint off-by-one). The exact bit patterns are not
// quoted verbatim in PORT_NOTES; these are the documented approximate inputs
// plus a couple of neighbors that also sit near a multiple of the divisor.
template <typename T> std::vector<std::pair<T, T>> remainder_regression() {
  std::vector<std::pair<T, T>> v;
  v.emplace_back(static_cast<T>(68.379), static_cast<T>(3.5066));
  // Neighbors on the dividend side to bracket the nint(a/b) boundary.
  v.emplace_back(detail::next_up(static_cast<T>(68.379)), static_cast<T>(3.5066));
  v.emplace_back(detail::next_down(static_cast<T>(68.379)), static_cast<T>(3.5066));
  return v;
}

// PORT_NOTES §3c: atanh via log loses precision for small |a| (log evaluated
// near 1). The Taylor branch (|a| < 0.5) is what these stress.
template <typename T> std::vector<T> atanh_small() {
  return { static_cast<T>(0.0), static_cast<T>(1e-6), static_cast<T>(1e-3),
           static_cast<T>(0.1), static_cast<T>(0.25), static_cast<T>(0.49),
           static_cast<T>(-1e-3), static_cast<T>(-0.25), static_cast<T>(-0.49) };
}

// PORT_NOTES §3b: sinh/cosh cancel for small |a| ((e^a - e^-a)/2). The Taylor
// branch (|a| < 0.5) is what these stress.
template <typename T> std::vector<T> sinh_cosh_small() {
  return { static_cast<T>(0.0), static_cast<T>(1e-6), static_cast<T>(1e-3),
           static_cast<T>(0.1), static_cast<T>(0.3), static_cast<T>(0.49),
           static_cast<T>(-1e-3), static_cast<T>(-0.3), static_cast<T>(-0.49) };
}

// PORT_NOTES §3a / §5: sin/cos/tan near multiples of π need joint sin/cos
// doublings; near ±π they're conditioning-limited. Cover ±π, ±2π, ±3π, ±π/2 and
// nextafter neighbors so the argument-reduction paths are hit.
template <typename T> std::vector<T> trig_near_pi() {
  std::vector<T> v;
  const T pi = detail::pi_v<T>();
  const T anchors[] = { pi, -pi, detail::twopi<T>(), -detail::twopi<T>(),
                        static_cast<T>(3) * pi, static_cast<T>(-3) * pi,
                        detail::halfpi<T>(), -detail::halfpi<T>() };
  for (T x : anchors) {
    detail::append(v, { x, detail::next_up(x), detail::next_down(x) });
  }
  return v;
}

// ============================================================================
// Bundlers — "throw the whole corpus at this op". Honor CorpusFlags.
// ============================================================================

namespace detail {

// Drop, in-place, any element whose class the flags turned off. Applied to the
// assembled bundle so incidental members (a zero produced by nextafter(0,+inf),
// a subnormal produced by the same) obey the flags too.
template <typename T>
inline void filter_by_flags(std::vector<T>& v, const CorpusFlags& f) {
  auto drop = [&f](T x) {
    if (!f.include_nan        && std::isnan(x))    return true;
    if (!f.include_inf        && std::isinf(x))    return true;
    if (!f.include_zero       && x == static_cast<T>(0)) return true;
    if (!f.include_subnormals && x != static_cast<T>(0) && std::isfinite(x) &&
        std::fabs(x) < std::numeric_limits<T>::min())    return true;
    return false;
  };
  v.erase(std::remove_if(v.begin(), v.end(), drop), v.end());
}

// Pair variant: drop a pair if either component's class is turned off.
template <typename T>
inline void filter_by_flags(std::vector<std::pair<T, T>>& v, const CorpusFlags& f) {
  auto drop_scalar = [&f](T x) {
    if (!f.include_nan        && std::isnan(x))    return true;
    if (!f.include_inf        && std::isinf(x))    return true;
    if (!f.include_zero       && x == static_cast<T>(0)) return true;
    if (!f.include_subnormals && x != static_cast<T>(0) && std::isfinite(x) &&
        std::fabs(x) < std::numeric_limits<T>::min())    return true;
    return false;
  };
  auto drop = [&drop_scalar](const std::pair<T, T>& p) {
    return drop_scalar(p.first) || drop_scalar(p.second);
  };
  v.erase(std::remove_if(v.begin(), v.end(), drop), v.end());
}

}  // namespace detail

// Full unary corpus: every finite category always, plus zero/inf/subnormal/NaN
// families gated by flags. Includes the PORT_NOTES §3/§4 unary regression
// families so a bundler-only test still exercises them.
template <typename T>
std::vector<T> unary(CorpusFlags flags = {}) {
  std::vector<T> v;
  auto add = [&v](const std::vector<T>& s) { v.insert(v.end(), s.begin(), s.end()); };

  add(finite_specials<T>());
  add(powers_of_two<T>());
  add(nextafter_neighbors<T>());

  // PORT_NOTES regression families (unary).
  add(exp_overflow<T>());
  add(nint_half_integer<T>());
  add(atanh_small<T>());
  add(sinh_cosh_small<T>());
  add(trig_near_pi<T>());

  if (flags.include_zero)       add(zeros<T>());
  if (flags.include_inf)        add(infinities<T>());
  if (flags.include_subnormals) add(subnormals<T>());
  if (flags.include_nan)        add(nans<T>());

  // Flags are authoritative over the WHOLE bundle, not just the dedicated
  // families: several always-on categories emit these classes incidentally
  // (e.g. nextafter around 0 yields both a literal 0 and denorm_min). Filter so
  // a caller that turned a class off never sees it, regardless of source.
  detail::filter_by_flags(v, flags);
  return v;
}

// Full binary corpus: near-cancellation and huge/tiny mixes always, plus the
// remainder regression pair. Zero/inf/subnormal/NaN are folded in as (special,
// special) and (special, 1) pairs when their flags are set, so binary ops see
// e.g. (inf, 1), (0, 0), (nan, 1).
template <typename T>
std::vector<std::pair<T, T>> binary(CorpusFlags flags = {}) {
  std::vector<std::pair<T, T>> v;
  auto addp = [&v](const std::vector<std::pair<T, T>>& s) { v.insert(v.end(), s.begin(), s.end()); };

  addp(near_cancellation<T>());
  addp(huge_tiny<T>());
  addp(remainder_regression<T>());

  // Cross specials against 1 and against themselves, gated by flags.
  const T one = static_cast<T>(1);
  auto cross = [&v, one](const std::vector<T>& specials) {
    for (T s : specials) {
      v.emplace_back(s, one);
      v.emplace_back(one, s);
      v.emplace_back(s, s);
    }
  };
  if (flags.include_zero)       cross(zeros<T>());
  if (flags.include_inf)        cross(infinities<T>());
  if (flags.include_subnormals) cross(subnormals<T>());
  if (flags.include_nan)        cross(nans<T>());

  // Authoritative filter over the whole bundle (see unary()): huge_tiny and
  // near_cancellation can emit an incidental subnormal/zero component.
  detail::filter_by_flags(v, flags);
  return v;
}

}  // namespace corpus
}  // namespace kokkos_ep
