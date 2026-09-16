// ============================================================================
// corpus_test.cpp — smoke test for the corner-case corpus (T0.2).
// ============================================================================
//
// This is corpus-SCAFFOLDING validation, mirroring hello_test.cpp's role for the
// harness: it checks that corpus.hpp loads, the categories/flags behave, and the
// PORT_NOTES §4 regression accessors contain the specific called-out values. It
// does NOT run any DD math op — DD correctness against these inputs begins in
// Phase 1 (T1.x). No __float128 oracle is touched: corpus data itself needs no
// oracle. (This used to be phrased as "needs no KOKKOS_EP_HAVE_QUADMATH guard,
// unlike hello_test" — that guard is gone from every test, see test_utils.hpp.)
// ============================================================================

#include "test_utils.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using namespace kokkos_ep;
namespace cp = kokkos_ep::corpus;

// Does vector v contain an element for which pred(x) is true?
template <typename T, typename Pred>
static bool any_of(const std::vector<T>& v, Pred pred) {
  for (const T& x : v) if (pred(x)) return true;
  return false;
}

// Approximate presence check for a scalar value (corpus entries are exact
// constants, but constants like 88.72 narrow differently per T, so compare with
// a small relative tolerance).
template <typename T>
static bool contains_approx(const std::vector<T>& v, double target, double tol = 1e-4) {
  return any_of(v, [target, tol](T x) {
    return std::isfinite((double)x) &&
           std::fabs((double)x - target) <= tol * (1.0 + std::fabs(target));
  });
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;
  // No Kokkos::initialize: this test only inspects host-side corpus vectors.

  // -- Bundler flag behavior (double) ----------------------------------------
  {
    cp::CorpusFlags on;    // include_inf/zero/subnormals default true; nan false
    auto u = cp::unary<double>(on);
    KOKKOS_EP_ASSERT(!u.empty(), "unary<double>() should be non-empty");

    KOKKOS_EP_ASSERT(any_of(u, [](double x){ return std::isinf(x); }),
                     "include_inf=true should yield +/-inf");
    KOKKOS_EP_ASSERT(any_of(u, [](double x){ return x == 0.0; }),
                     "include_zero=true should yield zero");
    KOKKOS_EP_ASSERT(any_of(u, [](double x){ return x != 0.0 && std::fabs(x) < std::numeric_limits<double>::min(); }),
                     "include_subnormals=true should yield a subnormal");
    KOKKOS_EP_ASSERT(!any_of(u, [](double x){ return std::isnan(x); }),
                     "include_nan=false should yield no NaN");

    cp::CorpusFlags off;
    off.include_inf = false;
    off.include_zero = false;
    off.include_subnormals = false;
    off.include_nan = false;
    auto u2 = cp::unary<double>(off);
    KOKKOS_EP_ASSERT(!any_of(u2, [](double x){ return std::isinf(x); }),
                     "include_inf=false should yield no inf");
    KOKKOS_EP_ASSERT(!any_of(u2, [](double x){ return x == 0.0; }),
                     "include_zero=false should yield no zero");
    KOKKOS_EP_ASSERT(!any_of(u2, [](double x){ return x != 0.0 && std::fabs(x) < std::numeric_limits<double>::min(); }),
                     "include_subnormals=false should yield no subnormal");

    cp::CorpusFlags with_nan;
    with_nan.include_nan = true;
    auto u3 = cp::unary<double>(with_nan);
    KOKKOS_EP_ASSERT(any_of(u3, [](double x){ return std::isnan(x); }),
                     "include_nan=true should yield a NaN");
  }

  // -- Bundler works for float too (precision-parametric) --------------------
  {
    auto uf = cp::unary<float>();
    KOKKOS_EP_ASSERT(!uf.empty(), "unary<float>() should be non-empty");
    auto bf = cp::binary<float>();
    KOKKOS_EP_ASSERT(!bf.empty(), "binary<float>() should be non-empty");
  }

  // -- Binary bundler --------------------------------------------------------
  {
    auto b = cp::binary<double>();
    KOKKOS_EP_ASSERT(!b.empty(), "binary<double>() should be non-empty");
  }

  // -- PORT_NOTES §4 named regression accessors non-empty + spot-check values-
  {
    auto eo = cp::exp_overflow<double>();
    KOKKOS_EP_ASSERT(!eo.empty(), "exp_overflow<double>() should be non-empty");
    // §4a: the 88.72 input that made FF exp return NaN must be present.
    KOKKOS_EP_ASSERT(contains_approx(eo, 88.72),
                     "exp_overflow should contain the 88.72 splitter-overflow input");
    KOKKOS_EP_ASSERT(any_of(eo, [](double x){ return x > 79.4; }),
                     "exp_overflow should contain inputs > 79.4 (PORT_NOTES §4a)");

    auto nh = cp::nint_half_integer<double>();
    KOKKOS_EP_ASSERT(!nh.empty(), "nint_half_integer<double>() should be non-empty");
    // §4b: the 19.4999993 offender that rounded to 20 instead of 19.
    KOKKOS_EP_ASSERT(contains_approx(nh, 19.4999993, 1e-3),
                     "nint_half_integer should contain the ~19.4999993 offender (PORT_NOTES §4b)");

    auto rr = cp::remainder_regression<double>();
    KOKKOS_EP_ASSERT(!rr.empty(), "remainder_regression<double>() should be non-empty");
    // §4b: remainder(68.379..., 3.5066...).
    KOKKOS_EP_ASSERT(any_of(rr, [](std::pair<double,double> p){
                        return std::fabs(p.first - 68.379) < 1e-2 &&
                               std::fabs(p.second - 3.5066) < 1e-2; }),
                     "remainder_regression should contain the (68.379, 3.5066) pair (PORT_NOTES §4b)");

    auto as = cp::atanh_small<double>();
    KOKKOS_EP_ASSERT(!as.empty(), "atanh_small<double>() should be non-empty");
    KOKKOS_EP_ASSERT(any_of(as, [](double x){ return std::fabs(x) < 0.5; }),
                     "atanh_small should contain |a| < 0.5 (PORT_NOTES §3c)");

    auto sc = cp::sinh_cosh_small<double>();
    KOKKOS_EP_ASSERT(!sc.empty(), "sinh_cosh_small<double>() should be non-empty");
    KOKKOS_EP_ASSERT(any_of(sc, [](double x){ return std::fabs(x) < 0.5; }),
                     "sinh_cosh_small should contain |a| < 0.5 (PORT_NOTES §3b)");

    auto tp = cp::trig_near_pi<double>();
    KOKKOS_EP_ASSERT(!tp.empty(), "trig_near_pi<double>() should be non-empty");
    KOKKOS_EP_ASSERT(contains_approx(tp, 3.14159265358979, 1e-6),
                     "trig_near_pi should contain a value near +pi (PORT_NOTES §3a)");
  }

  // -- expected-min-drop registry (PORT_NOTES §5) ----------------------------
  {
    // §5 conditioning-limited ops -> non-null.
    for (const char* op : {"sub", "fdim", "fma", "asin", "acos", "atanh", "remainder"}) {
      KOKKOS_EP_ASSERT(lookup_expected_min_drop(op) != nullptr,
                       "conditioning-limited op should be in the expected-min-drop registry");
    }
    // add is NOT conditioning-limited -> null.
    KOKKOS_EP_ASSERT(lookup_expected_min_drop("add") == nullptr,
                     "add is not conditioning-limited; should NOT be in the registry");
    // A registry hit should carry a reason string.
    const auto* sub_ann = lookup_expected_min_drop("sub");
    KOKKOS_EP_ASSERT(sub_ann != nullptr && sub_ann->reason != nullptr,
                     "expected-min-drop entry should carry a reason");
  }

  std::printf("corpus_test: all corpus-scaffolding checks passed\n");
  return ep_exit_code();
}
