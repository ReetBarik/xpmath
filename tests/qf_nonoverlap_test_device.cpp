// ============================================================================
// qf_nonoverlap_test_device.cpp — the DEVICE half of T3.2's non-overlap tests.
// ============================================================================
// Added by CORE_PLAN section C4 step 2 (chunk D), mirroring chunk C's
// dd_property_test_device.cpp / ff_property_test_device.cpp.
//
// WHY THE SPLIT
//   tests/qf_nonoverlap_test.cpp ran Test A (every QF op, host), Test B (a
//   five-op device tripwire through Kokkos::parallel_for) and Test C (named
//   corner cases, host) in one TU. The invariant it checks —
//   |f_{i+1}| <= 1/2 ulp(f_i) on a length-4 expansion — is ORACLE-INDEPENDENT:
//   it is arithmetic on the four FP32 words themselves and needs no reference
//   value at all. The 128-bit type entered that file through exactly one door,
//   make_wide_input(), which ENRICHES a nominal double into a full-width 4-word
//   operand. One input-construction helper made the whole TU, device tripwire
//   included, unbuildable under nvcc (S6: nvcc gives every TU a device pass and
//   rejects std::vector<__float128> in it).
//
// WHAT REPLACES make_wide_input, AND WHY IT IS NOT A WEAKER INPUT
//   The enrichment adds tail terms at relative 2^-28 / 2^-56 / 2^-84 and then
//   decomposes the wide value into four ordered FP32 words by successive
//   round-to-nearest. It needs a CARRIER with ~96+ bits of significand; it does
//   not need binary128 specifically. This file carries it in xp::DoubleDouble
//   (2 x FP64, 106 bits) instead — the library's own extended type, which
//   compiles for a device by construction.
//
//   That is a carrier swap, not a weakening of the test. What the construction
//   has to deliver is a NORMALIZED, ORDERED, FOUR-WORD operand, so that the op
//   under test is exercised rather than renorm (T3.1's job). 106 bits delivers
//   that with room to spare: after w0/w1/w2 are peeled off, the residual still
//   holds ~35 bits, so w3 is nonzero and the operand is genuinely full width.
//   The construction is not claiming to represent any particular target value
//   to the last bit — no assertion anywhere reads the input's exact value — so
//   the two bits by which 106 falls short of the 108 an exact 2^-84 tail would
//   want change nothing that is measured.
//
//   NOTHING HERE SCORES. There is no oracle, no digit count and no tolerance:
//   every verdict below is the 1/2-ulp comparison classify_nonoverlap() makes on
//   the output words, which is the same verdict the host half makes. This is not
//   a second scorer; it is the SAME check on a different execution space, which
//   is exactly what a tripwire is for.
//
// WHAT STAYED IN THE HOST HALF
//   Test A (every QF op, 30 unary + 4 two-out + 17 binary + the four bespoke
//   forms) and Test C (the 13 named corner cases) both build their inputs with
//   the binary128 make_wide_input and remain there. They lose nothing: they
//   never ran on a device. Only Test B moved, and it moved intact — same five
//   ops, same seeds, same 10^5 inputs, same gate.
// ============================================================================

#include "device_harness.hpp"
#include "test_utils_device.hpp"
#include <xp/qf_math.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace kokkos_ep;

// test_utils_device.hpp declares `namespace dd = xp;` (and the DD/FF tags) but
// has no QF tag — a TODO there. Rather than add a traits specialization to
// shared harness state, declare the alias locally, as qf_eft_test.cpp does.
// dd:: and qf:: name the same namespace; both spellings are used below where
// they say something about intent (dd:: = the wide carrier, qf:: = the ops).
namespace qf = xp;

static const int nd = 100'000;   // 10^5 random inputs per op, as in the host half

// ----------------------------------------------------------------------------
// The Priest length-4 non-overlap invariant and its domain. Copied VERBATIM
// from the host half (qf_nonoverlap_test.cpp), which copied it verbatim from
// qf_eft_test.cpp, so all three layers speak the same invariant. None of it
// touches a type wider than a machine word.
// ----------------------------------------------------------------------------

// Mathematical 1/2 ulp of a normal float, from its binade exponent. frexp writes
// x = m * 2^e with m in [0.5, 1); for FP32's 24-bit significand ulp(x) = 2^(e-24),
// so 1/2 ulp = 2^(e-25).
static double half_ulp(float x) {
  if (x == 0.0f) return 0.0;
  int e;
  std::frexp((double)x, &e);
  return std::ldexp(1.0, e - 25);
}

// UNDERFLOW-TAIL gate: |f_i| >= 2^-102 is required for 1/2 ulp(f_i) to be a
// normal float; below that subnormal quantization makes the comparison ill-posed.
// Gated a hair higher at 2^-100 for margin (identical to T3.1 / T3.2 / T2.2).
static constexpr float kUnderflowTail = 0x1p-100f;
static bool pair_checkable(float hi) {
  if (!std::isfinite(hi)) return false;
  if (hi == 0.0f) return false;                                         // trailing zero
  if (std::fabs(hi) < std::numeric_limits<float>::min()) return false;  // subnormal
  if (std::fabs(hi) < kUnderflowTail) return false;                     // underflow tail
  return true;
}

// GATE POSTURE. Identical to the host half and set by the same flag: QD's renorm
// provably delivers only the WEAKER Shewchuk non-overlap |f_{i+1}| <= ulp(f_i),
// so a word in (1/2 ulp, ulp] is expected rather than a defect (PORT_NOTES_QF
// §16). Only >ulp overlaps and packing breaks are fatal. Every strict deviation
// is still counted and printed. To change the posture, flip this flag HERE AND
// IN THE HOST HALF — do not edit the check.
static constexpr bool kStrictPriestGate = false;

enum NovlClass { NOVL_OK = 0, NOVL_WEAK = 1, NOVL_FAIL = 2 };
static NovlClass classify_nonoverlap(float b0, float b1, float b2, float b3,
                                     int* skips, double* worst_ratio, int* worst_idx) {
  const float b[4] = {b0, b1, b2, b3};
  bool seen_zero = false;
  for (int i = 0; i < 4; ++i) {
    if (b[i] == 0.0f) seen_zero = true;
    else if (seen_zero) return NOVL_FAIL;   // nonzero after zero -> not packed
  }
  NovlClass cls = NOVL_OK;
  for (int i = 0; i < 3; ++i) {
    if (b[i] == 0.0f) break;                 // trailing zeros: trivially holds
    if (!pair_checkable(b[i])) { if (skips) ++*skips; continue; }
    const double hu  = half_ulp(b[i]);
    const double nxt = std::fabs((double)b[i + 1]);
    if (nxt > hu) {
      const double ratio = nxt / hu;         // 1.0 = 1/2 ulp, 2.0 = ulp
      if (worst_ratio && ratio > *worst_ratio) { *worst_ratio = ratio;
                                                 if (worst_idx) *worst_idx = i; }
      if (nxt > 2.0 * hu) return NOVL_FAIL;  // exceeds ulp -> genuine overlap
      cls = NOVL_WEAK;
    }
  }
  return cls;
}

static bool result_checkable(const qf::QuadFloat& d) {
  if (std::isnan(d.f0)) return false;
  if (std::isinf(d.f0)) return false;
  if (d.f0 != 0.0f && std::isfinite(d.f0) &&
      std::fabs(d.f0) < std::numeric_limits<float>::min())
    return false;                       // subnormal leading word
  if (d.f0 != 0.0f && std::fabs(d.f0) < kUnderflowTail)
    return false;                       // near-underflow tail: check ill-posed
  return true;
}

static NovlClass classify_result(const qf::QuadFloat& d, double* ratio, int* idx) {
  return classify_nonoverlap(d.f0, d.f1, d.f2, d.f3, nullptr, ratio, idx);
}

// ----------------------------------------------------------------------------
// Counters and printers (same 4-word hex format as the host half).
// ----------------------------------------------------------------------------
struct InvCount { long tested = 0; long skipped = 0; long failures = 0;
                  long weak = 0; double worst_ratio = 0.0; int worst_idx = -1; };

struct InvSummary {
  std::string name;
  long tested = 0, skipped = 0, failures = 0, weak = 0;
  double worst_ratio = 0.0; int worst_idx = -1;
};

static NovlClass grade(const qf::QuadFloat& d, InvCount& c, double* ratio, int* idx) {
  double r = 0.0; int i = -1;
  NovlClass cl = classify_result(d, &r, &i);
  if (cl == NOVL_FAIL) ++c.failures;
  else if (cl == NOVL_WEAK) {
    ++c.weak;
    if (r > c.worst_ratio) { c.worst_ratio = r; c.worst_idx = i; }
  }
  if (ratio) *ratio = r;
  if (idx)   *idx = i;
  return cl;
}

static uint32_t fbits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(float));
  return b;
}

static void print_fail_unary(const char* op, double x, const qf::QuadFloat& d) {
  std::printf("    FAIL %-12s x=%.9g  f=[%.9g %.9g %.9g %.9g]  "
              "bits=[0x%08x 0x%08x 0x%08x 0x%08x]\n",
              op, x, d.f0, d.f1, d.f2, d.f3,
              fbits(d.f0), fbits(d.f1), fbits(d.f2), fbits(d.f3));
}

static void print_fail_binary(const char* op, double a, double b,
                              const qf::QuadFloat& d) {
  std::printf("    FAIL %-12s a=%.9g b=%.9g  f=[%.9g %.9g %.9g %.9g]  "
              "bits=[0x%08x 0x%08x 0x%08x 0x%08x]\n",
              op, a, b, d.f0, d.f1, d.f2, d.f3,
              fbits(d.f0), fbits(d.f1), fbits(d.f2), fbits(d.f3));
}

static void print_weak(const char* op, double x, const qf::QuadFloat& d,
                       double ratio, int idx) {
  std::printf("    %s %-12s x=%.9g  ratio=%.6f @idx%d (1.0=1/2ulp,2.0=ulp)  "
              "f=[%.9g %.9g %.9g %.9g]\n",
              kStrictPriestGate ? "FAIL(weak-norm)" : "WEAK-NORM",
              op, x, ratio, idx, d.f0, d.f1, d.f2, d.f3);
}

// ----------------------------------------------------------------------------
// Input construction — the host half's make_wide_input with xp::DoubleDouble as
// the carrier in place of binary128. See this file's header for why that is a
// carrier swap and not a weakening.
//
// The tail terms sit at relative 2^-28 / 2^-56 / 2^-84, each below 1/2 ulp of
// f0, so f0 stays == (float)x and the four words come out ordered and
// non-overlapping. Rounding a DoubleDouble to float through its leading word is
// exact except at an exact tie broken by the low word — an input-generation
// detail that no assertion below reads.
// ----------------------------------------------------------------------------
static qf::QuadFloat make_wide_input(double x, std::mt19937_64& g) {
  std::uniform_real_distribution<double> dt(-1.0, 1.0);
  dd::DoubleDouble v(x);
  v = dd::add(v, dd::DoubleDouble(x * (dt(g) * 0x1p-28)));   // fills word 1
  v = dd::add(v, dd::DoubleDouble(x * (dt(g) * 0x1p-56)));   // fills word 2
  v = dd::add(v, dd::DoubleDouble(x * (dt(g) * 0x1p-84)));   // fills word 3
  float w0 = (float)v.hi;  v = dd::subtract(v, dd::DoubleDouble((double)w0));
  float w1 = (float)v.hi;  v = dd::subtract(v, dd::DoubleDouble((double)w1));
  float w2 = (float)v.hi;  v = dd::subtract(v, dd::DoubleDouble((double)w2));
  float w3 = (float)v.hi;
  return qf::QuadFloat(w0, w1, w2, w3);
}

// ----------------------------------------------------------------------------
// A QF value crossing the bus is four FP32 word arrays. Quad4 owns them,
// Quad4Ptr is the trivially-copyable device-side view a kernel captures.
// ----------------------------------------------------------------------------
struct Quad4Ptr {
  float* w0; float* w1; float* w2; float* w3;
  XPMATH_INLINE_FUNCTION void store(std::size_t i, const qf::QuadFloat& v) const {
    w0[i] = v.f0; w1[i] = v.f1; w2[i] = v.f2; w3[i] = v.f3;
  }
  XPMATH_INLINE_FUNCTION qf::QuadFloat load(std::size_t i) const {
    return qf::QuadFloat(w0[i], w1[i], w2[i], w3[i]);
  }
};

struct Quad4 {
  xpt::buffer<float> w0, w1, w2, w3;
  explicit Quad4(std::size_t n) : w0(n), w1(n), w2(n), w3(n) {}
  Quad4Ptr dev() { return Quad4Ptr{w0.device(), w1.device(), w2.device(), w3.device()}; }
  void set(std::size_t i, const qf::QuadFloat& v) {
    w0.host()[i] = v.f0; w1.host()[i] = v.f1; w2.host()[i] = v.f2; w3.host()[i] = v.f3;
  }
  // Not const: xpt::buffer::host() is a non-const accessor (the harness
  // hands out a writable staging pointer and has no const overload).
  qf::QuadFloat get(std::size_t i) {
    return qf::QuadFloat(w0.host()[i], w1.host()[i], w2.host()[i], w3.host()[i]);
  }
  void to_device()   { w0.to_device();   w1.to_device();   w2.to_device();   w3.to_device(); }
  void from_device() { w0.from_device(); w1.from_device(); w2.from_device(); w3.from_device(); }
};

// ----------------------------------------------------------------------------
// The five tripwire ops, as tag structs with a device-callable apply().
// ----------------------------------------------------------------------------
struct OpAdd {
  static constexpr const char* name = "add";
  XPMATH_INLINE_FUNCTION static qf::QuadFloat apply(const qf::QuadFloat& a,
                                                    const qf::QuadFloat& b) {
    return qf::add(a, b);
  }
};
struct OpMultiply {
  static constexpr const char* name = "multiply";
  XPMATH_INLINE_FUNCTION static qf::QuadFloat apply(const qf::QuadFloat& a,
                                                    const qf::QuadFloat& b) {
    return qf::multiply(a, b);
  }
};
struct OpSqrt {
  static constexpr const char* name = "sqrt";
  XPMATH_INLINE_FUNCTION static qf::QuadFloat apply(const qf::QuadFloat& a) {
    return qf::sqrt(a);
  }
};
struct OpExp {
  static constexpr const char* name = "exp";
  XPMATH_INLINE_FUNCTION static qf::QuadFloat apply(const qf::QuadFloat& a) {
    return qf::exp(a);
  }
};
struct OpSin {
  static constexpr const char* name = "sin";
  XPMATH_INLINE_FUNCTION static qf::QuadFloat apply(const qf::QuadFloat& a) {
    return qf::sin(a);
  }
};

// Kernels: trivially-copyable structs holding raw device pointers, NOT lambdas —
// device_harness.hpp passes the functor by value into a __global__ and
// deliberately does not require nvcc --extended-lambda of its callers.
template <class Op>
struct UnaryOpKernel {
  Quad4Ptr in, out;
  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    out.store(i, Op::apply(in.load(i)));
  }
};
template <class Op>
struct BinaryOpKernel {
  Quad4Ptr a, b, out;
  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    out.store(i, Op::apply(a.load(i), b.load(i)));
  }
};

// ----------------------------------------------------------------------------
// Domain predicates on the NOMINAL double (suppressing out-of-domain calls and
// thus QF's domain-guard prints). Copied from the host half.
// ----------------------------------------------------------------------------
static bool dom2_any(double a, double b) { return std::isfinite(a) && std::isfinite(b); }
static bool dom_nonneg(double x) { return std::isfinite(x) && x >= 0.0; }
static bool dom_exp(double x)    { return std::isfinite(x) && x < 88.0; }
// Trig: QF sincos has no tiny-argument stall; only the |a.f0| >= 1e30 "too large"
// guard bounds it. Cap at 1e29 with margin.
static bool dom_trig(double x)   { return std::isfinite(x) && std::fabs(x) < 1e29; }

// ----------------------------------------------------------------------------
// Runners.
// ----------------------------------------------------------------------------
template <class Op>
static InvSummary run_unary(uint64_t seed, double lo, double hi,
                            bool (*in_domain)(double)) {
  const std::size_t n = (std::size_t)nd;
  std::vector<double> hx(n);
  Quad4 in(n), out(n);
  {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> gen(lo, hi);
    for (std::size_t i = 0; i < n; ++i) {
      hx[i] = gen(g);
      in.set(i, make_wide_input(hx[i], g));
    }
  }
  in.to_device();
  xpt::parallel_for_n(n, UnaryOpKernel<Op>{in.dev(), out.dev()});
  out.from_device();

  InvCount c; int samples_left = 3;
  for (std::size_t i = 0; i < n; ++i) {
    if (!in_domain(hx[i])) { ++c.skipped; continue; }
    qf::QuadFloat d = out.get(i);
    if (!result_checkable(d)) { ++c.skipped; continue; }
    ++c.tested;
    double ratio = 0.0; int idx = -1;
    NovlClass cl = grade(d, c, &ratio, &idx);
    if (cl == NOVL_FAIL && samples_left > 0) {
      print_fail_unary(Op::name, hx[i], d); --samples_left;
    } else if (cl == NOVL_WEAK && samples_left > 0) {
      print_weak(Op::name, hx[i], d, ratio, idx); --samples_left;
    }
  }
  std::printf("  [device] %-12s tested=%-8ld skipped=%-8ld failures=%ld weak=%ld worst=%.4f\n",
              Op::name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
  return InvSummary{std::string("device:") + Op::name, c.tested, c.skipped, c.failures,
                    c.weak, c.worst_ratio, c.worst_idx};
}

template <class Op>
static InvSummary run_binary(uint64_t seed, double lo, double hi,
                             bool (*in_domain)(double, double)) {
  const std::size_t n = (std::size_t)nd;
  std::vector<double> ha(n), hb(n);
  Quad4 qa(n), qb(n), out(n);
  {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> gen(lo, hi);
    for (std::size_t i = 0; i < n; ++i) {
      ha[i] = gen(g); qa.set(i, make_wide_input(ha[i], g));
      hb[i] = gen(g); qb.set(i, make_wide_input(hb[i], g));
    }
  }
  qa.to_device(); qb.to_device();
  xpt::parallel_for_n(n, BinaryOpKernel<Op>{qa.dev(), qb.dev(), out.dev()});
  out.from_device();

  InvCount c; int samples_left = 3;
  for (std::size_t i = 0; i < n; ++i) {
    if (!in_domain(ha[i], hb[i])) { ++c.skipped; continue; }
    qf::QuadFloat d = out.get(i);
    if (!result_checkable(d)) { ++c.skipped; continue; }
    ++c.tested;
    double ratio = 0.0; int idx = -1;
    NovlClass cl = grade(d, c, &ratio, &idx);
    if (cl == NOVL_FAIL && samples_left > 0) {
      print_fail_binary(Op::name, ha[i], hb[i], d); --samples_left;
    } else if (cl == NOVL_WEAK && samples_left > 0) {
      print_weak(Op::name, ha[i], d, ratio, idx); --samples_left;
    }
  }
  std::printf("  [device] %-12s tested=%-8ld skipped=%-8ld failures=%ld weak=%ld worst=%.4f\n",
              Op::name, c.tested, c.skipped, c.failures, c.weak, c.worst_ratio);
  return InvSummary{std::string("device:") + Op::name, c.tested, c.skipped, c.failures,
                    c.weak, c.worst_ratio, c.worst_idx};
}

// ============================================================================
int main(int, char**) {
  std::printf("=== qf_nonoverlap_test_device (T3.2, C4 device half): Priest "
              "length-4 non-overlap |f_{i+1}| <= 1/2 ulp(f_i) on the device ===\n");
  std::printf("execution space: %s\n", xpt::where_name());
  std::printf("Oracle-independent (mathematical 1/2-ulp check). Inputs enriched to "
              "full 4-word width through a DoubleDouble carrier — see this file's "
              "header.\n\n");

  std::printf("[Test B] device tripwire (5 ops, 10^5 random each)\n");
  std::vector<InvSummary> summary;
  summary.push_back(run_binary<OpAdd>(55501ULL, -1e8, 1e8, dom2_any));
  summary.push_back(run_binary<OpMultiply>(55502ULL, -1e6, 1e6, dom2_any));
  summary.push_back(run_unary<OpSqrt>(55503ULL, 0.0, 1e8, dom_nonneg));
  summary.push_back(run_unary<OpExp>(55504ULL, -88.0, 87.5, dom_exp));
  summary.push_back(run_unary<OpSin>(55505ULL, -1000.0, 1000.0, dom_trig));

  std::printf("\n=== Summary (op : tested / skipped / failures / weak / worst : status) ===\n");
  std::printf("    gate posture: %s (kStrictPriestGate=%s)\n",
              kStrictPriestGate ? "STRICT Priest 1/2-ulp -- WEAK counts as failure"
                                : "WEAK Shewchuk <=ulp -- WEAK tolerated",
              kStrictPriestGate ? "true" : "false");
  long total_tested = 0, total_skipped = 0, total_failures = 0, total_weak = 0;
  double overall_worst = 0.0;
  for (const auto& s : summary) {
    total_tested += s.tested; total_skipped += s.skipped;
    total_failures += s.failures; total_weak += s.weak;
    if (s.worst_ratio > overall_worst) overall_worst = s.worst_ratio;
    long op_gate_fail = s.failures + (kStrictPriestGate ? s.weak : 0);
    std::printf("  %-24s %11ld %11ld %9ld %8ld %8.4f   %s\n",
                s.name.c_str(), s.tested, s.skipped, s.failures, s.weak, s.worst_ratio,
                op_gate_fail == 0 ? "OK" : "FAIL");
  }
  std::printf("  %-24s %11ld %11ld %9ld %8ld %8.4f\n",
              "TOTAL", total_tested, total_skipped, total_failures, total_weak, overall_worst);

  const long gate_fail = total_failures + (kStrictPriestGate ? total_weak : 0);
  if (total_weak > 0)
    std::printf("\n  NOTE: %ld weak-normalization deviation(s) in (1/2 ulp, ulp] "
                "(worst ratio %.4f). These are QD renorm's Shewchuk-weak non-overlap,\n"
                "  not per-op bugs (see PORT_NOTES / T3.2 report). Under the strict "
                "Priest gate they count as failures.\n", total_weak, overall_worst);

  // A run that checked nothing is not a pass: if every result came back
  // uncheckable (a dead launch leaves the output buffers at whatever the
  // allocator returned) every count above reads 0 failures.
  KOKKOS_EP_ASSERT(total_tested > 0,
                   "no device result was checkable — the tripwire proved nothing");
  KOKKOS_EP_ASSERT(gate_fail == 0,
                   "one or more QF ops violated the length-4 non-overlap gate on the device");
  // A nonzero vendor code is a TEST FAILURE, not a warning. Sticky since process
  // start, so a later good call cannot erase it.
  KOKKOS_EP_ASSERT(xpt::last_error() == 0,
                   "device harness reported a nonzero vendor error code");

  int rc = ep_exit_code();
  std::printf("\n=== qf_nonoverlap_test_device: %s ===\n",
              rc == 0 ? "ALL PASSED" : "FAILURES PRESENT");
  return rc;
}
