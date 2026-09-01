// ============================================================================
// tf_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for TF.  Plan S10 Phase 2.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// TF's arithmetic is built on Dekker's error-free transforms. Two of them are
// TWO-ROUNDED-OPERATION sequences whose error term is meaningful only because a
// product and its subtraction are DISTINCT rounded steps:
//
//   tf_two_prod(a,b)  ->  p = fl(a*b),  e = ((a1*b1 - p) + a1*b2 + a2*b1) + a2*b2
//   tf_two_sqr (a)    ->  q = fl(a*a),  e = ((hi*hi  - q) + 2*hi*lo)      + lo*lo
//                                             ^^^^^^^^^^^ the contraction hazard
//   (Veltkamp splitter 8193.0f = 2^13 + 1, reused from ff_math.hpp.)
//
// If the compiler CONTRACTS `a1*b1 - p` (resp. `hi*hi - q`) into a single fused
// multiply-add, the rounding Dekker's algebra depends on never happens and the
// "error" term is silently wrong. This test builds the tf_math.hpp primitives
// under BOTH contraction settings and cross-checks against a contraction-immune
// FP64 oracle.
//
//   * contraction-OFF build  -> the error terms must be EXACT. FAIL-GATES.
//   * contraction-ON  build  -> the compiler is ALLOWED to contract. The verdict
//                               is REPORTED, not gated — the guard's job is to
//                               make the compiler's behavior VISIBLE, not forbid
//                               it. The ONE genuinely-bad outcome (a
//                               nonzero-but-wrong error term) fails the ON target.
//
// This is the TF analogue of qf_fma_guard_test.cpp (T3.5), ff_fma_guard_test.cpp
// (T2.5), and dd_fma_guard_test.cpp (T1.5). Every design decision T3.5 locked in
// carries over — single-source/two-targets, a contraction-immune FP64 oracle (no
// quadmath), a twoSum CONTROL, host + device passes, OFF gates / ON reports with
// a committed baseline.
//
// TF DIVERGENCE FROM FF: DIRECT CALL TO SHIPPED PRIMITIVES (same as QF)
// ----------------------------------------------------------------------
// ff_fma_guard_test had to DUPLICATE FF's Dekker twoProduct because ff_math.hpp
// embeds it inside multiply() with no standalone primitive to call. tf_math.hpp
// is different: it EXPOSES the shipped primitives as free functions (tf_two_prod
// / tf_two_sqr / tf_two_sum). So this test calls the ACTUAL shipped code under
// the ON flags — strictly stronger for a contraction guard, as it characterizes
// whether GCC contracts *tf_math.hpp's own source*, not a copy. This mirrors
// T3.5's same divergence from T2.5. (Rule 4 is trivially respected — we only
// #include, never edit.)
//
// tf_two_sqr IS ALSO GUARDED (same as QF). FF had no squaring EFT; TF ships
// tf_two_sqr, a SECOND Dekker sequence with a `hi*hi - q` contraction hazard, so
// this test guards both — matching T3.5's op surface.
//
// THREE-WAY CLASSIFICATION (the ON reporter's refinement)
// --------------------------------------------------------
// For each in-domain input, with got = (shipped hi, shipped lo), exact = the
// contraction-immune FP64 product, and e_ref = the unique exact residual
// float(exact - got.hi):
//
//   id_ok := ((double)got.hi + (double)got.lo == exact)
//
//   * e_ref == 0 -> TRIVIAL (uninformative; both correct and contracted yield lo==0).
//   * e_ref != 0 and  id_ok                 -> ERR_NONZERO_CORRECT
//   * e_ref != 0 and !id_ok and got.lo == 0 -> ERR_ZERO (classic contraction signature)
//   * e_ref != 0 and !id_ok and got.lo != 0 -> ERR_NONZERO_WRONG (genuinely broken)
//
//   OFF gate: ERR_ZERO == 0 && ERR_NONZERO_WRONG == 0   (F := their sum).
//   ON  PASS: ERR_NONZERO_WRONG == 0
//
// A SINGLE SOURCE, TWO TARGETS
// ----------------------------
// This one file is compiled TWICE into two executables:
//   tf_fma_guard_test              (kokkos_ep_add_eft_test             -> OFF)
//   tf_fma_guard_test_contract_on  (kokkos_ep_add_eft_test_contract_on -> ON)
// The only per-variant knobs are compile definitions the helpers set:
//   KOKKOS_EP_CONTRACTION_MODE=0 (OFF) or =1 (ON)
//   KOKKOS_EP_BASELINE_PATH=... (ON only, drift detection)

#include "test_utils.hpp"
#include "corpus.hpp"
#include <xp/tf_math.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

using namespace kokkos_ep;

// TF types live in xp::
namespace tf = xp;

// ----------------------------------------------------------------------------
// Oracle comparisons (FP64, provably exact)
// ----------------------------------------------------------------------------
inline bool sum_is_exact(float a, float b) {
  float e;
  float s = xp::tf_two_sum(a, b, e);
  return (double)s + (double)e == (double)a + (double)b;
}

inline bool prod_is_exact(float a, float b) {
  float e;
  float p = xp::tf_two_prod(a, b, e);
  return (double)p + (double)e == (double)a * (double)b;
}

inline bool sqr_is_exact(float a) {
  float e;
  float q = xp::tf_two_sqr(a, e);
  return (double)q + (double)e == (double)a * (double)a;
}

// ----------------------------------------------------------------------------
// Domain predicates
// ----------------------------------------------------------------------------
inline bool sum_in_domain(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  return std::isfinite(a + b);
}

inline float split_safe_max() {
  return std::numeric_limits<float>::max() / 8193.0f;
}

inline bool prod_in_domain(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  const float fmin = std::numeric_limits<float>::min();
  auto normal_or_zero = [fmin](float x) {
    return x == 0.0f || std::fabs(x) >= fmin;
  };
  if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
  const float ssm = split_safe_max();
  if (std::fabs(a) >= ssm || std::fabs(b) >= ssm) return false;
  if (a == 0.0f || b == 0.0f) return true;
  double tp = (double)a * (double)b;
  double mag = tp < 0.0 ? -tp : tp;
  const double hi_lim = (double)std::numeric_limits<float>::max();
  const double lo_lim = std::ldexp(1.0, -102);
  return mag <= hi_lim && mag >= lo_lim;
}

// ----------------------------------------------------------------------------
// Three-way classification (ON reporter)
// ----------------------------------------------------------------------------
#if KOKKOS_EP_CONTRACTION_MODE == 1
enum class Verdict { TRIVIAL, ERR_NONZERO_CORRECT, ERR_ZERO, ERR_NONZERO_WRONG };

static Verdict classify_prod(float a, float b) {
  float e_got;
  float p_got = xp::tf_two_prod(a, b, e_got);
  double exact = (double)a * (double)b;
  float e_ref = (float)(exact - (double)p_got);
  if (e_ref == 0.0f) return Verdict::TRIVIAL;
  bool id_ok = ((double)p_got + (double)e_got == exact);
  if (id_ok) return Verdict::ERR_NONZERO_CORRECT;
  if (e_got == 0.0f) return Verdict::ERR_ZERO;
  return Verdict::ERR_NONZERO_WRONG;
}

static Verdict classify_sqr(float a) {
  float e_got;
  float q_got = xp::tf_two_sqr(a, e_got);
  double exact = (double)a * (double)a;
  float e_ref = (float)(exact - (double)q_got);
  if (e_ref == 0.0f) return Verdict::TRIVIAL;
  bool id_ok = ((double)q_got + (double)e_got == exact);
  if (id_ok) return Verdict::ERR_NONZERO_CORRECT;
  if (e_got == 0.0f) return Verdict::ERR_ZERO;
  return Verdict::ERR_NONZERO_WRONG;
}
#endif

// ----------------------------------------------------------------------------
// Accumulator
// ----------------------------------------------------------------------------
struct EftCount {
  long tested = 0;
  long skipped = 0;
  long failures = 0;
#if KOKKOS_EP_CONTRACTION_MODE == 1
  long trivial = 0;
  long correct = 0;
  long zero = 0;
  long wrong = 0;
#endif
};

// ----------------------------------------------------------------------------
// Host batches
// ----------------------------------------------------------------------------
enum class Op { Sum, Prod, Sqr };

inline void check_pair(Op op, float a, float b, EftCount& c) {
  bool in_domain = (op == Op::Sum) ? sum_in_domain(a, b) : prod_in_domain(a, b);
  if (!in_domain) { ++c.skipped; return; }
  ++c.tested;

#if KOKKOS_EP_CONTRACTION_MODE == 0
  // OFF: gate on exact
  bool ok;
  switch (op) {
    case Op::Sum:  ok = sum_is_exact(a, b);  break;
    case Op::Prod: ok = prod_is_exact(a, b); break;
    case Op::Sqr:  ok = sqr_is_exact(a);     break;
  }
  if (!ok) ++c.failures;
#else
  // ON: classify
  Verdict v;
  switch (op) {
    case Op::Prod: v = classify_prod(a, b); break;
    case Op::Sqr:  v = classify_sqr(a);     break;
    default: return;  // Sum is control-only, not classified
  }
  switch (v) {
    case Verdict::TRIVIAL:            ++c.trivial; break;
    case Verdict::ERR_NONZERO_CORRECT: ++c.correct; break;
    case Verdict::ERR_ZERO:            ++c.zero;    break;
    case Verdict::ERR_NONZERO_WRONG:   ++c.wrong;   break;
  }
#endif
}

static EftCount run_host_batches(Op op, const char* label) {
  EftCount total;
  const float R = (op == Op::Sum) ? 1e30f : 1e18f;

  // Uniform broad
  std::mt19937_64 gen(12345ULL);
  std::uniform_real_distribution<float> d(-R, R);
  for (int i = 0; i < 1'000'000; ++i) {
    float a = d(gen), b = d(gen);
    check_pair(op, a, b, total);
  }

  // Uniform narrow
  std::uniform_real_distribution<float> d_narrow(-1.0f, 1.0f);
  for (int i = 0; i < 500'000; ++i) {
    float a = d_narrow(gen), b = d_narrow(gen);
    check_pair(op, a, b, total);
  }

  // Corpus
  corpus::CorpusFlags flags;
  std::vector<float> xs = corpus::unary<float>(flags);
  const size_t N = xs.size();
  const size_t kMaxPairs = 250'000;
  size_t made = 0;
  for (size_t i = 0; i < N && made < kMaxPairs; ++i) {
    for (size_t j = i + 1; j < N && made < kMaxPairs; ++j) {
      check_pair(op, xs[i], xs[j], total);
      ++made;
    }
  }

  return total;
}

// ----------------------------------------------------------------------------
// Device parity
// ----------------------------------------------------------------------------
static EftCount run_device_parity() {
  EftCount R;
  using exec_space = Kokkos::DefaultExecutionSpace;
  const int nd = 200'000;

  std::vector<float> ha(nd), hb(nd);
  std::mt19937_64 gen(99999ULL);
  std::uniform_real_distribution<float> d(-1e18f, 1e18f);
  for (int i = 0; i < nd; ++i) { ha[i] = d(gen); hb[i] = d(gen); }

  Kokkos::View<float*, exec_space> va("va", nd), vb("vb", nd);
  Kokkos::View<float*, exec_space> s_hi("s_hi", nd), s_lo("s_lo", nd);
  Kokkos::View<float*, exec_space> p_hi("p_hi", nd), p_lo("p_lo", nd);
  Kokkos::View<float*, exec_space> q_hi("q_hi", nd), q_lo("q_lo", nd);

  auto hva = Kokkos::create_mirror_view(va);
  auto hvb = Kokkos::create_mirror_view(vb);
  for (int i = 0; i < nd; ++i) { hva(i) = ha[i]; hvb(i) = hb[i]; }
  Kokkos::deep_copy(va, hva);  Kokkos::deep_copy(vb, hvb);

  Kokkos::parallel_for("tf_fma_guard_device", Kokkos::RangePolicy<exec_space>(0, nd),
    KOKKOS_LAMBDA(int i) {
      float es, ep, eq;
      s_hi(i) = xp::tf_two_sum(va(i), vb(i), es);  s_lo(i) = es;
      p_hi(i) = xp::tf_two_prod(va(i), vb(i), ep); p_lo(i) = ep;
      q_hi(i) = xp::tf_two_sqr(va(i), eq);         q_lo(i) = eq;
    });
  Kokkos::fence();

  auto hshi = Kokkos::create_mirror_view(s_hi);
  auto hslo = Kokkos::create_mirror_view(s_lo);
  auto hphi = Kokkos::create_mirror_view(p_hi);
  auto hplo = Kokkos::create_mirror_view(p_lo);
  auto hqhi = Kokkos::create_mirror_view(q_hi);
  auto hqlo = Kokkos::create_mirror_view(q_lo);
  Kokkos::deep_copy(hshi, s_hi); Kokkos::deep_copy(hslo, s_lo);
  Kokkos::deep_copy(hphi, p_hi); Kokkos::deep_copy(hplo, p_lo);
  Kokkos::deep_copy(hqhi, q_hi); Kokkos::deep_copy(hqlo, q_lo);

  long sum_fail = 0, prod_fail = 0, sqr_fail = 0;
  long sum_skip = 0, prod_skip = 0, sqr_skip = 0;
#if KOKKOS_EP_CONTRACTION_MODE == 1
  long prod_tri = 0, prod_cor = 0, prod_zero = 0, prod_wrong = 0;
  long sqr_tri = 0, sqr_cor = 0, sqr_zero = 0, sqr_wrong = 0;
#endif

  for (int i = 0; i < nd; ++i) {
    float a = ha[i], b = hb[i];

    // twoSum
    if (sum_in_domain(a, b)) {
      if ((double)hshi(i) + (double)hslo(i) != (double)a + (double)b) ++sum_fail;
    } else ++sum_skip;

    // twoProd
    if (prod_in_domain(a, b)) {
#if KOKKOS_EP_CONTRACTION_MODE == 0
      if ((double)hphi(i) + (double)hplo(i) != (double)a * (double)b) ++prod_fail;
#else
      double exact = (double)a * (double)b;
      float e_ref = (float)(exact - (double)hphi(i));
      if (e_ref == 0.0f) { ++prod_tri; continue; }
      bool id_ok = ((double)hphi(i) + (double)hplo(i) == exact);
      if (id_ok) ++prod_cor;
      else if (hplo(i) == 0.0f) ++prod_zero;
      else ++prod_wrong;
#endif
    } else ++prod_skip;

    // twoSqr
    if (prod_in_domain(a, a)) {
#if KOKKOS_EP_CONTRACTION_MODE == 0
      if ((double)hqhi(i) + (double)hqlo(i) != (double)a * (double)a) ++sqr_fail;
#else
      double exact = (double)a * (double)a;
      float e_ref = (float)(exact - (double)hqhi(i));
      if (e_ref == 0.0f) { ++sqr_tri; continue; }
      bool id_ok = ((double)hqhi(i) + (double)hqlo(i) == exact);
      if (id_ok) ++sqr_cor;
      else if (hqlo(i) == 0.0f) ++sqr_zero;
      else ++sqr_wrong;
#endif
    } else ++sqr_skip;
  }

  R.tested = 3 * nd;
  R.skipped = (int)(sum_skip + prod_skip + sqr_skip);
#if KOKKOS_EP_CONTRACTION_MODE == 0
  R.failures = (int)(sum_fail + prod_fail + sqr_fail);
#else
  R.trivial = prod_tri + sqr_tri;
  R.correct = prod_cor + sqr_cor;
  R.zero = prod_zero + sqr_zero;
  R.wrong = prod_wrong + sqr_wrong;
  R.failures = (int)R.wrong;
#endif

  return R;
}

// ============================================================================
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
#if KOKKOS_EP_CONTRACTION_MODE == 0
    std::printf("=== tf_fma_guard_test: contraction OFF (gate on exact) ===\n");
#else
    std::printf("=== tf_fma_guard_test_contract_on: contraction ON (report) ===\n");
#endif
    std::printf("Oracle: FP64 (exact, 25-bit sum / 48-bit product fit in 53-bit FP64)\n");
    std::printf("Splitter: 8193.0f = 2^13 + 1 (reused from ff_math.hpp)\n\n");

    // twoSum CONTROL
    std::printf("[Control] tf_two_sum\n");
    EftCount sum_h = run_host_batches(Op::Sum, "Sum");
    std::printf("  Host: tested=%ld, skip=%ld, fail=%ld\n", sum_h.tested, sum_h.skipped, sum_h.failures);
    KOKKOS_EP_ASSERT(sum_h.failures == 0, "tf_two_sum control failed");

    // twoProd
    std::printf("\n[Test] tf_two_prod\n");
    EftCount prod_h = run_host_batches(Op::Prod, "Prod");
#if KOKKOS_EP_CONTRACTION_MODE == 0
    std::printf("  Host: tested=%ld, skip=%ld, fail=%ld\n", prod_h.tested, prod_h.skipped, prod_h.failures);
    KOKKOS_EP_ASSERT(prod_h.failures == 0, "tf_two_prod OFF not bit-exact");
#else
    std::printf("  Host: tested=%ld, trivial=%ld, correct=%ld, zero=%ld, wrong=%ld\n",
                prod_h.tested, prod_h.trivial, prod_h.correct, prod_h.zero, prod_h.wrong);
#endif

    // twoSqr
    std::printf("\n[Test] tf_two_sqr\n");
    EftCount sqr_h = run_host_batches(Op::Sqr, "Sqr");
#if KOKKOS_EP_CONTRACTION_MODE == 0
    std::printf("  Host: tested=%ld, skip=%ld, fail=%ld\n", sqr_h.tested, sqr_h.skipped, sqr_h.failures);
    KOKKOS_EP_ASSERT(sqr_h.failures == 0, "tf_two_sqr OFF not bit-exact");
#else
    std::printf("  Host: tested=%ld, trivial=%ld, correct=%ld, zero=%ld, wrong=%ld\n",
                sqr_h.tested, sqr_h.trivial, sqr_h.correct, sqr_h.zero, sqr_h.wrong);
#endif

    // Device
    std::printf("\n[Device] %s\n", Kokkos::DefaultExecutionSpace::name());
    EftCount dev = run_device_parity();
#if KOKKOS_EP_CONTRACTION_MODE == 0
    std::printf("  Device: tested=%ld, skip=%d, fail=%ld\n", dev.tested, dev.skipped, dev.failures);
    KOKKOS_EP_ASSERT(dev.failures == 0, "device EFT parity mismatch");
#else
    std::printf("  Device: tested=%ld, trivial=%ld, correct=%ld, zero=%ld, wrong=%ld\n",
                dev.tested, dev.trivial, dev.correct, dev.zero, dev.wrong);
#endif

#if KOKKOS_EP_CONTRACTION_MODE == 0
    long F = prod_h.failures + sqr_h.failures + dev.failures;
    KOKKOS_EP_ASSERT(F == 0, "OFF: at least one Dekker primitive broke");
    rc = ep_exit_code();
#else
    long W_total = prod_h.wrong + sqr_h.wrong + dev.wrong;
    if (W_total > 0) {
      std::printf("\nERROR: ERR_NONZERO_WRONG detected (nonzero but violates Dekker identity)\n");
      rc = ep_exit_code();
    } else {
      std::printf("\nON target: ERR_NONZERO_WRONG == 0 -> PASS\n");
    }
#endif

    std::printf("\n=== tf_fma_guard_test: %s ===\n", rc == 0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return rc;
}
