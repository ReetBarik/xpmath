// ============================================================================
// tf_fma_guard_test.cpp — Layer 5 (FMA-contraction guard) for TF.  Plan S10 Phase 2.
// ============================================================================
//
// WHAT THIS TEST IS AND WHY IT EXISTS
// -----------------------------------
// TF's production tf_two_prod / tf_two_sqr (include/xp/tf_math.hpp, backing
// multiply, sqr, multiply_scalar, and through multiply_scalar the long-division
// quotient digits of divide()) compute the exact product/square error term via
// an FMA-based TwoProduct, not Dekker/Veltkamp splitting:
//
//   tf_two_prod(a,b)  ->  p = fl(a*b),  e = fma(a, b, -p)
//   tf_two_sqr (a)    ->  q = fl(a*a),  e = fma(a, a, -q)   // tf_two_prod(a,a)
//
// This is correct provided (a) `p` (resp. `q`) is the correctly-rounded product
// and (b) the single rounding inside `fma` is the ONLY rounding — i.e. the
// compiler is not allowed to further reassociate the already-explicit `fma`
// call, and the volatile guards inside `detail::eft_mul`/`detail::eft_fma`
// (include/xp/config.hpp) are not allowed to be optimized away. Those guards
// exist precisely because aggressive compilers (gfx90a under ROCm 7.0.2 even
// with -ffp-contract=off; g++ 13.3 under -ffp-contract=fast/-ffast-math) have
// been observed to fuse or fold adjacent rounded ops that look, to a
// real-number-algebra optimizer, like they cancel to zero. This test builds the
// tf_math.hpp primitives under BOTH contraction settings and cross-checks
// against a contraction-immune FP64 oracle.
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
// __float128), a twoSum CONTROL, host + device passes, OFF gates / ON reports with
// a committed baseline.
//
// TF DIVERGENCE FROM FF: DIRECT CALL TO SHIPPED PRIMITIVES (same as QF)
// ----------------------------------------------------------------------
// ff_fma_guard_test originally had to DUPLICATE FF's Dekker twoProduct because
// ff_math.hpp embedded it inside multiply() with no standalone primitive to
// call (that mirror was retired when FF migrated to FMA; see
// ff_fma_guard_test.cpp's own "WHY THIS NO LONGER TESTS A LOCAL DEKKER
// MIRROR"). tf_math.hpp never had that problem: it has always EXPOSED its EFT
// primitives as free functions (tf_two_prod / tf_two_sqr / tf_two_sum). So this
// test has always called the ACTUAL shipped code under the ON flags —
// strictly stronger for a contraction guard, as it characterizes whether GCC
// contracts *tf_math.hpp's own source* (Dekker's, historically; FMA's, now),
// not a copy that could drift. This mirrors T3.5's same divergence from T2.5.
// (Rule 4 is trivially respected — we only #include, never edit.)
//
// tf_two_sqr IS ALSO GUARDED (same as QF). FF had no squaring EFT; TF ships
// tf_two_sqr, a SECOND FMA-based TwoProduct instance (tf_two_prod(a,a)) with
// the same reassociation hazard, so this test guards both — matching T3.5's op
// surface.
//
// DOMAIN WIDENING vs. the pre-migration version of this file
// ------------------------------------------------------------
// The old Dekker-based domain predicate excluded |operand| >= split_safe_max
// (~4.1528e34 = FLT_MAX/8193, ~2^114.9998) because SPLITTING an operand that
// large overflows the splitter itself — a hazard specific to Veltkamp splitting
// that does not exist for FMA-based tf_two_prod/tf_two_sqr (they never split).
// That exclusion is removed below (see prod_in_domain). What remains —
// rejecting subnormal operands and bounding the TRUE product away from FP32's
// overflow/underflow edges — is NOT splitter-specific: it is the IEEE-754
// "exact FMA residual" precondition (Boldo & Muller), the same requirement an
// FMA-based TwoProduct has that a Dekker one does. It stays.
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
//   tf_fma_guard_test              (xpm_add_device_eft_test             -> OFF)
//   tf_fma_guard_test_contract_on  (xpm_add_device_eft_test_contract_on -> ON)
// The only per-variant knobs are compile definitions the helpers set:
//   XPMATH_CONTRACTION_MODE=0 (OFF) or =1 (ON)
//   XPMATH_BASELINE_PATH=... (ON only, drift detection)

// LAUNCH MECHANISM: tests/device_harness.hpp, not Kokkos (CORE_PLAN C4 step 3).
// Nothing this test measures changed -- same primitives, same inputs, same
// seeds, same FP64 oracle, same OFF-gates/ON-reports contract. What changed is
// that a GPU can now be asked whether xp::tf_two_prod holds up on it without a
// Kokkos install standing in the way of the answer.
#include "device_harness.hpp"
#include "test_utils_device.hpp"
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

using namespace xpmath;

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
// Domain predicates — the "no overflow / no underflow" region the FMA-based
// TwoProduct is proven over. A pair outside the domain is SKIPPED, not counted
// (EFT skip-not-fail discipline). The splitter-overflow exclusion this
// predicate used to carry (split_safe_max) is GONE — see "DOMAIN WIDENING" in
// the file header: tf_two_prod/tf_two_sqr never split, so there is no splitter
// left to overflow.
// ----------------------------------------------------------------------------
inline bool sum_in_domain(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  return std::isfinite(a + b);
}

// twoProd/twoSqr domain: finite normal-or-zero operands with the TRUE product
// comfortably inside the normal range (so both p and its error term e are
// representable without underflow). The FMA-based TwoProduct is exact only
// absent underflow/overflow of the true product (Boldo & Muller; Muller et al.,
// "Handbook of Floating-Point Arithmetic", §4.4 — the same precondition a
// Dekker-split TwoProduct has). The predicate is evaluated on the EXACT (double)
// product, not the rounded FP32 product (which would flush to 0 in the underflow
// regime and let underflowing pairs masquerade as in-domain). Subnormal OPERANDS
// are excluded — an exact FMA residual against a subnormal operand is itself a
// separate, narrower guarantee this predicate does not attempt to certify.
// Threshold FLT_MIN * 2^24 = 2^-102 keeps e >= ~2^-126.
inline bool prod_in_domain(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  const float fmin = std::numeric_limits<float>::min();   // smallest normal FP32
  auto normal_or_zero = [fmin](float x) {
    return x == 0.0f || std::fabs(x) >= fmin;             // reject subnormals
  };
  if (!normal_or_zero(a) || !normal_or_zero(b)) return false;
  if (a == 0.0f || b == 0.0f) return true;                       // exact product 0
  double tp = (double)a * (double)b;                             // exact (48 <= 53)
  double mag = tp < 0.0 ? -tp : tp;
  const double hi_lim = (double)std::numeric_limits<float>::max();
  const double lo_lim = std::ldexp(1.0, -102);
  if (mag > hi_lim) return false;   // product (and p) would overflow FP32
  if (mag < lo_lim) return false;   // error term would fall into FP32 subnormals
  return true;
}

// twoSum oracle-faithfulness predicate (CONTROL pass only). FP32-SPECIFIC —
// same guard T2.5's ff_fma_guard_test / T3.5's qf_fma_guard_test carry (no DD
// counterpart). A product-domain pair like (a = FLT_MIN = 2^-126, b = 2^24)
// has an in-range PRODUCT (2^-102) yet an exact SUM spanning 2^24 down to
// 2^-126 — far beyond the FP64 oracle's 53-bit mantissa. The FP32 twoSum is
// CORRECT there (hi = 2^24, lo = 2^-126, non-overlapping); it is the FP64
// *decomposition oracle* that collapses the tiny tail, so the exact FP32 lo
// would look like a false "mismatch". A pair is in the control's domain only
// when the exact FP64 sum is faithful, i.e. the FP64 twoSum error term is zero
// (the double sum carries no rounding). This excludes ONLY wide-exponent sums,
// never a pair where FP32 twoSum is actually wrong. (tf_two_prod / tf_two_sqr
// need no such guard — their exact product is always <= 48 bits, so the FP64
// oracle is unconditionally faithful.)
//
// This was latent and unaddressed before this chunk (TF's guard test gated its
// twoSum CONTROL on sum_in_domain alone, same gap Chunk B found and fixed in
// dd_fma_guard_test.cpp for DD — unlike FF/QF, which already carried it).
// Needed regardless of the FMA migration's domain widening above: it is a
// general FP64-oracle-faithfulness check, not a threshold tied to the old
// splitter bound (matching ff_fma_guard_test's/qf_fma_guard_test's identical
// finding for their own word width).
inline bool sum_oracle_faithful(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  double da = (double)a, db = (double)b;
  double s  = da + db;
  if (!std::isfinite(s)) return false;             // sum must not overflow (twoSum domain)
  double e   = s - da;                             // FP64 twoSum error term;
  double err = (db - e) + (da - (s - e));          // zero <=> the double sum is exact.
  return err == 0.0;
}

// ----------------------------------------------------------------------------
// Three-way classification (ON reporter)
// ----------------------------------------------------------------------------
#if XPMATH_CONTRACTION_MODE == 1
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
#if XPMATH_CONTRACTION_MODE == 1
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
  // Op::Sqr squares `a` and IGNORES `b`, so its domain must be evaluated at
  // (a, a) — not at (a, b). Gating the sqr path on prod_in_domain(a, b) admitted
  // pairs whose PRODUCT is in range while a*a is not: e.g. a = FLT_MIN with
  // b = 2^24 has |a*b| = 2^-102 (in domain) but a*a = 1.4e-76, ~2^-252, which
  // flushes to zero in FP32 so the exact residual is unrepresentable and
  // tf_two_sqr's error term is legitimately inexact. That is the underflow
  // exclusion prod_in_domain already encodes, applied to the wrong pair. QF gets
  // this right by construction — qf_fma_guard_test.cpp:356/362/367
  // (build_sqr_inputs) filters every squaring input through prod_in_domain(a, a),
  // and its named cases do the same at :511. The device pass below (line ~343)
  // was already correct; only this host path was not.
  bool in_domain;
  switch (op) {
    // sum_oracle_faithful: see its comment — excludes wide-exponent-gap pairs
    // where the FP64 oracle itself can't witness the exact FP32 twoSum result.
    case Op::Sum:  in_domain = sum_in_domain(a, b) && sum_oracle_faithful(a, b);  break;
    case Op::Prod: in_domain = prod_in_domain(a, b);  break;
    case Op::Sqr:  in_domain = prod_in_domain(a, a);  break;
  }
  if (!in_domain) { ++c.skipped; return; }
  ++c.tested;

#if XPMATH_CONTRACTION_MODE == 0
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
// The kernel, as a trivially-copyable struct rather than a lambda: the harness
// passes F by value into a __global__ and deliberately does not require
// nvcc --extended-lambda of its callers.
struct GuardKernel {
  const float* a;
  const float* b;
  float* s_hi; float* s_lo;
  float* p_hi; float* p_lo;
  float* q_hi; float* q_lo;

  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    float es, ep, eq;
    s_hi[i] = xp::tf_two_sum(a[i], b[i], es);  s_lo[i] = es;
    p_hi[i] = xp::tf_two_prod(a[i], b[i], ep); p_lo[i] = ep;
    q_hi[i] = xp::tf_two_sqr(a[i], eq);        q_lo[i] = eq;
  }
};

static EftCount run_device_parity() {
  EftCount R;
  const int nd = 200'000;

  std::vector<float> ha(nd), hb(nd);
  std::mt19937_64 gen(99999ULL);
  std::uniform_real_distribution<float> d(-1e18f, 1e18f);
  for (int i = 0; i < nd; ++i) { ha[i] = d(gen); hb[i] = d(gen); }

  xpt::buffer<float> va(nd), vb(nd);
  xpt::buffer<float> s_hi(nd), s_lo(nd), p_hi(nd), p_lo(nd), q_hi(nd), q_lo(nd);
  for (int i = 0; i < nd; ++i) { va.host()[i] = ha[i]; vb.host()[i] = hb[i]; }
  va.to_device(); vb.to_device();

  xpt::parallel_for_n(static_cast<std::size_t>(nd),
      GuardKernel{va.device(), vb.device(),
                  s_hi.device(), s_lo.device(),
                  p_hi.device(), p_lo.device(),
                  q_hi.device(), q_lo.device()});

  s_hi.from_device(); s_lo.from_device();
  p_hi.from_device(); p_lo.from_device();
  q_hi.from_device(); q_lo.from_device();

  const float* hshi = s_hi.host();
  const float* hslo = s_lo.host();
  const float* hphi = p_hi.host();
  const float* hplo = p_lo.host();
  const float* hqhi = q_hi.host();
  const float* hqlo = q_lo.host();

  long sum_fail = 0;
#if XPMATH_CONTRACTION_MODE == 0
  long prod_fail = 0, sqr_fail = 0;
#endif
  long sum_skip = 0, prod_skip = 0, sqr_skip = 0;
#if XPMATH_CONTRACTION_MODE == 1
  long prod_tri = 0, prod_cor = 0, prod_zero = 0, prod_wrong = 0;
  long sqr_tri = 0, sqr_cor = 0, sqr_zero = 0, sqr_wrong = 0;
#endif

  for (int i = 0; i < nd; ++i) {
    float a = ha[i], b = hb[i];

    // twoSum
    if (sum_in_domain(a, b) && sum_oracle_faithful(a, b)) {
      if ((double)hshi[i] + (double)hslo[i] != (double)a + (double)b) ++sum_fail;
    } else ++sum_skip;

    // twoProd
    if (prod_in_domain(a, b)) {
#if XPMATH_CONTRACTION_MODE == 0
      if ((double)hphi[i] + (double)hplo[i] != (double)a * (double)b) ++prod_fail;
#else
      double exact = (double)a * (double)b;
      float e_ref = (float)(exact - (double)hphi[i]);
      if (e_ref == 0.0f) { ++prod_tri; continue; }
      bool id_ok = ((double)hphi[i] + (double)hplo[i] == exact);
      if (id_ok) ++prod_cor;
      else if (hplo[i] == 0.0f) ++prod_zero;
      else ++prod_wrong;
#endif
    } else ++prod_skip;

    // twoSqr
    if (prod_in_domain(a, a)) {
#if XPMATH_CONTRACTION_MODE == 0
      if ((double)hqhi[i] + (double)hqlo[i] != (double)a * (double)a) ++sqr_fail;
#else
      double exact = (double)a * (double)a;
      float e_ref = (float)(exact - (double)hqhi[i]);
      if (e_ref == 0.0f) { ++sqr_tri; continue; }
      bool id_ok = ((double)hqhi[i] + (double)hqlo[i] == exact);
      if (id_ok) ++sqr_cor;
      else if (hqlo[i] == 0.0f) ++sqr_zero;
      else ++sqr_wrong;
#endif
    } else ++sqr_skip;
  }

  R.tested = 3 * nd;
  R.skipped = (int)(sum_skip + prod_skip + sqr_skip);
#if XPMATH_CONTRACTION_MODE == 0
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

// ----------------------------------------------------------------------------
// Baseline mechanism (ON variant only). Records the observed mismatch count so a
// LATER build with a different compiler/ISA that changes contraction behavior is
// caught as drift. WARN-only (never fails) -- the ON variant is a reporter.
//
// TF was the one backend without this. DD (T1.5), FF (T2.5) and QF all carried
// check_baseline() plus a <backend>_fma_guard_baseline.txt; TF carried neither,
// so tf_fma_guard_test_contract_on ran, printed its counts and exited 0 with
// nothing compared against anything. A reporter that reports to no one is the
// same shape of defect as a gate that cannot fail.
//
// Deliberately identical in behaviour to the other three: same file format (one
// integer on the first non-comment line), same graceful degradation when the
// file is missing or unparseable, same warn-never-fail posture. Four backends,
// one contract.
// ----------------------------------------------------------------------------
#if XPMATH_CONTRACTION_MODE == 1
#  ifdef XPMATH_BASELINE_PATH
static void check_baseline(long observed) {
  const char* path = XPMATH_BASELINE_PATH;
  std::ifstream f(path);
  if (!f) {
    std::printf("  baseline: no file at %s\n", path);
    std::printf("            record this run by writing \"%ld\" as the first\n", observed);
    std::printf("            non-comment line of that file to arm drift detection.\n");
    return;
  }
  long baseline = -1;
  std::string line;
  bool got = false;
  while (std::getline(f, line)) {
    size_t p = line.find_first_not_of(" \t");
    if (p == std::string::npos || line[p] == '#') continue;   // blank / comment
    try { baseline = std::stol(line.substr(p)); got = true; } catch (...) {}
    break;
  }
  if (!got) {
    std::printf("  baseline: %s present but unparseable; skipping drift check\n", path);
    return;
  }
  if (observed == baseline) {
    std::printf("  baseline: OK — observed mismatch count %ld matches baseline\n", observed);
  } else {
    std::printf("  baseline: *** DRIFT *** observed=%ld baseline=%ld\n", observed, baseline);
    std::printf("            contraction behavior changed since the baseline was\n");
    std::printf("            recorded (compiler/ISA/flag change). This is a WARNING,\n");
    std::printf("            not a failure. If the new value is correct, update %s.\n", path);
  }
}
#  endif
#endif

// ============================================================================
int main(int, char**) {
  int rc = 0;
  {
#if XPMATH_CONTRACTION_MODE == 0
    std::printf("=== tf_fma_guard_test: contraction OFF (gate on exact) ===\n");
#else
    std::printf("=== tf_fma_guard_test_contract_on: contraction ON (report) ===\n");
#endif
    std::printf("Oracle: FP64 (exact, 25-bit sum / 48-bit product fit in 53-bit FP64)\n");
    std::printf("TwoProduct: FMA-based (p = fl(a*b), e = fma(a,b,-p)); primitives called directly from tf_math.hpp\n\n");

    // twoSum CONTROL
    std::printf("[Control] tf_two_sum\n");
    EftCount sum_h = run_host_batches(Op::Sum, "Sum");
    std::printf("  Host: tested=%ld, skip=%ld, fail=%ld\n", sum_h.tested, sum_h.skipped, sum_h.failures);
    XPMATH_ASSERT(sum_h.failures == 0, "tf_two_sum control failed");

    // twoProd
    std::printf("\n[Test] tf_two_prod\n");
    EftCount prod_h = run_host_batches(Op::Prod, "Prod");
#if XPMATH_CONTRACTION_MODE == 0
    std::printf("  Host: tested=%ld, skip=%ld, fail=%ld\n", prod_h.tested, prod_h.skipped, prod_h.failures);
    XPMATH_ASSERT(prod_h.failures == 0, "tf_two_prod OFF not bit-exact");
#else
    std::printf("  Host: tested=%ld, trivial=%ld, correct=%ld, zero=%ld, wrong=%ld\n",
                prod_h.tested, prod_h.trivial, prod_h.correct, prod_h.zero, prod_h.wrong);
#endif

    // twoSqr
    std::printf("\n[Test] tf_two_sqr\n");
    EftCount sqr_h = run_host_batches(Op::Sqr, "Sqr");
#if XPMATH_CONTRACTION_MODE == 0
    std::printf("  Host: tested=%ld, skip=%ld, fail=%ld\n", sqr_h.tested, sqr_h.skipped, sqr_h.failures);
    XPMATH_ASSERT(sqr_h.failures == 0, "tf_two_sqr OFF not bit-exact");
#else
    std::printf("  Host: tested=%ld, trivial=%ld, correct=%ld, zero=%ld, wrong=%ld\n",
                sqr_h.tested, sqr_h.trivial, sqr_h.correct, sqr_h.zero, sqr_h.wrong);
#endif

    // Device
    std::printf("\n[Device] %s\n", xpt::where_name());
    EftCount dev = run_device_parity();
    // A nonzero vendor code is a TEST FAILURE, not a warning: a launch that
    // never ran leaves every output buffer at whatever the allocator handed
    // back, and "all zeros compared equal" is exactly how a dead device pass
    // looks from the comparison loop. Sticky since process start, so a later
    // clean call cannot erase an earlier failure.
    XPMATH_ASSERT(xpt::last_error() == 0,
                     "device harness reported a nonzero vendor error code");
#if XPMATH_CONTRACTION_MODE == 0
    std::printf("  Device: tested=%ld, skip=%d, fail=%ld\n", dev.tested, dev.skipped, dev.failures);
    XPMATH_ASSERT(dev.failures == 0, "device EFT parity mismatch");
#else
    std::printf("  Device: tested=%ld, trivial=%ld, correct=%ld, zero=%ld, wrong=%ld\n",
                dev.tested, dev.trivial, dev.correct, dev.zero, dev.wrong);
#endif

#if XPMATH_CONTRACTION_MODE == 0
    long F = prod_h.failures + sqr_h.failures + dev.failures;
    XPMATH_ASSERT(F == 0, "OFF: at least one FMA-based TwoProduct primitive broke");
    rc = ep_exit_code();
#else
    // The drift counter is the ON-mode analogue of the OFF-mode failure count:
    // zero-but-should-not-be plus nonzero-but-wrong, over both host primitives
    // and the device pass. Mirrors qf_fma_guard_test's F() = err_zero + err_wrong.
    const long F_drift = prod_h.zero + prod_h.wrong
                       + sqr_h.zero  + sqr_h.wrong
                       + dev.zero    + dev.wrong;
#  ifdef XPMATH_BASELINE_PATH
    check_baseline(F_drift);
#  endif
    long W_total = prod_h.wrong + sqr_h.wrong + dev.wrong;
    if (W_total > 0) {
      std::printf("\nERROR: ERR_NONZERO_WRONG detected (nonzero but violates TwoProduct identity)\n");
    } else {
      std::printf("\nON target: ERR_NONZERO_WRONG == 0 -> PASS\n");
    }
    // ep_exit_code() is consulted on BOTH paths now. It used to be read only
    // inside the W_total > 0 branch, so a failed XPMATH_ASSERT -- the twoSum
    // control, and as of C4 the harness-error check -- could print ASSERT FAILED
    // and still exit 0 whenever W_total happened to be zero. An assertion whose
    // entire job is to fire on a silent device has to be able to reach rc.
    // (This does not change the W_total verdict itself: W_total > 0 with no
    // failed assertion still exits 0, exactly as before. The ON target is a
    // reporter.)
    rc = ep_exit_code();
#endif

    std::printf("\n=== tf_fma_guard_test: %s ===\n", rc == 0 ? "PASS" : "FAIL");
  }
  return rc;
}
