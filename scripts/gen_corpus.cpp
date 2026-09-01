// ============================================================================
// gen_corpus.cpp — shared cross-backend validation corpus generator
// ============================================================================
//
// WHAT THIS PRODUCES
//   One binary file holding, for every operation in the union of the DD / FF /
//   QF op inventories, N input elements plus the __float128 reference value for
//   each element. Layout and loader: tests/corpus_binary.hpp.
//
// WHY IT EXISTS — COMPARABILITY, NOT SPEED
//   Today each demo (src/demo_*.cpp) generates its own inputs from its own seed
//   via its own fill_inputs(), so DD, FF and QF are scored on DIFFERENT numbers
//   and their accuracy tables are not directly comparable. A fourth backend (TF,
//   3xFP32, 72 bits) is planned and will need the same treatment. This tool
//   fixes ONE set of inputs and ONE reference per (op, element) so every backend
//   is scored against identical data, and so TF arrives with a ready-made target.
//
//   THIS IS EXPLICITLY NOT A PERFORMANCE OPTIMIZATION. Measured on this
//   hardware the __float128 oracle is about 1% of a QF demo's runtime (QF
//   transcendentals run 12000-23000 ns/op against roughly 100-200 ns for a
//   quadmath call), so caching the reference saves nothing meaningful. Anyone
//   who later reads this file as a speed feature has misread it. The value is
//   cross-backend comparability and a fixed target for new backends.
//
// BUILD (standalone host tool; NOT wired into the CMake build, same posture as
// scripts/gen_qf_constants.cpp — run from the repository root):
//
//   g++ -std=c++17 -fext-numeric-literals -O2 scripts/gen_corpus.cpp \
//       -lquadmath -o scripts/gen_corpus
//
//   (-fext-numeric-literals is required for the __float128 'Q' suffix under
//    -std=c++17; drop it if building with -std=gnu++NN. Toolchain of record is
//    gcc/13.3.0 per scripts/prepare.sh, but the source is plain C++17 and has
//    been checked to build — and to produce byte-identical output — with both
//    the system g++ 7.5 and g++ 14.)
//
// RUN
//   ./scripts/gen_corpus                          # default: n = 1,000,000 per op
//   ./scripts/gen_corpus --n 1000                 # small file for smoke tests
//   ./scripts/gen_corpus --real-only              # skip the 24 complex ops
//   ./scripts/gen_corpus --with-nan               # opt NaN inputs in
//   ./scripts/gen_corpus --verify scripts/xp_corpus.bin
//
//   Default output path is scripts/xp_corpus.bin, which .gitignore excludes.
//   The generator is committed; the corpus it emits never is.
//
// ON-DISK SIZE
//   Exactly 2280 bytes per element across all 63 ops (1048 with --real-only),
//   so the file is linear in --n:
//       --n       all 63 ops     --real-only
//       10,000        21.7 MiB        10.0 MiB
//       100,000      217.4 MiB        99.9 MiB
//       1,000,000  2,174.0 MiB (2.12 GiB)   999.5 MiB
//   Per element an op costs 16 B of __float128 reference plus 8 B per real
//   operand (32 B and 16 B respectively for complex). Reduce with --n, or with
//   --real-only: the 24 complex ops are 54% of the bytes.
//
// ============================================================================
// INPUT DISTRIBUTION
// ============================================================================
// Deliberately NOT the demos' fill_inputs(): those draw uniform-real over a
// narrow interval, which clusters every sample within a decade or two of 1.0 and
// never lands on a corner case. Each op's N elements are the union of three
// families, laid out in this fixed order so a consumer can score them
// separately (the op record carries n_corner and n_cancel):
//
//   [0, n_corner)                    (b) CORNER CASES
//   [n_corner, n_corner + n_cancel)  (c) CANCELLATION-PRONE PAIRS
//   [n_corner + n_cancel, N)         (a) LOG-UNIFORM MAGNITUDES
//
// PROPORTIONS
//   (b) corner   — every distinct corner value the op's domain admits, capped at
//                  N/2. The corner sets are fixed-size (~300 values for unary
//                  ops, ~1000 operand pairs for binary ops), so at the default
//                  N = 1,000,000 they are roughly 0.03%-0.1% of the elements:
//                  GUARANTEED PRESENT but statistically invisible in a mean.
//                  That is intentional — score them via n_corner if you want
//                  them to count for more.
//   (c) cancel   — 15% of N for binary and ternary ops; 0 for unary ops.
//   (a) loguni   — the remainder: ~85% for binary/ternary, ~100% for unary.
//
// (a) LOG-UNIFORM MAGNITUDES
//   x = +/- m * 2^e with e uniform over a per-op integer window and m uniform
//   over [1, 2). Every binade is equally likely, so all magnitudes are
//   represented instead of clustering near 1.0. Both signs, unless the op's
//   domain forbids one.
//
//   The default window is e in [-100, 100]. The upper bound is chosen so that
//   results stay inside the FP32 normal range: FF, QF and the planned TF all
//   narrow their inputs to FP32, so a corpus value above ~2^127 or below ~2^-126
//   would be an artifact of the corpus rather than a test of the backend.
//   2^100 leaves room for the op itself to grow the magnitude (hypot of two
//   2^100 operands is 2^100.5; add of two is 2^101).
//
// (b) CORNER CASES
//   Taken from tests/corpus.hpp via its accessors — not reimplemented here. The
//   accessors are precision-parametric, and BOTH instantiations are used:
//     corpus::...<float>()   widened to double — the FP32-world corners that
//                            matter to FF / QF / TF (FP32 denormals, 2^+-126,
//                            FLT_MAX, float-ulp neighbors)
//     corpus::...<double>()  the FP64-world corners that matter to DD
//   deduplicated by bit pattern, insertion order preserved.
//
//   Unary ops draw from zeros, infinities, subnormals, powers_of_two,
//   nextafter_neighbors, finite_specials plus the PORT_NOTES regression families
//   corpus::unary() bundles in (exp_overflow, nint_half_integer, atanh_small,
//   sinh_cosh_small, trig_near_pi). Binary ops additionally draw from
//   near_cancellation, huge_tiny and remainder_regression via corpus::binary(),
//   and from a deterministic stride-pairing of the unary corner list with itself
//   so every unary corner appears in both operand slots. Ternary (fma) extends
//   the binary pairs with a third operand chosen to cancel the product.
//
//   NaN is opt-in in corpus.hpp (CorpusFlags::include_nan) and is OFF by
//   default here; --with-nan turns it on and sets kFlagWithNan in the header.
//
// (c) CANCELLATION-PRONE PAIRS
//   b = +/- a * (1 + eps) with eps = +/- 2^-k, k uniform over [1, 53] — so the
//   relative separation spans a few ulps up to order 1, as the corpus.hpp
//   near_cancellation family does for its fixed anchors but across the op's full
//   magnitude window. The sign is chosen so the op's own cancelling combination
//   is the one exercised: b = -a(1+eps) for add, b = +a(1+eps) for sub / fdim /
//   fmod / remainder / hypot. For mul, div, pow, copysign, fmax and fmin
//   "cancellation" has no meaning, but near-equal operands are still a corner
//   (fmax/fmin tie-breaks, div and pow near 1, fmod near 0), so those get
//   b = a(1+eps) too. fma gets c = -(a*b)(1+eps).
//
// ============================================================================
// PER-OP DOMAIN POLICY
// ============================================================================
// A corpus element whose reference is NaN teaches nothing — every backend
// "matches" or every backend fails, depending on the scorer. So inputs are
// constrained to each op's mathematical domain. Two mechanisms, applied in this
// order to every element regardless of which family it came from:
//
//   1. REPAIR (deterministic, documented, applied first). Per op:
//        sqrt, log, log2, log10   a <- |a|.  a = 0 is KEPT: log(0) = -inf is a
//                                 legitimate limit, not an error.
//        log1p                    a < -1 -> a <- 1/a, which lands in (-1, 0) and
//                                 preserves the sign. a = -1 is kept (-inf).
//        sin, cos, tan            |a| = inf -> a <- +-1 (no limit at infinity).
//        asin, acos, atanh        |a| > 1 -> a <- 1/a (maps outside the domain
//                                 back inside, sign-preserving; +-inf -> 0).
//                                 |a| = 1 is kept: atanh(+-1) = +-inf.
//        acosh                    a < 1 -> a <- 1 + |a|.
//        pow                      a <- |a| (negative base with a non-integer
//                                 exponent is NaN); a = 0 -> a = 1 (0^negative
//                                 is inf, 0^0 is 1 — both defined, but the
//                                 whole column would be degenerate). Then b is
//                                 rescaled so |b * log2(a)| <= 120, keeping the
//                                 result inside the FP32 range every backend
//                                 must be able to represent.
//        fmod, remainder          b = 0 -> b = 1 (NaN); |a| = inf -> a = 0 (NaN).
//        add, sub                 inf +- inf of the NaN-producing sign -> b = 1.
//        mul                      0 * inf -> the zero operand becomes 1.
//        div                      0/0 and inf/inf -> b = 1.
//        fdim                     fdim(inf, inf) -> b = 0.
//        fma                      the inf/0 product and inf-inf sum cases.
//        complex div, log, log10  z = 0 -> z = 1.
//        complex exp,             the component that acts as the trigonometric
//        sin, cos, tan,           ANGLE (imaginary part for exp/sinh/cosh/tanh,
//        sinh, cosh, tanh         real part for sin/cos/tan): +-inf -> +-1.
//        complex pow              infinite components -> +-1; base 0 -> 1;
//                                 exponent rescaled as real pow.
//        complex atan             z = +-i    -> real part nudged off the pole.
//        complex atanh            z = +-1    -> imag part nudged off the pole.
//        complex polar            operand a is (r, theta): both +-inf -> +-1
//                                 (infinite angle, and inf * sin(0) = NaN).
//
//   2. RESAMPLE SAFETY NET (fires zero times as shipped — checked at --n 20000,
//      which saturates the corner pool, for the default, --with-nan,
//      --real-only and --complex-only; counted and reported if it ever does). After the reference is computed, an element whose reference is NaN
//      while --with-nan is off is redrawn from the op's log-uniform family, up
//      to 8 times. This is a backstop for a domain case the explicit repairs
//      missed — the per-op "resampled" count in the run summary is the evidence
//      that the repairs are complete. A nonzero count means a repair is missing,
//      not that the corpus is wrong.
//
// Ops whose reference can legitimately be +-inf (log(0), atanh(+-1),
// exp(DBL_MAX), pow overflow at a corner input) keep it. Infinite references are
// meaningful — a backend either reproduces the overflow or it does not.
//
// PER-OP LOG-UNIFORM EXPONENT WINDOWS (family (a)); rationale for the narrow
// ones is FP32 representability of the RESULT, so no backend is scored on an
// answer it structurally cannot hold:
//   exp, exp2, sinh, cosh        [-40,  5]  |x| <= 64 -> e^64 = 6e27 < FLT_MAX
//   exp10                        [-30,  4]  |x| <= 32 -> 1e32   < FLT_MAX
//   expm1                        [-60,  3]
//   sin, cos, tan                [-60,  8]  |x| <= 512, ~80 periods; beyond that
//                                           FP32 argument reduction is measuring
//                                           the input's own ulp, not the kernel
//   tanh                         [-60,  5]
//   asin, acos, atanh            [-100,-1]  |x| < 1 by construction, plus a 25%
//                                           enrichment at x = +-(1 - 2^-k),
//                                           k in [1,53], for the ill-conditioned
//                                           edge those three all have
//   acosh                        1 + m*2^e, e in [-60, 100]
//   ceil, floor, round, trunc    [-10, 53]  above 2^53 they are the identity
//   mul, div, fma(a,b)           [-50, 50]  product/quotient stays under 2^101
//   pow exponent b               [-10,  6]  then clamped (see repair)
//   everything else              [-100,100]
//   complex ops                  analogous, with the real and imaginary parts
//                                windowed separately (an imaginary part is an
//                                angle for exp/sin/cos, a growth exponent for
//                                sinh/cosh) — see kComplex below.
//
// ============================================================================
// DETERMINISM
// ============================================================================
// Same seed + same options => byte-identical file. Guaranteed by:
//   * std::mt19937_64 only, explicitly seeded; nothing time-, address- or
//     hardware-dependent anywhere in the tool.
//   * NO std::uniform_real_distribution / std::uniform_int_distribution — their
//     output is implementation-defined and would make the file compiler- and
//     libstdc++-version-dependent. Mantissas and exponents are derived from the
//     raw 64-bit engine output by fixed arithmetic (see struct Rng).
//   * Each op gets its OWN stream, seeded by splitmix64(seed ^ FNV(op name)).
//     Adding, removing or reordering an op therefore cannot perturb any other
//     op's data.
//   * No unordered containers on any path that affects output ordering.
//   Caveat: the REFERENCE values come from libquadmath, so byte-identity is
//   guaranteed for a given libquadmath. A libquadmath upgrade that changes a
//   transcendental by an ulp would change the file — which is exactly what the
//   generator_version staleness guard is for: bump kGeneratorVersion in
//   tests/corpus_binary.hpp and every consumer will refuse the old file.
//
// ============================================================================

#include "../tests/corpus_binary.hpp"
#include "../tests/corpus.hpp"

#include <quadmath.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cb = kokkos_ep::corpus_binary;
namespace cc = kokkos_ep::corpus;

namespace {

const char* const kDefaultOut = "scripts/xp_corpus.bin";
const uint64_t    kDefaultSeed = 12345ull;   // same default as the demos
const uint64_t    kDefaultN    = 1000000ull;

// Fraction of a binary/ternary op's elements devoted to family (c).
const double kCancelFraction = 0.15;

// ---------------------------------------------------------------------------
// Deterministic sampling primitives (see DETERMINISM above).
// ---------------------------------------------------------------------------
uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

uint64_t fnv1a(const char* s) {
  uint64_t h = cb::kChecksumSeed;
  for (; *s; ++s) { h ^= uint64_t((unsigned char)*s); h *= cb::kChecksumPrime; }
  return h;
}

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t s) : g(s) {}

  uint64_t next() { return g(); }
  // [0,1) from the top 53 bits — no std distribution, so no libstdc++ dependence.
  double unit() { return double(g() >> 11) * (1.0 / 9007199254740992.0); }
  int    in(int lo, int hi) { return lo + int(g() % uint64_t(hi - lo + 1)); }
  bool   coin() { return (g() & 1ull) != 0ull; }

  // m * 2^e, m uniform in [1,2), e uniform in [lo,hi]. Positive.
  double logunif(int lo, int hi) { return std::ldexp(1.0 + unit(), in(lo, hi)); }
  double slogunif(int lo, int hi) { const double v = logunif(lo, hi); return coin() ? v : -v; }
};

// ---------------------------------------------------------------------------
// Op inventory. Re-derived from src/demo_qf_real.cpp (39 real ops) and
// src/demo_qf_complex.cpp (24 complex ops); DD and FF expose the same sets.
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

struct RealSpec {
  const char* name;
  int         nops;              // 1, 2 or 3
  int         a_lo, a_hi;        // log-uniform exponent window for operand a
  int         b_lo, b_hi;        // ... for operand b (ignored when nops < 2)
};

// clang-format off
const RealSpec kReal[R_COUNT] = {
  /* add       */ { "add",       2, -100, 100, -100, 100 },
  /* sub       */ { "sub",       2, -100, 100, -100, 100 },
  /* mul       */ { "mul",       2,  -50,  50,  -50,  50 },
  /* div       */ { "div",       2,  -50,  50,  -50,  50 },
  /* sqrt      */ { "sqrt",      1, -100, 100,    0,   0 },
  /* abs       */ { "abs",       1, -100, 100,    0,   0 },
  /* exp       */ { "exp",       1,  -40,   5,    0,   0 },
  /* log       */ { "log",       1, -100, 100,    0,   0 },
  /* exp2      */ { "exp2",      1,  -40,   5,    0,   0 },
  /* exp10     */ { "exp10",     1,  -30,   4,    0,   0 },
  /* expm1     */ { "expm1",     1,  -60,   3,    0,   0 },
  /* log2      */ { "log2",      1, -100, 100,    0,   0 },
  /* log10     */ { "log10",     1, -100, 100,    0,   0 },
  /* log1p     */ { "log1p",     1, -100, 100,    0,   0 },
  /* sin       */ { "sin",       1,  -60,   8,    0,   0 },
  /* cos       */ { "cos",       1,  -60,   8,    0,   0 },
  /* tan       */ { "tan",       1,  -60,   8,    0,   0 },
  /* asin      */ { "asin",      1, -100,  -1,    0,   0 },
  /* acos      */ { "acos",      1, -100,  -1,    0,   0 },
  /* atan      */ { "atan",      1, -100, 100,    0,   0 },
  /* sinh      */ { "sinh",      1,  -40,   5,    0,   0 },
  /* cosh      */ { "cosh",      1,  -40,   5,    0,   0 },
  /* tanh      */ { "tanh",      1,  -60,   5,    0,   0 },
  /* acosh     */ { "acosh",     1,  -60, 100,    0,   0 },
  /* asinh     */ { "asinh",     1, -100, 100,    0,   0 },
  /* atanh     */ { "atanh",     1, -100,  -1,    0,   0 },
  /* pow       */ { "pow",       2, -100, 100,  -10,   6 },
  /* hypot     */ { "hypot",     2, -100, 100, -100, 100 },
  /* fmod      */ { "fmod",      2, -100, 100, -100, 100 },
  /* remainder */ { "remainder", 2, -100, 100, -100, 100 },
  /* copysign  */ { "copysign",  2, -100, 100, -100, 100 },
  /* fmax      */ { "fmax",      2, -100, 100, -100, 100 },
  /* fmin      */ { "fmin",      2, -100, 100, -100, 100 },
  /* fdim      */ { "fdim",      2, -100, 100, -100, 100 },
  /* fma       */ { "fma",       3,  -50,  50,  -50,  50 },
  /* ceil      */ { "ceil",      1,  -10,  53,    0,   0 },
  /* floor     */ { "floor",     1,  -10,  53,    0,   0 },
  /* round     */ { "round",     1,  -10,  53,    0,   0 },
  /* trunc     */ { "trunc",     1,  -10,  53,    0,   0 },
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

struct ComplexSpec {
  const char* name;
  int         nops;                // 1 or 2
  int         re_lo, re_hi;        // window for the real part of operand a
  int         im_lo, im_hi;        // window for the imaginary part of operand a
  int         b_re_lo, b_re_hi;    // operand b (pow's exponent has its own)
  int         b_im_lo, b_im_hi;
};

// clang-format off
const ComplexSpec kComplex[C_COUNT] = {
  /* add   */ { "add",   2,  -50,  50,  -50,  50,  -50,  50,  -50,  50 },
  /* sub   */ { "sub",   2,  -50,  50,  -50,  50,  -50,  50,  -50,  50 },
  /* mul   */ { "mul",   2,  -40,  40,  -40,  40,  -40,  40,  -40,  40 },
  /* div   */ { "div",   2,  -40,  40,  -40,  40,  -40,  40,  -40,  40 },
  /* abs   */ { "abs",   1, -100, 100, -100, 100,    0,   0,    0,   0 },
  /* conj  */ { "conj",  1, -100, 100, -100, 100,    0,   0,    0,   0 },
  /* sqrt  */ { "sqrt",  1, -100, 100, -100, 100,    0,   0,    0,   0 },
  /* exp   */ { "exp",   1,  -40,   5,  -60,   4,    0,   0,    0,   0 },
  /* log   */ { "log",   1, -100, 100, -100, 100,    0,   0,    0,   0 },
  /* log10 */ { "log10", 1, -100, 100, -100, 100,    0,   0,    0,   0 },
  /* sin   */ { "sin",   1,  -60,   6,  -40,   4,    0,   0,    0,   0 },
  /* cos   */ { "cos",   1,  -60,   6,  -40,   4,    0,   0,    0,   0 },
  /* tan   */ { "tan",   1,  -60,   6,  -40,   4,    0,   0,    0,   0 },
  /* asin  */ { "asin",  1,  -60,  20,  -60,  20,    0,   0,    0,   0 },
  /* acos  */ { "acos",  1,  -60,  20,  -60,  20,    0,   0,    0,   0 },
  /* atan  */ { "atan",  1,  -60,  20,  -60,  20,    0,   0,    0,   0 },
  /* sinh  */ { "sinh",  1,  -40,   4,  -60,   6,    0,   0,    0,   0 },
  /* cosh  */ { "cosh",  1,  -40,   4,  -60,   6,    0,   0,    0,   0 },
  /* tanh  */ { "tanh",  1,  -40,   4,  -60,   6,    0,   0,    0,   0 },
  /* asinh */ { "asinh", 1,  -60,  20,  -60,  20,    0,   0,    0,   0 },
  /* acosh */ { "acosh", 1,  -60,  20,  -60,  20,    0,   0,    0,   0 },
  /* atanh */ { "atanh", 1,  -60,  20,  -60,  20,    0,   0,    0,   0 },
  /* pow   */ { "pow",   2,  -40,  40,  -40,  40,  -10,   4,  -10,   4 },
  /* polar */ { "polar", 1,  -60,  60,  -60,   3,    0,   0,    0,   0 },
};
// clang-format on

// ---------------------------------------------------------------------------
// Corner-case harvesting from tests/corpus.hpp.
//
// Both instantiations (<float> widened to double, and <double>) are collected so
// the FP32-world corners that matter to FF/QF/TF and the FP64-world corners that
// matter to DD are both present. Dedup is by bit pattern; insertion order is
// preserved so the output stays deterministic.
// ---------------------------------------------------------------------------
uint64_t bits_of(double x) { uint64_t b; std::memcpy(&b, &x, 8); return b; }

void push_unique(std::vector<double>& out, std::set<uint64_t>& seen, double x) {
  if (seen.insert(bits_of(x)).second) out.push_back(x);
}

std::vector<double> unary_corners(bool with_nan) {
  cc::CorpusFlags flags;
  flags.include_nan = with_nan;
  std::vector<double> out;
  std::set<uint64_t>  seen;
  for (float v : cc::unary<float>(flags))   push_unique(out, seen, double(v));
  for (double v : cc::unary<double>(flags)) push_unique(out, seen, v);
  return out;
}

std::vector<std::pair<double, double> > binary_corners(bool with_nan) {
  cc::CorpusFlags flags;
  flags.include_nan = with_nan;
  std::vector<std::pair<double, double> > out;
  std::set<std::pair<uint64_t, uint64_t> > seen;
  auto push = [&](double a, double b) {
    if (seen.insert(std::make_pair(bits_of(a), bits_of(b))).second)
      out.push_back(std::make_pair(a, b));
  };
  for (const std::pair<float, float>& p : cc::binary<float>(flags))
    push(double(p.first), double(p.second));
  for (const std::pair<double, double>& p : cc::binary<double>(flags))
    push(p.first, p.second);

  // Stride-pair the unary corner list with itself so every unary corner appears
  // in BOTH operand slots. Stride 7 is coprime with any plausible list length,
  // so the pairing is a permutation, not a fixed point.
  const std::vector<double> u = unary_corners(with_nan);
  const size_t              n = u.size();
  for (size_t i = 0; i < n; ++i) {
    const double partner = u[(i * 7 + 3) % n];
    push(u[i], partner);
    push(partner, u[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Per-op domain repair (mechanism 1 of the domain policy; see header comment).
// ---------------------------------------------------------------------------
bool is_inf(double x) { return std::isinf(x); }

// Map +-inf onto +-1: an in-domain, sign-preserving, deterministic stand-in for
// the ops where an infinite argument is simply undefined (sin/cos/tan have no
// limit at infinity; complex pow of an infinite base or exponent is NaN).
double tame_inf(double x) { return std::isinf(x) ? std::copysign(1.0, x) : x; }

void repair_real(int id, double& a, double& b, double& c) {
  using std::fabs;
  switch (id) {
    case R_Add:
      if (is_inf(a) && is_inf(b) && ((a > 0) != (b > 0))) b = 1.0;   // inf + (-inf)
      break;
    case R_Sub:
      if (is_inf(a) && is_inf(b) && ((a > 0) == (b > 0))) b = 1.0;   // inf - inf
      break;
    case R_Mul:
      if (a == 0.0 && is_inf(b)) b = 1.0;                            // 0 * inf
      if (is_inf(a) && b == 0.0) a = 1.0;
      break;
    case R_Div:
      if ((a == 0.0 && b == 0.0) || (is_inf(a) && is_inf(b))) b = 1.0;
      break;
    case R_Sqrt: case R_Log: case R_Log2: case R_Log10:
      a = fabs(a);                                                   // 0 -> -inf is kept
      break;
    case R_Log1p:
      if (a < -1.0) a = 1.0 / a;                                     // back into (-1, 0)
      break;
    case R_Sin: case R_Cos: case R_Tan:
      a = tame_inf(a);                                               // sin(inf) is NaN
      break;
    case R_Asin: case R_Acos: case R_Atanh:
      if (fabs(a) > 1.0) a = 1.0 / a;                                // +-inf -> +-0
      break;
    case R_Acosh:
      if (!(a >= 1.0)) a = 1.0 + fabs(a);
      break;
    case R_Pow: {
      a = fabs(a);
      if (a == 0.0) a = 1.0;
      if (a != 1.0 && std::isfinite(a) && std::isfinite(b) && b != 0.0) {
        const double mag = fabs(b * std::log2(a));
        if (mag > 120.0) b *= 120.0 / mag;   // keep |a^b| inside the FP32 range
      }
      break;
    }
    case R_Fmod: case R_Remainder:
      if (b == 0.0) b = 1.0;
      if (is_inf(a)) a = 0.0;
      break;
    case R_Fdim:
      if (is_inf(a) && is_inf(b) && ((a > 0) == (b > 0))) b = 0.0;
      break;
    case R_Fma:
      if (a == 0.0 && is_inf(b)) b = 1.0;
      if (is_inf(a) && b == 0.0) a = 1.0;
      if ((is_inf(a) || is_inf(b)) && is_inf(c)) c = 0.0;
      break;
    default:
      break;
  }
}

void repair_complex(int id, double& are, double& aim, double& bre, double& bim) {
  using std::fabs;
  switch (id) {
    case C_Div:
      if (bre == 0.0 && bim == 0.0) bre = 1.0;
      break;
    case C_Log: case C_Log10:
      if (are == 0.0 && aim == 0.0) are = 1.0;
      break;
    // An infinite ANGLE is NaN for the same reason real sin(inf) is: the
    // trigonometric argument is the real part for sin/cos/tan and the imaginary
    // part for their hyperbolic counterparts (and for exp).
    case C_Exp:
      aim = tame_inf(aim);
      break;
    case C_Sin: case C_Cos: case C_Tan:
      are = tame_inf(are);
      break;
    case C_Sinh: case C_Cosh: case C_Tanh:
      aim = tame_inf(aim);
      break;
    case C_Polar:
      // Operand a is (r, theta). An infinite angle is NaN as above; an infinite
      // modulus is NaN too whenever the sine or cosine is exactly zero
      // (inf * 0), which theta = 0 hits, so both components are tamed.
      are = tame_inf(are);
      aim = tame_inf(aim);
      break;
    case C_Atan:
      if (are == 0.0 && fabs(aim) == 1.0) are = 0.5;          // z = +-i is a pole
      break;
    case C_Atanh:
      if (aim == 0.0 && fabs(are) == 1.0) aim = 0.5;          // z = +-1 is a pole
      break;
    case C_Pow: {
      are = tame_inf(are); aim = tame_inf(aim);
      bre = tame_inf(bre); bim = tame_inf(bim);
      if (are == 0.0 && aim == 0.0) are = 1.0;
      const double mod = std::hypot(are, aim);
      if (std::isfinite(mod) && mod > 0.0 && mod != 1.0 &&
          std::isfinite(bre) && std::isfinite(bim)) {
        const double la  = std::log2(mod);
        const double mag = std::hypot(bre, bim) * fabs(la);
        if (mag > 120.0) { const double s = 120.0 / mag; bre *= s; bim *= s; }
      }
      break;
    }
    default:
      break;
  }
}

// ---------------------------------------------------------------------------
// Random sampling (family (a)) and cancellation sampling (family (c)).
// ---------------------------------------------------------------------------
struct Triple { double a, b, c; };

Triple sample_real_random(int id, Rng& rng) {
  const RealSpec& s = kReal[id];
  Triple t; t.a = 0.0; t.b = 0.0; t.c = 0.0;

  switch (id) {
    case R_Asin: case R_Acos: case R_Atanh:
      // 25% enrichment at the ill-conditioned |x| -> 1 edge; otherwise
      // log-uniform inside the unit interval.
      if ((rng.next() & 3ull) == 0ull) {
        const double v = 1.0 - std::ldexp(1.0, -rng.in(1, 53));
        t.a = rng.coin() ? v : -v;
      } else {
        t.a = rng.slogunif(s.a_lo, s.a_hi);
      }
      break;
    case R_Acosh:
      t.a = 1.0 + rng.logunif(s.a_lo, s.a_hi);
      break;
    case R_Log1p:
      // Positive side spans the full window; negative side is confined to
      // (-1, 0) so the value stays in the domain without a repair.
      t.a = rng.coin() ? rng.logunif(s.a_lo, s.a_hi) : -rng.logunif(s.a_lo, -1);
      break;
    case R_Sqrt: case R_Log: case R_Log2: case R_Log10:
      t.a = rng.logunif(s.a_lo, s.a_hi);   // positive by domain
      break;
    default:
      t.a = rng.slogunif(s.a_lo, s.a_hi);
      break;
  }

  if (s.nops >= 2) t.b = rng.slogunif(s.b_lo, s.b_hi);
  if (s.nops >= 3) t.c = rng.slogunif(-100, 100);
  return t;
}

Triple sample_real_cancel(int id, Rng& rng) {
  const RealSpec& s = kReal[id];
  Triple t = sample_real_random(id, rng);
  // eps = +- 2^-k, k in [1, 53]: a few ulps of separation up to order 1.
  const double eps = std::ldexp(1.0, -rng.in(1, 53)) * (rng.coin() ? 1.0 : -1.0);
  if (s.nops == 3) {                       // fma: make c cancel the product
    t.c = -(t.a * t.b) * (1.0 + eps);
  } else if (id == R_Add) {                // a + b cancels when b ~ -a
    t.b = -t.a * (1.0 + eps);
  } else {
    t.b = t.a * (1.0 + eps);
  }
  return t;
}

struct Quad { double are, aim, bre, bim; };

Quad sample_complex_random(int id, Rng& rng) {
  const ComplexSpec& s = kComplex[id];
  Quad q; q.are = 0.0; q.aim = 0.0; q.bre = 0.0; q.bim = 0.0;
  if (id == C_Polar) {
    q.are = rng.logunif(s.re_lo, s.re_hi);      // modulus r >= 0
    q.aim = rng.slogunif(s.im_lo, s.im_hi);     // angle theta
    return q;
  }
  q.are = rng.slogunif(s.re_lo, s.re_hi);
  q.aim = rng.slogunif(s.im_lo, s.im_hi);
  if (s.nops >= 2) {
    q.bre = rng.slogunif(s.b_re_lo, s.b_re_hi);
    q.bim = rng.slogunif(s.b_im_lo, s.b_im_hi);
  }
  return q;
}

Quad sample_complex_cancel(int id, Rng& rng) {
  Quad q = sample_complex_random(id, rng);
  const double eps = std::ldexp(1.0, -rng.in(1, 53)) * (rng.coin() ? 1.0 : -1.0);
  const double s   = (id == C_Add) ? -(1.0 + eps) : (1.0 + eps);
  q.bre = q.are * s;
  q.bim = q.aim * s;
  return q;
}

// ---------------------------------------------------------------------------
// __float128 references (quadmath). Same op semantics as the demos'
// host_quadmath_reference(), expressed with the bare quadmath entry points
// because this tool does not link Kokkos.
// ---------------------------------------------------------------------------
__float128 reference_real(int id, double da, double db, double dc) {
  const __float128 a = (__float128)da, b = (__float128)db, c = (__float128)dc;
  switch (id) {
    case R_Add:       return a + b;
    case R_Sub:       return a - b;
    case R_Mul:       return a * b;
    case R_Div:       return a / b;
    case R_Sqrt:      return sqrtq(a);
    case R_Abs:       return fabsq(a);
    case R_Exp:       return expq(a);
    case R_Log:       return logq(a);
    // powq, not exp2q/exp10q: exp10q does not exist and exp2q is absent from
    // older libquadmath (gcc 7). Using powq unconditionally keeps the file
    // byte-identical across the toolchains this repo builds with.
    case R_Exp2:      return powq((__float128)2.0, a);
    case R_Exp10:     return powq((__float128)10.0, a);
    case R_Expm1:     return expm1q(a);
    case R_Log2:      return log2q(a);
    case R_Log10:     return log10q(a);
    case R_Log1p:     return log1pq(a);
    case R_Sin:       return sinq(a);
    case R_Cos:       return cosq(a);
    case R_Tan:       return tanq(a);
    case R_Asin:      return asinq(a);
    case R_Acos:      return acosq(a);
    case R_Atan:      return atanq(a);
    case R_Sinh:      return sinhq(a);
    case R_Cosh:      return coshq(a);
    case R_Tanh:      return tanhq(a);
    case R_Acosh:     return acoshq(a);
    case R_Asinh:     return asinhq(a);
    case R_Atanh:     return atanhq(a);
    case R_Pow:       return powq(a, b);
    case R_Hypot:     return hypotq(a, b);
    case R_Fmod:      return fmodq(a, b);
    case R_Remainder: return remainderq(a, b);
    case R_Copysign:  return copysignq(a, b);
    case R_Fmax:      return fmaxq(a, b);
    case R_Fmin:      return fminq(a, b);
    case R_Fdim:      return fdimq(a, b);
    case R_Fma:       return fmaq(a, b, c);
    case R_Ceil:      return ceilq(a);
    case R_Floor:     return floorq(a);
    case R_Round:     return roundq(a);
    case R_Trunc:     return truncq(a);
  }
  return (__float128)0.0;
}

void reference_complex(int id, double are, double aim, double bre, double bim,
                       __float128& out_re, __float128& out_im) {
  __complex128 za; __real__ za = (__float128)are; __imag__ za = (__float128)aim;
  __complex128 zb; __real__ zb = (__float128)bre; __imag__ zb = (__float128)bim;
  __complex128 r;

  switch (id) {
    case C_Add:   r = za + zb;      break;
    case C_Sub:   r = za - zb;      break;
    case C_Mul:   r = za * zb;      break;
    case C_Div:   r = za / zb;      break;
    case C_Abs:   out_re = cabsq(za); out_im = (__float128)0.0; return;
    case C_Conj:  r = conjq(za);    break;
    case C_Sqrt:  r = csqrtq(za);   break;
    case C_Exp:   r = cexpq(za);    break;
    case C_Log:   r = clogq(za);    break;
    case C_Log10: r = clog10q(za);  break;
    case C_Sin:   r = csinq(za);    break;
    case C_Cos:   r = ccosq(za);    break;
    case C_Tan:   r = ctanq(za);    break;
    case C_Asin:  r = casinq(za);   break;
    case C_Acos:  r = cacosq(za);   break;
    case C_Atan:  r = catanq(za);   break;
    case C_Sinh:  r = csinhq(za);   break;
    case C_Cosh:  r = ccoshq(za);   break;
    case C_Tanh:  r = ctanhq(za);   break;
    case C_Asinh: r = casinhq(za);  break;
    case C_Acosh: r = cacoshq(za);  break;
    case C_Atanh: r = catanhq(za);  break;
    case C_Pow:   r = cpowq(za, zb); break;
    case C_Polar: {
      // Operand a carries (r, theta), matching demo_qf_complex.cpp's polar().
      const __float128 rad = (__float128)are, th = (__float128)aim;
      out_re = rad * cosq(th);
      out_im = rad * sinq(th);
      return;
    }
    default: r = (__complex128)0; break;
  }
  out_re = crealq(r);
  out_im = cimagq(r);
}

// ---------------------------------------------------------------------------
// Block generation.
// ---------------------------------------------------------------------------
struct Options {
  uint64_t    n        = kDefaultN;
  uint64_t    seed     = kDefaultSeed;
  bool        with_nan = false;
  bool        do_real  = true;
  bool        do_cplx  = true;
  std::string out      = kDefaultOut;
  std::string verify;                 // non-empty => verify mode
};

struct BlockStats {
  uint64_t n_corner   = 0;
  uint64_t n_cancel   = 0;
  uint64_t resampled  = 0;   // NaN safety net firings — expected to be 0
  uint64_t nonfinite  = 0;   // references that are +-inf (legitimate)
};

uint64_t stream_seed(uint64_t base, const char* name, uint8_t kind) {
  return splitmix64(base ^ fnv1a(name) ^ (uint64_t(kind) * 0x9E3779B97F4A7C15ull));
}

// Assemble one real op's data block: ref[n] then a[n] [b[n]] [c[n]], padded.
std::vector<unsigned char> build_real_block(int id, const Options& opt,
                                            const std::vector<double>& ucorner,
                                            const std::vector<std::pair<double, double> >& bcorner,
                                            BlockStats& st) {
  const RealSpec& s = kReal[id];
  const uint64_t  n = opt.n;
  Rng             rng(stream_seed(opt.seed, s.name, cb::kKindReal));

  const uint64_t corner_avail = (s.nops == 1) ? ucorner.size() : bcorner.size();
  const uint64_t corner_cap   = (n < 2) ? n : n / 2;
  const uint64_t n_corner     = corner_avail < corner_cap ? corner_avail : corner_cap;
  uint64_t n_cancel = 0;
  if (s.nops >= 2) {
    n_cancel = uint64_t(double(n) * kCancelFraction);
    if (n_corner + n_cancel > n) n_cancel = n - n_corner;
  }
  st.n_corner = n_corner;
  st.n_cancel = n_cancel;

  std::vector<double>     a(n), b(s.nops >= 2 ? n : 0), c(s.nops >= 3 ? n : 0);
  std::vector<__float128> ref(n);

  for (uint64_t i = 0; i < n; ++i) {
    Triple t;
    if (i < n_corner) {
      if (s.nops == 1) { t.a = ucorner[i]; t.b = 0.0; t.c = 0.0; }
      else {
        t.a = bcorner[i].first;
        t.b = bcorner[i].second;
        // fma's third operand cancels the product, the hardest case for it.
        t.c = (s.nops >= 3) ? -(t.a * t.b) : 0.0;
      }
    } else if (i < n_corner + n_cancel) {
      t = sample_real_cancel(id, rng);
    } else {
      t = sample_real_random(id, rng);
    }

    repair_real(id, t.a, t.b, t.c);
    __float128 r = reference_real(id, t.a, t.b, t.c);

    // Safety net: an element whose reference is NaN teaches nothing.
    for (int attempt = 0; attempt < 8 && isnanq(r) && !opt.with_nan; ++attempt) {
      t = sample_real_random(id, rng);
      repair_real(id, t.a, t.b, t.c);
      r = reference_real(id, t.a, t.b, t.c);
      ++st.resampled;
    }
    if (!finiteq(r) && !isnanq(r)) ++st.nonfinite;

    a[i] = t.a;
    if (s.nops >= 2) b[i] = t.b;
    if (s.nops >= 3) c[i] = t.c;
    ref[i] = r;
  }

  std::vector<unsigned char> blk(cb::block_bytes(cb::kKindReal, uint8_t(s.nops), n), 0);
  size_t off = 0;
  std::memcpy(blk.data() + off, ref.data(), 16 * size_t(n)); off += 16 * size_t(n);
  std::memcpy(blk.data() + off, a.data(), 8 * size_t(n));    off += 8 * size_t(n);
  if (s.nops >= 2) { std::memcpy(blk.data() + off, b.data(), 8 * size_t(n)); off += 8 * size_t(n); }
  if (s.nops >= 3) { std::memcpy(blk.data() + off, c.data(), 8 * size_t(n)); off += 8 * size_t(n); }
  return blk;
}

// Assemble one complex op's data block: ref[2n] then a[2n] [b[2n]], padded.
std::vector<unsigned char> build_complex_block(int id, const Options& opt,
                                               const std::vector<double>& ucorner,
                                               BlockStats& st) {
  const ComplexSpec& s = kComplex[id];
  const uint64_t     n = opt.n;
  Rng                rng(stream_seed(opt.seed, s.name, cb::kKindComplex));

  // Complex corners: put each real corner on the real axis, the imaginary axis
  // and both diagonals — that is where the branch cuts and the sign-of-zero
  // conventions live.
  std::vector<std::pair<double, double> > zc;
  zc.reserve(ucorner.size() * 4);
  for (size_t i = 0; i < ucorner.size(); ++i) {
    const double v = ucorner[i];
    zc.push_back(std::make_pair(v, 0.0));
    zc.push_back(std::make_pair(0.0, v));
    zc.push_back(std::make_pair(v, v));
    zc.push_back(std::make_pair(v, -v));
  }

  const uint64_t corner_cap = (n < 2) ? n : n / 2;
  const uint64_t n_corner   = zc.size() < corner_cap ? zc.size() : corner_cap;
  uint64_t       n_cancel   = 0;
  if (s.nops >= 2) {
    n_cancel = uint64_t(double(n) * kCancelFraction);
    if (n_corner + n_cancel > n) n_cancel = n - n_corner;
  }
  st.n_corner = n_corner;
  st.n_cancel = n_cancel;

  std::vector<double>     za(2 * n), zb(s.nops >= 2 ? 2 * n : 0);
  std::vector<__float128> ref(2 * n);

  for (uint64_t i = 0; i < n; ++i) {
    Quad q;
    if (i < n_corner) {
      q.are = zc[i].first;
      q.aim = zc[i].second;
      if (s.nops >= 2) {
        // Stride-pair the corner list against itself for the second operand.
        const std::pair<double, double>& p = zc[(size_t(i) * 7 + 3) % zc.size()];
        q.bre = p.first;
        q.bim = p.second;
      } else {
        q.bre = 0.0; q.bim = 0.0;
      }
    } else if (i < n_corner + n_cancel) {
      q = sample_complex_cancel(id, rng);
    } else {
      q = sample_complex_random(id, rng);
    }

    repair_complex(id, q.are, q.aim, q.bre, q.bim);
    __float128 rr, ri;
    reference_complex(id, q.are, q.aim, q.bre, q.bim, rr, ri);

    for (int attempt = 0; attempt < 8 && (isnanq(rr) || isnanq(ri)) && !opt.with_nan; ++attempt) {
      q = sample_complex_random(id, rng);
      repair_complex(id, q.are, q.aim, q.bre, q.bim);
      reference_complex(id, q.are, q.aim, q.bre, q.bim, rr, ri);
      ++st.resampled;
    }
    if ((!finiteq(rr) && !isnanq(rr)) || (!finiteq(ri) && !isnanq(ri))) ++st.nonfinite;

    za[2*i] = q.are; za[2*i+1] = q.aim;
    if (s.nops >= 2) { zb[2*i] = q.bre; zb[2*i+1] = q.bim; }
    ref[2*i] = rr; ref[2*i+1] = ri;
  }

  std::vector<unsigned char> blk(cb::block_bytes(cb::kKindComplex, uint8_t(s.nops), n), 0);
  size_t off = 0;
  std::memcpy(blk.data() + off, ref.data(), 32 * size_t(n)); off += 32 * size_t(n);
  std::memcpy(blk.data() + off, za.data(), 16 * size_t(n));  off += 16 * size_t(n);
  if (s.nops >= 2) std::memcpy(blk.data() + off, zb.data(), 16 * size_t(n));
  return blk;
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------
void usage(const char* argv0) {
  std::fprintf(stderr,
    "Usage: %s [--n N] [--seed N] [--out PATH] [--with-nan]\n"
    "          [--real-only | --complex-only]\n"
    "       %s --verify PATH\n"
    "\n"
    "  --n N          elements per op (default %llu)\n"
    "  --seed N       RNG seed (default %llu)\n"
    "  --out PATH     output file (default %s)\n"
    "  --with-nan     include NaN inputs (corpus::CorpusFlags::include_nan)\n"
    "  --real-only    emit the 39 real ops only\n"
    "  --complex-only emit the 24 complex ops only\n"
    "  --verify PATH  validate an existing corpus (header + full checksum)\n",
    argv0, argv0, (unsigned long long)kDefaultN, (unsigned long long)kDefaultSeed,
    kDefaultOut);
}

bool parse_args(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto need = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "Missing value after %s\n", what); return nullptr; }
      return argv[++i];
    };
    if (s == "--help" || s == "-h") return false;
    else if (s == "--n") {
      const char* v = need("--n"); if (!v) return false;
      o.n = std::strtoull(v, nullptr, 10);
      if (o.n == 0) { std::fprintf(stderr, "--n must be > 0\n"); return false; }
    } else if (s == "--seed") {
      const char* v = need("--seed"); if (!v) return false;
      o.seed = std::strtoull(v, nullptr, 10);
    } else if (s == "--out") {
      const char* v = need("--out"); if (!v) return false;
      o.out = v;
    } else if (s == "--verify") {
      const char* v = need("--verify"); if (!v) return false;
      o.verify = v;
    } else if (s == "--with-nan")     { o.with_nan = true; }
    else if (s == "--real-only")      { o.do_cplx = false; }
    else if (s == "--complex-only")   { o.do_real = false; }
    else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); return false; }
  }
  if (!o.do_real && !o.do_cplx) {
    std::fprintf(stderr, "--real-only and --complex-only are mutually exclusive\n");
    return false;
  }
  return true;
}

int run_verify(const std::string& path) {
  try {
    cb::Corpus corpus(path);
    const cb::Header& h = corpus.header();
    char gv[49]; std::memcpy(gv, h.generator_version, 48); gv[48] = '\0';
    std::printf("corpus:            %s\n", path.c_str());
    std::printf("  format_version:  %u\n", h.format_version);
    std::printf("  generator:       %s\n", gv);
    std::printf("  elements per op: %llu\n", (unsigned long long)h.n_elems);
    std::printf("  ops:             %u\n", h.n_ops);
    std::printf("  seed:            %llu\n", (unsigned long long)h.seed);
    std::printf("  with_nan:        %s\n", (h.flags & cb::kFlagWithNan) ? "yes" : "no");
    std::printf("  payload:         %llu bytes\n", (unsigned long long)h.payload_bytes);
    std::printf("  checksum:        %s\n", cb::Corpus::hex(h.payload_checksum).c_str());
    std::fflush(stdout);
    const uint64_t c = corpus.verify_checksum();
    std::printf("  checksum OK:     %s\n", cb::Corpus::hex(c).c_str());
    std::printf("VERIFY OK\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "VERIFY FAILED: %s\n", e.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, opt)) { usage(argv[0]); return 2; }
  if (!opt.verify.empty()) return run_verify(opt.verify);

  // ---- op table -----------------------------------------------------------
  std::vector<cb::OpRecord> recs;
  std::vector<int>          ids;      // parallel: index into kReal / kComplex
  auto add_rec = [&](const char* name, uint8_t kind, uint8_t nops, int id) {
    cb::OpRecord r;
    std::memset(&r, 0, sizeof(r));
    std::strncpy(r.name, name, sizeof(r.name) - 1);
    r.kind = kind;
    r.n_operands = nops;
    recs.push_back(r);
    ids.push_back(id);
  };
  if (opt.do_real)
    for (int i = 0; i < R_COUNT; ++i) add_rec(kReal[i].name, cb::kKindReal, uint8_t(kReal[i].nops), i);
  if (opt.do_cplx)
    for (int i = 0; i < C_COUNT; ++i) add_rec(kComplex[i].name, cb::kKindComplex, uint8_t(kComplex[i].nops), i);

  const uint64_t first = sizeof(cb::Header) + uint64_t(recs.size()) * sizeof(cb::OpRecord);
  uint64_t off = first, payload = 0;
  for (size_t i = 0; i < recs.size(); ++i) {
    recs[i].offset = off;
    recs[i].bytes  = cb::block_bytes(recs[i].kind, recs[i].n_operands, opt.n);
    off     += recs[i].bytes;
    payload += recs[i].bytes;
  }

  std::printf("gen_corpus %s\n", cb::kGeneratorVersion);
  std::printf("  out=%s  n=%llu  seed=%llu  ops=%zu  with_nan=%s\n",
              opt.out.c_str(), (unsigned long long)opt.n, (unsigned long long)opt.seed,
              recs.size(), opt.with_nan ? "yes" : "no");
  std::printf("  projected size: %.2f MiB\n", double(first + payload) / (1024.0 * 1024.0));
  std::fflush(stdout);

  // ---- header (checksum patched at the end) --------------------------------
  cb::Header hdr;
  std::memset(&hdr, 0, sizeof(hdr));
  std::memcpy(hdr.magic, cb::kMagic, 8);
  hdr.format_version   = cb::kFormatVersion;
  hdr.header_bytes     = uint32_t(sizeof(cb::Header));
  hdr.n_elems          = opt.n;
  hdr.n_ops            = uint32_t(recs.size());
  hdr.op_record_bytes  = uint32_t(sizeof(cb::OpRecord));
  hdr.seed             = opt.seed;
  hdr.flags            = opt.with_nan ? cb::kFlagWithNan : 0ull;
  hdr.payload_bytes    = payload;
  hdr.payload_checksum = 0;
  std::strncpy(hdr.generator_version, cb::kGeneratorVersion, sizeof(hdr.generator_version) - 1);

  std::FILE* f = std::fopen(opt.out.c_str(), "wb");
  if (!f) { std::fprintf(stderr, "cannot open %s for writing\n", opt.out.c_str()); return 1; }
  if (std::fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
      std::fwrite(recs.data(), sizeof(cb::OpRecord), recs.size(), f) != recs.size()) {
    std::fprintf(stderr, "write failed (header)\n"); std::fclose(f); return 1;
  }

  // ---- blocks --------------------------------------------------------------
  const std::vector<double>                     ucorner = unary_corners(opt.with_nan);
  const std::vector<std::pair<double, double> > bcorner = binary_corners(opt.with_nan);
  std::printf("  corner pool: %zu unary values, %zu operand pairs\n",
              ucorner.size(), bcorner.size());
  std::fflush(stdout);

  uint64_t checksum = cb::kChecksumSeed;
  uint64_t total_resampled = 0, total_nonfinite = 0;
  for (size_t i = 0; i < recs.size(); ++i) {
    BlockStats st;
    std::vector<unsigned char> blk =
        (recs[i].kind == cb::kKindReal)
            ? build_real_block(ids[i], opt, ucorner, bcorner, st)
            : build_complex_block(ids[i], opt, ucorner, st);
    recs[i].n_corner = st.n_corner;
    recs[i].n_cancel = st.n_cancel;
    total_resampled += st.resampled;
    total_nonfinite += st.nonfinite;

    if (blk.size() != recs[i].bytes) { std::fprintf(stderr, "internal: block size mismatch\n"); std::fclose(f); return 1; }
    checksum = cb::checksum_update(checksum, blk.data(), blk.size());
    if (std::fwrite(blk.data(), 1, blk.size(), f) != blk.size()) {
      std::fprintf(stderr, "write failed (%s)\n", recs[i].name); std::fclose(f); return 1;
    }
    std::printf("  %-8s %-7s n_corner=%-6llu n_cancel=%-8llu inf_refs=%-8llu resampled=%llu\n",
                recs[i].kind == cb::kKindReal ? "real" : "complex", recs[i].name,
                (unsigned long long)st.n_corner, (unsigned long long)st.n_cancel,
                (unsigned long long)st.nonfinite, (unsigned long long)st.resampled);
    std::fflush(stdout);
  }

  // ---- patch header + op table (n_corner / n_cancel / checksum) ------------
  hdr.payload_checksum = checksum;
  if (std::fseek(f, 0, SEEK_SET) != 0 ||
      std::fwrite(&hdr, sizeof(hdr), 1, f) != 1 ||
      std::fwrite(recs.data(), sizeof(cb::OpRecord), recs.size(), f) != recs.size()) {
    std::fprintf(stderr, "write failed (header patch)\n"); std::fclose(f); return 1;
  }
  if (std::fclose(f) != 0) { std::fprintf(stderr, "close failed\n"); return 1; }

  std::printf("\nwrote %s  (%llu bytes, %.2f MiB)\n", opt.out.c_str(),
              (unsigned long long)(first + payload),
              double(first + payload) / (1024.0 * 1024.0));
  std::printf("checksum %s\n", cb::Corpus::hex(checksum).c_str());
  std::printf("infinite references: %llu   NaN-safety-net resamples: %llu%s\n",
              (unsigned long long)total_nonfinite, (unsigned long long)total_resampled,
              total_resampled ? "   <-- a per-op domain repair is missing" : "");
  return 0;
}
