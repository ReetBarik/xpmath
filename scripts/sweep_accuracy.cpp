// ============================================================================
// sweep_accuracy.cpp — dense per-point accuracy sweep + monotone regression gate
// ============================================================================
//
// WHAT THIS PRODUCES
//   One CSV row per (backend, kind, op, grid point) giving the measured decimal
//   digits of agreement with a __float128 / __complex128 oracle:
//
//       backend,kind,op,point,digits
//       DD,r,add,0,31.00
//       ...
//
//   4 backends (DD, FF, QF, TF) x 63 ops (39 real + 24 complex) x a fixed
//   deterministic grid. Written to validation/sweep/sweep_baseline.csv, which
//   IS committed, together with validation/sweep/sweep_grid.csv, a manifest of
//   the grid values so a diff of the baseline can be read without rerunning.
//
// WHY IT EXISTS — THE TOLERANCE TABLE CANNOT SEE A PARTIAL LOSS
//   The per-op tolerance tables in tests/*_accuracy_test.cpp and
//   tests/*_complex_accuracy_test.cpp gate on the MEAN over a corpus, against a
//   threshold. That protects against a score falling BELOW a threshold. It does
//   not protect against a score falling AT ALL. A fix that repairs complex acosh
//   by 30 digits while quietly costing complex exp two digits leaves exp's mean
//   comfortably above its gate, and nothing goes red.
//
//   This tool records the score at every individual grid point, once, before any
//   numerical fix lands. `--baseline` then re-runs the sweep and reports every
//   point whose digits DECREASED. That is the monotone gate, and it is the only
//   thing in the repo that can catch a silent trade.
//
//   It is NOT a replacement for the tolerance tables. Those say "this op is this
//   accurate"; this says "this op is no less accurate than it was".
//
// RELATIONSHIP TO scripts/gen_corpus.cpp
//   Same posture (standalone host tool, not wired into CMake, libquadmath
//   allowed) and the same determinism discipline. The deterministic sampling
//   primitives — splitmix64, fnv1a, struct Rng, stream_seed — and the per-op
//   domain repair table are taken from gen_corpus.cpp rather than reinvented;
//   see the DETERMINISM and PER-OP DOMAIN POLICY sections of that file for the
//   reasoning behind each.
//
//   The difference is the INPUT SET. gen_corpus draws a large random sample from
//   a log-uniform distribution and stores it; this walks a small FIXED grid
//   chosen to land on the places where these implementations are known or
//   suspected to break — branch cuts, signed zeros, multiples of pi, the
//   neighbourhood of 1, and the extremes of the magnitude range. A random
//   corpus finds defects by volume; this grid finds them by aim.
//
// BUILD (from the repository root)
//   g++ -std=c++17 -fext-numeric-literals -O2 -Iinclude
//       scripts/sweep_accuracy.cpp -lquadmath -o scripts/sweep_accuracy
//
//   (-fext-numeric-literals is required for __float128 under -std=c++17.
//    Toolchain of record is gcc/13.3.0 per scripts/prepare.sh.)
//
// RUN
//   ./scripts/sweep_accuracy                                  # write the default baseline
//   ./scripts/sweep_accuracy --out /tmp/fresh.csv             # write elsewhere
//   ./scripts/sweep_accuracy --baseline validation/sweep/sweep_baseline.csv
//                                                             # MONOTONE GATE: exit 1 on any decrease
//   ./scripts/sweep_accuracy --grid-out validation/sweep/sweep_grid.csv
//   ./scripts/sweep_accuracy --summary                        # per-(backend,op) table only
//
// ============================================================================
// THE GRID
// ============================================================================
// Two grids, each shared by every op of its kind so that a point id means the
// same input everywhere and cross-op comparison is possible.
//
// REAL GRID (1588 points), three families in this fixed order:
//
//   [0, 242)      (1) LOG SWEEP. |x| = 10^e for e = -30, -29.5, ... , +30, both
//                     signs. Half-decade steps; 61 whole decades and 60 halves.
//                     Covers the magnitude range the task calls for and, at the
//                     small end, deliberately runs past the point where QF's and
//                     TF's trailing FP32 limbs go subnormal. Those points score
//                     badly. That is a real property of a 4xFP32 layout and it
//                     is recorded, not excluded.
//
//   [242, 563)    (2) LINEAR SWEEP. x = i/20 for i = -160 .. +160, i.e. -8.00 to
//                     +8.00 in steps of 0.05. Dense coverage of the region where
//                     the argument reductions and the near-1 behaviour live.
//
//   [563, 1588)   (3) ULP NEIGHBOURHOODS. 205 anchors { 0, +-1, +-pi/2,
//                     +-k*pi for k = 1..100 }, each with its -2, -1, 0, +1, +2
//                     ulp neighbours (std::nextafter). This is the family that
//                     catches KI-2 (nint at a one-ulp-below-a-tie) and KI-4 (DD
//                     sin sign flip near odd multiples of pi) — both need an
//                     input a random corpus will not land on.
//
//   [1588, 1652)  (4) NEAR ONE. +-(1 -+ 10^-k) for k = 1..16. The branch point of
//                     asin/acos/atanh, approached geometrically rather than in
//                     ulps, so the whole ramp is sampled and not just its foot.
//
// COMPLEX GRID (1472 points), two families:
//
//   [0, 384)      (1) POLAR. |z| in { 1e-8, 1e-4, 0.5, 0.9, 0.99, 1, 1.01, 1.1,
//                     2, 10, 1e4, 1e8 } x 32 equally spaced arguments. The
//                     moduli cluster around 1 because that is where the inverse
//                     functions' branch points sit.
//
//   [384, 1472)   (2) BRANCH CUTS. 17 anchors on the two axes that carry every
//                     cut in the inventory:
//                        real:   -100 -10 -2 -1 -0.5 0 +0.5 +1 +2 +10 +100
//                        imag:   -10i -2i -1i +1i +2i +10i
//                     Each anchor is approached PERPENDICULAR to its axis from
//                     both sides at 10^-p for p = 0..30 (62 points), plus the
//                     two signed zeros +0.0 and -0.0 (2 points) = 64 per anchor.
//
//                     The signed zeros are the point of this family. KI-5 (d) is
//                     invisible without them: complex asin returns the same
//                     branch for Im = +0 and Im = -0, so it is wrong on exactly
//                     one side of every real cut, and a grid that only ever
//                     writes 0.0 sees half the defect at best.
//
//   [1472, 1780)  (3) REAL AXIS, GEOMETRIC. +-10^-k and +-10^k, and +-(1 -+ 10^-k),
//                     each at BOTH signed zeros. Family (2) puts 31 decades of
//                     approach at 17 fixed anchors; this puts the anchors
//                     themselves on a geometric ladder, which is what the KI-5
//                     ramps need — (a) asinh degrades as Re(z) -> -inf and (b)
//                     atanh as z -> 0, so both are measured by how DEEP the
//                     ladder goes, not by how closely a cut is approached.
//
// Families are APPENDED, never inserted. Extending the grid therefore does not
// renumber existing points, so a `point` id keeps meaning the same input across
// grid revisions and a regenerated baseline diffs as inserted rows rather than
// as whole-file churn. (Compare mode still refuses a baseline whose row set does
// not match exactly — see compare_baseline; the invariant is about diff
// readability and cross-revision interpretation, not about relaxing the gate.)
//
// GRID SIZE — REDUCED FROM THE 10k/op TARGET, DELIBERATELY
//   The brief asked for order 10k points per op per backend. That would be
//   63 ops x 4 backends x 10k = 2.5M rows, a ~28 MB CSV added to git history on
//   every regeneration — and the whole workflow here is "regenerate and diff
//   after each fix", so history growth is a recurring cost, not a one-off.
//
//   1652 real / 1780 complex points gives ~429k rows and a ~8.7 MB CSV (~1.3 MB
//   once git packs it) while still containing EVERY family the brief named, at
//   full resolution in the families that matter: all 100 multiples of pi, all 31
//   decades of cut approach, both signed zeros. What was traded away is density
//   in the two families where density buys least — the log sweep is at
//   half-decade rather than tenth-decade steps, and the linear sweep is at 0.05
//   rather than 0.01. A defect that a 0.01 linear step finds and a 0.05 step
//   misses would have to be narrower than 5e-2 in argument and not be anchored
//   at 0, +-1, +-pi/2 or a multiple of pi, which is not the shape of anything in
//   KNOWN_ISSUES.md.
//
//   Runtime is not the binding constraint: a full sweep is well under a minute.
//   File size is. Raise kLinearStepInv / add log-sweep subdivisions here if that
//   trade ever stops being the right one.
//
// ============================================================================
// SCORING
// ============================================================================
// Every backend result is widened to __float128 by summing its limbs, which is
// EXACT: the limbs are non-overlapping and 2xFP64 = 106, 4xFP32 = 96,
// 3xFP32 = 72 and 2xFP32 = 48 significant bits all fit binary128's 113. So the
// measured error is the backend's, with nothing added by the scorer.
//
//   digits = -log10( |got - ref| / |ref| ),  clamped to [0, cap]
//   caps: DD 31.00, QF 29.00, TF 21.70, FF 14.00  (the caps the ctest suite uses)
//
// Non-finite and zero references, chosen to keep every cell a plain number so
// the file stays diffable and the gate stays a simple numeric comparison:
//
//   ref NaN         -> cap if got is NaN, else 0
//   ref +-inf       -> cap if got is the same infinity, else 0
//   got non-finite while ref is finite -> 0
//   ref exactly 0   -> cap if got is exactly 0, else 0
//
// For COMPLEX ops the two components are scored separately and the point takes
// min(d_re, d_im) — the weaker component decides, matching
// tests/*_complex_accuracy_test.cpp. That file's zero-reference rule is carried
// over too: a component whose reference is exactly zero is scored ABSOLUTELY
// against the magnitude of the OTHER component. Without it every purely-real
// result reads a spurious 0.00, which is a scoring artifact and not a defect.
//
// RED CELLS ARE THE POINT
//   KI-1, KI-2, KI-4 and the four KI-5 defects are all unfixed as of this
//   baseline. They appear here as 0.00 cells and they are recorded AS THEY ARE.
//   Nothing is excluded, clamped or annotated away: a baseline that hides the
//   current defects cannot later prove a fix worked.
//
//   A 0.00 is NOT automatically a defect, though, and the ulp family in
//   particular is full of legitimate ones. sin(x) for x within a couple of ulps
//   of k*pi is ~1e-16, and recovering it needs a pi wider than the backend's
//   own; FF carries 48 bits, so 992 of its 1025 ulp-family sin points read 0.00
//   and every one of them is FF being honest about a 2xFP32 layout. Read a cell
//   against KNOWN_ISSUES.md and against the other backends before calling it a
//   bug — the gate's job is to catch a 0.00 that APPEARS, not to indict the ones
//   already here.
//
// ============================================================================
// DETERMINISM
// ============================================================================
// Same build + same libm/libquadmath => byte-identical CSV. Guaranteed by:
//   * the grid being a fixed arithmetic construction, not a sample;
//   * second and third operands for the binary/ternary ops coming from
//     std::mt19937_64 with a per-op stream seed (gen_corpus's stream_seed), and
//     from NO std::uniform_*_distribution, whose output is implementation
//     defined;
//   * no unordered containers on any path that affects output order.
// Caveat, identical to gen_corpus's: the references come from libquadmath and
// the grid's transcendental anchors (pi, the polar family) from libm, so a
// toolchain change can move cells. This is not hypothetical — running the same
// binary against the system /usr/lib64/libquadmath.so.0 instead of gcc/13.3.0's
// moves 11 of the 428592 cells by 0.01 digits, which compare mode would
// otherwise report as eleven mystifying regressions.
//
// So the baseline records an ORACLE FINGERPRINT: an FNV hash over the raw bits
// of a fixed set of libquadmath results. Compare mode checks it and WARNS on a
// mismatch. A warning and not a failure, because a toolchain upgrade is a
// legitimate reason to re-baseline, but the operator has to be told that the
// hairline diffs are the oracle moving and not the library regressing.
//
// GENERATE AND COMPARE UNDER THE TOOLCHAIN OF RECORD:
//   module use /soft/modulefiles && module load gcc/13.3.0
// kFormatVersion is the separate, coarser guard: bump it when the grid or the
// column set changes.
//
// ============================================================================

#define XPMATH_ENABLE_DIAGNOSTICS 0

#include "../include/xp/dd_complex.hpp"
#include "../include/xp/ff_complex.hpp"
#include "../include/xp/qf_complex.hpp"
#include "../include/xp/tf_complex.hpp"

#include <quadmath.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <string>
#include <random>
#include <string>
#include <vector>

namespace {

const char* const kFormatVersion = "xp-sweep-2";
const char* const kDefaultOut    = "validation/sweep/sweep_baseline.csv";
// Triage threshold: a point scoring below this fraction of its backend's cap is
// classified. 0.50 is a judgment call — see the CLASSIFICATION section.
const double      kDefaultClassifyFrac = 0.50;
const uint64_t    kDefaultSeed   = 12345ull;   // same default as gen_corpus and the demos

// ---------------------------------------------------------------------------
// Deterministic sampling primitives. Copied from scripts/gen_corpus.cpp; see the
// DETERMINISM section of that file for why each choice is what it is.
// ---------------------------------------------------------------------------
uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

uint64_t fnv1a(const char* s) {
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) { h ^= uint64_t((unsigned char)*s); h *= 1099511628211ull; }
  return h;
}

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t s) : g(s) {}
  double unit() { return double(g() >> 11) * (1.0 / 9007199254740992.0); }
  int    in(int lo, int hi) { return lo + int(g() % uint64_t(hi - lo + 1)); }
  bool   coin() { return (g() & 1ull) != 0ull; }
  double logunif(int lo, int hi) { return std::ldexp(1.0 + unit(), in(lo, hi)); }
  double slogunif(int lo, int hi) { const double v = logunif(lo, hi); return coin() ? v : -v; }
};

uint64_t stream_seed(uint64_t base, const char* name, unsigned kind) {
  return splitmix64(base ^ fnv1a(name) ^ (uint64_t(kind) * 0x9E3779B97F4A7C15ull));
}

// ---------------------------------------------------------------------------
// Oracle fingerprint — hashes the raw bits of a fixed set of libquadmath results
// so that "the reference moved" is distinguishable from "the library
// regressed". See the DETERMINISM section above. Deliberately covers the
// complex inverse functions, which is where the two libquadmath builds on this
// machine actually differ.
// ---------------------------------------------------------------------------
uint64_t oracle_fingerprint() {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&h](__float128 v) {
    unsigned char b[sizeof(__float128)];
    std::memcpy(b, &v, sizeof(b));
    for (size_t i = 0; i < sizeof(b); ++i) { h ^= b[i]; h *= 1099511628211ull; }
  };
  auto mixc = [&mix](__complex128 z) { mix(crealq(z)); mix(cimagq(z)); };
  for (int k = 1; k <= 8; ++k) {
    const __float128 x = (__float128)k / (__float128)7.0;
    mix(sqrtq(x)); mix(logq(x)); mix(expq(x)); mix(log10q(x)); mix(log1pq(x));
    mix(sinq(x)); mix(cosq(x)); mix(atanq(x)); mix(tanhq(x));
    mix(acoshq((__float128)1 + x)); mix(powq(x, (__float128)3.5));
    __complex128 z; __real__ z = x; __imag__ z = (__float128)0.25 * (__float128)k;
    mixc(csqrtq(z)); mixc(clogq(z)); mixc(clog10q(z)); mixc(cexpq(z));
    mixc(casinq(z)); mixc(cacosq(z)); mixc(cacoshq(z));
    mixc(casinhq(z)); mixc(catanhq(z)); mixc(cpowq(z, z));
  }
  return h;
}

// ---------------------------------------------------------------------------
// Op inventory. Identical order and naming to scripts/gen_corpus.cpp, so a
// sweep row and a corpus record refer to the same operation.
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
const RealSpec kReal[R_COUNT] = {
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
const ComplexSpec kComplex[C_COUNT] = {
  {"add", 2}, {"sub", 2}, {"mul", 2}, {"div", 2},
  {"abs", 1}, {"conj", 1}, {"sqrt", 1}, {"exp", 1}, {"log", 1}, {"log10", 1},
  {"sin", 1}, {"cos", 1}, {"tan", 1}, {"asin", 1}, {"acos", 1}, {"atan", 1},
  {"sinh", 1}, {"cosh", 1}, {"tanh", 1}, {"asinh", 1}, {"acosh", 1}, {"atanh", 1},
  {"pow", 2}, {"polar", 1},
};
// clang-format on

// ---------------------------------------------------------------------------
// Grid construction.
// ---------------------------------------------------------------------------
const int kLinearHalfSpan = 160;   // x = i/20, i in [-160, 160]  ->  [-8, 8] step 0.05
const int kUlpRadius      = 2;     // -2 .. +2 ulp around every anchor
const int kMaxPiMultiple  = 100;   // +-k*pi for k = 1..100
const int kCutDecades     = 30;    // approach a cut at 10^-p, p = 0..30

struct GridPoint {
  double      re, im;      // im unused for the real grid
  const char* family;
};

double pow10i(int e) { return std::pow(10.0, double(e)); }

std::vector<GridPoint> build_real_grid() {
  std::vector<GridPoint> g;

  // (1) log sweep, half-decade steps, both signs.
  for (int e = -30; e <= 30; ++e) {
    const double v = pow10i(e);
    g.push_back({ v, 0.0, "log"});
    g.push_back({-v, 0.0, "log"});
    if (e < 30) {
      const double h = 3.1622776601683795 * v;   // sqrt(10) * 10^e
      g.push_back({ h, 0.0, "log"});
      g.push_back({-h, 0.0, "log"});
    }
  }

  // (2) dense linear sweep.
  for (int i = -kLinearHalfSpan; i <= kLinearHalfSpan; ++i)
    g.push_back({double(i) / 20.0, 0.0, "linear"});

  // (3) few-ulp neighbourhoods of the anchors that matter.
  std::vector<double> anchors;
  anchors.push_back(0.0);
  anchors.push_back(1.0);   anchors.push_back(-1.0);
  anchors.push_back(M_PI_2); anchors.push_back(-M_PI_2);
  for (int k = 1; k <= kMaxPiMultiple; ++k) {
    anchors.push_back(double(k) * M_PI);
    anchors.push_back(-double(k) * M_PI);
  }
  for (size_t i = 0; i < anchors.size(); ++i) {
    for (int n = -kUlpRadius; n <= kUlpRadius; ++n) {
      double v = anchors[i];
      for (int s = 0; s < (n < 0 ? -n : n); ++s)
        v = std::nextafter(v, n < 0 ? -HUGE_VAL : HUGE_VAL);
      g.push_back({v, 0.0, "ulp"});
    }
  }

  // (4) geometric approach to the branch point at +-1, from both sides.
  for (int k = 1; k <= 16; ++k) {
    const double d = pow10i(-k);
    g.push_back({ 1.0 - d, 0.0, "near1"});
    g.push_back({-1.0 + d, 0.0, "near1"});
    g.push_back({ 1.0 + d, 0.0, "near1"});
    g.push_back({-1.0 - d, 0.0, "near1"});
  }
  return g;
}

std::vector<GridPoint> build_complex_grid() {
  std::vector<GridPoint> g;

  // (1) polar shells.
  const double kMod[] = {1e-8, 1e-4, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 10.0, 1e4, 1e8};
  const int    kArgs  = 32;
  for (size_t m = 0; m < sizeof(kMod) / sizeof(kMod[0]); ++m)
    for (int j = 0; j < kArgs; ++j) {
      const double th = 2.0 * M_PI * double(j) / double(kArgs);
      g.push_back({kMod[m] * std::cos(th), kMod[m] * std::sin(th), "polar"});
    }

  // (2) branch cuts: perpendicular approach from both sides, plus both signed zeros.
  const double kRealAnchor[] = {-100.0, -10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0};
  const double kImagAnchor[] = {-10.0, -2.0, -1.0, 1.0, 2.0, 10.0};

  for (size_t i = 0; i < sizeof(kRealAnchor) / sizeof(kRealAnchor[0]); ++i) {
    const double x = kRealAnchor[i];
    g.push_back({x, 0.0,  "cut-re"});          // Im = +0
    g.push_back({x, -0.0, "cut-re"});          // Im = -0  <-- KI-5 (d) needs this
    for (int p = 0; p <= kCutDecades; ++p) {
      const double d = pow10i(-p);
      g.push_back({x,  d, "cut-re"});
      g.push_back({x, -d, "cut-re"});
    }
  }
  for (size_t i = 0; i < sizeof(kImagAnchor) / sizeof(kImagAnchor[0]); ++i) {
    const double y = kImagAnchor[i];
    g.push_back({0.0,  y, "cut-im"});
    g.push_back({-0.0, y, "cut-im"});
    for (int p = 0; p <= kCutDecades; ++p) {
      const double d = pow10i(-p);
      g.push_back({ d, y, "cut-im"});
      g.push_back({-d, y, "cut-im"});
    }
  }

  // (3) the real axis on a geometric ladder, at both signed zeros. This is the
  // family that measures the KI-5 (a) and (b) ramps to their far ends, and the
  // KI-5 (c) ramp into the branch point at 1.
  {
    std::vector<double> x;
    for (int k = 1; k <= 30; ++k) { const double d = pow10i(-k); x.push_back(d); x.push_back(-d); }
    for (int k = 1; k <= 15; ++k) { const double d = pow10i( k); x.push_back(d); x.push_back(-d); }
    for (int k = 1; k <= 16; ++k) {
      const double d = pow10i(-k);
      x.push_back(1.0 - d); x.push_back(-1.0 + d);
      x.push_back(1.0 + d); x.push_back(-1.0 - d);
    }
    for (size_t i = 0; i < x.size(); ++i) {
      g.push_back({x[i],  0.0, "axis"});
      g.push_back({x[i], -0.0, "axis"});
    }
  }
  return g;
}

// ---------------------------------------------------------------------------
// Per-op domain repair, applied to the shared grid value so that a point is
// evaluated somewhere the op is defined. Transcribed from gen_corpus.cpp's
// repair_real(); see its PER-OP DOMAIN POLICY section for the rationale.
//
// Note the asymmetry with the complex side: NO repair is applied to the complex
// grid. The complex grid exists precisely to sit on branch cuts and poles, and
// repairing it away would delete the thing being measured. Complex points whose
// reference is NaN or infinite are scored by the non-finite rules instead.
// ---------------------------------------------------------------------------
bool is_inf(double x) { return std::isinf(x); }
double tame_inf(double x) { return std::isinf(x) ? std::copysign(1.0, x) : x; }

void repair_real(int id, double& a, double& b, double& c) {
  using std::fabs;
  switch (id) {
    case R_Add:  if (is_inf(a) && is_inf(b) && ((a > 0) != (b > 0))) b = 1.0; break;
    case R_Sub:  if (is_inf(a) && is_inf(b) && ((a > 0) == (b > 0))) b = 1.0; break;
    case R_Mul:  if (a == 0.0 && is_inf(b)) b = 1.0;
                 if (is_inf(a) && b == 0.0) a = 1.0;
                 break;
    case R_Div:  if ((a == 0.0 && b == 0.0) || (is_inf(a) && is_inf(b))) b = 1.0; break;
    case R_Sqrt: case R_Log: case R_Log2: case R_Log10:
                 a = fabs(a); break;
    case R_Log1p: if (a < -1.0) a = 1.0 / a; break;
    case R_Sin: case R_Cos: case R_Tan:
                 a = tame_inf(a); break;
    case R_Asin: case R_Acos: case R_Atanh:
                 if (fabs(a) > 1.0) a = 1.0 / a;
                 break;
    case R_Acosh: if (!(a >= 1.0)) a = 1.0 + fabs(a); break;
    case R_Pow: {
      a = fabs(a);
      if (a == 0.0) a = 1.0;
      if (a != 1.0 && std::isfinite(a) && std::isfinite(b) && b != 0.0) {
        const double mag = fabs(b * std::log2(a));
        if (mag > 120.0) b *= 120.0 / mag;
      }
      break;
    }
    case R_Fmod: case R_Remainder:
                 if (b == 0.0) b = 1.0;
                 if (is_inf(a)) a = 0.0;
                 break;
    case R_Fdim: if (is_inf(a) && is_inf(b) && ((a > 0) == (b > 0))) b = 0.0; break;
    case R_Fma:  if (a == 0.0 && is_inf(b)) b = 1.0;
                 if (is_inf(a) && b == 0.0) a = 1.0;
                 if ((is_inf(a) || is_inf(b)) && is_inf(c)) c = 0.0;
                 break;
    default: break;
  }
}

// Second/third operands for the binary and ternary real ops. Family split
// mirrors gen_corpus: roughly one point in seven is a deliberately cancelling
// pair, the rest are log-uniform over the op's window.
void fill_real_operands(int id, size_t i, double a, Rng& rng, double& b, double& c) {
  const RealSpec& s = kReal[id];
  b = 0.0; c = 0.0;
  if (s.nops < 2) return;

  const bool cancel = (i % 7 == 1);
  if (cancel) {
    const double eps = std::ldexp(1.0, -rng.in(1, 53)) * (rng.coin() ? 1.0 : -1.0);
    b = (id == R_Add) ? -a * (1.0 + eps) : a * (1.0 + eps);
  } else {
    b = rng.slogunif(s.b_lo, s.b_hi);
  }
  if (s.nops >= 3) c = cancel ? -(a * b) * (1.0 + std::ldexp(1.0, -rng.in(1, 53)))
                              : rng.slogunif(-100, 100);
}

// Second operand for the binary complex ops. The complex grid is stride-paired
// against itself (stride 7, coprime with the grid length, so the pairing is a
// permutation and never a fixed point), with the same one-in-seven cancelling
// family. pow's exponent is drawn small and then clamped so |a^b| stays inside
// the FP32 range every backend must be able to hold — same clamp as gen_corpus.
void fill_complex_operands(int id, size_t i, const std::vector<GridPoint>& grid,
                           double are, double aim, Rng& rng, double& bre, double& bim) {
  bre = 0.0; bim = 0.0;
  if (kComplex[id].nops < 2) return;

  if (id == C_Pow) {
    bre = rng.slogunif(-10, 4);
    bim = rng.slogunif(-10, 4);
    const double mod = std::hypot(are, aim);
    if (std::isfinite(mod) && mod > 0.0 && mod != 1.0) {
      const double la  = std::log2(mod);
      const double mag = std::hypot(bre, bim) * std::fabs(la);
      if (mag > 120.0) { const double s = 120.0 / mag; bre *= s; bim *= s; }
    }
    return;
  }
  if (i % 7 == 1) {
    const double eps = std::ldexp(1.0, -rng.in(1, 53)) * (rng.coin() ? 1.0 : -1.0);
    const double s   = (id == C_Add) ? -(1.0 + eps) : (1.0 + eps);
    bre = are * s; bim = aim * s;
    return;
  }
  const GridPoint& p = grid[(i * 7 + 3) % grid.size()];
  bre = p.re; bim = p.im;
}

// ---------------------------------------------------------------------------
// __float128 oracle. Same op semantics as gen_corpus.cpp's reference_real /
// reference_complex, including its use of powq for exp2/exp10 (exp10q does not
// exist and exp2q is absent from older libquadmath).
// ---------------------------------------------------------------------------
// KI-37.  `round` is the one op where the oracle and the library follow
// DIFFERENT conventions, and the oracle was the wrong one.  C's `round` — and
// therefore libquadmath's `roundq` — breaks ties AWAY from zero, so it scores
// round(-6.5) as -7 and round(-0.5) as -1.  KI-20 deliberately gave the library
// IEEE 754 `roundToIntegralTiesToEven`, under which those are -6 and -0, and
// the library implements that correctly.  The two conventions differ at the
// exact half-integers and nowhere else, which is why the disagreement was
// exactly 8 points per backend.  Round the tie to even here so the oracle
// scores the convention the library actually implements.
//
// Written out rather than delegating to rintq/nearbyintq: those honour the
// DYNAMIC rounding mode, and the sweep must not depend on the mode a caller
// happens to leave set.
__float128 round_ties_even_q(__float128 a) {
  if (isnanq(a) || !finiteq(a) || a == 0) return a;
  const __float128 t = truncq(a);                    // toward zero, exact
  const __float128 f = a - t;                        // |f| < 1, exact (Sterbenz)
  const __float128 af = fabsq(f);
  if (af < (__float128)0.5) return t;
  if (af > (__float128)0.5) return t + copysignq((__float128)1.0, a);
  // Exact tie: keep the even neighbour.
  const __float128 h = t * (__float128)0.5;          // exact
  if (h == truncq(h)) return t;                      // t already even
  return t + copysignq((__float128)1.0, a);
}

// The quad-argument form. --classify needs to evaluate the oracle at PERTURBED
// inputs, which are quad and not exactly representable as double, so the body
// lives here and the double entry point below just widens and forwards.
__float128 reference_real_q(int id, __float128 a, __float128 b, __float128 c) {
  switch (id) {
    case R_Add:       return a + b;
    case R_Sub:       return a - b;
    case R_Mul:       return a * b;
    case R_Div:       return a / b;
    case R_Sqrt:      return sqrtq(a);
    case R_Abs:       return fabsq(a);
    case R_Exp:       return expq(a);
    case R_Log:       return logq(a);
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
    case R_Round:     return round_ties_even_q(a);           // KI-37
    case R_Trunc:     return truncq(a);
  }
  return (__float128)0.0;
}

__float128 reference_real(int id, double da, double db, double dc) {
  return reference_real_q(id, (__float128)da, (__float128)db, (__float128)dc);
}

void reference_complex_q(int id, __float128 are, __float128 aim,
                         __float128 bre, __float128 bim,
                         __float128& out_re, __float128& out_im) {
  __complex128 za; __real__ za = are; __imag__ za = aim;
  __complex128 zb; __real__ zb = bre; __imag__ zb = bim;
  __complex128 r;
  switch (id) {
    case C_Add:   r = za + zb;       break;
    case C_Sub:   r = za - zb;       break;
    case C_Mul:   r = za * zb;       break;
    case C_Div:   r = za / zb;       break;
    case C_Abs:   out_re = cabsq(za); out_im = (__float128)0.0; return;
    case C_Conj:  r = conjq(za);     break;
    case C_Sqrt:  r = csqrtq(za);    break;
    case C_Exp:   r = cexpq(za);     break;
    case C_Log:   r = clogq(za);     break;
    case C_Log10: r = clog10q(za);   break;
    case C_Sin:   r = csinq(za);     break;
    case C_Cos:   r = ccosq(za);     break;
    case C_Tan:   r = ctanq(za);     break;
    case C_Asin:  r = casinq(za);    break;
    case C_Acos:  r = cacosq(za);    break;
    case C_Atan:  r = catanq(za);    break;
    case C_Sinh:  r = csinhq(za);    break;
    case C_Cosh:  r = ccoshq(za);    break;
    case C_Tanh:  r = ctanhq(za);    break;
    case C_Asinh: r = casinhq(za);   break;
    case C_Acosh: r = cacoshq(za);   break;
    case C_Atanh: r = catanhq(za);   break;
    case C_Pow:   r = cpowq(za, zb); break;
    case C_Polar: {
      const __float128 rad = are, th = aim;
      out_re = rad * cosq(th);
      out_im = rad * sinq(th);
      return;
    }
    default: r = (__complex128)0; break;
  }
  out_re = crealq(r);
  out_im = cimagq(r);
}

void reference_complex(int id, double are, double aim, double bre, double bim,
                       __float128& out_re, __float128& out_im) {
  reference_complex_q(id, (__float128)are, (__float128)aim,
                      (__float128)bre, (__float128)bim, out_re, out_im);
}

// ---------------------------------------------------------------------------
// Widening a backend value to __float128. Exact — see SCORING above.
// ---------------------------------------------------------------------------
__float128 to_q(const xp::DoubleDouble& x) {
  if (!std::isfinite(x.hi)) return (__float128)x.hi;
  return (__float128)x.hi + (__float128)x.lo;
}
__float128 to_q(const xp::FloatFloat& x) {
  if (!std::isfinite(x.hi)) return (__float128)x.hi;
  return (__float128)x.hi + (__float128)x.lo;
}
__float128 to_q(const xp::TripleFloat& x) {
  if (!std::isfinite(x.f0)) return (__float128)x.f0;
  return ((__float128)x.f0 + (__float128)x.f1) + (__float128)x.f2;
}
__float128 to_q(const xp::QuadFloat& x) {
  if (!std::isfinite(x.f0)) return (__float128)x.f0;
  return (((__float128)x.f0 + (__float128)x.f1) + (__float128)x.f2) + (__float128)x.f3;
}

double score_scalar(__float128 got, __float128 ref, double cap) {
  if (isnanq(ref)) return isnanq(got) ? cap : 0.0;
  if (!finiteq(ref))                                     // ref is +-inf
    return (!finiteq(got) && !isnanq(got) && signbitq(got) == signbitq(ref)) ? cap : 0.0;
  if (isnanq(got) || !finiteq(got)) return 0.0;
  if (ref == 0) return got == 0 ? cap : 0.0;
  if (got == ref) return cap;
  const __float128 rel = fabsq(got - ref) / fabsq(ref);
  const double     d   = -(double)log10q(rel);
  if (!(d > 0.0)) return 0.0;
  return d > cap ? cap : d;
}

// Complex component score, carrying tests/*_complex_accuracy_test.cpp's rule for
// a component whose reference is exactly zero: measure it ABSOLUTELY against the
// magnitude of the other component rather than relatively against zero.
double score_component(__float128 got, __float128 ref, __float128 other, double cap) {
  if (finiteq(ref) && ref == 0 && finiteq(other) && other != 0) {
    if (got == 0) return cap;
    if (isnanq(got) || !finiteq(got)) return 0.0;
    const double d = -(double)log10q(fabsq(got) / fabsq(other));
    if (!(d > 0.0)) return 0.0;
    return d > cap ? cap : d;
  }
  return score_scalar(got, ref, cap);
}

// ===========================================================================
// ULP SCORING (see docs/ULP_METRIC.md)
// ===========================================================================
// The digit score above is a RELATIVE error, and a relative error goes vacuous
// exactly where the bugs are: near a zero of the function. DD sin(3*pi) scores
// 16.04 digits — a 21-digit gate passes it — while being wrong by 7.4e15 ulps.
// KI-4 (DD sin returning the wrong SIGN near odd multiples of pi) lived in that
// blind spot and the accuracy tests never saw it.
//
// So every point is ALSO scored in ulps of the true value:
//
//     ulp(true) = |true| * 2^-p        p = the expansion's significand bits
//     ulps      = |got - true| / ulp(true)
//
//   p:  DD 106 (2xFP64)   QF 96 (4xFP32)   TF 72 (3xFP32)   FF 48 (2xFP32)
//
// Those are the limb counts times the word significand (53 and 24 bits
// including the hidden bit), which is the same arithmetic the SCORING section
// above already relies on to argue that widening to binary128 is exact.
//
// ZERO / NON-FINITE REFERENCE. ulp(0) is not defined, so:
//   ref == 0    -> 0 ulps if got == 0, otherwise UNSCORABLE (not gated).
//                  Scoring |got| against ulp(0) = 0 would make every miss
//                  infinite and every op with a zero in its range ungateable;
//                  the digit score already handles this cell exactly.
//   ref inf/nan -> UNSCORABLE. Same reasoning; the digit score covers it.
//   got inf/nan while ref is finite -> +inf ulps (a real, gateable failure).
double kUnscorableUlps() { return -1.0; }   // sentinel: not measurable here

double ulps_scalar(__float128 got, __float128 ref, int sig_bits) {
  if (isnanq(ref) || !finiteq(ref)) return kUnscorableUlps();
  if (ref == 0) return (got == 0) ? 0.0 : kUnscorableUlps();
  if (isnanq(got) || !finiteq(got)) return HUGE_VAL;
  if (got == ref) return 0.0;
  // |got - ref| / (|ref| * 2^-p), formed as (|got-ref|/|ref|) * 2^p so that the
  // scaling cannot overflow or underflow binary128 for any in-range ref.
  const __float128 rel = fabsq(got - ref) / fabsq(ref);
  return (double)ldexpq(rel, sig_bits);
}

// ===========================================================================
// CONDITION-AWARE ULP BOUNDS
// ===========================================================================
// A FLAT ulp gate is wrong, and wrong by orders of magnitude. Take DD sin at
// x = 3*pi (rounded to double, then held exactly by DD):
//
//     kappa = |x * cot(x)| = 9.4248 / 3.673940e-16  ~  2.57e16
//
// DD carries pi to ~106 bits; resolving sin(x) there would need ~47 decimal
// digits of pi. The argument reduction therefore perturbs x by ~1 ulp, and that
// perturbation is amplified by kappa. ~2.6e16 ulps of error is INHERENT. The
// measured 7.4e15 is four times BETTER than the limit. A flat gate fails that
// point; a condition-aware gate passes it, correctly.
//
// So the bound at every point is
//
//     expected_ulps = 1 + kappa(f, x)
//
// the 1 being the final rounding and kappa the amplification of the one-ulp
// perturbation the backend cannot avoid carrying (in the argument, in its stored
// constants, or in both). The gate is
//
//     measured <= kUlpAllowance * expected
//
// kUlpAllowance is a small documented slack for the implementation's own
// internal roundoff — a polynomial evaluation is not a single correctly-rounded
// operation. See the ALLOWANCE note below for what the data actually needed.
//
// kappa is DERIVED ANALYTICALLY per op, never fitted to a measurement. For a
// unary f, kappa = |x f'(x) / f(x)|. For a multi-argument f the partial
// condition numbers add: kappa = sum_i |x_i (df/dx_i) / f|, which is the
// standard first-order bound on the relative output perturbation induced by
// independent one-ulp relative perturbations of the inputs.
//
// EVERY real op in the inventory has an elementary kappa and all 39 are
// implemented below. Two carry a stated caveat:
//
//   fmod / remainder   f = a - b*n with n integral. n is LOCALLY constant, so
//                      kappa = (|a| + |b n|) / |f| holds away from the jump in
//                      n; at the jump the function is discontinuous and no
//                      first-order bound applies. Points that land on the jump
//                      surface as failures, which is the honest outcome.
//   ceil/floor/round/trunc   f' = 0 almost everywhere, so kappa = 0 and the
//                      bound is 1 ulp — i.e. "exact", which is what these ops
//                      are required to be. At the integers they jump; same
//                      caveat, same honest outcome.
//
// COMPLEX OPS ARE NOT GATED ON ULPS. They are MEASURED in ulps (of the complex
// modulus) and reported, but they stay on the digit gate. Deriving kappa for a
// complex op is elementary in the same way, but the complex scorer's
// zero-component rule (score a zero component absolutely against the other
// component) has no ulp analogue that is not invented, and the whole point of
// this change is to stop inventing tolerances. An honest partial conversion
// beats a fabricated one.
//
// ALLOWANCE — WHAT THE DATA NEEDED. Started at 8x per the brief. See
// docs/ULP_METRIC.md for the measured distribution and the final value.
const double kUlpAllowance = 8.0;

const __float128 kQLn2  = 0.693147180559945309417232121458176568Q;
const __float128 kQLn10 = 2.30258509299404568401799145468436421Q;

// kappa for a real op at (a, b, c). Returns false when no analytic kappa is
// available for this op — no such op exists today; the hook stays so that
// adding an op cannot silently acquire a fabricated bound.
bool kappa_real(int id, __float128 a, __float128 b, __float128 c, double& kappa) {
  const __float128 aa = fabsq(a), ab = fabsq(b);
  __float128 k;
  switch (id) {
    // --- algebraic ---------------------------------------------------------
    case R_Add:   k = (a + b == 0) ? HUGE_VALQ : (aa + ab) / fabsq(a + b);      break;
    case R_Sub:   k = (a - b == 0) ? HUGE_VALQ : (aa + ab) / fabsq(a - b);      break;
    case R_Mul:   k = 2;                                                        break;  // |a f_a/f| + |b f_b/f|
    case R_Div:   k = 2;                                                        break;
    case R_Sqrt:  k = 0.5Q;                                                     break;  // x*(1/2 x^-1/2)/x^1/2
    case R_Abs:   k = 1;                                                        break;
    case R_Hypot: k = 1;                                                        break;  // (a^2+b^2)/h^2
    case R_Fma: { const __float128 p = a * b, f = p + c;
                  k = (f == 0) ? HUGE_VALQ : (2 * fabsq(p) + fabsq(c)) / fabsq(f); } break;
    case R_Fdim: { if (!(a > b)) { k = 0; break; }
                   k = (a - b == 0) ? HUGE_VALQ : (aa + ab) / fabsq(a - b); }    break;
    case R_Copysign: case R_Fmax: case R_Fmin: k = 1;                            break;
    case R_Ceil: case R_Floor: case R_Round: case R_Trunc: k = 0;                break;
    case R_Fmod: case R_Remainder: {
      if (b == 0) { k = HUGE_VALQ; break; }
      const __float128 n = (id == R_Fmod) ? truncq(a / b) : nearbyintq(a / b);
      const __float128 f = a - b * n;
      k = (f == 0) ? HUGE_VALQ : (aa + fabsq(b * n)) / fabsq(f);
    } break;

    // --- exponentials: f = base^x, x f'/f = x ln(base) ----------------------
    case R_Exp:   k = aa;                                                        break;
    case R_Exp2:  k = aa * kQLn2;                                                break;
    case R_Exp10: k = aa * kQLn10;                                               break;
    case R_Expm1: { const __float128 f = expm1q(a);
                    k = (f == 0) ? HUGE_VALQ : fabsq(a * expq(a) / f); }         break;

    // --- logarithms: f = log_base(x), x f'/f = 1/ln(x) ----------------------
    case R_Log: case R_Log2: case R_Log10: {
      const __float128 l = logq(a);
      k = (l == 0) ? HUGE_VALQ : 1 / fabsq(l);
    } break;
    case R_Log1p: { const __float128 f = log1pq(a);
                    k = (f == 0 || a == -1) ? HUGE_VALQ
                                            : fabsq(a / ((1 + a) * f)); }        break;

    // --- circular -----------------------------------------------------------
    case R_Sin:  { const __float128 s = sinq(a);
                   k = (s == 0) ? HUGE_VALQ : fabsq(a * cosq(a) / s); }          break;
    case R_Cos:  { const __float128 co = cosq(a);
                   k = (co == 0) ? HUGE_VALQ : fabsq(a * sinq(a) / co); }        break;
    case R_Tan:  { const __float128 d = sinq(a) * cosq(a);
                   k = (d == 0) ? HUGE_VALQ : fabsq(a / d); }                    break;
    case R_Asin: { const __float128 f = asinq(a), r = 1 - a * a;
                   k = (f == 0 || r <= 0) ? HUGE_VALQ : fabsq(a / (sqrtq(r) * f)); } break;
    case R_Acos: { const __float128 f = acosq(a), r = 1 - a * a;
                   k = (f == 0 || r <= 0) ? HUGE_VALQ : fabsq(a / (sqrtq(r) * f)); } break;
    case R_Atan: { const __float128 f = atanq(a);
                   k = (f == 0) ? HUGE_VALQ : fabsq(a / ((1 + a * a) * f)); }    break;

    // --- hyperbolic ---------------------------------------------------------
    case R_Sinh: { const __float128 s = sinhq(a);
                   k = (s == 0) ? HUGE_VALQ : fabsq(a * coshq(a) / s); }         break;
    case R_Cosh: { const __float128 co = coshq(a);
                   k = (co == 0) ? HUGE_VALQ : fabsq(a * sinhq(a) / co); }       break;
    case R_Tanh: { const __float128 d = sinhq(a) * coshq(a);
                   k = (d == 0) ? HUGE_VALQ : fabsq(a / d); }                    break;
    case R_Asinh:{ const __float128 f = asinhq(a);
                   k = (f == 0) ? HUGE_VALQ : fabsq(a / (sqrtq(1 + a * a) * f)); } break;
    case R_Acosh:{ const __float128 f = acoshq(a), r = a * a - 1;
                   k = (f == 0 || r <= 0) ? HUGE_VALQ : fabsq(a / (sqrtq(r) * f)); } break;
    case R_Atanh:{ const __float128 f = atanhq(a), r = 1 - a * a;
                   k = (f == 0 || r == 0) ? HUGE_VALQ : fabsq(a / (r * f)); }    break;

    // --- pow: a df/da / f = b;  b df/db / f = b ln a ------------------------
    case R_Pow:  { if (a <= 0) { k = fabsq(b); break; }      // ln a undefined; a-partial only
                   k = fabsq(b) + fabsq(b * logq(a)); }                          break;

    default: return false;
  }
  (void)c;
  if (isnanq(k)) return false;
  kappa = (double)k;
  return true;
}

// ---------------------------------------------------------------------------
// THE EXP-FAMILY ALGORITHM FLOOR  (step 1c)
// ---------------------------------------------------------------------------
// kappa is the condition number of the MATHEMATICAL map. It bounds what an
// idealised, correctly-rounded implementation would cost. That is the right
// bound for an op whose algorithm is exact or faithfully rounded — but exp is
// not one of those, in any of the four backends. All four use the same
// textbook skeleton:
//
//     s = x - m*ln2 ; r = s / 2^nq ; t = Taylor(r) ; SQUARE t nq TIMES
//
// (dd_math.hpp:374, ff_math.hpp:606, qf_math.hpp:927, tf_math.hpp:844.)
// Squaring propagates relative error at gain 2: (1+e)^2 = 1 + 2e + O(e^2). So
// nq squarings multiply whatever relative error the Taylor sum had by 2^nq.
// With the Taylor sum accurate to a few ulps, the OUTPUT of exp is a few * 2^nq
// ulps out — by construction, at every argument, however well conditioned.
//
// nq per backend, read from the headers, NOT fitted: DD 6, QF 6, TF 5, FF 4.
//
// This is what makes DD exp2 at grid point 110 read 156 ulps while kappa is
// 0.0022: 156 / 2^6 = 2.4, i.e. the Taylor sum was good to ~2.4 ulps and the
// six squarings turned that into 156. Step 1a called that a FAIL against a
// bound of 1.002; it is not a defect, it is the price of the reduction scheme,
// and the bound was simply missing a term. The measured ratios against 2^nq:
// DD 2.44, QF 0.47, TF 0.19, FF 0.03 — all inside the 8x allowance.
//
// Applied to every op that routes through exp(): exp, exp2, exp10 (exp of a
// scaled argument), pow (exp(b log a)), expm1, sinh, cosh, tanh (via sinhcosh
// / expm1). CONSERVATIVE on two of those: expm1 and sinhcosh take a direct
// Taylor branch for small |x| with no squarings at all, so the term is charged
// where it is not owed. That is a deliberate, stated looseness on one branch of
// two ops, and it is a far smaller distortion than leaving the whole exp family
// permanently red for a non-defect.
//
// NOT applied to sin/cos/tan, which use the same reduce-and-reconstruct shape
// (tf_math.hpp:917 uses nq = 4 with double-angle) but reconstruct through
// Chebyshev-like recurrences whose gain is not simply 2^nq. Deriving that gain
// honestly is step-1d work; charging them 2^nq on the strength of the analogy
// would be exactly the fitted-tolerance move this metric exists to remove.
//
// ---------------------------------------------------------------------------
// THE TRIG-FAMILY ABSOLUTE FLOOR  (consolidation; closes the step-1c deferral)
// ---------------------------------------------------------------------------
// step 1c refused to charge the circular family 2^nq "on the strength of the
// analogy" with exp, because the doubling recurrence is not a squaring. It is
// not — and once written out it gives a DIFFERENT and stricter law, which is
// why the analogy was rightly refused and the derivation is worth doing.
//
// All four backends reduce s3 = x - 2pi*nint(x/2pi), scale r = s3/2^nq, Taylor
// both series, then apply the JOINT doubling nq times (dd_math.hpp:812,
// ff_math.hpp:789, qf_math.hpp:1146, tf_math.hpp:966):
//
//     sin(2u) = 2 sin(u) cos(u)        cos(2u) = cos(u)^2 - sin(u)^2
//
// Track the ABSOLUTE error of the PAIR, e = max(|dsin|, |dcos|). One step:
//     |d(2sc)|      <= 2(|c| |ds| + |s| |dc|) <= 2e (|s| + |c|) <= 2e*sqrt(2)
//     |d(c^2-s^2)|  <= 2(|c||dc| + |s||ds|)   <= 2e
// and both outputs are themselves bounded by 1, so each step also adds one
// rounding of a quantity of magnitude <= 1, i.e. 2^-p absolute. The absolute
// error therefore at most DOUBLES per step and picks up 2^-p:
//
//     e_{j+1} <= 2 e_j + 2^-p   =>   e_nq <= 2^nq * (e_0 + 2^-p)
//
// with e_0 the Taylor sum's own absolute error, a couple of 2^-p. So
//
//     ABSOLUTE error of the (sin, cos) pair  ~  2^nq * 2^-p
//
// This is an ABSOLUTE floor, like the log family's, NOT a relative one like
// exp's — the difference the analogy would have got wrong. Two consequences
// the exp analogy does not have:
//
//   * it is divided by |f| to become ulps, so it explodes at the zeros of sin
//     and cos rather than being flat;
//   * it SATURATES. sin and cos live in [-1, 1], so the absolute error can
//     never exceed 2 however bad the reduction is. That ceiling is a property
//     of the CODOMAIN, and it is what bounds the huge-argument points.
//
// The reduction itself contributes a second absolute term. 2pi is stored to p
// bits, so 2pi*nint(x/2pi) carries an absolute error of |x|*2^-p, which lands
// directly on s3 and propagates to sin/cos with gain <= 1 (|f'| <= 1). Hence
//
//     Aulp(x) = min( 2^nq + |x| , 2^(p+1) )        [in units of 2^-p]
//
// and, for the forward pair, floor_ulps = Aulp / |f|.
//
// MEASURED against this tree, all at ratio < 1 (i.e. the law is an upper bound
// and is not slack by orders of magnitude):
//     DD sin  x=1.01     Aulp=32+1.01,  |f|=0.847 -> 39.0   measured 22.8
//     FF cos  x=pi(dbl)  Aulp=16+3.14,  |f|=1.00  -> 19.1   measured 13.8
//     DD asin x=0.3183   (inverse form below)     -> 104    measured 51.5
//     DD atan x~1        (inverse form below)     -> 57.5   measured 54.5
//
// The inverse functions are Newton iterations ON the forward pair (the same
// relationship log has to exp), so they inherit Aulp through the derivative of
// the forward function at the answer:
//
//     asin, acos : db = A / |cos b|  = A / sqrt(1-x^2)
//     atan       : db = A (1+|x|) |cos b|,  |cos b| = 1/sqrt(1+x^2)
//     tan = s/c  : dt = A (1 + |t|) / |c|
//
// divided in each case by the result's own magnitude to reach ulps.
//
// NOT applied to the hyperbolic family: sinh/cosh/tanh go through exp, not the
// circular doubling, and are already charged the exp-family 2^nq.
//
// One further format limit, and it is the KI-40 shape exactly -- it lives in an
// INTERMEDIATE that no input-side probe can see. The reduction multiple
// n = nint(x/2pi) is formed by the per-word round-to-nearest-integer trick, so
// it must be an exactly representable integer of ONE WORD: |x|/2pi < 2^s, with
// s the WORD significand (53 for an FP64 limb, 24 for an FP32 limb) -- NOT the
// expansion's p. Past that, nint saturates, the headers say so out loud
// ("DDNINT: argument too large", dd_math.hpp:820), the residual is garbage and
// the codomain guard returns the identity point (1, 0). The absolute error is
// then the full codomain diameter, and the saturated Aulp is the honest bound.
// Note this is keyed off the WORD, so DD -- with the most bits overall -- hits
// it at 2pi*2^53 while carrying 106 bits of precision, which is why DD reads
// WORSE than QF at x = 1e17. An input-width model predicts the opposite.
double trig_floor_ulps(int id, int trig_nq, int sig_bits, int word_bits,
                       __float128 aq, __float128 ref) {
  const double x  = std::fabs((double)aq);
  const double f  = std::fabs((double)ref);
  if (!(f > 0.0) || !std::isfinite(f)) return 1.0;
  const bool reduction_formable =
      (x / 6.283185307179586) < std::exp2((double)word_bits);
  // Absolute error of the (sin, cos) pair, in units of 2^-p. Saturates at the
  // codomain diameter 2 = 2^(p+1) * 2^-p: however lost the reduction is, sin
  // and cos cannot be more than 2 apart from the truth.
  double A = reduction_formable ? std::exp2((double)trig_nq) + x
                                : std::exp2((double)sig_bits + 1.0);
  const double sat = std::exp2((double)sig_bits + 1.0);
  if (!(A < sat)) A = sat;
  switch (id) {
    case R_Sin: case R_Cos:
      return A / f;
    case R_Tan: {
      // dt = A (1+|t|)/|c|, in ulps of t. c = cos(x) recovered as t/sin — use
      // the reference tan and the identity |c| = 1/sqrt(1+t^2) (sign-free).
      const double t = f;
      const double c = 1.0 / std::sqrt(1.0 + t * t);
      return (c > 0.0) ? A * (1.0 + t) / (c * t) : HUGE_VAL;
    }
    case R_Asin: case R_Acos: {
      const double s = std::sqrt(std::fabs(1.0 - x * x));
      return (s > 0.0) ? A / (s * f) : HUGE_VAL;
    }
    case R_Atan:
      return A * (1.0 + x) / (std::sqrt(1.0 + x * x) * f);
    default:
      return 1.0;
  }
}

double algo_floor_ulps(int id, int exp_squarings, __float128 a) {
  const double g = std::exp2((double)exp_squarings);   // the squaring gain
  switch (id) {
    case R_Exp: case R_Exp2: case R_Exp10: case R_Expm1:
    case R_Pow: case R_Sinh: case R_Cosh: case R_Tanh:
      return g;
    // The log family is Newton ON exp — dd:379, ff:70, qf:32, tf:35 all iterate
    // b <- b + (a - exp(b))/exp(b). The iteration cannot converge past the
    // accuracy of the exp it is inverting, so it leaves an ABSOLUTE error of
    // about g * 2^-p in the natural logarithm. Converting an absolute error to
    // ulps of the result divides by the result's own magnitude, so the floor is
    // g / |ln a| — which blows up near ln a = 0 (a -> 1) exactly as observed.
    // log2 and log10 are the same natural log rescaled by a constant; both the
    // absolute error and |f| scale together, so the ulp floor is unchanged.
    case R_Log: case R_Log2: case R_Log10: {
      if (a <= 0) return g;
      const double l = std::fabs((double)logq(a));
      return (l > 0.0) ? g / l : g * std::exp2((double)exp_squarings);
    }
    case R_Log1p: {
      if (a <= -1) return g;
      const double l = std::fabs((double)log1pq(a));
      return (l > 0.0) ? g / l : g * std::exp2((double)exp_squarings);
    }
    default:
      return 1.0;
  }
}

// ---------------------------------------------------------------------------
// THE CONSOLIDATED BOUND
// ---------------------------------------------------------------------------
// One expression, four terms, every one of them derived from the FORMAT (word
// type, limb count, exponent range), the ALGORITHM's published structure (the
// reduction depth nq read out of the headers), or the MATHEMATICAL condition
// number. No term is fitted to a measurement of this implementation.
//
//     bound_ulps = ( res + kappa * in_delta + kappa_int ) * gain + abs_floor
//
//   res       resolution floor at |ref| — 1 ulp normally, growing as 1/|ref|
//             once the trailing limbs go subnormal (subnormal_floor_ulps)
//   kappa     condition number of the mathematical map
//   in_delta  the input perturbation the format cannot avoid, in ulps
//   kappa_int INTERMEDIATE width: the algorithm rounds its intermediates into
//             the same p bits, so an intermediate I that is later cancelled
//             down to f contributes |I|/|f| ulps of the RESULT. This is the
//             term KI-40 is about; it is what makes a 53-bit double input
//             expensive for a 48/72/96-bit format even though the input itself
//             is held exactly.
//   gain      the algorithm's own amplification (2^nq for the reduce-square
//             families), which multiplies everything upstream of it rather
//             than sitting beside it — squaring amplifies the storage floor
//             and the conditioning error alike. Outside the subnormal band and
//             for gain == 1 this reduces EXACTLY to the previous bound.
//   abs_floor an additive absolute error the algorithm leaves behind that is
//             not a multiple of anything upstream (the log family's Newton
//             residue, the circular family's doubling residue).
struct Bound {
  double res, kappa, in_delta, kappa_int, gain, abs_floor, value;
};

double bound_value(const Bound& b) {
  return (b.res + b.kappa * b.in_delta + b.kappa_int) * b.gain + b.abs_floor;
}

// The multiplicative amplification of the reduce-and-recombine families, read
// out of the headers (nq), NOT fitted. exp squares nq times at gain 2 per
// squaring; log is Newton on exp and inherits it; pow/sinh/cosh/tanh route
// through exp. Everything else is 1.
double algo_gain(int id, int exp_nq) {
  switch (id) {
    case R_Exp: case R_Exp2: case R_Exp10: case R_Expm1:
    case R_Pow: case R_Sinh: case R_Cosh: case R_Tanh:
    case R_Log: case R_Log2: case R_Log10: case R_Log1p:
      return std::exp2((double)exp_nq);
    default:
      return 1.0;
  }
}

// Every complex transcendental is built from the REAL exp / sincos of the
// components (z = x+iy -> exp(x)*(cos y, sin y) and so on), so it inherits the
// real family's reduce-and-recombine amplification. Read from the same nq.
double complex_algo_gain(int id, int exp_nq) {
  switch (id) {
    case C_Add: case C_Sub: case C_Mul: case C_Div:
    case C_Abs: case C_Conj: case C_Sqrt:
      return 1.0;
    default:
      return std::exp2((double)exp_nq);
  }
}

// ---------------------------------------------------------------------------
// COMPLEX: kappa BY NUMERICAL DIFFERENTIATION OF THE ORACLE
// ---------------------------------------------------------------------------
// The real ops carry hand-derived, per-op kappa formulae (docs/CORRECTNESS.md).
// Writing 24 more by hand for the complex ops is where the previous attempt
// stopped, and it left the complex half of the sweep on the digit gate -- which
// is where KI-1 and KI-5 hid. There is no need to write them: kappa is a
// property of the MATHEMATICAL map, and the map is already implemented to 113
// bits in the oracle. Differentiate that.
//
//     kappa = sum over operands of  | (f(z_j(1+h)) - f(z_j)) / (f * h) |
//
// evaluated in binary128 with h = 2^-40, which leaves ~73 bits of headroom
// below the oracle's own precision, so the quotient is a derivative and not
// noise. This is the same first-order sum-of-partials the real formulae are,
// obtained by measurement OF THE ORACLE rather than of the implementation --
// the distinction that matters. It cannot encode an implementation defect as
// a specification, because the implementation is never called.
double kappa_complex_numeric(int id, __float128 are, __float128 aim,
                             __float128 bre, __float128 bim,
                             __float128 fre, __float128 fim) {
  const __float128 h = ldexpq((__float128)1.0, -40);
  const __float128 fm = hypotq(fre, fim);
  if (fm == 0 || !finiteq(fm)) return -1.0;            // no bound issuable
  // Four independent REAL directions, not two complex ones. Perturbing
  // a -> a(1+h) scales re and im together, which is a purely RADIAL move and
  // cannot see a tangential sensitivity. log(z) near |z| = 1 is the canonical
  // case: log|z| is flat there while arg(z) is not, so a radial probe reports
  // kappa ~ 1 for an operand that is in fact ill-conditioned.
  double kappa = 0.0;
  for (int j = 0; j < 4; ++j) {
    __float128 pa_re = are, pa_im = aim, pb_re = bre, pb_im = bim;
    switch (j) {
      case 0:  pa_re = are * (1 + h); break;
      case 1:  pa_im = aim * (1 + h); break;
      case 2:  pb_re = bre * (1 + h); break;
      default: pb_im = bim * (1 + h); break;
    }
    __float128 gre = 0, gim = 0;
    reference_complex_q(id, pa_re, pa_im, pb_re, pb_im, gre, gim);
    if (!finiteq(gre) || !finiteq(gim)) return -1.0;
    const __float128 d = hypotq(gre - fre, gim - fim) / (fm * h);
    if (!finiteq(d)) return -1.0;
    kappa += (double)d;
  }
  return kappa;
}

// The intermediates the complex algorithms actually form. Only the algebraic
// ops have any that are not the answer itself: the bilinear products of a
// complex multiply and the components of a divide. These are exactly the
// products that need 2 x 53 bits when the grid hands the backend two doubles,
// and exactly the ones that cancel -- KI-39's mechanism B, KI-40's clause 2.
int complex_intermediates(int id, __float128 are, __float128 aim,
                          __float128 bre, __float128 bim, __float128* out) {
  switch (id) {
    case C_Mul:
      out[0] = are * bre; out[1] = aim * bim;
      out[2] = are * bim; out[3] = aim * bre;   return 4;
    case C_Div: {
      const __float128 q = bre * bre + bim * bim;
      if (q == 0) return 0;
      out[0] = (are * bre) / q; out[1] = (aim * bim) / q;
      out[2] = (aim * bre) / q; out[3] = (are * bim) / q;
      // q itself is held in the working format. |b|^2 leaves the FP32 limb
      // range long before b does, so the divisor is a format limit living
      // entirely in an intermediate -- invisible from the operands.
      out[4] = q;                                          return 5;
    }
    case C_Abs:
      out[0] = are * are; out[1] = aim * aim;   return 2;
    // The inverse trig/hyperbolic family is log(z + sqrt(1 -+ z^2)) or
    // 1/2 log((1+z)/(1-z)). Both form a SUM whose addends can be much larger
    // than the sum -- 1 - z^2 near z = +-1, 1 - z near z = 1. The addend is
    // the intermediate the algorithm must hold in p bits, so it is the term
    // that sets the cancellation, exactly as a*b does for fma.
    case C_Asin: case C_Acos: case C_Asinh: case C_Acosh: {
      const __float128 zr = are * are - aim * aim;
      const __float128 zi = 2 * are * aim;
      out[0] = hypotq(zr, zi); out[1] = 1;       return 2;
    }
    case C_Atan: case C_Atanh:
      out[0] = hypotq(are, aim); out[1] = 1;     return 2;
    default:
      return 0;
  }
}

// ---------------------------------------------------------------------------
// INTERMEDIATES — the quantities the algorithm must hold in p bits on the way
// to the answer. Read off each op's expression tree, not guessed: these are the
// products and squares the headers actually form. Used for two things, clause
// (3) above (does the intermediate fit the exponent range at all) and the
// kappa_int cancellation term (how much of it survives into the answer).
// ---------------------------------------------------------------------------
int real_intermediates(int id, __float128 a, __float128 b, __float128 c,
                       __float128* out) {
  switch (id) {
    case R_Fma:    out[0] = a * b;  out[1] = c;      return 2;   // a*b + c
    case R_Hypot:  out[0] = a * a;  out[1] = b * b;  return 2;   // sqrt(a^2+b^2)
    case R_Mul:    out[0] = a * b;                   return 1;
    case R_Div:    out[0] = a / b;                   return 1;
    default:       (void)c;                          return 0;
  }
}

// kappa_int: the cancellation an intermediate suffers before it reaches the
// answer. Each intermediate is stored to p bits, so it carries |I| * 2^-p of
// absolute error; against a result of magnitude |f| that is |I|/|f| ulps.
// Where the intermediates cancel (|f| << |I|) this is the dominant term, and
// it is invisible to any probe that perturbs only the INPUTS -- which is
// exactly the blind spot KI-40 records.
double kappa_intermediate(const __float128* inter, int ninter, __float128 ref) {
  if (ninter == 0 || ref == 0 || !finiteq(ref)) return 0.0;
  const __float128 f = fabsq(ref);
  __float128 worst = 0;
  for (int k = 0; k < ninter; ++k) {
    if (!finiteq(inter[k])) continue;
    const __float128 r = fabsq(inter[k]) / f;
    if (r > worst) worst = r;
  }
  // An intermediate no larger than the answer cancels nothing; charge 0 there
  // so well-behaved ops keep the bound they had.
  return (worst > 1) ? (double)worst - 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// Backend bindings. Every math entry point is spelled the same way in all four
// headers, so one template body serves all of them.
// ---------------------------------------------------------------------------
// `range` carries what --classify needs to decide whether a point was OUTSIDE
// the backend's reach before the algorithm ever ran. Every field is a property
// of the WORD type and the limb count, not of any implementation:
//
//   min_sub    smallest nonzero magnitude the leading word can hold at all
//   min_norm   smallest normal magnitude of the leading word
//   max_val    largest finite magnitude of the leading word
//   full_prec  min_norm * 2^((limbs-1) * limb_bits) — the magnitude below which
//              the TRAILING limbs go subnormal, so the pair/triple/quad stops
//              carrying its nominal digit count. This is the number quoted as
//              "~1e-292" for DD and "~2e-31" for FF.
//   eps        2^-(limbs * limb_bits), the type's own unit roundoff, used as the
//              perturbation size for the conditioning probe.
struct Range {
  double min_sub, min_norm, max_val, full_prec, eps;
  int    limbs, limb_bits;
  // log10 of the three bounds. Every magnitude test in the classifier is done
  // in log space: a binary128 reference like exp(-1000) = 5e-435 is a perfectly
  // ordinary number to the oracle but collapses to 0.0 the moment it is cast to
  // double, which would silently misfile every underflow as UNEXPLAINED.
  double l10_min_sub, l10_full_prec, l10_max;
};

void fill_log_bounds(Range& r) {
  r.l10_min_sub   = std::log10(r.min_sub);
  r.l10_full_prec = std::log10(r.full_prec);
  r.l10_max       = std::log10(r.max_val);
}

// ---------------------------------------------------------------------------
// THE SUBNORMAL-LOW-WORD FLOOR  (step 1c)
// ---------------------------------------------------------------------------
// An N-limb expansion is N words of the base type and NOTHING ELSE. The value
// it represents is the exact sum of those words, so the finest ABSOLUTE
// increment it can express at any magnitude is the base type's smallest
// subnormal, 2^Emin_sub (2^-1074 for FP64 words, 2^-149 for FP32). The same
// bound applies to every residual an algorithm forms internally: a Dekker
// two-product or a Newton correction at scale |x|*2^-p is a plain word, and
// below 2^Emin_sub it flushes.
//
// So at magnitude |v| the format's achievable RELATIVE resolution is
//
//     res(|v|) = max(2^-p, 2^Emin_sub / |v|)
//
// and, expressed in the NOMINAL ulps the metric counts (units of |v| * 2^-p),
// the unavoidable error floor is
//
//     floor_ulps(|v|) = max(1, 2^(Emin_sub + p) / |v|)
//
// The two arms cross at |v| = 2^(Emin_sub + p) = 2 * Range::full_prec, where
// full_prec = min_norm * 2^((limbs-1)*limb_bits) is the magnitude below which
// the TRAILING limb can no longer be normal. (The factor 2 is the hidden bit:
// the last limb loses normality at full_prec, and one nominal ulp of headroom
// is gone one binade earlier.) Above the crossing the function returns 1 and
// the bound is byte-identically the step-1a bound. Below it the return value
// grows as 1/|v| and saturates at 2^p — the leading word is then itself the
// smallest subnormal and the expansion carries exactly one bit — so this is
// bounded, never infinite, and cannot silently swallow an op.
//
// The cliffs, DERIVED FROM THE FORMAT, not fitted. Per backend, the |hi| below
// which the trailing limb cannot be normal (= full_prec), and the crossing:
//
//     backend  p    Emin_sub   full_prec = 2^(Emin+(N-1)s)   crossing 2^(Emin_sub+p)
//     DD       106  -1074      2^-969  = 2.0042e-292         2^-968  = 4.0079e-292
//     QF        96   -149      2^-54   = 5.5511e-17          2^-53   = 1.1102e-16
//     TF        72   -149      2^-78   = 3.3087e-24          2^-77   = 6.6174e-24
//     FF        48   -149      2^-102  = 1.9722e-31          2^-101  = 3.9443e-31
//
// This is an INHERENT FORMAT LIMIT, not an implementation defect, and it is the
// same kind of statement UNDERFLOW and OVERFLOW already make.
double subnormal_floor_ulps(__float128 v, const Range& rg, int sig_bits) {
  if (v == 0 || !finiteq(v)) return 1.0;
  // log2 in quad: a binary128 magnitude like exp(-1000) is fine to the oracle
  // but collapses to 0.0 the instant it is cast to double.
  const double l2   = (double)log2q(fabsq(v));
  double       lfl  = std::log2(rg.min_sub) + (double)sig_bits - l2;
  // Clamp at 2^p. Once |v| drops below the word type's smallest subnormal the
  // format holds nothing at all there, and "holds nothing" is exactly 2^p
  // nominal ulps of error — you cannot be more wrong than the whole value.
  // Without this the bound runs away as 1/|v| and becomes vacuous.
  if (lfl > (double)sig_bits) lfl = (double)sig_bits;
  return (lfl <= 0.0) ? 1.0 : std::exp2(lfl);
}

// double words: 53-bit significand, min normal 2^-1022, min subnormal 2^-1074.
// float  words: 24-bit significand, min normal 2^-126,  min subnormal 2^-149.
Range range_double(int limbs) {
  Range r;
  r.limb_bits = 53;
  r.limbs     = limbs;
  r.min_sub   = std::ldexp(1.0, -1074);
  r.min_norm  = std::ldexp(1.0, -1022);
  r.max_val   = 1.7976931348623157e308;
  r.full_prec = std::ldexp(r.min_norm, (limbs - 1) * r.limb_bits);
  r.eps       = std::ldexp(1.0, -limbs * r.limb_bits);
  fill_log_bounds(r);
  return r;
}
Range range_float(int limbs) {
  Range r;
  r.limb_bits = 24;
  r.limbs     = limbs;
  r.min_sub   = std::ldexp(1.0, -149);
  r.min_norm  = std::ldexp(1.0, -126);
  r.max_val   = 3.4028234663852886e38;
  r.full_prec = std::ldexp(r.min_norm, (limbs - 1) * r.limb_bits);
  r.eps       = std::ldexp(1.0, -limbs * r.limb_bits);
  fill_log_bounds(r);
  return r;
}

// ---------------------------------------------------------------------------
// THE FOUR-CLAUSE UNRESOLVED PREDICATE  (replaces five classifier categories)
// ---------------------------------------------------------------------------
// The old apparatus had UNDERFLOW, OVERFLOW, ARG_RANGE, CONDITIONING and
// UNEXPLAINED, plus the SUBNORMAL_LIMB and JUMP_UNRESOLVED region labels bolted
// on beside them — seven names for "the measurement does not mean what it
// usually means here", each with its own rule. Every one of them is a special
// case of a single statement:
//
//     the format cannot carry this point, so no verdict is issuable.
//
// Four clauses, each a pure predicate over the format's exponent range and the
// derived bound. Nothing keys off an op name or an exclusion list.
//
//   (1) the ANSWER does not fit: |ref| is zero, non-finite, or outside
//       [min_sub, max_val] of the leading word.      (was UNDERFLOW/OVERFLOW)
//   (2) the QUESTION does not fit: some exact operand is outside that range,
//       so the backend was handed a different problem. (was ARG_RANGE-low)
//   (3) the WORK does not fit: some intermediate the algorithm must form is
//       outside the range — a product that overflows the word, a square that
//       flushes.                                      (was ARG_RANGE-high)
//   (4) the BOUND ITSELF says there is nothing left: it already predicts >= 2^p
//       ulps, i.e. >= 100% relative error. A verdict against a bound that
//       permits total loss adds no information.  (was SUBNORMAL_LIMB and
//       JUMP_UNRESOLVED, both of which reach 2^p exactly)
//
// Clause 4 is the load-bearing one: it is why JUMP_UNRESOLVED needs no label.
// An fmod whose integral quotient the format cannot resolve is charged its
// honest absolute error, one modulus |b|, which is >= |f| by definition of a
// remainder — so the bound reaches 2^p and clause 4 fires by arithmetic.
bool out_of_format(__float128 v, const Range& rg) {
  if (v == 0) return false;                      // exact zero is representable
  if (!finiteq(v)) return true;
  const __float128 m = fabsq(v);
  return m > (__float128)rg.max_val || m < (__float128)rg.min_sub;
}

bool point_unresolved(double bound, __float128 ref, const __float128* exact,
                      int nops, const __float128* inter, int ninter,
                      const Range& rg, int sig_bits) {
  if (ref == 0 || !finiteq(ref) || out_of_format(ref, rg)) return true;   // (1)
  for (int k = 0; k < nops; ++k)
    if (out_of_format(exact[k], rg)) return true;                         // (2)
  for (int k = 0; k < ninter; ++k)
    if (out_of_format(inter[k], rg)) return true;                         // (3)
  return !(bound < std::exp2((double)sig_bits));                          // (4)
}

// sig_bits() is limbs x word-significand (FP64 53, FP32 24, hidden bit
// included) — the p in ulp(true) = |true| * 2^-p. Same arithmetic the SCORING
// section uses to argue that widening to binary128 is exact.
struct BackendDD { using S = xp::DoubleDouble; using Z = xp::DoubleDoubleComplex;
                   static const char* name() { return "DD"; } static double cap() { return 31.0; }
                   static int sig_bits() { return 106; }   // 2 x 53
                   static int exp_squarings() { return 6; }  // dd_math.hpp:374
                   static int trig_squarings() { return 5; } // dd_math.hpp sincos nq
                   static Range range() { return range_double(2); } };
struct BackendFF { using S = xp::FloatFloat;   using Z = xp::FloatFloatComplex;
                   static const char* name() { return "FF"; } static double cap() { return 14.0; }
                   static int sig_bits() { return 48; }    // 2 x 24
                   static int exp_squarings() { return 4; }  // ff_math.hpp:606
                   static int trig_squarings() { return 4; } // ff_math.hpp sincos nq
                   static Range range() { return range_float(2); } };
struct BackendQF { using S = xp::QuadFloat;    using Z = xp::QuadFloatComplex;
                   static const char* name() { return "QF"; } static double cap() { return 29.0; }
                   static int sig_bits() { return 96; }    // 4 x 24
                   static int exp_squarings() { return 6; }  // qf_math.hpp:927
                   static int trig_squarings() { return 5; } // qf_math.hpp sincos nq
                   static Range range() { return range_float(4); } };
struct BackendTF { using S = xp::TripleFloat;  using Z = xp::TripleFloatComplex;
                   static const char* name() { return "TF"; } static double cap() { return 21.7; }
                   static int sig_bits() { return 72; }    // 3 x 24
                   static int exp_squarings() { return 5; }  // tf_math.hpp:844
                   static int trig_squarings() { return 4; } // tf_math.hpp sincos nq
                   static Range range() { return range_float(3); } };

template <class S>
S eval_real(int id, const S& a, const S& b, const S& c) {
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
// (complex abs), so the caller knows the imaginary reference is a true zero and
// not a discarded component.
template <class S, class Z>
Z eval_complex(int id, const Z& a, const Z& b, bool& is_real) {
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
    // theta — the convention src/demo_*_complex.cpp and gen_corpus both use.
    case C_Polar: return xp::polar(a.re, a.im);
  }
  return a;
}

// ===========================================================================

// ---------------------------------------------------------------------------
// A scored row, in emission order.
// ---------------------------------------------------------------------------
// state: S scored, U unresolved (no verdict issuable), N unscorable, X no
// analytic/numeric kappa. One row per (backend, kind, op, point) -- the single
// measurement of record.
struct Row {
  const char* backend;
  char        kind;        // 'r' or 'c'
  const char* op;
  int         point;
  double      digits;
  double ulps, bound;
  char   state;
};

// One (backend, op) aggregate, for the human-readable summary.
struct Cell {
  const char* backend;
  char        kind;
  const char* op;
  double      cap;
  double      sum;
  double      min;
  long        n;
  long        n_zero;      // points scoring exactly 0.00 — total loss
};

// ---------------------------------------------------------------------------
// ULP GATE (--ulp). Bookkeeping for the condition-aware ulp verdict.
//
// MINIMUM, NOT MEAN. Every aggregate below is a WORST-POINT reduction. The old
// tolerance tables gate on the mean over a cell, which cannot see an op that is
// exact at 1600 points and returns nothing at 52 — that still averages ~30 and
// passes. Here a single bad point fails its region.
//
// REGIONS. A failing point is run through the existing --classify machinery and
// filed under its class. UNDERFLOW and OVERFLOW are REPORTED BUT NOT GATED: in
// those bands |true| * 2^-p is not the backend's actual resolution (the trailing
// limbs are subnormal, or the value is outside the leading word's range), so the
// metric's own premise has failed and a verdict there would be noise. Since
// 2026-09-04 JUMP_UNRESOLVED joins them, for the same reason: see
// jump_unresolved(). The other classes — ARG_RANGE, CONDITIONING, UNEXPLAINED,
// and OK (the point was never bad enough to classify) — are gated.
// Filing per region is what stops one
// legitimate conditioning loss from forcing the whole op's bound open.
// ---------------------------------------------------------------------------
const char* const kRegionOK = "OK";
// The single "no verdict issuable" label, produced by point_unresolved().
// It replaces UNDERFLOW, OVERFLOW, ARG_RANGE, SUBNORMAL_LIMB and
// JUMP_UNRESOLVED, which were five names for one fact.
const char* const kRegionUnresolved = "UNRESOLVED";
const char* const kRegionAboveBound = "ABOVE_BOUND";

// Step 1c: the third format-derived exemption class, alongside UNDERFLOW and
// OVERFLOW. The point is in the band where the expansion's trailing limb cannot
// be normal, so |true| * 2^-p is not the backend's actual resolution and the
// metric's own premise has failed. See subnormal_floor_ulps().
const char* const kRegionSubnormalLimb = "SUBNORMAL_LIMB";

// KI-10/KI-15: the operands straddle a jump of fmod/remainder at a distance
// the format cannot resolve. Derivation and measurements: jump_unresolved().
const char* const kRegionJumpUnresolved = "JUMP_UNRESOLVED";

// KI-10/KI-15, 2026-09-04: the fourth format-derived exemption, and the only
// one that is a DISCONTINUITY rather than a resolution floor.
//
// DERIVATION. fmod/remainder are f(a,b) = a - n*b with n = trunc(a/b) (resp.
// nearbyint). The backend does not see (a,b); it sees the stored pair
// A = a(1+alpha), B = b(1+beta) with |alpha|,|beta| <= 2^-p, p = sig_bits.
// Only FF is affected: p = 48 < 53, so a double operand does not survive
// storage. DD (106), QF (96) and TF (72) all hold a double exactly, alpha =
// beta = 0, and this predicate can never fire for them.
//
//   Regime 1 - n unchanged. R = A - nB = r + a*alpha - n*b*beta, so
//     |R - r| <= (|a| + |n b|) 2^-p, i.e. at most (|a| + |n b|)/|r| = kappa
//   ulps of r. This is EXACTLY the kappa already returned by kappa_real, so
//   the ordinary bound covers regime 1 and no exemption is wanted. (Measured:
//   every such point passes; the FF cell's worst regime-1 ratio is well under
//   the allowance.)
//
//   Regime 2 - n changes. trunc/nearbyint are discontinuous. When a/b lies
//   within 2|a/b|2^-p of an integer, trunc(A/B) can differ from trunc(a/b) by
//   one, and then
//     R - r = -/+ b     regardless of how small the perturbation was,
//   which is 2^p |b/r| ulps of r. Since |r| <= |b| always, that is >= 2^p ulps
//   -- ZERO correct digits -- and it is a JUMP, so the first-order kappa does
//   not bound it even in principle. The op comment above kappa_real already
//   states the caveat ("at the jump the function is discontinuous and no
//   first-order bound applies"); this predicate is that caveat made testable.
//
// The test below is not a distance heuristic: it recomputes n at the exact
// operands and at the stored operands and asks whether they differ. If they
// do, the format provably cannot tell which side of the jump the point is on,
// so the metric has no verdict to give and the point is exempt.
//
// MEASURED, 2026-09-04, all 6 points this fires on (FF fmod, grid points 708,
// 743, 883, 995, 1548, 1590 -- all in the i%7==1 "cancelling" family, where
// b = a(1+eps)): eps ranges 2.22e-16 .. 8.88e-16, every one BELOW 2^-48 =
// 3.55e-15, so round48(a) == round48(b) bit-for-bit. Exact a/b < 1 gives n = 0
// and f = a; stored A/B == 1 exactly gives n = 1 and f = 0. Measured error
// 2.815e14 ulps = 2^48 at all six, i.e. the regime-2 ceiling is ATTAINED, not
// exceeded. The returned value is bit-identical to fmodq evaluated at the
// backend's own stored operands at all six. There is nothing to fix in the
// implementation; a 2xFP32 pair cannot resolve the two operands apart.
bool jump_unresolved(int id, const __float128* stored, const __float128* exact) {
  if (id != R_Fmod && id != R_Remainder) return false;
  if (stored[1] == 0 || exact[1] == 0) return false;
  const __float128 qe = exact[0] / exact[1], qs = stored[0] / stored[1];
  if (!finiteq(qe) || !finiteq(qs)) return false;
  const __float128 ne = (id == R_Fmod) ? truncq(qe) : nearbyintq(qe);
  const __float128 ns = (id == R_Fmod) ? truncq(qs) : nearbyintq(qs);
  return ne != ns;
}

struct UlpFail {
  const char* backend; char kind; const char* op; int point;
  const char* region;
  double ulps, kappa, expected, ratio, digits;
  double out_floor, in_delta, a0;   // step 1c: the two ends of the bound, and x
};

struct UlpCell {
  const char* backend; char kind; const char* op;
  long   n_scored, n_unscorable, n_ungated, n_unresolved, n_fail;
  double worst_ratio;  int worst_point;
  double worst_ulps, worst_expected, worst_kappa, worst_digits;
  double max_ulps;     int max_ulps_point;   // biggest raw ulp error, gated or not
};

struct UlpCtx {
  double                allowance;
  std::vector<UlpCell>* cells;
  std::vector<UlpFail>* fails;
  // --ulp-explain OP:POINT — record the full breakdown for one grid point on
  // every backend, pass or fail. This is the microscope the metric is argued
  // with; without it a claim about a single point cannot be reproduced.
  const char*           explain_op;
  int                   explain_point;
  std::vector<UlpFail>* explains;
};

UlpCell make_ulp_cell(const char* be, char kind, const char* op) {
  UlpCell u;
  u.backend = be; u.kind = kind; u.op = op;
  u.n_scored = u.n_unscorable = u.n_ungated = u.n_unresolved = u.n_fail = 0;
  u.worst_ratio = 0.0; u.worst_point = -1;
  u.worst_ulps = u.worst_expected = u.worst_kappa = u.worst_digits = 0.0;
  u.max_ulps = 0.0; u.max_ulps_point = -1;
  return u;
}

// Input perturbation the backend cannot avoid, expressed in ulps of the input.
// Normally 1 (the stored value is the input, rounded). But a double input does
// NOT fit FF's 48 bits, so for FF the stored operand is already off by more than
// one ulp of the FF format before the algorithm runs; charge that honestly
// instead of pretending the argument was exact.
// Relative distance between the operand the backend actually stored and the
// exact operand the oracle was asked about. This is the "different question"
// term in the bound: the backend is not wrong for answering the question it was
// given.
double storage_rel_err(__float128 stored, __float128 exact) {
  if (exact == 0) return (stored == 0) ? 0.0 : 1.0;
  if (!finiteq(exact) || !finiteq(stored)) return 0.0;
  const __float128 e = fabsq(stored - exact) / fabsq(exact);
  return (double)e;
}

double input_delta_ulps(const __float128* stored, const __float128* exact,
                        int nops, int sig_bits) {
  double worst = 1.0;
  for (int i = 0; i < nops; ++i) {
    const double e = storage_rel_err(stored[i], exact[i]);
    const double in_ulps = std::ldexp(e, sig_bits);
    if (in_ulps > worst) worst = in_ulps;
  }
  return worst;
}

template <class B>
void sweep_real(int id, const std::vector<GridPoint>& grid,
                const std::vector<double>& a_in, const std::vector<double>& b_in,
                const std::vector<double>& c_in, const std::vector<__float128>& ref,
                std::vector<Row>& rows, std::vector<Cell>& cells,
                const UlpCtx* ux) {
  typedef typename B::S S;
  Cell cell = {B::name(), 'r', kReal[id].name, B::cap(), 0.0, B::cap(), 0, 0};
  UlpCell ucell = make_ulp_cell(B::name(), 'r', kReal[id].name);
  const Range rg = B::range();
  const int   sb_bits = B::sig_bits();
  const int   nops    = kReal[id].nops;
  for (size_t i = 0; i < grid.size(); ++i) {
    const S sa(a_in[i]), sb(b_in[i]), sc(c_in[i]);
    const S r = eval_real<S>(id, sa, sb, sc);
    const double d = score_scalar(to_q(r), ref[i], B::cap());
    rows.push_back({B::name(), 'r', kReal[id].name, int(i), d, -1.0, -1.0, 'N'});
    Row& row_ = rows.back();
    // --- condition-aware ulp verdict -------------------------------------
    if (ux) {
      const double m = ulps_scalar(to_q(r), ref[i], sb_bits);
      double kappa = 0.0;
      row_.ulps = m;
      if (m == kUnscorableUlps()) {
        ++ucell.n_unscorable; row_.state = 'N';
      } else if (!kappa_real(id, (__float128)a_in[i], (__float128)b_in[i],
                             (__float128)c_in[i], kappa)) {
        ++ucell.n_ungated; row_.state = 'X';    // no analytic bound
      } else {
        const __float128 stored[3] = {to_q(sa), to_q(sb), to_q(sc)};
        const __float128 exact[3]  = {(__float128)a_in[i], (__float128)b_in[i],
                                      (__float128)c_in[i]};
        // Step 1c. Both ends of the bound pick up the subnormal-low-word floor:
        //   * out_floor — the result cannot be STORED finer than 2^Emin_sub;
        //   * in_floor  — no operand can be RESOLVED finer than that either, and
        //                 the residuals an algorithm forms from it live at the
        //                 operand's scale, so the floor enters through kappa.
        // Above the cliff both are exactly 1 and this is the step-1a bound.
        const double out_floor = subnormal_floor_ulps(ref[i], rg, sb_bits);
        double in_floor = input_delta_ulps(stored, exact, nops, sb_bits);
        for (int k = 0; k < nops; ++k) {
          const double f = subnormal_floor_ulps(exact[k], rg, sb_bits);
          if (f > in_floor) in_floor = f;
        }
        double af = algo_floor_ulps(id, B::exp_squarings(), (__float128)a_in[i]);
        {   // the circular family's own absolute residue (see trig_floor_ulps)
          const double tf = trig_floor_ulps(id, B::trig_squarings(), sb_bits,
                                            rg.limb_bits, (__float128)a_in[i], ref[i]);
          if (tf > af) af = tf;
        }
        // Intermediates: clause (3) of the unresolved predicate, and the
        // kappa_int cancellation term (KI-40).
        __float128 inter[3];
        const int  ninter = real_intermediates(id, exact[0], exact[1], exact[2],
                                               inter);
        double k_int = kappa_intermediate(inter, ninter, ref[i]);
        // KI-10/KI-15 as a BOUND rather than a label. When the stored operands
        // put the integral quotient on the other side of the jump, the returned
        // remainder is a different branch of a discontinuous function and the
        // absolute error is up to one modulus |b|. In ulps of f that is
        // 2^p*|b|/|f|, which is >= 2^p because |f| < |b| for every remainder --
        // so clause (4) fires and the point is unresolved, by arithmetic
        // rather than by an exemption list.
        if (jump_unresolved(id, stored, exact) && ref[i] != 0) {
          const double jump = std::ldexp((double)(fabsq(exact[1]) / fabsq(ref[i])),
                                         sb_bits);
          if (jump > k_int) k_int = jump;
        }
        Bound bd;
        bd.res       = out_floor;
        bd.kappa     = kappa;
        bd.in_delta  = in_floor;
        bd.kappa_int = k_int;
        bd.gain      = algo_gain(id, B::exp_squarings());
        bd.abs_floor = (af > 1.0) ? af : 0.0;
        bd.value     = bound_value(bd);
        const double exp_u = bd.value;
        if (point_unresolved(exp_u, ref[i], exact, nops, inter, ninter,
                             rg, sb_bits)) {
          ++ucell.n_unresolved; row_.state = 'U'; row_.bound = exp_u;
          if (ux->explains && ux->explain_point == int(i) &&
              std::strcmp(ux->explain_op, kReal[id].name) == 0)
            ux->explains->push_back({B::name(), 'r', kReal[id].name, int(i),
                                     kRegionUnresolved, m, kappa, exp_u, 0.0, d,
                                     out_floor > af ? out_floor : af,
                                     in_floor, a_in[i]});
          cell.sum += d;
          if (d < cell.min) cell.min = d;
          if (d == 0.0) ++cell.n_zero;
          ++cell.n;
          continue;
        }
        const double ratio = (exp_u > 0.0) ? m / exp_u : HUGE_VAL;
        ++ucell.n_scored; row_.state = 'S'; row_.bound = exp_u;
        if (m > ucell.max_ulps) { ucell.max_ulps = m; ucell.max_ulps_point = int(i); }
        if (ratio > ucell.worst_ratio) {
          ucell.worst_ratio = ratio;    ucell.worst_point    = int(i);
          ucell.worst_ulps  = m;        ucell.worst_expected = exp_u;
          ucell.worst_kappa = kappa;    ucell.worst_digits   = d;
        }
        if (ux->explains && ux->explain_point == int(i) &&
            std::strcmp(ux->explain_op, kReal[id].name) == 0)
          ux->explains->push_back({B::name(), 'r', kReal[id].name, int(i), kRegionOK,
                                   m, kappa, exp_u, ratio, d,
                                   out_floor > af ? out_floor : af,
                                   in_floor, a_in[i]});
        if (ratio > ux->allowance) {
          // ONE verdict. The point is above its derived bound; there is no
          // second axis to file it under, because every reason a point could
          // legitimately be above the OLD bound is now a TERM IN the bound and
          // every reason it could be unmeasurable is a clause of
          // point_unresolved(). What is left is the finding itself.
          ++ucell.n_fail;
          ux->fails->push_back({B::name(), 'r', kReal[id].name, int(i),
                                kRegionAboveBound, m, kappa, exp_u, ratio, d,
                                out_floor > af ? out_floor : af,
                                in_floor, a_in[i]});
        }
      }
    }
    cell.sum += d;
    if (d < cell.min) cell.min = d;
    if (d == 0.0) ++cell.n_zero;
    ++cell.n;
  }
  cells.push_back(cell);
  if (ux) ux->cells->push_back(ucell);
}

template <class B>
void sweep_complex(int id, const std::vector<GridPoint>& grid,
                   const std::vector<double>& b_re, const std::vector<double>& b_im,
                   const std::vector<__float128>& ref_re, const std::vector<__float128>& ref_im,
                   std::vector<Row>& rows, std::vector<Cell>& cells,
                   const UlpCtx* ux) {
  typedef typename B::S S;
  typedef typename B::Z Z;
  Cell cell = {B::name(), 'c', kComplex[id].name, B::cap(), 0.0, B::cap(), 0, 0};
  UlpCell ucell = make_ulp_cell(B::name(), 'c', kComplex[id].name);
  const Range rg = B::range();
  for (size_t i = 0; i < grid.size(); ++i) {
    const Z a{S(grid[i].re), S(grid[i].im)};
    const Z b{S(b_re[i]), S(b_im[i])};
    bool     is_real = false;
    const Z  r  = eval_complex<S, Z>(id, a, b, is_real);
    const double dre = score_component(to_q(r.re), ref_re[i], ref_im[i], B::cap());
    const double dim = is_real ? B::cap()
                               : score_component(to_q(r.im), ref_im[i], ref_re[i], B::cap());
    const double d = dre < dim ? dre : dim;
    rows.push_back({B::name(), 'c', kComplex[id].name, int(i), d, -1.0, -1.0, 'N'});
    Row& row_ = rows.back();
    // --- ulp VERDICT for complex: the same bound as the real ops ----------
    // Modulus form: |got - ref| / (|ref| * 2^-p), judged against the same
    // consolidated bound, with kappa obtained by differentiating the oracle
    // (kappa_complex_numeric) instead of by hand.
    if (ux) {
      const __float128 e_re = to_q(r.re) - ref_re[i], e_im = to_q(r.im) - ref_im[i];
      const __float128 mref = hypotq(ref_re[i], ref_im[i]);
      const bool bad = isnanq(e_re) || isnanq(e_im) ||
                       isnanq(ref_re[i]) || isnanq(ref_im[i]) ||
                       !finiteq(ref_re[i]) || !finiteq(ref_im[i]);
      if (bad || mref == 0) {
        ++ucell.n_unscorable;
      } else {
        const double m = (double)ldexpq(hypotq(e_re, e_im) / mref, B::sig_bits());
        row_.ulps = m;
        if (m > ucell.max_ulps) { ucell.max_ulps = m; ucell.max_ulps_point = int(i); }
        const int sb_bits = B::sig_bits();
        const double kappa = kappa_complex_numeric(id, (__float128)grid[i].re,
                                                   (__float128)grid[i].im,
                                                   (__float128)b_re[i],
                                                   (__float128)b_im[i],
                                                   ref_re[i], ref_im[i]);
        if (kappa < 0.0) {
          ++ucell.n_ungated; row_.state = 'X';   // no derivative -> no bound
        } else {
          const __float128 exact[4] = {(__float128)grid[i].re, (__float128)grid[i].im,
                                       (__float128)b_re[i], (__float128)b_im[i]};
          const __float128 stored[4] = {to_q(a.re), to_q(a.im), to_q(b.re), to_q(b.im)};
          // Only the operands this op actually consumes: a unary op must not be
          // charged the storage error or the subnormal floor of an unused b,
          // which the grid fills with an arbitrary magnitude.
          const int nop4 = (kComplex[id].nops >= 2) ? 4 : 2;
          // res: the resolution floor at the RESULT's modulus, and the same
          // floor on each operand, exactly as the real path does. This is the
          // whole of KI-39 mechanism A -- an FP32-worded expansion at 1e-30 has
          // a subnormal trailing limb and cannot compute there, however many
          // limbs it has.
          const double out_floor = subnormal_floor_ulps(mref, rg, sb_bits);
          double in_floor = input_delta_ulps(stored, exact, nop4, sb_bits);
          for (int k = 0; k < nop4; ++k) {
            const double f2 = subnormal_floor_ulps(exact[k], rg, sb_bits);
            if (f2 > in_floor) in_floor = f2;
          }
          __float128 inter[8];
          const int ninter = complex_intermediates(id, exact[0], exact[1],
                                                   exact[2], exact[3], inter);
          const double k_int = kappa_intermediate(inter, ninter, mref);
          Bound bd;
          bd.res = out_floor; bd.kappa = kappa; bd.in_delta = in_floor;
          bd.kappa_int = k_int;
          bd.gain = complex_algo_gain(id, B::exp_squarings());
          bd.abs_floor = 0.0;
          bd.value = bound_value(bd);
          row_.bound = bd.value;
          if (point_unresolved(bd.value, mref, exact, nop4, inter, ninter,
                               rg, sb_bits)) {
            ++ucell.n_unresolved; row_.state = 'U';
          } else {
            const double ratio = (bd.value > 0.0) ? m / bd.value : HUGE_VAL;
            ++ucell.n_scored; row_.state = 'S';
            if (ratio > ucell.worst_ratio) {
              ucell.worst_ratio = ratio;  ucell.worst_point    = int(i);
              ucell.worst_ulps  = m;      ucell.worst_expected = bd.value;
              ucell.worst_kappa = kappa;  ucell.worst_digits   = d;
            }
            if (ratio > ux->allowance) {
              ++ucell.n_fail;
              ux->fails->push_back({B::name(), 'c', kComplex[id].name, int(i),
                                    kRegionAboveBound, m, kappa, bd.value, ratio,
                                    d, out_floor, in_floor, grid[i].re});
            }
          }
        }
      }
    }
    cell.sum += d;
    if (d < cell.min) cell.min = d;
    if (d == 0.0) ++cell.n_zero;
    ++cell.n;
  }
  cells.push_back(cell);
  if (ux) ux->cells->push_back(ucell);
}

// ---------------------------------------------------------------------------
// The sweep proper. Inputs and oracle references are computed ONCE per op and
// shared across the four backends, so every backend is scored on identical data
// and against an identical reference — the comparability property gen_corpus
// exists to provide, applied to a grid instead of a sample.
// ---------------------------------------------------------------------------
void run_sweep(uint64_t seed, const std::vector<GridPoint>& rgrid,
               const std::vector<GridPoint>& cgrid,
               std::vector<Row>& rows, std::vector<Cell>& cells,
               const UlpCtx* ux) {
  const size_t nr = rgrid.size(), nc = cgrid.size();
  rows.reserve(4 * (R_COUNT * nr + C_COUNT * nc));

  for (int id = 0; id < R_COUNT; ++id) {
    Rng rng(stream_seed(seed, kReal[id].name, 0u));
    std::vector<double>     a(nr), b(nr), c(nr);
    std::vector<__float128> ref(nr);
    for (size_t i = 0; i < nr; ++i) {
      double av = rgrid[i].re, bv, cv;
      fill_real_operands(id, i, av, rng, bv, cv);
      repair_real(id, av, bv, cv);
      a[i] = av; b[i] = bv; c[i] = cv;
      ref[i] = reference_real(id, av, bv, cv);
    }
    sweep_real<BackendDD>(id, rgrid, a, b, c, ref, rows, cells, ux);
    sweep_real<BackendFF>(id, rgrid, a, b, c, ref, rows, cells, ux);
    sweep_real<BackendQF>(id, rgrid, a, b, c, ref, rows, cells, ux);
    sweep_real<BackendTF>(id, rgrid, a, b, c, ref, rows, cells, ux);
  }

  for (int id = 0; id < C_COUNT; ++id) {
    Rng rng(stream_seed(seed, kComplex[id].name, 1u));
    std::vector<double>     bre(nc), bim(nc);
    std::vector<__float128> rre(nc), rim(nc);
    for (size_t i = 0; i < nc; ++i) {
      fill_complex_operands(id, i, cgrid, cgrid[i].re, cgrid[i].im, rng, bre[i], bim[i]);
      reference_complex(id, cgrid[i].re, cgrid[i].im, bre[i], bim[i], rre[i], rim[i]);
    }
    sweep_complex<BackendDD>(id, cgrid, bre, bim, rre, rim, rows, cells, ux);
    sweep_complex<BackendFF>(id, cgrid, bre, bim, rre, rim, rows, cells, ux);
    sweep_complex<BackendQF>(id, cgrid, bre, bim, rre, rim, rows, cells, ux);
    sweep_complex<BackendTF>(id, cgrid, bre, bim, rre, rim, rows, cells, ux);
  }
}

// ---------------------------------------------------------------------------
// I/O.
// ---------------------------------------------------------------------------
bool write_baseline(const std::string& path, const std::vector<Row>& rows,
                    size_t nr, size_t nc, uint64_t seed) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { std::fprintf(stderr, "cannot open %s for writing\n", path.c_str()); return false; }
  std::fprintf(f, "# %s\n", kFormatVersion);
  std::fprintf(f, "# dense accuracy sweep baseline; see scripts/sweep_accuracy.cpp\n");
  std::fprintf(f, "# grid: real=%zu complex=%zu  seed=%llu\n",
               nr, nc, (unsigned long long)seed);
  std::fprintf(f, "# caps: DD=31.00 FF=14.00 QF=29.00 TF=21.70\n");
  std::fprintf(f, "# rows: %zu\n", rows.size());
  std::fprintf(f, "# oracle-fingerprint: %016llx\n",
               (unsigned long long)oracle_fingerprint());
  std::fprintf(f, "# built-by: gcc %s   (regenerate under the toolchain of record:\n"
                  "#   module use /soft/modulefiles && module load gcc/13.3.0)\n",
               __VERSION__);
  std::fprintf(f, "# ulps  = |got-ref| / (|ref| * 2^-p), the measurement of record\n");
  std::fprintf(f, "# bound = the derived bound; see docs/CORRECTNESS.md\n");
  std::fprintf(f, "# state = S scored | U unresolved | N unscorable | X no kappa\n");
  std::fprintf(f, "backend,kind,op,point,digits,ulps,bound,state\n");
  char buf[96];
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    const int  n = std::snprintf(buf, sizeof(buf), "%s,%c,%s,%d,%.2f,%.6g,%.6g,%c\n",
                                 r.backend, r.kind, r.op, r.point, r.digits,
                                 r.ulps, r.bound, r.state);
    if (std::fwrite(buf, 1, size_t(n), f) != size_t(n)) {
      std::fprintf(stderr, "write failed\n"); std::fclose(f); return false;
    }
  }
  if (std::fclose(f) != 0) { std::fprintf(stderr, "close failed\n"); return false; }
  return true;
}

bool write_grid(const std::string& path, const std::vector<GridPoint>& rg,
                const std::vector<GridPoint>& cg) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { std::fprintf(stderr, "cannot open %s for writing\n", path.c_str()); return false; }
  std::fprintf(f, "# %s grid manifest\n", kFormatVersion);
  std::fprintf(f, "# the shared input grid the baseline's `point` column indexes.\n");
  std::fprintf(f, "# `re`/`im` are the FIRST operand; second and third operands are\n");
  std::fprintf(f, "# per-op and derived deterministically — see scripts/sweep_accuracy.cpp.\n");
  std::fprintf(f, "# real inputs are additionally passed through the per-op domain repair.\n");
  std::fprintf(f, "kind,point,family,re,im\n");
  for (size_t i = 0; i < rg.size(); ++i)
    std::fprintf(f, "r,%zu,%s,%.17g,\n", i, rg[i].family, rg[i].re);
  for (size_t i = 0; i < cg.size(); ++i)
    std::fprintf(f, "c,%zu,%s,%.17g,%.17g\n", i, cg[i].family, cg[i].re, cg[i].im);
  if (std::fclose(f) != 0) { std::fprintf(stderr, "close failed\n"); return false; }
  return true;
}

// Classification CSV. One row per point below the threshold, carrying enough
// evidence to re-derive the verdict by hand: the inputs the oracle saw, the
// reference, what the backend returned, and the three measured ceilings
// (repr = digits the input survived storage with, range = digits the RESULT's
// magnitude leaves room for, ach = digits any implementation could achieve).

void print_summary(const std::vector<Cell>& cells) {
  std::printf("\n  %-3s %-4s %-9s %8s %8s %8s %8s %8s\n",
              "be", "kind", "op", "mean", "min", "cap", "zeros", "n");
  std::printf("  --- ---- --------- -------- -------- -------- -------- --------\n");
  int below_cap = 0, with_zero = 0;
  for (size_t i = 0; i < cells.size(); ++i) {
    const Cell& c = cells[i];
    const double mean = c.n ? c.sum / double(c.n) : 0.0;
    if (mean < c.cap - 0.005) ++below_cap;
    if (c.n_zero > 0) ++with_zero;
    std::printf("  %-3s %-4s %-9s %8.2f %8.2f %8.2f %8ld %8ld\n",
                c.backend, c.kind == 'r' ? "real" : "cplx", c.op,
                mean, c.min, c.cap, c.n_zero, c.n);
  }
  std::printf("\n  (backend, op) cells            : %zu\n", cells.size());
  std::printf("  ... with mean below their cap  : %d\n", below_cap);
  std::printf("  ... containing a 0.00 point    : %d\n", with_zero);
}

// ---------------------------------------------------------------------------
// Compare mode — THE MONOTONE GATE.
//
// A fresh sweep is run in memory and diffed against a recorded baseline. Any
// point whose digits DECREASED is a regression and the tool exits nonzero.
// Increases are counted and reported but never fail: getting better is the
// point of the fixes this gate exists to protect.
//
// The baseline is compared POSITIONALLY, not by lookup: both sides are produced
// by the same deterministic emission order, so a key mismatch means the grid or
// the op inventory changed, which invalidates the comparison outright and is
// reported as an error rather than silently skipped.
// ---------------------------------------------------------------------------
int compare_baseline(const std::string& path, const std::vector<Row>& fresh) {
  // The committed baseline is gzipped -- 428,592 rows is 8.7 MB of text and
  // 2.6 MB compressed, and it is rewritten on every accepted change, so the
  // uncompressed form dominates repository growth. Read it through gzip -dc
  // rather than keeping a second, decompressed copy on disk.
  const bool gz = path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
  std::FILE* f = nullptr;
  if (gz) {
    const std::string cmd = "gzip -dc '" + path + "'";
    f = popen(cmd.c_str(), "r");
  } else {
    f = std::fopen(path.c_str(), "rb");
  }
  if (!f) { std::fprintf(stderr, "cannot open baseline %s\n", path.c_str()); return 2; }

  char   line[256];
  size_t idx = 0, parsed = 0;
  long   decreased = 0, increased = 0, worst_shown = 0;
  long   state_moved = 0, state_shown = 0;
  double worst_drop = 0.0;
  bool   saw_header = false, structural = false;

  const uint64_t fp_now = oracle_fingerprint();
  bool           fp_seen = false, fp_ok = true;

  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == '#') {
      unsigned long long fp = 0;
      if (std::sscanf(line, "# oracle-fingerprint: %llx", &fp) == 1) {
        fp_seen = true;
        fp_ok   = (uint64_t(fp) == fp_now);
        if (!fp_ok)
          std::fprintf(stderr,
            "WARNING: oracle fingerprint mismatch — baseline %016llx, this run %016llx.\n"
            "  libquadmath is not the one the baseline was recorded against, so the\n"
            "  REFERENCE moved, not necessarily the library. Expect a scatter of 0.01\n"
            "  differences that are not regressions. Load the toolchain of record\n"
            "  (module use /soft/modulefiles && module load gcc/13.3.0) and re-run.\n",
            fp, (unsigned long long)fp_now);
      }
      continue;
    }
    if (!saw_header && std::strncmp(line, "backend,", 8) == 0) { saw_header = true; continue; }

    char be[16], op[32]; char kind = 0; int point = 0;
    double dig = 0.0, base_ulps = 0.0, base_bound = 0.0; char st = 0;
    if (std::sscanf(line, "%15[^,],%c,%31[^,],%d,%lf,%lf,%lf,%c", be, &kind, op,
                    &point, &dig, &base_ulps, &base_bound, &st) != 8) {
      std::fprintf(stderr, "malformed baseline line %zu: %s", parsed + 1, line);
      structural = true; break;
    }
    ++parsed;

    if (idx >= fresh.size()) {
      std::fprintf(stderr, "baseline has more rows than the fresh sweep "
                           "(%zu vs %zu) — the grid changed\n", parsed, fresh.size());
      structural = true; break;
    }
    const Row& r = fresh[idx];
    if (std::strcmp(be, r.backend) != 0 || kind != r.kind ||
        std::strcmp(op, r.op) != 0 || point != r.point) {
      std::fprintf(stderr,
                   "baseline row %zu is %s,%c,%s,%d but the fresh sweep has %s,%c,%s,%d\n"
                   "  the grid or the op inventory changed; the comparison is invalid\n",
                   idx, be, kind, op, point, r.backend, r.kind, r.op, r.point);
      structural = true; break;
    }

    // THE MONOTONE GATE, in ulps. Larger is worse, so a regression is a point
    // whose error GREW. Two things are deliberately not regressions:
    //
    //  * a move of less than kNoiseFactor. The oracle is libquadmath, whose own
    //    last place moves under a compiler or glibc change, and re-running the
    //    identical binary moves roughly twenty of the 428,592 rows in the last
    //    place of the score. kNoiseFactor = 1.26 is 10^0.1, i.e. exactly the
    //    "sub-0.1-digit" floor that scatter lives in, expressed multiplicatively
    //    because ulps are a ratio. Choosing a tolerance over run-twice-and-
    //    intersect is deliberate: intersecting doubles a four-second gate's
    //    cost to suppress noise that a stated threshold suppresses for free,
    //    and a stated threshold is auditable -- you can read 1.26 in this file.
    //
    //  * anything at or below kSubUlpFloor. Below one ulp the score is a
    //    rounding coin-flip, and 0.2 -> 0.4 ulps is not a defect.
    //
    // A point that changes STATE is always reported: scored -> unresolved means
    // the bound stopped being issuable, which is a real change even though no
    // ulp count got worse.
    const double kNoiseFactor = 1.2589254;   // 10^0.1: one tenth of a digit
    const double kSubUlpFloor = 1.0;
    const double fresh_ulps = r.ulps;
    const bool both_scored = (st == 'S' && r.state == 'S');
    if (both_scored && fresh_ulps > kSubUlpFloor &&
        fresh_ulps > base_ulps * kNoiseFactor) {
      ++decreased;
      const double drop = (base_ulps > 0.0)
                        ? std::log10(fresh_ulps / base_ulps) : 0.0;
      if (drop > worst_drop) worst_drop = drop;
      if (worst_shown < 40) {
        std::printf("  REGRESSION  %s %c %-9s point %-5d  %.6g -> %.6g ulps "
                    "(-%.2f digits)\n", be, kind, op, point,
                    base_ulps, fresh_ulps, drop);
        ++worst_shown;
      }
    } else if (st != r.state) {
      ++state_moved;
      if (state_shown < 20) {
        std::printf("  STATE       %s %c %-9s point %-5d  %c -> %c\n",
                    be, kind, op, point, st, r.state);
        ++state_shown;
      }
      if (st == 'S' && r.state != 'S') ++decreased;   // lost a verdict
    } else if (both_scored && fresh_ulps * kNoiseFactor < base_ulps) {
      ++increased;
    }
    (void)dig; (void)base_bound;
    ++idx;
  }
  if (gz) pclose(f); else std::fclose(f);

  if (structural) return 2;
  if (idx != fresh.size()) {
    std::fprintf(stderr, "baseline has %zu rows, the fresh sweep has %zu — "
                         "the grid changed\n", idx, fresh.size());
    return 2;
  }

  std::printf("\n  compared      : %zu points against %s\n", idx, path.c_str());
  std::printf("  decreased     : %ld%s\n", decreased,
              decreased ? "   <-- MONOTONE GATE FAILED" : "");
  std::printf("  increased     : %ld\n", increased);
  std::printf("  unchanged     : %zu\n", idx - size_t(decreased) - size_t(increased));
  std::printf("  state moved   : %ld\n", state_moved);
  if (decreased) {
    std::printf("  worst drop    : %.2f digits equivalent\n", worst_drop);
    if (decreased > worst_shown)
      std::printf("  (%ld further regressions not listed)\n", decreased - worst_shown);
  }
  if (!fp_seen)
    std::printf("  note          : baseline predates the oracle fingerprint; "
                "a toolchain mismatch cannot be detected\n");
  else if (!fp_ok)
    std::printf("  note          : ORACLE FINGERPRINT MISMATCH — see the warning above; "
                "hairline diffs are the reference moving\n");
  std::printf("\nRESULT: %s\n", decreased
    ? "FAIL — a point got worse by more than a tenth of a digit"
    : "PASS — no point above the 0.1-digit noise floor got worse");
  return decreased ? 1 : 0;
}


// ---------------------------------------------------------------------------
// THE OPEN-DEFECT REGISTER
// ---------------------------------------------------------------------------
// The absolute gate says "no point above its derived bound". Some points ARE
// above it -- those are the open defects, and the honest thing is to keep the
// bound where the derivation puts it and carry the exceptions in a file you can
// read, rather than to loosen the bound until they disappear. That is the whole
// difference between this and the per-op tolerance tables it replaces: a
// tolerance hides the count, a register prints it.
//
// The register is checked in BOTH directions. A point above bound that is not
// listed fails the gate (a new defect). A listed point that is no longer above
// bound also fails (the register has rotted and must be shrunk). So it can only
// get smaller, and it cannot silently absorb a regression.
//
// Format: one "BACKEND kind op point" per line; '#' comments.
bool region_is_gated(const char* r);


int apply_register(const std::string& path, const std::vector<UlpFail>& fails) {
  std::vector<std::string> listed, seen, unlisted, stale;
  char key[96];
  for (size_t i = 0; i < fails.size(); ++i) {
    if (!region_is_gated(fails[i].region)) continue;
    std::snprintf(key, sizeof(key), "%s %c %s %d", fails[i].backend,
                  fails[i].kind, fails[i].op, fails[i].point);
    seen.push_back(key);
  }
  std::sort(seen.begin(), seen.end());
  if (path.empty()) {
    std::printf("\nRESULT: %s — absolute gate (no register given)\n",
                seen.empty() ? "PASS" : "FAIL");
    return seen.empty() ? 0 : 4;
  }
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "cannot open register %s\n", path.c_str()); return 2; }
  char line[160];
  while (std::fgets(line, sizeof(line), f)) {
    char* h = std::strchr(line, '#'); if (h) *h = 0;
    char be[16], op[32]; char kind = 0; int pt = 0;
    if (std::sscanf(line, "%15s %c %31s %d", be, &kind, op, &pt) == 4) {
      std::snprintf(key, sizeof(key), "%s %c %s %d", be, kind, op, pt);
      listed.push_back(key);
    }
  }
  std::fclose(f);
  std::sort(listed.begin(), listed.end());
  std::set_difference(seen.begin(), seen.end(), listed.begin(), listed.end(),
                      std::back_inserter(unlisted));
  std::set_difference(listed.begin(), listed.end(), seen.begin(), seen.end(),
                      std::back_inserter(stale));
  std::printf("\n  register      : %s\n", path.c_str());
  std::printf("  above bound   : %zu point(s)\n", seen.size());
  std::printf("  registered    : %zu\n", listed.size());
  std::printf("  NEW (unlisted): %zu%s\n", unlisted.size(),
              unlisted.empty() ? "" : "   <-- ABSOLUTE GATE FAILED");
  for (size_t i = 0; i < unlisted.size() && i < 40; ++i)
    std::printf("      + %s\n", unlisted[i].c_str());
  std::printf("  STALE (fixed) : %zu%s\n", stale.size(),
              stale.empty() ? "" : "   <-- shrink the register");
  for (size_t i = 0; i < stale.size() && i < 40; ++i)
    std::printf("      - %s\n", stale[i].c_str());
  const bool ok = unlisted.empty() && stale.empty();
  std::printf("\nRESULT: %s — absolute gate (every point at or below its "
              "derived bound, except the %zu registered)\n",
              ok ? "PASS" : "FAIL", listed.size());
  return ok ? 0 : 4;
}

// ---------------------------------------------------------------------------
// --ulp report. Worst point per (backend, op), then the failures grouped by
// (backend, op, region). Returns the gated-failure count.
// ---------------------------------------------------------------------------
// Every row that reaches the failure list is ABOVE_BOUND: the unresolved
// points never get there. Kept as a named predicate so the report reads the
// same, but there is now exactly one gated region and no exemptions.
bool region_is_gated(const char* r) { return std::strcmp(r, kRegionUnresolved) != 0; }

long print_ulp_report(const std::vector<UlpCell>& cells,
                      const std::vector<UlpFail>& fails,
                      const std::vector<UlpFail>& explains,
                      double allowance) {
  if (!explains.empty()) {
    std::printf("\n--- --ulp-explain: %s point %d, both metrics -------------------\n",
                explains[0].op, explains[0].point);
    std::printf("  x = %.17g\n", explains[0].a0);
    std::printf("  %-4s %8s  %12s  %10s  %10s  %12s  %12s  %9s  %s\n",
                "be", "digits", "kappa", "out_floor", "in_delta", "measured u",
                "expected u", "ratio", "verdict");
    for (size_t i = 0; i < explains.size(); ++i) {
      const UlpFail& e = explains[i];
      std::printf("  %-4s %8.2f  %12.4g  %10.4g  %10.4g  %12.4g  %12.4g  %9.4g  %s\n",
                  e.backend, e.digits, e.kappa, e.out_floor, e.in_delta, e.ulps,
                  e.expected, e.ratio, e.ratio <= allowance ? "PASS" : "FAIL");
    }
  }

  std::printf("\n=== ULP GATE  (measured <= %.1f x (out_floor + kappa*in_delta), worst point per region) ===\n",
              allowance);
  std::printf("%-4s %-2s %-11s %9s %9s %10s %12s %12s %9s\n",
              "be", "k", "op", "scored", "ungated", "unscorable", "worst ulps",
              "expected", "ratio");
  long gated_fail_cells = 0;
  for (size_t i = 0; i < cells.size(); ++i) {
    const UlpCell& u = cells[i];
    if (u.n_scored == 0) {
      std::printf("%-4s %-2c %-11s %9ld %9ld %10ld %12.4g %12s %9s   (digit gate)\n",
                  u.backend, u.kind, u.op, u.n_scored, u.n_ungated, u.n_unscorable,
                  u.max_ulps, "-", "-");
      continue;
    }
    std::printf("%-4s %-2c %-11s %9ld %9ld %10ld %12.4g %12.4g %9.3g%s\n",
                u.backend, u.kind, u.op, u.n_scored, u.n_ungated, u.n_unscorable,
                u.worst_ulps, u.worst_expected, u.worst_ratio,
                u.worst_ratio > allowance ? "  <-- over" : "");
  }

  // Failures grouped by (backend, op, region); worst point in each region wins.
  std::printf("\n--- failures by region (worst point in each) ---------------------\n");
  std::vector<size_t> seen;
  long gated = 0, exempt = 0;
  for (size_t i = 0; i < fails.size(); ++i) {
    bool dup = false;
    for (size_t j = 0; j < seen.size(); ++j) {
      const UlpFail& s = fails[seen[j]];
      if (s.backend == fails[i].backend && s.kind == fails[i].kind &&
          std::strcmp(s.op, fails[i].op) == 0 &&
          std::strcmp(s.region, fails[i].region) == 0) {
        dup = true;
        if (fails[i].ratio > s.ratio) seen[j] = i;
        break;
      }
    }
    if (!dup) seen.push_back(i);
    if (region_is_gated(fails[i].region)) ++gated; else ++exempt;
  }
  if (seen.empty()) std::printf("  none\n");
  std::printf("  %-4s %-2s %-11s %-13s %7s %12s %12s %12s %9s %s\n",
              "be", "k", "op", "region", "point", "digits", "measured u",
              "expected u", "ratio", "verdict");
  for (size_t j = 0; j < seen.size(); ++j) {
    const UlpFail& f = fails[seen[j]];
    const bool g = region_is_gated(f.region);
    if (g) ++gated_fail_cells;
    std::printf("  %-4s %-2c %-11s %-13s %7d %12.2f %12.4g %12.4g %9.4g %s\n",
                f.backend, f.kind, f.op, f.region, f.point, f.digits, f.ulps,
                f.expected, f.ratio, g ? "GATED-FAIL" : "exempt (metric n/a)");
  }
  std::printf("\n  above bound   : %ld point(s) in %ld (backend,op) cells\n",
              gated, gated_fail_cells);
  std::printf("  unresolved    : %ld point(s) — the format cannot carry the "
              "question, the answer, or the work\n", exempt);
  // No verdict here: the verdict is apply_register's, which knows which of
  // these are already-open defects. Printing a second RESULT line would be a
  // second opinion, and one measurement means one verdict.
  return gated_fail_cells;
}

// ---------------------------------------------------------------------------
void usage(const char* argv0) {
  std::fprintf(stderr,
    "Usage: %s --ulp --register PATH          the ABSOLUTE gate\n"
    "       %s --baseline PATH                the MONOTONE gate\n"
    "       %s [--out PATH] [--grid-out PATH] [--seed N] [--summary]\n"
    "\n"
    "  --out PATH        write the baseline CSV here. No default: without this\n"
    "                    flag nothing is written.\n"
    "  --grid-out PATH   also write the grid manifest here\n"
    "  --seed N          RNG seed for the derived operands (default %llu)\n"
    "  --summary         print the per-(backend, op) table (implied unless --quiet)\n"
    "  --quiet           suppress the per-(backend, op) table\n"
    "  --baseline PATH   MONOTONE GATE: re-run the sweep, diff against PATH (.gz\n"
    "                    accepted) and exit 1 if any point's error GREW by more\n"
    "                    than a tenth of a digit. Writes nothing.\n"
    "  --ulp             score every point in ulps against the derived bound and\n"
    "                    report the worst point per (backend, op). Never a mean.\n"
    "  --register PATH   ABSOLUTE GATE: PATH lists the known above-bound points.\n"
    "                    Checked both ways -- an unlisted point above bound and a\n"
    "                    listed point no longer above it both exit 4.\n"
    "  --ulp-dump PATH   with --ulp, write EVERY above-bound point to PATH as CSV\n"
    "  --ulp-allowance A slack multiplier on the bound (default %.1f)\n"
    "  --ulp-explain O:P print the full both-metric breakdown for real op O at grid\n"
    "                    point P on all four backends (implies --ulp)\n",
    argv0, argv0, argv0, (unsigned long long)kDefaultSeed, kUlpAllowance);
}

}  // namespace

int main(int argc, char** argv) {
  // No default output path. Defaulting to the committed baseline meant that
  // running the tool to LOOK at something silently rewrote the thing every
  // monotone comparison is measured against -- and under ctest, whose working
  // directory is the build tree, the same default made a passing gate exit 1
  // because the relative path did not resolve. Writing the baseline is now an
  // explicit --out.
  std::string out, grid_out, baseline, explain_op, ulp_dump;
  std::string ulp_register;
  uint64_t    seed = kDefaultSeed;
  bool        quiet = false, out_set = false, ulp_gate = false;
  double      ulp_allowance = kUlpAllowance;
  int         explain_point = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto need = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "Missing value after %s\n", what); std::exit(2); }
      return argv[++i];
    };
    if (s == "--help" || s == "-h") { usage(argv[0]); return 0; }
    else if (s == "--out")       { out = need("--out"); out_set = true; }
    else if (s == "--grid-out")  { grid_out = need("--grid-out"); }
    else if (s == "--baseline")  { baseline = need("--baseline"); }
    else if (s == "--ulp")       { ulp_gate = true; }
    else if (s == "--ulp-allowance") { ulp_allowance = std::atof(need("--ulp-allowance")); }
    else if (s == "--ulp-dump") { ulp_dump = need("--ulp-dump"); }
    else if (s == "--register") { ulp_register = need("--register"); ulp_gate = true; }
    else if (s == "--ulp-explain") {
      const std::string v = need("--ulp-explain");
      const size_t colon = v.rfind(':');
      if (colon == std::string::npos) {
        std::fprintf(stderr, "--ulp-explain wants OP:POINT\n"); return 2;
      }
      explain_op = v.substr(0, colon);
      explain_point = std::atoi(v.c_str() + colon + 1);
      ulp_gate = true;
    }
    else if (s == "--seed")      { seed = std::strtoull(need("--seed"), nullptr, 10); }
    else if (s == "--summary")   { quiet = false; }
    else if (s == "--quiet")     { quiet = true; }
    else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); usage(argv[0]); return 2; }
  }
  (void)out_set;

  const std::vector<GridPoint> rgrid = build_real_grid();
  const std::vector<GridPoint> cgrid = build_complex_grid();

  std::printf("sweep_accuracy %s\n", kFormatVersion);
  std::printf("  grid: %zu real points x %d ops, %zu complex points x %d ops, 4 backends\n",
              rgrid.size(), int(R_COUNT), cgrid.size(), int(C_COUNT));
  std::printf("  rows: %zu   seed: %llu\n",
              4 * (size_t(R_COUNT) * rgrid.size() + size_t(C_COUNT) * cgrid.size()),
              (unsigned long long)seed);
  std::fflush(stdout);

  std::vector<Row>      rows;
  std::vector<Cell>     cells;
  std::vector<UlpCell>  ulp_cells;
  std::vector<UlpFail>  ulp_fails, ulp_explains;
  UlpCtx                ux = {ulp_allowance, &ulp_cells, &ulp_fails,
                              explain_op.c_str(), explain_point,
                              explain_point >= 0 ? &ulp_explains : nullptr};
  // The ulp verdict is now THE measurement, not a mode: every run computes it,
  // so the baseline it writes always carries it.
  run_sweep(seed, rgrid, cgrid, rows, cells, &ux);

  if (!quiet) print_summary(cells);

  if (ulp_gate) {
    const long bad = print_ulp_report(ulp_cells, ulp_fails, ulp_explains, ulp_allowance);
    // --ulp-dump: EVERY failing point, not the per-region worst the report
    // prints. Triage of a 100+-cell population cannot be done off the summary;
    // this is the raw material the step-1c classification was tabulated from.
    if (!ulp_dump.empty()) {
      FILE* f = std::fopen(ulp_dump.c_str(), "w");
      if (!f) { std::fprintf(stderr, "cannot write %s\n", ulp_dump.c_str()); return 1; }
      std::fprintf(f, "backend,kind,op,point,region,gated,x,digits,kappa,"
                      "out_floor,in_delta,ulps,expected,ratio\n");
      for (size_t i = 0; i < ulp_fails.size(); ++i) {
        const UlpFail& e = ulp_fails[i];
        std::fprintf(f, "%s,%c,%s,%d,%s,%d,%.17g,%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                     e.backend, e.kind, e.op, e.point, e.region,
                     region_is_gated(e.region) ? 1 : 0, e.a0, e.digits, e.kappa,
                     e.out_floor, e.in_delta, e.ulps, e.expected, e.ratio);
      }
      std::fclose(f);
      std::printf("wrote %s  (%zu rows)\n", ulp_dump.c_str(), ulp_fails.size());
    }
    const int rc = apply_register(ulp_register, ulp_fails);
    if (!out.empty()) {
      if (!write_baseline(out, rows, rgrid.size(), cgrid.size(), seed)) return 1;
      std::printf("wrote %s  (%zu rows)\n", out.c_str(), rows.size());
    }
    (void)bad;
    return rc;
  }

  if (!baseline.empty()) return compare_baseline(baseline, rows);


  if (!write_baseline(out, rows, rgrid.size(), cgrid.size(), seed)) return 1;
  std::printf("\nwrote %s  (%zu rows)\n", out.c_str(), rows.size());
  if (!grid_out.empty()) {
    if (!write_grid(grid_out, rgrid, cgrid)) return 1;
    std::printf("wrote %s  (%zu rows)\n", grid_out.c_str(), rgrid.size() + cgrid.size());
  }
  return 0;
}
