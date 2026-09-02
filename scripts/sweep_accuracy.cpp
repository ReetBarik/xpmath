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
#include <random>
#include <string>
#include <vector>

namespace {

const char* const kFormatVersion = "xp-sweep-1";
const char* const kDefaultOut    = "validation/sweep/sweep_baseline.csv";
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

// ---------------------------------------------------------------------------
// Backend bindings. Every math entry point is spelled the same way in all four
// headers, so one template body serves all of them.
// ---------------------------------------------------------------------------
struct BackendDD { using S = xp::DoubleDouble; using Z = xp::DoubleDoubleComplex;
                   static const char* name() { return "DD"; } static double cap() { return 31.0; } };
struct BackendFF { using S = xp::FloatFloat;   using Z = xp::FloatFloatComplex;
                   static const char* name() { return "FF"; } static double cap() { return 14.0; } };
struct BackendQF { using S = xp::QuadFloat;    using Z = xp::QuadFloatComplex;
                   static const char* name() { return "QF"; } static double cap() { return 29.0; } };
struct BackendTF { using S = xp::TripleFloat;  using Z = xp::TripleFloatComplex;
                   static const char* name() { return "TF"; } static double cap() { return 21.7; } };

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

// ---------------------------------------------------------------------------
// A scored row, in emission order.
// ---------------------------------------------------------------------------
struct Row {
  const char* backend;
  char        kind;        // 'r' or 'c'
  const char* op;
  int         point;
  double      digits;
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

template <class B>
void sweep_real(int id, const std::vector<GridPoint>& grid,
                const std::vector<double>& a_in, const std::vector<double>& b_in,
                const std::vector<double>& c_in, const std::vector<__float128>& ref,
                std::vector<Row>& rows, std::vector<Cell>& cells) {
  typedef typename B::S S;
  Cell cell = {B::name(), 'r', kReal[id].name, B::cap(), 0.0, B::cap(), 0, 0};
  for (size_t i = 0; i < grid.size(); ++i) {
    const S r = eval_real<S>(id, S(a_in[i]), S(b_in[i]), S(c_in[i]));
    const double d = score_scalar(to_q(r), ref[i], B::cap());
    rows.push_back({B::name(), 'r', kReal[id].name, int(i), d});
    cell.sum += d;
    if (d < cell.min) cell.min = d;
    if (d == 0.0) ++cell.n_zero;
    ++cell.n;
  }
  cells.push_back(cell);
}

template <class B>
void sweep_complex(int id, const std::vector<GridPoint>& grid,
                   const std::vector<double>& b_re, const std::vector<double>& b_im,
                   const std::vector<__float128>& ref_re, const std::vector<__float128>& ref_im,
                   std::vector<Row>& rows, std::vector<Cell>& cells) {
  typedef typename B::S S;
  typedef typename B::Z Z;
  Cell cell = {B::name(), 'c', kComplex[id].name, B::cap(), 0.0, B::cap(), 0, 0};
  for (size_t i = 0; i < grid.size(); ++i) {
    const Z a{S(grid[i].re), S(grid[i].im)};
    const Z b{S(b_re[i]), S(b_im[i])};
    bool     is_real = false;
    const Z  r  = eval_complex<S, Z>(id, a, b, is_real);
    const double dre = score_component(to_q(r.re), ref_re[i], ref_im[i], B::cap());
    const double dim = is_real ? B::cap()
                               : score_component(to_q(r.im), ref_im[i], ref_re[i], B::cap());
    const double d = dre < dim ? dre : dim;
    rows.push_back({B::name(), 'c', kComplex[id].name, int(i), d});
    cell.sum += d;
    if (d < cell.min) cell.min = d;
    if (d == 0.0) ++cell.n_zero;
    ++cell.n;
  }
  cells.push_back(cell);
}

// ---------------------------------------------------------------------------
// The sweep proper. Inputs and oracle references are computed ONCE per op and
// shared across the four backends, so every backend is scored on identical data
// and against an identical reference — the comparability property gen_corpus
// exists to provide, applied to a grid instead of a sample.
// ---------------------------------------------------------------------------
void run_sweep(uint64_t seed, const std::vector<GridPoint>& rgrid,
               const std::vector<GridPoint>& cgrid,
               std::vector<Row>& rows, std::vector<Cell>& cells) {
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
    sweep_real<BackendDD>(id, rgrid, a, b, c, ref, rows, cells);
    sweep_real<BackendFF>(id, rgrid, a, b, c, ref, rows, cells);
    sweep_real<BackendQF>(id, rgrid, a, b, c, ref, rows, cells);
    sweep_real<BackendTF>(id, rgrid, a, b, c, ref, rows, cells);
  }

  for (int id = 0; id < C_COUNT; ++id) {
    Rng rng(stream_seed(seed, kComplex[id].name, 1u));
    std::vector<double>     bre(nc), bim(nc);
    std::vector<__float128> rre(nc), rim(nc);
    for (size_t i = 0; i < nc; ++i) {
      fill_complex_operands(id, i, cgrid, cgrid[i].re, cgrid[i].im, rng, bre[i], bim[i]);
      reference_complex(id, cgrid[i].re, cgrid[i].im, bre[i], bim[i], rre[i], rim[i]);
    }
    sweep_complex<BackendDD>(id, cgrid, bre, bim, rre, rim, rows, cells);
    sweep_complex<BackendFF>(id, cgrid, bre, bim, rre, rim, rows, cells);
    sweep_complex<BackendQF>(id, cgrid, bre, bim, rre, rim, rows, cells);
    sweep_complex<BackendTF>(id, cgrid, bre, bim, rre, rim, rows, cells);
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
  std::fprintf(f, "# a 0.00 cell is a recorded present-day defect, not a gap; "
                  "see docs/KNOWN_ISSUES.md\n");
  std::fprintf(f, "backend,kind,op,point,digits\n");
  char buf[96];
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    const int  n = std::snprintf(buf, sizeof(buf), "%s,%c,%s,%d,%.2f\n",
                                 r.backend, r.kind, r.op, r.point, r.digits);
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
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "cannot open baseline %s\n", path.c_str()); return 2; }

  char   line[256];
  size_t idx = 0, parsed = 0;
  long   decreased = 0, increased = 0, worst_shown = 0;
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

    char be[16], op[32]; char kind = 0; int point = 0; double dig = 0.0;
    if (std::sscanf(line, "%15[^,],%c,%31[^,],%d,%lf", be, &kind, op, &point, &dig) != 5) {
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

    // Round the fresh score through the same 2-decimal text the baseline stores,
    // so an unchanged point compares exactly and a 0.01 move is still detectable.
    // The epsilon below only absorbs the decimal round-trip.
    char tmp[16];
    std::snprintf(tmp, sizeof(tmp), "%.2f", r.digits);
    const double fresh_dig = std::atof(tmp);
    if (fresh_dig < dig - 1e-9) {
      ++decreased;
      const double drop = dig - fresh_dig;
      if (drop > worst_drop) worst_drop = drop;
      if (worst_shown < 40) {
        std::printf("  REGRESSION  %s %c %-9s point %-5d  %.2f -> %.2f  (-%.2f)\n",
                    be, kind, op, point, dig, fresh_dig, drop);
        ++worst_shown;
      }
    } else if (fresh_dig > dig + 1e-9) {
      ++increased;
    }
    ++idx;
  }
  std::fclose(f);

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
  if (decreased) {
    std::printf("  worst drop    : %.2f digits\n", worst_drop);
    if (decreased > worst_shown)
      std::printf("  (%ld further regressions not listed)\n", decreased - worst_shown);
  }
  if (!fp_seen)
    std::printf("  note          : baseline predates the oracle fingerprint; "
                "a toolchain mismatch cannot be detected\n");
  else if (!fp_ok)
    std::printf("  note          : ORACLE FINGERPRINT MISMATCH — see the warning above; "
                "hairline diffs are the reference moving\n");
  std::printf("\nRESULT: %s\n", decreased ? "FAIL — accuracy decreased" : "PASS — no point decreased");
  return decreased ? 1 : 0;
}

// ---------------------------------------------------------------------------
void usage(const char* argv0) {
  std::fprintf(stderr,
    "Usage: %s [--out PATH] [--grid-out PATH] [--seed N] [--summary]\n"
    "       %s --baseline PATH\n"
    "\n"
    "  --out PATH        write the baseline CSV here (default %s)\n"
    "  --grid-out PATH   also write the grid manifest here\n"
    "  --seed N          RNG seed for the derived operands (default %llu)\n"
    "  --summary         print the per-(backend, op) table (implied unless --quiet)\n"
    "  --quiet           suppress the per-(backend, op) table\n"
    "  --baseline PATH   MONOTONE GATE: re-run the sweep, diff against PATH and\n"
    "                    exit 1 if ANY point's digits decreased. Writes nothing.\n",
    argv0, argv0, kDefaultOut, (unsigned long long)kDefaultSeed);
}

}  // namespace

int main(int argc, char** argv) {
  std::string out = kDefaultOut, grid_out, baseline;
  uint64_t    seed = kDefaultSeed;
  bool        quiet = false, out_set = false;

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

  std::vector<Row>  rows;
  std::vector<Cell> cells;
  run_sweep(seed, rgrid, cgrid, rows, cells);

  if (!quiet) print_summary(cells);

  if (!baseline.empty()) return compare_baseline(baseline, rows);

  if (!write_baseline(out, rows, rgrid.size(), cgrid.size(), seed)) return 1;
  std::printf("\nwrote %s  (%zu rows)\n", out.c_str(), rows.size());
  if (!grid_out.empty()) {
    if (!write_grid(grid_out, rgrid, cgrid)) return 1;
    std::printf("wrote %s  (%zu rows)\n", grid_out.c_str(), rgrid.size() + cgrid.size());
  }
  return 0;
}
