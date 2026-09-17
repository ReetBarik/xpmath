// ============================================================================
// scripts/sweep_inputs.hpp — the sweep's INPUTS, derived identically to the
//                            host scorer's, plus the committed grid's reader
// ============================================================================
// Added by CORE_PLAN section C6 (chunk A).
//
// WHAT THIS IS
// Everything that decides WHICH numbers an operation is evaluated at: the
// deterministic RNG, the per-op second/third operand streams, the per-op domain
// repair, and a parser for validation/sweep/sweep_grid.csv.
//
// WHY IT MATTERS MORE THAN IT LOOKS
// C6 is one scorer with two producers. The scorer computes the MPFR reference
// from its own copy of these inputs and joins it to the device's result on
// (backend, kind, op, point). If the two derivations disagree by one RNG draw,
// nothing fails loudly: the reference simply answers a different question than
// the device was asked, and the whole device baseline is quietly wrong. So the
// inputs are as much a shared contract as the ops are, and they live here for
// the same reason.
//
// THIS IS HOST CODE. It uses <random>, <vector> and <string> and runs on the
// CPU before anything is copied to a device. It must stay out of every functor.
// It carries no __float128 and no oracle, so a device translation unit may
// include it without tripping scripts/check_device_tu_purity.sh — but nothing
// it declares may be called from a kernel.
//
// THE GRID IS READ, NEVER REGENERATED. validation/sweep/sweep_grid.csv is the
// committed manifest the baseline's `point` column indexes. A second generator
// is a second grid: it would drift the instant either copy of build_real_grid()
// was touched, and the drift would present as an accuracy finding. So this
// header parses the file and validates it, and there is deliberately no code
// here that could produce a grid point.
//
// PROVENANCE, AND THE DUPLICATE THAT IS STILL OPEN
// splitmix64, fnv1a, Rng, stream_seed, repair_real, fill_real_operands and
// fill_complex_operands are transcribed VERBATIM from
// scripts/sweep_accuracy.cpp as of `1056341`; only the namespace changed.
// That file still carries its own copies because C6 chunk A may not modify it.
// See the same paragraph in scripts/sweep_ops.hpp for who closes that and how
// the current agreement was measured.
// ============================================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "sweep_ops.hpp"

namespace xpsweep {

// The same default as scripts/sweep_accuracy.cpp, gen_corpus and the demos.
// A device run with a different seed produces results the committed baseline
// cannot be joined to, so this is not a knob to turn casually.
inline const uint64_t kDefaultSeed = 12345ull;

// ---------------------------------------------------------------------------
// Deterministic sampling primitives.
// ---------------------------------------------------------------------------
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

inline uint64_t fnv1a(const char* s) {
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

inline uint64_t stream_seed(uint64_t base, const char* name, unsigned kind) {
  return splitmix64(base ^ fnv1a(name) ^ (uint64_t(kind) * 0x9E3779B97F4A7C15ull));
}

// ---------------------------------------------------------------------------
// The grid, as read off the committed manifest.
// ---------------------------------------------------------------------------
// `family` is carried because the manifest has it and a diagnostic that can say
// "hardred point 1693" is worth more than one that says "point 1693". Nothing
// numeric depends on it.
struct GridPoint {
  double      re, im;      // im is 0.0 for the real grid
  std::string family;
};

struct Grid {
  std::vector<GridPoint> real;
  std::vector<GridPoint> complx;
};

// Reads `kind,point,family,re,im`, skipping `#` comments and the header line.
//
// IT CANNOT SUCCEED ON SILENCE. An unreadable file, an empty file, a file with
// no `r` rows or no `c` rows, and a file whose `point` column is not 0..n-1 in
// order are all hard failures with a named reason — the manifest is the one
// input whose corruption would otherwise show up as an accuracy result.
inline bool read_grid(const std::string& path, Grid& out, std::string& err) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open grid manifest: " + path; return false; }

  char line[1024];
  long lineno = 0;
  while (std::fgets(line, sizeof(line), f)) {
    ++lineno;
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
    if (std::strncmp(line, "kind,", 5) == 0) continue;   // the column header

    // kind,point,family,re,im  — five fields, the last possibly empty.
    char* p = line;
    char* field[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    int   nf = 0;
    field[nf++] = p;
    for (; *p && nf < 5; ++p) {
      if (*p == ',') { *p = '\0'; field[nf++] = p + 1; }
    }
    if (nf < 5) {
      char msg[128];
      std::snprintf(msg, sizeof msg, "grid line %ld: expected 5 comma-separated fields, got %d",
                    lineno, nf);
      err = msg; std::fclose(f); return false;
    }
    // Strip the trailing newline from the last field.
    for (char* q = field[4]; *q; ++q) if (*q == '\n' || *q == '\r') { *q = '\0'; break; }

    GridPoint gp;
    gp.re     = std::strtod(field[3], nullptr);
    gp.im     = (field[4][0] == '\0') ? 0.0 : std::strtod(field[4], nullptr);
    gp.family = field[2];
    const long idx = std::strtol(field[1], nullptr, 10);

    std::vector<GridPoint>* dst = nullptr;
    if (std::strcmp(field[0], "r") == 0)      dst = &out.real;
    else if (std::strcmp(field[0], "c") == 0) dst = &out.complx;
    else {
      char msg[128];
      std::snprintf(msg, sizeof msg, "grid line %ld: kind must be r or c, got '%s'",
                    lineno, field[0]);
      err = msg; std::fclose(f); return false;
    }
    if (idx != (long)dst->size()) {
      char msg[160];
      std::snprintf(msg, sizeof msg,
                    "grid line %ld: kind %s point column is %ld but %zu rows of that kind "
                    "have been read — the manifest must be dense and in order, because "
                    "`point` is a positional index",
                    lineno, field[0], idx, dst->size());
      err = msg; std::fclose(f); return false;
    }
    dst->push_back(gp);
  }
  std::fclose(f);

  if (out.real.empty() || out.complx.empty()) {
    char msg[160];
    std::snprintf(msg, sizeof msg,
                  "grid manifest %s yielded %zu real and %zu complex points; both must be "
                  "nonzero", path.c_str(), out.real.size(), out.complx.size());
    err = msg; return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Per-op domain repair, applied to the shared grid value so that a point is
// evaluated somewhere the op is defined.
//
// Note the asymmetry with the complex side: NO repair is applied to the complex
// grid. The complex grid exists precisely to sit on branch cuts and poles, and
// repairing it away would delete the thing being measured.
// ---------------------------------------------------------------------------
inline bool is_inf(double x) { return std::isinf(x); }
inline double tame_inf(double x) { return std::isinf(x) ? std::copysign(1.0, x) : x; }

inline void repair_real(int id, double& a, double& b, double& c) {
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

// Second/third operands for the binary and ternary real ops. Roughly one point
// in seven is a deliberately cancelling pair, the rest are log-uniform over the
// op's window.
inline void fill_real_operands(int id, size_t i, double a, Rng& rng, double& b, double& c) {
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
// the FP32 range every backend must be able to hold.
inline void fill_complex_operands(int id, size_t i, const std::vector<GridPoint>& grid,
                                  double are, double aim, Rng& rng,
                                  double& bre, double& bim) {
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

}  // namespace xpsweep
