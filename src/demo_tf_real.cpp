// Kokkos triple-float demo — real ops.
// Benchmarks Kokkos::Experimental::TripleFloat against the host __float128
// oracle (Kokkos-wrapped quadmath, T0.0).
//
// Adapted from src/demo_qf_real.cpp (T3.0b) for the TF backend (S10 P5):
// QuadFloat -> TripleFloat, ~29-digit -> ~21.7-digit accuracy cap, and a per-op
// pass/fail verdict (RC 0 on all-pass, RC 1 on any non-conditioning-limited
// miss) as the S10 P5 deliverable.
//
// ============================================================
// TF usage reference  (namespace tf = Kokkos::Experimental)
// ============================================================
//
// Construction
//   tf::TripleFloat x;               // zero
//   tf::TripleFloat x(1.5f);         // from float
//   tf::TripleFloat x(hi, mid, lo);  // from three float components
//
// Arithmetic operators
//   x + y,  x - y,  x * y,  x / y    // TripleFloat op TripleFloat -> TripleFloat
//   x + b,  x - b,  x * b,  x / b    // TripleFloat op float  -> TripleFloat
//   a + y,  a - y,  a * y,  a / y    // float  op TripleFloat -> TripleFloat
//   -x                                // unary negation
//   x += y, x -= y, x *= y, x /= y   // TripleFloat op TripleFloat
//   x += b, x -= b, x *= b, x /= b   // TripleFloat op float
//
// Math functions  (all KOKKOS_INLINE_FUNCTION, host + device)
//   tf::abs(x)
//   tf::sqrt(x)
//   tf::exp(x),   tf::exp2(x),  tf::exp10(x), tf::expm1(x)
//   tf::log(x),   tf::log2(x),  tf::log10(x), tf::log1p(x)
//   tf::sin(x),   tf::cos(x),   tf::tan(x)
//   tf::asin(x),  tf::acos(x),  tf::atan(x),  tf::atan2(y, x)
//   tf::sinh(x),  tf::cosh(x),  tf::tanh(x)
//   tf::asinh(x), tf::acosh(x), tf::atanh(x)
//   tf::pow(x, y)                // TripleFloat exponent
//   tf::pow_int(x, n)            // int exponent
//   tf::hypot(x, y)
//   tf::ceil(x),  tf::floor(x),  tf::round(x), tf::trunc(x)
//   tf::fmod(x, y),  tf::remainder(x, y)
//   tf::fma(x, y, z)
//   tf::fmax(x, y),  tf::fmin(x, y),  tf::fdim(x, y)
//   tf::copysign(x, y)
//   tf::sincos(x, c, s)          // void; writes c=cos(x), s=sin(x)
//   tf::sinhcosh(x, ch, sh)      // void; writes ch=cosh(x), sh=sinh(x)
//
// Constants
//   tf::TripleFloat_pi()           // pi
//   tf::TripleFloat_e()            // e
//   tf::TripleFloat_log2()         // ln 2
//   tf::TripleFloat_log10()        // ln 10
//   tf::TripleFloat_sqrt2()        // sqrt(2)
//   tf::TripleFloat_euler_gamma()  // Euler-Mascheroni gamma
// ============================================================

#include <Kokkos_Core.hpp>

// Host __float128 oracle via Kokkos's quadmath overloads (namespace Kokkos).
// This header transitively includes <quadmath.h> when Kokkos was built with
// Kokkos_ENABLE_LIBQUADMATH=ON, so ::*q symbols remain available too.
#include <impl/Kokkos_QuadPrecisionMath.hpp>

#include <tf_math.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int      kWarmupRuns     = 2;
constexpr int      kDefaultRepeats = 5;
constexpr uint64_t kDefaultSeed    = 12345ULL;
constexpr double   kMaxDigits      = 21.7; // TF max ~21.68 decimal digits (24-bit FP32 mantissa x 3 = ~72 bits, u=2^-72)
// Per-op pass gate for the RC-0/RC-1 verdict (mean digits vs __float128 oracle).
// Set below TF's ~21.7 ceiling to leave room for argument-reduction / conditioning
// losses (QD's own margins). Ops whose mean is conditioning-limited below this
// are listed in kConditioningLimited and exempted from the fail verdict.
constexpr double   kPassMeanDigits  = 19.0;

// clang-format off
enum class Op {
  Add, Sub, Mul, Div,
  Sqrt, Abs, Exp, Log, Exp2, Exp10, Expm1, Log2, Log10, Log1p,
  Sin, Cos, Tan, Asin, Acos, Atan,
  Sinh, Cosh, Tanh, Acosh, Asinh, Atanh,
  Pow, Hypot, Fmod, Remainder, Copysign, Fmax, Fmin, Fdim,
  Fma,
  Ceil, Floor, Round, Trunc,
};

static const Op kAllOps[] = {
  Op::Add, Op::Sub, Op::Mul, Op::Div,
  Op::Sqrt, Op::Abs, Op::Exp, Op::Log, Op::Exp2, Op::Exp10, Op::Expm1,
  Op::Log2, Op::Log10, Op::Log1p,
  Op::Sin, Op::Cos, Op::Tan, Op::Asin, Op::Acos, Op::Atan,
  Op::Sinh, Op::Cosh, Op::Tanh, Op::Acosh, Op::Asinh, Op::Atanh,
  Op::Pow, Op::Hypot, Op::Fmod, Op::Remainder, Op::Copysign,
  Op::Fmax, Op::Fmin, Op::Fdim,
  Op::Fma,
  Op::Ceil, Op::Floor, Op::Round, Op::Trunc,
};
// clang-format on

struct Config {
  Op       op      = Op::Add;
  bool     all_ops = false;
  int      batch   = 1'000'000;
  int      repeats = kDefaultRepeats;
  uint64_t seed    = kDefaultSeed;
};

bool parse_op(const std::string& s, Op& out) {
  // clang-format off
  if (s == "add")       { out = Op::Add;       return true; }
  if (s == "sub")       { out = Op::Sub;       return true; }
  if (s == "mul")       { out = Op::Mul;       return true; }
  if (s == "div")       { out = Op::Div;       return true; }
  if (s == "sqrt")      { out = Op::Sqrt;      return true; }
  if (s == "abs")       { out = Op::Abs;       return true; }
  if (s == "exp")       { out = Op::Exp;       return true; }
  if (s == "log")       { out = Op::Log;       return true; }
  if (s == "exp2")      { out = Op::Exp2;      return true; }
  if (s == "exp10")     { out = Op::Exp10;     return true; }
  if (s == "expm1")     { out = Op::Expm1;     return true; }
  if (s == "log2")      { out = Op::Log2;      return true; }
  if (s == "log10")     { out = Op::Log10;     return true; }
  if (s == "log1p")     { out = Op::Log1p;     return true; }
  if (s == "sin")       { out = Op::Sin;       return true; }
  if (s == "cos")       { out = Op::Cos;       return true; }
  if (s == "tan")       { out = Op::Tan;       return true; }
  if (s == "asin")      { out = Op::Asin;      return true; }
  if (s == "acos")      { out = Op::Acos;      return true; }
  if (s == "atan")      { out = Op::Atan;      return true; }
  if (s == "sinh")      { out = Op::Sinh;      return true; }
  if (s == "cosh")      { out = Op::Cosh;      return true; }
  if (s == "tanh")      { out = Op::Tanh;      return true; }
  if (s == "acosh")     { out = Op::Acosh;     return true; }
  if (s == "asinh")     { out = Op::Asinh;     return true; }
  if (s == "atanh")     { out = Op::Atanh;     return true; }
  if (s == "pow")       { out = Op::Pow;       return true; }
  if (s == "hypot")     { out = Op::Hypot;     return true; }
  if (s == "fmod")      { out = Op::Fmod;      return true; }
  if (s == "remainder") { out = Op::Remainder; return true; }
  if (s == "copysign")  { out = Op::Copysign;  return true; }
  if (s == "fmax")      { out = Op::Fmax;      return true; }
  if (s == "fmin")      { out = Op::Fmin;      return true; }
  if (s == "fdim")      { out = Op::Fdim;      return true; }
  if (s == "fma")       { out = Op::Fma;       return true; }
  if (s == "ceil")      { out = Op::Ceil;      return true; }
  if (s == "floor")     { out = Op::Floor;     return true; }
  if (s == "round")     { out = Op::Round;     return true; }
  if (s == "trunc")     { out = Op::Trunc;     return true; }
  // clang-format on
  return false;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Add:       return "add";
    case Op::Sub:       return "sub";
    case Op::Mul:       return "mul";
    case Op::Div:       return "div";
    case Op::Sqrt:      return "sqrt";
    case Op::Abs:       return "abs";
    case Op::Exp:       return "exp";
    case Op::Log:       return "log";
    case Op::Exp2:      return "exp2";
    case Op::Exp10:     return "exp10";
    case Op::Expm1:     return "expm1";
    case Op::Log2:      return "log2";
    case Op::Log10:     return "log10";
    case Op::Log1p:     return "log1p";
    case Op::Sin:       return "sin";
    case Op::Cos:       return "cos";
    case Op::Tan:       return "tan";
    case Op::Asin:      return "asin";
    case Op::Acos:      return "acos";
    case Op::Atan:      return "atan";
    case Op::Sinh:      return "sinh";
    case Op::Cosh:      return "cosh";
    case Op::Tanh:      return "tanh";
    case Op::Acosh:     return "acosh";
    case Op::Asinh:     return "asinh";
    case Op::Atanh:     return "atanh";
    case Op::Pow:       return "pow";
    case Op::Hypot:     return "hypot";
    case Op::Fmod:      return "fmod";
    case Op::Remainder: return "remainder";
    case Op::Copysign:  return "copysign";
    case Op::Fmax:      return "fmax";
    case Op::Fmin:      return "fmin";
    case Op::Fdim:      return "fdim";
    case Op::Fma:       return "fma";
    case Op::Ceil:      return "ceil";
    case Op::Floor:     return "floor";
    case Op::Round:     return "round";
    case Op::Trunc:     return "trunc";
  }
  return "?";
}

// Ops whose mean accuracy is bounded by the operation's condition number (not by
// TF quality), so a below-gate mean is expected, not a failure. Grounded in
// PORT_NOTES.md §5 (the FP32 conditioning list, which applies unchanged to TF):
//   * exp: e^a near FP32's smallest normal (a ~= -88) pushes the low-order TF
//     words into FP32 denormal range, capping accuracy in the tail.
//   * asin/acos near |a|=1: derivative 1/sqrt(1-a^2) -> inf.
//   * atanh near |a|=1: 1/(1-a^2) -> inf.
//   * fmod/remainder near a multiple of b: a - b*nint(a/b) cancels; relative
//     precision bounded by eps*|a|/|result|, unbounded as result -> 0.
//   * sub/fdim/fma: random near-cancellation, one digit lost per matched digit.
bool is_conditioning_limited(Op op) {
  switch (op) {
    case Op::Exp: case Op::Asin: case Op::Acos: case Op::Atanh:
    case Op::Fmod: case Op::Remainder:
    case Op::Sub: case Op::Fdim: case Op::Fma:
      return true;
    default:
      return false;
  }
}

void print_usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [--op <name>] [--batch N] [--repeats N] [--seed N]\n"
    << "  Omit --op to run all operations and print a complete table.\n"
    << "  Operations: add sub mul div sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p\n"
    << "              sin cos tan asin acos atan sinh cosh tanh acosh asinh atanh\n"
    << "              pow fmod remainder fmax fmin fdim fma\n"
    << "              ceil floor round trunc\n"
    << "  Defaults: batch=1000000 repeats=" << kDefaultRepeats << " seed=" << kDefaultSeed << "\n"
    << "  Warmup runs (fixed): " << kWarmupRuns << "\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") return false;
    auto need = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { std::cerr << "Missing value after " << what << "\n"; return nullptr; }
      return argv[++i];
    };
    if (a == "--op") {
      const char* v = need("--op"); if (!v) return false;
      if (!parse_op(v, cfg.op)) { std::cerr << "Unknown op: " << v << "\n"; return false; }
      cfg.all_ops = false;
    } else if (a == "--batch") {
      const char* v = need("--batch"); if (!v) return false;
      cfg.batch = std::atoi(v);
      if (cfg.batch <= 0) { std::cerr << "Invalid --batch\n"; return false; }
    } else if (a == "--repeats") {
      const char* v = need("--repeats"); if (!v) return false;
      cfg.repeats = std::atoi(v);
      if (cfg.repeats <= 0) { std::cerr << "Invalid --repeats\n"; return false; }
    } else if (a == "--seed") {
      const char* v = need("--seed"); if (!v) return false;
      cfg.seed = static_cast<uint64_t>(std::strtoull(v, nullptr, 10));
    } else {
      std::cerr << "Unknown argument: " << a << "\n"; return false;
    }
  }
  return true;
}

// Inputs are sampled in double precision then narrowed to FP32 for TF and to FP64 for the
// FP64 baseline. The quadmath reference uses the doubles directly (sufficient for the TF
// accuracy budget of ~21.7 digits).
void fill_inputs(Op op, double* ha, double* hb, double* hc, int n, uint64_t seed) {
  std::mt19937_64 gen(seed);
  auto unary = [&](double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    for (int i = 0; i < n; ++i) { ha[i] = d(gen); hb[i] = 0.0; }
  };
  auto binary = [&](double lo_a, double hi_a, double lo_b, double hi_b) {
    std::uniform_real_distribution<double> da(lo_a, hi_a), db(lo_b, hi_b);
    for (int i = 0; i < n; ++i) { ha[i] = da(gen); hb[i] = db(gen); }
  };
  constexpr double pi = 3.14159265358979323846;
  switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: binary(0.1, 10.0, 0.1, 10.0); break;
    case Op::Sqrt:      unary(1e-16, 1e8);    break;
    case Op::Abs:       unary(-1e8,  1e8);    break;
    case Op::Exp:       unary(-80.0, 80.0);   break;
    case Op::Log:       unary(1e-16, 1e16);   break;
    case Op::Exp2:      unary(-100.0, 100.0); break;
    case Op::Exp10:     unary(-30.0,  30.0);  break;
    case Op::Expm1:     unary(-1.0,   1.0);   break;
    case Op::Log2: case Op::Log10: unary(1e-16, 1e16); break;
    case Op::Log1p:     unary(-0.999, 1e16);  break;
    case Op::Sin: case Op::Cos: unary(-pi, pi); break;
    case Op::Tan:       unary(-1.4,   1.4);   break;
    case Op::Asin: case Op::Acos: unary(-1.0, 1.0); break;
    case Op::Atan:      unary(-1e8,   1e8);   break;
    case Op::Sinh: case Op::Cosh: unary(-20.0, 20.0); break;
    case Op::Tanh:      unary(-5.0,   5.0);   break;
    case Op::Acosh:     unary(1.0,   1e12);   break;
    case Op::Asinh:     unary(-1e8,   1e8);   break;
    case Op::Atanh:     unary(-0.999, 0.999); break;
    case Op::Pow:       binary(0.5, 20.0, 0.1, 5.0);   break;
    case Op::Hypot:     binary(0.0, 1e8, 0.0, 1e8);    break;
    case Op::Fmod:      binary(0.1, 100.0, 0.1, 10.0); break;
    case Op::Remainder: binary(0.1, 100.0, 0.1, 10.0); break;
    case Op::Copysign:  binary(-1e8, 1e8, -1.0, 1.0);  break;
    case Op::Fmax: case Op::Fmin: case Op::Fdim: binary(-1e8, 1e8, -1e8, 1e8); break;
    case Op::Fma: {
      std::uniform_real_distribution<double> da(0.1,10.0), db(0.1,10.0), dc(-10.0,10.0);
      for (int i = 0; i < n; ++i) { ha[i]=da(gen); hb[i]=db(gen); hc[i]=dc(gen); }
      break;
    }
    case Op::Ceil: case Op::Floor: case Op::Round:
    case Op::Trunc: unary(-1e6, 1e6); break;
  }
}

void host_quadmath_reference(Op op, const double* ha, const double* hb, const double* hc,
                             __float128* out, int n) {
  for (int i = 0; i < n; ++i) {
    __float128 fa = (__float128)ha[i], fb = (__float128)hb[i], fc = (__float128)hc[i];
    switch (op) {
      case Op::Add:       out[i] = fa + fb;                      break;
      case Op::Sub:       out[i] = fa - fb;                      break;
      case Op::Mul:       out[i] = fa * fb;                      break;
      case Op::Div:       out[i] = fa / fb;                      break;
      case Op::Sqrt:      out[i] = Kokkos::sqrt(fa);                    break;
      case Op::Abs:       out[i] = Kokkos::abs(fa);                    break;
      case Op::Exp:       out[i] = Kokkos::exp(fa);                     break;
      case Op::Log:       out[i] = Kokkos::log(fa);                     break;
      case Op::Exp2:      out[i] = Kokkos::exp2(fa);                    break;
      case Op::Exp10:     out[i] = Kokkos::pow((__float128)10.0, fa);   break;
      case Op::Expm1:     out[i] = Kokkos::expm1(fa);                   break;
      case Op::Log2:      out[i] = Kokkos::log2(fa);                    break;
      case Op::Log10:     out[i] = Kokkos::log10(fa);                   break;
      case Op::Log1p:     out[i] = Kokkos::log1p(fa);                   break;
      case Op::Sin:       out[i] = Kokkos::sin(fa);                     break;
      case Op::Cos:       out[i] = Kokkos::cos(fa);                     break;
      case Op::Tan:       out[i] = Kokkos::tan(fa);                     break;
      case Op::Asin:      out[i] = Kokkos::asin(fa);                    break;
      case Op::Acos:      out[i] = Kokkos::acos(fa);                    break;
      case Op::Atan:      out[i] = Kokkos::atan(fa);                    break;
      case Op::Sinh:      out[i] = Kokkos::sinh(fa);                    break;
      case Op::Cosh:      out[i] = Kokkos::cosh(fa);                    break;
      case Op::Tanh:      out[i] = Kokkos::tanh(fa);                    break;
      case Op::Acosh:     out[i] = Kokkos::acosh(fa);                   break;
      case Op::Asinh:     out[i] = Kokkos::asinh(fa);                   break;
      case Op::Atanh:     out[i] = Kokkos::atanh(fa);                   break;
      case Op::Pow:       out[i] = Kokkos::pow(fa, fb);                 break;
      case Op::Hypot:     out[i] = Kokkos::hypot(fa, fb);               break;
      case Op::Fmod:      out[i] = Kokkos::fmod(fa, fb);                break;
      case Op::Remainder: out[i] = Kokkos::remainder(fa, fb);           break;
      case Op::Copysign:  out[i] = Kokkos::copysign(fa, fb);            break;
      case Op::Fmax:      out[i] = Kokkos::fmax(fa, fb);                break;
      case Op::Fmin:      out[i] = Kokkos::fmin(fa, fb);                break;
      case Op::Fdim:      out[i] = Kokkos::fdim(fa, fb);                break;
      case Op::Fma:       out[i] = Kokkos::fma(fa, fb, fc);             break;
      case Op::Ceil:      out[i] = Kokkos::ceil(fa);                    break;
      case Op::Floor:     out[i] = Kokkos::floor(fa);                   break;
      case Op::Round:     out[i] = Kokkos::round(fa);                   break;
      case Op::Trunc:     out[i] = Kokkos::trunc(fa);                   break;
    }
  }
}

// ---- Timing ----------------------------------------------------------------

struct TimeStats { double min_s = 0, max_s = 0, median_s = 0, mean_s = 0; };

TimeStats summarize_times(std::vector<double> t) {
  if (t.empty()) return {};
  std::sort(t.begin(), t.end());
  TimeStats s;
  s.min_s    = t.front();
  s.max_s    = t.back();
  size_t n   = t.size();
  s.median_s = (n % 2 == 1) ? t[n/2] : 0.5*(t[n/2-1]+t[n/2]);
  s.mean_s   = std::accumulate(t.begin(), t.end(), 0.0) / (double)n;
  return s;
}

using wall_clock = std::chrono::high_resolution_clock;

template <typename Launch>
TimeStats time_kernel_fence(int repeats, Launch&& launch) {
  for (int w = 0; w < kWarmupRuns; ++w) { launch(); Kokkos::fence(); }
  std::vector<double> times;
  times.reserve((size_t)repeats);
  for (int r = 0; r < repeats; ++r) {
    auto t0 = wall_clock::now();
    launch(); Kokkos::fence();
    times.push_back(std::chrono::duration<double>(wall_clock::now()-t0).count());
  }
  return summarize_times(std::move(times));
}

// ---- Accuracy --------------------------------------------------------------

struct AccStats { double min_d = kMaxDigits, max_d = 0, mean_d = 0, median_d = 0; };

static double element_digits(__float128 dev, __float128 ref) {
  if (Kokkos::isnan(dev) || Kokkos::isnan(ref)) return 0.0;
  if (Kokkos::isinf(ref)) return (Kokkos::isinf(dev) && (dev>0)==(ref>0)) ? kMaxDigits : 0.0;
  if (ref == (__float128)0.0) return (dev == (__float128)0.0) ? kMaxDigits : 0.0;
  __float128 rel = Kokkos::abs((dev - ref) / ref);
  if (rel == (__float128)0.0) return kMaxDigits;
  double d = -(double)Kokkos::log10(rel);
  return d < 0.0 ? 0.0 : (d > kMaxDigits ? kMaxDigits : d);
}

// Convert TripleFloat to __float128 for accuracy comparison
static __float128 tf_to_q(Kokkos::Experimental::TripleFloat x) {
  return (__float128)x.f0 + (__float128)x.f1 + (__float128)x.f2;
}

AccStats compute_accuracy_tf(const __float128* ref, const Kokkos::Experimental::TripleFloat* dev, int n) {
  std::vector<double> digits((size_t)n);
  for (int i = 0; i < n; ++i)
    digits[i] = element_digits(tf_to_q(dev[i]), ref[i]);
  std::sort(digits.begin(), digits.end());
  AccStats s;
  s.min_d  = digits.front();
  s.max_d  = digits.back();
  s.mean_d = std::accumulate(digits.begin(), digits.end(), 0.0) / (double)n;
  size_t m = digits.size();
  s.median_d = (m%2==1) ? digits[m/2] : 0.5*(digits[m/2-1]+digits[m/2]);
  return s;
}

AccStats compute_accuracy_dbl(const __float128* ref, const double* dev, int n) {
  std::vector<double> digits((size_t)n);
  for (int i = 0; i < n; ++i)
    digits[i] = element_digits((__float128)dev[i], ref[i]);
  std::sort(digits.begin(), digits.end());
  AccStats s;
  s.min_d  = digits.front();
  s.max_d  = digits.back();
  s.mean_d = std::accumulate(digits.begin(), digits.end(), 0.0) / (double)n;
  size_t m = digits.size();
  s.median_d = (m%2==1) ? digits[m/2] : 0.5*(digits[m/2-1]+digits[m/2]);
  return s;
}

// ---- Per-op runner ---------------------------------------------------------

struct OpResult { Op op; TimeStats tf_timing, dbl_timing; AccStats tf_acc, dbl_acc; };

using exec_space = Kokkos::DefaultExecutionSpace;
using policy_1d  = Kokkos::RangePolicy<exec_space>;
using vtf        = Kokkos::View<Kokkos::Experimental::TripleFloat*, Kokkos::LayoutRight, exec_space>;
using vdbl       = Kokkos::View<double*,             Kokkos::LayoutRight, exec_space>;

namespace tf = Kokkos::Experimental;

OpResult run_op(Op op, const Config& cfg) {
  const int n = cfg.batch;

  std::vector<double>     ha(n), hb(n), hc(n, 0.0);
  std::vector<__float128> href(n);

  fill_inputs(op, ha.data(), hb.data(), hc.data(), n, cfg.seed);
  host_quadmath_reference(op, ha.data(), hb.data(), hc.data(), href.data(), n);

  vtf  atf("atf",n), btf("btf",n), ctf("ctf",n), rtf("rtf",n);
  vdbl ad("ad",n), bd("bd",n), cd("cd",n), rd("rd",n);
  {
    auto matf=Kokkos::create_mirror_view(atf), mbtf=Kokkos::create_mirror_view(btf);
    auto mctf=Kokkos::create_mirror_view(ctf);
    auto mad=Kokkos::create_mirror_view(ad), mbd=Kokkos::create_mirror_view(bd);
    auto mcd=Kokkos::create_mirror_view(cd);
    for (int i=0; i<n; ++i) {
      // Use TripleFloat(double) so the TF input faithfully encodes the FP64 value
      // (Route-A successive split into f0..f2). A double carries only 53 bits, so
      // f0,f1 hold it exactly and f2 falls to 0 — the input is exact in TF, and
      // accuracy is bounded only by the TF math, not the input encoding.
      matf(i)=tf::TripleFloat(ha[i]); mbtf(i)=tf::TripleFloat(hb[i]);
      mctf(i)=tf::TripleFloat(hc[i]);
      mad(i)=ha[i]; mbd(i)=hb[i]; mcd(i)=hc[i];
    }
    Kokkos::deep_copy(atf,matf); Kokkos::deep_copy(btf,mbtf);
    Kokkos::deep_copy(ctf,mctf);
    Kokkos::deep_copy(ad,mad); Kokkos::deep_copy(bd,mbd); Kokkos::deep_copy(cd,mcd);
  }

  policy_1d pol(0, n);
  TimeStats st_tf, st_dbl;

  // ---- TF kernels ------------------------------------------------------------
  switch (op) {
    case Op::Add:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_add",pol,KOKKOS_LAMBDA(int i){rtf(i)=atf(i)+btf(i);});}); break;
    case Op::Sub:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_sub",pol,KOKKOS_LAMBDA(int i){rtf(i)=atf(i)-btf(i);});}); break;
    case Op::Mul:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_mul",pol,KOKKOS_LAMBDA(int i){rtf(i)=atf(i)*btf(i);});}); break;
    case Op::Div:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_div",pol,KOKKOS_LAMBDA(int i){rtf(i)=atf(i)/btf(i);});}); break;
    case Op::Sqrt:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_sqrt",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::sqrt(atf(i));});}); break;
    case Op::Abs:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_abs",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::abs(atf(i));});}); break;
    case Op::Exp:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_exp",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::exp(atf(i));});}); break;
    case Op::Log:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_log",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::log(atf(i));});}); break;
    case Op::Exp2:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_exp2",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::exp2(atf(i));});}); break;
    case Op::Exp10:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_exp10",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::exp10(atf(i));});}); break;
    case Op::Expm1:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_expm1",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::expm1(atf(i));});}); break;
    case Op::Log2:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_log2",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::log2(atf(i));});}); break;
    case Op::Log10:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_log10",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::log10(atf(i));});}); break;
    case Op::Log1p:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_log1p",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::log1p(atf(i));});}); break;
    case Op::Sin:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_sin",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::sin(atf(i));});}); break;
    case Op::Cos:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_cos",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::cos(atf(i));});}); break;
    case Op::Tan:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_tan",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::tan(atf(i));});}); break;
    case Op::Asin:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_asin",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::asin(atf(i));});}); break;
    case Op::Acos:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_acos",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::acos(atf(i));});}); break;
    case Op::Atan:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_atan",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::atan(atf(i));});}); break;
    case Op::Sinh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_sinh",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::sinh(atf(i));});}); break;
    case Op::Cosh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_cosh",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::cosh(atf(i));});}); break;
    case Op::Tanh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_tanh",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::tanh(atf(i));});}); break;
    case Op::Acosh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_acosh",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::acosh(atf(i));});}); break;
    case Op::Asinh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_asinh",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::asinh(atf(i));});}); break;
    case Op::Atanh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_atanh",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::atanh(atf(i));});}); break;
    case Op::Pow:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_pow",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::pow(atf(i),btf(i));});}); break;
    case Op::Hypot:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_hypot",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::hypot(atf(i),btf(i));});}); break;
    case Op::Fmod:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_fmod",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::fmod(atf(i),btf(i));});}); break;
    case Op::Remainder:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_rem",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::remainder(atf(i),btf(i));});}); break;
    case Op::Copysign:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_copysign",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::copysign(atf(i),btf(i));});}); break;
    case Op::Fmax:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_fmax",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::fmax(atf(i),btf(i));});}); break;
    case Op::Fmin:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_fmin",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::fmin(atf(i),btf(i));});}); break;
    case Op::Fdim:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_fdim",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::fdim(atf(i),btf(i));});}); break;
    case Op::Fma:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_fma",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::fma(atf(i),btf(i),ctf(i));});}); break;
    case Op::Ceil:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_ceil",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::ceil(atf(i));});}); break;
    case Op::Floor:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_floor",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::floor(atf(i));});}); break;
    case Op::Round:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_round",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::round(atf(i));});}); break;
    case Op::Trunc:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tf_trunc",pol,KOKKOS_LAMBDA(int i){rtf(i)=tf::trunc(atf(i));});}); break;
  }

  // ---- Double kernels --------------------------------------------------------
  switch (op) {
    case Op::Add:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_add",pol,KOKKOS_LAMBDA(int i){rd(i)=ad(i)+bd(i);});}); break;
    case Op::Sub:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_sub",pol,KOKKOS_LAMBDA(int i){rd(i)=ad(i)-bd(i);});}); break;
    case Op::Mul:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_mul",pol,KOKKOS_LAMBDA(int i){rd(i)=ad(i)*bd(i);});}); break;
    case Op::Div:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_div",pol,KOKKOS_LAMBDA(int i){rd(i)=ad(i)/bd(i);});}); break;
    case Op::Sqrt:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_sqrt",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::sqrt(ad(i));});}); break;
    case Op::Abs:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_abs",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::fabs(ad(i));});}); break;
    case Op::Exp:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_exp",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::exp(ad(i));});}); break;
    case Op::Log:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_log",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::log(ad(i));});}); break;
    case Op::Exp2:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_exp2",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::exp2(ad(i));});}); break;
    case Op::Exp10:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_exp10",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::pow(10.0,ad(i));});}); break;
    case Op::Expm1:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_expm1",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::expm1(ad(i));});}); break;
    case Op::Log2:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_log2",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::log2(ad(i));});}); break;
    case Op::Log10:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_log10",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::log10(ad(i));});}); break;
    case Op::Log1p:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_log1p",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::log1p(ad(i));});}); break;
    case Op::Sin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_sin",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::sin(ad(i));});}); break;
    case Op::Cos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_cos",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::cos(ad(i));});}); break;
    case Op::Tan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_tan",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::tan(ad(i));});}); break;
    case Op::Asin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_asin",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::asin(ad(i));});}); break;
    case Op::Acos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_acos",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::acos(ad(i));});}); break;
    case Op::Atan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_atan",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::atan(ad(i));});}); break;
    case Op::Sinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_sinh",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::sinh(ad(i));});}); break;
    case Op::Cosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_cosh",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::cosh(ad(i));});}); break;
    case Op::Tanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_tanh",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::tanh(ad(i));});}); break;
    case Op::Acosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_acosh",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::acosh(ad(i));});}); break;
    case Op::Asinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_asinh",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::asinh(ad(i));});}); break;
    case Op::Atanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_atanh",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::atanh(ad(i));});}); break;
    case Op::Pow:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_pow",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::pow(ad(i),bd(i));});}); break;
    case Op::Hypot:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_hypot",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::hypot(ad(i),bd(i));});}); break;
    case Op::Fmod:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_fmod",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::fmod(ad(i),bd(i));});}); break;
    case Op::Remainder:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_rem",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::remainder(ad(i),bd(i));});}); break;
    case Op::Copysign:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_copysign",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::copysign(ad(i),bd(i));});}); break;
    case Op::Fmax:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_fmax",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::fmax(ad(i),bd(i));});}); break;
    case Op::Fmin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_fmin",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::fmin(ad(i),bd(i));});}); break;
    case Op::Fdim:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_fdim",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::fdim(ad(i),bd(i));});}); break;
    case Op::Fma:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_fma",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::fma(ad(i),bd(i),cd(i));});}); break;
    case Op::Ceil:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_ceil",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::ceil(ad(i));});}); break;
    case Op::Floor:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_floor",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::floor(ad(i));});}); break;
    case Op::Round:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_round",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::round(ad(i));});}); break;
    case Op::Trunc:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_trunc",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::trunc(ad(i));});}); break;
  }

  auto mrtf = Kokkos::create_mirror_view(rtf);
  Kokkos::deep_copy(mrtf, rtf);
  auto mrd = Kokkos::create_mirror_view(rd);
  Kokkos::deep_copy(mrd, rd);

  AccStats tf_acc  = compute_accuracy_tf(href.data(), mrtf.data(), n);
  AccStats dbl_acc = compute_accuracy_dbl(href.data(), mrd.data(), n);
  return {op, st_tf, st_dbl, tf_acc, dbl_acc};
}

// ---- Table printing --------------------------------------------------------

static constexpr int kOpW = 10;
static constexpr int kTW  =  9;
static constexpr int kAW  =  7;
static constexpr int kTimeSec = 4*kTW + 3;
static constexpr int kAccSec  = 4*kAW + 3;
static constexpr int kBkndW   = kTimeSec + 1 + kAccSec;

static std::string dashes(int n) { return std::string((size_t)n, '-'); }

static std::string center(const std::string& s, int w) {
  int pad = w - (int)s.size();
  int lp  = pad / 2, rp = pad - lp;
  return std::string((size_t)lp,' ') + s + std::string((size_t)rp,' ');
}

static void print_sep_real() {
  std::cout << '-' << dashes(kOpW) << "-+"
            << dashes(kTimeSec) << "+" << dashes(kAccSec) << "+"
            << dashes(kTimeSec) << "+" << dashes(kAccSec) << "+\n";
}

static void print_header_real() {
  using std::cout;
  cout << ' ' << std::string(kOpW,' ')
       << " |" << center("Kokkos TF (triple-float)", kBkndW)
       << "|" << center("CUDA FP64", kBkndW) << "|\n";
  cout << ' ' << std::string(kOpW,' ')
       << " |" << center("Time (ms)", kTimeSec) << "|" << center("Accuracy (digits)", kAccSec)
       << "|" << center("Time (ms)", kTimeSec) << "|" << center("Accuracy (digits)", kAccSec) << "|\n";
  print_sep_real();
  cout << ' ' << std::left << std::setw(kOpW) << ""
       << " |" << center("Min",kTW) << "|" << center("Max",kTW)
       << "|" << center("Med",kTW)  << "|" << center("Mean",kTW)
       << "|" << center("Min",kAW)  << "|" << center("Max",kAW)
       << "|" << center("Med",kAW)  << "|" << center("Mean",kAW)
       << "|" << center("Min",kTW)  << "|" << center("Max",kTW)
       << "|" << center("Med",kTW)  << "|" << center("Mean",kTW)
       << "|" << center("Min",kAW)  << "|" << center("Max",kAW)
       << "|" << center("Med",kAW)  << "|" << center("Mean",kAW) << "|\n";
  cout << '=' << dashes(kOpW) << "=+"
       << dashes(kTimeSec) << "+" << dashes(kAccSec) << "+"
       << dashes(kTimeSec) << "+" << dashes(kAccSec) << "+\n";
}

static void print_row_real(const OpResult& r) {
  using std::cout; using std::setw; using std::right; using std::fixed; using std::setprecision;
  auto T = [](double s) { return s * 1000.0; };
  cout << ' ' << std::left << std::setw(kOpW) << op_name(r.op) << " |"
       << right << fixed << setprecision(4)
       << setw(kTW) << T(r.tf_timing.min_s)    << "|"
       << setw(kTW) << T(r.tf_timing.max_s)    << "|"
       << setw(kTW) << T(r.tf_timing.median_s) << "|"
       << setw(kTW) << T(r.tf_timing.mean_s)   << "|"
       << setprecision(2)
       << setw(kAW) << r.tf_acc.min_d    << "|"
       << setw(kAW) << r.tf_acc.max_d    << "|"
       << setw(kAW) << r.tf_acc.median_d << "|"
       << setw(kAW) << r.tf_acc.mean_d   << "|"
       << setprecision(4)
       << setw(kTW) << T(r.dbl_timing.min_s)    << "|"
       << setw(kTW) << T(r.dbl_timing.max_s)    << "|"
       << setw(kTW) << T(r.dbl_timing.median_s) << "|"
       << setw(kTW) << T(r.dbl_timing.mean_s)   << "|"
       << setprecision(2)
       << setw(kAW) << r.dbl_acc.min_d    << "|"
       << setw(kAW) << r.dbl_acc.max_d    << "|"
       << setw(kAW) << r.dbl_acc.median_d << "|"
       << setw(kAW) << r.dbl_acc.mean_d   << "|\n";
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  cfg.all_ops = true;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--op") { cfg.all_ops = false; break; }

  if (!parse_args(argc, argv, cfg)) {
    print_usage(argv[0]);
    return 1;
  }

  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nbatch=" << cfg.batch << "  repeats=" << cfg.repeats
              << "  seed=" << cfg.seed << "  warmup=" << kWarmupRuns
              << "  timing=kernel+fence\n\n";
    print_header_real();

    std::vector<OpResult> results;
    if (cfg.all_ops) {
      for (Op op : kAllOps) results.push_back(run_op(op, cfg));
    } else {
      results.push_back(run_op(cfg.op, cfg));
    }
    for (const OpResult& r : results) { print_row_real(r); print_sep_real(); }
    std::cout << "\n";

    // ---- Pass/fail verdict (S10 P5 deliverable) --------------------------------
    // Each op must reach kPassMeanDigits mean accuracy vs the __float128 oracle,
    // unless it is conditioning-limited (is_conditioning_limited). Any other
    // shortfall sets RC 1.
    std::cout << "Accuracy verdict (gate: mean >= " << std::fixed << std::setprecision(1)
              << kPassMeanDigits << " digits vs __float128; conditioning-limited ops exempt)\n";
    int n_pass = 0, n_cond = 0, n_fail = 0;
    for (const OpResult& r : results) {
      bool cond = is_conditioning_limited(r.op);
      bool ok   = r.tf_acc.mean_d >= kPassMeanDigits;
      const char* tag;
      if (ok)             { tag = "PASS";        ++n_pass; }
      else if (cond)      { tag = "PASS (cond)"; ++n_cond; }
      else                { tag = "FAIL";        ++n_fail; rc = 1; }
      if (!ok) {
        std::cout << "  " << std::left << std::setw(10) << op_name(r.op)
                  << " mean=" << std::right << std::fixed << std::setprecision(2)
                  << std::setw(6) << r.tf_acc.mean_d
                  << "  min=" << std::setw(6) << r.tf_acc.min_d
                  << "  " << tag << "\n";
      }
    }
    std::cout << "\nSummary: " << n_pass << " pass, " << n_cond
              << " pass (conditioning-limited), " << n_fail << " fail  ->  RC "
              << rc << "\n\n";
  }
  Kokkos::finalize();
  return rc;
}
