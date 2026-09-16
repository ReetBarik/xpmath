// Kokkos quad-float demo — real ops. TIMING AND SMOKE ONLY.
// Times Kokkos::Experimental::QuadFloat against FP64, op by op.
//
// THIS DEMO DOES NOT MEASURE ACCURACY, DELIBERATELY. The accuracy record is
// validation/sweep/, scored in ulps against an MPFR/MPC oracle with one verdict
// per point (docs/CORRECTNESS.md). The oracle columns this demo used to print
// duplicated that at lower resolution, and cost the project its libquadmath
// dependency: they routed through impl/Kokkos_QuadPrecisionMath.hpp, which
// exists only in a Kokkos built with Kokkos_ENABLE_LIBQUADMATH=ON. The RC-0/RC-1
// accuracy verdict they fed went with them: the contract allows exactly one
// verdict per point, and the sweep gates are the ones that issue it.
//
// ============================================================
// QF usage reference  (namespace qf = Kokkos::Experimental)
// ============================================================
//
// Construction
//   qf::QuadFloat x;               // zero
//   qf::QuadFloat x(1.5f);         // from float
//   qf::QuadFloat x(hi, lo);       // from two float components
//
// Arithmetic operators
//   x + y,  x - y,  x * y,  x / y    // QuadFloat op QuadFloat -> QuadFloat
//   x + b,  x - b,  x * b,  x / b    // QuadFloat op float  -> QuadFloat
//   a + y,  a - y,  a * y,  a / y    // float  op QuadFloat -> QuadFloat
//   -x                                // unary negation
//   x += y, x -= y, x *= y, x /= y   // QuadFloat op QuadFloat
//   x += b, x -= b, x *= b, x /= b   // QuadFloat op float
//
// Math functions  (all KOKKOS_INLINE_FUNCTION, host + device)
//   qf::abs(x)
//   qf::sqrt(x)
//   qf::exp(x),   qf::exp2(x),  qf::exp10(x), qf::expm1(x)
//   qf::log(x),   qf::log2(x),  qf::log10(x), qf::log1p(x)
//   qf::sin(x),   qf::cos(x),   qf::tan(x)
//   qf::asin(x),  qf::acos(x),  qf::atan(x),  qf::atan2(y, x)
//   qf::sinh(x),  qf::cosh(x),  qf::tanh(x)
//   qf::asinh(x), qf::acosh(x), qf::atanh(x)
//   qf::pow(x, y)                // QuadFloat exponent
//   qf::pow_int(x, n)            // int exponent
//   qf::hypot(x, y)
//   qf::erf(x),   qf::erfc(x)
//   qf::tgamma(x)
//   qf::ceil(x),  qf::floor(x),  qf::round(x), qf::trunc(x)
//   qf::fmod(x, y),  qf::remainder(x, y)
//   qf::fma(x, y, z)
//   qf::fmax(x, y),  qf::fmin(x, y),  qf::fdim(x, y)
//   qf::copysign(x, y)
//   qf::zeta(s)                  // Riemann zeta
//   qf::bessel_j0(x), qf::bessel_j1(x), qf::bessel_jn(n, x)
//   qf::bessel_y0(x), qf::bessel_y1(x), qf::bessel_yn(n, x)
//   qf::sincos(x, c, s)          // void; writes c=cos(x), s=sin(x)
//   qf::sinhcosh(x, ch, sh)      // void; writes ch=cosh(x), sh=sinh(x)
//
// Constants
//   qf::QuadFloat_pi()           // pi
//   qf::QuadFloat_e()            // e
//   qf::QuadFloat_log2()         // ln 2
//   qf::QuadFloat_log10()        // ln 10
//   qf::QuadFloat_sqrt2()        // sqrt(2)
//   qf::QuadFloat_euler_gamma()  // Euler-Mascheroni gamma
// ============================================================

#include <Kokkos_Core.hpp>

#include <qf_math.hpp>

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

void print_usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [--op <name>] [--batch N] [--repeats N] [--seed N]\n"
    << "  Omit --op to run all operations and print a complete table.\n"
    << "  Operations: add sub mul div sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p\n"
    << "              sin cos tan asin acos atan sinh cosh tanh acosh asinh atanh\n"
    << "              pow hypot fmod remainder copysign fmax fmin fdim fma\n"
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

// Inputs are sampled in double precision then narrowed to FP32 for QF and to FP64 for the
// FP64 baseline.
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

// ---- Per-op runner ---------------------------------------------------------

struct OpResult { Op op; TimeStats qf_timing, dbl_timing; };

using exec_space = Kokkos::DefaultExecutionSpace;
using policy_1d  = Kokkos::RangePolicy<exec_space>;
using vqf        = Kokkos::View<Kokkos::Experimental::QuadFloat*, Kokkos::LayoutRight, exec_space>;
using vdbl       = Kokkos::View<double*,             Kokkos::LayoutRight, exec_space>;

namespace qf = Kokkos::Experimental;

OpResult run_op(Op op, const Config& cfg) {
  const int n = cfg.batch;

  std::vector<double> ha(n), hb(n), hc(n, 0.0);

  fill_inputs(op, ha.data(), hb.data(), hc.data(), n, cfg.seed);

  vqf  aqf("aqf",n), bqf("bqf",n), cqf("cqf",n), rqf("rqf",n);
  vdbl ad("ad",n), bd("bd",n), cd("cd",n), rd("rd",n);
  {
    auto maqf=Kokkos::create_mirror_view(aqf), mbqf=Kokkos::create_mirror_view(bqf);
    auto mcqf=Kokkos::create_mirror_view(cqf);
    auto mad=Kokkos::create_mirror_view(ad), mbd=Kokkos::create_mirror_view(bd);
    auto mcd=Kokkos::create_mirror_view(cd);
    for (int i=0; i<n; ++i) {
      // Use QuadFloat(double) so the QF input faithfully encodes the FP64 value
      // (Route-A successive split into f0..f3). A double carries only 53 bits, so
      // f0,f1 hold it exactly and f2,f3 fall to 0 — the input is exact in QF, and
      // accuracy is bounded only by the QF math, not the input encoding.
      maqf(i)=qf::QuadFloat(ha[i]); mbqf(i)=qf::QuadFloat(hb[i]);
      mcqf(i)=qf::QuadFloat(hc[i]);
      mad(i)=ha[i]; mbd(i)=hb[i]; mcd(i)=hc[i];
    }
    Kokkos::deep_copy(aqf,maqf); Kokkos::deep_copy(bqf,mbqf);
    Kokkos::deep_copy(cqf,mcqf);
    Kokkos::deep_copy(ad,mad); Kokkos::deep_copy(bd,mbd); Kokkos::deep_copy(cd,mcd);
  }

  policy_1d pol(0, n);
  TimeStats st_qf, st_dbl;

  // ---- QF kernels ------------------------------------------------------------
  switch (op) {
    case Op::Add:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_add",pol,KOKKOS_LAMBDA(int i){rqf(i)=aqf(i)+bqf(i);});}); break;
    case Op::Sub:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_sub",pol,KOKKOS_LAMBDA(int i){rqf(i)=aqf(i)-bqf(i);});}); break;
    case Op::Mul:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_mul",pol,KOKKOS_LAMBDA(int i){rqf(i)=aqf(i)*bqf(i);});}); break;
    case Op::Div:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_div",pol,KOKKOS_LAMBDA(int i){rqf(i)=aqf(i)/bqf(i);});}); break;
    case Op::Sqrt:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_sqrt",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::sqrt(aqf(i));});}); break;
    case Op::Abs:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_abs",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::abs(aqf(i));});}); break;
    case Op::Exp:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_exp",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::exp(aqf(i));});}); break;
    case Op::Log:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_log",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::log(aqf(i));});}); break;
    case Op::Exp2:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_exp2",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::exp2(aqf(i));});}); break;
    case Op::Exp10:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_exp10",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::exp10(aqf(i));});}); break;
    case Op::Expm1:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_expm1",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::expm1(aqf(i));});}); break;
    case Op::Log2:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_log2",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::log2(aqf(i));});}); break;
    case Op::Log10:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_log10",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::log10(aqf(i));});}); break;
    case Op::Log1p:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_log1p",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::log1p(aqf(i));});}); break;
    case Op::Sin:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_sin",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::sin(aqf(i));});}); break;
    case Op::Cos:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_cos",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::cos(aqf(i));});}); break;
    case Op::Tan:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_tan",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::tan(aqf(i));});}); break;
    case Op::Asin:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_asin",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::asin(aqf(i));});}); break;
    case Op::Acos:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_acos",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::acos(aqf(i));});}); break;
    case Op::Atan:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_atan",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::atan(aqf(i));});}); break;
    case Op::Sinh:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_sinh",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::sinh(aqf(i));});}); break;
    case Op::Cosh:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_cosh",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::cosh(aqf(i));});}); break;
    case Op::Tanh:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_tanh",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::tanh(aqf(i));});}); break;
    case Op::Acosh:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_acosh",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::acosh(aqf(i));});}); break;
    case Op::Asinh:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_asinh",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::asinh(aqf(i));});}); break;
    case Op::Atanh:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_atanh",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::atanh(aqf(i));});}); break;
    case Op::Pow:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_pow",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::pow(aqf(i),bqf(i));});}); break;
    case Op::Hypot:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_hypot",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::hypot(aqf(i),bqf(i));});}); break;
    case Op::Fmod:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_fmod",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::fmod(aqf(i),bqf(i));});}); break;
    case Op::Remainder:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_rem",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::remainder(aqf(i),bqf(i));});}); break;
    case Op::Copysign:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_cs",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::copysign(aqf(i),bqf(i));});}); break;
    case Op::Fmax:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_fmax",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::fmax(aqf(i),bqf(i));});}); break;
    case Op::Fmin:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_fmin",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::fmin(aqf(i),bqf(i));});}); break;
    case Op::Fdim:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_fdim",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::fdim(aqf(i),bqf(i));});}); break;
    case Op::Fma:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_fma",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::fma(aqf(i),bqf(i),cqf(i));});}); break;
    case Op::Ceil:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_ceil",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::ceil(aqf(i));});}); break;
    case Op::Floor:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_floor",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::floor(aqf(i));});}); break;
    case Op::Round:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_round",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::round(aqf(i));});}); break;
    case Op::Trunc:
      st_qf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qf_trunc",pol,KOKKOS_LAMBDA(int i){rqf(i)=qf::trunc(aqf(i));});}); break;
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
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dbl_cs",pol,KOKKOS_LAMBDA(int i){rd(i)=Kokkos::copysign(ad(i),bd(i));});}); break;
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

  // Smoke: bring the results back host-side. Nothing scores them -- this is
  // here so the demo still exercises the whole round trip (kernel output ->
  // mirror -> host) for a user-defined type, which is where a broken deep_copy
  // would show up. It sits outside the timed region.
  auto mrqf = Kokkos::create_mirror_view(rqf);
  Kokkos::deep_copy(mrqf, rqf);
  auto mrd = Kokkos::create_mirror_view(rd);
  Kokkos::deep_copy(mrd, rd);

  return {op, st_qf, st_dbl};
}

// ---- Table printing --------------------------------------------------------

static constexpr int kOpW = 10;
static constexpr int kTW  =  9;
static constexpr int kTimeSec = 4*kTW + 3;
static constexpr int kBkndW   = kTimeSec;

static std::string dashes(int n) { return std::string((size_t)n, '-'); }

static std::string center(const std::string& s, int w) {
  int pad = w - (int)s.size();
  int lp  = pad / 2, rp = pad - lp;
  return std::string((size_t)lp,' ') + s + std::string((size_t)rp,' ');
}

static void print_sep_real() {
  std::cout << '-' << dashes(kOpW) << "-+"
            << dashes(kTimeSec) << "+" << dashes(kTimeSec) << "+\n";
}

static void print_header_real() {
  using std::cout;
  cout << ' ' << std::string(kOpW,' ')
       << " |" << center("Kokkos QF (quad-float)", kBkndW)
       << "|" << center("CUDA FP64", kBkndW) << "|\n";
  cout << ' ' << std::string(kOpW,' ')
       << " |" << center("Time (ms)", kTimeSec)
       << "|" << center("Time (ms)", kTimeSec) << "|\n";
  print_sep_real();
  cout << ' ' << std::left << std::setw(kOpW) << ""
       << " |" << center("Min",kTW) << "|" << center("Max",kTW)
       << "|" << center("Med",kTW)  << "|" << center("Mean",kTW)
       << "|" << center("Min",kTW)  << "|" << center("Max",kTW)
       << "|" << center("Med",kTW)  << "|" << center("Mean",kTW) << "|\n";
  cout << '=' << dashes(kOpW) << "=+"
       << dashes(kTimeSec) << "+" << dashes(kTimeSec) << "+\n";
}

static void print_row_real(const OpResult& r) {
  using std::cout; using std::setw; using std::right; using std::fixed; using std::setprecision;
  auto T = [](double s) { return s * 1000.0; };
  cout << ' ' << std::left << std::setw(kOpW) << op_name(r.op) << " |"
       << right << fixed << setprecision(4)
       << setw(kTW) << T(r.qf_timing.min_s)    << "|"
       << setw(kTW) << T(r.qf_timing.max_s)    << "|"
       << setw(kTW) << T(r.qf_timing.median_s) << "|"
       << setw(kTW) << T(r.qf_timing.mean_s)   << "|"
       << setw(kTW) << T(r.dbl_timing.min_s)    << "|"
       << setw(kTW) << T(r.dbl_timing.max_s)    << "|"
       << setw(kTW) << T(r.dbl_timing.median_s) << "|"
       << setw(kTW) << T(r.dbl_timing.mean_s)   << "|\n";
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
  }
  Kokkos::finalize();
  return 0;
}
