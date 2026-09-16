// Kokkos DD (double-double) real ops demo -- TIMING AND SMOKE ONLY.
// Each operation is run on double-double (DD) and FP64. The table shows
// slowdown vs FP64 (min/max/med/mean across repeats).
//
// THIS DEMO DOES NOT MEASURE ACCURACY, DELIBERATELY. The accuracy record is
// validation/sweep/, scored in ulps against an MPFR/MPC oracle with one verdict
// per point (docs/CORRECTNESS.md). The oracle columns this demo used to print
// duplicated that at lower resolution, and cost the project its libquadmath
// dependency: they routed through impl/Kokkos_QuadPrecisionMath.hpp, which
// exists only in a Kokkos built with Kokkos_ENABLE_LIBQUADMATH=ON.

#include <Kokkos_Core.hpp>

#include <dd_math.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace dd    = Kokkos::Experimental;

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

// ---- Slowdown stats --------------------------------------------------------

struct SlowdownStats { double min_x = 1, max_x = 1, median_x = 1, mean_x = 1; };

static SlowdownStats compute_slowdown(const TimeStats& backend, const TimeStats& fp64) {
  auto sdiv = [](double a, double b) { return b > 0.0 ? a/b : 1.0; };
  return { sdiv(backend.min_s, fp64.min_s),
           sdiv(backend.max_s, fp64.max_s),
           sdiv(backend.median_s, fp64.median_s),
           sdiv(backend.mean_s, fp64.mean_s) };
}

static std::string fmt_slow(double x) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%.1fx", x);
  return buf;
}

// ---- Per-op result ---------------------------------------------------------

struct OpResult {
  Op        op;
  TimeStats dd_timing, fp64_timing;
};

// ---- View types ------------------------------------------------------------

using exec_space = Kokkos::DefaultExecutionSpace;
using policy_1d  = Kokkos::RangePolicy<exec_space>;
using vdd_t      = Kokkos::View<dd::DoubleDouble*,    Kokkos::LayoutRight, exec_space>;
using vdbl       = Kokkos::View<double*,          Kokkos::LayoutRight, exec_space>;

OpResult run_op(Op op, const Config& cfg) {
  const int n = cfg.batch;

  std::vector<double> ha(n), hb(n), hc(n, 0.0);

  fill_inputs(op, ha.data(), hb.data(), hc.data(), n, cfg.seed);

  vdd_t  add("add",n),   bdd("bdd",n),   cdd("cdd",n),   rdd("rdd",n);
  vdbl   ad("ad",n),     bd("bd",n),     cd("cd",n),     rd("rd",n);

  {
    auto madd=Kokkos::create_mirror_view(add), mbdd=Kokkos::create_mirror_view(bdd);
    auto mcdd=Kokkos::create_mirror_view(cdd);
    auto mad=Kokkos::create_mirror_view(ad), mbd=Kokkos::create_mirror_view(bd);
    auto mcd=Kokkos::create_mirror_view(cd);
    for (int i=0; i<n; ++i) {
      madd(i) = dd::DoubleDouble(ha[i]);
      mbdd(i) = dd::DoubleDouble(hb[i]);
      mcdd(i) = dd::DoubleDouble(hc[i]);
      mad(i)  = ha[i]; mbd(i) = hb[i]; mcd(i) = hc[i];
    }
    Kokkos::deep_copy(add,madd); Kokkos::deep_copy(bdd,mbdd); Kokkos::deep_copy(cdd,mcdd);
    Kokkos::deep_copy(ad,mad); Kokkos::deep_copy(bd,mbd); Kokkos::deep_copy(cd,mcd);
  }

  policy_1d pol(0, n);
  TimeStats st_dd, st_dbl;

  // ---- DD kernels ------------------------------------------------------------
  switch (op) {
    case Op::Add:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_add",pol,KOKKOS_LAMBDA(int i){rdd(i)=add(i)+bdd(i);});}); break;
    case Op::Sub:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_sub",pol,KOKKOS_LAMBDA(int i){rdd(i)=add(i)-bdd(i);});}); break;
    case Op::Mul:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_mul",pol,KOKKOS_LAMBDA(int i){rdd(i)=add(i)*bdd(i);});}); break;
    case Op::Div:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_div",pol,KOKKOS_LAMBDA(int i){rdd(i)=add(i)/bdd(i);});}); break;
    case Op::Sqrt:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_sqrt",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::sqrt(add(i));});}); break;
    case Op::Abs:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_abs",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::abs(add(i));});}); break;
    case Op::Exp:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_exp",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::exp(add(i));});}); break;
    case Op::Log:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_log",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::log(add(i));});}); break;
    case Op::Exp2:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_exp2",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::exp2(add(i));});}); break;
    case Op::Exp10:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_exp10",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::exp10(add(i));});}); break;
    case Op::Expm1:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_expm1",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::expm1(add(i));});}); break;
    case Op::Log2:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_log2",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::log2(add(i));});}); break;
    case Op::Log10:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_log10",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::log10(add(i));});}); break;
    case Op::Log1p:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_log1p",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::log1p(add(i));});}); break;
    case Op::Sin:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_sin",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::sin(add(i));});}); break;
    case Op::Cos:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_cos",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::cos(add(i));});}); break;
    case Op::Tan:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_tan",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::tan(add(i));});}); break;
    case Op::Asin:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_asin",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::asin(add(i));});}); break;
    case Op::Acos:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_acos",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::acos(add(i));});}); break;
    case Op::Atan:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_atan",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::atan(add(i));});}); break;
    case Op::Sinh:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_sinh",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::sinh(add(i));});}); break;
    case Op::Cosh:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_cosh",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::cosh(add(i));});}); break;
    case Op::Tanh:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_tanh",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::tanh(add(i));});}); break;
    case Op::Acosh:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_acosh",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::acosh(add(i));});}); break;
    case Op::Asinh:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_asinh",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::asinh(add(i));});}); break;
    case Op::Atanh:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_atanh",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::atanh(add(i));});}); break;
    case Op::Pow:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_pow",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::pow(add(i),bdd(i));});}); break;
    case Op::Hypot:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_hypot",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::hypot(add(i),bdd(i));});}); break;
    case Op::Fmod:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_fmod",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::fmod(add(i),bdd(i));});}); break;
    case Op::Remainder:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_rem",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::remainder(add(i),bdd(i));});}); break;
    case Op::Copysign:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_cs",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::copysign(add(i),bdd(i));});}); break;
    case Op::Fmax:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_fmax",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::fmax(add(i),bdd(i));});}); break;
    case Op::Fmin:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_fmin",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::fmin(add(i),bdd(i));});}); break;
    case Op::Fdim:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_fdim",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::fdim(add(i),bdd(i));});}); break;
    case Op::Fma:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_fma",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::fma(add(i),bdd(i),cdd(i));});}); break;
    case Op::Ceil:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_ceil",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::ceil(add(i));});}); break;
    case Op::Floor:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_floor",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::floor(add(i));});}); break;
    case Op::Round:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_round",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::round(add(i));});}); break;
    case Op::Trunc:
      st_dd=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dd_trunc",pol,KOKKOS_LAMBDA(int i){rdd(i)=dd::trunc(add(i));});}); break;
  }

  // ---- FP64 kernels (timing baseline only) -----------------------------------
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
  auto mrdd  = Kokkos::create_mirror_view(rdd);  Kokkos::deep_copy(mrdd, rdd);

  return {op, st_dd, st_dbl};
}

// ---- Table printing --------------------------------------------------------
// Layout: op(10) | DD: slow×4 |
// kSW=7 fits "100.0x" (6 chars) right-aligned.

static constexpr int kOpW     = 10;
static constexpr int kSW      =  7;
static constexpr int kSlowSec = 4*kSW + 3;   // 31
static constexpr int kBkndW   = kSlowSec;    // 31

static std::string dashes(int n) { return std::string((size_t)n, '-'); }

static std::string center(const std::string& s, int w) {
  int pad = w - (int)s.size();
  int lp  = pad / 2, rp = pad - lp;
  return std::string((size_t)lp,' ') + s + std::string((size_t)rp,' ');
}

static void print_sep_real() {
  std::cout << '-' << dashes(kOpW) << "-+" << dashes(kSlowSec) << "+\n";
}

static void print_header_real() {
  using std::cout;
  cout << ' ' << std::string(kOpW,' ')
       << " |" << center("Kokkos DD (double-double)", kBkndW) << "|\n";
  cout << ' ' << std::string(kOpW,' ')
       << " |" << center("Slowdown vs FP64", kSlowSec) << "|\n";
  print_sep_real();
  cout << ' ' << std::left << std::setw(kOpW) << ""
       << " |" << center("Min",kSW) << "|" << center("Max",kSW)
       << "|" << center("Med",kSW)  << "|" << center("Mean",kSW) << "|\n";
  cout << '=' << dashes(kOpW) << "=+" << dashes(kSlowSec) << "+\n";
}

static void print_row_real(const OpResult& r) {
  using std::cout; using std::setw; using std::right;
  SlowdownStats sd = compute_slowdown(r.dd_timing,    r.fp64_timing);
  cout << ' ' << std::left << std::setw(kOpW) << op_name(r.op) << " |"
       << right
       << setw(kSW) << fmt_slow(sd.min_x)    << "|"
       << setw(kSW) << fmt_slow(sd.max_x)    << "|"
       << setw(kSW) << fmt_slow(sd.median_x) << "|"
       << setw(kSW) << fmt_slow(sd.mean_x)   << "|\n";
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
    if (cfg.all_ops) {
      for (Op op : kAllOps) {
        print_row_real(run_op(op, cfg));
        print_sep_real();
      }
    } else {
      print_row_real(run_op(cfg.op, cfg));
      print_sep_real();
    }
    std::cout << "\n";
  }
  Kokkos::finalize();
  return 0;
}
