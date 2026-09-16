// Kokkos FF (float-float) complex ops demo -- TIMING AND SMOKE ONLY.
// Each operation is run on Kokkos::Experimental::FloatFloatComplex and on
// Kokkos::complex<double>. The table shows wall time for each, one row per op
// (the real and imag parts share a kernel, so they shared a timing figure even
// when this demo printed two rows).
//
// THIS DEMO DOES NOT MEASURE ACCURACY, DELIBERATELY. The accuracy record is
// validation/sweep/, scored in ulps against an MPFR/MPC oracle with one verdict
// per point (docs/CORRECTNESS.md). The oracle columns this demo used to print
// duplicated that at lower resolution, and were the single most expensive part
// of the project's libquadmath dependency: the __complex128 overloads they
// called came from impl/Kokkos_ComplexQuadPrecisionMath.hpp, a header that is
// NOT upstream in Kokkos and had to be patched into every Kokkos install by
// hand. Deleting the columns deleted the patch.
//
// ============================================================
// FF usage reference  (namespace ff = Kokkos::Experimental)
// ============================================================
//
// Construction
//   ff::FloatFloatComplex z;                  // zero
//   ff::FloatFloatComplex z(1.5f);            // real from float
//   ff::FloatFloatComplex z(x);               // real from FloatFloat
//   ff::FloatFloatComplex z(1.0f, 2.0f);      // from two floats (re, im)
//   ff::FloatFloatComplex z(x, y);            // from two FloatFloats (re, im)
//
// Arithmetic operators
//   z + w,  z - w,  z * w,  z / w    // FloatFloatComplex op FloatFloatComplex -> FloatFloatComplex
//   -z                                // unary negation
//   z += w, z -= w, z *= w, z /= w   // FloatFloatComplex op FloatFloatComplex
//
// Complex math functions  (all KOKKOS_INLINE_FUNCTION, host + device)
//   ff::abs(z)                        // -> FloatFloat (modulus)
//   ff::conj(z)                       // -> FloatFloatComplex
//   ff::sqrt(z)                       // -> FloatFloatComplex
//   ff::exp(z),   ff::log(z),   ff::log10(z)
//   ff::sin(z),   ff::cos(z),   ff::tan(z)
//   ff::asin(z),  ff::acos(z),  ff::atan(z)
//   ff::sinh(z),  ff::cosh(z),  ff::tanh(z)
//   ff::asinh(z), ff::acosh(z), ff::atanh(z)
//   ff::pow(z, w)                     // FloatFloatComplex exponent
//   ff::polar(r, theta)               // r*exp(i*theta), r and theta are FloatFloat
//
// Real math functions also available (see demo_ff_real.cpp)
//
// Constants
//   ff::FloatFloat_pi()           // pi
//   ff::FloatFloat_e()            // e
//   ff::FloatFloat_log2()         // ln 2
//   ff::FloatFloat_log10()        // ln 10
//   ff::FloatFloat_sqrt2()        // sqrt(2)
//   ff::FloatFloat_euler_gamma()  // Euler-Mascheroni gamma
// ============================================================

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>


#include <ff_math.hpp>
#include <ff_complex.hpp>

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
  Abs, Conj, Sqrt, Exp, Log, Log10,
  Sin, Cos, Tan, Asin, Acos, Atan,
  Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
  Pow, Polar,
};

static const Op kAllOps[] = {
  Op::Add, Op::Sub, Op::Mul, Op::Div,
  Op::Abs, Op::Conj, Op::Sqrt, Op::Exp, Op::Log, Op::Log10,
  Op::Sin, Op::Cos, Op::Tan, Op::Asin, Op::Acos, Op::Atan,
  Op::Sinh, Op::Cosh, Op::Tanh, Op::Asinh, Op::Acosh, Op::Atanh,
  Op::Pow, Op::Polar,
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
  if (s=="add")   {out=Op::Add;   return true;}
  if (s=="sub")   {out=Op::Sub;   return true;}
  if (s=="mul")   {out=Op::Mul;   return true;}
  if (s=="div")   {out=Op::Div;   return true;}
  if (s=="abs")   {out=Op::Abs;   return true;}
  if (s=="conj")  {out=Op::Conj;  return true;}
  if (s=="sqrt")  {out=Op::Sqrt;  return true;}
  if (s=="exp")   {out=Op::Exp;   return true;}
  if (s=="log")   {out=Op::Log;   return true;}
  if (s=="log10") {out=Op::Log10; return true;}
  if (s=="sin")   {out=Op::Sin;   return true;}
  if (s=="cos")   {out=Op::Cos;   return true;}
  if (s=="tan")   {out=Op::Tan;   return true;}
  if (s=="asin")  {out=Op::Asin;  return true;}
  if (s=="acos")  {out=Op::Acos;  return true;}
  if (s=="atan")  {out=Op::Atan;  return true;}
  if (s=="sinh")  {out=Op::Sinh;  return true;}
  if (s=="cosh")  {out=Op::Cosh;  return true;}
  if (s=="tanh")  {out=Op::Tanh;  return true;}
  if (s=="asinh") {out=Op::Asinh; return true;}
  if (s=="acosh") {out=Op::Acosh; return true;}
  if (s=="atanh") {out=Op::Atanh; return true;}
  if (s=="pow")   {out=Op::Pow;   return true;}
  if (s=="polar") {out=Op::Polar; return true;}
  // clang-format on
  return false;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Add:   return "add";
    case Op::Sub:   return "sub";
    case Op::Mul:   return "mul";
    case Op::Div:   return "div";
    case Op::Abs:   return "abs";
    case Op::Conj:  return "conj";
    case Op::Sqrt:  return "sqrt";
    case Op::Exp:   return "exp";
    case Op::Log:   return "log";
    case Op::Log10: return "log10";
    case Op::Sin:   return "sin";
    case Op::Cos:   return "cos";
    case Op::Tan:   return "tan";
    case Op::Asin:  return "asin";
    case Op::Acos:  return "acos";
    case Op::Atan:  return "atan";
    case Op::Sinh:  return "sinh";
    case Op::Cosh:  return "cosh";
    case Op::Tanh:  return "tanh";
    case Op::Asinh: return "asinh";
    case Op::Acosh: return "acosh";
    case Op::Atanh: return "atanh";
    case Op::Pow:   return "pow";
    case Op::Polar: return "polar";
  }
  return "?";
}

void print_usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [--op <name>] [--batch N] [--repeats N] [--seed N]\n"
    << "  Omit --op to run all operations and print complete tables.\n"
    << "  Operations: add sub mul div abs conj sqrt exp log log10\n"
    << "              sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow polar\n"
    << "  Defaults: batch=1000000 repeats=" << kDefaultRepeats << " seed=" << kDefaultSeed << "\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") return false;
    auto need = [&](const char* w) -> const char* {
      if (i+1>=argc){std::cerr<<"Missing value after "<<w<<"\n";return nullptr;}
      return argv[++i];
    };
    if (a=="--op") {
      const char* v=need("--op"); if(!v) return false;
      if(!parse_op(v,cfg.op)){std::cerr<<"Unknown op: "<<v<<"\n";return false;}
      cfg.all_ops=false;
    } else if (a=="--batch") {
      const char* v=need("--batch"); if(!v) return false;
      cfg.batch=std::atoi(v);
      if(cfg.batch<=0){std::cerr<<"Invalid --batch\n";return false;}
    } else if (a=="--repeats") {
      const char* v=need("--repeats"); if(!v) return false;
      cfg.repeats=std::atoi(v);
      if(cfg.repeats<=0){std::cerr<<"Invalid --repeats\n";return false;}
    } else if (a=="--seed") {
      const char* v=need("--seed"); if(!v) return false;
      cfg.seed=static_cast<uint64_t>(std::strtoull(v,nullptr,10));
    } else {
      std::cerr<<"Unknown argument: "<<a<<"\n"; return false;
    }
  }
  return true;
}

// ---- Input generation -------------------------------------------------------

void fill_inputs(Op op, double* ha_re, double* ha_im,
                         double* hb_re, double* hb_im, int n, uint64_t seed) {
  std::mt19937_64 gen(seed);
  constexpr double pi = 3.14159265358979323846;

  auto fill_a = [&](double lo_re, double hi_re, double lo_im, double hi_im) {
    std::uniform_real_distribution<double> dr(lo_re,hi_re), di(lo_im,hi_im);
    for (int i=0;i<n;++i){ha_re[i]=dr(gen);ha_im[i]=di(gen);hb_re[i]=0;hb_im[i]=0;}
  };
  auto fill_ab = [&](double lo_re, double hi_re, double lo_im, double hi_im) {
    std::uniform_real_distribution<double> dr(lo_re,hi_re), di(lo_im,hi_im);
    for (int i=0;i<n;++i){ha_re[i]=dr(gen);ha_im[i]=di(gen);hb_re[i]=dr(gen);hb_im[i]=di(gen);}
  };

  switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: fill_ab(0.1,10.0,0.1,10.0); break;
    case Op::Abs: case Op::Conj: fill_a(-10,10,-10,10); break;
    case Op::Sqrt:  fill_a(-10,10,-10,10); break;
    case Op::Exp:   fill_a(-10,10,-pi,pi); break;
    case Op::Log: case Op::Log10: fill_a(0.1,10,-10,10); break;
    case Op::Sin: case Op::Cos: case Op::Tan: fill_a(-pi,pi,-2,2); break;
    case Op::Asin: case Op::Acos: fill_a(-1,1,-1,1); break;
    case Op::Atan:  fill_a(-10,10,-0.9,0.9); break;
    case Op::Sinh: case Op::Cosh: case Op::Tanh: fill_a(-5,5,-pi,pi); break;
    case Op::Asinh: fill_a(-10,10,-10,10); break;
    case Op::Acosh: fill_a(0,10,-5,5); break;
    case Op::Atanh: fill_a(-0.9,0.9,-0.9,0.9); break;
    case Op::Pow: {
      std::uniform_real_distribution<double> dbre(0.1,10),dbim(-5,5),dere(0,3),deim(-1,1);
      for (int i=0;i<n;++i){ha_re[i]=dbre(gen);ha_im[i]=dbim(gen);hb_re[i]=dere(gen);hb_im[i]=deim(gen);}
      break;
    }
    case Op::Polar: {
      std::uniform_real_distribution<double> dr(0.01,100), dth(-pi,pi);
      for (int i=0;i<n;++i){ha_re[i]=dr(gen);ha_im[i]=dth(gen);hb_re[i]=0;hb_im[i]=0;}
      break;
    }
  }
}

// ---- Timing ----------------------------------------------------------------

struct TimeStats { double min_s=0, max_s=0, median_s=0, mean_s=0; };

TimeStats summarize_times(std::vector<double> t) {
  if (t.empty()) return {};
  std::sort(t.begin(), t.end());
  TimeStats s;
  s.min_s    = t.front();
  s.max_s    = t.back();
  size_t n   = t.size();
  s.median_s = (n%2==1) ? t[n/2] : 0.5*(t[n/2-1]+t[n/2]);
  s.mean_s   = std::accumulate(t.begin(),t.end(),0.0)/(double)n;
  return s;
}

using wall_clock = std::chrono::high_resolution_clock;

template <typename Launch>
TimeStats time_kernel_fence(int repeats, Launch&& launch) {
  for (int w=0;w<kWarmupRuns;++w){launch();Kokkos::fence();}
  std::vector<double> times; times.reserve((size_t)repeats);
  for (int r=0;r<repeats;++r) {
    auto t0=wall_clock::now(); launch(); Kokkos::fence();
    times.push_back(std::chrono::duration<double>(wall_clock::now()-t0).count());
  }
  return summarize_times(std::move(times));
}

// ---- Per-op result ---------------------------------------------------------

struct ComplexOpResult {
  Op        op;
  TimeStats ff_timing, dbl_timing;
};

// ---- Execution space -------------------------------------------------------

using exec_space = Kokkos::DefaultExecutionSpace;
using policy_1d  = Kokkos::RangePolicy<exec_space>;
using vffc       = Kokkos::View<Kokkos::Experimental::FloatFloatComplex*,    Kokkos::LayoutRight, exec_space>;
using vff        = Kokkos::View<Kokkos::Experimental::FloatFloat*,       Kokkos::LayoutRight, exec_space>;
using vdc        = Kokkos::View<Kokkos::complex<double>*,  Kokkos::LayoutRight, exec_space>;

namespace ff = Kokkos::Experimental;

ComplexOpResult run_op(Op op, const Config& cfg) {
  const int n = cfg.batch;

  std::vector<double> ha_re(n),ha_im(n),hb_re(n),hb_im(n);
  fill_inputs(op, ha_re.data(),ha_im.data(), hb_re.data(),hb_im.data(), n, cfg.seed);

  vffc qa("qa",n), qb("qb",n), qr("qr",n);
  vdc  da("da",n), db("db",n), dr("dr",n);
  vff  qra("qra",n), qrb("qrb",n);  // for polar: r and theta as FloatFloat

  {
    auto mqa=Kokkos::create_mirror_view(qa), mqb=Kokkos::create_mirror_view(qb);
    auto mda=Kokkos::create_mirror_view(da), mdb=Kokkos::create_mirror_view(db);
    auto mqra=Kokkos::create_mirror_view(qra), mqrb=Kokkos::create_mirror_view(qrb);
    for (int i=0;i<n;++i) {
      // Use FloatFloat(double) so the FF input faithfully encodes the FP64 value
      // to ~14 digits (Route A split). Casting through (float) first would
      // populate only hi and bound accuracy at ~7.2 digits.
      mqa(i)=ff::FloatFloatComplex(ff::FloatFloat(ha_re[i]), ff::FloatFloat(ha_im[i]));
      mqb(i)=ff::FloatFloatComplex(ff::FloatFloat(hb_re[i]), ff::FloatFloat(hb_im[i]));
      mda(i)=Kokkos::complex<double>(ha_re[i],ha_im[i]);
      mdb(i)=Kokkos::complex<double>(hb_re[i],hb_im[i]);
      mqra(i)=ff::FloatFloat(ha_re[i]);   // polar r
      mqrb(i)=ff::FloatFloat(ha_im[i]);   // polar theta
    }
    Kokkos::deep_copy(qa,mqa); Kokkos::deep_copy(qb,mqb);
    Kokkos::deep_copy(da,mda); Kokkos::deep_copy(db,mdb);
    Kokkos::deep_copy(qra,mqra); Kokkos::deep_copy(qrb,mqrb);
  }

  policy_1d pol(0,n);
  TimeStats st_ff, st_dbl;

  // ---- FF kernels -------------------------------------------------------
  switch (op) {
    case Op::Add:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_add",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)+qb(i);});}); break;
    case Op::Sub:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_sub",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)-qb(i);});}); break;
    case Op::Mul:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_mul",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)*qb(i);});}); break;
    case Op::Div:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_div",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)/qb(i);});}); break;
    case Op::Abs:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_abs",pol,KOKKOS_LAMBDA(int i){
        qr(i)=ff::FloatFloatComplex(ff::abs(qa(i)), ff::FloatFloat(0.0f));});}); break;
    case Op::Conj:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_conj",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::conj(qa(i));});}); break;
    case Op::Sqrt:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_sqrt",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::sqrt(qa(i));});}); break;
    case Op::Exp:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_exp",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::exp(qa(i));});}); break;
    case Op::Log:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_log",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::log(qa(i));});}); break;
    case Op::Log10:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_log10",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::log10(qa(i));});}); break;
    case Op::Sin:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_sin",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::sin(qa(i));});}); break;
    case Op::Cos:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_cos",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::cos(qa(i));});}); break;
    case Op::Tan:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_tan",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::tan(qa(i));});}); break;
    case Op::Asin:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_asin",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::asin(qa(i));});}); break;
    case Op::Acos:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_acos",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::acos(qa(i));});}); break;
    case Op::Atan:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_atan",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::atan(qa(i));});}); break;
    case Op::Sinh:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_sinh",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::sinh(qa(i));});}); break;
    case Op::Cosh:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_cosh",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::cosh(qa(i));});}); break;
    case Op::Tanh:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_tanh",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::tanh(qa(i));});}); break;
    case Op::Asinh:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_asinh",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::asinh(qa(i));});}); break;
    case Op::Acosh:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_acosh",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::acosh(qa(i));});}); break;
    case Op::Atanh:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_atanh",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::atanh(qa(i));});}); break;
    case Op::Pow:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_pow",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::pow(qa(i),qb(i));});}); break;
    case Op::Polar:
      st_ff=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("fc_polar",pol,KOKKOS_LAMBDA(int i){qr(i)=ff::polar(qra(i),qrb(i));});}); break;
  }

  // ---- double complex kernels ----------------------------------------------
  switch (op) {
    case Op::Add:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_add",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)+db(i);});}); break;
    case Op::Sub:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_sub",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)-db(i);});}); break;
    case Op::Mul:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_mul",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)*db(i);});}); break;
    case Op::Div:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_div",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)/db(i);});}); break;
    case Op::Abs:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_abs",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::complex<double>(Kokkos::abs(da(i)),0.0);});}); break;
    case Op::Conj:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_conj",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::conj(da(i));});}); break;
    case Op::Sqrt:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_sqrt",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sqrt(da(i));});}); break;
    case Op::Exp:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_exp",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::exp(da(i));});}); break;
    case Op::Log:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_log",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::log(da(i));});}); break;
    case Op::Log10:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_log10",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::log10(da(i));});}); break;
    case Op::Sin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_sin",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sin(da(i));});}); break;
    case Op::Cos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_cos",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::cos(da(i));});}); break;
    case Op::Tan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_tan",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::tan(da(i));});}); break;
    case Op::Asin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_asin",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::asin(da(i));});}); break;
    case Op::Acos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_acos",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::acos(da(i));});}); break;
    case Op::Atan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_atan",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::atan(da(i));});}); break;
    case Op::Sinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_sinh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sinh(da(i));});}); break;
    case Op::Cosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_cosh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::cosh(da(i));});}); break;
    case Op::Tanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_tanh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::tanh(da(i));});}); break;
    case Op::Asinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_asinh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::asinh(da(i));});}); break;
    case Op::Acosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_acosh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::acosh(da(i));});}); break;
    case Op::Atanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_atanh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::atanh(da(i));});}); break;
    case Op::Pow:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_pow",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::pow(da(i),db(i));});}); break;
    case Op::Polar:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("ddc_polar",pol,KOKKOS_LAMBDA(int i){
        dr(i)=Kokkos::complex<double>(da(i).real()*Kokkos::cos(da(i).imag()),
                                      da(i).real()*Kokkos::sin(da(i).imag()));
      });}); break;
  }

  // ---- Download (smoke) ------------------------------------------------------
  // Nothing scores these. The readback stays so the demo still exercises the
  // whole round trip (kernel output -> mirror -> host) for a user-defined type,
  // which is where a broken deep_copy would show up. Outside the timed region.
  auto mqr=Kokkos::create_mirror_view(qr); Kokkos::deep_copy(mqr,qr);
  auto mdr=Kokkos::create_mirror_view(dr); Kokkos::deep_copy(mdr,dr);

  return { op, st_ff, st_dbl };
}

// ---- Table printing --------------------------------------------------------
// One row per op. This used to be two rows, real part and imag part, because
// the two carried separate accuracy figures; they never carried separate
// timings (one kernel computes both), so with the accuracy columns gone the
// split has nothing left to distinguish.

static constexpr int kOpW = 12;
static constexpr int kTW  =  9;
static constexpr int kTSec = 4*kTW + 3;
static constexpr int kBkW  = kTSec;

static std::string dashes(int n) { return std::string((size_t)n,'-'); }

static std::string center(const std::string& s, int w) {
  int pad=w-(int)s.size(), lp=pad/2, rp=pad-lp;
  return std::string((size_t)lp,' ')+s+std::string((size_t)rp,' ');
}

static void print_sep() {
  std::cout
    << '-' << dashes(kOpW) << "-+"
    << dashes(kTSec) << "+" << dashes(kTSec) << "+\n";
}

static void print_header(const char* lbl1, const char* lbl2) {
  using std::cout;
  cout << ' ' << std::string(kOpW,' ') << " |"
       << center(lbl1, kBkW) << "|"
       << center(lbl2, kBkW) << "|\n";
  cout << ' ' << std::string(kOpW,' ') << " |"
       << center("Time (ms)", kTSec) << "|"
       << center("Time (ms)", kTSec) << "|\n";
  print_sep();
  cout << ' ' << std::left << std::setw(kOpW) << "" << " |"
       << center("Min",kTW) << "|" << center("Max",kTW) << "|"
       << center("Med",kTW) << "|" << center("Mean",kTW) << "|"
       << center("Min",kTW) << "|" << center("Max",kTW) << "|"
       << center("Med",kTW) << "|" << center("Mean",kTW) << "|\n";
  print_sep();
}

static void print_complex_op_rows(const ComplexOpResult& r) {
  using std::cout; using std::setw; using std::right; using std::fixed; using std::setprecision;
  auto T=[](double s){return s*1000.0;};
  const TimeStats& t1 = r.ff_timing;
  const TimeStats& t2 = r.dbl_timing;
  cout << ' ' << std::left << std::setw(kOpW) << op_name(r.op) << " |"
       << right << fixed << setprecision(4)
       << setw(kTW)<<T(t1.min_s)   <<"|"<< setw(kTW)<<T(t1.max_s)    <<"|"
       << setw(kTW)<<T(t1.median_s)<<"|"<< setw(kTW)<<T(t1.mean_s)   <<"|"
       << setw(kTW)<<T(t2.min_s)   <<"|"<< setw(kTW)<<T(t2.max_s)    <<"|"
       << setw(kTW)<<T(t2.median_s)<<"|"<< setw(kTW)<<T(t2.mean_s)   <<"|\n";
  print_sep();
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  cfg.all_ops = true;
  for (int i=1;i<argc;++i)
    if (std::string(argv[i])=="--op") { cfg.all_ops=false; break; }

  if (!parse_args(argc, argv, cfg)) { print_usage(argv[0]); return 1; }

  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nbatch=" << cfg.batch << "  repeats=" << cfg.repeats
              << "  seed=" << cfg.seed << "  warmup=" << kWarmupRuns
              << "  timing=kernel+fence\n\n";
    print_header("Kokkos FF (float-float)", "CUDA FP64");
    if (cfg.all_ops) {
      for (Op op : kAllOps) {
        print_complex_op_rows(run_op(op, cfg));
      }
    } else {
      print_complex_op_rows(run_op(cfg.op, cfg));
    }
    std::cout << "\n";
  }
  Kokkos::finalize();
  return 0;
}
