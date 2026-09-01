// Kokkos triple-float demo — complex ops.
// Benchmarks Kokkos::Experimental::TripleFloatComplex against the host
// __complex128 oracle (Kokkos-wrapped quadmath, T0.0/T0.3).
//
// Adapted from src/demo_qf_complex.cpp (T3.0c) for the TF backend (S10 P5):
// QuadFloatComplex -> TripleFloatComplex, ~29-digit -> ~21.7-digit accuracy cap, and
// a per-op pass/fail verdict (RC 0 on all-pass, RC 1 on any non-conditioning-
// limited miss) mirroring demo_tf_real.cpp.
//
// ============================================================
// TF complex usage reference  (namespace tf = Kokkos::Experimental)
// ============================================================
//
// Construction
//   tf::TripleFloatComplex z;                  // zero
//   tf::TripleFloatComplex z(1.5f);            // real from float
//   tf::TripleFloatComplex z(x);               // real from TripleFloat
//   tf::TripleFloatComplex z(1.0f, 2.0f);      // from two floats (re, im)
//   tf::TripleFloatComplex z(x, y);            // from two TripleFloats (re, im)
//
// Arithmetic operators
//   z + w,  z - w,  z * w,  z / w    // TripleFloatComplex op TripleFloatComplex
//   -z                                // unary negation
//   z += w, z -= w, z *= w, z /= w
//
// Complex math functions  (all KOKKOS_INLINE_FUNCTION, host + device)
//   tf::abs(z)   -> TripleFloat        tf::norm(z) -> TripleFloat (|z|²)
//   tf::arg(z)   -> TripleFloat        tf::conj(z) -> TripleFloatComplex
//   tf::sqrt(z)
//   tf::exp(z),   tf::log(z),   tf::log10(z)
//   tf::sin(z),   tf::cos(z),   tf::tan(z)
//   tf::asin(z),  tf::acos(z),  tf::atan(z)
//   tf::sinh(z),  tf::cosh(z),  tf::tanh(z)
//   tf::asinh(z), tf::acosh(z), tf::atanh(z)
//   tf::pow(z, w)                     // TripleFloatComplex exponent
//   tf::polar(r, theta)               // r*exp(i*theta), r and theta are TripleFloat
//
// Constants
//   tf::TripleFloat_pi()  tf::TripleFloat_e()  tf::TripleFloat_log2()
//   tf::TripleFloat_log10()  tf::TripleFloat_sqrt2()  tf::TripleFloat_euler_gamma()
// ============================================================

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

// Host __float128 oracle via Kokkos's quadmath overloads (namespace Kokkos).
#include <impl/Kokkos_QuadPrecisionMath.hpp>
// Complex __complex128 oracle overloads (namespace Kokkos), local repo extension
// (patches/kokkos_complex_quad_math.hpp — see patches/README.md). Same plumbing
// as the FF/QF complex demos.
#include <impl/Kokkos_ComplexQuadPrecisionMath.hpp>

#include <tf_math.hpp>
#include <tf_complex.hpp>

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
constexpr double   kMaxDigits      = 21.7; // TF max ~21.68 decimal digits (3×24-bit FP32 mantissa = ~72 bits, u=2^-72)
// Per-op pass gate for the RC-0/RC-1 verdict (mean digits vs __complex128
// oracle, real+imag pooled). Set below TF's ~21.7 ceiling to leave room for
// argument-reduction / conditioning losses, matching demo_tf_real.cpp.
constexpr double   kPassMeanDigits = 19.0;

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

// Complex ops whose mean accuracy is condition-number-bounded (not TF-quality-
// bounded), so a below-gate mean is expected, not a failure. Grounded in
// PORT_NOTES.md §5 (the FP32 conditioning list) carried to the complex layer:
//   * sub: random near-cancellation of z-w, one digit lost per matched digit.
//   * div: relative error amplified when |w| is small (fill uses |w| down to ~0.1).
//   * tan: sin(z)/cos(z) blows up near cos(z)=0 (odd multiples of pi/2 in re).
//   * asin/acos: derivative 1/sqrt(1-z^2) -> inf near the ±1 branch points.
//   * atan/atanh: 1/(1±z²) blow-up near the ±i / ±1 branch points.
//   * pow: exp(w·log z) — log's arg + w-multiply compound the conditioning.
//   * log/log10: |log z| relative error unbounded as |z| -> 1 (real part -> 0).
bool is_conditioning_limited(Op op) {
  switch (op) {
    case Op::Sub: case Op::Div: case Op::Tan:
    case Op::Asin: case Op::Acos: case Op::Atan: case Op::Atanh:
    case Op::Pow: case Op::Log: case Op::Log10:
      return true;
    default:
      return false;
  }
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

// ---- Input generation (identical ranges to demo_qf_complex.cpp) -------------

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

// ---- Host quadmath reference ------------------------------------------------

void host_quadmath_reference(Op op,
                             const double* ha_re, const double* ha_im,
                             const double* hb_re, const double* hb_im,
                             __float128* out_re, __float128* out_im, int n) {
  for (int i = 0; i < n; ++i) {
    __complex128 za; __real__ za=(__float128)ha_re[i]; __imag__ za=(__float128)ha_im[i];
    __complex128 zb; __real__ zb=(__float128)hb_re[i]; __imag__ zb=(__float128)hb_im[i];
    __complex128 res = 0.0q;
    switch (op) {
      case Op::Add:   res = za + zb;       break;
      case Op::Sub:   res = za - zb;       break;
      case Op::Mul:   res = za * zb;       break;
      case Op::Div:   res = za / zb;       break;
      case Op::Abs:   out_re[i]=Kokkos::abs(za); out_im[i]=0.0q; continue;
      case Op::Conj:  res = Kokkos::conj(za);     break;
      case Op::Sqrt:  res = Kokkos::sqrt(za);    break;
      case Op::Exp:   res = Kokkos::exp(za);     break;
      case Op::Log:   res = Kokkos::log(za);     break;
      case Op::Log10: res = Kokkos::log10(za);   break;
      case Op::Sin:   res = Kokkos::sin(za);     break;
      case Op::Cos:   res = Kokkos::cos(za);     break;
      case Op::Tan:   res = Kokkos::tan(za);     break;
      case Op::Asin:  res = Kokkos::asin(za);    break;
      case Op::Acos:  res = Kokkos::acos(za);    break;
      case Op::Atan:  res = Kokkos::atan(za);    break;
      case Op::Sinh:  res = Kokkos::sinh(za);    break;
      case Op::Cosh:  res = Kokkos::cosh(za);    break;
      case Op::Tanh:  res = Kokkos::tanh(za);    break;
      case Op::Asinh: res = Kokkos::asinh(za);   break;
      case Op::Acosh: res = Kokkos::acosh(za);   break;
      case Op::Atanh: res = Kokkos::atanh(za);   break;
      case Op::Pow:   res = Kokkos::pow(za,zb);  break;
      case Op::Polar: {
        __float128 r=(__float128)ha_re[i], th=(__float128)ha_im[i];
        out_re[i]=r*Kokkos::cos(th); out_im[i]=r*Kokkos::sin(th); continue;
      }
    }
    out_re[i]=Kokkos::real(res); out_im[i]=Kokkos::imag(res);
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

// ---- Accuracy --------------------------------------------------------------

struct AccStats { double min_d=kMaxDigits, max_d=0, mean_d=0, median_d=0; };

static double element_digits(__float128 dev, __float128 ref) {
  if (Kokkos::isnan(dev)||Kokkos::isnan(ref)) return 0.0;
  if (Kokkos::isinf(ref)) return (Kokkos::isinf(dev)&&(dev>0)==(ref>0))?kMaxDigits:0.0;
  if (ref==(__float128)0.0) return (dev==(__float128)0.0)?kMaxDigits:0.0;
  __float128 rel=Kokkos::abs((dev-ref)/ref);
  if (rel==(__float128)0.0) return kMaxDigits;
  double d=-(double)Kokkos::log10(rel);
  return d<0.0?0.0:(d>kMaxDigits?kMaxDigits:d);
}

static __float128 tf_to_q(Kokkos::Experimental::TripleFloat x) {
  return (__float128)x.f0 + (__float128)x.f1 + (__float128)x.f2;
}

AccStats compute_acc_tf(const Kokkos::Experimental::TripleFloat* dev, const __float128* ref, int n) {
  std::vector<double> digs((size_t)n);
  for (int i=0;i<n;++i) digs[i]=element_digits(tf_to_q(dev[i]),ref[i]);
  std::sort(digs.begin(),digs.end());
  AccStats s;
  s.min_d=digs.front(); s.max_d=digs.back();
  s.mean_d=std::accumulate(digs.begin(),digs.end(),0.0)/(double)n;
  size_t m=digs.size();
  s.median_d=(m%2==1)?digs[m/2]:0.5*(digs[m/2-1]+digs[m/2]);
  return s;
}

AccStats compute_acc_dbl(const double* dev, const __float128* ref, int n) {
  std::vector<double> digs((size_t)n);
  for (int i=0;i<n;++i) digs[i]=element_digits((__float128)dev[i],ref[i]);
  std::sort(digs.begin(),digs.end());
  AccStats s;
  s.min_d=digs.front(); s.max_d=digs.back();
  s.mean_d=std::accumulate(digs.begin(),digs.end(),0.0)/(double)n;
  size_t m=digs.size();
  s.median_d=(m%2==1)?digs[m/2]:0.5*(digs[m/2-1]+digs[m/2]);
  return s;
}

// ---- Per-op result ---------------------------------------------------------

struct ComplexOpResult {
  Op        op;
  TimeStats tf_timing, dbl_timing;
  AccStats  tf_re, tf_im;
  AccStats  dbl_re, dbl_im;
};

// ---- Execution space -------------------------------------------------------

using exec_space = Kokkos::DefaultExecutionSpace;
using policy_1d  = Kokkos::RangePolicy<exec_space>;
using vtfc       = Kokkos::View<Kokkos::Experimental::TripleFloatComplex*, Kokkos::LayoutRight, exec_space>;
using vtf        = Kokkos::View<Kokkos::Experimental::TripleFloat*,        Kokkos::LayoutRight, exec_space>;
using vdc        = Kokkos::View<Kokkos::complex<double>*,                Kokkos::LayoutRight, exec_space>;

namespace tf = Kokkos::Experimental;

ComplexOpResult run_op(Op op, const Config& cfg) {
  const int n = cfg.batch;

  std::vector<double> ha_re(n),ha_im(n),hb_re(n),hb_im(n);
  fill_inputs(op, ha_re.data(),ha_im.data(), hb_re.data(),hb_im.data(), n, cfg.seed);

  std::vector<__float128> href_re(n), href_im(n);
  host_quadmath_reference(op, ha_re.data(),ha_im.data(), hb_re.data(),hb_im.data(),
                          href_re.data(), href_im.data(), n);

  vtfc qa("qa",n), qb("qb",n), qr("qr",n);
  vdc  da("da",n), db("db",n), dr("dr",n);
  vtf  qra("qra",n), qrb("qrb",n);  // for polar: r and theta as TripleFloat

  {
    auto mqa=Kokkos::create_mirror_view(qa), mqb=Kokkos::create_mirror_view(qb);
    auto mda=Kokkos::create_mirror_view(da), mdb=Kokkos::create_mirror_view(db);
    auto mqra=Kokkos::create_mirror_view(qra), mqrb=Kokkos::create_mirror_view(qrb);
    for (int i=0;i<n;++i) {
      // Use TripleFloat(double) so the TF input faithfully encodes the FP64 value
      // (Route-A successive split into f0..f2). A double carries only 53 bits, so
      // f0,f1 hold it exactly — the input is exact in TF, and accuracy is bounded
      // only by the TF complex math, not the input encoding.
      mqa(i)=tf::TripleFloatComplex(tf::TripleFloat(ha_re[i]), tf::TripleFloat(ha_im[i]));
      mqb(i)=tf::TripleFloatComplex(tf::TripleFloat(hb_re[i]), tf::TripleFloat(hb_im[i]));
      mda(i)=Kokkos::complex<double>(ha_re[i],ha_im[i]);
      mdb(i)=Kokkos::complex<double>(hb_re[i],hb_im[i]);
      mqra(i)=tf::TripleFloat(ha_re[i]);   // polar r
      mqrb(i)=tf::TripleFloat(ha_im[i]);   // polar theta
    }
    Kokkos::deep_copy(qa,mqa); Kokkos::deep_copy(qb,mqb);
    Kokkos::deep_copy(da,mda); Kokkos::deep_copy(db,mdb);
    Kokkos::deep_copy(qra,mqra); Kokkos::deep_copy(qrb,mqrb);
  }

  policy_1d pol(0,n);
  TimeStats st_tf, st_dbl;

  // ---- TF complex kernels ---------------------------------------------------
  switch (op) {
    case Op::Add:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_add",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)+qb(i);});}); break;
    case Op::Sub:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_sub",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)-qb(i);});}); break;
    case Op::Mul:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_mul",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)*qb(i);});}); break;
    case Op::Div:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_div",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)/qb(i);});}); break;
    case Op::Abs:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_abs",pol,KOKKOS_LAMBDA(int i){
        qr(i)=tf::TripleFloatComplex(tf::abs(qa(i)), tf::TripleFloat(0.0f));});}); break;
    case Op::Conj:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_conj",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::conj(qa(i));});}); break;
    case Op::Sqrt:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_sqrt",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::sqrt(qa(i));});}); break;
    case Op::Exp:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_exp",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::exp(qa(i));});}); break;
    case Op::Log:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_log",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::log(qa(i));});}); break;
    case Op::Log10:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_log10",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::log10(qa(i));});}); break;
    case Op::Sin:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_sin",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::sin(qa(i));});}); break;
    case Op::Cos:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_cos",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::cos(qa(i));});}); break;
    case Op::Tan:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_tan",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::tan(qa(i));});}); break;
    case Op::Asin:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_asin",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::asin(qa(i));});}); break;
    case Op::Acos:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_acos",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::acos(qa(i));});}); break;
    case Op::Atan:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_atan",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::atan(qa(i));});}); break;
    case Op::Sinh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_sinh",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::sinh(qa(i));});}); break;
    case Op::Cosh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_cosh",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::cosh(qa(i));});}); break;
    case Op::Tanh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_tanh",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::tanh(qa(i));});}); break;
    case Op::Asinh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_asinh",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::asinh(qa(i));});}); break;
    case Op::Acosh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_acosh",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::acosh(qa(i));});}); break;
    case Op::Atanh:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_atanh",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::atanh(qa(i));});}); break;
    case Op::Pow:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_pow",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::pow(qa(i),qb(i));});}); break;
    case Op::Polar:
      st_tf=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("tc_polar",pol,KOKKOS_LAMBDA(int i){qr(i)=tf::polar(qra(i),qrb(i));});}); break;
  }

  // ---- double complex kernels ----------------------------------------------
  switch (op) {
    case Op::Add:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_add",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)+db(i);});}); break;
    case Op::Sub:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sub",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)-db(i);});}); break;
    case Op::Mul:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_mul",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)*db(i);});}); break;
    case Op::Div:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_div",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)/db(i);});}); break;
    case Op::Abs:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_abs",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::complex<double>(Kokkos::abs(da(i)),0.0);});}); break;
    case Op::Conj:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_conj",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::conj(da(i));});}); break;
    case Op::Sqrt:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sqrt",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sqrt(da(i));});}); break;
    case Op::Exp:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_exp",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::exp(da(i));});}); break;
    case Op::Log:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_log",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::log(da(i));});}); break;
    case Op::Log10:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_log10",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::log10(da(i));});}); break;
    case Op::Sin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sin",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sin(da(i));});}); break;
    case Op::Cos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_cos",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::cos(da(i));});}); break;
    case Op::Tan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_tan",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::tan(da(i));});}); break;
    case Op::Asin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_asin",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::asin(da(i));});}); break;
    case Op::Acos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_acos",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::acos(da(i));});}); break;
    case Op::Atan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_atan",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::atan(da(i));});}); break;
    case Op::Sinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sinh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sinh(da(i));});}); break;
    case Op::Cosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_cosh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::cosh(da(i));});}); break;
    case Op::Tanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_tanh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::tanh(da(i));});}); break;
    case Op::Asinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_asinh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::asinh(da(i));});}); break;
    case Op::Acosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_acosh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::acosh(da(i));});}); break;
    case Op::Atanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_atanh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::atanh(da(i));});}); break;
    case Op::Pow:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_pow",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::pow(da(i),db(i));});}); break;
    case Op::Polar:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_polar",pol,KOKKOS_LAMBDA(int i){
        dr(i)=Kokkos::complex<double>(da(i).real()*Kokkos::cos(da(i).imag()),
                                      da(i).real()*Kokkos::sin(da(i).imag()));
      });}); break;
  }

  // ---- Download and compute accuracy ---------------------------------------
  auto mqr=Kokkos::create_mirror_view(qr); Kokkos::deep_copy(mqr,qr);
  auto mdr=Kokkos::create_mirror_view(dr); Kokkos::deep_copy(mdr,dr);

  std::vector<tf::TripleFloat> qr_re(n), qr_im(n);
  std::vector<double>        dr_re(n), dr_im(n);
  for (int i=0;i<n;++i) {
    qr_re[i]=mqr(i).re; qr_im[i]=mqr(i).im;
    dr_re[i]=mdr(i).real(); dr_im[i]=mdr(i).imag();
  }

  return { op, st_tf, st_dbl,
           compute_acc_tf(qr_re.data(), href_re.data(), n),
           compute_acc_tf(qr_im.data(), href_im.data(), n),
           compute_acc_dbl(dr_re.data(), href_re.data(), n),
           compute_acc_dbl(dr_im.data(), href_im.data(), n) };
}

// ---- Table printing --------------------------------------------------------

static constexpr int kOpW = 12;
static constexpr int kTW  =  9;
static constexpr int kAW  =  7;
static constexpr int kTSec = 4*kTW + 3;
static constexpr int kASec = 4*kAW + 3;
static constexpr int kBkW  = kTSec + 1 + kASec;

static std::string dashes(int n) { return std::string((size_t)n,'-'); }

static std::string center(const std::string& s, int w) {
  int pad=w-(int)s.size(), lp=pad/2, rp=pad-lp;
  return std::string((size_t)lp,' ')+s+std::string((size_t)rp,' ');
}

static void print_sep() {
  std::cout
    << '-' << dashes(kOpW) << "-+"
    << dashes(kTSec) << "+" << dashes(kASec) << "+"
    << dashes(kTSec) << "+" << dashes(kASec) << "+\n";
}

static void print_header(const char* lbl1, const char* lbl2, const char* acc_label) {
  using std::cout;
  cout << ' ' << std::string(kOpW,' ') << " |"
       << center(lbl1, kBkW) << "|"
       << center(lbl2, kBkW) << "|\n";
  cout << ' ' << std::string(kOpW,' ') << " |"
       << center("Time (ms)", kTSec) << "|" << center(acc_label, kASec) << "|"
       << center("Time (ms)", kTSec) << "|" << center(acc_label, kASec) << "|\n";
  print_sep();
  cout << ' ' << std::left << std::setw(kOpW) << "" << " |"
       << center("Min",kTW) << "|" << center("Max",kTW) << "|"
       << center("Med",kTW) << "|" << center("Mean",kTW) << "|"
       << center("Min",kAW) << "|" << center("Max",kAW) << "|"
       << center("Med",kAW) << "|" << center("Mean",kAW) << "|"
       << center("Min",kTW) << "|" << center("Max",kTW) << "|"
       << center("Med",kTW) << "|" << center("Mean",kTW) << "|"
       << center("Min",kAW) << "|" << center("Max",kAW) << "|"
       << center("Med",kAW) << "|" << center("Mean",kAW) << "|\n";
  print_sep();
}

static void print_row(const char* name,
                      const TimeStats& t1, const AccStats& a1,
                      const TimeStats& t2, const AccStats& a2) {
  using std::cout; using std::setw; using std::right; using std::fixed; using std::setprecision;
  auto T=[](double s){return s*1000.0;};
  cout << ' ' << std::left << std::setw(kOpW) << name << " |"
       << right << fixed << setprecision(4)
       << setw(kTW)<<T(t1.min_s)   <<"|"<< setw(kTW)<<T(t1.max_s)    <<"|"
       << setw(kTW)<<T(t1.median_s)<<"|"<< setw(kTW)<<T(t1.mean_s)   <<"|"
       << setprecision(2)
       << setw(kAW)<<a1.min_d      <<"|"<< setw(kAW)<<a1.max_d       <<"|"
       << setw(kAW)<<a1.median_d   <<"|"<< setw(kAW)<<a1.mean_d      <<"|"
       << setprecision(4)
       << setw(kTW)<<T(t2.min_s)   <<"|"<< setw(kTW)<<T(t2.max_s)    <<"|"
       << setw(kTW)<<T(t2.median_s)<<"|"<< setw(kTW)<<T(t2.mean_s)   <<"|"
       << setprecision(2)
       << setw(kAW)<<a2.min_d      <<"|"<< setw(kAW)<<a2.max_d       <<"|"
       << setw(kAW)<<a2.median_d   <<"|"<< setw(kAW)<<a2.mean_d      <<"|\n";
}

static void print_inner_sep() {
  std::cout << ' ' << std::string(kOpW, ' ') << " +"
            << dashes(kTSec) << "+" << dashes(kASec) << "+"
            << dashes(kTSec) << "+" << dashes(kASec) << "+\n";
}

static void print_complex_op_rows(const ComplexOpResult& r) {
  std::string re_n = std::string(op_name(r.op)) + " (real)";
  std::string im_n = std::string(op_name(r.op)) + " (imag)";
  print_row(re_n.c_str(), r.tf_timing, r.tf_re, r.dbl_timing, r.dbl_re);
  print_inner_sep();
  print_row(im_n.c_str(), r.tf_timing, r.tf_im, r.dbl_timing, r.dbl_im);
  print_sep();
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  cfg.all_ops = true;
  for (int i=1;i<argc;++i)
    if (std::string(argv[i])=="--op") { cfg.all_ops=false; break; }

  if (!parse_args(argc, argv, cfg)) { print_usage(argv[0]); return 1; }

  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nbatch=" << cfg.batch << "  repeats=" << cfg.repeats
              << "  seed=" << cfg.seed << "  warmup=" << kWarmupRuns
              << "  timing=kernel+fence\n\n";
    print_header("Kokkos TF (triple-float)", "CUDA FP64", "Accuracy (dig)");

    std::vector<ComplexOpResult> results;
    if (cfg.all_ops) {
      for (Op op : kAllOps) results.push_back(run_op(op, cfg));
    } else {
      results.push_back(run_op(cfg.op, cfg));
    }
    for (const ComplexOpResult& r : results) print_complex_op_rows(r);
    std::cout << "\n";

    // ---- Pass/fail verdict (S10 P5 deliverable) --------------------------------
    // Each op must reach kPassMeanDigits mean accuracy vs the __complex128 oracle
    // in BOTH the real and imag components, unless it is conditioning-limited
    // (is_conditioning_limited). The pooled mean (real+imag averaged) is the
    // reported figure; any non-conditioning shortfall sets RC 1.
    std::cout << "Accuracy verdict (gate: mean >= " << std::fixed << std::setprecision(1)
              << kPassMeanDigits << " digits vs __complex128, real & imag; "
              << "conditioning-limited ops exempt)\n";
    int n_pass = 0, n_cond = 0, n_fail = 0;
    for (const ComplexOpResult& r : results) {
      bool cond = is_conditioning_limited(r.op);
      double mean_pooled = 0.5*(r.tf_re.mean_d + r.tf_im.mean_d);
      double min_pooled  = std::min(r.tf_re.mean_d, r.tf_im.mean_d);
      bool ok = min_pooled >= kPassMeanDigits;
      const char* tag;
      if (ok)        { tag = "PASS";        ++n_pass; }
      else if (cond) { tag = "PASS (cond)"; ++n_cond; }
      else           { tag = "FAIL";        ++n_fail; rc = 1; }
      if (!ok) {
        std::cout << "  " << std::left << std::setw(10) << op_name(r.op)
                  << " mean(re,im)=" << std::right << std::fixed << std::setprecision(2)
                  << std::setw(6) << r.tf_re.mean_d << "," << std::setw(6) << r.tf_im.mean_d
                  << "  pooled=" << std::setw(6) << mean_pooled
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
