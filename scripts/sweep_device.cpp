// ============================================================================
// scripts/sweep_device.cpp — the DEVICE producer for the accuracy sweep
// ============================================================================
// Added by CORE_PLAN section C6 (chunk A).
//
//   sweep_device --out <path> [--grid <path>] [--seed N] [--dump-inputs OP:POINT]
//
// WHAT IT DOES
// Evaluates all four backends (DD, FF, QF, TF) over all 63 operations (39 real
// + 24 complex) at every point of the committed grid, on whatever device
// tests/device_harness.hpp was compiled for — HIP, CUDA, or a serial host loop
// — and writes the RAW RESULT LIMBS out as hexadecimal bit patterns.
//
// WHAT IT DOES NOT DO, AND THIS IS THE POINT
// IT COMPUTES. IT DOES NOT SCORE. There is no ulp arithmetic in this file, no
// tolerance, no bound, no verdict and no oracle. docs/CORRECTNESS.md permits
// EXACTLY ONE SCORER, and that is scripts/sweep_accuracy.cpp. C6's design is
// one scorer with TWO PRODUCERS: this binary produces results on the device,
// the host scorer produces its own and holds the MPFR/MPC reference, and the
// scorer alone decides whether a number is good. If you are about to add a
// comparison here that needs a tolerance, you are about to add a second
// opinion — put it in the scorer instead.
//
// (A bit-exactness check is a different thing and is allowed anywhere: `diff`
// of two limb columns issues no verdict about accuracy. Plan step 7's serial
// self-test is exactly that, and it lives outside this file.)
//
// WHY IT READS THE GRID INSTEAD OF GENERATING IT
// validation/sweep/sweep_grid.csv is the manifest the baseline's `point` column
// indexes. A second generator is a second grid: the two copies would drift the
// moment either was touched, and the drift would present as an accuracy
// finding rather than as the bookkeeping error it is. So the grid is READ. See
// scripts/sweep_inputs.hpp, which also validates it loudly.
//
// --out IS REQUIRED, ON PURPOSE
// scripts/sweep_accuracy.cpp defaulted its --out to the empty string, which
// meant it computed an entire sweep and only then failed to write it; it also
// once overwrote the committed baseline in place. Both hazards are avoided the
// same way: there is NO default output path here, the check happens before any
// computation, and the refusal exits 2 with a message that names the fix.
//
// TWO PLACES THE PLAN WAS WRONG, AND WHAT WAS DONE INSTEAD
//
//  1. The plan puts the binary at <build>/scripts/sweep_device. There is no
//     scripts/ directory in the build tree — the top-level CMakeLists.txt has
//     exactly one add_subdirectory(), `tests`. This target is therefore
//     registered in tests/CMakeLists.txt and lands at <build>/tests/sweep_device,
//     alongside sweep_accuracy, which is built the same way and for the same
//     reason (a target built by the build system cannot be a stale hand-compiled
//     binary sitting in the source tree).
//
//  2. The plan's row is `backend,kind,op,point,limb0..limb3`. Four limb columns
//     cannot hold a QF COMPLEX result: that is four FP32 words of real part
//     plus four of imaginary, eight in all. The schema here is
//     limb0..limb7, with limb0-3 the REAL component's words and limb4-7 the
//     IMAGINARY component's, unused columns empty. Real-realm rows leave
//     limb4-7 empty always. This keeps the plan's rule that the transfer be
//     lossless and unambiguous, and keeps ONE ROW PER (backend, kind, op,
//     point) — which is the baseline's own row identity (docs/CORRECTNESS.md),
//     so the scorer can join on it directly.
//
// HEX, NOT %a AND NOT DECIMAL. Each limb is printed as the raw IEEE-754 bit
// pattern of its word: %016llx for an FP64 limb, %08x for an FP32 one. A NaN
// payload, a signed zero and a subnormal all survive that and only that.
//
// ONE KERNEL PER (BACKEND, OP), BY CONSTRUCTION
// xpsweep::eval_real / eval_complex take the op id as a runtime argument, so
// that their text matches the scorer's copy exactly. This file nevertheless
// instantiates them at a COMPILE-TIME id, which folds the switch to its single
// live case. That is not a micro-optimisation: TD-1
// (docs/ROCM_BRANCH_RELAXATION_BUG.md) is a gfx90a miscompile of an OVERSIZED
// device callee, and a 39-case switch inlined into one kernel is precisely the
// shape that triggers it. 252 small kernels is the safe arrangement; one giant
// one is the unsafe arrangement that happens to be less typing.
//
// THE DOMAIN DIAGNOSTICS ARE COMPILED OUT (XPMATH_ENABLE_DIAGNOSTICS=0, set on
// the target). The sweep evaluates every op at overflow, at poles and outside
// its domain on purpose, so the numeric headers' ~40 one-line diagnostics fire
// in bulk — ~269 KB of stdout on a host run, and on a GPU a device-side printf
// per offending thread, which costs registers everywhere and can be dropped
// under load. It changes no result: the diagnostics-ON and diagnostics-OFF
// builds' 436,080-row outputs were compared with `cmp` and are byte-identical.
//
// NO __float128. This is a device TU and is listed in
// scripts/check_device_tu_purity.sh's FILES array, which is what makes that a
// standing claim rather than an intention.
// ============================================================================

#include "sweep_inputs.hpp"      // host-side input derivation + the grid reader
#include "sweep_ops.hpp"         // the op inventory and the two evaluators

#include "device_harness.hpp"    // xpt::buffer / parallel_for_n / last_error

#include <xp/dd_complex.hpp>
#include <xp/ff_complex.hpp>
#include <xp/qf_complex.hpp>
#include <xp/tf_complex.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

#ifndef XPMATH_SWEEP_GRID_DEFAULT
// Only a fallback for a hand-compile; the CMake target defines the real path.
#define XPMATH_SWEEP_GRID_DEFAULT "validation/sweep/sweep_grid.csv"
#endif

namespace {

// ---------------------------------------------------------------------------
// Limb layout per backend scalar: the word type, how many of them, and how to
// write them out of a value. This is the ONLY place that knows a DoubleDouble
// is {hi, lo} and a QuadFloat is {f0..f3}.
// ---------------------------------------------------------------------------
template <class S> struct Limbs;

template <> struct Limbs<xp::DoubleDouble> {
  using W = double;
  static constexpr int n = 2;
  static XPMATH_INLINE_FUNCTION void store(const xp::DoubleDouble& v, W* o) {
    o[0] = v.hi; o[1] = v.lo;
  }
};
template <> struct Limbs<xp::FloatFloat> {
  using W = float;
  static constexpr int n = 2;
  static XPMATH_INLINE_FUNCTION void store(const xp::FloatFloat& v, W* o) {
    o[0] = v.hi; o[1] = v.lo;
  }
};
template <> struct Limbs<xp::TripleFloat> {
  using W = float;
  static constexpr int n = 3;
  static XPMATH_INLINE_FUNCTION void store(const xp::TripleFloat& v, W* o) {
    o[0] = v.f0; o[1] = v.f1; o[2] = v.f2;
  }
};
template <> struct Limbs<xp::QuadFloat> {
  using W = float;
  static constexpr int n = 4;
  static XPMATH_INLINE_FUNCTION void store(const xp::QuadFloat& v, W* o) {
    o[0] = v.f0; o[1] = v.f1; o[2] = v.f2; o[3] = v.f3;
  }
};

// The four backends, named exactly as the baseline's `backend` column spells
// them. The scorer joins on this string.
struct BeDD { using S = xp::DoubleDouble; using Z = xp::DoubleDoubleComplex;
              static const char* name() { return "DD"; } };
struct BeFF { using S = xp::FloatFloat;   using Z = xp::FloatFloatComplex;
              static const char* name() { return "FF"; } };
struct BeQF { using S = xp::QuadFloat;    using Z = xp::QuadFloatComplex;
              static const char* name() { return "QF"; } };
struct BeTF { using S = xp::TripleFloat;  using Z = xp::TripleFloatComplex;
              static const char* name() { return "TF"; } };

// ---------------------------------------------------------------------------
// The kernels. Trivially copyable structs holding device pointers, which is the
// shape tests/device_harness.hpp requires (it passes the functor BY VALUE into
// the launch and deliberately does not support a __device__ lambda).
//
// Operand construction is `const S sa(a[i])` — the same one-argument conversion
// the host scorer performs, so a backend that cannot hold a double exactly is
// handed exactly the value it would be handed there.
// ---------------------------------------------------------------------------
template <class S, int ID>
struct RealKernel {
  using W = typename Limbs<S>::W;
  const double* a;
  const double* b;
  const double* c;
  W*            out;

  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    const S sa(a[i]), sb(b[i]), sc(c[i]);
    const S r = xpsweep::eval_real<S>(ID, sa, sb, sc);
    Limbs<S>::store(r, out + i * Limbs<S>::n);
  }
};

template <class S, class Z, int ID>
struct ComplexKernel {
  using W = typename Limbs<S>::W;
  const double* are;
  const double* aim;
  const double* bre;
  const double* bim;
  W*            out;

  XPMATH_INLINE_FUNCTION void operator()(std::size_t i) const {
    const Z a{S(are[i]), S(aim[i])};
    const Z b{S(bre[i]), S(bim[i])};
    // eval_complex reports "the result is a real scalar in the real slot" for
    // complex abs. The raw file records what the library returned and the
    // scorer interprets it, so the flag is consumed and dropped here; it exists
    // so this evaluator stays textually identical to the scorer's.
    bool    is_real = false;
    const Z r = xpsweep::eval_complex<S, Z>(ID, a, b, is_real);
    (void)is_real;
    const int n = Limbs<S>::n;
    Limbs<S>::store(r.re, out + i * (2 * n));
    Limbs<S>::store(r.im, out + i * (2 * n) + n);
  }
};

// ---------------------------------------------------------------------------
// Row formatting. Host side only.
// ---------------------------------------------------------------------------
inline void hex_of(char* dst, double w) {
  uint64_t u;
  std::memcpy(&u, &w, sizeof u);
  std::snprintf(dst, 24, "%016llx", (unsigned long long)u);
}
inline void hex_of(char* dst, float w) {
  uint32_t u;
  std::memcpy(&u, &w, sizeof u);
  std::snprintf(dst, 24, "%08x", (unsigned)u);
}

struct Cols {
  char c[8][24];
  Cols() { for (int k = 0; k < 8; ++k) c[k][0] = '\0'; }
};

inline void put_row(std::FILE* f, const char* be, char kind, const char* op,
                    std::size_t point, const Cols& x) {
  std::fprintf(f, "%s,%c,%s,%zu,%s,%s,%s,%s,%s,%s,%s,%s\n", be, kind, op, point,
               x.c[0], x.c[1], x.c[2], x.c[3], x.c[4], x.c[5], x.c[6], x.c[7]);
}

template <class S>
void write_real_rows(std::FILE* f, const char* be, const char* op,
                     std::size_t n, const typename Limbs<S>::W* out) {
  const int L = Limbs<S>::n;
  for (std::size_t i = 0; i < n; ++i) {
    Cols x;
    for (int k = 0; k < L; ++k) hex_of(x.c[k], out[i * L + k]);
    put_row(f, be, 'r', op, i, x);
  }
}

template <class S>
void write_complex_rows(std::FILE* f, const char* be, const char* op,
                        std::size_t n, const typename Limbs<S>::W* out) {
  const int L = Limbs<S>::n;
  for (std::size_t i = 0; i < n; ++i) {
    Cols x;
    for (int k = 0; k < L; ++k) {
      hex_of(x.c[k],     out[i * (2 * L) + k]);        // real component
      hex_of(x.c[4 + k], out[i * (2 * L) + L + k]);    // imaginary component
    }
    put_row(f, be, 'c', op, i, x);
  }
}

// ---------------------------------------------------------------------------
// Driving one op.
// ---------------------------------------------------------------------------
struct RunCtx {
  const xpsweep::Grid* grid;
  uint64_t             seed;
  std::FILE*           out;
  const char*          dump_op;     // --dump-inputs OP:POINT, or nullptr
  int                  dump_point;
  char                 dump_kind;   // 'r' or 'c'
};

template <class B, int ID>
void real_backend(const RunCtx& ctx, xpt::buffer<double>& A,
                  xpt::buffer<double>& Bb, xpt::buffer<double>& C) {
  using S = typename B::S;
  using W = typename Limbs<S>::W;
  const std::size_t n = A.size();
  xpt::buffer<W>    O(n * Limbs<S>::n);
  RealKernel<S, ID> k{A.device(), Bb.device(), C.device(), O.device()};
  xpt::parallel_for_n(n, k);
  O.from_device();
  write_real_rows<S>(ctx.out, B::name(), xpsweep::kReal[ID].name, n, O.host());
}

template <int ID>
void real_op(const RunCtx& ctx) {
  const std::vector<xpsweep::GridPoint>& g = ctx.grid->real;
  const std::size_t                      n = g.size();

  xpt::buffer<double> A(n), Bb(n), C(n);

  // The operand stream, derived exactly as scripts/sweep_accuracy.cpp derives
  // it: one Rng per (op, realm), drawn in point order, fill BEFORE repair, and
  // fill reading the UNREPAIRED a. Every one of those is load-bearing — a
  // single extra or reordered draw silently shifts every later point, and the
  // scorer would then be referencing a different question than was asked.
  xpsweep::Rng rng(xpsweep::stream_seed(ctx.seed, xpsweep::kReal[ID].name, 0u));
  for (std::size_t i = 0; i < n; ++i) {
    double av = g[i].re, bv = 0.0, cv = 0.0;
    xpsweep::fill_real_operands(ID, i, av, rng, bv, cv);
    xpsweep::repair_real(ID, av, bv, cv);
    A.host()[i] = av; Bb.host()[i] = bv; C.host()[i] = cv;
  }
  A.to_device(); Bb.to_device(); C.to_device();

  if (ctx.dump_op && ctx.dump_kind == 'r' && ctx.dump_point >= 0 &&
      std::size_t(ctx.dump_point) < n &&
      std::strcmp(ctx.dump_op, xpsweep::kReal[ID].name) == 0) {
    const std::size_t i = std::size_t(ctx.dump_point);
    std::printf("INPUTS r %s point %d\n", xpsweep::kReal[ID].name, ctx.dump_point);
    std::printf("  a      = %.36g\n", A.host()[i]);
    if (xpsweep::kReal[ID].nops >= 2) std::printf("  b      = %.36g\n", Bb.host()[i]);
    if (xpsweep::kReal[ID].nops >= 3) std::printf("  c      = %.36g\n", C.host()[i]);
  }

  real_backend<BeDD, ID>(ctx, A, Bb, C);
  real_backend<BeFF, ID>(ctx, A, Bb, C);
  real_backend<BeQF, ID>(ctx, A, Bb, C);
  real_backend<BeTF, ID>(ctx, A, Bb, C);
}

template <class B, int ID>
void complex_backend(const RunCtx& ctx, xpt::buffer<double>& Are,
                     xpt::buffer<double>& Aim, xpt::buffer<double>& Bre,
                     xpt::buffer<double>& Bim) {
  using S = typename B::S;
  using Z = typename B::Z;
  using W = typename Limbs<S>::W;
  const std::size_t n = Are.size();
  xpt::buffer<W>    O(n * 2 * Limbs<S>::n);
  ComplexKernel<S, Z, ID> k{Are.device(), Aim.device(), Bre.device(),
                            Bim.device(), O.device()};
  xpt::parallel_for_n(n, k);
  O.from_device();
  write_complex_rows<S>(ctx.out, B::name(), xpsweep::kComplex[ID].name, n, O.host());
}

template <int ID>
void complex_op(const RunCtx& ctx) {
  const std::vector<xpsweep::GridPoint>& g = ctx.grid->complx;
  const std::size_t                      n = g.size();

  xpt::buffer<double> Are(n), Aim(n), Bre(n), Bim(n);

  // Realm tag 1 in stream_seed, against the real realm's 0. The complex grid is
  // NOT repaired: it is placed on branch cuts and poles deliberately, and
  // repairing it would delete the thing being measured.
  xpsweep::Rng rng(xpsweep::stream_seed(ctx.seed, xpsweep::kComplex[ID].name, 1u));
  for (std::size_t i = 0; i < n; ++i) {
    double br = 0.0, bi = 0.0;
    xpsweep::fill_complex_operands(ID, i, g, g[i].re, g[i].im, rng, br, bi);
    Are.host()[i] = g[i].re; Aim.host()[i] = g[i].im;
    Bre.host()[i] = br;      Bim.host()[i] = bi;
  }
  Are.to_device(); Aim.to_device(); Bre.to_device(); Bim.to_device();

  if (ctx.dump_op && ctx.dump_kind == 'c' && ctx.dump_point >= 0 &&
      std::size_t(ctx.dump_point) < n &&
      std::strcmp(ctx.dump_op, xpsweep::kComplex[ID].name) == 0) {
    const std::size_t i = std::size_t(ctx.dump_point);
    std::printf("INPUTS c %s point %d\n", xpsweep::kComplex[ID].name, ctx.dump_point);
    std::printf("  a.re   = %.36g\n", Are.host()[i]);
    std::printf("  a.im   = %.36g\n", Aim.host()[i]);
    if (xpsweep::kComplex[ID].nops >= 2) {
      std::printf("  b.re   = %.36g\n", Bre.host()[i]);
      std::printf("  b.im   = %.36g\n", Bim.host()[i]);
    }
  }

  complex_backend<BeDD, ID>(ctx, Are, Aim, Bre, Bim);
  complex_backend<BeFF, ID>(ctx, Are, Aim, Bre, Bim);
  complex_backend<BeQF, ID>(ctx, Are, Aim, Bre, Bim);
  complex_backend<BeTF, ID>(ctx, Are, Aim, Bre, Bim);
}

// Compile-time unrolls, ascending, so the emitted row order matches the host
// scorer's run_sweep(): op-major, backend-minor, real realm before complex.
// Nothing joins on order — every row carries its full key — but a file whose
// order matches is a file a human can diff.
template <int ID> struct RealAll {
  static void run(const RunCtx& c) { RealAll<ID - 1>::run(c); real_op<ID>(c); }
};
template <> struct RealAll<-1> { static void run(const RunCtx&) {} };

template <int ID> struct ComplexAll {
  static void run(const RunCtx& c) { ComplexAll<ID - 1>::run(c); complex_op<ID>(c); }
};
template <> struct ComplexAll<-1> { static void run(const RunCtx&) {} };

void usage(std::FILE* f) {
  std::fprintf(f,
    "sweep_device -- evaluate every backend x op x grid point on the device and\n"
    "                write the raw result limbs. It does NOT score: the verdict\n"
    "                belongs to scripts/sweep_accuracy.cpp (docs/CORRECTNESS.md).\n"
    "\n"
    "  --out <path>          REQUIRED. Where to write the results CSV.\n"
    "                        Never a path under validation/sweep/.\n"
    "  --grid <path>         grid manifest (default: the committed one)\n"
    "  --seed N              operand seed (default %llu; the baseline's seed)\n"
    "  --dump-inputs [r:|c:]OP:POINT\n"
    "                        print one point's operands; cross-check against\n"
    "                        sweep_accuracy --dump-operands OP:POINT. Most op\n"
    "                        names exist in BOTH realms, so prefix r: or c: to\n"
    "                        say which; bare defaults to real when the name is\n"
    "                        a real op, complex otherwise.\n"
    "  --help                this text\n"
    "\n"
    "Row: backend,kind,op,point,limb0..limb7 -- limb0-3 the real component's\n"
    "words, limb4-7 the imaginary component's, unused columns empty, each limb\n"
    "the raw IEEE-754 bit pattern in hex (%%016llx for FP64, %%08x for FP32).\n",
    (unsigned long long)xpsweep::kDefaultSeed);
}

}  // namespace

int main(int argc, char** argv) {
  std::string out;
  std::string grid = XPMATH_SWEEP_GRID_DEFAULT;
  std::string dump;
  uint64_t    seed = xpsweep::kDefaultSeed;

  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "sweep_device: %s wants a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "--out")             out  = need("--out");
    else if (s == "--grid")       grid = need("--grid");
    else if (s == "--seed")       seed = std::strtoull(need("--seed").c_str(), nullptr, 10);
    else if (s == "--dump-inputs") dump = need("--dump-inputs");
    else if (s == "--help" || s == "-h") { usage(stdout); return 0; }
    else {
      std::fprintf(stderr, "sweep_device: unknown argument '%s'\n", s.c_str());
      usage(stderr);
      return 2;
    }
  }

  // THE REFUSAL, BEFORE ANY WORK. See this file's header: the scorer's --out
  // defaulted to "" and so computed a whole sweep before failing to write it.
  if (out.empty()) {
    std::fprintf(stderr,
      "sweep_device: --out <path> is REQUIRED and was not given.\n"
      "  Refusing to compute anything: this run produces hundreds of thousands\n"
      "  of rows and there is deliberately no default destination. The scorer\n"
      "  learned this the expensive way -- its --out defaulted to the empty\n"
      "  string, so it computed an entire sweep and only then failed to write.\n"
      "  Pass --out /tmp/<something>.csv .\n"
      "  Never pass a path under validation/sweep/ -- that is the committed\n"
      "  record, and this tool would overwrite it in place.\n");
    return 2;
  }

  xpsweep::Grid g;
  std::string   err;
  if (!xpsweep::read_grid(grid, g, err)) {
    std::fprintf(stderr, "sweep_device: %s\n", err.c_str());
    return 4;
  }

  const std::size_t nr = g.real.size();
  const std::size_t nc = g.complx.size();
  const std::size_t expected_rows =
      nr * std::size_t(xpsweep::R_COUNT) * 4u + nc * std::size_t(xpsweep::C_COUNT) * 4u;

  std::FILE* f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "sweep_device: cannot open %s for writing\n", out.c_str());
    return 4;
  }
  // 436k fprintf calls; give stdio something to work with.
  static char iobuf[1 << 20];
  std::setvbuf(f, iobuf, _IOFBF, sizeof iobuf);

  std::fprintf(f, "# xp-device-1\n");
  std::fprintf(f, "# raw device results; see scripts/sweep_device.cpp. NOT SCORED.\n");
  std::fprintf(f, "# where: %s\n", xpt::where_name());
  std::fprintf(f, "# grid: real=%zu complex=%zu  seed=%llu\n", nr, nc,
               (unsigned long long)seed);
  std::fprintf(f, "# grid-file: %s\n", grid.c_str());
  std::fprintf(f, "# backends: DD FF QF TF\n");
  std::fprintf(f, "# rows: %zu   (= %zu real x %d ops x 4 + %zu complex x %d ops x 4)\n",
               expected_rows, nr, int(xpsweep::R_COUNT), nc, int(xpsweep::C_COUNT));
  std::fprintf(f, "backend,kind,op,point,limb0,limb1,limb2,limb3,limb4,limb5,limb6,limb7\n");

  RunCtx ctx;
  ctx.grid       = &g;
  ctx.seed       = seed;
  ctx.out        = f;
  ctx.dump_op    = nullptr;
  ctx.dump_point = -1;
  ctx.dump_kind  = 'r';

  std::string dump_name;
  if (!dump.empty()) {
    // An optional realm prefix. Sixteen of the 24 complex op names are also
    // real op names, so without this the complex operand stream of `mul` or
    // `pow` cannot be asked for at all — and the complex stream is the half
    // with the interesting derivation (the stride-7 self-pairing and pow's
    // magnitude clamp). Bare OP:POINT keeps working and picks the real realm
    // when the name is a real op.
    char forced = '\0';
    if (dump.size() > 2 && dump[1] == ':' && (dump[0] == 'r' || dump[0] == 'c')) {
      forced = dump[0];
      dump   = dump.substr(2);
    }
    const std::size_t colon = dump.rfind(':');
    if (colon == std::string::npos) {
      std::fprintf(stderr, "sweep_device: --dump-inputs wants [r:|c:]OP:POINT\n");
      std::fclose(f);
      return 2;
    }
    dump_name      = dump.substr(0, colon);
    ctx.dump_point = std::atoi(dump.c_str() + colon + 1);
    ctx.dump_op    = dump_name.c_str();
    if (forced) {
      ctx.dump_kind = forced;
    } else {
      bool in_real = false;
      for (int k = 0; k < xpsweep::R_COUNT; ++k)
        if (dump_name == xpsweep::kReal[k].name) in_real = true;
      ctx.dump_kind = in_real ? 'r' : 'c';
    }
  }

  RealAll<xpsweep::R_COUNT - 1>::run(ctx);
  ComplexAll<xpsweep::C_COUNT - 1>::run(ctx);

  std::fclose(f);

  // A launch that never ran must not look like a launch that ran clean. The
  // harness's error slot is STICKY and holds the FIRST nonzero vendor code.
  const int e = xpt::last_error();
  if (e != 0) {
    std::fprintf(stderr,
                 "sweep_device: device error %d on backend '%s' -- the results in\n"
                 "  %s are NOT trustworthy and must not be scored.\n",
                 e, xpt::where_name(), out.c_str());
    return 3;
  }

  std::printf("sweep_device: where=%s  grid real=%zu complex=%zu  seed=%llu\n",
              xpt::where_name(), nr, nc, (unsigned long long)seed);
  std::printf("sweep_device: wrote %zu rows to %s\n", expected_rows, out.c_str());
  return 0;
}
