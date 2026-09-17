// probe_complex_oracle.cpp — audit the COMPLEX ORACLE itself.
//
// WHY THIS EXISTS.  scripts/sweep_accuracy.cpp scores every complex row against
// libquadmath (csinq, cacosq, ...).  That oracle has been the defect before:
// KI-36 found libquadmath's complex divide is itself round-then-cancel, so the
// "error" the sweep attributed to the library was the reference moving.  Before
// any complex-trig fix is designed, the oracle has to be shown to be a decade
// better than the thing it is judging, or the whole target is noise.
//
// WHAT IT MEASURES.  For each complex trig op and each of the 1780 complex grid
// points, it computes the answer twice:
//
//   quad : the sweep's own oracle, libquadmath __complex128 (113-bit words)
//   mpc  : MPC at 400 bits, the same modulus-relative comparison the sweep uses
//
// and reports |quad - mpc| / |mpc| expressed in DD ulps (x 2^106) -- DD being
// the most demanding backend, so a figure below 1 here means the oracle is
// adequate for every backend, and a figure above the residual being chased
// means the residual is the oracle.
//
// WHY MPC AND NOT MPFR.  MPFR has no complex type, which is why the sweep never
// gained a wide complex reference.  MPC 1.1.0 is installed here, it is built on
// MPFR, and -- checked, not assumed, see the note below -- it gets the C99
// Annex G signed-zero cuts right.
//
// THE SIGNED-ZERO TRAP, which killed the previous attempt at a wide complex
// oracle (1,996 false defects concentrated in sqrt/log/acosh).  That attempt
// widened the INPUT by summing the expansion's limbs, and (-0.0) + (+0.0) is
// +0.0 in round-to-nearest, so a z = (-100, -0) arrived at the oracle as
// (-100, +0) and the oracle correctly answered for the conjugate.  That is a
// defect in the widening, NOT in the reference library:
//
//     $ mpc_set_d_d(z, -100.0, -0.0); mpc_sqrt(w, z)
//       sqrt(-100, -0) = (0, -10)          <- correct, C99 Annex G
//
// This probe never widens an expansion.  It reads the grid's doubles straight
// from validation/sweep/sweep_grid.csv via strtod, which round-trips -0 as -0,
// and hands those doubles to both references.  Both therefore see the identical
// input including the sign of every zero, so a disagreement between them cannot
// be a widening artifact.
//
// MODES
//   (default)        every op x every complex grid point; per-op summary
//   --op NAME        restrict to one op
//   --point OP N     dump one point in full: both references, both components
//   --top N          list the N worst points overall
//   --prec BITS      MPC working precision (default 400)
//   --selfcheck      run MPC at 400 and at 800 bits and report the largest
//                    disagreement between them.  This is the probe's own
//                    convergence check: if 400 bits were not enough, the two
//                    would differ at the level being reported.
//   --poison         NEGATIVE CONTROL.  Perturbs the quad answer by exactly one
//                    DD ulp of the modulus and requires the probe to report
//                    1.0 +- 0.01 DD ulps.  A probe that reports ~0 with the
//                    poison applied is not comparing anything, and the run
//                    exits nonzero.  Without this the "oracle is clean" reading
//                    and "the comparison is dead" reading are indistinguishable.
//
// MEASURED, 2026-09-11, MPC 1.1.0 at 400 bits, all 12 complex trig ops x the
// 1780-point complex grid (20,867 scorable pairs after the tan/tanh skip):
//
//   op      npts   >0.1ulp   >1ulp     worst    worst point
//   sin     1748         0       0   0.01725   141
//   cos     1750         0       0   0.02025   1224
//   tan     1730         0       0   0.02441   710
//   asin    1778         0       0   0.02400   11
//   acos    1775         0       0   0.01500   79
//   atan    1772         0       0   0.02328   23
//   sinh    1704         0       0   0.01639   269
//   cosh    1706         0       0   0.02025   584
//   tanh    1682         0       0   0.02995   269
//   asinh   1778         0       0   0.02697   60
//   acosh   1775         0       0   0.01500   79
//   atanh   1769         0       0   0.02019   1574
//
//   --selfcheck  MPC 400 vs 800 bits: 0 DD ulps, exactly. Converged.
//   --poison     1.00 DD ulp injected into 13,947 clean points, read back as
//                1.00 with worst deviation 0.0078. The comparison is live.
//
// CONCLUSION: the complex oracle is clean for trig. Worst discrepancy anywhere
// is 0.030 DD ulps -- 33x below the 1-ulp line and ~100x below the 3-6 ulp
// residual the complex trig cells actually carry. Whatever those cells are
// measuring, it is the LIBRARY. This is the opposite of the KI-36 finding for
// complex divide, and it had to be checked rather than assumed either way.
//
// Build (needs MPC + MPFR; not part of the CMake build):
//   g++ -O2 -std=c++17 -fext-numeric-literals \
//       scripts/probe_complex_oracle.cpp -o /tmp/probe_complex_oracle \
//       -lmpc -lmpfr -lgmp -lquadmath
//   /tmp/probe_complex_oracle --selfcheck
//   /tmp/probe_complex_oracle --poison
//   /tmp/probe_complex_oracle
//   /tmp/probe_complex_oracle --point acos 374

// mpfr_get_float128 is behind this switch; it is what lets the MPC answer come
// back at the oracle's own width instead of being truncated to a double on the
// way out, which would inject ~2^53 DD ulps and hide everything.
#define MPFR_WANT_FLOAT128
#include <mpc.h>
#include <mpfr.h>
#include <quadmath.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* const kGrid = "validation/sweep/sweep_grid.csv";

// DD is the tightest backend: 2 x 53 significand bits.  Every figure this probe
// prints is in units of a DD ulp of the RESULT MODULUS, which is exactly the
// unit sweep_accuracy.cpp uses for its complex rows.
const int kDDSigBits = 106;

struct Op {
  const char* name;
  __complex128 (*q)(__complex128);
  int (*m)(mpc_ptr, mpc_srcptr, mpc_rnd_t);
};

const Op kOps[] = {
    {"sin", csinq, mpc_sin},       {"cos", ccosq, mpc_cos},
    {"tan", ctanq, mpc_tan},       {"asin", casinq, mpc_asin},
    {"acos", cacosq, mpc_acos},    {"atan", catanq, mpc_atan},
    {"sinh", csinhq, mpc_sinh},    {"cosh", ccoshq, mpc_cosh},
    {"tanh", ctanhq, mpc_tanh},    {"asinh", casinhq, mpc_asinh},
    {"acosh", cacoshq, mpc_acosh}, {"atanh", catanhq, mpc_atanh},
};
const int kNOps = (int)(sizeof kOps / sizeof kOps[0]);

struct Pt {
  int    idx;
  double re, im;
  std::string family;
};

// The complex half of the shared grid manifest, in `point` order.
std::vector<Pt> load_grid() {
  std::vector<Pt> v;
  FILE* f = std::fopen(kGrid, "r");
  if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", kGrid); std::exit(2); }
  char line[512];
  while (std::fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == 'k') continue;
    char* p = line;
    if (*p != 'c') continue;
    // kind,point,family,re,im
    char* fields[5]; int nf = 0;
    fields[nf++] = p;
    for (char* q = p; *q && nf < 5; ++q) if (*q == ',') { *q = 0; fields[nf++] = q + 1; }
    if (nf < 5) continue;
    Pt pt;
    pt.idx    = std::atoi(fields[1]);
    pt.family = fields[2];
    pt.re     = std::strtod(fields[3], nullptr);   // round-trips -0 as -0
    pt.im     = std::strtod(fields[4], nullptr);
    v.push_back(pt);
  }
  std::fclose(f);
  return v;
}

// |a - b| / |b|, scaled to DD ulps.  Identical in shape to the sweep's complex
// metric (modulus-relative), so a figure here is directly comparable to a
// sweep `ulps` column.
double ulps_dd(__float128 are, __float128 aim, __float128 bre, __float128 bim) {
  const __float128 mb = hypotq(bre, bim);
  if (mb == 0 || !finiteq(mb)) return -1.0;
  const __float128 e = hypotq(are - bre, aim - bim);
  if (!finiteq(e)) return -1.0;
  return (double)ldexpq(e / mb, kDDSigBits);
}

// MPC reference at `prec` bits.  The doubles go in exactly (mpc_set_d_d is
// exact for any double and carries the sign of a zero), so the only error in
// the answer is MPC's own, at 2^-prec.
void ref_mpc(const Op& op, double re, double im, mpfr_prec_t prec,
             __float128& out_re, __float128& out_im) {
  mpc_t z, w;
  mpc_init2(z, prec); mpc_init2(w, prec);
  mpc_set_d_d(z, re, im, MPC_RNDNN);
  op.m(w, z, MPC_RNDNN);
  out_re = mpfr_get_float128(mpc_realref(w), MPFR_RNDN);
  out_im = mpfr_get_float128(mpc_imagref(w), MPFR_RNDN);
  mpc_clear(z); mpc_clear(w);
}

void ref_quad(const Op& op, double re, double im,
              __float128& out_re, __float128& out_im) {
  __complex128 z;
  __real__ z = (__float128)re;
  __imag__ z = (__float128)im;
  const __complex128 r = op.q(z);
  out_re = crealq(r);
  out_im = cimagq(r);
}

void show(const char* tag, __float128 re, __float128 im) {
  char a[64], b[64];
  quadmath_snprintf(a, sizeof a, "%.36Qg", re);
  quadmath_snprintf(b, sizeof b, "%.36Qg", im);
  std::printf("  %-6s (%s, %s)\n", tag, a, b);
}

struct Worst { double u = -1.0; int op = -1, pt = -1; };

// MPC's tan/tanh do not terminate in useful time on the far side of the grid,
// and the reason is not a bug in MPC -- it is Ziv's loop doing exactly what it
// promises.  At grid point 353, z = (9.8e7, 1.95e7), the true Re tan(z) is
// ~ 2 sin(2x) e^{-2y} = e^{-3.9e7}, so resolving the real component to a
// correctly-rounded result demands ~5.6e7 bits of working precision, and MPC
// keeps doubling until it gets them.
//
// SKIPPING IS SOUND HERE, and it is a statement about the FORMAT rather than a
// convenience.  binary128's smallest subnormal is ~6.5e-4966.  Once 2|y| passes
// 11400 (= 4951 decades) the true real component is below that, so the oracle's
// own answer is an exact zero and there is nothing left for it to get wrong
// that binary128 can express -- both references agree trivially.  The
// threshold is therefore |y| > 5700 for tan and |x| > 5700 for tanh, one
// component of a decade inside the format's floor.
//
// COST, counted rather than waved at: 48 of 1780 points for tan and 96 of 1780
// for tanh.  None of them is a point this task is chasing -- the worst complex
// tan cells sit at pt 458 z=(-10, 1e-4) and pt 1708 z=(1, 0), and the worst
// tanh cells at pt 1164 z=(1e-5, -2) and pt 180 z=(-0.707, -0.707), all four
// orders of magnitude inside the cut.  The skipped points are printed.
const double kMpcTanCeiling = 5700.0;

bool mpc_would_grind(const char* op, double re, double im) {
  if (!std::strcmp(op, "tan"))  return std::fabs(im) > kMpcTanCeiling;
  if (!std::strcmp(op, "tanh")) return std::fabs(re) > kMpcTanCeiling;
  return false;
}

int run(int argc, char** argv) {
  const char* only_op   = nullptr;
  const char* dump_op   = nullptr;
  int         dump_pt   = -1;
  int         top_n     = 0;
  mpfr_prec_t prec      = 400;
  bool        selfcheck = false, poison = false, progress = false;
  int         skip_pt   = -1;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--op") && i + 1 < argc)          only_op = argv[++i];
    else if (!std::strcmp(argv[i], "--point") && i + 2 < argc) { dump_op = argv[++i]; dump_pt = std::atoi(argv[++i]); }
    else if (!std::strcmp(argv[i], "--top") && i + 1 < argc)     top_n = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--prec") && i + 1 < argc)    prec = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "--selfcheck"))               selfcheck = true;
    else if (!std::strcmp(argv[i], "--poison"))                  poison = true;
    else if (!std::strcmp(argv[i], "--progress"))                progress = true;
    else if (!std::strcmp(argv[i], "--skip") && i + 1 < argc)    skip_pt = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }

  const std::vector<Pt> grid = load_grid();
  std::printf("probe_complex_oracle: %zu complex grid points, MPC %s at %ld bits\n",
              grid.size(), mpc_get_version(), (long)prec);
  std::printf("all figures are MODULUS-RELATIVE and in DD ulps (x 2^%d)\n\n", kDDSigBits);

  // ---- one point, in full ------------------------------------------------
  if (dump_op) {
    for (int o = 0; o < kNOps; ++o) {
      if (std::strcmp(kOps[o].name, dump_op)) continue;
      for (const Pt& p : grid) {
        if (p.idx != dump_pt) continue;
        __float128 qr, qi, mr, mi;
        ref_quad(kOps[o], p.re, p.im, qr, qi);
        ref_mpc(kOps[o], p.re, p.im, prec, mr, mi);
        std::printf("c %s point %d  family=%s\n", dump_op, dump_pt, p.family.c_str());
        std::printf("  z      = (%.17g, %.17g)\n", p.re, p.im);
        show("quad", qr, qi);
        show("mpc", mr, mi);
        std::printf("  oracle discrepancy = %.6g DD ulps\n", ulps_dd(qr, qi, mr, mi));
        return 0;
      }
    }
    std::fprintf(stderr, "no such op/point\n");
    return 2;
  }

  // ---- MPC convergence: 400 bits vs 800 bits -----------------------------
  if (selfcheck) {
    Worst w;
    for (int o = 0; o < kNOps; ++o) {
      for (const Pt& p : grid) {
        if (mpc_would_grind(kOps[o].name, p.re, p.im)) continue;
        __float128 ar, ai, br, bi;
        ref_mpc(kOps[o], p.re, p.im, 400, ar, ai);
        ref_mpc(kOps[o], p.re, p.im, 800, br, bi);
        const double u = ulps_dd(ar, ai, br, bi);
        if (u > w.u) { w.u = u; w.op = o; w.pt = p.idx; }
      }
    }
    std::printf("SELFCHECK  MPC 400 vs 800 bits, worst over all ops x points:\n");
    std::printf("  %.6g DD ulps  (c %s point %d)\n", w.u,
                w.op >= 0 ? kOps[w.op].name : "?", w.pt);
    std::printf("  %s\n", w.u < 1e-6 ? "400 bits is converged; the probe's own error is "
                                       "not what it is reporting."
                                     : "NOT CONVERGED -- raise --prec before believing "
                                       "anything else this probe prints.");
    return w.u < 1e-6 ? 0 : 1;
  }

  // ---- negative control ---------------------------------------------------
  // Nudge the quad answer by exactly one DD ulp of the modulus, along the
  // modulus direction, and require the probe to see it.  A perturbation
  // smaller than this would risk being absorbed by binary128 rounding of the
  // sum; one full DD ulp is 2^7 binary128 ulps, so the injected value survives
  // exactly and the expected reading is 1.0.
  if (poison) {
    int bad = 0, n = 0;
    double worst_dev = 0.0;
    for (int o = 0; o < kNOps; ++o) {
      for (const Pt& p : grid) {
        if (mpc_would_grind(kOps[o].name, p.re, p.im)) continue;
        __float128 qr, qi, mr, mi;
        ref_quad(kOps[o], p.re, p.im, qr, qi);
        ref_mpc(kOps[o], p.re, p.im, prec, mr, mi);
        const double clean = ulps_dd(qr, qi, mr, mi);
        if (clean < 0.0 || clean > 1e-3) continue;   // only points the probe calls clean
        const __float128 mod = hypotq(mr, mi);
        if (mod == 0 || !finiteq(mod)) continue;
        // move the real component by one DD ulp OF THE MODULUS
        const __float128 kick = ldexpq(mod, -kDDSigBits);
        const double got = ulps_dd(qr + kick, qi, mr, mi);
        ++n;
        const double dev = std::fabs(got - 1.0);
        if (dev > worst_dev) worst_dev = dev;
        if (dev > 0.01) ++bad;
      }
    }
    std::printf("POISON  injected exactly 1.00 DD ulp into %d otherwise-clean points\n", n);
    std::printf("  worst deviation from the expected 1.00 reading: %.6g\n", worst_dev);
    std::printf("  points misreporting by more than 0.01 ulp: %d\n", bad);
    if (n == 0) { std::printf("  FAIL: no clean points to poison -- the control proved nothing.\n"); return 1; }
    if (bad)    { std::printf("  FAIL: the comparison does not see an injected error.\n"); return 1; }
    std::printf("  PASS: the comparison is live, so a ~0 reading elsewhere is a real\n"
                "        agreement and not a dead code path.\n");
    return 0;
  }

  // ---- the survey ---------------------------------------------------------
  struct Rec { double u; int op, pt; };
  std::vector<Rec> all;
  std::printf("%-6s %7s %8s %8s %8s   %-6s %s\n", "op", "npts", ">0.1ulp", ">1ulp", "worst", "wpt", "note");
  Worst gw;
  for (int o = 0; o < kNOps; ++o) {
    if (only_op && std::strcmp(kOps[o].name, only_op)) continue;
    int n = 0, n01 = 0, n1 = 0, nskip = 0;
    double worst = -1.0; int wp = -1;
    for (const Pt& p : grid) {
      if (progress) { std::printf("  ... %s pt %d\n", kOps[o].name, p.idx); std::fflush(stdout); }
      if (skip_pt >= 0 && p.idx == skip_pt) continue;
      if (mpc_would_grind(kOps[o].name, p.re, p.im)) { ++nskip; continue; }
      __float128 qr, qi, mr, mi;
      ref_quad(kOps[o], p.re, p.im, qr, qi);
      ref_mpc(kOps[o], p.re, p.im, prec, mr, mi);
      const double u = ulps_dd(qr, qi, mr, mi);
      if (u < 0.0) continue;                       // no modulus -> nothing to score
      ++n;
      if (u > 0.1) ++n01;
      if (u > 1.0) ++n1;
      if (u > worst) { worst = u; wp = p.idx; }
      all.push_back({u, o, p.idx});
      if (u > gw.u) { gw.u = u; gw.op = o; gw.pt = p.idx; }
    }
    std::printf("%-6s %7d %8d %8d %8.4g   %-6d %s\n", kOps[o].name, n, n01, n1, worst, wp,
                nskip ? (std::to_string(nskip) + " skipped past the binary128 floor").c_str() : "");
  }
  std::printf("\nworst overall: %.6g DD ulps at c %s point %d\n", gw.u,
              gw.op >= 0 ? kOps[gw.op].name : "?", gw.pt);

  if (top_n > 0) {
    std::sort(all.begin(), all.end(), [](const Rec& a, const Rec& b) { return a.u > b.u; });
    std::printf("\ntop %d oracle discrepancies:\n", top_n);
    for (int i = 0; i < top_n && i < (int)all.size(); ++i) {
      const Pt* p = nullptr;
      for (const Pt& q : grid) if (q.idx == all[i].pt) { p = &q; break; }
      std::printf("  %-6s pt %4d  %12.6g DD ulps   z = (%.17g, %.17g)  %s\n",
                  kOps[all[i].op].name, all[i].pt, all[i].u,
                  p ? p->re : 0.0, p ? p->im : 0.0, p ? p->family.c_str() : "?");
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
