// probe_composition_floor.cpp -- how much of the residual complex-trig error is
// INHERENT TO THE FORMULA rather than to the implementation of it.
//
// WHAT QUESTION THIS ANSWERS.  probe_complex_input_rounding.cpp established
// that the correctly rounded answer reads at most 0.492 ulps on the sweep's
// modulus-relative complex metric, so the ~8,200 scored complex-trig rows above
// 1 ulp are real library error and not a threshold artefact.  That does NOT yet
// say they are fixable.  The library does not compute a complex sine directly;
// it composes real building blocks:
//
//     sin(x+iy)  = sin(x) *cosh(y) + i cos(x) *sinh(y)
//     cos(x+iy)  = cos(x) *cosh(y) - i sin(x) *sinh(y)
//     sinh(x+iy) = sinh(x)* cos(y) + i cosh(x)* sin(y)
//     cosh(x+iy) = cosh(x)* cos(y) + i sinh(x)* sin(y)
//
// (dd_complex.hpp, and the FF/QF/TF siblings are the same four lines; QF and TF
// only swap the sincos out-parameter order.)  Each real call lands within half
// an ulp of its own true value AT BEST, each product rounds again, and the two
// components are then measured against the MODULUS.  An implementation built
// from perfect real functions therefore cannot read 0.5; it reads whatever this
// composition costs.  That number -- not ulps_ideal -- is what the shipped
// library has to be compared against before any of its rows is called fixable.
//
// HOW IT IS COMPUTED.  Per row, at 400 bits throughout:
//
//   1. widen the STORED operand, sign-preservingly, to (x, y).
//   2. evaluate the four real building blocks EXACTLY.
//   3. round each to the backend's format -- this is the correctly rounded real
//      library the backend does not have.
//   4. form the two products and round each to the format.
//   5. score the pair against the exact complex reference at the stored
//      operand, with the sweep's own modulus-relative metric.
//
// Step 3 is the whole point.  Skipping it (--poison-exact) must collapse the
// reading back to the correctly rounded floor.
//
// This is an OPTIMISTIC floor and is meant to be.  It charges one rounding per
// building block and one per product; a real DD multiply is slightly worse than
// correctly rounded, and sincos costs more than half an ulp.  So where the
// shipped library already sits at this floor there is provably nothing to win
// from a better implementation of the same formula, and where it sits well
// above it, the gap is real headroom.  A floor that errs low can only make the
// library look worse, never better.
//
// SCOPE.  The four product-form ops only: sin, cos, sinh, cosh.  tan and tanh
// are a quotient of two of these and their error is dominated by the divide and
// by the |Im z| >= 2 asymptotic branch, which is a different mechanism and is
// not modelled here.  The inverse family is not a product form at all.
//
// RESULT.  Over the 26,296 scored rows of the product family, 2,284 read above
// 1 ulp.  Of those:
//
//     2,186 (95.7%) have a building block whose OWN real ulp error is at least
//                   half the complex reading, and
//         0 (0.0%)  would still be above 1 ulp if the building blocks were
//                   correctly rounded -- the largest composition floor over the
//                   whole set is 0.9568.
//
// Per cell the worst real building-block error tracks the worst complex reading
// almost exactly: DD 2.945 vs 3.247, FF 3.52 vs 3.625, QF 1.447 vs 2.035, TF
// 4.344 vs 3.821, against composition floors of 0.90, 0.96, 0.15 and 0.31.
//
// The residual in complex sin/cos/sinh/cosh is therefore the error of the REAL
// sincos and sinhcosh, passed through unchanged.  It is not the complex
// composition, and it cannot be reached from dd_complex.hpp or its siblings --
// those four functions are already at their floor.
//
// NOT A GATE.  docs/CORRECTNESS.md allows one measurement and one verdict; this
// is neither.  It is a diagnosis of rows the sweep has already judged.
//
// POISONS, both of which must pass before believing a number here.
//
//   --poison-exact   skips step 3, so the building blocks stay exact and only
//                    the final pair is rounded.  The reading must then collapse
//                    to the correctly rounded floor -- every row at or below
//                    0.5 ulps.  This is the check that the four formulas above
//                    are transcribed CORRECTLY: a swapped sin/cos or a wrong
//                    sign does not produce a slightly worse floor, it produces
//                    a reading of order 2^p, and the check fails loudly.
//
//   --poison-blocks  perturbs one rounded building block (the first) by one ulp
//                    of the format before the product.  The composition floor
//                    must then RISE on essentially every row.  Zero movement
//                    would mean the building-block rounding is not reaching the
//                    result and the floor is measuring nothing.
//
// BUILD.  -DXPMATH_ENABLE_DIAGNOSTICS=0 silences the ~40 domain-diagnostic
// printfs (config.hpp:121 -- return values are unaffected); the outer grid radii
// overflow real exp on every backend and the diagnostics bury the report.
//
//   g++ -O2 -std=c++17 -fext-numeric-literals -DXPMATH_ENABLE_DIAGNOSTICS=0 \
//       -I include scripts/probe_composition_floor.cpp -o /tmp/probe_cf \
//       -lmpc -lmpfr -lgmp
//
// RUN from the repo root (it reads validation/sweep/sweep_grid.csv):
//   /tmp/probe_cf --poison-exact
//   /tmp/probe_cf --poison-blocks
//   /tmp/probe_cf --csv /tmp/cf.csv
//
// Join the CSV on (backend, op, point) against the sweep baseline to restrict
// the reading to rows the sweep actually SCORES (state=='S').

#include <mpc.h>
#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xp/dd_complex.hpp"
#include "xp/ff_complex.hpp"
#include "xp/qf_complex.hpp"
#include "xp/tf_complex.hpp"

namespace {

const char* const kGrid = "validation/sweep/sweep_grid.csv";
const mpfr_prec_t kPrec = 400;

bool g_poison_exact = false;    // skip the building-block rounding
bool g_poison_blocks = false;   // nudge one rounded building block by an ulp

// ---------------------------------------------------------------- backends

template <class T>
struct Tr;
template <>
struct Tr<xp::DoubleDouble> {
    using C = xp::DoubleDoubleComplex;
    static const char* name() { return "DD"; }
    static int sig_bits() { return 106; }   // sweep_accuracy.cpp:2025
    static bool is_fp32() { return false; }
    static int nlimb() { return 2; }
    static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::FloatFloat> {
    using C = xp::FloatFloatComplex;
    static const char* name() { return "FF"; }
    static int sig_bits() { return 48; }    // sweep_accuracy.cpp:2031
    static bool is_fp32() { return true; }
    static int nlimb() { return 2; }
    static double limb(const xp::FloatFloat& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::QuadFloat> {
    using C = xp::QuadFloatComplex;
    static const char* name() { return "QF"; }
    static int sig_bits() { return 96; }    // sweep_accuracy.cpp:2037
    static bool is_fp32() { return true; }
    static int nlimb() { return 4; }
    static double limb(const xp::QuadFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : i == 2 ? v.f2 : v.f3;
    }
};
template <>
struct Tr<xp::TripleFloat> {
    using C = xp::TripleFloatComplex;
    static const char* name() { return "TF"; }
    static int sig_bits() { return 72; }    // sweep_accuracy.cpp:2043
    static bool is_fp32() { return true; }
    static int nlimb() { return 3; }
    static double limb(const xp::TripleFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : v.f2;
    }
};

// Sign-preserving widen.  The plain limb sum destroys the sign of a zero --
// (-0.0) + (+0.0) is +0.0 -- which is the trap sweep_accuracy.cpp:2549 names as
// the reason the complex path is not scored at the stored operand.
template <class T>
void widen(const T& v, mpfr_ptr out) {
    mpfr_set_d(out, Tr<T>::limb(v, 0), MPFR_RNDN);
    for (int i = 1; i < Tr<T>::nlimb(); ++i) mpfr_add_d(out, out, Tr<T>::limb(v, i), MPFR_RNDN);
    if (mpfr_zero_p(out) && std::signbit(Tr<T>::limb(v, 0)))
        mpfr_setsign(out, out, 1, MPFR_RNDN);
}
template <class T>
void widen_c(const typename Tr<T>::C& z, mpc_ptr out) {
    widen(z.re, mpc_realref(out));
    widen(z.im, mpc_imagref(out));
}

bool g_unrep = false;   // set when the leading word overflows the format

// Round an exact value to the backend's format the way the backend holds a
// number: greedily peel off the nearest IEEE word, subtract, repeat.
template <class T>
void round_to_format(mpfr_srcptr x, mpfr_ptr out) {
    static bool init = false;
    static mpfr_t rem;
    if (!init) { mpfr_init2(rem, kPrec); init = true; }
    mpfr_set(rem, x, MPFR_RNDN);
    mpfr_set_zero(out, mpfr_signbit(x) ? -1 : 1);
    for (int i = 0; i < Tr<T>::nlimb(); ++i) {
        const double w =
            Tr<T>::is_fp32() ? (double)mpfr_get_flt(rem, MPFR_RNDN) : mpfr_get_d(rem, MPFR_RNDN);
        if (!std::isfinite(w)) {
            if (i == 0) g_unrep = true;
            break;
        }
        if (w == 0.0) break;
        mpfr_add_d(out, out, w, MPFR_RNDN);
        mpfr_sub_d(rem, rem, w, MPFR_RNDN);
    }
}

// ---------------------------------------------------------------- metric

// The sweep's complex metric, entirely at 400 bits: |got-ref|/|ref| * 2^p.
// Mirrors scripts/sweep_accuracy.cpp:2105-2140.  Returns -1 where the metric
// has no answer (non-finite, or a zero reference).
double ulps_mpc(mpc_srcptr got, mpc_srcptr ref, int p) {
    static bool init = false;
    static mpc_t d;
    static mpfr_t e, m;
    if (!init) { mpc_init2(d, kPrec); mpfr_init2(e, kPrec); mpfr_init2(m, kPrec); init = true; }
    mpc_abs(m, ref, MPFR_RNDN);
    if (!mpfr_number_p(m) || mpfr_zero_p(m)) return -1.0;
    mpc_sub(d, got, ref, MPC_RNDNN);
    mpc_abs(e, d, MPFR_RNDN);
    if (!mpfr_number_p(e)) return -1.0;
    mpfr_div(e, e, m, MPFR_RNDN);
    mpfr_mul_2si(e, e, p, MPFR_RNDN);
    return mpfr_get_d(e, MPFR_RNDN);
}

// ---------------------------------------------------------------- ops

// Every op here has the shape  re = A(x)*B(y),  im = s * C(x)*D(y).
using Rfn = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

struct Op {
    const char* name;
    Rfn A, B, C, D;
    int s;   // sign of the imaginary part
    xp::DoubleDoubleComplex (*dd)(xp::DoubleDoubleComplex);
    xp::FloatFloatComplex (*ff)(xp::FloatFloatComplex);
    xp::QuadFloatComplex (*qf)(xp::QuadFloatComplex);
    xp::TripleFloatComplex (*tf)(xp::TripleFloatComplex);
};

const Op kOps[] = {
    {"sin", mpfr_sin, mpfr_cosh, mpfr_cos, mpfr_sinh, +1,
     (xp::DoubleDoubleComplex (*)(xp::DoubleDoubleComplex)) & xp::sin,
     (xp::FloatFloatComplex (*)(xp::FloatFloatComplex)) & xp::sin,
     (xp::QuadFloatComplex (*)(xp::QuadFloatComplex)) & xp::sin,
     (xp::TripleFloatComplex (*)(xp::TripleFloatComplex)) & xp::sin},
    {"cos", mpfr_cos, mpfr_cosh, mpfr_sin, mpfr_sinh, -1,
     (xp::DoubleDoubleComplex (*)(xp::DoubleDoubleComplex)) & xp::cos,
     (xp::FloatFloatComplex (*)(xp::FloatFloatComplex)) & xp::cos,
     (xp::QuadFloatComplex (*)(xp::QuadFloatComplex)) & xp::cos,
     (xp::TripleFloatComplex (*)(xp::TripleFloatComplex)) & xp::cos},
    {"sinh", mpfr_sinh, mpfr_cos, mpfr_cosh, mpfr_sin, +1,
     (xp::DoubleDoubleComplex (*)(xp::DoubleDoubleComplex)) & xp::sinh,
     (xp::FloatFloatComplex (*)(xp::FloatFloatComplex)) & xp::sinh,
     (xp::QuadFloatComplex (*)(xp::QuadFloatComplex)) & xp::sinh,
     (xp::TripleFloatComplex (*)(xp::TripleFloatComplex)) & xp::sinh},
    {"cosh", mpfr_cosh, mpfr_cos, mpfr_sinh, mpfr_sin, +1,
     (xp::DoubleDoubleComplex (*)(xp::DoubleDoubleComplex)) & xp::cosh,
     (xp::FloatFloatComplex (*)(xp::FloatFloatComplex)) & xp::cosh,
     (xp::QuadFloatComplex (*)(xp::QuadFloatComplex)) & xp::cosh,
     (xp::TripleFloatComplex (*)(xp::TripleFloatComplex)) & xp::cosh},
};
const int kNOps = (int)(sizeof(kOps) / sizeof(kOps[0]));

template <class T>
typename Tr<T>::C call(const Op& o, const typename Tr<T>::C& z);
template <>
xp::DoubleDoubleComplex call<xp::DoubleDouble>(const Op& o, const xp::DoubleDoubleComplex& z) {
    return o.dd(z);
}
template <>
xp::FloatFloatComplex call<xp::FloatFloat>(const Op& o, const xp::FloatFloatComplex& z) {
    return o.ff(z);
}
template <>
xp::QuadFloatComplex call<xp::QuadFloat>(const Op& o, const xp::QuadFloatComplex& z) {
    return o.qf(z);
}
template <>
xp::TripleFloatComplex call<xp::TripleFloat>(const Op& o, const xp::TripleFloatComplex& z) {
    return o.tf(z);
}

// The four real building blocks AS THE LIBRARY COMPUTES THEM, widened.  sin/cos
// come from one sincos call and sinh/cosh from one sinhcosh call, which is
// exactly how the complex headers obtain them.  NOTE the out-parameter order
// differs between backends -- DD and FF are (cos, sin) and (cosh, sinh); QF and
// TF are (sin, cos) and (sinh, cosh).  Getting that wrong is a known trap, so
// the wrappers below are per-backend rather than one template.
template <class T>
void lib_sincos(const T& v, T& sn, T& cs);
template <>
void lib_sincos(const xp::DoubleDouble& v, xp::DoubleDouble& sn, xp::DoubleDouble& cs) {
    xp::sincos(v, cs, sn);
}
template <>
void lib_sincos(const xp::FloatFloat& v, xp::FloatFloat& sn, xp::FloatFloat& cs) {
    xp::sincos(v, cs, sn);
}
template <>
void lib_sincos(const xp::QuadFloat& v, xp::QuadFloat& sn, xp::QuadFloat& cs) {
    xp::sincos(v, sn, cs);
}
template <>
void lib_sincos(const xp::TripleFloat& v, xp::TripleFloat& sn, xp::TripleFloat& cs) {
    xp::sincos(v, sn, cs);
}
template <class T>
void lib_sinhcosh(const T& v, T& sn, T& cs);
template <>
void lib_sinhcosh(const xp::DoubleDouble& v, xp::DoubleDouble& sn, xp::DoubleDouble& cs) {
    xp::sinhcosh(v, cs, sn);
}
template <>
void lib_sinhcosh(const xp::FloatFloat& v, xp::FloatFloat& sn, xp::FloatFloat& cs) {
    xp::sinhcosh(v, cs, sn);
}
template <>
void lib_sinhcosh(const xp::QuadFloat& v, xp::QuadFloat& sn, xp::QuadFloat& cs) {
    xp::sinhcosh(v, sn, cs);
}
template <>
void lib_sinhcosh(const xp::TripleFloat& v, xp::TripleFloat& sn, xp::TripleFloat& cs) {
    xp::sinhcosh(v, sn, cs);
}

// A REAL relative ulp reading, |got-ref|/|ref| * 2^p, at 400 bits.
double ulps_real(mpfr_srcptr got, mpfr_srcptr ref, int p) {
    static bool init = false;
    static mpfr_t e;
    if (!init) { mpfr_init2(e, kPrec); init = true; }
    if (!mpfr_number_p(ref) || mpfr_zero_p(ref) || !mpfr_number_p(got)) return -1.0;
    mpfr_sub(e, got, ref, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    mpfr_div(e, e, ref, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    mpfr_mul_2si(e, e, p, MPFR_RNDN);
    return mpfr_get_d(e, MPFR_RNDN);
}

// ---------------------------------------------------------------- grid

struct Pt {
    int idx;
    double re, im;
    std::string family;
};

std::vector<Pt> load_grid() {
    std::vector<Pt> v;
    FILE* f = std::fopen(kGrid, "r");
    if (!f) {
        std::fprintf(stderr, "cannot open %s (run from the repo root)\n", kGrid);
        std::exit(2);
    }
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] != 'c' || line[1] != ',') continue;
        char* fields[5];
        int nf = 0;
        fields[nf++] = line;
        for (char* q = line; *q && nf < 5; ++q)
            if (*q == ',') { *q = 0; fields[nf++] = q + 1; }
        if (nf < 5) continue;
        Pt p;
        p.idx = std::atoi(fields[1]);
        p.family = fields[2];
        p.re = std::strtod(fields[3], nullptr);   // round-trips -0 as -0
        p.im = std::strtod(fields[4], nullptr);
        v.push_back(p);
    }
    std::fclose(f);
    return v;
}

// ---------------------------------------------------------------- rows

struct Row {
    const char* bk;
    const char* op;
    int pt;
    const char* family;
    double u_lib;    // what the shipped library reads
    double u_comp;   // what correctly rounded building blocks would read
    double u_blk;    // worst REAL ulp among the four building blocks
};
std::vector<Row> g_rows;

long g_exact_over = 0;      // --poison-exact rows above 0.5
double g_exact_worst = 0.0;
long g_blocks_same = 0;     // --poison-blocks rows that did not move
long g_blocks_seen = 0;

// ---------------------------------------------------------------- run

template <class T>
void run_backend(const std::vector<Pt>& grid) {
    using C = typename Tr<T>::C;
    const int p = Tr<T>::sig_bits();

    mpc_t zs, rs, lib, comp;
    mpfr_t a, b, c, d, ar, br, cr, dr, prod, k, wa, wb, wc, wd;
    mpc_init2(zs, kPrec);
    mpc_init2(rs, kPrec);
    mpc_init2(lib, kPrec);
    mpc_init2(comp, kPrec);
    mpfr_inits2(kPrec, a, b, c, d, ar, br, cr, dr, prod, k, wa, wb, wc, wd, (mpfr_ptr)0);
    // one ulp of the format, as a multiplier, for --poison-blocks
    mpfr_set_ui(k, 1, MPFR_RNDN);
    mpfr_mul_2si(k, k, -p, MPFR_RNDN);
    mpfr_add_ui(k, k, 1, MPFR_RNDN);

    for (int o = 0; o < kNOps; ++o) {
        for (const Pt& pt : grid) {
            const C z(T(pt.re), T(pt.im));
            const C r = call<T>(kOps[o], z);

            widen_c<T>(z, zs);
            // exact complex reference at the operand the library actually held
            switch (o) {
                case 0: mpc_sin(rs, zs, MPC_RNDNN); break;
                case 1: mpc_cos(rs, zs, MPC_RNDNN); break;
                case 2: mpc_sinh(rs, zs, MPC_RNDNN); break;
                default: mpc_cosh(rs, zs, MPC_RNDNN); break;
            }
            widen_c<T>(r, lib);

            // the four real building blocks, exactly
            kOps[o].A(a, mpc_realref(zs), MPFR_RNDN);
            kOps[o].B(b, mpc_imagref(zs), MPFR_RNDN);
            kOps[o].C(c, mpc_realref(zs), MPFR_RNDN);
            kOps[o].D(d, mpc_imagref(zs), MPFR_RNDN);

            g_unrep = false;
            if (g_poison_exact) {
                mpfr_set(ar, a, MPFR_RNDN);
                mpfr_set(br, b, MPFR_RNDN);
                mpfr_set(cr, c, MPFR_RNDN);
                mpfr_set(dr, d, MPFR_RNDN);
            } else {
                round_to_format<T>(a, ar);
                round_to_format<T>(b, br);
                round_to_format<T>(c, cr);
                round_to_format<T>(d, dr);
                if (g_poison_blocks) mpfr_mul(ar, ar, k, MPFR_RNDN);
            }

            mpfr_mul(prod, ar, br, MPFR_RNDN);
            round_to_format<T>(prod, mpc_realref(comp));
            mpfr_mul(prod, cr, dr, MPFR_RNDN);
            if (kOps[o].s < 0) mpfr_neg(prod, prod, MPFR_RNDN);
            round_to_format<T>(prod, mpc_imagref(comp));

            // The same four building blocks AS THE LIBRARY COMPUTES THEM.  For
            // sin and cos the library calls sincos(x) and sinhcosh(y); for sinh
            // and cosh it calls sinhcosh(x) and sincos(y).
            T sx, cx, sy, cy;
            if (o < 2) { lib_sincos<T>(z.re, sx, cx); lib_sinhcosh<T>(z.im, sy, cy); }
            else       { lib_sinhcosh<T>(z.re, sx, cx); lib_sincos<T>(z.im, sy, cy); }
            // A = f(x), B = g(y), C = h(x), D = k(y) in the order kOps declares.
            // o=0 sin:  A=sin x  B=cosh y  C=cos x   D=sinh y
            // o=1 cos:  A=cos x  B=cosh y  C=sin x   D=sinh y
            // o=2 sinh: A=sinh x B=cos y   C=cosh x  D=sin y
            // o=3 cosh: A=cosh x B=cos y   C=sinh x  D=sin y
            widen(o == 0 || o == 2 ? sx : cx, wa);
            widen(o < 2 ? cy : cy, wb);
            widen(o == 0 || o == 2 ? cx : sx, wc);
            widen(sy, wd);
            double ub = 0.0;
            for (const auto& pr : {std::make_pair((mpfr_ptr)wa, (mpfr_ptr)a),
                                   std::make_pair((mpfr_ptr)wb, (mpfr_ptr)b),
                                   std::make_pair((mpfr_ptr)wc, (mpfr_ptr)c),
                                   std::make_pair((mpfr_ptr)wd, (mpfr_ptr)d)}) {
                const double u = ulps_real(pr.first, pr.second, p);
                if (u >= 0) ub = std::max(ub, u);
            }

            const double ul = ulps_mpc(lib, rs, p);
            const double uc = ulps_mpc(comp, rs, p);
            if (ul < 0 || uc < 0 || g_unrep) continue;

            if (g_poison_exact) {
                // With exact building blocks only the final pair is rounded, so
                // this IS the correctly rounded answer and must read <= 0.5.  A
                // mistranscribed formula reads of order 2^p instead.
                if (uc > 0.5) {
                    ++g_exact_over;
                    g_exact_worst = std::max(g_exact_worst, uc);
                }
                continue;
            }
            g_rows.push_back(
                {Tr<T>::name(), kOps[o].name, pt.idx, pt.family.c_str(), ul, uc, ub});
        }
    }
    mpfr_clears(a, b, c, d, ar, br, cr, dr, prod, k, wa, wb, wc, wd, (mpfr_ptr)0);
    mpc_clear(zs);
    mpc_clear(rs);
    mpc_clear(lib);
    mpc_clear(comp);
}

double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const char* csv = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--poison-exact")) g_poison_exact = true;
        else if (!std::strcmp(argv[i], "--poison-blocks")) g_poison_blocks = true;
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
    }

    const std::vector<Pt> grid = load_grid();
    std::printf("# grid %zu complex points, MPC %s at %d bits\n", grid.size(), MPC_VERSION_STRING,
                (int)kPrec);

    run_backend<xp::DoubleDouble>(grid);
    run_backend<xp::FloatFloat>(grid);
    run_backend<xp::QuadFloat>(grid);
    run_backend<xp::TripleFloat>(grid);

    if (g_poison_exact) {
        std::printf("POISON-EXACT: building blocks left exact, only the final pair\n"
                    "rounded.  Every row must then read <= 0.5 ulps -- that is the\n"
                    "correctly rounded floor, and a mistranscribed formula cannot\n"
                    "reach it.\n");
        std::printf("  rows above 0.5: %ld   worst: %.6g\n", g_exact_over, g_exact_worst);
        std::printf("%s\n", g_exact_over == 0 ? "PASS -- the four formulas are transcribed correctly."
                                              : "FAIL");
        return g_exact_over == 0 ? 0 : 1;
    }

    if (g_poison_blocks) {
        // Re-run clean into a second vector to compare against.
        std::vector<Row> poisoned;
        poisoned.swap(g_rows);
        g_poison_blocks = false;
        run_backend<xp::DoubleDouble>(grid);
        run_backend<xp::FloatFloat>(grid);
        run_backend<xp::QuadFloat>(grid);
        run_backend<xp::TripleFloat>(grid);
        const size_t n = std::min(poisoned.size(), g_rows.size());
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ++g_blocks_seen;
            if (poisoned[i].u_comp <= g_rows[i].u_comp) ++g_blocks_same;
            else worst = std::max(worst, poisoned[i].u_comp - g_rows[i].u_comp);
        }
        std::printf("POISON-BLOCKS: one rounded building block nudged by an ulp of the\n"
                    "format.  The composition floor must rise -- if it does not, the\n"
                    "building-block rounding is not reaching the result.\n");
        std::printf("  rows that did NOT rise: %ld of %ld (%.2f%%)   worst rise: %.4g\n",
                    g_blocks_same, g_blocks_seen, 100.0 * g_blocks_same / (g_blocks_seen ? g_blocks_seen : 1),
                    worst);
        // A nudge of one ulp cannot move a row whose product is dominated by the
        // OTHER component, so some rows legitimately do not rise; requiring a
        // majority is the honest bar.
        const bool ok = g_blocks_same * 2 < g_blocks_seen;
        std::printf("%s\n", ok ? "PASS -- the building-block rounding reaches the result."
                               : "FAIL -- the floor is not sensitive to its own inputs.");
        return ok ? 0 : 1;
    }

    if (csv) {
        FILE* f = std::fopen(csv, "w");
        if (!f) { std::perror(csv); return 2; }
        std::fprintf(f, "backend,op,point,family,ulps_lib,ulps_comp,ulps_block\n");
        for (const Row& r : g_rows)
            std::fprintf(f, "%s,%s,%d,%s,%.6g,%.6g,%.6g\n", r.bk, r.op, r.pt, r.family,
                         r.u_lib, r.u_comp, r.u_blk);
        std::fclose(f);
        std::printf("wrote %zu rows to %s\n", g_rows.size(), csv);
    }

    std::printf("\nrows measured: %zu\n", g_rows.size());
    std::printf("\nCOMPOSITION FLOOR vs SHIPPED LIBRARY (all grid rows, not just scored)\n");
    std::printf("%-4s %-5s %7s %9s %9s %9s %9s %9s %8s\n", "bk", "op", "rows", "med lib",
                "med flr", "max lib", "max flr", "max blk", "lib>2flr");
    for (const char* bk : {"DD", "FF", "QF", "TF"}) {
        for (int o = 0; o < kNOps; ++o) {
            std::vector<double> L, F;
            double ml = 0, mf = 0, mb = 0;
            long over = 0;
            for (const Row& r : g_rows) {
                if (std::strcmp(r.bk, bk) || std::strcmp(r.op, kOps[o].name)) continue;
                L.push_back(r.u_lib);
                F.push_back(r.u_comp);
                ml = std::max(ml, r.u_lib);
                mf = std::max(mf, r.u_comp);
                mb = std::max(mb, r.u_blk);
                if (r.u_lib > 2.0 * r.u_comp && r.u_lib > 1.0) ++over;
            }
            if (L.empty()) continue;
            const size_t n = L.size();
            std::printf("%-4s %-5s %7zu %9.4f %9.4f %9.4g %9.4g %9.4g %8ld\n", bk,
                        kOps[o].name, n, median(L), median(F), ml, mf, mb, over);
        }
    }
    return 0;
}
