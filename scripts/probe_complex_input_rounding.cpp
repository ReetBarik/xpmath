// probe_complex_input_rounding.cpp -- how much of a complex-trig sweep row is
// the LIBRARY and how much is the GRID POINT NOT FITTING IN THE FORMAT.
//
// WHY.  sweep_accuracy.cpp scores every complex row against the oracle at the
// GRID input, not at the operand the backend actually stored:
//
//     scripts/sweep_accuracy.cpp:2530  "LAYER 0, COMPLEX: NOT APPLIED.
//                                       Deliberately still the grid oracle."
//
// and the comment there says why -- to_q() widens by summing limbs,
// (-0.0) + (+0.0) = +0.0, so a stored-input oracle answers for the CONJUGATE at
// every branch-cut point, which produced 1,996 false defects when it was tried.
//
// The consequence is that on FF (48 bits) a grid point given as a 53-bit double
// is NOT the number the library was handed, and where the operation amplifies
// that difference the row charges the library for it.  z = 0.999 on the real
// axis is the clean example: acos'(x) = -1/sqrt(1-x^2) = -22.4 there, and an FF
// storage error of ~2^-49 relative emerges as ~250 ulps of a result whose
// modulus is only 0.0447.
//
// This probe measures both readings at once, at the backend's own sig_bits:
//
//     ulps_grid    |lib - MPC(grid double)|   / |MPC(grid double)|   * 2^p
//     ulps_stored  |lib - MPC(stored value)|  / |MPC(stored value)|  * 2^p
//
// ulps_grid reproduces the sweep's question and is the metric of record.
// ulps_stored is the library's own error with the input question removed.  The
// gap between them is what no change to the library can remove.
//
// THE SECOND FLOOR.  Even at the stored input there is a reading the metric
// cannot go below.  A backend value of modulus M is a sum of IEEE words, so its
// smallest nonzero increment is the format's smallest subnormal, D -- 2^-149 for
// the FP32 backends (FF, TF, QF), 2^-1074 for DD.  No algorithm can place the
// answer closer than D/2 to the truth, and the metric divides by M*2^-p, so
//
//     floor_ulps = 2^p * D / M
//
// is a property of the FORMAT AT THAT MAGNITUDE, not of the code.  For QF
// (p = 96, D = 2^-149) that is 2^-53 / M, which at M = 1e-21 is 1.11e5 ulps.
// The `xfloor` column below is ulps_stored / floor_ulps; a row reading 2.000 or
// 4.000 there is at the representational floor and no library change can move
// it.  A row reading 1e+13 there is a real defect.
//
// EVERYTHING IS MPFR AT 400 BITS.  An earlier draft did the arithmetic in
// __float128 and its own --poison caught it: __float128 carries 113 bits, so
// injecting a 2^-106 perturbation into a DD value by multiplication rounds away
// 2^-113 / 2^-106 = 2^-7 of it, and the poison read 0.99219 instead of 1.00000
// on 40,831 rows.  400 bits leaves 294 bits of headroom over DD and the poison
// reads 1.000000 everywhere.  This is why the poison is in the file.
//
// THE SIGNED ZERO.  The stored-input reference is only trustworthy if widening
// preserves the sign of a zero limb, which plain summation does not.  widen()
// carries the sign of limb 0 explicitly when the sum is zero.  That is the fix
// sweep_accuracy.cpp:2549 names as the precondition for ever applying a stored
// oracle, and --poison-sign is the measurement it asks for.
//
// NOT A GATE.  Nothing here issues a verdict; docs/CORRECTNESS.md allows one
// measurement and one verdict and this is neither.  It is a diagnosis of rows
// the sweep has already judged.
//
// POISONS, all three of which must pass before believing a number here.
//
//   --poison        replaces every library result with ref_stored * (1 + 2^-p),
//                   computed at 400 bits.  Every ulps_stored reading must then
//                   be 1.000000.  Anything else proves the metric, the widen or
//                   the reference is not doing what this file claims.
//                   Measured on the shipped tree: 0 misses, worst miss 0.
//                   THIS POISON HAS ALREADY EARNED ITS KEEP -- it is what caught
//                   the __float128 draft described above.
//
//   --poison-sign   disables the sign-carrying line in widen(), i.e. reverts to
//                   the plain limb sum sweep_accuracy.cpp warns about, and
//                   counts widened components whose sign moves.  Zero would mean
//                   the signed-zero handling here is decorative and the
//                   branch-cut rows cannot be trusted.  Measured: 684 of 14,240.
//
//   --poison-floor  gives every backend FP64's exponent range, so 2^p*D/|ref|
//                   drops by 2^925 and no row can be within 8x of it.  The
//                   REPRESENTATIONAL FLOOR count must fall to zero, proving the
//                   classification is driven by FP32's actual subnormal limit
//                   and not by kFloorSlack being generous.  Measured: 36 -> 0.
//
// BUILD.  -DXPMATH_ENABLE_DIAGNOSTICS=0 silences the ~40 domain-diagnostic
// printfs (config.hpp:121 -- return values are unaffected); the outer grid radii
// overflow real exp on every backend and the diagnostics bury the report.
//
//   g++ -O2 -std=c++17 -fext-numeric-literals -DXPMATH_ENABLE_DIAGNOSTICS=0 \
//       -I include scripts/probe_complex_input_rounding.cpp -o /tmp/probe_cir \
//       -lmpc -lmpfr -lgmp
//
// RUN from the repo root (it reads validation/sweep/sweep_grid.csv):
//   /tmp/probe_cir --poison
//   /tmp/probe_cir --poison-sign
//   /tmp/probe_cir --poison-floor
//   /tmp/probe_cir --above 8
//   /tmp/probe_cir --cell FF acos --list 20
//   /tmp/probe_cir --csv /tmp/cir.csv    # + floor_ulps and xfloor per row
//
// The CSV is the useful output: join it on (backend, op, point) against the
// sweep baseline to restrict the reading to rows the sweep actually SCORES.
// Done that way the probe reproduces the sweep exactly -- 242 rows above 8
// ulps, identical membership, and no row above 1 ulp disagreeing with the
// libquadmath oracle by more than 2.0%.

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

// How far above 2^p*D/|ref| a row may read and still be called floor-limited.
// 8.0 is not a new judgement: it is kUlpAllowance, the slack the absolute gate
// already applies to its own derived bound (sweep_accuracy.cpp:876).  The
// xfloor column is printed so the reader never has to take the classification
// on trust -- the QF rows it catches read 2.000 and 4.000, not 7.9.
const double kFloorSlack = 8.0;

bool g_poison = false;
bool g_drop_sign = false;   // --poison-sign: revert widen() to the plain sum
// --poison-floor: pretend the FP32 backends have FP64's exponent range.  The
// floor classification must then catch NOTHING, which is what proves it is
// driven by the actual subnormal limit and not by kFloorSlack being generous.
bool g_poison_floor = false;

// ---------------------------------------------------------------- backends

template <class T>
struct Tr;
template <>
struct Tr<xp::DoubleDouble> {
    using C = xp::DoubleDoubleComplex;
    static const char* name() { return "DD"; }
    static int sig_bits() { return 106; }   // sweep_accuracy.cpp:2025
    static int denorm_exp() { return -1074; }   // FP64 smallest subnormal
    static int nlimb() { return 2; }
    static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::FloatFloat> {
    using C = xp::FloatFloatComplex;
    static const char* name() { return "FF"; }
    static int sig_bits() { return 48; }    // sweep_accuracy.cpp:2031
    static int denorm_exp() { return -149; }   // FP32 smallest subnormal
    static int nlimb() { return 2; }
    static double limb(const xp::FloatFloat& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::QuadFloat> {
    using C = xp::QuadFloatComplex;
    static const char* name() { return "QF"; }
    static int sig_bits() { return 96; }    // sweep_accuracy.cpp:2037
    static int denorm_exp() { return -149; }   // FP32 smallest subnormal
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
    static int denorm_exp() { return -149; }   // FP32 smallest subnormal
    static int nlimb() { return 3; }
    static double limb(const xp::TripleFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : v.f2;
    }
};

// Sign-preserving widen.  The limb sum is exact at 400 bits; the last line is
// the part plain summation gets wrong -- see THE SIGNED ZERO above.
template <class T>
void widen(const T& v, mpfr_ptr out) {
    mpfr_set_d(out, Tr<T>::limb(v, 0), MPFR_RNDN);
    for (int i = 1; i < Tr<T>::nlimb(); ++i)
        mpfr_add_d(out, out, Tr<T>::limb(v, i), MPFR_RNDN);
    if (!g_drop_sign && mpfr_zero_p(out) && std::signbit(Tr<T>::limb(v, 0)))
        mpfr_setsign(out, out, 1, MPFR_RNDN);
}

template <class T>
void widen_c(const typename Tr<T>::C& z, mpc_ptr out) {
    widen(z.re, mpc_realref(out));
    widen(z.im, mpc_imagref(out));
}

// ---------------------------------------------------------------- ops

struct Op {
    const char* name;
    int (*m)(mpc_ptr, mpc_srcptr, mpc_rnd_t);
};
const Op kOps[] = {
    {"sin", mpc_sin},     {"cos", mpc_cos},     {"tan", mpc_tan},
    {"asin", mpc_asin},   {"acos", mpc_acos},   {"atan", mpc_atan},
    {"sinh", mpc_sinh},   {"cosh", mpc_cosh},   {"tanh", mpc_tanh},
    {"asinh", mpc_asinh}, {"acosh", mpc_acosh}, {"atanh", mpc_atanh},
};
const int kNOps = (int)(sizeof kOps / sizeof kOps[0]);

template <class T>
typename Tr<T>::C apply(int op, const typename Tr<T>::C& z) {
    switch (op) {
        case 0: return xp::sin(z);
        case 1: return xp::cos(z);
        case 2: return xp::tan(z);
        case 3: return xp::asin(z);
        case 4: return xp::acos(z);
        case 5: return xp::atan(z);
        case 6: return xp::sinh(z);
        case 7: return xp::cosh(z);
        case 8: return xp::tanh(z);
        case 9: return xp::asinh(z);
        case 10: return xp::acosh(z);
        default: return xp::atanh(z);
    }
}

// mpc_tan/mpc_tanh drive Ziv's loop for a very long time near the poles of the
// grid; probe_complex_oracle.cpp carries the same ceiling for the same reason.
const double kMpcTanCeiling = 5700.0;
bool would_grind(int op, double re, double im) {
    if (std::strcmp(kOps[op].name, "tan") == 0) return std::fabs(im) > kMpcTanCeiling;
    if (std::strcmp(kOps[op].name, "tanh") == 0) return std::fabs(re) > kMpcTanCeiling;
    return false;
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

// ---------------------------------------------------------------- metric

// Modulus-relative ulps, entirely at kPrec bits.  Same shape as
// sweep_accuracy.cpp:2105-2140: |got - ref| / |ref| * 2^sig_bits, one number for
// the complex pair rather than one per component.  Returns -1 when the row
// carries no question (zero reference, or a non-finite anywhere).
double ulps_mpc(mpc_srcptr got, mpc_srcptr ref, int p) {
    static bool init = false;
    static mpc_t d;
    static mpfr_t e, m;
    if (!init) {
        mpc_init2(d, kPrec);
        mpfr_init2(e, kPrec);
        mpfr_init2(m, kPrec);
        init = true;
    }
    mpc_abs(m, ref, MPFR_RNDN);
    if (!mpfr_number_p(m) || mpfr_zero_p(m)) return -1.0;
    mpc_sub(d, got, ref, MPC_RNDNN);
    mpc_abs(e, d, MPFR_RNDN);
    if (!mpfr_number_p(e)) return -1.0;
    mpfr_div(e, e, m, MPFR_RNDN);
    mpfr_mul_2si(e, e, p, MPFR_RNDN);
    return mpfr_get_d(e, MPFR_RNDN);
}

// ---------------------------------------------------------------- rows

struct Row {
    const char* bk;
    const char* op;
    int pt;
    const char* family;
    double u_grid, u_stored;
    double floor_u;   // 2^p * D / |ref_stored| -- see THE SECOND FLOOR
    double xfloor() const { return floor_u > 0 ? u_stored / floor_u : 0.0; }
};

// |ref| as a double, saturating rather than trapping on the outer grid radii.
double modulus(mpc_srcptr z) {
    static bool init = false;
    static mpfr_t m;
    if (!init) { mpfr_init2(m, kPrec); init = true; }
    mpc_abs(m, z, MPFR_RNDN);
    return mpfr_get_d(m, MPFR_RNDN);
}

std::vector<Row> g_rows;
int g_poison_bad = 0;
double g_poison_worst = 0.0;
long g_sign_moved = 0, g_sign_seen = 0;

template <class T>
void run_backend(const std::vector<Pt>& grid, const char* only_bk, const char* only_op) {
    using C = typename Tr<T>::C;
    const int p = Tr<T>::sig_bits();
    if (only_bk && std::strcmp(only_bk, Tr<T>::name())) return;

    mpc_t zg, zs, rg, rs, lib;
    mpc_init2(zg, kPrec);
    mpc_init2(zs, kPrec);
    mpc_init2(rg, kPrec);
    mpc_init2(rs, kPrec);
    mpc_init2(lib, kPrec);
    mpfr_t k;
    mpfr_init2(k, kPrec);
    mpfr_set_ui_2exp(k, 1, -p, MPFR_RNDN);   // 2^-p, exact
    mpfr_add_ui(k, k, 1, MPFR_RNDN);         // 1 + 2^-p, exact at 400 bits

    for (int o = 0; o < kNOps; ++o) {
        if (only_op && std::strcmp(only_op, kOps[o].name)) continue;
        for (const Pt& pt : grid) {
            if (would_grind(o, pt.re, pt.im)) continue;

            // Operand built exactly as sweep_accuracy.cpp builds it: from the
            // DOUBLE, so the expansion splits it across limbs and FF/TF drop the
            // bits they cannot hold.
            const C z(T(pt.re), T(pt.im));
            const C r = apply<T>(o, z);

            // reference at the GRID double -- the sweep's own question
            mpc_set_d_d(zg, pt.re, pt.im, MPC_RNDNN);
            kOps[o].m(rg, zg, MPC_RNDNN);

            // reference at the STORED operand -- the library's own question
            widen_c<T>(z, zs);
            kOps[o].m(rs, zs, MPC_RNDNN);

            widen_c<T>(r, lib);

            if (g_poison) {
                // Replace the library answer with the stored reference nudged by
                // exactly one ulp of the format.  ulps_stored must read 1.000000.
                mpc_mul_fr(lib, rs, k, MPC_RNDNN);
                const double u = ulps_mpc(lib, rs, p);
                if (u < 0) continue;
                const double rel = std::fabs(u - 1.0);
                if (rel > g_poison_worst) g_poison_worst = rel;
                if (rel > 1e-9) ++g_poison_bad;
                continue;
            }

            const double ug = ulps_mpc(lib, rg, p);
            const double us = ulps_mpc(lib, rs, p);
            if (ug < 0 || us < 0) continue;
            // 2^p * D / M, entirely in the exponent so it does not underflow.
            const double M = modulus(rs);
            const int de = g_poison_floor ? -1074 : Tr<T>::denorm_exp();
            const double fl =
                (M > 0 && std::isfinite(M)) ? std::ldexp(1.0, p + de) / M : 0.0;
            g_rows.push_back(
                {Tr<T>::name(), kOps[o].name, pt.idx, pt.family.c_str(), ug, us, fl});
        }
    }
    mpfr_clear(k);
    mpc_clear(zg);
    mpc_clear(zs);
    mpc_clear(rg);
    mpc_clear(rs);
    mpc_clear(lib);
}

// --poison-sign: count widened components whose sign moves when the
// sign-carrying line is dropped.
template <class T>
void run_sign_probe(const std::vector<Pt>& grid) {
    mpfr_t a, b;
    mpfr_init2(a, kPrec);
    mpfr_init2(b, kPrec);
    for (const Pt& pt : grid) {
        const T vals[2] = {T(pt.re), T(pt.im)};
        for (const T& v : vals) {
            ++g_sign_seen;
            g_drop_sign = false;
            widen(v, a);
            g_drop_sign = true;
            widen(v, b);
            if (mpfr_signbit(a) != mpfr_signbit(b)) ++g_sign_moved;
        }
    }
    g_drop_sign = false;
    mpfr_clear(a);
    mpfr_clear(b);
}

}  // namespace

int main(int argc, char** argv) {
    const char* only_bk = nullptr;
    const char* only_op = nullptr;
    const char* csv = nullptr;
    double above = 8.0;
    int list = 0;
    bool sign_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--poison")) g_poison = true;
        else if (!std::strcmp(argv[i], "--poison-sign")) sign_mode = true;
        else if (!std::strcmp(argv[i], "--poison-floor")) g_poison_floor = true;
        else if (!std::strcmp(argv[i], "--above") && i + 1 < argc) above = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--list") && i + 1 < argc) list = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!std::strcmp(argv[i], "--cell") && i + 2 < argc) {
            only_bk = argv[++i];
            only_op = argv[++i];
        }
    }

    const std::vector<Pt> grid = load_grid();
    std::printf("# grid %zu complex points, MPC %s at %ld bits\n", grid.size(),
                mpc_get_version(), (long)kPrec);
    if (g_poison_floor)
        std::printf("POISON-FLOOR: every backend is given FP64's exponent range.\n"
                    "A correct classification must now find ZERO floor-limited rows.\n");

    if (sign_mode) {
        run_sign_probe<xp::DoubleDouble>(grid);
        run_sign_probe<xp::FloatFloat>(grid);
        run_sign_probe<xp::QuadFloat>(grid);
        run_sign_probe<xp::TripleFloat>(grid);
        std::printf("POISON-SIGN: %ld of %ld widened components change sign when the\n"
                    "sign-carrying line is dropped.\n", g_sign_moved, g_sign_seen);
        std::printf("%s\n", g_sign_moved > 0
                    ? "PASS -- the signed-zero handling is load-bearing."
                    : "FAIL -- it is decorative; branch-cut rows here are not trustworthy.");
        return g_sign_moved > 0 ? 0 : 1;
    }

    run_backend<xp::DoubleDouble>(grid, only_bk, only_op);
    run_backend<xp::FloatFloat>(grid, only_bk, only_op);
    run_backend<xp::QuadFloat>(grid, only_bk, only_op);
    run_backend<xp::TripleFloat>(grid, only_bk, only_op);

    if (g_poison) {
        std::printf("POISON: one injected ulp read back on ulps_stored.\n");
        std::printf("  misses (>1e-9 relative): %d   worst relative miss: %.3g\n",
                    g_poison_bad, g_poison_worst);
        std::printf("%s\n", g_poison_bad == 0 ? "PASS" : "FAIL");
        return g_poison_bad == 0 ? 0 : 1;
    }

    if (csv) {
        FILE* f = std::fopen(csv, "w");
        if (!f) { std::perror(csv); return 2; }
        std::fprintf(f, "backend,op,point,family,ulps_grid,ulps_stored,floor_ulps,xfloor\n");
        for (const Row& r : g_rows)
            std::fprintf(f, "%s,%s,%d,%s,%.6g,%.6g,%.6g,%.6g\n", r.bk, r.op, r.pt,
                         r.family, r.u_grid, r.u_stored, r.floor_u, r.xfloor());
        std::fclose(f);
        std::printf("wrote %zu rows to %s\n", g_rows.size(), csv);
    }

    std::printf("\nrows measured: %zu\n", g_rows.size());
    std::printf("\nCELLS WITH ROWS ABOVE %.4g ulps (grid oracle = the sweep's own question)\n",
                above);
    std::printf("  n>thr  rows the sweep's metric puts above the threshold\n"
                "  stored rows still above it once the oracle is moved to the STORED operand\n"
                "         (the difference is input rounding -- no library change reaches it)\n"
                "  float  of those, rows within %gx the format's representational floor\n"
                "         (the difference is real library error)\n\n", (double)kFloorSlack);
    std::printf("%-3s %-6s %-9s %7s %7s %7s   %13s %13s %9s\n", "bk", "op", "family",
                "n>thr", "stored", "floor", "worst grid", "worst stored", "xfloor");
    std::vector<size_t> idx(g_rows.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [](size_t a, size_t b) {
        const Row& x = g_rows[a];
        const Row& y = g_rows[b];
        int c = std::strcmp(x.bk, y.bk);
        if (c) return c < 0;
        c = std::strcmp(x.op, y.op);
        if (c) return c < 0;
        return std::strcmp(x.family, y.family) < 0;
    });
    size_t i = 0;
    while (i < idx.size()) {
        size_t j = i;
        const Row& h = g_rows[idx[i]];
        while (j < idx.size() && !std::strcmp(g_rows[idx[j]].bk, h.bk) &&
               !std::strcmp(g_rows[idx[j]].op, h.op) &&
               !std::strcmp(g_rows[idx[j]].family, h.family))
            ++j;
        int n = 0, still = 0, atfloor = 0;
        double wg = 0, ws = 0, wx = 0;
        for (size_t k = i; k < j; ++k) {
            const Row& r = g_rows[idx[k]];
            if (r.u_grid <= above) continue;
            ++n;
            if (r.u_stored > above) {
                ++still;
                if (r.xfloor() > 0 && r.xfloor() <= kFloorSlack) ++atfloor;
            }
            if (r.u_grid > wg) { wg = r.u_grid; ws = r.u_stored; wx = r.xfloor(); }
        }
        if (n)
            std::printf("%-3s %-6s %-9s %7d %7d %7d   %13.5g %13.5g %9.4g\n", h.bk, h.op,
                        h.family, n, still, atfloor, wg, ws, wx);
        i = j;
    }

    int tot = 0, rem = 0, fl = 0;
    for (const Row& r : g_rows)
        if (r.u_grid > above) {
            ++tot;
            if (r.u_stored > above) {
                ++rem;
                if (r.xfloor() > 0 && r.xfloor() <= kFloorSlack) ++fl;
            }
        }
    std::printf("\nabove %.4g ulps, scored by the grid oracle: %d rows\n", above, tot);
    std::printf("  %4d (%5.1f%%) INPUT ROUNDING -- gone once the oracle moves to the stored\n"
                "                    operand; the format cannot hold the grid double\n",
                tot - rem, tot ? 100.0 * (tot - rem) / tot : 0.0);
    std::printf("  %4d (%5.1f%%) REPRESENTATIONAL FLOOR -- within %gx of 2^p*D/|ref|; the\n"
                "                    format cannot hold the ANSWER at that magnitude\n",
                fl, tot ? 100.0 * fl / tot : 0.0, (double)kFloorSlack);
    std::printf("  %4d (%5.1f%%) LIBRARY ERROR -- neither; these are the real defects\n",
                rem - fl, tot ? 100.0 * (rem - fl) / tot : 0.0);

    if (list) {
        std::printf("\nWORST %d LIBRARY-ERROR ROWS (input rounding and the floor removed)\n",
                    list);
        std::vector<size_t> lib;
        for (size_t t = 0; t < g_rows.size(); ++t) {
            const Row& r = g_rows[t];
            if (r.u_grid > above && r.u_stored > above &&
                !(r.xfloor() > 0 && r.xfloor() <= kFloorSlack))
                lib.push_back(t);
        }
        std::sort(lib.begin(), lib.end(),
                  [](size_t a, size_t b) { return g_rows[a].u_stored > g_rows[b].u_stored; });
        std::printf("%-3s %-6s %6s %-9s %13s %13s %11s\n", "bk", "op", "pt", "family",
                    "ulps_grid", "ulps_stored", "xfloor");
        for (int t = 0; t < list && t < (int)lib.size(); ++t) {
            const Row& r = g_rows[lib[t]];
            std::printf("%-3s %-6s %6d %-9s %13.5g %13.5g %11.4g\n", r.bk, r.op, r.pt,
                        r.family, r.u_grid, r.u_stored, r.xfloor());
        }
    }
    return 0;
}
