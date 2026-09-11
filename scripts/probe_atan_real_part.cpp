// probe_atan_real_part.cpp -- ablate the two closed forms for Re atan(x+iy),
// in each backend's own arithmetic, before either of them is shipped.
//
// THE DEFECT.  QF atan reads 7380 ulps at z = (1e-23, -1) and QF atanh reads
// the same 7380 at z = (-1, 1e-23) -- 48 scored rows between them, the largest
// surviving library-error cluster in complex trig above 8 ulps.  atanh's
// imaginary part IS atan's real part (atanh(z) = i atan(-iz)), so it is one
// mechanism seen twice, and both headers compute it the same way:
//
//     FORM A   Re atan = 0.5 * atan2(2x, (1-y)(1+y) - x^2)
//
// The factored (1-y)(1+y) is already there and is already right -- it is exact
// by Sterbenz on both factors, and the comment in the header explains why the
// unfactored 1-(x^2+y^2) is not.  The failure is one term further in.  At
// x = 1e-23 the product x^2 = 1e-46 is BELOW FP32's smallest subnormal,
// 2^-149 = 1.4e-45, so multiply(z.re, z.re) flushes to zero.  With y = -1
// exactly, (1-y)(1+y) is exactly zero too, so the second argument is 0 - 0 = 0
// and atan2(2e-23, 0) returns pi/2.  The correct second argument is -1e-46,
// which puts the angle at pi/2 + 5e-24, so Re lands on pi/4 instead of
// pi/4 + 2.5e-24.  Measured component error 2.522e5 ulps; 2.5e-24 / (pi/4) *
// 2^96 = 2.5e5.  The arithmetic and the measurement agree.
//
// This is NOT a representational floor of the answer.  pi/4 + 2.5e-24 is
// comfortably representable in QuadFloat -- it is 2.5e-24 relative to 0.785,
// i.e. 3.2e-24, and QF resolves 2^-96 = 1.3e-29.  Only the INTERMEDIATE x^2 is
// unrepresentable.  A formulation that never forms x^2 is therefore not
// fighting the format.
//
//     FORM B   Re atan = 0.5 * (atan2(x, 1-y) + atan2(x, 1+y))
//
// from atan(z) = (i/2)[log(1-iz) - log(1+iz)], taking imaginary parts of the
// two logs.  At the same point: atan2(1e-23, 2) = 5e-24 and atan2(1e-23, 0) =
// pi/2, summing to pi/2 + 5e-24 and halving to exactly the right answer.  No
// square is formed, so nothing underflows.
//
// Form B also has no cancellation anywhere: both terms carry the sign of x, so
// the sum is a sum of like-signed quantities, never a difference.
//
// AND IT GETS THE CUT RIGHT ON ITS OWN.  Form A needs an explicit override for
// Re(z) a signed zero with |Im z| > 1, because atan2 is handed the zero but
// does not carry its sign into the result; the header installs +-pi/2 by hand
// and records that the monotone gate is what caught the omission.  Form B needs
// no override: at z = +0 + 2i it reads atan2(+0,-1) + atan2(+0,3) = pi + 0, so
// pi/2; at z = -0 + 2i it reads atan2(-0,-1) + atan2(-0,3) = -pi + -0, so
// -pi/2.  The signed zero flows through atan2's own y-argument, which is where
// atan2 does respect it.  The override is kept anyway -- see below.
//
// WHAT THIS PROBE MEASURES.  All three forms, evaluated in each backend's own
// arithmetic with the backend's own xp_atan2_safe, over the whole 1780-point
// complex grid, scored against MPC's Re atan at the STORED operand.  A form is
// only worth shipping if it is better where it is supposed to be better and no
// worse anywhere else; that second half is the point of running the full grid
// rather than the four points the defect lives at.
//
// RESULT, and what shipped.  Form B is NOT a straight win: it fixes the cut but
// it is worse nearly everywhere else, taking the QF maximum from 2.58e7 to
// 1.56e8, TF from 1.5 to 9.3, DD point 6 from 1.57 to 5.25, and every median
// with it.  So form C ships -- form A, with form B substituted only where the
// x^2 is provably gone AND the quantity it is subtracted from has collapsed
// too.  Measured over the whole grid, per backend, on the real part alone:
//
//     bk   median A   median C      max A         max B        max C
//     DD     0.1083     0.1083      5.6453        5.2486       5.6453
//     FF     0.1901     0.1901      3.4307        5.1629       3.4307
//     QF     0.0141     0.0141      2.5753e+07    1.5556e+08   2.5753e+07
//     TF     0.0392     0.0392      1.5350        9.2721       1.5350
//
// DD and FF are bit-identical to A -- the guard never fires on them at FP64
// width.  The guard is shipped on all four backends anyway, so the four headers
// do not diverge on a branch cut.  QF's max is UNCHANGED at 2.58e7 because that
// maximum is not the defect: it is QF point 8, z = (6.1e-25, 1e-08), where the
// component floor 2^-53/6.1e-25 is 1.8e8 and BOTH forms sit below it -- KI-33's
// representational floor, not a library error.  Requiring both conditions is
// what keeps the guard off that point; an earlier version gated on the lost
// x^2 alone and made it 6x worse for nothing.
//
// The 48 grid rows this was built for went 7380.3 -> 0.0646 modulus-relative
// ulps in the sweep.  The only rows that moved the wrong way are 8 TF rows at
// 0.077219 -> 0.077308, a change in the fifth figure, 13x below the 1-ulp line.
//
// form_a() below is a transcription of what the headers shipped BEFORE that
// change, kept so the ablation stays runnable and re-checkable.
//
// NOT A GATE.  docs/CORRECTNESS.md allows one measurement and one verdict; this
// is neither.
//
// POISON.  --poison swaps form B's two denominators, computing
// 0.5*(atan2(x,1+y) + atan2(x,1-y)) with the 1-y and 1+y exchanged relative to
// the derivation.  That is a real transcription error of exactly the kind this
// probe exists to catch, and it must show up as a large regression -- if the
// two forms read the same under it, the probe is not sensitive to which
// denominator goes where and proves nothing about form B being right.
// (The sum is symmetric, so the poison instead swaps ONE of them for its
// negation: 0.5*(atan2(x,1-y) + atan2(x,-(1+y))), which is the same class of
// slip and is not symmetric.)
//
// BUILD.  -DXPMATH_ENABLE_DIAGNOSTICS=0 silences the domain printfs
// (config.hpp:121 -- return values are unaffected).
//
//   g++ -O2 -std=c++17 -fext-numeric-literals -DXPMATH_ENABLE_DIAGNOSTICS=0 \
//       -I include scripts/probe_atan_real_part.cpp -o /tmp/probe_atan \
//       -lmpc -lmpfr -lgmp
//
// RUN from the repo root:
//   /tmp/probe_atan
//   /tmp/probe_atan --poison
//   /tmp/probe_atan --csv /tmp/atan.csv

#include <mpc.h>
#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "xp/dd_complex.hpp"
#include "xp/ff_complex.hpp"
#include "xp/qf_complex.hpp"
#include "xp/tf_complex.hpp"

namespace {

const char* const kGrid = "validation/sweep/sweep_grid.csv";
const mpfr_prec_t kPrec = 400;
bool g_poison = false;

template <class T>
struct Tr;
template <>
struct Tr<xp::DoubleDouble> {
    static const char* name() { return "DD"; }
    static int sig_bits() { return 106; }
    static int nlimb() { return 2; }
    static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::FloatFloat> {
    static const char* name() { return "FF"; }
    static int sig_bits() { return 48; }
    static int nlimb() { return 2; }
    static double limb(const xp::FloatFloat& v, int i) { return i ? v.lo : v.hi; }
};
template <>
struct Tr<xp::QuadFloat> {
    static const char* name() { return "QF"; }
    static int sig_bits() { return 96; }
    static int nlimb() { return 4; }
    static double limb(const xp::QuadFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : i == 2 ? v.f2 : v.f3;
    }
};
template <>
struct Tr<xp::TripleFloat> {
    static const char* name() { return "TF"; }
    static int sig_bits() { return 72; }
    static int nlimb() { return 3; }
    static double limb(const xp::TripleFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : v.f2;
    }
};

template <class T>
void widen(const T& v, mpfr_ptr out) {
    mpfr_set_d(out, Tr<T>::limb(v, 0), MPFR_RNDN);
    for (int i = 1; i < Tr<T>::nlimb(); ++i) mpfr_add_d(out, out, Tr<T>::limb(v, i), MPFR_RNDN);
    if (mpfr_zero_p(out) && std::signbit(Tr<T>::limb(v, 0)))
        mpfr_setsign(out, out, 1, MPFR_RNDN);
}

// leading word of an expansion, for the branch tests the headers use
template <class T>
double lead(const T& v) {
    return Tr<T>::limb(v, 0);
}

double ulps_real(mpfr_srcptr got, mpfr_srcptr ref, int p) {
    static bool init = false;
    static mpfr_t e;
    if (!init) { mpfr_init2(e, kPrec); init = true; }
    if (!mpfr_number_p(ref) || !mpfr_number_p(got)) return -1.0;
    if (mpfr_zero_p(ref)) return mpfr_zero_p(got) ? 0.0 : -1.0;
    mpfr_sub(e, got, ref, MPFR_RNDN);
    mpfr_div(e, e, ref, MPFR_RNDN);
    mpfr_abs(e, e, MPFR_RNDN);
    mpfr_mul_2si(e, e, p, MPFR_RNDN);
    return mpfr_get_d(e, MPFR_RNDN);
}

// FORM A, transcribed from the shipped headers (dd_complex.hpp:994-1006 and the
// FF/QF/TF siblings), INCLUDING the signed-zero cut override.
template <class T>
T form_a(const T& x, const T& y) {
    const T one(static_cast<decltype(lead(x))>(1.0));
    const T x2 = xp::multiply(x, x);
    T twox = xp::multiply_scalar(x, static_cast<decltype(lead(x))>(2.0));
    if (lead(x) == 0.0) twox = x;
    const T d2 = xp::subtract(xp::multiply(xp::subtract(one, y), xp::add(one, y)), x2);
    T re = xp::multiply_scalar(xp::xp_atan2_safe(twox, d2),
                               static_cast<decltype(lead(x))>(0.5));
    if (lead(x) == 0.0 && lead(d2) < 0.0) {
        re = xp::multiply_scalar(xp::xp_atan2_safe(T(static_cast<decltype(lead(x))>(1.0)),
                                                   T(static_cast<decltype(lead(x))>(0.0))),
                                 static_cast<decltype(lead(x))>(1.0));
        if (std::signbit(lead(x))) re = xp::negate(re);
    }
    return re;
}

// FORM B.  0.5 * (atan2(x, 1-y) + atan2(x, 1+y)).
template <class T>
T form_b(const T& x, const T& y) {
    using S = decltype(lead(x));
    const T one(static_cast<S>(1.0));
    T omy = xp::subtract(one, y);
    T opy = xp::add(one, y);
    if (g_poison) opy = xp::negate(opy);
    return xp::multiply_scalar(
        xp::add(xp::xp_atan2_safe(x, omy), xp::xp_atan2_safe(x, opy)), static_cast<S>(0.5));
}

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
        p.re = std::strtod(fields[3], nullptr);
        p.im = std::strtod(fields[4], nullptr);
        v.push_back(p);
    }
    std::fclose(f);
    return v;
}

// FORM C.  Form A, except where form A has provably lost its second argument.
// The trigger is the failure itself, not a magnitude guess: x^2 is formed, and
// if its LEADING WORD has left the format's normal range -- zero, or subnormal,
// where an FP32 word carries only a handful of bits -- then (1-y)(1+y) - x^2
// cannot be trusted at |y| ~ 1 and form B is used instead.  Everywhere else
// form A is kept unchanged, which is what keeps form B's regressions out.
template <class T>
bool x2_lost(const T& x2) {
    const double w = std::fabs(lead(x2));
    const double tiny = std::is_same<T, xp::DoubleDouble>::value ? 2.2250738585072014e-308
                                                                 : 1.1754943508222875e-38;
    return w < tiny;   // covers w == 0 and the subnormal range
}

template <class T>
T form_c(const T& x, const T& y) {
    if (lead(x) == 0.0) return form_a<T>(x, y);
    const T one(static_cast<decltype(lead(x))>(1.0));
    // BOTH conditions are needed.  x^2 leaving the normal range is not by
    // itself a defect: at z = (6.1e-25, 1e-08) the product underflows too, but
    // (1-y)(1+y) is ~1 there and the lost x^2 was genuinely negligible against
    // it -- dropping it is the CORRECT answer, and switching forms there made
    // the reading 6x worse for nothing.  The lost term only matters when the
    // quantity it is subtracted from has itself collapsed, which is what
    // happens at |y| = 1, on the cut, where the whole answer is the x^2.
    const T pr = xp::multiply(xp::subtract(one, y), xp::add(one, y));
    if (x2_lost(xp::multiply(x, x)) && x2_lost(pr)) return form_b<T>(x, y);
    return form_a<T>(x, y);
}

struct Row {
    const char* bk;
    int pt;
    const char* family;
    double a, b, c;
};
std::vector<Row> g_rows;

template <class T>
void run(const std::vector<Pt>& grid) {
    const int p = Tr<T>::sig_bits();
    mpc_t zs, r;
    mpfr_t wa, wb, wc;
    mpc_init2(zs, kPrec);
    mpc_init2(r, kPrec);
    mpfr_inits2(kPrec, wa, wb, wc, (mpfr_ptr)0);
    for (const Pt& pt : grid) {
        const T x(pt.re), y(pt.im);
        widen(x, mpc_realref(zs));
        widen(y, mpc_imagref(zs));
        mpc_atan(r, zs, MPC_RNDNN);
        const T a = form_a<T>(x, y);
        const T b = form_b<T>(x, y);
        const T c = form_c<T>(x, y);
        widen(a, wa);
        widen(b, wb);
        widen(c, wc);
        const double ua = ulps_real(wa, mpc_realref(r), p);
        const double ub = ulps_real(wb, mpc_realref(r), p);
        const double uc = ulps_real(wc, mpc_realref(r), p);
        if (ua < 0 || ub < 0 || uc < 0) continue;
        g_rows.push_back({Tr<T>::name(), pt.idx, pt.family.c_str(), ua, ub, uc});
    }
    mpfr_clears(wa, wb, wc, (mpfr_ptr)0);
    mpc_clear(zs);
    mpc_clear(r);
}

double med(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const char* csv = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--poison")) g_poison = true;
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
    }
    const std::vector<Pt> grid = load_grid();
    std::printf("# grid %zu complex points, MPC %s at %d bits%s\n", grid.size(),
                MPC_VERSION_STRING, (int)kPrec, g_poison ? "   [POISONED form B]" : "");

    run<xp::DoubleDouble>(grid);
    run<xp::FloatFloat>(grid);
    run<xp::QuadFloat>(grid);
    run<xp::TripleFloat>(grid);

    if (csv) {
        FILE* f = std::fopen(csv, "w");
        if (!f) { std::perror(csv); return 2; }
        std::fprintf(f, "backend,point,family,ulps_a,ulps_b,ulps_c\n");
        for (const Row& r : g_rows)
            std::fprintf(f, "%s,%d,%s,%.6g,%.6g,%.6g\n", r.bk, r.pt, r.family, r.a, r.b, r.c);
        std::fclose(f);
        std::printf("wrote %zu rows to %s\n", g_rows.size(), csv);
    }

    std::printf("\nRe atan(x+iy): FORM A (shipped) vs FORM B (two atan2)\n");
    std::printf("%-4s %7s %9s %9s %11s %11s %11s %8s %8s\n", "bk", "rows", "med A", "med C",
                "max A", "max B", "max C", "C worse", "C >2x A");
    long tot_worse = 0;
    for (const char* bk : {"DD", "FF", "QF", "TF"}) {
        std::vector<double> A, C;
        double ma = 0, mb = 0, mc = 0;
        long worse = 0, worse2 = 0;
        for (const Row& r : g_rows) {
            if (std::strcmp(r.bk, bk)) continue;
            A.push_back(r.a);
            C.push_back(r.c);
            ma = std::max(ma, r.a);
            mb = std::max(mb, r.b);
            mc = std::max(mc, r.c);
            if (r.c > r.a) ++worse;
            if (r.c > 2.0 * r.a && r.c > 1.0) ++worse2;
        }
        if (A.empty()) continue;
        tot_worse += worse2;
        std::printf("%-4s %7zu %9.4f %9.4f %11.5g %11.5g %11.5g %8ld %8ld\n", bk, A.size(),
                    med(A), med(C), ma, mb, mc, worse, worse2);
    }

    std::printf("\nthe four defect points\n");
    for (const Row& r : g_rows)
        if (r.pt == 1264 || r.pt == 624)
            std::printf("  %s pt %-5d %-7s A %12.6g   B %12.6g   C %12.6g\n", r.bk, r.pt,
                        r.family, r.a, r.b, r.c);

    if (g_poison) {
        std::printf("\nPOISON: form B's second denominator negated.  It must now be\n"
                    "materially worse than form A somewhere, or the probe cannot tell\n"
                    "a right transcription from a wrong one.\n");
        std::printf("  rows where C is above 1 ulp and more than 2x A: %ld\n", tot_worse);
        std::printf("%s\n", tot_worse > 0 ? "PASS -- the comparison is live." : "FAIL");
        return tot_worse > 0 ? 0 : 1;
    }
    return 0;
}
