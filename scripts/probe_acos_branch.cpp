// probe_acos_branch.cpp -- the seven BRANCH CHECK values in the acos() header
// comment of the four *_complex.hpp headers, checked mechanically instead of by
// hand, on all four backends, INCLUDING THE SIGN OF EVERY ZERO COMPONENT.
//
// Why this exists.  Re acos is now atan2(leg, Re z) with leg = sqrt(a^2 - x^2),
// a MAGNITUDE that is exactly zero on the real cut (|Re z| > 1, Im z = +-0).
// There the sign of that zero decides the whole real part -- atan2(+0, x < 0) is
// +pi and atan2(-0, x < 0) is -pi -- so the header forces it positive.  A hand
// check of seven values is not a check; this is.
//
// The sign of a zero is compared with std::signbit, not with ==.  Comparing
// +0.0 == -0.0 succeeds and is exactly the hole this probe exists to close.
//
// Reference values are the exact closed forms, not an oracle: acos(2) = i*ln(2 +
// sqrt(3)) is available to any precision from __float128 acosh, and the four cut
// points are pi and that value in the four sign combinations.  Nothing here
// depends on libquadmath's complex functions or on MPC.
//
// POISON, two of them.
//
// (1) --poison flips the expected sign of every zero component.  A build whose
// signed-zero handling is correct then FAILS: 24 of 56 components, on the clean
// tree.  That shows the sign comparison is live and is not a tautology.
//
// (2) The header's guard itself was poisoned by DELETING the
// `if (leg == 0) leg = +0` line from all four headers and re-running.  Result:
//
//     FAIL DD z = 2-0i  Re: got -0.000000e+00, want +0
//     FAIL DD z = -2-0i Re: got -3.14159...e+00 want 3.14159...e+00  rel 2.000e+00
//     FAIL FF z = 2-0i  Re: got -0.000000e+00, want +0
//     FAIL FF z = -2-0i Re: got -3.14159...e+00 want 3.14159...e+00  rel 2.000e+00
//
// so the guard is load-bearing on DD and FF: without it acos(-2 - 0i) returns
// -pi, the wrong sheet.  On QF and TF the same deletion changes nothing, because
// their renormalisation absorbs the -0 on its own ((-0) + (+0) = +0, the KI-10
// trap).  The guard is therefore PROVEN necessary on two backends and DEFENSIVE
// on the other two -- it is not claimed to be load-bearing everywhere, and it
// stays on all four so the four headers do not diverge on a signed zero.
//
//   g++ -O2 -std=c++17 -fext-numeric-literals -I include \
//       scripts/probe_acos_branch.cpp -o /tmp/probe_acos_branch -lquadmath
//
// Exit status is 0 only if every component of every case matches, value and
// sign of zero, on all four backends.

#include <quadmath.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "xp/dd_complex.hpp"
#include "xp/ff_complex.hpp"
#include "xp/qf_complex.hpp"
#include "xp/tf_complex.hpp"

namespace {

// ---------------------------------------------------------------- backends

template <class T>
struct Tr;
template <>
struct Tr<xp::DoubleDouble> {
    using C = xp::DoubleDoubleComplex;
    static const char* name() { return "DD"; }
    static int nlimb() { return 2; }
    static double limb(const xp::DoubleDouble& v, int i) { return i ? v.lo : v.hi; }
    // digits the format can carry, used only to size the value tolerance
    static double digits() { return 31.00; }
};
template <>
struct Tr<xp::FloatFloat> {
    using C = xp::FloatFloatComplex;
    static const char* name() { return "FF"; }
    static int nlimb() { return 2; }
    static double limb(const xp::FloatFloat& v, int i) { return i ? v.lo : v.hi; }
    static double digits() { return 14.00; }
};
template <>
struct Tr<xp::QuadFloat> {
    using C = xp::QuadFloatComplex;
    static const char* name() { return "QF"; }
    static int nlimb() { return 4; }
    static double limb(const xp::QuadFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : i == 2 ? v.f2 : v.f3;
    }
    static double digits() { return 29.00; }
};
template <>
struct Tr<xp::TripleFloat> {
    using C = xp::TripleFloatComplex;
    static const char* name() { return "TF"; }
    static int nlimb() { return 3; }
    static double limb(const xp::TripleFloat& v, int i) {
        return i == 0 ? v.f0 : i == 1 ? v.f1 : v.f2;
    }
    static double digits() { return 21.70; }
};

// Widen an expansion to __float128 by summing limbs.  This is the SAME widening
// that destroys the sign of a zero -- (-0.0) + (+0.0) is +0.0 -- which is why
// the sign is read off limb 0 separately below and never from the sum.
template <class T>
__float128 widen(const T& v) {
    __float128 s = 0;
    for (int i = 0; i < Tr<T>::nlimb(); ++i) s += (__float128)Tr<T>::limb(v, i);
    return s;
}
template <class T>
bool lead_negative(const T& v) {
    return std::signbit(Tr<T>::limb(v, 0));
}

// ---------------------------------------------------------------- cases

struct Case {
    const char* label;
    double re, im;     // input, written so that -0.0 survives as -0.0
    int want;          // index into the expected table below
};

// Expected values, built from exact closed forms at __float128 precision.
struct Expect {
    __float128 re, im;
    int rezero;   // 0 = not a zero, +1 = must be +0, -1 = must be -0
    int imzero;
};

const __float128 kPi =
    3.14159265358979323846264338327950288419716939937510582097494Q;

// acosh(2) = ln(2 + sqrt(3)) = 1.3169578969248167...
__float128 acosh2() { return acoshq(2.0Q); }

// The seven header cases.  Ordering matches the BRANCH CHECK comment.
Expect expected(int i) {
    const __float128 L = acosh2();
    switch (i) {
        // The three cases with a zero imaginary part all want -0, not +0.  That
        // is not a guess: it is what C99 Annex G asks for (Im acos = -Im asin and
        // Im asin(x + i0) = +0 for |x| <= 1), it is what libquadmath's cacosq
        // returns for all three, and it is what this header returned before the
        // real part was reformulated -- Im acos is the untouched exact identity.
        case 0: return {kPi / 2, 0, 0, -1};             // z = 0        -> pi/2 - 0i
        case 1: return {0, 0, +1, -1};                  // z = 1        -> +0 - 0i
        case 2: return {kPi, 0, 0, -1};                 // z = -1       -> pi - 0i
        case 3: return {0, -L, +1, 0};                  // z = 2 + 0i   -> -1.3170i
        case 4: return {0, +L, +1, 0};                  // z = 2 - 0i   -> +1.3170i
        case 5: return {kPi, -L, 0, 0};                 // z = -2 + 0i  -> pi - 1.3170i
        default: return {kPi, +L, 0, 0};                // z = -2 - 0i  -> pi + 1.3170i
    }
}

const Case kCases[] = {
    {"z = 0",       0.0, 0.0, 0},
    {"z = 1",       1.0, 0.0, 1},
    {"z = -1",     -1.0, 0.0, 2},
    {"z = 2+0i",    2.0, 0.0, 3},
    {"z = 2-0i",    2.0, -0.0, 4},
    {"z = -2+0i",  -2.0, 0.0, 5},
    {"z = -2-0i",  -2.0, -0.0, 6},
};

int g_fail = 0;
bool g_poison = false;

// One component.  `want_zero` is 0 / +1 / -1 as above.
template <class T>
void check_component(const char* backend, const char* label, const char* part,
                     const T& got, __float128 want, int want_zero) {
    if (g_poison && want_zero != 0) want_zero = -want_zero;

    const __float128 g = widen(got);

    if (want_zero != 0) {
        // Value must be exactly zero AND carry the demanded sign.  Read the sign
        // off limb 0: the widening above cannot be trusted to preserve it.
        const bool is_zero = (g == 0);
        const bool neg = lead_negative(got);
        const bool sign_ok = (want_zero < 0) == neg;
        if (!is_zero || !sign_ok) {
            std::printf("  FAIL %s %-9s %s: got %s%.6Qe, want %c0\n", backend, label,
                        part, neg ? "-" : "+", fabsq(g), want_zero < 0 ? '-' : '+');
            ++g_fail;
        }
        return;
    }

    // Non-zero component: relative, at the format's own width less a two-digit
    // margin.  This probe is a BRANCH check, not an accuracy measurement -- the
    // sweep is where accuracy is judged -- so the tolerance is deliberately
    // loose and only has to catch a wrong branch, which is off by pi or by a
    // sign, not by an ulp.
    const __float128 tol = powq(10.0Q, -(Tr<T>::digits() - 2.0));
    const __float128 rel = fabsq(g - want) / fabsq(want);
    if (!(rel <= tol)) {
        std::printf("  FAIL %s %-9s %s: got %.30Qe want %.30Qe  rel %.3Qe > %.3Qe\n",
                    backend, label, part, g, want, rel, tol);
        ++g_fail;
    }
}

template <class T>
void run() {
    using C = typename Tr<T>::C;
    const char* bk = Tr<T>::name();
    for (const Case& c : kCases) {
        // Construct from the DOUBLE so the expansion splits it across limbs, the
        // same way sweep_accuracy.cpp builds its operands.  -0.0 survives T(-0.0)
        // because the leading limb is set from the double directly.
        const C z(T(c.re), T(c.im));
        const C w = xp::acos(z);
        const Expect e = expected(c.want);
        check_component<T>(bk, c.label, "Re", w.re, e.re, e.rezero);
        check_component<T>(bk, c.label, "Im", w.im, e.im, e.imzero);
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--poison") == 0) g_poison = true;

    if (g_poison)
        std::printf("POISON: every expected zero-sign is flipped; a correct build MUST fail.\n");

    run<xp::DoubleDouble>();
    run<xp::FloatFloat>();
    run<xp::QuadFloat>();
    run<xp::TripleFloat>();

    const int cases = (int)(sizeof(kCases) / sizeof(kCases[0])) * 4 * 2;
    if (g_poison) {
        // Under poison the run is EXPECTED to fail.  Report the inversion so the
        // caller can assert on exit status either way.
        std::printf("poisoned run: %d of %d components failed\n", g_fail, cases);
        return g_fail > 0 ? 0 : 1;
    }
    std::printf("%s: %d components checked, %d failed\n", g_fail ? "FAIL" : "PASS",
                cases, g_fail);
    return g_fail ? 1 : 0;
}
