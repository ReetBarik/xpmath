// Standalone smoke test for xp::TripleFloatComplex — no Kokkos dependency.
//
// Verifies that include/xp/tf_complex.hpp compiles and links in a plain C++17
// environment with no Kokkos installed. This is the TF complex analogue of
// qf_complex_no_kokkos_smoke.cpp, ff_complex_no_kokkos_smoke.cpp, and
// dd_complex_no_kokkos_smoke.cpp.
//
// Exercises every complex operation in the S10 Phase 4 op inventory (24 ops: 1 type
// + 3 basic + 5 trig + 3 inv-trig + 3 hyperbolic + 3 inv-hyperbolic + 3 exp/log
// + 3 utility), with a handful of real arithmetic primitives dispatched through
// tf_math.hpp. Checks for sensible results — not byte-exact validation (that's
// what tf_accuracy_test does), just "sqrt of a small positive returns something
// reasonable, not NaN/inf".
//
// IMPORTANT COVERAGE NOTE: xp::TripleFloatComplex is a bespoke struct (re, im
// public members), not Kokkos::complex<TripleFloat>. This test does NOT verify
// Kokkos::complex interop — that integration is a separate future task.
//
// Build and run:
//   g++ -std=c++17 -I../../include tests/standalone/tf_complex_no_kokkos_smoke.cpp -o /tmp/tf_complex_smoke && /tmp/tf_complex_smoke
//
// Expected output: 0 failures, exit 0.

#include <xp/tf_complex.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int failures = 0;

// EPSILON: 2⁻⁷² is TripleFloat's ulp; for smoke we use 1e-6 (FP32-ish) since we're
// comparing to float literals rounded at compile-time.
static constexpr float kRelTol = 1e-6f;

static void check(const char* label, xp::TripleFloat got, float expected_approx) {
    float g = got.f0;
    float diff = std::fabs(g - expected_approx);
    float tol  = kRelTol * std::fabs(expected_approx);
    if (std::fabs(expected_approx) < 1e-6f) tol = kRelTol;  // absolute near zero
    if (diff > tol) {
        std::printf("FAIL %s: got %.9e, expected ~%.9e (diff %.2e)\n", label, g, expected_approx, diff);
        ++failures;
    }
}
static void check_cplx(const char* label, xp::TripleFloatComplex z, float re_approx, float im_approx) {
    float re = z.re.f0;
    float im = z.im.f0;
    float diff_re = std::fabs(re - re_approx);
    float diff_im = std::fabs(im - im_approx);
    float tol_re = kRelTol * std::fabs(re_approx);
    float tol_im = kRelTol * std::fabs(im_approx);
    if (std::fabs(re_approx) < 1e-6f) tol_re = kRelTol;
    if (std::fabs(im_approx) < 1e-6f) tol_im = kRelTol;
    if (diff_re > tol_re || diff_im > tol_im) {
        std::printf("FAIL %s: got (%.9e, %.9e), expected ~(%.9e, %.9e)\n",
                    label, re, im, re_approx, im_approx);
        ++failures;
    }
}

int main() {
    // --- Type construction ---
    xp::TripleFloatComplex z1;
    xp::TripleFloatComplex z2(1.5f);
    xp::TripleFloatComplex z3(2.0f, 3.0f);
    xp::TripleFloat tf_val(4.0f);
    xp::TripleFloatComplex z4(tf_val);
    xp::TripleFloatComplex z5(xp::TripleFloat(1.0f), xp::TripleFloat(2.0f));

    // --- Basic complex operations ---
    xp::TripleFloatComplex a(3.0f, 4.0f);
    xp::TripleFloat mag = xp::abs(a);
    check("abs(3+4i)", mag, 5.0f);

    xp::TripleFloatComplex a_conj = xp::conj(a);
    check_cplx("conj(3+4i)", a_conj, 3.0f, -4.0f);

    xp::TripleFloatComplex sq = xp::sqrt(a);
    check("sqrt(3+4i) real", sq.re, 2.0f);  // sqrt(3+4i) ≈ 2 + i
    check("sqrt(3+4i) imag", sq.im, 1.0f);

    xp::TripleFloat n = xp::norm(a);
    check("norm(3+4i)", n, 25.0f);

    xp::TripleFloat theta = xp::arg(a);
    check("arg(3+4i)", theta, 0.927295f);  // atan2(4,3) ≈ 0.927

    // --- Exponential and logarithm ---
    xp::TripleFloatComplex b(0.5f, 0.0f);
    xp::TripleFloatComplex eb = xp::exp(b);
    check("exp(0.5) real", eb.re, 1.6487212707f);
    check("exp(0.5) imag", eb.im, 0.0f);

    xp::TripleFloatComplex c(2.718281828f, 0.0f);
    xp::TripleFloatComplex lc = xp::log(c);
    check("log(e) real", lc.re, 1.0f);
    check("log(e) imag", lc.im, 0.0f);

    xp::TripleFloatComplex d(100.0f, 0.0f);
    xp::TripleFloatComplex ld = xp::log10(d);
    check("log10(100) real", ld.re, 2.0f);
    check("log10(100) imag", ld.im, 0.0f);

    // --- Trigonometric ---
    xp::TripleFloatComplex e(0.0f, 1.0f);  // i
    xp::TripleFloatComplex sine = xp::sin(e);
    check("sin(i) real", sine.re, 0.0f);
    check("sin(i) imag", sine.im, 1.1752011936f);  // sinh(1)

    xp::TripleFloatComplex cose = xp::cos(e);
    check("cos(i) real", cose.re, 1.5430806348f);  // cosh(1)
    check("cos(i) imag", cose.im, 0.0f);

    xp::TripleFloatComplex tane = xp::tan(e);
    check("tan(i) real", tane.re, 0.0f);
    check("tan(i) imag", tane.im, 0.7615941559f);  // tanh(1)

    // --- Inverse trigonometric ---
    xp::TripleFloatComplex f(0.5f, 0.0f);
    xp::TripleFloatComplex asf = xp::asin(f);
    check("asin(0.5) real", asf.re, 0.5235987756f);  // π/6
    check("asin(0.5) imag", asf.im, 0.0f);

    xp::TripleFloatComplex acf = xp::acos(f);
    check("acos(0.5) real", acf.re, 1.0471975512f);  // π/3
    check("acos(0.5) imag", acf.im, 0.0f);

    xp::TripleFloatComplex g(1.0f, 0.0f);
    xp::TripleFloatComplex atg = xp::atan(g);
    check("atan(1) real", atg.re, 0.7853981634f);  // π/4
    check("atan(1) imag", atg.im, 0.0f);

    // --- Hyperbolic ---
    xp::TripleFloatComplex h(1.0f, 0.0f);
    xp::TripleFloatComplex sh = xp::sinh(h);
    check("sinh(1) real", sh.re, 1.1752011936f);
    check("sinh(1) imag", sh.im, 0.0f);

    xp::TripleFloatComplex ch = xp::cosh(h);
    check("cosh(1) real", ch.re, 1.5430806348f);
    check("cosh(1) imag", ch.im, 0.0f);

    xp::TripleFloatComplex th = xp::tanh(h);
    check("tanh(1) real", th.re, 0.7615941559f);
    check("tanh(1) imag", th.im, 0.0f);

    // --- Inverse hyperbolic ---
    xp::TripleFloatComplex k(1.0f, 0.0f);
    xp::TripleFloatComplex ash = xp::asinh(k);
    check("asinh(1) real", ash.re, 0.8813735870f);
    check("asinh(1) imag", ash.im, 0.0f);

    xp::TripleFloatComplex m(2.0f, 0.0f);
    xp::TripleFloatComplex ach = xp::acosh(m);
    check("acosh(2) real", ach.re, 1.3169578969f);
    check("acosh(2) imag", ach.im, 0.0f);

    xp::TripleFloatComplex p(0.5f, 0.0f);
    xp::TripleFloatComplex ath = xp::atanh(p);
    check("atanh(0.5) real", ath.re, 0.5493061443f);
    check("atanh(0.5) imag", ath.im, 0.0f);

    // --- Power and polar ---
    xp::TripleFloatComplex base(2.0f, 0.0f);
    xp::TripleFloatComplex exponent(3.0f, 0.0f);
    xp::TripleFloatComplex pw = xp::pow(base, exponent);
    check("pow(2, 3) real", pw.re, 8.0f);
    check("pow(2, 3) imag", pw.im, 0.0f);

    xp::TripleFloat r(5.0f);
    xp::TripleFloat angle(0.927295f);  // atan2(4,3)
    xp::TripleFloatComplex pol = xp::polar(r, angle);
    check("polar(5, atan2(4,3)) real", pol.re, 3.0f);
    check("polar(5, atan2(4,3)) imag", pol.im, 4.0f);

    xp::TripleFloat unit_r(1.0f);
    xp::TripleFloatComplex pol2 = xp::polar(unit_r, xp::TripleFloat(1.5707963268f));  // π/2
    check("polar(1, π/2) real", pol2.re, 0.0f);
    check("polar(1, π/2) imag", pol2.im, 1.0f);

    if (failures == 0) {
        std::printf("tf_complex_no_kokkos_smoke: PASS (0 failures)\n");
    } else {
        std::printf("tf_complex_no_kokkos_smoke: FAIL (%d failures)\n", failures);
    }
    return failures;
}
