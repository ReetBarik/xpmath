// Standalone smoke test for xp::QuadFloatComplex — no Kokkos dependency.
//
// Verifies that include/xp/qf_complex.hpp compiles and links in a plain C++17
// environment with no Kokkos installed. This is the QF complex analogue of
// dd_complex_no_kokkos_smoke.cpp and ff_complex_no_kokkos_smoke.cpp.
//
// Exercises every complex operation in the T3.0c op inventory (24 ops: 1 type
// + 3 basic + 5 trig + 3 inv-trig + 3 hyperbolic + 3 inv-hyperbolic + 3 exp/log
// + 3 utility), with a handful of real arithmetic primitives dispatched through
// qf_math.hpp. Checks for sensible results — not byte-exact validation (that's
// what qf_accuracy_test does), just "sqrt of a small positive returns something
// reasonable, not NaN/inf".
//
// IMPORTANT COVERAGE NOTE (from the T3.0c complex task): xp::QuadFloatComplex
// is a bespoke struct (re, im public members), not Kokkos::complex<QuadFloat>.
// This test does NOT verify Kokkos::complex interop — that integration is a
// separate future task.
//
// Build and run:
//   g++ -std=c++17 -I../../include tests/standalone/qf_complex_no_kokkos_smoke.cpp -o /tmp/qf_complex_smoke && /tmp/qf_complex_smoke
//
// Expected output: 0 failures, exit 0.

#include <xp/qf_complex.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int failures = 0;

// EPSILON: 2⁻⁹⁶ is QuadFloat's ulp; for smoke we use 1e-6 (FP32-ish) since we're
// comparing to float literals rounded at compile-time.
static constexpr float kRelTol = 1e-6f;

static void check(const char* label, xp::QuadFloat got, float expected_approx) {
    float g = got.f0;
    float diff = std::fabs(g - expected_approx);
    float tol  = kRelTol * std::fabs(expected_approx);
    if (std::fabs(expected_approx) < 1e-6f) tol = kRelTol;  // absolute near zero
    if (diff > tol) {
        std::printf("FAIL %s: got %.9e, expected ~%.9e (diff %.2e)\n", label, g, expected_approx, diff);
        ++failures;
    }
}
static void check_cplx(const char* label, xp::QuadFloatComplex z, float re_approx, float im_approx) {
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
    xp::QuadFloatComplex z1;
    xp::QuadFloatComplex z2(1.5f);
    xp::QuadFloatComplex z3(2.0f, 3.0f);
    xp::QuadFloat qf_val(4.0f);
    xp::QuadFloatComplex z4(qf_val);
    xp::QuadFloatComplex z5(xp::QuadFloat(1.0f), xp::QuadFloat(2.0f));

    // --- Basic complex operations ---
    xp::QuadFloatComplex a(3.0f, 4.0f);
    xp::QuadFloat mag = xp::abs(a);
    check("abs(3+4i)", mag, 5.0f);

    xp::QuadFloatComplex a_conj = xp::conj(a);
    check_cplx("conj(3+4i)", a_conj, 3.0f, -4.0f);

    xp::QuadFloatComplex sq = xp::sqrt(a);
    check("sqrt(3+4i) real", sq.re, 2.0f);  // sqrt(3+4i) ≈ 2 + i
    check("sqrt(3+4i) imag", sq.im, 1.0f);

    xp::QuadFloat n = xp::norm(a);
    check("norm(3+4i)", n, 25.0f);

    xp::QuadFloat theta = xp::arg(a);
    check("arg(3+4i)", theta, 0.927295f);  // atan2(4,3) ≈ 0.927

    // --- Exponential and logarithm ---
    xp::QuadFloatComplex b(0.5f, 0.0f);
    xp::QuadFloatComplex eb = xp::exp(b);
    check("exp(0.5) real", eb.re, 1.6487212707f);
    check("exp(0.5) imag", eb.im, 0.0f);

    xp::QuadFloatComplex c(2.718281828f, 0.0f);
    xp::QuadFloatComplex lc = xp::log(c);
    check("log(e) real", lc.re, 1.0f);
    check("log(e) imag", lc.im, 0.0f);

    xp::QuadFloatComplex d(100.0f, 0.0f);
    xp::QuadFloatComplex ld = xp::log10(d);
    check("log10(100) real", ld.re, 2.0f);
    check("log10(100) imag", ld.im, 0.0f);

    // --- Trigonometric ---
    xp::QuadFloatComplex e(0.0f, 1.0f);  // i
    xp::QuadFloatComplex sine = xp::sin(e);
    check("sin(i) real", sine.re, 0.0f);
    check("sin(i) imag", sine.im, 1.1752011936f);  // sinh(1)

    xp::QuadFloatComplex cose = xp::cos(e);
    check("cos(i) real", cose.re, 1.5430806348f);  // cosh(1)
    check("cos(i) imag", cose.im, 0.0f);

    xp::QuadFloatComplex tane = xp::tan(e);
    check("tan(i) real", tane.re, 0.0f);
    check("tan(i) imag", tane.im, 0.7615941559f);  // tanh(1)

    // --- Inverse trigonometric ---
    xp::QuadFloatComplex f(0.5f, 0.0f);
    xp::QuadFloatComplex asf = xp::asin(f);
    check("asin(0.5) real", asf.re, 0.5235987756f);  // π/6
    check("asin(0.5) imag", asf.im, 0.0f);

    xp::QuadFloatComplex acf = xp::acos(f);
    check("acos(0.5) real", acf.re, 1.0471975512f);  // π/3
    check("acos(0.5) imag", acf.im, 0.0f);

    xp::QuadFloatComplex g(1.0f, 0.0f);
    xp::QuadFloatComplex atg = xp::atan(g);
    check("atan(1) real", atg.re, 0.7853981634f);  // π/4
    check("atan(1) imag", atg.im, 0.0f);

    // --- Hyperbolic ---
    xp::QuadFloatComplex h(1.0f, 0.0f);
    xp::QuadFloatComplex sh = xp::sinh(h);
    check("sinh(1) real", sh.re, 1.1752011936f);
    check("sinh(1) imag", sh.im, 0.0f);

    xp::QuadFloatComplex ch = xp::cosh(h);
    check("cosh(1) real", ch.re, 1.5430806348f);
    check("cosh(1) imag", ch.im, 0.0f);

    xp::QuadFloatComplex th = xp::tanh(h);
    check("tanh(1) real", th.re, 0.7615941559f);
    check("tanh(1) imag", th.im, 0.0f);

    // --- Inverse hyperbolic ---
    xp::QuadFloatComplex k(1.0f, 0.0f);
    xp::QuadFloatComplex ash = xp::asinh(k);
    check("asinh(1) real", ash.re, 0.8813735870f);
    check("asinh(1) imag", ash.im, 0.0f);

    xp::QuadFloatComplex m(2.0f, 0.0f);
    xp::QuadFloatComplex ach = xp::acosh(m);
    check("acosh(2) real", ach.re, 1.3169578969f);
    check("acosh(2) imag", ach.im, 0.0f);

    xp::QuadFloatComplex p(0.5f, 0.0f);
    xp::QuadFloatComplex ath = xp::atanh(p);
    check("atanh(0.5) real", ath.re, 0.5493061443f);
    check("atanh(0.5) imag", ath.im, 0.0f);

    // --- Power and polar ---
    xp::QuadFloatComplex base(2.0f, 0.0f);
    xp::QuadFloatComplex exponent(3.0f, 0.0f);
    xp::QuadFloatComplex pw = xp::pow(base, exponent);
    check("pow(2, 3) real", pw.re, 8.0f);
    check("pow(2, 3) imag", pw.im, 0.0f);

    xp::QuadFloat r(5.0f);
    xp::QuadFloat angle(0.927295f);  // atan2(4,3)
    xp::QuadFloatComplex pol = xp::polar(r, angle);
    check("polar(5, atan2(4,3)) real", pol.re, 3.0f);
    check("polar(5, atan2(4,3)) imag", pol.im, 4.0f);

    xp::QuadFloat unit_r(1.0f);
    xp::QuadFloatComplex pol2 = xp::polar(unit_r, xp::QuadFloat(1.5707963268f));  // π/2
    check("polar(1, π/2) real", pol2.re, 0.0f);
    check("polar(1, π/2) imag", pol2.im, 1.0f);

    if (failures == 0) {
        std::printf("qf_complex_no_kokkos_smoke: PASS (0 failures)\n");
    } else {
        std::printf("qf_complex_no_kokkos_smoke: FAIL (%d failures)\n", failures);
    }
    return failures;
}
