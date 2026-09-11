// ============================================================================
// pow_domain_test.cpp — pow()'s non-positive-base contract, all four backends.
// ============================================================================
//
// WHY THIS EXISTS AND WHY THE SWEEP CANNOT REPLACE IT
// ----------------------------------------------------
// scripts/sweep_accuracy.cpp:529-537 (repair_real, case R_Pow) forces
// a = fabs(a) and maps a == 0 to 1 before any backend sees it. No grid point
// therefore ever presents pow with a zero or negative base, and the whole
// domain contract is structurally invisible to the sweep -- which is how TF
// shipped without the guard the other three have.
//
// TF's bare body returned exp(b * log(0)) = exp(b * 0) = 1 for pow(0, b),
// against 0 from DD/FF/QF, with only TFLOG's diagnostic printed. This test
// pins the shared contract so the next backend cannot drift the same way.
//
// THE CONTRACT (as DD/FF/QF have implemented it since the KI-19-era audit):
//     a > 0            -> exp(b * log a), the normal path
//     a == 0, b > 0    -> 0        (0^positive = 0)
//     a == 0, b <= 0   -> 0, with a diagnostic   (0^0 and 0^negative are not
//                                                 defined; the library reports
//                                                 rather than inventing 1 or inf)
//     a < 0            -> 0, with a diagnostic   (real pow of a negative base
//                                                 is complex; use the complex
//                                                 header)
//
// This is a CONTRACT test, not an accuracy test: every assertion is an exact
// comparison against a specified value, so it issues no competing opinion on
// accuracy (docs/CORRECTNESS.md:5-6 reserves that for the sweep).
// ============================================================================
#include <cstdio>
#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>
#include <xp/tf_math.hpp>
#include <xp/qf_math.hpp>

using namespace xp;

static int failures = 0;
static int checks   = 0;

static void expect_zero(const char* what, double got) {
    ++checks;
    if (got != 0.0) {
        ++failures;
        std::printf("  FAIL: %s -> %g, expected 0\n", what, got);
    }
}

int main() {
    std::printf("pow_domain_test - non-positive-base contract, four backends\n");
    std::printf("(diagnostics printed below are EXPECTED; they are the contract)\n\n");

    // ---- a == 0, b > 0 : exactly zero, no diagnostic --------------------
    std::printf("a == 0, b > 0  (expect 0, silent)\n");
    expect_zero("DD pow(0, 2)", xp::pow(DoubleDouble(0.0), DoubleDouble(2.0)).hi);
    expect_zero("FF pow(0, 2)", (double)xp::pow(FloatFloat(0.0f), FloatFloat(2.0f)).hi);
    expect_zero("TF pow(0, 2)", (double)xp::pow(TripleFloat(0.0f), TripleFloat(2.0f)).f0);
    expect_zero("QF pow(0, 2)", (double)xp::pow(QuadFloat(0.0f), QuadFloat(2.0f)).f0);

    // ---- a == 0, b <= 0 : zero WITH a diagnostic ------------------------
    std::printf("\na == 0, b <= 0  (expect 0 + diagnostic)\n");
    expect_zero("DD pow(0, -1)", xp::pow(DoubleDouble(0.0), DoubleDouble(-1.0)).hi);
    expect_zero("FF pow(0, -1)", (double)xp::pow(FloatFloat(0.0f), FloatFloat(-1.0f)).hi);
    expect_zero("TF pow(0, -1)", (double)xp::pow(TripleFloat(0.0f), TripleFloat(-1.0f)).f0);
    expect_zero("QF pow(0, -1)", (double)xp::pow(QuadFloat(0.0f), QuadFloat(-1.0f)).f0);

    // ---- a < 0 : zero WITH a diagnostic ---------------------------------
    // This is the case TF got wrong: without the guard it computed
    // exp(b * log(-2)) = exp(b * 0) = 1.
    std::printf("\na < 0  (expect 0 + diagnostic; TF returned 1 before the guard)\n");
    expect_zero("DD pow(-2, 3)", xp::pow(DoubleDouble(-2.0), DoubleDouble(3.0)).hi);
    expect_zero("FF pow(-2, 3)", (double)xp::pow(FloatFloat(-2.0f), FloatFloat(3.0f)).hi);
    expect_zero("TF pow(-2, 3)", (double)xp::pow(TripleFloat(-2.0f), TripleFloat(3.0f)).f0);
    expect_zero("QF pow(-2, 3)", (double)xp::pow(QuadFloat(-2.0f), QuadFloat(3.0f)).f0);

    expect_zero("DD pow(-0.5, 2)", xp::pow(DoubleDouble(-0.5), DoubleDouble(2.0)).hi);
    expect_zero("FF pow(-0.5, 2)", (double)xp::pow(FloatFloat(-0.5f), FloatFloat(2.0f)).hi);
    expect_zero("TF pow(-0.5, 2)", (double)xp::pow(TripleFloat(-0.5f), TripleFloat(2.0f)).f0);
    expect_zero("QF pow(-0.5, 2)", (double)xp::pow(QuadFloat(-0.5f), QuadFloat(2.0f)).f0);

    // ---- the positive path still works ----------------------------------
    // Not an accuracy assertion: 2^3 = 8 is exactly representable in every
    // backend, so this is a bit-equality like the rest of the file.
    std::printf("\na > 0  (the normal path must be untouched)\n");
    {
        ++checks;
        const double dd = xp::pow(DoubleDouble(2.0), DoubleDouble(3.0)).hi;
        if (dd != 8.0) { ++failures; std::printf("  FAIL: DD pow(2,3) -> %g, expected 8\n", dd); }
        ++checks;
        const float ff = xp::pow(FloatFloat(2.0f), FloatFloat(3.0f)).hi;
        if (ff != 8.0f) { ++failures; std::printf("  FAIL: FF pow(2,3) -> %g, expected 8\n", (double)ff); }
        ++checks;
        const float tf = xp::pow(TripleFloat(2.0f), TripleFloat(3.0f)).f0;
        if (tf != 8.0f) { ++failures; std::printf("  FAIL: TF pow(2,3) -> %g, expected 8\n", (double)tf); }
        ++checks;
        const float qf = xp::pow(QuadFloat(2.0f), QuadFloat(3.0f)).f0;
        if (qf != 8.0f) { ++failures; std::printf("  FAIL: QF pow(2,3) -> %g, expected 8\n", (double)qf); }
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
