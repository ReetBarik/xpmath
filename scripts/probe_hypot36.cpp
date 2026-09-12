// ===========================================================================
// probe_hypot36 -- the one REAL row in the div-lift regression set.
//
// scripts/probe_div_downstream.cpp answers the complex side of "why does a
// more accurate real divide make the caller worse".  Eighteen of the nineteen
// rows sweep_monotone_gate reports are complex and are in that probe's reach.
// The nineteenth is not:
//
//     REGRESSION  QF r hypot  point 36   16916.8 -> 60371 ulps
//
// This probe answers that row on its own terms, and it does so by INCLUDING
// scripts/sweep_accuracy.cpp rather than paraphrasing it.  The operand pair at
// real point 36 is drawn from a per-op RNG stream (fill_real_operands, the
// i%7==1 cancelling family), so any hand-transcription of the inputs would be
// a guess.  Including the sweep gives the genuine grid, the genuine operand
// fill, the genuine oracle and the genuine ulps_scalar -- there is no second
// copy of any of them to drift.
//
// WHAT IS MEASURED, four arms of hypot at those exact operands:
//
//   lift    xp::hypot as shipped at d7ac5af (divide() with the power-of-two lift)
//   plain   the same body with detail::qf_divide_core -- the pre-lift divide
//   exact   the same body with t = n/m rounded exactly, from MPFR at 400 bits
//   FLOOR*  the exact 400-bit hypot greedily rounded into four floats: the
//           unbeatable answer in this format, whatever the algorithm
//
// plus the repo's OWN subnormal_floor_ulps() at the result magnitude -- the
// error floor sweep_accuracy already derives from the format constants and
// already folds into this row's bound.
//
// SELF-CHECKS, all hard:
//   * the `lift` arm must equal xp::hypot(a,b) BIT FOR BIT (the transcription
//     of hypot's ten-line body is otherwise unproven),
//   * `lift` must reproduce the post-lift CSV's 60371 ulps and `plain` the
//     baseline CSV's 16916.8, to the CSV's own printed precision,
//   * a WRECK arm perturbs t by (1+2^-20) and must move the score, so an
//     apparatus that cannot tell the arms apart fails instead of agreeing.
// Any of these failing exits nonzero and the numbers below it mean nothing.
//
// Build: scripts/probe_hypot36.sh
// ===========================================================================

#define main sweep_accuracy_main_renamed
#include "sweep_accuracy.cpp"
#undef main

#include <mpfr.h>

namespace {

constexpr int kPrec = 400;

// --- the exactly-rounded quotient, and the greedy round of any exact value --
//
// Greedy round-to-nearest into four floats IS the format floor: each word takes
// the nearest float to what is left, so no other four-float expansion is closer.
xp::QuadFloat greedy_qf(mpfr_srcptr x) {
    mpfr_t r; mpfr_init2(r, kPrec); mpfr_set(r, x, MPFR_RNDN);
    float w[4];
    for (int k = 0; k < 4; ++k) {
        w[k] = (float)mpfr_get_d(r, MPFR_RNDN);
        if (!std::isfinite(w[k])) { w[k] = 0.0f; }
        mpfr_t t; mpfr_init2(t, kPrec); mpfr_set_d(t, (double)w[k], MPFR_RNDN);
        mpfr_sub(r, r, t, MPFR_RNDN); mpfr_clear(t);
    }
    mpfr_clear(r);
    return xp::QuadFloat(w[0], w[1], w[2], w[3]);
}

// __float128 -> mpfr, EXACTLY.  Not via long double: x87's 64-bit mantissa
// truncates a binary128 and would put a 2^-64 relative error into the format
// floor, which at 1.4e-21 is 7.8e-41 -- three orders ABOVE the errors this
// probe is measuring, i.e. it would make the unbeatable arm the worst one.
// 113 bits fit in three doubles, so three Dekker-style peels are exact.
void set_q(mpfr_t out, __float128 x) {
    mpfr_set_d(out, 0.0, MPFR_RNDN);
    __float128 r = x;
    for (int k = 0; k < 3 && r != 0; ++k) {
        const double d = (double)r;
        if (!std::isfinite(d)) break;
        mpfr_t t; mpfr_init2(t, kPrec); mpfr_set_d(t, d, MPFR_RNDN);
        mpfr_add(out, out, t, MPFR_RNDN); mpfr_clear(t);
        r -= (__float128)d;
    }
}

void set_qf(mpfr_t out, const xp::QuadFloat& v) {
    mpfr_set_d(out, (double)v.f0, MPFR_RNDN);
    const float rest[3] = {v.f1, v.f2, v.f3};
    for (float f : rest) {
        mpfr_t t; mpfr_init2(t, kPrec); mpfr_set_d(t, (double)f, MPFR_RNDN);
        mpfr_add(out, out, t, MPFR_RNDN); mpfr_clear(t);
    }
}

xp::QuadFloat exact_quotient(const xp::QuadFloat& n, const xp::QuadFloat& m) {
    mpfr_t mn, mm, mq;
    mpfr_init2(mn, kPrec); mpfr_init2(mm, kPrec); mpfr_init2(mq, kPrec);
    set_qf(mn, n); set_qf(mm, m);
    mpfr_div(mq, mn, mm, MPFR_RNDN);
    xp::QuadFloat q = greedy_qf(mq);
    mpfr_clear(mn); mpfr_clear(mm); mpfr_clear(mq);
    return q;
}

// --- hypot's body, transcribed from qf_math.hpp:1789-1800 -------------------
//
// The ONLY line that differs between arms is the one marked.  Everything else
// -- the abs, the inf tests, the m/n ordering, the [kQFSqLo, kQFSqHi] gate that
// decides whether a divide happens at all, the closing multiply -- is the
// shipped text.  MODE 0 is checked bit-for-bit against xp::hypot below, which
// is what makes the other three arms about hypot and not about this function.
enum Mode { M_LIFT = 0, M_PLAIN = 1, M_EXACT = 2, M_WRECK = 3 };

bool g_divided = false;   // did the scaled branch (and therefore divide) run?

xp::QuadFloat hypot_arm(xp::QuadFloat a, xp::QuadFloat b, Mode mode) {
    using xp::QuadFloat;
    g_divided = false;
    QuadFloat x = xp::abs(a);
    QuadFloat y = xp::abs(b);
    if (xp::detail::isinf(x.f0)) return x;
    if (xp::detail::isinf(y.f0)) return y;
    QuadFloat m = (x.f0 < y.f0) ? y : x;
    QuadFloat n = (x.f0 < y.f0) ? x : y;
    if (m.f0 == 0.0f) return QuadFloat(0.0f);
    if (m.f0 <= xp::detail::kQFSqHi && m.f0 >= xp::detail::kQFSqLo)
        return xp::sqrt(xp::add(xp::multiply(a, a), xp::multiply(b, b)));
    g_divided = true;
    QuadFloat t;                                          // <<< the one line
    switch (mode) {
        case M_LIFT:  t = xp::divide(n, m);                       break;
        case M_PLAIN: t = xp::detail::qf_divide_core(n, m);       break;
        case M_EXACT: t = exact_quotient(n, m);                   break;
        case M_WRECK: t = xp::multiply(xp::divide(n, m),
                                       QuadFloat(1.0f + 0x1p-20f)); break;
    }
    return xp::multiply(m, xp::sqrt(xp::add(QuadFloat(1.0f),
                                            xp::multiply(t, t))));
}

bool same_words(const xp::QuadFloat& p, const xp::QuadFloat& q) {
    return p.f0 == q.f0 && p.f1 == q.f1 && p.f2 == q.f2 && p.f3 == q.f3;
}

// How many of the four words have fallen out of the FP32 normal range?  This is
// the whole question at 1e-21: a word below 1.17549435e-38f is subnormal and
// carries fewer than 24 bits; a word below FLT_TRUE_MIN carries none.
int subnormal_words(const xp::QuadFloat& v) {
    int n = 0;
    const float w[4] = {v.f0, v.f1, v.f2, v.f3};
    for (float f : w)
        if (f != 0.0f && std::fabs(f) < 1.17549435e-38f) ++n;
    return n;
}

void print_words(const char* tag, const xp::QuadFloat& v) {
    std::printf("    %-8s f0 %-14.7g f1 %-14.7g f2 %-14.7g f3 %-14.7g   "
                "subnormal words %d\n",
                tag, (double)v.f0, (double)v.f1, (double)v.f2, (double)v.f3,
                subnormal_words(v));
}

}  // namespace

int main() {
    // ---- the genuine operands, from the sweep's own grid and operand fill ---
    const std::vector<GridPoint> rgrid = build_real_grid();
    const int    id = R_Hypot;
    const size_t pt = 36;
    if (pt >= rgrid.size()) { std::printf("grid too small\n"); return 1; }

    Rng rng(stream_seed(12345ULL, kReal[id].name, 0u));
    double a = 0.0, b = 0.0, c = 0.0;
    for (size_t i = 0; i <= pt; ++i) {          // the stream is sequential
        double av = rgrid[i].re, bv, cv;
        fill_real_operands(id, i, av, rng, bv, cv);
        repair_real(id, av, bv, cv);
        if (i == pt) { a = av; b = bv; c = cv; }
    }
    (void)c;

    const xp::QuadFloat A(a), B(b);
    const __float128 ref = reference_real_q(id, to_q(A), to_q(B), (__float128)0);
    const int  sb   = BackendQF::sig_bits();
    const Range rg  = BackendQF::range();

    std::printf("=== QF r hypot point 36 ===============================\n");
    std::printf("  a   = %.17g\n  b   = %.17g\n", a, b);
    std::printf("  b/a = 1 + %.17g   (the i%%7==1 cancelling family)\n", b / a - 1.0);
    std::printf("  ref = %.21Lg\n", (long double)ref);
    print_words("a", A);
    print_words("b", B);

    // ---- SELF-CHECK 1: the transcription is hypot -------------------------
    const xp::QuadFloat shipped = xp::hypot(A, B);
    const xp::QuadFloat r_lift  = hypot_arm(A, B, M_LIFT);
    const bool divided = g_divided;
    if (!same_words(shipped, r_lift)) {
        std::printf("FAIL: the transcribed body does not reproduce xp::hypot "
                    "bit for bit; nothing below is about hypot.\n");
        return 1;
    }
    std::printf("  scaled branch taken (a divide happens): %s\n",
                divided ? "yes" : "NO -- this row never divides");

    const xp::QuadFloat r_plain = hypot_arm(A, B, M_PLAIN);
    const xp::QuadFloat r_exact = hypot_arm(A, B, M_EXACT);
    const xp::QuadFloat r_wreck = hypot_arm(A, B, M_WRECK);

    mpfr_t mref; mpfr_init2(mref, kPrec);
    set_q(mref, ref);
    const xp::QuadFloat r_floor = greedy_qf(mref);
    mpfr_clear(mref);

    struct Arm { const char* tag; xp::QuadFloat v; };
    const Arm arms[5] = {{"plain",  r_plain}, {"lift",   r_lift},
                         {"exact",  r_exact}, {"FLOOR*", r_floor},
                         {"WRECK",  r_wreck}};
    double u_plain = 0.0, u_lift = 0.0, u_floor = 0.0;
    // The last column is the point of the whole probe: the absolute error as a
    // multiple of FLT_TRUE_MIN, the smallest number any FP32 word can hold.
    // Below 1.0 the error is smaller than the format's own smallest increment
    // and no expansion of floats can express the correction.
    const double kTrueMin = 1.4012984643248171e-45;
    std::printf("\n  arm       ulps            abs err       x FLT_TRUE_MIN\n");
    for (const Arm& arm : arms) {
        const __float128 g = to_q(arm.v);
        const double u = ulps_scalar(g, ref, sb);
        const double e = (double)fabsq(g - ref);
        std::printf("    %-8s %-15.6g %-13.6g %.4g\n", arm.tag, u, e, e / kTrueMin);
        if (!std::strcmp(arm.tag, "plain"))  u_plain = u;
        if (!std::strcmp(arm.tag, "lift"))   u_lift  = u;
        if (!std::strcmp(arm.tag, "FLOOR*")) u_floor = u;
    }

    // ---- SELF-CHECK 2: the arms reproduce the two CSVs --------------------
    // 16916.8 and 60371 are what validation/sweep/sweep_baseline.csv.gz and the
    // post-lift sweep print, to six significant figures.
    const double csv_plain = 16916.8, csv_lift = 60371.0;
    const bool ok_plain = std::fabs(u_plain - csv_plain) <= 0.5;
    const bool ok_lift  = std::fabs(u_lift  - csv_lift)  <= 0.5;
    std::printf("\n  CSV check: plain %.6g vs baseline %.6g  %s ;  "
                "lift %.6g vs post-lift %.6g  %s\n",
                u_plain, csv_plain, ok_plain ? "OK" : "MISMATCH",
                u_lift,  csv_lift,  ok_lift  ? "OK" : "MISMATCH");

    // ---- SELF-CHECK 3: the apparatus can tell the arms apart --------------
    const bool wreck_moved = !same_words(r_wreck, r_lift);
    std::printf("  wreck check: perturbing t by 1+2^-20 moves the result: %s\n",
                wreck_moved ? "yes" : "NO -- this probe cannot fail");

    // ---- the format's own floor at this magnitude -------------------------
    //
    // subnormal_floor_ulps is sweep_accuracy.cpp:1952, unmodified, called on the
    // same ref and the same Range the sweep uses for QF.  It is already a term
    // in this row's bound; printing it alone says how much of the bound it is.
    const double floor_ulps = subnormal_floor_ulps(ref, rg, sb);
    const double ulp_abs    = (double)fabsq(ref) * std::exp2(-(double)sb);
    std::printf("\n  format floor at |ref| = %.6g:\n", (double)fabsq(ref));
    std::printf("    subnormal_floor_ulps (sweep_accuracy.cpp:1952) : %.6g ulps\n",
                floor_ulps);
    std::printf("    1 nominal ulp of the metric                    : %.6g abs\n",
                ulp_abs);
    std::printf("    FLT_TRUE_MIN                                   : %.6g abs\n",
                1.4012984643248171e-45);
    std::printf("    row's bound from the CSV                       : 188310 ulps\n");
    std::printf("    plain %.6g / floor = %.4g ;  lift %.6g / floor = %.4g\n",
                u_plain, u_plain / floor_ulps, u_lift, u_lift / floor_ulps);

    print_words("t plain", xp::detail::qf_divide_core(xp::abs(A), xp::abs(B)));
    print_words("t lift",  xp::divide(xp::abs(A), xp::abs(B)));

    // ---- SELF-CHECK 4: FLOOR* is unbeatable -------------------------------
    // Greedy rounding of the exact value is by construction the closest thing
    // this format holds.  If any algorithmic arm beats it, the floor arm is
    // computed wrongly and the "no algorithm can do better" claim is unearned.
    const bool floor_is_floor = (u_floor <= u_plain + 1e-9) &&
                                (u_floor <= u_lift  + 1e-9);
    std::printf("  floor check: FLOOR* (%.6g) is at or below every algorithmic "
                "arm: %s\n", u_floor, floor_is_floor ? "yes" : "NO -- floor arm is wrong");

    if (!ok_plain || !ok_lift || !wreck_moved || !divided || !floor_is_floor) {
        std::printf("\nPROBE FAILED ITS OWN CHECKS\n");
        return 1;
    }
    std::printf("\nall self-checks passed\n");
    return 0;
}
