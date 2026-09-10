// ============================================================================
// trig_reduction_test.cpp — structural test for the Payne-Hanek reduction.
// ============================================================================
//
// WHAT THIS TESTS, AND WHAT IT DELIBERATELY DOES NOT
// --------------------------------------------------
// docs/CORRECTNESS.md:5-6 — "nothing else in the repository issues a competing
// opinion" on accuracy. This test therefore issues NO accuracy verdict on
// sin/cos/tan. It tests the REDUCTION ENGINE in isolation: given x, does
// xp_ph_reduce return the exact quadrant and a fraction good to one ulp of the
// output expansion? That is a property of include/xp/trig_reduction.hpp and
// include/xp/trig_reduction_data.hpp alone, and the sweep never sees it.
//
// GROUND TRUTH IS COMPUTED HERE, NOT CITED
// ----------------------------------------
// This file links mpfr and gmp and recomputes 2/pi, pi/2, the exact fraction
// and the exact quadrant at 4000 bits for every input it drives. It cites no
// number it cannot re-derive. (Contrast tests/exp_reduction_test.cpp:193, which
// names a /tmp path as the authority for a tail it cannot resolve in
// binary128 — that is the antipattern this file exists not to repeat.)
//
// The decimal mantissas below ARE cited, from
// scripts/gen_trig_reduction_constants.cpp --measure. But they are cited as
// INPUTS, not as authorities: nothing is asserted on the grounds that they are
// the argmax. P5 re-derives each one's exact fraction and asserts P6/P7 about
// it directly, and P4 proves each is a value the backend can actually hold.
// If the generator's search were wrong, these would simply be less demanding
// inputs than the true worst case — the assertions would still be sound, and
// running the generator reproduces or refutes the choice in four seconds.
//
// WHY EACH ASSERTION EXISTS (each maps to a way the reduction can rot)
// --------------------------------------------------------------------
// P1  table content    someone regenerates against a different constant, or
//                      hand-edits a chunk, or the two chunkings drift apart
// P2  table depth      someone trims the table "because the tail is small"
// P3  pi/2 expansion   someone drops or rounds a word of the final multiply
// P4  input reach      the pinning inputs must be values the format HOLDS,
//                      or P6/P7 are testing something no user can reach
// P5  quadrant         n mod 4 is what selects sin/cos and its sign; a wrong
//                      quadrant is a 100%-error answer, not a rounding
// P6  accuracy         the actual property, at the inputs that stress it most
// P7  depth is pinned  one chunk fewer must MISS — otherwise the shipped
//                      table has slack and P6 proves nothing about the depth
// P8  guard is pinned  the naive guard p+4 (no cancellation allowance) must
//                      MISS at the deepest-cancellation input — otherwise the
//                      measured C in kPhGuard* is decoration
//
// THE CRITERION, AND WHY IT IS NOT "RELATIVE ERROR BELOW 2^-p"
// -------------------------------------------------------------
// The output is an expansion of nfrac machine words, so the smallest nonzero
// quantity it can carry at magnitude |f| is
//     ulp(f) = max(|f| * 2^-p, smallest positive subnormal)
// and at the extreme cancellations this test drives, the subnormal floor is
// the binding one for the FP32 backends (QF reaches |f| ~ 2^-95, where
// |f|*2^-96 is 2^-191 and the float floor is 2^-149). Asking for relative
// error below 2^-p there is asking the format for bits it does not have; it
// would fail TF and QF for being formats. So the check is ABSOLUTE against
// that floor, expressed in ulps of it.
//
// THE POISON (validation/trig_reduction_selftest.sh)
// ---------------------------------------------------
//   1  add 1 to the LAST double chunk        must be caught by P1
//   2  set mantissa bit 0 of the last pi/2   must be caught by P3
//   3  reduce with ntab-1 in the SHIPPED arm must be caught by P6
//   4  reduce with guard = p+4 in the same   must be caught by P6
// Poisons 1 and 2 perturb this file's own copies, in the shape of
// tests/exp_reduction_test.cpp; 3 and 4 perturb the arguments handed to the
// shipped engine. Note the byte-identical trap that file documents does not
// arise for 1 (the chunks are integers below 2^24, so +1 always moves the
// value) and is defeated for 2 by setting a bit that is 0 in every shipped
// pi/2 word rather than by re-rounding to a nominal width.
//
// P7 and P8 are permanently-armed negative arms and need no poison of their
// own: they ARE the poison, asserted every run, in the shape of
// exp_reduction_test.cpp's "one piece fewer would also pass" checks.
// ============================================================================
#include <xp/trig_reduction.hpp>

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <mpfr.h>
#include <gmp.h>

using namespace xp::detail;

#ifdef XPMATH_POISON_TRIG_REDUCTION
#  define POISON XPMATH_POISON_TRIG_REDUCTION
#else
#  define POISON 0
#endif

static int failures = 0;
static int checks   = 0;

static void ok(bool cond, const char* what) {
    ++checks;
    if (!cond) { ++failures; std::printf("  FAIL: %s\n", what); }
}

static const mpfr_prec_t kWork = 4000;
static mpfr_t g_2opi;      // 2/pi
static mpfr_t g_pio2;      // pi/2

static double l2(mpfr_srcptr v) {
    if (mpfr_sgn(v) == 0) return -1e30;
    mpfr_t t; mpfr_init2(t, kWork);
    mpfr_abs(t, v, MPFR_RNDN);
    mpfr_log2(t, t, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

// ---------------------------------------------------------------------------
// P1 — the 2/pi table is the exact truncation of 2/pi at its own depth
// ---------------------------------------------------------------------------
//
// "Exact truncation" is the whole assertion, and it is a two-sided one: the
// residual frac(2/pi) - sum(T[k] * 2^(-cb(k+1))) must lie in [0, 2^(-cb*n)).
// Perturbing ANY chunk by even 1 in either direction leaves that interval, at
// any depth, including the last — which is why this check does not need to
// know where a corruption might be.
static void check_ipio2(const char* nm, const std::vector<long>& t, int cb) {
    const int n = (int)t.size();
    int rng = 0;
    for (int k = 0; k < n; ++k)
        if (t[k] < 0 || t[k] >= (1L << cb)) ++rng;
    ok(rng == 0, "P1 every chunk is an integer in [0, 2^cb)");

    mpfr_t s, u, f;
    mpfr_inits2(kWork, s, u, f, (mpfr_ptr)0);
    mpfr_set_ui(s, 0, MPFR_RNDN);
    for (int k = 0; k < n; ++k) {
        mpfr_set_si(u, t[k], MPFR_RNDN);
        mpfr_div_2ui(u, u, (unsigned long)(cb * (k + 1)), MPFR_RNDN);
        mpfr_add(s, s, u, MPFR_RNDN);
    }
    mpfr_frac(f, g_2opi, MPFR_RNDN);      // 2/pi = 0.6366..., drops a leading 0
    mpfr_sub(u, f, s, MPFR_RNDN);         // residual
    const int sign_ok = mpfr_sgn(u) >= 0;
    mpfr_t lim; mpfr_init2(lim, kWork);
    mpfr_set_ui(lim, 1, MPFR_RNDN);
    mpfr_div_2ui(lim, lim, (unsigned long)(cb * n), MPFR_RNDN);
    const int mag_ok = mpfr_cmp(u, lim) < 0;
    std::printf("   %s table: %d chunks x %d bits, residual 2^%.2f (limit 2^%d)\n",
                nm, n, cb, l2(u), -cb * n);
    ok(sign_ok && mag_ok, "P1 table is the exact truncation of 2/pi");
    mpfr_clears(s, u, f, lim, (mpfr_ptr)0);
}

// ---------------------------------------------------------------------------
// P5/P6/P7 driver — reduce one input and score it against MPFR
// ---------------------------------------------------------------------------
static void narrow_set(double* d, mpfr_srcptr r) { *d = mpfr_get_d(r, MPFR_RNDN); }
static void narrow_set(float*  d, mpfr_srcptr r) { *d = mpfr_get_flt(r, MPFR_RNDN); }

struct Meas {
    double ulps;        // error in ulps of the output expansion
    double log2x;
    double log2f;
    double floor_bits;
    int    q_mine;
    int    q_ref;
    bool   q_ok;
    bool   exact_input; // the nw words reproduce m * 2^e with no loss
};

// Score an expansion that is already in hand. `xwant` is what the words were
// meant to represent; pass a null pointer when the words themselves ARE the
// definition (the span sweep), in which case exact_input is trivially true.
template <typename S>
static Meas score(const S* w, int nw, int p, int nfrac, int guard, int ntab,
                  mpfr_srcptr xwant) {
    Meas r;
    const double sub = (sizeof(S) == 4) ? -149.0 : -1074.0;

    mpfr_t x, v, fref, fmine, t;
    mpfr_inits2(kWork, x, v, fref, fmine, t, (mpfr_ptr)0);
    mpfr_set_ui(x, 0, MPFR_RNDN);
    for (int i = 0; i < nw; ++i) {
        mpfr_set_d(t, (double)w[i], MPFR_RNDN);
        mpfr_add(x, x, t, MPFR_RNDN);       // exact: a sum of machine words
    }
    if (xwant) { mpfr_sub(t, x, xwant, MPFR_RNDN); r.exact_input = (mpfr_sgn(t) == 0); }
    else       { r.exact_input = true; }
    r.log2x = l2(x);

    // exact reference: v = x * 2/pi, n = rint(v), f = v - n, q = n mod 4
    mpfr_mul(v, x, g_2opi, MPFR_RNDN);
    mpfr_round(t, v);
    mpfr_sub(fref, v, t, MPFR_RNDN);
    { mpz_t n; mpz_init(n); mpfr_get_z(n, t, MPFR_RNDN);
      long q = (long)mpz_fdiv_ui(n, 4);
      if (mpz_sgn(n) < 0) { mpz_neg(n, n); q = (4 - (long)mpz_fdiv_ui(n, 4)) & 3; }
      r.q_ref = (int)q; mpz_clear(n); }
    r.log2f = l2(fref);

    r.floor_bits = r.log2f - (double)p;
    if (r.floor_bits < sub) r.floor_bits = sub;

    S fr[8];
    r.q_mine = xp_ph_reduce<S>(w, nw, guard, ntab, fr, nfrac);
    mpfr_set_ui(fmine, 0, MPFR_RNDN);
    for (int i = 0; i < nfrac; ++i) {
        mpfr_set_d(t, (double)fr[i], MPFR_RNDN);
        mpfr_add(fmine, fmine, t, MPFR_RNDN);
    }
    // (q, f) and (q+1, f-1) name the SAME angle. Accept the shifted pair and
    // fold the shift into f, so a genuine quadrant error still shows as one.
    r.q_ok = true;
    if (r.q_mine != r.q_ref) {
        if (((r.q_mine + 1) & 3) == r.q_ref)      mpfr_sub_ui(fmine, fmine, 1, MPFR_RNDN);
        else if (((r.q_mine + 3) & 3) == r.q_ref) mpfr_add_ui(fmine, fmine, 1, MPFR_RNDN);
        else r.q_ok = false;
    }

    mpfr_sub(t, fmine, fref, MPFR_RNDN);
    const double abserr = (mpfr_sgn(t) == 0) ? -1e30 : l2(t);
    r.ulps = std::exp2(abserr - r.floor_bits);

    mpfr_clears(x, v, fref, fmine, t, (mpfr_ptr)0);
    return r;
}

// x = m * 2^e, greedily split into the backend's own nw-word expansion.
template <typename S>
static Meas drive(const char* mstr, long e, int p, int nw, int nfrac,
                  int guard, int ntab) {
    mpz_t m; mpz_init_set_str(m, mstr, 10);
    mpfr_t x, t, u;
    mpfr_inits2(kWork, x, t, u, (mpfr_ptr)0);
    mpfr_set_z(x, m, MPFR_RNDN);
    mpfr_mul_2si(x, x, e, MPFR_RNDN);       // exact
    S w[8];
    mpfr_set(t, x, MPFR_RNDN);
    for (int i = 0; i < nw; ++i) {
        narrow_set(&w[i], t);
        mpfr_set_d(u, (double)w[i], MPFR_RNDN);
        mpfr_sub(t, t, u, MPFR_RNDN);
    }
    const Meas r = score<S>(w, nw, p, nfrac, guard, ntab, x);
    mpfr_clears(x, t, u, (mpfr_ptr)0);
    mpz_clear(m);
    return r;
}

// ---------------------------------------------------------------------------

struct Pin {
    const char* nm;
    const char* m;      // decimal mantissa; x = m * 2^e
    long        e;
    int         p;      // significand bits of the backend
    int         nw;     // words in the input expansion
    int         nfrac;  // words in the output expansion
    int         guard;  // shipped guard
    int         ntab;   // shipped table depth
    bool        is32;
};

// argmax of log2(|x|/|f|): what the table DEPTH is sized by, and therefore
// what the one-chunk-short arm has to get wrong.
static const Pin kPins[] = {
    {"DD", "54146676858324748940541860376576", 917, 106, 2, 3, kPhGuardDD, kPhChunksDD, false},
    {"FF", "267873697790651",                   77,  48, 2, 2, kPhGuardFF, kPhChunksFF, true },
    {"TF", "4454573411458703862623",            56,  72, 3, 3, kPhGuardTF, kPhChunksTF, true },
    {"QF", "78191413402364241149357698304",     32,  96, 4, 4, kPhGuardQF, kPhChunksQF, true },
};

// argmax of log2(1/|f|): what the GUARD is sized by. These do NOT pin the
// table depth — the shipped and one-short tables agree here — which is exactly
// why the two maxima are measured separately.
static const Pin kDeep[] = {
    {"DD", "77828009278254995876762849461631", 263, 106, 2, 3, kPhGuardDD, kPhChunksDD, false},
    {"FF", "260373991191917",                   59,  48, 2, 2, kPhGuardFF, kPhChunksFF, true },
    {"TF", "4301426596981124853717",           -52,  72, 3, 3, kPhGuardTF, kPhChunksTF, true },
    {"QF", "55510843683165024412902151171",    -74,  96, 4, 4, kPhGuardQF, kPhChunksQF, true },
};

static Meas run(const Pin& c, int guard, int ntab) {
    return c.is32 ? drive<float >(c.m, c.e, c.p, c.nw, c.nfrac, guard, ntab)
                  : drive<double>(c.m, c.e, c.p, c.nw, c.nfrac, guard, ntab);
}

// the shipped arm, with the poison hooks applied to its arguments
static int g_dntab  = 0;
static int g_naive_guard = 0;
static Meas run_shipped(const Pin& c) {
    return run(c, g_naive_guard ? c.p + 4 : c.guard, c.ntab + g_dntab);
}

int main() {
    mpfr_init2(g_2opi, kWork);
    mpfr_init2(g_pio2, kWork);
    { mpfr_t pi; mpfr_init2(pi, kWork); mpfr_const_pi(pi, MPFR_RNDN);
      mpfr_set_ui(g_2opi, 2, MPFR_RNDN); mpfr_div(g_2opi, g_2opi, pi, MPFR_RNDN);
      mpfr_div_2ui(g_pio2, pi, 1, MPFR_RNDN);
      mpfr_clear(pi); }

#if POISON == 3
    g_dntab = -1;
#elif POISON == 4
    g_naive_guard = 1;
#endif

    // ---------------------------------------------------------------- P1
    std::printf("P1  2/pi tables\n");
    {
        std::vector<long> td(kPhIpio2LenD), tf(kPhIpio2LenF);
        for (int k = 0; k < kPhIpio2LenD; ++k) td[k] = (long)xp_ph_ipio2_d(k);
        for (int k = 0; k < kPhIpio2LenF; ++k) tf[k] = (long)xp_ph_ipio2_f(k);

        // every chunk must have come back as an exact integer
        int nonint = 0;
        for (int k = 0; k < kPhIpio2LenD; ++k) if ((double)td[k] != xp_ph_ipio2_d(k)) ++nonint;
        for (int k = 0; k < kPhIpio2LenF; ++k) if ((float)tf[k]  != xp_ph_ipio2_f(k)) ++nonint;
        ok(nonint == 0, "P1 every table entry is an exact integer in its word");

#if POISON == 1
        td[kPhIpio2LenD - 1] += 1;   // the deepest chunk: the hardest to see
#endif
        check_ipio2("FP64", td, kPhChunkBits64);
        check_ipio2("FP32", tf, kPhChunkBits32);

        // the two chunkings must describe the same constant: 24 = 2 * 12
        ok(tf[0] * 4096 + tf[1] == td[0],
           "P1 the 12-bit and 24-bit chunkings agree on the leading 24 bits");
        // and the leading word must be fdlibm's published 0xA2F983 — an
        // independent source for the one value MPFR and the generator share
        ok(td[0] == 0xA2F983L, "P1 leading FP64 chunk is fdlibm's 0xA2F983");

        // digit products must be exact in the working word
        ok(2 * kPhChunkBits64 <= 53, "P1 2*24 <= 53: FP64 digit products are exact");
        ok(2 * kPhChunkBits32 <= 24, "P1 2*12 <= 24: FP32 digit products are exact");
    }

    // ---------------------------------------------------------------- P2
    // The shipped depth must cover every backend that reads the table.
    std::printf("P2  table depth covers every backend\n");
    ok(kPhChunksDD <= kPhIpio2LenD, "P2 DD depth within the FP64 table");
    ok(kPhChunksFF <= kPhIpio2LenF, "P2 FF depth within the FP32 table");
    ok(kPhChunksTF <= kPhIpio2LenF, "P2 TF depth within the FP32 table");
    ok(kPhChunksQF <= kPhIpio2LenF, "P2 QF depth within the FP32 table");
    // and the FP32 table must be exactly as deep as its deepest consumer:
    // a longer one would make P7's FP32 arms vacuous
    ok(kPhIpio2LenF == kPhChunksQF, "P2 FP32 table is exactly its deepest consumer");
    ok(kPhIpio2LenD == kPhChunksDD, "P2 FP64 table is exactly its deepest consumer");

    // ---------------------------------------------------------------- P3
    std::printf("P3  pi/2 expansions\n");
    {
        // pi/2 enters as the final multiply r = f * (pi/2), so its RELATIVE
        // error lands directly in r. The widest FP64 consumer is DD (p=106)
        // and the widest FP32 one is QF (p=96); allow 4 bits of slack, the
        // same kSlackBits the guard carries.
        //
        // HONEST MARGIN. Two doubles reach 2^-109.04, so the FP64 N-1 arm
        // below misses the 110-bit requirement by 0.96 bits — deterministic,
        // but thin, and it depends on the slack being at least 3.04. Two
        // doubles are 3.04 bits past p=106, so ANY nonzero slack keeps three
        // words the right answer; only a slack of 0..3 would unpin the count.
        // The FP32 arms are not close: three floats reach 2^-76.33 against a
        // 100-bit requirement.
        const double reqd = 106.0 + 4.0;
        const double reqf =  96.0 + 4.0;

        double pd[8]; float pf[8];
        for (int i = 0; i < kPhPio2WordsD; ++i) pd[i] = xp_ph_pio2_d(i);
        for (int i = 0; i < kPhPio2WordsF; ++i) pf[i] = xp_ph_pio2_f(i);
#if POISON == 2
        // Set mantissa bit 0 of the SECOND word of each expansion. Second and
        // not last: the last FP64 word has 53 bits of slack over the
        // requirement, so corrupting it is invisible and would be a poison
        // that tests nothing — exactly the trap exp_reduction_test.cpp
        // documents, in a different disguise. Bit 0 is 0 in every shipped
        // word, so setting it always moves the value.
        { uint64_t b; std::memcpy(&b, &pd[1], 8); b ^= 1ull; std::memcpy(&pd[1], &b, 8); }
        { uint32_t b; std::memcpy(&b, &pf[1], 4); b ^= 1u;   std::memcpy(&pf[1], &b, 4); }
#endif
        mpfr_t s, t;
        mpfr_inits2(kWork, s, t, (mpfr_ptr)0);

        for (int arm = 0; arm < 2; ++arm) {           // shipped, then one short
            const int nd = kPhPio2WordsD - arm, nf = kPhPio2WordsF - arm;
            mpfr_set_ui(s, 0, MPFR_RNDN);
            for (int i = 0; i < nd; ++i) { mpfr_set_d(t, pd[i], MPFR_RNDN); mpfr_add(s, s, t, MPFR_RNDN); }
            mpfr_sub(t, s, g_pio2, MPFR_RNDN);
            const double rd = l2(t);
            mpfr_set_ui(s, 0, MPFR_RNDN);
            for (int i = 0; i < nf; ++i) { mpfr_set_d(t, (double)pf[i], MPFR_RNDN); mpfr_add(s, s, t, MPFR_RNDN); }
            mpfr_sub(t, s, g_pio2, MPFR_RNDN);
            const double rf = l2(t);
            std::printf("   %-8s FP64 %d words 2^%8.2f (need 2^%.0f)   FP32 %d words 2^%8.2f (need 2^%.0f)\n",
                        arm ? "N-1" : "shipped", nd, rd, -reqd, nf, rf, -reqf);
            if (!arm) {
                ok(rd < -reqd, "P3 FP64 pi/2 expansion reaches DD's p+4");
                ok(rf < -reqf, "P3 FP32 pi/2 expansion reaches QF's p+4");
            } else {
                ok(!(rd < -reqd), "P3 one FP64 word fewer would also pass — count is not pinned");
                ok(!(rf < -reqf), "P3 one FP32 word fewer would also pass — count is not pinned");
            }
        }
        mpfr_clears(s, t, (mpfr_ptr)0);

        // Non-overlap. A greedy round-to-nearest split leaves each word at
        // most a half ulp of its predecessor, so |w[i+1]| <= |w[i]| * 2^-(b-1)
        // with b the format's significand width. Without this the words could
        // sum to the right value while overlapping, and the backend's
        // expansion arithmetic assumes they do not.
        int lap = 0;
        for (int i = 0; i + 1 < kPhPio2WordsD; ++i)
            if (!(std::fabs(pd[i + 1]) <= std::ldexp(std::fabs(pd[i]), -52))) ++lap;
        for (int i = 0; i + 1 < kPhPio2WordsF; ++i)
            if (!(std::fabs((double)pf[i + 1]) <= std::ldexp(std::fabs((double)pf[i]), -23))) ++lap;
        ok(lap == 0, "P3 pi/2 words are non-overlapping");
    }

    // ------------------------------------------------------- P4/P5/P6/P7
    std::printf("P4/P5/P6/P7  reduction at the depth-pinning inputs\n");
    for (const Pin& c : kPins) {
        const Meas s = run_shipped(c);
        const Meas n = run(c, c.guard, c.ntab - 1);
        std::printf("   %s  log2|x|=%9.3f  log2|f|=%8.3f  ulp=2^%-8.2f  q=%d  "
                    "shipped %9.3g ulp   N-1 %9.3g ulp\n",
                    c.nm, s.log2x, s.log2f, s.floor_bits, s.q_ref, s.ulps, n.ulps);
        ok(s.exact_input, "P4 the pinning input is a value the backend can hold");
        ok(s.q_ok,        "P5 quadrant matches the exact n mod 4");
        ok(s.ulps <= 1.0, "P6 shipped table is within 1 ulp at the pin");
        ok(!(n.ulps <= 1.0),
           "P7 one chunk fewer would also pass — the depth is not pinned");
    }

    // ---------------------------------------------------------------- P8
    std::printf("P8  reduction at the deepest-cancellation inputs\n");
    for (const Pin& c : kDeep) {
        const Meas s = run_shipped(c);
        const Meas g = run(c, c.p + 4, c.ntab);     // guard with no C allowance
        std::printf("   %s  log2|f|=%8.3f  ulp=2^%-8.2f  q=%d  "
                    "shipped %9.3g ulp   guard=p+4 %9.3g ulp\n",
                    c.nm, s.log2f, s.floor_bits, s.q_ref, s.ulps, g.ulps);
        ok(s.exact_input, "P4 the cancellation input is a value the backend can hold");
        ok(s.q_ok,        "P5 quadrant matches the exact n mod 4");
        ok(s.ulps <= 1.0, "P6 shipped guard is within 1 ulp at the deepest cancellation");
        ok(!(g.ulps <= 1.0),
           "P8 guard p+4 would also pass — the measured C is not load-bearing");
    }

    // ---------------------------------------------------------------- P5
    // A handful of ordinary and adversarial magnitudes, for the quadrant.
    // These are not sized by anything; they are here so that a reduction that
    // is precise but off by a quadrant cannot hide behind the extreme inputs.
    std::printf("P5  quadrant on assorted magnitudes (DD path)\n");
    {
        struct A { const char* m; long e; };
        static const A as[] = {
            {"6381956970095103", 797},        // the published IEEE-double worst case
            {"1", 0}, {"3", 0}, {"5", 1}, {"355", -7},
            {"7074237752028440", -51},        // ~pi
            {"1", 1023}, {"1", 1000}, {"1", -60},
            {"4503599627370495", 970},        // largest odd-ish mantissa near DBL_MAX
        };
        int bad = 0, over = 0;
        for (const A& a : as) {
            const Meas r = drive<double>(a.m, a.e, 106, 2, 3, kPhGuardDD, kPhChunksDD);
            if (!r.q_ok) ++bad;
            if (!(r.ulps <= 1.0)) ++over;
        }
        std::printf("   %d inputs: %d wrong quadrant, %d above 1 ulp\n",
                    (int)(sizeof(as) / sizeof(as[0])), bad, over);
        ok(bad == 0,  "P5 quadrant correct on every assorted magnitude");
        ok(over == 0, "P6 within 1 ulp on every assorted magnitude");
    }

    // ---------------------------------------------------------------- P9
    // WIDE-SPAN AND GAPPED INPUTS. The sizing scan enumerates x = m * 2^e with
    // a single contiguous mantissa, so it says nothing about an expansion whose
    // words are far apart — and a DoubleDouble hi + lo with lo 900 binades
    // below hi is a perfectly legal value. This is the one family the sizing
    // argument does not cover, so it is MEASURED here rather than assumed.
    //
    // The engine drops any word below 2^-guard absolute (the s > jmin+nacc-3
    // skip in xp_ph_reduce). That is safe by construction and not by luck: a
    // dropped word contributes under 2^-(p+C+4) to f, while ulp(f) is at least
    // 2^-(p+C) because |f| >= 2^-C. The span sweep is what checks the
    // construction against the code.
    std::printf("P9  wide-span and gapped expansions\n");
    {
        static const int spans[] = {53, 54, 60, 80, 106, 150, 223, 224, 300,
                                    500, 700, 900, 1000};
        double worst = 0; int worst_span = 0, bad = 0;
        for (int sp : spans) {
            for (int hi_e = 40; hi_e <= 1000; hi_e += 240) {
                if (hi_e - sp < -1070) continue;
                for (int sg = 0; sg < 2; ++sg) {
                    double w[2] = { std::ldexp(1.3, hi_e),
                                    std::ldexp(sg ? -1.7 : 1.7, hi_e - sp) };
                    const Meas r = score<double>(w, 2, 106, 3, kPhGuardDD,
                                                 kPhChunksDD, nullptr);
                    if (!r.q_ok) ++bad;
                    if (r.ulps > worst) { worst = r.ulps; worst_span = sp; }
                }
            }
        }
        std::printf("   DD 2-word, spans %d..%d: worst %9.3g ulp (at span %d), "
                    "%d wrong quadrant\n", spans[0], spans[12], worst,
                    worst_span, bad);
        ok(bad == 0,      "P9 quadrant correct on every span");
        ok(worst <= 1.0,  "P9 within 1 ulp on every span");

        // and the same for the widest FP32 backend, whose words can be
        // subnormal at the bottom of the range
        double worstq = 0; int badq = 0;
        for (int sp = 24; sp <= 140; sp += 4) {
            for (int hi_e = 10; hi_e <= 120; hi_e += 55) {
                float w[4] = { std::ldexp(1.3f, hi_e),
                               std::ldexp(-1.7f, hi_e - sp),
                               std::ldexp(1.1f, hi_e - 2 * sp),
                               std::ldexp(-1.9f, hi_e - 3 * sp) };
                // ldexp already flushes the deepest words to +-0 below 2^-149;
                // the engine has to cope with zero words either way
                const Meas r = score<float>(w, 4, 96, 4, kPhGuardQF,
                                            kPhChunksQF, nullptr);
                if (!r.q_ok) ++badq;
                if (r.ulps > worstq) worstq = r.ulps;
            }
        }
        std::printf("   QF 4-word, spans 24..140: worst %9.3g ulp, "
                    "%d wrong quadrant\n", worstq, badq);
        ok(badq == 0,     "P9 quadrant correct on every QF span");
        ok(worstq <= 1.0, "P9 within 1 ulp on every QF span");
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
#if POISON != 0
    if (failures == 0) {
        std::printf("POISON %d WAS NOT DETECTED — the test is blind\n", POISON);
        return 1;
    }
    std::printf("poison %d correctly detected\n", POISON);
    return 0;   // under poison, failing the assertions is SUCCESS
#else
    return failures ? 1 : 0;
#endif
}
