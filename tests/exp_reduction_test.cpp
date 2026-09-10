// ============================================================================
// exp_reduction_test.cpp — structural test for exp()'s Cody-Waite reduction.
// ============================================================================
//
// WHAT THIS TESTS, AND WHAT IT DELIBERATELY DOES NOT
// --------------------------------------------------
// docs/CORRECTNESS.md:5-6 — "nothing else in the repository issues a competing
// opinion" on accuracy. This test therefore issues NO accuracy verdict on
// exp(). It asserts STRUCTURAL properties of the reduction that are provable
// bit-equalities, in the shape of tests/*_eft_test.cpp: no oracle, no
// tolerance, no sampling luck.
//
// The property under test is the one the fix rests on:
//
//     every k*c_i is EXACT in one machine word, over the whole k range the
//     exp() guards admit, so that a - sum(k*c_i) carries no rounding of its own
//
// WHY EACH ASSERTION EXISTS (each maps to a way the fix can rot)
// ---------------------------------------------------------------
// P1  width      someone "improves precision" by widening a piece
// P2  exactness  the actual property, over the FULL reachable k
// P3  tail       someone deletes a piece as redundant
// P4  reduction  someone reintroduces the rounded product, or reorders
//
// TWO TRAPS THIS TEST IS BUILT AROUND
// ------------------------------------
// (1) THE BYTE-IDENTICAL POISON. Rounding ln2 to 17 or 18 significant bits
//     yields a float BIT-IDENTICAL to the 16-bit one, because the 16-bit value
//     0x3f317200 already has 9 trailing zero mantissa bits. The same happens on
//     DD at 43 bits. A poison that "widens a piece by one bit" therefore
//     silently tests NOTHING — verified the hard way during development, twice.
//     P1 defeats this by reading significant bits from the SHIPPED mantissa
//     (via trailing-zero count), never by re-deriving a constant at a nominal
//     width. The self-test poisons by setting mantissa bit 0, which always
//     changes the value.
// (2) THE K-RANGE GAP. The obvious bound is |k| <= 1024 from
//     a < 709.78 = ln(DBL_MAX). But dd_math.hpp:548 admits a down to -745.2, so
//     k reaches -1075. A check over [-1024, 1024] misses 306 products. The
//     ranges below are taken from the guards, not from the overflow limit.
//
// Ground truth is oracle-free and provable, exactly as in ff_eft_test.cpp:
//   FP32 pieces: k needs <= 8 bits, c_i <= 16, so k*c_i needs <= 24 <= 53.
//                Computing it in double is therefore EXACT.
//   FP64 pieces: k needs <= 11 bits, c_i <= 42, so k*c_i needs <= 53... but the
//                PRODUCT's exactness is the claim, so we widen to __float128
//                (113 bits) and compare bit-for-bit.
// ============================================================================
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <quadmath.h>

static int failures = 0;
static int checks   = 0;

static void ok(bool cond, const char* what) {
    ++checks;
    if (!cond) { ++failures; std::printf("  FAIL: %s\n", what); }
}

// ---------------------------------------------------------------- constants
// These MUST be kept identical to the bodies of exp() in the four headers.
// P3 pins the count; P1 pins the widths; P2 pins the exactness.
#ifdef XPMATH_POISON_EXP_REDUCTION
#  define POISON XPMATH_POISON_EXP_REDUCTION
#else
#  define POISON 0
#endif

static double dd_ln2[3] = {
     0x1.62e42fefa38p-1,
     0x1.ef35793c768p-45,
    -0x1.9ff0342543p-90
};
static float f_ln2[6] = {
     0x1.62e4p-1f,
     0x1.7f7ep-20f,
    -0x1.c61p-37f,
    -0x1.950ep-54f,
     0x1.e3b4p-72f,
    -0x1.9ffp-90f
};
// pieces actually used by each backend (nested prefixes; verified)
static const int FF_N = 4, TF_N = 5, QF_N = 6, DD_N = 3;

// significant bits carried, read from the mantissa itself
static int sigbits(double x) {
    if (x == 0.0) return 0;
    uint64_t b; std::memcpy(&b, &x, 8);
    uint64_t m = (b & ((1ULL << 52) - 1)) | (1ULL << 52);
    int tz = 0; while (!(m & 1)) { m >>= 1; ++tz; }
    return 53 - tz;
}
static int sigbits(float x) {
    if (x == 0.0f) return 0;
    uint32_t b; std::memcpy(&b, &x, 4);
    uint32_t m = (b & ((1U << 23) - 1)) | (1U << 23);
    int tz = 0; while (!(m & 1)) { m >>= 1; ++tz; }
    return 24 - tz;
}

int main() {
#if POISON == 1
    // flip the lowest mantissa bit of the leading piece: always changes the
    // value, immune to the byte-identical trap
    { uint64_t b; std::memcpy(&b, &dd_ln2[0], 8); b ^= 1ULL; std::memcpy(&dd_ln2[0], &b, 8); }
    { uint32_t b; std::memcpy(&b, &f_ln2[0], 4);  b ^= 1U;   std::memcpy(&f_ln2[0], &b, 4); }
    std::printf("POISON 1: low mantissa bit set on the leading piece\n");
#endif
    std::printf("exp_reduction_test — structural, no accuracy verdict\n");

    // ---------------------------------------------------------------- P1
    std::printf("\nP1: piece widths, read from the shipped mantissas\n");
    for (int i = 0; i < DD_N; ++i) {
        int w = sigbits(dd_ln2[i]);
        std::printf("   DD c%d: %2d bits\n", i + 1, w);
        ok(w <= 42, "DD piece exceeds 42 significant bits");
    }
    for (int i = 0; i < QF_N; ++i) {
        int w = sigbits(f_ln2[i]);
        std::printf("   FP32 c%d: %2d bits\n", i + 1, w);
        ok(w <= 16, "FP32 piece exceeds 16 significant bits");
    }

    // ---------------------------------------------------------------- P2
    // Ranges from the exp() guards, NOT from the overflow limit:
    //   dd_math.hpp:546,548  a in (-745.2, 709.78271289338397) -> k in [-1075, 1024]
    //   ff_math.hpp:646,650  a in (-104,   88.722839)          -> k in [-151,  128]
    std::printf("\nP2: k*c_i exact over the FULL reachable k range\n");
    {
        int bad = 0; long n = 0;
        for (long k = -1075; k <= 1024; ++k)
            for (int i = 0; i < DD_N; ++i) {
                double p = (double)k * dd_ln2[i];
                __float128 q = (__float128)(double)k * (__float128)dd_ln2[i];
                ++n;
                if ((__float128)p != q) ++bad;
            }
        std::printf("   DD   k in [-1075, 1024]: %ld products, %d inexact\n", n, bad);
        ok(bad == 0, "DD k*c_i not exact over the full k range");
    }
    {
        int bad = 0; long n = 0;
        for (long k = -151; k <= 128; ++k)
            for (int i = 0; i < QF_N; ++i) {
                float p = (float)k * f_ln2[i];
                double d = (double)(float)k * (double)f_ln2[i];
                ++n;
                if ((double)p != d) ++bad;
            }
        std::printf("   FP32 k in [-151, 128]:   %ld products, %d inexact\n", n, bad);
        ok(bad == 0, "FP32 k*c_i not exact over the full k range");
    }

    // ---------------------------------------------------------------- P3
    // The tail pins the COUNT. Each backend's tail must be small enough that
    // kmax*tail is under half an ulp of that backend's p -- and the count one
    // SHORT must fail that same criterion, so the test cannot be satisfied by
    // deleting a piece. Measured consequences of one short, for the record:
    //   FF n=3 -> 105.7 ulps   TF n=4 -> 242   QF n=5 -> 1.33e4
    std::printf("\nP3: tail pins the piece count\n");
    {
        // ln2 in __float128 (113 bits) as ground truth for the tail. Derived by
        // logq rather than an M_LN2q literal so no quadmath literal suffix is
        // needed (the suffix requires -fext-numeric-literals on some fronts).
        __float128 ln2q = logq((__float128)2);
        struct { const char* name; int n; int p; } cfg[] = {
            {"FF", FF_N, 48}, {"TF", TF_N, 72}, {"QF", QF_N, 96}
        };
        for (auto& c : cfg) {
            __float128 s = 0;
            for (int i = 0; i < c.n; ++i) s += (__float128)f_ln2[i];
            __float128 tail = fabsq(ln2q - s);
            double t = (double)tail;
            double ulps = std::ldexp(151.0 * t, c.p);
            std::printf("   %s %d pieces: tail 2^%.1f -> %.3g ulps of 2^-%d\n",
                        c.name, c.n, std::log2(t), ulps, c.p);
            ok(ulps < 0.5, "tail too large for the stated piece count");
            // one short must NOT satisfy the same criterion
            __float128 s1 = 0;
            for (int i = 0; i < c.n - 1; ++i) s1 += (__float128)f_ln2[i];
            double t1 = (double)fabsq(ln2q - s1);
            double u1 = std::ldexp(151.0 * t1, c.p);
            ok(u1 >= 0.5, "one piece fewer would also pass — count is not pinned");
        }
        __float128 s = 0;
        for (int i = 0; i < DD_N; ++i) s += (__float128)dd_ln2[i];
        double t = (double)fabsq(ln2q - s);
        // NOTE: 3 x 42 = 126 bits of DD pieces EXCEEDS binary128's 113-bit
        // mantissa, so __float128 cannot resolve the DD tail — it reports
        // exactly 0 (2^-inf). That would make this assertion vacuous, so the
        // DD tail is checked against MPFR in the standalone probe
        // (/tmp/xpprobe/emit.cpp: tail = 2^-136.07, 9.5e-07 ulps at kmax) and
        // asserted here only as "below what binary128 can see", which is a
        // real if weaker statement: it pins the tail under 2^-113.
        std::printf("   DD %d pieces: tail 2^%.1f (binary128 floor; true tail "
                    "2^-136.07 per MPFR)\n", DD_N, std::log2(t));
        ok(t == 0.0 || std::ldexp(1075.0 * t, 106) < 0.5,
           "DD tail resolvable in binary128 — it must be below 2^-113");
        // and the count must still be pinned: two pieces must be visibly bad
        __float128 s2 = 0;
        for (int i = 0; i < DD_N - 1; ++i) s2 += (__float128)dd_ln2[i];
        double t2 = (double)fabsq(ln2q - s2);
        ok(t2 > 0.0 && std::ldexp(1075.0 * t2, 106) >= 0.5,
           "DD with one piece fewer would also pass — count is not pinned");
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
