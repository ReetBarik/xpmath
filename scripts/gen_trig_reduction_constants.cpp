// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// ============================================================================
// gen_trig_reduction_constants.cpp — generator for include/xp/trig_reduction_data.hpp
// ============================================================================
//
// Emits the 2/pi chunk tables and the pi/2 expansions that Payne-Hanek
// argument reduction needs, AND derives their sizes from a measurement rather
// than from a rule of thumb.
//
// WHY THIS FILE IS COMMITTED
// --------------------------
// tests/exp_reduction_test.cpp:193 cites a /tmp path as the authority for a
// value it cannot verify. That is the thing this file exists not to repeat:
// every constant in trig_reduction_data.hpp is reproducible by
//
//     g++ -O2 -o gen scripts/gen_trig_reduction_constants.cpp -lmpfr -lgmp
//     ./gen --emit > include/xp/trig_reduction_data.hpp
//
// and every SIZE in it is printed, with its derivation, by ./gen --measure.
//
// THE SIZING ARGUMENT
// -------------------
// Payne-Hanek computes f = frac(x * 2/pi) and the quadrant n mod 4, then
// r = f * pi/2. TWO sizes come out of this and they are not the same number.
//
// TABLE DEPTH. Truncating the 2/pi table at depth bits leaves an absolute
// error of at most |x| * 2^-depth in the product, hence in f. The reduced
// argument r is proportional to f, so the RELATIVE error of r is
// (|x| * 2^-depth) / |f|. Requiring that to stay under 2^-p gives
//
//     depth  >=  D  +  p  +  slack,     D = max over x of log2(|x| / |f|)
//
// GUARD. The engine also carries only a finite number of fractional bits of
// x*(2/pi) below the binary point. To know an f of size 2^-C to a relative
// 2^-p it must carry
//
//     guard  >=  p  +  C  +  slack,     C = max over x of log2(1 / |f|)
//
// C is the worst-case cancellation of the format: the largest number of bits
// that can vanish when a representable x is reduced mod pi/2. It is NOT a
// free parameter and it is NOT 2p — it is measured here, per backend.
//
// D is NOT max log2|x| + C. Those two maxima are attained at different inputs
// (the deepest cancellations sit near |x| = 1, where the deep chunks are never
// reached), and combining them ships a table that no input can exercise. The
// symptom is silent: with an over-deep table, a one-chunk-short table is still
// correct, so the N-1 poison in tests/trig_reduction_test.cpp tests nothing.
// Both are measured in the same scan, separately, and reported separately.
//
// HOW C IS MEASURED
// -----------------
// For a fixed binade exponent e the reachable values are m * 2^e with m an
// integer in [1, N). Writing alpha = frac(2^e * 2/pi), the quantity that
// cancels is ||m * alpha||, the distance from m*alpha to the nearest integer.
// By the theory of best approximations of the second kind (three-distance
// theorem), min over 1 <= m <= N of ||m*alpha|| is attained at the largest
// continued-fraction convergent denominator of alpha that is <= N. So one CF
// expansion per exponent settles that exponent exactly, and scanning the
// format's exponent range settles the format.
//
// N is 2^p, and p is exactly the number of significand bits a normalised
// expansion carries with its words adjacent: 2*53 for DoubleDouble, 2*24 / 3*24
// / 4*24 for the float expansions. So every m in [1, N) IS a mantissa the
// backend can hold, which matters twice over: the sizing is not inflated by
// values the format cannot reach, and the argmax is an input the poison test in
// tests/trig_reduction_test.cpp can actually feed to the engine.
//
// VALIDATION OF THE METHOD. Run with p = 52 (mantissas < 2^53, i.e. exactly the
// IEEE double grid) the search returns m = 6381956970095103 at e = 797 — the
// published worst case for double argument reduction, which is also the value
// scripts/sweep_accuracy.cpp cites in its grid family (5) comment. The method
// rediscovers it from first principles, having been told only 2/pi.
//
// WHY CHUNKS ARE 24 BITS (FP64 PATH) AND 12 BITS (FP32 PATH)
// -----------------------------------------------------------
// The reduction convolves input mantissa digits against table digits. Both are
// signed and bounded by 2^(cb-1) and 2^cb, so a product needs 2*cb bits and
// must be EXACT in the working word:
//     FP64 path: 2*24 = 48 <= 53   ok
//     FP32 path: 2*12 = 24 <= 24   ok
// A 24-bit chunking of the FP32 path would need double arithmetic; keeping the
// FP32 backends in float costs twice the chunks and is worth it.
// ============================================================================
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <mpfr.h>
#include <gmp.h>

// Working precision for the constant itself: deeper than the deepest table
// (DD's, ~1272 bits) with room to spare.
static const mpfr_prec_t kWork = 4096;
// Working precision for the continued-fraction search: the convergent
// denominators reach 2^107 and the residues 2^-120, so 6000 bits is ~50x the
// span actually exercised.
static const mpfr_prec_t kSearch = 6000;

static const int kSlackBits = 4;   // signed-digit carry bookkeeping

// ---------------------------------------------------------------------------
// C measurement
// ---------------------------------------------------------------------------

// One exponent's worth of the scan.
//
// alpha = frac(2^e * 2/pi); the inputs at this exponent are x = m * 2^e for
// integer m in [1, N). Two different things are extracted, because two
// different sizes depend on them:
//
//   C  = max over m of log2(1 / ||m*alpha||)          -> sizes the GUARD
//   D  = max over m of (log2 m + log2(1 / ||m*alpha||))
//      = max over m of log2(|x| / |f|) - e            -> sizes the TABLE
//
// C and D are NOT attained at the same m and must not be combined. The guard is
// a depth below the BINARY POINT, so it only cares how small |f| gets. The
// table is a depth below the TOP OF 2/pi, and the product x*(2/pi) truncated
// there carries absolute error |x| * 2^-depth, so what it cares about is the
// RATIO |x|/|f|. A large m helps the ratio and hurts nothing else.
//
// CANDIDATE SET. min ||m*alpha|| over m <= N is attained at the largest
// continued-fraction convergent denominator <= N (best approximation of the
// second kind). For the ratio, the candidates are the best approximations of
// the FIRST kind, i.e. the convergents together with the semiconvergents
// q_{k-1} + j*q_k. Within one CF step, ||q_{k-1} + j*q_k|| decreases
// monotonically in j while m increases, so only the largest admissible j at
// each step can win; that is the single extra candidate scored per step. The
// step that overruns N is scored too, which is what covers m in (q_L, N].
//
// A provable upper bound over ALL m <= N is log2 N + C -- every m is at most N
// and no m cancels deeper than the best approximation does. That bound is
// reported next to the measured D so the gap between "what an input can
// actually do" and "what nothing can exceed" is visible rather than assumed.
struct ExpScan {
  double C = -1e30;      // log2(1/||m*alpha||) at the deepest-cancelling m
  double D = -1e30;      // max of log2 m + log2(1/||m*alpha||)
  double bound = -1e30;  // log2 N + C, the provable ceiling on D
  double R = -1e30;      // max of log2 m + min(p + C(m), floor_bits): the
                         // TABLE requirement, capped at what the output format
                         // can represent at all
  double Rm = 0;         // log2 m at the argmax of R
  double RC = 0;         // log2(1/||m*alpha||) there
  bool   skip = false;   // alpha == 0: x is an exact multiple of pi/2 here
  std::string cm, rm;    // the two argmax mantissas
};

// log2|v| for a positive mpfr value.
static double l2(mpfr_srcptr v) {
  mpfr_t t; mpfr_init2(t, 64);
  mpfr_log2(t, v, MPFR_RNDN);
  double r = mpfr_get_d(t, MPFR_RNDN);
  mpfr_clear(t);
  return r;
}

static ExpScan scan_at_exp(mpfr_srcptr two_over_pi, long e, mpz_srcptr N,
                           int p, double floor_bits) {
  ExpScan out;
  mpfr_t alpha, x, t, v;
  mpfr_inits2(kSearch, alpha, x, t, v, (mpfr_ptr)0);
  mpfr_mul_2si(alpha, two_over_pi, e, MPFR_RNDN);
  mpfr_frac(alpha, alpha, MPFR_RNDN);
  if (mpfr_sgn(alpha) == 0) {
    out.skip = true;
    mpfr_clears(alpha, x, t, v, (mpfr_ptr)0);
    return out;
  }

  mpz_t q_prev, q, q_new, a_z, j_z, m_z, cm, rm, room;
  mpz_inits(q_prev, q, q_new, a_z, j_z, m_z, cm, rm, room, (mpz_ptr)0);
  mpz_set_ui(q_prev, 0);
  mpz_set_ui(q, 1);
  mpz_set_ui(cm, 1);
  mpz_set_ui(rm, 1);

  // score one candidate m: update C (needs the smallest ||m*alpha||) and D
  // (needs the largest log2 m + log2(1/||m*alpha||)).
  auto score = [&](mpz_srcptr m) {
    if (mpz_sgn(m) <= 0 || mpz_cmp(m, N) > 0) return;
    mpfr_set_z(v, m, MPFR_RNDN);
    mpfr_mul(v, v, alpha, MPFR_RNDN);
    mpfr_round(t, v);
    mpfr_sub(v, v, t, MPFR_RNDN);
    mpfr_abs(v, v, MPFR_RNDN);
    if (mpfr_sgn(v) == 0) return;              // exact hit; no finite C
    const double c = -l2(v);
    if (c > out.C) { out.C = c; mpz_set(cm, m); }
    mpfr_set_z(t, m, MPFR_RNDN);
    const double lm = l2(t);
    const double d = lm + c;
    if (d > out.D) out.D = d;
    // What the table actually has to deliver: absolute error |x| * 2^-depth
    // must sit under the smallest thing the OUTPUT can hold at this magnitude,
    // which is |f| * 2^-p until |f| * 2^-p drops below the format's smallest
    // subnormal, and the subnormal after that.
    double want = (double)p + c;
    if (want > floor_bits) want = floor_bits;
    const double r = lm + want;
    if (r > out.R) { out.R = r; out.Rm = lm; out.RC = c; mpz_set(rm, m); }
  };

  mpz_set_ui(m_z, 1); score(m_z);
  mpfr_set(x, alpha, MPFR_RNDN);

  for (int it = 0; it < 2000; ++it) {
    if (mpfr_sgn(x) == 0) break;
    mpfr_ui_div(x, 1, x, MPFR_RNDN);
    mpfr_floor(t, x);
    mpfr_get_z(a_z, t, MPFR_RNDD);
    mpfr_sub(x, x, t, MPFR_RNDN);

    // largest semiconvergent q_prev + j*q that still fits in [1, N]
    mpz_sub(room, N, q_prev);
    if (mpz_sgn(room) >= 0) {
      mpz_fdiv_q(j_z, room, q);
      if (mpz_cmp(j_z, a_z) > 0) mpz_set(j_z, a_z);
      if (mpz_sgn(j_z) > 0) {
        mpz_mul(m_z, j_z, q);
        mpz_add(m_z, m_z, q_prev);
        score(m_z);
      }
    }

    mpz_mul(q_new, a_z, q);
    mpz_add(q_new, q_new, q_prev);
    if (mpz_cmp(q_new, N) > 0) break;
    mpz_set(q_prev, q);
    mpz_set(q, q_new);
    score(q);
  }

  if (out.C > -1e29) {
    mpfr_set_z(t, N, MPFR_RNDN);
    out.bound = l2(t) + out.C;
    char* s = mpz_get_str(nullptr, 10, cm); out.cm = s; free(s);
    s = mpz_get_str(nullptr, 10, rm);       out.rm = s; free(s);
  } else {
    out.skip = true;
  }
  mpz_clears(q_prev, q, q_new, a_z, j_z, m_z, cm, rm, room, (mpz_ptr)0);
  mpfr_clears(alpha, x, t, v, (mpfr_ptr)0);
  return out;
}

// VALIDATION OF THE CANDIDATE SET.
//
// scan_at_exp scores O(log N) values of m and claims the maximum of
// log2 m + log2(1/||m*alpha||) over ALL m <= N. For C alone that claim is a
// theorem. For the ratio it rests on the semiconvergents being the only places
// the running maximum can move, which is an argument, not a proof -- so it is
// checked the way everything else in this file is checked: by brute force at a
// size where brute force is possible. Every m from 1 to 2^nbits is enumerated
// at each of a spread of exponents and the two answers compared.
//
// Returns the largest discrepancy in bits (0.0 means the enumeration never
// missed a maximum). A nonzero result invalidates the table sizing and the
// caller refuses to emit.
static double validate_candidate_set(mpfr_srcptr two_over_pi, int nbits,
                                     long e_lo, long e_hi, long e_step) {
  mpz_t N; mpz_init(N);
  mpz_ui_pow_ui(N, 2, (unsigned long)nbits);
  const long M = 1L << nbits;
  double worst_gap = 0.0;
  mpfr_t alpha, v, t;
  mpfr_inits2(256, alpha, v, t, (mpfr_ptr)0);
  for (long e = e_lo; e <= e_hi; e += e_step) {
    mpfr_mul_2si(alpha, two_over_pi, e, MPFR_RNDN);
    mpfr_frac(alpha, alpha, MPFR_RNDN);
    if (mpfr_sgn(alpha) == 0) continue;
    double brute = -1e30;
    for (long m = 1; m <= M; ++m) {
      mpfr_mul_si(v, alpha, m, MPFR_RNDN);
      mpfr_round(t, v);
      mpfr_sub(v, v, t, MPFR_RNDN);
      mpfr_abs(v, v, MPFR_RNDN);
      if (mpfr_sgn(v) == 0) continue;
      const double d = std::log2((double)m) - l2(v);
      if (d > brute) brute = d;
    }
    const ExpScan sc = scan_at_exp(two_over_pi, e, N, 0, 1e30);
    if (sc.skip) continue;
    const double gap = brute - sc.D;
    if (gap > worst_gap) worst_gap = gap;
  }
  mpfr_clears(alpha, v, t, (mpfr_ptr)0);
  mpz_clear(N);
  return worst_gap;
}

struct Backend {
  const char* name;
  int   p;          // significand bits of the expansion
  long  max_exp2;   // format ceiling: every finite |x| is < 2^max_exp2
  int   chunk_bits; // 24 (FP64 path) or 12 (FP32 path)
  double floor_bits;// -log2 of the smallest positive value a word can hold
                    // (1074 for FP64, 149 for FP32) -- the output cannot carry
                    // anything finer, so the table need not deliver it
  // filled by measure()
  long   emax;       // top of the scanned exponent range, = max_exp2 - p - 1
  double C;          // max over x of log2(1/|f|)        -- sizes the GUARD
  long   worst_e;    // argmax of C
  std::string worst_m;
  double D;          // max over x of log2(|x|/|f|)
  double R;          // max over x of log2|x| + min(p + C(x), floor_bits)
                     //                                  -- sizes the TABLE
  long   pin_e;      // argmax of R
  std::string pin_m;
  double pin_C;      // log2(1/|f|) at the pinning input
  double pin_log2x;  // log2|x| at the pinning input
  double bound;      // provable ceiling on D (log2 N + C, maximised over e)
  int    chunks;     // derived table depth, in chunks
  double need_bits;  // derived table depth, in bits, before rounding up
  int    guard;      // fractional bits of x*(2/pi) the engine must carry
};

// THE SCAN RANGE. A value of the format is x = m * 2^e with the mantissa m in
// [1, 2^p) and m * 2^e < 2^max_exp2, so e never exceeds max_exp2 - p. Letting m
// range over all of [1, N) at every e in [-p, emax] covers every such value at
// least once (values with fewer significant bits simply appear at several e),
// and admits nothing whose magnitude the format cannot hold. Taking e up to
// max_exp2 instead -- which an earlier draft of this file did -- lets |x| reach
// 2^(max_exp2 + p), and sizes the table for arguments that cannot exist.
//
// This is the value set of a NORMALISED expansion. DoubleDouble and friends can
// also hold hi + lo pairs with an arbitrary gap between the words, e.g.
// 2^1000 + 2^-1000, whose reduction can cancel far more than C bits. Those are
// out of scope for the sizing and the limit is measured, not assumed: see the
// span sweep in tests/trig_reduction_test.cpp.
static void measure(mpfr_srcptr two_over_pi, Backend& b) {
  mpz_t N;
  mpz_init(N);
  mpz_ui_pow_ui(N, 2, (unsigned long)b.p);
  b.emax = b.max_exp2 - b.p;
  double bestC = -1e30, bestD = -1e30, bestB = -1e30, bestR = -1e30;
  long ce = 0, re = 0;
  std::string cm, rm;
  double pinC = 0, pinX = 0;
  // Only |x| >= 1 can cancel: below pi/4 the reduction is the identity and no
  // table is consulted at all.
  for (long e = -b.p; e <= b.emax; ++e) {
    ExpScan sc = scan_at_exp(two_over_pi, e, N, b.p, b.floor_bits);
    if (sc.skip) continue;
    if (sc.C > bestC) { bestC = sc.C; ce = e; cm = sc.cm; }
    if ((double)e + sc.D > bestD) bestD = (double)e + sc.D;
    if ((double)e + sc.bound > bestB) bestB = (double)e + sc.bound;
    const double r = (double)e + sc.R;
    if (r > bestR) {
      bestR = r; re = e; rm = sc.rm; pinC = sc.RC; pinX = (double)e + sc.Rm;
    }
  }
  b.C = bestC; b.worst_e = ce; b.worst_m = cm;
  b.D = bestD; b.bound = bestB;
  b.R = bestR; b.pin_e = re; b.pin_m = rm; b.pin_C = pinC; b.pin_log2x = pinX;
  // NO SLACK IS ADDED HERE, and that is deliberate. Slack in the table is
  // invisible: an extra chunk beyond what any input can reach makes a
  // one-chunk-short table correct, and the N-1 arm of the poison test then
  // proves nothing. Rounding R up to a whole chunk already leaves headroom,
  // and how much is printed per backend rather than assumed. The slack that
  // signed-digit carry bookkeeping genuinely needs goes in the GUARD, where it
  // costs accumulator words rather than table depth.
  b.need_bits = b.R;
  b.chunks = (int)std::ceil(b.need_bits / (double)b.chunk_bits);
  b.guard = (int)std::ceil((double)b.p + b.C + (double)kSlackBits);
  mpz_clear(N);
}

// ---------------------------------------------------------------------------
// chunk extraction
// ---------------------------------------------------------------------------

// 2/pi = sum_k chunk[k] * 2^(-cb*(k+1)), each chunk an integer in [0, 2^cb).
static void chunks_of_two_over_pi(mpfr_srcptr two_over_pi, int cb, int n,
                                  std::vector<long>& out) {
  out.assign(n, 0);
  mpfr_t x, t;
  mpfr_inits2(kWork, x, t, (mpfr_ptr)0);
  mpfr_set(x, two_over_pi, MPFR_RNDN);
  mpfr_frac(x, x, MPFR_RNDN);      // 2/pi = 0.63..., so this only drops a 0
  for (int k = 0; k < n; ++k) {
    mpfr_mul_2si(x, x, cb, MPFR_RNDN);
    mpfr_floor(t, x);
    out[k] = mpfr_get_si(t, MPFR_RNDD);
    mpfr_sub(x, x, t, MPFR_RNDN);
  }
  mpfr_clears(x, t, (mpfr_ptr)0);
}

static double narrow(mpfr_srcptr r, double*) { return mpfr_get_d(r, MPFR_RNDN); }
static float  narrow(mpfr_srcptr r, float*)  { return mpfr_get_flt(r, MPFR_RNDN); }

// pi/2 as a non-overlapping expansion of `n` doubles or floats. Returns log2
// of |pi/2 - sum|, or -1e30 when the expansion is exact.
template <typename T>
static double pio2_expansion(mpfr_srcptr pi, int n, std::vector<T>& out) {
  out.assign(n, T(0));
  mpfr_t r, t;
  mpfr_inits2(kWork, r, t, (mpfr_ptr)0);
  mpfr_div_2ui(r, pi, 1, MPFR_RNDN);      // pi/2, exact
  for (int i = 0; i < n; ++i) {
    T v = narrow(r, (T*)nullptr);
    out[i] = v;
    mpfr_set_d(t, (double)v, MPFR_RNDN);
    mpfr_sub(r, r, t, MPFR_RNDN);
  }
  double lg = -1e30;
  if (mpfr_sgn(r) != 0) { mpfr_abs(t, r, MPFR_RNDN); lg = l2(t); }
  mpfr_clears(r, t, (mpfr_ptr)0);
  return lg;
}

// Smallest word count whose residual clears 2^-req, sized FROM THE REQUIREMENT
// and not padded: one word fewer must miss it, or the count-is-pinned
// assertion in tests/trig_reduction_test.cpp proves nothing. Returns 0 if no
// count within reach clears it (the FP32 expansion stalls at the subnormal
// floor, 2^-150.02, after six words).
template <typename T>
static int pio2_size_for(mpfr_srcptr pi, double req, std::vector<T>& out) {
  for (int n = 1; n <= 16; ++n)
    if (pio2_expansion<T>(pi, n, out) < -req) return n;
  out.clear();
  return 0;
}

// significant bits carried, read from the mantissa itself (never re-derived
// from a nominal width — see the byte-identical trap in exp_reduction_test.cpp)
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

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  bool do_emit = false, do_measure = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--emit"))    do_emit = true;
    else if (!std::strcmp(argv[i], "--measure")) do_measure = true;
    else { std::fprintf(stderr, "usage: %s [--measure] [--emit]\n", argv[0]); return 2; }
  }
  if (!do_emit && !do_measure) do_measure = true;

  mpfr_t pi, two_over_pi;
  mpfr_inits2(kWork, pi, two_over_pi, (mpfr_ptr)0);
  mpfr_const_pi(pi, MPFR_RNDN);
  mpfr_set_ui(two_over_pi, 2, MPFR_RNDN);
  mpfr_div(two_over_pi, two_over_pi, pi, MPFR_RNDN);

  // The FP32 backends share one table; the deepest (QF) contains the others as
  // prefixes, exactly as the ln2 pieces already do.
  Backend be[] = {
    {"DD", 106, 1024, 24, 1074, 0, 0, 0, "", 0, 0, 0, "", 0, 0, 0, 0, 0, 0},
    {"FF",  48,  128, 12,  149, 0, 0, 0, "", 0, 0, 0, "", 0, 0, 0, 0, 0, 0},
    {"TF",  72,  128, 12,  149, 0, 0, 0, "", 0, 0, 0, "", 0, 0, 0, 0, 0, 0},
    {"QF",  96,  128, 12,  149, 0, 0, 0, "", 0, 0, 0, "", 0, 0, 0, 0, 0, 0},
  };
  const int nbe = 4;
  for (int i = 0; i < nbe; ++i) measure(two_over_pi, be[i]);

  // Method validation: p = 52 reproduces the published double worst case.
  Backend dbl = {"double", 53, 1024, 24, 1074, 0, 0, 0, "", 0, 0, 0, "", 0, 0, 0, 0, 0, 0};
  measure(two_over_pi, dbl);

  if (do_measure) {
    std::printf("gen_trig_reduction_constants — measured sizing\n\n");
    std::printf("METHOD VALIDATION (IEEE double, mantissa < 2^53):\n");
    std::printf("  worst case  m = %s  at e = %ld\n", dbl.worst_m.c_str(), dbl.worst_e);
    std::printf("  C = %.3f bits\n", dbl.C);
    std::printf("  expected    m = 6381956970095103  at e = 797   (published)\n");
    std::printf("  %s\n\n",
                dbl.worst_m == "6381956970095103" && dbl.worst_e == 797
                  ? "MATCH — the search reproduces the literature from 2/pi alone"
                  : "*** MISMATCH — do not trust the sizing below ***");

    {
      const double gap = validate_candidate_set(two_over_pi, 14, -40, 200, 7);
      std::printf("CANDIDATE-SET VALIDATION (brute force over all m <= 2^14, 35 exponents):\n");
      // 2^-20 of a bit is 45 orders of magnitude below one chunk: this is a
      // rounding tolerance on two log2() calls, not a tolerance on the result.
      std::printf("  largest amount by which brute force beat the enumeration: %.3e bits\n", gap);
      std::printf("  %s\n\n", gap < 1e-6
                    ? "MATCH — the enumeration found every maximum"
                    : "*** MISS — the table sizing below is not trustworthy ***");
    }

    std::printf("%-4s %4s %6s %9s %10s %10s %9s %7s %8s %7s\n", "be", "p", "e<=",
                "C(meas)", "D(meas)", "D(bound)", "need", "chunks", "headroom",
                "guard");
    for (int i = 0; i < nbe; ++i) {
      const Backend& b = be[i];
      std::printf("%-4s %4d %6ld %9.3f %10.1f %10.1f %9.1f %7d %8.1f %7d\n",
                  b.name, b.p, b.emax, b.C, b.D, b.bound, b.need_bits,
                  b.chunks, b.chunks * b.chunk_bits - b.need_bits, b.guard);
    }
    std::printf("\n  guard = p + C + %d      C = max over x of log2(1/|f|)\n", kSlackBits);
    std::printf("  need  = max over x of  log2|x| + min(p + C(x), %s)\n", "floor");
    std::printf("          floor = 1074 (FP64 words) / 149 (FP32 words): the output is an\n");
    std::printf("          expansion of machine words and cannot carry anything finer, so\n");
    std::printf("          the table is not asked to deliver it either.\n");
    std::printf("  headroom is what rounding up to whole chunks left over. No slack is\n");
    std::printf("  added on top: an unreachable chunk would make the N-1 poison vacuous.\n\n");
    for (int i = 0; i < nbe; ++i)
      std::printf("  %-2s deepest cancellation: C = %8.3f  at e = %5ld  m = %s\n",
                  be[i].name, be[i].C, be[i].worst_e, be[i].worst_m.c_str());
    std::printf("\n  The inputs that PIN the table depth. These, and not the rows above,\n");
    std::printf("  are what a one-chunk-short table has to get wrong --\n");
    std::printf("  tests/trig_reduction_test.cpp drives exactly these:\n");
    for (int i = 0; i < nbe; ++i)
      std::printf("  %-2s need = %8.3f  log2|x| = %8.3f  C = %8.3f  e = %5ld  m = %s\n",
                  be[i].name, be[i].R, be[i].pin_log2x, be[i].pin_C, be[i].pin_e,
                  be[i].pin_m.c_str());
    std::printf("\n  N-1 check:\n");
    for (int i = 0; i < nbe; ++i) {
      const Backend& b = be[i];
      const int nm1 = (b.chunks - 1) * b.chunk_bits;
      std::printf("  %-2s shipped %5d bits, N-1 = %5d bits, pinning input needs %8.1f  -> %s\n",
                  b.name, b.chunks * b.chunk_bits, nm1, b.need_bits,
                  (nm1 < b.need_bits) ? "N-1 IS SHORT (poison bites)"
                                      : "*** N-1 STILL SUFFICES: the poison tests nothing ***");
    }
  }

  if (do_emit) {
    const int nd = be[0].chunks;                       // DD, 24-bit chunks
    int nf = 0;
    for (int i = 1; i < nbe; ++i) if (be[i].chunks > nf) nf = be[i].chunks;

    std::vector<long> cd, cf;
    chunks_of_two_over_pi(two_over_pi, 24, nd, cd);
    chunks_of_two_over_pi(two_over_pi, 12, nf, cf);
    // pi/2 must carry p + slack bits RELATIVE, because f * (pi/2) inherits its
    // relative error; the widest FP64 consumer is DD and the widest FP32 one
    // is QF.
    const double reqd = (double)be[0].p + kSlackBits;
    double reqf = 0;
    for (int i = 1; i < nbe; ++i)
      if ((double)be[i].p + kSlackBits > reqf) reqf = (double)be[i].p + kSlackBits;
    std::vector<double> pd; const int npd = pio2_size_for<double>(pi, reqd, pd);
    std::vector<float>  pf; const int npf = pio2_size_for<float>(pi, reqf, pf);
    const double residd = pio2_expansion<double>(pi, npd, pd);
    const double residf = pio2_expansion<float>(pi, npf, pf);

    // ---- self-checks before a single byte is emitted --------------------
    int bad = 0;
    for (int k = 0; k < nd; ++k)
      if (cd[k] < 0 || cd[k] >= (1L << 24)) { std::fprintf(stderr, "FP64 chunk %d out of range\n", k); ++bad; }
    for (int k = 0; k < nf; ++k)
      if (cf[k] < 0 || cf[k] >= (1L << 12)) { std::fprintf(stderr, "FP32 chunk %d out of range\n", k); ++bad; }
    // the two chunkings must describe the SAME constant on their common depth
    {
      const int common = (nd * 24 < nf * 12) ? nd * 24 : nf * 12;
      mpfr_t a, b, t;
      mpfr_inits2(kWork, a, b, t, (mpfr_ptr)0);
      mpfr_set_ui(a, 0, MPFR_RNDN); mpfr_set_ui(b, 0, MPFR_RNDN);
      for (int k = 0; k < nd && (k + 1) * 24 <= common; ++k) {
        mpfr_set_si(t, cd[k], MPFR_RNDN); mpfr_div_2ui(t, t, 24 * (k + 1), MPFR_RNDN);
        mpfr_add(a, a, t, MPFR_RNDN);
      }
      for (int k = 0; k < nf && (k + 1) * 12 <= common; ++k) {
        mpfr_set_si(t, cf[k], MPFR_RNDN); mpfr_div_2ui(t, t, 12 * (k + 1), MPFR_RNDN);
        mpfr_add(b, b, t, MPFR_RNDN);
      }
      mpfr_sub(t, a, b, MPFR_RNDN);
      if (mpfr_sgn(t) != 0) { std::fprintf(stderr, "the 24-bit and 12-bit chunkings disagree\n"); ++bad; }
      mpfr_clears(a, b, t, (mpfr_ptr)0);
    }
    for (size_t i = 0; i < pd.size(); ++i) if (sigbits(pd[i]) > 53) ++bad;
    for (size_t i = 0; i < pf.size(); ++i) if (sigbits(pf[i]) > 24) ++bad;
    if (npd == 0 || npf == 0) {
      std::fprintf(stderr, "no pi/2 expansion reaches the required precision\n");
      ++bad;
    } else {
      // one word fewer MUST miss the requirement, or the count is unpinned
      std::vector<double> sd; std::vector<float> sf;
      if (npd > 1 && pio2_expansion<double>(pi, npd - 1, sd) < -reqd) {
        std::fprintf(stderr, "pi/2 double expansion is one word too long\n"); ++bad;
      }
      if (npf > 1 && pio2_expansion<float>(pi, npf - 1, sf) < -reqf) {
        std::fprintf(stderr, "pi/2 float expansion is one word too long\n"); ++bad;
      }
    }
    // the sizes above are only as good as the enumeration that produced them
    if (validate_candidate_set(two_over_pi, 14, -40, 200, 7) >= 1e-6) {
      std::fprintf(stderr, "candidate-set validation failed; the sizing is not trustworthy\n");
      ++bad;
    }
    if (dbl.worst_m != "6381956970095103" || dbl.worst_e != 797) {
      std::fprintf(stderr, "the search no longer reproduces the published double worst case\n");
      ++bad;
    }
    if (bad) { std::fprintf(stderr, "%d self-check failures; nothing emitted\n", bad); return 1; }

    std::printf("// SPDX-License-Identifier: LicenseRef-DHB-License\n");
    std::printf("// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC\n");
    std::printf("//\n// GENERATED by scripts/gen_trig_reduction_constants.cpp --emit\n");
    std::printf("// Do not edit by hand. Regenerate with:\n");
    std::printf("//   g++ -O2 -o gen scripts/gen_trig_reduction_constants.cpp -lmpfr -lgmp\n");
    std::printf("//   ./gen --emit > include/xp/trig_reduction_data.hpp\n//\n");
    std::printf("// Sizes below are MEASURED, not assumed. ./gen --measure prints the\n");
    std::printf("// derivation; the short version, with C the worst-case cancellation found\n");
    std::printf("// by a continued-fraction search over each format's own value set:\n//\n");
    std::printf("//   guard = p + C + %d      C = max over x of log2(1/|f|), MEASURED\n", kSlackBits);
    std::printf("//   depth = max over x of  log2|x| + min(p + C(x), floor)\n");
    std::printf("//           floor = 1074 (FP64 words) / 149 (FP32 words) -- the output is\n");
    std::printf("//           an expansion of machine words and cannot carry finer, so the\n");
    std::printf("//           table is not asked to. NO slack is added to the depth: an\n");
    std::printf("//           unreachable chunk would make the N-1 poison vacuous. The\n");
    std::printf("//           headroom column of --measure is what rounding to whole\n");
    std::printf("//           chunks happens to leave.\n//\n");
    std::printf("// The two maxima are NOT the same input and must not be combined. The guard\n");
    std::printf("// is a depth below the BINARY POINT, so all it cares about is how small |f|\n");
    std::printf("// gets. The table is a depth below the top of 2/pi, and truncating it there\n");
    std::printf("// costs |x| * 2^-depth absolute, so what it cares about is the RATIO |x|/|f|.\n");
    std::printf("// Sizing the table as max log2|x| + max C ships chunks no input can reach,\n");
    std::printf("// and then a one-chunk-short table is still correct -- which is to say the\n");
    std::printf("// N-1 poison in tests/trig_reduction_test.cpp would test nothing.\n//\n");
    std::printf("//   %-4s %6s %8s %9s %10s %10s %8s %7s %9s %7s\n", "be", "p", "e<=",
                "C", "D(meas)", "D(bound)", "need", "chunks", "headroom", "guard");
    for (int i = 0; i < nbe; ++i)
      std::printf("//   %-4s %6d %8ld %9.3f %10.1f %10.1f %8.1f %7d %9.1f %7d\n",
                  be[i].name, be[i].p, be[i].emax, be[i].C, be[i].D, be[i].bound,
                  be[i].need_bits, be[i].chunks,
                  (double)(be[i].chunks * be[i].chunk_bits) - be[i].need_bits,
                  be[i].guard);
    std::printf("//\n// The inputs that PIN the table depth (argmax of log2(|x|/|f|)) -- these,\n");
    std::printf("// and not the deepest-cancellation rows, are what the N-1 arm has to get\n");
    std::printf("// wrong. tests/trig_reduction_test.cpp drives exactly these values:\n//\n");
    for (int i = 0; i < nbe; ++i)
      std::printf("//   %-2s  e = %5ld  log2|x| = %8.3f  C = %8.3f  m = %s\n",
                  be[i].name, be[i].pin_e, be[i].pin_log2x, be[i].pin_C,
                  be[i].pin_m.c_str());
    std::printf("//\n// The search validates against the published IEEE-double worst case\n");
    std::printf("// m = %s at e = %ld, which it rediscovers from 2/pi alone.\n",
                dbl.worst_m.c_str(), dbl.worst_e);
    std::printf("// ============================================================================\n");
    std::printf("#ifndef XP_TRIG_REDUCTION_DATA_HPP\n#define XP_TRIG_REDUCTION_DATA_HPP\n\n");
    std::printf("#include <xp/config.hpp>\n\nnamespace xp {\nnamespace detail {\n\n");

    std::printf("// 2/pi = sum_k kPhIpio2D[k] * 2^(-24*(k+1))\n");
    std::printf("inline constexpr int kPhChunkBits64 = 24;\n");
    std::printf("inline constexpr int kPhChunksDD    = %d;\n\n", be[0].chunks);
    std::printf("// 2/pi = sum_k kPhIpio2F[k] * 2^(-12*(k+1))\n");
    std::printf("inline constexpr int kPhChunkBits32 = 12;\n");
    for (int i = 1; i < nbe; ++i)
      std::printf("inline constexpr int kPhChunks%s    = %d;\n", be[i].name, be[i].chunks);
    std::printf("\n");

    std::printf("// physical lengths of the emitted tables: the deepest backend of each\n");
    std::printf("// type. Reading past these is out of bounds.\n");
    std::printf("inline constexpr int kPhIpio2LenD  = %d;\n", nd);
    std::printf("inline constexpr int kPhIpio2LenF  = %d;\n", nf);
    std::printf("inline constexpr int kPhPio2WordsD = %d;\n", npd);
    std::printf("inline constexpr int kPhPio2WordsF = %d;\n\n", npf);

    std::printf("// Fractional bits of x*(2/pi) the engine must carry: p + C + %d, with C the\n", kSlackBits);
    std::printf("// MEASURED worst-case cancellation (not 2p, not a rule of thumb).\n");
    for (int i = 0; i < nbe; ++i)
      std::printf("inline constexpr int kPhGuard%s      = %d;   // p=%d  C=%.3f\n",
                  be[i].name, be[i].guard, be[i].p, be[i].C);
    std::printf("\n");

    std::printf("XPMATH_INLINE_FUNCTION double xp_ph_ipio2_d(int k) {\n");
    std::printf("    constexpr double t[%d] = {\n", nd);
    for (int k = 0; k < nd; ++k)
      std::printf("        %8ld.0%s%s", cd[k], k + 1 < nd ? "," : "",
                  (k % 6 == 5 || k + 1 == nd) ? "\n" : "");
    std::printf("    };\n    return t[k];\n}\n\n");

    std::printf("XPMATH_INLINE_FUNCTION float xp_ph_ipio2_f(int k) {\n");
    std::printf("    constexpr float t[%d] = {\n", nf);
    for (int k = 0; k < nf; ++k)
      std::printf("        %5ld.0f%s%s", cf[k], k + 1 < nf ? "," : "",
                  (k % 8 == 7 || k + 1 == nf) ? "\n" : "");
    std::printf("    };\n    return t[k];\n}\n\n");

    std::printf("// pi/2 as a non-overlapping expansion. The word counts are the SMALLEST\n");
    std::printf("// that clear p + %d bits for the widest consumer of each type (DD needs\n", kSlackBits);
    std::printf("// %.0f, QF needs %.0f); one word fewer misses it, which is what makes the\n", reqd, reqf);
    std::printf("// count-is-pinned assertion in tests/trig_reduction_test.cpp bite.\n");
    std::printf("// MEASURED residuals: 2^%.2f (FP64, %d words) and 2^%.2f (FP32, %d words).\n",
                residd, npd, residf, npf);
    std::printf("XPMATH_INLINE_FUNCTION double xp_ph_pio2_d(int k) {\n    constexpr double t[%d] = {",
                (int)pd.size());
    for (size_t i = 0; i < pd.size(); ++i) std::printf("%s\n        %.20a", i ? "," : "", pd[i]);
    std::printf("\n    };\n    return t[k];\n}\n\n");
    std::printf("XPMATH_INLINE_FUNCTION float xp_ph_pio2_f(int k) {\n    constexpr float t[%d] = {",
                (int)pf.size());
    for (size_t i = 0; i < pf.size(); ++i) std::printf("%s\n        %.12af", i ? "," : "", pf[i]);
    std::printf("\n    };\n    return t[k];\n}\n\n");

    std::printf("}  // namespace detail\n}  // namespace xp\n\n#endif  // XP_TRIG_REDUCTION_DATA_HPP\n");
  }

  mpfr_clears(pi, two_over_pi, (mpfr_ptr)0);
  return 0;
}
