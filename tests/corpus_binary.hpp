#pragma once
// ============================================================================
// corpus_binary.hpp — on-disk format + loader for the shared cross-backend
//                     validation corpus produced by scripts/gen_corpus.cpp
// ============================================================================
//
// NO SPDX header: like tests/corpus.hpp, this is DOWNSTREAM-ONLY test
// scaffolding. It is not written for upstreaming to Kokkos.
//
// WHAT THIS IS
//   The single source of truth for the corpus binary layout. Both the generator
//   (scripts/gen_corpus.cpp) and every consumer include this header, so the
//   writer and the reader can never drift apart.
//
// WHY A SHARED CORPUS EXISTS
//   Today each demo generates its own inputs from its own seed, so DD, FF and QF
//   are scored on DIFFERENT numbers and their accuracy tables are not directly
//   comparable. The corpus fixes one set of inputs, plus one __float128
//   reference per (op, element), for every backend to be scored against.
//
//   This is NOT a performance optimization. On this hardware the __float128
//   oracle is ~1% of a QF demo's runtime, so caching references saves nothing
//   worth having. See the header comment of scripts/gen_corpus.cpp.
//
// STALENESS GUARD (the reason this header is strict)
//   A cached reference that silently goes stale is worse than no cache at all:
//   the suite stays green while scoring against wrong numbers. So the loader
//   FAILS LOUDLY — it throws CorpusError, it never warns-and-continues — on:
//     * wrong magic                     (not a corpus file)
//     * format_version mismatch         (layout changed)
//     * generator_version mismatch      (SAMPLING or REFERENCE MATH changed;
//                                        this is the one that catches a stale
//                                        file produced by an older generator)
//     * header_bytes / op_record_bytes mismatch (ABI/padding drift)
//     * declared payload size vs. actual file size mismatch (truncated file)
//     * a requested op missing from the file (require_ops)
//   Content corruption is caught by verify_checksum(), which is a separate,
//   explicit call because it re-reads the whole (multi-GB) payload.
//
// INTENDED CONSUMER USAGE
//     #include "corpus_binary.hpp"
//     namespace cb = kokkos_ep::corpus_binary;
//
//     cb::Corpus corpus("scripts/xp_corpus.bin");   // throws if stale/corrupt
//     corpus.require_ops({"add", "sin", "pow"}, {}); // throws if any missing
//
//     cb::RealOpData d = corpus.load_real("sin");
//     for (uint64_t i = 0; i < d.n; ++i) {
//       MyBackend y = my_sin(MyBackend(d.a[i]));
//       cb::QuadRef r = cb::decode_ref128(d.ref[i]);       // no __float128
//       double words[2] = {y.hi, y.lo};
//       double dig = cb::ref_digits(words, 2, r, /*cap=*/31.0);
//     }
//     // d.n_corner / d.n_cancel let a consumer score the families separately:
//     //   [0, n_corner)                    corner cases (tests/corpus.hpp)
//     //   [n_corner, n_corner + n_cancel)  cancellation-prone operand pairs
//     //   [n_corner + n_cancel, n)         log-uniform random magnitudes
//
// NO libquadmath ON THE CONSUMER SIDE (this is the point of the split)
//   The GENERATOR (scripts/gen_corpus.cpp) needs __float128 and libquadmath —
//   that is where the reference math happens and it is correct that it stays
//   there. A CONSUMER only ever needs to COMPARE against a reference, and a
//   reference on disk is just sixteen bytes of IEEE-754 binary128. So this
//   header never mentions __float128 on the path a consumer walks: references
//   load as `Ref128` (raw bytes), `decode_ref128()` turns those bytes into an
//   EXACT three-double expansion with a separate binary exponent, and
//   `ref_digits()` scores a backend value against it in plain `double`
//   arithmetic (Knuth two-sum expansions — exact, no wide type required).
//
//   Why this matters: hipcc and icpx are clang-based and the GCC-flavoured
//   quadmath oracle is simply unavailable there, so every oracle-dependent test
//   runtime-SKIPs (exit 77) on an AMD or Intel toolchain and those runs report
//   no accuracy signal at all. With the split, the corpus is generated ONCE on
//   an x86_64 GCC host and consumed anywhere, by a TU that needs neither
//   -lquadmath nor -fext-numeric-literals.
//
//   A `__float128` convenience conversion is still offered, but only behind
//   `#if defined(KOKKOS_EP_CORPUS_HAVE_FLOAT128)`, and nothing in this header
//   depends on it. It exists so a host that DOES have quadmath can cross-check
//   the decoder (see tests/README.md); a consumer must not reach for it.
//
// HOST ONLY. The reader uses <cstdio> and std::vector, so it belongs in host
//   scoring code, never in a .cpp compiled for a GPU backend. That is a
//   host/device split, not a quadmath one.
//
// ENDIANNESS. The format is raw little-endian IEEE data. The generator is
//   x86_64-only (libquadmath), so no byte-swapping layer is provided; the
//   binary128 decoder below reads the 16 bytes little-endian to match.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Availability of the OPTIONAL __float128 conveniences. Never required.
#if defined(__SIZEOF_FLOAT128__)
#  define KOKKOS_EP_CORPUS_HAVE_FLOAT128 1
#endif

namespace kokkos_ep {
namespace corpus_binary {

// ----------------------------------------------------------------------------
// Format identity. Bump kFormatVersion when the LAYOUT changes; bump
// kGeneratorVersion when the SAMPLING or the REFERENCE MATH changes (a file's
// bytes are still readable, but its numbers no longer mean what a current
// consumer thinks they mean). Both are checked on open.
// ----------------------------------------------------------------------------
static const char     kMagic[8]         = {'X', 'P', 'C', 'O', 'R', 'P', 'U', 'S'};
static const uint32_t kFormatVersion    = 1u;
static const char     kGeneratorVersion[] = "gen_corpus/1.0.0";

enum : uint8_t {
  kKindReal    = 0,
  kKindComplex = 1,
};

enum : uint64_t {
  kFlagWithNan = 1ull << 0,  // NaN inputs were included (corpus::CorpusFlags::include_nan)
};

// ----------------------------------------------------------------------------
// File layout
//
//   [Header            : 128 bytes                     ]
//   [OpRecord[n_ops]   : 64 bytes each                 ]
//   [data block 0      : at OpRecord[0].offset         ]
//   [data block 1      : ...                           ]
//   ...
//
// A data block always stores the __float128 reference array FIRST (so it lands
// on the 16-byte alignment __float128 wants), then the double operand arrays,
// then padding up to a multiple of 16 bytes:
//
//   real op, k operands:      ref[n]  a[n]  [b[n]]  [c[n]]        pad
//                             16n B   8n B   8n B    8n B
//   complex op, k operands:   ref[2n] a[2n] [b[2n]]                pad
//                             32n B   16n B  16n B
//
// Complex arrays are interleaved (re, im, re, im, ...).
// ----------------------------------------------------------------------------
struct Header {
  char     magic[8];              //   0  "XPCORPUS"
  uint32_t format_version;        //   8
  uint32_t header_bytes;          //  12  sizeof(Header)
  uint64_t n_elems;               //  16  elements per op
  uint32_t n_ops;                 //  24  number of OpRecords that follow
  uint32_t op_record_bytes;       //  28  sizeof(OpRecord)
  uint64_t seed;                  //  32  RNG seed the generator was given
  uint64_t flags;                 //  40  kFlag* bitmask
  uint64_t payload_bytes;         //  48  total bytes of all data blocks
  uint64_t payload_checksum;      //  56  FNV-1a-64 (word-wise, see checksum())
  char     generator_version[48]; //  64  NUL-padded, must equal kGeneratorVersion
  uint64_t reserved[2];           // 112
};                                // 128

struct OpRecord {
  char     name[24];              //   0  NUL-padded op name ("add", "sin", ...)
  uint8_t  kind;                  //  24  kKindReal / kKindComplex
  uint8_t  n_operands;            //  25  1..3
  uint16_t reserved16;            //  26
  uint32_t reserved32;            //  28
  uint64_t offset;                //  32  absolute byte offset of the data block
  uint64_t bytes;                 //  40  data block size, including padding
  uint64_t n_corner;              //  48  leading elements drawn from corpus.hpp
  uint64_t n_cancel;              //  56  elements after those that near-cancel
};                                // 64

static_assert(sizeof(Header) == 128,  "corpus Header layout drifted");
static_assert(sizeof(OpRecord) == 64, "corpus OpRecord layout drifted");

// ----------------------------------------------------------------------------
// Payload checksum. FNV-1a with a 64-bit-word stride (little-endian) plus a
// byte tail; defined here so writer and reader are literally the same code.
// ----------------------------------------------------------------------------
static const uint64_t kChecksumSeed  = 1469598103934665603ull;  // FNV offset basis
static const uint64_t kChecksumPrime = 1099511628211ull;        // FNV prime

inline uint64_t checksum_update(uint64_t h, const void* data, size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    h ^= w;
    h *= kChecksumPrime;
  }
  for (; i < bytes; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= kChecksumPrime;
  }
  return h;
}

// Block size helper — shared by writer and reader so they cannot disagree.
inline uint64_t block_bytes(uint8_t kind, uint8_t n_operands, uint64_t n) {
  const uint64_t lanes = (kind == kKindComplex) ? 2u : 1u;
  uint64_t b = 16u * lanes * n                        // reference array
             + 8u * lanes * n * uint64_t(n_operands); // operand arrays
  return (b + 15u) & ~uint64_t(15u);                  // pad to 16
}

// ============================================================================
// QUADMATH-FREE REFERENCE VALUES
// ============================================================================
// On disk a reference is one IEEE-754 binary128: 16 little-endian bytes,
// 1 sign + 15 exponent + 112 stored fraction bits. `Ref128` is exactly those
// bytes, so loading needs no wide type; `decode_ref128()` unpacks them.
//
// WHY AN EXPANSION AND NOT A DOUBLE PAIR
//   A hi/lo double pair carries 106 bits. DD's own resolution is 104 bits and
//   its accuracy cap is 31.0 decimal digits (103 bits), which leaves under one
//   decimal digit of headroom — thin enough that the reference would start
//   contributing to the number being reported. Three doubles carry 159 bits,
//   which holds all 113 bits of a binary128 EXACTLY, so the decode is lossless
//   and the oracle stays ~10 digits clear of DD, exactly as the __float128
//   oracle was.
//
//   The three components are the MANTISSA only; the binary exponent is kept
//   separately in `e`. That is what makes the decode total: a binary128
//   exponent runs to +-16383 and would overflow a double outright.
//   ----------------------------------------------------------------------
//   value  =  sign * (w[0] + w[1] + w[2]) * 2^e,   with w[0] + w[1] + w[2]
//   in [1, 2) for every finite nonzero value (subnormal binary128 included —
//   those are normalized on decode).
// ----------------------------------------------------------------------------
struct Ref128 {
  unsigned char b[16];
};
static_assert(sizeof(Ref128) == 16, "Ref128 must be exactly one binary128");

struct QuadRef {
  enum Class { kFinite = 0, kInf = 1, kNan = 2 };
  int    cls     = kFinite;
  int    sign    = 1;       // +1 / -1 (meaningful for kFinite and kInf)
  bool   is_zero = false;
  int    e       = 0;       // binary exponent (see the identity above)
  double w[3]    = {0.0, 0.0, 0.0};
};

inline QuadRef decode_ref128(const Ref128& r) {
  QuadRef q;
  const unsigned char* b = r.b;

  q.sign = (b[15] & 0x80u) ? -1 : 1;
  const uint32_t exp = (uint32_t(b[15] & 0x7Fu) << 8) | uint32_t(b[14]);

  // Fraction: 112 bits = bytes 0..13, little-endian. `fl` is bits 63..0,
  // `fh` is bits 111..64 (48 bits).
  uint64_t fl = 0, fh = 0;
  std::memcpy(&fl, b, 8);
  std::memcpy(&fh, b + 8, 6);

  if (exp == 0x7FFFu) {
    q.cls = (fl | fh) ? QuadRef::kNan : QuadRef::kInf;
    return q;
  }

  // Significand as a 113-bit integer M, split across (mh : ml).
  uint64_t mh = fh, ml = fl;
  int      e;
  if (exp != 0u) {
    mh |= (uint64_t(1) << 48);          // implicit leading bit -> M bit 112
    e = int(exp) - 16383;
  } else {
    if ((mh | ml) == 0u) { q.is_zero = true; return q; }
    // Subnormal binary128: no implicit bit. Shift the leading 1 up to bit 112
    // so the [1,2) mantissa invariant holds for every finite nonzero value.
    e = -16382;
    while ((mh & (uint64_t(1) << 48)) == 0u) {
      mh = (mh << 1) | (ml >> 63);
      ml <<= 1;
      --e;
    }
  }
  q.e = e;

  // M = m0*2^60 + m1*2^7 + m2, each chunk < 2^53 so each is exact in a double.
  //   m0 = M[112:60] (53 bits)   m1 = M[59:7] (53 bits)   m2 = M[6:0] (7 bits)
  const uint64_t kMask53 = (uint64_t(1) << 53) - 1u;
  const uint64_t m0 = (mh << 4) | (ml >> 60);
  const uint64_t m1 = (ml >> 7) & kMask53;
  const uint64_t m2 = ml & uint64_t(0x7F);

  // mantissa = M * 2^-112, in [1, 2).
  q.w[0] = std::ldexp(double(m0), -52);
  q.w[1] = std::ldexp(double(m1), -105);
  q.w[2] = std::ldexp(double(m2), -112);
  return q;
}

// ----------------------------------------------------------------------------
// Exact expansion arithmetic (Knuth two-sum / Shewchuk grow-expansion).
//
// Scoring needs |got - ref| to a few significant digits when got and ref agree
// to ~104 bits, so the difference MUST be formed exactly — a plain
// double-precision subtraction would return zero and report infinite accuracy.
// Both operands are exact sums of a handful of doubles, so a nonoverlapping
// expansion built by repeated grow-expansion holds the difference exactly, and
// the components can then simply be added up: they are already sorted by
// magnitude, so the sum's leading digits are correct.
//
// No multiplication anywhere, so FMA contraction cannot break these (unlike the
// Dekker twoProduct the *_eft_test targets have to compile with contraction
// off).
// ----------------------------------------------------------------------------
inline void ref_two_sum(double a, double b, double& s, double& err) {
  s             = a + b;
  const double v = s - a;
  err            = (a - (s - v)) + (b - v);
}

// Maximum expansion length: the longest backend value is QF (4 words) and a
// reference is 3, so 7 inputs -> at most 8 components. 12 is slack.
struct RefExpansion {
  double e[12];
  int    n = 0;

  void add(double x) {
    if (x == 0.0) return;
    double q = x;
    int    m = 0;
    for (int i = 0; i < n; ++i) {
      double s, err;
      ref_two_sum(q, e[i], s, err);
      q = s;
      if (err != 0.0) e[m++] = err;
    }
    if (q != 0.0) e[m++] = q;
    n = m;
  }

  double total() const {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += e[i];   // components ascend in magnitude
    return s;
  }
};

// ----------------------------------------------------------------------------
// ref_digits — decimal digits of agreement between a backend value and a
// binary128 reference, with NO wide float type anywhere.
//
//   got[0 .. n-1]  the backend's component words, widened to double. Every
//                  backend in this repo stores its value as an exact sum of
//                  FP32 or FP64 words, so this loses nothing: DD passes
//                  {hi, lo}, FF {hi, lo}, TF three floats, QF four.
//   cap            the backend's own precision ceiling (DD 31, QF 29, TF 21.7,
//                  FF 14). A bit-exact result returns exactly `cap`, matching
//                  the convention every *_accuracy_test already uses.
//
// Non-finite handling mirrors test_utils.hpp::digits_of_accuracy: NaN scores
// the cap only against NaN, an infinity only against a same-signed infinity,
// and an exact zero only against an exact zero.
// ----------------------------------------------------------------------------
inline double ref_digits(const double* got, int n, const QuadRef& r, double cap) {
  bool got_nan = false, got_inf = false, got_zero = true;
  double got_lead = 0.0;
  for (int i = 0; i < n; ++i) {
    if (std::isnan(got[i])) got_nan = true;
    else if (std::isinf(got[i])) { got_inf = true; if (got_lead == 0.0) got_lead = got[i]; }
    if (got[i] != 0.0) got_zero = false;
  }

  if (r.cls == QuadRef::kNan) return got_nan ? cap : 0.0;
  if (got_nan) return 0.0;
  if (r.cls == QuadRef::kInf)
    return (got_inf && ((got_lead > 0.0) == (r.sign > 0))) ? cap : 0.0;
  if (got_inf) return 0.0;
  if (r.is_zero) return got_zero ? cap : 0.0;

  // Work in units of 2^e so neither side can overflow or underflow: the
  // reference mantissa is in [1, 2) by construction and a backend value that
  // agrees with it is too. ldexp is exact.
  RefExpansion d;
  for (int i = 0; i < n; ++i) {
    const double s = std::ldexp(got[i], -r.e);
    if (std::isinf(s)) return 0.0;          // got is astronomically too large
    d.add(s);
  }
  for (int i = 0; i < 3; ++i) d.add(-double(r.sign) * r.w[i]);

  const double diff   = std::fabs(d.total());
  const double refmag = r.w[0] + r.w[1] + r.w[2];   // in [1, 2)
  if (diff == 0.0) return cap;

  const double dig = -std::log10(diff / refmag);
  if (!(dig > 0.0)) return 0.0;
  return dig > cap ? cap : dig;
}

// Magnitude of a reference as a base-2 exponent, without ever materializing the
// value. Domain predicates use this to exclude elements a backend structurally
// cannot hold (an FP32 word tops out near 2^128), which is an EXPONENT-RANGE
// limit, not an accuracy one — the same argument tf_accuracy_test.cpp's
// sqr_in_domain makes.
inline int ref_exponent(const QuadRef& r) { return r.e; }

#if defined(KOKKOS_EP_CORPUS_HAVE_FLOAT128)
// OPTIONAL cross-check only — see the header comment. Nothing in this file or
// in any consumer needs this; it exists so a quadmath-capable host can assert
// decode_ref128() against the real thing.
inline __float128 ref_to_float128(const Ref128& r) {
  __float128 v;
  std::memcpy(&v, r.b, 16);
  return v;
}
#endif

// ----------------------------------------------------------------------------
// Errors. One type, always thrown, message always names the file.
// ----------------------------------------------------------------------------
struct CorpusError : std::runtime_error {
  explicit CorpusError(const std::string& what) : std::runtime_error(what) {}
};

// ----------------------------------------------------------------------------
// Loaded data.
// ----------------------------------------------------------------------------
// `ref*` fields are RAW binary128 bytes — run them through decode_ref128() (or
// straight into ref_digits()). They were std::vector<__float128> before the
// quadmath split; nothing on the consumer side needs the wide type.
struct RealOpData {
  std::string          name;
  uint64_t             n          = 0;
  uint64_t             n_corner   = 0;
  uint64_t             n_cancel   = 0;
  int                  n_operands = 1;
  std::vector<double>  a, b, c;    // b/c empty when the op does not use them
  std::vector<Ref128>  ref;
};

struct ComplexOpData {
  std::string          name;
  uint64_t             n          = 0;
  uint64_t             n_corner   = 0;
  uint64_t             n_cancel   = 0;
  int                  n_operands = 1;
  std::vector<double>  a_re, a_im, b_re, b_im;   // b_* empty for unary ops
  std::vector<Ref128>  ref_re, ref_im;
};

// ----------------------------------------------------------------------------
// Reader.
// ----------------------------------------------------------------------------
class Corpus {
 public:
  explicit Corpus(const std::string& path) : path_(path) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) fail("cannot open corpus file");

    if (std::fread(&hdr_, sizeof(hdr_), 1, f_) != 1) fail("short read on header");

    if (std::memcmp(hdr_.magic, kMagic, 8) != 0)
      fail("bad magic — not a corpus file (expected \"XPCORPUS\")");
    if (hdr_.header_bytes != sizeof(Header))
      fail("header_bytes " + std::to_string(hdr_.header_bytes) + " != " +
           std::to_string(sizeof(Header)) + " — struct layout drifted; rebuild consumer");
    if (hdr_.op_record_bytes != sizeof(OpRecord))
      fail("op_record_bytes " + std::to_string(hdr_.op_record_bytes) + " != " +
           std::to_string(sizeof(OpRecord)) + " — struct layout drifted; rebuild consumer");
    if (hdr_.format_version != kFormatVersion)
      fail("format_version " + std::to_string(hdr_.format_version) + " != expected " +
           std::to_string(kFormatVersion) + " — STALE CORPUS, regenerate with scripts/gen_corpus");

    char gv[49];
    std::memcpy(gv, hdr_.generator_version, 48);
    gv[48] = '\0';
    if (std::strcmp(gv, kGeneratorVersion) != 0)
      fail(std::string("generator_version \"") + gv + "\" != expected \"" + kGeneratorVersion +
           "\" — STALE CORPUS (sampling or reference math changed), regenerate with "
           "scripts/gen_corpus");

    if (hdr_.n_ops == 0) fail("header declares zero ops");
    if (hdr_.n_elems == 0) fail("header declares zero elements per op");

    ops_.resize(hdr_.n_ops);
    if (std::fread(ops_.data(), sizeof(OpRecord), hdr_.n_ops, f_) != hdr_.n_ops)
      fail("short read on op table");

    // Structural cross-check: every block sized and placed as the format says,
    // and the file is exactly as long as the header claims.
    const uint64_t first = sizeof(Header) + uint64_t(hdr_.n_ops) * sizeof(OpRecord);
    uint64_t expect_off = first, payload = 0;
    for (uint32_t i = 0; i < hdr_.n_ops; ++i) {
      const OpRecord& r = ops_[i];
      if (r.kind != kKindReal && r.kind != kKindComplex)
        fail("op " + op_name(r) + ": bad kind " + std::to_string(int(r.kind)));
      if (r.n_operands < 1 || r.n_operands > 3)
        fail("op " + op_name(r) + ": bad n_operands " + std::to_string(int(r.n_operands)));
      const uint64_t want = block_bytes(r.kind, r.n_operands, hdr_.n_elems);
      if (r.bytes != want)
        fail("op " + op_name(r) + ": block is " + std::to_string(r.bytes) +
             " bytes, format says " + std::to_string(want));
      if (r.offset != expect_off)
        fail("op " + op_name(r) + ": block offset " + std::to_string(r.offset) +
             " != expected " + std::to_string(expect_off));
      if (r.n_corner + r.n_cancel > hdr_.n_elems)
        fail("op " + op_name(r) + ": n_corner + n_cancel exceeds n_elems");
      expect_off += r.bytes;
      payload += r.bytes;
    }
    if (payload != hdr_.payload_bytes)
      fail("payload_bytes " + std::to_string(hdr_.payload_bytes) + " != sum of blocks " +
           std::to_string(payload));

    if (std::fseek(f_, 0, SEEK_END) != 0) fail("seek to end failed");
    const long end = std::ftell(f_);
    if (end < 0) fail("ftell failed");
    if (uint64_t(end) != first + payload)
      fail("file is " + std::to_string(end) + " bytes, header implies " +
           std::to_string(first + payload) + " — TRUNCATED OR APPENDED-TO");
  }

  ~Corpus() { if (f_) std::fclose(f_); }
  Corpus(const Corpus&)            = delete;
  Corpus& operator=(const Corpus&) = delete;

  const Header&                header() const { return hdr_; }
  const std::vector<OpRecord>& ops()    const { return ops_; }
  const std::string&           path()   const { return path_; }

  const OpRecord* find(const std::string& name, uint8_t kind) const {
    for (const OpRecord& r : ops_)
      if (r.kind == kind && op_name(r) == name) return &r;
    return nullptr;
  }

  // Assert the file actually carries the ops a consumer is about to score. A
  // corpus generated before an op existed would otherwise just be "missing".
  void require_ops(const std::vector<std::string>& real,
                   const std::vector<std::string>& cplx) const {
    std::string missing;
    for (const std::string& s : real)
      if (!find(s, kKindReal)) missing += " real:" + s;
    for (const std::string& s : cplx)
      if (!find(s, kKindComplex)) missing += " complex:" + s;
    if (!missing.empty())
      fail("corpus is missing required ops —" + missing +
           " — STALE CORPUS, regenerate with scripts/gen_corpus");
  }

  RealOpData load_real(const std::string& name) {
    const OpRecord* r = find(name, kKindReal);
    if (!r) fail("no real op \"" + name + "\" in corpus");
    const uint64_t n = hdr_.n_elems;
    RealOpData d;
    d.name = name; d.n = n; d.n_corner = r->n_corner; d.n_cancel = r->n_cancel;
    d.n_operands = r->n_operands;
    seek(r->offset, name);
    d.ref.resize(n);            read_into(d.ref.data(), 16 * n, name);
    d.a.resize(n);              read_into(d.a.data(), 8 * n, name);
    if (r->n_operands >= 2) { d.b.resize(n); read_into(d.b.data(), 8 * n, name); }
    if (r->n_operands >= 3) { d.c.resize(n); read_into(d.c.data(), 8 * n, name); }
    return d;
  }

  ComplexOpData load_complex(const std::string& name) {
    const OpRecord* r = find(name, kKindComplex);
    if (!r) fail("no complex op \"" + name + "\" in corpus");
    const uint64_t n = hdr_.n_elems;
    ComplexOpData d;
    d.name = name; d.n = n; d.n_corner = r->n_corner; d.n_cancel = r->n_cancel;
    d.n_operands = r->n_operands;
    seek(r->offset, name);

    std::vector<Ref128> zref(2 * n);
    read_into(zref.data(), 32 * n, name);
    d.ref_re.resize(n); d.ref_im.resize(n);
    for (uint64_t i = 0; i < n; ++i) { d.ref_re[i] = zref[2*i]; d.ref_im[i] = zref[2*i+1]; }

    std::vector<double> za(2 * n);
    read_into(za.data(), 16 * n, name);
    d.a_re.resize(n); d.a_im.resize(n);
    for (uint64_t i = 0; i < n; ++i) { d.a_re[i] = za[2*i]; d.a_im[i] = za[2*i+1]; }

    if (r->n_operands >= 2) {
      read_into(za.data(), 16 * n, name);
      d.b_re.resize(n); d.b_im.resize(n);
      for (uint64_t i = 0; i < n; ++i) { d.b_re[i] = za[2*i]; d.b_im[i] = za[2*i+1]; }
    }
    return d;
  }

  // Full content check: re-reads the payload and compares against the header's
  // checksum. Separate from the constructor because at the default 1e6
  // elements/op the payload is multiple GB.
  uint64_t verify_checksum() {
    const uint64_t first = sizeof(Header) + uint64_t(hdr_.n_ops) * sizeof(OpRecord);
    seek(first, "<payload>");
    std::vector<unsigned char> buf(1u << 22);  // 4 MiB
    uint64_t h = kChecksumSeed, left = hdr_.payload_bytes;
    while (left) {
      const size_t want = size_t(left < buf.size() ? left : buf.size());
      if (std::fread(buf.data(), 1, want, f_) != want) fail("short read verifying checksum");
      h = checksum_update(h, buf.data(), want);
      left -= want;
    }
    if (h != hdr_.payload_checksum)
      fail("payload checksum " + hex(h) + " != header " + hex(hdr_.payload_checksum) +
           " — CORPUS CORRUPT, regenerate with scripts/gen_corpus");
    return h;
  }

  static std::string op_name(const OpRecord& r) {
    char b[25];
    std::memcpy(b, r.name, 24);
    b[24] = '\0';
    return std::string(b);
  }

  static std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%016llx", (unsigned long long)v);
    return std::string(b);
  }

 private:
  void fail(const std::string& msg) const {
    throw CorpusError("corpus_binary: " + path_ + ": " + msg);
  }
  void seek(uint64_t off, const std::string& who) {
    if (std::fseek(f_, long(off), SEEK_SET) != 0) fail("seek failed for " + who);
  }
  void read_into(void* dst, uint64_t bytes, const std::string& who) {
    if (std::fread(dst, 1, size_t(bytes), f_) != size_t(bytes))
      fail("short read loading " + who);
  }

  std::string           path_;
  std::FILE*            f_ = nullptr;
  Header                hdr_{};
  std::vector<OpRecord> ops_;
};

}  // namespace corpus_binary
}  // namespace kokkos_ep
