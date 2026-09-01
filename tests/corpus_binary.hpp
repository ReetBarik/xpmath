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
//       MyBackend y = my_sin(MyBackend(d.a[i]));     // score against d.ref[i]
//     }
//     // d.n_corner / d.n_cancel let a consumer score the families separately:
//     //   [0, n_corner)                    corner cases (tests/corpus.hpp)
//     //   [n_corner, n_corner + n_cancel)  cancellation-prone operand pairs
//     //   [n_corner + n_cancel, n)         log-uniform random magnitudes
//
// HOST ONLY. This header uses __float128 (libquadmath), which is x86_64 host
//   only and must not appear in a device translation unit. Include it from host
//   scoring code, never from a .cpp compiled for a GPU backend.
//
// ENDIANNESS. The format is raw little-endian IEEE data. The project is already
//   x86_64-only (libquadmath), so no byte-swapping layer is provided.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__SIZEOF_FLOAT128__)
#error "corpus_binary.hpp requires __float128 (x86_64 + libquadmath)"
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

// ----------------------------------------------------------------------------
// Errors. One type, always thrown, message always names the file.
// ----------------------------------------------------------------------------
struct CorpusError : std::runtime_error {
  explicit CorpusError(const std::string& what) : std::runtime_error(what) {}
};

// ----------------------------------------------------------------------------
// Loaded data.
// ----------------------------------------------------------------------------
struct RealOpData {
  std::string             name;
  uint64_t                n         = 0;
  uint64_t                n_corner  = 0;
  uint64_t                n_cancel  = 0;
  int                     n_operands = 1;
  std::vector<double>     a, b, c;    // b/c empty when the op does not use them
  std::vector<__float128> ref;
};

struct ComplexOpData {
  std::string             name;
  uint64_t                n         = 0;
  uint64_t                n_corner  = 0;
  uint64_t                n_cancel  = 0;
  int                     n_operands = 1;
  std::vector<double>     a_re, a_im, b_re, b_im;   // b_* empty for unary ops
  std::vector<__float128> ref_re, ref_im;
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

    std::vector<__float128> zref(2 * n);
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
