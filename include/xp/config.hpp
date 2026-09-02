// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// Portability configuration for the standalone extended-precision core.
//
// NAMING RATIONALE (ratified via S2 naming memo + S3):
// xp:: = extended precision, the companion to MxP (mixed precision).
// The X-for-extended convention traces through XBLAS ("Extended and Mixed
// Precision BLAS", Li/Demmel et al., ACM TOMS 28(2) 2002), whose reference
// implementation uses double-double, and whose report is hosted by David H.
// Bailey -- author of the DDFUN this repo ports and the QD the QF backend
// descends from. Short namespace (xp::), long macro prefix (XPMATH_) mirrors
// Kokkos' own Kokkos:: / KOKKOS_ split, and avoids a hazardous two-character
// macro prefix in the global namespace.

#pragma once

// ============================================================
// What this header is for
// ============================================================
// The extended-precision headers (dd/ff/qf) were written against Kokkos and
// used exactly five Kokkos facilities: KOKKOS_INLINE_FUNCTION, a set of
// scalar math wrappers (Kokkos::fabs/sqrt/log/...), Kokkos::printf, the
// CUDA-only `#ifndef __CUDA_ARCH__` idiom, and Kokkos::complex (complex
// headers only, out of scope here). This header reimplements the first four
// with zero Kokkos dependency, so the numeric core is usable from plain
// C++17, CUDA, HIP and SYCL. Kokkos then consumes the same core through a
// thin compat wrapper (third_party/include/dd_math.hpp).
//
// Design rule: this header must never include a Kokkos header, and must
// compile with `g++ -std=c++17` on a machine with no Kokkos installed.

#include <cmath>   // scalar math dispatch below
#include <cstdio>  // XPMATH_PRINTF

// ============================================================
// 1. On-device detection
// ============================================================
// Replaces the CUDA-only `#ifndef __CUDA_ARCH__` idiom used throughout the
// original headers. Each vendor spells "this is the device compilation pass"
// differently; XPMATH_ON_DEVICE unifies the three so a single guard covers
// CUDA, HIP and SYCL instead of silently only covering CUDA.
//
// Rationale for each token:
//   __CUDA_ARCH__          — nvcc defines it only in the device pass.
//   __HIP_DEVICE_COMPILE__ — hipcc's documented equivalent (__HIP_ARCH__ is
//                            NOT equivalent: it is set in both passes).
//   __SYCL_DEVICE_ONLY__   — DPC++/oneAPI device pass of the single-source
//                            compile.
// Defined as a value-less object-like macro tested with `defined()`, so a
// consumer cannot accidentally get `0` treated as "on device".
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || \
    defined(__SYCL_DEVICE_ONLY__)
#define XPMATH_ON_DEVICE
#endif

// A narrower predicate: "the vendor's global-namespace device intrinsics and
// device math library are available". CUDA and HIP both provide the classic
// __longlong_as_double / ::sqrt style global-namespace device entry points;
// SYCL does not, and instead guarantees the std:: math subset in device code.
// Keeping this separate from XPMATH_ON_DEVICE is what lets SYCL take the
// portable path everywhere rather than a CUDA-shaped one.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define XPMATH_ON_DEVICE_CUDA_OR_HIP
#endif

// ============================================================
// 2. XPMATH_INLINE_FUNCTION
// ============================================================
// The direct replacement for KOKKOS_INLINE_FUNCTION (~720 uses across the six
// headers). Every function in the library is a header-inline leaf, so the
// annotation is the only thing that differs per backend.
//
//   __CUDACC__ / __HIPCC__ : `__host__ __device__ inline`. Both macros are
//       defined in BOTH compilation passes (unlike __CUDA_ARCH__), which is
//       exactly what is wanted — the same declaration must be emitted for the
//       host and device pass or the two passes disagree on the symbol.
//   SYCL_LANGUAGE_VERSION  : plain `inline`. SYCL is single-source with no
//       host/device function attributes; SYCL_EXTERNAL exists only for
//       *non*-inline functions compiled in a separate TU, which never applies
//       to a header-only inline library and would break plain-C++ reuse.
//   otherwise              : plain `inline` (host C++, OpenMP, OpenMP-target,
//       and Kokkos's Serial/OpenMP/Threads backends all need nothing more).
//
// A consumer may pre-define XPMATH_INLINE_FUNCTION to force a specific
// spelling (e.g. to inject __forceinline__); the guard below respects it.
#if !defined(XPMATH_INLINE_FUNCTION)
#if defined(__CUDACC__) || defined(__HIPCC__)
#define XPMATH_INLINE_FUNCTION __host__ __device__ inline
#elif defined(SYCL_LANGUAGE_VERSION)
#define XPMATH_INLINE_FUNCTION inline
#else
#define XPMATH_INLINE_FUNCTION inline
#endif
#endif

// ============================================================
// 3. XPMATH_PRINTF — diagnostic policy
// ============================================================
// The numeric headers print a one-line diagnostic on a domain violation
// (~40 sites, e.g. "DDSQRT: negative argument"). Kokkos::printf provided a
// backend-uniform printf; this is the dependency-free equivalent.
//
// Policy:
//   * Host, CUDA and HIP: plain `::printf`. CUDA and HIP both provide a
//     device-side printf in the GLOBAL namespace with C semantics; std::printf
//     is NOT device-callable, hence the explicit `::`.
//   * SYCL: no-op. SYCL 2020 has no portable device printf — the available
//     spellings (sycl::ext::oneapi::experimental::printf, or an ext_oneapi
//     stream) are vendor extensions, and requiring one would make the header
//     depend on a SYCL header. CAVEAT, documented rather than hidden: on a
//     SYCL device the domain diagnostics are silently dropped. The RETURN
//     VALUES of the guarded functions are unaffected — every diagnostic site
//     returns the same value with or without the print — so this is a
//     debuggability loss, not a numerical one.
//   * XPMATH_ENABLE_DIAGNOSTICS: defaults to 1 (today's behaviour, which the
//     byte-identical gate depends on). Define it to 0 to compile every
//     diagnostic out entirely — worth doing in a hot device kernel, where an
//     unreachable printf still costs registers and a format-string constant.
//
// Variadic-macro form (not a function) so that at 0 the arguments are never
// evaluated and the format strings are not emitted into the binary.
#if !defined(XPMATH_ENABLE_DIAGNOSTICS)
#define XPMATH_ENABLE_DIAGNOSTICS 1
#endif

#if XPMATH_ENABLE_DIAGNOSTICS && !defined(__SYCL_DEVICE_ONLY__)
#define XPMATH_PRINTF(...) ::printf(__VA_ARGS__)
#else
#define XPMATH_PRINTF(...) ((void)0)
#endif

namespace xp {
namespace detail {

// ============================================================
// 4. Scalar math dispatch
// ============================================================
// The library calls ordinary float/double math on the COMPONENTS of an
// extended-precision value (never on the extended type itself). Previously
// these went through Kokkos::fabs/exp/floor/sqrt/log/isinf/ceil/copysign/
// atan/isfinite, which exist precisely to be host+device valid. These
// using-declarations are the dependency-free replacement.
//
// Why using-declarations and not forwarding functions: a using-declaration
// makes `detail::sqrt` an ALIAS for the chosen overload set, so there is no
// extra inline frame, no risk of the wrapper's signature narrowing an
// overload, and nothing for a debug build to fail to inline. It is also
// literally zero code.
//
// Why the split:
//   * CUDA/HIP device pass -> the global-namespace `::` overloads. These are
//     the vendor's __device__ implementations. std::sqrt(double) happens to
//     alias ::sqrt in libstdc++, but that is an implementation detail and it
//     does NOT hold for every overload or every standard library.
//   * everywhere else (host, and SYCL device) -> `std::`. SYCL 2020 §4.17.5
//     guarantees the std:: math subset is usable in device code, and on the
//     host std:: is the only spelling the standard guarantees at all.
//
// Why `detail::` qualification at the call sites: namespace xp also
// defines exp/log/sqrt/ceil/floor/copysign/atan for the EXTENDED types. An
// unqualified call from inside xp would drag those into the overload set;
// `detail::` makes the scalar intent explicit and unambiguous.
//
// The set is the union over all six backends (dd/ff/qf, real + complex), so
// some entries are unused by any single header — that is deliberate: this is
// the one place the policy is stated, and S5 must not have to reopen it.
// dd_math.hpp additionally needs atan2 and ldexp; both are included below.
//
// `rint` (KI-2, 2026-09-02) is the round-to-nearest-integer primitive used by
// qf_math.hpp's and tf_math.hpp's `*_nint`. It is one hardware instruction on
// every target this library compiles for — x86 `roundsd`/`roundss`, ARM
// `frintn`, PTX `cvt.rni`, AMD `v_rndne` — and unlike `floor(x + 0.5)` it never
// double-rounds, which is the whole of KI-2. It honours the CURRENT rounding
// mode rather than hard-coding nearest; that is a feature, not a hazard,
// because under a directed mode `floor(x + 0.5)` is simply wrong.
#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)
using ::atan;
using ::atan2;
using ::ceil;
using ::copysign;
using ::exp;
using ::fabs;
using ::floor;
using ::isfinite;
using ::isinf;
using ::ldexp;
using ::log;
using ::rint;
using ::sqrt;
#else
using std::atan;
using std::atan2;
using std::ceil;
using std::copysign;
using std::exp;
using std::fabs;
using std::floor;
using std::isfinite;
using std::isinf;
using std::ldexp;
using std::log;
using std::rint;
using std::sqrt;
#endif

}  // namespace detail
}  // namespace xp
