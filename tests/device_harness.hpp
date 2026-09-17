// ============================================================================
// tests/device_harness.hpp — a Kokkos-free CUDA / HIP / serial launch harness
// ============================================================================
// Added by CORE_PLAN section C3.
//
// WHY THIS EXISTS
// Every device measurement this repository has ever made went through Kokkos.
// That was fine while Kokkos was a dependency; it is not fine now that
// include/xp/ is the artifact and Kokkos is one consumer of it. A device test
// that needs Kokkos to launch cannot answer "does the standalone core run on
// this GPU", because a Kokkos build failure and a library defect look the same
// from outside. This header is the smallest thing that launches a kernel
// without Kokkos, so C4-C8 can measure the core on hardware directly.
//
// It is a TEST harness, not part of the library. Nothing in include/xp/
// includes it and nothing ever should.
//
// WHAT IT IS NOT
// It is not a scorer. docs/CORRECTNESS.md allows one measurement and one
// verdict per point; this header produces raw results and the existing host
// scorer judges them. There is deliberately no ulp arithmetic anywhere below.
//
// BACKEND SELECTION, AT COMPILE TIME
//   __HIPCC__  -> HIP        (checked FIRST: hipcc targeting NVIDIA defines
//                             both __HIPCC__ and __CUDACC__, and there the HIP
//                             API is the one that is actually present)
//   __CUDACC__ -> CUDA
//   otherwise  -> a serial host loop
//
// THE THREE RULES, AND WHERE THEY ARE ENFORCED
//   1. Every launch is followed by a fence before from_device(). Enforced by
//      parallel_for_n() itself, which fences before it returns, so the
//      invariant does not depend on a caller remembering. The public fence()
//      remains for callers that want an explicit synchronisation point.
//   2. last_error() is checked after every launch and a nonzero value is a
//      TEST FAILURE, not a warning. Every vendor call below records its code.
//   3. The serial backend is a REAL fallback running the SAME functor. It also
//      keeps two genuinely separate allocations and memcpys between them, so a
//      test that forgets to_device()/from_device() fails on a host build too,
//      rather than passing everywhere a GPU is absent and only failing on
//      hardware nobody runs in CI.
//
// last_error() IS STICKY, AND THAT IS ON PURPOSE. It returns the FIRST nonzero
// vendor code seen since the process started, 0 if there has been none. A
// literal "last" would be destructive: cudaGetLastError()/hipGetLastError()
// CLEAR the error they report, so one later successful call would erase the
// evidence of an earlier failed one and the test would pass on a broken run.
//
// NO __float128 AND NO KOKKOS. The first is enforced by
// scripts/check_device_tu_purity.sh (ctest target device_tu_purity), which
// carries this file in its FILES list. The second is structural: the only
// project header included here is the xp core's config.hpp.
// ============================================================================

#pragma once

// XPMATH_INLINE_FUNCTION, for the functors callers hand to parallel_for_n.
#include <xp/config.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(__HIPCC__)
#define XPT_BACKEND_HIP 1
// Explicit, even though include/xp/config.hpp now includes this too (C3 step 1,
// for the __double_as_longlong intrinsics). The include is guarded and
// idempotent, and this header calls hipMalloc/hipMemcpy by name: a test harness
// should not be silently relying on the library under test to declare the API
// it uses.
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#define XPT_BACKEND_CUDA 1
// No cuda_runtime.h include, for the same reason config.hpp does not have one:
// nvcc force-includes it into every TU, so an explicit include buys nothing and
// adds a failure mode for clang-CUDA, whose header layout differs.
#else
#define XPT_BACKEND_HOST 1
#endif

namespace xpt {

// ---------------------------------------------------------------------------
// Which backend was compiled in
// ---------------------------------------------------------------------------

enum class Where { Host, Cuda, Hip };

constexpr Where where() {
#if defined(XPT_BACKEND_HIP)
  return Where::Hip;
#elif defined(XPT_BACKEND_CUDA)
  return Where::Cuda;
#else
  return Where::Host;
#endif
}

inline const char* where_name() {
  switch (where()) {
    case Where::Hip:  return "hip";
    case Where::Cuda: return "cuda";
    case Where::Host: return "host-serial";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Error state
// ---------------------------------------------------------------------------

namespace detail {

// Not a vendor code. CUDA and HIP error enumerators are non-negative, so a
// negative sentinel cannot collide with one.
constexpr int kAllocFailed = -1;

inline int& error_slot() {
  static int e = 0;
  return e;
}

inline void record(int e) {
  if (e != 0 && error_slot() == 0) error_slot() = e;
}

// The vendor's own "did the last call fail" query, which CLEARS as it reports.
// Called exactly once per launch, immediately, and funnelled into the sticky
// slot above so the clearing does not lose anything.
inline int vendor_last_error() {
#if defined(XPT_BACKEND_HIP)
  return static_cast<int>(hipGetLastError());
#elif defined(XPT_BACKEND_CUDA)
  return static_cast<int>(cudaGetLastError());
#else
  return 0;
#endif
}

}  // namespace detail

inline int last_error() { return detail::error_slot(); }

// ---------------------------------------------------------------------------
// Fence
// ---------------------------------------------------------------------------
// A no-op on the serial backend, which is correct rather than a stub: the loop
// has already run to completion by the time parallel_for_n returns.

inline void fence() {
#if defined(XPT_BACKEND_HIP)
  detail::record(static_cast<int>(hipDeviceSynchronize()));
#elif defined(XPT_BACKEND_CUDA)
  detail::record(static_cast<int>(cudaDeviceSynchronize()));
#endif
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

namespace detail {

inline void* dev_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
#if defined(XPT_BACKEND_HIP)
  void* p = nullptr;
  record(static_cast<int>(hipMalloc(&p, bytes)));
  return p;
#elif defined(XPT_BACKEND_CUDA)
  void* p = nullptr;
  record(static_cast<int>(cudaMalloc(&p, bytes)));
  return p;
#else
  return std::malloc(bytes);
#endif
}

inline void dev_free(void* p) {
  if (p == nullptr) return;
#if defined(XPT_BACKEND_HIP)
  record(static_cast<int>(hipFree(p)));
#elif defined(XPT_BACKEND_CUDA)
  record(static_cast<int>(cudaFree(p)));
#else
  std::free(p);
#endif
}

inline void copy_h2d(void* dst, const void* src, std::size_t bytes) {
  if (bytes == 0 || dst == nullptr || src == nullptr) return;
#if defined(XPT_BACKEND_HIP)
  record(static_cast<int>(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice)));
#elif defined(XPT_BACKEND_CUDA)
  record(static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)));
#else
  std::memcpy(dst, src, bytes);
#endif
}

inline void copy_d2h(void* dst, const void* src, std::size_t bytes) {
  if (bytes == 0 || dst == nullptr || src == nullptr) return;
#if defined(XPT_BACKEND_HIP)
  record(static_cast<int>(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost)));
#elif defined(XPT_BACKEND_CUDA)
  record(static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)));
#else
  std::memcpy(dst, src, bytes);
#endif
}

}  // namespace detail

// A paired host + device allocation of n elements of T.
//
// Non-copyable: it owns device memory, and a shallow copy would double-free it
// on a backend where that is a device-side fault rather than a host crash.
// Not movable either — no test needs it, and the surface C3 specifies does not
// include it.
template <class T>
class buffer {
  static_assert(std::is_trivially_copyable<T>::value,
                "xpt::buffer copies with memcpy/cudaMemcpy, so T must be "
                "trivially copyable; a type with a nontrivial copy would be "
                "silently bit-blitted across the PCIe bus.");

 public:
  explicit buffer(std::size_t n)
      : n_(n),
        host_(n ? static_cast<T*>(std::malloc(n * sizeof(T))) : nullptr),
        dev_(n ? static_cast<T*>(detail::dev_alloc(n * sizeof(T))) : nullptr) {
    // An allocation failure must be visible through last_error(), not through a
    // segfault fifty lines later in the caller's assertion loop.
    if (n != 0 && (host_ == nullptr || dev_ == nullptr)) {
      detail::record(detail::kAllocFailed);
    }
  }

  ~buffer() {
    std::free(host_);
    detail::dev_free(dev_);
  }

  buffer(const buffer&)            = delete;
  buffer& operator=(const buffer&) = delete;

  T*          host() { return host_; }
  T*          device() { return dev_; }
  std::size_t size() const { return n_; }

  void to_device() { detail::copy_h2d(dev_, host_, n_ * sizeof(T)); }
  void from_device() { detail::copy_d2h(host_, dev_, n_ * sizeof(T)); }

 private:
  std::size_t n_;
  T*          host_;
  T*          dev_;
};

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
// F is a functor whose `operator()(std::size_t) const` is annotated
// XPMATH_INLINE_FUNCTION, and which is trivially copyable because it is passed
// BY VALUE into the kernel. A struct holding raw device pointers is the
// intended shape. A __device__ lambda would need nvcc --extended-lambda, which
// this harness deliberately does not require of its callers.

#if !defined(XPT_BACKEND_HOST)
namespace detail {
template <class F>
__global__ void launch_n(std::size_t n, F f) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}
}  // namespace detail
#endif

template <class F>
void parallel_for_n(std::size_t n, F f) {
  if (n == 0) return;
#if defined(XPT_BACKEND_HOST)
  for (std::size_t i = 0; i < n; ++i) f(i);
#else
  const unsigned block = 256u;
  const unsigned grid  = static_cast<unsigned>((n + block - 1) / block);
  detail::launch_n<<<grid, block>>>(n, f);
  // Rule 2: the launch's own error, taken immediately, before anything else can
  // clear it. This is the one that catches "too many resources requested" and
  // "invalid device function" — failures that never reach the kernel body.
  detail::record(detail::vendor_last_error());
  // Rule 1: fence here, so from_device() cannot race a live kernel no matter
  // what the caller does.
  fence();
#endif
}

}  // namespace xpt
