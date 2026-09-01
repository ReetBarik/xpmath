# S5 PHASE NOTES (work-in-progress — delete this file when S5 completes)

This scratch file accumulates notes from each S5 phase. When all five headers are converted, the final S5 phase will consolidate these into the real STATUS block and delete this file.

---

## Phase 1: ff_math.hpp (FF real)

**Converted:** `third_party/include/ff_math.hpp` (1316 lines) → `include/xp/ff_math.hpp` (1288 lines standalone) + `third_party/include/ff_math.hpp` (191 lines compat wrapper).

**Wrapper site count:** 60 explicit `using xp::` declarations (1 type alias + 59 functions: 6 constants, 8 primitives, 4 basic math, 8 exp/log, 9 trig, 7 hyperbolic, 9 multi-arg, 4 rounding, 4 special functions).

**Mechanical translation sites:**
- 163 occurrences of `KOKKOS_INLINE_FUNCTION` → `XPMATH_INLINE_FUNCTION`
- 3 occurrences of `#ifndef __CUDA_ARCH__` → `#if !defined(XPMATH_ON_DEVICE)`
- 1 from_bits guard `#ifndef __CUDA_ARCH__` → `#if defined(XPMATH_ON_DEVICE_CUDA_OR_HIP)` (narrower, same as S2 dd_math pattern for __int_as_float intrinsic)
- Scalar math replacements: `Kokkos::fabs` → `detail::fabs`, `Kokkos::isfinite` → `detail::isfinite`, `Kokkos::isinf` → `detail::isinf`, `Kokkos::sqrt` → `detail::sqrt`, `Kokkos::log` → `detail::log`, `Kokkos::copysign` → `detail::copysign`, `Kokkos::atan2` → `detail::atan2`
- 21 occurrences of `Kokkos::printf` → `XPMATH_PRINTF`
- Namespace: `namespace Kokkos { namespace Experimental {` → `namespace xp {` (one level, matching dd_math pattern)

**Non-mechanical sites:** None. Every conversion was a mechanical token swap. No arithmetic changes, no constant edits, no algorithm modifications.

**Compat wrapper additions:** Added `erfc_asymptotic_sum` and `incgamma` to the explicit using-declarations list (these were internal helpers in the original but are part of the public API because they're used by ff_accuracy_test).

**Smoke test extension:** Created `tests/standalone/ff_no_kokkos_smoke.cpp` (106 lines, mirrors dd_no_kokkos_smoke.cpp for FloatFloat instead of DoubleDouble). Updated `scripts/check_standalone_no_kokkos.sh` to compile and run both DD and FF smoke tests, and to preprocess both headers in stage 3. Script now reports "standalone no-Kokkos compile smoke (DD + FF)" and runs [1/5] through [5/5] stages.

**Gates:**
1. Build clean: ✓ (all targets built without errors)
2. ctest 23/23: ✓ (100% tests passed, 0 failed out of 23, total time 134.95 sec)
3. Byte-identical gate on `kokkos_ep_demo_ff`: ✓
   - BEFORE: 89 lines (captured at HEAD 3375aa1)
   - AFTER: 89 lines
   - FF accuracy columns (fields 6-9 in pipe-delimited output): BYTE-IDENTICAL
4. Byte-identical gate on `kokkos_ep_demo_ff_complex`: ✓
   - BEFORE: 105 lines (captured at HEAD 3375aa1)
   - AFTER: 105 lines
   - FF accuracy columns (fields 6-9 in pipe-delimited output): BYTE-IDENTICAL
5. No-Kokkos smoke extended to FF: ✓
   - DD smoke: PASS (0 failures)
   - FF smoke: PASS (0 failures)
   - Preprocessed output: 50690 lines, 0 Kokkos hits
   - include/xp/*.hpp naming Kokkos only in comments: ok

**Deviations and surprises:** None. The pattern from S2/S3 applied cleanly. The automated sed-based conversion script handled 99% of the work correctly; manual fixes were limited to the 5 scalar math sites that the initial regex missed (sqrt, log, copysign, atan2 in the middle of complex expressions).

**What the next phase must know:** The compat wrapper pattern is now proven on both DD and FF. The explicit `using xp::` declaration list is verbose (60 declarations for FF) but it is the auditable, documented API surface that S4 will defend. The `erfc_asymptotic_sum` and `incgamma` functions are exposed because they're tested — don't silently drop them from the wrapper.
