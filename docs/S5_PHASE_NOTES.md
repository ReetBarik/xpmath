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

### Capture note (added during phase 1 review)

`validation/s5/before_ff_real.txt` carries a trailing `[exited with code 0]` line
from the phase-1 capture shell. It is not demo output. Every accuracy row is
identical between the before/after captures; the only diff is that stray line.

### Gate tooling replaced

`validation/s3/strip_timing.sh` did not work on the FF tables — see the header of
`validation/strip_timing.sh` for the full explanation. Use the new one from here on.

## Phase 2 — qf_math.hpp (QF real)

**Converted:** `third_party/include/qf_math.hpp` → `include/xp/qf_math.hpp`
(1243 lines, standalone, zero Kokkos outside comments) plus a compat wrapper at
the old path with **65 `using xp::` sites**. `include/xp/config.hpp` reused
unchanged — it is now shared by DD, FF and QF.

**Gate results (all green):**

| gate | result |
|---|---|
| build + ctest | 23/23, 0 failed, 134.70 s |
| byte-identical `kokkos_ep_demo_qf` | IDENTICAL |
| byte-identical `kokkos_ep_demo_qf_complex` | IDENTICAL |
| no-Kokkos smoke (DD + FF + QF) | PASS — 0 Kokkos hits in 55 587 preprocessed lines |
| `include/xp/qf_math.hpp` Kokkos refs outside comments | 0 |

**Non-mechanical sites:** none. The auto-generated 4×FP32 `from_bits` constants
were copied verbatim; no expression was reordered.

**Deviation — baseline captured from a pristine worktree.** The phase-2 authoring
session was killed by its wall clock *after* completing the conversion but *before*
its BEFORE captures finished, leaving two 0-byte files. Because the working tree was
by then already converted, the baseline could not simply be re-run. It was instead
captured from a detached `git worktree` at the pre-conversion commit (4fba10d), with
an assertion that `include/xp/qf_math.hpp` was absent from that tree before building.
The comparison is therefore genuinely pre- vs post-conversion.

**Timing note for the remaining phases.** QF is by far the most expensive backend to
gate: the sum of median per-op times is ~219 µs, so at `--batch 1000000 --repeats 5`
each QF demo is ~18 minutes of pure kernel time, and the before+after pair for the two
QF demos took roughly 2.5 hours. Three sessions in a row have now died on demo
captures rather than on conversion work. **For phases 3–5 the baseline captures should
be taken outside the authoring session**, so the session only has to convert and can
finish inside its budget.

---

## Phase 3 — dd_complex.hpp (DD complex)

**Converted:** `third_party/include/dd_complex.hpp` → `include/xp/dd_complex.hpp`
(288 lines, standalone, zero Kokkos outside comments) plus a compat wrapper at
the old path with **20 `using xp::` sites** (1 type alias + 19 functions). 
`include/xp/config.hpp` reused unchanged — it is now shared by DD real, DD complex, FF and QF.

**Wrapper site count:** 20 explicit `using xp::` declarations (1 type alias + 19 functions:
abs, acos, acosh, asin, asinh, atan, atanh, conj, cos, cosh, exp, log, log10, polar, pow, sin, sinh, sqrt, tan, tanh).

**Mechanical translation sites:**
- 41 occurrences of `KOKKOS_INLINE_FUNCTION` → `XPMATH_INLINE_FUNCTION`
- 2 occurrences of `#ifndef __CUDA_ARCH__` → `#if !defined(XPMATH_ON_DEVICE)` (the ostream overload guards)
- 1 occurrence of `Kokkos::printf` → `XPMATH_PRINTF`
- Namespace: `namespace Kokkos { namespace Experimental {` → `namespace xp {` (one level)
- Include: `#include <dd_math.hpp>` → `#include <xp/dd_math.hpp>`

**Non-mechanical sites:** None. Every conversion was a mechanical token swap. No arithmetic changes, no algorithm modifications.

**Kokkos::complex finding (confirmed):** All three complex headers (dd_complex.hpp, ff_complex.hpp, qf_complex.hpp) reference `Kokkos::complex` **only in comments** (lines 23, 39, 320 in dd_complex.hpp; all explain the relationship to Kokkos, not code). There is no actual Kokkos::complex interop to relocate — the complex types are bespoke structs. The ~7 Kokkos::complex sites flagged in the plan were all comment text. This finding transfers to phases 4 and 5.

**Smoke test extension:** Created `tests/standalone/dd_complex_no_kokkos_smoke.cpp` (117 lines, mirrors dd_no_kokkos_smoke.cpp for DoubleDoubleComplex). Updated `scripts/check_standalone_no_kokkos.sh` to compile and run DD + DD complex + FF + QF smoke tests (9 stages total, was 7), and to preprocess all four headers in the Kokkos-free gate. Script now reports "standalone no-Kokkos compile smoke (DD + DD complex + FF + QF)".

**Gates:**
1. Build clean: ✓ (all targets built without errors)
2. ctest 23/23: ✓ (100% tests passed, 0 failed out of 23, total time 134.91 sec)
3. Byte-identical gate on `kokkos_ep_demo_complex`: ✓
   - BEFORE: 105 lines (captured at HEAD ab4f72f)
   - AFTER: 105 lines
   - DD complex accuracy columns (fields 6-9 in pipe-delimited output): BYTE-IDENTICAL
4. Byte-identical gate on `kokkos_ep_demo`: ✓
   - BEFORE: 89 lines (captured at HEAD ab4f72f)
   - AFTER: 89 lines
   - DD real accuracy columns (fields 6-9 in pipe-delimited output): BYTE-IDENTICAL
5. No-Kokkos smoke extended to DD complex: ✓
   - DD real smoke: PASS (0 failures)
   - DD complex smoke: PASS (0 failures)
   - FF smoke: PASS (0 failures)
   - QF smoke: PASS (0 failures)
   - Preprocessed output: 51,978 lines, 0 Kokkos hits
   - include/xp/*.hpp naming Kokkos only in comments: ok

**Baseline captures:** BEFORE baselines provided pre-session at HEAD ab4f72f in validation/s5p3/before_dd_complex.txt and before_dd_real.txt. AFTER captures completed in ~6 minutes total.

**Deviations and surprises:** None. The pattern from phases 1-2 applied cleanly. The dd_complex.hpp header is simpler than the real math headers (no scalar math dispatch sites, no constants, fewer functions), so the conversion was straightforward.
