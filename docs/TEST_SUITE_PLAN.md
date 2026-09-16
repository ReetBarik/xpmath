# Kokkos extended-precision demo — exhaustive test suite plan

**Repo:** `ReetBarik/kokkos-extended-precision-demo`
**Date locked:** 2026-07-31 (session with Minion)
**Owner:** Reet (implementer: cluster-Claude, sequenced from this doc)

## Context and premises

### Repo state as of 2026-07-31

Five branches exist:

- `main` — DD (double-double) + CUDA emulated FP128. No tests. This is where
  everything eventually lands (DD + FF + QF, all benchmarked against
  quadmath).
- `ddfunKokkos` — DD only. No tests.
- `fffunKokkos` — **FF (float-float) already implemented.** Working demos
  for real and complex ops. Ad-hoc validation via `scripts/test_ffmul.cpp`
  (single-op standalone) and `scripts/probe_op.cpp` (debugging tool, not a
  gating test). `PORT_NOTES.md` (220 lines) documents five categories of
  precision bugs found during the DD→FF port and how they were fixed. Not
  merged to main yet.
- `qffunKokkos` — **does not exist yet.** Will hold QF (quad-float,
  4×FP32) as a mechanical port of QD's `qd_real.cc`.
- `CUDAFP128Kokkos` — CUDA emulated FP128 only. Sibling branch,
  **explicitly excluded** from the portable-test-suite plan because it's
  NVIDIA-only and non-portable outside Blackwell (sm_100). Independent
  benchmarking track.

### Precision targets

| Backend | Representation | Bits | Digits | Priority |
|---|---|---|---|---|
| DD  | 2×FP64 | ~106 | ~31.9 | ships on `main` |
| FF  | 2×FP32 | ~48  | ~14.4 | ships on `main` |
| QF  | 4×FP32 | ~96  | ~28.9 | ships on `main` |
| TF  | 3×FP32 | ~72  | ~21.7 | **SKIPPED** — hardest algorithmically, no distinctive niche between FF and QF |

### Oracle: Kokkos-wrapped quadmath (VERIFIED, adopted)

Kokkos ships `core/src/impl/Kokkos_QuadPrecisionMath.hpp` (~110 lines).
Provides `__float128` overloads in `namespace Kokkos` for every math
function needed as oracle: `abs, fma, exp, log, sqrt, pow, sin, cos, tan,
asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, erf,
tgamma, ceil, floor, round, trunc, ldexp, scalbn, copysign,
isinf, isnan, signbit`.

- Gated by `KOKKOS_ENABLE_LIBQUADMATH` CMake option (default ON on Linux
  per `cmake/kokkos_tpls.cmake`).
- Host-only: `inline`, not `KOKKOS_INLINE_FUNCTION`. Perfect for oracle
  role — libquadmath doesn't exist on GPUs, but oracle computation happens
  on host anyway.
- Linked automatically via `kokkos_link_tpl(kokkoscore PUBLIC LIBQUADMATH)`.
- Header is in `impl/` — private convention, may want to include via
  `Kokkos_MathematicalFunctions.hpp` if the overloads leak there when
  `KOKKOS_ENABLE_LIBQUADMATH` is set (verify at implementation time).

**Precision headroom check:**

- Oracle: 34 digits
- DD target: 31.9 digits → 2.1 digit headroom (tight but usable for
  accuracy; MAYBE insufficient for tight-bound verification)
- FF target: 14.4 digits → 20 digit headroom (comfortable)
- QF target: 28.9 digits → 5.1 digit headroom (adequate for pass/fail;
  MPFR ≥150 bits recommended as optional secondary oracle for tight-bound
  work)

MPFR is an optional secondary oracle behind a CMake flag; do not couple
default builds to it.

#### Complex oracle (T0.3)

> **SUPERSEDED 2026-09-16.** The complex demos no longer have an oracle. Their
> accuracy columns were deleted along with `patches/kokkos_complex_quad_math.hpp`,
> `scripts/smoke_kokkos_complex_quad.cpp`, the `check_cxx_source_compiles` probe
> and the `Kokkos_ENABLE_LIBQUADMATH=ON` requirement; the accuracy record is
> `validation/sweep/` (`docs/CORRECTNESS.md`). Nothing below is a live
> instruction — it is kept as the record of why the patch existed. See
> `patches/README.md`.

Kokkos ships **no** `__complex128` wrapper upstream — `Kokkos_QuadPrecisionMath.hpp`
covers only real `__float128`. The complex demo (`src/demo_complex.cpp`) needs a
`__complex128` oracle (`cexpq`, `csqrtq`, `csinq`, …).

To keep the complex oracle plumbing symmetric with the real one (T0.0), this repo
carries a **local extension header**,
`impl/Kokkos_ComplexQuadPrecisionMath.hpp`, that adds `__complex128` overloads in
`namespace Kokkos` (`Kokkos::exp`, `Kokkos::sqrt`, `Kokkos::conj`, `Kokkos::real`,
…). It is applied to Reet's local Kokkos install (dropped into the Kokkos source
tree's `core/src/impl/` before build/install).

- A repo-side copy lives at `patches/kokkos_complex_quad_math.hpp` with a
  `patches/README.md` explaining what it is, that it is **not upstream**, how to
  apply it, and which Kokkos version it was tested against (5.1.0, LIBQUADMATH ON).
- Each overload is a **one-line forward** to the corresponding `::c<fn>q`
  function, so the wrapper is **bit-exact against `::c<fn>q` by construction**.
  Verified by `scripts/smoke_kokkos_complex_quad.cpp` (compares `Kokkos::exp`
  against `::cexpq` bit-for-bit).
- `CMakeLists.txt` probes the Kokkos install for this header
  (`check_cxx_source_compiles`) and warns-and-continues if absent — the same
  graceful-degradation posture used for LIBQUADMATH itself. Only
  `kokkos_ep_demo_complex` depends on it.
- Kept local (not sent upstream) until the test suite stabilizes (Reet's call).

### Test framework

Recommend CTest + a lightweight header (or GoogleTest if you don't mind
the dep). Do NOT continue the `printf + return code` pattern from
`test_ffmul.cpp` beyond that single file — unmaintainable at
6 layers × 3 backends = 18 test binaries. Framework decision is part of
T0.1.

## The six test layers

Every backend gets all six layers. Layers are independent and can be
implemented in parallel within a phase.

1. **EFT unit tests.** Test `twoSum`, Dekker `twoProd` (or `twoProdFMA`)
   at the primitive level. Ground truth: for FF the FP64 sum/product is
   exact and provable; for DD/QF the quadmath sum/product is exact.
   Assert bit-equality in higher precision. 10⁶ random + full corner
   corpus.

2. **Non-overlap invariant checks.** For every op output, assert
   `fl(hi + lo) == hi` bit-exactly (DD/FF), or the length-4 Priest
   invariant `|f_{i+1}| ≤ ½ ulp(f_i)` for QF. Instrument in each op's
   return path under a debug-build flag OR as a wrapper test that
   consumes op outputs. 10⁶ inputs per op.

3. **Property/identity tests.** `a - a == 0`, `a * 1 == a`,
   `sqrt(a)² ≈ a`, `exp(log(a)) ≈ a`, `sin² + cos² ≈ 1`, commutativity
   of `mul` where the algorithm is symmetric. No oracle needed. Fast.

4. **Differential accuracy vs. quadmath oracle.** Per op, measure
   `max(rel_err / u^N)` across 10⁶ random + corpus inputs. Compare
   against published bounds. For DDFUN the bounds don't cite cleanly
   from Joldes-Muller-Popescu (JMP uses `ieee_add`, DDFUN doesn't) — so
   for each op, comment must cite either a published bound or
   `"observed empirically, no proven bound"`. **No silent citation of
   inapplicable bounds.**

5. **FMA-contraction guard.** Compile EFT tests with `--fmad=false` on
   device and `-ffp-contract=off` on host; verify pass. Then with
   contraction on, verify no silent breakage. Precedent:
   `scripts/test_ffmul.cpp` uses `-ffloat-store` for the host case.

6. **End-to-end cancellation-prone kernels with known answers.**
   `√(x²+1) − x` for large x (compare to rearranged
   `1/(√(x²+1) + x)`), Σ 1/k² → π²/6, Machin's formula for π, partial
   sums of alternating series. Assert digit-count vs quadmath oracle.

## Corner-case corpus (used by every layer)

Host-side generator producing arrays of:

- Subnormals (esp. FP32 subnormals for FF/QF)
- ±0, ±inf, NaN (only where op semantics allow)
- Powers of two
- `nextafter` neighbors (u apart, 2u apart, 10u apart)
- Near-cancellation pairs (a, a·(1 + ε) for small ε)
- Huge/tiny magnitude mixes (crossing splitter overflow — see
  PORT_NOTES §4a: FF `exp` broke at input > 79 because
  `b * 8193.0f` overflowed when b ≈ 2¹¹⁵)
- Half-integer boundaries (see PORT_NOTES §4b: FF `ffnint` off-by-one
  because 2⁴⁷ magic-constant trick fails at 24-bit mantissa; affects
  `nint`, `floor`, `ceil`, `round`, `trunc`, `fmod`, sincos arg reduction)

**Explicit regression corpus from PORT_NOTES:**

- `exp` at input > 79 (FF splitter overflow, historical bug 4a)
- `remainder(68.379…, 3.5066…)` giving −1.7533 vs +1.7533 (FF `ffnint`
  off-by-one, historical bug 4b)
- `atanh(a)` for `|a| < 0.5` (cancellation branch, PORT_NOTES §3c)
- `sinh(a)`, `cosh(a)` for `|a| < 0.5` (Taylor branch, §3b)
- `sin(x)`, `cos(x)`, `tan(x)` near multiples of π (joint sin/cos
  doublings needed, §3a)

## PORT_NOTES §5 — conditioning limits (NOT bugs, DO NOT fail-gate on)

Test suite must classify these as "expected min drop", not treat them as
regressions:

- `sub`, `fdim`, `fma` under near-cancellation (unavoidable, matches FP64)
- `asin`, `acos` near `|a| = 1` (derivative → ∞)
- `atanh` near `|a| = 1` (similar)
- `remainder` near multiples of `b` (result → 0 with fixed absolute error)
- `exp` at output denormal range (FF `lo` falls into FP32 subnormal)
- `sin`/`cos` near `±π` (needs triple-float π for arg reduction — out of
  scope)

Suite should report **min AND mean digits** for each op. Fail gate on
mean; separately report min with expected-min annotations.

## Sequencing and task breakdown

Total: 23 tasks, ~3-4 focused weeks. Sequential within phases; parallel
within a phase after the first task lands.

### Phase 0 — Foundation (2 tasks + 1 pre-task)

**T0.-1: Extract FP128 to sibling branch; make `main` portable. (DONE)**

- Executed 2026-07-31. FP128 code confirmed intact on `CUDAFP128Kokkos`
  (headers byte-identical to `main`'s copies before removal).
- Removed `third_party/include/NVIDIA_emulated_quad/` from `main`.
- Stripped all FP128 device-backend paths from `src/demo_real.cpp` and
  `src/demo_complex.cpp` (namespace alias, Views, kernels, accuracy
  columns). Complex quadmath oracle calls (`cexpq`, `csqrtq`, …) left
  intact — they are the reference, not the FP128 backend.
- `main` now builds and runs DD-only on a Serial Kokkos (no CUDA
  required). New Serial BEFORE baseline captured for the T0.0 diff.
- Note: `scripts/build_with_kokkos.sh` still forces
  `Kokkos_ENABLE_CUDA` + `Kokkos_ARCH_BLACKWELL100`; that is the
  remaining CUDA coupling on `main` (not `CMakeLists.txt`), flagged as a
  follow-up.

**T0.0: Migrate quadmath oracle to Kokkos wrapper. (DONE)**

- Executed. Task commit `1188418`. Status was never recorded when the task
  landed; added retroactively in the bucket-1 doc-hygiene pass. Evidence:
  `find_library(QUADMATH_LIBRARY ...)` and the x86_64 gate are gone from
  `CMakeLists.txt`, and `tests/test_utils.hpp` carries the Kokkos-wrapped
  quadmath oracle behind `KOKKOS_EP_HAVE_QUADMATH`.

- **SUPERSEDED (B2). `KOKKOS_EP_HAVE_QUADMATH` AND THE SKIP-77 PLUMBING NO
  LONGER EXIST.** Every DONE block below that says a file is
  `#ifdef KOKKOS_EP_HAVE_QUADMATH`-guarded, runtime-SKIPs 77, or "runs
  unconditionally, unlike the gated tests" describes the suite as it was, and is
  left as written rather than retconned. What changed: no test calls a `*q`
  function any more, so there is nothing for the gate to answer to.
  `__float128` stays as the CARRIER (a compiler type; libgcc supplies its
  arithmetic), and the elementary functions moved to `__builtin_*f128` or, for
  the one genuine reference, to MPFR. The gate had to GO rather than merely
  being redundant: on a Kokkos without the TPL it reported `Skipped`, which
  reads as a pass in a summary line. See the header of `tests/test_utils.hpp`.

- Remove `find_library(QUADMATH_LIBRARY ...)` and the x86_64 gate from
  `CMakeLists.txt`. Replace with detection of
  `Kokkos_ENABLE_LIBQUADMATH` and fail gracefully (skip tests, not
  configure error) if off.
- Include Kokkos's quadmath overloads
  (`impl/Kokkos_QuadPrecisionMath.hpp` or its public umbrella if
  exposed via `Kokkos_MathematicalFunctions.hpp`).
- Convert all `expq`/`logq`/`sinq`/... calls in `src/demo_real.cpp` and
  `src/demo_complex.cpp` to `Kokkos::exp((__float128)…)` style.
- Verify demo accuracy numbers are byte-identical after the refactor
  (regression check).
- Deliverable: `main` builds and runs identically, but oracle plumbing
  is now via Kokkos.
- Do on `main` first, then cherry-pick to `ddfunKokkos`, `fffunKokkos`.

**T0.1: Test harness skeleton (backend-parameterized). (DONE)**

- Executed. Task commit `6435ab1`. Status was never recorded when the task
  landed; added retroactively in the bucket-1 doc-hygiene pass. Evidence:
  `tests/` exists with `tests/CMakeLists.txt` registering 33 targets via the
  `kokkos_ep_add_*` helpers; framework choice is CTest + `test_utils.hpp`.

- Create `tests/` directory, wire into `CMakeLists.txt` with
  `enable_testing()` and `add_test`.
- Pick and adopt test framework (CTest + lightweight header, or
  GoogleTest — document choice).
- Design `test_utils.hpp` with:
  - Backend type traits template (Backend = DD | FF | QF), so a single
    test file can instantiate across backends.
  - Random input generators seeded reproducibly.
  - `rel_err<Backend>(dut_result, quad_reference)` and
    `digits_of_accuracy` computation.
  - Kokkos device-side test runner pattern that copies inputs to device,
    runs op in parallel_for, copies results back to host for oracle
    comparison.
  - Pass/fail reporting with min/mean/max digit stats.
- Deliverable: `tests/hello_test.cpp` exercises harness on a trivial
  identity for DD. Verification: `ctest` runs, passes.
- Port `scripts/test_ffmul.cpp` structure as a reference example
  (rename to `tests/reference_ffmul_pattern.cpp` — the pattern this
  harness generalizes).

**T0.4: Rename DD to `Kokkos::Experimental::DoubleDouble` for upstream-readiness. (DONE)**

- Executed 2026-08-01. Commit `49c87df`.
- Pure refactor, no arithmetic changes. Done BEFORE T0.2 lands corpus code so the
  eventual Kokkos PR is a mechanical namespace move, not a rewrite. See the new
  "Upstreaming considerations" section under "Deliverable at end of Phase 3".
- **Namespace + type.** `namespace quad::ddfun` → `namespace Kokkos::Experimental`;
  `struct ddouble` → `struct DoubleDouble`; `struct ddcomplex` →
  `struct DoubleDoubleComplex` (bespoke struct, NOT `Kokkos::complex<DoubleDouble>`
  yet — that integration is a separate future task).
- **STL-style free functions.** `ddadd→add`, `ddsub→subtract`, `ddmul→multiply`,
  `dddiv→divide` (all still reachable via operators); `ddmuld→multiply_scalar`,
  `dddivd→divide_scalar` (internal helpers); `ddneg→negate`;
  `ddnint→round_to_nearest_int`; `ddang→angle` (internal DDFUN atan2(y,x)
  primitive; public `atan2(y,x)` wrapper unchanged); `powi→pow_int`;
  `ddmuldd→two_prod`. Transcendentals already had STL names (exp/log/sin/…) and
  did not rename. Internal `ddexpint→expint`, `ddincgamma→incgamma`.
- **Constants.** `dd_pi()`→`DoubleDouble_pi()`, and likewise `DoubleDouble_e()`,
  `DoubleDouble_log2()`, `DoubleDouble_log10()`, `DoubleDouble_sqrt2()`,
  `DoubleDouble_euler_gamma()`. Chose the free-function form over a
  `constants::pi<DoubleDouble>()` template: mirrors Kokkos's existing M_PI-style
  accessors, reads shorter at the call site, and these are runtime-built from
  IEEE-754 bit patterns so they cannot be constexpr template variables.
- **Factory.** `make_dd(hi,lo)` → static `DoubleDouble::from_bits(hi,lo)`:
  namespaced to the type, discoverable, no free-function symbol.
- **ADL + `Kokkos::` re-exposure.** Every single/two-output math function is
  ADL-findable (argument namespace is `Kokkos::Experimental`) AND re-exposed under
  `namespace Kokkos` via one-line forwards at the bottom of each header, mirroring
  `impl/Kokkos_QuadPrecisionMath.hpp`'s `__float128` overloads — so
  `Kokkos::exp(dd)` works identically to `Kokkos::exp(double)` /
  `Kokkos::exp(__float128)`. `add`/`subtract`/`multiply`/`divide` are NOT
  re-exposed under `Kokkos` (operators + explicit ADL only).
- **SPDX headers.** Kokkos SPDX + copyright header added to
  `third_party/include/dd_math.hpp`, `dd_complex.hpp`; `patches/` copy already
  had one (T0.3). Downstream files (demos, tests, harness, scripts) deliberately
  left header-free.
- **Callers.** Demos and tests use `namespace dd = Kokkos::Experimental;` alias
  (ergonomic call-site sugar; short `dd::` names retained). `BackendTraits<DD>::type`
  = `Kokkos::Experimental::DoubleDouble`.
- **Regression gate.** BEFORE (`HEAD~1`) vs AFTER accuracy-columns-only diff of
  both demos (`--batch 500000 --repeats 5 --seed 12345`) is **empty** —
  byte-identical behavior (rule 7). `ctest` (hello_test) still passes.

**T0.2: Corner-case corpus. (DONE)**

- Executed 2026-08-01. Commit `5ffa52a`.
- **`tests/corpus.hpp` (new).** Pure DATA + a tiny iteration API, downstream-only
  (NO SPDX header — never upstreamed, unlike `dd_math.hpp`). Precision-parametric
  on the scalar type (`double` for DD, `float` for FF/QF later; only these two).
- **Return shape decision:** materialized `std::vector<T>` (unary) /
  `std::vector<std::pair<T,T>>` (binary), NOT the `InputDist` generator-functor
  shape. Rationale: corpus entries are deterministic constants (a specific
  subnormal, the literal 88.72 that broke FF `exp`), so a vector the caller
  iterates — one element == one deterministic test input — is the natural
  representation; generator functors model random draws, of which there are none
  here.
- **Categories:** subnormals (min/mid/largest denormal), ±0, ±inf, quiet NaN
  (opt-in), powers of two, `nextafter` neighbors (anchors 0/1/π/e/1e6/1e-6, ±2
  ulp), near-cancellation pairs, huge/tiny mixes (ratios 1e6..1e30),
  half-integer boundaries.
- **Two API styles (both shipped):** bundlers `unary<T>(flags)` /
  `binary<T>(flags)` (for T\*.2 invariant sweeps) and named accessors (for T\*.4
  accuracy tests so failures cite a specific PORT_NOTES bug). `CorpusFlags`
  (`include_nan=false`, `include_inf/zero/subnormals=true`) is authoritative over
  the *whole* assembled bundle (incidental members from other categories are
  filtered too).
- **PORT_NOTES §4/§3 regression accessors (verbatim named entries):**
  `exp_overflow` ({79.5,80,85,88.7,88.72}, §4a), `nint_half_integer`
  (19.4999993 + k±0.5 neighbors for k∈{0,1,2,10,100,1000,19}, §4b),
  `remainder_regression` ((68.379, 3.5066) + neighbors, §4b), `atanh_small`
  (§3c), `sinh_cosh_small` (§3b), `trig_near_pi` (±π/±2π/±3π/±π/2 neighbors, §3a).
  Category accessors (`subnormals`/`powers_of_two`/`nextafter_neighbors`/
  `finite_specials`/`zeros`/`infinities`/`nans`/`near_cancellation`/`huge_tiny`)
  are also individually callable.
- **`tests/test_utils.hpp` (edit).** `TODO(T0.2)` block replaced with an
  integration note; `#include "corpus.hpp"` added at file scope (corpus declares
  its own `namespace kokkos_ep::corpus`). Added corpus-pass runners
  `run_unary_op_on_corpus` / `run_binary_op_on_corpus` (same
  host→device→host→oracle pipeline as the random runners, driven by a corpus
  vector); the generator-based runners are unchanged for the random pass. Added
  the PORT_NOTES §5 expected-min-drop registry: `ExpectedMinDropAnnotation` +
  `lookup_expected_min_drop(op_name)`, preloaded with sub, fdim, fma, asin, acos,
  atanh, remainder, exp, sin, cos, tan. **Registry decision:** static constexpr
  table + linear scan — the set is tiny and fixed at compile time, so a table is
  allocation-free and reads as data next to `std::map` / an if/else chain.
- **`tests/corpus_test.cpp` (new).** Corpus-scaffolding smoke test (no DD op run,
  no oracle touched → no LIBQUADMATH guard needed). Verifies bundlers non-empty,
  flag behavior (inf/zero/subnormal present/absent per flag, NaN opt-in), the six
  named regression accessors non-empty with spot-checked values (88.72,
  19.4999993, (68.379,3.5066), |a|<0.5, near +π), and the registry
  (sub/fdim/fma/asin/acos/atanh/remainder non-null; add null).
  `kokkos_ep_add_test(corpus_test)` registered in `tests/CMakeLists.txt`.
- **Gate:** `cmake --build` clean; `ctest -V` shows hello_test AND corpus_test
  passing; `kokkos_ep_demo --batch 100` unchanged (no demo regression).

**T0.3: Add local Kokkos `__complex128` wrapper; route complex oracle
through Kokkos. (DONE)**

- Executed 2026-08-01. Commit `0b11585`.
- **A — header + install.** Wrote
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp` (21 `__complex128` overloads in
  `namespace Kokkos`: `abs, real, imag, conj, exp, log, log10, pow, sqrt, sin,
  cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh`), mirroring
  `Kokkos_QuadPrecisionMath.hpp` (SPDX header, include guards,
  `KOKKOS_ENABLE_LIBQUADMATH` gate, `#error` on missing `__float128`). Each
  overload is a one-line forward to `::c<fn>q`. `arg`/`cargq` omitted (demo does
  not use it). Repo-side copy saved to `patches/kokkos_complex_quad_math.hpp` +
  `patches/README.md`. Dropped into the local Kokkos 5.1.0 source tree and
  reinstalled; confirmed present at
  `<prefix>/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp` (install's
  `FILES_MATCHING PATTERN "*.hpp"` picks it up automatically — no build-system
  change needed). Smoke test `scripts/smoke_kokkos_complex_quad.cpp` compiles,
  runs, and confirms `Kokkos::exp` is bit-exact against `::cexpq`.
- **B — demo migration.** `src/demo_complex.cpp` now includes
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp` and routes all 21 oracle calls
  through `Kokkos::<fn>((__complex128)…)` (`cabsq→Kokkos::abs`,
  `conjq→Kokkos::conj`, `csqrtq→Kokkos::sqrt`, `cexpq→Kokkos::exp`,
  `clogq→Kokkos::log`, `clog10q→Kokkos::log10`, `csinq/ccosq/ctanq→sin/cos/tan`,
  `casinq/cacosq/catanq→asin/acos/atan`, `csinhq/ccoshq/ctanhq→sinh/cosh/tanh`,
  `casinhq/cacoshq/catanhq→asinh/acosh/atanh`, `cpowq→Kokkos::pow`,
  `crealq/cimagq→Kokkos::real/imag`). No DD-side arithmetic changed.
  `CMakeLists.txt` probes the install for the wrapper via
  `check_cxx_source_compiles` (linking `Kokkos::kokkos` so libquadmath resolves)
  and warns-and-continues if absent, pointing at `patches/README.md`. Only
  `kokkos_ep_demo_complex` is affected; `kokkos_ep_demo` (real) builds unchanged.
- **Regression gate.** BEFORE (`HEAD~1`) vs AFTER accuracy-columns-only diff of
  `kokkos_ep_demo_complex --batch 500000 --repeats 5 --seed 12345` is **empty**
  — behavior byte-identical, as required (rule 7).
- Bit-exactness is by construction (one-line forwards), so the refactor cannot
  change oracle values.

**T0.5: License hygiene — correct DDFUN attribution; add repo LICENSE /
NOTICE for dual-license posture. (DONE)**

- Executed 2026-08-01. Commit `e1b4de0`.
- **Problem fixed.** T0.4 had mechanically stamped Kokkos's
  `Apache-2.0 WITH LLVM-exception` SPDX onto `third_party/include/dd_math.hpp`
  and `dd_complex.hpp` to match Kokkos style. Those two files are C++/Kokkos
  ports of DDFUN v04 and are governed by David H. Bailey's **DHB-License**
  (modified BSD-3-Clause + §3 grant-back), not Kokkos's license. A Kokkos SPDX on
  a DDFUN derivative does not relicense DDFUN; it makes the file's status
  ambiguous. T0.5 corrects this and establishes the repo's overall dual-license
  posture.
- **Header rewrite.** Removed the two T0.4 SPDX lines from `dd_math.hpp` and
  `dd_complex.hpp`; replaced with a `LicenseRef-DHB-License` block (David H.
  Bailey copyright 2024 + "Modifications Copyright (c) 2026 UChicago Argonne,
  LLC"), DDFUN v04 provenance, and a modifications list. Additive only — no code
  inside the headers changed.
- **License files added.** Top-level `LICENSE` (Apache-2.0, verbatim from
  apache.org); `LICENSES/Apache-2.0.txt` (identical duplicate for SPDX tooling);
  `LICENSES/LicenseRef-DHB-License.txt` (verbatim byte-for-byte from
  davidhbailey.com, CRLF preserved, 24 lines,
  SHA-256 `8e167fe5…5fa5200`); `NOTICE.md` (plain-English per-file mapping, §3
  grant-back explanation, contacts).
- **`patches/kokkos_complex_quad_math.hpp`** verified correct
  (`Apache-2.0 WITH LLVM-exception`, Kokkos-style extension, not a DDFUN
  derivative) and left unchanged. `patches/README.md` gained a "License" section
  explaining why it matches Kokkos's license.
- **This doc.** Added the "Licensing" section above (dual-license posture,
  rationale, §3 grant-back, file-to-license mapping, Phase 2/3 FF/QF lineage
  checkboxes, Argonne review note); corrected the "Upstreaming considerations"
  SPDX bullet, which had assumed the DDFUN files were Apache-2.0.
- **Scope-out.** No code changed; no outreach to David H. Bailey or Argonne
  tech-transfer (Reet's follow-ups); no touching of FF/QF headers on sibling
  branches (their license lineage is a Phase 2/3 kickoff item).
- **Gate.** `grep -rn "SPDX-License-Identifier"` returns exactly the expected
  set (two DHB-License headers, one Apache-2.0-WITH-LLVM-exception patch, plus
  the license-text files themselves). Build unchanged; `ctest -V` shows
  hello_test AND corpus_test passing; demo smoke test unchanged.

### Phase 1 — DD validation (6 tasks)

**T1.1: EFT unit tests for DD. (DONE)**

- Executed 2026-08-01. Commit `30d8a40`.
- **`tests/dd_eft_test.cpp` (new, ~470 lines).** Layer-1 EFT unit test for the
  two error-free transforms every DD op is built on: the twoSum inside
  `DoubleDouble` `add` (dd_math.hpp:178-185) and the Dekker twoProduct inside
  `multiply` (dd_math.hpp:197-211, ≡ `two_prod` dd_math.hpp:270-278).
- **EFT primitive extraction — mirror-and-comment.** `add`/`multiply` embed the
  transforms in longer sequences that also fold in the input `.lo` components; for
  EFT testing we want the transform of two RAW doubles. So `two_sum` /
  `two_prod_dekker` are duplicated into the test file (bit-identical to the
  embedded transforms when `.lo == 0`), each with a comment citing the source
  lines. `dd_math.hpp` is NOT modified (rule 4). The duplication doubles as
  documentation of exactly what is under test.
- **Ground truth `__float128` — provable, not approximate.** Exact FP64 sum ≤54
  bits, exact FP64 product ≤106 bits, both ≤ binary128's 113-bit mantissa, so
  widening operands and computing sum/product in `__float128` is exact. Assert
  `(float128)hi + (float128)lo == (float128)a {+,*} (float128)b` bit-exactly.
  Oracle via Kokkos LIBQUADMATH; runtime-SKIP (77) if absent (hello_test pattern).
- **Coverage.** Test A (twoSum) and Test B (Dekker twoProd) each run 4 corpora:
  10⁶ uniform in [-1e100,1e100], 10⁶ uniform in [-1,1], 10⁵ `|a|≫|b|` pairs
  (`b=a·2^-k`, k∈[1,60]), and the full `corpus::unary<double>()` cross-product
  (i<j). Result: **A tested 2,113,526 / 0 failures; B tested 2,110,183 / 0
  failures.** Test C: 14/14 named hard cases (exact cancellation, both-subnormal,
  ±0, Bailey's `1.0 + 2^-53` → `lo` exactly 2^-53, π·π/e·e/√2·√2 spot-checks).
  Test D: device parity via `parallel_for` (Serial here → reduces to host, as the
  spec allows; catches device FP differences on CUDA/HIP/SYCL) — 200,000/200,000.
- **Out-of-domain inputs are SKIPPED, not failed.** twoSum skips only non-finite
  pairs and overflowing sums. Dekker twoProduct additionally skips subnormal
  operands, splitter-overflow magnitudes (`|x| ≥ 2^996`, checked BEFORE the
  zero-product shortcut because `x·0` still splits `x`), and products that overflow
  or gradually underflow (error term would fall subnormal). These are documented
  limits of Dekker's method (Dekker 1971; Muller et al. HFPA §4.4), not defects in
  `multiply`. The domain predicate tests the EXACT (float128) product, not the
  rounded FP64 product (which flushes to 0 under gradual underflow). No unguarded
  splitter-overflow bug was found in `ddmul` itself.
- **FMA-contraction posture.** `tests/CMakeLists.txt` gains
  `kokkos_ep_add_eft_test(<name>)` = `kokkos_ep_add_test` + per-target
  `-ffp-contract=off` (GNU/Clang, `COMPILE_LANGUAGE:CXX`), `-fp-model=precise`
  (Intel), `--fmad=false` (nvcc, `COMPILE_LANGUAGE:CUDA`, gated on
  `Kokkos_ENABLE_CUDA`). Per-target, not global, so demos/other layers keep normal
  flags. Verified `-ffp-contract=off` present in the verbose compile line. Reused
  by FF (T2.1) and QF (T3.1).
- **Gate.** `cmake --build` clean, zero warnings from `dd_eft_test.cpp`;
  `ctest -V` shows hello_test, corpus_test, dd_eft_test all Passed;
  `kokkos_ep_demo --batch 100 --repeats 1` unchanged (no regression).
- **Scope-out.** Only the twoSum-in-`add` and Dekker-twoProd-in-`multiply` EFTs
  (Layer 1). Higher-level ops → T1.2/T1.3/T1.4. No FF/QF EFTs (T2.1/T3.1). No MPFR
  (quadmath is provably exact for DD EFTs). No T1.5 contraction matrix. Demos
  untouched.

**T1.2: Non-overlap invariant checks for DD. (DONE)**

- Executed 2026-08-01. Commit `0786b65`.
- **`tests/dd_invariant_test.cpp` (new, ~700 lines).** Layer-2 output-invariant
  test. For every DD op that returns a double-double it asserts the NON-OVERLAP
  invariant `fl(hi + lo) == hi` bit-exactly, evaluated in **raw FP64** (a single
  hardware add + compare, `|lo| <= ½ ulp(hi)`). Op inventory (50 checked rows):
  unary — abs, negate, sqrt, round_to_nearest_int, ceil, floor, round, trunc, exp,
  exp2, exp10, expm1, log, log2, log10, log1p, sin, cos, tan, asin, acos, atan,
  sinh, cosh, tanh, asinh, acosh, atanh, erf, erfc, tgamma; binary — add, subtract,
  multiply, divide, pow, atan2, hypot, fmod, remainder, copysign, fmax, fmin, fdim;
  ternary — fma; two-output — sincos (cos, sin) and sinhcosh (cosh, sinh), each
  output checked separately; integer-scalar — pow_int(dd,int). `dd_math.hpp` NOT
  modified (rule 4).
- **Oracle-independent by design — no `__float128`, no `KOKKOS_EP_HAVE_QUADMATH`
  guard.** The invariant is a statement ABOUT FP64 rounding ("adding lo back into
  hi does not move hi"). A `__float128` promotion would test the exact real sum — a
  DIFFERENT property that would flag a perfectly-normalized pair with nonzero lo as
  "unequal". So the check is deliberately `(d.hi + d.lo) == d.hi` in `double`, and
  the TU builds and RUNS even on a quadmath-less Kokkos (unlike the T1.1/T1.5 EFT
  tests, which need the exact-product oracle). This is the ONE Phase-1 test with no
  quadmath dependency. Accuracy-vs-oracle is the separate concern of T1.4.
- **Two passes per op.** (a) 10⁶ op-appropriate random inputs; (b) a full corpus
  pass via `corpus::unary<double>()` / `corpus::binary<double>()` (bundler style,
  `include_zero=true`, `include_inf=false`, `include_nan=false`; subnormal INPUTS
  left on — a subnormal RESULT is filtered by `result_checkable`, not here). Total
  ~50.5M invariant checks, 1367 skipped, **0 failures**.
- **Skip-not-fail domain gating.** A result is SKIPPED (counted, not failed) when
  `hi` is NaN / ±inf / subnormal, or the input is outside the op's mathematical
  domain (log of ≤0, asin of |x|>1, …). Each op carries an input-domain predicate
  that gates BOTH the random generator's rare escapes and every corpus value BEFORE
  the call — which additionally suppresses `dd_math.hpp`'s internal domain-guard
  diagnostics (`DDEXP`/`DDCSSNR`/… `Kokkos::printf`s) on saturation inputs the op
  clamps. The final run emits **zero** guard-diagnostic lines. Notable predicate
  bounds, each mathematically motivated (see file comments): log-family restricted
  to normal `x ∈ [1e-100, 1e100]` (keeps |log x| < ~230 so the internal refining
  exp stays inside its Taylor iteration budget); `pow` base to the same window (its
  internal `log(a)` hits the same limit for tiny/subnormal a); `atan2` gated on the
  larger operand magnitude `m` (m=0 → atan2(0,0), else `1e-150 ≤ m ≤ 1e150`) so
  `r=√(a²+b²)` neither overflows nor UNDERFLOWS to 0 — the latter (both operands
  subnormal/smallest-normal) is what drove `angle`'s internal sincos to its
  iteration limit, confirmed by a standalone corpus probe.
- **Per-op reporting + failure forensics.** Each op prints
  `op: tested=N skipped/failures` and a summary table with an `OK`/`FAIL` column;
  the first ≤3 failures per op dump input+output bit patterns in the
  `probe_op.cpp` hex format (`0x%016llx`). Two `KOKKOS_EP_ASSERT`s gate total
  failures == 0.
- **Device pass.** 5 ops (add, multiply, sqrt, exp, sin) additionally run on device
  at 10⁵ random inputs each (host default = Serial on this build; parallel_for +
  KOKKOS_LAMBDA), all `OK`.
- **Test C — explicit PORT_NOTES §4 regressions.** exp at 79.5/80/85/88.7/88.72
  (invariant holds AND result not NaN); round_to_nearest_int(19.4999993) and
  half-integer boundaries k+0.5 (invariant); remainder(68.379, 3.5066) (invariant
  AND sign). **Deviation, reported with evidence:** the original task text expected
  a *positive* remainder here, but at FP64 `a/b = 19.50008… > 19.5`, so nint=20 and
  the correct remainder is **negative** (-1.7529999999999983). Verified by TWO
  independent FP64/quadmath oracles (libm `remainder()` and `remainderq()`); DD
  agrees with both. The "positive" expectation was FP32-specific (there
  `a/b ≈ 19.4999993 < 19.5`, per PORT_NOTES §4b's `ffnint` context — the literals
  round to opposite sides of 19.5 at FP32 vs FP64). The invariant held throughout,
  so this was a bug in the *test's expectation*, not in `dd_math` → report-and-stop
  did not apply; Test C now gates DD's sign against `std::remainder(a,b)`.
- **`tests/CMakeLists.txt` (edit).** Registered with the plain
  `kokkos_ep_add_test(dd_invariant_test)` helper — NO contraction flags (the
  invariant holds regardless of FMA contraction; this is not an EFT test).
- **`tests/README.md` (edit).** Added the `dd_invariant_test` registry row and a
  "Non-overlap invariant (Layer 2)" section mirroring the Layer-1/Layer-5 sections.
- **Gate.** `cmake --build` clean, zero warnings; `ctest -V` shows all SIX tests
  (hello_test, corpus_test, dd_invariant_test, dd_eft_test, dd_fma_guard_test,
  dd_fma_guard_test_contract_on) Passed; per-op summary shows 0 failures and 0
  guard-diagnostic lines; `kokkos_ep_demo --batch 100 --repeats 1 --seed 12345`
  runs cleanly (DD table, RC 0).
- **Scope-out.** DD real ops only — no complex ops, no accuracy-vs-oracle (T1.4),
  no device parity beyond the 5 named ops. `dd_math.hpp` untouched; demos untouched.
  No invariant violations found (nothing to report-and-stop on).
- Independent of T1.1.

**T1.3: Property/identity tests for DD. (DONE)**

- Executed 2026-08-01. Commit `42b802b`.
- **`tests/dd_property_test.cpp` (new, ~560 lines).** Layer-3 algebraic-identity
  test: do the DD ops compose the way the algebra says they should? Identities are
  split by whether verification needs the `__float128` oracle at all.
  `dd_math.hpp` NOT modified (rule 4).
- **Group A — bit-exact, no oracle (10⁶ random + full finite corpus).** Two sides
  must produce the IDENTICAL `(hi,lo)` bit pattern → raw `==`, no tolerance, no
  quadmath. A1 `add(a,negate(a))==0`, A2 `a-a==0` (operator-), A3 `a·1==a`,
  A4 `a·(-1)==negate(a)`, A5 `abs` sign branches, A6 `negate(negate(a))==a`,
  A8 add commutativity `add(a,b)==add(b,a)`. **7 identities, 0 failures.**
  Failures (none) would dump input+output bit patterns in `probe_op.cpp` hex.
- **Classification by inspection + empirics — one demotion.** A7 (multiply
  commutativity) was DEMOTED from Group A to Group B (as B0): Dekker twoProduct's
  partial-sum chain `(((a1·b1−c11)+a1·b2)+a2·b1)+a2·b2` reorders under operand
  swap and FP add is non-associative, so `multiply(a,b)` and `multiply(b,a)` agree
  to ~31 digits but are NOT guaranteed bit-identical (per the task's "demote rather
  than force-pass" rule). A8 (add commutativity) STAYS in Group A: Knuth twoSum's
  error term is exact and order-independent, so add commutativity IS bit-exact
  (confirmed: 0 failures over 10⁶). B0_mul_comm runs at min=mean=31.00 — near-exact
  but not asserted bit-exact.
- **Dekker domain restriction (skip-not-fail).** A3/A4 (which call `multiply`)
  gate on `|x| < 2^996` via a `dom_dekker` predicate: the first run FAILED on
  corpus values ≥6.7e299 (DBL_MAX etc.) where the Veltkamp split `a.hi·(2^27+1)`
  overflows to inf→nan. This is the documented Dekker splitter-overflow limit from
  T1.1, NOT a bug in `multiply` → per "out-of-domain SKIPPED not failed" each skips
  6 corpus values (`skipped=6`), 0 failures. Group A add/negate/abs identities use
  the full finite corpus (`dom_all`).
- **Group B — tolerance vs `__float128` oracle (`#ifdef KOKKOS_EP_HAVE_QUADMATH`;
  runtime-SKIP 77 otherwise).** 13 identities scored with `digits_of_accuracy`,
  **fail-gated on the MEAN** against `tolerance_digits = -log10(N·u²)` (u=2⁻⁵³;
  ≈25.91 at N=10⁶, ≈26.91 at N=10⁵). Per-identity proven bounds (2u²/4u²/10u²)
  cited in code comments (rule 5). B0 mul-comm, B1 sqrt²≈a, B2 exp(log a)≈a,
  B3 log(exp a)≈a, B4 sin²+cos²≈1, B5 sin(−a)==−sin a, B6 cos(−a)==cos a,
  B7 tan·cos≈sin, B8 2·sin·cos≈sin(2a), B9 exp(a)·exp(−a)≈1, B10 hypot²≈a²+b²,
  B11 pow(a,2)≈a·a, B12 atanh(a)≈½(log(1+a)−log(1−a)). **All 13 pass**; lowest
  mins B8=20.45 and B12=23.94 (still ≫ tolerance), all means ≥29.77. Gating on the
  mean (not min) keeps conditioning-limited samples (B4 near ±π·k, PORT_NOTES §5)
  from false-failing. B5/B6 are empirically bit-exact (min 31.00) but conservatively
  kept in Group B.
- **Deviations (justified).** B3/B9 domain narrowed from the task's `[-700,700]` to
  `[-290,290]` because `dd_math.hpp`'s `exp` clamps at `a.hi≥300` (an intermediate
  `exp(-a)`/`exp(a)` past ~±300 would saturate) — the identity is still exercised
  across 580 orders of magnitude in the exponent. B12 phrased as the equivalence
  `atanh(a)−½(log(1+a)−log(1−a))≈0` mapped through `digits_of_accuracy` (both sides
  finite, `|a|<0.5`).
- **Test C — named-constant regressions (target ≥30 digits).** C1 `|sin(π)|≤ε`
  (softened via `lookup_expected_min_drop("sin")`; `|sin(π)|=1.08e-31`, zero_digits
  30.97), C2 `log(e)≈1` (30.46), C3 `exp(log2)≈2` (31.00), C4 `√2·√2≈2` (31.00),
  C5 `log(10)≈log10` constant (30.18). C6 (euler_gamma/digamma) SKIPPED — no digamma
  op and no independent DD oracle for the constant. **5/5 pass.**
- **Device pass.** 3 Group A (A1, A3, A5) bit-exact + 2 Group B (B1, B4) digit
  checks rerun on device at 10⁵ inputs (Serial here; parallel_for + KOKKOS_LAMBDA
  ships hi/lo back). All PASS (device B1/B4 min 30.86).
- **Anti-tests (documented in-source, deliberately NOT tested).** Associativity of
  `add` and distributivity across large-magnitude cancellations are FALSE for any
  finite-precision format (rounding is grouping-dependent) — asserting them would
  test IEEE rounding, not the DD port.
- **`tests/CMakeLists.txt` (edit).** Registered with the plain
  `kokkos_ep_add_test(dd_property_test)` helper — NO contraction flags (not an EFT
  test; identities hold regardless of FMA contraction).
- **`tests/README.md` (edit).** Added the `dd_property_test` registry row and a
  "Property/identity tests (Layer 3)" section (Group A/B/Test C, the A7→B0
  demotion, anti-tests).
- **Gate.** `cmake --build` clean, zero warnings from `dd_property_test.cpp`;
  `ctest -V` shows all SEVEN tests (hello_test, corpus_test, dd_invariant_test,
  dd_property_test, dd_eft_test, dd_fma_guard_test, dd_fma_guard_test_contract_on)
  Passed; `kokkos_ep_demo --batch 100 --repeats 1 --seed 12345` runs cleanly.
- **Scope-out.** DD real ops only — no complex identities (`dd_complex.hpp`); no
  associativity; no ≥3-op compositions beyond those listed; device parity limited
  to 3 Group A + 2 Group B; quadmath NOT used as the Group A oracle. `dd_math.hpp`
  untouched; demos untouched. No bugs found (nothing to report-and-stop on; the
  A3/A4 corpus failures were the known Dekker domain limit, handled by skipping).
- Independent of T1.2/T1.5/T1.6.

**T1.4: Differential accuracy for DD vs quadmath. (DONE, RED)**

`(DONE, RED)` is deliberate: the task shipped its deliverable (the test) and the
test is doing its job — it flags three REAL `dd_math.hpp` accuracy defects and
fails on them. The red is the point; it is the durable regression gate for the
follow-up bug tasks B1/B2/B3 below.

- Executed 2026-08-01. Commit `f6dbbd2`.
- **`tests/dd_accuracy_test.cpp` (new, ~760 lines).** Layer-4 per-op differential
  accuracy vs the `__float128` oracle: 10⁶ random + corpus per op, each element
  scored in digits = −log₁₀(rel_err) capped at 31 (u² = 2⁻¹⁰⁶), mean-gated at
  ~25.91 digits, with EXPECTED-MIN-DROP semantics for PORT_NOTES §5
  conditioning-limited ops. Whole file `KOKKOS_EP_HAVE_QUADMATH`-guarded; SKIP
  (77) without quadmath. Registered with the plain `kokkos_ep_add_test` helper
  (not an EFT test). **`tests/CMakeLists.txt` / `tests/README.md` (edit).**
- **Op inventory — same ~50-row set as T1.2, verbatim** (see the T1.2 DONE block
  for the full enumeration; not duplicated here). By category:
  - *unary* — abs, negate, sqrt, round_to_nearest_int, ceil, floor, round, trunc,
    exp, exp2, exp10, expm1, log, log2, log10, log1p, sin, cos, tan, asin, acos,
    atan, sinh, cosh, tanh, asinh, acosh, atanh, erf, erfc, tgamma
  - *two-output* — sincos.cos, sincos.sin, sinhcosh.cosh, sinhcosh.sinh
  - *binary* — add, subtract, multiply, divide, pow, atan2, hypot, fmod,
    remainder, copysign, fmax, fmin, fdim
  - *ternary* — fma
  - *integer-scalar* — pow_int
- **Two-pass shape.** Random pass uses the T1.2 domain predicates/ranges
  verbatim; corpus pass uses the PORT_NOTES §3/§4 named accessors where they
  exist (`exp_overflow`, `nint_half_integer`, `remainder_regression`,
  `atanh_small`, `sinh_cosh_small`, `trig_near_pi`) with `unary<T>()` /
  `binary<T>()` bundler fallback otherwise. ~50M random inputs scored (10⁶ × 50
  ops) plus ~878 corpus/skip-filtered inputs across all ops (per-op corpus counts:
  ~165 unary bundler / ~67 binary bundler / named-accessor sizes for the rest).
- **Tolerance rationale.** `tolerance_digits = −log₁₀(N · u²)` with u² = 2⁻¹⁰⁶ and
  N = 10⁶ → **25.91**. Same formula as T1.3 Group B. Single uniform tolerance +
  the PORT_NOTES §5 registry only — **no per-op tolerance overrides** (that would
  defeat the point of a differential-accuracy gate).
- **Results — 47 PASS, 3 FAIL (real signals, not test artifacts).**
  - The 47 passing ops land at **mean 29.4–31.0 digits**. EXPECTED-MIN-DROP
    entries (mean cleared; sanctioned low min surfaced with its registry reason):
    exp (output-denormal lo), sin/cos/tan (near ±π needs triple-float reduction),
    asin/acos (derivative → ∞ near |a|=1), atanh (1/(1−a²) blows up near |a|=1).
  - **tgamma — mean=14.56 / tol=25.91 FAIL.** Lanczos g=7 with FP64 coefficient
    constants (`dd_math.hpp:729-738`: 676.5203681218851, …) caps the result at
    ~15 digits regardless of the enclosing DD arithmetic. Logged as **B1** below.
  - **erfc — mean=19.50 / tol=25.91 FAIL.** `erfc(z)=subtract(1, erf(z))`
    (`dd_math.hpp:718-720`) is catastrophic cancellation as erf(z) → 1 (26.4
    digits at x=3, 18.2 at x=5, 0 at x=8). Logged as **B2** below.
  - **erf — mean=24.64 / tol=25.91 FAIL.** 30–31 digits for |z| ≤ 5, but the
    large-|z| asymptotic branch (`dd_math.hpp:698-715`) collapses to ~5 digits at
    x=8, dragging the uniform(−10,10) random-pass mean below tol. Logged as **B3**
    below.
  - The three failures were verified **library-side, not oracle-side**, by an
    independent standalone probe (separate TU calling quadmath `erfq`/`erfcq`/
    `tgammaq` directly) that reproduced the same digit counts; the same oracle
    machinery scores 47 other ops at 29.4–31.0, so oracle and domains are sound.
- **Rule 4 posture.** `dd_math.hpp` NOT modified. The three failures are real
  defects and stay flagged RED in the shipped test until B1/B2/B3 land; the test
  is the durable regression gate for those fixes. The failing ops are NOT
  skipped/disabled/xfailed.
- **Scope-out.** No `dd_complex.hpp`; no FF/QF; no MPFR; no new corpus categories;
  no per-op tolerance overrides. Op inventory not re-derived (taken from T1.2).
- Depends on T0.1, T0.2.

**T1.5: FMA-contraction guard for DD. (DONE)**

- Executed 2026-08-01. Commit `863f997`.
- **`tests/dd_fma_guard_test.cpp` (new, ~340 lines).** Layer-5 positive test of
  the FMA-contraction posture T1.1 adopts defensively. Builds the identical Dekker
  `twoProduct` (mirrored from dd_math.hpp:197-211 / `two_prod` 270-278, copied
  verbatim from `dd_eft_test.cpp` per rule "tests are standalone; duplication is
  acceptable") under BOTH contraction settings and cross-checks against a
  contraction-immune `__float128` oracle. `twoSum` is included as a labeled CONTROL
  (no mul-then-± adjacency → contraction-immune → must stay exact both ways).
  `dd_math.hpp` NOT modified (rule 4).
- **Source-layout choice: single-source, two targets** (not two-sources-with-
  shared-header). The whole point is to run the IDENTICAL body over IDENTICAL
  inputs under different flags; compiling the same bytes twice makes "identical" a
  build-system guarantee, not a claim a reviewer must verify across two files that
  can drift. Per-variant behavior is selected by compile definitions the CMake
  helpers set: `KOKKOS_EP_CONTRACTION_MODE` (0 = OFF/gate, 1 = ON/report) and
  `KOKKOS_EP_BASELINE_PATH` (ON only).
- **Contraction-immune oracle.** Reference `(p_ref, e_ref)` built from the exact
  `__float128` product (`p_ref = (double)(f128)a*(f128)b`; `e_ref` = exact residual,
  fits in a double since the 106-bit FP64 product fits binary128's 113-bit
  mantissa). Computed via a single f128 multiply — no mul-then-add adjacency for a
  compiler to contract, so the ground truth cannot itself be corrupted.
- **`tests/CMakeLists.txt` (edit).** Added companion helper
  `kokkos_ep_add_eft_test_contract_on(name)` mirroring `kokkos_ep_add_eft_test`
  but forcing contraction ON into a distinct `<name>_contract_on` target:
  `-ffp-contract=fast` (GNU/Clang — the spelling both honor identically; GCC's
  `=on` is accepted too but `=fast` is chosen for clang parity), `-fp-model=fast`
  (Intel), `--fmad=true` (nvcc default, stated explicitly). Also threads
  `KOKKOS_EP_CONTRACTION_MODE=0` into `kokkos_ep_add_eft_test` (harmless for
  `dd_eft_test`, which ignores it). Registered BOTH variants:
  `kokkos_ep_add_eft_test(dd_fma_guard_test)` (OFF) and
  `kokkos_ep_add_eft_test_contract_on(dd_fma_guard_test)` (ON, suffixed target, no
  name clash).
- **Regression posture / baseline: implemented, not deferred.** OFF variant
  fail-gates (`KOKKOS_EP_ASSERT F == 0` — stronger than T1.1). ON variant is a
  reporter: always exits 0, prints `tested=N exact=M mismatches=F`, and compares
  `F` to `tests/dd_fma_guard_baseline.txt` (committed with the observed count),
  printing `baseline: OK` or `*** DRIFT ***`. Drift is WARN-only (a compiler/ISA
  upgrade changing contraction behavior is a signal to investigate, not a CI
  failure). Chose to implement (~30 lines) rather than defer: it is the only thing
  that turns the ON variant from a one-shot print into a cross-upgrade sentinel,
  which is the stated goal; missing/unparseable baseline degrades gracefully to
  "print + hint", still exit 0.
- **Observed contraction-ON mismatch count on this compiler: `F = 0`**
  (GCC 13.3.0, `-O3 -ffp-contract=fast`, baseline x86-64 Serial, 220,366 in-domain
  checks: 110,183 host + 110,183 device). **Nuance worth recording:** `F = 0` here
  is NOT merely "GCC declines to contract." Under `-mfma` GCC *does* emit 8 FMA
  instructions for the Dekker sequence (verified in `-S` output: 5× `vfmsub*sd` +
  3× `vfmadd*sd`), yet `F` stays 0 — Veltkamp splitting (split = 2^27+1) makes each
  partial product `a1*b1`/`a1*b2`/`a2*b1`/`a2*b2` ≤52 bits and thus exactly
  representable, so fusing `partial ± accumulator` changes no rounding. At the
  project's actual build flags (no `-mfma`/`-march`) GCC emits plain mul+sub and
  contracts nothing. Either way the shipped Dekker error term is bit-exact on this
  toolchain; `-ffp-contract=off` is belt+suspenders here, and T1.5 will catch any
  future toolchain where that stops being true.
- **Sensitivity verified.** A deliberately-broken twoProduct (dropped `a2*b2`
  term) compiled under the ON posture flags 209,310/220,366 mismatches and still
  exits 0 — confirming the guard actually detects a wrong error term (it is not a
  vacuous pass) and that the ON variant reports rather than gates.
- **Gate.** `cmake --build` clean, zero warnings; `ctest -V` shows all 5 tests
  (hello_test, corpus_test, dd_eft_test, dd_fma_guard_test [OFF, asserts],
  dd_fma_guard_test_contract_on [ON, exits 0 with report]) Passed;
  `kokkos_ep_demo --batch 100 --repeats 1` unchanged.
- **Scope-out.** Dekker twoProduct ONLY (the one contraction-hazard primitive); no
  extension to log/sin/etc.; no rebuilding Kokkos under different contraction
  settings (per-target flags suffice); no runtime `#pragma FP_CONTRACT` (build-flag
  approach is cleaner and covers CUDA). `dd_math.hpp` untouched; demos untouched.

**T1.6: End-to-end cancellation kernels for DD. (DONE)**

- Executed 2026-08-01. Commit `73d9f0a`.
- **`tests/dd_e2e_test.cpp` (new, ~380 lines).** Layer-6 end-to-end test: four
  classic cancellation-hostile kernels evaluated in DD and scored in digits of
  accuracy against `__float128` / closed-form oracles, mean-gated at 28.0 digits.
  Whole file is `#ifdef KOKKOS_EP_HAVE_QUADMATH`; runtime-SKIP 77 without quadmath.
  Host-side only (the kernels are inherently serial reductions/recurrences).
  `dd_math.hpp` NOT modified (rule 4).
- **Two-oracle strategy (K2, K4).** Each sum is scored twice. The
  DD-vs-quadmath-partial-sum comparison carries the arithmetic-precision claim:
  identical N, identical summation order, identical terms, so it isolates DD's
  accumulation quality from truncation. The DD-vs-closed-form comparison
  (K2 vs π²/6, K4 vs ln 2) is a truncation-limited sanity check, gated at
  `truncation_floor − 1` digit of slack. At N=10⁶ the floor is ~6 digits: the
  Basel tail Σ_{N+1}^∞ 1/k² ≈ 1/N, and the alternating-series error is bounded by
  the first omitted term ≈ 1/N.
- **K1 deviation from the literal spec (justified, confirmed with plan owner).**
  The T1.6 spec named the naive `√(x²+1) − x` as the DUT and expected ~31 digits;
  that expectation is numerically false and NOT a library defect. Catastrophic
  cancellation loses ~2·log₁₀(x) digits regardless of arithmetic precision
  (Higham, *Accuracy and Stability of Numerical Algorithms*, §1.7). Ship shape:
  `K1_stable` gates `1/(√(x²+1) + x)` vs f128 — the algebraic rearrangement that
  eliminates the cancellation; `K1_naive_report` computes and reports the naive
  form in DD AND FP64 per magnitude to demonstrate DD's ~16-digit lift under a
  hostile algorithm. Together — flat ~31 digits on the stable form (competent
  algorithm) plus the ~16-digit naive shift FP64→DD (hostile algorithm) —
  demonstrate DD's ~31-digit precision end to end.
- **Per-kernel results (mean_digits / tolerance 28.0).** `K1_stable` 31.00
  (harness cap; uncapped ~32.6), `K2_basel` 29.48, `K3_machin` 28.09,
  `K4_alt_harmonic` 29.56 — all PASS. `K1_naive_report` DD {21.25, 16.44, 1.54}
  vs FP64 {5.12, 0.00, 0.00} at x ∈ {1e6, 1e10, 1e15}. K2 sanity vs π²/6 6.22
  digits (truncation floor 6); K4 sanity vs ln 2 6.14 digits (truncation floor 6).
- **K3 margin note.** `K3_machin` clears the 28.0 gate by ~0.09 digits.
  Acceptable because the kernel is fully deterministic (no RNG, fixed constants),
  so 28.09 is reproducible run-to-run, not a flaky sample. No proven DDFUN `atan`
  bound is available — "observed empirically" per rule 5. Flagged for revisit if a
  future toolchain shift (Clang vs GCC, different libm) slips it below 28.0.
- **Tolerance rationale.** DD targets ~31.9 digits, the harness caps at 31
  (`BackendTraits<DD>::max_digits`), leaving ~3 digits of headroom for accumulated
  round-off in composed / 10⁶-term kernels → 28.0, applied uniformly to the
  arithmetic-precision comparisons.
- **`tests/CMakeLists.txt` (edit).** Registered with the plain
  `kokkos_ep_add_test(dd_e2e_test)` helper (a regular test, not an EFT test —
  no contraction flags). **`tests/README.md` (edit).** Registry table row added.
- **Scope-out.** No `dd_complex.hpp`, no FF/QF backends, no per-op differential
  accuracy (that is the T1.4 sibling). `dd_math.hpp` untouched; demos untouched.

### Follow-up bug tasks (from T1.4, T2.3, and T2.4)

Real extended-precision library defects surfaced by the property/accuracy tests
(B1–B3 from T1.4's `dd_accuracy_test`; B4 from T2.3's `ff_property_test`; B5–B7
from T2.4's `ff_accuracy_test`). Each is a library-side fix (rule 4: the surfacing
task reported, did not patch). The corresponding test is the durable acceptance
gate. Pick up in any order after Phase 2/3, not now; these are stubs, one
screenful each.

**B1: `tgamma` — Lanczos coefficients at DD precision (higher-order Lanczos, g≈20.32 / N=24). (DONE)**

- **Scope revised (2026-08-06).** The stub as originally written scoped the fix to
  "promote the nine g=7 coefficients to DD precision, keep the g=7 structure". That
  is **mathematically incapable of clearing the 25.91 gate**: the g=7 / n=9 Lanczos
  approximation has an intrinsic *truncation-order* ceiling independent of how the
  coefficients are stored. Empirical: exact 25-digit coefficients via 9-node Cauchy
  solve, mpmath at 80-digit precision, worst-case **13.03 digits** over
  a∈[0.55,150] (mid-range peaks ~19.6, large-a floor ~13). So coefficient
  promotion alone reaches at most ~13–19 digits. The original root cause
  ("`double` coefficients cap the result at ~15 digits") was *incomplete* — it
  correctly identified a real defect, but the storage precision and the
  approximation order both cap near ~15, and only the latter is binding once the
  former is fixed. This is the substantive difference from the FF sibling **B7**,
  where the ~12.84-digit target sat *under* the g=7 ceiling and coefficient
  promotion alone was sufficient. Scope is therefore *widened* (not invalidated):
  same defect, same op, larger fix. Precedent for stub revision: B4 → B8.
- **Read first:** `dd_math.hpp:723-751` (tgamma), how `DoubleDouble_pi()` etc.
  build DD constants via `from_bits(hi, lo)`; `tests/dd_accuracy_test.cpp` tgamma
  row (random pass is `uniform(0.1, 50)` × 10⁶, so the **mean is dominated by
  large a** — precisely where the g=7 ceiling is worst); B7 (FF sibling); this block.
- **Root cause.** `tgamma` uses Lanczos g=7 with FP64 coefficient constants
  (`dd_math.hpp:729-738`: `676.5203681218851`, …), measured mean 14.56, min ~0
  near poles. Two compounding caps: (i) `double` coefficients truncate to ~15.65
  digits; (ii) the g=7 / n=9 approximation order itself caps at ~13–19 digits.
  Fixing (i) alone leaves (ii) binding.
- **Fix options.** **(a) [selected] Higher-order Lanczos**, g≈20.32 / N=24, the
  Boost `lanczos24m113` parameter shape (published, well-tested). Keeps the
  already-validated series structure — `c_0 + Σ c_k/(x+k)` with the
  `sqrt(2π)·t^(x+0.5)·e^−t` prefactor — and the same "promote coefficients to DD
  via `from_bits(hi, lo)` split" mechanic as B7; only the order and coefficient
  count change (24 instead of 9). (b) [rejected] g=7 with coefficient promotion
  as originally scoped — capped at ~13 digits, cannot clear the gate. (c)
  [rejected] Switch to Stirling asymptotic + reflection (QD's `qd_real.cc`) —
  larger diff and a new algorithm to validate, unnecessary when the Lanczos
  structure still reaches DD precision at higher order.
- **Coefficient-form caveat (matters for choosing g).** Boost's published
  lanczos24m113 coefficients are for the **rational** (ratio-of-polynomials)
  evaluation form, which is *immune* to cancellation. This project's tgamma uses
  the **partial-fraction** form, whose coefficients grow as max|c_k| ~ 10^(g/2)
  while the sum stays O(1) — so raising g trades truncation error for
  **cancellation** error, and the two cross. The coefficients must therefore be
  re-derived for the partial-fraction form at the chosen g (not transcribed from
  Boost), and g itself is a tuning parameter with an interior optimum. The
  implementing task should sweep (g, N) under simulated DD arithmetic and pick the
  optimum, reporting the measured mean for the chosen point.
- **Background / deliverables.** New DD coefficient constants + tgamma rework in
  `dd_math.hpp` (its own task/branch), plus a committed generator script so the
  2·N magic numbers are reproducible rather than transcribed. Cite the coefficient
  source and derivation method in comments.
- **Acceptance gate (unchanged).** `dd_accuracy_test` tgamma mean ≥ 25.91; full
  ctest green; no other op regresses.
- **Scope-out.** tgamma only (not lgamma/digamma); no new test file.
- **STATUS (resolved).** See DONE block below (commit `9c91b16`). The numbers and
  line refs above are **pre-fix** and retained as the historical problem
  statement: tgamma mean is now **28.30**, not 14.56, and tgamma lives at
  `dd_math.hpp:912-962` (the DD coefficient pairs at ~920-937), not the cited
  `723-751` / `729-738`. Root-cause finding on close: the `double` coefficient
  literals were a real defect but **not the binding one** — the g=7/n=9
  *truncation order* caps at ~13 digits at large `a` regardless of how exactly
  the coefficients are stored, so fixing storage alone could never clear the
  gate. Shipped **g=14/N=17**, not the prescribed g≈20.32/N=24: Boost's
  `lanczos24m113` g is derived for the **rational** evaluation form and does not
  transfer to this repo's **partial-fraction** form, where max|c_k| ~ 10^(g/2)
  against an O(1) sum makes cancellation grow with g (g=20.32 measured 26.17 vs
  g=14's 28.30). See DONE for full detail; the general lesson is consolidated as
  `PORT_NOTES.md` §4f.

- **DONE (2026-08-06).** Promoted DD `tgamma` to Lanczos **g=14 / N=17** with
  DD-precision coefficients (`from_bits(hi, lo)` pairs). Acceptance gate **PASS**:
  `dd_accuracy_test` tgamma mean **14.56 → 28.30**. **Fourth post-Phase-3
  library-side bug FIX task** (after B8, B7, B5) and the **first DD-side one** —
  every prior fix was FF-side.
- **Commits.** `402bed2` (stub rewrite: scope widened to higher-order Lanczos) →
  `9c91b16` (task commit, branch `b1-dd-tgamma-lanczos-higher-order`) → `ad51e1c`
  (merge to `main`, branch deleted) → this DONE block + its docs-pointer
  (`045f2c9`).
- **Result.** tgamma mean 14.56 → **28.30** (gate ≥ 25.91, **+2.39-digit
  margin**), min 0.00 → 0.00, n=1000077, skipped=88. Of the 50 scored rows
  **exactly one changed**; the other 49 — including the preserved erf 24.64 /
  erfc 19.50 REDs and every EXPECTED-MIN-DROP row — are digit-for-digit
  identical. Full ctest **22/23, 1703s**; the sole RED is `#6 dd_accuracy_test`,
  now failing on **erf/erfc only** (was erf/erfc/tgamma). All six demos build and
  run RC 0 (tgamma is not in the demo op inventory, so that gate is build+run
  only). Zero `dd_math.hpp` warnings under the project's flags plus
  `-O3 -Wall -Wextra`.
- **Root cause (the original stub's diagnosis was incomplete).** The stub blamed
  the `double` coefficient literals alone. That is a real defect but **not the
  binding one**: g=7 / n=9 has an intrinsic *truncation-order* ceiling of ~13
  digits at large `a`, proven by recomputing the g=7 set to 25 exact digits
  (9-node Cauchy solve, mpmath at 80-digit precision) and re-measuring — worst
  case **13.03** over a∈[0.55,150]. Those recomputed values reproduce the shipped
  literals to 16 digits, confirming the coefficients were already the correct g=7
  set. Storage cap (~15.65 digits) and order cap (~13–19) both sit near ~15, so
  fixing only storage leaves the order binding. This is exactly why **B7**'s
  identical fix succeeded for FF — its 12.84 target sits *under* the g=7 ceiling —
  and could not work here. Fix keeps the series structure
  (`c_0 + Σ c_k/(x+k)` with the `sqrt(2π)·t^(x+0.5)·e^−t` prefactor), the DD
  accumulator (already DD, untouched as in B7), and the reflection path
  (`DoubleDouble_pi()`, already a full two-word constant); only the order, term
  count, and coefficient storage change. `g+1/2 = 14.5` is exact in binary, so `t`
  carries no representation error into `pow(t, x+0.5)` where `x` would amplify it.
- **Deviation 1 (accepted on the merits).** Shipped **g=14 / N=17** instead of the
  prescribed g≈20.32 / N=24. This repo's tgamma uses the **partial-fraction** form,
  where max|c_k| ~ 10^(g/2) against an O(1) sum, so cancellation grows with g (at
  g=20.32, max|c_k| ~ 1e10.5 → ~10.5 digits lost). Boost's `lanczos24m113`
  coefficients target the **rational** (ratio-of-polynomials) form, which is
  structurally immune to that cancellation — **the g does not transfer between
  forms**. Raising g therefore trades truncation error against cancellation error,
  crossing at an interior optimum near g=14. Both configurations were implemented
  and measured on real hardware: g=20.32/N=24 → **26.17** (+0.26 margin, thin for a
  durable gate); g=14/N=17 → **28.30** (+2.39 margin, and 7 fewer DD divisions per
  call). Simulation under 106-bit arithmetic predicted 25.87 / 28.28 vs 26.17 /
  28.30 measured, so the model is calibrated. Revert to the Boost parameter point
  is one command:
  `python3.12 scripts/gen_dd_lanczos_coeffs.py --g 20.3209821879863739013671875 --n 24`.
- **Deviation 2 (accepted).** `scripts/gen_dd_lanczos_coeffs.py` committed (new,
  non-library). Because the coefficients are *derived* rather than transcribed, the
  34 hex magic numbers would otherwise be unverifiable; the script regenerates the
  18 shipped `from_bits` pairs **bit-for-bit** (verified), carries the (g, N) sweep
  behind `--scan`, and documents the cancellation-vs-truncation tradeoff. Requires
  mpmath; the interpreter on the target machine is `python3.12` (the default
  `python3` is 3.6 and has no mpmath).
- **Coefficient source.** Derived for the partial-fraction form by the committed
  generator: exact 17-node Cauchy solve at 150-digit working precision (mpmath
  1.3.0), split via `hi=(double)c`, `lo=(double)(c−hi)`. Method reference:
  C. Lanczos, "A Precision Approximation of the Gamma Function", *J. SIAM Numer.
  Anal. B* **1** (1964) 86–96; P. Godfrey (2001), "A note on the computation of the
  convergent Lanczos complex Gamma approximation".
- **PORT_NOTES (deferred, batched).** Fourth companion draft, joining B7/B8/B5 for
  the eventual consolidation pass. Draft text is parked in the `9c91b16` commit
  body — `PORT_NOTES.md` was **not** extended inline.
- **Follow-ups.** None. B1 does **not** unblock B2/B3 — DD erf/erfc remain
  independent open tasks and are now the only ctest RED.
- **Rule 4 respected.** `dd_math.hpp` is the ONLY library file touched; no
  `ff_math.hpp` / `qf_math.hpp` / `*_complex.hpp`, and no `tests/` changes (the
  25.91 gate was already encoded in `dd_accuracy_test`'s tolerance table).

**B2: `erfc` — direct computation for large |z|.**

- **Read first:** `dd_math.hpp:698-720` (erf asymptotic branch + erfc); Boost.Math
  / DDFUN erfc reference for the cutoff; `tests/dd_accuracy_test.cpp` erfc row.
- **Root cause.** `erfc(z) = subtract(DoubleDouble(1.0), erf(z))`
  (`dd_math.hpp:718-720`) is catastrophic cancellation as erf(z) → 1 (26.4 digits
  at x=3, 18.2 at x=5, 0 at x=8; mean 19.50).
- **Fix.** Direct asymptotic/continued-fraction erfc for |z| above a threshold
  (~0.5–1; Boost uses |z| > 0.5); fall back to `1 − erf(z)` only for small |z|
  where erf is far from 1. Likely shares the asymptotic-region code path with B3.
- **Acceptance gate.** `dd_accuracy_test` erfc mean ≥ 25.91; full ctest green; no
  other op regresses.
- **Scope-out.** erfc path only; cross-reference B3 for the shared branch.
- **STATUS (resolved).** See DONE block below (commit `ebed8c7`). The numbers and
  line refs above are **pre-fix** and retained as the historical problem
  statement: erfc mean is now **27.97**, and erfc lives at `dd_math.hpp:850-885`
  (`kDirectMin` at 853, `kUnderflowMax` at 862), not the cited `698-720` /
  `718-720`. Note the **two-stage history** behind the "mean 19.50" figure above:
  **B3** lifted it 19.50 → **24.87** without touching `erfc()` at all — making
  `erf`'s asymptotic branch live routed `1 − erf` through `erf`'s `lo` word,
  which recovers ~16 digits over |z| ∈ [5.25, 8.5] and then stalls at exactly the
  53 bits a `double` lo word holds — and B2 then closed the residual 24.87 →
  27.97. Root-cause finding on close: the defect is **wider than this stub
  states, and a ramp rather than a cliff**. Loss is `log10(1/erfc(z))` by
  construction, so it starts near z ~ 1 (22.95 digits at z = 4), and *every*
  point from z = 3.68 through 7.70 sits under the 25.91 gate. The prescribed
  direct-asymptotic fix provably cannot reach that band — the expansion's
  optimal-truncation floor is worth only ~11 digits at z = 5, worse than the
  subtract it would replace — so [3.68, 7.70] remains sub-gate and the row passes
  on the mean, which is what `dd_accuracy_test` actually encodes. Also: threshold
  shipped at **6.5**, not this stub's "~0.5–1; Boost uses |z| > 0.5" — Boost's
  0.5 cut selects a rational minimax approximation, a different algorithm whose
  accuracy floor is not tied to z, so that number does not transfer to the series
  this stub prescribes. **B6 must not be re-scoped to close the residual band.**
  See DONE for full detail; consolidated as `PORT_NOTES.md` §4h.

**B2: DD erfc direct asymptotic expansion for |z| ≥ 6.5. (DONE)**

- Executed 2026-08-07. Task commit `ebed8c7`, merge `8225671`.
- **`third_party/include/dd_math.hpp` (edit, +129/-19).** Factored
  `erfc_asymptotic_sum(az, z2)` out of erf()'s asymptotic branch
  (A&S 7.1.23, optimal truncation, returns the raw sum; scaling
  stays at call sites so erf()'s arithmetic is preserved verbatim).
  `erfc()`: for `z ≥ kDirectMin = 6.5`, scale directly by
  `e^{-z²}/√π` and never form `1 - erf`. For `z > kUnderflowMax =
  27.25` return +0 (also catches +inf via `!(z.hi <= …)`). Below
  6.5 falls through to the original `subtract(1, erf(z))` path;
  negative z is unchanged (erfc(-|z|) ~ 2, subtract is benign).
- **Scaling form: `multiply(sum, e^{-z²})` not `divide(sum, e^{z²})`.**
  Forced by two DDFUN-port limits: (a) `exp` has a hard `|arg| ≥
  300 → 0/inf` guard (would zero erfc from z = 17.32 up), (b)
  Dekker splitter overflows for `b.hi > DBL_MAX/(2²⁷+1) ~ 1.3e300`
  reached at z = 26.29 (would turn divide into NaN). Above z² =
  300 the same exp guard is cleared by quartering the argument and
  squaring twice (max `|z²/4|` = 185.7 at kUnderflowMax); costs ~2
  bits — measured 30.45 digits at z = 20, 29.38 at z = 26.
- **Threshold derivation (kDirectMin = 6.5).** Uniform(-10, 10)
  mean is flat ±0.03 digit across [5.6, 6.6] (27.99 down to 27.96)
  and decides nothing. Cut placed on the pointwise rule instead:
  where the direct series stops regressing anything vs the
  fallback it replaces. Fallback above z = 6.0 is B3's lo-word
  shelf (mean 16.53 digits over [6.3, 8.5], scatter to 19.20); the
  direct series rises ~5.3 digits per unit z and clears that
  envelope at z ~ 6.44. 6.5 sits just above; over 6001 grid points
  on [6, 9] at 0.0005 spacing NO point scores worse than shipped.
  Rejected alternative kDirectMin = 5.75 (minimax cut): mean-
  optimal and lifts the composite trough 13.55 → 14.50 digits, but
  regresses 34/721 grid points over (4, 10] by up to 1.86 digits.
  Selection rule is "no regressions", so 6.5.
- **Verification.** dd_accuracy_test erfc row: mean 24.87 → 27.97
  (gate 25.91, +2.06). Min unchanged at −0.00 (structural: corpus
  feeds erfc 79.5..100.5 where true erfc is 1e-2747..1e-4389,
  unrepresentable in any double-based type; the row gates on mean
  only). All 49 other scored rows digit-for-digit identical vs a
  baseline binary built from main's `dd_math.hpp`. erf row
  unchanged 30.67 / 0.94, verified bit-for-bit over 236,014 points
  (0.0005 grid over [-9, 9], 200k pseudorandom, plus specials).
  ctest 22/23 → 23/23. All six demos
  (`kokkos_ep_demo{,_complex,_ff,_ff_complex,_qf,_qf_complex}`)
  RC 0. Zero `dd_math.hpp` warnings under project flags plus
  `-O3 -Wall -Wextra`. Specials all checked (±0, ±min subnormal,
  ±DBL_MIN, ±1, seam neighbours 6.4999/6.5/6.5000001, 8.5,
  17.3/17.33, 27.25 ± eps, 1e8, 1e150, 1e153, 1e200, ±DBL_MAX,
  ±inf).
- **Blast radius.** grep-verified: erfc has no demo consumer and
  no `dd_complex.hpp` consumer, so only `dd_accuracy_test` and
  `dd_invariant_test` (both green) are affected.
- **DEVIATION 1 — the defect is wider than the brief states, and
  this fix by construction cannot close all of it.** Reported, not
  silently narrowed. Brief characterised post-B3 erfc as "benign
  for |z| ≤ 5.25, ~16 digits over [5.25, 8.5], hard 0 above 8.5";
  measured, the first clause is too generous — the loss is
  `log10(1/erfc(z))` and starts at z ~ 1 (erfc is 22.95 digits at
  z = 4, 18.39 at z = 5). On a 0.02 grid, scattered points fall
  under the 25.91 gate from z = 2.90, and EVERY point is under it
  from z = 3.68 through z = 7.70. The prescribed fix (direct
  asymptotic above a threshold) provably cannot reach that band:
  the asymptotic expansion's optimal-truncation floor is worth
  only ~11 digits at z = 5, worse than the subtract it would
  replace. Closing that band needs a different algorithm — a
  Lentz continued fraction (A&S 7.1.14) or a triple-double erf —
  which is a larger change than this task scopes. The row now
  PASSES on the mean (the gate `dd_accuracy_test` actually
  encodes) with +2.06 digits of margin, and every point that was
  reachable by the prescribed fix is fixed. The composite trough
  remains 13.55 digits at z = 5.915 — pre-existing `1 - erf`
  behaviour, left exactly as it was, NOT introduced here. Flagged
  so future B6 (FF erfc quality-lift) doesn't get re-scoped to
  try to close it either.
- **DEVIATION 2 — threshold 6.5, not the stub's "~0.5-1; Boost
  uses |z| > 0.5".** Boost's 0.5 cut selects a rational minimax
  approximation, a different algorithm with no accuracy floor
  tied to z. This series' floor is ~`e^{-z²}`; at z = 0.5 it is
  worth under 1 digit. The stub number doesn't transfer to the
  algorithm the stub (and this task) prescribes. Full derivation
  above.
- **DEVIATION 3 — erf() was touched, under the brief's explicit
  sanction, and is verified bit-identical.** The brief allows
  "optionally factors a shared static helper … your judgment
  call". Taken: duplicating a 12-line divergent series with hand-
  tuned truncation in two functions is the drift pattern that
  produced this bug family. Extraction is pure — erf()'s scaling
  expression (`divide by sqrt(pi)*exp(z2)`) is left verbatim at
  the call site rather than moved into the helper, precisely so
  erf's arithmetic is unchanged. Verified over 236,014 points,
  every hi and lo word identical to main.
- **Rule 4.** `third_party/include/dd_math.hpp` is the only file
  touched. `dd_complex.hpp`, `ff_math.hpp`, `ff_complex.hpp`,
  `qf_math.hpp`, `qf_complex.hpp`, and all of `tests/` untouched.
  No tolerance changed; the 25.91 gate was already encoded in
  `dd_accuracy_test`'s table.
- **PORT_NOTES draft — sixth deferred companion** (joining
  B7 f-suffix, B8 divide-splitter, B5 t2/t3 growth, B1 DD-Lanczos,
  B3 erf truncated series §4c for the eventual batch consolidation
  pass). Draft body: §4d "erfc: when the identity is the bug" —
  covers the `1 - erf` catastrophic-cancellation trap
  (log10(1/erfc(z)) digit loss by construction at every
  precision), the lo-word-shelf ceiling B3 hit, the direct-
  asymptotic fix, and the cross-format note on DDFUN-port limits
  (`|arg| ≥ 300` exp guard + Dekker splitter overflow at
  DBL_MAX/(2²⁷+1)) that surface only once output range extends
  past ~1e-130. General lesson: when a special function is
  defined as an identity over another one (`1 - erf`, `1 - cos`,
  `log(1 + x)`, `exp(x) - 1`), check the identity's conditioning
  before tuning the callee. `PORT_NOTES.md` NOT extended inline.
- **Backlog after this closes.** ctest fully GREEN across FF, DD,
  QF. Remaining: B6 (FF erfc quality-lift — inherits DEV1's
  pointwise-band limitation, do not re-scope), PORT_NOTES
  consolidation pass (six batched drafts), optional stale-branch
  prune. Reet's next call.
- Depends on T1.4 (regression gate), B3 (erf asymptotic branch
  this factors from).

**B3: `erf` — asymptotic branch for |z| > ~8. (DONE)**

- **Read first:** `dd_math.hpp:669-716` (erf, both branches);
  `tests/dd_accuracy_test.cpp` erf row; B2 (shared asymptotic-region concern).
- **Root cause.** erf delivers 30–31 digits for |z| ≤ 5 but the large-|z|
  asymptotic branch (`dd_math.hpp:698-715`) collapses to ~5 digits at x=8,
  dragging the uniform(−10,10) random-pass mean to 24.64. The moderate-|z| Taylor
  branch is fine; the fix is scoped to the large-|z| asymptotic branch.
- **Fix.** Repair/replace the asymptotic expansion for |z| > ~8 (convergence /
  term-ordering); likely a shared fix with B2's direct-erfc asymptotic path.
- **Acceptance gate.** `dd_accuracy_test` erf mean ≥ 25.91; full ctest green; no
  other op regresses.
- **Scope-out.** erf large-|z| branch only; coordinate with B2.
- **STATUS (resolved).** See DONE block below (commit `2bb18e3`). The numbers and
  line refs above are **pre-fix** and retained as the historical problem
  statement: erf mean is now **30.67**, not 24.64, and erf lives at
  `dd_math.hpp:739-797` (`kTaylorMax` at 763), not the cited `669-716` /
  `698-715`. Root-cause finding on close: **both halves of the diagnosis above
  were falsified.** The large-|z| asymptotic branch this stub blames was
  **unreachable dead code** — its `|z| < 9.0` guard cannot be false once the
  `|z| > 8.5` saturation four lines above has returned — and the Taylor branch
  this stub calls "fine" was the actual defect: its `k ≤ 100` cap truncates a
  series needing `k ≈ z² + 50` terms (182 at |z| = 8). A truncated partial sum
  degrades *continuously*, which is why the observed 30.01 → 3.17 fall is a
  smooth ramp and not the cliff a broken branch would produce — the tell that
  should have redirected the diagnosis earlier. The fix therefore **necessarily
  touched the Taylor loop this stub scopes out**; the hole spans |z| ∈ [4.9, 8.5]
  and cannot be closed from the asymptotic side alone. B2 interaction: this fix
  lifted erfc 19.50 → 24.87 downstream through the untouched `1 − erf`, but did
  **not** close B2. See DONE for full detail; consolidated as `PORT_NOTES.md`
  §4g.

- **DONE (2026-08-06).** Raised the Taylor iteration cap, lifted the asymptotic
  branch out of dead code by moving the switchover to |z| = 6, and added
  optimal truncation to the (divergent) asymptotic series. Acceptance gate
  **PASS**: `dd_accuracy_test` erf mean **24.64 → 30.67**. **Fifth post-Phase-3
  library-side bug FIX task** (after B8, B7, B5, B1) and the **second DD-side
  one**.
- **Commits.** `2bb18e3` (task commit, branch `b3-dd-erf-asymptotic-branch`) →
  `8fb1d2d` (merge to `main`, branch deleted) → this DONE block + its
  docs-pointer (`bd61bf0`).
- **Result.** erf mean 24.64 → **30.67** (gate ≥ 25.91, **+4.76-digit margin**),
  n=1000165, skipped=0. erf min 0.94 → **0.94, unchanged** — the input is the
  smallest subnormal `4.94e-324` from the corpus (denormal conditioning,
  pre-existing, bit-identical to baseline; erf gates on the mean only, and the
  random-pass worst is 29.35 at z = −1.9102). All **48 other scored rows are
  digit-for-digit identical**, diffed row by row against a stash-rebuilt
  baseline binary. Full ctest **22/23** (was 21/23); the sole RED is
  `#6 dd_accuracy_test`, now failing on **erfc only** (was erf/erfc). All six
  demos build and run RC 0 — erf/erfc are header-only special functions with no
  demo and no `dd_complex.hpp` consumer (grep-verified), so the blast radius is
  `dd_accuracy_test` alone. Zero `dd_math.hpp` warnings under the project's
  flags plus `-O3 -Wall -Wextra`. Spot-checks vs quadmath (digits, clamp 33):
  z=6 18.46 → **32.55**, z=7 10.42 → **33.00**, z=8 5.03 → **33.00**,
  z=8.5 3.17 → **33.00**.
- **Root cause (differs from the original stub).** The stub blamed the large-|z|
  asymptotic branch and asserted the Taylor branch "is fine". Reality-first
  check falsified both halves. **(a)** Old lines 698–715 were **unreachable dead
  code**: the `|z| < 9.0` guard cannot be false once the `|z| > 8.5` saturation
  four lines above has returned. Had it run it would also have been wrong — the
  erfc asymptotic series (A&S 7.1.23) is **divergent**, so its relative-eps exit
  could never fire and the loop would have summed far past the smallest term.
  **(b)** The ~5-digit reading at x=8 came from the **Taylor** branch, whose
  `k ≤ 100` cap truncates a series needing `k ≈ z² + 50` terms (measured against
  an mpmath reference: 106 at |z|=5, 130 at 6, 182 at 8, 197 at 8.5). A
  truncated partial sum degrades *continuously*, which matches the observed
  30.01 → 3.17 ramp exactly — a smooth fall, not the cliff a broken branch
  would produce.
- **Fix.** (1) **Switchover at |z| = 6** (new named constant `kTaylorMax`)
  replacing the dead `< 9.0`, making the asymptotic branch live for
  6 ≤ |z| ≤ 8.5. Derivation: the asymptotic expansion's optimal-truncation floor
  is its smallest term ~ e^−z²; carried through erf = 1 − erfc that is an
  absolute error ≈ e^−2z²/(z·√π), which first drops below u² = 2⁻¹⁰⁶ ≈ 1.23e-32
  at |z| ≈ 5.97. Empirical crossover sweep over (0, 8.5] on a 0.005 grid
  confirms 5.75–7.0 all give the same minimum, **29.49 digits at z = 0.435** — a
  generic-roundoff point, not a branch artifact — so 6.0 sits mid-window
  (5.5 → 27.46, 5.0 → 22.86). No dip at the seam: erf(5.95) = 29.70,
  erf(6.05) = 31.00. (2) **Taylor cap 100 → 200** (measured need 130 as
  |z| → 6⁻; sensitivity: cap 100 → 18.46, 120 → 28.35, ≥130 → 29.49; ~1.5×
  margin). (3) **Optimal truncation** on the asymptotic branch: stop as soon as
  |term| stops decreasing, standard for a divergent expansion, with the
  relative-eps test demoted to a secondary exit (measured worst case 72 terms at
  |z| = 8.5 against a k = 100 cap). (4) **Term recurrence** — see Deviation 2.
  Series identities cited in-source: A&S 7.1.6 (Taylor), A&S 7.1.23
  (asymptotic erfc), the same pair as the FF sibling B5.
- **Deviation 1 (accepted on the merits — stub falsified).** The root cause
  differs from the stub, and the fix therefore **necessarily touches the Taylor
  loop the stub scoped out**. This was forced, not elective: the accuracy hole
  spans |z| ∈ [4.9, 8.5] and cannot be closed from the asymptotic side alone
  (the asymptotic branch at |z| = 4.9 is worth only ~22 digits, under gate). The
  widening stays strictly inside `erf()`'s body in `dd_math.hpp` — no other
  function, no other file, no test change, no tolerance change. Precedent for
  shipping-with-deviation rather than report-and-stop: **B1**'s stub prescribed
  g≈20.32/N=24 and reality selected g=14/N=17; **B5**'s stub prescribed an
  overflow-probe recipe its own fix made moot. Both shipped with the deviation
  documented.
- **Deviation 2 (accepted — robustness win).** B5's overflow-safe term
  recurrence is **NOT load-bearing at DD**. FP64 has the headroom: measured at
  `kTaylorMax = 6.0` the old separate-accumulator form peaks at 3.3e238
  (numerator 2^k·z^(2k+1)) and 1.7e255 ((2k+1)!!) against `DBL_MAX` 1.8e308, and
  scores **digit-for-digit identically** to the recurrence at every cap from 130
  through 240. So the overflow half of B5 does not transfer as a *correctness*
  fix. Adopted anyway for **cap independence**: with separate accumulators
  (2k+1)!! reaches `DBL_MAX` near k ≈ 150, so the cap and the switchover would
  be pinned within ~10% of each other, and at `kTaylorMax = 7` the old form
  returns **NaN outright** (measured, cap 200 and cap 400 alike). It is also
  cheaper — one DD/double divide per term instead of a full DD/DD divide — and
  it *replaces* machinery rather than adding it. Minimal-diff alternative is
  documented in the task commit: drop the recurrence and set the Taylor cap to
  140, which also clears the gate, at the cost of a ~7% margin on the cap and a
  silent-NaN landmine one z-unit above the switchover.
- **B2 interaction (improved, NOT closed).** erfc mean **19.50 → 24.87**,
  downstream through the **untouched** `erfc = subtract(1, erf)`: on the
  asymptotic branch erf returns `1 − erfc_val` with `erfc_val` small enough to
  sit exactly in the DD `lo` word, so the outer subtract recovers ~16 digits
  over |z| ∈ [5.25, 8.5] (z=6.0: 15.89, z=8.0: 16.24, versus 1.79 / 0.00
  before). Above |z| = 8.5 erf still saturates to exactly 1, so erfc returns
  exactly 0 and scores 0 digits — B2's catastrophic-cancellation defect is
  intact and still holds the mean under gate. erfc min −0.00 → −0.00,
  unchanged. **B3 does not close B2**; B2 is now the sole open ctest RED and
  remains an independent task.
- **PORT_NOTES (deferred, batched).** Fifth companion draft (§4c, "`erf`: a
  truncated series that looks like a bad expansion"), joining B7/B8/B5/B1 for
  the eventual consolidation pass. Draft text is parked in the `2bb18e3` commit
  body — `PORT_NOTES.md` was **not** extended inline.
- **Follow-ups.** None B3-owned. Backlog: **B2** (DD erfc — the sole remaining
  ctest RED), **B6** (FF erfc quality-lift), and the PORT_NOTES consolidation
  pass (five batched drafts now).
- **Rule 4 respected.** `dd_math.hpp` is the ONLY file touched, and only
  `erf()`'s body within it; no `erfc()` / `tgamma()` / other DD functions, no
  `dd_complex.hpp`, no `ff_math.hpp` / `qf_math.hpp` / `*_complex.hpp`, and no
  `tests/` changes (the 25.91 gate was already encoded in `dd_accuracy_test`'s
  tolerance table).

**B4: FF `exp` — original scope INVALIDATED by empirical investigation. Superseded by B8. (SUPERSEDED)**

- **Status: closed without code fix.** The stub's originally-hypothesized root
  cause (Taylor-eps constant `1.0e-15f` finer than FF's ~3.55e-15 resolution,
  causing terms to never fall below the convergence threshold) was empirically
  falsified during B4's execution attempt (2026-08-05).
- **Investigation summary (from cluster-Claude's B4 attempt, uncommitted).**
  Instrumented `ff_math.hpp:349` exp Taylor loop with per-stall categorization
  across three eps values (`1.0e-15f` original, `3.55e-15f` = u_FF, `5.0e-15f`
  margin). Ran T2.3's B3/B9 identity tests over `[-85, 85]` under each. Result:
  **31,085 FFEXP iteration-limit stalls at every eps value** (identical count),
  with **0% Taylor-plateau stalls and 100% NaN-input stalls**. Changing eps had
  zero observable effect on stall count or on `ff_accuracy_test` exp mean (13.2962
  digits both). The eps constant is cosmetically imperfect but functionally a no-op.
- **Real root cause (surfaced by B4's investigation).** `divide()` at
  `ff_math.hpp:209` uses the FP32 Dekker splitter (`split = 8193.0f`). For divisor
  `b` with `|b.hi| > FLT_MAX / 8193 ≈ 4.15e34`, the internal `b.hi * split`
  overflows to `±∞`, and the subsequent `(b.hi * split) - b.hi = ∞ − ∞ = NaN`.
  Inside `log()`'s Newton iteration, the divisor is `exp(b) ≈ a`, which exceeds the
  splitter overflow threshold for `x ≳ 79.7`. The NaN then propagates back into
  `exp()` on the next Newton step; NaN comparisons in the Taylor convergence check
  always evaluate false, so the loop never breaks and stalls at the 60-iteration
  cap. **Same bug class as PORT_NOTES §4a** (exp large-input NaN from splitter
  overflow), living in a different site (divide's splitter, not exp's final scaling).
- **Empirical clean ceiling for FF exp inputs (from a probe sweep during B4's
  investigation): x ≈ ±79.7.** Measured: `[-79.7, 79.7]` → 0 NaN; `[-80, 80]` →
  891; `[-85, 85]` → 15,491. FP32's own exp overflow threshold (~±88 =
  FLT_MAX_EXP·ln2) is NOT the binding constraint — the FF divide splitter overflow
  bites first.
- **Superseded by B8** (FF `divide` splitter overflow), which fixes the real defect
  and enables the honest B3/B9 domain restoration. No changes to `ff_math.hpp` under
  the B4 label; the eps swap is not shipped (functionally a no-op, defensible as
  hygiene but not honest as a bug fix).
- **T2.3 B3/B9 domain narrowing stays as-is** (`[-69, 69]`) until B8 lands, at which
  point B3/B9 restore to `[-79, 79]` (the true clean ceiling, safety-margined one
  integer under `79.7`).

**B5: FF `erf` — asymptotic branch broken + FP32 Taylor overflow. (DONE)**

- **Read first:** `ff_math.hpp:694` (erf, both branches — the large-|z|
  asymptotic branch in particular); `tests/ff_accuracy_test.cpp` erf row; B3 (DD
  sibling — shared asymptotic-region concern); this block.
- **Root cause.** erf returns **NaN across the smooth well-conditioned range
  ~[1.9, 6]** (mean 3.94, min 0.00). Two failures compound at FP32: (a) the Taylor
  branch overflows earlier than DD does because FP32's exponent range is ~10³⁸ vs
  FP64's ~10³⁰⁸, so intermediate term products blow up at |z|~2 rather than the DD
  Taylor's clean cutoff; (b) the large-|z| asymptotic branch is broken (same defect
  DD's B3 flags, port-inherited).
- **Fix.** Same shape as B3 (repair/replace the large-|z| asymptotic expansion),
  plus lower the Taylor→asymptotic switchover threshold to well inside FP32's
  overflow-safe region. Likely shares the asymptotic-region code path with B6 (as
  B3 shares with B2).
- **Acceptance gate.** `ff_accuracy_test` erf mean ≥ 8.45; full ctest green; no
  other op regresses; verify erf(2.0) is finite and matches the oracle to ≥ FF's
  precision floor.
- **Scope-out.** erf both branches only; coordinate with B6.
- **STATUS (resolved).** See DONE block below (commit `1013cb1`). The numbers and
  line refs above are **pre-fix** and retained as the historical problem
  statement: erf mean is now **13.72**, not 3.94, the NaN band over ~[1.9, 6] is
  gone (`erf(2.0)` = 13.98 digits, finite), and erf lives at
  `ff_math.hpp:716-786` (`kTaylorMax` at 749), not the cited `694`. Root-cause
  finding on close: **the diagnosis above was confirmed — both halves — but the
  fix path diverged from the recipe.** (a) and (b) are one defect wearing two
  faces: separately-grown `t2 = 2^k z^{2k+1}` / `t3 = (2k+1)!!` intermediates
  overflow FP32 at k ~ 26–31, and the divergent asymptotic series runs to its
  k=60 cap where `(2k−1)!!` does the same. A single **term recurrence** fixes
  both by keeping every intermediate O(term). Consequence: the recurrence
  **eliminated the overflow outright**, so the Taylor→asymptotic switchover
  (4.0 → 3.5) is set by the convergence/accuracy crossover, *not* by the
  Taylor-overflow probe the task prompt prescribed to bound it — the probe was
  made moot by its own fix. Downstream side effect (anticipated, not gated):
  erfc lifted 4.09 → 11.85 through the untouched `subtract(1, erf(z))`, flipping
  `ff_accuracy_test` RED → GREEN and **de-gating B6**, which survives only as a
  quality-lift task for erfc's min. See DONE for full detail; consolidated as
  `PORT_NOTES.md` §4e.

- Executed 2026-08-06. Task commit `1013cb1` on branch
  `b5-ff-erf-asymptotic-plus-taylor` (now merged to `main` via merge commit
  `c3a512b` and the branch deleted); DONE block + docs-pointer this bundle
  (T3.5/T3.6/B8/B7-style two-commit clean-GREEN close). **Third post-Phase-3
  library-side bug FIX task** (after B8 and B7). **MILESTONE: this task flips
  `ff_accuracy_test` from RED to GREEN — all originally-preserved FF-side REDs are
  now resolved; the sole remaining ctest RED is `dd_accuracy_test`, which is DD-side
  only (pending B1/B2/B3).**
- **What shipped (1 file, +55/−27 LOC).** `third_party/include/ff_math.hpp` only:
  the `erf()` body (both branches) plus the B5 comment block. **Rule 4 respected**
  — the ONE library file the B5 stub authorizes; no other `*_math.hpp` /
  `*_complex.hpp` (and no `dd_math.hpp`, confirming DD tgamma/erf/erfc stay
  untouched pending B1/B2/B3).
- **Root cause (both defects, both structural DD→FF port artifacts).** (a) **Taylor
  branch:** the DD port accumulates each term from a separately-grown numerator
  (`t2 = 2^k z^{2k+1}`) and denominator (`t3 = (2k+1)!!`). At FP64 those
  intermediates stay finite; at FP32 (max ~3.4e38) both overflow around k~26–31 for
  `|z|` in [2, 4), so `t2/t3 → inf/inf = NaN` before the convergence test fires.
  (b) **Asymptotic branch:** the erfc asymptotic series is *divergent*, so the
  relative-eps convergence test can never trip; the loop ran to its fixed k=60 cap
  where `(2k−1)!!` overflows FP32 → `inf/inf = NaN`. This is the FF sibling of the
  still-open DD-B3 (`dd_math.hpp:698` collapses to ~5 digits there rather than NaN,
  because FP64 has the exponent headroom to avoid overflow).
- **Fix technique (three parts, unified via the recurrence).**
  1. **Term-recurrence rewrite**, both branches. Each term is now derived from the
     previous via its running ratio: Taylor `term_k = term_{k−1} · 2z² / (2k+1)`;
     asymptotic `term_k = term_{k−1} · −(2k−1) / (2z²)`. Every intermediate stays
     O(term) → FP32 overflow is STRUCTURALLY impossible for `|z| ≤ 6` (Taylor sum
     bounded by `(√π/2)·e^{z²}`, ~3.8e15 at `|z|=6`, well under `FLT_MAX`).
     Reference: Abramowitz & Stegun 7.1.6 (Taylor) and 7.1.23 (asymptotic erfc),
     cited inline (DLMF 7.12.1 is the same expansion).
  2. **Optimal truncation** on the asymptotic branch. The sum stops as soon as
     `|term_k| > |term_{k−1}|` — the smallest-term criterion, standard textbook
     technique for asymptotic expansions. Adding beyond the smallest term both loses
     accuracy and (at FP32) eventually overflows. `erf = 1 − erfc` on that branch.
  3. **Switchover 4.0 → 3.5** (`kTaylorMax`). Reality-first deviation from the
     task-prompt recipe (which asked for a Taylor-overflow probe to bound the
     threshold): the recurrence eliminated the overflow entirely, so overflow is no
     longer the binding constraint. 3.5 is chosen for the convergence-within-cap +
     accuracy crossover — Taylor converges within the iteration cap up to `|z| ~ 4`;
     asymptotic erf (`erf = 1 − tiny erfc`) clears 8.45 digits for `|z| >~ 3`. 3.5
     sits inside both windows with ~0.5 margin; no dip at the boundary
     (`erf(3.5) = 11.59` digits). `eps` 1e-15f → 1e-14f (~ FF's 2^-46 relative
     resolution; finer values can never fire — the B4 lesson carried forward). Third
     instance of the "reality-first check" discipline paying off (precedents: T3.4
     pow §10, B8 unscale-by-`s`, B7 `FloatFloat(double)` constructor).
- **Deviations from the B5 task prompt (all minor, all justified; precedent:
  B8/B7).** (1) **Switchover-threshold derivation** — the prompt asked for a
  Taylor-overflow probe with ~80% margin; the recurrence removed the overflow, so
  the threshold is now derived from convergence/accuracy crossover empirics instead
  (recorded above). (2) **Taylor-branch iteration cap raised 60 → 100** (asymptotic
  cap stays 60): with the recurrence the Taylor loop converges quickly (~10–30
  iterations at `|z| ~ 3`), so the 100 cap is defensive headroom, not a design
  change; documented in the commit. (3) **erfc accidentally lifted RED → GREEN** as a
  downstream side-effect of the erf repair (mean 4.09 → 11.85 via the untouched
  `subtract(1, erf(z))` chain) — anticipated by the prompt ("stay RED or IMPROVE —
  report the erfc delta explicitly, do NOT gate on it either way"); reported per
  instruction.
- **Acceptance-gate summary (all PASS).**
  - **`ff_accuracy_test` erf: mean 4.13 → 13.72** (gate ≥ 8.45, clears by +5.27).
    Spot-checks all finite: `erf(2.0) = 13.98d`, `erf(3.0) = 13.31d`,
    `erf(5.0) = 15.00d` (capped), `erf(7.0) = 15.00d` (saturated, `|z| > 6`).
    Binding gate MET.
  - **`ff_accuracy_test` erfc downstream: mean 4.09 → 11.85** (also clears the 8.45
    gate purely via the untouched `subtract(1, erf(z))` path). erfc **min still
    −0.00** — catastrophic cancellation as `erf(z) → 1` at large `|z|` (the B2
    pattern) persists. **erfc IS NO LONGER GATE-BLOCKING** — B6 (direct large-`|z|`
    erfc) remains worth doing to lift the min, but is no longer required for the
    accuracy gate. `erfc()` body was NOT touched under this task.
  - **No other op regressed.** Pre/post diff of all 50 `ff_accuracy_test` rows: ONLY
    erf and erfc changed. The other 48 rows are digit-for-digit identical.
  - **Full ctest: 22/23 pass, 1737 s. `ff_accuracy_test` flipped RED → GREEN**
    (2 failures → 0). Sole remaining failure is `dd_accuracy_test` (DD erf 24.64 /
    erfc 19.50 / tgamma 14.56 — pending B1/B2/B3), digit-for-digit unchanged
    (`dd_math.hpp` untouched).
  - **All six demos** (`kokkos_ep_demo`, `_complex`, `_ff`, `_ff_complex`, `_qf`,
    `_qf_complex`) build and run **RC 0**. **Zero new warnings** from `ff_math.hpp`
    under `-O3 -Wall -Wextra`.
- **B6 status update (reported, NOT closed).** erfc's mean now clears the 8.45 gate
  via the repaired erf, so B6 is no longer required for the FF accuracy gate. B6
  remains open as a **quality-lift task** (direct large-`|z|` erfc computation would
  lift erfc min from −0.00 to ~8–10 digits). Per the post-B5 sequencing decision,
  **B6 is DEFERRED until after DD-side B1/B2/B3 land** — the DD siblings have
  analogous erfc characteristics, and a unified DD/FF fix pattern may emerge from
  doing DD first. B6's stub stays as-is (no invalidation) and is reordered later in
  the sequence.
- **Follow-ups deferred (not shipped, per scope-out).** (1) A companion PORT_NOTES
  section documenting the DD→FF port artifact (separately-grown `t2`/`t3` → FP32
  overflow) as a general pattern to watch for in future DD→FF ports — drafted in the
  commit body for Reet's ratification, NOT shipped inline. (2) No probe of other
  Taylor/asymptotic series in `ff_math.hpp` (sin/cos/asin/atan/etc.) for the same
  DD-port artifact — per the B4/B8 precedent, hypothesized-but-unverified defects get
  their own tasks with proper failure gates, not silent scope expansion.
- **Post-B5 sequence (REORDERED — B6 deferred).** Remaining FF library-fix stub:
  **B6** (erfc, quality-lift, no longer gate-blocking; DEFERRED until after
  DD-side). Next executable tasks: **DD-side B1 / B2 / B3** — DD tgamma Lanczos, DD
  erfc direct-large-`|z|`, DD erf asymptotic-branch — independent of the FF-side and
  workable in any order among themselves. B3 shares the fix pattern with B5
  (asymptotic-branch repair + Taylor recurrence, adapted from FP32 to FP64
  headroom); B1 shares the pattern with B7 (Lanczos coefficient promotion, but DD
  needs true DD-precision coefficients since `double` doesn't exceed DD's precision
  floor); B2 is direct-large-`|z|` erfc, the template for the eventual B6. **After
  DD-side: B6** as the final FF quality-lift task, informed by the DD-side fixes.
- See `1013cb1` (merged via the B5 merge commit) for the code diff and this DONE block for the outcome.

**B6: FF `erfc` — direct computation for large |z|.**

- **Read first:** `ff_math.hpp:738` (erfc = `subtract(1, erf(z))`); B2 (DD sibling
  — direct-large-|z| pattern); B5 (upstream erf NaN blocker + shared asymptotic
  path); `tests/ff_accuracy_test.cpp` erfc row.
- **Root cause.** `erfc(z) = subtract(1, erf(z))` inherits B5's NaN when erf itself
  returns NaN, plus catastrophic cancellation as erf(z) → 1 (the B2 pattern,
  compounded). Mean 3.91, min 0.00.
- **Fix.** Direct asymptotic/continued-fraction erfc for |z| above a threshold
  (same shape as B2), likely shares the asymptotic-region code path with B5's
  repaired branch. Downstream of B5 — restore erf first, then measure erfc, then
  decide whether the direct erfc path is still needed.
- **Acceptance gate.** `ff_accuracy_test` erfc mean ≥ 8.45; full ctest green; no
  other op regresses.
- **Scope-out.** erfc path only; cross-reference B5.
- **STATUS (resolved).** See DONE block below (commit `243c302`). The numbers and
  line refs above are **pre-fix** and retained as the historical problem
  statement: erfc mean is now **11.97**, and erfc lives at `ff_math.hpp:887-930`
  (`kDirectMin` at 890, `kUnderflowMax` at 899), not the cited `738`. Note the
  **two-stage history** behind the "mean 3.91, min 0.00" figure above: **B5**
  lifted it 3.91 → **11.85** without touching `erfc()` at all — repairing erf's
  NaN removed the inherited-NaN half of the root cause named above — and B6 then
  closed the residual 11.85 → **11.97**. That small row delta understates the
  fix: the real defect was a **cliff, not the DD sibling's shelf**. FF's `erf`
  saturates at |z| = 6.0 and its lo word is only 24 bits, so erfc returned
  literal +0 for every z ≥ 6.02; the 5.24M representable floats in [6.001, 9.0]
  went from a 0.000 exhaustive mean to **13.201**. The row mean barely moves
  because `ff_accuracy_test` samples uniform(-6, 6) and hardly reaches the cliff.
  Also: threshold shipped at **kDirectMin = 5.75** with an upper clamp
  **kUnderflowMax = 10.05** (above which +0 is the only representable answer);
  the stub's "same shape as B2" held, but B2's 6.5 did not transfer, and B2's
  grid-based threshold method actively fails at FP32 — see DEVIATION 2. The
  residual mid-z pointwise band **was not attacked**, per B2's explicit
  instruction not to re-scope. See DONE for full detail; `PORT_NOTES.md` §4i.

**B6: FF erfc direct asymptotic expansion for z >= 5.75. (DONE)**

- Executed 2026-08-07. Task commit `243c302`, merge `8e162cc`.
- **`third_party/include/ff_math.hpp` (edit, +156/-16).** Factored
  `erfc_asymptotic_sum(az, z2)` out of erf()'s asymptotic branch
  (A&S 7.1.23, optimal truncation, returns the raw sum; scaling
  stays at call sites so erf()'s arithmetic is preserved verbatim,
  bit-identical over 14M inputs). `erfc()`: for `z >= kDirectMin =
  5.75`, scale directly by `e^{-z²}/√π` and never form `1 - erf`.
  For `z > kUnderflowMax = 10.05` return +0 (also catches +inf/NaN
  via `!(z.hi <= …)`). Below 5.75 falls through to the original
  `subtract(FloatFloat(1.0f), erf(z))` path; negative z is unchanged
  (erfc(-|z|) ~ 2, subtract is benign). Mirrors B2 (DD sibling) in
  shape.
- **Scaling form: `multiply(sum, e^{-z²})` not `divide(sum, e^{z²})`**.
  Forced by two FP32 failure sites. (a) `multiply()`'s Dekker splitter
  overflows: `multiply(sqrt(pi), exp(z2))` computes `conb = b.hi *
  8193` with `b.hi = e^{z²}`, passing `FLT_MAX/(8193+1) = 4.15e34` at
  z = 8.93 and giving `inf - inf = NaN`. This is the §4d bug class,
  but at multiply()'s splitter, which B8 did NOT scale — only
  divide()'s was fixed. (b) ff `exp()` has a hard `|arg| >= 88` guard
  (FP32's finite range) at z = 9.38, returns 0 AND prints "FFEXP:
  argument too large", so the quotient is 1/0. Measured: divide form
  returns NaN for every z >= 8.95 and prints from z >= 9.38. Multiply
  form has neither failure. Above z² = 88 the guard is cleared by
  quartering the argument and squaring twice (max `|z²/4|` = 25.25 at
  kUnderflowMax); costs ~2 bits of exp's relative error, moot because
  e^{-z²} is already subnormal past 9.38.
- **Threshold derivation (kDirectMin = 5.75).** Same rule as B2:
  place the cut where the direct series stops regressing anything vs
  the fallback at every measured point. Fallback above z = 3.5 is
  erf's 24-bit lo-word channel (per-float mean 7.80 digits over
  [5.4, 6.0], scatter to 13.40). Direct series climbs ~4 digits per
  unit z through that band. Exhaustive per-float enumeration over
  [5.4, 6.001] (1,260,389 inputs) finds exactly 2 regressors, highest
  at z = 5.6338043 by 0.106 digit; kDirectMin = 5.75 sits 0.116 above.
  Over all 7,235,176 representable floats in [5.7, 10.3] NO input
  scores worse than shipped. Rejected alternatives: (i) 4.9, which a
  0.00005-spaced grid says is clean — but exhaustive enumeration finds
  the fallback's per-float lucky spikes that a grid steps over (see
  DEV2). (ii) Minimax cut at 4.0-4.5: mean-optimal (12.36 vs 11.97)
  but regresses 814/6201 measured points by up to 1.93 digits. The
  no-regressions guarantee is worth ~0.40 digit of mean on a row that
  clears its gate by +3.52.
- **Verification.** ff_accuracy_test erfc row: mean 11.85 → 11.97
  (gate 8.45, +3.52). Row min unchanged at -0.00, corpus-structural
  (see DEV1). All 49 other rows digit-for-digit identical vs a
  baseline binary built from main's `ff_math.hpp`. erf row unchanged
  and bit-identical over 14,081,211 inputs (every representable float
  in [3.4, 6.1] ± negation, plus a 1e-4 sweep over [-12, 12] and
  specials). ctest 23/23 PASS. All six demos
  (`kokkos_ep_demo{,_complex,_ff,_ff_complex,_qf,_qf_complex}`) RC 0.
  Zero new `ff_math.hpp` warnings under `-O3 -Wall -Wextra`.

  Exhaustive per-float far-tail lift (quadmath oracle):

     window            floats      pre min / mean     post min / mean
     [5.75,  6.001]    526,386     -0.00 /  7.784     12.539 / 13.502
     [6.001, 9.0]    5,240,784     -0.00 /  0.000      8.197 / 13.201
     [9.0,  10.05]   1,101,006     -0.00 /  0.000     -0.00  /  4.711
     [10.05, 11.0]     996,148     -0.00 /  0.000     -0.00  /  0.000

  Zero NaNs in all four windows. The [6.001, 9.0] window — 5.24M
  floats scoring literal 0 pre-B6, now averaging 13.20 digits — is
  the real lift the row mean hides (the row samples uniform(-6, 6)
  and barely reaches into it).

- **Blast radius.** grep-verified: erfc has no demo consumer and no
  `ff_complex.hpp` consumer, so only `ff_accuracy_test` (only affected
  test) is meaningfully touched.
- **DEVIATION 1 — row min stays -0.00, corpus-structural.** Inherited
  from B2 verbatim: 14 corpus entries (z = 10.5 ×3, 19.5 ×3, 79.5, 80,
  85, 88.7, 88.72, 100.5 ×3) have true erfc between 7.0e-50 and
  ~1e-4389, all below FP32's smallest subnormal 1.4e-45. +0 is the
  only representable answer while the `__float128` oracle still
  resolves them. Six OTHER corpus entries did lift: 2π neighbours
  (6.28318501/48/96) 0 → 13.32/13.32/13.07 digits; 3π neighbours
  (9.42477703/798/894) 0 → 5.79/5.16/5.65. Read the far-tail mean or a
  windowed min over the representable range, not the row min.
- **DEVIATION 2 — grid sampling is the wrong instrument at FP32; the
  no-regressions rule is not free here.** Two findings, both worth
  recording as method-lessons for future FP32 threshold work.
  (i) A 0.00005-spaced grid over [4.85, 6.6] says kDirectMin = 4.9 is
  clean (1 regressing point in 35,001, worth 0.020 digit). It is not:
  exhaustive per-float enumeration finds the fallback's lucky spikes
  the grid steps over, and the true highest regressor at z = 5.6338043
  is 5.3× worse than the grid's worst. B2's 0.0005 grid was adequate
  for DD because DD's fallback had a genuine 53-bit shelf with a
  narrow envelope; FF's 24-bit lo word gives scatter, not a band. At
  FP32 the interesting range is only a few million floats — enumerate
  them. (ii) B2 could apply the pointwise-no-regressions rule at zero
  cost because DD's row mean was flat ±0.03 digit across its candidate
  window. FF's is not: predicted row mean rises monotonically with a
  lower cut (11.84 with no direct path, 11.96 at 5.75, 12.26 at 5.0,
  12.36 at 4.5-4.0 plateau, 12.21 at 3.0). So 5.75 leaves ~0.40 digit
  of mean on the table vs the mean-optimal ~4.0. Trade taken on
  purpose: no-regressions guarantee stands.
- **DEVIATION 3 — divide-overflow attribution was wrong; there is a
  live B8-class site at `multiply()`'s splitter.** The task brief cited
  divide()'s splitter (PORT_NOTES §4d, which B8 fixed with a scaled
  splitter). Actual failing site is `multiply()`'s splitter inside
  `multiply(sqrt(pi), exp(z2))`, which B8 left unscaled. Same 4.15e34
  threshold, same 8193 splitter, different call site. Conclusion (use
  multiply-by-e^{-z²}) is unchanged and correct — only the attribution
  moves. **Backlog item, not fixed here** (multiply()'s splitter is
  out of B6's Rule-4 scope, and multiply-by-e^{-z²} dodges it on the
  shipped path). Any future extended-range special function that forms
  a large FF intermediate and then multiplies is exposed to the same
  bug at multiply()'s splitter. Worth its own scoped task later.
- **DEVIATION 4 — kUnderflowMax is 10.05, not the brief's estimated
  ~10.3.** The brief's `sqrt(ln(1/1.4e-45)) ~ 10.3` omitted the
  `1/(z·√π)` prefactor and round-to-nearest half-ulp. Measured crossing
  is 10.05: oracle `erfc(10.04) = 9.3e-46`, `erfc(10.06) = 6.2e-46`,
  and the series independently first returns exactly 0 at z = 10.04.
  Both derivations agree at 10.05. Trivial correction.
- **Scope held — mid-z pointwise trough NOT attacked.** Documented
  in-source as KNOWN LIMITATION, inherited verbatim from B2 DEV1. On
  a 0.02 grid, erfc scores under the 8.45 gate at scattered points
  from z ~ 2.94 and at every point from z ~ 3.2 to kDirectMin. Direct
  asymptotic cannot help (its optimal-truncation floor is 4.04 digits
  at z = 3.0, at or below the subtract it would replace). One new
  in-source finding worth recording: the global trough (5.459 digits
  at z = 3.5) is NOT cancellation — it is erf's own Taylor→asymptotic
  seam, where the fallback and the direct series return the identical
  value. Closing the band needs a Lentz continued fraction (A&S
  7.1.14) or a triple-float erf.
- **Rule 4.** `third_party/include/ff_math.hpp` is the only file
  touched. `ff_complex.hpp`, `dd_math.hpp`, `dd_complex.hpp`,
  `qf_math.hpp`, `qf_complex.hpp`, and all of `tests/` untouched. No
  tolerance changed; the 8.45 gate was already encoded in
  `ff_accuracy_test`'s table.
- **PORT_NOTES §4i shipped inline this cycle** (see PORT_NOTES.md
  edit in the same commit). Batched-deferral pattern retired with the
  consolidation pass (`abeb4c9` + `8b845ed`); subsequent B-task closes
  ship their §4x inline in the DONE commit.
- **Backlog after this closes.** All library-side gate defects
  closed across FF/DD/QF. Remaining: (a) new B8-class site in
  `multiply()`'s Dekker splitter surfaced by B6 DEV3 — needs its
  own scoped task; (b) optional stale-branch prune (three merged
  task branches b5/b7/b8 with remotes gone are safe to
  `git branch -d`; four long-lived backend branches
  ddfunKokkos/fffunKokkos/CUDAFP128Kokkos/+1 require explicit ask
  per USER.md rule).
- Depends on T1.4 (regression gate), B5 (FF erf asymptotic branch
  this factors from), B2 (DD sibling and pattern template).

**B7: FF `tgamma` — Lanczos coefficients at FF precision. (DONE)**

- **Read first:** `ff_math.hpp:749` (Lanczos coefficients, `c1=676.5203681218851f`,
  …); B1 (DD sibling — same defect, DD-flavored); `tests/ff_accuracy_test.cpp`
  tgamma row; this block.
- **Root cause.** Uniformly ~6 digits (mean 6.10, min 5.94, flat across
  [1e-3, 23)). The Lanczos g=7 coefficients are stored as `float` literals, capping
  the accumulator at FP32's ~7-digit ceiling regardless of the enclosing FF
  arithmetic. Worse than DD's B1 case because FF's mantissa is 24 bits vs DD's 53,
  so the coefficient-truncation penalty is more severe.
- **Fix.** Promote the Lanczos coefficients to FF-precision constants — compute
  once to ~14+ digits and hardcode as `FloatFloat` pairs (analogous to DD's B1 fix
  template, via `FloatFloat::from_bits(hi, lo)` or the FF equivalent constructor).
  Preference: option (a) from B1 (keep the Lanczos structure, promote coefficient
  precision).
- **Acceptance gate.** `ff_accuracy_test` tgamma mean ≥ 8.45; full ctest green; no
  other op regresses.
- **Scope-out.** tgamma only (not lgamma/digamma); no new test file.
- **STATUS (resolved).** See DONE block below (commit `926e056`). The numbers and
  line refs above are **pre-fix** and retained as the historical problem
  statement: tgamma mean is now **12.84** (min **11.42**), not 6.10 / 5.94, and
  tgamma lives at `ff_math.hpp:793-834` with the promoted coefficients at
  ~809-818, not the cited `749`. Root-cause finding on close: the diagnosis above
  was **correct and complete** — the defect really was single-`float` coefficient
  literals — but two details of the prescription were off. (1) The abbreviated
  "`c1=676.5203681218851f`, …" notation undercounted: the Lanczos g=7 body has
  **nine** coefficients (c0–c8) plus the `sqrt(2π)` leading factor, ten constants
  in all, and the leading factor matters because it multiplies the whole result.
  (2) No explicit `from_bits(hi, lo)` split was needed — the existing
  `FloatFloat(double)` constructor at `ff_math.hpp:94` already performs exactly
  that split, so the fix reduced to **deleting the `f` suffixes**; because
  `double`'s 53-bit mantissa exceeds FF's 48, the result is FF-exact. The
  accompanying "rewrite the accumulator arithmetic" step was a no-op: the
  accumulator was already FF. Contrast the DD sibling **B1**, where the analogous
  coefficient promotion is *necessary but not sufficient*. See DONE for full
  detail; consolidated as `PORT_NOTES.md` §4c.

- Executed 2026-08-05. Task commit `926e056` on branch
  `b7-ff-tgamma-lanczos-precision` (now merged to `main` via merge commit `08086c2`
  and the branch deleted); DONE block + docs-pointer this bundle (T3.5/T3.6/B8-style
  two-commit clean-GREEN close). **Second post-Phase-3 library-side bug FIX task**
  (after B8): like B8, licensed by rule 4 to edit the ONE library file the stub
  authorizes.
- **What shipped (1 file, +22/−10 LOC).** `third_party/include/ff_math.hpp` only:
  the `tgamma()` Lanczos-coefficient promotion plus the B7 comment block. **Rule 4
  respected** — the ONE library file the B7 stub authorizes; no other `*_math.hpp` /
  `*_complex.hpp` (and no `dd_math.hpp`, confirming the DD tgamma stays untouched).
- **Fix technique — remove the erroneous `f` suffixes.** The 9 Lanczos g=7
  coefficients (`c0..c8`) AND the `sqrt(2·π)` leading factor (10 constants total)
  were `float` literals. Promoted to `double`, each `FloatFloat(cN)` call binds the
  `FloatFloat(double)` constructor at `ff_math.hpp:94`, which performs the split
  `hi = (float)d; lo = (float)(d − (double)hi)` — a full FF pair (~14 digits). Because
  `double`'s 53-bit mantissa exceeds FF's 48 bits, the split is **FF-exact**
  (per-coefficient FF representation error `< 2^-48`). The Lanczos g=7 structure is
  unchanged; the reflection path uses `FloatFloat_pi()` (already a full two-word FF
  constant, correctly left untouched). Coefficient source: Godfrey g=7 (P. Godfrey
  2001; identical values to `dd_math.hpp`'s DD tgamma, Boost.Math, and Wikipedia
  "Lanczos approximation").
- **Design insight worth recording.** The B7 stub anticipated an explicit
  `from_bits(hi, lo)` split with hardcoded bit patterns. Reality-first inspection
  found the `FloatFloat(double)` constructor already performs that exact split, so the
  fix reduces to removing `f` suffixes — mechanically cleaner, less to review, less
  bit-rot risk. This is the pattern-recognition analogue of B8's "unscale by `s` not
  `1/s`" catch: inspecting existing infrastructure beat the prompt's speculative recipe.
- **Deviations from the B7 task prompt (all minor, all justified; precedent: B8's
  prompt-error catch).** (1) **Coefficient count** — the task prompt cited FIVE
  (`c1..c5`, following the stub's abbreviated "c1=…, …" notation); the actual Lanczos
  g=7 body has NINE (`c0..c8`) plus the `sqrt(2·π)` leading factor. All 10 promoted
  (prompt undercount, not scope drift). (2) **"Rewrite accumulator arithmetic to use
  FF multiply/add"** was a no-op — the accumulator already used FF add/divide/multiply;
  reality was verified and nothing was done there. (3) **`sqrt(2·π)` leading factor**
  promoted alongside the coefficients: it multiplies the whole result, so as a single
  `float` it would have capped the product at ~7 digits (a legitimate companion
  Lanczos parameter, in-scope per the stub's guidance).
- **Base-branch discrepancy (recorded honestly).** The B7 task prompt cited `main`
  tip as `cee94ec` and said it "includes the B8 fix." At prompt-send time `main` was
  actually at `61851e3` (pre-B8-merge) — the B8 branch merge had not yet been
  performed. The fix was correctly branched from actual-main (`61851e3`), correctly
  reasoned independent of B8 (tgamma coefficients vs `divide()` splitter; small
  in-domain tgamma divisors never approach the splitter-overflow band), and the
  discrepancy was flagged in the task commit. The subsequent merge into `main` (merge
  commit `08086c2`) reconciles both fixes in the correct order. Third instance of the
  "reality-first check" discipline paying off (precedents: T3.4 pow §10, B8
  unscale-by-`s`).
- **Acceptance-gate summary (all PASS).**
  - **`ff_accuracy_test` tgamma: mean 6.10 → 12.84** (gate ≥ 8.45, clears by +4.39),
    **min 5.94 → 11.42** (+5.48), n = 1,000,060 (unchanged). Binding gate MET.
  - **No other op regressed.** Pre/post diff of all 50 `ff_accuracy_test` rows: ONLY
    tgamma changed. The other 49 rows — including the deliberately-preserved erf 3.94 /
    erfc 3.91 REDs (B5/B6, pending) and every EXPECTED-MIN-DROP row — are digit-for-digit
    identical.
  - **Full ctest: 21/23 pass, 1757 s.** The 2 failures are exactly the
    deliberately-preserved REDs: `dd_accuracy_test` (DD erf 24.64 / erfc 19.50 / tgamma
    14.56 — DD tgamma UNCHANGED, confirming `dd_math.hpp` untouched) and
    `ff_accuracy_test` (B5/B6 FF erf/erfc; its tgamma row now PASSES).
  - **All six demos** (`kokkos_ep_demo`, `_complex`, `_ff`, `_ff_complex`, `_qf`,
    `_qf_complex`) build and run **RC 0**; tgamma is not in the demo op inventory, so
    the demo gate is build+run only. **Zero new warnings** from `ff_math.hpp` under
    `-O3 -Wall -Wextra`.
- **Follow-ups deferred (not shipped, per scope-out).** (1) A companion PORT_NOTES
  section documenting the DD→FF `f`-suffix port artifact — drafted separately for
  Reet's ratification, NOT inline. The draft notes DD's B1 sibling is DIFFERENT: DD's
  coefficients are already `double`, so DD has ~15-digit precision but needs true
  DD-precision coefficients to reach ~30; FF only needed the `float`→`double`
  promotion because `double` already exceeds FF's precision floor. (2) No probe of
  other Lanczos-adjacent constants in `ff_math.hpp` (e.g. sinh/cosh/asin coefficient
  literals) — per the B4/B8 precedent, hypothesized-but-unverified defects get their
  own tasks with proper failure gates, not silent scope expansion.
- **Post-B7 sequence.** Remaining FF library-fix stubs: **B5** (FF erf — asymptotic
  branch broken + FP32 Taylor overflow; the DD-B3 sibling), then **B6** (FF erfc —
  downstream of B5, direct computation for large `|z|`; the DD-B2 sibling). B5 is the
  next executable task. B1/B2/B3 (DD-side erf/erfc/tgamma) remain independent and can
  be worked in any order relative to the FF set.
- See `926e056` (merged via the B7 merge commit) for the code diff and this DONE block for the outcome.

**B8: FF `divide` — Dekker splitter overflow at large divisors (surfaced by B4). (DONE)**

- **Read first:** `ff_math.hpp:209` (divide, the Dekker splitter with
  `split = 8193.0f`); PORT_NOTES §4a (exp large-input NaN from splitter overflow —
  same bug class, different site); B4 stub above (the surfacing investigation,
  including the instrumentation table and the empirical clean ceiling); T2.3 B3/B9
  domain narrowing (the tests to restore after fix).
- **Root cause.** `ff_math.hpp:209` divide performs `b.hi * split - b.hi` inside its
  Dekker splitting step to extract the high half of the divisor. For
  `|b.hi| > FLT_MAX / 8193 ≈ 4.15e34`, `b.hi * split` overflows to `±∞`; the
  subtraction `∞ − ∞ = NaN` corrupts the split, and the NaN propagates into every
  downstream product / difference / accumulation. Because divide is called by
  `log()`'s Newton iteration with divisor `exp(b) ≈ a`, any `log(x)` for
  `x ≳ e^79.7 ≈ 1.5e34` produces NaN via this path, which then feeds back into
  `exp()` on the next Newton step and hangs the Taylor loop at the 60-iteration cap
  (surfaced by B4).
- **Empirical evidence (from B4's investigation).** Probe over the B3/B9 identity
  tests: `[-79.7, 79.7]` → 0 FFEXP iteration-limit prints; `[-80, 80]` → 891 prints;
  `[-85, 85]` → 15,491 prints. Linear-ish growth with the divide-domain violation
  frequency.
- **Fix options.** (a) **Scaled splitter** — detect `|b.hi| > FLT_MAX / (split + 1)`
  and pre-scale `b` down (divide by a power of 2), do the split, unscale the result.
  Matches PORT_NOTES §4a's shape (§4a scaled the input in exp's final scaling). Small
  diff, preserves the Dekker structure, framework-agnostic. **Preference: (a).**
  (b) Alternative divide algorithm for the overflow-region operands (e.g. long
  division / non-Dekker path). Larger diff; only if (a) fails.
- **Acceptance gate.** `ff_math.hpp:209` divide produces finite outputs (no NaN, no
  inf unless the true quotient overflows FP32) for all in-domain divisors `|b.hi|`
  up to `FLT_MAX` (safety-margined; the exact ceiling is determined by FP32's
  overflow, NOT by 4.15e34). T2.3 `ff_property_test` B3 (log(exp a)≈a) and B9
  (exp(a)·exp(-a)≈1) domains restored from `[-69, 69]` to `[-79, 79]`
  (safety-margined one integer under 79.7). **Zero `FFEXP: iteration limit` prints**
  across the restored B3/B9 range. Full ctest green — the 2 deliberately-preserved
  REDs (`dd_accuracy_test`, `ff_accuracy_test`) stay digit-for-digit unchanged
  EXCEPT `ff_accuracy_test`'s exp / log rows may IMPROVE (report delta, do not gate
  on it). All four demos build and run RC 0.
- **Scope-out.** `ff_math.hpp:209` divide only. Do NOT touch other splitter-using
  primitives (multiply, sqr, two_prod, etc.) unless the same bug is empirically
  demonstrated there under a probe; per the B4 precedent, hypothesized-but-unverified
  bugs get their own tasks, not silent scope expansion. Do NOT touch `qf_math.hpp` /
  `dd_math.hpp` / any `*_complex.hpp`. Do NOT modify PORT_NOTES §4a (this is a NEW
  site, same class — reference §4a, don't rewrite it). If a new PORT_NOTES section is
  warranted (documenting the divide-site fix as a companion to §4a), draft it
  separately for Reet's ratification, don't ship inline.
- **Cross-references.** Same bug class as PORT_NOTES §4a; may share a fix pattern
  (scaled input) with §4a's exp final scaling. Not related to B1/B2/B3 (DD-side,
  transcendental accuracy). Independent of B5/B6/B7 (FF erf/erfc/tgamma) — those can
  be worked in any order relative to B8.

- Executed 2026-08-05. Task commit `b2cff7d` on branch
  `b8-ff-divide-splitter-overflow` (**NOT merged** — Reet decides merge posture
  separately); DONE block + docs-pointer this bundle (T3.5/T3.6-style two-commit
  clean-GREEN close). **First library-side bug FIX task** (B1–B7 are still open
  stubs): unlike the Txx *validation* tasks, B8 is licensed by rule 4 to edit the
  library — it is the "separate task with clear scope" that fixes what T2.3's B4
  investigation surfaced.
- **What shipped (3 files, +59/−33 LOC).** `third_party/include/ff_math.hpp`
  (+23/−1): the `divide()` scaled-splitter fix (fix option (a) from the stub) plus a
  ~16-line comment block. `tests/ff_property_test.cpp` (+27/−26): B3/B9 domain
  restoration + rationale-block rewrite. `tests/README.md` (+9/−6): B3/B9 registry
  text updated with the B4→B8 root-cause history. **`ff_math.hpp` is the sole library
  file touched** (rule 4 respected — no `dd_math.hpp` / `qf_math.hpp` / any
  `*_complex.hpp`).
- **Fix technique — scaled splitter (mirrors PORT_NOTES §4a).** When
  `|b.hi| > FLT_MAX / (split + 1) ≈ 4.15e34` (the overflow-hazard band), pre-scale the
  divisor down by the exact power of two `s = 2^-64` (power-of-2 multiplication does
  not round, so `b`'s full FF precision is preserved), run the UNCHANGED Dekker split,
  then unscale the quotient. `2^-64` gives ~15 orders of headroom: the largest
  `|b.hi| ≈ FLT_MAX = 2^128` maps to `2^64`, and `2^64 · split ≈ 2^77 ≪ FLT_MAX`, so
  neither the divisor split nor the internal quotient's split can overflow. The
  splitter constant (`8193.0f`) and the divide algorithm are otherwise untouched, and
  **the non-overflow path (`s = 1.0f`) is bit-identical to the prior code** (verified:
  0 mismatches over a dense `[1e-10, 1e34]` sweep) — same bug class as §4a, different
  site (divide's splitter, not exp's final scaling).
- **Prompt-error catch (justified deviation).** The task prompt's step 5 said
  "unscale by `1/s`". That is mathematically wrong: `q = a/(b·s) = (a/b)/s`, so
  recovering `a/b` requires **multiplying** `q` by `s`, not by `1/s` (unscaling by
  `1/s` would yield `(a/b)/s²`). The shipped code multiplies by `s`, confirmed correct
  by the direct probe (correct quotients to FF precision at `b.hi` up to `FLT_MAX`);
  the deviation is flagged in the commit message. All other prompt steps followed as
  written — prompt-error hygiene, not scope drift (precedent: T3.4 pow 30-ulp, T3.2
  Priest→Shewchuk-weak).
- **Test-domain restoration.** `ff_property_test` B3 (`log(exp a)≈a`) and B9
  (`exp(a)·exp(−a)≈1`) restored `[-69,69]` → `[-79,79]` — one integer under the ~79.7
  empirical clean ceiling, NOT pushed to the `[-85,85]` the B4 stub originally
  anticipated (the honest ceiling is `79.7`, so `79` is the safety-margined restore).
  The header rationale block and README B3/B9 registry text now carry the B4→B8
  root-cause history in place of the old B4-narrowing note.
- **Acceptance-gate summary (all PASS).**
  - **Direct probe** (shipped `ff_math.hpp`, `|b.hi|` swept `1e30 → FLT_MAX`):
    post-fix **0 NaN / 0 inf everywhere**; pre-fix **392 + 782 NaN** on the same
    sweep. Endpoints `b.hi = FLT_MAX` and `b.hi = 5e34` finite and correct post-fix
    (NaN pre-fix).
  - **`ff_property_test` B3/B9 on `[-79,79]`: PASS, ZERO `FFEXP: iteration limit`
    prints** across the whole run (`grep -c FFEXP = 0`) — the stub's binding gate.
  - **Full ctest: 21/23 pass, 139.70 s.** The 2 failures are exactly the
    deliberately-preserved REDs (`dd_accuracy_test` and `ff_accuracy_test` on
    erf/erfc/tgamma → B1/B2/B3, B5/B6/B7), **digit-for-digit unchanged**.
  - **`ff_accuracy_test` exp/log rows unchanged** pre vs post (exp mean 13.20, log
    mean 14.00): the random accuracy domain never reaches the `>79.7` overflow region,
    so no delta — expected, and per the stub the exp/log improvement was reported, not
    gated.
  - **All four demos** (`kokkos_ep_demo` DD, `_ff`, `_qf`, `_qf_complex`) build and
    run **RC 0**; **zero new warnings** from `ff_math.hpp` under the build's `-O3`
    flags.
- **Rule 4 respected.** `ff_math.hpp` is the ONE library file edited (the B-task
  fix mandate); no other `*_math.hpp` / `*_complex.hpp` touched. **PORT_NOTES §4a left
  unmodified** — B8 is a NEW site of the same class, so it references §4a rather than
  rewriting it.
- **Follow-ups deferred (not shipped, per scope-out).** (1) A PORT_NOTES §4c-style
  companion note documenting the divide-site fix alongside §4a — drafted separately
  for Reet's ratification, NOT inline (per the stub's "if a new PORT_NOTES section is
  warranted, draft it separately"). (2) The other splitter-using primitives
  (`multiply`, `sqr`, `two_prod`) were NOT touched — per the B4 precedent,
  hypothesized-but-unverified overflow in those sites gets its own probe + B-task, not
  silent scope expansion.
- **Post-B8 sequence.** Remaining FF library-fix stubs in suggested order: **B7**
  (tgamma FP32 Lanczos, self-contained), then **B5** (erf, the asymptotic-branch
  blocker), then **B6** (erfc, downstream of B5). B1/B2/B3 (DD-side erf/erfc/tgamma)
  are independent and can be worked in any order relative to the FF set.
- See `b2cff7d` (branch `b8-ff-divide-splitter-overflow`) for the code diff and this DONE block for the outcome.

**B9: FF Dekker-splitter guard for multiply() and siblings. (DONE)**

- Executed 2026-08-07. Task commit `1767cf9`, merge `191bfff`.
- **Origin.** Surfaced by B6 (commit `243c302`) as DEV3: after B8 scaled
  divide()'s Dekker splitter, multiply() and its siblings still had the same
  overflow at 4.15e34. B6 dodged the exposure in its shipped path by rewriting
  `divide(sum, multiply(sqrt(pi), exp(z²)))` as `multiply(sum, exp(-z²))/sqrt(pi)`,
  so no shipped path today hits multiply()'s splitter — but the latent trap
  remained for any future extended-range special function that forms a large FF
  intermediate and multiplies it. B9 closes it.
- **`third_party/include/ff_math.hpp` (edit, +101/-5).** Four splitter sites
  reviewed and guarded (see SIBLING AUDIT below). Fix pattern is B8's verbatim:
  when an operand's `|.hi|` is in the overflow-hazard band
  (`> kSplitOverflowThresh = 4.1528233e34f`, the constant B8 already defined at
  line 227), pre-scale the offending operand(s) by an exact power of two (2⁻⁶⁴,
  chosen for ~15 orders of headroom — `2⁶⁴ * 8193 ≈ 2⁷⁷ ≪ FLT_MAX`), run the
  UNCHANGED Dekker split, then unscale the product by the exact compensating
  factor. Non-overflow paths (all operands safe) are bit-identical to the current
  code, verified at 20M pairs per site.
- **SIBLING AUDIT** — all four splitter sites addressed:
  - **multiply() ~line 194** — FIXED. B6 erfc-shape reachability demonstrated
    (multiply(1e35f, 3.7e-6f) → -nan pre-B9, bit-exact vs `__float128` post-B9).
  - **multiply_scalar() ~line 255** — FIXED. Public `operator*`; in-header scalar
    callers pass small constants but the FF side is unconstrained by caller
    context.
  - **divide_scalar() ~line 271** — FIXED, divisor half only. Scaling `b` inflates
    the intermediate `t1` by 2⁶⁴ but stays 11 orders under the hazard band, so
    B8's-style divisor scaling suffices. Quotient-scaling gap identical to
    divide() — see DEV1 below.
  - **two_prod() ~line 289** — FIXED, though currently unreachable in-header.
    `sqrt()` is its only caller and passes `≤1.9e19`. Guarded anyway because
    two_prod is a public exactness primitive on the FF API surface.
- **Design decision — no shared helper.** The 3–4 line scaled-splitter dance is
  inline at each of the four sites; duplication is cheaper than a helper's
  function-call boundary and return-tuple API at a hot arithmetic primitive.
  Explicitly agreed with FIX SHAPE point 3.
- **Verification.**
  - Bit-identity: 20,000,576 pairs per site, single-binary A/B against verbatim
    pre-B9 bodies — **0 mismatches on all four sites** in the non-overflow domain.
    Load-bearing acceptance claim; holds.
  - ff_accuracy_test all 50 rows, min and mean: **zero delta** (largest |Δ|
    0.0000), statuses identical, both runs ALL PASSED.
  - Positive test: `multiply(1e35f, 3.7e-6f)` returns `-nan` on main, bit-exact vs
    `__float128` on B9. two_prod exactness in the hazard band: ~5.7M products go
    100% NaN → 100% bit-exact.
  - Full ctest: 23/23 PASS (was 23/23 pre-B9). All six demos
    (`kokkos_ep_demo{,_complex,_ff,_ff_complex,_qf,_qf_complex}`) RC 0. Zero new
    `ff_math.hpp` warnings under `-O3 -Wall -Wextra`.
- **DEVIATION 1 — divide() is only half-fixed; new B10 backlog task.** Discovered
  mid-work: B8 scaled divide()'s divisor, but divide() also splits the quotient
  estimate `s1 = a.hi/b.hi`, which is still unguarded. `divide(1e35f, 1.0f)`
  returns NaN pre-B9 AND post-B9 — no large operand anywhere, just a large
  quotient. Not patched here per report-and-stop directive: needs numerator
  scaling, interacts with B8's existing divisor scale, and an overflowing quotient
  is unrecoverable regardless. divide_scalar carries the identical gap.
  **Deferred to B10** as one unified design pass covering both divide() and
  divide_scalar()'s quotient-scaling exposure. Documented at the divide() site
  in-source.
- **DEVIATION 2 — both-operands-in-band corner is unchanged (self-caught,
  corrected pre-commit).** multiply() already returned NaN for any overflowing
  product independent of the splitter (`multiply(1e30, 1e30)` was NaN on main too
  — product overflow, not splitter overflow). That corner now returns `(inf, -inf)`
  instead of NaN. First in-code comment overclaimed this as a fix; corrected
  before commit. Not a defect, just scope precision — B9 addresses
  splitter-overflow-inside-a-finite-product, not the pre-existing
  finite-inputs-overflowing-product case.
- **DEVIATION 3 — no shared helper**, agreed with FIX SHAPE point 3. (Not really a
  deviation, restated for the record.)
- **Prompt-error corrections cluster-Claude flagged:**
  - The brief called splitter site ~289 the "pow chain"; it is actually
    `two_prod`. `pow_int` reaches the splitter only via `multiply()`. Corrected in
    the DONE block above.
  - Sweep methodology gotcha: a uniform draw over exponents ≤2¹¹⁴ leaks past the
    4.1528e34 threshold (top bucket reaches 4.1538e34). First bit-identity sweep
    reported spurious mismatches from exactly that; fixed by rejection-sampling on
    the value, not the exponent. Method-lesson worth recording in §4j.
- **Codegen artifact (not a semantic issue).** Two separately-compiled binaries
  A/B'd showed 44 (multiply, multiply_scalar, two_prod combined) + 109 (divide)
  divergences, all quiet-NaN sign bit only. Divide showed 109 despite being a
  comment-only edit, which is the clincher that this is compiler codegen around
  NaN sign bits, not semantics. At `-O0` everything matches. Recorded for the
  record; no action.
- **Header comment lines 11-15 lightly updated** to name the scaled-splitter
  guards, per the brief's judgment-call sanction.
- **tests/ff_eft_test.cpp:16, 127 cite drifted line numbers** (comments only,
  Rule-4 out of scope, flagged not touched). Fold into next housekeeping pass.
- **Device-path caveat.** Kokkos install is Serial-only (KOKKOS_ENABLE_CUDA
  undefined); all verification is host-path. Fix adds no new construct —
  `ldexpf` / `Kokkos::fabs` already appear in this header from B8 — but the device
  path is unexercised here, as it was for B8.
- **Rule 4.** `third_party/include/ff_math.hpp` is the only file touched.
  `ff_complex.hpp`, `dd_math.hpp`, `dd_complex.hpp`, `qf_math.hpp`,
  `qf_complex.hpp`, and all of `tests/` untouched. No tolerance changed; no
  accuracy gate touched.
- **Backlog after this closes.** (a) **B10** — divide() and divide_scalar()
  quotient-scaling gap (DEV1), one unified design pass. (b) Optional housekeeping:
  ff_eft_test.cpp line-ref drift.
- Depends on B8 (divide splitter scaled), B6 (surfaced DEV3), B2
  (identity-is-the-bug lens).

**B10: FF divide() + divide_scalar() quotient-splitter guard. (DONE)**

- Executed 2026-08-08. Task commit `3ca298e`, merge `febf46b`.
- **Origin.** Surfaced by B9 (commit `1767cf9`) as DEV1: after B8 scaled
  divide()'s DIVISOR splitter, divide() still had a second unguarded splitter on
  the QUOTIENT ESTIMATE `s1 = a.hi / b.hi`, which could land in the 4.15e34
  hazard band with BOTH operands far below it. `divide(1e35f, 1.0f)` returned NaN
  with no large operand anywhere. divide_scalar carried the identical gap. B9
  deferred rather than widening; B10 closes both sites in one unified design pass.
- **`third_party/include/ff_math.hpp` (edit, +83/-21, 18 code lines).** Three-zone
  classification on `s1` (the quotient estimate), applied identically at divide()
  and divide_scalar():
  - **Zone A** — `|s1| ≤ kSplitOverflowThresh`. Safe. Untouched, bit-identical to
    pre-B10 (verified at 20M pairs).
  - **Zone B** — `kSplitOverflowThresh < |s1| < ∞`. Splitter would overflow but
    the true quotient IS representable. Pre-scale the NUMERATOR down by `2^-32`,
    run the unchanged body, unscale the quotient by `2^32`. Was NaN; now correct.
  - **Zone C** — `|s1| == ∞`. True quotient overflows FP32; no scaling recovers a
    finite answer. Return correctly-signed ±inf instead of NaN. Deliberate
    SEMANTIC upgrade (honest overflow signal instead of NaN poison), NOT a
    bit-identical fix — unlike B8/B9, B10 cannot promise "correct everywhere in
    the band" because part of the band has no representable answer at all.
- **Design decisions worth noting explicitly:**
  - **Scale factor 2^-32 (not B8/B9's 2^-64).** B8/B9's scale lands on operands
    that are huge by definition; B10's lands on the NUMERATOR, which in Zone B can
    be as small as `|b.hi| · 4.15e34 ≈ 2^-34` (b.hi subnormal). 2^-32 keeps ~19
    binades of splitter headroom AND ~36 binades of underflow margin
    (a.lo ≥ 2^-90 in the worst corner, clear of 2^-126). 2^-64 would give 51
    binades of splitter headroom but under 4 of underflow — would flush a.lo
    subnormal.
  - **Scale the NUMERATOR, not the divisor.** The divisor is B8's site; `s1` is
    what has to shrink. Multiplying `a` by an exact power of two scales s1, every
    Dekker intermediate, and the final (hi, lo) by exactly that factor. Recovery
    via two independent exact factors (`s` and `1/sn`), never as one reciprocal —
    same sign trap B8/B9 flagged.
  - **Zone classification on `s1` itself is overflow-safe by construction.** IEEE
    division never traps, delivers correctly-signed ±inf exactly when the quotient
    exceeds FLT_MAX, and yields NaN only for 0/0 and inf/inf — both of which fail
    `fabs(s1) > thresh` and fall through to the unchanged body (where they
    produced NaN before and still do). No `|a.hi|` vs `|b.hi| * thresh` product is
    needed.
  - **B8 and B10 guards are mutually exclusive** (see DEV3). B8 firing forces
    `|a.hi/b.hi| ≤ split+1 = 8194`, so post-B8-scale `|s1| ≤ 1.5e23`, eleven
    orders below the B10 threshold. B10 can never fire on top of B8. Unscale is
    still written as two independent factors so it would compose correctly if that
    ever changed.
  - **No shared helper across the two sites** (per B9's rule established for the
    multiply family). Duplication is cheaper than the abstraction at a hot
    arithmetic primitive.
- **Verification.**
  - Zone A bit-identical: 20,000,000 pairs at each site, single-binary A/B vs
    verbatim pre-B10 bodies. Rejection-sampled on the RATIO value per B9's
    grid-vs-value lesson (3,444,127 draws rejected). Plus 7,020,506 sliver cases
    where pre-B10 was finite. **0 mismatches at both sites.**
  - Zone B correctness: `divide(1e35f, 1.0f)` and `divide_scalar(1e35f, 2.0f)`
    bit-exact vs `__float128`; `divide(1e30f, 1e-5f)` 16.81 digits. All three were
    NaN pre-B10.
  - Zone C: 8 cases (4 sign combos × 2 sites) return correctly-signed ±inf, 0 NaN.
  - B8-interaction check: `divide(1e37f, 1e35f)` returns 15.28 digits,
    bit-identical to pre-B10 (confirms DEV3 — guards do not compose in practice).
  - ff_accuracy_test all 50 rows: **49 byte-identical**. The divide row's printed
    MIN moves 0.00 → -0.00; mean, n, skipped, tolerance, status all unchanged
    (14.00 / 1000063 / 4 / 8.45 / PASS). This is a **sort-tie-break artifact, not
    a numeric change** — see KNOWN QUIRK below.
  - Full ctest: 23/23 PASS (was 23/23 pre-B10). All six demos
    (`kokkos_ep_demo{,_complex,_ff,_ff_complex,_qf,_qf_complex}`) RC 0. Zero new
    `ff_math.hpp` warnings under `-O3 -Wall -Wextra`.
  - Downstream log() sweep: byte-identical pre/post at `-O0` across
    x ∈ {1e-30 … 3.4e38} including B8's motivating x ≳ e^79.7 case. Single -O3
    delta is a NaN sign bit at x = 3.4e38 (the §4j codegen artifact, confirmed by
    `-O0` agreement). 200k-point sweep: mean 14.86, min 10.11, 0 non-finite.
- **KNOWN QUIRK, non-numeric.** The divide row's printed MIN moving 0.00 → -0.00
  is not a regression, and future diffs of ff_accuracy_test should not treat it as
  one. `compute_stats` does `std::sort + v.front()`; +0 and -0 compare equal, so
  which zero surfaces depends on the multiset. Per-element scoring of the
  huge_tiny corpus pairs shows why the multiset changed:

      a, b              ratio        pre         post
      1.5e18, 1.5e-18   1e36 (B)     NaN, 0.00   finite, 14.00  ← LIFT
      1.5e24, 1.5e-24   1e48 (C)     NaN, 0.00   +inf,   0.00
      1.5e30, 1.5e-30   1e60 (C)     NaN, 0.00   +inf,   0.00

  Improved 1, worse 0, unchanged 9. The `+0.0` count dropped 5→4, and the
  tie-break landed on a pre-existing `-0.0`. The Zone B pair lifted; the Zone C
  pairs correctly took the ±inf-overflow path and still score 0 digits (as they
  must — the true quotient is unrepresentable in FP32). One corpus element in
  1,000,063 lifting is too small to move the printed mean.
- **DEVIATION 1 — Zone B scale is 2^-32, not the brief's suggested B8/B9-analogous
  2^-64.** Rationale above under Design decisions. Documented in-source.
- **DEVIATION 3 — B8 and B10 guards are mutually exclusive; the brief's
  "composition scenario" cannot occur.** Rationale above. The brief's FIX SHAPE
  point 5 assumed the two could fire together and required working out the
  combined scale; in practice they can't, so no composition math is needed.
  Cluster-Claude wrote the unscale as two independent factors anyway
  (belt-and-braces). Documented in-source.
- **DEVIATION 4 — Zone C incidentally makes 1/0, inf/1, 1/FLT_MIN, 1/denorm_min
  IEEE-correct** (return ±inf where pre-B10 returned NaN). 0/0 and inf/inf
  correctly stay NaN. Not aimed at, no test gate, just a consequence of returning
  correctly-signed ±inf whenever `|s1| == inf`.
- **DEVIATION 6 — B10 reaches WIDER than the splitter it was scoped to fix.** In
  Zone B, when `|a.hi|` is within a rounding step of FLT_MAX, it is the Dekker
  term `a1*b1` that overflows — not `cona`. 62 such cases appeared in the sliver
  sweep, all NaN before, all correct now. Discovered because cluster-Claude's
  first sliver pass produced 18 apparent mismatches; investigated instead of
  dismissed. This means B10 quietly closes a superset of the reported gap;
  positive scope surprise, no defect. Documented at the site.
- **DEVIATION 8 — device path unexercised.** Kokkos install is Serial-only;
  `Kokkos::isinf` is device-callable by inspection but not by test. Same caveat as
  B6/B8/B9.
- **NEW BACKLOG ITEM, found not fixed** — `divide(1.0f, inf)` returns NaN where
  IEEE says +0. Root cause is B8's divisor guard (`conb = inf*split = inf`), NOT
  B10's territory — §4d/§B8 direct extension. Correctly declined to widen B10 to
  fix a divisor-side issue; recorded here as follow-up work. Not urgent — no
  shipped path today hits `divide(x, inf)`.
- **Rule 4.** `third_party/include/ff_math.hpp` is the only file touched.
  `ff_complex.hpp`, `dd_math.hpp`, `dd_complex.hpp`, `qf_math.hpp`,
  `qf_complex.hpp`, and all of `tests/` untouched. No tolerance changed; no
  accuracy gate touched.
- **Backlog after this closes.** (a) `divide(x, inf)` IEEE-correctness (see above;
  §4d/§B8 territory). (b) Optional housekeeping: `tests/ff_eft_test.cpp:16, 127`
  cite drifted line numbers, comments only, Rule-4 out of scope but folds into any
  future housekeeping pass.
- Depends on B8 (divisor splitter scaled — mutually-exclusive guard, referenced),
  B9 (surfaced DEV1, established the no-shared-helper rule and the
  rejection-sampling method lesson), B6 (identity-is-the-bug lens; established
  inline §4x shipping post-consolidation).

**B11: FF divide() + divide_scalar() non-finite-divisor precondition. (DONE)**

- Executed 2026-08-09. Task commit `6fdf719`; merged to `main` by FAST-FORWARD, so
  there is no separate merge SHA — `6fdf719` is both the task commit and main's
  tip at close. (B9/B10 used `--no-ff` merge commits; this one did not, per the
  merge brief's "fast-forward if possible". Noted so the pattern break is on the
  record and nobody hunts for a missing merge commit.)
- **Origin.** Filed by B10 (commit `3ca298e`) as its "NEW BACKLOG ITEM, found not
  fixed" (:2046-2050) and in PORT_NOTES §4k "Related backlog". B10 correctly
  declined to widen: the defect is divisor-side (§4d/§B8 territory), and B10's
  design pass was quotient-side.
- **Root cause — B8's own guard, firing on an input it was never designed for.**
  The divisor guard classifies on `|b.hi| > kSplitOverflowThresh`, which is TRUE
  for `b.hi = ±inf`, so it scales the divisor by `2^-64` — and `inf * 2^-64` is
  still `inf`. The unchanged Dekker split then computes `conb = inf * split = inf`
  and `b1 = conb - (conb - b.hi) = inf - inf = NaN`. `divide(1.0f, inf)` returned
  NaN where IEEE 754-2019 §6.1 requires `+0`.
  Worth recording: **the quotient path was already correct.** `s1 = a.hi / b.hi`
  evaluates to the right `±0`; it just gets discarded by the NaN coming out of the
  divisor's split. Nothing needed to be computed differently — only reached
  differently.
- **`third_party/include/ff_math.hpp` (edit, +28/-0, 2 code lines).** One
  precondition at the top of each of `divide()` and `divide_scalar()`, ahead of
  every splitter by construction:

      if (!Kokkos::isfinite(b.hi)) return FloatFloat(a.hi / b.hi, 0.0f);

  The bare float quotient IS the whole answer for every non-finite divisor:
  `finite/±inf` gives a correctly-signed zero (including `±0/±inf`, where IEEE
  preserves the numerator's zero sign), `±inf/±inf` gives NaN, and a NaN divisor
  propagates. `lo = 0.0f` because the result is exact — B10's Zone C convention.
- **Design decisions worth noting explicitly:**
  - **Return the quotient, NOT a `copysign` form.** The brief suggested
    `copysign(0.0f, a.hi / b.hi)`. That is wrong: it flattens the invalid
    `inf/inf` case to a signed zero instead of leaving it NaN, silently breaking
    a case the same brief separately asked to preserve. Returning the quotient
    directly yields all four sign combinations AND both invalid cases from one
    IEEE operation, with no case analysis to get wrong. See DEV1.
  - **Divide-by-ZERO deliberately untouched.** `b.hi = 0` is finite, falls
    through the guard, and keeps B10's Zone C semantics (`±inf` for a nonzero
    numerator, NaN for `0/0`). The two questions look adjacent and are not: Zone C
    is about a quotient too large to represent; B11 is about a divisor that breaks
    the splitter. Verified by regression rows, not by inspection alone.
  - **Precondition, not another zone.** B8/B9/B10 all scale-and-unscale. B11 does
    not, because there is nothing to recover — the answer is exact and needs no FF
    arithmetic. Cheapest correct fix, and it cannot perturb any finite path.
- **Verification.**
  - Finite-divisor bit-identity: **20,000,625 pairs per function, 0 mismatches**
    at both sites, single-binary A/B vs verbatim pre-B11 bodies (B9/B10
    methodology — one TU, so both sides share inlining context and FP flags).
    Sampling spans the FULL finite exponent range with rejection on non-finite, so
    it covers Zones A, B and C — strictly more than the "Zone A" the brief asked
    for — plus a 24×24 edge grid (±0, ±FLT_MIN, ±denorm_min, ±FLT_MAX, both sides
    of the 4.1528233e34 threshold, 8193, 1e35).
  - **Harness validity (positive control).** "0 mismatches" is worthless from a
    comparator that cannot see a difference, so the same comparator was pointed at
    non-finite divisors, where pre/post MUST differ: 36 compared, 36 mismatches.
    Recorded because prior B-tasks asserted bit-identity without this check.
  - IEEE cases, pre → post, both functions: `divide(±1, +inf)` NaN → ±0;
    `divide(±1, -inf)` NaN → ∓0; `divide(±0, ±inf)` NaN → ±0 with the numerator's
    zero sign preserved; subnormal / FLT_MIN / 1e-30 / 1e30 / 1e35 / FLT_MAX over
    ±inf all NaN → correctly-signed 0; `divide(±inf, ±inf)` NaN → NaN (unchanged,
    correct); NaN divisor NaN → NaN (unchanged, correct); divide-by-zero ±inf →
    ±inf (Zone C, unchanged). No case raised an exception.
  - **New test is RED pre-fix, GREEN post-fix.** Built against pre-B11
    `ff_math.hpp` it fails **34 of 62** checks (every inf-divisor row, both
    functions) and exits RC=1; against the fixed header, 62/62 PASS, RC=0. The 28
    NaN-divisor and divide-by-zero rows pass in BOTH builds, which is what
    confirms the blast radius is confined to inf divisors.
  - ff_accuracy_test: **byte-identical** pre vs post (all rows, min/mean/status —
    a clean `diff`, unlike B10, which had the -0.00 tie-break quirk).
  - **No measurable cost** despite sitting in a hot primitive: ff_accuracy_test at
    `-O3` runs 138.16s pre vs 137.75s post (0.3% faster, i.e. noise), reproducible
    across repeat runs.
  - Full ctest: **23/23 PASS, RC=0** in both trees — `build/` (no `-O3`) 1613.93s,
    `build_opt/` (`-O3 -DNDEBUG`) 630.01s. All six demos RC 0, zero NaN mentions.
  - DD/QF insulated by construction — re-confirmed that `dd_math.hpp`,
    `qf_math.hpp` and `qf_complex.hpp` name `ff_math.hpp` only in COMMENTS and
    `#include` no FF header (the careless-grep trap §4j warned about).
- **DEVIATION 1 — the brief's suggested fix form was wrong, and was not used.**
  `copysign(0.0f, a.hi / b.hi)` breaks `inf/inf`. Rationale above. Reported rather
  than silently substituted.
- **DEVIATION 2 — a test file WAS touched, unlike every prior B-task.** B1–B10 all
  close with "all of `tests/` untouched". B11 extends
  `tests/ff_property_test.cpp` with **Group S**, an explicit IEEE expectation
  table for divide/divide_scalar (31 cases × 2 functions = 62 checks), gated by
  `KOKKOS_EP_ASSERT` alongside Group A and running unconditionally (no
  `__float128` needed — this is exact-semantics checking, not accuracy-vs-oracle).
  Sanctioned by the task brief, which preferred extending an existing test over
  adding a file. No tolerance and no accuracy gate changed.
  Group S is a **table, not a computed reference**: the point is to encode what
  IEEE mandates independently, rather than recompute the quotient with the same
  primitive under test.
- **Why this survived B8, B9 AND B10 — the coverage hole is the real finding.**
  Group A's corpus sets `include_zero = true, include_inf = false, include_nan =
  false`, and `ff_invariant_test` SKIPS non-finite `hi` by construction (its
  non-overlap invariant is undefined there). Between them, **no FF test had ever
  pinned down what division does with an infinite divisor.** Three consecutive
  splitter tasks worked in that blind spot. Worth remembering when the next
  primitive gets a guard: check whether any test can see the guarded input class
  at all.
- **DEVIATION 3 — the merge brief's "item 2" did not exist; nothing was landed for
  it.** The brief described a "B9 DEV3 multiply-splitter followup". B9's DEVIATION
  3 is "no shared helper" (:1903-1904), which carries no deferred work. The
  multiply-splitter-in-a-special-function-chain item is **B6 DEV3** (:1551-1562,
  backlog at :1589-1592), and **B9 is the task that closed it** — B9's commit
  message closes that backlog entry by name. All five Dekker-splitter sites in
  `ff_math.hpp` are guarded today (grepped, not assumed): multiply (both
  operands), multiply_scalar, two_prod, divide (divisor B8 / quotient B10),
  divide_scalar (divisor B9 / quotient B10). Reported and stopped rather than
  inventing a defect to fix.
- **Rule 4.** `third_party/include/ff_math.hpp` and — by explicit sanction —
  `tests/ff_property_test.cpp`. `ff_complex.hpp`, `dd_math.hpp`, `dd_complex.hpp`,
  `qf_math.hpp`, `qf_complex.hpp` untouched. No tolerance changed; no accuracy
  gate touched. PORT_NOTES §4l shipped inline in the task commit.
- **Backlog after this closes.** Both remaining items are cosmetic and were closed
  in the post-B11 cleanup pass (see below), not by a B-task: (a)
  `tests/ff_eft_test.cpp:16, 127` line-ref drift — flagged by B9, re-filed by B10,
  moved again by B11; (b) a stale parenthetical in `ff_math.hpp`'s erfc block
  claiming multiply's splitter was never scaled, which B9 falsified. **No library
  defect remains open across FF, DD or QF.**
- Depends on B8 (the guard whose ±inf blind spot this closes), B10 (filed the
  item; its Zone C convention for `lo` and its untouched divide-by-zero path are
  both reused), B9 (single-binary A/B bit-identity methodology).
- **NUMBERING NOTE — "B11" is overloaded, and this is the second time.** This
  block is B11 in the **library-fix B-series** (B1…B11). There is an unrelated
  **B11 in T2.3's property-test identity numbering** — `pow(a,2) ≈ a·a`, at
  :2515 and :3349. Different namespace, same digit, no relation. B9 hit exactly
  this collision with its own digit (`exp(a)·exp(−a) ≈ 1`, :583) and recorded the
  warning only in its commit message, where it was easy to miss — so it is on the
  page this time. The two series are not going to stop colliding; read the
  surrounding section, not the digit.
- Cross-ref: PORT_NOTES.md §4l.

### Phase 2 — FF validation (6 tasks)

FF library is already implemented on `fffunKokkos`. Phase 2 = validate
+ merge, not implement.

**T2.0: Merge FF backend into main behind Kokkos::Experimental namespace. (DONE)**

- Executed 2026-08-02. Commit `dad92ef` (task); DONE block + docs-pointer `7441670`.
- Brought the mechanically-DD→FF-translated float-float library from
  `fffunKokkos` onto `main`, behind the same `Kokkos::Experimental` namespace
  and STL-style API the DD backend adopted in T0.4. Non-trivial because the FF
  files predate every Phase-0 convention (T0.0 oracle plumbing, T0.3 complex
  oracle, T0.4 namespace/rename, T0.5 license posture) and had to be brought
  into conformance with all four without touching a single arithmetic body:
  the port's correctness rests on the PORT_NOTES bug-hunt, so any behavioral
  edit would invalidate that validation. 15 files, +3110/−11; 9 FF-only files
  created, 6 existing files modified, 0 deleted.
- **Merge strategy.** Cherry-pick of the FF work, NOT a `git merge` of
  `fffunKokkos` — `main` had already diverged via the entire Phase-0/Phase-1
  sequence, so a merge would have dragged pre-T0.0 versions of shared files
  back in. For every file that exists on both branches (`CMakeLists.txt`,
  `README.md`, `CLAUDE.md`, `.gitignore`, `NOTICE.md`, `TEST_SUITE_PLAN.md`,
  `probe_op.cpp`) **main wins** and the FF content is folded in by hand; the 9
  FF-only files (`ff_math.hpp` 924, `ff_complex.hpp` 313, `demo_ff_real.cpp`
  746, `demo_ff_complex.cpp` 650, `PORT_NOTES.md` 220, `test_ffmul.cpp` 110,
  `gen_ff_constants.cpp` 46, `run_all_ff_ops.sh` 21, `run_all_ff_complex_ops.sh`
  21) are copied over verbatim.
- **Namespace + type renaming (mirror T0.4).** Both FF headers received the
  full T0.4-style rename with no arithmetic bodies changed: `namespace
  quad::ffun` → `Kokkos::Experimental`; `ffloat` → `FloatFloat`; `ffcomplex` →
  `FloatFloatComplex`; `ffadd`/`ffsub`/`ffmul`/`ffdiv` →
  `add`/`subtract`/`multiply`/`divide`; `ffmulf`/`ffdivf` →
  `multiply_scalar`/`divide_scalar`; `ffmulff` → `two_prod`; `ffneg` →
  `negate`; `ffnint` → `round_to_nearest_int`; `ffang` → `angle`; `powi` →
  `pow_int`; `ffexpint`/`ffincgamma` → `expint`/`incgamma`; `ff_pi()…` →
  `FloatFloat_pi()…`; `make_ff` → `FloatFloat::from_bits`. Each header ends
  with a bottom-of-header `namespace Kokkos { … }` re-exposure block
  (~40 one-line forwards) mirroring `impl/Kokkos_QuadPrecisionMath.hpp`, so
  `Kokkos::exp(ff)` works identically to `Kokkos::exp(double)`;
  `add`/`subtract`/`multiply`/`divide` are deliberately NOT re-exposed under
  `Kokkos` (operators + ADL only, same posture as DD).
- **Licensing.** The load-bearing decision: FF descends from DDFUN via a
  mechanical DD→FF translation — `ff_math.hpp` line 3 reads "Copyright (c) 2024
  David H. Bailey — DDFUN v04 (original algorithms)" and its header comment
  states it "is a mechanical translation of dd_math.hpp … from double-double
  (2×FP64) to float-float (2×FP32)" — so it inherits the **DHB-License** under
  §3 grant-back, NOT independent authorship. Both FF headers therefore carry
  `LicenseRef-DHB-License` with a PORT_NOTES-referencing attribution string,
  mirroring T0.5's DD treatment. `NOTICE.md` gained 2 FF file-to-license rows
  and its DDFUN paragraph now enumerates all four derivative files
  (dd_math/dd_complex/ff_math/ff_complex). The `TEST_SUITE_PLAN.md` Licensing
  section's T2.0-kickoff checkbox was flipped to `[x]` with the FF-lineage
  rationale recorded inline. (All of the above landed in `dad92ef`.)
- **CMake wiring.** 2 new targets: `kokkos_ep_demo_ff` (from
  `demo_ff_real.cpp`) and `kokkos_ep_demo_ff_complex` (from
  `demo_ff_complex.cpp`), both linking `Kokkos::kokkos` only (the T0.0 route —
  quadmath comes through Kokkos, not a `find_library`). Deprecates
  `fffunKokkos`'s ad-hoc `kokkos_ff_demo` / `kokkos_ff_demo_complex` names. The
  install target list is expanded to all 4 demos.
- **Demo adaptation.** Real oracle migrated to the T0.0 posture
  (`Kokkos::exp((__float128)…)`); complex oracle migrated to the T0.3 posture
  (`Kokkos::exp((__complex128)…)` plus an `#include` of
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp`). Call sites use the
  `namespace ff = Kokkos::Experimental;` alias. No `KOKKOS_EP_HAVE_QUADMATH`
  gate on the include: `main`'s DD demos (`demo_real.cpp`/`demo_complex.cpp`)
  don't gate the oracle include either, so the matched (correct) posture is an
  unconditional include — the original prompt overspecified a gate that would
  have diverged from the DD demos.
- **Acceptance gate — all pass.** Build clean with zero warnings on
  FF-affected files; all 4 demos build; both FF demos run on Serial and exit 0;
  `ctest` 8/9 Passed — the one failure is the deliberately-preserved T1.4 RED
  (erf 24.64 / erfc 19.50 / tgamma 14.56, pending B1/B2/B3), with `dd_math.hpp`
  untouched so rule 4 is respected; the FF accuracy-columns diff pre/post-rename
  is **empty** across 39 real + 48 complex rows (`--batch 500000 --repeats 5
  --seed 12345`, confirming rule 7 — the rename is non-behavioral); SPDX grep
  returns exactly 4 `LicenseRef-DHB-License` + 1 `Apache-2.0-WITH-LLVM-exception`,
  no mis-stamped headers.
- **Deviations (justified).** (1) `run_all_ff_complex_ops.sh` referenced the
  deprecated `kokkos_ff_demo` name — updated to `kokkos_ep_demo_ff_complex` to
  match the T2.0 target naming (the task explicitly deprecates the ad-hoc
  names). (2) `gen_ff_constants.cpp`'s template string still emits the old
  `make_ff`/`ffloat` spellings — left verbatim because it is an out-of-scope,
  run-once constant-regeneration tool (not built by CMake, gitignored binary);
  flagged for a T2.x refresh if it is ever re-run. (3) No
  `KOKKOS_EP_HAVE_QUADMATH` gate — see Demo adaptation above.
- **Scope-out.** No modifications to `ff_math.hpp` / `ff_complex.hpp` bodies —
  T0.4 renames only. No FF test files (those are T2.1–T2.6). No harness
  extension (`BackendTraits<FF>` is T2.1's problem). No corpus extension
  (already parametric per T0.2). No QF work (Phase 3). No B1/B2/B3 work
  (deferred to after Phase 2/3). No `probe_op.cpp` merge (main wins). The
  `fffunKokkos` branch is left in place as a historical reference until Phase 3
  completes.
- **Bugs found in FF code.** None — this was a mechanical rename, and the
  byte-identical accuracy diff confirms it was non-behavioral.
- Depends on T0.0, T0.3, T0.4, T0.5.

**T2.1: EFT unit tests for FF. (DONE)**

- Executed 2026-08-02. Commit `4e025b0` (task); DONE block + docs-pointer `8c9a00f`.
- Layer-1 EFT test for FF, mirroring T1.1's shape (same four-corpora
  coverage plus named hard cases and device parity). The one material
  divergence is the oracle: plain **FP64** instead of quadmath —
  algebraically exact rather than merely higher-precision, since the
  exact FP32 sum (≤25 bits) and product (24+24 = 48 bits) both fit
  inside FP64's 53-bit mantissa. That makes it a *stronger* oracle than
  DD's quadmath (exact, not sub-ulp), so the test needs no LIBQUADMATH
  and runs unconditionally. Total scored ~4.62M inputs, 0 failures
  across all four test corpora.
- **`tests/ff_eft_test.cpp` (new, 560 LOC).** Layer-1 EFT unit test for the
  two error-free transforms every FF op is built on: the twoSum inside
  `Kokkos::Experimental::add` and the Dekker twoProduct inside `multiply`
  (splitter = `8193.0f = 2^13+1` for FP32's 24-bit mantissa; cited from
  `ff_math.hpp:192`).
- **EFT primitive extraction — mirror-and-comment.** Duplicated `two_sum` /
  `two_prod_dekker` from `ff_math.hpp` `add`/`multiply` into the test file,
  bit-identical to the embedded transforms when `.lo == 0.0f`. Same rationale
  as T1.1: transforms are embedded in longer op sequences; EFT test wants the
  transform of two RAW floats. `ff_math.hpp` NOT modified (rule 4).
- **Ground truth `double` — provable, not approximate.** Exact FP32 sum ≤25
  bits, exact FP32 product ≤48 bits, both well inside FP64's 53-bit mantissa,
  so widening operands to `double` and computing sum/product in `double` is
  exact. Asserts `(double)hi + (double)lo == (double)a {+,*} (double)b`
  bit-exactly. **Stronger oracle than DD's quadmath** (exact, not sub-ulp).
  No `KOKKOS_EP_HAVE_QUADMATH` guard; no runtime SKIP-77; runs unconditionally
  on every build. This is the sole deliberate divergence from T1.1's shape.
- **Coverage — ~4.62M inputs, 0 failures.** Test A (twoSum): 2,113,526 tested
  / 335 skipped / 0 failures. Test B (Dekker twoProduct): 2,110,205 / 3,656 /
  0. Test C (named hard cases including exact cancellation, both-subnormal,
  ±0, `1.0f + 2^-24 → lo = 2^-24`): 14 passed / 1 skipped / 0 failed. Test D
  (device parity via `parallel_for`, 200k inputs × 2 transforms = 400,000
  scored): 400,000 / 0 / 0. Per-op broad random range: 1e30 for twoSum, 1e18
  for twoProduct (deviation from prompt — see below).
- **Out-of-domain inputs are SKIPPED, not failed.** Same posture as T1.1
  with FP32-specific bounds: twoSum skips non-finite pairs and overflowing
  sums; Dekker twoProduct additionally skips subnormal operands, splitter-
  overflow magnitudes (`|x| ≥ FLT_MAX/(2^13+1) ≈ 2^114.9998`, derived from
  `two_prod_dekker`'s first op `cona = a * 8193.0f`; empirically verified
  2^114·8193 finite, 2^115·8193 = inf, matching PORT_NOTES §4a's exp
  splitter-overflow mechanism at b~2^115), and products that overflow or
  gradually underflow (error term subnormal — cite Dekker 1971; Muller et
  al. HFPA §4.4; predicate evaluated on the exact `double` product, not the
  rounded FP32 product, so underflowing pairs cannot masquerade as
  in-domain). Underflow floor: `|p| ≥ 2^-102 = FLT_MIN·2^24`.
- **Harness extension.** `tests/test_utils.hpp` gains
  `BackendTraits<FF>` (+41 LOC), field-for-field mirror of `BackendTraits<DD>`:
  `type = Kokkos::Experimental::FloatFloat`, `u = 2^-24`, `u_squared = 2^-48`,
  `max_digits = 14` (capped at FP32-double-word precision, matching
  `kMaxDigits` in `demo_ff_real.cpp`), `to_quad(x)` widener under the
  quadmath guard. `struct FF{}` tag replaces the T0.2 TODO placeholder.
  Included even though `ff_eft_test.cpp` does not consume the harness runners
  (its FP64 oracle is self-contained) — unblocks T2.2/T2.3/T2.4/T2.6.
- **CMake.** Registered via `kokkos_ep_add_eft_test(ff_eft_test)` — the
  contraction-OFF helper T1.1 established. Verified `-ffp-contract=off`
  reached the compile line via `flags.make`. No contraction-ON reporter
  (T2.5's problem, mirroring T1.5's `dd_fma_guard_test_contract_on`).
- **`tests/README.md` (+34 LOC).** Registry row + "FF EFT (Layer 1, Phase 2)"
  section mirroring the T1.1 write-up shape.
- **Acceptance gate — all pass.** `cmake --build` clean, zero warnings from
  `ff_eft_test.cpp`; `ctest -V` 9/10 Passed — the single failure is the
  deliberately-preserved T1.4 RED on `dd_accuracy_test` (erf 24.64 / erfc
  19.50 / tgamma 14.56 vs tol 25.91, digit-for-digit unchanged, pending
  B1/B2/B3); all 8 previously-passing DD tests still green; new `ff_eft_test`
  Passed with 0 failures across A/B/C/D. Both FF demos (`kokkos_ep_demo_ff`,
  `kokkos_ep_demo_ff_complex`) still build and run cleanly (RC 0) —
  `BackendTraits<FF>` addition did not disturb them.
- **Deviations (justified).** (1) Splitter is `2^13+1 = 8193.0f`, NOT
  `2^12+1` as the T2.1 prompt and T2.0-inherited `ff_math.hpp:12` license
  header state (`2^12+1 = 4097`, not 8193). Test mirrors the shipped
  `8193.0f` and cites it correctly at `ff_math.hpp:192` (which reads
  `2^13 + 1`). The stale `2^12+1` typo in `ff_math.hpp:12`'s license header
  is fixed in THIS commit (docs-only, no arithmetic change, no rule-4
  concern). (2) Per-op broad random range: 1e30 for twoSum, 1e18 for
  twoProduct — FP32's 6×-narrower exponent range overflows nearly every
  product at 1e30 (a·b = 1e60 ≫ FLT_MAX ≈ 3.4e38), giving vacuous coverage
  (first build tested 1/1e6 twoProduct pairs, 0 device twoProduct pairs).
  1e18 keeps products ≤1e36 < FLT_MAX and operands well below the splitter
  bound. Rationale documented in-code via `broad_bound()`. (3) No
  `KOKKOS_EP_HAVE_QUADMATH` guard — as prompted; the FP64 oracle is exact
  and unconditional.
- **Scope-out.** EFT primitives only (twoSum + Dekker twoProduct); no
  higher-level ops (sqrt/exp/log/trig/transcendentals — T2.2/T2.4). No FF
  complex EFTs (out of scope). No contraction-ON reporter (T2.5). No new
  corpus categories (`corpus.hpp` already parametric per T0.2). No harness
  runner extension beyond `BackendTraits<FF>`. No QF work. No B1/B2/B3
  work. `ff_math.hpp` / `ff_complex.hpp` arithmetic bodies untouched
  (rule 4).
- **Bugs found in FF code.** None during the EFT test. One docs-only typo
  in `ff_math.hpp`'s license header (`2^12+1` → `2^13+1`) fixed inline
  in this commit; traced back to boilerplate inherited from the T2.0
  prompt.
- Depends on T0.2, T1.1, T2.0.

**T2.2: Non-overlap invariant checks for FF. (DONE)**

- Executed 2026-08-03. Commit `f56cc2c` (task); DONE block + docs-pointer `c3c1921`.
- Layer-2 output-invariant test for FF, the FP32 analogue of T1.2
  (`dd_invariant_test`): asserts the non-overlap invariant `fl(hi+lo)==hi`
  evaluated in RAW FP32 — `(f.hi + f.lo) == f.hi` in `float` — for every
  `ff_math.hpp` op that returns a `FloatFloat`. Oracle-independent: no
  `__float128`, no `KOKKOS_EP_HAVE_QUADMATH` gate; runs unconditionally like T1.2.
  Total 48,787,956 inputs checked across all 50 ops, **0 failures**.
- **`tests/ff_invariant_test.cpp` (new, 830 LOC).** Structure mirrors T1.2
  verbatim: Test A (every op, two passes — 10^6 op-appropriate random inputs +
  full corner-case corpus `corpus::unary/binary<float>`, zero on / inf,nan off),
  Test B (device tripwire: add/multiply/sqrt/exp/sin at 10^5 via `parallel_for`,
  results copied back and checked on host), Test C (PORT_NOTES §4 named
  regressions). The one type change vs T1.2: the invariant is FP32-typed (DD uses
  `double`). No `__float128` promotion — that tests the exact real sum, a different
  property.
- **50-op inventory, none missing.** All 50 ops from T1.2's inventory are present
  on the FF side (31 unary + 13 binary + fma + 4 two-output components + pow_int);
  the report-and-stop-if-missing contingency never fired.
- **FP32-narrower domain predicates, re-derived from `ff_math.hpp` (not copied from
  T1.2).** FP32's exponent range is ~6× narrower than FP64 and the port has
  FP32-specific hazards T1.2 never hit. Skip-not-fail gating keeps out-of-domain
  inputs — and the domain-guard diagnostics `ff_math.hpp` would otherwise print —
  out of the run. Material tightenings: `exp < 88` (not DD's 300); trig family
  lower floor `|x| >= 1e-25` (FP32 `sincos` Taylor iteration-limit on tiny args, no
  T1.2 counterpart); `atan2` per-operand floor `|·| >= 1e-18`; log-window
  `[1e-34, 1e34]`; `sinh`/`cosh < 40`, `tanh < 20`, `tgamma ∈ [1e-3, 23)`.
- **Registered via plain `kokkos_ep_add_test(ff_invariant_test)`** — no contraction
  flags; the invariant holds regardless of FMA contraction (not an EFT test).
- **`tests/README.md` (+101 LOC).** Registry row + "FF non-overlap invariant
  (Layer 2, Phase 2)" section documenting the FP32-narrower predicates and every
  finding below.

- **FINDING 1 — underflow-tail round-to-even hole (harness improvement, NOT a
  masking fix; cross-cutting — see known-lurking issue).** The first full run
  reported "failures" on `exp`/`exp2`/`exp10` (and `device:exp`) for very negative
  args (e.g. `exp(-84.32) → hi=2.40e-37, lo=-1.121e-44`). These are NOT
  normalization defects: each failing `lo` is a *subnormal* landing EXACTLY on
  `±½ ulp(hi)`, where round-to-even flips `fl(hi+lo)` off `hi` by one ulp even
  though the mathematical non-overlap `|lo| ≤ ½ ulp(hi)` still holds. Systematic
  only once `|hi| < 2^-102`, where the tie value `½ ulp(hi) = 2^(e-24)` is itself
  subnormal so `lo` quantizes straight onto the tie. This is a property of
  double-word arithmetic in the FP32 denormal tail — universal to DD/FF/QF — not an
  `ff_math.hpp` bug (Rule 4 never engaged). Resolved with an output-side, general
  skip-not-fail guard in `result_checkable` (`kUnderflowTail = 2^-100`, a 4× margin
  above `2^-102`); it does NOT mask a real overlap — a normal-range
  `|lo| > ½ ulp(hi)` is still checked.
- **FINDING 2 — `remainder(68.379f, 3.5066f)` sign.** The prompt (following
  PORT_NOTES §4b) expected a POSITIVE FP32 remainder (+1.7533) on the premise
  `a/b ≈ 19.4999993 < 19.5` at FP32 → `nint=19`. That premise does NOT hold for
  these literals: at FP32 `68.379f/3.5066f = 19.5000858` (> 19.5), so `nint=20` and
  the remainder is NEGATIVE (−1.75300026). `std::remainderf` agrees exactly and the
  shipped FF `remainder` reproduces it — FF is correct; the historical `ffnint` bug
  must have been about a different input. Sign gated against `std::remainderf`
  (same-precision oracle), NOT `std::remainder` (which would compare an FP32 op to
  the FP64 answer).
- **FINDING 3 — nint literal `19.4999993f`.** Rounds to EXACTLY `19.5f` at FP32
  (`lo=0`) → nint 20 (correct); the historical 19-vs-20 distinction only appears
  when `19.4999993` is carried in the FF pair via the Route-A double split
  (`hi=19.5, lo=-7e-7`, total < 19.5), where the fixed `ffnint` returns 19. Test C
  checks both constructions.
- **exp §4a Test C — two DISTINCT roles, made explicit in the output.**
  `79.5/80/85` are the load-bearing NaN-pre-fix regression cases (the §4a bug was
  NaN-from-splitter-overflow: `b * 8193.0f` overflowed FP32); `88.7/88.72` are
  edge-of-saturation GUARD cases past the `a.hi >= 88` guard — they saturate to +0
  (invariant trivially holds, not-NaN), emitting the run's only 2 `FFEXP`
  diagnostics, which are EXPECTED and NORMAL. Documented safety-guard behavior is a
  PASS, not a report-and-stop.

- **KNOWN-LURKING ISSUE (cross-cutting; follow-up, NOT urgent, NOT a blocker).**
  The underflow-tail round-to-even hole (Finding 1) is NOT FF-specific — it is a
  property of the `fl(hi+lo)==hi` *evaluation* itself, so DD (and a future QF) can
  hit it too. `dd_invariant_test` (T1.2) simply never tripped it across ~50.5M
  inputs because FP64's exponent range is ~6× wider, keeping the denormal tail out
  of reach for realistic random inputs; FF surfaced it because FP32's narrower
  range brings the tail into reach at ordinary `exp`-of-negative inputs.
  **Follow-up:** give `dd_invariant_test` the same `kUnderflowTail`-style guard so
  it is not surprised by the same hole if the DD op inventory grows or a future
  random seed lands in the tail. Tracked here; no code change in this task.
- **Acceptance gate — all pass (Serial).** `cmake --build` clean, zero warnings;
  `ff_invariant_test` EXIT=0, 0 failures over 48,787,956 checked inputs (Test A
  48 ops + fma + pow_int, Test B 5 device ops, Test C 51/51). Only 2 internal
  `ff_math.hpp` diagnostics in the whole run — the two Test C edge-of-saturation
  `FFEXP` prints; Test A/B are diagnostic-clean. `ctest` 10/11 green incl.
  `ff_invariant_test`; the sole failure is the deliberately-preserved T1.4 RED on
  `dd_accuracy_test` (erf 24.64 / erfc 19.50 / tgamma 14.56 vs tol 25.91,
  digit-for-digit unchanged, pending B1/B2/B3); all previously-passing tests still
  green. Both FF demos (`kokkos_ep_demo_ff`, `kokkos_ep_demo_ff_complex`) still
  build and run cleanly (RC 0).
- **Deviations (justified).** (1) The `kUnderflowTail` output-side guard is an
  addition to T1.2's `result_checkable` skip criteria not present in the T1.2
  template — see Finding 1. (2) Test C `exp(88.7)/(88.72)` treated as
  PASS-on-saturation (guard fires → +0, not NaN) rather than asserting a finite
  non-NaN value the ≥88 guard makes impossible — the §4a NaN-fix is proven by
  79.5/80/85 instead. Both are documented in-code and in README.
- **Scope-out.** Real FF ops only — `ff_complex.hpp` out of scope; accuracy-vs-
  oracle is T2.4. No new corpus categories (`corpus.hpp` already parametric per
  T0.2). No QF work. No B1/B2/B3 work. `ff_math.hpp` / `ff_complex.hpp` NOT
  modified (rule 4).
- **Bugs found in FF code.** None. All findings are either correct FF behavior
  (Findings 2, 3), a fundamental double-word property resolved test-side (Finding
  1), or documented safety-guard behavior (exp §4a saturation).
- Depends on T0.2, T1.2, T2.0, T2.1.

**T2.3: Property/identity tests for FF. (DONE)**

- Same identities as T1.3, adjusted for FF's ~14 digit precision.
- No oracle needed.

- Executed 2026-08-03. Commit `c49572b` (task); DONE block + docs-pointer `1c4a59c`.
- **`tests/ff_property_test.cpp` (new, 799 LOC).** Layer-3 algebraic-identity
  test, the FP32 analogue of T1.3 (`dd_property_test`): do the FF ops compose the
  way the algebra says? Structure mirrors T1.3 verbatim; the mechanical change is
  the precision scale (`u = 2⁻²⁴` for FF vs DD's `2⁻⁵³`). Group A 7 bit-exact +
  Group B 13 tolerance + Test C 5/5, **0 failures across host + device**, **zero
  diagnostic prints**. `ff_math.hpp` / `ff_complex.hpp` NOT modified (rule 4).
- **Group A — bit-exact, no oracle (10⁶ random + full finite corpus, raw `==`).**
  A1 `add(a,negate(a))==0`, A2 `a-a==0`, A3 `a·1==a`, A4 `a·(-1)==negate(a)`,
  A5 `abs` sign branches, A6 `negate(negate(a))==a`, A8 add commutativity.
  **7 identities, 14.00 digits, 0 failures.** A1–A6 use Route-A `FloatFloat(double)`
  operands (nonzero `lo`); A8 uses single-float operands (`lo==0`, the FF analogue
  of DD's single-double convention, since `add`'s `+a.lo+b.lo` tail reorders under
  swap). A3/A4 call `multiply` → Dekker-domain-gated (`dom_dekker`), each **skips 22
  Dekker-split corpus entries** past `split_safe_max()`; the rest use `dom_all`.
  A7 mul-commutativity **demoted to B0** by design (Dekker cross-term reorders under
  swap), exactly as in T1.3. Denormal-tail mismatches (`<2⁻¹⁰⁰`) count as skipped.
- **Group B — tolerance vs `__float128` oracle (`#ifdef KOKKOS_EP_HAVE_QUADMATH`;
  runtime-SKIP otherwise).** 13 identities scored with `digits_of_accuracy`,
  **fail-gated on the MEAN** against `tolerance_digits = -log10(N·u²)` = **8.45** at
  N=10⁶ (`u²=2⁻⁴⁸`; computed at runtime from `BackendTraits<FF>::u_squared`, NOT
  hardcoded). B0 mul_comm, B1 sqrt²≈a, B2 exp(log a)≈a, B3 log(exp a)≈a,
  B4 sin²+cos²≈1, B5 sin(−a)==−sin a, B6 cos(−a)==cos a, B7 tan·cos≈sin,
  B8 2·sin·cos≈sin(2a), B9 exp(a)·exp(−a)≈1, B10 hypot²≈a²+b², B11 pow(a,2)≈a·a,
  B12 atanh(a)≈½(log(1+a)−log(1−a)). Per-identity proven bounds (2u²/4u²/10u²) cited
  in comments (rule 5). **All 13 pass; means 13.10–14.00, clearing the 8.45 floor
  with room.**
- **Test C — named-constant regressions.** C2 `log(e)≈1`, C3 `exp(log2)≈2`,
  C4 `√2·√2≈2`, C5 `log(10)≈log10` constant, plus C1 `|sin(π)|≤ε` (softened for
  arg-reduction conditioning). **5/5 pass.** C6 (euler_gamma/digamma) SKIPPED — no
  digamma op and no independent FF oracle for the constant.
- **Device pass.** 3 Group A (A1, A3, A5) bit-exact + 2 Group B (B1, B4) digit
  checks rerun on device at 10⁵ inputs (Serial here; `parallel_for` + `KOKKOS_LAMBDA`
  ships `hi`/`lo` back). All PASS.
- **B3/B9 narrowing (surfaced-not-hidden; see B4).** The outer `exp` in these
  identities respects `ff_math.hpp`'s `a.hi≥88` guard, but B3's `log(exp(x))` calls
  `log`, whose internal Newton iteration invokes `exp` on generic large-`|x|` args
  and **stalls there** — not at the top-level guard. Domain narrowed
  `[-85, 85]→[-69, 69]` (B2's clean `|log(x)|≤69` ceiling); B3 min `0.00→9.80`,
  B9 min `0.00→12.56`. In-source B4 citations on both identities mark the
  restoration path; cross-reference the B4 stub in the follow-up-bug-tasks
  subsection.
- **B4 discovery (library defect, deferred per rule 4).** `ff_math.hpp` `exp`'s
  Taylor convergence `eps = 1e-15f` is finer than FloatFloat's resolution
  `2⁻⁴⁸ ≈ 3.55e-15`; for ~3.1% of generic large-`|x|` args the series never drops
  below `eps`, hits the 60-iteration cap, prints `FFEXP: iteration limit`, and
  returns a wrong `0` (absorbed by `log`'s Newton — the identity accuracy is
  preserved but stdout is spammed, 31,093 prints pre-narrow → 0 post-narrow). A
  direct DD→FF port artifact: DD's `exp` uses `eps=1e-32` matching its `1.2e-32`
  resolution with cap 100, and is clean. Logged as **B4** in the
  follow-up-bug-tasks subsection (`from T1.4 and T2.3`).
- **CMake / README.** Registered via plain `kokkos_ep_add_test(ff_property_test)`
  (+5) — NO contraction flags (identities are FMA-contraction agnostic).
  `tests/README.md` (+49) gains the registry row + "FF property/identity (Layer 3,
  Phase 2)" section. `docs/TEST_SUITE_PLAN.md` (+34/−6): the B4 stub plus the
  follow-up subsection rename `(from T1.4)` → `(from T1.4 and T2.3)`.
- **Acceptance gate — all pass (Serial).** `cmake --build` clean, zero warnings;
  `ctest` 12 tests / 685 s, `ff_property_test` #10 **Passed**, all 11 prior tests
  still green; the sole red is the deliberately-preserved T1.4 RED on
  `dd_accuracy_test` (erf 24.64 / erfc 19.50 / tgamma 14.56 — digit-for-digit
  unchanged, no regression). 0 failures across Group A/B/Test C host + device;
  **zero** `FFEXP`/`FFCSSNR`/`FFNINT` diagnostic prints in the whole run (was
  31,093 pre-narrow). Both FF demos (`kokkos_ep_demo_ff`,
  `kokkos_ep_demo_ff_complex`) still build and run cleanly (RC 0).
- **Deviations (justified).** (1) Tolerance floor is **8.45** (runtime formula
  `-log10(10⁶·2⁻⁴⁸)`), NOT the plan's **8.24** estimate — tightened not relaxed,
  documented in-source. (2) B3/B9 domain `[-85, 85]→[-69, 69]` per the
  option-2-adapted verdict (surface-and-defer, rule 4: no `ff_math.hpp` fix). (3)
  The task-commit body was authored by cluster-Claude (its drafted middle was elided
  in handoff).
- **Scope-out.** Real FF ops only — no complex identities (`ff_complex.hpp`); no
  associativity/distributivity anti-tests (documented in-source as deliberately
  untested — FALSE for any finite-precision format); device parity limited to 3
  Group A + 2 Group B; quadmath NOT used as the Group A oracle. No B1–B4 fixes.
- **Bugs found in FF code.** One (B4, above) — surfaced, reported, and deferred per
  rule 4; not patched in this task.
- See `c49572b` for the code diff and this DONE block for the outcome; B4 stub in
  the follow-up-bug-tasks subsection.
- Depends on T0.2, T1.3, T2.0, T2.1.

**T2.4: Differential accuracy for FF vs quadmath. (DONE, RED)**

- Per-op `max(rel_err / u²)` where `u = 2⁻²⁴`.
- For FF, published bounds from CAMPARY / Joldes-Muller-Popescu apply
  more directly (FF is closer to canonical double-word than DD's
  DDFUN variant is). Cite where applicable.
- Report min AND mean; annotate PORT_NOTES §5 conditioning-limited
  ops as expected-min-drops.
- Expected mean: 13.3–14.0 digits per PORT_NOTES.

`(DONE, RED)` is deliberate, exactly as in T1.4: the task shipped its deliverable
(the test) and the test is doing its job — it flags three REAL `ff_math.hpp`
accuracy defects and fails on them. The red is the point; it is the durable
regression gate for the follow-up bug tasks B5/B6/B7 below (the FF siblings of
T1.4's DD B1/B2/B3).

- Executed 2026-08-03. Commit `12020a4` (task); DONE block + three B-task stubs
  (B5/B6/B7) + subsection rename `571b26b`; docs-pointer follows.
- **`tests/ff_accuracy_test.cpp` (new, 816 LOC).** Layer-4 per-op differential
  accuracy vs the `__float128` oracle, the FF analogue of `dd_accuracy_test`
  (T1.4). Mirrors that test's structure verbatim; the mechanical changes are the
  precision scale (`u = 2⁻²⁴`, `u² = 2⁻⁴⁸`, digits capped at 14, mean-gated at
  ~8.45) and the FP32-narrower op domains taken **verbatim from T2.2**
  (`ff_invariant_test`: exp guards at `a.hi≥88` not DD's 300, trig carries a
  tiny-arg lower bound, sinh/cosh cap at |x|<40, tanh at |x|<20, log window
  `[1e-34,1e34]`, erf/erfc `[-6,6]`, tgamma `[1e-3,23)`), with the FP32 corpus
  accessors (`corpus::unary<float>` / `<float>` named accessors). Shared
  PORT_NOTES §5 registry → EXPECTED-MIN-DROP (conditioning is a property of the
  algorithm, not the width, so DD and FF read the same table). Whole file
  `KOKKOS_EP_HAVE_QUADMATH`-guarded; SKIP (77) without quadmath. Registered with
  the plain `kokkos_ep_add_test` helper (not an EFT test).
  **`tests/CMakeLists.txt` (+5) / `tests/README.md` (+62) (edit).**
- **Op inventory — same ~50-row set as T1.4/T2.2, verbatim** (see the T1.2/T1.4
  DONE blocks for the full enumeration; not duplicated here). Categories: unary
  (abs…tgamma), two-output (sincos/sinhcosh, per component), binary (add…fdim),
  ternary (fma), integer-scalar (pow_int).
- **Tolerance rationale.** `tolerance_digits = −log₁₀(N · u²)` with u² = 2⁻⁴⁸ and
  N = 10⁶ → **8.45**. Same formula as T1.4 / T2.3 Group B, computed at runtime
  from `BackendTraits<FF>::u_squared`. Single uniform tolerance + the PORT_NOTES
  §5 registry only — **no per-op tolerance overrides**.
- **Results — 47 PASS, 3 FAIL (real signals, not test artifacts).** 50 ops total.
  Mean-digit distribution: **34 ops at ~13.5–14.0**; **8 EXPECTED-MIN-DROP**
  §5-sanctioned ops clear the 8.45 floor with low mins surfaced (exp
  output-denormal lo; sin/cos/tan near ±π; asin/acos derivative→∞ near |a|=1;
  atanh 1/(1−a²) near |a|=1; subtract/fdim/fma near-cancellation; remainder near
  multiples of b); **3 FAIL**.
  - **erf — mean=3.94 / min=0.00 / tol=8.45 FAIL.** Returns **NaN across the
    smooth well-conditioned range ~[1.9, 6]** (`ff_math.hpp:694`). Two FP32
    failures compound: the Taylor branch overflows far earlier than DD's (FP32
    exponent range ~10³⁸ vs FP64 ~10³⁰⁸), and the large-|z| asymptotic branch is
    broken (port-inherited, same defect DD's B3 flags). Logged as **B5** below.
  - **erfc — mean=3.91 / min=0.00 / tol=8.45 FAIL.** `erfc(z)=subtract(1, erf(z))`
    (`ff_math.hpp:738`) inherits erf's NaN plus catastrophic cancellation as
    erf(z) → 1 (the T1.4 B2 pattern, compounded). Logged as **B6** below.
  - **tgamma — mean=6.10 / min=5.94 / tol=8.45 FAIL.** Uniformly ~6 digits, flat
    across [1e-3, 23). Lanczos g=7 with **FP32** coefficient constants
    (`ff_math.hpp:749`: 676.5203681218851f, …) caps the accumulator at FP32's
    ~7-digit ceiling regardless of the enclosing FF arithmetic — worse than DD's
    B1 (24-bit mantissa vs 53-bit). Logged as **B7** below.
  - The three failures were verified **library-side, not oracle-side**, by an
    independent standalone probe (separate TU calling quadmath `erfq`/`erfcq`/
    `tgammaq` directly) that reproduced clean oracle values across the failing
    ranges (erf(2)=0.995…, tgamma(5)=24); the same oracle machinery scores 47
    other ops at ~13.5–14.0, so oracle and domains are sound.
- **Rule 4 posture.** `ff_math.hpp` / `ff_complex.hpp` NOT modified. The three
  failures are real defects and stay flagged RED in the shipped test until
  B5/B6/B7 land; the test is the durable regression gate for those fixes. The
  failing ops are NOT skipped/disabled/xfailed.
- **Acceptance-gate results.** Build clean, zero warnings from
  `ff_accuracy_test.cpp`; `ctest` **11/13 green**. The only two REDs are
  `ff_accuracy_test` (new, this task) and `dd_accuracy_test` (the
  deliberately-preserved T1.4 RED: erf 24.64 / erfc 19.50 / tgamma 14.56 —
  digit-for-digit unchanged, no regression); all previously-passing tests
  including `ff_property_test` still green. Both FF demos (`kokkos_ep_demo_ff`,
  `kokkos_ep_demo_ff_complex`) build and run cleanly (RC 0). **Zero** FFEXP /
  FFCSSNR / FFNINT (or any) diagnostic prints during the run.
- **Scope-out.** Real FF ops only — no `ff_complex.hpp`; no DD/QF; no MPFR; no new
  corpus categories; no per-op tolerance overrides. Op inventory not re-derived
  (taken from T2.2). PORT_NOTES §5 not modified (the 8 low-min ops are sanctioned
  by the existing registry).
- Depends on T0.1, T0.2, T2.0, T2.1.
- See `12020a4` for the code diff; B5/B6/B7 stubs in the follow-up-bug-tasks
  subsection.

**T2.5: FMA-contraction guard for FF. (DONE)**

- Same as T1.5 but for FF. Precedent: `test_ffmul.cpp` compiles with
  `-ffloat-store` for this reason.

- Executed 2026-08-03. Commit `e293de7` (task); DONE block + docs-pointer `b98e29e`.
  Closes Phase 2 (7 of 7 tasks).
- **`tests/ff_fma_guard_test.cpp` (new, 514 LOC).** Layer-5 positive test of the
  FMA-contraction posture T2.1 adopts defensively, the FP32 analogue of T1.5
  (`dd_fma_guard_test`). Builds the identical FF Dekker `twoProduct` (mirrored from
  `ff_math.hpp:193-207` / `two_prod` 266-274, copied verbatim from `ff_eft_test.cpp`
  per rule "tests are standalone; duplication is acceptable"; splitter `8193.0f` =
  2¹³+1 for FP32's 24-bit mantissa) under BOTH contraction settings and cross-checks
  against a contraction-immune oracle. `twoSum` is included as a labeled CONTROL (no
  mul-then-± adjacency → contraction-immune → must stay exact both ways). Structure
  mirrors T1.5 verbatim: single-source/two-targets, `KOKKOS_EP_CONTRACTION_MODE`
  (0 = OFF/gate, 1 = ON/report) + `KOKKOS_EP_BASELINE_PATH` (ON only), OFF fail-gates
  (`KOKKOS_EP_ASSERT F == 0`), ON reports (exits 0, baseline drift check). `ff_math.hpp`
  / `ff_complex.hpp` NOT modified (rule 4).
- **Test structure.** FF Dekker `twoProduct` bit-exact check `a*b == hi + lo` on
  **220,410 in-domain checks** (110,205 host + 110,205 device — 10⁵ random in
  `[-1e18f, 1e18f]` + corpus cross-product, each pair filtered by an FP32-narrowed
  `prod_in_domain`); plus a `twoSum` CONTROL over the same input list with an
  FP32-specific oracle-faithful guard (see deviation 2). Both variants registered:
  `kokkos_ep_add_eft_test(ff_fma_guard_test)` (OFF) and
  `kokkos_ep_add_eft_test_contract_on(ff_fma_guard_test)` (ON, suffixed target).
- **Acceptance-gate results.** Contraction-off variant: **PASS, 0 misses**
  (`twoProduct` 0 mismatches host + device; `twoSum` control 0 mismatches). Contraction-
  on variant: **PASS reporter, `F = 0`**, baseline armed. Both variants register in
  ctest (**#15 `ff_fma_guard_test`, #16 `ff_fma_guard_test_contract_on`**) and both
  show **Passed** at the ctest-summary level. **ctest: 14 pass / 2 fail** — the 2 REDs
  are the deliberately-preserved `dd_accuracy_test` (T1.4: erf 24.64 / erfc 19.50 /
  tgamma 14.56 — digit-for-digit unchanged) and `ff_accuracy_test` (T2.4: erf 3.94 /
  erfc 3.91 / tgamma 6.10 — digit-for-digit unchanged). Both FF demos
  (`kokkos_ep_demo_ff`, `kokkos_ep_demo_ff_complex`) build and run RC 0. `cmake --build`
  clean, zero warnings from `ff_fma_guard_test.cpp`; zero unexpected diagnostic prints.
- **Deviation 1 — contraction-ON variant is a REPORTER, not `WILL_FAIL`.** The drafted
  T2.5 prompt guessed the ON variant would flag `> 0` mismatches and register `WILL_FAIL`.
  That is not T1.5's actual shape: T1.5's ON variant is a **reporter** because GCC 13.3.0
  on baseline x86-64 emits plain mul+sub and contracts nothing → `F = 0` for DD. The FF
  Dekker `twoProduct` is the same algorithm at FP32 on the same toolchain and behaves
  identically: **`F = 0`**. This is a T1.5-verbatim mirror, not a weakening — the guard's
  realness is proven by T1.5's separate *sensitivity* check (a deliberately-broken
  `twoProduct` flags ~95% of checks under the ON flags, yet still exits 0), not by
  production compiler behavior. A representative failing input pair `> 0` **does not
  exist on this toolchain** for either backend; claiming one would be fabrication. The
  drafting error was corrected up front by the "read §T1.5 first" instruction.
- **Deviation 2 — FP32-specific `twoSum` oracle guard (`sum_oracle_faithful`), no DD
  counterpart.** The first build showed **84 `twoSum` "mismatches" under BOTH postures**
  — posture-independent, so not a contraction effect but a real oracle limitation.
  Root cause: T1.5's DD guard reuses its `twoProduct` input list for the `twoSum` control
  unchanged, which is safe at FP64 scale but not at FP32. A product-domain pair like
  `(FLT_MIN = 2⁻¹²⁶, 2²⁴)` has an in-range *product* (2⁻¹⁰², admitted by
  `prod_in_domain`) yet an exact *sum* spanning 2²⁴ down to 2⁻¹²⁶ — **174 significant
  bits**, far beyond the FP64 oracle's 53-bit mantissa. The FP32 `twoSum` is **correct**
  there (`hi = 2²⁴`, `lo = 2⁻¹²⁶`, both representable and non-overlapping); it is the
  FP64 *decomposition oracle* that collapses the tiny tail (`(double)a + (double)b`
  rounds the addend away → `lo == 0`), so the exact FP32 `lo` looks like a false
  "mismatch". Fix: the control skips pairs the oracle cannot witness — a pair is in
  domain only when the FP64 `twoSum` error term is zero (the double sum carries no
  rounding). This is the EFT skip-not-fail discipline applied to the *oracle*, excluding
  ONLY wide-exponent sums, never a pair where FP32 `twoSum` is actually wrong. After the
  guard: 0 `twoSum` mismatches. `twoProduct` needs no such guard — its exact product is
  always ≤48 bits, so the FP64 oracle is unconditionally faithful (confirmed: 0
  `twoProduct` mismatches over 220,410 checks).
- **Deviation 3 — FP64 oracle, no quadmath gate (inherited from T2.1).** Ground truth is
  plain **FP64**, not `__float128`: the exact FP32 product needs ≤48 bits, which fits
  FP64's 53-bit mantissa, so the reference `(p, e)` decomposition is *algebraically
  exact* — a stronger oracle than DD's quadmath, needing no external library. Both
  variants therefore run **unconditionally** (no `KOKKOS_EP_HAVE_QUADMATH` gate, no
  runtime SKIP-77), unlike DD's `dd_fma_guard_test` which SKIPs without LIBQUADMATH.
  Same posture T2.1's `ff_eft_test` established.
- **Baseline / CMake.** `tests/ff_fma_guard_baseline.txt` (new, 26 lines, first
  non-comment line = `0`) armed for future regression detection: each ON run compares
  its live count to the baseline and prints `baseline: OK` or `*** DRIFT ***` (WARN-only,
  never a failure). Generalized the `kokkos_ep_add_eft_test_contract_on` helper to derive
  the baseline path from the target name (`<base>_baseline.txt`), replacing T1.5's
  hardcoded `dd_fma_guard_baseline.txt` — so the DD and FF guards share one helper
  (touches the DD path too; `tests/CMakeLists.txt` **+15/−1**). `tests/README.md`
  **+60**: two registry rows + "FF FMA-contraction guard (Layer 5, Phase 2)" section.
- **Scope-out.** FF Dekker `twoProduct` ONLY (the one contraction-hazard primitive); no
  complex, no other ops; no rebuilding Kokkos under different contraction settings
  (per-target flags suffice). No new B-task stubs — 0 misses, nothing to log.
  `ff_math.hpp` / `ff_complex.hpp` untouched; demos untouched.
- Depends on T0.1, T0.2, T2.0, T2.1.
- **Closes Phase 2. 7 of 7 tasks DONE:** T2.0 (`dad92ef`), T2.1 (`4e025b0`),
  T2.2 (`f56cc2c`), T2.3 (`c49572b`), T2.4 (`12020a4`, RED-by-design), T2.5
  (`e293de7`), T2.6 (`6cb2211`). Follow-up bug tasks B4/B5/B6/B7 logged for
  post-Phase-3 fixes.
- See `e293de7` for the code diff and this DONE block for the outcome. Phase 2
  complete; Phase 3 (QF from scratch) next.

**T2.6: End-to-end cancellation kernels for FF. (DONE)**

- Same kernels as T1.6, expect ~14 digits accuracy.

- Executed 2026-08-03. Commit `6cb2211` (task); DONE block + docs-pointer `d144394`.
  **Note: executed out of plan-doc order** — this is T2.6 (end-to-end cancellation
  kernels, the T1.6 analogue); T2.5 (FMA-contraction guard for FF, the T1.5
  analogue) remains the final open Phase-2 task. Prompt-labeling artifact (the task
  prompt said "T2.5" but its body unambiguously described the `dd_e2e_test`
  analogue), not a scope change.
- **`tests/ff_cancellation_test.cpp` (new, 413 LOC).** Layer-6 end-to-end test, the
  FP32 analogue of T1.6 (`dd_e2e_test`): four classic cancellation-hostile kernels
  evaluated in FF and scored in digits of accuracy against `__float128` /
  closed-form oracles, mean-gated at 11.0 digits. Structure mirrors T1.6 verbatim;
  the mechanical change is the precision scale (FF `max_digits = 14` vs DD's 31).
  Whole file is `#ifdef KOKKOS_EP_HAVE_QUADMATH`; runtime-SKIP 77 without quadmath.
  Host-side only (the kernels are inherently serial reductions/recurrences).
  `ff_math.hpp` / `ff_complex.hpp` NOT modified (rule 4).
- **Two-oracle strategy (K2, K4).** Same split as T1.6. The
  FF-vs-quadmath-partial-sum comparison carries the arithmetic-precision claim:
  identical N, identical summation order, identical terms, so it isolates FF's
  accumulation quality from truncation. The FF-vs-closed-form comparison
  (K2 vs π²/6, K4 vs ln 2) is a truncation-limited sanity check, gated at
  `truncation_floor − 1` digit of slack. At N=10⁶ the floor is ~6 digits: the
  Basel tail Σ_{N+1}^∞ 1/k² ≈ 1/N, and the alternating-series error is bounded by
  the first omitted term ≈ 1/N.
- **Tolerance rationale.** FF's harness cap is `BackendTraits<FF>::max_digits = 14`
  (u² = 2⁻⁴⁸ ≈ 14.45 decimal digits); the SAME "cap − 3" formula T1.6 used gives
  `14 − 3 = 11.0` (DD used `31 − 3 = 28.0`), leaving ~3 digits of headroom for
  accumulated round-off in composed / 10⁶-term kernels, applied uniformly to the
  arithmetic-precision comparisons. Computed from `max_digits` at compile time, not
  hardcoded.
- **Per-kernel results (mean_digits / tolerance 11.0).** `K1_stable` 14.00/14.00
  (harness cap) — PASS; `K2_basel` 12.70 — PASS by +1.70; `K3_machin` 14.00
  (harness cap) — PASS; `K4_alt_harmonic` 11.50 — PASS by +0.50. `K1_naive_report`
  FF {10.27, 7.64, 8.60} vs FP32 {3.28, 0.00, 0.00} at x ∈ {1e2, 1e4, 1e6} — a
  ~+7-to-+10-digit FF-over-FP32 lift. K2 sanity vs π²/6 6.22 digits (truncation
  floor 6); K4 sanity vs ln 2 6.14 digits (truncation floor 6). **4 PASS / 0 RED.**
- **K1 measurement deviations (justified; measurement wins over spec, per the T1.6
  K1 precedent).** Two FP32-forced deviations from the literal T1.6 recipe, both
  documented in-source and in the README, both preserving the test's intent:
  (1) **Naive baseline = FP32, not FP64.** FF's demonstrable lift is over its
  1-word base scalar (FP32, ~7 digits), symmetric to DD's lift over FP64. FP64
  (~16 digits) is *wider* than FF (~14), so an FP64 baseline would be dishonest —
  it would "win" the naive contest while saying nothing about FF. (2) **Magnitudes
  {1e2, 1e4, 1e6}, not T1.6's {1e6, 1e10, 1e15}.** The cancellation gradient sits
  ~3 decades lower at FP32: plain FP32 loses the "+1" in `x²+1` once `x² > 2²⁴`
  (x ≳ 4100), so at T1.6's magnitudes both stable and naive read exactly 0 at the
  FP32 base — no visible lift. At {1e2, 1e4, 1e6} the "+1" is retained in FF at all
  three and the FP32→FF lift is visible across the whole sweep (FP32 naive
  collapses to 0 at x ≥ 1e4, FF retains ~7-10 digits).
- **K2/K4 iteration counts.** Kept at N = 10⁶ (no FP32 rescaling needed): at that N
  the smallest term (1/N = 1e-6 for K4, 1e-12 for K2) stays well above FF's
  running-sum resolution, so no term stalls into the precision floor — verified by
  the clean 12.70 / 11.50 arithmetic-precision means; the FP32 iteration-bound
  concern the plan flags does not bite here.
- **Contraction posture.** Registered with the plain `kokkos_ep_add_test`
  helper (mirroring `dd_e2e_test`), NOT the EFT helper. K1's naive `√(x²+1)−x` has
  a mul-then-sub adjacency an FMA *could* contract, but K1_naive is
  reported-not-gated, and the gated path K1_stable = `1/(√(x²+1)+x)` has no
  subtractive-cancellation adjacency. The Dekker-`twoProduct` hazard the EFT helper
  guards is on no gated path here; documented in the CMake comment.
- **`tests/CMakeLists.txt` (+12).** `kokkos_ep_add_test(ff_cancellation_test)`.
  **`tests/README.md` (+71).** Registry row + "FF end-to-end cancellation (Layer 6,
  Phase 2)" section (DD baseline + FF Phase-2 subsection).
- **Acceptance gate — all pass.** `cmake --build` clean, zero warnings from
  `ff_cancellation_test.cpp`; `ctest` **12/14 green**, `ff_cancellation_test` #12
  **Passed**, all 11 previously-passing tests still green. The only 2 REDs are the
  deliberately-preserved `dd_accuracy_test` (T1.4: erf 24.64 / erfc 19.50 / tgamma
  14.56 — digit-for-digit unchanged) and `ff_accuracy_test` (T2.4: erf 3.94 /
  erfc 3.91 / tgamma 6.10 — digit-for-digit unchanged). Both FF demos
  (`kokkos_ep_demo_ff`, `kokkos_ep_demo_ff_complex`) build and run cleanly (RC 0).
  **Zero** `FFEXP` / `FFCSSNR` / `FFNINT` diagnostic prints in the whole run.
- **Scope-out.** Real FF ops only — no `ff_complex.hpp`, no DD/QF backends, no
  per-op differential accuracy (that is the T2.4 sibling). `ff_math.hpp` /
  `ff_complex.hpp` untouched; demos untouched. No library defects surfaced — no new
  B-tasks.
- Depends on T0.1, T0.2, T2.0, T2.1.
- See `6cb2211` for the code diff and this DONE block for the outcome; **T2.5
  (FMA-contraction guard for FF) remains open as the final Phase-2 task.**

### Phase 3 — QF implementation + validation (8 tasks)

QF does not exist yet. Phase 3 = build + validate. Model after QD's
`qd_real.cc`.

**T3.0a: QF library — arithmetic and renormalization. (DONE)**

- Create branch `qffunKokkos` from `fffunKokkos` (inherits FF
  infrastructure).
- Follow the T0.4 naming convention: type + math under
  `namespace Kokkos::Experimental`, STL-style free-function names
  (`add`/`subtract`/`multiply`/`divide`/`negate`/`round_to_nearest_int`/
  `pow_int`), constants as `QuadFloat_pi()` etc., `QuadFloat::from_bits(...)`
  factory, and a bottom-of-header `namespace Kokkos` re-exposure block so
  `Kokkos::exp(qf)` works. FF gets the same treatment in T2.0.
- Create `third_party/include/qf_math.hpp` with:
  - `QuadFloat` struct: 4 × float components (f0, f1, f2, f3).
  - `renorm_4` (length-5 → length-4 Priest normalization; port from
    QD Hida-Li-Bailey Algorithm 3).
  - `add` (QD's `sloppy_add` + optional `ieee_add`; both from QD).
  - `subtract`.
  - `multiply` (16 partial products, keep down to weight u³, `renorm_4`).
  - `divide` (Newton, 3 iterations from FP32 → ~96 bits).
  - `sqrt` (Newton, 3 iterations).
  - `negate`, `abs`.
  - Constants (`QuadFloat_pi`, `QuadFloat_e`, `QuadFloat_log2`,
    `QuadFloat_log10`, `QuadFloat_sqrt2`, `QuadFloat_euler_gamma`) —
    regenerate from MPFR or quadmath as 4 × FP32 bit patterns (extend
    `scripts/gen_ff_constants.cpp` pattern).
- Every function `KOKKOS_INLINE_FUNCTION`, same style as
  `dd_math.hpp` and `ff_math.hpp`.
- **Preamble mandate: cluster-Claude must read `qd/src/qd_real.cc`
  from QD 2.3.24 upstream BEFORE porting.** Do not hallucinate QD
  internals. Cite QD source location in comments.
- Deliverable: `qf_math.hpp` compiles, arithmetic ops work, minimal
  standalone smoke test (like `scripts/test_ffmul.cpp` but for QF).

- Executed 2026-08-03. Branch `qffunKokkos` (forked from `main@b0d3cdf`);
  task commit `d47947b`; DONE block + docs-pointer `974744e`. First Phase-3
  task; first from-scratch implementation in the project (all prior tasks
  validated existing code).
- **Branching decision.** Created from `main` (not `fffunKokkos` — the
  plan-doc's original "from `fffunKokkos`" wording predated T2.0's merge of
  FF onto `main` behind `Kokkos::Experimental`, so `main` already carries the
  full FF infrastructure this branch inherits).
- **What shipped (6 files, +1332).** `qf_math.hpp` (756),
  `scripts/attic/test_qfmul.cpp` (243), `docs/PORT_NOTES_QF.md` (186),
  `scripts/attic/gen_qf_constants.cpp` (84),
  `LICENSES/LicenseRef-LBNL-BSD-License.txt` (61), `.gitignore` (+2).
- **QD 2.3.24 source citations (every non-trivial routine cites its QD
  location, in-header):** `renorm` / `renorm_4` (Priest/Alg-3,
  `qd_inline.h:127-177`), `three_sum` / `two_sum` / `two_prod` (QD
  by-reference EFTs, `qd/include/qd/inline.h`), `ieee_add`
  (`qd_inline.h:286-336`), `sloppy_add` (the active default,
  `qd_inline.h:338-405`), `multiply` = `sloppy_mul`
  (`qd_inline.h:567-599`), `sqr` (`qd_inline.h:674-715`), `divide` =
  `sloppy_div` / long division (`qd_real.cpp:693-712`), `accurate_div`
  (`qd_real.cpp:714-736`), `sqrt` = Heron (`qd_real.cpp:738-785`), `nint`
  (`inline.h:116-120`), `pow_int` (`qd_real.cpp:48-86`).
- **Two source-fidelity deviations (both accepted, both cited; rule 6):**
  - **`divide` = long division (not Newton) + `sqrt` = Heron (not Karp
    reciprocal-Newton).** The prompt described the wrong algorithms; QD
    2.3.24 actually uses classical long division for `divide` and Heron's
    method for `sqrt`. Ported QD's real code per the QD-source-fidelity rule
    (non-negotiable). Prompt drafting error corrected, documented in
    PORT_NOTES §0a/§0b.
  - **License = LBNL-BSD (not DHB).** The prompt said "copy ff's DHB
    header"; the plan-doc's own lineage rule says QF (QD-derived) →
    `LicenseRef-LBNL-BSD-License`. Cluster-Claude followed the plan-doc, not
    the prompt — a DHB attribution on a QD derivative would misattribute
    copyright.
- **Smoke-test results (`scripts/test_qfmul.cpp` vs `__float128`).** 5 ops
  PASS: add / subtract 29.00 / 29.00, multiply 29.00 / 28.42, divide 28.99
  / 27.78, sqrt 29.00 / 28.18 — all above the 28-digit target.
- **Newton/iteration-count justification.** `sqrt` Heron: 3 iterations
  (24 → 48 → 96 bits, saturating at QF width). `divide` long division: 4
  quotient digits (sloppy) / 5 (accurate), per-digit convergence
  24 → 48 → 72 → 96. Algebra in PORT_NOTES §0c.
- **PORT_NOTES_QF.md §1–§5 gotchas:** splitter reuse (§1),
  splitter-overflow limit not ported (§2), `sloppy_add` safety at the
  narrower FP32 exponent (§3), the FF `ffnint` bug does NOT recur at QF
  (§4), constant generation precision (§5).
- **Compile.** `qf_math.hpp` zero-warning under GCC 13.3.0 + Kokkos 5.1
  (C++20, `-Wall -Wextra -Wpedantic`). `dd_math.hpp` / `ff_math.hpp` /
  `ff_complex.hpp` untouched. CMake QF demo target deferred to T3.0b (a
  39-op demo needs transcendentals; no demo source exists yet).
- See `d47947b` for the code diff and this DONE block for the outcome.

**T3.0b: QF library — transcendentals. (DONE)**

- Add to `qf_math.hpp`:
  - `exp` (Taylor with argument reduction; more terms than QD's FP64
    version — derive term count for FP32-error-accumulation ~28-digit
    target, document in comments).
  - `log` (Newton on `exp`).
  - `log10`, `log2`, `log1p`, `exp2`, `exp10`, `expm1`.
  - `sin`, `cos`, `tan`, `sincos` (argument reduction, joint sin/cos
    doublings — apply PORT_NOTES §3a lesson from FF).
  - `asin`, `acos`, `atan`, `atan2`.
  - `sinh`, `cosh`, `tanh` (Taylor branch for `|a| < 0.5` per
    PORT_NOTES §3b).
  - `asinh`, `acosh`, `atanh` (Taylor branch for `|a| < 0.5` per
    PORT_NOTES §3c).
  - `pow`, `pow_int`, `hypot`, `fmod`, `remainder`, `copysign`, `fmax`,
    `fmin`, `fdim`, `fma`, `ceil`, `floor`, `round`, `trunc`,
    `round_to_nearest_int`.
- **Apply PORT_NOTES lessons proactively:**
  - `round_to_nearest_int`: don't use magic-constant trick with 2⁹⁵ —
    FP32 mantissa won't absorb it. Convert to FP64 or FP128 for
    rounding, like FF's
    `ffnint` does.
  - `exp` scaling: use direct scaling for final `ldexp`, not
    `multiply(s, ldexpf(1, nz))` — splitter would overflow.
- Deliverable: `qf_math.hpp` complete, `demo_qf_real.cpp` and
  `demo_qf_complex.cpp` (adapted from FF equivalents) run and produce
  ~28-29 digits accuracy against quadmath.

- Executed 2026-08-03. Task commit `d50099b` on `qffunKokkos` (branch tip
  advanced by exactly one commit vs T3.0a); DONE block + docs-pointer `974744e`.
- **What shipped (4 files, +1475 / −5).** `qf_math.hpp` (+511
  transcendentals), `src/demo_qf_real.cpp` (new, 804),
  `docs/PORT_NOTES_QF.md` (+156, §6–§11), `CMakeLists.txt` (+9, registers
  `kokkos_ep_demo_qf` mirroring the FF demo target).
- **QD 2.3.24 source citations per transcendental (60 citations grepped in
  `qf_math.hpp`):** `exp` (`qd_real.cpp:925-983`), `log`
  (`qd_real.cpp:986-1011`), `log10` (`qd_real.cpp:1025`), `sincos` /
  `sin` / `cos` (`qd_real.cpp:2298-2360`), `tan` (`qd_real.cpp:2473`),
  `atan2` / `angle` (`qd_real.cpp:2393-2460`), `atan` (`qd_real.cpp:2389`),
  `asin` (`qd_real.cpp:2479`), `acos` (`qd_real.cpp:2494`), `sinh` / `cosh`
  (`qd_real.cpp:2509-2547`), `tanh` (`qd_real.cpp`), `asinh`
  (`qd_real.cpp:2576`), `acosh` (`qd_real.cpp:2580`), `atanh`
  (`qd_real.cpp:2589`), `pow` (`qd_real.cpp:655`), `fmod`
  (`qd_real.cpp:2598`), `remainder` = `drem` (`qd_real.cpp:2462`), `floor`
  (`qd_real.cpp:136-157`), `ceil` (`qd_real.cpp:159-180`), `trunc`,
  `round` (QD `nint`, `qd_real.cpp:96`, T3.0a).
- **No-QD-analogue compositions honestly flagged.** `log2`, `log1p`,
  `exp2`, `exp10`, `expm1`, `hypot`, `copysign`, `fmax` / `fmin` / `fdim` /
  `fma` have no direct QD routine; each is composed from ported primitives
  and noted as such in-header (not passed off as a faithful port).
- **Demo results — 39/39 PASS on `demo_qf_real.cpp` vs `__float128`
  oracle.** Per-op means: arithmetic / rounding / data ops 29.00; log-family
  28.99; sin / cos 28.57, tan 28.89; asin 28.68 / acos 28.85 / atan 28.98;
  sinh / cosh 28.2, tanh 28.87; asinh / acosh 28.95–28.98, atanh 28.72; pow
  27.77; exp2 / exp10 26.79; exp 25.99 (min 10.5 — denormal tail at
  a ≈ −80, PORT_NOTES §5 / §10). An independent 2000-sample host probe agrees
  within 0.1 digit.
- **`exp` term-count derivation.** `|r| ≤ (log2/2)/2^nq`; need
  `|r|^N/N! < u = 2⁻⁹⁶`. With `nq=6` (`|r| ≤ 5.4e-3`), `N=11` terms. QD's
  FP64 `exp` uses `nq=16` + a 15-entry `inv_fact` table capped at < 9 terms;
  QF uses more Taylor terms but far fewer squarings (6 vs 16) and no
  factorial table. Full derivation in PORT_NOTES §7.
- **`sinh` / `cosh` threshold decision.** Kept at 0.5 (FF's), not shifted to
  QD's 0.05. Cancellation loss ≈ `log₁₀(1/|a|)`: ~0.3 digits at 0.5 vs ~1.3
  at 0.05. At QF's 29-digit budget the wider Taylor coverage is safer.
  Derivation in PORT_NOTES §8.
- **§4b remainder-sign observation.** No FP32 sign divergence observed at
  QF. `remainder` scores mean 29.00 vs oracle (reuses QD `nint` =
  `floor(d+0.5)`, which per T3.0a §4 avoids the FF `ffnint` bug; matches
  `std::remainderf`). No fix introduced. Documented in PORT_NOTES §9.
- **Source-fidelity deviation (accepted, called out prominently; rule 6):
  table-free port of `sin` / `cos` / `exp`.** QD 2.3.24 uses lookup tables:
  `inv_fact[15]` for `exp`, `sin_table[256]` / `cos_table[256]` for
  `sin` / `cos` / `sincos`. QF is table-free — Taylor with divide-by-k
  accumulating `1/k!` inline, joint `sincos` via doublings, no lookup. Three
  reasons: (1) the T3.0b task + PORT_NOTES §3a explicitly directed joint
  doublings + a "more terms than QD" Taylor; (2) 256-entry × 4-word FP32
  tables (2048 constants) would be device-hostile on the GPU/accelerator
  targets Kokkos serves; (3) QF's 4-word π makes mod-2π reduction accurate
  enough that tables aren't needed — bonus: near-π `sin` / `cos` beats FF's
  §5 conditioning ceiling.
- **T3.6 archaeology reminder.** QF is now algorithmically different from QD
  in its transcendental inner loops. When T3.6 validates QF against QD
  reference results (or any other QF-vs-QD comparison), differences are
  **expected and by design, not defects.** PORT_NOTES §6 documents the
  divergence in full.
- **`qf_complex.hpp` deferred to a new §T3.0c slot** (see below): the
  plan-doc names `demo_qf_complex.cpp` in the T3.0b deliverable but not the
  header, so the complex QF backend is carved out as its own task rather than
  half-shipped here.
- **PORT_NOTES_QF.md new sections:** §6 table-free divergence, §7 `exp`
  term count + coarse-eps to avoid the FF `exp`-stall bug, §8 `sinh`
  threshold, §9 remainder sign, §10 `exp` denormal tail / §4a scaling, §11
  demo verdict + measured accuracy.
- **Compile.** Zero-warning under GCC 13.3.0 + Kokkos 5.1 (C++20,
  `-Wall -Wextra -Wpedantic`); the only notes are pre-existing `-Wpedantic`
  Q-suffix notes from the Kokkos `__float128` oracle header (identical in the
  FF demo).
- **Acceptance-gate results.** Branch tip advanced by exactly one commit;
  `qf_math.hpp` + demo compile clean; `demo_qf_real` compiles + runs RC 0,
  mean ≥ 28 for well-conditioned ops; `main` ctest 14/16 green (400s Serial,
  the 2 pre-existing REDs T1.4 / T2.4 unchanged); every non-trivial
  transcendental cites its QD 2.3.24 location; FF PORT_NOTES §3a / §3b / §3c /
  §4a applied proactively; §4b documented; `dd_math.hpp` / `ff_math.hpp` /
  `ff_complex.hpp` untouched; `main` untouched (no QF files on `main`).
- See `d50099b` for the code diff and this DONE block for the outcome.
  `qf_complex.hpp` and `demo_qf_complex.cpp` deferred to new §T3.0c slot.

**T3.0c: QF complex library. (DONE)**

- Port `third_party/include/ff_complex.hpp` mechanically to QF, mirroring the
  DD→FF complex port done pre-T2.0 on `fffunKokkos`.
- Create `third_party/include/qf_complex.hpp` under
  `namespace Kokkos::Experimental` (STL-style API matching `ff_complex.hpp` /
  `dd_complex.hpp`): `QuadFloatComplex` struct with two `QuadFloat`
  components; all standard complex ops (add / subtract / multiply / divide /
  negate / conjugate / abs / norm / arg); complex transcendentals (exp, log,
  sin, cos, sinh, cosh, sqrt, pow, polar).
- Preserve the table-free posture from T3.0b — no `sin_table` / `cos_table` /
  `inv_fact` in the complex header either.
- Apply FF PORT_NOTES §3–§5 lessons proactively (same as T3.0b): bare-scalar
  promotions via literal-lift, `sqrt` internal ½ / 2 constants, ±1 via
  literal, real→complex imaginary padding via `literal(0)`.
- LBNL-BSD license header (same lineage rule as T3.0a).
- **Preamble mandate:** read QD 2.3.24's `qd/src/qd_complex.cpp` (if present)
  or `dd/src/dd_complex.cpp` cover-to-cover BEFORE porting. If QD lacks a
  complex header entirely and only FF's `ff_complex.hpp` is the reference,
  note that.
- Deliverable: `qf_complex.hpp` complete, `src/demo_qf_complex.cpp` (adapted
  from `demo_ff_complex.cpp`) runs and produces ~28–29 digits accuracy
  against quadmath.

- Executed 2026-08-04. Task commit `74f28a2` on `qffunKokkos` (branch tip
  advanced by exactly one commit vs T3.0b); DONE block + docs-pointer `01bdde9`.
  Completes Phase 3's build sub-phase (§T3.0a–c); T3.1 onward is validation.
- **What shipped (4 files, +1236 / −5).** `third_party/include/qf_complex.hpp`
  (new, 422): `Kokkos::Experimental::QuadFloatComplex` (two `QuadFloat`
  components) + full complex math, composing on the T3.0a/T3.0b `qf_math.hpp`
  real arithmetic and table-free transcendentals. `src/demo_qf_complex.cpp`
  (new, 703, adapted from `demo_ff_complex.cpp` + the T3.0b `demo_qf_real.cpp`
  verdict). `CMakeLists.txt` (+11 / −5, registers `kokkos_ep_demo_qf_complex`
  mirroring the FF-complex target + install entry). `docs/PORT_NOTES_QF.md`
  (+100, §12–§15 + the complex-layer license-lineage note).
- **Op inventory (all `KOKKOS_INLINE_FUNCTION`).** Standard ops
  (add / subtract / multiply / divide / negate / conjugate / abs / norm / arg)
  + transcendentals (sqrt, exp, log, log10, sin, cos, tan, asin, acos, atan,
  sinh, cosh, tanh, asinh, acosh, atanh, pow, polar) — the full
  `ff_complex.hpp` inventory. Every non-trivial function cites its
  `ff_complex.hpp` (and, where deeper, `dd_complex.hpp`) line range in-header.
- **Table-free posture preserved (T3.0b lineage).** No `sin_table` /
  `cos_table` / `inv_fact` in the complex header; every transcendental routes
  through the table-free real `qf_math.hpp` primitives. FF PORT_NOTES §3–§5
  lessons applied proactively (bare-scalar promotion via literal-lift, `sqrt`
  internal ½ / 2 constants, ±1 via literal, real→complex imaginary padding via
  `literal(0)`).
- **QD 2.3.24 source finding (preamble mandate).** The tarball ships **NO**
  `qd/src/qd_complex.cpp` or `dd/src/dd_complex.cpp` — a `grep -ril complex`
  over the whole 2.3.24 tree matches only NEWS / README / TODO / Fortran
  files; `qd/include/qd/` + `qd/src/` carry the `qd_real` / `dd_real` real
  types and the `c_dd` / `c_qd` C-linkage wrappers of the real types only. QD
  leaves quad-double complex for users to layer on. So `ff_complex.hpp` +
  `dd_complex.hpp` are the **sole** algorithm references (no QD complex routine
  exists to cite); PORT_NOTES §13.
- **Deviation 1 — `sincos` / `sinhcosh` OUTPUT-ARGUMENT ORDER swap (silent-
  wrong-answer trap; PORT_NOTES §12).** `ff_math.hpp` writes `sincos(a, x, y)`
  as `x = cos, y = sin` (and `sinhcosh` as `x = cosh, y = sinh`); `qf_math.hpp`
  names them **sin-first** (`sincos(a, sin_a, cos_a)`) / **sinh-first**. Porting
  `ff_complex.hpp`'s `sincos(z.im, c, s)` verbatim would bind `c` to sin and
  `s` to cos — every complex exp/sin/cos/sinh/cosh/tanh/polar would silently
  swap its components. Fixed by passing the local variables in **swapped
  positional order** at every call site (`sincos(z.im, s, c)`,
  `sinhcosh(…, sb, cb)`, …) so the local names stay identical to
  `ff_complex.hpp` (`c` = cos, `s` = sin, `cb` = cosh, `sb` = sinh) and the
  downstream algebra is byte-for-byte the same. Documented in the header
  preamble and inline at each call site; no numeric deviation once applied.
- **Deviation 2 — license lineage LBNL-BSD, not DHB (PORT_NOTES "License
  lineage (complex layer — T3.0c)").** Two precedents weighed: **(a)** follow
  scalar dispatch (LBNL-BSD, as T3.0a chose for `qf_math.hpp`) vs **(b)** follow
  the structural template (DHB, as `ff_complex.hpp` inherits from
  `dd_complex.hpp`'s DDFUN heritage). **Chose (a) LBNL-BSD.** The header
  contains no DDFUN/DHB-original arithmetic — the complex composition formulas
  are textbook identities and every non-trivial numeric step is a QD-derived
  QuadFloat op; a DHB header would misattribute copyright to Bailey personally
  for a file whose substance is the LBNL institutional QD package. Keeps the
  whole QF backend (`qf_math.hpp` + `qf_complex.hpp`) under one consistent
  license. Flagged in PORT_NOTES for review; Reet confirmed. (`NOTICE.md` gains
  a `qf_complex.hpp` LBNL-BSD row at the T3.x QF→`main` merge, out of scope.)
- **`norm` / `arg` are additions, not ports (PORT_NOTES §14).** `ff_complex` /
  `dd_complex` ship only `abs` + `conj` as standalone basic ops and expose the
  angle solely inside `log()`; `qf_complex.hpp` adds `norm(z) = re²+im²` (the
  *squared* magnitude — C++ `std::norm`, no sqrt) and `arg(z) = atan2(im, re)`
  per `std::complex` conventions, flagged in-header as inventory additions with
  no upstream line to cite.
- **Acceptance-gate results.** `src/demo_qf_complex.cpp` exercises all 24
  complex ops vs the `__complex128` oracle and returns RC 0 iff every op meets
  a mean ≥ 24.0-digit gate in BOTH real and imag components (conditioning-
  limited ops exempt — the complex list of PORT_NOTES §15: sub, div, tan,
  asin / acos / atan / atanh, log / log10, pow). Run `--batch 5000 --repeats 2`
  (Serial/host): **24 pass / 0 fail / 0 conditioning-exempt, RC 0** — every op
  cleared the gate on its own, so none needed the exemption. Well-conditioned
  means sit at the 28.12–29.00-digit QF ceiling; the lowest means (atan-imag
  26.81, tanh-imag 27.70, pow 27.8) are the branch-cut / compounded-conditioning
  cases, all still above the 24-digit gate. Full per-op real/imag table in
  PORT_NOTES §15.
- **Rule 4 respected.** `dd_math.hpp` / `dd_complex.hpp` / `ff_math.hpp` /
  `ff_complex.hpp` / `qf_math.hpp` all untouched (no needed helper surfaced);
  `main` untouched (no QF files on `main`). `main` ctest 14/16 green (the 2
  pre-existing REDs T1.4 `dd_accuracy` + T2.4 `ff_accuracy` unchanged).
- **Compile.** `qf_complex.hpp` zero-warning under GCC 13.3.0 + Kokkos 5.1
  (C++20, `-Wall -Wextra -Wpedantic`); the only notes are the 2 pre-existing
  `-Wpedantic` Q-suffix notes from the Kokkos `__complex128` oracle header,
  identical in count to `demo_ff_complex.cpp`.
- **Next task: T3.1 (EFT unit tests for QF)** — the first Phase-3 *validation*
  task, and the first test that *runs on* the QF backend rather than authoring
  it (`twoSum` / Dekker `twoProd` / `renorm_4` at primitive level).
- See `74f28a2` for the code diff and this DONE block for the outcome.

**T3.1: EFT unit tests for QF. (DONE)**

- Test `twoSum`, Dekker `twoProd` at primitive level (same as FF —
  QF reuses FF's primitives internally).
- If any tests already cover this from T2.1, cross-reference; add QF
  wrapper tests as needed.
- Also test `renorm_4` at the primitive level: input a random
  length-5 unnormalized expansion, verify output satisfies Priest
  invariant `|f_{i+1}| ≤ ½ ulp(f_i)` AND equals input as real number
  (within QF truncation threshold).

- Executed 2026-08-04. Task commit `3c40cf7` on `qffunKokkos` (branch tip
  advanced by exactly one commit vs T3.0c); DONE block + docs-pointer `15518e5`.
  First Phase-3 *validation* task — opens the T3.1–T3.6 validation sequence
  that runs *on* the QF backend rather than authoring it (T3.0a–c built it).
- **What shipped (3 files, +918 LOC).** `tests/qf_eft_test.cpp` (new, 861):
  Layer-1 EFT unit test calling the SHIPPED `qf_math.hpp` primitives directly.
  `tests/CMakeLists.txt` (+10, adds `kokkos_ep_add_eft_test(qf_eft_test)` — the
  contraction-OFF helper, same as `ff_eft_test` / `dd_eft_test`).
  `tests/README.md` (+47, registry row + a "QF EFT" section documenting the two
  structural deviations from T2.1).
- **EFT inventory covered (all shipped free functions in
  `Kokkos::Experimental`).** `qf_two_sum`, `qf_quick_two_sum`, Dekker
  `qf_two_prod`, `qf_two_sqr` (`qf_math.hpp:118-158`); `renorm` (length-4 →
  length-4) and `renorm_4` (length-5 → length-4) (`qf_math.hpp:182-257`). QD
  2.3.24 lineage: `qd/include/qd/inline.h` (twoSum / twoProd / two_sqr),
  `qd/include/qd/qd_inline.h` (renorm 4-word / 5-word).
- **Oracle strategy per EFT.** FP32 primitives use **plain FP64, provable-
  exact** (not merely higher-precision): the exact FP32 sum needs ≤25 bits and
  the exact FP32 product ≤48 bits, both inside FP64's 53-bit mantissa, so
  `(double)s+(double)e == (double)a+(double)b` (and the product form) is a
  *bit-equality* — no quadmath, runs unconditionally, same posture as T2.1.
  renorm/renorm_4 value-preservation uses a **two-tier** oracle: exact FP64 for
  53-bit-source inputs (renorm drops nothing → bit-exact `sum(b_i)==x`), plus a
  wide-spread `__float128` check (input = ordered decomposition of a ~113-bit
  value where renorm genuinely truncates the tail) verifying relative agreement
  within the QF truncation threshold (rel ≤ 2⁻⁸⁸; observed max ~1.6e-30 ≈ 2⁻⁹⁹).
- **Acceptance-gate verdict.** `ctest -R qf_eft` → **100% passed, 0 failed,
  11.5 s**. Zero-warning under GCC 13.3.0 + Kokkos 5.1 (C++20,
  `-Wall -Wextra -Wpedantic`). Per-EFT counts (tested / skipped / failed):
  `qf_two_sum` 2,113,526 / 335 / 0; `qf_quick_two_sum` 2,113,526 / 335 / 0;
  `qf_two_prod` 2,110,205 / 3,656 / 0; `qf_two_sqr` 2,111,573 / 2,288 / 0;
  `renorm_4` bounded 1,000,000 / 0 / 0; `renorm` bounded 1,000,000 / 0 / 0;
  `renorm_4` wide-spread 1,000,000 / 540 pair-skips / 0; named-case corner
  tests 21 / 1 skip / 0; device parity 600,000 / 0 / 0. Skips are out-of-domain
  (splitter overflow, subnormal operands, underflow tail), not failures.
- **Contraction-OFF posture confirmed** via the `kokkos_ep_add_eft_test` helper
  (same as `ff_eft_test` / `dd_eft_test`): Dekker `twoProduct` needs `a1*b1 - p`
  to be TWO distinct rounded ops, so `-ffp-contract=off` (host) / `--fmad=false`
  (CUDA) is mandatory or the error term collapses to zero. T3.5 later builds the
  contraction-ON reporter mirror; T3.1 needs only the OFF posture.
- **Corner-case coverage.** Zero, ±ulp, near-cancellation, subnormals, ±inf,
  and NaN-propagation-without-crash, as named hard cases. The device-parity
  block re-runs ordered-input EFTs through `Kokkos::parallel_for` (Serial
  backend here) against an exact-double per-element oracle → host/device bit-
  parity.
- **Deviation 1 — no mirror-and-comment.** T2.1 had to *duplicate* FF's twoSum /
  Dekker twoProduct into the test file because `ff_math.hpp` embeds them inside
  the longer `add()`/`multiply()` sequences (no standalone primitive to call).
  `qf_math.hpp` instead EXPOSES the shipped primitives as free functions in
  `Kokkos::Experimental`, so T3.1 calls the ACTUAL shipped code — strictly
  stronger than a mirror (which can drift from the header). Rule 4 trivially
  respected (only `#include`, never edit). Documented at
  `qf_eft_test.cpp:29-56`.
- **Deviation 2 — renorm inputs must be magnitude-ordered (T3.1-original
  methodology).** No FF analogue exists for `renorm_4`. renorm/renorm_4 are QD's
  renormalization step; their `quick_two_sum` cascade (`qf_quick_two_sum`
  requires |a| ≥ |b|) ASSUMES a magnitude-ordered "sloppy" expansion — exactly
  what `add`/`multiply`/`divide`/`QuadFloat(double)` produce. So the test
  generates every renorm input by successive FP32 decomposition of a wider
  scalar (FP64 for the exact oracle, `__float128` for the wide-spread oracle),
  yielding the ordered magnitude-decreasing shape callers feed. Documented at
  `qf_eft_test.cpp:69-95`.
- **Test-bug caught during development — NOT a library defect, no B-task.** The
  first-draft renorm generator fed ARBITRARY UNORDERED words, violating the
  `quick_two_sum` precondition, and produced spurious value/overlap failures.
  Fixed in the test (ordered decomposition, above). Three `__float128`-oracle
  probes confirmed `renorm`/`renorm_4` are CORRECT on properly-ordered input
  (max rel 1.6e-30 < u = 2⁻⁹⁶). Called out explicitly to distinguish a test-
  authoring lesson from a real `qf_math.hpp` defect — this is the former, so no
  follow-up B-task is warranted.
- **Cross-reference to T2.1 in-source** (`qf_eft_test.cpp:23-56`): the FF-EFT
  test (commit `4e025b0`) covers FP32 twoSum / Dekker twoProduct against
  bit-identical mirrors; T3.1 re-exercises the same primitives against the
  separately-compiled QF-side symbols (guards against a QF copy drifting from
  FF's) and ADDS `qf_two_sqr` + the renorm family.
- **Rule 4 respected.** No touch to `dd_math.hpp` / `dd_complex.hpp` /
  `ff_math.hpp` / `ff_complex.hpp` / `qf_math.hpp` / `qf_complex.hpp` /
  `PORT_NOTES_QF.md` — only test-side files changed (T3.1 is a test, not a
  port). `main` untouched; `qffunKokkos` not merged.
- **Next task: T3.2 (non-overlap invariant checks for QF)** — the T2.2 analogue
  for the QF backend; second Phase-3 validation task (Priest length-4 invariant
  across every QF op, 10⁶ inputs + corpus).
- See `3c40cf7` for the code diff and this DONE block for the outcome.

**T3.2: Non-overlap invariant checks for QF. (DONE)**

- Priest length-4 invariant: `|f_{i+1}| ≤ ½ ulp(f_i)` for i = 0,1,2.
- Every QF op, 10⁶ inputs + corpus.
- Report failures with 4-component bit patterns.

- Executed 2026-08-04. Task commit `4627336` on `qffunKokkos`; gate-flip commit
  `353193d`; PORT_NOTES §16 commit `104027c`; DONE block + docs-pointer `9da35c2`.
  Branch tip advanced by exactly four commits vs T3.1 (task, gate-flip, §16,
  DONE-block; +docs-pointer). Second Phase-3 *validation* task — the T2.2
  analogue for the QF backend, running the length-4 non-overlap invariant *on*
  the shipped `qf_math.hpp`/`qf_complex.hpp`.
- **What shipped (3 files, +1093 LOC).** `tests/qf_nonoverlap_test.cpp` (new,
  1021): the invariant test — carries NO oracle (a normalization defect is
  self-evident in the output words), runs unconditionally even on a
  quadmath-less Kokkos; `__float128` is used ONLY to enrich inputs to full
  ~96-bit width (KOKKOS_EP_HAVE_QUADMATH-gated). `tests/CMakeLists.txt` (+11,
  adds `kokkos_ep_add_test(qf_nonoverlap_test)` — the PLAIN helper, contraction
  posture irrelevant here). `tests/README.md` (+61, registry row + a "QF
  non-overlap" section).
- **Op inventory (54 host ops + 5 device tripwire).** vs T2.2's 50 FF host ops,
  **+4** = `ieee_add` / `sloppy_add` / `divide_accurate` (the QD-specific
  add/div variants qf_math exposes that FF does not) — the fourth delta is the
  `sqr` split from `multiply`. Breakdown: 29 unary + 4 joint-component
  (sincos.sin/.cos, sinhcosh.sinh/.cosh) + 17 binary + 4 special-form + 5 device
  tripwire (add/multiply/sqrt/exp/sin re-run through `Kokkos::parallel_for`,
  Serial backend).
- **Invariant form (deviation 1 vs T2.2).** T2.2 (FF) checks a length-2 invariant
  in **bit form**; T3.2 checks the mathematical ½-ulp form via `frexp` on the
  length-4 chain (`half_ulp = 2^(e-25)` for FP32) at i = 0,1,2. The
  mathematical-vs-bit form is the deliberate deviation for the longer expansion.
- **Two-tier classifier.** `classify_nonoverlap()` returns NOVL_OK
  (`|f_{i+1}| ≤ ½ ulp`), NOVL_WEAK (`½ ulp < · ≤ ulp`, the QD weak-normalization
  band), NOVL_FAIL (`> ulp`, packing break — nonzero word after a zero — or
  NaN/inf leak in a checkable slot). NOVL_FAIL is **always fatal**. Whether
  NOVL_WEAK is fatal is the single flag `kStrictPriestGate`, **default `false`
  (Shewchuk-weak) per PORT_NOTES §16** after Reet's review; `true` (strict
  Priest) is retained as a diagnostic switch. WEAK deviations are counted per-op
  with worst-ratio reported under **either** gate — nothing is hidden.
- **Acceptance-gate verdict (AFTER the §16 flip).** `ctest -R qf_nonoverlap` →
  **100% passed, 0 failed, 320.11 s** (~5m21s). Zero-warning under GCC 13.3.0 +
  Kokkos 5.1 (`-Wall -Wextra -Wpedantic`). *Historical strict-Priest counts*
  (`kStrictPriestGate = true`, same corpus, retained for the record): the test
  shipped deterministically RED — **19 WEAK / 0 FAIL** across 11,234,222 checked,
  worst ratio **1.375×** (fmod); those WEAK results are QD's baseline
  normalization form, not regressions (see PORT_NOTES §16).
- **Per-op counts (post-flip GREEN run).** 11,234,222 checked / 72,362 skipped /
  **0 NOVL_FAIL** / 19 WEAK (worst 1.3750). Device tripwire 500,000 checked
  (5 ops × 10⁵, exp skips its out-of-range tail), 0 FAIL. Corner cases 13/13
  (zero / ±ulp / subnormal / ±inf / NaN — the non-checkable ones pass as
  `inv=ok`). Skips are SKIP-not-fail: f0 NaN/±inf, f0 subnormal, underflow tail
  (`|f0| < 2⁻¹⁰⁰`), or out-of-math-domain input (log ≤ 0, asin |x|>1, …).
- **Corpus + input construction (deviations 2–3).** `kRandomN = 2×10⁵` per op
  **+ full corner corpus** (deviation 3 — reduced from the plan's 10⁶ to keep
  ctest at ~5m21s vs an extrapolated ~13.5 min; still 11.2M total checks). The
  random pass builds each input with `make_wide_input()` (reused from T3.1: a
  ~96-bit `__float128` → 4 magnitude-ordered FP32 words, leading word
  `f0 == (float)x` so domain predicates written against nominal `x` stay valid);
  the corpus pass uses the `QuadFloat(double)` constructor (deviation 2 — ordered
  inputs are mandatory because renorm's `quick_two_sum` cascade assumes
  `|a| ≥ |b|`; feeding unordered words would test renorm, T3.1's job, and raise
  spurious failures on correct passthrough ops).
- **Contraction-OFF NOT required.** Unlike T3.1/T3.5, this test uses the plain
  `kokkos_ep_add_test` helper, not `_add_eft_test`: the invariant reads only the
  *output* words' overlap structure, which does not depend on whether an internal
  Dekker product contracted — so the FMA posture is irrelevant here.
- **Benign diagnostics.** `angle` / `atan2` emit 18 benign `QFCSSNR: argument too
  large` guard prints on far-out-of-range corpus inputs (the QF analogue of FF's
  FFEXP guard) — informational, not failures. No tiny-argument trig lower bound
  is imposed (the invariant is well-posed down to the underflow tail).
- **Root-cause classification — WEAK is QD baseline, not a regression, not a
  B-task.** The 19 WEAK deviations are systemic (shared `renorm`/`renorm_4` path,
  a `quick_two_sum` cascade), match the published Shewchuk-weak bound, and are
  unfixable without replacing QD's normalization — barred by Rule 4. Filed as a
  **normalization-form clarification (PORT_NOTES §16)**, deliberately distinct
  from the §5 accuracy/conditioning registry; real precision loss is assessed in
  T3.4, not by this invariant.
- **Rule 4 respected.** No touch to any `*_math.hpp` / `*_complex.hpp` in any of
  the four commits (gate-flip is test-only; §16 + DONE block + docs-pointer are
  docs-only). `main` untouched; `qffunKokkos` not merged.
- **Next task: T3.3 (property/identity tests for QF)** — the T1.3/T2.3 analogue,
  adjusted for QF's ~29-digit precision; third Phase-3 validation task.
- See `4627336` (test), `353193d` (gate flip), `104027c` (§16) for the code/doc
  diffs and this DONE block for the outcome.

**T3.3: Property/identity tests for QF. (DONE)**

- Same identities as T1.3/T2.3, adjusted for QF's ~29 digit precision.

- Executed 2026-08-04. Task commit `2c25370` on `qffunKokkos`; DONE block +
  docs-pointer `d867ab3`. Branch tip advanced by exactly three commits vs T3.2
  (task, DONE block, docs-pointer — no gate-flip, no PORT_NOTES entry: nothing
  algorithmic surfaced). Third Phase-3 *validation* task — the T1.3/T2.3 analogue
  for the QF backend (**3/6 Phase-3 validation tasks done**: T3.1 EFT, T3.2
  non-overlap, T3.3 property).
- **What shipped (3 files, +1026 LOC).** `tests/qf_property_test.cpp` (new, 948):
  the algebraic-identity test — Group A carries NO oracle (a broken identity is
  self-evident in the 4-word `==`), Group B scores against `__float128`
  (`KOKKOS_EP_HAVE_QUADMATH`-gated, runtime-SKIP otherwise). `tests/CMakeLists.txt`
  (+14, adds `kokkos_ep_add_test(qf_property_test)` — the PLAIN helper,
  contraction posture irrelevant: identities are FMA-contraction agnostic).
  `tests/README.md` (+64, registry row + a "QF property/identity (Layer 3,
  Phase 3)" section).
- **Group A inventory — bit-exact, no oracle (~200 K random per identity + full
  finite corpus, raw 4-word `==`). 12 identities, 0 failures.** A1
  `add(a,negate(a))==0`; A2 `a-a==0`; A3 `a+0==a`; A4 `a-0==a`; A5 `a·1==a`
  (dom_dekker); A6 `a·0==0` → +0 (dom_dekker); A7 `a·(-1)==negate(a)` (dom_dekker);
  A8 `negate(negate(a))==a`; A9 `abs` sign branches; A10 `abs(-a)==abs(a)`; A11
  **add commutativity on WIDE 4-word operands** (a QF strengthening — see below);
  A12 `mul_pwr2(mul_pwr2(a,2ᵏ),2⁻ᵏ)==a` ±2ᵏ round-trip. Skips are the corpus's
  FLT_MAX / splitter-overflow (`|x|≥FLT_MAX/8193≈4.15e34`) / denormal-tail
  (`|f0|<2⁻¹⁰⁰`) entries — range limits, not defects (SKIP-not-fail).
- **Group B inventory — tolerance vs `__float128`, MEAN-gated. 15 identities, all
  clear their gates.** (tol ulp / min-digits / mean-digits): B0 mul commutativity
  demoted (10 / 28.30 / 29.00); B1 `sqrt(a)²≈a` (10 / 14.34 / 28.03); B2
  `exp(log a)≈a` (**30 §10** / 26.48 / 27.94); B3 `log(exp a)≈a` (10 / 26.35 /
  28.77); B4 `sin²+cos²≈1` (10 / 26.89 / 28.39); B5 `cosh²−sinh²≈1` (10 / 24.59 /
  28.20); B6 `sin(a+b)` formula (10 / 23.59 / 28.28); B7 `cos(a+b)` formula (10 /
  23.77 / 28.29); B8 `tanh≈sinh/cosh` (10 / 0.00 / 28.86); B9 `asin(sin a)≈a` (10 /
  26.80 / 28.68); B10 `atan(tan a)≈a` (10 / 27.22 / 28.69); B11 `pow(a,2)≈a·a`
  (**30 §10** / 26.19 / 27.79); B12 `sqrt(a²)≈|a|` (10 / 28.24 / 29.00); B13
  `hypot²≈a²+b²` (10 / 27.82 / 28.98); B14 `exp(a+eps)≈exp(a)(1+eps)` (10 / 26.40 /
  28.06). **Mean-gated posture:** the low mins on B1/B8 are conditioning-driven
  (division near 0, sqrt at the denormal edge), while the means stay high — the
  gate is on the mean by design.
- **Tolerance model (deviation from T2.3).** QF uses an **absolute ulp-of-U =
  2⁻⁹⁶ floor** — `digits(k ulp) = 96·log10(2) − log10(k)`, so **10 ulp = 27.90
  digits, 30 ulp = 27.42** — NOT the DD/FF `−log10(N·u²)` *statistical* floor. QF
  is a quad-word whose resolution IS U, so the statistical formula (which assumes
  a double-word `u²` error model) would hand back a much looser ~23.6 floor that
  would hide real drift. The absolute floor is also N-independent, so the
  wall-time-driven `kRandomN` reduction does not move the gate. A genuine
  methodological improvement over T2.3, documented in-source.
- **The two 30-ulp gates cite PORT_NOTES_QF §10** (exp output-denormal tail): B2
  `exp(log·)` and B11 `pow(x,2)` — the latter compounds `exp(y·log·)` conditioning
  on top of the same tail. Marked EXEMPT via the in-source
  `lookup_expected_min_drop("exp")` annotation. No new PORT_NOTES entry invented,
  no tolerance silently loosened; the §10 tail is a pre-existing, already-documented
  characterization.
- **B14 = the T2.3 B4 regression guard — does NOT recur in QF.** FF's B4 was a
  real `ff_math.hpp` bug (exp Taylor convergence `eps=1e-15f` finer than FF's
  ~3.55e-15 resolution → iteration-cap stall / wrong-0 return). `qf_math.hpp`
  exp uses `eps=1e-28f`, deliberately coarser than QF's U=2⁻⁹⁶ (an authoring-time
  fix per PORT_NOTES_QF §7/§10). B14 holds to mean 28.06 with **no stall** and
  **zero** diagnostic prints. **No QF B-task filed** — B14 stays as a durable
  regression guard against the defect ever re-porting.
- **A11 = a QF strengthening over FF/DD.** Add commutativity is bit-exact on
  FULL-WIDTH 4-word operands (verified 0/3×10⁶ mismatches), where FF/DD could only
  claim it for single-word operands (their `+a.lo+b.lo` tail reorders under swap).
  A real QF property upgrade promoted into Group A, not a spec bug. Only `a·b==b·a`
  still reorders (Dekker cross-term) → demoted to Group B (B0), exactly as in T1.3/
  T2.3.
- **Test C corner cases: 5/5 pass** (target ≥27 of 29 digits; C6 euler_gamma/
  digamma SKIPPED — no QF digamma op / no independent oracle). **Device pass: 3
  Group A + 2 Group B, all PASS** (Serial; `parallel_for` + `KOKKOS_LAMBDA` ships
  4 float words back). **0 diagnostic prints** in the whole run.
- **Deviations from T2.3 (all justified, all in-source).** (1) Tolerance model =
  absolute ulp-of-U floor, not the `−log10(N·u²)` statistical floor. (2) A11 add
  commutativity promoted into Group A on wide operands. (3) `kRandomN = 2×10⁵`
  (same wall-time reduction as T3.2; the absolute ulp floor is N-independent so
  the gate is unaffected). (4) Real QF ops only — `qf_complex` out of scope,
  matching T2.3's real-only posture.
- **Acceptance-gate verdict.** `ctest qf_property_test` → **PASS, 190.84 s**
  (~3m11s). Zero-warning under GCC 13.3.0 + Kokkos 5.1 (`-Wall -Wextra
  -Wpedantic`).
- **RED cases / B-tasks: none.** No library defects surfaced. No new §16-style
  algorithmic characterizations. No tolerance-setting judgment calls needed
  beyond the documented model.
- **Rule 4 respected.** No touch to any `*_math.hpp` / `*_complex.hpp` /
  `PORT_NOTES_QF.md` — only test-side files changed. `main` untouched;
  `qffunKokkos` not merged.
- **Next task: T3.4 (differential accuracy for QF vs quadmath)** — the T2.4
  analogue, the heavy Phase-3 task where real precision defects are expected to
  surface (T2.4 shipped RED with B5/B6/B7 for FF erf/erfc/tgamma). Ready to draft
  when Reet asks.
- See `2c25370` for the code diff and this DONE block for the outcome.

**T3.4: Differential accuracy for QF vs quadmath. (DONE)**

- Original stub (superseded by what shipped, kept for provenance): Per-op
  `max(rel_err / u⁴)` where `u = 2⁻²⁴`; quadmath oracle (5-digit headroom); MPFR
  ≥150 bits as an optional secondary oracle behind a CMake flag; Hida-Li-Bailey
  `u⁴` bounds per op; report min AND mean. **Deviations, all deliberate:** (1) the
  `rel_err / u⁴` framing is a double-word-precision-squared error model — wrong for
  a *quad-word*. QF's resolution IS `U = 2⁻⁹⁶` (four FP32 limbs ≈ 96 bits), not
  `u⁴`, so T3.4 gates on T3.3's **absolute ulp-of-U** floor (`digits(k ulp) =
  96·log10(2) − log10(k)`; 10 ulp = 27.90, 30 ulp = 27.42), not `u⁴`. (2) **MPFR
  not added:** the __float128 oracle carries ~34 digits, ~5 above QF's ~29 — ample
  pass/fail headroom, exactly as the stub allows ("adequate for pass/fail"); a
  second oracle behind a CMake flag was unnecessary and no op needed tight-bound
  verification. (3) `u = 2⁻²⁴` in the stub is FF's roundoff (copy-paste from the
  T2.4 template); QF's is `2⁻⁹⁶`.

- Executed 2026-08-05. Task commit `faf35d5` on `qffunKokkos`; DONE block +
  docs-pointer `7792dc0`. Fourth Phase-3 *validation* task — the T2.4/T1.4
  analogue for the QF backend (**4/6 Phase-3 validation tasks done**: T3.1 EFT,
  T3.2 non-overlap, T3.3 property, T3.4 accuracy).
- **What shipped (3 files, +~1040 LOC).** `tests/qf_accuracy_test.cpp` (new, ~900):
  the differential-accuracy test, whole file `#ifdef KOKKOS_EP_HAVE_QUADMATH` with
  a runtime-SKIP-77 fallback. `tests/CMakeLists.txt` (+11, adds
  `kokkos_ep_add_test(qf_accuracy_test)` — PLAIN helper, not EFT: accuracy scoring
  has no contractible Dekker adjacency to protect). `tests/README.md` (registry
  row + a "QF differential accuracy (Layer 4, Phase 3)" section).
- **Op surface: 49 scored + 5 skipped = T3.2's 54-op ceiling** (reconciles exactly).
  Scored: 29 unary (abs/negate/sqr/sqrt, round-family ×5, exp-family ×4, log-family
  ×4, trig ×3, inverse-trig ×3, hyperbolic ×3, inverse-hyperbolic ×3), 4 two-output
  components (sincos.sin/.cos, sinhcosh.sinh/.cosh — SIN/SINH first per §12), 13
  binary (add/subtract/multiply/divide/pow/atan2/hypot/fmod/remainder/copysign/
  fmax/fmin/fdim), 3 custom (multiply_scalar/fma/pow_int). Skipped with in-source
  rationale: `sloppy_add` (add() is its public alias), `ieee_add` (internal, no
  distinct public op), `divide_accurate` (internal, divide() wraps it), `mul_pwr2`
  (exact power-of-2 scaling; exact by construction, covered bit-exactly by T3.3
  A12), `angle` (identical to atan2). **vs T2.4 (~50 scored):** the counts match
  once QF's arithmetic variants are folded (sloppy_add/ieee_add/divide_accurate
  skipped as non-distinct) and QF's absent erf/erfc/tgamma are dropped — see below.
- **Three passes per op, four regimes.** narrow random (Route-A `QuadFloat(double)`
  over the op's §11 demo domain) + broad random (same domain enriched to a full
  ~96-bit ordered QuadFloat via `make_wide_input`, copied from `qf_nonoverlap_test`)
  + corpus/near-edge (the PORT_NOTES §3/§4 named accessor where one exists —
  `exp_overflow`, `trig_near_pi`, `sinh_cosh_small`, `atanh_small`,
  `remainder_regression` — else the generic bundler). Combined per op: min over all,
  mean count-weighted. Out-of-domain corpus values SKIP (counted, not failed).
- **Oracle at the exact widened input (a QF-specific twist over T1.4/T2.4).** DD/FF
  evaluate the oracle at the nominal `(float128)x` — exact for them because their
  precision is coarser than a double. QF's ~29 digits are FINER than a double, so
  T3.4 evaluates the oracle at `qf_to_q(input)` (sum of the four input words). The
  narrow regime coincides with `(float128)x` exactly; the broad regime carries
  sub-double bits, where the widened reference is the only honest choice.
- **Tolerance model = T3.3's absolute ulp-of-U floor, MEAN-gated.** 10 ulp = 27.90
  (default); 30 ulp = 27.42 for the exp-family output-denormal tail (PORT_NOTES_QF
  §10): exp/exp2/exp10, expm1 kept at 10 ulp (result ≥ −1, no denormal tail), and
  **pow** (see below). Low mins for the shared PORT_NOTES §5 registry
  (`lookup_expected_min_drop`) report EXPECTED-MIN-DROP: sub/fma/asin/acos/atanh/
  remainder/exp/sin/cos/tan (add is not in the registry).
- **exp-family domain narrowing (documented deviation from §11).** §11 samples exp
  over [−80,80] → mean 25.99 (its negative tail lands in the FP32 output-denormal
  band, §10, below even the 30-ulp floor). To gate the MEAN honestly the random
  passes narrow the negative end to keep the quad-word result in FP32 normal range
  (exp [−35,80], exp2 [−50,120], exp10 [−15,30]); the excluded tail is the §10
  limit (min-drop exempt), exercised at the HIGH edge by `exp_overflow`. This is
  why T3.4's exp mean (28.19) is higher than §11's 25.99 — **not** a §11 contradiction,
  a deliberate honest-domain choice, documented in-source; §11's per-op numbers are
  NOT modified.
- **Round-family: `Kokkos::round/ceil/floor/trunc` oracle (deviation from T1.4/T2.4).**
  qf round/round_to_nearest_int use round-**half-up** (floor(d+0.5)); T1.4/T2.4 used
  a `nearbyint` ties-to-even oracle because DD/FF nint rounds to even. On continuous
  random inputs (and the generic corpus, no exact half-integers) all conventions
  agree — ties are measure-zero — so the oracle matches `src/demo_qf_real.cpp`. The
  `nint_half_integer` corpus is deliberately NOT used (exact-tie semantics differ per
  rounding, out of scope here).
- **The one judgment call: pow → 30-ulp (§10) tier, EXEMPT.** Under the 10-ulp
  default pow lands mean **27.76** (min 25.81) — 0.14 digits under the 27.90 gate,
  the sole would-be RED. Root cause is NOT a defect: `qf::pow(a,b) = exp(b·log a)`,
  so pow's accuracy is bounded above by the internal exp (mean 28.19, already
  §10-gated at 30 ulp) and drops ~1.2 further digits under the pow relative
  condition number `κ = |b·ln a|` (Higham §3.4; reaches ~15 over the demo domain
  a∈[0.5,20], b∈[0.1,5]) — 28.19 − log10(κ) ≈ 27.76, the accuracy of a *correct*
  exp-log pow. So pow is an exp-family op and is gated at 30 ulp under the **same
  §10 output-denormal-tail limit** its internal exp obeys — "documented conditioning
  limit → cite §10, mark EXEMPT," not silently loosened (cited inline + in header).
  **Flagged for Reet:** if pow is instead to be held to the 10-ulp default, it
  becomes a RED to investigate; `lookup_expected_min_drop` is intentionally NOT
  consulted for pow (no registry mutation).
- **Per-op results (min / mean / gate, all PASS).** Structural (bit-exact, mean
  29.00): abs, negate, add, subtract, copysign, fmax, fmin, fdim, multiply_scalar.
  Arithmetic ~29.00: sqr, multiply, divide (28.99), hypot (29.00), sqrt (29.00).
  round-family 29.00. log-family 28.86–28.91 (min 21.67, near-1 conditioning).
  trig 28.56–28.88 (sin/cos min ~21.8 near ±π → EXP-MIN-DROP; tan/asin min ~0 →
  EXP-MIN-DROP). hyperbolic 28.21–28.87. inverse-hyperbolic 28.71–28.98 (atanh
  EXP-MIN-DROP). exp-family: exp 28.19 (EXP-MIN-DROP, 30-ulp), exp2 28.15, exp10
  28.11, expm1 28.54. fma 29.00 (min 28.21, EXP-MIN-DROP). remainder 28.97 (min
  22.94, EXP-MIN-DROP); fmod 28.98. atan2 28.82; pow_int 28.88. **pow 27.76 (30-ulp
  tier, PASS).** Lowest mean = pow (27.76), then exp10/exp2/exp/sinh family
  (28.1–28.2). ~9.8M inputs scored per op-family; total_skipped 1000 (out-of-domain
  corpus + pow 0^neg + splitter guards), failures **0**, device_failures **0**.
- **Device parity.** Every pass already runs through `Kokkos::parallel_for` on the
  Serial `DefaultExecutionSpace` (the whole test IS the device path). An explicit
  checkpoint re-runs add/multiply/sqrt/exp/sin on 5×10⁴ fresh-seeded inputs — 5/5
  PASS (exp 28.19 at its 30-ulp gate).
- **`kNarrowN = kBroadN = 10⁵` (2×10⁵ random/op)**, tuned down from the plan's 10⁶
  for the same wall-time reason as T3.2/T3.3; the absolute ulp floor is
  N-independent so the reduction does not shift any gate.
- **Acceptance-gate verdict.** `ctest qf_accuracy_test` → **PASS, 310.35 s**
  (~5m10s). Zero-warning under GCC 13.3.0 + Kokkos 5.1 (`-Wall -Wextra -Wpedantic`).
- **RED cases / B-tasks: none.** `qf_math.hpp` has **no** erf/erfc/tgamma, so T2.4's
  three REDs (its B5/B6/B7 — erf/erfc large-|z| + FP32 Lanczos) have no QF
  counterpart and cannot recur. The one sub-gate op (pow) is a documented
  conditioning limit (§10), EXEMPT, not a defect — no B-task filed. Contrast T1.4
  (3 REDs) and T2.4 (3 REDs): T3.4 ships **clean GREEN** because the QF port never
  ported the three defective special functions.
- **§11 discrepancies.** T3.4's exp mean (28.19) exceeds §11's exp mean (25.99)
  because T3.4 narrows the exp negative domain out of the §10 denormal tail (§11
  samples [−80,80] including it). This is a *domain* difference, not a defect and
  not a §11 error — §11's numbers stand unmodified; the difference is documented
  in-source and here.
- **Rule 4 respected.** No touch to any `*_math.hpp` / `*_complex.hpp`; §5 not
  extended; §11 numbers unchanged; no new PORT_NOTES_QF section created. Only
  test-side files changed. `main` untouched; `qffunKokkos` not merged.
- **Next task: T3.5 (FMA-contraction guard for QF)** — the T1.5/T2.5 analogue over
  QF's Dekker sequences (`qf_two_prod`/`qf_two_sqr`), the contraction-ON reporter
  mirror of T3.1's `qf_eft_test`. Ready to draft when Reet asks.
- See `faf35d5` for the code diff and this DONE block for the outcome.

**T3.5: FMA-contraction guard for QF. (DONE)**

- Same as T1.5/T2.5 for QF's Dekker sequences.

- Executed 2026-08-05. Task commit `c71690b` on `qffunKokkos`; DONE block +
  docs-pointer this bundle (T3.4-style two-commit clean-GREEN close). Fifth Phase-3
  *validation* task — the T1.5/T2.5 analogue for the QF backend (**5/6 Phase-3
  validation tasks done**: T3.1 EFT, T3.2 non-overlap, T3.3 property, T3.4 accuracy,
  T3.5 FMA-guard).
- **What shipped (4 files, +830 LOC).** `tests/qf_fma_guard_test.cpp` (new, +714):
  the Layer-5 positive test of the FMA-contraction posture QF's EFTs depend on.
  `tests/qf_fma_guard_baseline.txt` (new, +31): the contraction-ON drift baseline,
  armed at `0`. `tests/CMakeLists.txt` (+13): the two-target dual registration.
  `tests/README.md` (+72): two registry rows + a "QF FMA-contraction guard (Layer 5,
  Phase 3)" section.
- **Two-target dual-registration pattern (OFF gate + ON reporter).**
  `kokkos_ep_add_eft_test(qf_fma_guard_test)` builds `qf_fma_guard_test`
  (`-ffp-contract=off`, `KOKKOS_EP_CONTRACTION_MODE=0`, **fail-gates**);
  `kokkos_ep_add_eft_test_contract_on(qf_fma_guard_test)` builds
  `qf_fma_guard_test_contract_on` (`-ffp-contract=fast`,
  `KOKKOS_EP_CONTRACTION_MODE=1` + `KOKKOS_EP_BASELINE_PATH`, **reports**). Both
  helpers are the shared ones introduced by T1.5 and generalized by T2.5 (baseline
  path derived from target name) — **no helper churn in T3.5**, single-source /
  two-targets exactly as DD/FF.
- **Per-op acceptance-gate numbers.** `qf_two_prod` (`a1*b1 − p`): **511,988 checks**
  (host + device). `qf_two_sqr` (`hi*hi − q`): **399,286 checks** (host + device).
  `F = ERR_ZERO + ERR_NONZERO_WRONG = 0` under **both** postures; **ERR_NONZERO_WRONG
  = 0** under both postures. `qf_two_sum` CONTROL (contraction-immune) exact under
  both postures (260,121 tested, 84 wide-exponent sums skipped by the FP32
  oracle-faithful guard). No named-case corner failure (11 pass / 3 skip / 0 fail).
  `ctest -R qf_fma_guard` → **2/2 PASS, 0.85 s** (#21 / #22, 0.42 s each).
- **GCC 13.3.0 contraction observation.** On baseline x86-64 (no `-mfma` /
  `-march=native`) GCC does **NOT** contract `a1*b1 − p` (qf_two_prod) or `hi*hi − q`
  (qf_two_sqr): `ERR_ZERO = 0` for both under `-ffp-contract=fast`. Same outcome as
  T1.5 (DD) and T2.5 (FF) — `-ffp-contract=off` is belt+suspenders on this ISA
  target. Because the test calls the SHIPPED primitives directly (see below), this
  characterizes GCC's contraction of `qf_math.hpp`'s own source, not a copy.
- **Baseline armed at `0`** in `tests/qf_fma_guard_baseline.txt`: each ON run compares
  its live `F` to the baseline and prints `baseline: OK` or `*** DRIFT ***`
  (WARN-only, never a failure). Drift means the toolchain's contraction behavior
  changed (compiler upgrade, new FMA-bearing ISA target, or flag change) — a signal
  to investigate, not a gate.
- **Deviation 1 — no mirror-and-comment; calls the SHIPPED `qf_math.hpp` primitives
  directly** (`qf_two_prod` / `qf_two_sqr` / `qf_two_sum`, `qf_math.hpp:118-158`).
  T2.5's FF guard mirrored the `twoProduct` sequence because `ff_math.hpp` embeds it
  inside `multiply()` with no free-function handle; `qf_math.hpp` exposes the EFTs as
  free functions, so T3.5 compiles the shipped source under the ON flags. Precedent:
  T3.1's `qf_eft_test` took the identical divergence from T2.1's `ff_eft_test`.
  Strictly stronger — a mirror can drift from the shipped code; this cannot.
- **Deviation 2 — `qf_two_sqr` is ALSO guarded** (`hi*hi − q`, a second Dekker
  sequence with its own contraction hazard). FF exposes no squaring EFT, so T2.5 was
  `twoProduct`-only. Matches T3.1's op surface (T3.1 likewise added `qf_two_sqr` over
  T2.1).
- **Design refinement over T2.5's binary F — three-way classification, ratified
  in-code.** Each observable input is bucketed TRIVIAL (`e_ref == 0`, the error term
  is legitimately zero) / **ERR_NONZERO_CORRECT** (Dekker identity holds) /
  **ERR_ZERO** (error term collapsed to zero, the contraction signature) /
  **ERR_NONZERO_WRONG** (error term nonzero but violates the Dekker identity — a
  genuinely-broken term). This distinguishes "contracted-to-zero" (informative — a
  correct implementation and a contracted one both yield `lo == 0` when `e_ref` is
  legitimately zero, so ZERO is not per se a fault of `qf_math.hpp`) from
  "contracted-to-wrong" (the ONE real bug case). Gate policy: **OFF fails on
  `ERR_ZERO || ERR_NONZERO_WRONG`** (the strict contraction gate); **ON PASSes iff
  `ERR_NONZERO_WRONG == 0`** (any ZERO/CORRECT mix acceptable — ZERO is informative,
  not a fault). T2.5's binary F could not tell the two apart and exited 0 on all
  contraction; T3.5 can, and fails the ON variant only on the genuinely-broken
  bucket. On this toolchain all buckets but CORRECT are 0, so ON passes either way.
- **FP64 oracle, no quadmath gate.** Ground truth is exact FP64: an FP32 product
  needs ≤48 bits ≤ FP64's 53-bit mantissa, so the `(hi, lo)` decomposition reference
  is algebraically exact and contraction-immune — no external library. Both variants
  therefore run **unconditionally** (no `KOKKOS_EP_HAVE_QUADMATH` gate, no runtime
  SKIP-77). Same posture `ff_fma_guard_test` / `qf_eft_test` established.
- **Rule 4 respected.** `qf_math.hpp` / `qf_complex.hpp` untouched (only `#include`d);
  no other `*_math.hpp` / `*_complex.hpp` touched. **PORT_NOTES_QF §5 NOT extended** —
  FMA contraction is a compiler characterization, not input conditioning; `tests/README.md`
  documents it correctly, so no PORT_NOTES drift. `main` untouched; `qffunKokkos` not
  merged.
- **Scope-out.** `qf_two_prod` + `qf_two_sqr` only — no complex EFTs, no other ops;
  `qf_two_sum` is a contraction-immune CONTROL, not a target under test. No new
  B-task stubs — 0 misses, nothing to log (clean GREEN, like T3.4).
- Depends on T0.1, T0.2, T3.0a, T3.0b, T3.1.
- **Phase 3 validation: 5 of 6 tasks DONE** (T3.1, T3.2, T3.3, T3.4, T3.5). T3.6
  remains (e2e cancellation kernels).
- See `c71690b` for the code diff and this DONE block for the outcome.

**T3.6: End-to-end cancellation kernels for QF. (DONE)**

- Same kernels as T1.6/T2.6, expect ~29 digits accuracy.

- Executed 2026-08-05. Task commit `64aac2d` on `qffunKokkos`; DONE block +
  docs-pointer this bundle (T3.5-style two-commit clean-GREEN close). **Sixth and
  final Phase-3 validation task — the T1.6/T2.6 analogue for the QF backend.**
- **What shipped (3 files, +511 LOC).** `tests/qf_cancellation_test.cpp` (new,
  +462): the Layer-6 end-to-end test — four classic cancellation-hostile kernels
  evaluated in QF (4×FP32, ~29 digits) and scored in digits of accuracy against
  `__float128` / closed-form oracles, mean-gated at 26.0 digits. `tests/CMakeLists.txt`
  (+12): the plain-helper registration + contraction rationale comment.
  `tests/README.md` (+37): registry row + a "QF end-to-end cancellation (Layer 6,
  Phase 3)" section. Structure mirrors T1.6 (`dd_e2e_test`) / T2.6
  (`ff_cancellation_test`) verbatim; the mechanical change is the precision scale
  (QF `kMaxDig = 29` vs DD's 31 / FF's 14).
- **The four kernels (host-side, inherently serial reductions/recurrences).**
  `K1` = `√(x²+1) − x` at x ∈ {1e2, 1e4, 1e6}; `K2` = `Σ 1/k²`, k=1..10⁶ (Basel,
  oracle π²/6); `K3` = Machin's `π = 16·atan(1/5) − 4·atan(1/239)` (oracle
  `QuadFloat_pi()`); `K4` = `Σ (−1)^(k+1)/k`, k=1..10⁶ (alternating harmonic, oracle
  ln 2). Whole file is `#ifdef KOKKOS_EP_HAVE_QUADMATH`; runtime-SKIP 77 without
  libquadmath. `qf_math.hpp` / `qf_complex.hpp` NOT modified (rule 4);
  `qf_complex.hpp` NOT included (real-only, matching T1.6/T2.6 discipline).
- **Two-oracle strategy (K2, K4).** Same split as T1.6/T2.6. The
  QF-vs-quadmath-partial-sum comparison carries the arithmetic-precision claim:
  identical N, identical summation order, identical terms, so it isolates QF's
  accumulation quality from truncation (gated at `kTol = 26.0`). The
  QF-vs-closed-form comparison (K2 vs π²/6, K4 vs ln 2) is a truncation-limited
  sanity check, gated at `truncation_floor − 1 = 5.0`. At N=10⁶ the floor is ~6
  digits: the Basel tail Σ_{N+1}^∞ 1/k² ≈ 1/N ≈ 1e-6, and the alternating-series
  error is bounded by the first omitted term ≈ 1/N.
- **Tolerance rationale.** `kTol = kMaxDig − 3 = 29 − 3 = 26.0`, the SAME "cap − 3"
  formula T1.6 (`31 − 3 = 28.0`) and T2.6 (`14 − 3 = 11.0`) used — ~3 digits of
  headroom for accumulated round-off in composed / 10⁶-term kernels, applied
  uniformly to the arithmetic-precision comparisons. Computed from `kMaxDig` at
  compile time, not hardcoded.
- **QF-local harness helpers (correct call, avoids shared-harness churn).**
  `test_utils.hpp` carries `BackendTraits<DD>` and `<FF>` but NOT `<QF>`
  (`test_utils.hpp:81` is a `TODO(Phase 3)` placeholder; the primary template is
  undefined). Rather than touch the shared harness other tasks own (rule 1/4), this
  file carries the QF-local `kMaxDig = 29.0` / `qf_to_q` / `qf_digits` helpers
  directly — IDENTICAL to `qf_accuracy_test` (T3.4) and `qf_property_test` (T3.3),
  which established the pattern. `kMaxDig` matches those files and
  `src/demo_qf_real.cpp`.
- **K2/K4 iteration counts.** Kept at N = 10⁶. The smallest term (1/N = 1e-6 for K4,
  1/N² = 1e-12 for K2) sits ~15 decades above QF's `u = 2⁻⁹⁶ ≈ 1.3e-29`, so no term
  stalls into the precision floor — the FP32-narrow term-stall concern that gated
  T2.6's calibration does not bite QF at all (QF's `u` is far finer than FF's).
- **K1 baseline = FP32, not FP64 (same rationale as T2.6).** T1.6 compared naive-DD
  against naive-FP64 (DD's 1-word base). The faithful QF mirror compares naive-QF
  (4×FP32) against naive-**FP32** (QF's 1-word base). FP64 (~16 digits) can be
  *wider* than the target-scaled QF quad-word for the small-x cases, so an FP64
  baseline would misrepresent QF's lift — the honest lift is over the 1-word base
  scalar. `K1_stable = 1/(√(x²+1)+x)` is the GATED DUT (algebraically
  cancellation-free); `K1_naive_report` is REPORTED, not gated.
- **K1 magnitude set {1e2, 1e4, 1e6} — pilot-calibrated (per the T3.6 prompt).**
  The prompt flagged that if K1_naive showed uniform ~29-digit reads at this set (no
  gradient), it should extend upward to {1e2, 1e6, 1e10}. A pilot confirmed the
  cancellation gradient IS present at {1e2, 1e4, 1e6}: naive-QF reads
  {28.23, 23.50, 23.87} (NOT uniform 29), so no extension is needed. The "+1"
  collapse that drives the FP32 baseline to 0 is a FP32-**high-word** arithmetic
  property (plain FP32 loses the "+1" in `x²+1` once `x² > 2²⁴`, x ≳ 4100), not a
  composed-quad-word property — matching T2.6's set exactly.
- **Per-kernel results (mean_digits / tolerance 26.0).** `K1_stable` 29.00 (harness
  cap; uncapped would exceed), `K2_basel` 27.67, `K3_machin` 28.73, `K4_alt_harmonic`
  28.64 — all PASS. `K1_naive_report` QF {28.23, 23.50, 23.87} vs FP32
  {3.28, 0.00, 0.00} at x ∈ {1e2, 1e4, 1e6}. K2 sanity vs π²/6 6.22 digits
  (truncation floor 6); K4 sanity vs ln 2 6.14 digits (truncation floor 6).
  **4 PASS / 0 RED.**
- **K3 margin note (contrast with T1.6).** `K3_machin` clears the 26.0 gate by
  **+2.73 digits** — MUCH more headroom than T1.6's DD K3 (+0.09), because QF's atan
  path has no PORT_NOTES_QF §10 denormal-tail hazard on Machin's small arguments
  (1/5, 1/239 are well inside FP32 normal range at every word). Fully deterministic
  (no RNG, fixed constants), so reproducible run-to-run. No revisit flag needed at
  this margin.
- **Contraction posture.** Registered with the plain `kokkos_ep_add_test` helper
  (mirroring `dd_e2e_test` / `ff_cancellation_test`), NOT the EFT helper. K1's naive
  `√(x²+1)−x` has a mul-then-sub adjacency an FMA *could* contract, but K1_naive is
  reported-not-gated (a contraction-induced shift can't flip pass/fail), and the
  gated path `K1_stable = 1/(√(x²+1)+x)` has no subtractive-cancellation adjacency.
  The Dekker-`twoProduct` hazard the EFT helper guards is on no gated path here — it
  is covered by `qf_fma_guard_test` (T3.5). Documented in the CMake comment.
- **Rule 4 respected.** `qf_math.hpp` / `qf_complex.hpp` / any other `*_math.hpp`
  untouched (only `#include`d). **No PORT_NOTES_QF changes** — T3.6 surfaced no
  library defects or new algorithmic characterizations. `main` untouched;
  `qffunKokkos` not merged.
- **Scope-out.** Real QF ops only — no `qf_complex.hpp` (matches T1.6/T2.6 real-only
  discipline); no per-op differential accuracy (that is the T3.4 sibling). No new
  B-task stubs — 0 misses, nothing to log (clean GREEN, like T3.4/T3.5).
- **Acceptance gate — all pass.** `cmake --build` clean, zero warnings from
  `qf_cancellation_test.cpp`; `qf_cancellation_test` **PASS 4.87 s**; **20/20
  non-accuracy tests green** (the 3 excluded `*_accuracy` tests are the known set,
  unaffected by this task).
- Depends on T0.1, T0.2, T3.0a, T3.0b, T3.1.
- **Phase 3 validation COMPLETE: 6 of 6 tasks DONE** (T3.1 EFT, T3.2 non-overlap,
  T3.3 property, T3.4 accuracy, T3.5 FMA-guard, T3.6 e2e cancellation). Next:
  `qffunKokkos` → `main` merge task (separate, on Reet's ask).
- See `64aac2d` for the code diff and this DONE block for the outcome.

## Rules for cluster-Claude implementation

Bake into launcher wrapper for every task:

1. **One task = one PR = one branch.** No task modifies files owned by
   another concurrent task. Phase 2/3 tasks target their respective
   branches (`fffunKokkos` merges → `main`; `qffunKokkos` builds → merges → `main`).
2. **Every task starts by reading:**
   - This file (`projects/kokkos_ep_test_suite.md`).
   - `tests/README.md` (created by T0.1).
   - The relevant `*_math.hpp` for the backend under test.
   - `PORT_NOTES.md` (for anything touching FF, and as reference for QF).
   - **NEVER the whole repo.** Context budget is precious.
3. **Verification gate is a passing `ctest`, not "it looks right".**
   Task done when new test file(s) pass in CI.
4. **If a task discovers a bug in existing code, it reports it and
   stops.** It does NOT fix. Bug fixes are separate tasks with clear
   scope. (Exception: T3.0a/T3.0b, which are building QF from scratch,
   can fix bugs in their own code as they go.)
5. **Numeric bounds cited in tests must have a source comment.** Format:
   ```
   // Bound: 4u² for double-word mul (Joldes-Muller-Popescu 2017, Thm 5.1)
   ```
   or:
   ```
   // Bound: observed empirically, no proven bound available for DDFUN variant
   ```
   Halts the "sounds plausible" failure mode.
6. **Never hallucinate library internals.** T3.0a/T3.0b in particular:
   read `qd_real.cc` from QD 2.3.24 (or clone if needed) BEFORE writing
   `qf_math.hpp`. Cite line numbers in comments.
7. **Preserve existing behavior across refactors.** T0.0 in particular:
   after refactor, demos must produce byte-identical accuracy numbers.
   Verify with `diff` on run output.
8. **Report format at task completion:**
   - What was implemented (files created/modified, LOC).
   - What tests pass (ctest output snippet).
   - What was NOT done and why (scope-out decisions).
   - Any bugs found in existing code (with reproduction, no fix).
   - Any deviations from this plan (must be justified).

## Parallelism graph

```
T0.0 ──┬─→ T0.1 ─┬─→ T1.1 ─┬─→ (T1.2, T1.3, T1.5, T1.6 all parallel)
       │        └→ T0.2 ─┤                                  ↓
       │                 └→ T1.4                    Phase 1 done
       │                                                    ↓
       └─→ (parallel with T0.1 if API locked)      T2.0 → T2.1 → (T2.2..T2.6 parallel)
                                                                              ↓
                                                                     Phase 2 done
                                                                              ↓
                                                              T3.0a → T3.0b → T3.1 → (T3.2..T3.6 parallel)
```

Phases sequential. Within phase: EFT layer (T*.1) first, then everything
else parallel.

## Sizing

- Phase 0: 3 tasks, ~1-2 days.
- Phase 1 (DD): 6 tasks, ~1 week.
- Phase 2 (FF): 6 tasks, ~1 week.
- Phase 3 (QF): 8 tasks, ~2 weeks (T3.0a/T3.0b are the biggest single
  tasks in the plan).

**Total: 23 tasks, ~3-4 focused weeks wall time.**

## Upstreaming considerations

The extended-precision types are being developed so that contributing them to
Kokkos later is a mechanical extraction rather than a rewrite. Locked in by T0.4
(DD) and inherited by FF (T2.0) and QF (T3.0):

- **Namespace + type names.** Types and their math functions live under
  `namespace Kokkos::Experimental` (`DoubleDouble`, and future `FloatFloat`,
  `QuadFloat`; complex as `DoubleDoubleComplex`, etc.). This is exactly where an
  upstream PR would place them, so the move is `git mv` + include-path fixups, not
  a symbol rewrite.
- **STL-style surface.** Arithmetic free functions use `add`/`subtract`/
  `multiply`/`divide`/`negate` (plus operators); transcendentals already match
  `<cmath>` (`exp`/`log`/`sin`/…); `round_to_nearest_int`, `pow_int`, `atan2`,
  and `two_prod` replace the DDFUN-Fortran `dd*` spellings. Constants are
  `DoubleDouble_pi()`-style free functions; bit-pattern construction is the
  static factory `DoubleDouble::from_bits(hi, lo)`.
- **ADL + `Kokkos::` re-exposure.** Every math function is ADL-findable via the
  argument's namespace AND re-exposed under `namespace Kokkos` (one-line
  forwards at the bottom of each header, mirroring
  `impl/Kokkos_QuadPrecisionMath.hpp`). So `Kokkos::exp(dd)` works identically to
  `Kokkos::exp(double)` / `Kokkos::exp(__float128)`. Arithmetic
  (`add`/`subtract`/`multiply`/`divide`) is reached via operators + explicit ADL
  only — deliberately NOT re-exposed as `Kokkos::add`.
- **SPDX + copyright headers** on the future-upstream files. Corrected in T0.5:
  the DDFUN-derived files (`third_party/include/dd_math.hpp`, `dd_complex.hpp`)
  carry the **DHB-License** (`SPDX-License-Identifier: LicenseRef-DHB-License`),
  NOT Kokkos's Apache-2.0 WITH LLVM-exception — they are ports of DDFUN and
  cannot be relicensed by us (see the "Licensing" section below). The
  Kokkos-style extension header `patches/kokkos_complex_quad_math.hpp` is the one
  file that genuinely carries `Apache-2.0 WITH LLVM-exception` (matching Kokkos
  for upstream compatibility). Whenever these types are upstreamed, the DDFUN
  attribution and DHB-License terms travel with the DDFUN-derived files; a Kokkos
  PR would need Kokkos maintainers' acceptance of the DHB-License for those two
  headers (or a clean-room reimplementation). Downstream-only files (demos,
  benchmarks, tests, this repo's harness, scripts) are deliberately header-free
  and covered by the top-level `LICENSE` (Apache-2.0).
- **Test structure.** Test logic is written against `BackendTraits<Backend>` so
  it could later drop into Kokkos's GoogleTest suite with minimal rewrite.
  Free-form `printf` is confined to the harness itself (`test_utils.hpp`), not
  the per-op test logic.
- **PR scope when it happens:** types + math functions + unit tests only. The
  demos, benchmarks, and this repo's CTest harness stay downstream. Opening a
  Kokkos PR is a future decision, made only after the test suite proves value.

## Licensing

Locked in by T0.5. This repository is **dual-licensed** (see `NOTICE.md` for the
authoritative, user-facing summary; this section is the rationale).

### Posture

- **Apache-2.0** — all Reet-authored original work: demos, tests, harness,
  corpus, scripts, docs. Covered by the top-level `LICENSE` (mirrored at
  `LICENSES/Apache-2.0.txt` for SPDX tooling). These files carry NO per-file SPDX
  header; the top-level `LICENSE` covers them.
- **DHB-License** (`LicenseRef-DHB-License`) — the two DDFUN-derived headers,
  `third_party/include/dd_math.hpp` and `dd_complex.hpp`. Full verbatim text in
  `LICENSES/LicenseRef-DHB-License.txt`.
- **Apache-2.0 WITH LLVM-exception** — `patches/kokkos_complex_quad_math.hpp`,
  the Kokkos-style complex-quadmath extension header. Matches Kokkos itself so it
  can be upstreamed verbatim.

### Why the split

DDFUN is authored by David H. Bailey and released under the **DHB-License**, a
modified BSD-3-Clause variant (Copyright (c) 2024 David H. Bailey). Our
`dd_math.hpp` / `dd_complex.hpp` are C++/Kokkos ports of DDFUN v04 — derivative
works. We cannot relicense DDFUN; the port inherits the DHB-License unchanged.
T0.4 had mechanically stamped Kokkos's Apache-2.0 WITH LLVM-exception onto these
two files to match Kokkos style; T0.5 corrected that, because a Kokkos SPDX on a
DDFUN derivative does not relicense DDFUN — it only makes the file's license
status ambiguous.

The complex-quadmath patch, by contrast, is a genuine Kokkos-style extension (a
companion to `Kokkos_QuadPrecisionMath.hpp`), not a DDFUN derivative, so it
correctly carries Kokkos's license.

### DHB-License §3 grant-back

The DHB-License adds a non-standard §3 to the usual BSD-3-Clause conditions: if
you publish modifications/enhancements to the DDFUN-derived files publicly
(which this repo does) without a separate written agreement, you thereby grant
David H. Bailey a non-exclusive, royalty-free, perpetual license to use, modify,
distribute, and sublicense those enhancements. In practice: anyone contributing
improvements to `dd_math.hpp` / `dd_complex.hpp` here grants Bailey the right to
fold them back into DDFUN, and anyone redistributing those files inherits this
term. (The DDFUN website also asks commercial users to contact the author at
`dhbailey@lbl.gov`; this is a courtesy pointer, not a term of the license text.)

### File-to-license mapping

| File(s) | License |
|---|---|
| `third_party/include/dd_math.hpp`, `dd_complex.hpp` | DHB-License (`LICENSES/LicenseRef-DHB-License.txt`) |
| `third_party/include/ff_math.hpp`, `ff_complex.hpp` | DHB-License (`LICENSES/LicenseRef-DHB-License.txt`) — DD→FF mechanical translation (T2.0) |
| `patches/kokkos_complex_quad_math.hpp` | Apache-2.0 WITH LLVM-exception |
| Everything else (demos, tests, harness, corpus, scripts, docs) | Apache-2.0 (`LICENSE`) |

### Phase 2/3 port lineage — FF and QF (RESOLVED)

FF inherits the **DHB-License** (DDFUN-derived); QF inherits the
**LBNL-BSD-License** (QD-derived). Both verified and applied — see the ticked
action items below. Kept as the record of the distinction; nothing here is open.

FF (`fffunKokkos`) and QF (`qffunKokkos`) each had their license verified
against their actual source tree **before** merging into `main`:

- If ported from **DDFUN** directly → inherits the **DHB-License** (Bailey,
  personal copyright, `dhbailey@lbl.gov`).
- If ported from **QD** (github.com/BL-highprecision/QD) → inherits the
  **different** `LBNL-BSD-License` (triple-authored Bailey/Li/Hida; LBNL
  *institutional* copyright; commercial contact `ipo@lbl.gov`, not Bailey
  personally).

These are distinct licenses with distinct copyright holders and contacts; the
correct header must be applied per source tree, not assumed uniform. Action
items:

- [x] **T2.0 kickoff:** FF port lineage verified — `ff_math.hpp` line 3 states
      "Mechanically ported from dd_math.hpp (DDFUN by David H. Bailey)", i.e. FF
      descends from DDFUN (not QD), so it inherits the **DHB-License** under §3
      grant-back (a mechanical DD→FF translation is a modification, not
      independent authorship). Both FF headers now carry `LicenseRef-DHB-License`,
      mirroring the T0.5 DD treatment, with a PORT_NOTES-referencing attribution.
      Applied in the T2.0 merge commit.
- [x] **T3.0a kickoff:** QF port lineage verified — QF is modeled on QD 2.3.24
      (`qd_real.cpp` / `qd_inline.h`), **not** on DDFUN, so it inherits the
      `LBNL-BSD-License` rather than the DHB-License that governs dd/ff. Applied:
      `qf_math.hpp` and `qf_complex.hpp` carry
      `SPDX-License-Identifier: LicenseRef-LBNL-BSD-License` with the lineage
      rationale in-header, `LICENSES/LicenseRef-LBNL-BSD-License.txt` holds the
      full text, and NOTICE.md maps both files to it (refreshed in `96880eb`).

**Both action items are closed** — this section is retained as the record of how
the two licenses were told apart, not as outstanding work.

### Argonne institutional review (Reet's follow-up)

Non-blocking for implementation, but must happen **before opening any Kokkos
PR**: a ~10-minute email to Argonne tech-transfer to confirm (a) Apache-2.0 for
the original code is approved, and (b) the DHB-License redistribution posture for
the DDFUN-derived files is acceptable. The "Modifications Copyright" holder used
for the ported files is "UChicago Argonne, LLC", the standard attribution form
for Argonne open-source releases.

## Deliverable at end of Phase 3

- `main` branch: DD, FF, QF all present, all benchmarked against Kokkos-
  wrapped quadmath oracle. `ctest` gates all three backends across all
  six test layers.
- ~~Four backend branches maintained (`ddfunKokkos`, `fffunKokkos`,
  `qffunKokkos`, `CUDAFP128Kokkos`) — synced periodically with `main`
  for their respective backends.~~ **Superseded.** Once DD, FF and QF all
  landed on `main`, the per-backend branches stopped being synced and
  became stale copies of merged work; `ddfunKokkos` and `fffunKokkos`
  were retired during earlier cleanups and `qffunKokkos` in the bucket-1/2
  close. **The intended steady state is two branches: `main` (all three
  portable backends) and `CUDAFP128Kokkos` (the sm_100-only FP128 backend,
  which cannot merge into `main` because it needs compute ≥ 10.0).**
  Ratified by Reet, 2026-08-09.
- `PORT_NOTES.md` extended with QF port notes (bugs found during T3.0
  development, per PORT_NOTES.md § template).
- README updated: three portable extended-precision backends
  (DD ~32 digits, QF ~29 digits, FF ~14 digits) benchmarked side-by-
  side against FP64 baseline, accuracy validated to published bounds.
- **Complex oracle dependency (T0.3).** ~~The complex demos' `__complex128`
  oracle depends on the local Kokkos extension header
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp`, which is **not upstream in
  Kokkos**. Future contributors must apply `patches/kokkos_complex_quad_math.hpp`
  per `patches/README.md` and rebuild Kokkos with
  `-DKokkos_ENABLE_LIBQUADMATH=ON` before building the complex demos; otherwise
  CMake warns and the complex-oracle compile fails.~~ **Retired 2026-09-16 —
  there is no complex oracle any more.** The demos are timing and smoke only,
  the patch and the probe are deleted, and any Kokkos ≥5.1 at C++20 builds all
  nine targets except `kokkos_ep_bench_cost`, which needs libquadmath because
  `__float128` is the thing it measures.
