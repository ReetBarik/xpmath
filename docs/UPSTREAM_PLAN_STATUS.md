# UPSTREAM_PLAN_STATUS.md

This file records the outcome of each sub-plan in `docs/UPSTREAM_PLAN.md`. Append a STATUS block after each sub-plan completes.

---

## S0 — Freeze tag + doc hygiene

**Commits:** (none — docs-only changes not yet committed)

**Tag:** `kokkos-native-freeze` (annotated, local only, not pushed) at commit `7f32f4e`

**Outcome:** Freeze point established, documentation updated to match current repository state. Annotated tag `kokkos-native-freeze` created on current `main` (commit 7f32f4e) as immutable fallback for any future Kokkos-core-native contribution work. Tag message documents this is the last fully Kokkos-native state before standalone restructure (`docs/UPSTREAM_PLAN.md`). Tag created locally only — not pushed.

`CLAUDE.md` rewritten to describe the actual repository: three portable backends (DD, FF, QF), seven executables (6 demos + 1 bench), two branches (`main` with all three backends, `CUDAFP128Kokkos` with sm_100-only CUDA FP128), 23 tests in the ctest suite. Old content described a stale CUDA-FP128+DD configuration with four executables and a three-branch structure that never existed on this branch. New version is short, points at README/TEST_SUITE_PLAN/PERF_PLAN/UPSTREAM_PLAN instead of duplicating their content.

`docs/PERF_PLAN.md` marked PARKED with a banner stating the arc is paused pending `docs/UPSTREAM_PLAN.md` completion (repo layout will change underneath it) and that Phase 1 must not be started. No other edits to that file.

**Deviations and surprises:** None. All verification passed: CMakeLists.txt shows 7 `add_executable` calls, tests/CMakeLists.txt shows 23 test registrations (14 plain + 6 EFT + 3 EFT-contract-on), two branches confirmed via `git branch -a`. Tag created exactly as specified, local only, no push attempted. Commit convention note: S0 is docs-only and the plan's commit-per-logical-unit rule applies at the sub-plan level — the next commit will bundle the CLAUDE.md + PERF_PLAN.md + STATUS.md changes as a single `upstream: S0 — freeze tag + doc hygiene` commit.

**Next sub-plan (S1/S2) must know:** The freeze tag is local only and must be pushed when the human reviews. `CLAUDE.md` now accurately describes the repository state as of commit 7f32f4e (HEAD of `main`). The 23-test count verified from tests/CMakeLists.txt matches what the plan states. PERF_PLAN.md is parked and should not be referenced until the upstream arc completes.

---

## S1 — CUDA device validation of the CURRENT code (A100 / sm_80)

**Commits:** `029180c` (authoring half — scaffolding: `validation/a100/README.md`, `build_login.sh`, `run_a100.sh`) + this commit (analysis half — curated artifacts, README RESULTS section, this STATUS block).

**Run:** Cobalt job `996777`, JLSE queue `gpu_a100`, node `gpu07`, `--mode script`, `-t 60`, 2026-08-31. Repo HEAD at build and run time: `029180ccedde6a68b0a70864c1e744234c8a2743`.

**Outcome.** The A100 device path works for DD and FF, and does not exist yet for anything else. Seven of the 23 registered ctest targets produced binaries; **all seven passed, with zero assertion failures**, in 115.77 s total. The other sixteen never compiled, because a `__float128` host oracle cannot survive nvcc's device compilation pass — a limitation of the *test and demo* code, not of the library headers. All six demos and `kokkos_ep_bench_cost` failed the same way, so S1 produced **no device accuracy tables** and deliverable 3's "compare demo accuracy against README Section 2" could not be performed. Within what did build the evidence is strong and unambiguous: `nvidia-smi` inside the job confirms `NVIDIA A100-PCIE-40GB` / driver `610.43.02`; the installed Kokkos 5.1.0 defines `KOKKOS_ENABLE_CUDA`, `KOKKOS_ENABLE_CUDA_CONSTEXPR`, `KOKKOS_ENABLE_CUDA_LAMBDA`, `KOKKOS_ENABLE_LIBQUADMATH` and `KOKKOS_ARCH_AMPERE80` (`sm_80`), built at C++20; and every test that prints the banner reported **`Execution space: Cuda`** — with device sections naming the space explicitly, e.g. `[Test B] device tripwire (5 ops, 10^5 random on Cuda)` and `[Device] 3 Group A (bit-exact) + 2 Group B on 100000 inputs (Cuda)`. `DefaultExecutionSpace` resolved to CUDA; this was **not** a silent Serial fallback, which was the single largest risk to the run's value. Rule 4 was honoured in full: no file under `third_party/include/`, `src/`, `tests/` or any `CMakeLists.txt` was touched. Every failure below is reported, not fixed.

### DEVIATIONS AND SURPRISES UP FRONT

**(a) Only 7 of 23 tests could be built, so S1's device coverage is PARTIAL. This is the honest result and it is not a pass.** S1's gate is "an honest, complete record", which is met — but the sub-plan's *purpose* (a pre-churn device baseline) is only half-served. What is covered:

| ctest test | time | execution space |
|---|---:|---|
| `corpus_test` | 0.03 s | host by design — **not device evidence** |
| `dd_invariant_test` | 45.14 s | **Cuda** — device tripwire 5 ops × 10⁵, 0 failures |
| `ff_invariant_test` | 39.14 s | **Cuda** — same shape, 0 failures |
| `dd_property_test` | 29.53 s | **Cuda** — 3 Group A bit-exact + 2 Group B on 10⁵, all PASS |
| `dd_e2e_test` | 0.55 s | **Cuda** init only, host math by design — **not device evidence** |
| `dd_eft_test` | 0.95 s | **Cuda** — Test D device parity 200 000/200 000 |
| `ff_eft_test` | 0.39 s | **Cuda** — Test D device parity 400 000/400 000 |

Not built (16): `hello_test`, `dd_accuracy_test`, `ff_property_test`, `ff_accuracy_test`, `ff_cancellation_test`, `dd_fma_guard_test`, `dd_fma_guard_test_contract_on`, `ff_fma_guard_test`, `ff_fma_guard_test_contract_on`, `qf_eft_test`, `qf_nonoverlap_test`, `qf_property_test`, `qf_accuracy_test`, `qf_fma_guard_test`, `qf_fma_guard_test_contract_on`, `qf_cancellation_test` — plus all six demos and `kokkos_ep_bench_cost`. Of those sixteen, only two TUs (`hello_test.cpp`, `dd_accuracy_test.cpp`) actually reached the compiler and errored; the rest were **never attempted**, because `make -j16` ran without `-k` and aborted once the first targets failed. So the fourteen unattempted tests are **UNKNOWN, not known-broken** — some may compile cleanly. Determining which was out of this session's read scope; a rerun with `cmake --build … -k` would settle it in one pass and is cheap.

**(b) The `ctest` line `30% tests passed, 16 tests failed out of 23` is an artifact of missing binaries and must never be quoted on its own.** Nothing ran and failed. ctest reports a missing executable (`Unable to find executable`, `***Not Run`, 0.00 sec) in its failure tally. The correct reading is **7 attempted, 7 passed, 16 never built**; `ctest` exit code 8. Anyone citing "30%" as a device pass rate is citing a build outcome as a numerical one.

**(c) `scripts/build_with_kokkos.sh` cannot build Kokkos 5.1 at all — it passes `-DCMAKE_CXX_STANDARD=17` (lines 67 and 74), and Kokkos 5.1.0 rejects it.** `cmake/kokkos_test_cxx_std.cmake:89` raises `FATAL_ERROR "Kokkos requires C++20 or newer but requested 17"`. The existing `~/kokkos-install*` trees all record `Kokkos_CXX_STANDARD 20`, confirming C++20 is what actually works. `validation/a100/build_login.sh` was corrected to `-DCMAKE_CXX_STANDARD=20` **for the Kokkos configure only**; the consuming repo correctly stays at C++17 (a C++17 TU links a C++20-built Kokkos fine, and that mirrors the working Serial installs). **`scripts/build_with_kokkos.sh` itself was NOT patched — out of scope for S1, reported only.** It is a live trap for anyone following `CLAUDE.md`'s documented build path against a 5.1 Kokkos.

**(d) Root cause of the 16 failures — a translation-unit coupling, with a proven fix routed to S6.** Kokkos 5.1 exports `INTERFACE_COMPILE_OPTIONS` containing `-extended-lambda;-Wext-lambda-captures-this;-expt-relaxed-constexpr;-arch=sm_80` (verified in `kokkos-install-cuda-sm80-quadmath/lib64/cmake/Kokkos/KokkosTargets.cmake:78`). Every TU linking `Kokkos::kokkos` therefore gets a **device** compilation pass. Any such TU that also declares a `std::vector<__float128>` makes nvcc instantiate `std::initializer_list<__float128>`, which errors: *"contains a 128-bit floating-point, which is not supported in device code"*. Exact first failing site: **`src/demo_ff_complex.cpp:387`**, two `std::vector<__float128>` named `href_re` / `href_im`; the build log names that line explicitly (`detected during instantiation of class "std::initializer_list<_E> [with _E=__float128]" at line 387`). Minimal reproduction on the login node (nvcc 12.9.86, gcc 13.3.0): a `.cu` with a sized `std::vector<__float128>` plus any `__global__` kernel, at `-arch=sm_80`, gives 3 errors. **Every flag combination tested still gives 3 errors** — none, `--expt-relaxed-constexpr`, `--extended-lambda`, both — and identically at `-std=c++17` and `-std=c++20`. **No compiler flag avoids this.**

  **The library headers are INNOCENT.** `third_party/include/dd_math.hpp`, `ff_math.hpp` and `qf_math.hpp` contain zero `__float128` *code* — only comments and one docstring. The coupling lives entirely in the oracle-scoring code in `tests/` and `src/`.

  **Proven fix (verified end to end on the login node, deliberately NOT applied — Rule 4):** split the translation unit. An `oracle.cpp` compiled by plain `g++ -std=c++20 -fext-numeric-literals` may use STL containers of `__float128` freely → 0 errors; a `kernel.cu` compiled by `nvcc -arch=sm_80` against an interface header that mentions only `double` → 0 errors; `nvcc kernel.o oracle.o -lquadmath` links and runs correctly. **Reet has ratified that this belongs in S6**, whose deliverable 6 already reads: *"Make the demo executables conditional on oracle availability in CMake (currently they hard-require quadmath and would fail to compile under clang-based toolchains — S8 needs this)."* Record it as an **S6 input**, not as S1 work. The scope is wider than S6 deliverable 6 currently states: it is not only *conditional on oracle availability* but *the oracle must not share a TU with device code*, and it affects the tests as well as the demos.

**(e) Secondary: the `--fmad` guard never engaged — C1 confirmed at 0.** `build_login.sh` section 6 never ran (section 5 failed first), so `contraction_flags.log` and `targets.log` were never written. The analysis session **regenerated `contraction_flags.log` post-hoc** from the surviving build tree (`compile_commands.json`, `CMakeCache.txt`) — no rebuild, no GPU job. Results against the README's predictions: **C1 = 0 occurrences of `--fmad`** (predicted 0 — the top-level `project(… LANGUAGES CXX)` means `$<COMPILE_LANGUAGE:CUDA>` never evaluates true under `nvcc_wrapper`-as-CXX; **confirmed**). **C2 = 9 occurrences of `-ffp-contract`**, exactly the predicted 6 `off` + 3 `fast`, on the host pass only. **C3 = `GNU` 13.3.0**, so the `-ffp-contract` and `-fext-numeric-literals` CMake branches did fire. **C4:** nvcc device codegen therefore ran at its default `--fmad=true` for every target — yet both built device-parity passes were clean (`dd_eft_test` 200 000/200 000, `ff_eft_test` 400 000/400 000). At sm_80 / nvcc 12.9.86 the Dekker `twoProduct` sequence was not contracted into an FMA *in practice*; that is an empirical observation for this compiler+arch, **not** a guarantee, and the intended guard is still not engaged. **C5** could not be evaluated: no `_contract_on` target built. `targets.log` is not recoverable without a rebuild.

**(f) Benign, so nobody chases it:** `logs/run/996777.error` (57 bytes) contains only `/home/rbarik/.bashrc: line 13: module: command not found` — Lmod is not initialized in the Cobalt shell, and `run_a100.sh` loads its own modules. Also, the job was submitted with `-t 60` rather than the README's recommended `-t 120`; harmless here (it finished in 2 min 5 s precisely because the long tests had no binaries), but a rerun with the suite actually built still needs `-t 120`.

### What S2 / S5 must know

**There IS a usable pre-churn device baseline — for DD and FF only, and it is GREEN with zero code changes.** `dd_invariant_test`, `ff_invariant_test`, `dd_property_test`, `dd_eft_test` and `ff_eft_test` all pass on A100 `sm_80` against unmodified `main` at `029180c`. If a mechanical restructure in S2 or S5 turns any of those red on device, **the restructure caused it** — that attribution is now sound, and those five tests are the device regression gate for the DD and FF work.

**There is NO device baseline for anything else, and the absence must not be read as either evidence or excuse.** Specifically: **no QF device baseline at all** (all six QF registrations failed to build), no baseline for any accuracy test, no baseline for any FMA-guard test, and no device accuracy tables from any demo. Two consequences, both binding:

1. **A future QF device failure must NOT be attributed to the restructure.** QF has never been observed running on a GPU. If QF breaks on device after S2/S5, the correct statement is "QF device behaviour is untested before and after" — establishing whether the restructure is responsible requires first getting a QF device baseline, which requires the S6 TU split.
2. Conversely, the restructure cannot claim device coverage it does not have. Any S4 RFC text should say "DD and FF validated on A100; QF and the oracle-scored suite blocked on a test-harness limitation, tracked in S6" — not "runs on device".

**Ordering implication.** The S6 TU split is the unlock for full device coverage. Whoever schedules S6 should know that completing it makes a *second* A100 run worthwhile — that rerun, not this one, is what produces the QF/accuracy/FMA-guard device baseline and the demo accuracy tables S1 deliverable 3 asked for. Run it with `cmake --build … -k` so one job enumerates every remaining failure instead of stopping at the first.

**Also carry forward:** `scripts/build_with_kokkos.sh` still hardcodes `Kokkos_ARCH_BLACKWELL100`, builds with plain `g++`, *and* passes the fatal `-DCMAKE_CXX_STANDARD=17`. `validation/a100/build_login.sh` is the working sm_80 recipe; treat it, not `scripts/`, as the reference for CUDA builds until someone owns fixing the script.

---

## S2 — DD vertical slice: standalone core + Kokkos wrapper + naming memo

**Commits:** `71de64a` (code half — `include/EPLIB/config.hpp`, `include/EPLIB/dd_math.hpp`, `third_party/include/dd_math.hpp` rewritten as a compat wrapper, `tests/standalone/dd_no_kokkos_smoke.cpp`, `scripts/check_standalone_no_kokkos.sh`, one `CMakeLists.txt` line) + this commit (closeout half — this STATUS block, the naming memo below, and `validation/s2/` holding the byte-identical gate evidence).

**Outcome.** The standalone architecture works on the DD slice, and it costs consumers nothing. `include/EPLIB/dd_math.hpp` (1081 lines, 123 `EPLIB_INLINE_FUNCTION` declarations) is the DD implementation in `namespace eplib` with zero Kokkos: it needs only the C++17 standard library, and `include/EPLIB/config.hpp` (192 lines) supplies the four facilities Kokkos used to — `EPLIB_INLINE_FUNCTION`, an `EPLIB_ON_DEVICE` predicate unifying CUDA/HIP/SYCL, a `eplib::detail::` scalar-math dispatch built from using-declarations rather than wrapper functions, and a compile-time-removable `EPLIB_PRINTF`. Every choice carries a one-line in-header rationale, because this header is the exhibit S4 has to defend. The old path `third_party/include/dd_math.hpp` is now a 194-line compat wrapper: one type alias, 68 explicit `using eplib::` declarations, and the 39 `Kokkos::`-namespace math forwarders relocated verbatim from the bottom of the old file. **`dd_complex.hpp`, all 23 tests and both DD demos compile unchanged** — the only non-header edit in the whole sub-plan is `CMakeLists.txt` appending `include/` to `THIRD_PARTY_INCLUDE`. Every gate passed: full **ctest 23/23 in 137.53 s** on the login-node host Serial build against `$HOME/kokkos-install-quadmath` (the pre-flight at `2fe6ce9`, before any S2 change, was also 23/23 in 134.79 s); the **byte-identical gate passed on both DD demos** at `--batch 1000000 --repeats 5 --seed 12345` (timing-stripped diffs empty, 87 lines real / 105 lines complex — re-verified from the committed captures while writing this block); and the **no-Kokkos compile smoke passed** all four of its stages under `g++ -std=c++17` — compile+link against an include path containing only `include/` (`third_party/include/` deliberately excluded so the compat wrapper cannot mask a missing dependency), run with `PASS (0 failures)` over 11 checks, `53628 preprocessed lines, 0 Kokkos hits`, and `include/EPLIB/*.hpp` naming Kokkos only inside comments.

### DEVIATIONS AND SURPRISES UP FRONT

**(a) The authoring session was killed by its wall-clock timeout after committing the code but before writing this STATUS block; the closeout was done in a separate session, and one gate had to be re-run.** The timeout landed while the *second* byte-identical demo was still executing, leaving `validation/s2/after_dd_complex.txt` at 0 bytes. It was re-run to completion (exit 0) and diffed: **the gate genuinely passes — it was an incomplete run, not a failure.** Record this as a budgeting fact, not an incident: S2's gate set is two 1M-batch × 5-repeat demo runs *before* the change, the same two *after*, plus a full 23-test ctest at ~137 s, on top of the actual authoring. **A later sub-plan running the same shape — and S5 runs it for five more headers, i.e. ten demo captures — must budget for it explicitly, or split authoring and gating into separate sessions from the start.** The cheap mitigation, which S5 should just adopt: capture the "before" set and commit it *first*, so a timeout mid-gate costs a re-run rather than a lost baseline.

**(b) `tests/standalone/dd_no_kokkos_smoke.cpp` is deliberately NOT registered with ctest.** This is a choice, not an omission, and its reasoning is in the file's own header comment: S2's gate is "full ctest 23/23", and silently making the suite 24 would change what that number means on either side of the restructure the number exists to police. The smoke test is therefore driven by `scripts/check_standalone_no_kokkos.sh`, which is run by hand and exits non-zero on failure. **Wiring it into the build system is left to S5/S6** — and whoever does it should move the count deliberately and say so, rather than letting 23 drift into 24 unannounced.

**(c) Four sites needed judgement rather than a token swap. None is numerical; all four are portability-surface changes, and none is exercised by any gate S2 ran.** The plan says to record non-mechanical sites instead of improvising, so:

1. **`DoubleDouble::from_bits` is guarded by `EPLIB_ON_DEVICE_CUDA_OR_HIP`, not the unified `EPLIB_ON_DEVICE`.** `__longlong_as_double` is a CUDA/HIP global-namespace intrinsic with no SYCL equivalent, so a SYCL device build must take the `std::memcpy` path. Both paths are the identical bit reinterpretation, so this is a portability fix with zero numerical content — but it is why `config.hpp` defines *two* device predicates instead of one, and S5 should reuse the narrow one for the same reason wherever the ff/qf headers touch intrinsics.
2. **`atan2` and `ldexp` were missing from the plan's dispatch list, and `ldexp` was invisible to the coupling census.** `ldexp` was never `Kokkos::`-qualified in the old header — it resolved to glibc's global-namespace overload — so a grep for `Kokkos::` could not see it. **This is the transferable warning for S5: the Common-context coupling census undercounts. Grep for bare unqualified math calls, not only for `Kokkos::`-qualified ones**, or a `std::`-only host build will look fine and a device build will not. Both now route through `eplib::detail`.
3. **The two `#ifndef __CUDA_ARCH__` guards became `#if !defined(EPLIB_ON_DEVICE)` — a deliberate widening.** They cover the `<iomanip>`/`<ostream>` includes and `operator<<`. Under a HIP or SYCL device pass, stream output now compiles *out* where the CUDA-only spelling would have left it in (and then failed, which is the point). Numerically inert, but it is a real API-surface change on non-CUDA device builds.
4. **`EPLIB_PRINTF` is a no-op under `__SYCL_DEVICE_ONLY__`.** SYCL 2020 has no portable device printf, and every available spelling is a vendor extension that would make the header depend on a SYCL header. So on a SYCL device the ~19 domain diagnostics ("DDSQRT: negative argument", …) are silently dropped. **Return values are unaffected** — every diagnostic site returns the same value with or without the print — so this is a debuggability loss, not a numerical one. It is documented in `config.hpp` §3 rather than hidden.

  **None of (c)1–4 is covered by an S2 gate.** Everything S2 ran was a host build (login-node Serial ctest, host demos, host `g++` smoke). CUDA/HIP/SYCL device compilation of the standalone core is **unbuilt and untested** — S1's A100 device baseline predates the restructure and was taken through the old header. First device build after the restructure is the real test of (c)1, (c)3 and (c)4.

**(d) Minor, so nobody is surprised by it later: `validation/s2/strip_timing.sh` does not do one thing its header comment claims.** The comment says the `batch=… timing=…` banner line is excluded; the awk does not exclude it, and the banner survives into the stripped output. Harmless here — that line is deterministic across runs, so both captures carry it identically and the diff is still empty — but if anyone changes the banner to include a hostname, a date or a thread count, the gate will start failing for a non-accuracy reason. The script is otherwise correct and directly reusable (see "What S5 must know").

### Gate results

| Gate | Result |
|---|---|
| full ctest, host Serial, post-restructure | **100% tests passed, 0 failed out of 23**, 137.53 s (pre-flight at `2fe6ce9`: 23/23, 134.79 s) |
| byte-identical, `kokkos_ep_demo` (DD real) | **PASS** — timing-stripped diff empty, 87 lines |
| byte-identical, `kokkos_ep_demo_complex` (DD complex) | **PASS** — timing-stripped diff empty, 105 lines (see deviation (a): the "after" capture was re-run) |
| no-Kokkos compile smoke, `g++ -std=c++17` | **PASS**, all 4 stages — compile+link with `-I include` only; run `PASS (0 failures)`; `53628 preprocessed lines, 0 Kokkos hits`; headers name Kokkos only in comments |
| non-mechanical sites recorded, not improvised | **PASS** — four, listed in (c) |

Evidence for the byte-identical gate is committed under `validation/s2/`: `before_dd_real.txt`, `after_dd_real.txt`, `before_dd_complex.txt`, `after_dd_complex.txt` and the `strip_timing.sh` stripper that defines the gate.

### What S3 must know — the exact rename scope

`EPLIB` / `eplib` is a placeholder in **7 files, 287 occurrences on 281 lines**. Enumerate them with:

```sh
grep -rIn --exclude-dir=build --exclude-dir=build_t00 --exclude-dir=.git \
          --exclude-dir=kokkos --exclude-dir=validation \
          -e 'EPLIB' -e 'eplib' .
```

**`--exclude-dir=kokkos` is required**, and the reason is a trap: the vendored `kokkos/cmake/fake_tribits.cmake` matches on `DEPLIB` / `DEPLIBS` (5 lines) — a substring, not a rename site. **Do not "fix" this with word boundaries**: `\bEPLIB\b` silently drops `EPLIB_INLINE_FUNCTION`, because `_` is a word character. Exclude the directory instead.

| file | occurrences |
|---|---:|
| `include/EPLIB/dd_math.hpp` | 150 |
| `third_party/include/dd_math.hpp` | 78 |
| `include/EPLIB/config.hpp` | 25 |
| `tests/standalone/dd_no_kokkos_smoke.cpp` | 15 |
| `docs/UPSTREAM_PLAN.md` | 14 |
| `scripts/check_standalone_no_kokkos.sh` | 3 |
| `CMakeLists.txt` | 2 |

Four token classes, by occurrence (`grep -rhoE`):

| token class | count | note |
|---|---:|---|
| `EPLIB_INLINE_FUNCTION` | 131 | the bulk of the sweep |
| `eplib::<name>` qualified | 92 | 78 of them in the compat wrapper's right-hand sides |
| `EPLIB_PRINTF` | 19 | |
| `EPLIB/` as an include path | 14 | these follow the `git mv` of the directory |
| `namespace eplib` | 7 | including `namespace detail` inside it |
| `EPLIB_ON_DEVICE` | 7 | |
| `EPLIB_ENABLE_DIAGNOSTICS` | 5 | consumer-facing knob — renaming it is an API break for anyone who set it |
| `EPLIB_ON_DEVICE_CUDA_OR_HIP` | 4 | |
| bare `eplib` / `EPLIB` in prose | 8 | comments and doc text |

So the sweep is exactly: **five macros** (`EPLIB_INLINE_FUNCTION`, `EPLIB_PRINTF`, `EPLIB_ON_DEVICE`, `EPLIB_ON_DEVICE_CUDA_OR_HIP`, `EPLIB_ENABLE_DIAGNOSTICS`), **one namespace** (`eplib`, plus `eplib::detail`), **one directory** (`include/EPLIB/` → `git mv`, with 14 include-path references following it), and **the prose**. Two things worth knowing before planning the work:

- **There are no include-guard macros to rename.** Both standalone headers use `#pragma once`. That class of the rename is empty.
- **Three non-header files hardcode the placeholder and are easy to miss:** `scripts/check_standalone_no_kokkos.sh` (the `include/EPLIB/*.hpp` glob in stage 4 and the `#include <EPLIB/dd_math.hpp>` in the stage-3 generated TU), `CMakeLists.txt:77-78`, and `docs/UPSTREAM_PLAN.md`. The plan text is S3's call to rewrite or leave.
- **Verification after the sweep is cheap and already built:** re-run `scripts/check_standalone_no_kokkos.sh` and the byte-identical gate. A pure rename must leave both untouched.

### What S5 must know

**The compat-wrapper pattern proven here is the template — reapply it verbatim to `ff_math`, `qf_math`, `dd_complex`, `ff_complex` and `qf_complex`.** The shape, in the order it has to be written:

1. Standalone core at `include/EPLIB/<name>.hpp`, `namespace eplib`, macros from `config.hpp`, zero Kokkos includes.
2. Thin wrapper left at the *old* `third_party/include/<name>.hpp` path, containing exactly three things: a type alias (`using FloatFloat = eplib::FloatFloat;` — an alias, so the two spellings name the *same* type and stay interchangeable across the `*_complex.hpp` boundary), explicit `using eplib::f;` declarations, and the relocated `Kokkos::`-namespace forwarder block.
3. **Explicit using-declarations, never a `using namespace eplib;` directive.** The reasons are written into the wrapper's header comment: the Kokkos-visible surface stays an auditable list; a using-*declaration* merges into Kokkos's own same-named `Experimental` overloads (`sqrt`, `exp`, … for `half_t`) where a using-*directive* gets qualified lookup subtly wrong; and the list is the exhibit for the S4 RFC.
4. **Free operators are deliberately absent from the wrapper.** `dd + dd`, `1.0 * dd` and `os << dd` are found by ADL through the argument's real namespace. Re-declaring them is redundant.

Calibration from the DD slice: one real header cost 68 using-declarations + 39 forwarders and left every consumer untouched. Three further notes:

- **`config.hpp` already declares the union dispatch set over all six backends**, deliberately, so S5 does not have to reopen the policy — *but* S2 still had to add `atan2` and `ldexp` beyond the plan's census (deviation (c)2). Expect ff/qf to surface more unqualified global-namespace math calls that no `Kokkos::` grep can find.
- **`Kokkos::complex` interop is untouched and unproven.** S2 covered a real header only; the three complex headers use `Kokkos::complex` (~7 sites) and that is the one dependency the wrapper pattern above has not yet been shown to absorb. Plan for it as its own problem, not as a fifth repetition.
- **The byte-identical gate is what catches drift, and the tooling is committed.** `validation/s2/strip_timing.sh` keys on the demos' shared 10-field pipe-delimited row layout and works unmodified on both the real and complex tables (verified on both), so S5 can reuse it as-is for the remaining four demos — subject to deviation (d).

---

## S2 NAMING MEMO — library / namespace / repo name

Reet proposed two candidates and asked for them to be checked rather than accepted. Both were checked and **both are recommended against**, for different reasons; a third is recommended, with a runner-up. **The code stays on the `EPLIB` / `eplib::` placeholder — Reet ratifies, S3 applies.**

The test applied throughout is the call site, because that is what a reviewer of the S4 RFC actually reads:

```c++
ns::DoubleDouble x = ns::DoubleDouble_pi();
ns::DoubleDouble y = ns::exp(x);      // <- the hard one
using ns::erfc;
```

#### Candidate A — ExPLib / `exp::` — **reject**

The *reasoning* is sound. `MxP` is genuine, current shorthand for mixed precision in exactly the audience S4 targets: [HPL-MxP](https://hpl-mxp.org/) is a live Top500-adjacent benchmark, with a September 2025 methods paper ([arXiv:2509.19618](https://arxiv.org/abs/2509.19618)) and El Capitan's 16.7 Exaflop/s headline on the [November 2025 TOP500 list](https://top500.org/lists/top500/2025/11/highs/). "ExP for extended precision" would read naturally to that audience. The *letters* are the problem.

- **Fatal, and independent of any collision: `exp` is a function this library ships.** `exp::exp(x)`, `exp::exp2(x)`, `exp::expm1(x)`; the compat wrapper would contain the line `using exp::exp;`; and `namespace exp = …` would collide with any local named `exp`. Even at the longer `explib::exp(x)` the stutter survives. A numerics library cannot take the name of one of its own functions.
- **The token is crowded on GitHub, and crowded in a bad neighbourhood.** [RI-SE/EXPLib](https://github.com/RI-SE) (RISE Research Institutes of Sweden — explainable/dependable deep-learning components) is confirmed; so are [jroimartin/explib](https://github.com/jroimartin/explib) (a pwntools-based CTF/exploit-development package), [b4zinga/Explib](https://github.com/b4zinga/Explib) ("collections of poc and exp"), [akivajp/explib-python](https://github.com/akivajp/explib-python) and [Navin7490/ExpLibrary](https://github.com/Navin7490/ExpLibrary). Two of the top five are *exploit* libraries, so "explib" carries a security-tooling connotation the project does not want.
- **The `Lib` suffix is dated in the company this project would keep.** `mdspan`, `desul`, `Kokkos`, `RAJA`, `Umpire` — none carries it. `libxprec` puts it in front, Unix-style, which still reads as current.

#### Candidate B — InPLib / `inp::`, "Intermediate Precision Library" — **reject**

- **Availability is the real advantage, and it is weaker than it looks.** No GitHub, PyPI or crates.io hit for "inplib" — correct. But the immediate neighbours are [c-robinson/iplib](https://github.com/c-robinson/iplib) (Go, IP addresses), [mlocati/ip-lib](https://github.com/mlocati/ip-lib) (PHP), [TSeals/IPlib](https://github.com/TSeals/IPlib) (image processing) and [plib](https://pypi.org/project/plib/) (PyPI). "inplib" is one character from a package in active use — a typo-collision story, not a clean-namespace story.
- **"Intermediate precision" is established terminology for a different concept**: the precision of *intermediate results within an expression* — x87 80-bit temporaries, unrounded FMA products, `FLT_EVAL_METHOD`, `-ffloat-store`. The canonical reference is Bruce Dawson's ["Intermediate Floating-Point Precision"](https://randomascii.wordpress.com/2012/03/21/intermediate-floating-point-precision/) (Random ASCII, 2012; [reprinted on Game Developer](https://www.gamedeveloper.com/programming/in-depth-intermediate-floating-point-precision)). The S4 audience is precisely the audience that knows this article. Naming the library after it invites the reader to expect evaluation-order semantics and find a 106-bit user-facing type instead.
- **It also mischaracterizes two thirds of the library.** DD is 106 bits and QF ~96 — *above* the native formats, which is what "extended" conveys. Only FF (2×FP32, ~48 bits) genuinely sits between FP32 and FP64. The term is right for one backend of three.
- **Call site:** `inp::DoubleDouble x;` reads as "input", and `inp` is a conventional parameter name in exactly the numeric kernels this library targets.

#### Candidate C — the XP family: repo `xpmath`, namespace `xp::`, macros `XPMATH_`, headers `include/xp/` — **RECOMMEND**

This keeps Reet's MxP echo and drops the `exp` clash: the RFC can say, in as many words, *"extended precision (XP), the companion to mixed precision (MxP)"*.

**Why X is the right letter, and specifically right for this audience.** X-for-extended is already this community's convention, and the lineage runs directly through this library's own sources:

- **XBLAS** — the "Extended and Mixed Precision BLAS" standard and reference implementation (Li, Demmel et al., *ACM TOMS* 28(2), 2002; [ACM](https://dl.acm.org/doi/10.1145/567806.567808)). Its reference implementation uses **double-double** for the extended-precision path — the same technique as this library's DD backend. The report is hosted on [davidhbailey.com](https://www.davidhbailey.com/dhbpapers/xblas-report.pdf) — David H. Bailey, whose DDFUN this repo ports and whose QD is the QF backend's ancestor. The naming and the code share a provenance.
- **[tuwien-cms/libxprec](https://github.com/tuwien-cms/libxprec)** — C++11, MIT, double-double, CMake target `XPrec::xprec`. Same abbreviation, same technique, independently arrived at.
- **x87's 80-bit format is "extended precision"** in the IEEE-754 sense; the association predates all of the above.

**Call site.** `xp::DoubleDouble x;`, `xp::exp(x)`, `xp::erfc(y)`, `using xp::sqrt;`, `xp::detail::fabs(a.hi)`, and the wrapper line `using DoubleDouble = xp::DoubleDouble;`. No token in the library collides with its namespace.

**Collisions, stated honestly:**

- `xplib` as a **repo** name has a near-neighbour in [FragXPL/xPLLib](https://github.com/FragXPL/xPLLib), a C++ library for the **xPL home-automation protocol**. Not a hard conflict, but it pollutes search — which is why the recommendation is `xpmath` (or `libxp`) for the *repo* and `xp` for the *namespace*, rather than `XPLib` for both.
- `xpmath`: [xpmath.com](https://www.xpmath.com/) is a math-games education site and [xp-framework/math](https://github.com/xp-framework/math) is a PHP big-number package. Different languages, different domains, no package-manager contention.
- No `xplib` package on PyPI or crates.io.
- **"XP" as a bare word means eXtreme Programming, and Windows XP.** This is a real problem for a repo literally named `XP` — and not a problem for a namespace token inside numeric code, where `xp::sqrt(a)` is unambiguous. It is a further argument for the qualified repo name.
- **The one genuine cost: adjacency to libxprec's `xprec::`.** Two portable C++ double-double libraries whose namespaces are `xp` and `xprec` are confusable, and a user could plausibly depend on both. They remain distinguishable at a glance, and the alternative (lengthening to `xprecision::`) buys clarity at the price of every call site. Recommend accepting it, with the RFC explicitly naming libxprec as prior art (which S4 has to do anyway — see below).

**Macro prefix: `XPMATH_`, not `XP_`.** Keep the namespace short and the macros long. A two- or three-character prefix in the global *macro* namespace is genuinely hazardous (`XP_` shows up in legacy Windows and game code, and macros ignore namespaces), and the short-namespace/long-macro split is what Kokkos itself does (`Kokkos::` vs `KOKKOS_`). Concretely: `XPMATH_INLINE_FUNCTION`, `XPMATH_PRINTF`, `XPMATH_ON_DEVICE`, `XPMATH_ON_DEVICE_CUDA_OR_HIP`, `XPMATH_ENABLE_DIAGNOSTICS`; CMake package and target `xpmath::xpmath`.

#### Candidate D — `xfp::` / repo `xfp`, "eXtended Floating-Point" — runner-up

Keeps the X, more distinctive and more greppable than a two-letter token, no clash with `exp`. Two drawbacks: **XFP is the well-known 10 Gb/s optical transceiver form factor** (e.g. [thiagoesteves/xfp_statem](https://github.com/thiagoesteves/xfp_statem)), which owns the acronym in search even though the domain is unrelated; and it carries the MxP echo less directly than XP does. Hold it in reserve if a two-letter namespace feels too aggressive.

#### Recommendation

**Candidate C, in this concrete spelling:** repo `xpmath` · namespace `xp::` (with `xp::detail::`) · macro prefix `XPMATH_` · headers under `include/xp/` · CMake `xpmath::xpmath`. Fallback: candidate D. Reject A (its own `exp` is the blocker, before the crowded GitHub token) and B (semantic collision with a well-established different meaning, and wrong for DD and QF).

#### Prior art the S4 RFC must cite regardless of the naming outcome

- **[tuwien-cms/libxprec](https://github.com/tuwien-cms/libxprec)** — C++11, MIT, header-plus-library double-double, CMake target `XPrec::xprec`, documented per-operation error bounds in multiples of u². **This is the closest existing thing to the DD backend, and the RFC must answer "why not just vendor libxprec?"** The honest answers are: it is DD only (no FF, no QF), it is not written for device execution across CUDA/HIP/SYCL, and it does not carry the Kokkos-facing API surface — but the question will be asked, so answer it first.
- **QD** (Hida, Li, Bailey) — the double-double/quad-double library the QF backend already ports. Prior art the repo is downstream of, not merely adjacent to.
- **[oprecomp/FloatX](https://github.com/oprecomp/FloatX)** — header-only C++ emulation of *low*-precision floating-point types. The opposite direction on the precision axis, the same "portable emulated FP type in C++" shape; useful as the counterexample that frames the space.
