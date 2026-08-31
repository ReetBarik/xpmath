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
