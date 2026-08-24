# Standalone + upstream plan

This document is the authoritative spec for the arc that (a) makes the
DD/FF/QF extended-precision library usable **without Kokkos** and (b)
proposes it **into Kokkos** as a vendored third-party library (the
mdspan/desul model: standalone core + thin Kokkos wrapper).

It is decomposed into sub-plans **S0–S9**. Each sub-plan is
self-contained: a fresh Claude session executes one sub-plan by reading
ONLY the **Common context** block below plus its own section. Do not
read this whole document from a work session, and do not read the whole
repo — context budget is precious.

## Ratified decisions (Reet, 2026-08-20)

- The restructure happens **in this repo**. The repo will be **renamed**
  to drop "kokkos" from its name; the new name is decided in S2/S3.
- The library namespace / name is **proposed in S2** (naming memo),
  ratified by Reet, applied in S3. Until then every sub-plan uses the
  placeholder `EPLIB` (macro prefix) / `eplib::` (namespace).
- `docs/PERF_PLAN.md` is **PARKED** until this arc lands (S0 adds the
  banner). Do not start any PERF_PLAN phase.
- Device/GPU work runs **hybrid**: the session prepares exact scripts
  and commands, Reet executes them on the GPU machines and pastes the
  output back, the session analyzes and writes the STATUS block.
- The known DD/FF `erfc` mid-z limitations (TEST_SUITE_PLAN /
  PORT_NOTES) are **out of scope** for the entire arc.

## Ratified decisions (Reet, 2026-08-24)

- A fourth backend, **TF** (`TripleFloat`, 3×FP32, 72 bits / ~21.7
  digits), is added to this arc as **S10**, sequenced **after S5 and
  before S6**. It fills the only gap in the precision ladder under a
  quad-precision ceiling: 48b (FF) → 53b (`double`) → **72b (TF)** →
  96b (QF) → 106b (DD).
- S10 is the **only sub-plan in this arc that adds arithmetic**, so it
  is explicitly outside the Rule 4 supersession described below. Its
  byte-identical gate protects the five existing backends; it does not
  apply to TF.
- Precisions **beyond quad (>113 bits) are out of scope** for this
  library. That bounds the backend set at the five existing plus TF, and
  keeps `__float128` sufficient as the oracle — no MPFR, no ARB.

---

## Common context (read this + your own section, nothing else)

**What this repo is.** Three portable extended-precision backends as
header-only C++ in `third_party/include/`: DD (`DoubleDouble`, 2×FP64,
~31 digits, `dd_math.hpp` + `dd_complex.hpp`), FF (`FloatFloat`, 2×FP32,
~14 digits, `ff_*.hpp`), QF (`QuadFloat`, 4×FP32, ~29 digits,
`qf_*.hpp`). All types/math currently live in
`namespace Kokkos::Experimental`, every function is
`KOKKOS_INLINE_FUNCTION`, and validation is a 23-test ctest suite in
`tests/` plus six demo executables in `src/` measured against a
`__float128` (libquadmath) host oracle. All 23 tests pass on `main`.

**End goals of this arc.** (1) A non-Kokkos user can copy or install the
headers and use the types in plain C++/CUDA/HIP with zero Kokkos
dependency. (2) Kokkos consumes the same library as a vendored TPL plus
a thin `Kokkos::Experimental` wrapper, so Kokkos users see today's API
unchanged.

**Where to run.** Doc-only sub-plans run anywhere. Sub-plans whose gates
build or run code (S2, S3, S5, S6) need the x86_64 Linux host with the
Kokkos 5.1 + `Kokkos_ENABLE_LIBQUADMATH=ON` install (see
`scripts/prepare.sh` / `patches/README.md` for the environment) — run
the session there, or fall back to the hybrid protocol (S1 describes
it). GPU sub-plans (S1, S8) are always hybrid.

**Conventions.**
- Commits: `upstream: S<n> — <one-line summary>`, one commit per logical
  unit. Work on a short-lived branch, merge to `main` when the sub-plan's
  gates pass. No long-lived parallel branches.
- Every sub-plan closes with a STATUS block appended to
  `docs/UPSTREAM_PLAN_STATUS.md` (create on first use): sub-plan ID,
  commit SHA(s), one-paragraph outcome, deviations and surprises up
  front, and anything the next sub-plan must know.
- The verification gates are stated per sub-plan. Never silently loosen
  a gate; a gate change is a reported deviation.
- **Rule 4 supersession.** `docs/TEST_SUITE_PLAN.md` Rule 4 froze the
  library headers. S2 and S5 explicitly lift that freeze for
  **mechanical restructuring only** — namespace moves, macro renames,
  include changes. Arithmetic/algorithm changes remain forbidden and the
  byte-identical gates enforce that. Any site where a mechanical
  translation is impossible: stop, record it in STATUS, do not improvise
  a numerical change.
- **Byte-identical gate** (used by S2, S3, S5): run the affected demo(s)
  before and after with identical args
  (`--batch 1000000 --repeats 5 --seed 12345`), strip timing lines, and
  `diff` the accuracy tables. They must be identical. Timing columns are
  exempt; accuracy columns are not.

**Measured Kokkos coupling in the library headers** (so sub-plans don't
re-derive it): `KOKKOS_INLINE_FUNCTION` ~720 uses; scalar math wrappers
`Kokkos::fabs/exp/floor/sqrt/log/isinf/ceil/copysign/atan/isfinite`
(~150 uses total, all on plain `float`/`double`); `Kokkos::printf` ~40
diagnostic sites; `Kokkos::complex` ~7 uses (complex headers only);
`#ifndef __CUDA_ARCH__` guards in all six headers; one `Kokkos::numbers`
reference. No Views, no parallel constructs, no Kokkos runtime calls.

---

## S0 — Freeze tag + doc hygiene

**Model: Sonnet.** Depends on: nothing. Runs anywhere.

**Context.** Before the restructure begins, the last fully Kokkos-native
state must be preserved as an immutable fallback (if Kokkos later asks
for a core-native contribution, work restarts from this point). Also,
`CLAUDE.md` is badly stale — it describes a CUDA-FP128+DD repo with four
executables, none of which matches `main` — and every future session
reads it, so it must be fixed before the multi-session arc starts.

**Read:** `CLAUDE.md`, `README.md` (Sections 2–5), `docs/PERF_PLAN.md`
(first 40 lines only).

**Deliverables.**
1. Annotated git tag `kokkos-native-freeze` on current `main`, pushed to
   origin. Tag message: one paragraph stating this is the last fully
   Kokkos-native state before the standalone restructure
   (`docs/UPSTREAM_PLAN.md`).
2. Rewrite `CLAUDE.md` to describe the actual repo: three backends
   (DD/FF/QF), seven executables, build/test commands, pointers to
   `README.md`, `docs/TEST_SUITE_PLAN.md`, `docs/PERF_PLAN.md` (parked),
   and this plan. Keep it short — CLAUDE.md is an index, not a manual.
3. Add a PARKED banner at the top of `docs/PERF_PLAN.md`: one paragraph
   stating the arc is paused pending `docs/UPSTREAM_PLAN.md` completion
   (repo layout will change under it); do not start Phase 1. Do not
   otherwise edit the file.
4. STATUS block.

**Gates:** none beyond review — docs and a tag only, no code changes.

---

## S1 — CUDA device validation of the CURRENT code

**Model: Opus** (device-divergence analysis needs numerical judgment).
Depends on: S0. **Hybrid execution** — this sub-plan defines the
protocol the arc reuses.

**Context.** The entire suite and all demos have only ever run on the
Serial (host) execution space. Two consumers need device evidence: the
Kokkos RFC (S4) — "does it run on device?" is the first question
maintainers will ask — and the restructure (S2/S5), which needs a
pre-churn device baseline so later device failures are attributable.
This sub-plan validates the code **as it is today**; it does not
restructure anything. Rule 4 applies in full here: if a device run
exposes a library bug, report it in STATUS and stop — do not patch
`third_party/include/*.hpp`.

**Hybrid protocol** (referenced by S8): the session writes runnable
scripts plus a numbered command list into `validation/<arch>/README.md`;
Reet executes on the GPU machine and pastes back full stdout/stderr (or
commits the log files); the session analyzes, saves curated artifacts
under `validation/<arch>/`, and writes the STATUS block. Iterate as
needed — expect 2–3 round trips.

**Read:** `README.md` Sections 4–5, `scripts/build_with_kokkos.sh`,
`scripts/prepare.sh`, `tests/CMakeLists.txt` (the EFT-helper comments
about `--fmad` flags), `tests/README.md` intro only.

**Deliverables.**
1. Build recipe for a CUDA-enabled Kokkos 5.1 + LIBQUADMATH install on
   the available NVIDIA machine (ask Reet which: A100 → `sm_80` /
   `Kokkos_ARCH_AMPERE80`; B200 → `sm_100` / `Kokkos_ARCH_BLACKWELL100`).
   Note `scripts/build_with_kokkos.sh` hardcodes BLACKWELL100 — provide
   per-arch command lines or a minimal parameterization; do NOT build a
   general multi-arch script (that ambition was deliberately dropped).
2. Run: full ctest suite + all six demos under the CUDA build. First
   determine and record what actually executes on device (demos use
   device Views + `parallel_for`; several tests are host loops — map
   which is which rather than assuming).
3. Analysis: per-test pass/fail table; any device-vs-Serial divergence
   explained (note the EFT/FMA-guard tests already carry
   `--fmad=false/true` handling in `tests/CMakeLists.txt` — check it
   engaged); demo accuracy tables compared against the README Section 2
   Serial numbers.
4. Artifacts committed under `validation/<arch>/` (logs, accuracy
   tables). STATUS block with the pass/fail table and any findings.

**Gates:** an honest, complete record. Failures are findings, not
blockers — reported, not fixed.

---

## S2 — DD vertical slice: standalone core + Kokkos wrapper + naming memo

**Model: Opus** (this sub-plan makes the design decisions every later
sub-plan copies). Depends on: S0. Independent of S1 (can run in
parallel). Build/test gates need the x86_64 Linux host.

**Context.** Prove the standalone architecture on ONE header —
`dd_math.hpp` — before sweeping all six. The slice also doubles as the
concrete exhibit in the Kokkos RFC (S4), so code quality and in-header
documentation matter. Rule 4 is lifted for this sub-plan, for mechanical
restructuring only (see Common context).

**Read:** `third_party/include/dd_math.hpp` in full;
`third_party/include/dd_complex.hpp` (only to confirm what it needs from
dd_math); the "Measured Kokkos coupling" block above.

**Deliverables.**
1. **Config header** `include/EPLIB/config.hpp` (placeholder path),
   self-contained, no Kokkos:
   - `EPLIB_INLINE_FUNCTION`: `__host__ __device__ inline` under
     `__CUDACC__`/`__HIPCC__`, SYCL-appropriate under
     `SYCL_LANGUAGE_VERSION`, plain `inline` otherwise.
   - An on-device detection macro unifying
     `__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__ || __SYCL_DEVICE_ONLY__`
     (replaces the CUDA-only `#ifndef __CUDA_ARCH__` idiom).
   - Internal scalar-math dispatch (`fabs`, `exp`, `floor`, `sqrt`,
     `log`, `isinf`, `ceil`, `copysign`, `atan`, `isfinite`) resolving
     to `std::`/builtins so it is valid in host AND device compilation.
   - `EPLIB_PRINTF` diagnostic policy: plain `printf` on host/CUDA/HIP,
     compile-time removable (`EPLIB_ENABLE_DIAGNOSTICS`), documented
     SYCL caveat.
   - Every choice documented in-header with one-line rationale.
2. `include/EPLIB/dd_math.hpp`: the DD header moved, namespace
   `eplib::`, macros swapped, ZERO Kokkos includes. Mechanical
   translation only.
3. **Compat wrapper** at the old path `third_party/include/dd_math.hpp`:
   includes the standalone header, provides
   `namespace Kokkos::Experimental` aliases (`using DoubleDouble =
   eplib::DoubleDouble;` etc.), and the `Kokkos::`-namespace math
   forwarders (relocate the re-exposure block currently at the bottom of
   dd_math.hpp). Requirement: `dd_complex.hpp`, all DD tests, and both
   DD demos compile **unchanged**.
4. **No-Kokkos compile smoke**: a TU including only
   `include/EPLIB/dd_math.hpp` compiled with plain `g++`/`clang++
   -std=c++17` on a machine with no Kokkos installed (the ARM Mac works
   — the standalone core has no quadmath/x86_64 dependency; only the
   demos/tests do).
5. **Naming memo** appended to the S2 STATUS block: 2–3 candidate
   library/namespace/repo names with collision checks (GitHub search,
   common C++ namespaces, package indexes) and a recommendation. The
   repo rename drops "kokkos" per the ratified decision. Reet ratifies →
   S3 applies.

**Gates:** full ctest 23/23 on the Linux host; byte-identical gate on
`kokkos_ep_demo` and `kokkos_ep_demo_complex`; compile smoke passes; any
non-mechanical site recorded in STATUS instead of improvised.

**Out of scope:** FF/QF/complex headers, CMake packaging, repo rename,
`Kokkos::complex` interop (flagged for S5).

---

## S3 — Name ratification + rename sweep

**Model: Sonnet.** Depends on: S2 **and Reet's decisions** (name choice
+ GitHub repo rename — Reet performs the rename in GitHub settings;
GitHub redirects old URLs automatically). Build/test gates need the
x86_64 Linux host.

**Context.** S2 shipped the slice under the `EPLIB`/`eplib::`
placeholder and a naming memo. Reet has picked the final name and
renamed the GitHub repo. This sub-plan is a mechanical sweep.

**Read:** the S2 STATUS block in `docs/UPSTREAM_PLAN_STATUS.md`; output
of `git grep -l -i eplib`.

**Deliverables.**
1. Placeholder → final name everywhere S2 introduced it (paths,
   namespace, macros, includes).
2. `git remote set-url origin` to the renamed repo; update the repo name
   where it appears forward-looking (README title/intro, CLAUDE.md,
   CMake project name). Do NOT rewrite historical STATUS blocks or
   PORT_NOTES — those are the record.
3. STATUS block.

**Gates:** build + ctest 23/23 green; byte-identical gate on the two DD
demos; `git grep -i eplib` returns nothing (or only this plan's
historical text).

---

## S4 — Kokkos RFC draft

**Model: Opus.** Depends on: S1 (device evidence), S2 (slice exhibit),
S3 (final names). Runs anywhere. **Reet posts it** — the session only
drafts.

**Context.** "RFC" = a design-proposal GitHub issue on `kokkos/kokkos`,
posted before any PR, asking the maintainers the questions whose answers
gate the upstream work. It must be short enough that a maintainer reads
it in five minutes, with links into this repo for depth.

**Read:** `README.md` Sections 1–2 and 4 (evidence to cite); the S1 and
S2 STATUS blocks; `NOTICE.md` (license mapping);
`LICENSES/LicenseRef-DHB-License.txt` §3 (the grant-back clause — quote
it accurately, do not paraphrase from memory).

**Deliverables.**
1. `docs/RFC_KOKKOS_UPSTREAM.md`, ready for Reet to paste as a GitHub
   issue. Structure:
   - Motivation (2 paragraphs): portable extended precision inside
     Kokkos kernels; what exists (three backends, op inventory, accuracy
     tables, 23-test suite, device run from S1).
     Note the planned fourth backend (TF, 3×FP32, 72b — S10) so the RFC
     describes the ladder Kokkos would actually consume: 48b / 53b /
     72b / 96b / 106b. S4 runs before S10, so state TF as planned, not
     as shipped.
   - Proposed consumption model: vendored TPL (like `tpls/mdspan`,
     desul) + thin `Kokkos::Experimental` wrapper preserving the
     existing-API spelling; link the S2 slice as the concrete exhibit.
   - The three gating questions, asked explicitly: (a) vendored-TPL vs
     core-native — which would Kokkos accept? (b) naming/API conventions
     the wrapper should follow; (c) licensing — DHB-License (personal
     copyright + §3 grant-back) and LBNL-BSD in a vendored directory:
     acceptable?
   - What we commit to providing: cross-vendor validation (S8), public
     CI, tagged releases for snapshot syncs.
2. STATUS block; once Reet posts, record the issue URL there.

**Gates:** Reet's review. Nothing merges to Kokkos here.

---

## S5 — Full restructure sweep: FF, QF, complex, tests, demos

**Model: Sonnet** (mechanical repetition of the S2 pattern under hard
gates; escalate via STATUS if a site resists mechanical translation).
Depends on: S3. Runs during the RFC wait — this work serves the
non-Kokkos goal regardless of Kokkos's answer. Needs the x86_64 Linux
host.

**Context.** Apply the S2 pattern to the remaining five headers:
`ff_math.hpp`, `qf_math.hpp`, `dd_complex.hpp`, `ff_complex.hpp`,
`qf_complex.hpp` — standalone core under `include/<name>/`, compat
wrappers at the old `third_party/include/` paths so tests and demos
compile unchanged. Rule 4 lifted, mechanical-only, same as S2.

**Read:** the S2 STATUS block (the pattern and any recorded traps);
`include/<name>/config.hpp` and `include/<name>/dd_math.hpp` (the
worked example); then each header as you convert it.

**Special case to handle (flagged from S2):** the complex headers
reference `Kokkos::complex` (~7 sites, likely interop
conversions). The standalone core must not reference it — move any
`Kokkos::complex` interop into the wrapper layer, and record in STATUS
exactly what moved.

**Deliverables.**
1. Five headers converted; wrappers in place; tests/demos compile
   unchanged (they keep including the old paths — the wrapper layer gets
   continuously validated by the whole suite, which is deliberate).
2. No-Kokkos compile smoke extended to ALL six standalone headers
   (real + complex).
3. STATUS block, including a table: header → converted, wrapper site
   count, anything non-mechanical.

**Gates:** ctest 23/23; byte-identical gate on **all six** demos; a
sanity run of `kokkos_ep_bench_cost` completes; compile smoke green; no
arithmetic changes.

---

## S10 — TF backend (TripleFloat, 3×FP32, 72 bits / ~21.7 digits)

**Model: Sonnet** (the DD→FF and QD→QF ports are the worked pattern;
escalate via STATUS on any site that resists mechanical translation).
Depends on: S5. **Runs before S6** — see "Why here" below. Needs the
x86_64 Linux host.

**Rule 4 posture — read this first.** S10 is the one sub-plan in this arc
that **adds arithmetic**. It therefore sits OUTSIDE the Rule 4
supersession that governs S2/S3/S5, which permits mechanical
restructuring only. The byte-identical gate still applies, but to the
**five existing backends**, which S10 must not perturb; it does not and
cannot apply to TF itself. This is a declared gate regime, not a silent
loosening.

**Context.** TF is a 72-bit rung between `double` (53b) and QF (96b).
The demand is measured, not speculative. A downstream consumer walks a
cost-ordered precision ladder and stops at the first rung that clears a
digit tolerance; that ladder currently steps 48b (FF) → 53b (double) →
96b (QF), and workloads whose conditioning destroys ~9–12 digits land in
the gap and are forced onto QF. Expansion cost scales ~O(k²) — roughly
90 flops/multiply at k=3 versus ~250 at k=4 — so those workloads overpay
~2.8×. Both are FP32-word types on the same units, so that cost ordering
follows from k alone and needs no per-arch benchmark.

The consumer-side evidence and routing data live with the consumer, not
here. This repo's interest is narrower and sufficient: 72 bits is a real
precision, it is the only gap in the ladder under a quad-precision
ceiling, and the library is the right place to supply it.

**Provenance.** QD 2.3.24 is parameterized in word count, so TF is the
k=3 FP32 instantiation of the same LBNL-BSD source QF descends from —
not a new lineage, and no new license obligation beyond what QF already
carries. Apply `docs/PORT_NOTES_QF.md`'s source-fidelity rule 6: port
QD's real code, record every divergence between task text and source.
Reuse FF's EFT primitives verbatim, per the same "reuse, do not
re-derive" mandate QF followed — `two_sum`/`quick_two_sum`/`two_prod`
are word-type properties, not word-count properties, and FF already has
them for FP32.

**Range is inherited, not improved.** TF is 3×FP32 and therefore a
member of the FP32 family for every range-safety purpose: usable
magnitudes ~[3.9e-31, 8.3e34], no better than FF at the top end and
worse at the bottom. Any consumer-side range guard that rejects FF must
reject TF identically. Do not describe TF as a range improvement
anywhere in the docs it touches.

**Oracle.** 72 bits < `__float128`'s 113, so the existing libquadmath
oracle validates TF with full margin. No new dependency, and no MPFR —
that constraint only binds above 113 bits, which this arc does not go.

**Read:** `docs/PORT_NOTES_QF.md` (the port pattern and its traps); the
S5 STATUS block (the post-restructure layout TF must be written into);
`ff_math.hpp` for the EFT primitives to reuse; `qf_math.hpp` for the
renormalization structure at k=4 that TF specializes to k=3.

**Deliverables.**
1. `tf_math.hpp` + `tf_complex.hpp`, written directly into the post-S5
   standalone core layout under `include/<name>/`, with compat wrappers
   at the `third_party/include/` paths matching the S5 convention.
2. `docs/PORT_NOTES_TF.md` in the shape of `PORT_NOTES_QF.md`:
   source-fidelity findings, divergences from QD, term-count and
   threshold derivations for k=3, anything non-mechanical.
3. Test suite extended to TF against the `__float128` oracle, including
   a `tf_eft_test` and an FMA-guard test mirroring the FF/QF pair.
4. A TF demo matching the existing per-backend demo shape.
5. STATUS block: op inventory, measured accuracy per op, any op where TF
   falls short of the ~21.7-digit ceiling and why.

**Gates:** ctest green including the new TF tests; TF resolves ≥ 21
digits on the ops the oracle can check, with any shortfall explained in
STATUS rather than waived; **the five existing backends byte-identical**
(TF is purely additive — if a demo's accuracy table moves, stop);
no-Kokkos compile smoke covers TF; TF compiles for device under the S1
CUDA recipe.

**Why here (S5 → S10 → S6).** S10 runs immediately after S5 and before
packaging for two reasons. First, post-S5 the headers are already
de-Kokkos-ified, so TF is written once into the standalone core instead
of becoming a seventh Kokkos-coupled header that a later sweep has to
convert. Second, S6/S7/S8 each enumerate the backend set — license
manifest and release tag (S6), compile-smoke lanes and device TUs (S7),
the per-arch oracle-free test list and validation matrix (S8). S8 in
particular is hybrid: adding a backend after it means re-running four
GPUs by hand or shipping a matrix with a hole. Doing TF first lets those
three stages enumerate the full set once, correctly, and lets `v0.1.0`
ship complete.

**Not renumbered deliberately.** S10 executes fifth-from-last despite its
number. The dependency graph is the authority on order — S1 already runs
out of numeric sequence — and renumbering S6–S9 would invalidate every
existing cross-reference and STATUS entry.

---

## S6 — Standalone packaging, example, oracle-conditional demos

**Model: Sonnet.** Depends on: S5 + S10 (TF must exist before the
backend set is packaged and tagged). Needs the x86_64 Linux host for the
gates.

**Context.** "Non-Kokkos users can use this" requires a consumable
package, not just decoupled headers.

**Read:** top-level `CMakeLists.txt`; the S5 STATUS block.

**Deliverables.**
1. CMake: the library as an installable header-only `INTERFACE` target
   (`<name>::<name>`) with package-config files so
   `find_package(<name>)` works from a fresh tree. Demos/tests consume
   it internally.
2. `examples/standalone/`: one plain-C++ example (compensated reduction
   or similar) with its own minimal CMakeLists that uses ONLY the
   installed package — no Kokkos anywhere. Header comment: usage
   example, not a benchmark.
3. README section "Using without Kokkos" (install, find_package,
   copy-the-headers alternative, the example).
4. License packaging: SPDX headers already in each file; ensure
   `LICENSES/` + `NOTICE.md` install alongside the headers.
5. Version macros in the config header + first release tag (Reet picks
   the number; suggest `v0.1.0`).
6. Make the demo executables conditional on oracle availability in
   CMake (currently they hard-require quadmath and would fail to compile
   under clang-based toolchains — S8 needs this).
7. STATUS block.

**Gates:** from an empty directory: install the package, build the
standalone example against it with no Kokkos on the system, run it.
Existing build + ctest still green.

---

## S7 — Public CI

**Model: Sonnet.** Depends on: S6 (and therefore S10 — the smoke lanes
below enumerate eight headers, including TF).

**Context.** Kokkos will ask how they know vendored snapshots are green;
a visible CI answers it before they ask. It also protects the
byte-identical/23-test invariants for the rest of the arc.

**Read:** `tests/README.md` build/run section; the S6 STATUS block.

**Deliverables.** GitHub Actions workflows:
1. x86_64 Linux: build Kokkos 5.1 Serial + LIBQUADMATH (cached), build
   repo `-O3 -DNDEBUG`, run full ctest (~630 s optimized — acceptable).
2. No-Kokkos lane: compile smoke of all eight standalone headers (six
   pre-S10 + `tf_math.hpp`/`tf_complex.hpp`) + build
   and run the standalone example, on ubuntu AND macos runners (the
   macOS ARM runner proves the core really is x86_64-independent).
3. Compile-only device lanes: nvcc container compiling a device TU of
   each header (no GPU needed); hipcc equivalent if a suitable container
   is straightforward — otherwise record as future work.
4. README badge. STATUS block.

**Gates:** CI green on `main`.

---

## S8 — Cross-vendor device matrix

**Model: Opus** (per-arch divergence analysis). Depends on: S5 + S6 +
S10 (TF must be in the tree before the matrix is run — this stage is
hybrid and re-running four GPUs to add a backend later is the cost S10's
placement exists to avoid).
**Hybrid execution** — reuse the S1 protocol verbatim.

**Context.** Validate the RESTRUCTURED code on the four targets: NVIDIA
A100 (CUDA, `Kokkos_ARCH_AMPERE80`), B200 (CUDA,
`Kokkos_ARCH_BLACKWELL100`), AMD MI250 (HIP, `Kokkos_ARCH_AMD_GFX90A`),
Intel PVC (SYCL, `Kokkos_ARCH_INTEL_PVC`). Known constraint: hipcc/icpx
are clang-based, so the GCC-flavored quadmath oracle
(`-fext-numeric-literals`, `Kokkos_ENABLE_LIBQUADMATH`) will likely be
unavailable there — oracle-dependent tests skip (exit 77) by design and
demos are oracle-conditional since S6. The meaningful per-arch signal is:
library headers compile for device; oracle-free tests (`ff_eft_test`,
`qf_eft_test`, `tf_eft_test`, the FF/QF/TF FMA guards, and the three
invariant tests) run green; demos run where the oracle exists.

**Read:** the S1 STATUS block (protocol + CUDA baseline); `validation/`
layout; `tests/README.md` "graceful degradation" section.

**Deliverables.**
1. Per-target build recipes (Kokkos + repo), executed by Reet per the
   hybrid protocol.
2. Per-target artifacts under `validation/<arch>/` and a consolidated
   matrix table (arch × {header compile, oracle-free tests, oracle
   tests, demos}) in the STATUS block.
3. CUDA re-run compared against the S1 baseline — restructure must not
   have moved device results.
4. Findings reported, not fixed (Rule 4 posture returns: the library is
   supposed to be stable again after S5).

**Gates:** honest matrix; CUDA parity with S1; every red cell has an
explanation or a filed finding.

---

## S9 — Kokkos PR packaging

**Model: Opus.** Depends on: **Kokkos's RFC answer** + S5–S8. Do not
start before the answer exists.

**Context.** Shape depends entirely on the RFC outcome, so this sub-plan
is deliberately thin — it begins by re-planning against what Kokkos
actually said.

**If Kokkos accepts the vendored-TPL model:** prepare the snapshot
(library headers + licenses at the release tag), write the wrapper
header to their stated conventions, follow their contribution/CI
process, open the PR citing the RFC, the accuracy tables, the 23-test
suite, and the S8 matrix.

**If Kokkos wants core-native instead:** start from the
`kokkos-native-freeze` tag, port post-freeze improvements onto it, and
write a fresh plan — do not force-fit this one.

**Read:** the RFC thread (URL in the S4 STATUS block); the S8 matrix;
then whatever Kokkos's contributing docs require.

**Gates:** the PR is theirs to review; this repo's gate is that
everything the PR claims is reproducible from committed artifacts.

---

## Decision points for Reet (not Claude tasks)

1. After S2: ratify the library/repo name; rename the GitHub repo.
2. After S4: review and post the RFC issue.
3. Before S10: confirm TF ships in `v0.1.0` rather than waiting for a
   later release. Deferring it re-opens the S6/S7/S8 enumeration cost
   that S10's placement is designed to avoid.
4. At S6: pick the first release version number.
5. On Kokkos's reply: choose the S9 shape.

## Sub-plan → model map

| Sub-plan | What | Model | Why |
|---|---|---|---|
| S0 | Freeze tag + CLAUDE.md + PERF_PLAN banner | Sonnet | Mechanical docs/tag work |
| S1 | CUDA device validation (current code) | Opus | Device-divergence analysis is numerically subtle |
| S2 | DD slice: config header, standalone core, wrapper, naming memo | Opus | The design decisions everything else copies |
| S3 | Name ratification sweep + repo rename follow-up | Sonnet | Mechanical rename under hard gates |
| S4 | Kokkos RFC draft | Opus | Persuasive technical writing; licensing precision |
| S5 | Full restructure sweep (FF/QF/complex) | Sonnet | Repetition of the proven S2 pattern; gates catch drift |
| S10 | TF backend (3×FP32, 72b) — runs after S5, before S6 | Sonnet | QD→QF port pattern reapplied at k=3; additive, gates catch drift |
| S6 | Packaging + standalone example + oracle-conditional demos | Sonnet | Standard CMake/packaging work |
| S7 | Public CI | Sonnet | Standard GitHub Actions work |
| S8 | Cross-vendor device matrix | Opus | Per-arch toolchain + numerical judgment |
| S9 | Kokkos PR (blocked on RFC answer) | Opus | Adapts to maintainer feedback; judgment-heavy |

## Dependency graph

```
S0 ──> S1 ──────────────┐
  └──> S2 ─(Reet)─> S3 ─┼─> S4 ─(Reet posts; wait)─────> S9
                    S3 ─┴─> S5 ─> S10 ─> S6 ─> S7         ^
                                          └──> S8 ────────┘
```

S1 and S2 run in parallel. S5, S10 and S6–S8 run during the RFC wait and
ship the non-Kokkos goal regardless of Kokkos's answer. Only S9 blocks
on Kokkos. S10 executes fifth-from-last despite its number: this graph,
not the numbering, is the authority on order.

---

## Appendix — How to run this plan

One sub-plan = one fresh Claude Code session. Before starting a session:

1. `git pull` on the machine you're using (sessions alternate between
   machines; an unpushed sub-plan is invisible to its successor).
2. Set the model per the table above (`/model opus` or `/model sonnet`).
3. Run the session on the machine the sub-plan needs: S2/S3/S5/S6 on the
   x86_64 Linux host (their gates run ctest + demo diffs in-session);
   S0/S4 anywhere; S1/S8 anywhere (hybrid — you execute the GPU
   commands and paste output back, expect 2–3 round trips).
4. When the sub-plan merges: confirm the STATUS block landed in
   `docs/UPSTREAM_PLAN_STATUS.md`, then `git push`.

Kickoff prompts (copy-paste verbatim; do not add context — the doc is
the spec and the STATUS file is the inter-session memory):

**S0** — Sonnet, any machine, no preconditions:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S0. Then execute S0. Do not read the rest of the plan or the
whole repo.
```

**S1** — Opus, any machine (hybrid), after S0. I will run your GPU
commands and paste output back:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S1. Read the S0 STATUS block in docs/UPSTREAM_PLAN_STATUS.md.
Then execute S1 using its hybrid protocol: prepare scripts and exact
commands for me to run on the GPU machine; I will paste the output
back. Ask me which NVIDIA machine (A100 or B200) is available before
writing the build recipe. Do not read the rest of the plan or the
whole repo.
```

**S2** — Opus, x86_64 Linux host, after S0 (parallel with S1). Start in
plan mode:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S2. Read the S0 STATUS block in docs/UPSTREAM_PLAN_STATUS.md.
Propose the config-header design and the slice migration plan for my
approval before editing any file, then execute S2. Do not read the
rest of the plan or the whole repo.
```

**S3** — Sonnet, x86_64 Linux host. BLOCKED until I have ratified the
name from S2's memo and renamed the GitHub repo. Replace <NAME> below:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S3. Read the S2 STATUS block in docs/UPSTREAM_PLAN_STATUS.md.
The ratified library/namespace name is <NAME> and the GitHub repo has
been renamed to <REPO-NAME>. Execute S3. Do not read the rest of the
plan or the whole repo.
```

**S4** — Opus, any machine, after S1 + S2 + S3:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S4. Read the S1, S2, and S3 STATUS blocks in
docs/UPSTREAM_PLAN_STATUS.md. Then execute S4 (draft only — I will
post the RFC myself). Do not read the rest of the plan or the whole
repo.
```

**S5** — Sonnet, x86_64 Linux host, after S3:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S5. Read the S2 and S3 STATUS blocks in
docs/UPSTREAM_PLAN_STATUS.md. Then execute S5. If any site resists a
mechanical translation, stop and record it in STATUS instead of
improvising. Do not read the rest of the plan or the whole repo.
```

**S6** — Sonnet, x86_64 Linux host, after S5:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S6. Read the S5 STATUS block in docs/UPSTREAM_PLAN_STATUS.md.
Then execute S6. Ask me for the release version number before tagging.
Do not read the rest of the plan or the whole repo.
```

**S7** — Sonnet, any machine (CI runs on GitHub), after S6:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S7. Read the S6 STATUS block in docs/UPSTREAM_PLAN_STATUS.md.
Then execute S7. Do not read the rest of the plan or the whole repo.
```

**S8** — Opus, any machine (hybrid), after S5 + S6. I will run your GPU
commands on each machine and paste output back:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S8. Read the S1, S5, and S6 STATUS blocks in
docs/UPSTREAM_PLAN_STATUS.md. Then execute S8 using the S1 hybrid
protocol, one target at a time; ask me which of the four machines to
start with. Do not read the rest of the plan or the whole repo.
```

**S9** — Opus, any machine. BLOCKED until Kokkos answers the RFC:
```
Read docs/UPSTREAM_PLAN.md — ONLY the "Common context" section and
section S9. Read the S4 and S8 STATUS blocks in
docs/UPSTREAM_PLAN_STATUS.md, then read the Kokkos RFC thread at the
URL recorded in the S4 STATUS block. Summarize Kokkos's answer and
propose the PR plan for my approval before implementing anything.
Do not read the rest of the plan or the whole repo.
```
