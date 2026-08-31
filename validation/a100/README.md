# `validation/a100/` — CUDA device validation on NVIDIA A100 (S1)

Scaffolding for **S1** of `docs/UPSTREAM_PLAN.md`: validate the code **as it is
today** on a real CUDA device, so (a) the Kokkos RFC in S4 can answer "does it
run on device?", and (b) S2/S5 have a pre-restructure device baseline.

**Rule 4 applies in full.** Nothing in `third_party/include/` is touched, and no
source, test, or CMake file is modified. If a device run exposes a library bug,
that is a **finding for the STATUS block**, never a patch.

**This directory is the authoring half.** Scripts and the command list are here;
Reet runs them on JLSE and hands the logs back; a later session analyzes and
writes the STATUS block. S1 is **not complete** until that happens.

## Target

| | |
|---|---|
| GPU | NVIDIA A100 → `sm_80` → `Kokkos_ARCH_AMPERE80`, `CMAKE_CUDA_ARCHITECTURES=80` |
| Machine | JLSE, queue `gpu_a100` (nodes `gpu06`, `gpu07`), project `pepper_hep` |
| Scheduler | **Cobalt** (max walltime `06:00:00`) |
| Modules | `gcc/13.3.0`, `cmake/3.28.3`, `cuda/12.9.1` via `module use /soft/modulefiles` — the same set `scripts/prepare.sh` loads |
| Kokkos | **a new install is required.** All three existing installs (`~/kokkos-install`, `~/kokkos-install-quadmath`, `~/kokkos-install-openmp`) are Serial-only — none defines `KOKKOS_ENABLE_CUDA`. |

`scripts/build_with_kokkos.sh` hardcodes `Kokkos_ARCH_BLACKWELL100` /
`CMAKE_CUDA_ARCHITECTURES=100` **and** builds the repo with plain `g++`, which
cannot compile against a CUDA-backed Kokkos. Per S1 deliverable 1 this directory
carries an explicit **sm_80-only** recipe rather than a general multi-arch
script.

## Split: build on the login node, run on the GPU

`nvcc` cross-compiles for `sm_80` with no A100 present, and compiling Kokkos +
this repo takes much longer than running the suite. So:

* **`build_login.sh`** — login node. Clones Kokkos 5.1, applies the local
  `__complex128` oracle header, installs CUDA+LIBQUADMATH Kokkos, builds this
  repo against it, and records the build-side evidence.
* **`run_a100.sh`** — inside the Cobalt job. Compiles nothing; runs `ctest` and
  the six demos and tees everything into `logs/run/`.

The repo build directory defaults to `~/kokkos-ep-build-a100`, deliberately
**outside** the repo: `.gitignore` covers `/build/` only, and S1 must not modify
files outside `validation/a100/`.

## Numbered command list

Run 1–3 on a **JLSE login node**, from the repo root.

```
1.  cd ~/kokkos-extended-precision-demo
    git rev-parse HEAD          # expect 9d23d90 (or the S1 commit on top of it)

2.  bash validation/a100/build_login.sh 2>&1 | tail -40
    #  ~30-60 min. Installs:
    #    Kokkos source  : ~/kokkos-src-5.1.0-cuda-sm80
    #    Kokkos install : ~/kokkos-install-cuda-sm80-quadmath
    #    repo build     : ~/kokkos-ep-build-a100
    #  Logs land in validation/a100/logs/build/.
    #  Override with JOBS=..., KOKKOS_PREFIX=..., REPO_BUILD=... if needed.

3.  cat validation/a100/logs/build/kokkos_config_defines.log
    cat validation/a100/logs/build/contraction_flags.log
    #  Sanity-check BEFORE burning the reservation:
    #    - KOKKOS_ENABLE_CUDA and KOKKOS_ENABLE_LIBQUADMATH are both defined
    #    - KOKKOS_ARCH_AMPERE80 is defined
    #    - all six demos + 23 test binaries exist (targets.log)
```

Then submit the run. **Script mode** (automatable — preferred):

```
4a. qsub -A pepper_hep -t 120 -n 1 -q gpu_a100 --mode script \
         validation/a100/run_a100.sh

    # If you overrode any path in step 2, pass it through — script mode does
    # not inherit your login shell:
    # qsub -A pepper_hep -t 120 -n 1 -q gpu_a100 --mode script \
    #      --env REPO_ROOT=$PWD:REPO_BUILD=$HOME/kokkos-ep-build-a100 \
    #      validation/a100/run_a100.sh
```

Or **interactive** (the usual habit):

```
4b. qsub -A pepper_hep -I -t 120 -n 1 -q gpu_a100
    # inside the interactive shell:
    cd ~/kokkos-extended-precision-demo
    bash validation/a100/run_a100.sh
```

```
5.  ls validation/a100/logs/run/
    #  00_env.log, 01_ctest_verbose.log, 01_ctest_summary.log,
    #  01_ctest_list.log, 01_ctest_rc.log, 02_kokkos_ep_demo*.log (x6)

6.  git add validation/a100/logs && git commit -m "upstream: S1 — A100 run logs"
    #  (or paste the logs back into the session)
```

**Walltime.** `-t 120` rather than the usual `-t 60`: `qf_accuracy_test` alone
takes ~310 s on Serial, and the six demos each run 10⁶ × 5 repeats with a
`__float128` **host** oracle whose cost does not shrink on a GPU. 120 min is
comfortable and well under the 360 min cap. If the job is killed on walltime,
the `ctest` log is already complete — it runs first, by design.

**Script-mode caveat.** If Cobalt executes the script off the compute node,
`run_a100.sh` aborts at step 0 with `FATAL: no NVIDIA device visible` rather
than silently producing a Serial run. If that happens, fall back to 4b and
report it — a Serial-fallback log would be worthless S1 evidence.

## What runs ON DEVICE vs. what is a host loop

**How this was determined** (not assumed): every `tests/*.cpp`, `src/*.cpp`, and
`tests/test_utils.hpp` was grepped for `parallel_for` / `parallel_reduce` /
`RangePolicy` / `Kokkos::View<` / `DefaultExecutionSpace`, and each hit was read
in context. The shared harness helpers `run_unary_op`, `run_binary_op`,
`run_unary_op_on_corpus`, `run_binary_op_on_corpus` in `tests/test_utils.hpp`
(lines 305–490) all declare `using exec_space = Kokkos::DefaultExecutionSpace`,
allocate `Kokkos::View<T*, LayoutRight, exec_space>`, `deep_copy` a host mirror
in, launch `Kokkos::parallel_for(..., RangePolicy<exec_space>(0,n), ...)`, and
mirror the results back — so **any test calling those helpers evaluates the math
on the device**. No test hardcodes `Kokkos::Serial`; the only `Serial` hits are
comments. Under a CUDA-enabled Kokkos, `DefaultExecutionSpace` is `Cuda`, and
every test prints it (`Execution space: %s`), which is why `ctest -V` is used.

### Tests (23 registered)

| ctest name | Device? | Where the device work is |
|---|---|---|
| `hello_test` | **device** | whole test is `run_unary_op<DD>` (harness helper) |
| `corpus_test` | host only | no Kokkos math at all — corpus-data checks (`corpus.hpp`); no `View`, no `parallel_for` |
| `dd_eft_test` | **mixed** | Tests A–C host loops; **Test D** device-parity `parallel_for("dd_eft_device")` |
| `ff_eft_test` | **mixed** | Tests A–C host; **Test D** `parallel_for("ff_eft_device")` |
| `qf_eft_test` | **mixed** | Tests A–D host; **Test E** `parallel_for("qf_eft_device")` |
| `dd_invariant_test` | **mixed** | host sweep over all ops + dedicated device passes `dd_inv_dev_unary` / `dd_inv_dev_binary` |
| `ff_invariant_test` | **mixed** | same shape: `ff_inv_dev_unary` / `ff_inv_dev_binary` |
| `qf_nonoverlap_test` | **mixed** | same shape: `qf_nonoverlap_dev_unary` / `qf_nonoverlap_dev_binary` |
| `dd_property_test` | **mixed** | Groups A/B host; one device pass `parallel_for("dd_prop_dev")` |
| `ff_property_test` | **mixed** | one device pass `ff_prop_dev` |
| `qf_property_test` | **mixed** | one device pass `qf_prop_dev` |
| `dd_accuracy_test` | **device** | every scored op goes through `run_unary_op` / `run_binary_op` (+ explicit `fma_acc` / `pow_int_acc` kernels); only oracle scoring is host |
| `ff_accuracy_test` | **device** | same |
| `qf_accuracy_test` | **device** | own 4-word-View helpers `qf_acc_unary` / `qf_acc_unary_corpus` / `qf_acc_binary` / `qf_acc_binary_corpus`, all `RangePolicy<DefaultExecutionSpace>` |
| `dd_e2e_test` | **host only** | explicit by design — "Kokkos is initialized/finalized … but no `parallel_for` is spawned" (`dd_e2e_test.cpp:62`); the four cancellation kernels are inherently serial |
| `ff_cancellation_test` | **host only** | same rationale (`ff_cancellation_test.cpp:61`) |
| `qf_cancellation_test` | **host only** | same rationale (`qf_cancellation_test.cpp:73`) |
| `dd_fma_guard_test` | **mixed** | host pass + `parallel_for("dd_fma_guard_device")` |
| `dd_fma_guard_test_contract_on` | **mixed** | same source, contraction-ON build |
| `ff_fma_guard_test` | **mixed** | host pass + `parallel_for("ff_fma_guard_device")` |
| `ff_fma_guard_test_contract_on` | **mixed** | same source, contraction-ON build |
| `qf_fma_guard_test` | **mixed** | host pass + `qf_fma_guard_prod_device` and `qf_fma_guard_sqr_device` |
| `qf_fma_guard_test_contract_on` | **mixed** | same source, contraction-ON build |

Totals: **4 pure host** (`corpus_test`, `dd_e2e_test`, `ff_cancellation_test`,
`qf_cancellation_test`), **4 essentially all-device** (`hello_test` + the three
accuracy tests), **15 mixed** host+device.

Consequence for S1: the three end-to-end cancellation tests and `corpus_test`
**cannot** produce device evidence. If they pass on the A100, that says nothing
about the device — do not count them as device coverage in the STATUS table.

### Demos (all six: device)

| demo | device kernels |
|---|---|
| `kokkos_ep_demo` (DD real) | 78 `parallel_for` sites, 2 device Views |
| `kokkos_ep_demo_complex` (DD complex) | 48 `parallel_for` sites, 3 device Views |
| `kokkos_ep_demo_ff` | 78 / 2 |
| `kokkos_ep_demo_ff_complex` | 48 / 3 |
| `kokkos_ep_demo_qf` | 78 / 2 |
| `kokkos_ep_demo_qf_complex` | 48 / 3 |

Each demo generates inputs on the host, `deep_copy`s to a device View, runs the
op in a `parallel_for`, copies back, and scores against the `__float128` /
`__complex128` **host** oracle. So the demos are the strongest device evidence
in the repo: the arithmetic itself runs on the A100.

**`kokkos_ep_bench_cost` is deliberately excluded** from the run. It has zero
`Kokkos::View` and zero `parallel_for` — it is a pure host timing loop, so it
would produce no device evidence and only consume reservation time.

## The `--fmad` handling in `tests/CMakeLists.txt` — what to check

`tests/CMakeLists.txt` carries two contraction postures for the EFT and
FMA-guard targets:

| helper | host flag | CUDA flag | line |
|---|---|---|---|
| `kokkos_ep_add_eft_test` | `-ffp-contract=off` under `$<COMPILE_LANGUAGE:CXX>` | `--fmad=false` under `$<COMPILE_LANGUAGE:CUDA>`, guarded by `if(Kokkos_ENABLE_CUDA)` | 70–82 |
| `kokkos_ep_add_eft_test_contract_on` | `-ffp-contract=fast` | `--fmad=true` (nvcc's default, stated for symmetry) | 118–128 |

Applied to: `dd_eft_test`, `ff_eft_test`, `qf_eft_test`, and both postures of
`dd_fma_guard_test` / `ff_fma_guard_test` / `qf_fma_guard_test`. The point is
Dekker's `twoProduct`: if `a1*b1 - c11` is contracted into a single FMA the
error term collapses to zero and the EFT silently breaks.

**The analysis half must confirm the flag actually engaged, and there is
specific reason to expect it did not.** Two things to reconcile:

1. **The generator expression.** The top-level `CMakeLists.txt:3` declares
   `project(kokkos-extended-precision-demo LANGUAGES CXX)` — CUDA is not an
   enabled language, and no source is marked `LANGUAGE CUDA`. A CUDA-backed
   Kokkos is consumed by compiling ordinary `.cpp` files with `nvcc_wrapper` as
   `CMAKE_CXX_COMPILER`, so `COMPILE_LANGUAGE` is `CXX` for every source in the
   project. If that holds, `$<COMPILE_LANGUAGE:CUDA>` never evaluates true and
   **`--fmad=false` is never emitted**, even though `if(Kokkos_ENABLE_CUDA)` is
   true.
2. **Where `-ffp-contract=off` lands.** `nvcc_wrapper`'s argument parser routes
   unrecognized flags through its default `*)` case into `xcompiler_args`, which
   are handed to nvcc as `-Xcompiler …` — i.e. to the **host** compiler only
   (verified by reading `~/kokkos-install-quadmath/bin/nvcc_wrapper`, the `*)`
   case near line 504 and the command assembly near lines 566–575). So
   `-ffp-contract=off` constrains the host pass and leaves nvcc's **device**
   codegen at its default `--fmad=true`. Note `nvcc_wrapper` *does* have an
   explicit `--fmad=*` case (near line 239) routing it to `cuda_args`, so the
   flag would work **if it were emitted** — the gap, if confirmed, is the
   generator expression, not the wrapper.

**Checks the analysis half must perform**, against
`logs/build/contraction_flags.log` (produced by `build_login.sh`, which dumps
the real `compile_commands.json` entries plus occurrence counts):

- **C1.** Does `--fmad` appear in any compile line? Predicted: **0
  occurrences**. Record the actual count.
- **C2.** Does `-ffp-contract=off` / `=fast` appear, and on which targets?
  Predicted: yes, on the six OFF targets and three ON targets respectively —
  but host-side only.
- **C3.** What is `CMAKE_CXX_COMPILER_ID` in the cache? If it is not `GNU`, the
  `-ffp-contract` lines never fired either, **and** the top-level
  `-fext-numeric-literals` branch was skipped (`build_login.sh` passes that flag
  explicitly so the build does not depend on the answer — but the answer still
  needs recording).
- **C4.** Reconcile with the runtime results. If C1 is 0 and the guard tests'
  **device** passes report collapsed error terms (`ERR_ZERO`) or mismatches,
  that is **explained by the build plumbing, not by a defect in
  `dd/ff/qf_math.hpp`**. Say so explicitly in STATUS; do not read it as a
  library finding.
- **C5.** The `_contract_on` variants compare against
  `tests/*_fma_guard_baseline.txt`, recorded under GCC/Serial. Drift warnings on
  an A100 are **expected and are reports, not failures** — the ON variants exit
  0 by design (QF's exits non-zero only on `ERR_NONZERO_WRONG`). Record the
  drift; do not regenerate the baselines in S1.

If C1 confirms the flag never engaged, the remedy (adding `CUDA` to `project()`
`LANGUAGES`, or attaching `--fmad=false` as a plain `CXX` flag that
`nvcc_wrapper` forwards to `cuda_args`) is **out of scope for S1** — S1 forbids
source/CMake changes. Record it as a finding and hand it to a later sub-plan.

## Anticipated failure modes (capture, don't fix)

Every one of these is a **finding**. Paste the error, do not patch.

* **`__float128` under nvcc.** The oracle is host-only, but the whole
  translation unit passes through nvcc's frontend. `Kokkos_ENABLE_LIBQUADMATH`
  combined with `Kokkos_ENABLE_CUDA` is not a well-trodden upstream
  configuration; a frontend failure on `quadmath.h`, on `1.0Q` literals, or on
  `__complex128` would block the demos and is itself S1-relevant evidence for
  the RFC.
* **The `__complex128` oracle header.** `patches/kokkos_complex_quad_math.hpp`
  is a local, non-upstream extension (`patches/README.md`).
  `build_login.sh` copies it into the Kokkos source tree before configuring and
  falls back to copying it into the install tree. If
  `KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER` comes back false in
  `logs/build/repo_configure.log`, the three `*_complex` demos will not build.
* **`Kokkos::printf` diagnostics.** The headers carry ~40 `Kokkos::printf` sites
  behind `#ifndef __CUDA_ARCH__` guards. On device those guards flip; watch for
  either lost diagnostics or a flood of per-thread output.
* **Runtime.** Device runs add kernel-launch overhead per op;
  `qf_accuracy_test` issues ~9 launches per op across ~49 ops. `CTEST_TIMEOUT`
  defaults to 1800 s per test.
* **`ctest` non-zero exit.** Expected to be possible. S1's gate is "an honest,
  complete record" — failures are findings, not blockers.

## Evidence the analysis half needs

Commit (or paste) all of:

| file | why |
|---|---|
| `logs/build/env.log` | toolchain versions + repo SHA the binaries were built from |
| `logs/build/kokkos_config_defines.log` | proves `KOKKOS_ENABLE_CUDA` + `KOKKOS_ENABLE_LIBQUADMATH` + `KOKKOS_ARCH_AMPERE80` |
| `logs/build/repo_configure.log` | LIBQUADMATH / complex-wrapper probe results, and any warnings |
| `logs/build/repo_build.log` | compile warnings and any target that failed to build |
| `logs/build/contraction_flags.log` | **checks C1–C3** above |
| `logs/run/00_env.log` | `nvidia-smi` — proves an A100 was actually present |
| `logs/run/01_ctest_verbose.log` | per-test pass/fail **and** each test's `Execution space:` banner (must read `Cuda`, not `Serial`) |
| `logs/run/01_ctest_summary.log`, `01_ctest_rc.log` | the pass/fail table for STATUS |
| `logs/run/02_kokkos_ep_demo*.log` (×6) | accuracy tables to diff against README Section 2's Serial numbers |

The analysis session then produces, per S1 deliverable 3–4:

1. a per-test pass/fail table, annotated with the device/host mapping above (a
   pass in `corpus_test` / the three cancellation tests is **not** device
   evidence);
2. every device-vs-Serial divergence explained, with the C1–C5 contraction
   findings folded in;
3. demo accuracy tables compared against README Section 2 — note that the
   byte-identical gate is a **S2/S3/S5** device-vs-host tool; here a divergence
   is expected to be *characterized*, not gated to zero, because device
   transcendental/rounding behavior can legitimately differ;
4. curated artifacts under `validation/a100/` and the STATUS block appended to
   `docs/UPSTREAM_PLAN_STATUS.md`.
