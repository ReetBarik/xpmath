# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Extended-precision arithmetic as header-only C++, in two layers:

**Standalone core — `include/xp/`.** Zero Kokkos. Needs only the C++17 standard
library, and compiles under plain `g++`/`clang++`, `nvcc` and `hipcc`.

| backend | type | words | digits | headers |
|---|---|---|---|---|
| DD | `xp::DoubleDouble` | 2×FP64 | ~31 | `dd_math.hpp`, `dd_complex.hpp` |
| FF | `xp::FloatFloat` | 2×FP32 | ~14 | `ff_math.hpp`, `ff_complex.hpp` |
| QF | `xp::QuadFloat` | 4×FP32 | ~29 | `qf_math.hpp`, `qf_complex.hpp` |
| TF | `xp::TripleFloat` | 3×FP32 | ~21.7 | `tf_math.hpp`, `tf_complex.hpp` |

`include/xp/config.hpp` is shared by all eight math/complex headers and supplies what Kokkos used
to: `XPMATH_INLINE_FUNCTION` (`__host__ __device__ inline` under CUDA/HIP), an
`XPMATH_ON_DEVICE` predicate unifying `__CUDA_ARCH__` / `__HIP_DEVICE_COMPILE__` /
`__SYCL_DEVICE_ONLY__`, an `xp::detail::` scalar-math dispatch, and a
compile-time-removable `XPMATH_PRINTF`. It also carries
`XPMATH_NOINLINE_FUNCTION`, which is not a Kokkos replacement but a gfx90a
codegen mitigation — see Platform Constraints.

**Kokkos compat wrappers — `third_party/include/`.** Same filenames as the core.
Each includes its `include/xp/` counterpart and re-exposes it as
`Kokkos::Experimental::DoubleDouble` etc. plus `Kokkos::`-namespace math
forwarders, so existing Kokkos code compiles unchanged. Tests and demos include
these paths, which keeps the wrapper layer continuously validated.

Validation: **one** measurement — error in ulps against an **MPFR/MPC oracle at
400 bits, carried in `__float128`** — with **one** verdict per point against a
bound derived from the format and the condition number, and **two** ctest gates
over that number (`sweep_absolute_gate`, `sweep_monotone_gate`), each with a
self-test target (`*_selftest`) that poisons its input and requires it to fail.
Read **docs/CORRECTNESS.md** before adding anything that judges correctness; the
whole point is that nothing else issues a competing verdict. 49 ctest targets
with Kokkos and **21 without** (`-DXPMATH_WITH_KOKKOS=OFF`), all passing on
`main` — asserted by CI, not assumed. The Kokkos-free 21 include both gates:
`sweep_accuracy` links no Kokkos and never needed it, but until
`XPMATH_WITH_KOKKOS` existed the build required it anyway.

**The sweep oracle was libquadmath and is not any more.** `sweep_accuracy` links
`-lmpc -lmpfr -lgmp` and **no `-lquadmath`**, and that is asserted rather than
hoped for — two acceptance checks in the header of `scripts/sweep_accuracy.cpp`
(`ldd | grep -i quadmath` empty, `nm -D --undefined-only | grep -E 'q$'` empty).
`__float128` is still the CARRIER for ulp arithmetic, widenings and CSV
formatting; its arithmetic comes from libgcc and its elementary functions from
glibc's `*f128`, neither of which is libquadmath. MPFR/MPC are therefore a HARD
requirement of `tests/CMakeLists.txt` — a missing `-dev` package is a configure
`FATAL_ERROR`, not four silently unregistered targets. `--oracle=mpfr` survives
as an accepted no-op (CI's monotone gate passes it to the parent binary, which
may predate the switch); `--oracle=quadmath` is REFUSED with an explanation.
`scripts/attic/gen_corpus.cpp` still scores against libquadmath and its numbers
are not comparable to the sweep's point for point.

## Executables

Eight targets in CMakeLists.txt, all unconditional. (The four complex demos used
to sit inside a `KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER` conditional; that probe and
the patch header behind it are gone.)

- `kokkos_ep_demo` — DD real (39 ops)
- `kokkos_ep_demo_complex` — DD complex (24 ops)
- `kokkos_ep_demo_ff` — FF real (39 ops)
- `kokkos_ep_demo_ff_complex` — FF complex (24 ops)
- `kokkos_ep_demo_qf` — QF real (39 ops)
- `kokkos_ep_demo_qf_complex` — QF complex (24 ops)
- `kokkos_ep_demo_tf` — TF real (39 ops)
- `kokkos_ep_demo_tf_complex` — TF complex (24 ops)

**The demos are TIMING AND SMOKE ONLY.** They print per-op wall time (DD: the
slowdown vs FP64) and nothing else. They used to print accuracy columns scored
against a host `__float128` oracle, and the QF/TF pairs used to issue an
RC-0/RC-1 verdict on them; all of it is deleted. The accuracy record is
`validation/sweep/`, and the correctness contract allows exactly one verdict per
point (`docs/CORRECTNESS.md`). Each demo still exercises every op it always did
and still reads its results back host-side — that round trip is the smoke.

**Never run the demos casually** — they are hours of kernel time.

## Branch Structure

| Branch | Backends | Requirement |
|---|---|---|
| `main` | DD + FF + QF + TF (portable) | any Kokkos-compatible hardware |
| `CUDAFP128Kokkos` | CUDA FP128 only | compute ≥ 10.0 (sm_100, Blackwell) |

`CUDAFP128Kokkos` cannot merge into `main` — it needs sm_100 and is kept as a
separate reference implementation.

Tag `kokkos-native-freeze` marks the last fully Kokkos-native state, before the
standalone extraction. If a Kokkos-core-native contribution is ever requested,
work restarts from there.

## Build

Requires Kokkos ≥5.1 built at **C++20**, GCC 13.3.0, CMake 3.28.3. The consuming
project stays at C++17.

**`Kokkos_ENABLE_LIBQUADMATH=ON` is NOT required any more, and neither is a
patched Kokkos.** It was, until the demos stopped printing accuracy columns: the
real oracle came through `impl/Kokkos_QuadPrecisionMath.hpp` (upstream, but only
compiled with that flag) and the complex oracle through
`impl/Kokkos_ComplexQuadPrecisionMath.hpp`, which is NOT upstream and had to be
patched into every Kokkos install by hand. A Kokkos with libquadmath OFF — the
`~/xpm_device/kokkos-hip-gfx90a` (MI250) install among them — now CONFIGURES
this project cleanly with no warning. Configures; that is a narrower claim than
builds, and the MI250 build state is now MEASURED rather than inherited: S8c
(Cobalt `1000943`, 2026-09-16) configured on the card and then **39 of 49
targets did not build**. See the S8c and STEP-2 CLOSEOUT blocks.

**The wrapper is `scripts/xpm_build.sh`. Target hardware is an ARGUMENT.**

```bash
scripts/xpm_build.sh --arch host  --build-dir /tmp/b_host
scripts/xpm_build.sh --arch a100  --build-dir /tmp/b_a100
scripts/xpm_build.sh --arch mi250 --build-dir /tmp/b_mi250
scripts/xpm_build.sh --arch host --no-kokkos --build-dir /tmp/b_nok
```

`--arch {host|a100|mi250}` is required and selects a data row: modules,
compiler, Kokkos prefix, `Kokkos_ARCH_*` macro, FP-contraction spelling.
`--kokkos use-existing` (default) points at the install already on this
machine; `--kokkos build` delegates to `scripts/build_with_kokkos.sh`.
`--opt` defaults to `O3` and travels as `CMAKE_CXX_FLAGS_RELEASE`.
Read the header block of that script before comparing numbers across two
arches — it states what is held identical (`-O` level, C++17, contraction off)
and what is not (compiler, execution space). The host oracle used to be on that
"not held identical" list, because the mi250 Kokkos has no libquadmath and the
oracle-scored tests runtime-SKIPped there. It is not any more: the oracle is
MPFR/MPC and links no libquadmath on any arch, so whether the Kokkos install has
it no longer changes which tests score.

**Measured, 2026-09-16:** `--arch host` 49/49; `--arch host --no-kokkos` 21/21;
`--arch a100` configures and then fails to compile, on the S6 `__float128`
device-TU blocker plus five standalone smokes that nvcc's frontend rejects
(`no operator "<<"`). Neither is this wrapper's to fix.

Every configure — the wrapper's or a bare `cmake -B build` — stamps
`build-info.txt` into the build directory with the arch, git HEAD (`-dirty`
when the tree is not clean), compiler and version, Kokkos prefix, the full
`CMAKE_CXX_FLAGS`, the `-O` level and a UTC timestamp. The `build_provenance`
ctest target fails when it is missing or short a field. Three campaign
artifacts this month could not be traced to a tree; that is the whole reason.

`scripts/build_with_kokkos.sh` still exists and still builds Kokkos from
source, but is now driven by its environment (`KOKKOS_BACKEND`,
`KOKKOS_ARCH_FLAG`, `KOKKOS_CUDA_ARCHITECTURES`, `KOKKOS_CXX`,
`KOKKOS_LIBQUADMATH`, `KOKKOS_ONLY`, `REPO_BUILD_DIR`). The Blackwell/sm_100
values are now just its defaults. Its HIP branch is now reachable — `EXTRA_FLAGS`
is chosen by backend instead of being assigned the CUDA row unconditionally —
but **still unexecuted**: nothing has yet built a Kokkos through it.

(The older C++17-vs-20 trap this section used to describe is FIXED. `dd6d00a`
changed both Kokkos configure lines to `-DCMAKE_CXX_STANDARD=20` on 2026-09-02.
The consuming repo still builds at C++17.)

A bare configure also works, and is what CI does:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
ctest --test-dir build
```

`scripts/check_standalone_no_kokkos.sh` proves the core stands alone: it compiles
each `include/xp/` header with plain `g++ -std=c++17` against an include path
containing **only** `include/` (deliberately excluding `third_party/include/`, so
a compat wrapper cannot mask a missing dependency), then greps the preprocessed
output for the token `Kokkos`.

## Running

```bash
./build/kokkos_ep_demo --batch 1000000 --repeats 5
./build/kokkos_ep_demo --op sin --batch 1000000 --repeats 5
./build/kokkos_ep_demo_ff_complex --batch 1000000 --repeats 5
```

Arguments: `--op <name>`, `--batch N` (default 1,000,000), `--repeats N`
(default 5), `--seed N` (default 12345).

**Timing note.** QF demos are by far the slowest — the sum of median per-op times
is ~219 µs, so each QF demo is ~18 minutes of kernel time at `--batch 1000000
--repeats 5`. Budget accordingly; capture them in the background.

## Validation conventions

**The byte-identical gate is RETIRED, because its subject no longer exists.**
It read: any mechanical restructure must leave the demo accuracy columns
unchanged — run the affected demos before and after with identical arguments,
strip timing with `validation/strip_timing.sh`, and diff. The demos do not print
accuracy columns any more. `validation/strip_timing.sh` still runs, and it does
not fail silently — it keys on number format and only accepts a row as data if
at least one two-decimal accuracy field survives. With none left, `kept == 0` on
every demo row, so every row prints VERBATIM, timings included. The diff it
feeds is therefore pure wall-clock jitter: it will report differences, none of
which mean anything. Do not invoke the gate to bless a change — a red diff from
it is now noise, and there is no accuracy content for it to protect.

What replaced it: the demos are timing and smoke only, so a restructure that
touches them is checked by *compiling and running one* at a small batch
(`--batch 1000 --repeats 1`), and everything about accuracy is checked by the
two ctest gates over `validation/sweep/` (`docs/CORRECTNESS.md`). Those were
always the stronger measurement — the demo columns duplicated them at lower
resolution.

**Shared corpus — `scripts/attic/gen_corpus.cpp`, on `main`, unused.** It emits one
shared set of inputs plus a `__float128` reference per (op, element), covering 39
real + 24 complex ops, so backends could be scored on identical data instead of
each demo generating its own. No `CMakeLists.txt` references it, nothing consumes
it, and the loader that used to pair with it (`tests/corpus_binary.hpp`) is gone;
the `corpus-generator` branch it was developed on no longer exists. Do not
confuse it with **`tests/corpus.hpp`**, which is live and unrelated: that is the
T0.2 corner-case corpus (subnormals, ±inf, half-integer boundaries, the FF
splitter-overflow inputs) consumed by `corpus_test` and the invariant tests.

## Working on accuracy

Read **docs/CORRECTNESS.md** first. The short version:

- One measurement: error in ulps against the MPFR/MPC oracle at 400 bits,
  quantised into `__float128`. The carrier is still `__float128`; the reference
  is not libquadmath any more.
- One verdict per point: at or below its derived bound, or above it.
- The gate allows `kUlpAllowance = 8.0` × the derived bound
  (`scripts/sweep_accuracy.cpp:1937`). A point can therefore exceed its raw
  derived bound and still pass; report both numbers if you claim "at the limit".
- ~8.9% of sweep rows carry state `U`/`N` and get **no verdict** (33,879 `U` +
  4,792 `N` of 436,080) — the format cannot carry the question. Absence of a
  defect there is not evidence of correctness.

**Traps that have cost real time:**

- `sweep_accuracy` with no `--out` does NOT overwrite the committed baseline.
  It computes the whole sweep and then exits 1 with `cannot open  for writing`,
  because the path defaults to the empty string. It overwrote in place once;
  `334cc9b` removed that use and left `kDefaultOut` stranded and unreferenced at
  `scripts/sweep_accuracy.cpp:273`. Corrected in `fb1435d` after measuring
  against a sentinel: exit 1, sentinel intact. `--grid-out` is the same shape.
  Still pass an explicit `--out /tmp/...` — the hazard that remains is handing it
  the committed path yourself.
- `write_baseline` uses plain `fopen` — it does NOT gzip. Writing `--out foo.gz`
  produces uncompressed bytes under a `.gz` name. Write plain, then `gzip -9 -c`.
- `ctest --test-dir` on a MISSING directory exits 0. Confirm the dir exists and
  that a nonzero test count ran, or the run proved nothing.
- **The run-time-library fingerprint trap is RETIRED, and the retirement is
  measured.** It used to be the first thing to suspect: the oracle fingerprint
  was a property of the libquadmath loaded at **run** time, not of the compiler
  that built the binary, because `sweep_accuracy` linked `-lquadmath` with no
  RPATH and resolved it through `LD_LIBRARY_PATH` — which is what the module
  sets. Every `/soft` gcc from 9.5.0 to 13.3.0 supplied the libquadmath of
  record and yielded `578322f998a329c8`; the system
  `/usr/lib64/libquadmath.so.0` and the ubuntu-24.04 CI runner both yielded
  `54901e8104607a77`, differing in six of the 168 fingerprint fixtures (all
  complex: `casinh` k=1,5, `csqrt` k=5, `casin`/`cacos`/`cacosh` k=7), so a
  correctly BUILT binary still mis-scored if it was RUN without the module: 390
  rows as spurious "increased". That cost two sessions.
  It cannot happen now. The binary links no libquadmath at all, and MPFR/MPC are
  correctly rounded, so the answer does not depend on which build of them is
  found. MEASURED, not inferred: the same binary run under `env -i` with no
  module and no `LD_LIBRARY_PATH` — resolving the SYSTEM `/usr/lib64`
  libstdc++ and `/lib64` libgcc_s — produced `44f18a4a959f6c29` and a sweep
  **byte-identical** in all 436,080 rows to the module-loaded run.
  Still load `gcc/13.3.0` before cmake: this says nothing about building, only
  about running, and the rest of the suite is not `sweep_accuracy`.
  **TWO INDEPENDENT MPFRs HAVE NOW BEEN RUN, AND THEY AGREE.** That used to read
  "a different MPFR *should* agree ... but nobody has run two". Somebody has:
  this login node's SUSE `/usr/lib64` MPFR and the ubuntu-24.04 GitHub runner's
  both produce fingerprint `44f18a4a959f6c29` and a comparison against
  `validation/sweep/sweep_baseline.csv.gz` with **0 bound and 0 digit drift over
  all 436,080 rows** (PR #26's monotone lane, 2026-09-16, gating step and
  non-gating step respectively). Correct rounding leaves nothing to disagree
  about, and now that is measured rather than expected. A fingerprint mismatch
  is therefore not explained by the environment — treat it as a real change in
  the reference, and expect to be able to name which commit moved it.
- The sweep is not bit-reproducible: ~23 rows shift between identical runs (as
  measured, on the 1,652-point real grid = 428,592 rows; `5f2fc90` widened it and
  the sweep is 436,080 rows today). That is what the monotone gate's noise floor
  is for.
- A STALE sweep binary yields structurally impossible results (cross-backend
  deltas from a single-backend change). Rebuild before believing a diff.

**The accuracy record is host-measured, but this code HAS run on GPUs.** Four
campaigns executed it on real hardware: A100/sm_80 under CUDA (S1 — 7 of the
then-23 ctest targets built and all 7 passed, though only 5 are device evidence,
and DD/FF only), MI250X/gfx90a under HIP (S8 bringup, JLSE jobs
1000794/1000851/1000909, 2026-09-13…15), which is where both ROCm defects in
Platform Constraints were found, and then the two step-2 wrapper validations of
2026-09-16: A100 again (S8b, Cobalt `1000938` — 38 passed / 2 failed / 9 not
built of 49, the five-test device gate GREEN) and MI250X again (S8c, Cobalt
`1000943` — 8 passed / 2 failed / **39 not built** of 49, and not one of the 8
compiles a line of gfx90a code). None produced an accuracy table — the
`__float128` oracle cannot share a translation unit with device code (S6) — so
every number in `validation/sweep/` and `docs/DOMAINS.md` is still CPU-measured,
and `docs/DOMAINS.md` remains host-only and wrong below ~1e-31 under FTZ.

`validation/a100/` is TRACKED AGAIN as of S8b and carries job `1000938`'s logs;
the older S1 artifacts under that path were pruned and the S1 record is still
its STATUS block. `validation/mi250/` carries S8c's. **Both campaign scripts
(`validation/a100/run_a100.sh`, `validation/mi250/run_mi250_build.sh`) ran
against `0230239`, which predates the oracle migration**, so their `ldd |
grep quadmath` step is now vacuous — `sweep_accuracy` links no libquadmath to
find. They are left byte-identical to what produced the committed logs. Do not
read a silent quadmath step in those logs as a failure, and do not reuse that
check in anything new: on a Kokkos-linked target `ldd` sees Kokkos's own
libquadmath, so the acceptance check that means something is the `nm -D
--undefined-only` one.

## Documentation

- **README.md** — operation inventory, measured accuracy tables, algorithm references
- **docs/CORRECTNESS.md** — the whole correctness contract: the one measurement,
  the derived bound, the two gates and their exact exit codes (1/2/3/4/5/6). Read
  this first before touching anything that judges correctness.
- **docs/CONSUMING.md** — installing/consuming `xpmath::xpmath` as a CMake package
- **docs/ULP_METRIC.md** — derivation behind the ulp metric and the bound terms
- **docs/ROCM_BRANCH_RELAXATION_BUG.md**, **docs/ROCM_RECURSIVE_DEVICE_STACK.md** —
  the two gfx90a defects, as drop-in upstream issue text plus repo-facing notes
- **docs/UPSTREAM_PLAN.md** — standalone extraction + Kokkos upstream arc (S0–S10), active
- **docs/history/KNOWN_ISSUES.md** — development history: 40 filed defects, 39 resolved.
  Does NOT ship and is NOT the open-defect list. Read before assuming a surprising
  result is new. (`docs/KNOWN_ISSUES.md` is a redirect stub kept only because eight
  `include/xp/*.hpp` headers cite the old path in comments.)
- **validation/sweep/open_defects.txt** — the ONLY list of open defects, enforced
  in both directions by the `sweep_absolute_gate` ctest target.
- **docs/UPSTREAM_PLAN_STATUS.md** — one STATUS block per completed sub-plan; read this before starting one.
  Blocks exist for S0–S3, S5, S7, S8, S8b, S8c, S10, and **STEP-2 CLOSEOUT**, which
  covers the arch wrapper, the two device validations and the oracle migration as
  one arc and lists what they do NOT cover. Read the closeout block first; it is
  the only place the not-covered list is written down. S8 remains **PARTIAL** —
  its own block says so, and S8b/S8c narrowed it without closing it.
- **docs/CORE_PLAN_STATUS.md** — STATUS blocks for the CORE arc (C0/C1/C2), parallel to UPSTREAM_PLAN_STATUS.md
- **docs/TOOLCHAIN_DEFECTS.md** — catalog of toolchain defects (gcc/nvcc/hipcc/libquadmath) with workarounds
- **docs/TEST_SUITE_PLAN.md** — test suite architecture and conventions
- **docs/PERF_PLAN.md** — performance measurement plan (PARKED pending the upstream restructure)

## Platform Constraints

- **`__float128` is an x86_64-ism, and CMake does NOT enforce it.** The claim
  that it did was checked and is false: there is no `CMAKE_SYSTEM_PROCESSOR`
  test anywhere in `CMakeLists.txt`, `tests/CMakeLists.txt`, or any `.cmake`
  file — only comments referring to a gate that was never written. What is
  actually x86-bound is now narrower than it was: `libquadmath` is gone from the
  sweep oracle (MPFR/MPC replaced it) and gone from the demos (they carry no
  oracle), and the remaining user is `__float128` as a CARRIER in
  `scripts/sweep_accuracy.cpp` — arithmetic from libgcc, elementary functions
  from glibc `*f128`, neither of them libquadmath. The library, `include/xp/`,
  has no such dependency and compiles anywhere. The enforcement that does exist
  is in CI: the x86 lanes are where those targets are built.
- **`std::vector<__float128>` will not compile under `nvcc`.** Kokkos exports
  `-arch=sm_XX` in its interface flags, so every consuming TU gets a device pass;
  nvcc then instantiates `std::initializer_list<__float128>` and rejects the
  128-bit float as "not supported in device code". No compiler flag avoids it.
  This blocks the demos and the oracle-scored tests under CUDA builds — the fix
  is splitting host-oracle and device code into separate translation units, and
  is scheduled for S6. See the S1 STATUS block.
- **Two gfx90a (ROCm 7.0.2) backend defects are mitigated in-tree and NOT filed
  upstream.** (1) `BranchRelaxation` scavenges the live return-address pair
  `s[30:31]` in an oversized device callee, so the callee returns into its own
  body — mitigated by `XPMATH_NOINLINE_FUNCTION`. (2) A recursive device call
  sets `uses_dynamic_stack` and provisions nothing, overrunning the 1024 B
  default HIP per-thread stack — mitigated by de-recursing every device
  self-call (`asinh`, `tanh`, `tgamma` now fold rather than recurse). **The two
  interact:** the `noinline` that fixes (1) turns each link of the chain into a
  real frame and made (2) worse, so any change to either must be re-measured
  against both. Guarded by `scripts/xpm_lint_device_asm.sh` — tiers 1-3 for (1),
  tier 4 for (2) — in the non-gating `device-hip` CI lane. Write-ups:
  `docs/ROCM_BRANCH_RELAXATION_BUG.md`, `docs/ROCM_RECURSIVE_DEVICE_STACK.md`.
