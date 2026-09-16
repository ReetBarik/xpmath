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

Validation: **one** measurement — error in ulps against a `__float128` /
`__complex128` host oracle — with **one** verdict per point against a bound
derived from the format and the condition number, and **two** ctest gates over
that number (`sweep_absolute_gate`, `sweep_monotone_gate`), each with a self-test
target (`*_selftest`) that poisons its input and requires it to fail. Read
**docs/CORRECTNESS.md** before adding anything that judges correctness; the whole
point is that nothing else issues a competing verdict. 48 ctest targets with
Kokkos and **20 without** (`-DXPMATH_WITH_KOKKOS=OFF`), all passing on `main` —
asserted by CI, not assumed, because four sit behind `if(XPMATH_MPFR_FOUND)`
and a missing optional dependency removes coverage while leaving the lane
green. The Kokkos-free 20 include both gates: `sweep_accuracy` links no Kokkos
and never needed it, but until `XPMATH_WITH_KOKKOS` existed the build required
it anyway.

## Executables

Nine targets in CMakeLists.txt (the four complex demos are inside a
`Kokkos_ENABLE_LIBQUADMATH` conditional):

- `kokkos_ep_demo` — DD real (39 ops)
- `kokkos_ep_demo_complex` — DD complex (24 ops)
- `kokkos_ep_demo_ff` — FF real (39 ops)
- `kokkos_ep_demo_ff_complex` — FF complex (24 ops)
- `kokkos_ep_demo_qf` — QF real (39 ops)
- `kokkos_ep_demo_qf_complex` — QF complex (24 ops)
- `kokkos_ep_demo_tf` — TF real (39 ops)
- `kokkos_ep_demo_tf_complex` — TF complex (24 ops)
- `kokkos_ep_bench_cost` — cost benchmark across backends

**Never run the demos casually** — they are hours of kernel time. The accuracy
record is `validation/sweep/`, not the demos.

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

Requires Kokkos ≥5.1 built at **C++20** with `Kokkos_ENABLE_LIBQUADMATH=ON`, GCC
13.3.0, CMake 3.28.3. The consuming project stays at C++17.

```bash
source scripts/prepare.sh
source scripts/build_with_kokkos.sh <install-dir>
```

**Known trap:** `scripts/build_with_kokkos.sh` is hardcoded to the **wrong
GPU** for this branch — `Kokkos_ARCH_BLACKWELL100` (line 45) plus
`-DCMAKE_CUDA_ARCHITECTURES=100` (line 77), i.e. sm_100, which is
`CUDAFP128Kokkos`'s requirement and nothing on `main`'s target list. It also
defines `HIP_EXTRA_FLAGS` (line 51) and never uses it: line 53 hardcodes
`EXTRA_FLAGS=$CUDA_EXTRA_FLAGS`, so the HIP path in that script has never been
executed. Edit both arch lines before using it for A100 (`AMPERE80` / `80`).

(The older C++17-vs-20 trap this paragraph used to describe is FIXED. `dd6d00a`
changed both Kokkos configure lines to `-DCMAKE_CXX_STANDARD=20` on 2026-09-02;
lines 67 and 74 read 20 today. The consuming repo still builds at C++17.)

If Kokkos is already installed:

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

**Byte-identical gate.** Any mechanical restructure must leave the demo accuracy
columns unchanged. Run the affected demos before and after with identical
arguments, strip timing with **`validation/strip_timing.sh`**, and diff. Timing
columns are exempt; accuracy columns are not.

Use `validation/strip_timing.sh`. (An older `validation/s3/strip_timing.sh`
hard-coded the DD table shape and silently stripped nothing from FF/QF layouts,
producing diffs full of wall-clock jitter; it and the rest of the per-sub-plan
capture directories were pruned once `validation/sweep/` became the record.)

**Shared corpus — `scripts/gen_corpus.cpp`, on `main`, unused.** It emits one
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

- One measurement: error in ulps against the `__float128` oracle.
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
- Build without the gcc module and a different libquadmath links: the oracle
  fingerprint moves off `578322f998a329c8` and ~328 rows read as spurious
  "increased". Load `gcc/13.3.0` in EVERY shell, before cmake.
- The sweep is not bit-reproducible: ~23 rows shift between identical runs (as
  measured, on the 1,652-point real grid = 428,592 rows; `5f2fc90` widened it and
  the sweep is 436,080 rows today). That is what the monotone gate's noise floor
  is for.
- A STALE sweep binary yields structurally impossible results (cross-backend
  deltas from a single-backend change). Rebuild before believing a diff.

**The accuracy record is host-measured, but this code HAS run on GPUs.** Two
campaigns executed it on real hardware: A100/sm_80 under CUDA (S1 — 7 of the
then-23 ctest targets built and all 7 passed, though only 5 are device evidence,
and DD/FF only), and MI250X/gfx90a under HIP (S8 bringup, JLSE jobs
1000794/1000851/1000909, 2026-09-13…15), which is where both ROCm defects in
Platform Constraints were found. Neither produced an accuracy table — the
`__float128` oracle cannot share a translation unit with device code — so every
number in `validation/sweep/` and `docs/DOMAINS.md` is still CPU-measured, and
`docs/DOMAINS.md` remains host-only and wrong below ~1e-31 under FTZ. The S1
artifacts under `validation/a100/` were pruned; the record is the S1 STATUS block.

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
  Blocks exist for S0–S3, S5, S7, S10. **S8 (cross-vendor device matrix) is the
  active sub-plan and has no block yet** — the gfx90a work above is S8 in flight.
- **docs/TEST_SUITE_PLAN.md** — test suite architecture and conventions
- **docs/PERF_PLAN.md** — performance measurement plan (PARKED pending the upstream restructure)

## Platform Constraints

- **`libquadmath` (host oracle) is x86_64 only**, and CMake enforces it. This
  constrains the *tests and demos*, not the library: `include/xp/` has no
  quadmath dependency and compiles anywhere.
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
