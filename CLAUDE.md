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
compile-time-removable `XPMATH_PRINTF`.

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
point is that nothing else issues a competing verdict. 31 ctest targets, all
passing on `main`.

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
| `corpus-generator` | shared validation corpus tool | — (unmerged, see below) |

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

**Known trap:** `scripts/build_with_kokkos.sh` passes `-DCMAKE_CXX_STANDARD=17`
when configuring Kokkos itself (lines 67, 74). Kokkos 5.1.0 rejects that —
`cmake/kokkos_test_cxx_std.cmake:89` raises "Kokkos requires C++20 or newer".
Use 20 for the Kokkos configure. Reported in the S1 STATUS block, not yet fixed.

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

**Shared corpus (branch `corpus-generator`, unmerged).**
`scripts/gen_corpus.cpp` emits one shared set of inputs plus a `__float128`
reference per (op, element), covering 39 real + 24 complex ops, so backends can be
scored on identical data instead of each demo generating its own. The generated
file is gitignored; the generator and `tests/corpus_binary.hpp` (loader with a
staleness guard) are committed. Nothing consumes it yet.

## Working on accuracy

Read **docs/CORRECTNESS.md** first. The short version:

- One measurement: error in ulps against the `__float128` oracle.
- One verdict per point: at or below its derived bound, or above it.
- The gate allows `kUlpAllowance = 8.0` × the derived bound
  (`scripts/sweep_accuracy.cpp:876`). A point can therefore exceed its raw
  derived bound and still pass; report both numbers if you claim "at the limit".
- ~8.5% of sweep rows carry state `U`/`N` and get **no verdict** — the format
  cannot carry the question. Absence of a defect there is not evidence of
  correctness.

**Traps that have cost real time:**

- `sweep_accuracy` with no mode flag OVERWRITES the committed baseline
  (`kDefaultOut`, `scripts/sweep_accuracy.cpp:249`). Always pass `--out /tmp/...`.
  `--grid-out` behaves the same way.
- `write_baseline` uses plain `fopen` — it does NOT gzip. Writing `--out foo.gz`
  produces uncompressed bytes under a `.gz` name. Write plain, then `gzip -9 -c`.
- `ctest --test-dir` on a MISSING directory exits 0. Confirm the dir exists and
  that a nonzero test count ran, or the run proved nothing.
- Build without the gcc module and a different libquadmath links: the oracle
  fingerprint moves off `578322f998a329c8` and ~328 rows read as spurious
  "increased". Load `gcc/13.3.0` in EVERY shell, before cmake.
- The sweep is not bit-reproducible: ~23 of 428,592 rows shift between identical
  runs. That is what the monotone gate's noise floor is for.
- A STALE sweep binary yields structurally impossible results (cross-backend
  deltas from a single-backend change). Rebuild before believing a diff.

**Nothing has ever executed on a GPU.** CI compiles for nvcc and hip; every
accuracy number in the repo is CPU-measured, and `docs/DOMAINS.md` is host-only
and wrong below ~1e-31 under FTZ.

## Documentation

- **README.md** — operation inventory, measured accuracy tables, algorithm references
- **docs/UPSTREAM_PLAN.md** — standalone extraction + Kokkos upstream arc (S0–S10), active
- **docs/history/KNOWN_ISSUES.md** — development history: 40 filed defects, 39 resolved.
  Does NOT ship and is NOT the open-defect list. Read before assuming a surprising
  result is new. (`docs/KNOWN_ISSUES.md` is a redirect stub kept only because eight
  `include/xp/*.hpp` headers cite the old path in comments.)
- **validation/sweep/open_defects.txt** — the ONLY list of open defects, enforced
  in both directions by the `sweep_absolute_gate` ctest target.
- **docs/UPSTREAM_PLAN_STATUS.md** — one STATUS block per completed sub-plan; read this before starting one
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
