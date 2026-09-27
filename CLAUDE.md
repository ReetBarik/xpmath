# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**xpmath** is a header-only C++17 extended-precision library. The artifact is
`include/xp/` — zero Kokkos, standard library only, compiles under plain
`g++`/`clang++`, `nvcc`, and `hipcc`.

| backend | type | words | digits | headers |
|---|---|---|---|---|
| DD | `xp::DoubleDouble` | 2×FP64 | ~31 | `dd_math.hpp`, `dd_complex.hpp` |
| FF | `xp::FloatFloat` | 2×FP32 | ~14 | `ff_math.hpp`, `ff_complex.hpp` |
| QF | `xp::QuadFloat` | 4×FP32 | ~29 | `qf_math.hpp`, `qf_complex.hpp` |
| TF | `xp::TripleFloat` | 3×FP32 | ~21.7 | `tf_math.hpp`, `tf_complex.hpp` |

`include/xp/config.hpp` supplies `XPMATH_INLINE_FUNCTION`
(`__host__ __device__ inline` under CUDA/HIP), `XPMATH_ON_DEVICE` (unifying
`__CUDA_ARCH__` / `__HIP_DEVICE_COMPILE__` / `__SYCL_DEVICE_ONLY__`),
`xp::detail::` scalar-math dispatch, compile-time-removable `XPMATH_PRINTF`,
and `XPMATH_NOINLINE_FUNCTION` (gfx90a codegen mitigation — see Platform
Constraints).

**This repository finds and links no Kokkos.** The `Kokkos::Experimental`
wrappers and the eight timing demos last existed at commit `158d618` and move
to **xpmath-kokkos**. `XPMATH_WITH_KOKKOS` is gone. Device tests launch through
`tests/device_harness.hpp`.

Validation: **one** measurement — error in ulps against an **MPFR/MPC oracle at
400 bits, carried in `__float128`** — with **one** verdict per point against a
bound derived from the format and the condition number, and **two** ctest gates
(`sweep_absolute_gate`, `sweep_monotone_gate`), each with a self-test that
poisons its input and must fail. Read **docs/CORRECTNESS.md** before adding
anything that judges correctness. **67 ctest targets** in a single-tree
configure — asserted by CI as a COUNT. Host vs device halves are selected by
`XPMATH_BUILD_HOST_TARGETS` / `XPMATH_BUILD_DEVICE_TARGETS` (both default ON);
`scripts/xpm_build.sh` drives two trees when the arch needs a device compiler.

**The sweep oracle is MPFR/MPC, not libquadmath.** `sweep_accuracy` links
`-lmpc -lmpfr -lgmp` and **no `-lquadmath`** (asserted in the binary's header
checks). `__float128` remains the carrier; MPFR/MPC are a hard
`FATAL_ERROR` requirement of `tests/CMakeLists.txt`. `--oracle=quadmath` is
refused.

## Examples (not demos)

There are **no** `kokkos_ep_demo*` targets in this tree.

- `examples/standalone/compensated_reduction.cpp` — host usage via
  `find_package(xpmath)` only (`examples/standalone/CMakeLists.txt`).
- `compensated_reduction_{cuda,hip}.cpp` — same fold through
  `tests/device_harness.hpp`. Header comments state they are usage examples,
  not benchmarks.
- `consumer_package` builds and runs the host example against an install
  prefix so the example cannot rot.

## Branch Structure

| Branch | Contents | Requirement |
|---|---|---|
| `main` | DD + FF + QF + TF (portable C++ library) | any host that builds the suite |
| `CUDAFP128Kokkos` | CUDA FP128 only (reference) | compute ≥ 10.0 (sm_100, Blackwell) |

`CUDAFP128Kokkos` cannot merge into `main`. Tag `kokkos-native-freeze` marks the
last fully Kokkos-native state before the standalone extraction.

## Build

GCC 13.3.0, CMake 3.28.3 (JLSE / CI). The library and consumers stay at **C++17**.
No Kokkos install is required.

```bash
module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3
export LD_LIBRARY_PATH=/soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64:$LD_LIBRARY_PATH

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build -j$(nproc)
```

**Two-tree wrapper** (one `CXX` per CMake project — host tools must not be
handed to `nvcc`/`hipcc`):

```bash
scripts/xpm_build.sh --arch host  --build-dir /tmp/b_host
scripts/xpm_build.sh --arch a100  --build-dir /tmp/b_a100
scripts/xpm_build.sh --arch mi250 --build-dir /tmp/b_mi250
```

`--arch` selects modules, compiler, and FP-contraction spelling. After CORE
C10, `--kokkos` / `--no-kokkos` are removed: this repository never finds Kokkos.
`--opt` defaults to `O3`. Read the script header before comparing numbers across
arches.

Every configure stamps `build-info.txt` (arch, git HEAD with `-dirty` when
needed, compiler, flags, `-O`, UTC timestamp). `build_provenance` fails if it
is missing or short a field.

`scripts/build_with_kokkos.sh` still builds a Kokkos *install* for other
workflows (`KOKKOS_ONLY=1` is what the wrapper used when this repo still linked
Kokkos). It is not required to build or test xpmath itself.

`scripts/check_standalone_no_kokkos.sh` compiles each `include/xp/` header with
plain `g++ -std=c++17` against **only** `-Iinclude`, then greps for `Kokkos`.

## Validation conventions

Accuracy lives only in `validation/sweep/` and the two ctest gates
(`docs/CORRECTNESS.md`). There are no demo accuracy columns and no byte-identical
demo gate. `validation/strip_timing.sh` remains for reading *historical*
captures; on modern output every row prints verbatim (timing jitter only) — do
not use it to bless a change.

**Shared corpus — `scripts/attic/gen_corpus.cpp`, unused.** Do not confuse it
with live **`tests/corpus.hpp`** (T0.2 corner cases for `corpus_test` and
invariant tests).

Host domain limits: `docs/DOMAINS.md` (generated; `domains_fresh`). Device
comparison: `docs/DEVICE_PRECISION.md` (generated; `device_domains_fresh`) from
`validation/sweep/sweep_baseline_{a100,mi250}.csv.gz`.

## Working on accuracy

Read **docs/CORRECTNESS.md** first.

- One measurement: error in ulps against MPFR/MPC at 400 bits, quantised into
  `__float128`.
- One verdict per point: at or below its derived bound, or above it.
- The gate allows `kUlpAllowance = 8.0` × the derived bound. A point can exceed
  its raw bound and still pass; report both if you claim "at the limit".
- ~8.9% of sweep rows carry state `U`/`N` and get **no verdict** — the format
  cannot carry the question.

**Traps that have cost real time:**

- `sweep_accuracy` with no `--out` exits 1 (`cannot open for writing`); it does
  not overwrite the committed baseline. Always pass `--out /tmp/...`.
- `write_baseline` uses plain `fopen` — it does **not** gzip. Write plain, then
  `gzip -9 -c`.
- `ctest --test-dir` on a **missing** directory exits 0. Confirm the dir exists
  and a nonzero test count ran.
- The old libquadmath run-time fingerprint trap is **retired** (binary links no
  libquadmath; two independent MPFRs agree on fingerprint `44f18a4a959f6c29`).
  Still load `gcc/13.3.0` before cmake for *building*.
- The sweep is not bit-reproducible (~tens of rows shift); that is what the
  monotone gate's noise floor is for.
- A STALE sweep binary yields structurally impossible diffs. Rebuild before
  believing a result.

## Documentation

- **README.md** — motivation, op inventory, layout, consuming overview
- **docs/CORRECTNESS.md** — correctness contract (read first for accuracy work)
- **docs/DOMAINS.md** — host domain limits (generated)
- **docs/DEVICE_PRECISION.md** — host vs A100 vs MI250X (generated)
- **docs/CONSUMING.md** — installing/consuming `xpmath::xpmath`
- **docs/ULP_METRIC.md** — ulp metric and bound terms
- **docs/ROCM_BRANCH_RELAXATION_BUG.md**, **docs/ROCM_RECURSIVE_DEVICE_STACK.md**
- **docs/UPSTREAM_PLAN.md** / **docs/UPSTREAM_PLAN_STATUS.md** — upstream arc
- **docs/CORE_PLAN_STATUS.md** — CORE arc STATUS blocks (C0–C10)
- **docs/history/KNOWN_ISSUES.md** — development history (not the open-defect list)
- **validation/sweep/open_defects.txt** — the ONLY open-defect list
- **docs/TOOLCHAIN_DEFECTS.md**, **docs/TEST_SUITE_PLAN.md**
- **docs/PERF_PLAN.md** — PARKED (no cost benchmark in this tree)

## Platform Constraints

- **`__float128` is an x86_64-ism, and CMake does NOT enforce it.** The library
  has no such dependency. The remaining user is `__float128` as a carrier in
  `scripts/sweep_accuracy.cpp` (libgcc arithmetic, glibc `*f128` — not
  libquadmath). CI's x86 lanes are where those targets build.
- **Host-oracle and device code must not share a TU.** `std::vector<__float128>`
  fails under `nvcc`'s device pass. Host halves include
  `tests/test_utils_host.hpp`; device halves include
  `tests/test_utils_device.hpp` and launch via `tests/device_harness.hpp`.
  Enforced by `scripts/check_device_tu_purity.sh`.
- **Two gfx90a (ROCm 7.0.2) defects are mitigated in-tree and NOT filed
  upstream.** (1) `BranchRelaxation` scavenges `s[30:31]` in an oversized device
  callee — mitigated by `XPMATH_NOINLINE_FUNCTION`. (2) Recursive device calls
  overrun the 1024 B HIP stack — mitigated by de-recursing (`asinh`, `tanh`,
  …). The two interact. Guarded by `scripts/xpm_lint_device_asm.sh` in the
  `device-hip` CI lane.
