# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Three portable extended-precision backends as header-only C++ in `third_party/include/`:

- **DD (DoubleDouble, 2×FP64)**: `Kokkos::Experimental::DoubleDouble`, ~31 decimal digits, `dd_math.hpp` + `dd_complex.hpp`
- **FF (FloatFloat, 2×FP32)**: `Kokkos::Experimental::FloatFloat`, ~14 decimal digits, `ff_math.hpp` + `ff_complex.hpp`
- **QF (QuadFloat, 4×FP32)**: `Kokkos::Experimental::QuadFloat`, ~29 decimal digits, `qf_math.hpp` + `qf_complex.hpp`

All types/math live in `namespace Kokkos::Experimental`, every function is `KOKKOS_INLINE_FUNCTION`, portable to any Kokkos execution space (CPU, GPU, everything Kokkos targets). Validation is a 23-test ctest suite in `tests/` plus seven demo executables in `src/`, measured against a `__float128` (libquadmath) host oracle. All 23 tests pass on `main`.

## Executables

Seven demo/benchmark targets defined in CMakeLists.txt:

- `kokkos_ep_demo` — DD real operations (39 ops)
- `kokkos_ep_demo_complex` — DD complex operations (24 ops)
- `kokkos_ep_demo_ff` — FF real operations (39 ops)
- `kokkos_ep_demo_ff_complex` — FF complex operations (24 ops)
- `kokkos_ep_demo_qf` — QF real operations (39 ops)
- `kokkos_ep_demo_qf_complex` — QF complex operations (24 ops)
- `kokkos_ep_bench_cost` — Cost benchmark across all backends

## Branch Structure

| Branch | Backends | GPU requirement |
|---|---|---|
| `main` | DD + FF + QF (portable) | any Kokkos-compatible hardware |
| `CUDAFP128Kokkos` | CUDA FP128 only (sm_100) | compute ≥ 10.0 (Blackwell) |

The `CUDAFP128Kokkos` branch cannot merge into `main` — it requires sm_100 hardware and is kept as a separate reference implementation.

## Build

Requires Kokkos ≥5.1 with `Kokkos_ENABLE_LIBQUADMATH=ON`, GCC 13.3.0, CMake 3.28.3. On Argonne systems:

```bash
source scripts/prepare.sh
source scripts/build_with_kokkos.sh <install-dir>
```

This outputs executables in `build/` and generates `setup.sh` with environment variables.

If Kokkos is already installed:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
ctest --test-dir build
```

## Running

```bash
# All real operations for DD
./build/kokkos_ep_demo --batch 1000000 --repeats 5

# Single operation
./build/kokkos_ep_demo --op sin --batch 1000000 --repeats 5

# Complex operations for FF
./build/kokkos_ep_demo_ff_complex --batch 1000000 --repeats 5
```

Arguments: `--op <name>`, `--batch N` (default: 1,000,000), `--repeats N` (default: 5), `--seed N` (default: 12345).

## Documentation

- **README.md** — Operation inventory, measured accuracy tables, algorithm references
- **docs/TEST_SUITE_PLAN.md** — Test suite architecture and conventions
- **docs/PERF_PLAN.md** — Performance measurement plan (PARKED pending upstream restructure)
- **docs/UPSTREAM_PLAN.md** — Standalone library extraction + Kokkos upstream contribution plan (active)

## Platform Constraint

`libquadmath` (host oracle) is x86_64 only. CMake enforces this — the project will not build on ARM or other platforms.
