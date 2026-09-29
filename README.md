# xpmath — Extended-precision arithmetic library

Header-only C++17: four software-emulated extended-precision backends that
compile under plain `g++`/`clang++`, `nvcc`, and `hipcc`. This repository is
the **C++ library**. A separate **xpmath-kokkos** repository will carry the
`Kokkos::Experimental` wrappers and timing demos; this tree finds and links no
Kokkos.

## Using xpmath in your own project

```cmake
find_package(xpmath 0.2 REQUIRED)
target_link_libraries(my_app PRIVATE xpmath::xpmath)
```

```cpp
#include <xp/dd_math.hpp>
xp::DoubleDouble y = xp::sqrt(xp::DoubleDouble(2.0));
```

See [docs/CONSUMING.md](docs/CONSUMING.md) for install instructions, the
versioning policy, and how to verify an install. A runnable host example lives
in [`examples/standalone/`](examples/standalone/) (compensated reduction via
`find_package(xpmath)` only); CUDA and HIP variants of the same fold illustrate
`tests/device_harness.hpp`.

[![CI](https://github.com/ReetBarik/xpmath/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ReetBarik/xpmath/actions/workflows/ci.yml)

CI covers generated-doc freshness, the standalone no-Kokkos core on Linux and
macOS/ARM, a full build + 65-target ctest suite, the 436,080-point monotone
accuracy gate, and per-header `nvcc` / `hipcc` device compiles (plus the gfx90a
asm guard on the HIP lane). See `.github/workflows/ci.yml`.

## Section 1 — Motivation

Many scientific and engineering applications — numerical linear algebra,
particle physics, long-running N-body and climate integrations, ill-conditioned
solvers — need arithmetic precision beyond what 64-bit IEEE double (FP64,
~16 decimal digits) can provide. On host CPUs that need has historically been
served by GCC's libquadmath (`__float128`). That path does not travel:
libquadmath is host-only and x86_64-only, so code needing extra precision
*inside* a portable compute kernel has had nowhere to go.

This repository provides four portable, software-emulated backends — **DD**
(double-double), **FF** (float-float), **QF** (quad-float), and **TF**
(triple-float) — as plain C++ headers under `include/xp/`. They carry no Kokkos
dependency and compile for host, CUDA, HIP, and SYCL device passes. Accuracy is
measured against an MPFR/MPC oracle at 400 bits: one measurement (error in
ulps), one verdict per point (`docs/CORRECTNESS.md`). The host domain table is
[`docs/DOMAINS.md`](docs/DOMAINS.md); host vs A100 vs MI250X is
[`docs/DEVICE_PRECISION.md`](docs/DEVICE_PRECISION.md).

## Section 2 — Backends: types, ops, measured accuracy

### Precision types

| Backend | C++ type | Precision (approx. decimal digits) | Headers |
|---|---|---|---|
| DD (double-double, 2×FP64) | `xp::DoubleDouble` | ~31 | `include/xp/dd_math.hpp`, `dd_complex.hpp` |
| FF (float-float, 2×FP32) | `xp::FloatFloat` | ~14 | `include/xp/ff_math.hpp`, `ff_complex.hpp` |
| QF (quad-float, 4×FP32) | `xp::QuadFloat` | ~29 | `include/xp/qf_math.hpp`, `qf_complex.hpp` |
| TF (triple-float, 3×FP32) | `xp::TripleFloat` | ~21.7 | `include/xp/tf_math.hpp`, `tf_complex.hpp` |

Every operation is documented in the header where it is defined. Complex layers
(`DoubleDoubleComplex`, `FloatFloatComplex`, `QuadFloatComplex`,
`TripleFloatComplex`) mirror the real-side surface.

### Operation inventory

All four backends expose the same 39 real operations:

| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary math | `sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh acosh asinh atanh` |
| 2-input | `pow hypot fmod remainder copysign fmax fmin fdim` |
| 3-input | `fma` |
| Rounding | `ceil floor round trunc` |

**Tie convention.** `round` breaks halfway cases **to even** on every backend —
IEEE 754 `roundToIntegralTiesToEven`, so `round(0.5) == 0`, `round(1.5) == 2`,
`round(2.5) == 2`. This is a deliberate divergence from C99 `round` /
libquadmath `roundq` and from QD 2.3.24's `nint`; see KI-20 in
`docs/history/KNOWN_ISSUES.md`. `remainder` is half-even too, as IEEE 754
requires of it.

All four complex layers expose the same 24 complex operations:

| Category | Operations |
|---|---|
| Arithmetic | `add sub mul div` |
| Unary | `abs conj sqrt exp log log10` |
| Trig | `sin cos tan asin acos atan` |
| Hyperbolic | `sinh cosh tanh asinh acosh atanh` |
| Power / construction | `pow polar` |

### Measured accuracy

**Do not look here for digit tables.** Older README tables were a pre-fix
demo capture scored against libquadmath; the demos are gone, those columns
cannot be regenerated, and they are not the contract.

The live record is:

| Document | What it answers |
|---|---|
| [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md) | One measurement, one verdict, the two ctest gates and their exit codes |
| [`docs/DOMAINS.md`](docs/DOMAINS.md) | Host: where each op holds ~90% of its digit cap (generated; `domains_fresh`) |
| [`docs/DEVICE_PRECISION.md`](docs/DEVICE_PRECISION.md) | Host vs A100 vs MI250X ulps (generated; `device_domains_fresh`) |
| `validation/sweep/sweep_baseline{,_a100,_mi250}.csv.gz` | The committed 436,080-row baselines the docs are generated from |

Absolute gate (`ulps ≤ 8 × derived bound`) on every present arch: host, a100,
and mi250 all report **0** points above bound. Open defects live only in
`validation/sweep/open_defects.txt`.

## Section 3 — Repository layout

```
include/xp/            header-only library (DD/FF/QF/TF real + complex)
examples/standalone/   compensated reduction (host + CUDA/HIP usage examples)
tests/                 65-target ctest suite (host halves, device halves, gates)
validation/sweep/      committed baselines, open-defect register, grids
docs/                  CORRECTNESS, DOMAINS, DEVICE_PRECISION, CONSUMING, …
scripts/               xpm_build.sh, sweep_accuracy, domain generators, checks
patches/               historical note only (no active Kokkos patches)
```

## Section 4 — Tests

**65 registered ctest targets**, asserted as a COUNT in CI (not assumed). MPFR,
MPC, and GMP are a hard configure requirement — a missing `-dev` package is
`FATAL_ERROR`, not four silently unregistered targets.

- **Accuracy gates** — `sweep_absolute_gate`, `sweep_monotone_gate`, and their
  `*_selftest` poisons (`docs/CORRECTNESS.md`).
- **Device baselines** — `sweep_device_gate_{a100,mi250}` plus selftest; domain
  freshness for host (`domains_fresh`) and device (`device_domains_fresh`).
- **Property / invariant / EFT / FMA-contraction / cancellation** — per-backend
  host and device halves (device launches go through
  `tests/device_harness.hpp`, not Kokkos).
- **Packaging** — `consumer_package` installs the tree and builds both a minimal
  consumer and `examples/standalone/` against that prefix.
- **Hygiene** — `device_tu_purity`, `build_provenance`, contraction-flag guards.

On a GPU node use the two-tree wrapper (CMake has one `CXX` per project):

```bash
scripts/xpm_build.sh --arch {host|a100|mi250} --build-dir /tmp/b
# /tmp/b/host    g++ — host-tagged tests
# /tmp/b/device  serial / nvcc / hipcc — device-tagged tests
```

`--arch host` still builds the device tree, with the harness's serial fallback,
so device-tagged tests run without a GPU. Design notes:
`docs/TEST_SUITE_PLAN.md`.

## Section 5 — Usage

### Build and install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/where/you/want/it
cmake --build build -j$(nproc)
cmake --install build
ctest --test-dir build -j$(nproc)
```

GCC 13.3.0 and CMake 3.28.3 are what CI and the JLSE wrapper use. The library
itself is C++17. On JLSE:

```bash
scripts/xpm_build.sh --arch host  --build-dir /tmp/b_host
scripts/xpm_build.sh --arch a100  --build-dir /tmp/b_a100
scripts/xpm_build.sh --arch mi250 --build-dir /tmp/b_mi250
```

Read that script's header before comparing numbers across arches: it states what
is held identical (`-O` level, C++17) and what is not (compiler, device
backend). Host and AMD builds pass `-ffp-contract=off`. NVIDIA device builds do
not pass `--fmad=false`. `eft_add`, `eft_sub`, and `eft_mul` stay rounded in
the helpers: `volatile` on the host and on AMD, `add.rn` / `sub.rn` / `mul.rn`
on CUDA. Every configure stamps `build-info.txt` into the build directory;
`build_provenance` fails if it is missing or short a field.

`scripts/check_standalone_no_kokkos.sh` compiles each `include/xp/` header with
plain `g++ -std=c++17` against an include path containing **only** `include/`,
then greps the preprocessed output for the token `Kokkos`.

There is no CMake x86_64 gate. What is x86-bound is `__float128` as a *carrier*
in `scripts/sweep_accuracy.cpp` (arithmetic from libgcc, `*f128` from glibc —
not libquadmath). The installable library has no such dependency.

### Host example

```cpp
#include <xp/dd_math.hpp>
#include <vector>

xp::DoubleDouble compensated_sum(const std::vector<double>& x) {
  xp::DoubleDouble s(0.0);
  for (double v : x) s = s + xp::DoubleDouble(v);
  return s;
}
```

See `examples/standalone/compensated_reduction.cpp` for a full program that
builds only against the installed package. Device variants of the same fold:
`compensated_reduction_{cuda,hip}.cpp`.

## Section 6 — Licensing

This repository is dual-licensed. Repository-default is Apache-2.0 (see
`LICENSE`). The DDFUN-derived headers carry `LicenseRef-DHB-License`; the
QD-derived QF headers carry `LicenseRef-LBNL-BSD-License`. Full mapping, license
texts, and the plain-English explanation of the DHB-License §3 grant-back clause
live in `NOTICE.md` and `LICENSES/`.

| File | License |
|---|---|
| `include/xp/dd_math.hpp`, `dd_complex.hpp` | `LicenseRef-DHB-License` |
| `include/xp/ff_math.hpp`, `ff_complex.hpp` | `LicenseRef-DHB-License` |
| `include/xp/qf_math.hpp`, `qf_complex.hpp` | `LicenseRef-LBNL-BSD-License` |
| `include/xp/tf_math.hpp`, `tf_complex.hpp` | `LicenseRef-LBNL-BSD-License` |
| Everything else | `Apache-2.0` |

## Section 7 — References

- **DDFUN v04** — David H. Bailey.
  <https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz>
- **QD 2.3.24** — Yozo Hida, Xiaoye S. Li, David H. Bailey (LBNL).
  <https://www.davidhbailey.com/dhbsoftware/qd-2.3.24.tar.gz>
- **Kokkos** — <https://github.com/kokkos/kokkos> (wrappers and demos will live
  in **xpmath-kokkos**, not here)

Repository owner: Reet Barik. DDFUN questions: David H. Bailey
(<dhbailey@lbl.gov>).
