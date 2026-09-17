# Consuming xpmath

xpmath is a header-only C++17 library: four extended-precision types built from
IEEE-754 scalars, with real and complex math on each.

| type | layout | precision |
|---|---|---|
| `xp::DoubleDouble` | 2 x FP64 | p = 106 |
| `xp::FloatFloat` | 2 x FP32 | p = 48 |
| `xp::TripleFloat` | 3 x FP32 | p = 72 |
| `xp::QuadFloat` | 4 x FP32 | p = 96 |

## Install

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/where/you/want/it
cmake --build build -j
cmake --install build
```

One directory, one compiler, everything — and that is the supported way to
build and install this project on a host. It is what CI runs. If you are
consuming xpmath, this is the whole of what you need; the two-tree section
below is about testing it on a GPU and does not change what gets installed.

### Building with Kokkos

Kokkos ≥5.1 built at C++20, and nothing else. In particular **no
`Kokkos_ENABLE_LIBQUADMATH=ON`**: that used to be a hard requirement because the
demos scored themselves against a host `__float128` oracle reached through
`impl/Kokkos_QuadPrecisionMath.hpp` (and, for the complex demos, a non-upstream
patch header this repo carried). Those accuracy columns are gone — the accuracy
record is `validation/sweep/` — so a Kokkos with libquadmath OFF, such as a
`hip/gfx90a` install, configures cleanly. No target here uses libquadmath.

### Building without Kokkos

The build itself no longer requires Kokkos:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXPMATH_WITH_KOKKOS=OFF
cmake --build build -j
ctest --test-dir build
```

That gets you the installable package, **both accuracy gates**, and every test
— no test in this repository links Kokkos any more. What it skips is the eight
demos, which are the only remaining Kokkos consumers and the only thing that
exercises the compat wrappers in `third_party/include/`.

This used to be impossible. `find_package(Kokkos REQUIRED)` sat at the top of
the top-level `CMakeLists.txt` and gated everything below it — including the
header-only export, its install rules, and the whole test suite — so the
Kokkos-free core could not be built, installed or tested without Kokkos, and
`sweep_accuracy`, which links no Kokkos at all, was unreachable without it.

### Building the device side: two trees, one command

CMake supports exactly one `CXX` compiler per project and offers no per-target
override, so compiling the host-side tools with `g++` and the device-side tests
with `nvcc`/`hipcc` means **two build trees**. Two CMake options select the
halves, both defaulting `ON`:

| option | default | selects |
|---|---|---|
| `XPMATH_BUILD_HOST_TARGETS` | `ON` | the oracle-scored tools, the accuracy gates, the host test halves |
| `XPMATH_BUILD_DEVICE_TARGETS` | `ON` | the `*_test_device` halves, the device-harness self-test, the demos |

**With both left alone you get the single-tree build above, unchanged.** That
is the point of the defaults: a consumer, an installer and CI see no
difference.

The wrapper drives both halves for a named GPU and returns one merged verdict:

```sh
scripts/xpm_build.sh --arch {host|a100|mi250} --build-dir <dir>
# <dir>/host    g++, no Kokkos, identical on every arch — 38 tests
# <dir>/device  the arch's compiler and Kokkos       — 24 tests
scripts/xpm_build.sh --arch a100 --build-dir <dir> --only device   # on a compute node
```

`--arch host` builds the device tree too, with the harness's serial backend, so
the device-side tests run without a GPU. Building the host tools with a device
compiler is the failure this exists to prevent: on A100 job `1000938`
`sweep_accuracy` was handed to `nvcc_wrapper`, failed on
`std::vector<__float128>`, and took both accuracy gates down with it.

## Use it from another project

```cmake
find_package(xpmath 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE xpmath::xpmath)
```

```cpp
#include <xp/dd_math.hpp>

xp::DoubleDouble x(1.0);
xp::DoubleDouble y = xp::sqrt(x / xp::DoubleDouble(3.0));
double approx = y.hi + y.lo;
```

If xpmath is not in a default prefix, point CMake at it:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/where/you/installed/it
```

## No Kokkos required

The exported package does **not** depend on Kokkos. The headers in `include/xp/`
never include a Kokkos header — every mention of Kokkos in them is a comment —
so a consumer gets the numeric core with no third-party libraries at all, not
even libquadmath.

Kokkos is used by this repository's own eight demos and by nothing else — its
device tests run on a Kokkos-free CUDA/HIP harness (`tests/device_harness.hpp`).
The demos reach Kokkos through a compat wrapper in
`third_party/include/` that is deliberately **not** installed. If you want
xpmath inside Kokkos kernels, bring your own Kokkos and use it alongside — the
types are annotated `XPMATH_INLINE_FUNCTION` and run on device.

## Versioning

`0.1.0`, exported with `COMPATIBILITY SameMajorVersion`. Because the major
version is 0, CMake treats **every 0.x as incompatible with every other 0.x** —
`find_package(xpmath 0.1)` succeeds and `find_package(xpmath 0.2)` fails. That
is deliberate: the API has never been consumed by anyone outside this repo, and
promising compatibility it has not earned would be worse than making the break
explicit.

## Accuracy

Every op is scored per grid point in ULPs against a binary128 (and optionally
MPFR 400-bit) oracle, over 436,080 rows, with a bound derived from the format
and the condition number rather than from the implementation. See
`docs/CORRECTNESS.md` for the contract and `docs/DOMAINS.md` for measured
per-op behaviour. Known-bad points are carried in
`validation/sweep/open_defects.txt`, which is currently empty.

**xpmath is pre-1.0 and its accuracy is uneven by op.** Some ops (complex `pow`,
real `pow`, real `asin`) have materially higher error rates than others; the
open issues track them. Check `docs/DOMAINS.md` before relying on a specific
op in a specific range.

## Verifying an install

```sh
tests/consumer/run_consumer_test.sh /where/you/installed/it
```

Builds a separate project against the installed package from outside the build
tree, and additionally requires two failures: an impossible version query must
be refused, and removing the installed headers must break the build. A
packaging test that only ever passes proves nothing about packaging.
