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

Kokkos is used by this repository's own demos and by its device test matrix
(7 nvcc targets plus hipcc in CI), through a compat wrapper in
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
