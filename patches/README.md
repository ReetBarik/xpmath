# Local Kokkos patches

**There are none, and that is the point of this file.**

This directory used to carry `kokkos_complex_quad_math.hpp`, a local extension
to Kokkos's `core/src/impl/Kokkos_QuadPrecisionMath.hpp` adding the
`__complex128` overloads (`Kokkos::exp`, `Kokkos::sqrt`, `Kokkos::conj`, …)
that Kokkos does not ship. Every overload was a one-line forward to the
corresponding `libquadmath` `::c<fn>q`. Building the four complex demos meant
copying the header into a Kokkos SOURCE tree and rebuilding Kokkos with
`-DKokkos_ENABLE_LIBQUADMATH=ON`, so the repo did not consume a Kokkos install
— it consumed a *patched* one.

It existed for one reason: the `__complex128` accuracy columns the complex
demos printed. Those columns are gone. The demos are timing and smoke only, and
the accuracy record is `validation/sweep/` — error in ulps against an MPFR/MPC
oracle at 400 bits, one verdict per point, two ctest gates over it
(`docs/CORRECTNESS.md`). The demo columns duplicated that measurement at lower
resolution while costing the project a patched Kokkos.

Deleting them deleted, in one move:

- `patches/kokkos_complex_quad_math.hpp` and its verifier
  `scripts/smoke_kokkos_complex_quad.cpp`;
- the `check_cxx_source_compiles` probe for
  `impl/Kokkos_ComplexQuadPrecisionMath.hpp` in `CMakeLists.txt`, and the
  `KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER` guards that left four demo targets
  unbuilt when it failed (KI-21);
- the `Kokkos_ENABLE_LIBQUADMATH=ON` requirement on the Kokkos install, and the
  configure-time WARNING when it was absent.

**Any Kokkos ≥5.1 built at C++20 now works, libquadmath or not.** The MI250
(`hip/gfx90a`) install, which has libquadmath OFF, is a first-class supported
target for configuration.

## The one remaining libquadmath consumer

`src/bench_cost.cpp` includes `<quadmath.h>` and calls `::expq` / `::powq`
directly. That is not an oracle: `__float128` is the *incumbent* the DD and QF
backends are timed against, so removing it would delete the measurement rather
than relocate it. `kokkos_ep_bench_cost` is therefore the single target in this
repo that still needs libquadmath at link time, and it is the reason CI still
builds its Kokkos with `-DKokkos_ENABLE_LIBQUADMATH=ON`. No demo, no test, and
no installed header needs it.

## If you need the header back

It is not lost — it is in the history. `git log --diff-filter=D --
patches/kokkos_complex_quad_math.hpp` finds the commit that removed it, and the
file is intact in that commit's parent, along with the version of this README
that explained how to apply it. It was Apache-2.0 WITH LLVM-exception, matching
Kokkos, so it could still be upstreamed verbatim.
