# `tests/` — the ctest suite

**61 registered targets, with Kokkos or without it — the same 61 either way.**
The count is asserted in CI (`.github/workflows/ci.yml`, `expected=61` in *both*
lanes), not assumed. That is how this lane once ran 34 targets while reporting
31. Four targets used to sit behind `if(XPMATH_MPFR_FOUND)`, so a runner missing
a `-dev` package lost coverage and stayed green; that guard is gone — MPFR/MPC
are the sweep oracle now, and a missing package is a configure `FATAL_ERROR`.

`-DXPMATH_WITH_KOKKOS=OFF` now drops **nothing**. It used to drop 28 of 52,
leaving 24, because those 28 targets only existed inside an
`if(XPMATH_WITH_KOKKOS)` block; "green without Kokkos" was therefore a much
weaker claim than "green". CORE_PLAN C4 removed the block: no test TU links
Kokkos, the device-side tests launch through `tests/device_harness.hpp` instead
of `Kokkos::parallel_for`, and the three `kokkos_ep_add_*` helpers that linked
`Kokkos::kokkos` are deleted. MEASURED on the C4 gate run — the two `ctest -N`
name lists are not merely the same length, they are identical as sets.

52 → 61 is not nine new tests. C4 split nine mixed translation units (eight
named in the plan, plus `hello_test`) into a host half and a device half, and
each half registers as its own target. The Kokkos wrapper layer in
`third_party/include/` is still exercised — by the eight `src/demo_*.cpp`
targets, until C10 moves them to the `xpmath-kokkos` repo.

`device_harness_test` is the self-test for `tests/device_harness.hpp`, the
Kokkos-free CUDA/HIP/serial launch harness C3 added for C4–C8 to measure
through. Under plain `g++` what runs is the harness's SERIAL fallback, which is
deliberate: that fallback is the path nobody exercises on a GPU node and the one
every device test will assume behaves like the vendor paths. It cannot pass on
silence — it poisons both sides of the output buffer before the launch and
requires every element to differ from the poison AND to equal an exact function
of its index, so a kernel that never ran, or a `from_device()` that copied
nothing, is red rather than quiet.

`device_tu_purity` needs no Kokkos install, which is deliberate rather than
incidental: it names the xp core rather than the `third_party/include` wrappers,
and the lane most likely to introduce a device-purity regression is the one
where nobody builds with Kokkos at all. It runs **two checks pointing in
opposite directions**, and neither implies the other:

- **forward** — no device TU may carry `__float128`. Each listed
  `tests/*_test_device.cpp` and `tests/test_utils_device.hpp` is preprocessed
  and fails if `__float128` reaches it *from a file in this repository*. The
  provenance qualifier is load-bearing: `/usr/include/bits/floatn.h` typedefs
  the type unconditionally on x86_64, so a name-blind grep can only ever be red.
- **mirror** (C4) — no host TU may contain a kernel launch
  (`Kokkos::parallel_for`, `xpt::parallel_for_n`, `KOKKOS_LAMBDA`). The host set
  is *derived*, not maintained: every top-level `tests/*.cpp` that the device
  list does not claim. A new test is therefore host by default and gets the
  mirror for free; giving it a launch turns the mirror red until it is added to
  the device list, which subjects it to the forward check. There is no third
  state. This also preprocesses rather than greps source, because several host
  halves explain in prose comments where their launch went and a raw grep cannot
  tell a comment from a call.

Both halves refuse to pass on an empty input set, and the gate has been seen
red: C4 poisoned a host TU with a real launch, watched it fail, and restored the
file byte-identically. It found one genuine defect on its first run —
`hello_test` had been taken off Kokkos but left mixed.

**What judges correctness is `docs/CORRECTNESS.md`.** Read it before adding
anything that issues a verdict; the whole point of the current arrangement is
that nothing else competes with the sweep gates. This file is a map of what is
registered, nothing more.

## One target here is not a test: `sweep_device`

`tests/CMakeLists.txt` builds `scripts/sweep_device.cpp` into
`<build>/tests/sweep_device` in the **device tree only** (CORE_PLAN C6 step 1).
It evaluates every backend × op × grid point through `tests/device_harness.hpp`
and writes the raw result limbs as hex bit patterns; the host `sweep_accuracy`
scores them. That is C6's shape — **one scorer, two producers** — and the
producer issues no verdict, holds no oracle and computes no ulps, because
`docs/CORRECTNESS.md` permits exactly one scorer.

It registers **no `add_test()`**, so the 61 above is unchanged and the side-tag
assertion does not apply to it (that assertion constrains registered tests, not
targets). The device gates that will run it are C6 step 5. Do not "fix" the
count on account of this target.

## The accuracy record

| Target | What it asserts |
|---|---|
| `sweep_absolute_gate` | No point is above its derived bound unless it is listed in `validation/sweep/open_defects.txt`. Checked in both directions: an unlisted point above bound fails, and a listed point no longer above bound also fails. |
| `sweep_monotone_gate` | No point is worse than `validation/sweep/sweep_baseline.csv.gz`. |
| `sweep_absolute_gate_selftest` | Poisons the register — an entry deleted, a bogus entry added, the register emptied — and requires the absolute gate to fail each time. |
| `sweep_monotone_gate_selftest` | Poisons the baseline and requires the monotone gate to fail through a **named exit code**: 1 regression, 2 coverage removed, 3 record drift, 5 improvement drift, 6 coverage growth. Also requires it to pass clean input. |

Exit codes are asserted exactly, never "nonzero" — a case that starts failing
through the wrong door has stopped testing what it was written for.

## Reduction, domain and oracle

| Target | What it asserts | Needs MPFR |
|---|---|---|
| `pow_domain_test` | `pow`'s domain predicates match the shipped header. | |
| `exp_reduction_test` | Cody–Waite reduction for `exp`. | |
| `exp_reduction_selftest` | Compile-time poison: rebuilds the source with a corrupted constant and requires the test to catch it. | |
| `trig_reduction_test` | Payne–Hanek reduction for the trig family. | yes |
| `trig_reduction_selftest` | Compile-time poison for the above. | yes |
| `oracle_conv_test` | Conversions in and out of the oracle types round-trip. | yes |
| `oracle_conv_selftest` | Compile-time poison, carrying the `why[]` table that asserts **which** check catches **which** poison. | yes |

The three `*_selftest` shell harnesses poison **source at compile time** and
assert on a grep marker; the poisoned binaries deliberately exit 0. This is a
different contract from `gate_selftest.sh`, which poisons **CSV data at runtime**
and asserts exit codes. Do not merge the two.

## Per-backend tests

Four backends: `dd` (2×FP64, p=106), `ff` (2×FP32, p=48), `qf` (4×FP32, p=96),
`tf` (3×FP32, p=72).

| Family | Targets | What it asserts |
|---|---|---|
| EFT | `dd_eft_test`, `ff_eft_test`, `qf_eft_test`, `tf_eft_test`, `tf_eft_test_contract_on` | `two_sum` / `two_product` are bit-exact, plus `two_sqr` and the `renorm` family on QF/TF. |
| Invariant | `dd_invariant_test`, `ff_invariant_test`, `qf_nonoverlap_test` | Component words do not overlap on the output of every op. |
| Property | `dd_property_test`, `ff_property_test`, `qf_property_test`, `tf_property_test` | Algebraic identities: one group bit-exact, one tolerance-gated, one named-constant regression. |
| FMA guard | `{dd,ff,qf,tf}_fma_guard_test` and their `_contract_on` variants | The Dekker products survive `-ffp-contract=off`; the `_contract_on` build reports what contraction collapses. |
| Cancellation / e2e | `dd_e2e_test`, `ff_cancellation_test`, `qf_cancellation_test`, `tf_cancellation_test` | Cancellation-heavy kernels: √(x²+1)−x, Σ1/k², Machin's π, alternating harmonic. |

## Packaging and generated artifacts

| Target | What it asserts |
|---|---|
| `consumer_package` | A separate CMake project can `find_package(xpmath)` against the install tree and compile against it. |
| `domains_fresh` | `docs/DOMAINS.md` still matches what the CSVs imply. |
| `build_provenance` | The build directory carries a `build-info.txt` naming the arch, git HEAD (with `-dirty`), the resolved compiler and version, the Kokkos prefix, the full `CMAKE_CXX_FLAGS`, the `-O` level and a UTC timestamp. Written by the top-level `CMakeLists.txt` on **every** configure, not by `scripts/xpm_build.sh` — a stamp only the wrapper wrote would be missing from exactly the builds nobody can trace. Judges presence and non-emptiness of the fields, never their values; see the header of `check_build_provenance.cmake`. |

## Scaffolding

`hello_test` / `hello_test_device` (harness plumbing on a trivial DD round-trip)
and `corpus_test` (the corner-case corpus itself). Neither runs a real math op.

The `hello_test` pair is the C4 split in miniature and worth reading as the
worked example. The host half asserts `DD(x) → binary128 == x` bit-exactly over
10^6 inputs; the device half launches an identity kernel over the *same* 10^6
inputs (same engine, same seed, distribution constructed per draw so the
sequences really do match) and asserts the returned limbs are bit-identical to
the limbs that went in. The pre-split file asserted `digits_of_accuracy(device
output, oracle) >= max_digits`; bit-equality is strictly stronger and needs no
oracle, so the device half carries no `__float128` and the claim survives the
split intact.

## Retired

The eight `*_accuracy_test` targets — `{dd,ff,qf,tf}_accuracy_test` and their
`*_complex_accuracy_test` counterparts — are **gone**, along with the
`kokkos_ep_add_standalone_test` helper. They gated on a per-op **mean**, which
cannot see a single point getting worse, and one of them
(`tf_accuracy_test`) shipped for weeks with an exit code that could not report a
failure at all. The sweep and its two gates replaced them. See
"What this replaced, and what it did not" in `docs/CORRECTNESS.md`.

## Build and run

```bash
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j$(nproc)
ctest --test-dir build
```

The sweep gates are the slow ones (~7 s per sweep run, and each self-test case
is one run — budget ~45 s per mode).
