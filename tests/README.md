# `tests/` — the ctest suite

49 registered targets with Kokkos, **21 without it**. The count is asserted in
CI (`.github/workflows/ci.yml`, `expected=49`), not assumed. That is how this
lane once ran 34 targets while reporting 31. Four targets used to sit behind
`if(XPMATH_MPFR_FOUND)`, so a runner missing a `-dev` package lost coverage and
stayed green; that guard is gone — MPFR/MPC are the sweep oracle now, and a
missing package is a configure `FATAL_ERROR`.

`-DXPMATH_WITH_KOKKOS=OFF` drops the 28 targets that link Kokkos and keeps the
other 21 — both gates, both gate self-tests, the oracle-conversion and reduction
pairs, `domains_fresh`, `build_provenance`, the consumer package test and the
seven standalone smokes. Measured: 21/21 in 169 s with no Kokkos installed.

**What judges correctness is `docs/CORRECTNESS.md`.** Read it before adding
anything that issues a verdict; the whole point of the current arrangement is
that nothing else competes with the sweep gates. This file is a map of what is
registered, nothing more.

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

`hello_test` (harness plumbing on a trivial DD round-trip) and `corpus_test`
(the corner-case corpus itself). Neither runs a real math op.

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
