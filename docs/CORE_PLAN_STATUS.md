# CORE_PLAN_STATUS.md

This file records the outcome of each section of the core plan — the arc in
which the C++ library becomes the standalone artifact and device execution
stops being a rumour. One STATUS block per completed section, appended in
completion order, newest last. It follows the convention of
`docs/UPSTREAM_PLAN_STATUS.md`: each block states what was done, what was
measured, and what the next section must know.

---

## C0 — Land the plans and capture the reference state

**Branch:** `core/c0-plan-landing`. **Base:** `main` @ `53cdb7f`.

**Outcome.** Green. C0 changes no code — it is a measurement section, and its
job is to establish that `main` is in the state the remaining ten sections
assume. All three gates pass at `53cdb7f`, so the counts of record stand and
later sections may rely on them.

**Gate 1** — `scripts/xpm_build.sh --arch host --build-dir /tmp/c0_host`, then
`ctest --test-dir /tmp/c0_host -j8 --timeout 1800`:

```
49/49 Test #38: sweep_monotone_gate_selftest .....   Passed  677.07 sec

100% tests passed, 0 tests failed out of 49

Label Time Summary:
packaging    =   5.09 sec*proc (1 test)

Total Test time (real) = 706.78 sec
```

**Gate 2** — `scripts/xpm_build.sh --arch host --no-kokkos --build-dir /tmp/c0_nok`,
then `ctest --test-dir /tmp/c0_nok -j8 --timeout 1800`:

```
21/21 Test #10: sweep_monotone_gate_selftest .....   Passed  675.96 sec

100% tests passed, 0 tests failed out of 21

Label Time Summary:
packaging    =   4.77 sec*proc (1 test)

Total Test time (real) = 675.98 sec
```

**Gate 3** — oracle fingerprint:

```
# oracle-fingerprint: 44f18a4a959f6c29
```

This matches the expected value, and matches the fingerprint recorded in the
header of the tracked `validation/sweep/sweep_baseline.csv.gz`.

**Deviation — the documented fingerprint command does not produce a
fingerprint.** The command of record was

```bash
/tmp/c0_host/tests/sweep_accuracy --oracle-selftest 2>&1 | grep -i fingerprint
```

`--oracle-selftest` prints no fingerprint. It is a conversion selftest that
returns immediately (`oracle conversion selftest: PASS (0 failures)  poison=0`)
and exits without running the sweep, so the `grep` matches nothing and the
pipeline is silent. An empty result there is a broken instrument, not a red
gate, and must not be read as one.

The fingerprint is emitted only by `write_baseline`, as a `# oracle-fingerprint:`
header line on the baseline it writes (`scripts/sweep_accuracy.cpp:3356`). The
command that actually measures a build's fingerprint is a full sweep written to
a throwaway path:

```bash
/tmp/c0_host/tests/sweep_accuracy --out /tmp/c0_fp_baseline.csv --quiet
grep -i oracle-fingerprint /tmp/c0_fp_baseline.csv
```

Note the explicit `--out` to a `/tmp` path — never the committed baseline path.
Later sections needing the fingerprint should use this form; the sweep takes
roughly three minutes on a login node.

**Notes for later sections.**

- The counts of record are confirmed at `53cdb7f`: 49/49 with Kokkos, 21/21
  without. A section that changes either count without saying so in advance has
  a bug.
- Both suites are dominated by `sweep_monotone_gate_selftest` at roughly eleven
  minutes. Budget wall time for it rather than assuming a fast `ctest`.
- `docs/UPSTREAM_PLAN.md` S6 is now marked SUPERSEDED, pointing at this file.
  S4 and S9 are untouched and belong to the Kokkos layer's plan; S8 stays
  PARTIAL and is narrowed, not closed, by C7 and C8.
- No Kokkos status file was created here: that file belongs to the
  `xpmath-kokkos` repository and is created there by K0.

---

## C1 — Retire libquadmath from everything this project builds

**Branch:** `core/c1-retire-libquadmath`. **Base:** `main` @ `93f4b53`.

**Outcome.** Green. Nothing this project builds calls libquadmath any more. The
last consumer, `src/bench_cost.cpp`, is deleted rather than ported — that is a
decision of record, not an omission: the precision-tier-peer cost comparison is
lost on purpose and `docs/PERF_PLAN.md` stays parked with a line saying so.
The LIBQUADMATH TPL is gone from the CI Kokkos build, so CI now exercises the
same Kokkos shape as the MI250 (hip/gfx90a) install.

**Counts are unchanged, as predicted.** `kokkos_ep_bench_cost` was an
executable, not a registered test.

**Gate 1** — `scripts/xpm_build.sh --arch host --build-dir /tmp/c1_host`, then
`ctest --test-dir /tmp/c1_host -j8 --timeout 1800`:

```
49/49 Test #38: sweep_monotone_gate_selftest .....   Passed  674.25 sec

100% tests passed, 0 tests failed out of 49

Label Time Summary:
packaging    =   4.89 sec*proc (1 test)

Total Test time (real) = 704.24 sec
```

**Gate 2** — `scripts/xpm_build.sh --arch host --no-kokkos --build-dir /tmp/c1_nok`,
then `ctest --test-dir /tmp/c1_nok -j8 --timeout 1800`:

```
21/21 Test #10: sweep_monotone_gate_selftest .....   Passed  674.18 sec

100% tests passed, 0 tests failed out of 21

Label Time Summary:
packaging    =   4.90 sec*proc (1 test)

Total Test time (real) = 674.20 sec
```

**Gate 3** — the new CI guard, run against both build trees:

```
############ /tmp/c1_host (Kokkos) ############
scanned 62 binaries
GUARD EXIT: 0
############ /tmp/c1_nok (no Kokkos) ############
scanned 28 binaries
GUARD EXIT: 0
```

**The plan's guard regex was wrong, and it took a real build to see it.** As
written, ` (c?[a-z0-9_]+q)$` fired on `/tmp/c1_host/tests/trig_reduction_selftest/gen`
with the symbol `__gmpz_fdiv_q` — GMP's integer floor-division, pulled in by
MPFR through the oracle, and nothing whatever to do with libquadmath. The
underscore class is what does it. The guard in this PR is anchored and forbids
underscores: ` (c?[a-z][a-z0-9]*q)$`.

That change was made against a measurement, not a guess:

| set | anchored regex matches |
|---|---|
| libquadmath's own exports ending in `q` | **92 of 92** |
| symbols defined by gmp, mpfr, mpc, libm, libstdc++, libc | **0** |

Every libquadmath export ending in `q` is bare lowercase (`expq`, `cpowq`,
`cabsq`); none contains an underscore or a capital. So the anchored form loses
no coverage and drops the one false positive.

**The guard was also validated in the positive direction**, because a guard
that has only ever been seen to pass is indistinguishable from a guard that
cannot fire. A deliberate `expq` caller was compiled and scanned; the guard
found `U expq` and fired. The poison's argument is derived from `argc` rather
than a literal, so it cannot be constant-folded away into a blind test. The
guard additionally asserts that its binary list is non-empty and echoes the
count it scanned, per §0: a gate whose pass condition is silence must prove it
ran.

**`ldd` remains the wrong check** and was confirmed to be so here: `ldd` on
`/tmp/c1_host/tests/dd_invariant_test` still reports
`libquadmath.so.0 => /usr/lib64/libquadmath.so.0` in a tree with zero bound
libquadmath symbols. That DT_NEEDED is Kokkos's property, from the TPL on the
JLSE Kokkos install, and it is exactly the false failure `tests/CMakeLists.txt`
lines 50-65 warn about. Do not "fix" a future red `ldd` by relaxing the symbol
guard.

**Moved to `scripts/attic/`** (six files, via `git mv`, none referenced by any
`CMakeLists.txt` — checked before each move), with a `README.md` recording that
their numbers are not comparable point-for-point with the sweep's, and that
`gen_qf_constants.cpp` is the provenance of committed header constants and must
not be deleted:

```
scripts/attic/gen_corpus.cpp
scripts/attic/gen_qf_constants.cpp
scripts/attic/probe_acos_branch.cpp
scripts/attic/probe_complex_oracle.cpp
scripts/attic/probe_cpow.cpp
scripts/attic/test_qfmul.cpp
```

**Notes for later sections.**

- CI's Kokkos is now built without the LIBQUADMATH TPL and the cache key moved
  to `...-serial-noquadmath-...-v2`. A stale `-v1` hit would have silently
  reintroduced the flag, which is why both tokens changed in one commit.
- The monotone lane's `-lquadmath` at `.github/workflows/ci.yml:296` is
  **deliberately untouched**. It compiles the PARENT commit, and parents that
  predate this section still need it.
- The `quadmath` hits remaining under `src/` and `tests/` are all historical
  prose in comments — eight demo-header lines explaining why the dependency was
  dropped, and one line in `tests/dd_invariant_test.cpp` citing `remainderq()`
  as a cross-check value. No `#include <quadmath.h>` and no `*q` call survives
  anywhere the project builds; the symbol guard is the enforcement, not the
  grep.
- `tests/` was not touched by this section at all (`git diff main -- tests/`
  is empty), by arrangement with C2.
