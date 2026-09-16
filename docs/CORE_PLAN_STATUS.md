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

---

## C2 — Split `test_utils.hpp` into a device-safe half and a host-oracle half

**Branch:** `core/c2-split-test-utils`. **Base:** `main` @ `93f4b53`, rebased
onto C1 (`a470834`) before merge so that this section's STATUS block lands after
C1's rather than conflicting with it. Both gates below were measured on
C2-over-`93f4b53`, before that rebase. The two sections are disjoint outside
`.github/workflows/ci.yml`, `.gitignore` and this file -- C1 touched no file
under `tests/` -- and the rebase resolved exactly one conflict, the
append-append on this file.

The combined tree was then MEASURED rather than assumed, since neither section's
gates had built the other's code. Both configurations were rebuilt on the
rebased tree and run with the three ~11-minute selftests excluded -- those had
already passed in both gates above and are re-run by CI:

```
C1+C2, --arch host        build 0 errors, 50 registered, 47/47 pass
C1+C2, --no-kokkos        build 0 errors, 22 registered, 19/19 pass
```

**Outcome.** Green. `tests/test_utils.hpp` is gone, split into
`tests/test_utils_device.hpp` (what a device translation unit may include) and
`tests/test_utils_host.hpp` (that header plus the oracle surface). No forwarding
shim was left at the old name, deliberately: a shim would let a device TU pull
in the oracle by accident, which is the failure this section exists to prevent.
Counts moved 49 -> 50 and 21 -> 22, both as the plan predicted, the one new
target being `device_tu_purity`.

**Gate 1** — `scripts/xpm_build.sh --arch host --build-dir /tmp/c2_host`, then
`ctest --test-dir /tmp/c2_host -j8 --timeout 1800`:

```
50/50 Test #38: sweep_monotone_gate_selftest .....   Passed  672.85 sec

100% tests passed, 0 tests failed out of 50
Label Time Summary:
packaging    =   4.91 sec*proc (1 test)
Total Test time (real) = 702.50 sec
```

**Gate 2** — `scripts/xpm_build.sh --arch host --no-kokkos --build-dir /tmp/c2_nok`,
then `ctest --test-dir /tmp/c2_nok -j8 --timeout 1800`:

```
22/22 Test #10: sweep_monotone_gate_selftest .....   Passed  674.30 sec

100% tests passed, 0 tests failed out of 22
Label Time Summary:
packaging    =   4.92 sec*proc (1 test)
Total Test time (real) = 674.31 sec
```

### What the split actually moved

`to_quad` was the hinge. It returns `__float128`, so it cannot sit in a
device-safe header, but it lived in `BackendTraits<B>` alongside `max_digits`
and `sig_bits`, which are device-safe and which device code needs. Splitting the
struct rather than the header would have duplicated the metadata. So
`BackendTraits<B>` stays whole in the device header and the host header adds
`OracleTraits<B> : BackendTraits<B>`, carrying `to_quad` alone. That is the
whole reason `dd_property_test.cpp` and `ff_property_test.cpp` show ~60 changed
lines each while every other TU shows one or two: they are
`BackendTraits<X>::to_quad` -> `OracleTraits<X>::to_quad` renames, not logic
changes.

### Three findings the next sections need

**1. It is 21 includers, not 22.** Measured at `93f4b53`:
`git grep -l 'include "test_utils.hpp"'` returns 21 test TUs. The 22nd file is
`tests/corpus.hpp`, which names the header in a comment and does not include it.
Its comment was repointed. Nothing else about the file changed.

**2. The plan's motivating example is wrong, and C4 must not inherit it.** The
C2 text says `hello_test` "contains no `__float128` of its own" and was an A100
casualty merely for including the header. It is not: `hello_test.cpp:41` uses
`float128` and `OracleTraits`, and `:56` uses `AccStats`. That was already true
at `93f4b53`, before this section touched the file. The same claim had been
written into `tests/test_utils_device.hpp`, `scripts/check_device_tu_purity.sh`
and `tests/CMakeLists.txt` during this section and was corrected in all three
before commit.

The correction matters because it changes what the split buys. **Splitting the
header rescues no currently-registered test for device.** Every host-classified
TU uses an oracle symbol of its own. What it buys is an honest boundary: five
TUs now include no oracle at all, and C4 has a header its device TUs can include
without dragging one in. The "9 of 49 not built on A100" figure is S8b's and is
unchanged by this section.

**3. Classification was done by compiling, and all 16 host TUs earned it.** The
plan requires classifying by compiling rather than reading, so each
host-classified TU was re-compiled with its include swapped to the device
header. All 16 failed, so none is over-assigned:

```
corpus_test 30 errors      dd_e2e 455      dd_eft 449      dd_fma_guard 435
dd_invariant 447           dd_property 590 ff_cancellation 455
ff_invariant 447           ff_property 590 hello_test 16   qf_cancellation 454
qf_eft 447                 qf_nonoverlap 465               qf_property 551
tf_cancellation 37         tf_property 27
```

The five device-classified TUs are `ff_eft_test`, `ff_fma_guard_test`,
`qf_fma_guard_test`, `tf_eft_test` and `tf_fma_guard_test`. `corpus_test`,
`tf_property_test` and `tf_cancellation_test` are the nearest misses at 30, 27
and 37 errors, and are the first places C4 should look for a cheap device half.

### Deviation from the plan, and why it is the plan's own intent

The plan says register `device_tu_purity` in the Kokkos-free set and add it to
the `no-kokkos-build` lane. As first written, the device header included
`<Kokkos_Core.hpp>` -- `BackendTraits<DD>::type` was
`Kokkos::Experimental::DoubleDouble` -- so the gate could not run without a
Kokkos install. MEASURED both ways before deciding:

```
with    -I<kokkos>/include : ok, 130341 preprocessed lines, 0 in-repo __float128
without                    : FAIL -- Kokkos_Core.hpp: No such file or directory
```

Rather than register it in the Kokkos set and defer, the device header was
pointed at the xp core (`<xp/dd_math.hpp>`) instead of the `third_party/include`
compat wrappers. This costs no type churn:
`third_party/include/dd_math.hpp:53` is `using DoubleDouble = xp::DoubleDouble`,
a true alias rather than a distinct wrapper, so the core spelling names exactly
the type the Kokkos spelling names. After the change the same probe reads
`ok, 56339 preprocessed lines` **with or without** Kokkos on the include path,
and the gate is registered in the Kokkos-free set as written.

The five device TUs each gained an explicit `#include <Kokkos_Core.hpp>`: they
drive `Kokkos::View` / `parallel_for` / `initialize` directly and had been
receiving the declaration transitively through the harness header. The host
header took over supplying `<Kokkos_Core.hpp>` and the two compat wrappers, so
every host TU sees exactly what it saw before the split.

`device_tu_purity` enters the `no-kokkos-build` CI lane by being registered in
the Kokkos-free set, not by a bespoke step -- which is stronger, since the
lane's `expected=22` assertion now fails if the gate ever silently unregisters.

### The gate can fail, and was made to

A gate nobody has seen go red is indistinguishable from one that cannot. The
device header was poisoned with `using poisoned_t = __float128;`, and:

```
poisoned : FAIL: tests/test_utils_device.hpp -- __float128 reaches a device TU
           from files in this repo:  1 .../tests/test_utils_device.hpp   rc=1
restored : ok  (56339 preprocessed lines, 0 in-repo __float128)          rc=0
```

with the file verified byte-identical afterwards.

`scripts/check_device_tu_purity.sh` is deliberately **not** `grep __float128`,
which is what one would write first and which would have been red from birth:
`/usr/include/bits/floatn.h` typedefs the type on x86_64 whether anyone uses it
or not. The script walks the preprocessor's `# <line> "<file>"` markers,
attributes each occurrence to its originating file, and fails only on hits from
inside this repository -- system headers excluded by provenance rather than by a
name allowlist that would rot.

**Notes for later sections.**

- `FILES` in `scripts/check_device_tu_purity.sh` is the maintained list and holds
  one entry today. C4 extends it with every `*_test_device.cpp` it creates; the
  script already accepts extra `-I` arguments for that.
- The counts of record are now **50 with Kokkos, 22 without**, asserted in
  `.github/workflows/ci.yml` (`expected=50`, `expected=22`) and stated in
  `tests/README.md`.
- Stale `tests/test_utils.hpp` citations remain in `docs/TEST_SUITE_PLAN.md`,
  `docs/PORT_NOTES_TF.md` (`:530`, `ep_exit_code`), `docs/UPSTREAM_PLAN_STATUS.md`
  and `docs/history/KNOWN_ISSUES.md`. The last two are historical records and
  correctly name the file as it was. The first two are C1's files this cycle and
  were left alone to avoid a conflict; they want a docs pass, not a C2 edit.
- `.claude/` is now in `.gitignore`. Worktrees created under
  `.claude/worktrees/` live inside the checkout, so without it a `git add -A` in
  the main checkout commits a whole parallel working tree.
- Gate 1's build predates three comment-only corrections (finding 2 above, in
  `tests/test_utils_device.hpp`, `scripts/check_device_tu_purity.sh` and
  `tests/CMakeLists.txt`) and the `.gitignore` line. None is code. Gate 2 built
  the final tree, and `device_tu_purity` was re-run against it directly.
