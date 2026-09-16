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
