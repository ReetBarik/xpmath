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
---

## C3 — A Kokkos-free device harness, and the two real gfx90a build blockers

**Branch:** `core/c3-device-harness`. **Base:** `main` @ `520280c`.

**Outcome.** Green. Both gfx90a build blockers that were still live are fixed, the
Kokkos-free device harness exists with a self-test that cannot pass on silence,
the `device-hip` CI lane is gating, and the TF standalone smoke gap is closed.
Counts moved 50 -> 52 and 22 -> 24 exactly as the section said they would.

**What the section fixed, and what it only narrowed.** S8c's "39 of 49 targets
do not build on gfx90a" was measured at `0230239` and two of its four causes
were already fixed on `main` before this section started. C3 fixes the two that
remained — the missing HIP runtime include and the unconditional
`-fext-numeric-literals` — so the live blocker is smaller again. **This section
does not measure it.** No gfx90a hardware was touched here; the number stays
C8's to produce, and nothing below should be read as a target count.

### Step 1 — the HIP bit-cast bug: PRIMARY path, not the fallback

`include/xp/trig_reduction.hpp` calls `__double_as_longlong` and
`__float_as_int` under `XPMATH_ON_DEVICE_CUDA_OR_HIP`, and no `xp/` header
included a vendor runtime. nvcc force-includes `cuda_runtime.h` into every TU
it compiles, so the names were always declared there; hipcc has no equivalent
behaviour, so under hipcc they were simply undefined.

The primary fix was taken: a guarded `#include <hip/hip_runtime.h>` under
`__HIPCC__`, placed in `include/xp/config.hpp` beside the
`XPMATH_ON_DEVICE_CUDA_OR_HIP` definition.

Three decisions inside that, recorded because the fallback is still available
and a later reader should know why it was not needed:

- **`__HIPCC__`, not `__HIP_DEVICE_COMPILE__`.** The latter is the device pass
  only. Both passes must see the same declarations or they disagree on the
  symbol, and `hip/hip_runtime.h` is written to be included in both.
- **No `cuda_runtime.h` companion.** nvcc already supplies it, so the include
  would buy nothing while adding a failure mode for clang-CUDA.
- **In `config.hpp`, not `trig_reduction.hpp`.** `config.hpp` is the header
  every other `xp/` header already includes, so the next intrinsic user costs
  nothing.

**The fallback was not exercised and its premise was not tested.** The fallback
is to delete the intrinsic branch and use `std::memcpy` unconditionally, then
prove via `scripts/xpm_lint_device_asm.sh` that the gfx90a codegen did not
regress. It is unnecessary only if the primary fix compiles under hipcc, and
**there is no hipcc on this login node** — nothing local can execute that test.
The evidence is the `device-hip` CI lane, which step 5 made gating in the same
commit precisely so this claim is checked by something rather than asserted.

### Step 5 — the `device-hip` lane is now gating

`continue-on-error: true` is gone from the lane and "UNVERIFIED, non-gating" is
gone from its name. The lane header comment keeps the history, because the
reason it was advisory is the argument for removing it: the lane is the only
automated thing that could have caught the step-1 defect, and because it was
advisory the defect instead survived to be found by hand on MI250X as S8c cause
2. An advisory lane that reports a real defect nobody must act on is a lane that
does not exist.

**The other `continue-on-error: true` in the file was left alone.** There are
two. The second belongs to "Report against the committed baseline (non-gating)"
inside the monotone-gate lane, and removing it would convert a deliberately
non-gating report into a gate that fails on coverage growth. Only the device-hip
one was removed.

### Deviation — the harness is compiled by BOTH device lanes, not just HIP

The plan's step 5 says to add `tests/device_harness.hpp` to the headers the hip
lane compiles. That was done. It was ALSO added to `device-nvcc`, as a ninth
matrix row (`kind: harness`), which the plan does not ask for.

The reason: C4–C8 measure the core THROUGH this harness, and with only the HIP
lane compiling it the entire CUDA path — `cudaMalloc`, `cudaMemcpy`,
`cudaDeviceSynchronize`, the `<<<>>>` launch — would ship having been compiled
by nothing at all. That is the same "unverified by construction" state step 5
exists to end, and C7 is an A100 section. Adding the row costs one container
job.

In both lanes the harness is **instantiated**, not merely included. A header of
templates that nobody instantiates compiles clean while being completely broken,
which is the §0 "passes on silence" trap in a different costume. Each lane
builds a functor struct, a `buffer<double>`, a `parallel_for_n` launch, a fence
and a copy-back, and compiles it `-c` for sm_80 / gfx90a respectively.

`tf_math.hpp` needed no adding: the hip lane already compiled all eight xp
headers.

### Deviation — `last_error()` is sticky, and the name does not say so

The surface C3 specifies is `int last_error(); // 0 == ok; vendor error code
otherwise`, and that spelling was kept verbatim. Its SEMANTICS are the first
nonzero code since process start, not the most recent one.

A literal "last" would be actively harmful here: `cudaGetLastError()` and
`hipGetLastError()` CLEAR the error as they report it, so one later successful
call would erase the evidence of an earlier failed one and a broken run would
test green. Every vendor call in the harness funnels its code into a sticky slot
instead, and `last_error()` reads that slot. The header documents it at the
point of definition. Recorded here because the name is the spec's and the
behaviour is not what the name suggests.

### The harness self-test cannot pass on silence, and was made to fail

`tests/device_harness_test.cpp` poisons the output buffer on BOTH sides before
the launch — host-side, then `to_device()`, so the poison is resident in device
memory — and the kernel writes `in[i]*2 + 1` with `in[i] = 0.5*i`, i.e. exactly
`i + 1`, a value that is always ≥ 1.0 against a poison of −12345.0. A kernel
that never runs leaves poison on the device; a `from_device()` that copies
nothing leaves poison on the host. Either is red.

The unreachability of the poison is asserted AT RUNTIME rather than argued in a
comment, so a later edit to either constant that made the poison reachable
would turn the check into a tautology loudly instead of quietly.

MEASURED both ways, kernel body neutered to `(void)i;` and then restored
byte-identical:

```
neutered  ->  the kernel actually wrote every element [FAIL]  4096 of 4096 elements still hold the poison
              every element is exactly i + 1          [FAIL]  4096 wrong; first at i=0: got -12345 want 1
              FAIL (2 failures)                              exit=1
              ctest: 9 - device_harness_test (Failed)

restored  ->  1/1 Test #9: device_harness_test .....  Passed  0.00 sec
              100% tests passed, 0 tests failed out of 1
```

### Gate

Fresh `rm -rf` build of each configuration, then the full suite. Both counts
were asserted before the suite was allowed to run, because `ctest --test-dir` on
a missing directory exits 0 and a suite that never ran is indistinguishable from
one that passed.

**Gate 1** — `scripts/xpm_build.sh --arch host --build-dir /tmp/c3_host`:

```
host: registered 52 (expected 52)

52/52 Test #40: sweep_monotone_gate_selftest .....   Passed  675.42 sec

100% tests passed, 0 tests failed out of 52

Total Test time (real) = 705.21 sec
```

**Gate 2** — `scripts/xpm_build.sh --arch host --no-kokkos --build-dir /tmp/c3_nok`:

```
nok: registered 24 (expected 24)

24/24 Test #12: sweep_monotone_gate_selftest .....   Passed  676.02 sec

100% tests passed, 0 tests failed out of 24

Total Test time (real) = 676.03 sec
```

**Gate 3** — `scripts/check_standalone_no_kokkos.sh`, now 18 steps over eight TUs
rather than 16 over seven:

```
[14/18] run TF ...
tf_no_kokkos_smoke: PASS (0 failures)
[17/18] preprocessed output contains no Kokkos ...
      ok (62307 preprocessed lines, 0 Kokkos hits)
[18/18] include/xp/*.hpp name Kokkos only in comments ...
      ok

=== PASS: the standalone core stands alone ===
```

**Gate 4** — `scripts/check_device_tu_purity.sh`, two files now:

```
device-TU purity: g++ 13.3.0, 2 file(s)
  ok   tests/test_utils_device.hpp  (56339 preprocessed lines, 0 in-repo __float128)
  ok   tests/device_harness.hpp     (22204 preprocessed lines, 0 in-repo __float128)
device-TU purity: PASS
```

**NOT a gate this section can run.** Nothing above executes hipcc or nvcc. The
two device lanes are the real gate for C3 and they run on the PR.

### Step 1's open question, closed by CI

The step-1 section above says the fallback's premise could not be tested
locally because
there is no hipcc on this login node, and that the `device-hip` lane is what
checks it. It has now checked it. On PR #30 @ `b00b79c`, the lane's FIRST
execution in `rocm/dev-ubuntu-24.04:latest` was green:

```
hipcc device compile                           PASS
```

That is three claims settled at once, and none of them was settled before:

- The primary fix compiles under a real hipcc. The `std::memcpy` fallback is not
  needed and was never applied.
- The container works. The lane's advisory status existed because a first red
  would have been indistinguishable from a broken image; the gating flip now
  rests on a green run rather than on the argument for one.
- The gfx90a codegen guard passes with the harness in the TU, so neither the
  branch-reach mitigation nor the de-recursion regressed under the new code.

`nvcc device compile (device_harness)` was green in the same run, so both vendor
paths of the harness are compiled. Neither is EXECUTED: that is C7 and C8.

### Notes for later sections

- **Counts of record are now 52 with Kokkos and 24 without**, asserted in
  `.github/workflows/ci.yml` (`expected=52`, `expected=24`) and stated in
  `tests/README.md`. C3 added `tf_no_kokkos_smoke` and `device_harness_test`.
- `CLAUDE.md` still says 49 / 21. It was already stale at 50 / 22 before this
  section, so C3 did not silently fold a two-cycle documentation drift into its
  own diff. It wants a docs pass.
- `scripts/check_device_tu_purity.sh`'s `FILES` list now holds two entries. C4
  extends it with every `*_test_device.cpp` it creates.
- `tests/device_harness.hpp` is a TEST header. It is not installed, it is not on
  the public include path, and the two device lanes reach it with an explicit
  `-Itests`. Nothing in `include/xp/` may ever include it.
- **The harness has been COMPILED for both vendors and EXECUTED on neither.**
  ctest runs its serial fallback; the CUDA and HIP paths are compile-only in CI.
  First execution on real hardware is C7 (A100) and C8 (MI250X).
- `-Wall -Wextra` on the standalone smokes surfaces two pre-existing
  set-but-unused warnings in `include/xp/tf_math.hpp` (`k_log2`) and
  `include/xp/tf_complex.hpp` (`y2`). Neither is C3's and neither was touched.
---

## C4 — Every test TU is host-only or device-only, and no test links Kokkos

**Branch:** `core/c4-tu-split`. **Base:** `main` @ `bd0fa8e` (the C3 merge).
Five commits, one per chunk: A `0df9f5a`, B `bb314a4`, C `c445086`, D `5cd68e7`,
E below.

**Outcome.** Green, and the section's real assertion holds:

> **61 registered ctest targets with Kokkos, and the same 61 without it.**

Not merely equal counts — the two `ctest -N` NAME LISTS are IDENTICAL as sets,
`diff` clean over all 61. `tests/CMakeLists.txt` has no `if(XPMATH_WITH_KOKKOS)`
block left for a target to move across, because no test TU links Kokkos any more.

52 → 61 is not nine new tests. It is nine mixed translation units split into a
host half and a device half, each half registering separately, minus the
arithmetic of the `_contract_on` companions that moved with them.

### The count was 52/24 and the gap was the point of the section

Before C4, 28 of 52 targets existed only when Kokkos did, so the `no-kokkos`
lane ran a strict subset and "green without Kokkos" was a materially weaker
statement than "green" — the two lanes could not contradict each other because
one could not see most of what the other ran. That is now closed. `expected=61`
is asserted in BOTH lanes of `.github/workflows/ci.yml`, and the comment at each
site says that a future divergence means a Kokkos-conditional target came back
and should be fixed rather than forked into two constants.

### Chunk E step 1 — the MIRROR check, and the defect it found on first run

`scripts/check_device_tu_purity.sh` now runs two checks pointing in opposite
directions. C2 wrote the forward one; C4 added the mirror:

```
FORWARD (part 1)   no DEVICE TU may carry __float128
MIRROR  (part 2)   no HOST TU may carry a kernel launch
```

Neither implies the other, and the mirror was not decoration. **On its first
run it went red on `tests/hello_test.cpp`, and it was right.** Chunk A had taken
that file off Kokkos and re-pointed its launch at `tests/device_harness.hpp`; it
built clean, ran green, and passed the forward check trivially because it is not
in `FILES`, so the forward check never looked at it. What it actually was is a
TU that includes `tests/test_utils_host.hpp` (binary128) AND launches a kernel —
the exact shape nvcc rejects under S6 and the exact shape this section exists to
eliminate. It was a NINTH mixed TU and the plan does not name it, because the
plan has it on the "links Kokkos and barely uses it" list instead, where lifting
the launch onto the harness looks like the whole job.

It was SPLIT, not exempted. `tests/hello_test.cpp` keeps the target name, the
oracle and the 10^6-input `DD(x) → binary128 == x` identity;
`tests/hello_test_device.cpp` carries the launch.

**Nothing was lost in that split, and the reasoning is the one that generalises.**
The old device pass asserted `digits_of_accuracy(device output, (float128)x) >=
max_digits`. The new one asserts the returned limbs are BIT-IDENTICAL to the
limbs that went in — which is strictly stronger, and needs no reference value,
so it fits in a device-pure TU. Both halves draw the same 10^6 inputs from the
same engine and seed, with the distribution constructed per draw to match
`test_utils_host.hpp`'s `uniform()` exactly rather than approximately.

Three design points in the mirror worth not re-deriving:

- **The host set is DERIVED, not maintained.** It is every top-level
  `tests/*.cpp` that `FILES` does not claim. One list partitions the tree, so a
  new test TU is host by default and gets the mirror for free; giving it a
  launch turns the mirror red until it is added to `FILES`; adding it to `FILES`
  subjects it to the forward check. There is no third state and no way to be in
  neither. A second, separately-maintained host list would have drifted.
- **It preprocesses; it does not grep source.** A raw
  `grep -E 'parallel_for|KOKKOS_LAMBDA' tests/*.cpp` returns hits in NINE host
  TUs on this tree, EIGHT of which are prose comments explaining where that
  file's launch went. The preprocessor strips comments, and attribution by
  `# <line> "<file>"` provenance is the same methodology part 1 already used.
  The limit is recorded at the call site: a STRING LITERAL naming a token is not
  stripped and WILL fail. Chunks C and D each hit this and reworded one `printf`.
- **Both halves refuse to pass on an empty input set** (§0). Part 1 fails if
  `FILES` is empty; part 1b fails if the `tests/*_test_device.cpp` glob matches
  zero files, and also fails if any match is missing from `FILES`; part 2 fails
  if `tests/*.cpp` is empty or if the derived host set is.

**Part 1b is new and is what makes `FILES` self-checking.** Before it, forgetting
to list a device half was silent. It is also why `tests/device_harness_test.cpp`
is now in `FILES` — C3's own harness self-test, the most device-shaped TU in the
tree, had simply never been added.

### The mirror has been seen red (§0)

`tests/pow_domain_test.cpp` was poisoned with a real launch — a
`device_harness.hpp` include, an `XpPoisonOp` functor and a live
`xpt::parallel_for_n(4, ...)` — then restored. MEASURED against the shipping
script, not an earlier draft:

```
poisoned:
  host-TU mirror: 33 tests/*.cpp total, 17 classified host
  FAIL: tests/pow_domain_test.cpp -- a kernel launch reaches this TU from files in this repo:
            1 .../tests/pow_domain_test.cpp
            1 .../tests/device_harness.hpp
  device-TU purity: FAIL                                               exit=1

restored:
  device-TU purity: PASS                                               exit=0
```

`sha256` before poison and after restore both
`da1b9bc3acc8f80767e278b69db9910edd18eced3e8610e61bb7aef8130526f9`, and
`git diff --name-only` on the file returns nothing. Note the diagnostic names
BOTH the TU and the header the token arrived through, which is what makes it
actionable rather than merely red.

### Chunk E step 2 — the three `kokkos_ep_add_*` helpers are gone

`kokkos_ep_add_test`, `kokkos_ep_add_eft_test` and
`kokkos_ep_add_eft_test_contract_on` are deleted, along with the
`THIRD_PARTY_INCLUDE` plumbing under `tests/`. A comment block stands where they
were, saying what still exercises the Kokkos wrapper layer: the eight
`src/demo_*.cpp` targets, until C10 moves them to the `xpmath-kokkos` repo.
`THIRD_PARTY_INCLUDE` remains SET by the top-level `CMakeLists.txt` for those
demos; the comment there that claimed `tests/` consumed it was corrected in the
same commit rather than left to rot.

### EVERY assertion that lost device coverage, with the reason

The plan's rule is that an assertion which cannot be expressed in a device
functor without the oracle is a host assertion, and that each one be named here.
Four, in two groups. **None was replaced by a substitute, deliberately.**

**1. `B1_sqrt_sq` and `B4_pythag`, on DD (chunk C), FF (chunk C) and QF (chunk
D)** — six instances of two checks. They scored `digits_of_accuracy` against the
binary128 oracle on the device side. They are gone from the device and the host
half runs both (DD/FF at 10^6 inputs, 10× the old device pass's 10^5; QF at
2×10^5). Two reasons they cannot come across, and the second is the one that
matters:

- They are not structural. "sqrt(a)² recovers a to N digits" is a claim about a
  REFERENCE VALUE and the only reference is the oracle. Unlike `two_sum`, which
  has an independent exact transform to cross-check against, `sqrt` has no
  second exact algorithm at these widths.
- **Host/device bit-parity is NOT a substitute and shipping it would be worse
  than shipping nothing.** A real GPU's `sqrt` and `sincos` may legitimately
  differ from the host libm in the last bits, so a parity check would go red on
  correct hardware — and the only way to make it green again is a tolerance,
  which is a SECOND SCORER. `docs/CORRECTNESS.md` allows one measurement and one
  verdict per point.

What is genuinely gone is the ability to notice a GPU whose `sqrt` or `sincos`
is accurate on the host and inaccurate on the device. The apparatus that covers
that end to end is `validation/sweep/`, and it is host-measured for the same
reason: the oracle cannot share a translation unit with device code (S6).

**2. FF Group S (chunk C) stays host-only, and that is a SCOPE decision rather
than a limitation.** It is a table of IEEE special-value requirements for divide
(`x / ±inf = ±0`, NaN propagation, `FLT_MAX` divisors). Non-finite behaviour is
precisely what a device backend's fast-math posture may legitimately change, so
asserting it on the device would assert something no platform promises.

**Three splits lost nothing at all** and are recorded so nobody re-audits them:
`qf_eft_test` Tests A–D are host batch/named-case work with no kernel (Test C's
binary128 truncation check is the single thing that made the TU mixed); Test E
runs once before and once after, same inputs, same order, same oracle. The
device halves of `dd_property_test`, `ff_property_test` and `qf_property_test`
all GREW: the old device passes ran three Group A identities each and now run
all seven / all seven / all twelve, each additionally sweeping the corner-case
corpus the host runners always swept and the device passes never did.

### Decisions taken where the plan was ambiguous or wrong

1. **`hello_test` is a ninth mixed TU.** The plan names eight and puts this one
   on a different list. Split rather than exempted — see above. The alternative
   was to leave it out of the mirror's host set, which is weakening the check to
   make something pass, and the plan forbids that in as many words.
2. **`ff_fma_guard_test` and `qf_fma_guard_test` were migrated WHOLE, not
   split.** Post-C2 each includes `test_utils_device.hpp` only, and each one's
   oracle is an exact FP64 product (a 48-bit result in a 53-bit mantissa), not a
   128-bit one. They are in `FILES` without a `_device` sibling. MEASURED before
   adding each: the wide type's name appeared only in prose except for one
   `printf` string literal per file, which the preprocessor does not strip;
   both were reworded.
3. **The eight `tests/*_test_device.cpp` TUs are now compiled by BOTH device CI
   lanes**, as one extra job each (`devtests` matrix row on `device-nvcc`, a new
   step on `device-hip`). The plan does not ask for this. The argument is C3's
   own, one section later: C4's entire claim is that these TUs can be given a
   device pass, and the in-tree gate for that claim preprocesses with **g++** and
   checks **one token**. That is necessary and not sufficient — it is a proxy
   checked by a compiler that is not the one whose rejection (S6) started this.
   Without the lanes the claim ships verified by nothing but its own proxy, and
   C5 builds a device tree on top of it. Both steps assert the glob is non-empty
   and matches the expected 8 before compiling, because `for f in <no matches>`
   compiles nothing and exits 0. `-x cu` / `-x hip` are required: the files are
   `.cpp` because the host g++ suite builds them too, and without the flag the
   vendor driver hands them to the host compiler and gives them no device pass —
   the one thing the step exists to do. The HIP step is kept SEPARATE from the
   existing `--save-temps` compile rather than folded into it, because that TU is
   the gfx90a codegen guard's subject and adding eight unrelated objects would
   change what tiers 1–4 are judging.
4. **`CLAUDE.md`'s counts were corrected here**, breaking C3's precedent of
   deferring them to a docs pass. C3 left 49/21 standing because it was already
   two cycles stale and folding that in would have hidden a drift inside its own
   diff. The situation changed: after C4 the sentence is not merely numerically
   stale but STRUCTURALLY false — it asserts a with-Kokkos/without-Kokkos subset
   relationship that no longer exists. One paragraph was rewritten; nothing else
   in that file was touched.
5. **Stale `kokkos_ep_add_*` references in `docs/TEST_SUITE_PLAN.md` and
   `docs/PORT_NOTES_TF.md` were left alone.** They are historical records of
   completed T-tasks and correctly name the helpers as they were at the time.
   They want the same docs pass C2 and C3 both asked for.
6. **The grammar of the gate's own success line was fixed** (`scan()` takes a
   singular and a plural subject). Trivial, but the mirror prints one line per
   file across 17 files and "0 in-repo a kernel launch" is the sort of thing that
   makes a reader doubt the gate rather than the sentence.

### Gate

Fresh `rm -rf` build of each configuration, then the FULL suite with nothing
excluded. Both directories were confirmed to EXIST and both counts asserted
BEFORE the suite was allowed to run, because `ctest --test-dir` on a missing
directory exits 0 and a suite that never ran is indistinguishable from one that
passed.

```
/tmp/c4_host exists, registered=61
/tmp/c4_nok  exists, registered=61
diff of the two `ctest -N` name lists: EMPTY (identical, not merely equal)
```

**Gate 1** — `scripts/xpm_build.sh --arch host --build-dir /tmp/c4_host`, then
`ctest --test-dir /tmp/c4_host -j8 --timeout 1800`:

```
host: registered 61 (expected 61)

61/61 Test #49: sweep_monotone_gate_selftest ...........   Passed  674.71 sec

100% tests passed, 0 tests failed out of 61

Label Time Summary:
packaging    =   3.97 sec*proc (1 test)

Total Test time (real) = 704.10 sec
```

**Gate 2** — `scripts/xpm_build.sh --arch host --no-kokkos --build-dir /tmp/c4_nok`,
then `ctest --test-dir /tmp/c4_nok -j8 --timeout 1800`:

```
nok: registered 61 (expected 61)

61/61 Test #49: sweep_monotone_gate_selftest ...........   Passed  672.58 sec

100% tests passed, 0 tests failed out of 61

Label Time Summary:
packaging    =   4.89 sec*proc (1 test)

Total Test time (real) = 702.00 sec
```

**Gate 3** — `scripts/check_device_tu_purity.sh`, now 18 files forward, 8 in the
`FILES` completeness check, 17 host in the mirror:

```
=== part 1: device TUs carry no __float128 ===
device-TU purity: g++ 13.3.0, 18 file(s)
  ok   tests/test_utils_device.hpp  (56339 preprocessed lines, 0 in-repo __float128)
  ... 18 files, all ok ...
=== part 1b: every tests/*_test_device.cpp is listed in FILES ===
  8 file(s) match tests/*_test_device.cpp
  ... 8 files, all in FILES ...
=== part 2 (mirror): host TUs contain no kernel launch ===
host-TU mirror: 33 tests/*.cpp total, 17 classified host
                (launch tokens: Kokkos::parallel_for|parallel_for_n|KOKKOS_LAMBDA)
  ... 17 files, all ok ...
device-TU purity: PASS
```

33 = 17 host + 16 device `.cpp` (`FILES` holds 18 entries, two of which are
headers). The partition is exhaustive by construction.

The contraction guard is unchanged by all of this and says so:
`contraction guard: 11 OFF targets, 6 ON targets (expected 11/6)`.

**NOT a gate this section can run.** Nothing above executes nvcc or hipcc —
there is neither on this login node. The two device lanes are the real gate for
decision 3 and they run on the PR.

### Notes for later sections

- **Counts of record are 61 and 61**, asserted in both lanes of
  `.github/workflows/ci.yml` and stated in `tests/README.md` and `CLAUDE.md`.
  The `expected=8` device-TU count now appears in THREE places — part 1b of the
  purity script, and the two new CI steps. They must move together; each CI step
  says so in its error message.
- **The host/device partition is now mechanically enforced in both directions.**
  Adding a test TU requires no bookkeeping unless it launches a kernel, in which
  case the mirror tells you exactly what to do. Do not add a second host list.
- **The device halves have been COMPILED for both vendors and EXECUTED on
  neither** — still. ctest runs the harness's serial fallback; the CUDA and HIP
  paths are compile-only. C5 gives them a device build tree, C7 (A100) and C8
  (MI250X) are first execution on hardware. What C4 adds over C3 is that the
  real test TUs, not a synthetic instantiation, are now what the lanes compile.
- **Six device-side accuracy assertions are gone and are not coming back**
  (`B1_sqrt_sq` / `B4_pythag` × DD, FF, QF). If a future section wants
  device-side accuracy, the answer is not a tolerance in a test — it is running
  the sweep on the device, which needs the oracle out of the TU, which is the
  same S6 problem. Do not re-litigate it with a parity check.
- The demos still link Kokkos and still include `third_party/include`. C4
  touched none of them, and they are now the ONLY thing keeping the compat
  wrapper layer compiled. C10 moves them.
