# Correctness — one measurement, one verdict, two gates

This is the whole correctness apparatus for the library. There is one number, one
verdict per point, two gates over that number, and one generated document. If you
want to know whether an operation is correct, run `ctest`; nothing else in the
repository issues a competing opinion.

## 1. The measurement

**Error in ulps against a `__float128` / `__complex128` oracle.**

```
ulps = |got - true| / (|true| * 2^-p)          p = 106 (DD), 96 (QF), 72 (TF), 48 (FF)
```

For complex ops, `|.|` is the complex modulus, so one number covers both
components. The digit score `-log10(relative error)` is still printed and is what
`docs/DOMAINS.md` displays, because a reader wants "about 29 digits" — but it is a
derived display value. Ulps is the number of record, and it is what both gates
compare.

The measurement engine is `scripts/sweep_accuracy.cpp`, built by CMake as the
`sweep_accuracy` target. It evaluates 39 real and 24 complex operations on four
backends over a fixed grid of 1,652 real and 1,780 complex points (seed 12345),
which is 428,592 scored points, in about eight seconds.

The oracle is libquadmath, and *which* libquadmath matters: the baseline carries
an `oracle-fingerprint` header and the gates warn when it does not match. The
toolchain of record is GCC 13.3.0, fingerprint `578322f998a329c8`.

## 2. The verdict

Each point gets exactly one of two verdicts: **at or below its bound**, or
**above it**. A third state, **unresolved**, means no verdict is issuable.

The bound is derived from the format and from the mathematics. It is never set
from what the implementation currently scores — that was the defect in the
per-op tolerance tables this replaces, which made the current behaviour the
specification.

```
bound = (residual + kappa * input_delta + kappa_intermediate) * gain + absolute_floor
```

| term | what it is |
|---|---|
| `residual` | the subnormal-low-word floor of the **result**: `max(1, 2^(Emin_sub+p)/|f|)`. Below the cliff (DD 2.0e-292, QF 5.6e-17, TF 3.3e-24, FF 2.0e-31) the low limb goes subnormal and resolution is absolute, not relative. |
| `kappa` | the condition number `\|x f'(x)/f(x)\|`, summed over operands. For complex ops it is obtained by differentiating the binary128 oracle at `h = 2^-40` in **four independent real directions** (re and im of each operand) — a radial-only probe cannot see a tangential sensitivity, which is exactly the case for `log` near `\|z\| = 1`. |
| `input_delta` | how far the stored operand is from the exact one, in ulps — including its own subnormal floor. This is what the op is actually being asked about. |
| `kappa_intermediate` | **the cancellation the algorithm carries internally**: `max\|I\| / \|f\| - 1` over the quantities `I` the algorithm must hold in `p` bits on the way to the answer. Read off each op's expression tree, not guessed. `fma` holds `a*b`; `hypot` holds `a²` and `b²`; complex `div` holds `q = \|b\|²` and the four scaled bilinear products; complex `asin/acos/asinh/acosh` hold `z²` against 1; complex `atan/atanh` hold `z` against 1. |
| `gain` | the algorithm's own amplification. The exp family squares its reduced argument `nq` times, and squaring doubles relative error, so the gain is `2^nq`. The log family is Newton-on-exp and inherits the same factor. Everything else is 1. |
| `absolute_floor` | the trig family's absolute floor. The joint sin/cos doubling recurrence gives `e_{j+1} <= 2 e_j + 2^-p`, hence `e_nq <= 2^nq (e_0 + 2^-p)`, saturating at the codomain diameter 2. Argument reduction adds `x` itself, and the reduction multiple `n = nint(x/2π)` must fit **one word** (53-bit FP64 limb, 24-bit FP32 limb) or the reduction is not formable at all. |

### Why intermediate width is a separate term

A format limit can live entirely inside the algorithm, where nothing about the
operands or the answer reveals it. The product of two 53-bit doubles needs 106
bits: DD has them, QF/TF/FF do not, and a bound derived only from input width and
output width sees none of that. The `kappa_intermediate` term is what makes those
visible, and clause (3) below is what catches the intermediate that leaves the
format's *range* rather than its precision — `q = |b|²` in complex division
underflows the FP32 limb long before `b` does.

### Unresolved

No verdict is issued when any of four things holds — one derived predicate that
replaced seven hand-maintained exemption names (`UNDERFLOW`, `OVERFLOW`,
`ARG_RANGE`, `CONDITIONING`, `UNEXPLAINED`, `SUBNORMAL_LIMB`, `JUMP_UNRESOLVED`):

1. **the answer does not fit** — the true result is zero, non-finite, or outside
   the format's range;
2. **the question does not fit** — an operand is outside the format's range;
3. **the work does not fit** — an intermediate is outside the format's range;
4. **the bound already says everything is lost** — the derived bound is at or
   above `2^p` ulps, so "at bound" would be vacuous.

Clause 4 is why the old jump-discontinuity exemption is gone: for `fmod` and
`remainder` the jump contributes `2^p |b|/|f|` to the bound and reaches clause 4
by arithmetic instead of by being on a list.

## 3. The two gates

Both are ctest targets. `ctest` alone is the whole answer. Each also has a
self-test target that poisons its input and requires it to fail — see the end of
this section.

### `sweep_absolute_gate` — no point above its bound

```
sweep_accuracy --ulp --register validation/sweep/open_defects.txt
```

Some points *are* above bound. Those are open defects, and they are carried by
name in `validation/sweep/open_defects.txt` rather than absorbed by loosening the
bound until they vanish. That is the difference between a register and a
tolerance: **a tolerance hides the count, a register prints it.**

The register is checked in **both** directions. A point above bound that is not
listed fails the gate — a new defect. A listed point that is no longer above
bound also fails — the register has rotted and must be shrunk. So the file can
only get smaller, and it cannot silently stop meaning anything.

### `sweep_monotone_gate` — no point got worse

```
sweep_accuracy --baseline validation/sweep/sweep_baseline.csv.gz
```

Compares every point's ulp count against the committed baseline. Larger is worse.

**The noise floor.** The oracle's own last place moves under a compiler or glibc
change, and re-running the identical binary moves roughly twenty of the 428,592
rows. The gate therefore reports a regression only when a point's error grows by
more than a factor of **1.2589254 = 10^0.1**, i.e. one tenth of a digit, and only
when the fresh value exceeds one ulp — below one ulp the score is a rounding
coin-flip and 0.2 → 0.4 ulps is not a defect.

A stated multiplicative threshold was chosen over run-twice-and-intersect
deliberately: intersecting doubles an eight-second gate's cost to suppress noise
that a threshold suppresses for free, and 1.26 is a number you can read in the
source and argue with. Intersection hides its own criterion inside a sampling
procedure.

A point that changes **state** (scored ↔ unresolved) is always reported, and
losing a verdict counts as a regression even though no ulp count grew.

**Every recorded field is read.** A baseline row is
`backend,kind,op,point,digits,ulps,bound,state`. The first four are matched
positionally; `ulps` carries the verdict. `digits`, `bound` and `state` are not
verdicts — they are the *record*, and they are checked for the separate question
of whether the file describes this build at all. A rerun under the toolchain of
record reproduces the committed baseline byte for byte, so any row whose bound
moved by more than 10^0.1 relatively, whose digit count moved by more than 0.1,
or whose state moved at all, means the baseline is stale or edited. The gate
then exits **3** and asks for a re-baseline, rather than comparing against a
standard that is no longer the standard. Exit 1 is a regression, exit 2 is a
grid or op-inventory change, exit 0 is a pass.

**A record can also be stale in the BETTER direction, and that is exit 5.** Rows
that are *better* than the baseline records mean the file has stopped describing
this build just as surely as rows that are worse — but the two cannot be treated
identically, because an accuracy fix legitimately improves rows and a gate that
blocks fixes gets routed around. The gate cannot tell a fix's own improvement
from unexplained drift: every row-level property was tried and none separates
them (magnitude — both unbounded; row count — the legitimate `atan2` fix moved
1,032 rows, *more* than the unexplained case; concentration — both single-cell;
state — the complex-div fix is state `U`, which is why `both_scored` was dropped
from the exemption in the first place).

So the gate stops asking whether the improvement is legitimate and asks instead
whether a committed artifact was just invalidated. `sweep_monotone_gate` compares
against the committed `.gz`, where an improvement means the record is stale and
the remedy — a re-baseline in its own commit — is already mandated; that call
site fails on 5. CI's monotone lane regenerates the parent baseline from the
parent's own source every run, so there is nothing durable to re-record and the
same signal is informational; that call site accepts 5 and still fails 1, 2 and
3. Same binary, same predicate, two call sites that differ *structurally* — which
is why this is an exit code and not a flag. A flag is asserted per run by
whoever is under pressure, and would be pasted in the first time CI went red.

One interlock: a different oracle improves thousands of rows at once and is not
a library improvement. Measured — `--oracle=mpfr` against the committed baseline
yields 1,631 ungated improvements. When improvement drift coincides with an
oracle fingerprint mismatch the gate returns **3**, not 5, because the reference
moved rather than the library.

This mattered: `digits` and `bound` used to be parsed and discarded, and only
the *losing* direction of a state move was counted. Rewriting the last numeric
column — `bound`, not `ulps` — of all 428,592 rows to zero printed
`unchanged: 428592 / RESULT: PASS`, and so did rewriting every state to `U`.

### `sweep_monotone_gate_selftest`, `sweep_absolute_gate_selftest`

```
validation/gate_selftest.sh <sweep_accuracy> monotone|absolute <workdir>
```

Both gates are negative assertions, and a negative assertion that has only ever
been observed passing is indistinguishable from one that cannot fail. So each
gate is run against deliberately poisoned inputs built in the **build**
directory — never in `validation/` — and is required to exit with a
specific, named code:

| mode | poison | must exit |
|---|---|---|
| monotone | `ulps` column zeroed | 1 (regression) |
| monotone | `bound` column zeroed | 3 (record drift) |
| monotone | `digits` column zeroed | 3 (record drift) |
| monotone | `digits` column inflated by 5 | 3 (record drift, up-direction) |
| monotone | `state` column all `U` | 3 (record drift) |
| monotone | baseline truncated to 1000 rows | 2 (grid changed) |
| monotone | baseline empty / header-only / not gzip | 2 (nothing to compare) |
| monotone | `ulps` column inflated x1000 | 5 (stale in the better direction) |
| monotone | ONE row rewritten to 1e6 ulps | 5 (the smallest unit that must fire) |
| monotone | `ulps` improved by 1.05x (sub-noise) | 0 — must stay silent |
| absolute | a registered defect deleted | 4 (unlisted point above bound) |
| absolute | a bogus entry added | 4 (stale register entry) |
| absolute | register emptied | 4 (every above-bound point unlisted) |

The self-test asserts the exact **exit code**, not merely nonzero. Four distinct
failure doors exist, and a case that starts failing through the wrong one has
stopped testing what it was written for — when exit 5 was added, `digits-inflated`
would otherwise have stayed green while silently changing which door it
exercised.

Two of these were measured failing to fire *before* the fix that made them fire:
the whole-file `ulps` inflation scored `increased: 397409` and exited 0, and the
single-row case scored `increased: 1` and exited 0. The single-row case is the
stronger of the two, because it cannot be satisfied by a future "N% of rows
moved" heuristic. The sub-noise row is the negative control: without it the gate
would fire on run-to-run wiggle.

Each mode also runs the **real** input and requires exit 0, so a gate wired to
fail unconditionally does not satisfy the self-test either.

## 4. The document

`docs/DOMAINS.md` is generated by `scripts/gen_domains.py` from the baseline, the
grid and the register — per backend per op, the usable range and where it
degrades. It is guarded by the `domains_fresh` ctest target, so it cannot drift
from the data.

## 5. Running it

```bash
module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3   # gcc FIRST
export LD_LIBRARY_PATH=/soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64:$LD_LIBRARY_PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/kokkos-install-quadmath
cmake --build build -j16
ctest --test-dir build -j8 --timeout 1800
```

The `LD_LIBRARY_PATH` export is not optional: without it the binary links the
system libquadmath, the oracle fingerprint stops matching `578322f998a329c8`, and
you get a scatter of hairline differences that are the reference moving rather
than the library.

### Re-baselining after an accepted change

```bash
build/tests/sweep_accuracy --ulp --register validation/sweep/open_defects.txt \
    --out /tmp/base.csv
gzip -9 -c /tmp/base.csv > validation/sweep/sweep_baseline.csv.gz
python3 scripts/gen_domains.py > docs/DOMAINS.md
```

**Trap:** `sweep_accuracy` with no `--out` writes the committed baseline in place,
and `--grid-out` does the same to the grid. Always pass an explicit path.

## 6. What this replaced, and what it did not

Retired: the eight mean-gated accuracy tests (`dd/ff/qf/tf_accuracy_test` and
their four complex counterparts), and the five-category classifier
(`UNDERFLOW` / `OVERFLOW` / `ARG_RANGE` / `CONDITIONING` / `UNEXPLAINED`) with its
`sweep_classified.csv` output. Reasons are in `tests/CMakeLists.txt` next to the
retirements.

Kept, untouched: every structural test. The bit-exact EFT tests
(`two_sum`/`two_prod`/`two_sqr`), the FMA-contraction guards in both postures,
the representation invariants, the QF and TF non-overlap tests, the property and
identity tests, and the cancellation kernels. Those assert exact algebraic facts,
not tolerances, so they were never part of the problem.

The development history of the forty defects found and fixed along the way is in
`docs/history/KNOWN_ISSUES.md`. It is history, not a list of open issues; the open
list is `validation/sweep/open_defects.txt`.
