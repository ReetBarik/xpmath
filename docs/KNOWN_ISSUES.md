# Known issues — deferred, to be picked up after the upstream arc

Findings that are real, reproduced, and deliberately **not** fixed at the time they
were found, because fixing them was out of scope for the sub-plan that surfaced
them. Each entry states the evidence, why it was deferred, and what closing it
would involve.

This file is the backlog. It is not a status document — see
`docs/UPSTREAM_PLAN_STATUS.md` for what each sub-plan actually did.

---

## STATUS TABLE

Read this first. It is the authoritative status of every entry; the bodies below
carry the evidence and the history.

**Marker convention (adopted 2026-09-04).** Every `## KI-n` heading carries
**exactly one** marker, and it is one of:

- **`[RESOLVED <commit>]`** — fixed, verified. `<commit>` is the sha that carried
  the fix. An entry that was fixed, reopened, and fixed again reads
  `[RESOLVED <final-commit>]`; the reopen history lives in the body, never in the
  heading. Where two shas were needed, both are listed, oldest first.
- **`[OPEN]`** — not fixed, or only partially fixed. A partial fix is OPEN.

No other marker is legal. `[REOPENED]` is not a heading state — an entry is
either currently resolved or currently open.

| KI | one-line summary | status | fixing commit |
|---|---|---|---|
| 1 | Complex `acosh` wrong throughout the left half-plane, all four backends | RESOLVED | `f421a0e` |
| 2 | `nint` wrong for the one-ulp-below-a-tie class | RESOLVED | `650aa16` |
| 3 | `build_with_kokkos.sh` passes C++17 to a Kokkos that requires C++20 | RESOLVED | `dd6d00a` |
| 4 | DD `sin` returns the wrong sign near odd multiples of π | RESOLVED | `8135332` |
| 5 | Four defects in the complex inverse family — (a) `asinh`, (b) `log1p`/`atanh`, (c) `acos`, (d) `asin` | RESOLVED | `650aa16` (a), `3968a9f` (b, c), `f421a0e` (d) |
| 6 | `exp` flushes to zero outside a hard ±300 guard, losing ~170 decades | RESOLVED | `ad82f4f` |
| 7 | `tanh` returns the wrong sign for \|x\| past the `exp` guard | RESOLVED | `ad82f4f` |
| 8 | `hypot` and complex `abs` unscaled — overflow/underflow far inside range | RESOLVED | `4cd7dcb`, then `7680809` |
| 9 | QF/TF `div` returns NaN when the quotient outruns the Dekker splitter | RESOLVED | `82427f6`, completed by `0ad44fe` (KI-19) |
| 10 | `fmod`/`remainder` lose half their digits on far-apart operands | RESOLVED | `fc7b0fb`, then `58fda3b` |
| 11 | Complex inverse family loses digits when one component ≪ the other | RESOLVED | `855292d` (`atan`/`atanh`), then batch-8 ¶ (`asin`/`acos`/`asinh`/`acosh`/`sqrt`) |
| 12 | FF `sincos` never converges below \|x\| ≈ 1e-20 and skips its out-params | RESOLVED | `0ad44fe` (headline), then `2e810c2` (reduction band) |
| 13 | `angle`/`asinh`/`acosh` form an unscaled sum of squares — NaN above 1.8e19 | RESOLVED | `7680809` |
| 14 | FF `floor`/`ceil`/`trunc`/`round` return zero for every \|x\| ≥ 2^47 | RESOLVED | `2e810c2` |
| 15 | `fmod`/`remainder` return the wrong sign, zero, or infinity | RESOLVED | `fc7b0fb`, then `58fda3b` |
| 16 | `atanh`'s domain guard tests only the leading word | RESOLVED | batch-5 † |
| 17 | TF `asinh` has no odd-symmetry branch | RESOLVED | batch-5 † |
| 18 | Complex `tan`/`tanh`/`atan`/`atanh` have no asymptotic branch | RESOLVED | `855292d` |
| 19 | QF/TF `divide` returns NaN where `inf` is correct, on quotient overflow | RESOLVED | `0ad44fe` |
| 20 | `round` is half-to-even, not half-away-from-zero | **OPEN** | — |
| 21 | Missing-complex-oracle path is a hard build failure, not degradation | RESOLVED | batch-8 ¶ |
| 22 | DD `asinh`/`atanh`/`sinh` and TF `expm1` collapse to the leading word at small x | RESOLVED | batch-5 † |
| 23 | QF `log`/`log2`/`log10`/`log1p` lose ~11 digits above \|x\| ≈ 1e29 | RESOLVED | batch-6 ‡ |
| 24 | FF `sinh`/`cosh` lose one full FP32 word near the `exp` range limit | RESOLVED | batch-7 § |
| 25 | QF and TF `atan` return a non-finite value at \|x\| ≈ 3.16e19 | RESOLVED | `0ad44fe` |
| 26 | TF `sin`/`cos`/`tan` and QF `cos`/`tan` non-finite at very large arguments | RESOLVED | `0ad44fe` |
| 27 | DD and FF `multiply` return NaN on genuine product overflow | RESOLVED | `2e810c2` |
| 28 | DD and FF complex squaring returns NaN where the true result is finite-or-inf | RESOLVED | batch-6 ‡ |
| 29 | `asinh` mid-band scatter: the odd reflection costs a few ulps for 1 ≲ \|x\| ≲ 20 | **OPEN** | — |
| 30 | DD `multiply` returns NaN whenever either operand exceeds ~1.34e300 (Dekker splitter) | RESOLVED | batch-7 § |
| 31 | DD `divide_scalar` returns a quotient 2^64 too large above 1.339e300 | RESOLVED | batch-7 § |
| 32 | FF complex `asin` returns 0 for its REAL part on the far real axis — `iz + sqrt(1-z^2)` cancels to (0,0) | **OPEN** | — |

**29 resolved, 3 open** (20, 29, 32).

† `batch-5` is the commit titled `fix: KI-16 value-based domain guards; KI-17 TF
`asinh` odd symmetry; KI-22 small-argument series` — a commit cannot record its
own sha, and this table ships inside it.

‡ `batch-6` is the commit titled `fix: KI-23 QF log family at large argument;
KI-28 complex multiply Annex G recovery` — same reason: a commit cannot record
its own sha, and this table ships inside it.

§ `batch-7` is the commit titled `fix: KI-30 scale multiply past the Dekker
splitter limit; KI-24 fixed` — same reason.

¶ `batch-8` is the commit titled `fix: KI-11 complex inverse small-component;
KI-21 vanilla-Kokkos degradation` — same reason.

Three sections that are not KI entries also live in this file and are neither
resolved nor open: the classifier verdict (2026-09-03), the soft-failure map
(2026-09-03), and the two "Findings from …" dividers.

---

## KI-1 — Complex `acosh` is wrong throughout the left half-plane, in ALL FOUR backends **[RESOLVED f421a0e]**

> **Scope corrected 2026-09-02.** The title used to say "on the negative real
> axis". The complex accuracy tests added on 2026-09-02 show the defect covers the
> whole of `Re(z) < 0`, not just the cut — see *Extent*, below.

**Severity: high (silently wrong answers), scope: pre-existing, not introduced by S10.**

### What

For real `z < -1`, complex `acosh` returns numerical noise. Measured against a
`cacoshq` (`__complex128`) oracle, with each backend given inputs at its own word
width:

```
point                          DD      QF      TF      FF
------------------------------------------------------------
2 + 0i        generic        31.00   29.00   20.90   13.90
5 + 3i        generic        30.49   27.84   21.70   14.00
0.5 + 0i      cut, |z|<1     30.24   28.91   21.70   13.48
-0.5 + 0i     cut, |z|<1     30.72   28.96   21.70   14.00
-1 + 0i       branch point   31.00   29.00   21.70   14.00
-2 + 0i       cut, z<-1       0.11    0.11    0.11    0.11
-5 + 0i       cut, z<-1       0.00    0.00    0.00    0.00
-100 + 0i     cut, z<-1       0.00    0.00    0.00    0.00
-2 + 1e-6 i   just off cut    0.00    0.00    0.00    0.00
```

Caps are DD 31, QF 29, TF 21.7, FF 14 digits; a value at the cap is exact. Every
backend is exact-to-cap until `z < -1`, then all four collapse together.

The last row matters: `-2 + 1e-6i` is **off** the cut and still scores 0.00, so
this is not a branch-convention disagreement about the cut itself — the result is
wrong in a neighbourhood of the negative real axis.

### Extent

The four `tests/*_complex_accuracy_test.cpp` files added 2026-09-02 score `acosh`
over 2000 shared corpus elements. Partitioning those elements by half-plane
(mean digits, cap in parentheses):

```
region                             n     DD(31)  FF(14)  QF(29)  TF(21.7)
--------------------------------------------------------------------------
Re(z) >= 0                       957      25.36   11.35   24.85    18.17
Re(z) <  0, Im(z) != 0           540       0.00    0.00    0.00     0.00
Re(z) <  0, Im(z) == 0            74       0.84    0.59    0.84     0.84
all elements in window          1571      14.98    6.78   14.93    10.73
```

`Re(z) < 0 ∧ Im(z) != 0` is the interior of the left half-plane — nowhere near the
cut — and it scores a flat **0.00 in every backend**. The defect is the branch
selection, not the cut convention. Worked point: `z = -5 + 3i` returns
`(-2.456, -2.593)` where the correct principal value is `(+2.4529, +2.5943)` — both
components sign-flipped, which is exactly the wrong branch rather than lost digits.

This is also why KI-1's own table above shows `-0.5 + 0i` and `-1 + 0i` scoring at
cap: for `|z| <= 1` the wrong branch happens to coincide with the right one.

Because the region is large and the failure total, the tests **exclude it by domain
predicate** (`acosh_in_domain`, one per test file, each naming KI-1) rather than
loosening the tolerance. Deleting those four predicates is the acceptance test for
the fix.

### Why

The shared identity, at `include/xp/tf_complex.hpp:352` and cited there verbatim
from `qf_complex.hpp:369-370`, `ff_complex.hpp:291-293`, `dd_complex.hpp:286-289`:

```c++
acosh(z) = log(z + sqrt(z*z - 1))
```

**The defect is that `sqrt(z²−1)` selects the WRONG BRANCH for `Re(z) < 0`**, not
catastrophic cancellation. The distinction matters: a cancellation fix would leave
the bug fully intact.

The falsifying argument: cancellation loses digits that scale with `|z|` (roughly
0.87 digits at `z = -2`, rising to 4.3 at `z = -100`). The measured scores above
are **flat at 0.11, 0.00, 0.00 across a 50× magnitude range**. A conditioning
failure cannot be flat.

Direct measurement in `std::complex<double>` against `std::acosh` reference:

```
z = -2        true  +1.316957896924817 +3.141592653589793i
              repo  -1.316957896924816 +3.141592653589793i    0.11 digits
              Kahan +1.316957896924817 +3.141592653589793i   16.19 digits

z = -5        true  +2.292431669561178 ...
              repo  -2.292431669561173 ...                    0.00 digits
              Kahan +2.292431669561178 ...                   17.00 digits

z = -100      true  +5.298292365610485 ...
              repo  -5.298292365610980 ...                    0.00 digits
              Kahan +5.298292365610484 ...                   15.84 digits

z = -2+1e-6i  true  +1.316957896925009 +3.141592076239524i
              repo  -1.316957896925009 -3.141592076239524i    0.00 digits
              Kahan +1.316957896925009 +3.141592076239524i   17.00 digits

z = +2        repo and Kahan both 17.00 (control: right half-plane is fine)
```

**Three observations:**
1. The real part is NEGATED, with magnitude correct to ~15 digits. The negation is
   not bit-exact (`-1.316957896924816` vs `...817`): a small cancellation effect
   is present, but it is a rounding artifact (~0.1 digit), not the cause.
2. At `z = -2 + 1e-6i` (off the cut), the **imaginary part is also sign-flipped**.
   Both components are inverted — the signature of a branch error, not lost digits.
3. The current form squares `z` and therefore overflows. At `z = 1e150` the repo
   form returns finite but Kahan's form stays finite; at `z = 1e160` and beyond,
   repo returns `INF/NaN` while Kahan remains finite. For the FP32-word backends
   (FF, TF, QF), the squaring limit is ~1.8e19 against an FP32 range of ~3.4e38,
   so Kahan buys roughly 19 decades of usable domain as a side effect (derived from
   the double case; not measured at FP32 word width).

**The fix is Kahan's branch-correct form** (Kahan 1987, "Branch Cuts for Complex
Elementary Functions"):

```c++
acosh(z) = 2·log( sqrt((z+1)/2) + sqrt((z−1)/2) )
```

It is branch-correct with **no case analysis**: for `Re(z) < 0` both roots are
near-purely-imaginary with the same sign; for `Re(z) > 0` both are near-real
positive, so the addition never subtracts. Verified above: it scores 15.84–17.00
where the current form scores 0.00–0.11.

**Prediction that did NOT reproduce:** Kahan's form was predicted to degrade near
`z = 1` (forming `1 + u` with `u` small). Measured in double, Kahan is **better**
there: `z = 1 + 1e-4` repo 12.83, Kahan 13.69; `z = 1 + 1e-10` repo 10.56, Kahan
11.64. This was checked at double precision only and may still require attention at
DD's 31-digit scale, where a series branch
`acosh(1+w) = sqrt(2w)·(1 − w/12 + 3w²/160 − ...)` may be needed for tiny `w`.

**Sibling risk — no longer a risk, now measured: see KI-5.** This paragraph used
to list four unverified predictions about the rest of the `log`/`sqrt` inverse
family. All four were measured on 2026-09-02 and all four reproduce, in all four
backends. They are filed as **KI-5**, below. Two of the open questions asked here
are answered there:

- complex `acos` is **not** implemented via `acosh` — it is `π/2 − asin`
  (`dd_complex.hpp:241-246`), so it does not inherit KI-1's branch bug. It
  inherits a different defect instead (KI-5 (c)), and a second one from `asin`
  (KI-5 (d)).
- a fifth defect that was *not* predicted — `asin` on the real branch cut — was
  found by a control point that failed. It is KI-5 (d).

### Why it was not fixed when found

Found during S10 phase 4 (TF complex), 2026-09-01, while investigating what looked
like a TF-only accuracy outlier. It is not TF-only: TF inherited a defect present
in DD since the original port and propagated through FF and QF. Fixing it means
changing **shipped DD/FF/QF numerical output**, which:

- the byte-identical gate exists specifically to prevent during the restructure
  sub-plans, and
- is outside S10's declared additive scope ("TF is purely additive — if a demo's
  accuracy table moves, stop").

So it was recorded rather than fixed. Deferring was the correct call for the
sub-plan; leaving it deferred permanently is not.

### Why nothing caught it

**There was no complex accuracy test in the suite.** All 30 ctest targets covered
real arithmetic. `dd_accuracy_test`, `ff_accuracy_test`, `qf_accuracy_test` and
`tf_accuracy_test` score real ops only; the complex code was exercised only by the
demos, which report a table nobody diffs against an oracle per-op.

This was arguably the larger finding, and it is now closed: the four
`*_complex_accuracy_test` targets added 2026-09-02 score all 24 complex ops per
backend against the shared `__float128` corpus, taking the suite to 34 tests. A
test like this would have caught KI-1 on the day DD was written.

### To close

1. Fix the branch selection in all four `*_complex.hpp` headers.
2. ~~Add complex accuracy tests with a per-op tolerance table.~~ **Done 2026-09-02**
   (`tests/{dd,ff,qf,tf}_complex_accuracy_test.cpp`). They currently exclude
   `Re(z) < 0` for `acosh`; the fix must delete that predicate from all four and
   they must still pass.
3. Re-baseline the affected demo accuracy tables — the byte-identical gate WILL
   trip, and that is the intended signal, not a regression.

Probe used to produce the table above: `/tmp/acosh_all.cpp` (throwaway, not
committed; it links all four headers and scores each against `cacoshq`).

### Resolution (2026-09-02, commit `f421a0e`)

`log(z + sqrt(z*z - 1))` replaced by Kahan's form in all four `*_complex.hpp`
headers:

```c++
acosh(z) = 2*log( sqrt((z+1)/2) + sqrt((z-1)/2) )
```

Halving is done per component with `multiply_scalar(..., 0.5)` rather than by a
complex multiply by `(0.5, 0)`: 0.5 is a power of two, so the component form is
exact where the complex multiply would round. Measured, the two spellings score
identically to 0.01 digit at every probe point, so this is a cost choice (it saves
four multiplies and two adds per call), not an accuracy one.

Per-point scores against a `cacoshq` oracle, each backend at its own word width
(caps DD 31, QF 29, TF 21.7, FF 14):

```
point            DD before  after   QF before  after   TF before  after   FF before  after
------------------------------------------------------------------------------------------
-2   + 0i            0.00   30.53       0.00   28.10       0.00   20.69       0.00   14.00
-5   + 0i            0.00   30.98       0.00   28.05       0.00   21.12       0.00   14.00
-100 + 0i            0.00   31.00       0.00   29.00       0.00   21.51       0.00   14.00
-5   + 3i            0.00   30.38       0.00   27.37       0.00   21.67       0.00   13.49
-2   + 1e-6i         0.00   23.08       0.00   23.08       0.00   20.79       0.00   14.00
 0   - 1i            0.00   30.23       0.00   28.39       0.00   20.78       0.00   13.85
+2   + 0i           31.00   30.53      29.00   28.10      20.90   20.69      13.90   14.00
+1.5 + 0i           30.04   30.44      29.00   27.62      21.70   20.24      13.30   13.61
+5   + 3i           30.54   30.38      27.83   27.37      21.70   21.67      13.43   13.49
 0.5 + 0i           30.75   31.00      28.91   28.96      21.70   21.62      13.48   13.97
-0.5 + 0i           31.00   30.75      28.96   28.91      21.70   21.70      14.00   13.48
 0   + 1i           30.10   30.23      29.00   28.39      20.40   20.78      13.37   13.85
1e18 + 0i           31.00   31.00       0.00   29.00       0.00   21.70       0.00   14.00
1e150+ 0i            0.00   19.26       0.00    0.00       0.00    0.00       0.00    0.00
```

`0 - 1i` is worth noting: it is not on the negative real axis at all, and the old
form scored 0.00 there while `0 + 1i` scored 30.10. The wrong sheet reaches the
whole lower imaginary axis too, which the original filing did not record.

The `1e18` row is the overflow side benefit, confirmed at FP32 word width and not
merely derived from the double case: QF, TF and FF all go 0.00 -> cap because
Kahan's form never forms `z*z`. DD gains at `1e150` (0.00 -> 19.26). The claim in
the filing above that the FP32 backends gain "roughly 19 decades" is confirmed in
direction; the `1e150` row shows the FP32 backends still lose the *whole value* at
that magnitude for unrelated reasons (their own `exp`/`log` range), so the gain is
bounded by the word type, not unlimited.

**The near-`z = 1` concern did NOT materialise at DD scale.** The filing flagged it
as unverified above double precision. Measured at all four widths, the two forms
are bit-identical there — `1 + 1e-4` scores 13.26 for both in DD/QF/TF, `1 + 1e-10`
scores 7.38 for both, and `1 + 1e-20` scores 0.00 for both. Those numbers are
conditioning of `acosh` near its branch point, not a property of either
formulation, and they are backend-independent (DD, QF and TF all read 13.26/7.38),
which is the tell. **No series branch was added**, and none is warranted by
measurement. FF alone differs and Kahan is the better one there (`1 + 1e-4`:
11.50 -> 11.73).

The four `acosh_in_domain` predicates were deleted from
`tests/{dd,ff,qf,tf}_complex_accuracy_test.cpp` — the acceptance test named in
"To close" — so `acosh` is now scored over all 1571 in-window corpus elements
instead of the 957 with `Re(z) >= 0`. Corpus means over the *same* full element
set, before and after:

```
backend   acosh before   after     verdict before
DD            16.41      28.43     FAIL (tol 25.06)
QF            15.42      26.46     FAIL (tol 24.55)
TF            11.27      19.30     FAIL (tol 17.87)
FF             7.05      11.78     FAIL (tol 11.05)
```

Tolerance rows ratcheted to 28.13 / 26.16 / 19.00 / 11.48 (0.30 margin, the file's
convention). ctest 34/34.

### Accepted regression — complex `acosh` on the imaginary axis near the origin

The monotone gate reports 2022 decreased points against 4293 increased, and
**every decreased point is complex `acosh`** — `asin` and `acos` decrease nowhere
(see KI-5 (d)'s resolution). The decreases are accepted, not fixed. Per-cell:

```
cell           mean before -> after    zeros before -> after    dec    inc
DD c acosh        14.36  ->  25.79        888  ->  130          534    978
QF c acosh        12.45  ->  22.98        916  ->  164          581    958
TF c acosh         9.31  ->  17.03        932  ->  209          462    992
FF c acosh         5.81  ->  10.59        964  ->  279          445    963
```

Restricted to `Re(z) < 0` (852 grid points), the mean goes 1.78 -> 25.49 (DD),
1.65 -> 23.12 (QF), 1.25 -> 17.02 (TF), 0.80 -> 10.72 (FF).

**Where the losses are.** 1706 of the 2022 are under 1 digit. The 44 drops above 10
digits are all one cluster, in the grid's `cut-re` family at `re = 0` with `|im|`
between 1e-30 and 1e-22 — the imaginary axis just off the origin. Worst points
(`validation/sweep/sweep_grid.csv`, complex kind):

```
grid pt   re   im        DD before -> after     QF               TF
   766     0   1e-30       31.00 ->  1.23     29.00 -> (n/a)   21.70 -> (n/a)
   764     0   1e-29       31.00 ->  2.44
   762     0   1e-28       31.00 ->  3.50
   760     0   1e-27       31.00 ->  4.40     29.00 ->  3.56   21.70 ->  0.00
   758     0   1e-26       31.00 ->  5.31     29.00 ->  4.40
   752     0   1e-24       31.00 ->  8.33     29.00 ->  6.80   21.70 ->  0.37
```

Over that cluster (`re == 0`, `|im| < 1e-15`, 32 points) the mean goes 15.02 ->
8.81 (DD), 11.89 -> 7.42 (QF), 8.23 -> 2.31 (TF), 4.16 -> 0.44 (FF).

**Why.** For `z = i*eps`, `acosh(z) = eps + i*pi/2 + O(eps^3)`: the real part is
smaller than the imaginary part by up to 30 orders of magnitude, and the scoring
rule measures it *relatively*, so recovering it needs the whole `eps` to survive.
The old form's intermediate `z + sqrt(z^2 - 1)` is *purely imaginary* — exactly
`i*(eps + 1)` — so the complex `abs` reduces to `|im|`, `1 + eps` is stored
losslessly in the expansion's low word, and `log` returns `eps` intact. Kahan's
form splits the same quantity across two components of equal size (both roots land
near `0.7071*(1 + i)`), so `abs` must compute `sqrt(re^2 + im^2)`, which rounds at
absolute ~1e-32 in DD — one percent of an `eps` of 1e-30. That is the measured
1.23 digits.

This is a conditioning loss in one component of one op on one line of the plane,
and it is the price of the identity, not a bug in the new code: any formulation
that does not keep the intermediate purely imaginary pays it. It is **accepted**
because the thing bought is not comparable — an entire half-plane of O(1)
sign-flipped results becomes correct. A wrong sign over `Re(z) < 0` is worse than a
lost 30th digit of a real part that is 1e-30 next to a `pi/2`.

Closing it later, if wanted, means giving complex `acosh` a dedicated small-`|z|`
branch using `acosh(z) = i*acos(z)` for `|z| << 1`, or a complex `log1p`. Both are
larger than this fix and neither is needed for correctness. Not filed as a new KI
because it is recorded here with coordinates and is reproducible from
`validation/sweep/`.

The remaining 1978 sub-10-digit drops are spread over the `cut-re`, `cut-im`,
`polar` and `axis` families and are ordinary rounding-path differences: Kahan's
form takes two `sqrt`s and one `log` where the old took one `sqrt`, one `log` and a
squaring, so points that happened to round favourably under one form do not under
the other. Mean drop across all 2022 is 1.0-1.2 digits.

---

## KI-2 — `nint` is wrong for the one-ulp-below-a-tie class **[RESOLVED 650aa16]**

**Severity: low as filed — actually MEDIUM in TF, see the resolution. Scope:
inherited from QD.**

`0.49999997f + 0.5f` rounds to exactly `1.0f`, so TF's `nint`, formulated as
`floor(x + 0.5)` per QD, returns 1 where the nearest integer is 0. A genuine wrong
answer, not a precision shortfall. FP64 QD has the same defect.

Found in S10 phase 3.5 and excluded by domain predicate rather than absorbed into a
tolerance — see `docs/PORT_NOTES_TF.md` §12f. Closing it means diverging from QD's
`nint` formulation, which the source-fidelity rule discourages without cause.

### Resolution (2026-09-02, commit `TBD-FIX`)

`floor(d + 0.5)` replaced by `detail::rint(d)` plus a tie-direction restore, in
`qf_math.hpp` (`qf_nint`) and `tf_math.hpp` (`tf_nint`, new). `rint` does the same
rounding in one instruction with no intermediate to double-round, so the hazard
cannot arise; it is one hardware instruction on every target (`roundss`, `frintn`,
`cvt.rni`, `v_rndne`), and it stays correct under a non-default rounding mode,
where `floor(x + 0.5)` silently does not. `detail::rint` was added to
`config.hpp`'s dispatch.

`rint` is ties-to-EVEN and QD's form is ties-toward-+INFINITY (toward +inf, not
away from zero: `floor(-2.5 + 0.5) = -2`). The tie DIRECTION is deliberately
preserved by the `d - r == 0.5f` line — it is user-visible through `round`, and
the multi-word tie corrections assume the leading word rounded up. KI-2 is the
near-tie *wrong answer*; the tie direction is a convention and changing it is not
part of closing KI-2. A zero-sign line likewise restores `+0.0` on `(-0.5, 0)`,
so the change is provably minimal.

**DD and FF were audited and are NOT affected.** Both use magic-constant forms
(`2^105 + 2^52` at DD, `2^52` in FP64 at FF), never `floor(x + 0.5)`, and both are
exact and ties-to-even over their whole supported range. FF's was nevertheless
swapped to `rint` to drop the magic constant and gain rounding-mode correctness;
that swap is bit-identical to the old form, *including* the zero-sign line, which
the audit below showed is load-bearing. (DD and FF are ties-to-even; QF and TF are
ties-toward-+infinity. The two families have always disagreed on ties.)

**TF's exposure was far wider than the filed one-input description — this is the
substantive finding.** `qf_nint` short-circuited integers (`if (d == floor(d))
return d;`) before reaching the floor form, so QF's only wrong input really was
`0.49999997f`. TF's `round_to_nearest_int` called `floor(a.fN + 0.5f)` **bare**,
with no such guard. For an odd integer `d` in `[2^23, 2^24)`, where ulp is exactly
1, `d + 0.5f` is a perfect tie that round-half-to-even resolves *upward* to
`d + 1`, so `floor` returned `d + 1` as the nearest integer **of an integer**.
That is 2^22 wrong inputs per limb, on every limb of every TF reduction — not one
input. The fix repairs all of them.

**Surviving `floor(x + 0.5f)`: exactly one, `tf_math.hpp`'s `exp` range
reduction, deliberately not converted.** Its guards bound the argument to
magnitude < 116, nowhere near the `[2^23, 2^24)` band, and a near-tie off-by-one
in a reduction quotient is harmless rather than wrong — `r` shifts by `ln2`, gets
divided by `2^nq`, stays inside the convergence radius, and the same `m` scales
back. Converting it would perturb a large input set to buy nothing.

**Accepted regressions.** The monotone gate records six points that decreased.
All are collateral of the *correct* new rounding and all are documented trades,
not silent losses:

| point | op | coordinates | before → after |
|---|---|---|---|
| `r/893`  | TF `remainder` | `-97.389372261283611` (−31π − 2 ulp) | 0.33 → 0.00 |
| `c/374`  | TF `polar`     | `-3.8268e7 - 9.2388e7i` | 14.83 → 14.57 |
| `c/375`  | TF `polar`     | `-1.9509e7 - 9.8079e7i` | 14.72 → 14.53 |
| `c/377`  | TF `polar`     | `+1.9509e7 - 9.8079e7i` | 14.72 → 14.53 |
| `c/378`  | TF `polar`     | `+3.8268e7 - 9.2388e7i` | 14.83 → 14.57 |
| `c/396`  | QF `pow`       | `-100 + 1e-5i` | 28.42 → 27.77 |

`r/893` is `a = -97.389372261283611` (−31π − 2 ulp), `b = 3.5976482930064347e-21`.
It is the point that *found* the odd-integer class: the third limb of the TF
quotient is `15430381`, an odd integer in `[2^23, 2^24)`, so the old code returned
`15430382` and the new one correctly returns `15430381`. The score dropped anyway
because **`remainder` is meaningless at this operand pair**: `|a/b| = 2.7e22`,
where TF's ulp is `11.46`, so `nint(a/b)` is undetermined by ~11 integer units and
the answer carries an uncertainty of `11·b ≈ 4e-20` against a true remainder of
`1.16e-21` — 34× the answer itself. Both 0.33 and 0.00 are noise; the old value
merely landed closer by luck. Verified directly by evaluating both formulations
side by side at that point.

The four `polar` points are large arguments (|z| ~ 1e8) whose `sincos` reduction
quotient reaches the same limb band; `c/396` is complex `pow`, which is
`exp(b·log(a))` and so goes through the same reduction. Note `c/377` has
**positive** real part and `pow`/`polar` never call `asinh` — that is what rules
out any KI-5(a) involvement and pins these five on the rounding change.
Sub-0.7-digit motion well inside the digit budget, on the correct-rounding side.
**Accepted.**

Net across the whole sweep: 884 points up, 31 down (the 6 here plus the 25 under
KI-5(a)), 427,677 unchanged.

---

## KI-3 — `scripts/build_with_kokkos.sh` cannot build Kokkos 5.1 **[RESOLVED dd6d00a]**

**Severity: low (documented workaround), scope: pre-existing.**

Lines 67 and 74 pass `-DCMAKE_CXX_STANDARD=17` when configuring **Kokkos itself**.
Kokkos 5.1.0 rejects this at `cmake/kokkos_test_cxx_std.cmake:89`: *"Kokkos requires
C++20 or newer but requested 17"*. The consuming project correctly stays at C++17.

Found in S1. `validation/a100/build_login.sh` was corrected to 20 for its own Kokkos
configure, and `CLAUDE.md` documents the trap, but the script itself was left alone
as out of scope. It remains a live trap for anyone following the documented build
path.

**Resolution:** Changed lines 67 and 74 from `-DCMAKE_CXX_STANDARD=17` to
`-DCMAKE_CXX_STANDARD=20` in the script. Line 103 correctly remains at 17 (it
configures this repo, not Kokkos). Fixed 2026-09-02, commit TBD.

---

## KI-4 — DD `sin` returns the WRONG SIGN near odd multiples of π **[RESOLVED 8135332]**

**Severity: high (silently wrong answers, sign flip), scope: DD only, pre-existing.**

### What

`xp::sin` on DoubleDouble returns a result of the correct magnitude but the wrong
sign at some arguments near `k·π`. Measured at the FP64 value nearest `k·π`
(so the true sine is ~1e-16, not zero), scored against `sinq`, cap in parentheses:

```
 k     DD(31)   QF(29)   TF(21.7)   sin(fl64(k*pi))     DD returns
--------------------------------------------------------------------
 1      14.87    12.85     6.34      +1.224647e-16     +1.224647e-16
 2      17.36    14.86     6.76      -2.449294e-16     -2.449294e-16
 3       0.00    13.28     6.60      +3.673940e-16     -3.673940e-16   <-- sign
 4      17.36    14.86     6.76      -4.898587e-16     -4.898587e-16
 5      16.04    13.72     7.45      +6.123234e-16     +6.123234e-16
 6      16.12    14.86     6.76      -7.347881e-16     -7.347881e-16
 7       0.00    13.51     6.72      +8.572528e-16     -8.572528e-16   <-- sign
 8      17.36    14.86     6.76      -9.797174e-16     -9.797174e-16
 9      15.66    14.55     6.74      +1.102182e-15     +1.102182e-15
10      16.04    14.86     6.76      -1.224647e-15     -1.224647e-15
11      16.52    14.44     7.34      +4.899825e-15     +4.899825e-15
12      16.12    14.86     6.76      -1.469576e-15     -1.469576e-15
```

Every backend loses roughly half its digits here — that part is ordinary
argument-reduction conditioning against a finite-precision π and is expected. The
DD rows at `k = 3` and `k = 7` are a different thing: the magnitude is right to
~16 digits and the **sign is inverted**.

### Why

`include/xp/dd_math.hpp:454-498`. DD's `sincos` computes only the cosine through
the double-angle recurrence and then recovers the sine at line 493-496:

```c++
s4  = multiply(s0, s0);                      // cos^2
s5  = subtract(DoubleDouble(1.0), s4);       // 1 - cos^2
s1t = sqrt(s5);                              // |sin|, non-negative
if (is < 0) { s1t.hi = -s1t.hi; s1t.lo = -s1t.lo; }
```

`sqrt` is non-negative, so the sign comes from `is = sign(s3)` (line 467), the sign
of the reduced argument. **That identity holds only for `|s3| < π`.** The reduction
is `s3 = a − 2π·nint(a/2π)`, and when `a/2π` is a near-tie at a half-integer the
rounding can land `|s3|` marginally *above* π, at which point `sin(s3)` and `s3`
have opposite signs. Instrumented:

```
 k=1  a/2pi = 0.5  nint=0  s3 = +3.1415926535897931  (|s3| < pi)  ok
 k=3  a/2pi = 1.5  nint=2  s3 = -3.1415926535897936  (|s3| > pi)  SIGN FLIP
 k=5  a/2pi = 2.5  nint=2  s3 = +3.1415926535897927  (|s3| < pi)  ok
 k=7  a/2pi = 3.5  nint=4  s3 = -3.1415926535897940  (|s3| > pi)  SIGN FLIP
```

FF, QF and TF are structurally immune: all three carry **both** the sine and the
cosine series through the joint doubling (`sin(2x) = 2·sin x·cos x`,
`cos(2x) = cos²x − sin²x`) and never reconstruct a sign. They show conditioning
loss at these points and nothing worse. DD is the only backend that reconstructs.

Closing it means giving DD the same joint-doubling formulation the other three use,
or deriving the sign from the quadrant rather than from `sign(s3)`.

### Why it was not fixed when found

Found 2026-09-02 while setting the complex `sin`/`exp`/`sinh` tolerances — those
rows have a `min` of 0.00 in `dd_complex_accuracy_test`, which is this. Fixing it
changes shipped DD output for `sin`, `cos`, `tan` and every complex function built
on them, so it is the same re-baselining decision as KI-1 and belongs with it.
The four complex tests gate on the **mean**, and the means pass comfortably, so no
domain exclusion was needed — the defect is reported, not swallowed.

### Why nothing caught it

`dd_accuracy_test` scores real `sin`/`cos`, but its corpus never lands on a
half-integer `a/2π` tie, so the sign flip needs a targeted input. Related in
flavour to KI-2: both are tie-breaking in a rounding step producing a wrong answer
rather than a precision shortfall.

Probes used: `/tmp/rb_sinpi.cpp`, `/tmp/rb_dds.cpp`, `/tmp/rb_dds2.cpp` (throwaway,
not committed).

### Resolution (2026-09-02, commit `8135332`)

**First, the scope as filed was too small.** KI-2 (`nint` mis-rounding, commit
`650aa16`) was the proximate trigger, and the entry above was written before that
landed, so the state was re-measured first. KI-2 did **not** fix this. Scoring DD
`sin` at `fl64(k·π)` for odd `k ≤ 101` plus ±2 ulp either side — 255 points against
a `sinq` oracle — found **25 sign flips**, not two:

```
k with a flipped sign at the k*pi point itself (off = 0):
  3  7 13 17 21 25 29 43 51 57 59 65 67 73 75 81 89 91 93 95
k with a flipped sign one ulp off:
  1(+1) 5(+1) 9(+1) 35(-1) 39(-1)
```

The table in the original entry only ran to `k = 12`, which is why it saw only
`k = 3, 7`. Every one of the 25 scored 0.00 digits; all other 230 points scored
14.87–19.44. After the fix: **0 sign flips, 0 points below 10 digits**, min 14.96
and max 19.44 over the same 255 points.

**The fix is structural, not a sign repair.** `include/xp/dd_math.hpp` `sincos`
now carries **both** series through the doubling, matching `ff_math.hpp`,
`qf_math.hpp` and `tf_math.hpp` line for line:

```
was:  cos-only,  c -> 2c^2 - 1,  then sin = sign(s3) * sqrt(1 - cos^2)
now:  (s,c) -> (2sc, c^2 - s^2), sin never reconstructed
```

Repairing only the argument reduction was rejected deliberately. Even with
`|s3| < π` enforced exactly, `sin = ±sqrt(1 − cos²)` amplifies the *relative* error
of `cos` by `cot²(s3)`, which diverges as `s3 → ±π`. At the originally reported
failing point `s3 ≈ 3.674e-16`, so `cot² ≈ 7.4e30` — about 30.9 digits of
amplification against DD's 31-digit cap. That would have bought the right sign
while leaving the magnitude as noise: a loud bug traded for a quiet one.

All four backends now share one algorithm shape.

**A second, latent defect was found and fixed in the same routine.** The joint form
needs a *relative* convergence test on the sine term (`|sterm| < eps·|sin_r|`), as
the siblings use. For subnormal `|a|`, the `r = s3/2^nq` scaling underflows `r` to
exactly zero, that test becomes vacuous (`0 < eps·0` is false), the series runs to
`itrmx`, and the bail-out `return` **left `x` and `y` unassigned** — uninitialized
output, not merely an inaccurate one. Caught by the monotone gate as DD real `cos`
dropping 31.00 → 0.00 at 416 grid points (worst: `r/567`, `a = 9.88e-324`).
Fixed by answering the degenerate case directly (`r == 0 ⟹ cos = 1, sin = a`) and
by turning the iteration-limit `return` into a `break` so `x`/`y` are always
written. **`ff`/`qf`/`tf` have the same unassigned-`return` shape on their
iteration-limit path**; their sweep grids do not reach it, and they were left
untouched per this task's scope. Worth a follow-up.

**Call sites.** Every DD trig entry point funnels through this one `sincos`, so the
change reaches all of them without further edits: real `sin`, `cos`, `tan`;
`angle()` (the atan2 primitive, which Newton-iterates on `sincos`) and therefore
`asin`, `acos`, `atan`, `atan2`; and in `dd_complex.hpp` the seven call sites at
lines 191/213/220/265/272/281/369 — complex `exp`, `sin`, `cos`, `tan`, `sinh`,
`cosh`, `tanh`, `polar` and `pow`. `dd_complex.hpp` itself needed no change.

**Measured effect.** Sweep, 428,592 points: **20,957 up, 3,473 down, 404,162
unchanged.** Every affected cell improves in both mean and 1st percentile:

```
cell            mean before -> after     min before -> after
DD r sin           19.605 -> 21.380         0.00 -> 0.00
DD r cos           29.704 -> 29.749         0.00 -> 0.00
DD r tan           19.435 -> 21.346         0.00 -> 0.81
DD r asin          25.999 -> 30.897         0.00 -> 30.20
DD r acos          30.581 -> 30.995        16.33 -> 30.30
DD r atan          29.738 -> 30.984         0.00 -> 30.17
DD c exp           24.163 -> 29.242         0.00 -> 0.00
DD c tan           23.740 -> 27.158         0.00 -> 0.00
DD c pow           26.491 -> 28.210         0.00 -> 0.00
```

The inverse-trig rows are the largest single gain: `angle()`'s Newton correction
divides by `sin_a`/`cos_a`, so a sign-inverted sine there was diverging the
iteration outright.

**Accepted regressions.** All 3,473 decreases are in DD trig or in functions built
on it, and every one is ≤ 2.98 digits. They are the ordinary ±ulp reshuffling of a
different summation order, not a loss of a correct digit — the cells they sit in
improve on both mean and 1st percentile (table above), and the count is dwarfed
6:1 by the increases. The largest is worth naming:

| point | op | coordinates | before → after |
|---|---|---|---|
| `r/1575` | DD `sin` | `-311.01767270538954` (−99π) | 19.44 → 16.46 |
| `r/1575` | DD `tan` | same | 19.44 → 16.46 |
| `c/1658` | DD `pow` | see grid | 30.91 → 29.84 |
| `r/460`  | DD `cos` | see grid | 31.00 → 30.20 |

`r/1575` is `−99π`, and the old 19.44 was a **fluke, not a capability**: its
neighbours under the old code scored 16.62 (`k = 97`) and 16.29 (`k = 101`), and
the new code returns 16.46 there — the value consistent with the ~16-digit
conditioning floor that argument reduction against a finite-precision π imposes at
`k·π`. The old algorithm happened to land closer at this one point while being
outright sign-wrong at `k = 93` and `k = 95` on either side of it. Trading a lucky
3 digits at one point for 25 corrected signs is the intended bargain. **Accepted.**

**Cost.** Measured, not estimated: 200,000 calls over `[-1000, 1000]`, `-O2`, GCC
13.3.0, three reps, before/after built from the same source otherwise —
**894.4 → 1009.3 ns/call, +12.9%**. Inside the 10–25% Reet pre-approved.

**Tolerance ratchet** (`tests/test_utils.hpp`, conditioning registry): `sin` and
`cos` floors raised `0.0 → 3.0`, `acos` `0.0 → 5.0`. The table is shared by the
DD, FF and QF accuracy tests (TF does not use it), so the floors are bounded by
the worst backend: sin min is DD 15.42 / FF 6.06 / QF 21.73, cos DD 15.24 / FF 6.02
/ QF 21.87, acos DD 29.96 / FF 10.08 / QF 25.91. `tan` and `asin` stay at 0.0 —
QF's min on those is genuinely −0.00 / 0.00. Note the honest limit of this
ratchet: `dd_accuracy_test`'s corpus still never lands on a half-integer `a/2π`
tie, so a 3.0 floor would not by itself have *caught* KI-4; it removes the
sanction for a total loss, and the sweep grid is what actually covers these points.

**Gate.** Monotone gate exits 1 on the 3,473 accepted decreases above; ctest 34/34.
A session-hygiene note for whoever runs it next: loading the `cmake/3.28.3` module
alongside `gcc/13.3.0` links a *different* libquadmath and changes the oracle
fingerprint (`578322f998a329c8` → `54901e8104607a77`), which the gate correctly
flags but which buries the real diff in reference noise. Build the sweep with
`gcc/13.3.0` only.

---

## KI-5 — Four more defects in the complex inverse-function family, in ALL FOUR backends **[RESOLVED 650aa16 / 3968a9f / f421a0e]**

> **All four are RESOLVED.** (a) and (d) 2026-09-02 commit `f421a0e`;
> (b) and (c) 2026-09-02, commit `3968a9f` — see the two resolution blocks below.
> (c) is resolved with a **documented accepted regression**; read that block before
> trusting `acos` at `|z| ~ 1e8` with `Im(z) < 0`.

**Severity: high ((a) and (d) reach total loss of the result), scope: all four
backends (DD, FF, QF, TF), pre-existing, same `log`/`sqrt` family as KI-1.**

### What

KI-1 (complex `acosh` on the wrong `sqrt` sheet for `Re(z) < 0`) is not isolated.
The complex inverse functions all share two building blocks — a `sqrt` of a
quantity that changes sheet, and a `log` of a quantity that approaches 1 — and
four more members of the family are now measured to be defective:

| | op | failure mode | worst measured |
|---|---|---|---|
| (a) | `asinh` | cancellation for `Re(z) << 0` | total loss |
| (b) | `atanh` | loss of the `log` argument's leading 1 as `z → 0` | ~3 digits at FF |
| (c) | `acos`  | cancellation in `π/2 − asin(z)` as `z → 1` | ~7 digits off cap |
| (d) | `asin`  | **wrong branch on the real cut** `\|Re(z)\| > 1, Im(z) = ±0` | total loss |

**Provenance, stated plainly because it affects how much to trust each row.**
(a), (b) and (c) were *predicted* by the shared Claude planning thread as
consequences of the formulations in the headers, and then confirmed by
measurement — the prediction came first. (d) was **not** predicted. It was
written into the probe as a positive control for (d)'s mirror image, on the
assumption that `asin(+2)` would be fine, and the control failed. It is the
sharpest defect of the four.

### How this was measured

Two independent layers, both on 2026-09-02, GCC 13.3.0:

1. **Plain `double` transcription.** The repo's formulations, copied verbatim out
   of the `*_complex.hpp` headers into `std::complex<double>`, scored against
   libstdc++ `std::asinh` / `std::atanh` / `std::acos` / `std::asin`. This
   isolates the *formulation* from any backend's own arithmetic.
2. **The shipped backends.** The same points evaluated through
   `xp::{DoubleDouble,FloatFloat,QuadFloat,TripleFloat}Complex` and scored
   against the `__complex128` oracle (`casinhq`, `catanhq`, `cacosq`, `casinq`),
   using the same componentwise `min(d_re, d_im)` convention as
   `tests/*_complex_accuracy_test.cpp` — including its rule that a component
   whose reference is exactly zero is scored *absolutely* against the magnitude
   of the other component. That rule matters: without it, every purely-real
   reference in the `acos` table below reads a spurious `0.00`.

Layer 2 is the authoritative one. Layer 1 is reported alongside it because a
defect that shows up in both is a property of the identity, not of a backend.

Caps: DD 31, QF 29, TF 21.7, FF 14 digits. A value at the cap is exact.

---

### (a) `asinh(z) = log(z + sqrt(z² + 1))` collapses for `Re(z) << 0`

`include/xp/dd_complex.hpp:290-293`, and the same line in the FF, QF and TF
headers.

Plain `double` transcription, against `std::asinh`:

```
z            got                        ref                        digits
-1e2         -5.298342365610802         -5.2983423656105888         13.40
-1e4         -9.903487541418464         -9.9034875550361274          8.86
-1e6        -14.508650124059839        -14.508657738524469           6.28
-1e8        -inf                       -19.113827924512311           0.00   <-- total loss
+1e6        +14.508657738524469        +14.508657738524469          17.00   <-- control, fine
```

The shipped backends (the DD ramp is slower than `double`'s because DD carries
106 bits into the cancellation, but it ends in the same place):

```
point            DD(31)   QF(29)   TF(21.7)   FF(14)
------------------------------------------------------
-1e2 + 0i         29.41    28.71     19.17     10.99
-1e4 + 0i         24.49    24.49     16.19      8.64
-1e6 + 0i         21.29    25.04     13.60      9.76
-1e8 + 0i         17.62    17.88     16.83      0.00
-1e10 + 0i        14.48     0.00      0.00      0.00
-1e12 + 0i         9.28     0.00      0.00      0.00
-1e15 + 0i         2.78     0.00      0.00      0.00
-1e2 + 1i         26.37    25.98     18.38     10.52   <-- off-axis, same defect
+1e6 + 0i         31.00    28.95     21.70     14.00   <-- control, at cap
```

**Cause — conditioning, NOT a branch error.** For `Re(z) < 0`, `sqrt(z² + 1) ≈ −z`,
so `z + sqrt(z² + 1)` is a difference of near-equal quantities and the leading
digits cancel. The loss scales with `|z|` — roughly two digits per decade, which
is exactly what the tables show, and is the *opposite* of KI-1's signature (KI-1
is flat across a 50× magnitude range, which is why it cannot be conditioning).
Once the cancellation consumes the whole word the argument of `log` reaches zero
and the result is `-inf`.

The last row of the backend table matters: `-1e2 + 1i` is off the real axis and
degrades identically, so this is a half-plane property, not a cut property.

**Remedy — exact, and one line.** `asinh` is an ODD function, so for `Re(z) < 0`
compute `-asinh(-z)`, which moves the evaluation into the well-conditioned right
half-plane. No new series and no new constants. *(This paragraph originally read
"free … no accuracy cost anywhere else". That was wrong, and the monotone gate
caught it — see the accepted regressions in the resolution below.)*

### (a) Resolution (2026-09-02, commit `TBD-FIX`)

Implemented in all four `*_complex.hpp` headers. The predicate is **not** the
plain `Re(z) < 0` the remedy proposed; it is gated on magnitude as well as sign:

```cpp
if (z.re.hi < 0 && (-z.re.hi > 4 || fabs(z.im.hi) > 4))   // kXpAsinhReflect = 4
    return -asinh(-z);
```

The loss the reflection removes is `log10(2|z|²)` digits; the reflection also
substitutes a different rounding path, worth a few tenths either way. Below
`|z| ~ 4` the loss removed is smaller than the churn introduced. Measured on the
1780-point complex grid: reflecting **unconditionally** moved 535 points down and
396 up in the `|z| <= 1` bin (worst `-6.77`, QF at `z = -1e-16`, where
`asinh(z) ≈ z` and there was never any cancellation to fix), while `|z| > 4` was
859 up against 25 down (best `+28.22`, DD at `z = -1e15`). Gating at 4 keeps all
859 improvements and removes 944 of the 969 regressions.

The test is L-infinity on the leading limbs — `max(-Re, |Im|) > 4` — not
`hypot()`: it cannot overflow at `z ~ 1e200`, costs two compares, and 4 is exactly
representable so the comparison is exact. The sign predicate is `Re(z) < 0`, so
`Re(z) == ±0` both keep the direct branch: there is no cancellation on the
imaginary axis, and `asinh`'s cuts live there, where the sign of a zero real part
selects the sheet — routing `-0` through a negation would rewrite that selection.
(The on-cut handling for `Re(z) == -0` is separately wrong; that is the `asinh`
analogue of KI-5 (d) and is **not** addressed here.)

**Effect — the defect is closed.** Digits against `__complex128`, after:

```
z         DD(31)   FF(14)   QF(29)   TF(21.7)     (before, double proxy)
-1e2       31.00    14.00    28.40     21.47          13.40
-1e4       31.00    14.00    29.00     21.50           8.86
-1e6       31.00    14.00    28.95     21.70           6.28
-1e8       31.00    14.00    29.00     21.70           0.00   <-- was total loss
+1e6       31.00    14.00    28.95     21.70          17.00   <-- control, unchanged
```

Every backend at or near cap, including the points that previously returned
`-inf`.

**Accepted regressions — 25 points, and they were NOT placeable away.** The
working hypothesis was that they sit near the `Re(z) = 0` branch boundary and
that moving the boundary would shed them. The grid coordinates refute it: every
one is far inside the reflected region, and they cluster on the structural
families the grid samples deliberately.

| backend | points | coordinates | worst |
|---|---|---|---|
| DD | `c/448,449,1594,1595` | `-10 ± 0i` (`cut-re`, `axis`) | 31.00 → 30.53 |
| DD | `c/299,309` | `-5.5557 ± 8.3147i` (`polar`, \|z\| = 10) | 30.62 → 30.17 |
| DD | `c/368` | `-1e8 + 1.2246e-8i` | 17.09 → 16.45 |
| DD | `c/1091,1121,1411,1441` | `-1, -1e-15 ± 10i` (`cut-im`) | 30.39 → 30.23 |
| QF | `c/384,385,1598,1599` | `-100 ± 0i` | 28.71 → 28.40 |
| QF | `c/492,493,504,505` | `-10 ± 1e-21i, -10 ± 1e-27i` | 22.24 → 21.77 |
| QF | `c/1093,1097,1413,1417` | `-0.1, -0.001 ± 10i` (`cut-im`) | 28.63 → 28.46 |
| TF | `c/446,447` | `-100 ± 1e-30i` | 13.00 → 12.34 |

Worst drop across all 25: **0.66 digits**. Three observations settle the
placement question:

1. **They are not boundary artifacts.** `|z|` ranges from 10 to 1e8 — one to seven
   decades past `kXpAsinhReflect`. Moving the threshold cannot reach them without
   also un-reflecting the region where the fix delivers +14 to +28 digits.
2. **They come in exact ± pairs** (`448/449`, `384/385`, `446/447`, `1594/1595`)
   straddling `Im = ±0`. That is the signature of an extra exact negation
   changing which side of a rounding boundary the `log` argument lands on — a
   half-ulp coin flip, not a conditioning failure. It is intrinsic to evaluating
   `-asinh(-z)` instead of `asinh(z)`: the two expressions are mathematically
   identical and cannot be made bit-identical.
3. **The trade is not close.** 0.66 digits lost at 25 points, against 859 points
   improved by up to +28.22 and four previously-dead points restored from 0.00 to
   cap. Losing half a digit at 30 costs nothing anyone can use; recovering a
   result that was `-inf` is the whole point of the fix.

**Decision: accepted, not placed away.** Recorded here rather than absorbed by
regenerating the baseline, so the trade stays visible in the sweep diff.

---

### (b) `atanh(z) = ½·log((1+z)/(1−z))` degrades as `z → 0`

`include/xp/dd_complex.hpp:298-300` and siblings.

Plain `double` transcription, against `std::atanh`:

```
z            got                        ref                        digits
1e-2         0.010000333353334716       0.010000333353334763        14.33
1e-6         1.0000000000059673e-06     1.0000000000003333e-06      11.25
1e-10        1.0000000826403711e-10     1e-10                        7.08
1e-14        9.9920072216263079e-15     1e-14                        3.10
0.5          0.54930614433405489        0.54930614433405489         17.00   <-- control
```

The shipped backends:

```
point            DD(31)   QF(29)   TF(21.7)   FF(14)
------------------------------------------------------
1e-2 + 0i         28.94    27.63     19.00     12.18
1e-6 + 0i         24.52    22.12     15.56      9.40
1e-10 + 0i        20.97    22.06     11.51      7.87
1e-14 + 0i        17.55    22.12     15.10      7.76
1e-6 + 1e-6i      18.67    22.58     15.05      9.39   <-- off-axis, same defect
1e-18 + 0i        31.00    22.04     15.11      7.34   <-- see the tail note
1e-25 + 0i        31.00    29.00     15.09      7.71
0.5 + 0i          30.74    27.89     21.33     13.61   <-- control
```

**Cause — conditioning.** As `z → 0` the ratio `(1+z)/(1−z) → 1`, so every digit
of `z` that sits below the leading 1 is rounded away before `log` ever sees it.
The result then reconstructs `2z` from a `log` argument that only retains
`bits(word) − log2(1/|z|)` significant bits.

**Tail note, so the non-monotonic DD rows are not misread as noise.** Below some
`|z|` the *true* `atanh(z)` rounds to `z` itself (the `z³/3` term falls under the
backend's resolution: `|z| < sqrt(3·2^-106) ≈ 1.9e-16` for DD), so any
implementation returning something within an ulp of `z` scores at cap again. DD
recovers to 31.00 at `1e-18` for that reason, not because the defect stops. QF
does *not* recover where the same argument says it should (22.04 at `1e-18`,
against a threshold of `sqrt(3·2^-96) ≈ 6.2e-15`); that residual is QF-specific
and is not diagnosed here.

**Remedy.** The `log1p` form, which never forms the leading 1:

```
atanh(z) = ½·log1p( 2z / (1 − z) )
```

This requires a complex `log1p`; none of the four headers has one, so closing (b)
means adding it. That is the largest piece of work in KI-5.

---

### (b) Resolution (2026-09-02, commit `3968a9f`)

Closed in all four `*_complex.hpp`, and it took a new **real** `log1p` as well as
the new complex one. Three pieces.

**1. The real `log1p` was never a `log1p`.** All four headers spelled it
`log(add(1, a))`, the exact composition a `log1p` exists to avoid. Replacing that
was a prerequisite, not a side quest: a complex `log1p` built on it would have
inherited the same loss. Two attempts were needed.

  * *Goldberg's correction alone is not enough.* `u = fl(1+a); log(u)*a/(u-1)`
    repairs the rounding of the ARGUMENT and assumes `log(u)` is accurate for the
    `u` it is handed. It is not, here: `log` seeds from the FP64 log of the
    leading limb and takes ONE Newton step `x <- x + a*exp(-x) - 1`, which for
    `u = 1 + e` returns `e` itself against a true `e - e²/2`. Measured:
    `log1p(4e-20)` on DD scored **19.70**, not 31 — precisely the `2·log10(1/e)`
    that one Newton step buys. Recorded here because the result looks like a
    working fix until it is measured.
  * *What shipped* is the series, for `|a| < 1/4`:
    `log1p(a) = 2·atanh(t)`, `t = a/(2+a)`, `atanh(t) = t + t³/3 + t⁵/5 + …`.
    The atanh form rather than the alternating `log(1+a)` series because its terms
    are all positive and it converges in `t²`: `|a| < 1/4` gives `|t| < 1/7`, about
    19 terms for DD. Outside `|a| < 1/4` the body is the ORIGINAL `log(1+a)`,
    unchanged. That is deliberate — the first cut applied Goldberg there too and
    the monotone gate caught it costing 583 points up to `-0.87` digits (TF at
    `a = -1.85`), because with no cancellation to undo the correction is two extra
    roundings and nothing else.

**2. Complex `log1p`, new in all four headers** (`Re = ½·log1p(2·Re(w) + |w|²)`,
`Im = atan2(Im(w), 1 + Re(w))`). The formulation is Kahan 1987's. **Divergence
from the sources, recorded as instructed:** QD 2.3.24 (`/tmp/qdsrc/QD`) has no
complex layer at all, so it offers nothing to copy or diverge from; Kahan's
residual weakness — the real part loses relative accuracy on the circle
`|1+w| = 1`, where `2·Re(w) + |w|²` itself cancels — is inherited knowingly. That
locus is measure-zero, the answer on it is `~0`, and every hypot-based alternative
loses the same digits in the same place.

**3. `atanh` rebuilt on it**, expanded into components so no complex divide is
needed either:

```
Re atanh(z) = ¼·log1p( 4x / ((1-x)² + y²) )
Im atanh(z) = ½·atan2( 2y, 1 - x² - y² )
```

**THRESHOLD (`kXpAtanhSmall = 0.0625`, L-infinity on the leading limbs).** The new
form is used only where the old one loses. `(1-x)² + y²` squares its operands, so
it overflows for `|z|` above sqrt of the word range (`~1.3e154` FP64-word,
`~1.8e19` FP32-word) where the old ratio form is finite — switching
unconditionally would trade a conditioning defect for an overflow defect. The
first cut used `0.5`; the gate showed 43 points across the four backends losing up
to `-1.09` digits (QF at `z = -0.49 - 0.098i`) to rounding churn in `0.0625 < |z|
< 0.5`, where the old form was already at cap. Tightening to `0.0625` — exactly
representable, so the compare is exact — keeps every small-`z` win and leaves
**one** atanh regression in the whole sweep.

**4. A second defect, found by measurement while fixing the first: the signed zero
on the cuts** — the `atanh` analogue of (d). Approaching `x > 1` from `Im = +0`
gives `Im atanh = +π/2` and from `Im = -0` gives `-π/2`; by oddness the same
`+π/2` holds for `x < -1` with `Im = +0`. The old form's complex divide destroyed
the sign of `Im(z)`'s zero and returned `+π/2` for both conventions, so **every
`x -0i` cut point scored 0.00 on the imaginary component, in all four backends**.
The sign is now read off `Im(z)` with `detail::copysign` before the arithmetic can
lose it, as in (d).

Measured, `atanh` on the shipped backends (min over the two components,
`/tmp` harness against the `__complex128` oracle; caps DD 31 / QF 29 / TF 21.7 /
FF 14):

```
point                DD before -> after   QF before -> after   TF before -> after   FF before -> after
-------------------------------------------------------------------------------------------------------
1e-2  + 0i             28.94 -> 31.00       27.63 -> 29.00       19.00 -> 21.70       12.18 -> 14.00
1e-6  + 0i             24.52 -> 31.00       22.12 -> 29.00       15.56 -> 21.70        9.40 -> 14.00
1e-10 + 0i             20.97 -> 31.00       22.06 -> 29.00       11.51 -> 21.70        7.87 -> 14.00
1e-14 + 0i             17.55 -> 31.00       22.12 -> 29.00       15.10 -> 21.70        7.76 -> 14.00
1e-20 + 0i             31.00 -> 31.00       29.00 -> 29.00       15.35 -> 21.70        7.50 -> 14.00
1e-8  + 1e-8i          23.08 -> 31.00       22.50 -> 29.00       16.19 -> 21.70        8.22 -> 14.00
1e-16 + 3e-17i         16.61 -> 31.00       22.23 -> 29.00       15.31 -> 21.70        7.77 -> 14.00
0.5   + 0i  [control]  30.74 -> 30.74       27.89 -> 27.89       21.33 -> 21.33       13.61 -> 13.61
0.3   + 0.7i [control] 29.91 -> 29.91       27.01 -> 27.01       20.76 -> 20.76       12.82 -> 12.82
 2 + 0i  [cut]         30.74 -> 30.74       27.89 -> 27.89       21.33 -> 21.33       13.61 -> 13.61
 2 - 0i  [cut]          0.00 -> 30.74        0.00 -> 27.89        0.00 -> 21.33        0.00 -> 13.61
-2 + 0i  [cut]         29.93 -> 29.93       28.59 -> 28.59       21.30 -> 21.30       13.69 -> 13.69
-2 - 0i  [cut]          0.00 -> 29.93        0.00 -> 28.59        0.00 -> 21.30        0.00 -> 13.69
```

Every point in the conditioning region reaches the type's cap; the four `-0i` cut
rows go from total loss to the same score as their `+0i` partners. Corpus means
over the 1780-point complex grid: DD 25.74 -> 27.35, QF 22.34 -> 23.49,
TF 16.07 -> 17.14, FF 9.75 -> 10.59. Real `log1p` means over the 1652-point real
grid: DD 29.11 -> 30.76, QF 26.93 -> 28.37, TF 20.04 -> 21.44, FF 12.60 -> 13.86.

**Real-header additions, stated explicitly because the task scope asked.** The
complex `log1p` genuinely needed one, so `log1p` in all four of `dd_math.hpp`,
`ff_math.hpp`, `qf_math.hpp` and `tf_math.hpp` was rewritten as described in (1).
No other real-header function was touched, and no new constants were added.

**Residual regressions from (b): two points in 428,592.** `TF r log1p` point 278
`21.70 -> 21.47` and `QF c atanh` point 16 `20.58 -> 20.31`, both series-vs-`log`
rounding churn at the type's cap. Accepted.

---

### (c) `acos(z) = π/2 − asin(z)` degrades as `z → 1`

`include/xp/dd_complex.hpp:241-246` — confirmed by reading, the implementation
really is this form:

```c++
DoubleDouble pi_over_2 = multiply_scalar(DoubleDouble_pi(), 0.5);
DoubleDoubleComplex asin_z = asin(z);
return DoubleDoubleComplex(subtract(pi_over_2, asin_z.re), negate(asin_z.im));
```

Plain `double` transcription, against `std::acos`:

```
z            digits
1 - 1e-2     16.52
1 - 1e-6     11.26
1 - 1e-10    10.68
1 - 1e-14     9.36
0            17.00   <-- control
```

The shipped backends:

```
point            DD(31)   QF(29)   TF(21.7)   FF(14)
------------------------------------------------------
1 - 1e-2          30.06    28.64     21.56     14.00
1 - 1e-6          27.55    25.30     18.05     10.36
1 - 1e-10         25.95    24.66     17.28      9.31
1 - 1e-14         24.01    18.82     15.69      7.20
0 + 0i            31.00    29.00     21.70     14.00   <-- control, at cap
```

**Cause — conditioning.** As `z → 1`, `asin(z) → π/2`, so `π/2 − asin(z)` is a
difference of near-equal quantities. The true `acos(1−ε) ≈ sqrt(2ε)` is small
while both operands are `O(1)`, so the subtraction discards the leading digits.

**This answers an open question left in KI-1.** Complex `acos` does *not* route
through `acosh`, so it does not inherit KI-1's branch bug. It inherits this
cancellation instead — and, separately, defect (d) below, because it is built on
`asin`.

**Remedy.** Evaluate `acos` directly rather than as a difference, e.g. Kahan's
`acos(z) = 2·atan( sqrt((1−z)/(1+z)) )` form or the `acos(z) = -i·log(z + i·sqrt(1−z²))`
identity with the subtraction restructured. Either is a formulation change, not a
one-liner, and it must not regress the well-conditioned interior.

---

### (c) Resolution (2026-09-02, commit `3968a9f`) — **with an accepted regression**

Closed in all four `*_complex.hpp`. `π/2 − asin` is gone.

**Which formulation, and why.** Kahan 1987's log form,
`acos(z) = -2i·log( sqrt((1+z)/2) + i·sqrt((1-z)/2) )` — the exact companion of
the `acosh` adopted for KI-1, so the two share one verified branch layout. It was
chosen over Kahan's other form `2·atan(sqrt((1-z)/(1+z)))` because the atan form
adds a complex `atan` on top of a complex divide by `(1+z)`, which is singular at
`z = -1` — the OTHER end of the principal interval — so it moves the bad point
rather than removing it. The log form needs only `sqrt` and `log`, never forms
`z²` or `1/(1+z)`, and so has no overflow at large `|z|` and no singularity at
either end.

**It is NOT routed through `acosh`.** `acos(z) = ±i·acosh(z)` does hold, but the
sign flips with the half-plane AND with the side of each cut, so sharing the body
would reintroduce exactly the case analysis Kahan's form exists to avoid. The two
stay separate; they are a handful of lines each.

**But the pure log form is only half right, and the monotone gate is what said
so.** `Im(acos z) = -2·ln|w|` with `w = sqrt((1+z)/2) + i·sqrt((1-z)/2)`, and
`|w| → 1` exactly where `Im(acos z) → 0`, i.e. for `z` near the real segment
`[-1,1]`. There the log is taken of a number whose information sits below the
leading 1 — the same disease as (b), relocated. Measured on the full sweep: 4016 decreases across the four backends, worst
`31.00 -> 1.23` on DD at grid point 766, `z = 0 + 1e-30i`. A strictly worse
trade than the defect being fixed.

**What shipped is per-component**, and the split is exact rather than a
compromise: `Im(acos z) = -Im(asin z)` is an identity (`π/2` is real), so the
`π/2 − asin` cancellation was only ever in the REAL part. So

```
Re(acos z) = 2·arg( sqrt((1+z)/2) + i·sqrt((1-z)/2) )    <- Kahan, well conditioned
Im(acos z) = -Im(asin z)                                  <- exact identity, no cancellation
```

which takes each form's good component and leaves both bad ones. This also
inherits (d)'s cut fix on the imaginary part for free. Decreases fell from 4016 to 949.

**`sqrt_signed_cut`, new helper in each complex header.** The header's complex
`sqrt` tests `z.im.hi < 0.0`, which is FALSE for `-0.0`, so for `u < 0` it puts
both zero conventions on the `+i` sheet. `acos` needs the corrected sheet at all
four cut points, so the correction is applied locally. `multiply_scalar` does not
carry a signed zero through either (it renormalizes, and `quick_two_sum(-0,+0)` is
`+0`), so the caller passes the sign it read off the original `Im(z)` rather than
trusting the halved copy. **The underlying `sqrt` defect is NOT fixed** — that
would move every other caller, `acosh` included, in one undocumented step. It is
recorded here as a known defect in its own right and left for a separate change.

**Branch verification, both half-planes and both sides of both cuts**, each
against the `__complex128` oracle: `z=0 -> π/2`; `z=1 -> 0`; `z=-1 -> π`;
`z=2+0i -> -1.3170i`; `z=2-0i -> +1.3170i`; `z=-2+0i -> π-1.3170i`;
`z=-2-0i -> π+1.3170i`; and the four interior points `(±0.3, ±0.7)`, all at or
within 1.5 digits of cap.

Measured, `acos` on the shipped backends (min over the two components,
standalone harness against the `__complex128` oracle):

```
point                 DD before -> after   QF before -> after   TF before -> after   FF before -> after
-------------------------------------------------------------------------------------------------------
1 - 1e-2                30.37 -> 31.00       28.64 -> 29.00       21.56 -> 21.70       14.00 -> 13.57
1 - 1e-6                28.88 -> 30.74       25.30 -> 29.00       18.05 -> 21.70       10.36 -> 10.26
1 - 1e-10               27.11 -> 31.00       24.66 -> 29.00       17.28 -> 21.70        9.31 -> 14.00
1 - 1e-14               25.01 -> 31.00       18.82 -> 25.67       15.69 -> 21.70        7.20 -> 14.00
2 + 1e-20i              11.31 -> 30.05        8.76 -> 23.40        2.06 -> 21.43        0.00 -> 14.00
1e-30 + 0i              31.00 -> 30.20       29.00 -> 29.00       21.70 -> 21.70        0.00 -> 14.00
1 + 1e-10i              25.72 -> 25.72       24.20 -> 24.30       16.69 -> 17.64        8.86 ->  9.71
1 + 1e-20i              20.78 -> 20.78        6.38 ->  5.25        6.38 ->  5.25        4.05 ->  5.25
1e6 + 1e6i              20.14 -> 20.44       21.69 -> 21.69       13.77 -> 13.77        9.77 ->  9.77
0 [control]             31.00 -> 31.00       29.00 -> 29.00       21.70 -> 21.70       14.00 -> 14.00
 2 + 0i  [cut]          30.05 -> 30.05       28.32 -> 28.32       21.43 -> 21.43       14.00 -> 14.00
 2 - 0i  [cut]          31.00 -> 31.00       29.00 -> 29.00       20.90 -> 20.90       13.90 -> 13.90
-1 + 0i  [cut]          31.00 -> 31.00       29.00 -> 29.00       21.70 -> 21.70       14.00 -> 14.00
```

Read the table with the component split in mind; it explains the wins and the
flat rows alike. The `z -> 1` block is the defect this closes, and it is the
REAL part that moves: at `1 - 1e-14`, DD 25.01 -> 31.00, QF 18.82 -> 25.67,
TF 15.69 -> 21.70, FF 7.20 -> 14.00. `2 + 1e-20i` -- just off the cut, where the
old path also drove `asin` through `1 - z^2`, so the loss compounded -- is the
largest single gain: DD up 18.7 digits, FF from total loss to cap. The FF
`1e-30` row likewise goes 0.00 -> 14.00.

The rows that DO NOT move are those whose minimum sits on the IMAGINARY
component, which now comes from `-Im(asin z)` and is therefore exactly what
`asin` scored before. `1e6 + 1e6i` is 20.44 / 21.69 / 13.77 / 9.77 on both
sides, and the cut rows are unchanged to the digit. This change does not claim
to improve `acos` at large `|z|` off the axis -- `asin` bounds it there.

Four rows go DOWN and are listed rather than dropped. FF `1 - 1e-2`
14.00 -> 13.57, FF `1 - 1e-6` 10.36 -> 10.26 and DD `1e-30` 31.00 -> 30.20 are
sub-digit rounding churn at cap. QF and TF at `1 + 1e-20i` go 6.38 -> 5.25:
that point is a hair off the branch point `z = 1`, where `sqrt((1-z)/2)` is
evaluated at ~1e-20 and neither form has digits to keep -- both numbers are
noise around a near-zero answer, and DD, which has the range to resolve it, is
unchanged at 20.78.

Corpus means over the 1780-point complex sweep grid: DD 26.52 -> 27.98,
QF 23.46 -> 24.53, TF 16.90 -> 18.10, FF 9.91 -> 11.05. On the complex accuracy
test's own 1572-element `acos` corpus the DD mean moves 27.58 -> 27.63 and, more
to the point, the per-element MINIMUM rises 0.00 -> 16.01: after this change no
DD `acos` corpus element is completely wrong.

#### ACCEPTED REGRESSION — `acos` at `|z| ~ 1e8` with `Im(z) < 0`

The monotone gate exits **1**: 949 decreases against 4945 increases over 428,592
points, worst `-7.85`. 947 of the 949 are complex `acos`; the other two belong to
(b) and are listed there. The distribution is 584 under `0.5` digits, 216 in
`0.5–1`, 88 in `1–4`, 61 above `4`.

**The worst cluster is one ring and one mechanism.** Every drop past `-6` digits
is on the `|z| = 1e8` polar ring with `Im(z) < 0`. Coordinates of the worst six
(op `c acos`, point indices into `validation/sweep/sweep_grid.csv`):

```
TF pt 381  z = ( 8.3147e7, -5.5557e7)   21.70 -> 13.85   -7.85
TF pt 380  z = ( 7.0711e7, -7.0711e7)   21.70 -> 13.97   -7.73
TF pt 371  z = (-8.3147e7, -5.5557e7)   21.70 -> 14.20   -7.50
DD pt 382  z = ( 9.2388e7, -3.8268e7)   31.00 -> 23.72   -7.28
FF pt 381  z = ( 8.3147e7, -5.5557e7)   13.56 ->  6.29   -7.27
TF pt 383  z = ( 9.8079e7, -1.9509e7)   21.12 -> 13.88   -7.24
```

**Mechanism.** For `|z| >> 1`, `(1+z)/2 ≈ z/2` and `(1-z)/2 ≈ -z/2`, so with
`θ = arg(z)` the two roots satisfy `rm ≈ ∓i·rp` for `θ ≷ 0`. Then
`w = rp + i·rm` is `2·rp` for `θ > 0` — and a DIFFERENCE OF NEAR-EQUAL ROOTS for
`θ < 0`. `arg(w)`, and hence `Re(acos z)`, loses the digits that cancellation
eats: about `log10|z|/2 + O(1)`, which is the ~7 observed at `|z| = 1e8`.

**A fix was attempted and measured worse, twice, so it is not shipped.**
`acos(conj z) = conj(acos z)` holds identically including on both cuts, so
reflecting the cancelling half-plane should remove the loss. Reflecting `Im < 0`
took the gate from 949 decreases to **1915** (worst `-14.56`); reflecting `Im > 0`
gave **1648** (worst `-7.28`). Both are worse than no reflection, so the
cancellation above is not the whole story — the reflection also moves the `asin`
evaluation, and something in that path costs more than the cancellation saves.
Diagnosing that is a separate piece of work and is left open. Recorded so the next
session does not spend the same hour rediscovering that the obvious fix regresses.

**Why accept.** The trade is 4945 points up against 949 down; every corpus mean in
every backend rises; the ops the change targets go from partial-to-total loss to
cap; and the loss is confined to a region where the OLD form was at cap for the
wrong reason — `π/2 − asin` happened not to cancel there. It is the same shape of
decision as KI-1's, which accepted 2022 decreases on a smaller win. Do not
re-baseline to hide it: the baseline is regenerated in a separate commit and this
block is the record of what moved.

---

### (d) `asin` returns the WRONG BRANCH on the real cut — signed-zero collapse

`include/xp/dd_complex.hpp:231-240` and siblings:

```c++
// asin(z) = -i * log(iz + sqrt(1 - z^2))
DoubleDoubleComplex one_minus_z2 = DoubleDoubleComplex(DoubleDouble(1.0)) - z*z;
DoubleDoubleComplex sum = iz + sqrt(one_minus_z2);
```

The shipped backends, on and beside the cut `|Re(z)| > 1`:

```
point            DD(31)   QF(29)   TF(21.7)   FF(14)
------------------------------------------------------
-2 + 0i           30.05    28.32     21.43     14.00
-5 + 0i           30.19    29.00     20.90     13.14
-100 + 0i         28.38    27.08     18.88     11.00
+2 + 0i            0.00     0.00      0.00      0.00   <-- written as a control
+5 + 0i            0.00     0.00      0.00      0.00
+100 + 0i          0.00     0.00      0.00      0.00
-2 - 0i            0.00     0.00      0.00      0.00   <-- the mirror case
+2 - 0i           31.00    29.00     20.90     13.90
+2 + 1e-20i       30.05    28.32     21.43     14.00   <-- just OFF the cut: fine
+2 - 1e-20i       30.32    29.00     20.90     13.90   <-- just OFF the cut: fine
```

`acos` inherits it exactly, being `π/2 − asin`: `acos(+2 + 0i)` scores 0.00 in all
four backends while `acos(-2 + 0i)` scores 30.05 / 28.32 / 21.43 / 14.00.

**Cause — DIAGNOSED, and it is not what the asymmetry first suggests.** The
original report flagged this as "fails for `x > 1`, works for `x < -1`" and left
it undiagnosed. The measurement above shows the real rule, and it is symmetric:

> The implementation returns the **same sheet for both signed zeros**. Each of
> `Re(z) > 1` and `Re(z) < -1` therefore gets exactly one of its two signed-zero
> conventions wrong. The original probe used `Im = +0` throughout, which is the
> good side for `x < -1` and the bad side for `x > 1`.

The mechanism is a signed zero destroyed by an IEEE subtraction. For real `z` with
`Im(z) = ±0`, `imag(z·z) = 2·Re·Im` carries the sign `sign(Re)·sign(Im)`, and the
correct limiting sheet of `sqrt(1 − z²)` depends on it. But the subtraction throws
it away — in round-to-nearest, `0 − (+0)` and `0 − (−0)` both give `+0`:

```
z=(-2,+0)  imag(z*z)=-0  imag(1-z*z)=+0  sqrt=(+0,+1.73205)
z=(-2,-0)  imag(z*z)=+0  imag(1-z*z)=+0  sqrt=(+0,+1.73205)
z=(+2,+0)  imag(z*z)=+0  imag(1-z*z)=+0  sqrt=(+0,+1.73205)
z=(+2,-0)  imag(z*z)=-0  imag(1-z*z)=+0  sqrt=(+0,+1.73205)
```

All four collapse to the `+i·sqrt(z²−1)` sheet. Working the two cases through by
hand at `|z| = 2`, with `sqrt(1−z²) = +1.732i`:

```
x = -2:  iz = (0,-2)   iz+sqrt = (0,-0.268)  log = (-1.317,-pi/2)  -i*log = (-pi/2,+1.317)
x = +2:  iz = (0,+2)   iz+sqrt = (0,+3.732)  log = (+1.317,+pi/2)  -i*log = (+pi/2,-1.317)
```

The `Im = +0` convention (C99 Annex G, and what libstdc++ and libquadmath
implement) wants `asin(+2+0i) = +pi/2 + 1.317i`. The implementation returns
`-1.317i` — sign-flipped, hence 0.00 digits.

**Scope is narrower than KI-1, and this is the practical difference between them.**
KI-1's `acosh` is wrong in a whole *neighbourhood* of the negative real axis
(`-2 + 1e-6i` scores 0.00). KI-5 (d) is wrong **only exactly on the cut**: the two
`±1e-20i` rows above are correct to cap. So (d) is a signed-zero / branch-cut
*convention* defect, and KI-1 is a genuine wrong-region defect. They need different
fixes and different tests.

**Remedy.** Preserve the sign of the imaginary part across the `1 − z²` step
rather than recomputing it — form the real and imaginary parts of `1 − z²`
explicitly and give the imaginary part the sign `-sign(Re)·sign(Im)` (with `Im`'s
signed zero respected via `signbit`, not via a comparison against 0), or adopt
Kahan's `asin(z) = atan2(...)`-based formulation, which handles the cut without
depending on a signed zero surviving an arithmetic operation. The fix must be
checked at `Im = -0` as well as `Im = +0`; testing only `+0` is what let this
survive.

### Resolution of (d) (2026-09-02, commit `f421a0e`)

**First question answered: signed zero DOES round-trip in the expansion types.**
This decided whether an honest fix was possible at all, so it was measured before
anything was changed. Probing the raw leading word (`.hi` for DD/FF, `.f0` for
QF/TF) — *not* a `__float128` round-trip, which silently sums `-0.0 + 0.0` to `+0.0`
and reports a false negative:

```
                     DD   FF   QF   TF     correct?
ctor(-0.0) signbit    1    1    1    1     yes -- construction preserves it
copy signbit          1    1    1    1     yes
negate(+0.0)          1    1    1    1     yes
negate(-0.0)          0    0    0    0     yes  (that is +0)
Complex(2, -0.0).im   1    1    1    1     yes -- it reaches the callee intact
multiply_scalar(-0,1) 0    0    0    0     NO  -- lost in arithmetic
Im(z*z) for (2,-0)    0    0    0    0     NO  -- should be -0
Im(1 - z*z)           0    0    0    0     NO  -- the documented collapse
```

So the C99 Annex G convention **is** honourable in these types. The sign survives
construction and copy and is intact on entry to `asin`; it is destroyed only by the
arithmetic on the way to `sqrt`. The fix therefore reads it off `Im(z)` at the top
rather than trying to recover it downstream — which is what the filing above
recommended, and it works.

**What changed.** In all four `*_complex.hpp` headers, `asin` still computes
`root = sqrt(1 - z*z)`, but on the cut it then puts the root on the sheet the input
selects:

```c++
if (z.im.hi == 0.0 && detail::fabs(z.re.hi) > 1.0) {
    const bool want_neg = (detail::copysign(1.0, z.re.hi) ==
                           detail::copysign(1.0, z.im.hi));
    const bool have_neg = (detail::copysign(1.0, root.im.hi) < 0.0);
    if (want_neg != have_neg) root.im = negate(root.im);
}
```

`Im(1 - z^2) = -2*Re*Im`, so the root is negative-imaginary exactly when `Re` and
`Im` share a sign. `detail::copysign` is used rather than `signbit` because
`copysign` is already in `config.hpp`'s dispatch and is device-safe on every
backend; adding `signbit` to the dispatch was avoidable. **No change to
`config.hpp`, and none to the real-arithmetic headers.**

The guard is `Im == 0 && |Re| > 1` — exactly the cut — and it is a **no-op on the
two conventions that were already correct**, so it cannot move any point off the
cut. The gate confirms this: `asin` and `acos` decrease at zero points in all four
backends.

**Measured, all four on-cut cases at both signed zeros:**

```
point         DD before  after   QF before  after   TF before  after   FF before  after
-2 +0i           30.05   30.05      28.32   28.32      21.43   21.43      14.00   14.00
-2 -0i            0.00   31.00       0.00   29.00       0.00   20.90       0.00   13.90
+2 +0i            0.00   30.05       0.00   28.32       0.00   21.43       0.00   14.00
+2 -0i           31.00   31.00      29.00   29.00      20.90   20.90      13.90   13.90
-100 +0i         28.38   28.38      27.08   27.08      18.88   18.88      11.00   11.00
-100 -0i          0.00   31.00       0.00   28.22       0.00   21.70       0.00   14.00
+100 +0i          0.00   28.38       0.00   27.08       0.00   18.88       0.00   11.00
+100 -0i         31.00   31.00      28.22   28.22      21.70   21.70      14.00   14.00
```

Every previously-failing convention now scores at or near cap, and every
previously-correct one is unchanged to the digit. Off-cut controls (`|Re| < 1`, and
`±1e-20i`) are unchanged. `-1 +0i` and `+1 -0i` — the branch points, where the
guard deliberately does not fire — score at cap in all four.

**`acos` is fixed for free, and it is confirmed by measurement, not assumed.**
`acos(z) = pi/2 - asin(z)` (`dd_complex.hpp:253-258`) was not touched. Its on-cut
column is identical to `asin`'s row for row in all four backends: `-2 -0i` goes
0.00 -> 31.00 in DD, `+2 +0i` 0.00 -> 30.05, and so on. Corpus means over the full
element set:

```
backend   asin before -> after     acos before -> after
DD           25.89  ->  27.60         25.88  ->  27.58
QF           24.38  ->  25.96         24.27  ->  25.85
TF           17.50  ->  18.65         17.42  ->  18.56
FF           10.41  ->  11.12         10.29  ->  10.99
```

Tolerance rows ratcheted accordingly in all four
`tests/*_complex_accuracy_test.cpp`. On the dense sweep the `asin` cell's zero
count goes 72 -> 0 in DD, and `acos`'s 74 -> 2.

**KI-5 (c) is NOT closed by this and was not touched.** `acos`'s remaining zeros,
and points such as `acos(2 + 1e-20i)` scoring 11.31 (DD) / 8.76 (QF) / 2.06 (TF) /
0.00 (FF), are the `pi/2 - asin` cancellation, which is (c) and is out of this
session's scope. Those scores are unchanged by this fix. Likewise (a)'s `asinh` was
already resolved and (b)'s `atanh` is untouched.

Probes used: `/tmp/ki1_probe.cpp`, `/tmp/ki1_probe2.cpp`, `/tmp/ki1_after.cpp`
(throwaway, not committed).

---

### Why nothing caught these

The four `tests/*_complex_accuracy_test.cpp` targets added 2026-09-02 do score
`asin`, `acos`, `asinh` and `atanh` — they gate on the **mean** over the shared
corpus, and the means pass. That is not a bug in the tests; it is the structural
limit of a mean-gated, randomly-sampled corpus:

- (a), (b) and (c) are ramps, worst at the extremes of the magnitude window; the
  corpus's log-uniform sampling puts only a few elements there and the mean
  absorbs them. Their `min` columns are already 0.00 and are currently attributed
  to conditioning in the tolerance-table comments — correctly, as it happens, but
  the comment does not say *how far* the conditioning goes.
- (d) needs an input that is exactly on the real axis with a specific signed zero
  AND `|Re| > 1`. The corpus's complex corner family does place real corners on
  the real axis, but not with the `Im = -0` variant, and a handful of such
  elements cannot move a mean over ~1500.

This is the gap the dense sweep baseline (`scripts/sweep_accuracy.cpp`,
`validation/sweep/`) is built to cover: it records a score **per grid point**, so
a defect at one point is visible whether or not the mean notices, and a later fix
cannot silently trade it for a loss somewhere else.

### Why these were not fixed when found

Same reason as KI-1, and they belong with it: every one of these fixes changes
shipped DD/FF/QF/TF numerical output, which trips the byte-identical gate and
re-baselines the demo accuracy tables. That is a deliberate, scheduled decision,
not something to slip in alongside a measurement pass. This session's remit was
measurement and documentation only.

### To close

1. **(a)** `asinh`: add the odd-function reflection for `Re(z) < 0`, in all four
   `*_complex.hpp` headers. Cheapest of the five.
2. ~~**(d)** `asin`: fix the signed-zero sheet selection; `acos` is fixed for
   free.~~ **Done 2026-09-02**, commit `f421a0e` — see "Resolution of (d)"
   above. `acos` was indeed fixed for free, confirmed by measurement.
3. **(c)** `acos`: restructure away from `π/2 − asin`.
4. **(b)** `atanh`: add a complex `log1p` and switch to the `log1p` form.
5. Re-run the dense sweep and diff against `validation/sweep/` — the monotone gate
   must report no point where any *other* op's digits decreased. This is the whole
   reason the baseline is recorded before the fixes, not after.
6. Re-baseline the affected demo accuracy tables. The byte-identical gate WILL
   trip; that is the intended signal.

Probes used: `/tmp/sweep_build/ki5_probe.cpp` (plain-`double` transcription) and
`/tmp/sweep_build/ki5_backends.cpp` (all four backends against quadmath) —
throwaway, not committed. Both are subsumed by `scripts/sweep_accuracy.cpp`,
which is committed and whose grid was extended to carry these exact points:

- the `axis` complex family puts `±10^±k` and `±(1 ∓ 10^-k)` on the real axis at
  **both** signed zeros, which is what (a), (b) and (c)'s ramps are measured
  along and what (d) needs;
- the `cut-re` family puts `±2`, `±10`, `±100` on the cut at both signed zeros,
  which is (d) and KI-1.

The sweep reproduces the per-backend tables above cell for cell — for instance
`asinh` at `−1e2`, `−1e4`, `−1e8` and `−1e12` reads 29.41 / 24.49 / 17.62 / 9.28
for DD in both, which is the cross-check that the two measurements are of the
same thing.

---

# Findings from the zero-scoring sweep triage (2026-09-02)

KI-6 through KI-11 all came out of one exercise: classifying every point in
`validation/sweep/sweep_baseline.csv` that scores below **50 % of its backend's
cap** (44,911 of 428,592 points) into a range limit, a conditioning limit, or
neither. The classifier is `scripts/sweep_accuracy --classify`; its output,
`validation/sweep/sweep_classified.csv`, carries one row per triaged point with
the inputs, the binary128 reference, what the backend returned, and the three
measured ceilings that justify the verdict. Re-run it to reproduce any number
quoted below.

**None of these were found by eyeballing.** Each is what remained after range
and conditioning were subtracted, and each is a case where the true result is
comfortably inside the backend's representable range and the operation is
well-conditioned there — so the loss is the algorithm's, not the format's.

**None of them were fixed when this block was written** — that session measured
and documented only. **KI-6 and KI-7 have since been resolved** (2026-09-02,
commit `ad82f4f`), **KI-8** (2026-09-02, commit `4cd7dcb`) and **KI-9**
(2026-09-02, commit `82427f6`) — but see **KI-19**, which shows the KI-9 fix
moved the ceiling rather than removing it. **KI-10** has since been resolved
(`58fda3b`) and **KI-11** likewise (`855292d`, completed in batch-8), and
KI-12…KI-20 were added on 2026-09-03 by the session that diagnosed the 9,238
UNEXPLAINED sweep points. `docs/DOMAINS.md`
records the resulting usable ranges per backend per op, and is generated from the
same data — it was regenerated after each fix, so the counts quoted in the KI-6,
KI-7, KI-8 and KI-9 *Extent* sections above are the pre-fix numbers and no longer
match the current CSVs.

---

## KI-6 — `exp` flushes to ZERO outside a hard ±300 argument guard, discarding ~170 representable decades **[RESOLVED ad82f4f]**

**Severity: high (silently returns 0 for a large, ordinary result), all backends,
worst on DD.**

### What

`exp` (and, through it, `exp2`, `exp10`, `sinh`, `cosh`) appears to carry a hard
argument guard near |x| = 300 and returns exactly `0` beyond it — regardless of
whether the true result is representable. For DD it plainly is: DD reaches
1.8e308, so every result from exp(300) up to exp(709) is thrown away.

```
backend op     point  input                 true result            returned
DD      exp    130    316.22776601683796    2.1675733645732e+137   0
DD      exp    131   -316.22776601683796    4.6134539958093e-138   0
DD      exp10  998    131.94689145077126    8.8489440828519e+131   0
DD      exp10  999    131.94689145077129    8.8489440828525e+131   0
```

Note `exp10`: the guard is applied to the *internal* `x·ln10`, so `exp10` loses
everything above |x| ≈ 130.3 even though DD holds 10^308. That is 565 DD grid
points — the single largest unexplained family on DD.

### Extent (points classified UNEXPLAINED / returned-zero)

```
DD exp10 565   DD exp 52   DD exp2 2
FF exp 36      FF sinh 36  FF cosh 36
```

The FF/QF/TF counts are small only because their own range limit (3.4e38) bites
first and those points are correctly classified OVERFLOW instead.

### Why it is not a range limit

The classifier tests the true result against the backend's word range before
anything else. These points passed that test: `range` = 31.00 in every row
above, i.e. the result's magnitude leaves room for the full 31 digits. The
`ach` column is 29.4, so the point is well-conditioned too.

### Closing it

Replace the guard with one derived from the *word* range (overflow at
x > 709.78 for a double-word backend, x > 88.7 for a float-word one) and return
±inf / 0 only past the real boundary. Check `exp2`, `exp10`, `sinh` and `cosh`
apply the guard to their own argument, not to the reduced one.

### Resolution (2026-09-02)

Fixed in **`ad82f4f`**, all four backends (`dd_math.hpp`, `ff_math.hpp`,
`qf_math.hpp`, `tf_math.hpp`).

**Two independent bugs, not one.** The guard was only half the story.

1. *The guard itself* was a shared ±300 (±80 for TF) constant with no relation to
   any type's range. Replaced with a limit derived from the word format:
   `> ln(DBL_MAX) = 709.78271289338397` → `+inf`, `< ln(2^-1074) ≈ -745.2` → `0`
   for DD; `> ln(FLT_MAX) = 88.722839f` → `+inf`, `< ln(2^-149) ≈ -104f` → `0`
   for FF/QF/TF. TF additionally returned the *sentinel* `1.0e30f` on overflow —
   a finite wrong answer, worse than 0; that is gone.

2. *The final scaling* would have overflowed anyway even with the guard widened.
   Every backend materialised `2^nz` as a scalar and multiplied through. For
   `nz > 1023` (DD) or `> 127` (FF/QF/TF) that scalar is `inf` before it ever
   reaches the mantissa, and symmetrically `0` at the bottom, so exp would have
   returned inf/0 across the last exponent decade regardless. The scaling is now
   per word via `ldexp`, which cannot saturate:

   ```cpp
   if (nz >= -1021 && nz <= 1023) {              // 2^nz exact and normal
       const double pow2 = detail::ldexp(1.0, nz);
       return DoubleDouble(s3.hi * pow2, s3.lo * pow2);
   }
   return DoubleDouble(detail::ldexp(s3.hi, nz), detail::ldexp(s3.lo, nz));
   ```

   The banded fast path is not cosmetic: GCC 13.3.0 inlines `ldexpf(1.0f, n)` as
   exponent construction but emits a real libm call for `ldexpf(mantissa, n)`.
   Taking the component-wise form unconditionally cost FF exp **+49%**
   (421 → 627 ns/call). With the band it is +1.5%. See *Cost*.

   FF carried a third defect on the same path: an iteration-limit escape
   `if (l1 == 60) return FloatFloat(0.0f);` that discarded the partial sum. It now
   falls through with what it has.

**Measured, quad oracle, digits of agreement (`-1.00` = returned inf, `-0.00` =
returned 0, `0.00` = finite but no correct digit):**

| backend | x | before | after |
|---|---|---|---|
| DD `exp` | 300 | −0.00 (returned 0) | **30.17** |
| DD `exp` | 316.22777 | −0.00 | **30.24** |
| DD `exp` | 700 | −0.00 | **30.34** |
| DD `exp` | 709.78 | −0.00 | **30.10** |
| DD `exp` | 710 | −0.00 | −1.00 (`+inf`, correct: > DBL_MAX) |
| DD `exp` | −300 | −0.00 | **30.34** |
| DD `exp` | −700 | −0.00 | **19.98** (subnormal-limited) |
| DD `exp10` | 200 | −0.00 | **29.07** |
| DD `exp10` | 307 | −0.00 | **29.00** |
| DD `exp10` | −300 | −0.00 | **23.77** |
| DD `exp2` | 1023 | −0.00 | **31.00** |
| FF `exp` | 88 | −0.00 | **13.10** |
| FF `exp` | 88.72 | −0.00 | **13.57** |
| FF `exp` | −88 | −0.00 | **7.17** |
| FF `exp10` | 38.5 | −0.00 | **13.40** |
| QF `exp` | 88.72 | −0.00 | **27.83** |
| QF `exp10` | 38.5 | −0.00 | **28.57** |
| TF `exp` | 80 | 0.00 (`1e30` sentinel) | **21.00** |
| TF `exp` | 88.72 | 0.00 (`1e30`) | **20.77** |
| TF `exp` | −80 | −0.00 | **11.10** |
| TF `exp10` | 38.5 | 0.00 (`1e30`) | **20.40** |

Every backend now returns a correct result over its whole representable range and
`+inf`/`0` only past the true word boundary.

### Divergence from QD 2.3.24 — deliberate, recorded

QD's `dd_real::exp` (`src/dd_real.cpp:217-220`) has the same *shape* of guard but
at the right place:

```cpp
if (a.x[0] <= -709.0) return 0.0;
if (a.x[0] >=  709.0) return dd_real::_inf;
```

and already scales with `ldexp(s, m)`. So the ±300 was **our** divergence from the
source, not QD's. We now go slightly *wider* than QD in both directions, which is
the second, intentional divergence:

- QD returns `_inf` at exactly 709.0, but `e^709 = 8.22e307 < DBL_MAX`. We carry
  to 709.78271289338397 and score 29.64 digits at 709.
- QD returns 0 below −709.0, but FP64 subnormals reach `2^-1074 ≈ 4.94e-324`
  ≈ `e^-745`. We carry to −745.2; accuracy degrades gracefully into the subnormal
  band (19.98 digits at −700, 16.29 at −708, 2.59 at −740) rather than falling off
  a cliff at −709.

QD has no FF/QF/TF analogue to diverge from.

### Downstream audit

Every function that composes `exp`, with its verdict:

| function | verdict |
|---|---|
| `exp2`, `exp10` | **sound composition, fixed for free.** They scale the argument and call `exp`; the guard was applied to the internal `x·ln10`, which is exactly why `exp10` lost everything above \|x\|≈130 on DD. Range-deriving the guard restored them with no edit. DD `exp10` now reaches 1e307, FF/QF/TF reach 3.16e38. |
| `expm1` | **sound, no change.** Small-argument path never reaches the guard; large-argument path inherits the fix. |
| `pow` | **sound, no change.** DD `pow(10,300)` = 1e300 at 27.34 digits, FF/QF/TF `pow(10,35)` at 13.76 / 26.33 / 19.52. |
| `sinh`, `cosh` | **broken, fixed.** Past the guard they were **NaN**, not zero — the two-exponential form evaluated `e^x − 1/e^x` with one operand flushed. Added a saturation branch above `ln(1/u)/2` per backend (`kDDHyperbolicSaturate = 40.0`, `kFFHyperbolicSaturate = 19.0f`, `kQFHyperbolicSaturate = 36.0f`, `kTFHyperbolicSaturate = 27.0f`) where sinh and cosh agree to the last bit and both equal `e^{\|x\|}/2`. QF was NaN even *below* the old guard: `1/e^{80}` = 1.8e-35 goes through `divide()`, whose Dekker splitter overflows at FP32. |
| `tanh` | **broken, needed its own fix — see KI-7.** |
| `log` | **broken *by* this fix, fixed.** `exp` now returns `+inf` on genuine overflow instead of 0, and `log`'s Newton step `(a − e^x)/e^x` evaluates `(inf − inf)/inf = NaN` on an infinite argument. `atanh(±1)` reaches it through `(1+a)/(1−a)`. Added `if (detail::isinf(a.hi)) return a;` to all four `log`s. This was caught by the monotone gate — TF `atanh` at grid points 120/422/570/1650 (input exactly `1`) dropped the full cap 21.70 → 0 — not by reasoning; see *Gate*. |
| complex `exp`, `sin`, `cos`, `tan`, `tanh` | **inherit the real fixes; not separately audited.** They call the real `exp`/`sinh`/`cosh` on `Re(z)`, so the range extension propagates. No complex-specific probe was run. KI-7's note of 562 complex `tanh` points with no correct digits should be re-checked when complex work is next opened. |

### Cost

Measured, not estimated: tight loop, 200,000 calls, `-O2`, GCC 13.3.0, best of 5
reps, before/after built from the same source otherwise, interleaved rounds
(rounds 1 and 3 agreed; round 2 showed a 2× outlier on the *same* binary — the
machine is noisy, the per-binary numbers are reproducible).

| op | before | after | Δ |
|---|---|---|---|
| DD `exp` | 835.8 ns | 830.3 ns | **−0.7%** |
| FF `exp` | 603.9 ns | 613.0 ns | **+1.5%** |
| QF `exp` | 3576.7 ns | 3599.0 ns | **+0.6%** |
| TF `exp` | 1344.5 ns | 1328.8 ns | **−1.2%** |
| DD `tanh` | 509.6 ns | 500.5 ns | **−1.8%** |
| DD `cosh` | 455.8 ns | 514.2 ns | **+12.8%** |

`exp` itself is free to within noise — the added work is two comparisons on a path
that already does 9 squarings. `cosh` pays 12.8% for the saturation branch, which
is the KI-4 bargain (Reet accepted 12.9% there to remove a sign error) applied to a
NaN. `tanh` got *faster*: past the saturation threshold it now returns a constant
instead of evaluating an exponential.

### Gate and tests

Monotone gate (`sweep_baseline.csv`, 428,592 points, 253 cells): **exit 0, zero
decreases, 6,501 increases.** No accepted regressions. ctest **34/34**.

Two gate failures happened on the way and are worth recording because both were
real defects the probe had missed:

- **894 decreases, worst 21.70** — TF `atanh(1)` → `log(+inf)` → NaN. Root cause
  above; fixed by the `log` infinity guard.
- **670 decreases, worst 1.25**, all on *negative* arguments — my first sinh/cosh
  saturation branch computed `exp(\|a\| − ln2)`, and perturbing the argument turns
  into relative output error. The baseline reached `e^{\|a\|}` via `1/exp(a)`,
  which was luckier there. Fixed by mirroring the baseline route exactly
  (`e = exp(a)`; `if (a < 0) e = 1/e`; then *exact* halving) and keeping the
  shifted-argument form only as an overflow fallback.

A residual class the gate could *not* catch, because it was NaN in the baseline
too: DD sinh/cosh at −700 and QF sinh at −80/−88 still returned NaN, because
`divide()`'s splitter overflows (DD above ~1.3e300, QF at FP32). The fallback
condition is now `(detail::isinf(e.hi) || e.hi != e.hi)` — the self-inequality NaN
test is used because only `isinf` and `ldexp` are confirmed present in
`config.hpp`'s `detail::` dispatch.

### Tolerance ratchet

`tests/{dd,ff,qf}_accuracy_test.cpp` gate `exp`/`exp2`/`exp10`/`tanh` by a domain
*filter* plus a sampler, not a digit tolerance, so the ratchet is a domain
widening — the rows now cover ground the old code could not reach:

| test | op | filter before → after | sampler before → after |
|---|---|---|---|
| dd | `exp` | `x < 300` → `x < 709` | `(-300,299)` → `(-708,709)` |
| dd | `exp2` | `\|x\| < 400` → `\|x\| < 1022` | `(-100,100)` → `(-1021,1022)` |
| dd | `exp10` | `\|x\| < 120` → `\|x\| < 307` | `(-80,80)` → `(-307,307)` |
| dd | `tanh` | `\|x\| < 300` → `isfinite(x)` | `(-50,50)` → `(-1e4,1e4)` |
| ff | `exp` | `x < 88` → `x < 88.7` | `(-88,87.5)` → `(-87,88.7)` |
| ff | `exp10` | `\|x\| < 38` → `\|x\| < 38.5` | `(-37,37)` → `(-37.5,38.5)` |
| ff | `tanh` | `\|x\| < 300` → `isfinite(x)` | `(-50,50)` → `(-1e4,1e4)` |
| qf | `exp` | `x < 88` → `x < 88.7` | `(-35,80)` → `(-35,88.7)` |
| qf | `exp10` | `x < 38` → `x < 38.5` | `(-15,30)` → `(-15,38.5)` |
| qf | `tanh` | `\|x\| < 300` → `isfinite(x)` | `(-50,50)` → `(-1e4,1e4)` |

All still pass. Two deliberate omissions: `tf_accuracy_test.cpp` uses min-drop rows
(`{"exp", 19.0}`) rather than filters and was left alone; `sinh`/`cosh` filters were
left alone as the riskier widening — their accuracy in the last decade is
subnormal-limited, not algorithm-limited, and a widened row would encode a floor
that is a property of the format.

---

## KI-7 — `tanh` returns the WRONG SIGN for |x| past the `exp` guard, in all four backends **[RESOLVED ad82f4f]**

**Severity: high (wrong sign is worse than no answer), all backends.**

### What

Downstream of KI-6, but a strictly worse symptom and worth its own entry: for
|x| beyond the guard, `tanh` returns **−1 where the answer is +1** and vice
versa.

```
backend op    point  input                 true result   returned
DD      tanh  130    316.22776601683796     1            -1
DD      tanh  131   -316.22776601683796    -1             1
```

The mechanism is visible in the numbers: with `exp(2x)` flushed to 0, a
`(e−1)/(e+1)` form evaluates to `(0−1)/(0+1) = −1` for every large positive x.
TF instead returns NaN (452 points), which is the same defect through a
different form.

### Extent

2,590 real points return the wrong sign; 452 more return NaN; complex `tanh`
adds 562 points with no correct digits. Per backend, unexplained `tanh` points:
**DD 804, QF 1206, FF 1180, TF 601.**

### Closing it

Fixing KI-6 fixes most of this, but `tanh` should also short-circuit to
±1 once |x| exceeds the point where the type cannot resolve 1 − tanh(x)
(|x| > 36 for FF, > 19 for a 48-bit type, > 358 for DD), which is both correct
and much faster than evaluating the exponential at all.

### Resolution (2026-09-02)

Fixed in **`ad82f4f`**, same commit as [KI-6](#ki-6).

**Was KI-7 downstream of KI-6? Partly — and the remainder is a separate defect.**
This was checked, not assumed, and the entry above ("Downstream of KI-6") is
half right:

- **Downstream part.** DD `tanh(316.22777)` is fixed by the KI-6 guard alone:
  `expm1(632.46)` is now a real number, so `expm1(2x)/(expm1(2x)+2)` evaluates
  correctly and the sign is right.
- **Separate part.** For DD `|x| > 354.9` the *doubled* argument `2x` exceeds
  `ln(DBL_MAX)` legitimately. `expm1` returns `+inf`, and `inf/(inf+2)` is **NaN**
  — a different wrong answer, reached through correct code, and unreachable by any
  amount of guard-widening. `tanh(400)` and `tanh(1e6)` fail here. The same holds
  in FF/QF above `|x| > 44.4`. So `tanh` needed its own saturation short-circuit
  in every backend, exactly as *Closing it* anticipated.

TF's variant is a third case: it forms `tanh = sinh/cosh` rather than from
`expm1`, so before the fix it returned the *correct* `+1` for large positive `x`
(numerator and denominator both hit the `1e30` sentinel and the ratio is 1) and
**NaN** for large negative `x`. Not a sign error on TF — a NaN — which is why the
KI-7 extent table counts TF separately at 601 points.

**The fix**, after the existing odd reflection:

```cpp
if (a.hi > kDDHyperbolicSaturate) return DoubleDouble(1.0);   // DD, FF, QF
```

```cpp
if (abs(a).f0 > kTFHyperbolicSaturate)                        // TF: no reflection
    return TripleFloat(a.f0 < 0.0f ? -1.0f : 1.0f);
```

The thresholds are the same per-backend `ln(1/u)/2` constants KI-6 introduced for
`sinhcosh` — 40.0 (DD), 19.0f (FF), 36.0f (QF), 27.0f (TF) — the point past which
the type cannot resolve `1 − tanh(x)` from 1, so `±1` is not an approximation but
the correctly rounded answer.

### Sign correctness, measured against the quad oracle

`sgn` is `got/ref`; a mismatch is the defect. `+0` denotes NaN (`sgn` of a NaN
reads 0).

| backend | x | before `sgn` | after `sgn` | after digits |
|---|---|---|---|---|
| DD | 100 | `+1/+1` ✓ | `+1/+1` ✓ | 31.00 |
| DD | 316.22777 | **`-1/+1` ✗** | `+1/+1` ✓ | 31.00 |
| DD | 400 | **`-1/+1` ✗** | `+1/+1` ✓ | 31.00 |
| DD | 1000 | **`-1/+1` ✗** | `+1/+1` ✓ | 31.00 |
| DD | 1000000 | **`-1/+1` ✗** | `+1/+1` ✓ | 31.00 |
| DD | −316.22777 | **`+1/-1` ✗** | `-1/-1` ✓ | 31.00 |
| DD | −400 | **`+1/-1` ✗** | `-1/-1` ✓ | 31.00 |
| DD | −1000 | **`+1/-1` ✗** | `-1/-1` ✓ | 31.00 |
| DD | −1000000 | **`+1/-1` ✗** | `-1/-1` ✓ | 31.00 |
| FF | 44 | **`-1/+1` ✗** | `+1/+1` ✓ | 14.00 |
| FF | 50 | **`-1/+1` ✗** | `+1/+1` ✓ | 14.00 |
| FF | 100 | **`-1/+1` ✗** | `+1/+1` ✓ | 14.00 |
| FF | 1000000 | **`-1/+1` ✗** | `+1/+1` ✓ | 14.00 |
| FF | −44 | **`+1/-1` ✗** | `-1/-1` ✓ | 14.00 |
| FF | −1000000 | **`+1/-1` ✗** | `-1/-1` ✓ | 14.00 |
| QF | 44 | **`-1/+1` ✗** | `+1/+1` ✓ | 29.00 |
| QF | 1000000 | **`-1/+1` ✗** | `+1/+1` ✓ | 29.00 |
| QF | −44 | **`+1/-1` ✗** | `-1/-1` ✓ | 29.00 |
| QF | −1000000 | **`+1/-1` ✗** | `-1/-1` ✓ | 29.00 |
| TF | 100 | `+1/+1` ✓ | `+1/+1` ✓ | 21.00 |
| TF | −100 | **`+0/-1` (NaN) ✗** | `-1/-1` ✓ | 21.00 |
| TF | −1000 | **`+0/-1` (NaN) ✗** | `-1/-1` ✓ | 21.00 |
| TF | −1000000 | **`+0/-1` (NaN) ✗** | `-1/-1` ✓ | 21.00 |

Every probed point out to ±1e6 has the correct sign in all four backends, and
scores at or near the type's cap.

### Cost

DD `tanh` **509.6 → 500.5 ns/call, −1.8%** (same harness as KI-6). The
short-circuit replaces an exponential with a constant, so the correct answer is
also the cheaper one — the opposite of the KI-4 trade.

### Gate and tests

Covered by the KI-6 run: monotone gate **exit 0**, zero decreases, 6,501
increases; ctest **34/34**. `tanh` rows in the DD/FF/QF accuracy tests were
ratcheted from `|x| < 300` / `uniform(-50,50)` to `isfinite(x)` /
`uniform(-1e4,1e4)`, so the suite now samples the region that was wrong.

---

## KI-8 — `hypot` and complex `abs` have NO scaling, so they overflow and underflow far inside the representable range **[RESOLVED 7680809]**

> **Heading history.** Fixed first at `4cd7dcb` (2026-09-02), **reopened
> 2026-09-03** when the low edge of that fix was found to be set against the
> wrong cliff, then fixed for good at `7680809` (2026-09-04). The heading read
> `[REOPENED 2026-09-03]` until 2026-09-04 even though the body below had
> recorded the second fix; under the marker convention at the top of this file a
> reopened-then-fixed entry reads `[RESOLVED <final commit>]` and keeps the
> reopen history here, in the body.
>
> Re-verified by measurement 2026-09-04: QF `hypot(1e-16, 1e-16)` = 1.4142135862e-16
> (13.85 → 29.00 digits) and QF `hypot(1e30, 1e30)` = 1.4142135837e+30 — both ends
> finite and correct.

### RESOLVED 2026-09-04 — commit `fix: KI-8 scale hypot/abs at BOTH ends; KI-13 remaining unscaled sums of squares`

(Referenced by subject, not sha: this doc ships INSIDE that commit, so its own
sha is not knowable while it is being written, and the session was constrained
to a single commit.)

**What the 2026-09-02 fix missed.** It added the gate, and it picked the gate's
LOW edge (`1.0e-18f`, `1.0e-150` on DD) against the wrong cliff: the point at
which the *square* stops being a normal **float**. For an expansion the binding
constraint is the point at which the square's LAST WORD stops being a normal
float. A W-word FP32 expansion of magnitude `v` carries its last word at
`~v·2^-24(W-1)`, so `m²` keeps all its words only while `m²·2^-24(W-1) ≥ FLT_MIN`:

| backend | words | derived low limit | old gate edge | width of the hole |
|---|---|---|---|---|
| FF | 2×FP32 | 2^-51 = 4.44e-16 | 1.0e-18f | ~2.4 decades |
| TF | 3×FP32 | 2^-39 = 1.82e-12 | 1.0e-18f | ~5.7 decades |
| QF | 4×FP32 | 2^-27 = 7.45e-9 | 1.0e-18f | ~9.1 decades |
| DD | 2×FP64 | 2^-484.5 = 1.5e-146 | 1.0e-150 | ~3.8 decades |

Inside that hole the direct form ran and silently shed low words. **DD was
genuinely exposed**, not merely in principle: `hypot(1e-150, 1e-150)` and the
matching complex `abs` scored 23.52 of 31 digits.

**Why the verification did not catch it.** The 2026-09-02 probes went
1e19…1e38 (above the gate) and 1e-25/1e-38 (below it, where the scaled tail
already ran and was already at the format limit). Neither ladder has a rung in
[1e-18, 4.4e-16] for FF or in [1e-18, 7.5e-9] for QF — the probe stepped
straight over the band the gate itself had created.

**The fix.** The gate's low edge is now the derived limit itself, to the bit
(`kFFSqLo = 2^-51`, `kTFSqLo = 2^-39`, `kQFSqLo = 2^-27`, `kDDSqLo = 4.0e-146`),
shared by `hypot`, complex `abs` and complex `operator/` in each backend. The
TAIL is unchanged — see below.

**Genuine floor (format limit, documented not fixed).** Below the point where
the expansion's own LAST WORD goes subnormal, precision is gone before any
arithmetic happens: FF 2^-102 ≈ 1.97e-31, TF 2^-78 ≈ 3.31e-24,
QF 2^-54 ≈ 5.55e-17, DD 2^-969 ≈ 5.6e-292. Everything ABOVE those now delivers
full precision; measurements below them (e.g. QF `hypot(1e-20)` at 25.50 of 29)
sit at the format limit and are correct as they stand.

**Rejected alternative, with measurements.** Replacing the divide-based
`m·sqrt(1+t²)` tail with an exact-power-of-two rescale plus a direct square
looks free — scaling every word by `s = 2^-e` loses no bit and `s² = 2^-2e` is
also a power of two, so the expression is scale-equivariant on paper. It is
not free in practice: `sqrt()` of a square does not round-trip, and when the
smaller operand is exactly ZERO the min/max tail returns `m` bit-for-bit while
the squared form returns `sqrt(m²)`, a couple of ulps off. Complex `asin`
amplifies that by `|z|²`. Measured on the 428,592-point sweep, the pow2-squared
tail cost:

| point | z | before | after (rejected) |
|---|---|---|---|
| QF c asin 1628/1630 | ±1e10 + 0i | 24.45 | 11.91 (−12.54) |
| QF c asin 1632/1634 | ±1e11 + 0i | 16.68 | 9.97 (−6.71) |
| QF c asin 1648/1650 | ±1e15 + 0i | 16.08 | 2.04 (−14.04) |

(`asin` routes through complex `sqrt`, whose `abs` sees `(-1e20, ±0)` — exactly
the zero-component case.) The tail therefore keeps the min/max form and only
the GATE moved, which is also the smaller change. A one-binade safety margin on
the low edge was likewise tried and reverted: it cost 0.13–0.74 digits at six
sweep points that landed inside the margin (QF c abs 7/17, c pow 13/25, c div
1/510) for no measured gain.

**Before/after, small magnitudes** (`hypot(x,x)` and complex `abs(x+xi)`,
digits vs the `__float128` oracle; identical for both call sites unless noted):

| backend | 1e-3 | 1e-8 | 1e-12 | 1e-14 | 1e-16 | 1e-17 | 1e-18 | 1e-20 |
|---|---|---|---|---|---|---|---|---|
| FF before | 14.00 | 14.00 | 14.00 | 14.00 | 13.88 | 11.38 | 9.01 | (cliff) |
| FF after | 14.00 | 14.00 | 14.00 | 14.00 | **14.00** | **14.00** | **14.00** | (cliff) |
| TF before | 21.70 | 21.70 | 21.27 | 18.92 | 13.88 | 11.32 | 9.14/10.51 | 14.55 |
| TF after | 21.70 | 21.70 | **21.70** | **21.70** | **21.70** | **21.70** | **21.70** | 14.55 |
| QF before | 29.00 | 29.00 | 22.14 | 17.61 | 13.88 | 11.32 | 9.14 | 25.50 |
| QF after | 29.00 | 29.00 | **29.00** | **29.00** | **29.00** | **28.42** | **26.99** | 25.50 |

DD, on its own ladder: `hypot`/`cabs` at 1e-150 went **23.52 → 31.00**; 1e-146,
1e-160, 1e-200, 1e-290 and 1e-300 were already at cap and are unchanged.

Just below each backend's representation cliff the numbers stay where they
were, as they must: QF 1e-20 → 25.50, TF 1e-24 → 21.21, FF 1e-32 → 13.42, all
identical before and after.

**Before/after, large magnitudes** — unchanged, i.e. the half that already
worked is intact: FF/TF/QF `hypot` and complex `abs` at 1e19, 1e20, 1e24, 1e30,
1e35, 1e38 all still sit at their caps (14.00 / 21.70 / 29.00), and DD at 1e30
through 1e300 all still 31.00.

**Family audit — recheck 2026-09-04.** The 2026-09-02 table below still stands;
the low-edge move changes three verdicts from "fixed at the top end" to "fixed
at both": `hypot`, complex `abs`, complex `operator/`. `operator/` is the one
that needed a judgement call — widening its low edge routes more denominators
through Smith's algorithm, which forms no square at all. Measured, that is a
net win by a wide margin: **+143 improved points against 2 newly decreased**,
so it stayed widened. Everything else in the table was re-read and its verdict
is unchanged: complex `sqrt` inherits the fixed `abs`; `log`/`log10`/`pow`
inherit it in turn; `norm` is unfixable by construction (|z|² overflows exactly
when its own answer does); `arg`, `polar`, `tan`/`tanh` and the gated `atanh`
branch form no exposed magnitude.

**Accepted decreases — monotone gate, 2026-09-04.** The read-only gate
(`sweep_accuracy --baseline validation/sweep/sweep_baseline.csv`) reports
**1,226 increased, 18 decreased, worst 0.56 digits** and therefore exits 1. The
18 are accepted, not fixed, and this is the record of that decision. Every one
sits in a complex chain whose intermediate magnitude falls just below the new
low edge, so the value now takes the min/max tail where it used to take the
direct square; the tail costs a fraction of a digit there because the intermediate
is close enough to the edge that the direct form had not yet shed a whole word.
All 18 are already 5–15 digits below their backend's cap for reasons of
conditioning, so the fraction of a digit is not the binding constraint at any
of them.

| op | points | grid coordinates | before → after |
|---|---|---|---|
| QF c asin | 602, 858 | cut-re, `-1 + 1e-12i` | 22.15 → 21.73 (−0.42) |
| QF c acos | 602 | cut-re, `-1 + 1e-12i` | 22.15 → 21.73 (−0.42) |
| QF c asinh | 1243, 1307 | cut-im | 22.15 → 21.73 (−0.42) |
| QF c acosh | 856, 857 | cut-im | 22.71 → 22.63 (−0.08) |
| QF c acosh | 598, 599 | cut-re | 22.49 → 22.48 (−0.01) |
| QF c pow | 20 | polar, `-7.07e-9 - 7.07e-9i` | 27.40 → 26.84 (−0.56) |
| QF c pow | 1505 | — | 27.47 → 27.36 (−0.11) |
| TF c asin | 610, 866 | cut-re | 15.19 → 15.17 (−0.02) |
| TF c acos | 610 | cut-re | 15.19 → 15.17 (−0.02) |
| TF c asinh | 1251, 1315 | cut-im | 15.19 → 15.17 (−0.02) |
| FF c pow | 740, 1533 | — | 13.50 → 13.34, 13.11 → 13.05 |

Mechanism, concretely, for the largest of them: `asin(-1 + 1e-12i)` forms
`1 - z² ≈ 2e-12`, and complex `sqrt` calls `abs` on that. `2e-12` is below
QF's new low edge `2^-27 = 7.45e-9`, so `abs` now takes the tail. The direct
square at that magnitude has `m²·2^-72 ≈ 8.5e-46`, i.e. it HAS lost its two
bottom words — the tail is the more defensible code path even though it scores
0.42 digits lower at this particular point, because the point's own answer is
conditioning-limited to ~22 digits either way.

**ULP gate** (`sweep_accuracy --ulp`): failing gated cells **67 → 62**, failing
gated points **5,787 → 5,451**. The three `hypot` UNEXPLAINED cells that
motivated the reopen are GONE (FF 5.76e3×, TF 6.88e10×, QF 7.26e14× over bound
→ no longer failing). `QF r atan` went from `inf` to 14.31× over bound;
`FF r acosh` from `inf` to 11.04×; `TF r atan`, `TF r acosh` and `QF r acosh`'s
`inf` cells cleared. `QF r asinh`/`acosh` improved from `inf` to 2.01e11× —
still failing, for the QF-`log` reason recorded under KI-13's residuals.



**Severity: high, FP32-word backends (FF, QF, TF); DD unaffected.**

### REOPENED 2026-09-03 — the underflow half was never fixed

The step-1c ULP triage (`docs/ULP_METRIC.md`, "Step 1c") re-measured this KI
against the condition-aware ulp gate. The **overflow** half holds. The
**underflow** half does not, on all three FP32-word backends:

| backend | `hypot` worst point | measured ulps | bound | over by | digits |
|---|---|---|---|---|---|
| QF | x = −1e−16 | 2.62e18 | 3.6e3 | **7.26e14x** | 13.71 of 29 |
| TF | x = −1e−16 | 6.19e12 | 90 | **6.88e10x** | 12.87 of 21.7 |
| FF | x = −1e−16 | 1.13e5 | 19.6 | **5.76e3x** | 8.17 of 14 |

QF loses 51.5 bits — two whole FP32 words. The mechanism is precisely the one
the KI names, at the small end: `hypot` forms `x²` before it scales, and
`x² = 1e−32` sits 51 bits below QF's subnormal-low-word cliff of 5.55e−17
(the per-backend cliff table is in `docs/ULP_METRIC.md`). The intermediate is
destroyed; the final result is nowhere near the band, so no format-limit
exemption can or should cover it. Scaling by a power of two before squaring —
which is what the KI's own fix does at the top end — would remove it.

Reopened, **not fixed**: step 1c was scoped to diagnosis. The measurement above
is the acceptance test for whoever closes it.

### What

`hypot(a,b)` and complex `abs(z)` appear to evaluate `sqrt(a²+b²)` directly.
The intermediate square overflows the FP32 word above ≈1.8e19 and underflows it
below ≈1.2e-23, in both cases while the *answer* is perfectly representable.

```
backend op      point  inputs                          true result    returned
FF      hypot   4      9.9999999999999994e-30,
                       1.5772958167803594e+21          1.5772958e+21  nan
FF      c abs   752    0 + 1e-23i                      1e-23          0
```

Both directions are the same missing scaling: `hypot` should divide through by
max(|a|,|b|) before squaring.

### Extent

```
op          FF    QF    TF    DD
hypot      309   315   312     0
complex abs 66    84    78     0
```

DD is clean because a double word squares safely across the whole grid; the
same defect is latent there and would appear beyond ±1e154.

### Closing it

Scale by the larger operand: `m = max(|a|,|b|); r = m·sqrt(1 + (n/m)²)`.
One extra division, and it removes both the overflow and the underflow limb.

### Resolution (2026-09-02)

Fixed in **`4cd7dcb`**, all four backends (`dd_math.hpp`, `ff_math.hpp`,
`qf_math.hpp`, `tf_math.hpp`, and the four `*_complex.hpp`).

**Is it a porting defect?** No. QD 2.3.24 has no `hypot` anywhere in
`include/qd/` or `src/`, and no complex header at all — the sole complex support
is the C bindings `c_dd.h`/`c_qd.h`, which expose real arithmetic only. There is
therefore no upstream scaling that this port dropped, and nothing to diverge
from: `hypot` and the whole `*_complex.hpp` family are original compositions and
this is an original defect in them.

**What was done.** `hypot(a,b)` now takes `x=|a|`, `y=|b|`, `m=max`, `n=min` and
returns `m·sqrt(1 + (n/m)²)`, which cannot overflow in the intermediate because
`n/m ∈ [0,1]`. Complex `abs` and complex `/` got the same treatment (the latter
via Smith's algorithm, 1962, for the `|denominator|²` formulation).

**The scaled form is gated, not unconditional.** It is used only for
`m` outside `[1e-18, 1e18]` on the FP32-word backends and `[1e-150, 1e150]` on
DD — i.e. only where the direct form actually breaks. Inside the gate the
original expression is evaluated bit-for-bit and the cost is two compares. Two
reasons, both measured on the 428,592-point sweep:

1. The scaled form is a touch less accurate (divide + square + sqrt + multiply
   versus square + add + sqrt), and there is nothing to win where the direct
   form works.
2. The backends do not agree on the squaring primitive. TF's `hypot` used
   `sqr()` where TF's complex `abs` used `multiply(a,a)`, and they are not
   interchangeable: routing complex `abs` through `hypot`'s `sqr()` cost up to
   **3.04 digits** at TF `c abs` points 736/737/1528–1531 (and propagated to
   `c sqrt`, `c log`, `c log10` — 115 decreases in all). Each call site
   therefore keeps its own primitive inside the gate and they share only the
   scaled tail. This is the same "fix the interval that is broken, do not churn
   the one that is not" reasoning as the `atanh` threshold in `*_complex.hpp`.

**inf/nan convention (C99 F.9.4.3).** `hypot(±inf, y)` is `+inf` for *any* `y`,
NaN included, so the infinity test comes first; both operands go through `abs()`
first so the returned infinity is always `+inf`. Otherwise a NaN operand
propagates to NaN through the arithmetic. `hypot(0,0) = 0`.

**Measured before/after.** Operand pairs spanning the format, scored against a
`__float128` oracle:

```
                                        BEFORE            AFTER        true
FF hypot(1e-30, 1.5772958e+21)          nan            1.577296e+21  1.577296e+21
FF hypot(0, 1e-23)                      0              1.000000e-23  1.000000e-23
FF hypot(1e-25, 1e-25)                  0              1.414214e-25  1.414214e-25
FF hypot(1e+30, 1e+30)                  inf            1.414214e+30  1.414214e+30
FF hypot(1e-38, 1e-38)                  0              1.414214e-38  1.414214e-38
QF/TF: identical to FF at all five points (same FP32 word range).
DD hypot(1e+200, 1e+200)                1.414214e+200  1.414214e+200 (unchanged)
DD hypot(1e-320, 1e-320)                1.414016e-320  1.414016e-320 (unchanged)
FF c abs(0 + 1e-23i)                    0              1.000000e-23  1.000000e-23
FF c abs(1e30 + 1e30i)                  inf            1.414214e+30  1.414214e+30
FF c div (1+i)/(1e30+1e30i)             nan            1.000000e-30  1.000000e-30
```

Answers that genuinely do overflow still report non-finite, never a wrong finite
value: `FF/QF/TF hypot(1e300, 1e300)` → `inf` (true 1.41e300, far past the FP32
word's 3.4e38). One residual wrinkle at the very edge: `QF/TF hypot(3e38, 3e38)`
(true 4.24e38, just past 3.4e38) returns **NaN** rather than `+inf`, because the
final `multiply(m, ·)` overflows inside renormalization and produces `inf-inf`.
FF returns `inf` correctly. Both are non-finite, so no wrong finite value
escapes; tightening NaN to +inf there would need an overflow probe in the QF/TF
renormalizers and is not worth a divide on every call. Recorded here rather than
opened as a new KI.

> **Addendum (KI-9, `82427f6`).** That residual wrinkle is now closed, and it
> was worse than "one edge point": KI-8's scaled path divides by `max(|a|,|b|)`,
> and QF/TF `divide` was returning NaN for any operand past 4.15e34 (KI-9). So
> `QF/TF hypot` was NaN from 1e35 *upward* — not only at the genuine-overflow
> edge — while FF, whose splitter guard was already in place, was correct there.
> The KI-8 design was sound; it was sitting on a broken primitive. With KI-9
> fixed, QF/TF now match FF at 1e35/1e36/1e37/1e38, and `hypot(3e38, 3e38)`
> returns `+inf` rather than NaN. It took no overflow probe in the
> renormalizers: setting the error term of an overflowed product to 0 in
> `two_prod`/`two_sqr` was enough.
>
> **The gate thresholds stay as they are.** They were not chosen to dodge the
> division defect, and now that it is fixed the scaled path is correct wherever
> it is taken — so widening is safe but not useful. The 3.04-digit / 115-point
> cost measured above comes from `sqr()` vs `multiply(a,a)`, which KI-9 does not
> touch. Full reasoning in KI-9's resolution.

**Sweep effect.** Monotone gate exit **0**: 3,003 points increased, **0**
decreased, 425,589 unchanged. Every returned-zero point in the two named cells
is gone:

```
zero-scoring points   BEFORE   AFTER
FF  hypot               309       0
QF  hypot               315       0
TF  hypot               312       0
DD  hypot                 0       0
FF  c abs                66       0
QF  c abs                84       0
TF  c abs                78       0
DD  c abs                 0       0
```

**DD was genuinely unaffected — verified, not assumed.** DD scored 31.00 mean
with zero 0.00 points on both `hypot` and `c abs` before the fix and after it,
and the direct-form probes above at 1e±200/1e±320 return the right answer
unscaled. The defect is latent there, not absent: a `double` word squares safely
only to ≈1.3e154, so DD received the same gated scaling with its own thresholds.
The sweep grid simply never reaches past 1e154.

**Family audit.** Everything that forms a sum of squares, a norm or a magnitude,
across all four `*_complex.hpp`:

| member | verdict |
|---|---|
| `hypot` (real) | **exposed — fixed**, scaled with gate |
| complex `abs` | **exposed — fixed**, own fast path + scaled tail |
| complex `operator/` | **exposed — fixed**, Smith's algorithm past the gate |
| complex `sqrt` | **exposed — fixed**; it recomputed `sqrt(re²+im²)` inline, now calls the fixed `abs(z)` |
| complex `log`, `log10` | sound *by inheritance* — they call `abs(z)`, and improved with it |
| complex `norm` (QF, TF only) | **sound — no fix possible.** `norm = \|z\|²` overflows exactly when its own answer does, and underflows exactly when its answer is unrepresentable. There is no scaling that preserves the result. |
| complex `arg` | sound — `atan2(im, re)`, forms no magnitude |
| `polar` | sound — `r·(cos θ + i sin θ)`, no squaring |
| complex `pow` | sound by inheritance — `exp(w·log z)` |
| complex `tan`/`tanh` (`denom = cb² + T2·sb²`) | sound — `cb`, `sb` are sine/cosine and `T2` a squared tanh, all bounded by 1; the sum cannot leave the range |
| complex `atanh` small branch (`den = (1-x)² + y²`) | sound — already gated to \|z\| < 0.0625 by KI-5(b), with a comment naming this exact overflow as the reason for the gate |
| complex `asinh`/`acosh`/`asin`/`acos` | sound — they route through the fixed `sqrt`/`log`/`abs`, and form no bare sum of squares of their own |

**Cost.** Not measured. A cheap measurement was not available: the cost
benchmark is a demo target and demos were out of scope for this session, and a
microbenchmark of an inlined header function under `-O2` would mostly measure
the harness. What can be stated from the code: inside the gate the fix adds two
floating-point compares on the leading words and nothing else — no divide, no
branch misprediction hazard for the common case — and the gate covers 36 decades
on the FP32-word backends and 300 on DD. The divide is paid only outside it.

**Ratchet.** `tests/tf_accuracy_test.cpp` `{"hypot", 20.0}` → `20.88`
(measured mean 21.18 on [-100,100]², a domain wholly inside the gate and so
bit-identical before and after; the 1.18-digit slack was never earned). The
`dd`/`ff`/`qf` hypot rows sit under shared family-class tolerances, not per-op
rows, and were left alone. The complex `abs`/`sqrt` rows already carry the
standard 0.30 headroom and did not move.

---

## KI-9 — QF/TF division returns NaN when the QUOTIENT exceeds the Dekker splitter's headroom **[RESOLVED 82427f6]**

> **Follow-up: [KI-19](#ki-19--qftf-divide-returns-nan-where-inf-is-correct-when-the-quotient-overflows-the-format-resolved-0ad44fe).**
> `82427f6` closed the *splitter* ceiling described here — the one at ≈8.3e34. It
> did not close the *format* ceiling above it: once the quotient itself overflows
> the FP32 exponent range, `divide` still returned NaN where `±inf` is correct.
> That residue was filed separately as **KI-19** and fixed at `0ad44fe`. KI-9 and
> KI-19 are two distinct defects at two distinct magnitudes, not one entry
> superseding the other; both are resolved, and KI-19 is the one that completed
> the work.

**Severity: medium, QF and TF only.** *(Re-rated to blocking once KI-8's scaled
`hypot` was built on top of it — see the Resolution.)*

### What

Real `div` returns NaN when the quotient exceeds ≈8.3e34 — which is `FLT_MAX /
4097`, the point at which the Dekker splitting constant used inside the FP32
`two_prod` overflows. The quotient itself is representable (FP32 reaches
3.4e38); it is the splitter that cannot hold it.

```
backend op   point  input                  true result             returned
QF      div  238    3.1622776601683791e+29 3.9467330694268561e+36  nan
TF      div  241   -1e+30                  5.2975997860710159e+35  nan
```

Complex `div` shows the same boundary: QF 43 NaN + 35 inf, TF 43 NaN + 35 inf.
This is the splitter-overflow hazard already recorded for QF multiply, reaching
division through its Newton/long-division refinement step.

### Closing it

Guard the splitter the way the QF multiply path was guarded: scale the operand
down by a power of two before splitting and scale the product back afterwards.

### Resolution (2026-09-02)

Fixed in **`82427f6`** (`qf_math.hpp`, `tf_math.hpp`). DD and FF were already
clean — FF had re-derived the same guard independently at B8/B9
(`ff_math.hpp:245`), and DD's FP64 words put the threshold at 6.7e299.

**It is a PORTING DEFECT.** QD 2.3.24 *does* guard its splitter:
`qd/include/qd/inline.h:66-83` branches on `_QD_SPLIT_THRESH` (6.69692879491417e+299
= 2^996), pre-scales by 2^-28, splits, and unscales `hi`/`lo` by 2^28. The QF and
TF ports dropped that branch deliberately and recorded the omission in
PORT_NOTES_QF §2 as an accepted simplification ("Large-magnitude splitter
overflow is NOT ported"). It is not an acceptable one, and this fix restores the
upstream behaviour at FP32: threshold `FLT_MAX / (split + 1)` = 4.1528233e34,
pre-scale 2^-14, unscale 2^14 — the FP32 analogue of QD's 2^(1024-996), which is
2^(128-114). Both scalings are exact powers of two, so the hazard path adds no
rounding and the non-hazard path is bit-identical to before.

**The break point is 4.15e34, not the 8.3e34 the entry above predicted.** The
splitter is 8193 = 2^13+1, not 4097 = 2^12+1, so the ceiling is FLT_MAX/8193.

**Multiplication shared the exposure; `sqrt` did not.** Fixing the split inside
`qf_two_prod`/`qf_two_sqr` (and the TF pair) covers `multiply`, `sqr`,
`multiply_scalar`, `divide`, `divide_scalar` and everything composed from them in
one place. `sqrt` (Heron) is only exposed transitively: it splits the iterate and
the quotient digit, both ≈ sqrt(a) ≈ 1e19 at the top of the range, so it needed
a > 1.7e69 to break and was never reachable. Measured, before and after —
`divide(x,x)`, `multiply(x,1/x)`, `a*a` and `sqrt(x)` swept over 10^0..10^38 in
quarter-decade steps, "first failing x" = first point off the exact answer:

```
                     divide(x,x)   multiply(x,1/x)   a*a        sqrt(x)
FF  before / after   none / none   none / none       none/none  none / none
QF  before           5.62e+34      5.62e+34          none       none
QF  after            none          none              none       none
TF  before           5.62e+34      5.62e+34          none       none
TF  after            none          none              none       none
```

(5.62e34 is the first sampled point past the true 4.153e34 boundary.)

**Genuine overflow now reports ±inf, not NaN.** Two further sites, both fixed in
the same commit:

- `two_prod`/`two_sqr` evaluated their residual as `inf - inf = NaN` whenever the
  product itself overflowed, so a correctly-infinite answer came back NaN-tailed.
  The error term of an overflowed product carries no information; it is now 0,
  via an early return that also skips two splits on that path.
- `sqr()` forms `2.0f * a.f0 * a.fk` left to right, so the *doubling* overflows on
  its own for `|a.f0| > FLT_MAX/2 = 1.7e38`, and `inf * 0` (the usual value of a
  trailing limb) is NaN. Whenever those doublings can overflow, `a.f0 * a.f0`
  already has, so one leading-limb test at the top returns ±inf and covers every
  internal site.

This closes the "residual wrinkle" recorded under KI-8: `QF/TF hypot(3e38, 3e38)`
(true 4.24e38, past the FP32 word range) now returns `+inf`, matching FF.

**KI-8 re-verified.** `hypot(a, a)`, the case that made this blocking:

```
a       true          FF (was OK)   QF before  QF after      TF before  TF after
1e+30   1.414214e+30  1.414214e+30  1.414e+30  1.414214e+30  1.414e+30  1.414214e+30
1e+35   1.414214e+35  1.414214e+35  nan        1.414214e+35  nan        1.414214e+35
1e+36   1.414214e+36  1.414214e+36  nan        1.414214e+36  nan        1.414214e+36
1e+37   1.414214e+37  1.414214e+37  nan        1.414214e+37  nan        1.414214e+37
1e+38   1.414214e+38  1.414214e+38  nan        1.414214e+38  nan        1.414214e+38
3e+38   inf (4.24e38) inf           nan        inf           nan        inf
```

**KI-8's gate thresholds are KEPT as they are.** Decision and reason: the gate
(`m` outside `[1e-18, 1e18]` for the FP32-word backends) is no longer standing in
for a broken primitive — the scaled path is now correct everywhere it is taken,
so widening it is *safe*. It is still not *better*. KI-8 measured that routing
the whole range through the scaled path costs up to **3.04 digits** on 115 points,
because `hypot` squares with `sqr()` while complex `abs` squares with
`multiply(a,a)` and the two are not interchangeable; KI-9 does nothing to change
that, since it fixed the splitter, not the choice of squaring primitive. The gate
is now a pure accuracy-and-cost choice rather than a defect workaround, which is
the right reason to keep it. Widening it would trade a measured 3.04-digit loss
for no gain, since the direct form inside the gate is already exact there.

**One regression found and fixed on the way — an accidental fallback.** The
KI-7 saturate branch in `sinhcosh` (both backends) computed `e = exp(a)`,
reciprocated it for `a < 0`, and fell back to the accurate `exp(|a| - ln2)` form
only when the result came back inf or NaN. The unfixed `divide()` was *supplying*
that NaN for `|a| ≳ 80`, so the accurate route was being taken by accident.
Repairing division removed the accident and 60 sweep points lost up to **21.9
digits**. The mechanism underneath is real and independent of KI-9: for large
negative `a`, `exp(a)` lands near 1e-35 where the trailing limbs are subnormal or
zero, so it holds barely 10 of QF's 29 digits, and no reciprocal can put them
back. The two routes now switch on where that happens rather than on a NaN:
the last limb of a k-word FP32 expansion sits at `f0 · 2^-24(k-1)`, so it stays
normal while `a > ln(FLT_MIN) + 24(k-1)·ln2` — **-37.4 for QF (k=4), -54.1 for TF
(k=3)**. The thresholds ship at **-40.0f** and **-55.0f**, the derived values with
a margin, because the limbs are not exactly 24 bits apart and the sweep still
favours the reciprocal at a = -37.70 (QF points 703/705/706) and a = -53.41 (TF
point 757). Above the floor the reciprocal is used and is up to 1.0 digit better;
below it the direct form is used and is up to 21.9 digits better.

**Sweep effect.** Monotone gate exit **0** against
`validation/sweep/sweep_baseline.csv` (428,592 points): **0 decreased**, 304
increased, 428,288 unchanged. No accepted decreases. `ctest` 34/34;
`check_standalone_no_kokkos.sh` PASS; `check_domains_fresh.sh` PASS after
regeneration. Triage UNEXPLAINED went **9,348 → 9,238** (−110). The 304 gains
are QF/TF `sinh`/`cosh` at large negative arguments and the QF/TF `div`, `mul`,
`hypot` and complex `abs`/`div` points in the 4.15e34–3.4e38 band; the
UNEXPLAINED drop is smaller than the gain count because many of those points were
already classified OVERFLOW or CONDITIONING rather than UNEXPLAINED.

**No accuracy-test tolerance was ratcheted.** The QF and TF accuracy tests
exercise `sinh`/`cosh` on |x| < 40 and `hypot` on ±1e6, both entirely inside the
untouched region — the split guard fires only above 4.15e34 and the sinhcosh
crossover only below −40 — so no measured row moved and there is nothing to
tighten against.

---

## KI-10 — `fmod` and `remainder` lose half their digits when the operands are many decades apart **[RESOLVED 58fda3b]**

**Severity: medium, all backends.**

### REOPENED 2026-09-03 — FF `fmod` still returns zero correct digits

Re-measured under the ulp gate in step 1c. `remainder` is clean on all four
backends and DD/QF/TF `fmod` are clean, so the iterative reduction that closed
this KI did most of its job. **FF `fmod` did not survive it:**

| backend | worst point | measured ulps | bound | digits |
|---|---|---|---|---|
| FF | x = 40.84 | 1.41e14 | 1.0 | **0.00 of 14** |

Zero correct digits, at a well-conditioned point that is not near a subnormal
band and carries no algorithmic floor — so none of the three exemptions in the
new metric touch it. The brief predicted this KI would look worst under ulps;
it does, though only on one backend of four.

This is the same defect class as KI-15 (below), which shares the fix commit and
is reopened with it. Reopened, **not fixed**.

### What

For `|a/b|` large, both ops drift from the exact answer — which is always
exactly representable, since `a − n·b` is an exact operation for an integral n.

```
backend op        point  inputs                            true result       returned
DD      fmod      94     3.1622776601683792e-07,
                         -2.8336394835590658e-27           4.1487575672e-28  4.1487575672e-28  (12.03 digits)
DD      remainder 67     -3.1622776601683796e-14,
                         2.0698650412754168e-30            5.3277127301e-31 -5.0216124763e-31  (0.00 digits)
```

239 real `remainder` points come back with the *wrong sign*, which for
`remainder` means the wrong multiple `n` was chosen — an off-by-one in the
round-to-nearest of the quotient, not a rounding error.

### Extent

Unexplained `fmod` + `remainder` points: **DD 673, QF 713, TF 838, FF 61.**

### Why it is not conditioning

`fmod` and `remainder` are classified ALGEBRAIC by the triage, meaning the
conditioning probe charges them only the operands' *actual* storage error. For
DD that error is zero — a double input is held exactly — so a correct DD
implementation has nothing to lose here. The 673 DD points are pure algorithm.

### Closing it

Compute the quotient in the extended type, take `n` with the already-fixed
`nint` (see KI-2), and form `a − n·b` with a fused/compensated product so the
subtraction is exact.

**Superseded in part by KI-15.** The 2026-09-03 hard-failure triage found that
these ops also return the wrong *sign*, exact zero, and infinity — 605 further
points — and that there are three distinct mechanisms, not one. KI-15 has the
traced reproducers; close the two together. The remedy above is not sufficient
on its own: no quotient held in the extended type is wide enough, so the fix
has to be an iterative scale-and-subtract reduction.

### RESOLVED 2026-09-03 — commit `fix: KI-10/KI-15 iterative fmod and remainder`, together with KI-15

Both ops in all four backends were rewritten as an exact iterative
scale-and-subtract that never forms a quotient. The algorithm, its loop bound,
its termination argument and its exactness argument are in the comment block
above `xp::detail::dd_fmod_abs` in `include/xp/dd_math.hpp`; the other three
headers point at it. In one line: find the unique `k` with
`B*2^k <= A < B*2^(k+1)`, then for `i = k .. 0` subtract `B*2^i` from the
running remainder whenever it fits. Every scaling is a componentwise power of
two (exact), and every subtraction happens with `B*2^i <= r < 2*B*2^i`, which
is Sterbenz's condition — so no step commits an error at a large intermediate,
which is exactly what the one-shot form could not arrange. Loop bound `k+1`,
at most 2098 iterations for FP64 words and 302 for FP32; `i` decreases by one
per iteration so termination is structural.

Measured mean digits, `|a/b|` sweep, oracle at the backend-rounded operands
(200 random pairs per cell; cap = the backend's representable digits):

| ratio | DD before/after | FF before/after | QF before/after | TF before/after |
|---|---|---|---|---|
| 1e0  | 31.00 / **31.00** | 13.99 / **14.00** | 29.00 / **29.00** | 21.00 / **21.00** |
| 1e6  | 31.00 / **31.00** |  8.51 / **14.00** | 29.00 / **29.00** | 21.00 / **21.00** |
| 1e12 | 31.00 / **31.00** |  2.56 / **14.00** | 29.00 / **29.00** | 10.88 / **21.00** |
| 1e18 | 12.95 / **31.00** |  0.00 / **14.00** | 11.54 / **29.00** |  5.54 / **21.00** |
| 1e24 |  7.18 / **31.00** |  0.00 / **14.00** |  5.90 / **29.00** |  0.06 / **21.00** |
| 1e30 |  1.85 / **31.00** |  0.00 / **14.00** |  0.09 / **29.00** |  0.00 / **21.00** |

`remainder` tracks `fmod` to within 0.05 digits at every cell and is omitted
for width; the same table with `remainder` is reproducible from the harness
described in the commit message. Every "after" figure is the measurement cap,
i.e. the returned value is bit-identical to the binary128 reference — the
reduction is exact, not merely improved.

**FF needed one extra thing.** The span argument says the running remainder
needs (word mantissa) + 2 bits — 55 of DD's 106, 26 of QF's 96, 26 of TF's 72,
but ~50 of FF's 48. FF is one to two bits too narrow to hold its own exact
intermediates, so a two-word loop rounds at every step (measured: 5 of 50 steps
inexact at ratio 1e15, fmod landing at 10.93 of 14). The FF loop therefore runs
the remainder in three floats and rounds to a `FloatFloat` once at the end.
That is the whole of the difference between the 10.93 above and the 14.00 in
the table, and it is why `ff_math.hpp` carries `ff_sub3`/`ff_ge3` that the
other three headers do not need.

**Read-only monotone gate**, `/tmp/sweep --baseline
validation/sweep/sweep_baseline.csv` (baseline deliberately NOT regenerated;
a follow-up session does that with `docs/DOMAINS.md`):

| cell | mean before | mean after | zero-digit points before | after |
|---|---|---|---|---|
| DD fmod      | 26.20 | **31.00** | 64  | **0**   |
| DD remainder | 25.27 | **31.00** | 157 | **0**   |
| FF fmod      |  9.07 | **9.42**  | 399 | **107** |
| FF remainder |  8.54 | **8.92**  | 433 | **242** |
| QF fmod      | 24.00 | **28.80** | 72  | **6**   |
| QF remainder | 23.43 | **28.75** | 81  | **9**   |
| TF fmod      | 16.43 | **21.55** | 223 | **6**   |
| TF remainder | 16.07 | **21.51** | 245 | **9**   |

3403 points increased, 191 decreased, worst drop 6.53 digits. The decreases are
accepted and explained below.

### Accepted decreases

**FF `fmod`/`remainder`, 188 points** (sampled coordinates from
`validation/sweep/sweep_grid.csv`: r/22 `3.1622776601683796e-25`, r/191
`-3.1622776601683795e+17`, r/403 `0.05`, r/495 `4.65`, r/643
`-18.849555921538766`; the second operand is drawn log-uniform over
`[1e-100,1e100]` by `scripts/sweep_accuracy.cpp:559`, so these are extreme-ratio
points). The sweep scores against the oracle at the *double* inputs, while FF
stores only ~48 bits of them. For `fmod` that storage error is amplified by the
full ratio: perturbing `a` by one FF ulp moves `a mod b` by `ulp(a)`, which is
`|a/b|` ulps of the answer. The ceiling is therefore `14 - log10|a/b|` digits
and nothing an implementation does can beat it. Measured on 300 random pairs
per ratio:

| ratio | vs FF-rounded inputs (new) | vs double inputs, old | vs double inputs, new | conditioning ceiling |
|---|---|---|---|---|
| 1e0  | 14.00 | 13.95 | 13.96 | 13.98 |
| 1e3  | 14.00 | 11.54 | 11.94 | 12.54 |
| 1e6  | 14.00 |  8.67 |  9.66 | 12.71 |
| 1e9  | 14.00 |  5.40 |  6.34 | 10.55 |
| 1e12 | 14.00 |  2.34 |  2.97 |  4.82 |

The new code is exact for the operands it is actually given (column 1, 14.00 =
cap, at every ratio) and beats the old code on the double-input score at every
ratio (columns 2 vs 3). The 188 individual decreases are points where the old
code's rounding error happened to cancel part of the storage error; that
cancellation was luck, it was not reproducible, and the cell means moved the
right way (9.07 -> 9.42, 8.54 -> 8.92) while the zero-digit counts collapsed
(399 -> 107, 433 -> 242). Accepted.

**DD `acosh`, 3 points** (r/389, r/415, r/435), each -0.01 digits. Not reachable
from this change — `acosh` does not call `fmod` or `remainder`, and no shared
code was touched. The gate run reports `ORACLE FINGERPRINT MISMATCH — baseline
578322f998a329c8, this run 54901e8104607a77`, i.e. the reference itself has
moved since the baseline was written; hairline diffs of this size are that
drift. Pre-existing, out of scope here, and the follow-up re-baseline session
will absorb it.

### QD 2.3.24 shares the defect

`qd_real.cpp:2597` (`fmod` = `a - b*aint(a/b)`) and `:2462` (`drem` =
`a - b*nint(a/b)`) are the exact shape this entry indicts, with no iteration and
no guard on the quotient's width, so upstream QD is wrong in the same way at
the same ratios. Diverging from the source here is deliberate and is recorded
in each header's comment. QD's `drem` additionally uses `nint`, which is
half-away-from-zero; C99/IEEE-754 `remainder` requires half-to-EVEN, so the
replacement carries the integral quotient's parity out of the reduction loop
and breaks ties on it (see KI-20 — `round` staying half-even is CORRECT and was
not "fixed" here).

### RESOLVED 2026-09-04 — commit `fix: KI-10/KI-15 fmod and remainder under the ulp metric`, together with KI-15

**No library code changed.** The 2026-09-03 reduction was already correct; what
was wrong was the *bound* the ulp metric held it to. Both are settled by
measurement below.

#### What the first fix missed, and why its verification did not catch it

The first fix's accepted-decreases section (above) derived a storage ceiling of
`14 - log10|a/b|` digits. That derivation is **right, and it is not the whole
story** — it is the first-order regime only. It models the storage error as a
*perturbation* of a smooth function, and `fmod`/`remainder` are not smooth: they
jump by `b` whenever the integral quotient `n` changes. The first verification
scored **cell means over 200–300 random pairs per ratio**, and a mean cannot see
a six-point discontinuity population. The ulp gate scores the **worst point per
cell**, so it landed on exactly those six and reported the full `2^48` ulps.

#### The storage-error ceiling, derived

Let `p = sig_bits`. The backend never sees the double operands `(a,b)`; it sees
`A = a(1+α)`, `B = b(1+β)` with `|α|,|β| <= 2^-p`. Only FF is affected: `p = 48
< 53`, so a double does not survive storage. DD (106), QF (96) and TF (72) all
hold a double exactly — `α = β = 0` — which is why all three score 21–31 digits
at the very points FF scores 0.00.

*Regime 1 — `n` unchanged.* With `n` locally constant,

```
R = A - nB = r + a·α - n·b·β        =>   |R - r| <= (|a| + |n b|)·2^-p
```

so the error is at most `(|a| + |n b|) / |r|` ulps of `r`. Since `|r| < |b|`,
that is `~2|a/b|` when `|r| ~ |b|`, i.e. **`digits <= 14.4 - log10(|a|/|r|)`** —
the first fix's ceiling, confirmed. It is also *exactly* `kappa_real`'s
`(|a| + |bn|)/|f|`, so the ulp gate **already charges it** and regime 1 needs no
exemption. Measured: every regime-1 point passes the gate.

*Regime 2 — `n` changes.* `trunc`/`nearbyint` are discontinuous. When `a/b` lies
within `2|a/b|·2^-p` of an integer, `trunc(A/B)` can differ from `trunc(a/b)` by
one, and then

```
R - r = ∓b        regardless of how small the perturbation was
```

which is `2^p·|b/r|` ulps of `r`. Because `|r| <= |b|` always, that is **`>= 2^p`
ulps — zero correct digits — and κ, a first-order quantity, does not bound it
even in principle.** The op comment above `kappa_real` always said so ("at the
jump the function is discontinuous and no first-order bound applies"); nothing
made it testable until now.

#### FF `fmod` at x = 40.84, before and after

All six failing points are in the `i%7 == 1` *cancelling* family, where the
sweep draws `b = a·(1+ε)` (`scripts/sweep_accuracy.cpp:557`):

| grid pt | a | ε = b/a − 1 | 2^-48 | exact a/b → n | stored A/B → n |
|---|---|---|---|---|---|
| 708  | 40.8407044966673    | 2.22e-16 | 3.55e-15 | 0.999999999999999826 → 0 | **1 exactly** → 1 |
| 743  | −50.265482457436704 | 4.44e-16 | 3.55e-15 | 0.999999999999999576 → 0 | **1 exactly** → 1 |
| 883  | −94.247779607693815 | 2.22e-16 | 3.55e-15 | 0.999999999999999849 → 0 | **1 exactly** → 1 |
| 995  | −128.80529879718151 | 2.22e-16 | 3.55e-15 | 0.999999999999999779 → 0 | **1 exactly** → 1 |
| 1548 | 304.73448739820981  | 8.88e-16 | 3.55e-15 | 0.999999999999999067 → 0 | **1 exactly** → 1 |
| 1590 | 1.1000000000000001  | 2.22e-16 | 3.55e-15 | 0.999999999999999798 → 0 | **1 exactly** → 1 |

Every `ε` is **below `2^-48 = 3.55e-15`**, so `round48(a) == round48(b)`
bit-for-bit: a 2×FP32 pair provably cannot hold `a` and `b` as different
numbers. Exact `a/b < 1` gives `n = 0` and `fmod = a`; the stored quotient is
exactly `1`, giving `n = 1` and `fmod = 0`.

| | before (HEAD 7680809) | after |
|---|---|---|
| FF `fmod` at grid pt 708 | 1.40737e14 ulps, 0.00/14 digits, **GATED-FAIL** (region CONDITIONING) | 1.40737e14 ulps, 0.00/14 digits, **exempt** (region JUMP_UNRESOLVED) |
| returned value | `0` | `0` (unchanged — no code changed) |
| oracle at FF's own stored operands | `0` | `0` — **bit-identical** |
| regime-2 ceiling `2^p·|b/r|` | `>= 2^48 = 2.815e14` | same |

The measured error is `2^48` ulps and the ceiling is `>= 2^48` ulps: **the
ceiling is attained, not exceeded.** DD/QF/TF at point 708 score 31.00 / 29.00 /
21.70 digits, as the derivation requires.

#### Nothing is below the ceiling — measured, all four backends

The only defensible standard for an algebraic op is: *the returned value is the
nearest value of the type to `f` evaluated at the backend's own stored
operands.* Anything worse is algorithm, whatever the format is. Two independent
sweeps:

| probe | scope | result |
|---|---|---|
| every `fmod`/`remainder` pair on the sweep grid | 1652 pts × 2 ops × 4 backends = 13,216 | DD/QF/TF **bit-exact at every point**. FF bit-exact at 3,256 of 3,304; the other 48 are inexact **only because the true remainder is not representable as a `FloatFloat`** — for each one the returned value equals the correctly-rounded nearest `FloatFloat`, `self_ulps == repr_ulps` to every printed digit, worst **0.4711 ulp**. |
| randomized stress, operands log-uniform over `1e-35..1e35`, both signs | 200,000 pairs × 2 ops × 4 backends = 1.6M evaluations | DD worst **0.0000** ulp, QF **0.0000**, TF **0.0000** (bit-exact); FF worst **0.5000** ulp (correctly rounded). **0 points over 0.5 ulp on any backend.** |

`fmod`/`remainder` are therefore correctly rounded everywhere tested. There is
no residue to fix and no accuracy-test tolerance to ratchet — no measured value
moved, because no library code changed.

#### The exemption

`scripts/sweep_accuracy.cpp` gains a fourth format-derived region,
**`JUMP_UNRESOLVED`**, alongside `UNDERFLOW`/`OVERFLOW`/`SUBNORMAL_LIMB`. It is
not a distance heuristic: `jump_unresolved()` recomputes `n` at the exact
operands and at the stored operands and fires only when they differ — i.e. only
when the format provably cannot tell which side of the jump the point is on. It
outranks the digit-shaped classifier's label for the same reason
`SUBNORMAL_LIMB` does: it is a property of the FORMAT. The derivation and these
measurements are in the comment above the function.

It fires on **exactly 6 points**, all FF `fmod`, with no collateral: gated
failing points 5451 → 5445, gated cells 62 → 61, and all eight
`fmod`/`remainder` cells are now clean.

#### Gates

* `ctest` — see the KI-15 block (same commit).
* **Monotone gate, read-only** against `validation/sweep/sweep_baseline.csv`:
  1,226 increased, **18 decreased**, worst drop 0.56 digits. The 18 are
  **byte-identical before and after this change** (verified by diffing the
  `--baseline` output of a binary built from HEAD against one built from this
  commit) and are all *complex* ops — `QF/TF/FF c asin/acos/acosh/asinh/pow`.
  A classifier region label cannot move a score; **this batch contributes zero
  decreases.** They are batch 1's already-accepted 18, documented under KI-8.
* Both `--ulp` runs used `LD_LIBRARY_PATH=/soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64`
  so the oracle fingerprint matches `578322f998a329c8`.

#### QD 2.3.24

Re-checked. `src/dd_real.cpp:802` and `src/qd_real.cpp:2598` are still
`fmod(a,b) = a - b*aint(a/b)`, and `src/qd_real.cpp:2462` is
`drem(a,b) = a - nint(a/b)*b`. Upstream QD **shares the original defect** and has
no notion of the jump regime either. The divergence remains deliberate.

---

## KI-11 — The complex inverse family loses most of its digits when one component is far smaller than the other **[RESOLVED 855292d / batch-8]**

**Severity: medium, all backends, 4,759 points. PARTIALLY RESOLVED by commit
`855292d` — `atan`/`atanh` fixed and measured; `asin`/`acos`/`asinh`/`acosh`/
`sqrt` NOT fixed and NOT inherent (see "What is left", below).**

### What

`atan`, `atanh`, `asin`, `acos`, `acosh`, `asinh` and `sqrt` on complex
arguments degrade — often to zero digits — when |Im z| ≪ |Re z| or vice versa,
approaching but not on a branch cut. The small component is the one that is
lost.

```
backend op      point  input          true result                 returned
DD      c atan  444    -100 + 1e-29i  -1.56079666 + 9.9990001e-34i
                                      -1.56079666 - 1.62349876e-33i
DD      c atan  446    -100 + 1e-30i  -1.56079666 + 9.9990001e-35i
                                      -1.56079666 - 1.62349876e-33i
```

The returned imaginary part is **identical for both inputs** and has the wrong
sign: the implementation has hit a noise floor near 1.6e-33 and is no longer a
function of its input at all. This is the `log`-form's cancellation — the term
carrying the small component is a difference of two quantities of size |z|²,
and nothing rescues it.

### Extent

4,759 unexplained points across the seven ops, spread over all four backends;
the largest single cell is complex `atan` (817 partial-digit-loss + 476
no-correct-digits).

### Why it is not conditioning

The probe perturbs each input component independently by the backend's own eps
and re-evaluates the binary128 oracle. For these points the true result barely
moves (`ach` = cap = 31.00 on the rows above), so the mathematics permits a full
31-digit answer here and the implementation is not delivering it.

### Closing it

The standard remedy is Kahan's formulation of the complex inverse functions,
which computes the small component from `log1p`/`atan2` of a rearranged
argument rather than from a cancelling difference. This overlaps the work
already done for KI-5; the cases here are the ones the KI-5 fixes did not cover
because they sit *off* the cut rather than on it.

### What was fixed

`atan` and `atanh` were moved off the ratio form
`(i/2)·log((1−iz)/(1+iz))` onto Kahan's component form, in all four
`include/xp/*_complex.hpp`:

```
Re atan(z)  = 0.5  · atan2(2x, (1−y)(1+y) − x²)
Im atan(z)  = 0.25 · log1p( 4y / (x² + (1−y)²) )
Re atanh(z) = 0.25 · log1p( 4x / ((1−x)² + y²) )
Im atanh(z) = 0.5  · atan2(2y, (1−x)(1+x) − y²)
```

with three supports that the bare formulas need:

* **`xp_log_hypot2(a, b)`** = `log(a² + b²)` as `2·log(s) + log1p((t/s)²)`,
  `s = max(|a|,|b|)`. Used when the `log1p` argument leaves `log1p`'s good
  range — when `4y/den ≥ 1e4` (the quotient then runs into the Dekker splitter
  ceiling of KI-19) or when it approaches −1 (`log1p(−1) = −inf`, which took
  DD `atanh(∓1 + εi)` from 31.00 to 18.89 and QF to NaN in an intermediate
  revision of this fix). The `2·log(s)` form is required rather than
  `log(a²+b²)` because at the branch points `a²+b²` is FP32-subnormal.
* **`xp_atan2_safe(a, b)`** — `atan2`/`angle` forms `hypot(a,b)` internally, so
  raw `1−x²−y²` at |z| ~ 1e10 squares to 1e40 and overflows an FP32 word. Both
  operands are scaled down by a common exact power of two first. Without it,
  sweep point 1628 (`axis`, 1e10 + 0i) went 14.00 → **0.00** on FF/QF/TF.
* **`(1−y)(1+y) − x²` rather than `1 − (x² + y²)`.** The additive form rounds
  `x² + y²` to one word *before* the cancellation, so at |y| ~ 1 — exactly the
  atan cut — the surviving `x²` keeps only word-0 precision. Measured on DD
  `atanh(−1 + 1e−8i)`: 24.72 digits additive, 31.00 factored.
* **C99 Annex G poles.** `catan(±0 ± 1i)` and `catanh(±1 ± 0i)` are infinite.
  No formulation built on the extended `log()` can produce that infinity — it
  prints `non-positive argument` and returns 0 — so both are intercepted at the
  top of the function. At `811e08c` these returned a finite `(0, 0)`.

Measured, per point, against `catanq`/`catanhq` (min over the two components,
capped at the backend width; **before** = `811e08c`, **after** = the fix commit):

```
                         -100      100      2.0      1e8      0.5     1e-20    1e-6
                        1e-29    1e-20    1e-20     1e-8    1e-25       100      -2
DD  atan   before        0.00     8.55    11.80     8.52     7.65     29.18   30.79
           after        31.00    31.00    31.00    31.00    31.00     31.00   30.75
DD  atanh  before       29.24    29.18    30.74    23.30    30.74      8.55   24.36
           after        31.00    31.00    30.87    31.00    30.87     31.00   31.00
FF  atan   before        0.00     0.00    -0.00     0.00    -0.00     12.18   14.00
           after        12.74    14.00    13.98    14.00    13.72     14.00   13.61
FF  atanh  before       12.32    12.18    13.61     7.02    13.61      0.00    7.74
           after        14.00    14.00    13.32    14.00    13.32     14.00   14.00
QF  atan   before        0.00     5.98     9.74     5.54     6.32     27.40   28.02
           after        12.74    21.48    24.85    21.25    27.98     29.00   27.89
QF  atanh  before       27.02    27.40    27.89    21.31    19.25      5.98   21.92
           after        29.00    29.00    28.26    29.00    19.25     21.48   29.00
TF  atan   before        0.00     0.08     2.39     0.00     0.00     19.11   20.77
           after        12.74    21.48    21.70    21.25    21.41     21.70   21.33
TF  atanh  before       18.87    19.11    21.33    14.55    19.61      0.08   15.25
           after        21.70    21.70    21.70    21.70    19.61     21.48   21.70
```

`-0.00` and `-1.00` in the *before* rows are the probe's NaN/inf codes, i.e.
KI-18 territory. The residual 12.74 at `(∓100, ±1e−29)` on the three FP32-word
backends is representability, not formulation: the answer's small component is
~1e−33 and the large one is ~1.56, so a single FP32 word cannot hold both — the
`log1p` argument itself is 1e−31 and lands in the subnormal range.

Sweep-wide effect on the affected cells (read-only monotone gate against
`validation/sweep/sweep_baseline.csv`, oracle fingerprint `578322f998a329c8`):
**10,915 points increased, 1,932 decreased, worst decrease −2.04 digits, none
below 3 digits of loss and none reaching zero.**

### The other five ops — settled by measurement (batch-8, 2026-09-04)

`asin`, `acos`, `asinh`, `acosh` and `sqrt` were left untouched by `855292d`
and their loss was *described* as conditioning without a derivation. It is not
conditioning. Each was probed against the binary128 oracle at its worst point:
kappa = |z f'(z)/f(z)| and the per-component kappa_c = max_j |(d f_c/d in_j)·in_j / f_c|,
both by central difference in binary128, then compared with the measured digits.

| op | worst probe point | failing component | kappa = \|z f'/f\| | component kappa_c | achievable (DD cap 31.00) | measured DD / FF / QF / TF | verdict |
|---|---|---|---|---|---|---|---|
| `asin`  | 0.5 + 1e-30i  | Im | 1.10  | 1.000 | 31.00 − log10(1.00) = **31.00** | 1.55 / 0.00 / 0.21 / 0.00 | **FIXABLE** |
| `acos`  | 0.5 + 1e-30i  | Im (= −Im asin) | 0.551 | 1.000 | **31.00** | 1.55 / 0.00 / 0.21 / 0.00 | **FIXABLE** |
| `asinh` | 1e-25 + 0.5i  | Re | 1.10  | 1.000 | **31.00** | 6.54 / 0.00 / 5.79 / 0.00 | **FIXABLE** |
| `acosh` | 0.5 + 1e-30i  | Re (= \|Im asin\|) | 0.551 | 1.000 | **31.00** | 1.19 / 0.00 / 0.12 / 0.00 | **FIXABLE** |
| `sqrt`  | −0.1 − 0i     | Im (sign)          | 0.500 | 0.500 | **31.30** | 0.00 / 0.00 / 0.00 / 0.00 | **FIXABLE** |

Every component kappa_c is 1.000 or below — a relative eps on the small input
component moves the scored output component by the *same* relative eps, and
log10(kappa) is 0.04 or less. **Nothing here is conditioning. The whole gap was
formulation.** Five for five; none of the split the entry allowed for
("partly fixable, partly inherent") turned out to exist.

**Mechanism 1 — the modulus rounds the answer away (asin, acos, asinh, acosh).**
`Im asin(z) = −log|w|` with `w = iz + sqrt(1−z²)`, and `|w| → 1` for every z near
the real segment [−1,1] — not just on it. At `z = 0.5 + 1e-30i` the true `|w|` is
`1 − 1.155e-30`, so forming `|w|` at all destroys `log10(1/1.155e-30) = 29.9`
digits before `log` is ever entered. `acosh`'s real part and `asinh`'s real part
are the *same quantity* under exact identities, which is why all four move
together.

**Mechanism 2 — `-0.0` fails a `< 0` test (sqrt, and the sheet it feeds).**
`sqrt`'s negative-real-axis branch chose its sheet with `if (z.im.hi < 0.0)`,
which is **false for `-0.0`**. Both zero conventions therefore landed on the `+i`
sheet, so `sqrt(-a - 0i)` returned `+i·sqrt(a)` — the sign of the whole answer
wrong, 0.00 digits, at every negative-real-axis grid point (C99 Annex G requires
`csqrt(-a ∓ 0i) = ∓ i·sqrt(a)`). Reading the sign off `copysign` instead is free
on the two conventions that were already right.

### The fix

Four headers, `include/xp/{dd,ff,qf,tf}_complex.hpp`, identically.

**`xp_asin_imag_mag(x, y)`** — Hull, Fairgrove & Tang (1997), *Implementing the
complex arcsine and arccosine functions using exception handling*, ACM TOMS
23(3):299–335. With `x = |Re z|`, `y = |Im z|`,

```
r = hypot(x+1, y),   s = hypot(|x-1|, y),   a = (r + s)/2,   |Im asin z| = acosh(a)
```

exactly — and `a → 1` is precisely the lossy region, so the code carries `a − 1`
and never `a`. Because `r` and `s` are exact distances,
`r = (x+1) + y²/(r + (x+1))` and `s = |x−1| + y²/(s + |x−1|)` hold identically,
giving

```
a - 1 = (max(x,1) - 1) + (1/2)*( y²/(r+(x+1)) + y²/(s+|x-1|) )
```

— a sum of **non-negative** terms at every x, so there is no cancellation left
to lose anything to. `acosh(1+m) = log1p(m + sqrt(m)·sqrt(m+2))` then keeps it,
through the real `log1p` KI-5(b) rebuilt.

Two implementation details that were each worth digits and are commented in the
source:

- The two `y²` terms are carried as `v = (1/d1 + 1/d2)/2`, so `sqrt(a−1)` is
  formed as `y·sqrt(v)` — **y is never squared**. `y²` goes subnormal in an FP32
  word below `|y| ~ 1e-19` and flushes to zero below `~1e-22`; the first cut of
  this fix did square, and took QF `asin` at `0.5 + 1e-30i` to `-0.00`.
- `sqrt(m·(m+2))` is evaluated as `sqrt(m)·sqrt(m+2)`, two separate roots, so
  the product cannot overflow at any `|z|`.

**`acosh`'s imaginary part** now takes its sign from `copysign(Im z)` on the
magnitude. `acosh(conj z) = conj acosh(z)` and the principal strip is
`Im ∈ (−π, π]`, so `sign(Im acosh z) = sign(Im z)` everywhere, signed zeros on
the cut included. Kahan's chain dropped that `-0.0` on the FP32-word backends:
QF and TF returned `acosh(-0.1 - 0i) = +1.670964i`, wrong sheet, 0.00 digits,
while DD and FF happened to keep it.

`acos` was **not** edited — it already delegates its imaginary part to
`negate(asin(z).im)`, so it inherits the fix.

### After

Probe (binary128 oracle, worst point per op, min of the two components):

| op | point | DD before → after | FF | QF | TF |
|---|---|---|---|---|---|
| `asin`  | 0.5 + 1e-30i | 1.55 → **31.00** | 0.00 → **13.97** | 0.21 → **14.68** | 0.00 → **14.68** |
| `acos`  | 0.5 + 1e-30i | 1.55 → **31.00** | 0.00 → **13.97** | 0.21 → **14.68** | 0.00 → **14.68** |
| `acosh` | 0.5 + 1e-30i | 1.19 → **31.00** | 0.00 → **13.97** | 0.12 → **14.68** | 0.00 → **14.68** |
| `asinh` | 1e-25 + 0.5i | 6.54 → **31.00** | 0.00 → **13.97** | 5.79 → **19.50** | 0.00 → **19.50** |
| `sqrt`  | −0.1 − 0i    | 0.00 → **31.00** | 0.00 → **14.00** | 0.00 → **29.00** | 0.00 → **21.70** |
| `acosh` | −0.1 − 0i    | 0.00 → **30.71** | 0.00 → **14.00** | 0.00 → **29.00** | 0.00 → **21.70** |
| `asinh` | 0 − 10i      | 0.00 → **30.57** | 0.00 → **14.00** | 0.00 → **28.28** | 0.00 → **21.36** |

**The QF/TF numbers at `1e-30` and `1e-25` are the FORMAT, and this time with the
derivation.** A QF/TF value of magnitude 2^e stores word k at magnitude
2^(e−24k), and any word below 2^−126 is an FP32 subnormal, so the usable
significand is `min(24·nwords, e + 126)` bits. At the answer's magnitude
1.155e-30 (e ≈ −100) that is 24 + 26 = 50 bits = **15.0 digits** (measured
14.68); at 1.155e-25 (e ≈ −83), 24 + 43 = 67 bits = **20.2 digits** (measured
19.50). Both are at the ceiling. DD's FP64 words have exponents down to 2^−1022
and so do not hit this at all — hence 31.00 across the board.

### Sweep-wide effect (batch-8)

Read-only comparison of the full 428,592-point sweep, this change alone (the
same binary run before and after the header edits, so no oracle-fingerprint
drift enters):

**14,474 points increased, 5,126 decreased.** Of the increases, **1,329 went
from zero correct digits to non-zero** and **2,679 gained 10 digits or more**.
Of the decreases, 1,029 give up ≥ 0.5 digit and 222 give up ≥ 1.0; the worst is
−5.45. Every decreased cell is in the complex inverse family (`asin`, `acos`,
`asinh`, `acosh`) except 8 `DD c log10` and 2 `DD c pow` points, which route
through the same `sqrt` sheet.

Worst decreases:

| cell | pt | family | z | before | after |
|---|---|---|---|---|---|
| QF c asinh | 1554/1555 | axis | −1e−21 ± 0i | 29.00 | 23.55 |
| QF c asin / acos | 748 | cut-re | 0 + 1e−21i | 29.00 | 23.55 |
| QF c asinh | 1536–1539 | axis | ±1e−17 ± 0i | 29.00 | 27.25 |
| QF c asin / acos | 740/741 | cut-re | 0 ± 1e−17i | 29.00 | 27.25 |
| TF c acosh | 237 | polar | −0.91461657 + 0.61112726i | 21.52 | 19.97 |
| QF c acosh | 129 | polar | 0.97097743 + 0.19313942i | 28.52 | 26.98 |

The −5.45 group is the same subnormal-limb ceiling derived above, seen from the
other side: at `|z| = 1e-21` the answer needs a fourth QF word at ~2e−43, which
is subnormal, so **any** general computation there is capped near 24 digits. The
old code scored 29.00 only because its shortcut returned the input expansion
*unchanged* — exact by luck of the identity `asinh(y) ≈ y`, not by carrying the
digits. The remaining ~5,000 decreases are sub-digit rounding churn from a
formula with more operations (two `hypot`, two `divide`, a `sqrt`, a `log1p`)
replacing one complex `log`.

Judgement: 5,126 points give up a median well under 0.5 digit — and four points
give up 5.45 to a documented format floor — so that 14,474 points improve,
1,329 of them from zero. Recorded, not silently taken.

### Residual, filed separately

FF complex `asin` at `z = 1e8 + 1e-8i` returns **0** for its real part (true
value π/2). This is not the small-component loss: `sum = iz + sqrt(1−z²)`
cancels to `(0, 0)` in FF's two FP32 words and `atan2(0,0) = 0`. It predates
this change (the old `log(sum).im` had the identical cancellation) and is
FF-only. Filed as **KI-32**.

### Accepted decreases (`855292d`, the `atan`/`atanh` half)

All are ≤ 2.04 digits and all sit on the `polar` family (|z| = 1, where
`1 − x² − y²` is an exact cancellation of two O(1) quantities, so the atan2
argument keeps only the rounding of the *inputs*; the ratio form kept that
cancellation in a single subtraction instead of two). Worst cells:

| cell | pt | family | z | before | after |
|---|---|---|---|---|---|
| QF c atanh | 150 | polar | −0.37885660 − 0.91464074i | 28.73 | 26.69 |
| QF c atan | 158 | polar | 0.91464074 − 0.37885660i | 28.91 | 27.05 |
| QF c atan | 219 | polar | 0.56112594 − 0.83978431i | 28.89 | 27.14 |
| FF c atanh | 137 | polar | −0.19313942 + 0.97097743i | 14.00 | 12.51 |
| DD c atan | 177 | polar | −0.98078528 − 0.19509032i | 31.00 | 29.66 |
| TF c atanh | 316 | polar | 7.07106781 − 7.07106781i | 21.66 | 20.51 |

Judgement: 1,932 points give up at most 2.04 digits so that 10,915 points —
including 4,500-odd that returned zero correct digits or NaN — come back at
full width. Recorded, not silently taken.

---

## KI-12 — FF `sincos` never converges below |x| ≈ 1e-20, and its bail-out returns WITHOUT writing its out-params **[RESOLVED 0ad44fe / 2e810c2]**

**Severity: high (silent 0 and silent NaN from ordinary in-range inputs), FF
only, 1,020 points — the single largest hard-failure root cause in the sweep.**

### What

`sincos` (`include/xp/ff_math.hpp:734`) runs a Taylor loop with a fixed
convergence test

```cpp
const float eps = 1.0e-15f;                                  // ff_math.hpp:736
...
if (detail::fabs(sterm.hi) < eps * detail::fabs(sin_r.hi) &&
    detail::fabs(cterm.hi) < eps) break;                     // ff_math.hpp:759
if (k == itrmx) { XPMATH_PRINTF("FFCSSNR: iteration limit\n"); return; }  // :761
```

Two independent faults compound:

1. **`eps = 1e-15f` is finer than FloatFloat's own resolution** (u = 2^-48 =
   3.55e-15). This is the same DD→FF port artefact already recorded for `exp`
   in the T2.3/B3 work — it was never swept out of `sincos`.
2. **The iteration-limit bail is a bare `return`.** `x` and `y` are never
   assigned, so the caller sees its default-constructed `FloatFloat(0,0)`.

For |a| ≲ 1e-20 the second Taylor term underflows FP32 to exactly 0, and the
right-hand side `eps * |sin_r.hi|` underflows to 0 as well, so `0 < 0` is false
forever: the loop always runs to `itrmx` and always bails. `sin` and `cos` then
return 0, and everything that divides by them returns NaN.

### Reproducer

```
$ g++ -std=c++17 -O2 -Iinclude repro.cpp && ./a.out
FFCSSNR: iteration limit
FF sin(1e-30)   = 0                    true 1.0000000000000001e-30
FFCSSNR: iteration limit
FF cos(1e-30)   = 0                    true 1
FFCSSNR: iteration limit
FF tan(1e-30)   = -nan                 true 1.0000000000000001e-30
FFCSSNR: iteration limit
FF atan(1e-30)  = -nan                 true 1.0000000000000001e-30
FF sin(1e-19)   = 1.0000000000000005e-19   (converges — above the cliff)
```

`atan` is collateral: `atan(a) = angle(FloatFloat(1), a)`
(`ff_math.hpp:827`), and `angle` (`ff_math.hpp`, the `sincos`-Newton loop)
divides by `cos_a`, which is the 0 that `sincos` just handed back —
`corr = divide(subtract(target, sin_a), cos_a)` → 0/0 → NaN.

### Extent

1,020 UNEXPLAINED FF points returning 0 or NaN. Direct: `sin` 6, `cos` 6,
`tan` 68+32, `atan` 50. Propagated through `angle`/`sincos` into the complex
layer: `polar` 40, `exp`/`sinh`/`cosh` 36 each, `sin`/`cos`/`tan` 32 each,
`asinh` 56, `asin` 36+9, `atanh` 30, `atan` 28, `pow` 24, `log` 24, `log10` 24,
`acos` 18+5, `acosh` 18. Every one of these has `repr` = `range` = 14.00 and
most have `ach` = 14.00 — the point is fully representable and fully
well-conditioned.

### Why it is not an inherent limit

1e-30 is an ordinary normal FP32 value (FLT_MIN = 1.18e-38) and sin(x) = x to
within FF's resolution there. There is nothing to compute; the routine simply
fails to notice it has already converged.

### Closing it

Three changes, any one of which removes most of the population: set `eps` to
the type's own resolution (2^-48) rather than a hard-coded 1e-15f; make the
bail-out assign `x` and `y` before returning; and short-circuit `|a| < sqrt(u)`
to `(cos, sin) = (1, a)`. Audit `qf_math.hpp`, `tf_math.hpp` and `dd_math.hpp`
for the bare-`return` bail — only the FF `eps` is mis-scaled, but the
uninitialised-out-param pattern should be checked in all four.

### RESOLVED 2026-09-04 — commit `fix: KI-12 sincos out-params; KI-14 integer ops past 2^47; KI-27 multiply overflow`

**Most of this entry was already closed by the PREVIOUS batch, commit `0ad44fe`
(KI-19/KI-25/KI-26) — not by this one.** That was verified before anything was
touched. `ff_math.hpp:sincos` already carried, under its KI-25 comment, both
halves of the "Closing it" list above: the `itrmx` arm's bare `return` had
already become `break`, and the vacuous `<` convergence test had already become
`<=`. Measured at `0ad44fe`, the headline symptoms are clean:

| x | FF `sin` @ `0ad44fe` | FF `cos` | FF `tan` | FF `atan` |
|---|---|---|---|---|
| 1e-20 | 9.999999682655225388968e-21 | 1 | 9.999999682655225388968e-21 | 9.999999682655225388968e-21 |
| 1e-30 | 1.000000003171076850971e-30 | 1 | 1.000000003171076850971e-30 | 1.000000003171076850971e-30 |

Both out-params written, both correct, no NaN. This batch adds nothing there.

#### What was still broken: the degenerate reduction band

Below the convergence problem sits a second, independent one that `0ad44fe` did
not touch. `sincos` scales its reduced argument by `r = s3 / 2^nq` before the
first Taylor term. Once `|s3| < 2^nq · FLT_MIN` the leading word of `r` is
**subnormal**, so `r` sheds mantissa bits before the series ever runs; for
subnormal `|a|` it underflows to 0 and the answer collapses. `dd_math.hpp`
already guarded the `r.hi == 0` half of this (`if (r.hi == 0.0)`), which is why
DD was correct here and FF/QF/TF were not.

| x | quantity | `0ad44fe` | after | correct |
|---|---|---|---|---|
| 1e-40 | FF/QF/TF `sin` | 9.999665841421894618112e-41 | **9.999946101114759581526e-41** | exact — the FP32 operand for 1e-40 is 9.999946101114759581526e-41 and `sin` of it equals it far beyond FF/QF/TF resolution |
| 1e-40 | FF `atan` | 1.000078688019335447177e-40 | **9.999946101114759581526e-41** | same |
| 1e-44 | FF/QF/TF `sin` | **0** | **9.809089250273719496466e-45** | exact (subnormal operand) |
| 1e-44 | FF `atan` | 1.681558157189780485108e-44 | **9.809089250273719496466e-45** | same |
| 1e-40, 1e-44 | all four `cos` | 1 | 1 | exact — a²/2 is under FLT_TRUE_MIN |
| 6e-08 | FF `cos` | 0.9999999999999982000001 | 0.9999999999999982000001 | unchanged (see below) |

The fix is a short circuit to `(cos, sin) = (1, a)` over exactly that band —
`|a| < 2^nq · FLT_MIN` for the FP32-limbed backends, `|a| < 2^nq · DBL_MIN` for
DD — applied to all four `sincos` bodies. DD keeps its `r.hi == 0` guard as
redundant documentation of the same corner.

#### The threshold: two wider cuts were tried and both cost accuracy

The entry above suggests short-circuiting at `|a| < sqrt(u)`. **That is wrong,
and it was measured to be wrong** — recorded here because it is the natural
first guess:

1. **A cut at the unit roundoff** (2^-25 for FF, i.e. `sqrt(u)` for u = 2^-48).
   A double-word near 1 is not a 48-bit format: `FloatFloat(1.0f, -e)` is a
   legal normalized pair for any `e` down to FLT_TRUE_MIN, so `cos(a) = 1 - a²/2`
   is representable EXACTLY while a²/2 ≥ 2^-149, and the series does return it.
   Measured: this cut turned FF `cos(1e-8)` from 1 - 5e-17 into 1, and FF
   `sin(1e-8)` from `a - a³/6` into `a`.
2. **A cut at 2^-75**, where a²/2 first falls under FLT_TRUE_MIN. `(1, a)` is
   genuinely the correctly-rounded pair there, and for a REAL argument the
   series already returns exactly that anyway — `r²` underflows to zero, so
   `sin_r = r`, `cos_r = 1`, and the nq exact doublings reproduce `(a, 1)` bit
   for bit. It is **not** a no-op for the COMPLEX callers, though. Scored
   against the sweep grid it moved **132 complex points down by up to 1.87
   digits** (worst: QF `c asin` 1398/1399 and QF `c asinh` 566/567,
   20.07 → 18.20; QF `c tan` 1274/1275, 17.64 → 16.01) while every gain from
   this commit stayed put. A cut that can only tie on its own domain and loses
   money downstream is not worth taking.

The band actually shipped is the one where the series demonstrably cannot do
better, and it produces **zero** sweep decreases (536 increases).

#### The out-parameter audit

Every early-return path in every multi-output function, all four backends —
25 paths across 8 functions:

| function | early-return paths | wrote both out-params @ `0ad44fe` | after |
|---|---|---|---|
| DD `sincos` | 6 (`a.hi==0`, small-arg, `\|a\|≥1e60`, `s3.hi==0`, `r.hi==0`, codomain) | yes, all 6 | yes |
| FF `sincos` | 5 (`a.hi==0`, small-arg, `\|a\|≥1e30f`, `s3.hi==0`, codomain) | yes, all 5 — the `itrmx` bare `return` that made this a HIGH was already gone at `0ad44fe` | yes |
| QF `sincos` | 5 (same shape) | yes, all 5 | yes |
| TF `sincos` | 3 (`a.f0==0`, small-arg, codomain) | yes, all 3 | yes |
| DD `sinhcosh` | 1 (saturate) | yes | yes |
| FF `sinhcosh` | 2 (Taylor, saturate) | yes | yes |
| QF `sinhcosh` | 2 (Taylor, saturate) | yes | yes |
| TF `sinhcosh` | 1 (saturate) | yes | yes |

Confirmed two ways: by reading every `return;` inside those bodies, and by a
poison-seeded probe (out-params pre-set to −12345 and checked after each call)
over x ∈ {0, 1e-44, 1e-40, 1e-30, 1e-20, 6e-8, 0.25, 1, 25, 400, ±1e30, ±1e300},
both signs. No path leaves an out-param unwritten, before or after.

#### One thing this batch changed that the entry did not ask for

FF and DD `sincos` had ASYMMETRIC large-argument fast paths — `a.hi >= 1.0e30f`
and `a.hi >= 1.0e60`, positive only — so `+1e30` took the fast path and `-1e30`
fell all the way through the reduction to the codomain guard. Both produced
`(1, 0)`, so this was never a wrong answer, but two code paths were reaching the
same result with nothing making them agree. Both now test `fabs`.

---

## KI-13 — `angle`, `asinh` and `acosh` still form an UNSCALED sum of squares, so they NaN at |x| ≳ 1.8e19 **[RESOLVED 7680809]**

### RESOLVED 2026-09-04 — commit `fix: KI-8 scale hypot/abs at BOTH ends; KI-13 remaining unscaled sums of squares`

(Referenced by subject, not sha: this doc ships INSIDE that commit, so its own
sha is not knowable while it is being written, and the session was constrained
to a single commit.)

Same root cause as KI-8, three different call sites. All three now rescale by
an EXACT power of two (`*_pow2_unit_scale` / `*_pow2_scale` in each
`*_math.hpp`), which changes no bit of an operand.

- **`angle`** — `atan2` is exactly scale-invariant, `atan2(ys, xs) = atan2(y, x)`
  for any `s > 0`, so one rescale of both operands is a free fix. Applied on the
  **HIGH side only**, and that asymmetry is deliberate: `r` is used only to
  normalise `(x/r, y/r)` before a 3-step Newton refinement that re-evaluates
  `sin`/`cos` of the iterate, so low words shed from `r` do not survive into the
  answer — rescaling at the low end only perturbs the seed. Measured: a
  two-sided gate cost **45 regressions** (worst 1.27 digits, QF c atan points
  1254/1318) to buy **35 improvements**. The high side is a real fix; without it
  the square OVERFLOWS and `atan`/`atan2`/`asin`/`acos`/complex `arg` return 0
  or NaN outright.
- **`asinh`** — above the gate, `asinh(a) = log(a) + log(1 + sqrt(1 + 1/a²))`,
  which forms `1/a²` (tiny, harmless) instead of `a²` (overflowing). TF's
  `asinh` has no odd reflection, so its branch works on `abs(a)` and re-applies
  the sign.
- **`acosh`** — likewise `acosh(a) = log(a) + log(1 + sqrt(1 - 1/a²))`.

**Before/after** (digits vs the `__float128` oracle; `atan` is the `angle`
call site):

| backend | op | 1e19 | 1e20 | 1e24 | 1e30 | 1e35 | 1e38 |
|---|---|---|---|---|---|---|---|
| FF | atan / asinh / acosh | 14.00 | **0.00 → 14.00** | **0.00 → 14.00** | **0.00 → 14.00** | **0.00 → 14.00** | **0.00 → 14.00** |
| TF | atan / asinh / acosh | 21.70 | **0.00 → 21.70** | **0.00 → 21.70** | **0.00 → 21.70** | **0.00 → 21.70** | **0.00 → 21.70** |
| QF | atan | 29.00 | **0.00 → 29.00** | **0.00 → 29.00** | **0.00 → 29.00** | **0.00 → 29.00** | **0.00 → 29.00** |
| QF | asinh / acosh | 27.51 → **28.93** | **0.00 → 27.18** | **0.00 → 23.37** | **0.00 → 17.97** | **0.00 → 12.36** | **0.00 → 8.93** |

DD, on its own ladder: `atan`, `asinh` and `acosh` at 1e160, 1e200 and 1e300 all
went **0.00 → 31.00**. Small magnitudes are unchanged for every backend.

**Residual, NOT closed by this commit and NOT started (out of scope — one KI at
a time).** Three separate defects surfaced while probing; each has a different
mechanism from KI-13's premature squaring, and each is a strict improvement over
the pre-fix 0.00/NaN:

1. **QF `asinh`/`acosh` decay above the band** — 27.18 at 1e20 falling to 8.93
   at 1e38. The reformulated branch is `log(a) + log(1 + sqrt(1 - 1/a²))`, whose
   second term is fine; the loss tracks QF `log()` at large argument. Not
   diagnosed further.
2. **FF `atan` returns 0.00 for x ≤ 1e-30** — small-argument path, not a sum of
   squares.
3. **FF `asinh` holds only ~7–8 of 14 digits for tiny x** — classic `log(1+a)`
   cancellation at the LOW end; KI-13's mechanism is at the high end.


**Severity: high, FF/QF/TF (the float-word backends), 402 real points.**

### What

KI-8 added scaling to `hypot` and complex `abs`. Three other call sites that
form `x² + y²` or `a² ± 1` were not touched:

```cpp
FloatFloat r = sqrt(add(multiply(x,x), multiply(y,y)));   // angle(), ff_math.hpp
return log(add(a, sqrt(add(multiply(a, a), FloatFloat(1.0f)))));   // asinh, :894
FloatFloat t1 = subtract(multiply(a, a), FloatFloat(1.0f));        // acosh, :898
```

`a²` overflows the FP32 word at |a| = sqrt(FLT_MAX) = 1.844e19, five decades
short of the 3.4e38 the format actually reaches. `sqrt(inf)` and `log(inf)`
then produce NaN.

### Reproducer

```
FF atan (3.1622776601683796e19) = -nan   true 1.5707963267948966
QF atan (3.1622776601683796e19) = -nan   true 1.5707963267948966
TF atan (3.1622776601683796e19) = -nan   true 1.5707963267948966
DD atan (1e19)                  = 1.5707963267948966        (DD's limit is 1.3e154 — off-grid)
FF asinh(3.1622776601683796e19) = -nan   true 45.593556493943836
FF acosh(3.1622776601683796e19) = -nan   true 45.593556493943836
QF asinh(1e20)                  = -nan   true 46.744849040440859
TF asinh(1e20)                  = -nan   true 46.744849040440859
```

`atan` reaches `angle` via `atan(a) = angle(FloatFloat(1), a)`, so `atan2`,
`acos`, `asin` and complex `arg` inherit the same ceiling.

### Extent

402 UNEXPLAINED real points returning NaN: FF/QF/TF × {`atan`, `asinh`,
`acosh`} at 44 points each, plus 6 stragglers. DD is clean here only because
its own overflow point is outside the sweep grid — the defect is in the shared
algorithm, not in the FF words.

### Why it is not a range limit

`range` = `repr` = `ach` = cap on every one of these rows. atan(3.16e19) is
π/2; asinh(1e20) is 46.74. Both results are tiny and exactly the kind of value
the backend is for. The classifier tested the input against the word range
first and it passed.

### Closing it

The same remedy KI-8 used: for `asinh`/`acosh`, switch to
`log(2a) + 1/(4a²) − …` above a threshold, or scale by `a` before squaring; for
`angle`, divide both components by `max(|x|,|y|)` before forming the sum of
squares, exactly as the fixed `hypot` now does.

---

## KI-14 — FF `floor`, `ceil`, `trunc` and `round` return ZERO for every |x| ≥ 2^47 **[RESOLVED 2e810c2]**

**Severity: high (returns 0 for an input that is already an exact integer),
FF only, 194 points.**

### What

`round_to_nearest_int` guards its double-precision reduction with a magnitude
cap and returns **zero** past it:

```cpp
if (detail::fabs(total) >= 1.40737488355328e14 /* 2^47 */) {
    XPMATH_PRINTF("FFNINT: argument too large\n");
    return FloatFloat(0.0f);                            // ff_math.hpp:544-547
}
```

`floor`, `ceil`, `trunc` and `round` (`ff_math.hpp:1021`, `:1026`, `:1031`,
`:1034`) are all thin wrappers over it, so all four inherit the cap.

The guard itself is defensible — the reduction is done in a `double` and 2^47
is where that stops being exact. The bug is the *return value*: every
FloatFloat with |x| ≥ 2^47 is already an exact integer, so the correct answer
is `a` itself, unchanged.

### Reproducer

```
FFNINT: argument too large
FF floor(1e15) = 0     true 1000000000000000
FFNINT: argument too large
FF trunc(1e15) = 0     true 1000000000000000
FF round_to_nearest_int(3.1622776601683794e14) = 0   true 316227766016838
QF floor(1e30) = 1e+30      TF floor(1e30) = 1e+30      DD floor(1e30) = 1e+30
```

The three other backends are correct at the same magnitudes, which isolates
this to the FF guard.

### Extent

194 UNEXPLAINED FF points returning 0: `round` 66, `trunc` 64, `floor` 32,
`ceil` 32. All at |x| ≥ 3.16e14, `ach` = `repr` = `range` = 14.00.

### Closing it

Change the four bail-outs to `return a;`. Note that this guard is *also* the
mechanism behind part of KI-15 — `fmod` calls `trunc(q)` and `remainder` calls
`round_to_nearest_int(q)` on a quotient that routinely exceeds 2^47.

### RESOLVED 2026-09-04 — commit `fix: KI-12 sincos out-params; KI-14 integer ops past 2^47; KI-27 multiply overflow`

#### The mechanism — it is NOT a 32/48-bit integer cast

The 2^47 number invites the guess that something is being cast to a 48-bit
integer. It is not. **2^47 is a vestige of the routine's PREVIOUS body.**
Before KI-2, `round_to_nearest_int` rounded with the FP64 magic-constant form
`(total + 2^52) - 2^52`, which stops rounding at all once `|total|` reaches
2^51; 2^47 was the margin someone picked below that, and the bail-out returned
`FloatFloat(0.0f)` because there was no fallback to fall back to. The KI-2
rewrite replaced the magic constant with `detail::rint`, which is exact at
EVERY magnitude — and left the cap, and its `return FloatFloat(0.0f)`, sitting
in front of it. The guard has been dead weight in front of a correct
implementation ever since.

#### The filed scope was incomplete: DD is affected too

The entry says "FF only, 194 points". **DD has the same zero-returning bail**,
at 2^105 = 4.06e31 instead of 2^47, and it is additionally **one-sided**:
`a.hi >= T105` let every large NEGATIVE argument through, so `floor(-2^105)`
was right while `floor(+2^105)` was 0 and `ceil(+2^105)` was 1. Unlike FF's,
DD's cap is *real* — the DDFUN magic constant genuinely stops discarding the
fraction there — so only the return value was wrong, not the cap.

QF and TF are **correct** at every magnitude: both use QD's component-wise
`nint`, which has no cap.

#### Before / after

`x` here is the FF/QF/TF operand (the nearest representable value), which is
why the 1e15 and 1e30 rows do not read as round decimals.

| x | backend | `floor` / `ceil` / `trunc` / `round` @ `0ad44fe` | after |
|---|---|---|---|
| 2^46 = 7.03687e13 | FF, QF, TF, DD | 70368744177664 (all four ops, correct) | unchanged |
| 2^47 = 1.40737e14 | **FF** | **0 / 1 / 0 / 0** + `FFNINT: argument too large` | **140737488355328** (all four) |
| 2^47 | QF, TF, DD | 140737488355328 (correct) | unchanged |
| 2^48 = 2.81475e14 | **FF** | **0 / 1 / 0 / 0** | **281474976710656** |
| 2^60 = 1.15292e18 | **FF** | **0 / 1 / 0 / 0** | **1152921504606846976** |
| 2^60 | QF, TF, DD | correct | unchanged |
| 1e15 | **FF** | **0 / 1 / 0 / 0** | **999999986991104** |
| 1e30 | **FF** | **0 / 1 / 0 / 0** | **1.000000015047466219877e+30** |
| 2^104 = 2.02824e31 | **FF** | **0 / 1 / 0 / 0** | **2.028240960365167042395e+31** |
| 2^105 = 4.05648e31 | **FF** and **DD** | **0 / 1 / 0 / 0** + `DDNINT: argument too large` | **4.056481920730334084789e+31** |
| 2^106 = 8.11296e31 | **FF** and **DD** | **0 / 1 / 0 / 0** | **8.112963841460668169579e+31** |
| −2^47, −2^60, −2^105, −2^106 | **FF** | **−1 / 0 / 0 / 0** | correct, sign preserved |
| −2^105, −2^106 | DD | already correct (the cap was one-sided) | unchanged |
| 2^60 + 0.5 (non-integral, far-apart limbs) | **FF** | **0 / 1 / — / 0** | **1152921504606846976 / …977 / — / …976** |
| 2^106 + 0.5 | **DD** | **0 / 1 / — / 0** | **8.112963841460668169579e+31** (all three) |

Tie semantics are untouched: 0.5 → 0, 1.5 → 2, 2.5 → 2 on DD/FF (ties-to-even,
KI-20), 0.5 → 1, 2.5 → 3 on QF/TF, and 0.49999997 → 0 everywhere, all identical
before and after.

#### The fix

Both bails now return `a.hi + rint(a.lo)` instead of zero. That is **exact**,
not a clamp: at those magnitudes `ulp(a.hi)` is at least 2, so `a.hi` is already
an exact EVEN integer and every fractional bit lives in `a.lo`. Ties-to-even on
the low word is therefore ties-to-even on the whole value, and adding the two
limbs is a `two_sum` and loses nothing.

#### Effect on the gates

Sweep: FF `ceil`, `floor`, `round` and `trunc` each gain **64 points**, no
decreases. ULP gate: the FF `ceil`, `floor` and `trunc` `UNEXPLAINED` cells go
away entirely — 53 → 50 gated cells, 5,299 → 5,043 gated points. FF `round`
remains a gated-fail cell for the separate half-to-even reason (KI-20).

---

## KI-15 — `fmod`/`remainder` do not merely lose digits: they return the WRONG SIGN, ZERO, or INFINITY **[RESOLVED 58fda3b]**

**Severity: high, all backends, 605 hard-failure points on top of the 2,285
soft ones already recorded as KI-10.**

### REOPENED 2026-09-03 — with KI-10, same cell, same fix commit

KI-10 and KI-15 were closed by one commit (`fix: KI-10/KI-15 iterative fmod and
remainder`). The step-1c ulp re-measurement leaves FF `fmod` at 1.41e14 ulps —
0.00 of 14 digits — at x = 40.84 (see KI-10's reopen block for the table).
Because that is a returned value with no correct digits, it cannot be
distinguished from this KI's wrong-value class by measurement alone, so both
are reopened together rather than guessing which of the two the residue belongs
to. Whoever closes them should re-run the hard-failure scan in this KI's
original evidence section, not just the ulp gate. Reopened, **not fixed**.

### What

KI-10 records these two ops as "losing half their digits". The sweep's
hard-failure triage shows the failure is worse than that, and that it has three
distinct mechanisms, not one. Both ops are two lines
(`ff_math.hpp:986`/`:992`, and the same shape in `dd_`/`qf_`/`tf_math.hpp`):

```cpp
FloatFloat q = divide(a, b);
FloatFloat qt = trunc(q);                 // or round_to_nearest_int(q)
return subtract(a, multiply(b, qt));
```

**(a) The reducer's magnitude cap fires.** On FF, `trunc(q)` for |q| ≥ 2^47
returns 0 (KI-14), so the result is `a − b·0 = a` — the input, unreduced.

**(b) The quotient does not fit the type.** `a/b` is held to the backend's
nominal digits, but `fmod` needs the *integral part* of it exactly. When
|a/b| exceeds what the type can hold as an integer, `trunc(q)` is not the true
floor and `a − b·trunc(q)` is not the remainder:

```
TF fmod(316227.76601683797, 1.3460648851493211e-20)
  q          = 2.3492758001911501e+25
  trunc(q)   = 2.3492758001911501e+25       (already integral — nothing to truncate)
  b*trunc(q) = 316227.76601683797           (== a exactly)
  fmod       = 0                             true 3.4955874873628251e-21
```

```
DD fmod(1e19, -5.583642853032536e-30)
  trunc(a/b) = -1.7909454926131054e+48       (49 digits; DD holds 31)
  fmod       = -2.8421709430404007e-14       true 4.5351364837126992e-30
```

**(c) `divide` itself returns NaN** on QF/TF when the quotient is large — see
KI-19. `fmod` then propagates the NaN.

### Reproducer / extent

605 UNEXPLAINED `fmod`+`remainder` points that are not digit loss:
wrong-sign 317, zero ~200, inf 54, plus NaN. Split by backend:
DD 137+11+2, TF 78+61+81+77+15+12, QF 14+11+29+27+15+12, FF 14+6.
Every one of the 317 wrong-sign points is `fmod` or `remainder`; no other op in
the sweep produces a wrong sign at all.

### Why it is not conditioning

`fmod` and `remainder` are ALGEBRAIC in the triage, so the probe charges them
only the operands' real storage error — zero for a DD point built from a
double. `a − n·b` is exactly representable for integral n. The DD points are
pure algorithm.

### Closing it

Do not form the quotient in one shot. Reduce iteratively: scale `b` up by
2^k until it straddles `a`, subtract, and repeat — the textbook exact-fmod
loop, which never needs a quotient wider than the type. This subsumes KI-10;
KI-10 and this entry should be closed together.

### RESOLVED 2026-09-03 — commit `fix: KI-10/KI-15 iterative fmod and remainder`, together with KI-10

Done as described. The algorithm, loop bound, termination and exactness
arguments live in the comment above `xp::detail::dd_fmod_abs`
(`include/xp/dd_math.hpp`); the KI-10 entry carries the measured before/after
tables and the accepted decreases. Each of this entry's three mechanisms was
re-tested explicitly, on all four backends, against `fmodq`/`remainderq`:

| mechanism | probe | result |
|---|---|---|
| (a) reducer magnitude cap — `trunc(q)` returns 0 for FF `|q| >= 2^47`, so the input came back unreduced | `fmod(1e30, 1e-30)` and the full sign matrix `(+-1e30, +-1e-30)` on DD/FF/QF/TF | PASS all 4 backends, 16 cases. No reducer is called at all now — the loop never forms `q`. |
| (b) quotient wider than the type — `TF fmod(316227.76601683797, 1.3460648851493211e-20)` returned exact 0, `DD fmod(1e19, -5.583642853032536e-30)` returned -2.84e-14 | both reproducers verbatim, plus `fmod(1e22,3)`, `fmod(1e5,1e-25)`, and the KI-10 header rows r/94 and r/67 | PASS all 4 backends. TF now returns 3.495587e-21 (true 3.495587e-21), DD 4.535136e-30 (true 4.535136e-30). |
| (c) `divide` returns NaN at large quotients on QF/TF (KI-19) and `fmod` propagates it | same extreme-ratio cases on QF and TF | PASS. `divide` is no longer on the path, so KI-19 cannot reach these two ops. |

**Wrong sign — the 317 points.** The sign matrix `fmod(+-1e30, +-1e-30)` and
`remainder(+-1e30, +-1e-30)`, 16 cases per backend, all return the correct sign
and the correct magnitude to the cap. The convention is now explicit and
enforced by construction rather than by whatever the quotient happened to
round to: `fmod` reduces `|a| mod |b|` and re-applies the sign of `a`;
`remainder` does the same and then makes one half-even correction step, so its
sign is a property of the answer and not of `a`.

**Wrong zero — the ~200 points.** Distinguished from correct zero. `fmod(7.5,
2.5)` returns `+0` and `fmod(-7.5, 2.5)` returns `-0` (correct: `a` is an exact
multiple of `b`, C99 gives zero with the sign of `a`), while the KI-15(b) TF
reproducer that used to return exact 0 now returns its true 3.495587e-21. PASS
on all four backends.

**Wrong infinity — the 54 points.** No path in the new bodies can produce an
infinity from finite operands: the only arithmetic is power-of-two scaling
bounded by `|a|` and Sterbenz-conditioned subtraction, and there is no division
and no multiplication. `fmod(a, +-inf)` returns `a`, which is C99-correct and
finite.

**Zero and infinity conventions implemented** (C99 7.12.10.1/7.12.10.2,
IEEE 754-2019 5.3.1), verified on all four backends:
`fmod(a,0)` and `remainder(a,0)` -> NaN with a diagnostic;
`fmod(+-inf,b)` and `remainder(+-inf,b)` -> NaN with a diagnostic;
`fmod(a,+-inf)` = `remainder(a,+-inf)` = `a` for finite `a`;
`fmod(+-0,b)` = `remainder(+-0,b)` = `+-0` for `b != 0`;
a NaN operand propagates. Half-even ties verified:
`remainder(5,2) = 1`, `remainder(3,2) = -1`, `remainder(1,2) = 1`,
`remainder(-3,2) = 1`.

Total across the four backends: 0 failures on 41 probes each.

### RESOLVED 2026-09-04 — commit `fix: KI-10/KI-15 fmod and remainder under the ulp metric`, together with KI-10

**No library code changed.** The residue that reopened this entry is not a
wrong value — it is the format's, and KI-10's resolution block carries the
derivation, the ceiling and the measurements. This block records the
hard-failure re-scan the reopen asked for.

#### What the first fix missed, and why its verification did not catch it

Nothing, on this entry's own terms: all three mechanisms stayed fixed. What the
first *verification* could not do was tell "lost all precision" from "returned
the wrong value" at 0.00 digits — which is precisely why the reopen bundled the
two entries and asked for the hard-failure scan to be re-run rather than the
ulp gate alone. It has been. The answer is that FF `fmod` at grid point 708
returns `0` **and `0` is the correct `fmod` of the two operands FF actually
holds**, bit-identically. It is a correct value for the wrong-by-storage inputs,
not a wrong value.

One further gap in the first verification, found and closed here: its evidence
table checked `fmod(-7.5, 2.5)` for signed zero — which passes for the wrong
reason, since the sign is re-applied from a *nonzero* `a` — but never checked a
**zero dividend**. `fmod(±0, b)` and `remainder(±0, b)` are now tested directly
and are correct on all four backends. (Watch the harness here: the sign of a
zero result cannot be read off the summed limbs, because `-0.0 + 0.0 = +0.0` in
IEEE. It must be read off the **leading word**. A first pass of this re-test
reported four spurious failures for exactly that reason.)

#### Re-test of every failure mode, all four backends

Reference `fmodq`/`remainderq` at the backend's **own stored operands**;
tolerance bit-exact, or `<= 0.5 ulp` where the true value is not representable
in the type. **516 checks, 0 failures.**

| mode | probes | result |
|---|---|---|
| **wrong sign** | full sign matrix `(±1e30, ±1e-30)`, both ops; `ratio ladder` `(±98.765, ±1.2345e-k)` for k over 11 decades `1e0…1e30`, both ops — 88 signed cases | PASS. `fmod` sign always `sign(a)`; `remainder` sign a property of the answer. |
| **returned zero** | KI-15(b) reproducers verbatim (`TF fmod(316227.766…, 1.346e-20)`, `DD fmod(1e19, -5.5836e-30)`), `fmod(1e22,3)`, `fmod(1e5,1e-25)`, KI-10 rows r/94 and r/67 | PASS. No spurious zero; every true-nonzero result is nonzero and correctly rounded. |
| **correct zero, correct sign** | `fmod(7.5,2.5) = +0`, `fmod(-7.5,2.5) = -0`, and **new:** `fmod(±0,3)`, `remainder(±0,3)` | PASS on all four backends (sign read off the leading word). |
| **returned infinity** | the whole ladder plus the extreme-ratio cases; any non-finite result from finite operands is a failure | PASS. 0 infinities from finite operands across 1.6M stress evaluations. |
| **(c) `divide` NaN (KI-19) propagating** | `fmod`/`remainder` at `(1e38, 1e-38)` and `(1e20, 1e-21)`, quotient ~1e41 (the case KI-9's grid never probed) | PASS on QF and TF. `divide` is still not on the path. |
| **C99 / IEEE-754 conventions** | `f(a,0) → NaN`; `f(±inf,b) → NaN`; `f(a,±inf) = a`; NaN propagation; `\|fmod\| < \|b\|`; `sign(fmod) = sign(a)`; `\|remainder\| <= \|b\|/2` — asserted on **every** probe, not just the convention cases | PASS, 4 backends. |
| **half-EVEN ties for `remainder`** | `1.5/1 → -0.5`, `2.5/1 → +0.5`, `0.5/1 → +0.5`, `3.5/1 → -0.5`, `7.5/5`; plus `remainder` sign not tied to `a` (`0.9/1 → -0.1`, `-0.9/1 → +0.1`) | PASS. Half-even is CORRECT here per IEEE 754-2019; **KI-20 (`round` half-even vs QD's half-away `nint`) is a separate open issue and was not touched.** |

Plus the 1.6M-evaluation randomized correctly-rounded stress tabulated in
KI-10's block: worst 0.0000 ulp on DD/QF/TF, 0.5000 ulp on FF, **0 points over
0.5 ulp anywhere**.

#### Gates

`ctest` **34/34**. Monotone gate and the ulp gate before/after: see KI-10's
block (same commit). QD 2.3.24 re-checked and still shares the original defect;
see KI-10's block.

---

## KI-16 — `atanh`'s domain guard tests only the LEADING word, so it rejects arguments that are strictly inside (−1, 1) **[RESOLVED batch-5]**

**Severity: medium, FF and QF, 46 points.**

### What

```cpp
if (detail::fabs(a.f0) >= 1.0f) { XPMATH_PRINTF("QFATANH: |argument| >= 1\n");
                                  return QuadFloat(0.0f); }   // qf_math.hpp:1298
if (detail::fabs(a.hi) >= 1.0f) { ... return FloatFloat(0.0f); }  // ff_math.hpp:904
```

The guard reads `a.f0` / `a.hi` — the *leading float word* — not the value the
multi-word type represents. For `a = 0.99999999999999978` the leading FP32 word
rounds to exactly `1.0f`, the guard trips, and the function returns 0 even
though the represented value is comfortably inside the domain and the answer is
18.37.

### Reproducer

```
QFATANH: |argument| >= 1
QF atanh(0.99999999999999978) = 0     true 18.368400284838551
FFATANH: |argument| >= 1
FF atanh(0.99999999999999978) = 0     true 18.368400284838551
```

### Extent

46 UNEXPLAINED points returning 0: QF `atanh` 42, FF `atanh` 4, all in the
`ulp` family within an ulp of ±1. `ach` = 14.5–14.8 against a QF cap of 29 —
the point is genuinely ill-conditioned to about half the type's digits, but 14
digits is not 0.

### Closing it

Compare against the reconstructed value (`a.f0 + a.f1 + …`, or the type's own
`>=` operator against `QuadFloat(1.0f)`) rather than the leading word. The same
leading-word pattern appears in `acos`/`asin`'s range check
(`ff_math.hpp:820`) and in `acosh`'s `a.hi < 1.0f` (`ff_math.hpp:899`) and
should be audited with it.

---

### Resolution — batch-5

Fixed by giving every ±1 domain guard a VALUE-based comparison. Each backend
gained a two-line helper next to `asin`:

```cpp
XPMATH_INLINE_FUNCTION int dd_cmp_one(DoubleDouble a) {
    const DoubleDouble d = subtract(a, DoubleDouble(1.0));
    if (d.hi > 0.0) return  1;
    if (d.hi < 0.0) return -1;
    return 0;
}
XPMATH_INLINE_FUNCTION int dd_cmp_abs_one(DoubleDouble a) { return dd_cmp_one(abs(a)); }
```

`subtract` is the exact-expansion difference, so the sign of its leading word is
the sign of the whole value — the comparison is exact at full format width. The
one-word test `detail::fabs(a.hi) >= 1.0` is both a false-reject (arguments just
inside the interval whose leading word rounds to 1.0) and a false-accept (the
mirror case just outside).

Converted, in all four backends: `asin`, `acos` (`|a| > 1`), `acosh` (`a < 1`),
`atanh` (`|a| > 1`), and DD/FF `zeta` (`s <= 1`). TF's `asin`/`acos` also had a
separate one-word test for the exact-±1 short-circuit
(`abs_a.f0 == 1.0f && abs_a.f1 == 0.0f && abs_a.f2 == 0.0f`); it now reuses the
single `tf_cmp_one` result.

**Two findings beyond the reported scope.**

1. TF had NO `acosh` and NO `atanh` domain guard at all. `|a| >= 1` fell through
   to a log of a non-positive quotient, so it printed `TFLOG`'s diagnostic and
   returned 0 by accident rather than by decision. Both guards added.

2. `|x| == 1` is NOT a domain error for `atanh` — it is the C99 Annex G pole,
   `atanh(±1) = ±inf`, which `dd_complex.hpp`'s `catanh` already honoured. The
   shipped DD/FF/QF guards rejected it along with `|x| > 1` and returned 0. TF,
   having no guard, reached ±inf by accident and was the only backend correct
   there — the dense sweep caught this immediately when the new guard regressed
   4 TF grid points from 21.70 digits to 0.00. All four now split the cases:

   ```cpp
   const int c_atanh = dd_cmp_abs_one(a);
   if (c_atanh > 0)  { XPMATH_PRINTF("DDATANH: |argument| > 1\n"); return DoubleDouble(0.0); }
   if (c_atanh == 0) return DoubleDouble(a.hi > 0.0 ? HUGE_VAL : -HUGE_VAL);
   ```

   Measured after: `DD/FF/QF/TF atanh(+1) = inf`, `atanh(-1) = -inf`. This lifts
   DD, FF and QF from 0.00 digits at those grid points; TF is restored.

**Audit of every domain guard in all four backends** (the reported defect's
pattern, checked everywhere it could recur):

| guard | compared against | read one word? | action |
|---|---|---|---|
| `sqrt` negative | `a.hi < 0` | no — exact | none needed |
| `log`, `log2`, `log10` non-positive | `a.hi <= 0` | no — exact | none needed |
| `pow` non-positive base | `a.hi == 0 && b.hi > 0` | no — exact | none needed |
| `npwr` zero base, negative exponent | `a.hi == 0`, `n` is an `int` | no — exact | none needed |
| `atan2` both zero | leading words `== 0` | no — exact | none needed |
| `asin`, `acos` out of range | `fabs(a.hi) > 1` | **YES** | value-based |
| `acosh` below 1 | `a.hi < 1` | **YES** | value-based (added for TF) |
| `atanh` outside (−1,1) | `fabs(a.hi) >= 1` | **YES** | value-based + pole (added for TF) |
| DD/FF `zeta` `s <= 1` | `s.hi <= 1` | **YES** | value-based |

The zero-comparisons are exact and need no change: for a normalized expansion
`sign(value) == sign(f0)` whenever `f0 != 0`, and `f0 == 0` implies the value is
zero. Only comparisons against a NONZERO constant can disagree with the value,
which is why the ±1 family is exactly the affected set.

**Measured, `atanh` just inside ±1** (every row's leading word rounds to exactly
`1.0f`, so every row was rejected before):

| argument | FF before | FF after | QF before | QF after |
|---|---|---|---|---|
| 1 − 2.2e−16 | 0.00 d / 2.815e14 ulp | 15.40 d / 0.11 ulp | 0.00 d / 7.923e28 ulp | 30.55 d / 0.02 ulp |
| 1 − 1.0e−15 | 0.00 / 2.815e14 | 15.37 / 0.12 | 0.00 / 7.923e28 | 30.59 / 0.02 |
| 1 − 1.0e−12 | 0.00 / 2.815e14 | 14.85 / 0.40 | 0.00 / 7.923e28 | 29.44 / 0.29 |
| 1 − 1.0e−09 | 0.00 / 2.815e14 | 14.79 / 0.45 | 0.00 / 7.923e28 | 30.25 / 0.04 |
| 1 − 1.0e−08 | 0.00 / 2.815e14 | 14.94 / 0.32 | 0.00 / 7.923e28 | 28.54 / 2.31 |
| 1 − 1.0e−07 | 0.00 / 2.815e14 | 14.55 / 0.79 | 0.00 / 7.923e28 | 28.89 / 1.03 |

FF's cap is 14.00 digits and QF's is 29.00, so every row is at or above full
format precision.

## KI-17 — TF `asinh` has no odd-symmetry branch, so negative arguments cancel to a non-positive log **[RESOLVED batch-5]**

**Severity: medium, TF only, 9 hard points plus the negative half of TF's soft
`asinh` population.**

### What

DD, FF and QF all open `asinh` with `if (a.hi < 0) return negate(asinh(negate(a)));`.
TF does not:

```cpp
XPMATH_INLINE_FUNCTION TripleFloat asinh(TripleFloat a) {
    return log(add(a, sqrt(add(sqr(a), TripleFloat(1.0f)))));   // tf_math.hpp:1188-1190
}
```

For a ≪ 0, `sqrt(a²+1) ≈ |a|` and `a + |a|` is a catastrophic cancellation that
lands on or below zero, so `log` sees a non-positive argument and bails.

### Reproducer

```
TFLOG: non-positive argument
TF asinh(-1e12) = 0                    true -28.324168296488494
TF asinh( 1e12) = 28.324168296488494   (correct — positive side is fine)
```

The asymmetry between the two lines is the whole diagnosis.

### Extent

9 UNEXPLAINED TF `asinh` points returning 0, at a = −1e12, −3.16e12, −3.16e14
and below; plus the negative-argument share of TF's soft `asinh` loss. FF/QF/DD
`asinh` do not fail this way at any magnitude.

### Closing it

Add the same odd-symmetry guard the other three backends already have. One
line.

---

### Resolution — batch-5

One line at the top of `tf_math.hpp`'s `asinh`:

```cpp
if (a.f0 < 0.0f) return negate(asinh(negate(a)));                   // KI-17
```

Negation is a sign flip on every word, so the reflection is exact and free. The
`kTFSqHi` branch below used to re-apply the sign itself; with the reflection in
front of it, it only ever sees `a >= 0`, so that bookkeeping is gone.

**Checked in the other three backends, as asked:** the REAL `asinh` in
`dd_math.hpp`, `ff_math.hpp` and `qf_math.hpp` each already opened with the same
`if (leading word < 0) return negate(asinh(negate(a)));`. TF was the sole
omission. (KI-5(a) fixed the COMPLEX case separately.)

**Measured, TF `asinh` at negative arguments** (cap 21.70 digits):

| x | before | after |
|---|---|---|
| −1e−12 | 13.81 d | 24.78 d / 0.0008 ulp |
| −1e−06 | 16.47 | 22.41 / 0.18 |
| −0.3 | 20.37 | 20.82 / 7.17 |
| −1 | 21.64 | 20.43 / 17.4 |
| −7 | 18.67 | 22.34 / 0.22 |
| −1e+03 | 18.67 | 22.02 / 0.45 |
| −1e+06 | 17.04 | 22.40 / 0.19 |
| −1e+12 | **0.00 / 4.722e21 ulp** | 22.79 / 0.077 |
| −3.16e+12 | **0.00 / 4.722e21** | 22.62 / 0.11 |
| −3.16e+14 | **0.00 / 4.722e21** | 22.21 / 0.29 |
| −1e+19 | 0.00 | 22.23 / 0.28 |
| −1e+30 | 0.00 | 22.54 / 0.14 |

Across the whole 1,652-point real grid, TF `asinh` goes from mean 19.989 to
mean 21.447 digits: 750 points improve, 92 decrease. Verified separately:
`asinh(-x)` is now BIT-EXACTLY `-asinh(x)` at x = 1, 0.3, 7, 1e6, 1e12.

The two rows that lose ground (−0.3 and −1) are the KI-29 mid-band scatter,
filed below: they now score exactly what `asinh(+1)` and `asinh(+0.3)` score,
which is the correct odd-symmetric behaviour, not a new defect.

## KI-18 — Complex `tan`/`tanh` and complex `atan`/`atanh` have no asymptotic branch, so they NaN where the true result is bounded **[RESOLVED 855292d]**

**Severity: medium, all backends, 172 + 96 points. RESOLVED by commit `855292d`.**

### What

**tan / tanh.** For large |Im z| (tan) or large |Re z| (tanh) the true result
converges to ±i or ±1 and is perfectly bounded, but the implementations form
`sinh`/`cosh` of the large component directly and overflow inside the
intermediate:

```
DD c tan(9807.8528 + 1950.90322i) = -nan-nani   true -2.27682598e-1695 + 1i
FF c tanh(-100 + 1e-29i)          = -inf-nani   true -1 + 5.53558611e-116i
```

The DD row's real part is below binary128's own floor, so the only recoverable
answer is `(±0, ±1)` — and that is exactly what a saturating branch would
return. Note `kFFHyperbolicSaturate = 19.0f` (`ff_math.hpp:822`) already does
this for *real* `tanh` (the KI-7 fix); the complex path never got it.

**atan / atanh at the cut endpoints.** `atan(z) = (i/2)·log((i+z)/(i−z))`
divides by a quantity that goes to zero as z → ±i, so the whole result NaNs
rather than just the component that is genuinely ill-determined:

```
QF c atan(1e-19 + 1i)   = nan + 22.221132i     true 0.785398163 + 22.221132i
TF c atanh(1 + 1e-19i)  = 22.221132 + nani     true 22.221132 + 0.785398163i
```

The *other* component comes back correct, which shows the routine got that far
and then divided by a computed zero.

### Extent

172 UNEXPLAINED complex `tan`/`tanh` NaN points (DD/FF/QF/TF `tan` 32 each,
FF `tanh` 44); 96 complex `atan`/`atanh` NaN points (FF 28+30, QF 24+24,
TF 24+24). The atan/atanh cases sit adjacent to KI-11's population but are a
different failure — KI-11 loses the small component, this loses everything.

### Closing it

For `tan`/`tanh`, saturate on the large component the way real `tanh` already
does. For `atan`/`atanh`, use Kahan's formulation (already the stated remedy
for KI-11), which computes the divergent component from `log1p` of a
rearranged argument instead of from a vanishing denominator.

### Resolution

**`tan`/`tanh`: doubled-angle asymptotic branch with the exponential factored
out.** For `tan(x + iy)` with `t = e^(−2|y|)`,

```
tan(x+iy) = [ 2t·sin2x  +  i·(1 − t²)·sgn y ] / (1 + t² + 2t·cos2x)
```

and `tanh` is the same with the roles of the two components swapped. Nothing
ever exponentiates the large component upward, so nothing overflows; `t`
underflowing to 0 is the *correct* limit (`±i` for `tan`, `±1` for `tanh`)
rather than a NaN. The denominator `1 + t² + 2t·cos2x ≥ (1−t)² > 0`, which also
removes the `(−inf, NaN)` that KI-12's zero-returning `sincos` used to produce
through the old `sin(z)/cos(z)` quotient.

**Threshold, per backend.** The branch is taken on `|Im z| ≥ kXpTanAsymptote`
(`tan`) / `|Re z| ≥ kXpTanAsymptote` (`tanh`). This is *not* set from the
exponent range — the asymptotic form never approaches the exponent range, and
setting it there would leave the whole intermediate zone on the overflowing
path. It is set at the crossover where the asymptotic form measures at least as
accurate as the direct quotient, established by walking a ladder at
y = 1, 1.5, 2, 2.5, 3, 4, 5 on each backend:

| backend | `kXpTanAsymptote` | why not 1.0 |
|---|---|---|
| DD | 2.0 | at y = 1 the direct quotient still wins |
| FF | 4.0 | at y = 1 FF `tan(1+1i)` fell 14.00 → 13.29 with a 1.0 threshold |
| QF | 2.0 | as DD |
| TF | 2.0 | as DD |

**`atan`/`atanh`:** the KI-11 component form (see that entry) — the divergent
component now comes from `log1p` of `4y/(x² + (1−y)²)` or from `xp_log_hypot2`,
neither of which divides by a computed zero. The C99 Annex G poles
`catan(±0 ± 1i)` and `catanh(±1 ± 0i)` are intercepted explicitly and return
the correct signed infinity; at `811e08c` they returned a finite `(0, 0)`.

### Re-test — every recorded NaN case, all four backends

Values from `catanq`/`ctanq`/`catanhq`/`ctanhq`. No cell returns NaN.

| point | DD | FF | QF | TF | true |
|---|---|---|---|---|---|
| `tan(9807.8528 + 1950.90322i)` | (0, 1) | (0, 1) | (0, 1) | (0, 1) | (−2.2769e−1695, 1) |
| `tan(1 + 50i)` | (6.7653e−44, 1) | (7.0065e−44, 1) | (7.0065e−44, 1) | (7.0065e−44, 1) | (6.7653e−44, 1) |
| `tan(2 + 1e5i)` | (0, 1) | (0, 1) | (0, 1) | (0, 1) | (−0, 1) |
| `tanh(−100 + 1e−29i)` | (−1, 5.5356e−116) | (−1, 0) | (−1, 0) | (−1, 0) | (−1, 5.5356e−116) |
| `tanh(1950.90322 + 9807.8528i)` | (1, 0) | (1, 0) | (1, 0) | (1, 0) | (1, −2.2769e−1695) |
| `tanh(1e5 + 2i)` | (1, 0) | (1, 0) | (1, 0) | (1, 0) | (1, −0) |
| `atan(1e−19 + 1i)` | FULL | FULL | FULL | FULL | (0.785398163, 22.221132) |
| `atan(0 + 1i)` | (0, inf) | (0, inf) | (0, inf) | (0, inf) | (0, inf) |
| `atanh(1 + 1e−19i)` | FULL | FULL | FULL | FULL | (22.221132, 0.785398163) |
| `atanh(1 + 0i)` | (inf, 0) | (inf, 0) | (inf, 0) | (inf, 0) | (inf, 0) |

The zeros that remain are the correct representable answer, not a failure:
−2.2769e−1695 is below binary128's own floor, and 5.5356e−116 is far below the
FP32 minimum subnormal 1.4e−45, so `0` is the only value a 2×/3×/4×FP32
expansion can carry. `tan(1 + 50i)`'s 7.0065e−44 versus 6.7653e−44 is 1.45
digits and is likewise the FP32 subnormal grid, not the formulation: `e^(−100)`
is 3.7e−44, four binades above the smallest subnormal.

### Accepted decreases

The threshold move is not free just below the crossover. All decreases are
≤ 1.76 digits and all sit on `cut-re`/`cut-im` at |large component| = 2…10:

| cell | pt | family | z | before | after |
|---|---|---|---|---|---|
| QF c tan | 1436 | cut-im | 1e−13 + 10i | 25.06 | 23.30 |
| QF c tanh | 490 | cut-re | −10 + 1e−20i | 17.95 | 16.28 |
| TF c tan | 1214 | cut-im | 1e−30 − 2i | 14.92 | 13.41 |
| TF c tanh | 574 | cut-re | −2 + 1e−30i | 14.92 | 13.41 |
| DD c tan | 1350 | cut-im | 0.01 + 2i | 30.95 | 30.35 |
| DD c tanh | 542 | cut-re | −2 + 1e−14i | 31.00 | 30.67 |

Totals across all four `tan`/`tanh` cells: 448 points decreased (worst −1.76),
1,244 increased, and 172 NaN points became finite and correct.

---

## KI-19 — QF/TF `divide` returns NaN where `inf` is correct when the quotient OVERFLOWS the format **[RESOLVED 0ad44fe]**

> **Retitled 2026-09-04.** This entry used to be titled *"KI-9 is NOT fully
> closed: QF `divide` still returns NaN once the quotient reaches ~1e41"*. With
> both entries marked resolved, that title left a reader unable to tell which
> superseded which — the answer being neither: they are two defects at two
> different magnitudes. The title now names **this** entry's own defect, and
> [KI-9](#ki-9--qftf-division-returns-nan-when-the-quotient-exceeds-the-dekker-splitters-headroom-resolved-82427f6)
> carries a forward pointer here. No finding, measurement or scope below was
> changed.

**Severity: high (contradicts a RESOLVED entry), QF *and TF and DD* (scope
corrected below), surfaced through `fmod`.**

### What

KI-9 records QF/TF division returning NaN when the quotient outruns the Dekker
splitter, and is marked RESOLVED at `82427f6`. It is better, but not closed —
the ceiling moved, it did not disappear:

```
QF divide(1e30,  1e-11) = -nan      quotient 1e41
QF divide(1e20,  1e-21) = -nan      quotient 1e41
QF divide(1.0,   1e-41) = -nan      quotient 1e41
QF divide(-1e13, 1.4568618853408916e-28) = -nan
```

Three different operand pairs with three different operand magnitudes and the
same quotient all fail, which isolates the trigger to the quotient rather than
to either operand. QF's own range reaches 3.4e38 per word and the type is
routinely used well past 1e41 in the accumulated form.

### Extent

Reached in the sweep through `fmod`/`remainder` (see KI-15c) rather than through
`div` directly — the sweep's `div` grid does not place a point at a quotient of
1e41. That is itself worth noting: **the sweep grid does not currently probe
this region for `div`**, so the residual escaped the KI-9 re-baseline.

### Closing it

Re-open KI-9. Establish the exact quotient at which the fixed path still
saturates, then extend the sweep's `div` grid to cover it so the next
re-baseline cannot miss it again.

### Resolution — commit `fix: KI-19/KI-25/KI-26 correct non-finite signalling in div, atan and trig` (2026-09-04)

**Why KI-9's verification missed this.** KI-9 scaled division past the Dekker
splitter limit and was verified with `divide(x, x)`, where the quotient is
always 1. That exercise varies *operand* magnitude across the whole range and
holds *quotient* magnitude fixed at a single value. The defect lives entirely in
quotient magnitude, so the verification could not have caught it no matter how
wide the operand sweep was. **The earlier KI-9 verification tested operand
magnitude but never quotient magnitude.**

**Scope correction.** Filed as QF-only. Re-verification shows the defect on
**QF, TF and DD**. DD was the worst of the three and was not previously
suspected — it lost *representable* quotients, not merely overflowing ones:

```
DD divide(1e301, 1.0)    = NaN     true 1e301, representable
DD divide(1e300, 1e-10)  = NaN     true 1e310, overflow -> should be +inf
DD divide(1.0,   0.0)    = NaN     should be +inf
DD divide(1.0,   inf)    = NaN     should be +0
```

FF already carried the correct three-zone structure from B8/B9/B10 and needed no
change; it is the model the other three now follow.

**The fix.** Two shapes, chosen per backend by what its division algorithm can
support.

- *DD* gets the full FF three-zone treatment in `divide` and `divide_scalar`: a
  non-finite-divisor guard; Zone A (quotient inside the splitter's reach) left
  bit-for-bit untouched; Zone B, where the quotient is representable but exceeds
  the splitter threshold `DBL_MAX/(split+1) = 1.3393857e300`, prescaled by an
  exact power of two (`2^-64`) and unscaled at the end, which recovers cases
  like `1e301/1.0`; Zone C, genuine overflow, returns the correctly signed
  `±inf` that the hardware `a.hi/b.hi` already produced.
- *QF* and *TF* use QD long division, whose first quotient digit `q0 = a.f0/b.f0`
  is already the IEEE-correct answer for every special case. A three-line guard
  short-circuits on it before the splitter is ever reached:

  ```cpp
  q0 = a.f0 / b.f0;
  if (!detail::isfinite(q0) || q0 == 0.0f) return QuadFloat(q0);
  ```

  This is deliberately *not* a scaling fix: once `q0` is `±inf` or `±0`, no
  amount of refinement can produce anything else, and the remaining digits would
  only manufacture the `inf - inf = NaN` that caused the original symptom.

**Contract now honoured on all four backends**: quotient overflow gives `±inf`
with the correct sign; quotient underflow gives `±0` with the correct sign;
`x/0` gives `±inf`; `NaN` appears only for `0/0` and `inf/inf`. The signed zero
was verified on the *leading word's* sign bit — an earlier probe summed all
component words and `-0.0f + 0.0f = +0.0` silently masked a correct `-0`.

---

## KI-20 — `round` is half-to-even, not half-away-from-zero **[OPEN]**

**Severity: low, DD/QF/TF, 4 points.**

### What

`round(x)` is specified by C99 as round-half-*away-from-zero*. All four
backends implement it as `round_to_nearest_int`, which on DD is the
magic-constant form

```cpp
if (a.hi > 0.0) return subtract(add(a, CON), CON);   // dd_math.hpp, round_to_nearest_int
```

— i.e. round-half-to-**even**, inherited from the hardware rounding mode.

```
DD round( 0.5) = 0    true  1
DD round(-0.5) = 0    true -1
```

### Extent

4 UNEXPLAINED points: DD 2, QF 1, TF 1, all at exactly ±0.5. The population is
tiny only because ±0.5 is the sole tie the sweep grid happens to land on; every
half-integer tie behaves the same way.

### Closing it

Give `round` its own implementation — `trunc(x) + copysign(1, x)` when the
fractional part is exactly ±0.5 — and leave `round_to_nearest_int` (which is
correctly half-to-even, and is what the argument reductions want) alone. This
is deliberately *not* KI-2, which was about `nint` one ulp below a tie.

---

## KI-21 — The missing-complex-oracle path is documented as graceful degradation but is a hard build failure **[RESOLVED batch-8]**

**Severity: medium (build/packaging, not numerics). Surfaced by S7 while
standing up CI against a vanilla Kokkos install.**

### What

`CMakeLists.txt:66-71` probes the Kokkos install for the local extension header
`impl/Kokkos_ComplexQuadPrecisionMath.hpp` (carried in `patches/`, not upstream)
and, when it is absent, emits a `message(WARNING)` whose own text describes the
posture as *"warn and continue — same graceful-degradation posture used above
for LIBQUADMATH itself"*.

It does not continue. The four `src/demo_*complex.cpp` files include that header
unconditionally, so configure succeeds with a warning and then the **build
fails**:

```
src/demo_complex.cpp:18:10:    fatal error: impl/Kokkos_ComplexQuadPrecisionMath.hpp: No such file or directory
src/demo_ff_complex.cpp:56:10: fatal error: impl/Kokkos_ComplexQuadPrecisionMath.hpp: No such file or directory
src/demo_qf_complex.cpp:51:10: fatal error: impl/Kokkos_ComplexQuadPrecisionMath.hpp: No such file or directory
src/demo_tf_complex.cpp:51:10: fatal error: impl/Kokkos_ComplexQuadPrecisionMath.hpp: No such file or directory
```

The result probed the `KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER` variable is stored
in is never consulted again — no target is guarded by it.

### Extent

Reproduced on `main` @ `35fd2c9` against a from-scratch Kokkos 5.1.0 built with
`-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_LIBQUADMATH=ON -DCMAKE_CXX_STANDARD=20`
and **no patch applied**. Because `make` stops at the first failing target,
27 of the 34 ctest targets are never linked and ctest reports them `***Not Run`
— i.e. the visible symptom is 27 missing tests, which reads like a test-harness
problem rather than a missing header.

This is invisible in day-to-day work because
`$HOME/kokkos-install-quadmath` has had the patch applied since T0.0. It only
appears when someone builds against a stock Kokkos — which is precisely what a
new contributor, a packager, or CI does first.

### Why it matters beyond CI

The upstream pitch is that this repo builds against an ordinary Kokkos. Today it
does not: it builds against Reet's patched one, and the failure mode gives no
hint that `patches/README.md` is the answer.

### Closing it

Either honour the documented posture — guard the four complex demo targets (and
only those) on `KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER`, so a stock Kokkos yields a
smaller but working build — or drop the pretence and make the missing header a
`message(FATAL_ERROR)` that names `patches/README.md`. The present middle
ground is the only option that is actively misleading.

**Worked around, not fixed, in CI:** `.github/workflows/ci.yml` copies
`patches/kokkos_complex_quad_math.hpp` into the Kokkos source tree before
configuring, and then asserts the header reached the install tree. With that
copy in place the lane is 34/34.

### Fixed (batch-8, 2026-09-04)

The first of the two options above — honour the documented posture. In
`CMakeLists.txt` each of the four complex demo targets is now wrapped in
`if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER) ... endif()`, appending its own name to
`XP_COMPLEX_DEMO_TARGETS` as it goes; the `install(TARGETS ...)` list names that
variable instead of the four literals, because naming a target that was never
added is itself a hard configure error. Nothing else changed — the four demos
are the only consumers of the header (`scripts/smoke_kokkos_complex_quad.cpp`
also includes it but is not a CMake target). The `message(WARNING)` now says
plainly which four targets are being skipped and that the ctest suite is
unaffected, instead of the vague "will not compile its __complex128 oracle".

**How this was verified — against a Kokkos install that really lacks the
header.** Not by simulation and not by deleting the include from a source file:

```
cp -al $HOME/kokkos-install-quadmath /tmp/kokkos-noheader        # hardlink copy
rm /tmp/kokkos-noheader/kokkos-install-quadmath/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp
cmake -S . -B /tmp/ki21_build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/tmp/kokkos-noheader/kokkos-install-quadmath
cmake --build /tmp/ki21_build -j8 && ctest --test-dir /tmp/ki21_build -j4 --timeout 1800
```

The hardlink copy leaves `$HOME/kokkos-install-quadmath` untouched (verified:
the original header is still present afterwards, and `find` shows zero copies
under `/tmp/kokkos-noheader`). Results:

- configure: **rc 0**, and the probe reports `Kokkos install is MISSING
  impl/Kokkos_ComplexQuadPrecisionMath.hpp; the four complex demos … are
  SKIPPED`.
- `cmake --build --target help` lists no `kokkos_ep_demo_*complex` target, and
  does still list `dd/ff/qf/tf_complex_accuracy_test` — those never needed the
  header.
- build: **rc 0**, whole tree.
- **`ctest`: 34/34 passed, 0 failed**, versus 7 passed and 27 `***Not Run`
  before. That is the degradation the warning has been promising since T0.0.

The CI workaround in `.github/workflows/ci.yml` is left in place, as intended:
it exists so the CI lane exercises the *complex demos*, not to paper over this.

---

## Classifier verdict (2026-09-03) — `--classify` was audited and left UNCHANGED

The 2026-09-03 triage was asked to fix `scripts/sweep_accuracy.cpp --classify`
if it was found to be misclassifying. It was audited and found **not** to be.
Recording the audit so the next session does not repeat it.

**The one rule that was challenged.** `classify_real` /`classify_complex` test
"the true result is finite and in range but the backend returned NaN/inf"
*before* the conditioning probe, on the stated grounds that "ill-conditioning
can cost every digit; it cannot turn a representable number into a NaN".

For **268 of the 2,187 hard-failure points** the probe would have said
CONDITIONING had it run first — `ach` is 0.00 (or near it) against caps of
14.00/21.70, i.e. the stored input does not determine a single digit of the
answer. The affected cells:

```
FF r tan  58 nan + 4 inf      FF c atanh 24    FF c atan 24
TF c atanh 24 (ach 0.14-2.87) TF c atan  14    TF r sin  14
TF r cos  14                  TF r tan   12    FF c pow/log/log10 4 each
```

**Judgement: leave the rule as it is.** A NaN and a wrong-but-finite number are
different behaviours with different consequences for a caller, and the
`returned-nan`/`returned-inf` buckets are exactly where you want that to be
visible. Reclassifying these 268 as CONDITIONING would hide FF `tan`'s
π/2 inf and TF's huge-argument NaNs inside a bucket labelled "inherent, no
implementation does better", which is not true of a NaN. The honest statement
is the one made here: 268 of the 2,187 are hard failures *at points that are
also ill-conditioned to zero digits*, and a fix session should weight them
accordingly.

To list them:

```bash
awk -F, '$8=="UNEXPLAINED" && ($9=="returned-nan"||$9=="returned-inf") \
         && ($10+0) < 0.5*($7+0)' validation/sweep/sweep_classified.csv
```

Because nothing was reclassified, `validation/sweep/*.csv` and
`docs/DOMAINS.md` are byte-identical to `fecfc23`.

---

## Soft-failure map (2026-09-03) — where the other 7,051 UNEXPLAINED points go

The 9,238 UNEXPLAINED points split 2,187 hard (KI-12…KI-20 above) and 7,051
soft (`partial-digit-loss` 4,816, `no-correct-digits` 2,235). The soft
population was characterised by cause, not point by point. It is 73% complex
(5,165 vs 1,886 real) and it is **not** a long tail — five causes cover 96% of
it.

| cause | points | share | KI |
|---|---:|---:|---|
| complex inverse family loses the small component | 3,733 | 53% | KI-11 |
| `fmod`/`remainder` at extreme operand ratios | 1,680 | 24% | KI-10 / KI-15 |
| complex `tanh`/`tan` with no asymptotic branch | 794 | 11% | KI-18 |
| complex `mul`/`div` cancellation | 460 | 7% | *new — see below* |
| tails of the hard-failure defects | 169 | 2% | KI-14, KI-17, KI-20 |
| everything else (`fma`, `log*` near 1, `hypot`, …) | 215 | 3% | — |

**The three headline ops named in the brief resolve as follows.**

*atan (1,511 total, 1,297 soft).* Essentially all complex, essentially all in
the `cut-re` family: 1,210 of the 1,297 are `c atan cut-re`, plus 63 `polar`.
This is **KI-11**, not a separate defect — the real `atan` soft population is
negligible once KI-13's NaNs are removed.

*fmod (1,085) and remainder (1,200).* **Yes, these are KI-10.** 1,680 of the
2,285 are soft and sit in the `ulp`/`linear`/`log` families at large |a/b|,
which is precisely KI-10's shape; the other 605 are the hard failures now filed
as KI-15. No reclassification is warranted — they are already correctly
UNEXPLAINED, because KI-10 is an open algorithmic defect and not an inherent
limit. What *is* warranted is closing KI-10 and KI-15 together, at which point
2,285 points (25% of the whole UNEXPLAINED population) disappear at once. This
is the highest-yield single fix in the backlog.

**The one cause with no KI: complex `mul` and `div`, 460 points**
(`c div` 346, `c mul` 114; families `cut-re` 133+46, `polar` 103, `cut-im` 73).
These are classified ALGEBRAIC, so the conditioning probe charges them only the
operands' real storage error — for DD that is zero. The shape is the textbook
one: `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` cancels in each component when the two
products are close, and the naive division formula compounds it. Deliberately
not filed as a KI here because this session's remit was the hard failures and
this cause was not traced to a reproducer; a fix session should confirm it
against `dd_complex.hpp:multiply`/`divide` first. It is the obvious candidate
for a Kahan/Smith-style compensated complex multiply and division.

**Method note.** These are attributions by (kind, op, family) cell against the
mechanisms traced above, not per-point diagnoses. The cell boundaries are
clean — no cell splits across two causes — but the ±2% "everything else" row
has not been looked at.

---

## KI-22 — DD `asinh`, `atanh`, `sinh` (and TF `expm1`) collapse to the leading word at small arguments **[RESOLVED batch-5]**

**Severity: medium. DD on three ops, TF on one. Filed 2026-09-03 by the step-1c
ULP triage; measured, not fixed.**

### What

At small |x| these functions are asymptotically linear — `asinh(x) → x`,
`atanh(x) → x`, `sinh(x) → x`, `expm1(x) → x` — so they are perfectly
conditioned (kappa = 1) and the extended format should return essentially the
full mantissa. Instead they return the leading word and nothing more.

| backend | op | x | measured ulps | bound | over by | digits (of cap) |
|---|---|---|---|---|---|---|
| DD | `asinh` | 1e−15 | 9.92e15 | 2.0 | 4.96e15x | **15.91** of 31 |
| DD | `atanh` | −3.162e−16 | 6.84e15 | 2.0 | 3.42e15x | **16.07** of 31 |
| DD | `sinh` | −3.162e−16 | 6.23e15 | 64 | 9.73e13x | **16.11** of 31 |
| TF | `expm1` | 1e−9 | 7.16e10 | 32 | 2.24e9x | **10.81** of 21.7 |

15.91, 16.07, 16.11 of 31 digits is one FP64 word out of two. 10.81 of 21.7 is
one FP32 word plus change out of three.

### Why it is a defect and not a format or algorithm limit

Three exemptions exist in the new metric and none applies:

- **Not the subnormal-low-word band.** DD's cliff is 2.0042e−292; these inputs
  are at 1e−15, 277 decades above it. The results are O(1e−15), likewise.
- **Not conditioning.** kappa = 1 for all four. This is the *best*-conditioned
  region these functions have.
- **Not the exp-family squaring floor.** `sinh` and `expm1` are charged their
  backend's full `2^nq` (64 for DD, 32 for TF) and still exceed it by 9.7e13x
  and 2.2e9x. `asinh` and `atanh` are `log`-composed and carry no `2^nq` at all.

The signature — exactly one word retained, across three different functions,
only at small argument — is the classic cancellation of a small-|x| branch that
is missing. `asinh(x) = log(x + sqrt(x²+1))` and `atanh(x) = ½log((1+x)/(1−x))`
both cancel catastrophically as x → 0 and need a Taylor branch; `sinh` needs one
below the threshold where `(e^x − e^−x)/2` cancels. DD `expm1` has such a branch
(it is clean); TF's threshold appears set too low.

### Reproducing

```
/tmp/sweep --ulp --ulp-explain asinh:0
/tmp/sweep --ulp --ulp-explain expm1:14
```

### Closing it

Add (or lower the threshold of) the small-argument Taylor branch on DD `asinh`,
DD `atanh`, DD `sinh`, TF `expm1`. Acceptance: the four cells above drop under
the 8x allowance without any change to the metric.

---

### Resolution — batch-5

**Where the digits went.** Every one of these four routes an O(x) answer through
an intermediate of magnitude ~1:

```
asinh(x) = log(x + sqrt(x^2+1))     -> forms 1 + x inside the log
atanh(x) = 1/2 log((1+x)/(1-x))     -> forms 1 +- x
sinh(x)  = (e^x - e^-x)/2           -> subtracts two numbers near 1
expm1(x) = e^x - 1                  -> same
```

Forming `1 + x` for |x| < 1 discards every bit of x below 2^-p relative to 1, so
what survives is p − log2(1/|x|) bits. At DD's p = 106 and x = 1e−15 that is
106 − 50 = 56 bits — ONE FP64 WORD OF TWO, which is exactly the 15.91 of 31
digits this entry measured. The loss is u/|x| in relative terms, and it has
nothing to do with conditioning: κ = 1 for all four here.

**The crossover is not a tuning constant.** It is Sterbenz's lemma: `1 − x` is
exact precisely when x ∈ [1/2, 2], and `1 + x` likewise keeps every bit of x
while |x| ≥ 1/2. At |x| = 1/2 the closed form is still within ~1 ulp; the first
bit is lost the moment |x| drops below it. So the series branch takes |x| < 1/2.
FF, QF and TF already used 1/2 for `atanh`, `sinh` and `expm1` — the same
number, now derived rather than inherited, and applied to what was missing it.

**Terms needed at the crossover, for full precision at |x| = 1/2:**

| function | tail after N terms | N (DD, p=106) | N (QF, 96) | N (TF, 72) | N (FF, 48) |
|---|---|---|---|---|---|
| `asinh` | coefficients ≤ 1, tail ~x^(2N) = 2^(−2N) | 53 | 48 | 36 | 24 |
| `atanh` | x^(2k+1)/(2k+1), same 2^(−2N) tail | 53 | 48 | 36 | 24 |
| `sinh` | x^(2N)/(2N+1)!, factorial dominates | 12 | 11 | 9 | 7 |
| `expm1` | x^(N−1)/N! | 26 | 24 | 19 | 14 |

Each loop carries a cap above its count and exits on a relative-tolerance test,
so only arguments actually AT the crossover pay the full count; at x = 1e−15
`asinh` converges in two terms.

**Scope was wider than the entry stated.** DD was missing the branch on `asinh`,
`atanh` and `sinh`; TF was missing it on `expm1`; and `asinh` was missing it in
ALL FOUR backends. KI-22 named only DD because the sweep grid happened to catch
DD first — the defect is identical in FF, QF and TF at those formats' own
magnitudes. Fixed everywhere, per the standard that a backend should deliver the
precision the format allows.

**`asinh` does not use the series, and the measurement is why.** A term count is
also a rounding-error count, and `asinh`/`atanh` have the slow 2^(−2N) tail, so
their crossover counts are the largest in the table. Measured, the `asinh` series
at |x| = 1/2 was WORSE than the closed form it replaced: the dense sweep flagged
88 TF `asinh` grid points losing up to 1.21 digits, because TF pays 36 terms of
rounding to avoid 1 bit of cancellation. Rather than retune the crossover, note
that this particular cancellation is removable in CLOSED FORM. With
s = sqrt(x²+1) and x ≥ 0,

```
x + s = 1 + x + (s^2 - 1)/(1 + s) = 1 + x + x^2/(1 + s)
```

so `asinh(x) = log1p(x + x^2/(1+s))` with every term non-negative. The leading 1
is never formed: `log1p` carries its own small-argument series (2·atanh(t),
t = z/(2+z)) and receives an O(x) argument. That form is now used below
|x| = 1/2 in all four backends. It is not free either — the extra divide and the
handoff to `log1p` cost a fixed ~0.4 digits, while the `1 + x` it replaces loses
log10(1/|x|); those cross near |x| = 0.4, so the same Sterbenz 1/2 bounds it.
`asinh` is the only one of the four with such an identity; `atanh`, `sinh` and
`expm1` keep the series.

**DD `sinhcosh` replaces only the sinh output.** `(e^a − e^-a)/2` subtracts two
numbers near 1 and keeps the O(a) difference, costing u/|a|; `(e^a + e^-a)/2`
adds them and costs nothing. Computing cosh from a Taylor sum only adds
rounding, and measurably did — 26 DD complex grid points (`cos`, `sin`, `tan`,
`sinh`, `cosh`) regressed up to 0.23 digits when the first attempt replaced both.
`ff/qf/tf_math.hpp`'s own branches replace both; DD's does not, and the sweep
says DD is right. (Whether the other three should follow is left open — they are
shipped behaviour and regress nothing.)

**Measured at small x, in digits and ulps** (caps: DD 31.00, TF 21.70, FF 14.00,
QF 29.00; `99.00 d / 0 ulp` means bit-exact against the `__float128` oracle):

| x | DD `asinh` before → after | DD `atanh` before → after | DD `sinh` before → after |
|---|---|---|---|
| 1e−04 | 26.87 → 31.66 d / 1.76 ulp | 27.05 → 33.03 / 0.08 | 27.00 → 32.30 / 0.41 |
| 1e−06 | 24.87 → 32.05 / 0.72 | 25.05 → 32.89 / 0.10 | 25.00 → 33.74 / 0.015 |
| 1e−09 | 21.87 → 32.90 / 0.10 | 22.05 → 99.00 / 0 | 21.25 → 33.75 / 0.015 |
| 1e−12 | 18.23 → 32.68 / 0.17 | 19.05 → 99.00 / 0 | 19.13 → 33.76 / 0.014 |
| 1e−15 | **15.91** → 31.89 / 1.04 | **16.27** → 33.77 / 0.014 | **16.13** → 99.00 / 0 |
| 1e−16 | 15.15 → 32.59 / 0.21 | 16.15 → 99.00 / 0 | 15.13 → 99.00 / 0 |
| 1e−18 | 13.15 → 99.00 / 0 | 14.05 → 99.00 / 0 | 13.13 → 99.00 / 0 |
| 1e−20 | 11.15 → 99.00 / 0 | 12.05 → 99.00 / 0 | 11.13 → 99.00 / 0 |

| x | TF `expm1` before → after | FF `asinh` after | QF `asinh` after | TF `asinh` after |
|---|---|---|---|---|
| 1e−04 | 18.16 → 22.40 d / 0.19 ulp | 14.44 / 1.02 | 30.24 / 0.045 | 22.86 / 0.066 |
| 1e−06 | 15.98 → 22.18 / 0.31 | 14.67 / 0.60 | 29.76 / 0.14 | 22.41 / 0.18 |
| 1e−09 | 10.81 → 24.23 / 0.003 | 18.78 / 4.7e−05 | 30.34 / 0.036 | 22.70 / 0.094 |
| 1e−12 | 13.45 → 23.45 / 0.017 | 24.78 / 4.7e−11 | 31.02 / 0.008 | 24.78 / 0.0008 |
| 1e−15 | 14.80 → 23.80 / 0.007 | — | — | — |
| 1e−20 | 15.35 → 22.87 / 0.064 | — | — | — |

**Whole-grid means** (1,652 real points each, before → after):

| cell | mean before | mean after | up | down |
|---|---|---|---|---|
| DD `atanh` | 28.671 | 30.985 | 1376 | 0 |
| DD `sinh` | 29.948 | 30.330 | 81 | 0 |
| DD `cosh` | 30.346 | 30.346 | 0 | 0 |
| DD `asinh` | 30.449 | 30.805 | 74 | 2 |
| FF `asinh` | 13.485 | 13.891 | 132 | 4 |
| QF `asinh` | 28.084 | 28.422 | 140 | 42 |
| TF `asinh` | 19.989 | 21.447 | 750 | 92 |
| TF `expm1` | 16.271 | 16.705 | 131 | 0 |
| DD complex `sin` | 28.160 | 29.886 | 461 | 3 |
| DD complex `sinh` | 27.720 | 29.103 | 380 | 4 |
| DD complex `cosh` | 28.018 | 29.120 | 317 | 2 |

The condition-aware ulp gate closes two cells on this entry alone — `DD r sinh`
and `TF r expm1` both leave UNEXPLAINED — and opens none.

## KI-23 — QF `log`, `log2`, `log10`, `log1p` lose ~11 digits above |x| ≈ 1e29 **[RESOLVED batch-6]**

**Severity: medium, QF only. Filed 2026-09-03 by the step-1c ULP triage;
measured, not fixed.**

### What

| backend | op | x | measured ulps | bound | over by | digits (of 29) |
|---|---|---|---|---|---|---|
| QF | `log` | 3.162e29 | 2.62e14 | 1.29e3 | 2.03e11x | **17.58** |
| QF | `log2` | 3.162e29 | 2.62e14 | 1.29e3 | 2.03e11x | 17.58 |
| QF | `log10` | 3.162e29 | 2.62e14 | 1.29e3 | 2.03e11x | 17.58 |
| QF | `log1p` | 3.162e29 | 2.62e14 | 1.29e3 | 2.03e11x | 17.58 |

All four are the same natural log rescaled, so one root cause. **DD, FF and TF
are all clean at the same inputs** — this is QF-specific, which is what makes it
a defect rather than a property of the algorithm.

The bound of 1.29e3 already includes the log-family floor derived in step 1c
(`2^nq / |ln x|` = 64 / 67.9, times the 8x allowance and the kappa term); the
measurement exceeds it by eleven orders of magnitude, so the Newton-on-`exp`
inheritance does not explain it.

Note the interaction with **KI-19** (QF `pow` at 1e30) — same backend, same
decade, and `pow` routes through `log`. They may share a root cause; that has
not been established, so they are filed separately.

### Reproducing

```
/tmp/sweep --ulp --ulp-explain log:59
```

### Closing it

Trace QF `log`'s argument handling above 1e29 — most likely the initial estimate
or the exponent extraction feeding Newton. Acceptance: the four cells drop under
8x, and check whether KI-19 closes with them.

### RESOLVED 2026-09-04 — commit `fix: KI-23 QF log family at large argument; KI-28 complex multiply Annex G recovery`

**None of the four filed hypotheses was right.** It is not the exponent
extraction, not a short constant, not the FP32 seed, and not a premature square.
The cause is **which argument `exp` is evaluated at inside the Newton step**, and
it was found exactly where this entry suggested looking — by diffing QF's `log`
against TF's.

#### The QF-vs-TF diagnosis

The two implementations use algebraically identical but numerically different
Newton corrections:

| backend | Newton step | `exp` is evaluated at |
|---|---|---|
| DD, FF, TF | `x += (a − e^x)/e^x`  (residual form) | **+x** |
| QF | `x += a·e^{−x} − 1`  (QD `qd_real.cpp:1007-1009`) | **−x** |

QF was the only backend carrying QD's form. For `a = 1e29`, `x = ln a = 66.8`, so
QF evaluates `exp(−66.8) = 2^−96.4`. QF's fourth word `f3` sits at `2^(E−72)`
relative to a value of magnitude `2^E`, i.e. at `2^−168.4` — **below FP32's
smallest subnormal `2^−149`**. The bottom words are not merely inaccurate, they
are *not representable*, and `qf_math.hpp`'s final componentwise scaling
(`s3.f3 * pow2`) flushes them to zero.

The measured signature confirms it exactly. Over `1e20 → 1e38` — 18 decades, 59.8
bits of exponent — the loss is 27.18 → 8.92 digits, i.e. 60.6 bits. **One bit of
result lost per bit of argument exponent**, which is precisely what a fixed
underflow floor at `2^−149` produces and is not what any conditioning or
seed-quality argument predicts.

#### The mirror image: three backends had the same defect at the other end

The residual form has the identical weakness reflected about `a = 1`: it
evaluates `e^x`, which for `a ≪ 1` is tiny and loses *its* low words. Measured on
the shipped tree before this fix:

| x | DD | FF | QF | TF |
|---|---|---|---|---|
| 1e−20 | 32.00 | 14.00 | 29.00 | 22.00 |
| 1e−30 | 32.00 | 14.00 | 29.00 | **18.02** |
| 1e−38 | 32.00 | **9.54** | 29.00 | **9.54** |
| 1e−300 | **26.92** | — | — | — |
| 1e−307 | **19.98** | — | — | — |

QF, the backend that fails at the large end, is the only one *clean* at the small
end — because QD's form is the one that evaluates `exp` at the large argument
there. **The filed scope was half the defect.** Both ends are fixed here, in all
four backends.

#### The fix

Each backend keeps its own form as the default and switches to the other only
where the default's `exp` argument genuinely underflows. The switch point is
derived from the format, not tuned: the last word sits at `2^(E−w)` for a value
of magnitude `2^E` (w = 53 for DD's `lo`, 24 for FF's, 72 for QF's `f3`, 48 for
TF's `f2`), so it leaves the normal range when `E − w < E_min`.

| backend | default form | switches to the other when | that is |
|---|---|---|---|
| DD | residual | `x < −969·ln2 = −671.7` | a < 1e−291 |
| FF | residual | `x < −102·ln2 = −70.7` | a < 2e−31 |
| TF | residual | `x < −78·ln2 = −54.1` | a < 3e−24 |
| QF | QD | `x > +54·ln2 = +37.4` | a > 1.7e16 |

Each branch is additionally unavailable past its format's `ln(MAX)`, where the
reciprocal itself overflows; there the default is the only finite option.

**The defaults are not symmetric, and that is a measured result, not an
oversight.** An earlier revision of this fix made the residual form the default
for QF too, on the reasoning that its correction carries *relative* rather than
*absolute* error. The sweep rejected it: **7,817 decreases**, up to 6.77 digits,
concentrated in QF `log`/`asinh`/`acosh`/`pow`. QF's `divide` is a long division
and the residual form needs one per Newton step; QD's choice of the
multiply-only form for `qd_real` is vindicated. Inverting QF's branch to match
took the same sweep to 5 decreases. DD, FF and TF have the opposite preference
and keep the residual form.

#### After — `log` family, digits and ulps (DD/FF/TF are the control)

Digits (format caps: DD 32, FF 14, QF 29, TF 22). QF before → after:

| op | x | QF before | QF after | DD | FF | TF |
|---|---|---|---|---|---|---|
| `log` | 1e20 | 27.18 | **29.00** | 31.89 | 14.00 | 22.00 |
| `log` | 1e25 | 22.17 | **29.00** | 32.00 | 14.00 | 22.00 |
| `log` | 1e29 | 18.43 | **29.00** | 32.00 | 14.00 | 22.00 |
| `log` | 1e32 | 15.05 | **29.00** | 31.71 | 14.00 | 22.00 |
| `log` | 1e35 | 12.35 | **28.82** | 32.00 | 14.00 | 22.00 |
| `log` | 1e38 | 8.92 | **29.00** | 31.68 | 14.00 | 22.00 |

`log2`, `log10` and `log1p` are the same natural log rescaled and move
identically (`log10` at 1e35 lands at 28.81, everything else matches to 0.01).

Ulps of the true value at each format's `u`:

| op | x | QF before | QF after | DD | FF | TF |
|---|---|---|---|---|---|---|
| `log` | 1e20 | 5.28e+01 | **5.40e−01** | 2.61e−01 | 2.29e−01 | 6.05e−02 |
| `log` | 1e25 | 5.33e+06 | **2.84e−01** | 1.74e−01 | 1.38e−02 | 2.60e−02 |
| `log` | 1e29 | 2.95e+10 | **9.75e−02** | 4.12e−02 | 1.72e−01 | 5.00e−03 |
| `log` | 1e32 | 7.06e+13 | **2.54e−01** | 3.97e−01 | 7.36e−02 | 1.18e−01 |
| `log` | 1e35 | 3.52e+16 | **1.19e+00** | 2.79e−02 | 1.05e−01 | 3.06e−02 |
| `log` | 1e38 | 9.49e+19 | **9.45e−02** | 4.23e−01 | 1.09e−01 | 5.40e−03 |

The four QF cells go from 2.03e11× over bound to sub-ulp. **DD, FF and TF were
always clean at these inputs and are unchanged by the QF branch**, which is what
made the defect diagnosable in the first place.

#### KI-13's residual observation: yes, `asinh`/`acosh` lift with it

KI-13's residuals recorded "QF `asinh`/`acosh` decay above the band (27.18@1e20
→ 8.93@1e38, tracks QF `log()` at large argument)". Measured here, it tracks it
to the second decimal — same root cause, and it closes with it:

| op | x | QF before | QF after |
|---|---|---|---|
| `asinh`/`acosh` | 1e20 | 27.18 | **29.00** |
| `asinh`/`acosh` | 1e25 | 22.18 | **29.00** |
| `asinh`/`acosh` | 1e29 | 18.43 | **29.00** |
| `asinh`/`acosh` | 1e32 | 15.05 | **29.00** |
| `asinh`/`acosh` | 1e35 | 12.36 | **28.84** |
| `asinh`/`acosh` | 1e38 | 8.93 | **29.00** |

Above the band both are `log(a) + log(1 + sqrt(1 ± 1/a²))`, so they inherit
`log`'s error directly. No change was needed in `asinh` or `acosh` themselves.

#### Small end, after

| x | DD | FF | QF | TF |
|---|---|---|---|---|
| 1e−30 | 32.00 | 14.00 | 29.00 | 22.00 (was 18.02) |
| 1e−38 | 32.00 | 14.00 (was 9.54) | 29.00 | 22.00 (was 9.54) |
| 1e−300 | 32.00 (was 26.92) | — | — | — |
| 1e−307 | 32.00 (was 19.98) | — | — | — |

`1e−45` sits at 8.13 digits on all three FP32 backends. That is the FORMAT: the
input itself is subnormal and cannot be represented to more than that, so no
implementation can do better. It is an improvement on QF's previous **NaN**
there.

#### KI-19

This entry asked whether KI-19 (QF `pow` at 1e30) closes with it. KI-19 was
already resolved separately at `0ad44fe`. QF `pow` does route through `log` and
does improve here — 33 sweep points gain — but `pow` also picks up 1 decrease
(see the decreases note below).

#### Gates

- ULP gate: **48 gated cells / 2,650 gated points → 44 / 2,390.** The four
  removed cells are exactly QF `log`, `log2`, `log10`, `log1p`.
- Monotone gate, measured against HEAD so prior batches' accepted decreases do
  not confound it: **485 increases, 5 decreases.** All five are `pow`, which
  routes through `log`: QF `pow` pt190 29.00→27.70, TF `pow` pt16 21.11→21.05,
  TF complex `pow` pts 1565/1566/1568 (worst 0.40). Accepted — `pow` is
  exp-family and carries the §10 conditioning caveat, and the same op gains 47
  points across the two backends.

---

## KI-24 — FF `sinh` and `cosh` lose one full FP32 word near the `exp` range limit **[RESOLVED batch-7]**

**Severity: low, FF only. Filed 2026-09-03 by the step-1c ULP triage;
measured, not fixed.**

### What

| backend | op | x | measured ulps | bound | over by | digits (of 14) |
|---|---|---|---|---|---|---|
| FF | `sinh` | −87.96 | 2.94e7 | 128 | 2.30e5x | **7.07** |
| FF | `cosh` | −87.96 | 2.94e7 | 128 | 2.30e5x | 7.07 |

7.07 of 14 digits is 23.5 bits — one FP32 word of two, gone. x = −87.96 is just
inside FP32's `exp` range limit (ln(FLT_MAX) ≈ 88.72), so `e^x` is at the very
bottom of the normal range and `e^−x` at the very top.

The bound already charges FF's full exp-family floor (2^4 = 16, times 8). The
overshoot is 2.3e5x beyond it, so the squaring gain does not explain it.

Low severity because it is confined to the last decade before the range limit,
where a caller has already accepted that the result is near overflow. Filed
because the loss is a clean whole word — a scaling artefact, not gradual
degradation — and because it is FF-only: DD, QF and TF are clean at the
corresponding point in their own ranges.

### Reproducing

```
/tmp/sweep --ulp --ulp-explain sinh:0
```

### RESOLVED 2026-09-04 — commit `fix: KI-30 scale multiply past the Dekker splitter limit; KI-24 fixed`

**Verdict: not format-forced. An implementation artefact, and fixed.**

The filing was right that the loss is a clean whole word and wrong about where
it came from. It is not the `e^x − e^{−x}` subtraction — as the brief noted,
`e^{−x}` has underflowed to zero there and contributes nothing. It is the
*intermediate*: for x = −87.96 the saturate branch computed `e = exp(a)` at
`a = −87.96`, giving 6.0e−39, which is **subnormal in FP32**. A subnormal `hi`
word cannot carry a `lo` word at `hi·2^−24` — that would be 3.6e−46, far below
FLT_MIN·2^−23 — so `e.lo` is flushed and `e` degrades to a bare FP32 float. The
code then reciprocated that half-width value, and the returned `sinh` inherited
its 7 digits.

**The result was never the problem.** `sinh(−87.96) = −8.3e37` is an ordinary
normal FP32 number with an ordinary normal `lo` word; there is no bit-budget
argument that forbids 14 digits there. The proof was already sitting inside the
same function: `x = −89` scored **13.79** digits, because `exp(−89)` returns
`+inf` on the reciprocal and tripped the `isinf` fallback into
`exp(|a| − ln 2)`, which never forms the subnormal. x = −88.7, one step away and
on the other side of that test, scored **7.01**. Two neighbouring inputs, same
conditioning, 6.8 digits apart — a branch artefact, not a format ceiling.

**The fix** is the transformation the brief pointed at, and the one QF and TF
already carry (`kQFReciprocalFloor = -40.0f`, `kTFReciprocalFloor = -55.0f`,
both from the KI-9 session): below a floor, drop the reciprocal and evaluate
`exp(|a|)` directly at full width, then halve. FF gains
`kFFReciprocalFloor = -75.0f`, DD gains `kDDReciprocalFloor = -672.0`.

FF, digits of 14:

| x | before | after |
|---|---|---|
| −80 | 11.10 | **14.23** |
| −85 | 9.22 | **14.91** |
| −87 | 7.86 | **12.77** |
| −87.96 (the filed point) | 7.07 | **13.2** |
| −88 | 7.17 | **13.10** |
| −88.5 | 7.71 | **13.20** |
| −88.7 | 7.01 | **13.44** |
| −89 (already on the `isinf` path) | 13.79 | 13.79 |

The positive side is bit-identical — `a.hi < 0.0` gates the whole change, and
for `a > 0` the new `exp(aa)` is literally `exp(a)`.

### The same defect in DD, found by the audit

DD has the identical shape at its own floor. DD's `lo` word goes subnormal when
`hi < DBL_MIN·2^53 = 2.00e−292`, i.e. `a < −671.7`, and DD's saturate branch was
reciprocating `exp(a)` right through that band. Digits of 31:

| x | before | after |
|---|---|---|
| −672 | 30.23 | 30.3 |
| −690 | 24.31 | ~30 |
| −700 | 19.98 | ~30 |
| −709.7 | 15.41 | ~29.9 |

The floor is set at −672, matching the derived 2.00e−292 crossover.

### Why the floors are where they are

The natural floor is the subnormal crossover itself (FF: `FLT_MIN·2^24` =
1.97e−31 at a = −70.7; DD: −671.7). Switching there unconditionally is not free:
the direct path is a different rounding sequence, so points *above* the
crossover that happen to score well on the reciprocal can lose a fraction of a
digit. Setting FF's floor at −70.7 produced **656 sub-digit monotone decreases**
(worst −1.12) across DD and FF real and complex `sinh`/`cosh`. Backing off to
−71 left 10; −75 left **zero**, and −78 and −80 also pass but close fewer points
(50 increases at −75, 40 at −78, 30 at −80). −75 is the most aggressive floor
that keeps the monotone gate clean, so that is the one shipped. The band between
the true crossover and the floor still loses part of a word, but the loss there
is under a digit, not the whole word this entry was filed for.

### Gate effect

The condition-aware ULP gate goes from 44 failing cells / 2,390 failing points
to **42 / 2,350**. The two closed cells are exactly `FF r sinh` and `FF r cosh`
— this entry's cells. No cell that passed before fails now.

---

## KI-25 — QF and TF `atan` return a non-finite value at |x| ≈ 3.16e19 **[RESOLVED 0ad44fe]**

**Severity: high (wrong value, not lost digits), QF and TF. Filed 2026-09-03 by
the step-1c ULP triage; measured, not fixed.**

### What

`atan(3.162e19)` should return π/2 to full precision. It is the best-conditioned
input the function has — kappa = 2e−20, since the derivative `1/(1+x²)` has
annihilated any input perturbation. QF and TF instead return a **non-finite**
value (the ulp measurement is `inf`, i.e. `got` is not finite). DD and FF are
correct at the same input.

| backend | op | x | got | expected |
|---|---|---|---|---|
| QF | `atan` | 3.162e19 | non-finite | 1.5707963… |
| TF | `atan` | 3.162e19 | non-finite | 1.5707963… |

### Relationship to KI-13

**KI-13** records QF/TF `asinh` and `acosh` failing at the same magnitude, and
attributes it to an unscaled `x² + 1` overflowing the FP32 word (x² at 1e39
against FLT_MAX 3.4e38). `atan` plausibly reaches the same helper — but `atan`
is **not** in KI-13's op list, and the shared-helper claim has not been traced.
Filed separately rather than folded into KI-13 on the strength of an analogy; if
tracing shows one root cause, merge them then.

This is the same failure *mechanism family* as KI-8 (squaring before scaling),
at the opposite end of the range.

### Reproducing

```
/tmp/sweep --ulp --ulp-explain atan:57
```

### Resolution — commit `fix: KI-19/KI-25/KI-26 correct non-finite signalling in div, atan and trig` (2026-09-04)

**Yes, KI-25 does share KI-13's mechanism, and was already closed by the KI-13
fix at `7680809`.** The tracing that this entry asked for and did not do:
`atan(a)` is `angle(1, a)`, and `angle()` forms `x² + y²` — the same unscaled
sum of squares KI-13 was about. Commit `7680809` ("scale hypot/abs at BOTH ends;
remaining unscaled sums of squares") added the high-side rescale at
`qf_math.hpp` `angle()` (`kQFSqHi = 1.0e18f`) and the TF counterpart. Measured
against the current tree, QF and TF `atan(3.162e19)` are finite and correct. The
`atan`-is-not-in-KI-13's-op-list objection was right about the *entry* and wrong
about the *code*: the op list was incomplete, not the diagnosis. **Merge this
entry's mechanism into KI-13's the next time the file is reorganised.**

Two things were nevertheless found and fixed here, neither of them the filed
symptom.

**(a) A separate FF exposure at the LOW end.** `FF atan(1e-30)` returned NaN.
Unrelated to sums of squares: `ff_math.hpp` `sincos` tested convergence with a
strict `<` and, on hitting `itrmx`, executed a bare `return` that left the `x`
and `y` out-parameters *never written*. The caller then read uninitialised
storage. Changed to `<=` plus `break`, so the partial result is always
published. `FF atan(1e-30)` now returns `1e-30`. Filed here rather than as a new
KI because it is the same op and the same reported symptom class (`atan`
non-finite), just at the opposite end of the range.

**(b) A codomain violation at the top of the range, on QF.** `atan` is bounded
by π/2 for every finite argument — that is a property of the function, so no
input may exceed it. `angle()`'s Newton refinement lands slightly past:

```
QF atan(3.4e38) - (true pi/2) = +4.73e-30      about +1.9 ulp
```

DD, FF and TF measured inside the bound at the same inputs, but the margin is
incidental rather than structural. All four now clamp `atan`'s magnitude to the
stored π/2. The clamp costs nothing in accuracy and in fact *gains*: the true
`atan(3.4e38)` is `π/2 - 2.9e-39`, far below QF's resolution, so the stored π/2
is the correctly-rounded answer and the excess drops from `+4.73e-30` to
`-8.47e-32` — which is the constant's own distance from π/2, i.e. the best the
format can do. Halving π is exact, so the clamp target is the format's nearest
value to π/2 and not a re-rounded approximation of it. **`atan2` is deliberately
not clamped**: its range is (−π, π], not [−π/2, π/2].

Verified `|atan(x)| <= π/2` against a `__float128` π/2 on all four backends at
3.162e19, 1e30, ±3.4e38, 1.7e308 (DD), and at 1e-30, −1e-30, 1e-45.

---

## KI-26 — TF `sin`/`cos`/`tan` and QF `cos`/`tan` return non-finite values at very large arguments **[RESOLVED 0ad44fe]**

**Severity: high (wrong value, not lost digits), QF and TF. Filed 2026-09-03 by
the step-1c ULP triage; measured, not fixed.**

### What

| backend | op | x | got |
|---|---|---|---|
| TF | `sin` | 3.162e25 | non-finite |
| TF | `cos` | 3.162e25 | non-finite |
| TF | `tan` | 3.162e25 | non-finite |
| QF | `cos` | 1e30 | non-finite |
| QF | `tan` | 1e30 | non-finite |

Losing accuracy at these arguments is expected and is **not** what is filed
here. Argument reduction mod 2π at x = 3.16e25 needs π to roughly
`p + log2(x)` ≈ 157 bits and TF carries 72, so the reduced argument is
genuinely underdetermined and any *finite* answer in [−1, 1] would be within
the (enormous) legitimate bound. Step 1c derives that bound —
`in_delta = |x|` ulps, in `docs/ULP_METRIC.md` — and under it DD `cos(1e17)`
passes at ratio 0.03.

What is filed is that these five return **non-finite**. `sin` and `cos` are
bounded by 1 for every finite input; returning inf or NaN is wrong under any
error model, at any argument, and a caller cannot defend against it. The
correct behaviour when reduction runs out of π is to return a value in range
and, ideally, to document the accuracy loss — not to produce something outside
the function's codomain.

DD is clean at its corresponding point; FF has a separate, unrelated small
failure at x = π that is part of the deferred trig-reconstruction group.

### Reproducing

```
/tmp/sweep --ulp --ulp-explain sin:64
/tmp/sweep --ulp --ulp-explain cos:59
```

### Closing it

Clamp the reduction path so that an under-determined reduction still yields an
in-codomain result. Acceptance: no `sin`/`cos` point on any backend returns a
value outside [−1, 1] or non-finite. The *accuracy* at these arguments is out of
scope for this KI and is covered by the deferred `in_delta = |x|` bound term.

### Resolution — commit `fix: KI-19/KI-25/KI-26 correct non-finite signalling in div, atan and trig` (2026-09-04)

**Which half is fixed and which is deliberately not.** Fixed: the **range
violation** — no `sin` or `cos` on any backend now returns a non-finite value or
a value outside [−1, 1], at any finite argument. Not fixed, and not attempted:
the **accuracy loss** at huge arguments. Argument reduction mod 2π against a
finite-precision π cannot do better than the number of π bits the format
carries, and no amount of code changes that. These are separate claims and the
fix addresses only the first.

**Scope correction.** Filed for QF and TF. DD was also affected and was not
listed: `DD sin(1e35)` and `DD sin(3.4e38)` returned NaN. FF's pre-guard existed
but returned the wrong constant. All four backends were changed.

**The fix — a post-computation codomain guard, not a pre-computation one.** At
the end of `sincos`, on the *computed result*:

1. If either `|sin|` or `|cos|` exceeds `1 + 2^-10`, the reduction is
   under-determined by a margin no rounding can explain. Emit the diagnostic and
   return the identity point `(sin, cos) = (0, 1)` — in range, and the honest
   statement that the phase is unknown.
2. Otherwise clamp any result whose magnitude sits marginally above 1 (leading
   word `> 1`, or `== 1` with a same-signed tail) to exactly ±1.

Two pre-guards that already existed also returned the wrong values and were
corrected: DD's `a.hi >= 1e60` and FF/QF's `|a| >= 1e30` arms returned
`(sin, cos) = (0, 0)`, which is not a point on the unit circle. They now return
`(0, 1)`.

**Why the guard is on the result and not on the reduced argument.** The first
design tested the reduced argument (`!(|s3| <= 4)`) and **regressed the monotone
gate by 30 points, worst −7.06 digits**. Diagnosed: at DD `sin`/`cos`/`tan`
around ±1e27 the `nint` residual is 4.317 — over the threshold — yet roughly
seven correct digits are still recoverable there. Testing the reduced argument
throws away answers that are degraded but useful. Testing the *result* fires
only where the answer is already outside the codomain and therefore already
unusable. With the second design the gate shows **18 decreases, worst 0.56
digits** — bit-for-bit batch 1's already-accepted set under KI-8, i.e. **this
change adds zero decreases** and adds 2,022 improvements.

**tan convention.** `tan` is deliberately **not** clamped. Unlike `sin` and
`cos`, `tan` has no bounded codomain — it is genuinely unbounded at its poles,
and returning `±inf` near an odd multiple of π/2 is the correct IEEE
signalling, not a defect. `tan` is computed as `sin/cos` and inherits the
in-range `sin`/`cos` from the guard above, so it can no longer produce a
non-finite value from a *reduction* failure while still producing one from a
genuine pole. That is the intended and deliberate distinction.

**Siblings audited.** `asin`, `acos`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`,
`acosh` were probed at 3.162e19, 1e30, 1e35 and 3.4e38 on all four backends and
are **sound**. `sinh`/`cosh` returning `±inf` at these arguments is correct —
the true value genuinely overflows the format. Complex counterparts: complex
divide now returns `±inf` on QF/TF/FF via the KI-19 fix; complex `arg` is
finite; complex `sin`/`cos` are in codomain.

---

## KI-27 — DD and FF `multiply` return NaN on genuine product overflow, where `±inf` is correct **[RESOLVED 2e810c2]**

**Severity: high (wrong kind of non-finite value), DD and FF. Filed 2026-09-04
while fixing KI-19; measured, not fixed — outside that batch's scope.**

### What

The same contract KI-19 restored for `divide` is violated by `multiply` on two
backends. When the true product overflows the format, `inf` is correct and `NaN`
is wrong: downstream code branching on `isinf()` takes the wrong path and never
learns the magnitude overflowed.

```
DD multiply(1e300, 1e300).hi = -nan     true 1e600, should be +inf
FF multiply(1e30,  1e30 ).hi = -nan     true 1e60,  should be +inf
QF multiply(1e30,  1e30 )    = inf      correct
```

QF gets it right, which shows the correct behaviour is reachable and this is a
defect rather than a format limit.

### Extent

Not surfaced by the sweep grid — like KI-19's quotient region, the `mul` grid
does not place a point at an overflowing product. Found by direct probe.

It has at least one visible downstream consequence: `DD` complex
`(1e300 + 1e300i) / (1e-10 + 1e-10i)` still returns NaN after the KI-19 divide
fix, because the complex division formula multiplies before it divides and the
NaN is manufactured in the `multiply`.

### Closing it

Same shape as the KI-19 DD fix. The Dekker `two_prod` splitter overflows before
the hardware product does, turning an `inf` into `inf - inf = NaN`. Guard on the
hardware product `a.hi * b.hi` first: if it is non-finite or zero, that is
already the IEEE-correct answer and the error-free transform must not run. Then
extend the `mul` sweep grid to cover an overflowing product so the next
re-baseline cannot miss it, exactly as KI-19 requires for `div`.

### RESOLVED 2026-09-04 — commit `fix: KI-12 sincos out-params; KI-14 integer ops past 2^47; KI-27 multiply overflow`

Fixed as prescribed, and as QF/TF already do it at `qf_two_prod`/`tf_two_prod`
rather than by inventing a third approach: guard on the hardware product first,
BEFORE any splitter or scaling runs, and return it as-is when it is non-finite
or zero.

```cpp
const double p = a.hi * b.hi;                                   // KI-27
if (!detail::isfinite(p) || p == 0.0) return DoubleDouble(p, 0.0);
```

Three cases, and `p` is already right in all three: `|p| = inf` → the true
product overflows, `±inf` with the IEEE sign is correct; `p = NaN` → a genuinely
undefined form (`0 · inf`), keep it; `p = 0` → the true product underflows,
`±0` is correct because `|a.lo·b.hi|` and `|a.hi·b.lo|` are each at most
`|a.hi·b.hi|·2^-53` and cannot lift a zero leading product back into range.

Applied to `multiply`, `multiply_scalar` and `two_prod` on **both** DD and FF.
DD and FF expose no `sqr` (only QF and TF do, and theirs were already correct);
the DD/FF squaring path is `multiply(a, a)`, which the same guard covers.

| probe | `0ad44fe` | after | correct |
|---|---|---|---|
| DD `multiply(1e300, 1e300)` | −nan | **inf** | inf |
| DD `multiply(−1e300, 1e300)` | −nan | **−inf** | −inf |
| DD `multiply(1e−300, 1e−300)` | 0 | 0 | 0 |
| DD `multiply_scalar(1e300, 1e300)` | −nan | **inf** | inf |
| DD `mul(2, inf)` | −nan | **inf** | inf |
| DD `mul(0, inf)` | −nan | −nan | NaN — genuinely undefined, correctly kept |
| DD `mul(0, −5)` | 0 | 0 | 0 |
| FF `multiply(1e30, 1e30)` | −nan | **inf** | inf |
| FF `multiply(−1e30, 1e30)` | −nan | **−inf** | −inf |
| FF `multiply(1e−30, 1e−30)` | 0 | 0 | 0 |
| FF `multiply_scalar(1e30, 1e30)` | −nan | **inf** | inf |
| FF `two_prod(1e30, 1e30)` | −nan | **inf** | inf |
| FF `mul(2, inf)` | −nan | **inf** | inf |
| FF `mul(0, inf)` | −nan | −nan | NaN |
| QF / TF `multiply`, `sqr`, `multiply_scalar` at ±overflow and underflow | inf / −inf / 0 | unchanged | already correct |

Downstream, the consequence named in "Extent" is closed:
`DD (1e300 + 1e300i) / (1e−10 + 1e−10i)` went **−nan → +inf** in the real part
(imaginary part 0, unchanged), and the FF analogue with 1e30 likewise.

`ff_math.hpp:multiply` previously carried a B9 "Scope" paragraph recording
genuine product overflow as deliberately out of scope; that paragraph is now
superseded in place.

**Not done, and still owed:** the sweep `mul` grid still places no point at an
overflowing product, so a re-baseline would not catch a regression here. That
is a grid change, and this batch was told not to re-baseline; it stays on the
same list as the KI-19 `div` grid extension.

---

## KI-28 — DD and FF complex SQUARING returns NaN where the true result is finite-or-infinite **[RESOLVED batch-6]**

**Severity: medium (wrong kind of non-finite value), DD and FF complex. Filed
2026-09-04 while fixing KI-27; measured, not fixed — outside that batch's
scope.**

### What

KI-27 fixed the `multiply` PRIMITIVE. The complex multiply FORMULA has an
independent instance of the same class of defect, and the primitive fix does
not reach it:

```
DD (1e300 + 1e300i)^2  ->  (-nan, -nan)      true (0, +inf)
FF (1e30  + 1e30i )^2  ->  (-nan, -nan)      true (0, +inf)
```

Measured at commit `0ad44fe` and again after the KI-27 fix — unchanged by it.

### Mechanism

The real part is formed as `re = a.re·b.re − a.im·b.im`. With `a = b` and both
components at 1e300, each product now correctly overflows to `+inf` (that is
KI-27 working), and the subtraction is then `inf − inf = NaN`. The imaginary
part `2·a.re·a.im` is `+inf` and correct in isolation, but the same expression
tree contaminates it.

The true value is exact and unambiguous: for `z = c(1 + i)`, `z² = 2c²i`, so the
real part is `0` and the imaginary part is `+inf` once `2c²` overflows. C99
Annex G specifies exactly this recovery for complex multiplication — when the
naive formula yields NaN and at least one operand is infinite, the result is
directed-infinite, not NaN.

### Extent

Not surfaced by the sweep grid: the complex `mul`/`sqr` grid places no point
where the componentwise products overflow. Found by direct probe. QF and TF
were not probed at the corresponding magnitude and should be checked before
scoping the fix.

### Closing it

Add the Annex G recovery to the complex multiply in `dd_complex.hpp` and
`ff_complex.hpp` (and QF/TF if they share it): after computing the naive
formula, if either output is NaN and either operand has an infinite component,
recompute with the infinities normalized to ±1/±0 and the finite parts to their
signs, then scale. Then extend the complex sweep grid to cover an overflowing
componentwise product, alongside the KI-19 `div` and KI-27 `mul` grid gaps.

### RESOLVED 2026-09-04 — commit `fix: KI-23 QF log family at large argument; KI-28 complex multiply Annex G recovery`

#### The filed scope was too narrow — all four backends needed it

This entry said "QF/TF unprobed". They were probed here, and **all four backends
carry the identical naive formula and all four fail**:

| backend | probe | before |
|---|---|---|
| DD | `(1e300 + 1e300i)²` | `(-nan, -nan)` |
| FF | `(1e30 + 1e30i)²` | `(-nan, -nan)` |
| QF | `(1e30 + 1e30i)²` | `(-nan, inf)` |
| TF | `(1e30 + 1e30i)²` | `(-nan, inf)` |

QF and TF differ only in that their imaginary part already survived — their
`multiply` primitive returns `+inf` on overflow, so `2·a·b` was correct while
`re = ac − bd` was still `inf − inf`. The real part is NaN on every backend.

#### What C99 Annex G actually requires — and where it is not enough

Annex G.5.1 gives an explicit recovery box for complex multiplication, and it is
narrower than this entry assumed. It fires when an **operand** has an infinite
component: each infinity is normalised to `±1`, each finite part to its sign,
any NaN in the *other* operand to a signed zero, and the result is recomputed as
`inf × (normalised product)`. That box is transcribed verbatim into all four
headers and handles e.g. `(inf + 1i)·(2 + 3i) → (inf, inf)`.

**It does not cover the filed case.** For `(1e300 + 1e300i)²` every operand
component is *finite*; the infinities appear only in the intermediate products.
Annex G reaches its third clause, finds no NaN operand to zero, and recomputes
`inf × (inf − inf)` = `inf × NaN` = **NaN**. glibc's `__muldc3` does exactly this
and returns NaN. Following Annex G to the letter would have left the filed defect
open.

So a second recovery is implemented, for finite operands whose products overflow:
recompute on operands scaled down by an exact power of two, then scale the result
back up in two steps, letting the overflow happen once, at the end, on the
component that genuinely overflows. `2^-513` is the largest DD scale that cannot
overflow (`|a|,|b| ≤ 2^1024` gives a scaled product `≤ 2^1022`, and a sum of two
of those `≤ 2^1023`); the FP32 backends use `2^-65` on the same argument. All
scaling is by powers of two, so the recomputation is exact except at the final
deliberate overflow. Components more than ~2^513 (~2^65) below their partner's
magnitude flush to zero, costing a relative term far below the format's
resolution — and the alternative is NaN.

Order of attempt: NaN operand → propagate NaN unchanged (NaN is the right
answer); infinite operand → Annex G.5.1 box; otherwise → scaled retry.

#### The contract, asserted

*For complex `a*b` where the true result has an infinite component and neither
operand is NaN, the result carries the correct infinity with the correct sign,
not NaN.* Measured after:

| backend | probe | before | after | true |
|---|---|---|---|---|
| DD | `(1e300 + 1e300i)²` | `(-nan, -nan)` | **`(0, +inf)`** | `(0, +inf)` |
| DD | `(1e300 + 0i)²` | `(-nan, 0)` | **`(+inf, 0)`** | `(+inf, 0)` |
| DD | `(0 + 1e300i)²` | `(-nan, 0)` | **`(-inf, 0)`** | `(-inf, 0)` |
| DD | `(inf + 1i)·(2 + 3i)` | `(-nan, -nan)` | **`(+inf, +inf)`** | `(+inf, +inf)` |
| DD | `(2 + 3i)·(inf + 1i)` | `(-nan, -nan)` | **`(+inf, +inf)`** | `(+inf, +inf)` |
| FF | `(1e30 + 1e30i)²` | `(-nan, -nan)` | **`(0, +inf)`** | `(0, +inf)` |
| FF | `(inf + 1i)·(2 + 3i)` | `(-nan, -nan)` | **`(+inf, +inf)`** | `(+inf, +inf)` |
| QF | `(1e30 + 1e30i)²` | `(-nan, inf)` | **`(0, +inf)`** | `(0, +inf)` |
| QF | `(inf + 1i)·(2 + 3i)` | `(inf, inf)` | `(+inf, +inf)` | `(+inf, +inf)` |
| TF | `(1e30 + 1e30i)²` | `(-nan, inf)` | **`(0, +inf)`** | `(0, +inf)` |
| TF | `(inf + 1i)·(2 + 3i)` | `(inf, inf)` | `(+inf, +inf)` | `(+inf, +inf)` |

Note the sign discrimination: `(1e300 + 0i)²` gives `+inf` and `(0 + 1e300i)²`
gives `−inf` in the real part, which is the whole point of recovering rather
than clamping.

#### One case that stays NaN, correctly

`(inf + 0i)·(0 − 1i)` returns `(-nan, -inf)`. This is **not** a residual defect:
the real part is `inf·0`, which is genuinely indeterminate, and Annex G's own box
produces NaN there (`inf × (1·0 − 0·(−1))` = `inf × 0`). The imaginary part is
`−inf` and correct. A projective reading of the complex infinity would give
`(0, −inf)`, but deviating from the standard on a case the standard explicitly
decides is not worth the divergence. Matching Annex G here is deliberate.

#### Not surfaced by the sweep

Unchanged from the filing: the complex `mul`/`sqr` grid still places no point
where the componentwise products overflow, so the sweep neither caught this nor
scores the fix. Extending the complex grid to cover an overflowing componentwise
product remains open, alongside the KI-19 `div` and KI-27 `mul` grid gaps. The
0 decreases this change contributes to the monotone gate therefore reflect a grid
gap, not a verification.

---

## KI-29 — `asinh`'s odd reflection costs a few ulps for 1 ≲ |x| ≲ 20, where the unreflected form's subtraction is Sterbenz-exact **[OPEN]**

**Severity: LOW.** Filed 2026-09-04 while fixing KI-17 and KI-22. Affects
`asinh` at negative arguments in a bounded mid-band, in all four backends;
worst measured loss 1.21 digits (TF), typically < 0.5.

**What happens.** For a < 0 the shipped expression `log(a + sqrt(a²+1))` forms
`s − |a|`, and Sterbenz's lemma makes that subtraction EXACT whenever
s ≤ 2|a|, i.e. |a| ≥ 1/√3. The reflected form `−log1p(|a| + a²/(1+s))` instead
rounds an addition of two numbers near |a|. The two are algebraically identical,
so the difference is pure rounding — but at isolated points in roughly
1 ≲ |x| ≲ 20 the unreflected form happens to land closer to the true value.

**Why it is not simply reverted.** Outside that band the unreflected form is
catastrophically worse — it is exactly KI-17's defect — and the reflection is
required for `asinh(−x) == −asinh(x)` to hold at all. Measured, at negative
arguments (digits, oracle `__float128`):

| \|a\| | DD unrefl | DD refl | FF unrefl | FF refl | QF unrefl | QF refl | TF unrefl | TF refl |
|---|---|---|---|---|---|---|---|---|
| 1.5 | 30.53 | 30.54 | 13.88 | 14.13 | 29.02 | 27.92 | 20.92 | 20.61 |
| 6 | 30.38 | 30.44 | 12.84 | 13.58 | 28.21 | 29.07 | 20.88 | 21.77 |
| 12 | 31.26 | 31.42 | 12.30 | 13.85 | 28.81 | 28.88 | 22.60 | 21.54 |
| 96 | 28.59 | 31.07 | 11.25 | 14.40 | 27.05 | 28.90 | 19.86 | 21.93 |
| 1536 | 26.27 | 31.39 | 8.59 | 14.48 | 24.31 | 29.60 | 16.72 | 22.12 |
| 12288 | 24.40 | 31.15 | 8.51 | 14.81 | 24.40 | 30.13 | 15.59 | 22.09 |

The reflected form wins almost everywhere and wins by 5–7 digits at large |a|;
it loses by a few ulps at scattered mid-band points. Across the real grid the
trade is 750 improvements against 92 decreases on TF, 140/42 on QF, 132/4 on FF
and 74/2 on DD, with every backend's mean rising. Those 140 decreases were
ACCEPTED in batch 5 on that basis and are the only decreases it adds.

**What closing this involves.** Selecting per point between the two algebraically
equal forms, which needs a criterion sharper than "is the subtraction exact" —
exactness holds throughout the band, yet the winner alternates, so the residual
is in `log`/`log1p`'s own rounding rather than in the argument. A doubled-`log`
error term, or an error-free `log1p` argument (keeping `x²/(1+s)` as an unevaluated
sum), would decide it. Until then the reflected form is the right default: it is
never catastrophic, and it is the only one that is odd.

## KI-30 — DD `multiply` returns NaN whenever either operand exceeds ~1.34e300, because Dekker's splitter overflows **[RESOLVED batch-7]**

**Severity: medium, DD only. Filed 2026-09-04 while fixing KI-23; measured, not
fixed — the KI-23 fix works around it locally rather than closing it.**

### What

`dd_math.hpp`'s `multiply` forms Dekker's split `(2^27 + 1)·x`. That product
overflows FP64 once `|x| > DBL_MAX/(2^27 + 1) ≈ 1.34e300`, and the resulting
`inf` propagates into the error term, so the whole DoubleDouble comes back NaN —
even though the true product is perfectly representable.

Measured on the shipped tree, with the true product exactly `1.0` in every row:

| call | result |
|---|---|
| `multiply(1e-296, 1e296)` | `1.0` (lo `-1.296e-17`) |
| `multiply(1e-298, 1e298)` | `1.0` (lo `8.409e-17`) |
| `multiply(1e-300, 1e300)` | `1.0` (lo `7.756e-17`) |
| `multiply(1e-302, 1e302)` | **`(-nan, -nan)`** |
| `multiply(1e-304, 1e304)` | **`(-nan, -nan)`** |
| `multiply(1e-306, 1e306)` | **`(-nan, -nan)`** |
| `multiply(1e-308, 1e308)` | **`(-nan, -nan)`** |

The cutover between 1e300 and 1e302 is exactly `ln(DBL_MAX/(2^27+1))`, which
identifies the splitter as the mechanism and rules out genuine overflow: the
answer is `1.0`, not a large number.

This is the **same class** as KI-9 (QF/TF `div` outrunning the Dekker splitter,
resolved at `82427f6`/`0ad44fe`) and the FP64 instance of it. KI-9 was fixed in
`divide`; `multiply` was not audited at the time. It is distinct from KI-27,
which was about `multiply` returning NaN where the product *genuinely* overflows
and `±inf` is correct — here nothing overflows but the splitter.

### Extent

DD only, and only for operands above ~1.34e300 — the top ~8 decades of FP64,
about 2.6% of the exponent range. The FP32 backends do not show it at the
corresponding point (`multiply(1e-38, 1e38)` is exact on FF, QF and TF), so
whatever guard they carry is absent from `dd_math.hpp`.

Not surfaced by the sweep: the real `mul` grid places no point with an operand
above 1.34e300.

### How KI-23 works around it

`dd_math.hpp`'s `log` now evaluates `e^{|b|}` for `a < 1e-291`, which runs up to
1.8e308 and hits this directly — it was the reason DD `log(1e-307)` first came
back NaN. The KI-23 fix rebalances *that one product* by an exact power of two
before multiplying (`s0` down by 2^10 up to three times, `aa` up by the same),
which is correct and exact but is local to `log`. Every other caller of DD
`multiply` is still exposed.

### Closing it

Guard the splitter in `dd_math.hpp`'s `two_prod`/`multiply` the way `divide` was
guarded for KI-9: scale the operand down by an exact power of two before
splitting and scale the two output words back afterwards, or use an FMA-based
`two_prod` where the compiler provides one. Then remove the local rebalance in
`log` and extend the real `mul` grid above 1.34e300.

### RESOLVED 2026-09-04 — commit `fix: KI-30 scale multiply past the Dekker splitter limit; KI-24 fixed`

Fixed at the primitive, by the route this section proposed and the one QF and TF
already used. `dd_math.hpp` gained **`dd_split()`** — the FP64 counterpart of
`qf_split`/`tf_split`, itself a port of QD's `qd::split` (QD 2.3.24
`qd/include/qd/inline.h:66-83`) whose threshold branch the DD port had dropped.
Above `DBL_MAX / (split + 1) = 1.3393857e300` it pre-scales by `2^-28`, splits,
and unscales both output words by `2^28`. All three constants are QD's own.

The scaling sits **inside the split**, not around the operand. That matters: the
product `p = a.hi * b.hi` is never touched, so the non-hazard path performs the
same three operations in the same order as the unguarded code it replaced and is
**bit-identical** to it. (This is `qf_split`'s design; `ff_math.hpp`'s B9 guard
takes the other route — scale the operand, unscale the result — which is also
exact but does perturb the hazard path.) The three unguarded DD sites —
`multiply`, `multiply_scalar` and `two_prod` — now all call it.

Measured on the fixed tree, with the true product exactly `1.0` in every row:

| call | before | after | true product representable? |
|---|---|---|---|
| `multiply(1e-296, 1e296)` | `1.0` | `1.0` | yes |
| `multiply(1e-300, 1e300)` | `1.0` | `1.0` | yes |
| `multiply(1e-301, 1e301)` | **NaN** | `1.0000000000000002` | yes |
| **`multiply(1e-302, 1e302)`** | **NaN** | **`1.0`** | **yes** |
| `multiply(1e-304, 1e304)` | **NaN** | `0.99999999999999989` | yes |
| `multiply(1e-306, 1e306)` | **NaN** | `1.0` | yes |
| `multiply(1e-308, 1e308)` | **NaN** | `0.99999999999999989` | yes |
| `multiply(2^-1020, 2^1020)` | **NaN** | `1.0` (exact) | yes |
| `multiply(1e300, 1e300)` | `+inf` | `+inf` | **no** — genuine overflow, KI-27 |

The rows that land one ulp off `1.0` do so because `1e±304` and `1e±308` are not
exact powers of ten in FP64, so their true product is not exactly `1.0` either;
the exact power-of-two pair `(2^-1020, 2^1020)` returns exactly `1.0`, which is
the clean check. The last row confirms KI-27's genuine-overflow semantics
survive: the guard on `p` still runs first and still returns `±inf`.

**Threshold, before and after.** DD `multiply`, `multiply_scalar` and `two_prod`
failed from `2^997` (1.339e300) and are now clean through `2^1023` (8.99e307),
i.e. to the top of the format. Full audit table below.

**The KI-23 local workaround was removed.** `dd_math.hpp`'s `log` carried a loop
that rebalanced `multiply(a, s0)` by `2^10` up to three times to keep `s0`
(which runs to 1.8e308) out of the hazard band. `multiply` now guards its own
splitter, so the rebalance is redundant; it is deleted and the call is a plain
`multiply(a, s0)`. DD `log` accuracy is unchanged by the removal — the monotone
gate reports zero decreased points across all 428,592.

### The splitter audit

Every Dekker/Veltkamp split site in all four backends, with the operand
magnitude at which the op returns a non-finite or wrong result while the true
result is still representable. Measured by binade scan
(`multiply(1/2^k, 2^k)` and friends, true result `1.0`).

| backend | site | guard before | threshold before | threshold after |
|---|---|---|---|---|
| DD | `multiply` | **none** | **fails from 2^997 (1.34e300)** | clean to 2^1023 |
| DD | `multiply_scalar` | **none** | **fails from 2^997 (1.34e300)** | clean to 2^1023 |
| DD | `two_prod(double,double)` | **none** | **fails from 2^997 (1.34e300)** | clean to 2^1023 |
| DD | `divide` | KI-19 (divisor + quotient) | clean to 2^1023 | clean to 2^1023 |
| DD | `divide_scalar` | KI-19, **unscale missing** | **wrong value from 2^997** — see KI-31 | clean to 2^1023 |
| DD | `sqrt` | via `two_prod`; operand ≤ 1.3e154 | clean to 2^1023 | clean to 2^1023 |
| FF | `multiply` | B9 (both operands) | clean to 2^127 | unchanged |
| FF | `multiply_scalar` | B9 | clean to 2^127 | unchanged |
| FF | `two_prod(float,float)` | B9 | clean to 2^127 | unchanged |
| FF | `divide` | B8/B10 | clean to 2^127 | unchanged |
| FF | `divide_scalar` | B8/B9 | clean to 2^127 | unchanged |
| QF | `qf_split` (all consumers) | KI-9 | clean to 2^127 | unchanged |
| QF | `multiply` / `multiply_scalar` / `sqr` | via `qf_split` | clean to 2^127 | unchanged |
| QF | `divide` / `divide_scalar` | via `qf_split` | clean to 2^127 | unchanged |
| TF | `tf_split` (all consumers) | KI-9 | clean to 2^127 | unchanged |
| TF | `multiply` / `multiply_scalar` / `sqr` | via `tf_split` | clean to 2^127 | unchanged |
| TF | `divide` / `divide_scalar` | via `tf_split` | clean to 2^127 | unchanged |

Notes on the audit:

- **DD was the only backend with unguarded split sites.** FF re-derived the
  guard independently as B8/B9/B10; QF and TF carry it inside `qf_split`/
  `tf_split`, which every one of their product and quotient paths routes
  through. There is no raw `* split` anywhere in QF or TF outside those two
  functions, and none at all in any of the four `*_complex.hpp` headers — the
  complex paths consume the splitter only through `multiply`/`divide`, so
  fixing the primitive fixes them.
- `2^115` (4.15e34) is the FP32 threshold and `2^997` (1.34e300) the FP64 one;
  both are `FORMAT_MAX / (split + 1)`.
- **`sqr` on QF and TF is clean.** An earlier scan of this audit flagged both
  from `2^100`, but that was the scan's own fault: `sqr(2^100) = 2^200`
  genuinely overflows FP32. Re-run over `k <= 63`, where the square is
  representable, QF and TF `sqr` are exact at every point.
- DD `sqrt` never reaches the hazard band — it splits `t2 ~ sqrt(a.hi)`, at most
  1.3e154 — but it now inherits the guard anyway through `two_prod`.

---

## KI-31 — DD `divide_scalar` returns a quotient 2^64 too large above 1.339e300 **[RESOLVED batch-7]**

**Severity: high — a silently wrong finite value, not a NaN. DD only. Found
2026-09-04 by the KI-30 splitter audit; fixed in the same commit.**

### What

`divide_scalar(a, b)` for `|a.hi|` or `|b|` above `1.3393857e300` returns a
result exactly `2^64` times too large. Not non-finite, not lost digits — a
plausible-looking finite number that is wrong by nineteen orders of magnitude.

| call | true quotient | returned before | returned after |
|---|---|---|---|
| `divide_scalar(2^997, 2^997)` | `1.0` | **`1.8446744073709552e19`** (= 2^64) | `1.0` |
| `divide_scalar(2^1000, 2^1000)` | `1.0` | **`1.8446744073709552e19`** | `1.0` |
| `divide_scalar(2^1023, 2^1023)` | `1.0` | **`1.8446744073709552e19`** | `1.0` |
| `divide_scalar(2^996, 2^996)` | `1.0` | `1.0` | `1.0` |

The threshold is `DBL_MAX / (split + 1)` — the same 1.339e300 as KI-30, because
it is the same hazard band.

### Mechanism

This is a **defect in the KI-19 fix, not in the original code**. KI-19 added a
scale-and-unscale guard to `divide_scalar`: when an operand is in the hazard
band it scales the dividend by `sd` and the divisor by `un`, computes the
quotient on the scaled values, and must multiply the two output words by
`sd * un` on the way out. The scaling was written; the unscale was not. The
function ended with a bare

```cpp
    return DoubleDouble(hi, lo);
```

so `sd` and `un` were computed, applied to the inputs, and then dropped. With
both operands in the band, `sd * un = 2^64`, which is exactly the observed
error. Below the band the guard does not fire, `sd = un = 1`, and the bug is
invisible — which is why it survived KI-19's own validation and why the sweep
grid, which stops well short of 1e300, never saw it.

The sibling `divide(DoubleDouble, DoubleDouble)` applies its unscale correctly;
only the scalar overload was missing it. FF, QF and TF `divide_scalar` are all
correct — measured clean through 2^127.

### Resolution

One line, restoring the unscale the guard intended:

```cpp
    return DoubleDouble(hi * sd * un, lo * sd * un);
```

Both `sd` and `un` are exact powers of two, so the correction introduces no
rounding. Verified over every binade from 2^900 to 2^1023 with the true quotient
held at 1.0. The monotone gate reports no change to any existing point, since no
gridded input reaches the band — which is also the reason this entry is filed:
the sweep cannot currently see this class of defect, and the `mul` and `div`
grids should be extended above 1e300 to cover it.

---

## KI-32 — FF complex `asin` returns ZERO for its real part on the far real axis **[OPEN]**

**Severity: medium (a full component lost, no NaN to warn you). FF only, found
while closing KI-11 in batch-8. Not caused by that change — the pre-fix code
had the identical cancellation.**

### What

```
FF asin(1e8 + 1e-8i)   ref = (1.570796e+00, 1.911383e+01)
                       got = (0.000000e+00, 1.911383e+01)     0.00 digits on Re
```

`Re asin(z) = arg(iz + sqrt(1 - z^2))`. At `z = 1e8 + 1e-8i` the root is
`(1e-8, -1e8)` and `iz` is `(-1e-8, 1e8)`, so **both** components of the sum are
a subtraction of two equal quantities. In FF's two FP32 words the real parts
cancel to exactly 0 (their true difference is ~1e-24, twenty orders below FF's
resolution at 1e-8) and the imaginary parts likewise, leaving `atan2(0, 0) = 0`.
DD, QF and TF have enough words to survive it and score 31.00 / 29.00 / 21.70 at
the same point.

### Why it is a separate entry from KI-11

KI-11 is the *small component* being destroyed by a modulus. This is the *large*
component being destroyed by a cancellation in the argument of `atan2`, it is
FF-only, and it is unchanged by the KI-11 fix (the old `log(sum).im` and the new
`atan2(sum.im, sum.re)` read the same `sum`). Conflating them would have made
the KI-11 close look conditional when it is not.

### Closing it

Compute `Re asin` from a form that does not build `iz + sqrt(1-z^2)`
explicitly — Hull, Fairgrove & Tang give
`Re asin(z) = atan2(x, Re sqrt(1-z^2))` with the same `r`/`s` machinery
`xp_asin_imag_mag()` already carries, which never forms the cancelling sum. The
helper is in place in all four headers, so this is a small change; it was left
out of batch-8 to keep that change's measurement clean.

---

## Note on resolution markers — CLOSED 2026-09-04

This section used to track a drift between what the headings said and what the
bodies said. That drift is now fixed and the convention is stated once, at the
top of this file, in the **STATUS TABLE** section. Every `## KI-n` heading
carries exactly one marker — `[RESOLVED <commit>]` or `[OPEN]` — and the status
table is the authoritative index.

### What the 2026-09-04 audit found

Four headings disagreed with their own bodies or with the measured behaviour of
the library. Each was verified against git history **and** a measurement before
being changed; none was taken on trust from the body text.

| KI | heading was | heading is now | how it was verified |
|---|---|---|---|
| 8 | `[REOPENED 2026-09-03]` | `[RESOLVED 7680809]` | measured: QF `hypot(1e-16,1e-16)` = 1.4142135862e-16 and `hypot(1e30,1e30)` = 1.4142135837e+30, both ends finite and correct |
| 13 | *(no marker — read as open)* | `[RESOLVED 7680809]` | measured: DD `asinh(1e30)` = `acosh(1e30)` = 69.7707, no NaN. Same commit and same root cause as KI-8 (unscaled sum of squares, different call sites) |
| 18 | *(no marker — read as open)* | `[RESOLVED 855292d]` | measured: DD complex `tanh(x+1i)` saturates to ±1 with a vanishing imaginary part across x = ±30, ±100, ±700 — no NaN anywhere on the real axis |
| 19 | `[RESOLVED]`, but titled *"KI-9 is NOT fully closed"* while KI-9 also read `[RESOLVED]` | retitled to name its own defect; KI-9 gained a forward pointer | git: `82427f6` fixed the splitter ceiling (KI-9), `0ad44fe` fixed the format ceiling above it (KI-19). Two defects, two magnitudes, neither supersedes the other |

**A fifth mismatch was looked for and one was found, outside this file.** KI-3
is correctly marked `[RESOLVED dd6d00a]` here and `scripts/build_with_kokkos.sh`
lines 67 and 74 do now read `-DCMAKE_CXX_STANDARD=20`, but the repository's
`CLAUDE.md` still documents the C++17 trap as *"Reported in the S1 STATUS block,
not yet fixed"*. `CLAUDE.md` was outside this batch's edit scope, so it is
recorded here rather than changed. Two lesser cases were also checked and are
**not** mismatches: KI-11 was genuinely `[OPEN]` (`855292d` fixed `atan`/`atanh`
only, leaving `asin`/`acos`/`asinh`/`acosh`/`sqrt` — batch-8 has since closed
those five, so the entry now reads `[RESOLVED 855292d / batch-8]`), and KI-12's
body correctly credits `0ad44fe` with most of the fix and `2e810c2` with the
rest.

### Body-text shas that were left as written

KI-2 and KI-5(a) carry the literal placeholder `TBD-FIX` in their Resolution
subsections, and KI-3, KI-6, KI-7 and KI-9 date their Resolution without a sha —
these are all commits that were being written at the time the text was. The
status table above supplies the real sha for each; the body prose was left
untouched so that the historical record of what each session knew stays intact.
