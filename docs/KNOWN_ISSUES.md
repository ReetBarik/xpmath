# Known issues — deferred, to be picked up after the upstream arc

Findings that are real, reproduced, and deliberately **not** fixed at the time they
were found, because fixing them was out of scope for the sub-plan that surfaced
them. Each entry states the evidence, why it was deferred, and what closing it
would involve.

This file is the backlog. It is not a status document — see
`docs/UPSTREAM_PLAN_STATUS.md` for what each sub-plan actually did.

---

## KI-1 — Complex `acosh` is wrong throughout the left half-plane, in ALL FOUR backends **[RESOLVED]**

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

## KI-2 — `nint` is wrong for the one-ulp-below-a-tie class **[RESOLVED]**

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

## KI-3 — `scripts/build_with_kokkos.sh` cannot build Kokkos 5.1 **[RESOLVED]**

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

## KI-4 — DD `sin` returns the WRONG SIGN near odd multiples of π **[RESOLVED]**

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

## KI-5 — Four more defects in the complex inverse-function family, in ALL FOUR backends **[RESOLVED]**

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
commit `ad82f4f`) and **KI-8** has since been resolved (2026-09-02, commit
`d02a3bb`); KI-9 through KI-11 remain open. `docs/DOMAINS.md` records the
resulting usable ranges per backend per op, and is generated from the same data —
it was regenerated after each fix, so the counts quoted in the KI-6, KI-7 and
KI-8 *Extent* sections above are the pre-fix numbers and no longer match the
current CSVs.

---

## KI-6 — `exp` flushes to ZERO outside a hard ±300 argument guard, discarding ~170 representable decades **[RESOLVED]**

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

## KI-7 — `tanh` returns the WRONG SIGN for |x| past the `exp` guard, in all four backends **[RESOLVED]**

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

## KI-8 — `hypot` and complex `abs` have NO scaling, so they overflow and underflow far inside the representable range **[RESOLVED]**

**Severity: high, FP32-word backends (FF, QF, TF); DD unaffected.**

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

Fixed in **`d02a3bb`**, all four backends (`dd_math.hpp`, `ff_math.hpp`,
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

## KI-9 — QF/TF division returns NaN when the QUOTIENT exceeds the Dekker splitter's headroom

**Severity: medium, QF and TF only.**

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

---

## KI-10 — `fmod` and `remainder` lose half their digits when the operands are many decades apart

**Severity: medium, all backends.**

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

---

## KI-11 — The complex inverse family loses most of its digits when one component is far smaller than the other

**Severity: medium, all backends, 4,759 points.**

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
