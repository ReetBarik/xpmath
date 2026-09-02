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

### Resolution (2026-09-02, commit `fe42970`)

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

## KI-5 — Four more defects in the complex inverse-function family, in ALL FOUR backends

> **(a) and (d) are RESOLVED.** (b) `atanh` and (c) `acos` direct form remain open.

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

### Resolution of (d) (2026-09-02, commit `fe42970`)

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
   free.~~ **Done 2026-09-02**, commit `fe42970` — see "Resolution of (d)"
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
