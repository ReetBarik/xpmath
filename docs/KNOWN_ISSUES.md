# Known issues — deferred, to be picked up after the upstream arc

Findings that are real, reproduced, and deliberately **not** fixed at the time they
were found, because fixing them was out of scope for the sub-plan that surfaced
them. Each entry states the evidence, why it was deferred, and what closing it
would involve.

This file is the backlog. It is not a status document — see
`docs/UPSTREAM_PLAN_STATUS.md` for what each sub-plan actually did.

---

## KI-1 — Complex `acosh` is wrong throughout the left half-plane, in ALL FOUR backends

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

---

## KI-2 — `nint` is wrong for the one-ulp-below-a-tie class

**Severity: low, scope: inherited from QD.**

`0.49999997f + 0.5f` rounds to exactly `1.0f`, so TF's `nint`, formulated as
`floor(x + 0.5)` per QD, returns 1 where the nearest integer is 0. A genuine wrong
answer, not a precision shortfall. FP64 QD has the same defect.

Found in S10 phase 3.5 and excluded by domain predicate rather than absorbed into a
tolerance — see `docs/PORT_NOTES_TF.md` §12f. Closing it means diverging from QD's
`nint` formulation, which the source-fidelity rule discourages without cause.

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

## KI-4 — DD `sin` returns the WRONG SIGN near odd multiples of π

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

---

## KI-5 — Four more defects in the complex inverse-function family, in ALL FOUR backends

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

**Remedy — exact, free, and one line.** `asinh` is an ODD function, so for
`Re(z) < 0` compute `-asinh(-z)`, which moves the evaluation into the
well-conditioned right half-plane. No new series, no new constants, no accuracy
cost anywhere else.

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
2. **(d)** `asin`: fix the signed-zero sheet selection; `acos` is fixed for free.
   Add on-cut test points at **both** `Im = +0` and `Im = -0`.
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
