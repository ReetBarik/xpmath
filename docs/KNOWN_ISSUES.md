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

For real `z < -1`, the principal square root returns the **positive** root, so
`z + sqrt(z²−1)` is a difference of two nearly equal quantities of opposite sign.
It cancels catastrophically, and the logarithm of what survives is noise. The
standard remedy is to select the branch on the sign of `Re(z)` — e.g. compute
`acosh(z) = ±log(z ± sqrt(z²−1))` with the sign chosen so the addition does not
cancel, or route through `2·log(sqrt((z+1)/2) + sqrt((z−1)/2))`.

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

## KI-3 — `scripts/build_with_kokkos.sh` cannot build Kokkos 5.1

**Severity: low (documented workaround), scope: pre-existing.**

Lines 67 and 74 pass `-DCMAKE_CXX_STANDARD=17` when configuring **Kokkos itself**.
Kokkos 5.1.0 rejects this at `cmake/kokkos_test_cxx_std.cmake:89`: *"Kokkos requires
C++20 or newer but requested 17"*. The consuming project correctly stays at C++17.

Found in S1. `validation/a100/build_login.sh` was corrected to 20 for its own Kokkos
configure, and `CLAUDE.md` documents the trap, but the script itself was left alone
as out of scope. It remains a live trap for anyone following the documented build
path.

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
