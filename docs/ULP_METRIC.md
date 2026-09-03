# The ULP metric and its condition-aware bounds

Step 1a+1b of the accuracy-metric replacement. **This document describes the
metric, not a re-baseline.** `validation/sweep/sweep_baseline.csv` and
`docs/DOMAINS.md` are deliberately untouched and still speak digits.

## Why the digit score had to be joined

The suite scored accuracy as relative error in decimal digits and gated on the
**mean** over a cell against a **per-op tolerance**. Three separate problems:

**Wrong metric.** Relative error goes vacuous near a zero of the function —
exactly where the bugs live. Measured on this tree, DD `sin`:

| point | argument | digit score | error in ulps |
|---|---|---|---|
| 610 | `3*pi`  | 15.90 | 1.02e16 |
| 650 | `7*pi`  | 16.76 | 1.40e15 |
| 1570 | `99*pi` | 16.46 | 2.83e15 |

A 21-digit gate passes all three. KI-4 — DD `sin` returning the **wrong sign**
near odd multiples of pi — lived in precisely this region, and the accuracy
tests never saw it.

**Wrong statistic.** A mean over a cell cannot see an op that is exact at 1600
points and returns nothing at 52. That still averages ~30 and passes.

**Wrong provenance.** The per-op tolerances were set *from measurement of the
current implementation*, which encodes the implementation's defects as the
specification. That is how KI-1 (complex `acosh` on the wrong `sqrt` branch, an
O(1) wrong value) survived four ports.

## The metric

For an N-word expansion with `p` significand bits:

```
ulp(true) = |true| * 2^-p
ulps      = |got - true| / ulp(true)
```

| backend | layout | p | source |
|---|---|---|---|
| DD | 2 x FP64 | **106** | 2 x 53 |
| QF | 4 x FP32 | **96**  | 4 x 24 |
| TF | 3 x FP32 | **72**  | 3 x 24 |
| FF | 2 x FP32 | **48**  | 2 x 24 |

`p` is limbs x word significand, hidden bit included. Verified against
`include/xp/config.hpp` and the four backend types: these are the same bit
counts the sweep already relies on to argue that widening a backend value to
binary128 is *exact* (all of 106/96/72/48 fit binary128's 113).

**Zero and non-finite references.** `ulp(0)` is not defined. `ref == 0` scores
0 ulps if `got == 0` and is otherwise marked **UNSCORABLE** and excluded from
the ulp statistic; `ref` infinite or NaN is likewise UNSCORABLE. Scoring against
a zero ulp would make every miss infinite and every op with a zero in its range
ungateable, and the digit score already handles those cells exactly. A
non-finite `got` against a finite `ref` is a real, reported `+inf`.

**The digit score is kept.** `docs/DOMAINS.md` and the whole KI history are
written in digits; discarding the digit score orphans all of it. Both metrics
are reported side by side, everywhere.

## Why a flat ulp threshold would be wrong

Take DD `sin` at `x = 3*pi`:

```
kappa = |x * cot(x)| = 9.4248 / 3.673940e-16  ~  2.565e16
```

DD carries pi to ~106 bits; resolving `sin(x)` there needs ~47 decimal digits of
pi. The argument reduction therefore perturbs `x` by ~1 ulp and that
perturbation is amplified by kappa. **~2.6e16 ulps of error is inherent.** The
measured 1.0e16 is 2.5x *better* than the limit. A flat threshold fails that
point; the metric and the threshold have to change together.

## The bound

```
expected_ulps = 1 + kappa(f, x) * delta
measured      <= kUlpAllowance * expected_ulps          kUlpAllowance = 8
```

- the `1` is the final rounding;
- `kappa` is the amplification of the one-ulp perturbation the backend cannot
  avoid carrying (in the argument, in its stored constants, or both);
- `delta` is that input perturbation *in ulps of the format*, normally 1. It is
  larger for FF, because a `double` input does not fit FF's 48 bits: the stored
  operand is already off by more than one FF ulp before the algorithm runs, and
  that is charged honestly rather than assumed away.

kappa is **derived analytically, never fitted to a measurement**. For unary `f`,
`kappa = |x f'(x) / f(x)|`. For multi-argument `f` the partial condition numbers
add: `kappa = sum_i |x_i (df/dx_i) / f|`, the standard first-order bound on the
relative output perturbation from independent one-ulp relative input
perturbations.

### kappa, per real op — all 39 derived

| ops | kappa |
|---|---|
| `add` | `(\|a\|+\|b\|)/\|a+b\|` |
| `sub`, `fdim` (a>b) | `(\|a\|+\|b\|)/\|a-b\|` |
| `mul`, `div` | `2` |
| `sqrt` | `1/2` |
| `abs`, `hypot`, `copysign`, `fmax`, `fmin` | `1` |
| `fma` | `(2\|ab\|+\|c\|)/\|ab+c\|` |
| `ceil`, `floor`, `round`, `trunc` | `0` (piecewise constant) |
| `fmod`, `remainder` | `(\|a\|+\|b n\|)/\|f\|`, `n` the integral quotient |
| `exp` | `\|x\|` |
| `exp2`, `exp10` | `\|x\| ln 2`, `\|x\| ln 10` |
| `expm1` | `\|x e^x/(e^x-1)\|` |
| `log`, `log2`, `log10` | `1/\|ln x\|` |
| `log1p` | `\|x/((1+x) ln(1+x))\|` |
| `sin` | `\|x cot x\|` |
| `cos` | `\|x tan x\|` |
| `tan` | `\|x/(sin x cos x)\|` |
| `asin`, `acos` | `\|x/(sqrt(1-x^2) f(x))\|` |
| `atan` | `\|x/((1+x^2) atan x)\|` |
| `sinh` | `\|x coth x\|` |
| `cosh` | `\|x tanh x\|` |
| `tanh` | `\|x/(sinh x cosh x)\|` |
| `asinh` | `\|x/(sqrt(1+x^2) asinh x)\|` |
| `acosh` | `\|x/(sqrt(x^2-1) acosh x)\|` |
| `atanh` | `\|x/((1-x^2) atanh x)\|` |
| `pow` | `\|b\| + \|b ln a\|` |

Two carry a stated caveat. `fmod`/`remainder` hold `n` locally constant, which
is valid away from the jump in `n`; at the jump the function is discontinuous
and no first-order bound applies. The rounding ops have `f' = 0` almost
everywhere, so the bound is 1 ulp — i.e. "exact", which is what they are
required to be. Points landing on either discontinuity surface as failures,
which is the honest outcome rather than a fabricated exemption.

### What is NOT converted, and why

**Complex ops are measured in ulps but stay on the digit gate.** They are scored
`|got - ref| / (|ref| * 2^-p)` on the complex modulus and reported, and no
verdict is issued. Deriving kappa for a complex op is elementary in the same
way, but the complex scorer's zero-component rule — score a zero component
*absolutely* against the magnitude of the other component — has no ulp analogue
that is not invented, and the entire point of this change is to stop inventing
tolerances. An honest partial conversion beats a fabricated one.

## Minimum, not mean; and regions

The gate reduces by **worst point**, never a mean. Failing points are then run
through the existing `--classify` machinery and filed under their class, and the
verdict is taken per `(backend, op, region)`. Filing per region is what stops
one legitimate conditioning loss from forcing the whole op's bound open.

`UNDERFLOW` and `OVERFLOW` are **reported but not gated**: in those bands
`|true| * 2^-p` is not the backend's actual resolution (the trailing limbs are
subnormal, or the value is outside the leading word's range), so the metric's
own premise has failed and a verdict there would be noise. `ARG_RANGE`,
`CONDITIONING`, `UNEXPLAINED` and `OK` are gated.

A failing point is classified **even when its digit score is healthy** — the
whole premise is that a point can read 16 digits and still be 7e15 ulps out.

## The allowance: 8x, and the data says so

The brief said to start at 8x and report what the data needs. Worst ratio per
`(backend, op)` over the 156 real cells:

| allowance | clean cells |
|---|---|
| 1x | 89 / 156 |
| 4x | 93 / 156 |
| **8x** | **95 / 156** |
| 16x | 100 / 156 |
| 32x | 103 / 156 |
| 64x | 105 / 156 |
| 256x | 106 / 156 |

The distribution is strongly bimodal. 89 cells never reach the bound at all, and
past ~64x the curve is flat: going from 64x to 256x recovers **one** cell.
The 61 cells still failing at 8x are failing by many orders of magnitude — they
are defects or genuinely out-of-band points, not slack. **8x stays.** Loosening
it buys nothing and would only start sanctioning real error.

## Where the gate runs

```
scripts/sweep_accuracy --ulp                    # the gate; exit 4 on failure
scripts/sweep_accuracy --ulp-allowance 16       # sensitivity
scripts/sweep_accuracy --ulp-explain sin:610    # one point, both metrics, all 4 backends
```

The gate lives in the sweep because the **region classification lives there**.
The accuracy tests have no classifier, so they **report** the worst-point ulp
error alongside the digit score and remain gated on digits. That split is
deliberate, not a shortfall of ambition: a gate without a region label would
have to be either flat (wrong) or per-op-tolerance-shaped (the provenance
problem this change exists to remove).

Converted to report ulps: the four real tests, `tests/{dd,ff,qf,tf}_accuracy_test.cpp`,
via `ulp_error<Backend>()` and the `AccStats` ulp fields in `tests/test_utils.hpp`.
Each op line now carries `worst_ulp=`, `ulp_n=` and `ulp_unscorable=`.

Not converted: the four `*_complex_accuracy_test.cpp` files. They score against a
`cb::QuadRef` triple-double from `tests/corpus_binary.hpp` rather than a
`__float128`, so an ulp measure there needs a reference-side helper in a file
outside this change's scope — and the complex zero-component rule above has no
ulp analogue that is not invented. Tracked for step 1c.

## The gate bites

`exp2` at grid point 110, argument `0.0031622776601683794` — tiny and
**perfectly conditioned** (kappa = 0.0022, so the bound is 1.002 ulps):

| backend | digit score | measured ulps | expected | ratio | verdict |
|---|---|---|---|---|---|
| DD | **29.71** | 156.4 | 1.002 | 156 | **FAIL** |
| FF | 14.00 (cap) | 0.4198 | 1.002 | 0.42 | PASS |
| QF | 27.42 | 29.95 | 1.002 | 29.9 | **FAIL** |
| TF | 20.90 | 5.923 | 1.002 | 5.91 | PASS |

DD scores 29.71 digits. That clears the DD gate of 25.91 by four digits and sits
within 1.3 digits of the 31-digit cap — the digit metric calls this essentially
perfect. It is 156 ulps wrong. The digit metric also ranks DD (29.71) *above* FF
(14.00) at a point where FF is correctly rounded and DD is not.

And the three `sin(k*pi)` points, which a flat ulp gate would wrongly fail, pass:

| point | digits (DD) | kappa | measured ulps | expected | ratio | verdict |
|---|---|---|---|---|---|---|
| `3*pi`  | 15.90 | 2.565e16 | 1.015e16 | 2.565e16 | 0.396 | PASS |
| `7*pi`  | 16.76 | 2.565e16 | 1.401e15 | 2.565e16 | 0.055 | PASS |
| `99*pi` | 16.46 | 2.440e16 | 2.829e15 | 2.440e16 | 0.116 | PASS |

All four backends pass all three. Note FF scores **0.00 digits** at every one of
them and passes the ulp gate comfortably — FF being honest about a 2xFP32
layout, which is exactly the judgment the digit metric could not make.

---

# Step 1c — triage of the 172, and the twelve KIs re-verified

Diagnosis and threshold work only. No KI was fixed, `sweep_baseline.csv` and
`docs/DOMAINS.md` are untouched, and the monotone gate still passes with
**0 decreased / 0 increased**.

## Reproducing any number below

```
module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3
g++ -std=c++17 -fext-numeric-literals -O2 -Iinclude \
    scripts/sweep_accuracy.cpp -lquadmath -o /tmp/sweep
export LD_LIBRARY_PATH=/soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64:$LD_LIBRARY_PATH
/tmp/sweep --ulp --ulp-dump /tmp/dump.csv
/tmp/sweep --ulp --ulp-explain exp2:110
```

**The `LD_LIBRARY_PATH` line is not optional.** Loading the gcc module fixes the
*compiler* but not the *runtime* `libquadmath`: the binary still resolves
`/usr/lib64/libquadmath.so.0`, whose `sinq`/`cosq` differ in the last bits. That
alone produces the oracle-fingerprint mismatch and the phantom "7 decreased" the
previous session saw. With the path set, the fingerprint matches
`578322f998a329c8` and the gate is clean. This is the same trap recorded for the
build; it bites the *run* too.

## What the 172 actually were

| stage | change | cells left |
|---|---|---|
| step 1a/1b as committed | — | **172** |
| + subnormal-low-word floor, both ends of the bound, + `SUBNORMAL_LIMB` class | (a) | **134** |
| + exp-family algorithm floor `2^nq` | (c) | **90** |
| + log-family floor `2^nq / \|ln x\|` | (c) | **67** |

Classification of all 172:

| class | cells | |
|---|---|---|
| **(a) SUBNORMAL-LOW-WORD** — inherent format limit | **41** | 38 cleared by the floor model or filed under the new exemption class; 3 more are `fma` whose exact product exceeds the FP32 word max |
| **(b) THRESHOLD TOO TIGHT** | **0** | see below |
| **(c) KAPPA / BOUND WRONG** | **89** | 67 fixed in this commit, 22 diagnosed with the derivation stated and the code fix deferred |
| **(d) REAL DEFECT** | **42** | 22 already filed as open KIs, 16 newly filed (KI-22…KI-26), 4 reopened (KI-8, KI-10/KI-15) |

**(a) does NOT dominate.** The brief expected it to; it is 41 of 172, second to
(c) at 89. The single largest cause of the red was not the subnormal band — it
was that the bound charged every op the cost of a correctly-rounded
implementation, and `exp` and `log` are not correctly rounded and never claimed
to be. Saying so plainly, as asked.

**(b) is empty, and that is a result, not an omission.** Not one cell was
resolved by "the allowance is wrong for this op." Every cell reduced to a
missing *term* in the bound, a format limit, or a defect. **The 8x allowance
stays at 8x**, unchanged, and no per-op allowance was introduced.

## (a) The subnormal-low-word floor, derived per backend

An N-limb expansion is N words of the base type and nothing else, so the finest
absolute increment it can express — and the smallest residual an algorithm can
form inside it — is the base type's smallest **subnormal**, `2^Emin_sub`. At
magnitude `|v|` the achievable relative resolution is therefore
`max(2^-p, 2^Emin_sub/|v|)`, i.e. an unavoidable error floor of

```
floor_ulps(|v|) = min( 2^p, max(1, 2^(Emin_sub + p) / |v|) )
```

in the nominal ulps the metric counts. The `2^p` clamp is the point at which the
format holds *nothing* there; without it the bound runs away as `1/|v|` and the
exemption becomes vacuous.

**The cliff per backend** — the `|hi|` below which the trailing limb cannot be
normal, which is exactly `Range::full_prec`:

| backend | p | Emin_sub | cliff = 2^(Emin+(N-1)s) | bound departs 1 at 2^(Emin_sub+p) |
|---|---|---|---|---|
| DD | 106 | −1074 | 2^−969 = **2.0042e−292** | 2^−968 = 4.0079e−292 |
| QF | 96 | −149 | 2^−54 = **5.5511e−17** | 2^−53 = 1.1102e−16 |
| TF | 72 | −149 | 2^−78 = **3.3087e−24** | 2^−77 = 6.6174e−24 |
| FF | 48 | −149 | 2^−102 = **1.9722e−31** | 2^−101 = 3.9443e−31 |

(The factor 2 between the two columns is the hidden bit.) The brief's
hand-derived DD cliff of ~5.6e−292 and FF cliff of ~2.5e−32 were the right
mechanism at the wrong binade; the figures above come from the format.

**The failing points lie below it.** Measured, via `--ulp-explain`:

| point | backend | x | cliff | in_delta | measured | bound | ratio |
|---|---|---|---|---|---|---|---|
| `sqrt:563` | DD | 9.881e−324 | 2.00e−292 | 4.056e31 | 7.192e15 | 2.028e31 | 3.5e−16 |
| `sqrt:0` | QF | 1e−30 | 5.55e−17 | 1.110e14 | 2.930e12 | 5.551e13 | 0.053 |
| `sqrt:0` | TF | 1e−30 | 3.31e−24 | 6.617e6 | 1.746e5 | 3.309e6 | 0.053 |
| `sqrt:0` | FF | 1e−30 | 1.97e−31 | 1 (above cliff) | 0.0101 | 1.5 | 0.007 |
| `mul:5` | QF | −1e−29 | 5.55e−17 | 1.110e13 | 5.328e20 | 1.659e21 | 0.32 |
| `mul:5` | TF | −1e−29 | 3.31e−24 | 6.617e5 | 3.176e13 | 9.889e13 | 0.32 |
| `div:34` | QF | 3.162e−22 | 5.55e−17 | 3.511e5 | 5.580e13 | 1.480e14 | 0.38 |

FF is the control: at `x = 1e−30` FF is the one backend still **above** its
cliff, its `in_delta` stays 1, and it is the one backend that does not blow up.
The model predicts which backends fail and by how much, from the format alone.

### Point 563

`validation/sweep/sweep_grid.csv` row `r,563,ulp,-9.8813129168249309e-324` —
i.e. **−2^−1073, the second-smallest FP64 subnormal**, from the `ulp` probe
family. Not on any log sweep, which is why the independent 1e−300..1e300 probe
never reproduced it.

DD holds that input *exactly* (it is one bit), so its storage error is zero and
step 1a charged `in_delta = 1`. But `sqrt`'s Newton correction forms the
residual `a − ax²` at scale `|a|·2^−53 = 2^−1126`, three binades below the
smallest FP64 subnormal, so it **flushes to zero** and the result is the leading
double only. 7.192e15 measured ≈ 2^52.7 — exactly one word's worth of accuracy,
and the 16.05 digits the brief flagged. Against the floor-widened bound of
2.028e31 it now passes with 15 orders of margin. **Inherent, not a defect.**

### The mul/div claim — (a), confirmed

`FF:5 1.9e6 ulps at 8.17 digits`, `QF:2 2.7e13 at 15.47`, `TF:31 6.4e8 at 12.87`
against a bound of 3. All three inputs are at 1e−29…1e−30, below the QF and TF
cliffs; the true products are at ~1e−58, far below all of them. Every one now
passes at ratio ~0.32. **Class (a). No defect in `mul` or `div`.**

## (c) The two bound terms that were missing

### The exp family: `2^nq`

All four backends implement `exp` as the same textbook skeleton —
`s = x − m·ln2; r = s/2^nq; t = Taylor(r);` **square t nq times**
(`dd_math.hpp:374`, `ff_math.hpp:606`, `qf_math.hpp:927`, `tf_math.hpp:844`).
Squaring propagates relative error at gain 2, so nq squarings multiply the
Taylor sum's error by `2^nq`. nq is **6 / 4 / 6 / 5** for DD / FF / QF / TF,
read from the headers, not fitted.

### The exp2 claim — (c), not a defect

`exp2` grid point 110, `x = 0.0031622776601683794`, kappa 0.0022:

| backend | nq | 2^nq | measured ulps | measured / 2^nq | verdict |
|---|---|---|---|---|---|
| DD | 6 | 64 | 156.4 | **2.44** | PASS |
| QF | 6 | 64 | 29.95 | **0.47** | PASS |
| TF | 5 | 32 | 5.923 | **0.19** | PASS |
| FF | 4 | 16 | 0.4198 | **0.03** | PASS |

Every backend lands within a small constant of its own `2^nq` — i.e. the Taylor
sum is good to a couple of ulps everywhere and the squarings do the rest. That
is the signature of a correct implementation of this algorithm, not of a defect.
Step 1a's bound of 1.002 was simply missing the term. **Wrong kappa (bound), not
a real defect.** Applied to `exp`, `exp2`, `exp10`, `expm1`, `pow`, `sinh`,
`cosh`, `tanh`; conservative on the small-|x| Taylor branches of `expm1` and
`sinhcosh`, which is stated in the code.

### The log family: `2^nq / |ln x|`

`log` is Newton **on** `exp` in all four backends (`dd:379`, `ff:70`, `qf:32`,
`tf:35` all iterate `b ← b + (a − exp(b))/exp(b)`). The iteration cannot
converge past the accuracy of the `exp` it inverts, so it leaves an *absolute*
error of ~`2^nq·2^-p` in the natural log; converting that to ulps of the result
divides by `|ln x|`. Verified across three decades of `|ln x|`:

| cell | x | \|ln x\| | measured ulps | 2^nq/\|ln x\| | ratio |
|---|---|---|---|---|---|
| DD `log` | 1.4 | 0.3365 | 889 | 190 | 4.7 |
| DD `log` | 37.7 | 3.630 | 62 | 17.6 | 3.5 |
| DD `log1p` | 0.4 | 0.3365 | 1010 | 190 | 5.3 |
| FF `log` | 2.65 | 0.9745 | 35.7 | 16.4 | 2.2 |
| TF `log` | 1−1e−9 | 1e−9 | 7.4e10 | 3.2e10 | 2.3 |

The `1/|ln x|` shape is the prediction, and it holds over eleven orders of
magnitude of the floor. `log2`/`log10` are the same natural log rescaled, so
both the absolute error and `|f|` scale together and the ulp floor is unchanged.

### (c) diagnosed, code fix deferred — 22 cells

Two derivations are established below but **not implemented**, because
implementing either needs a term whose gain I have not yet derived to the
standard the rest of this file is held to. Deferred to step 1d; the numbers are
here so the work is not re-done.

**Trig at large argument (4 cells: DD `cos`/`sin` at 1e17, QF `cos`/`tan` at
1e30).** `delta` is not 1 ulp of `x`. Reducing `x` mod 2π uses a stored π of
`p` bits, so the reduced argument carries an absolute error of `|x|·2^-p`, which
against a reduced argument of O(1) is **`|x|` ulps**. For DD `cos(1e17)`:
`in_delta = 1e17`, kappa 5.25e16, bound 5.25e33 against a measured 1.6e32 —
**ratio 0.03, i.e. it would pass.** The right term is `in_delta = |x|` for the
circular family; what stopped me implementing it is that it must not also apply
to the ops where the reduction is exact, and separating those cleanly is a
larger change than this commit's scope.

**Trig / inverse-trig reconstruction floor (18 cells: `sin`, `cos`, `asin`,
`acos`, `atan`, `asinh`, `acosh`, `atanh`, `pow` at ratios 8.5–66).** `sincos`
uses the same reduce-by-`2^nq` shape (`tf_math.hpp:917`, nq = 4) but
reconstructs through double-angle recurrences whose error gain is **not** simply
`2^nq`, and the inverse hyperbolics are `log`-composed so they inherit the
log-family floor through a `sqrt` and an `add`. Every one of these 18 sits
between 8.5x and 66x — small, bounded overshoots at 29–30 of 31 digits, the
profile of a missing constant rather than a defect. Charging them `2^nq` on the
strength of the analogy would be exactly the fitted-tolerance move this metric
exists to remove, so they stay red and honest until the gain is derived.

## The new exemption class

`SUBNORMAL_LIMB` joins `UNDERFLOW` and `OVERFLOW` as reported-but-not-gated. It
fires when any operand or the reference lies below the backend's `full_prec`
cliff — a pure format predicate, like the other two, keyed off no op name and no
exclusion list. It **subsumes the low-magnitude end of `ARG_RANGE`**: a value
below the word type's smallest subnormal is the extreme of the same continuum,
not a different fact. `ARG_RANGE` at the *high* end stays gated (3 cells).

It is deliberately the *backstop*, not the mechanism: the floor-widened bound
does most of the work (38 cells) and only 14 cells rest on the label. A bound
that scales with magnitude discriminates; a binary in-band/out-of-band label
does not, and a blunt exemption is how a real defect hides.

## Where the remaining 67 sit

| group | cells | disposition |
|---|---|---|
| open KIs already filed (KI-12/13/14/16/19/20) | 22 | (d), no action |
| newly filed KI-22…KI-26 | 16 | (d), filed with measurement |
| reopened KI-8, KI-10/KI-15 | 4 | (d), reopened |
| trig large-argument `in_delta` | 4 | (c), derived above, deferred |
| trig/inverse-trig reconstruction floor | 18 | (c), derived above, deferred |
| `fma` product exceeds the FP32 word max | 3 | (a) |

## The twelve fixed KIs, re-measured in ulps

Complex ops are ulp-*reported* and digit-*gated* (§"What is NOT converted"), so
for a complex-only KI no ulp verdict can be issued; that is stated rather than
papered over.

| KI | subject | ulp evidence | verdict |
|---|---|---|---|
| KI-1 | complex `acosh`, left half-plane | complex: reported only | HOLDS on digits; **no ulp verdict issuable** |
| KI-2 | `nint`, one-ulp-below-a-tie | no `round` point off a tie fails; the surviving `round` failures are *at* x = −0.5, which is KI-20 (half-to-even), a different class | **HOLDS** |
| KI-3 | `build_with_kokkos.sh` | not a numerical claim | HOLDS (n/a) |
| KI-4 | DD `sin` wrong sign near odd k·π | independently derived: `kappa = \|x cot x\|` at 3π = 9.4248 / 3.673940e−16 = **2.565e16**; measured **1.015e16** → ratio **0.396**. 7π 0.055, 99π 0.116. DD `sin` has no failing point anywhere near k·π | **HOLDS — confirmed**, and the brief's ~4x-inside estimate is right |
| KI-5 | complex inverse family, 4 defects | complex: reported only | HOLDS on digits; **no ulp verdict issuable** |
| KI-6 | `exp` ±300 guard | `exp` has **zero** gated failing cells, all four backends | **HOLDS** |
| KI-7 | `tanh` wrong sign past the guard | `tanh` has **zero** gated failing cells, all four backends | **HOLDS** |
| KI-8 | `hypot`/complex `abs` unscaled | **FF 5.76e3x, QF 7.26e14x, TF 6.88e10x over bound.** QF `hypot(−1e−16)` = 13.71 of 29 digits — 51 bits, two FP32 words, exactly the gap between `x² = 1e−32` and QF's cliff 5.55e−17 | **REOPENED** |
| KI-9 | QF/TF `divide` NaN | `div` has **zero** gated failing cells (KI-19 separately tracks the ~1e41 residual) | **HOLDS** |
| KI-10 | `fmod`/`remainder` at extreme operand ratios | **FF `fmod` 1.41e14 ulps at 0.00 digits, x = 40.84.** As the brief predicted, this is the one that looks worst under ulps | **REOPENED** |
| KI-15 | `fmod`/`remainder` wrong sign / zero / inf | same cell, same commit as KI-10; `remainder` itself is clean | **REOPENED with KI-10** |
| KI-18 | complex `tan`/`tanh`/`atan`/`atanh` asymptotic branch | complex: reported only. Worst complex `tan` ulps DD 6.0e15 / QF 1.33e15 / FF 2.82e14 / TF 7.68e13 | HOLDS on digits; **no ulp verdict issuable** |

Nine hold, three are reopened, and the three complex-only entries cannot be
adjudicated until the complex scorer is converted — which is the honest state,
and is why converting it is the top of step 1d.
