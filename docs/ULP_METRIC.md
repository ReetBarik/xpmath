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
