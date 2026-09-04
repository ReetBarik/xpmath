# Domain limits — where each operation can and cannot be trusted

**GENERATED FILE — do not edit.** Produced by `scripts/gen_domains.py` from
`validation/sweep/sweep_baseline.csv`, `validation/sweep/sweep_grid.csv` and
`validation/sweep/sweep_classified.csv`. Regenerate with:

```bash
scripts/gen_domains.py > docs/DOMAINS.md
```

`validation/check_domains_fresh.sh` fails if this file is not what the
generator currently emits. It is **not** wired into ctest — see the note at
the end of this file for whoever sets up CI in S7.

---

## How to read this

Every number below is measured, on a fixed grid of 1652 real and 1780 complex
inputs per op, scored against a `__float128` (binary128) oracle. A backend's
**cap** is the most digits its format can carry; a cell reports where the op
actually reaches that cap and where it does not.

| backend | words | cap (digits) | full-precision floor | word max | word min (subnormal) |
|---|---|---|---|---|---|
| DD | 2 x FP64 | ~31 | 2.0e-292 | 1.8e+308 | 4.9e-324 |
| QF | 4 x FP32 | ~29 | 5.6e-17 | 3.4e+38 | 1.4e-45 |
| TF | 3 x FP32 | ~21.7 | 3.3e-24 | 3.4e+38 | 1.4e-45 |
| FF | 2 x FP32 | ~14 | 2.0e-31 | 3.4e+38 | 1.4e-45 |

**Full-precision floor** is the magnitude below which the *trailing* limbs go
subnormal, so the type stops carrying its nominal digit count even though the
leading word is still fine. It is the number that matters for small inputs:
FF is out of digits below ~2e-31 while DD keeps all 31 down to ~2e-292.

Column meanings:

- **trusted |x|** — the widest *contiguous* band of input magnitude in which
  **every** grid point reaches at least 90% of cap. For complex ops the
  magnitude is |z|. `none` means no such band exists. The band is contiguous
  by construction, so it has no holes: a caller can rely on all of it.
- **at cap** — the share of *all* the op's grid points reaching 90% of cap,
  including any outside the band. A wide band with a low percentage means the
  op also works in places the single interval does not cover.
- **boundary** — the failing point nearest each end of the trusted band, with
  the digits measured there, so the degradation is quantified and not merely
  located.
- **fails** — how many of the op's points score below 50% of cap, and the
  dominant classification of those failures.

**Caveat for the binary ops** (`add sub mul div pow hypot fmod remainder`
`copysign fmax fmin fdim fma`, and the complex `add sub mul div pow`): the
magnitude axis is the **first operand only**. The second is drawn
log-uniformly over a per-op window, with one point in seven a deliberately
cancelling pair, so a failure at a given |x| may be caused by the operand
paired with it rather than by x. For those rows read the `fails` count and the
classification, and treat the band as indicative. The unary rows have no such
ambiguity.

Classifications, from `scripts/sweep_accuracy --classify`:

- **UNDERFLOW** — true result below the smallest magnitude the words hold
- **OVERFLOW** — true result above the largest magnitude the words hold
- **ARG_RANGE** — the input itself is outside what the words hold
- **CONDITIONING** — measured ill-conditioning: no implementation at this precision does better
- **UNEXPLAINED** — NOT explained by range or conditioning -- see docs/KNOWN_ISSUES.md

`UNDERFLOW`, `OVERFLOW` and `ARG_RANGE` are *format* limits: they are properties
of 2xFP32 or 4xFP32 and no implementation can remove them. `CONDITIONING` is a
*mathematical* limit, measured by perturbing each operand by the error the
backend actually carries and re-evaluating the oracle. **`UNEXPLAINED` is
neither** — those are the points where the answer was representable and the
problem well-conditioned, and the implementation still lost the digits. They
are filed as KI-6 through KI-11 in `docs/KNOWN_ISSUES.md`.

### Why grouped by operation family rather than by backend

The four backends share one op inventory and one grid, so their cells are
directly comparable — and the question a caller actually has is *"can this op
hold my range, and if not which backend can?"*. Putting the four backends on
adjacent rows answers that by eye. A backend-major layout would answer the
rarer question ("what does DD do across all 63 ops?") while forcing a reader
comparing FF against DD to page between two distant sections.

---

## Arithmetic and selection (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `add` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `add` | QF | 29.00 | 28.99 | 3e-30 .. 1e+30 | 100% | below 1e-30: 9.56 | 1 | UNDERFLOW |
| `add` | TF | 21.70 | 21.69 | 3e-30 .. 1e+30 | 100% | below 1e-30: 9.56 | 1 | UNDERFLOW |
| `add` | FF | 14.00 | 13.19 | 0.9999 .. 1 | 90% | below 0.999: 6.22; above 1.1: 6.36 | 100 | CONDITIONING |
| `sub` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `sub` | QF | 29.00 | 28.98 | 3e-30 .. 1e+30 | 100% | below 1e-30: 1.91 | 1 | UNDERFLOW |
| `sub` | TF | 21.70 | 21.69 | 3e-30 .. 1e+30 | 100% | below 1e-30: 1.91 | 1 | UNDERFLOW |
| `sub` | FF | 14.00 | 13.21 | 1 .. 1.05 | 89% | below 0.999: 3.01; above 1.571: 5.61 | 87 | CONDITIONING |
| `mul` | DD | 31.00 | 30.94 | 1e-30 .. 1e+30 | 100% | below 1e-323: 4.89 | 4 | ARG_RANGE |
| `mul` | QF | 29.00 | 28.36 | 1e-10 .. 3e+20 | 97% | below 1e-16: 13.67; above 1e+21: 0.00 | 35 | UNDERFLOW |
| `mul` | TF | 21.70 | 21.30 | 3e-14 .. 3e+20 | 97% | below 3e-18: 10.59; above 1e+21: 0.00 | 30 | UNDERFLOW |
| `mul` | FF | 14.00 | 13.79 | 1e-17 .. 3e+20 | 98% | below 3e-20: 6.70; above 1e+21: 0.00 | 25 | UNDERFLOW |
| `div` | DD | 31.00 | 30.95 | 1e-30 .. 1e+30 | 100% | below 1e-323: 14.55 | 4 | ARG_RANGE |
| `div` | QF | 29.00 | 28.50 | 1e-06 .. 1e+24 | 96% | below 1e-21: 11.81; above 3e+24: 0.00 | 17 | UNDERFLOW |
| `div` | TF | 21.70 | 21.47 | 1e-12 .. 1e+24 | 98% | below 1e-22: 9.81; above 3e+24: 0.00 | 12 | UNDERFLOW |
| `div` | FF | 14.00 | 13.92 | 3e-21 .. 1e+24 | 99% | below 1e-323: 0.00; above 3e+24: 0.00 | 6 | ARG_RANGE |
| `fma` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fma` | QF | 29.00 | 28.63 | 3e-09 .. 3e+20 | 98% | below 3e-13: 6.94; above 1e+21: 0.00 | 19 | ARG_RANGE |
| `fma` | TF | 21.70 | 21.46 | 1e-12 .. 3e+20 | 99% | below 3e-13: 6.94; above 1e+21: 0.00 | 19 | ARG_RANGE |
| `fma` | FF | 14.00 | 12.99 | 1 .. 1 | 89% | below 1: 0.00; above 1: 6.83 | 126 | CONDITIONING |
| `abs` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `abs` | QF | 29.00 | 28.93 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `abs` | TF | 21.70 | 21.65 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `abs` | FF | 14.00 | 13.97 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `copysign` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `copysign` | QF | 29.00 | 28.93 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `copysign` | TF | 21.70 | 21.65 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `copysign` | FF | 14.00 | 13.97 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `fmax` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fmax` | QF | 29.00 | 28.92 | 191.6 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `fmax` | TF | 21.70 | 21.65 | 191.6 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `fmax` | FF | 14.00 | 13.98 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `fmin` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fmin` | QF | 29.00 | 28.93 | 6.15 .. 216.8 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `fmin` | TF | 21.70 | 21.66 | 6.15 .. 216.8 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `fmin` | FF | 14.00 | 13.98 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `fdim` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fdim` | QF | 29.00 | 29.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fdim` | TF | 21.70 | 21.70 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fdim` | FF | 14.00 | 13.69 | 0.9999 .. 2.65 | 96% | below 3e-18: 0.00; above 3.6: 0.76 | 35 | CONDITIONING |
| `hypot` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `hypot` | QF | 29.00 | 28.95 | 1e-19 .. 1e+30 | 99% | -- | 0 | -- |
| `hypot` | TF | 21.70 | 21.69 | 1e-26 .. 1e+30 | 100% | -- | 0 | -- |
| `hypot` | FF | 14.00 | 14.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `add` — QF 1 pts (log sweep, |x| = 10^e); TF 1 pts (log sweep, |x| = 10^e); FF 100 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `sub` — QF 1 pts (log sweep, |x| = 10^e); TF 1 pts (log sweep, |x| = 10^e); FF 87 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `mul` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 35 pts (log sweep, |x| = 10^e); TF 30 pts (log sweep, |x| = 10^e); FF 25 pts (log sweep, |x| = 10^e)
- `div` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 17 pts (log sweep, |x| = 10^e); TF 12 pts (log sweep, |x| = 10^e); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fma` — QF 19 pts (log sweep, |x| = 10^e); TF 19 pts (log sweep, |x| = 10^e); FF 126 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `abs` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `copysign` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fmax` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fmin` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fdim` — FF 35 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Rounding and remainder (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `ceil` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `ceil` | QF | 29.00 | 28.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `ceil` | TF | 21.70 | 21.67 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `ceil` | FF | 14.00 | 13.98 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `floor` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `floor` | QF | 29.00 | 28.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `floor` | TF | 21.70 | 21.67 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `floor` | FF | 14.00 | 13.98 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `round` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `round` | QF | 29.00 | 29.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `round` | TF | 21.70 | 21.70 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `round` | FF | 14.00 | 14.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | QF | 29.00 | 29.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | TF | 21.70 | 21.70 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | FF | 14.00 | 14.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fmod` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `fmod` | QF | 29.00 | 28.80 | 40.84 .. 219.9 | 99% | below 40.84: 0.36; above 219.9: 0.15 | 12 | UNDERFLOW |
| `fmod` | TF | 21.70 | 21.55 | 40.84 .. 219.9 | 99% | below 40.84: 0.36; above 219.9: 0.15 | 12 | CONDITIONING |
| `fmod` | FF | 14.00 | 9.42 | 3e-30 .. 1e-25 | 58% | below 1e-323: 0.00; above 3e-16: 0.00 | 550 | CONDITIONING |
| `remainder` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `remainder` | QF | 29.00 | 28.75 | 6.283 .. 78.54 | 99% | below 6.283: 0.00; above 78.54: 0.76 | 15 | CONDITIONING |
| `remainder` | TF | 21.70 | 21.51 | 6.283 .. 78.54 | 99% | below 6.283: 0.00; above 78.54: 0.76 | 14 | CONDITIONING |
| `remainder` | FF | 14.00 | 8.92 | 3e-30 .. 1e-27 | 52% | below 1e-30: 2.81; above 3e-25: 1.69 | 596 | CONDITIONING |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `ceil` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `floor` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fmod` — QF 12 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 12 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 550 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `remainder` — QF 15 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 14 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 596 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Exponential, logarithmic and power (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `exp` | DD | 31.00 | 30.62 | 0 .. 316.2 | 100% | above 1000: 0.00 | 6 | OVERFLOW |
| `exp` | QF | 29.00 | 15.67 | 0 .. 43.98 | 52% | above 72.26: 14.21 | 759 | UNDERFLOW |
| `exp` | TF | 21.70 | 11.87 | 0 .. 59.69 | 53% | above 81.68: 10.03 | 745 | UNDERFLOW |
| `exp` | FF | 14.00 | 7.73 | 0 .. 62.83 | 54% | above 91.11: 0.00 | 730 | OVERFLOW |
| `exp2` | DD | 31.00 | 30.62 | 0 .. 1000 | 100% | above 3162: 0.00 | 4 | OVERFLOW |
| `exp2` | QF | 29.00 | 17.61 | 0 .. 62.83 | 57% | above 103.7: 13.94 | 647 | UNDERFLOW |
| `exp2` | TF | 21.70 | 13.40 | 0 .. 84.82 | 59% | above 116.2: 10.55 | 628 | UNDERFLOW |
| `exp2` | FF | 14.00 | 8.75 | 0 .. 94.25 | 61% | above 128.8: 0.00 | 608 | OVERFLOW |
| `exp10` | DD | 31.00 | 29.76 | 0 .. 295.3 | 97% | above 311: 0.00 | 26 | OVERFLOW |
| `exp10` | QF | 29.00 | 13.13 | 0 .. 18.85 | 44% | above 31.42: 13.96 | 903 | UNDERFLOW |
| `exp10` | TF | 21.70 | 9.89 | 0 .. 25.13 | 45% | above 34.56: 10.76 | 898 | UNDERFLOW |
| `exp10` | FF | 14.00 | 6.40 | 0 .. 31.62 | 46% | above 40.84: 0.00 | 888 | OVERFLOW |
| `expm1` | DD | 31.00 | 29.83 | 0 .. 316.2 | 97% | above 1000: 0.00 | 55 | OVERFLOW |
| `expm1` | QF | 29.00 | 22.42 | 3e-20 .. 87.96 | 76% | below 1e-323: 0.00; above 91.11: 0.00 | 369 | OVERFLOW |
| `expm1` | TF | 21.70 | 16.83 | 1e-30 .. 87.96 | 78% | below 1e-323: 0.00; above 91.11: 0.00 | 369 | OVERFLOW |
| `expm1` | FF | 14.00 | 10.39 | 1e-30 .. 62.83 | 74% | below 1e-323: 0.00; above 91.11: 0.00 | 421 | OVERFLOW |
| `log` | DD | 31.00 | 30.93 | 1e-30 .. 1e+30 | 100% | below 0: 0.00 | 2 | OVERFLOW |
| `log` | QF | 29.00 | 28.81 | 1 .. 1e+30 | 98% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log` | TF | 21.70 | 21.57 | 1.001 .. 1e+30 | 99% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log` | FF | 14.00 | 13.86 | 1.01 .. 1e+30 | 97% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log2` | DD | 31.00 | 30.93 | 1e-30 .. 1e+30 | 100% | below 0: 0.00 | 2 | OVERFLOW |
| `log2` | QF | 29.00 | 28.81 | 1 .. 1e+30 | 98% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log2` | TF | 21.70 | 21.57 | 1.001 .. 1e+30 | 99% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log2` | FF | 14.00 | 13.86 | 1.01 .. 1e+30 | 97% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log10` | DD | 31.00 | 30.93 | 1e-30 .. 1e+30 | 100% | below 0: 0.00 | 2 | OVERFLOW |
| `log10` | QF | 29.00 | 28.80 | 1 .. 1e+30 | 98% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log10` | TF | 21.70 | 21.57 | 1.001 .. 1e+30 | 99% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log10` | FF | 14.00 | 13.86 | 1.01 .. 1e+30 | 97% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log1p` | DD | 31.00 | 30.89 | 1 .. 1e+30 | 100% | below 1: 0.00 | 6 | OVERFLOW |
| `log1p` | QF | 29.00 | 28.67 | 1 .. 3e+18 | 97% | below 1: 0.00 | 8 | OVERFLOW |
| `log1p` | TF | 21.70 | 21.57 | 1 .. 1e+29 | 99% | below 1: 0.00 | 8 | OVERFLOW |
| `log1p` | FF | 14.00 | 13.91 | 1.001 .. 1e+30 | 99% | below 1: 0.00 | 8 | OVERFLOW |
| `pow` | DD | 31.00 | 30.79 | 1e-30 .. 1e+30 | 100% | below 5e-324: 15.19 | 2 | ARG_RANGE |
| `pow` | QF | 29.00 | 26.86 | 0.3162 .. 3.45 | 86% | below 3e-05: 14.48; above 3.65: 13.96 | 147 | UNDERFLOW |
| `pow` | TF | 21.70 | 20.62 | 0.0001 .. 3.6 | 87% | below 1e-323: 0.00; above 28.27: 9.78 | 7 | ARG_RANGE |
| `pow` | FF | 14.00 | 13.73 | 1e-30 .. 4.55 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `sqrt` | DD | 31.00 | 30.98 | 1e-30 .. 1e+30 | 100% | -- | 0 | -- |
| `sqrt` | QF | 29.00 | 28.70 | 3e-20 .. 1e+30 | 97% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `sqrt` | TF | 21.70 | 21.59 | 3e-26 .. 1e+30 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `sqrt` | FF | 14.00 | 13.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `exp` — DD 6 pts (log sweep, |x| = 10^e); QF 759 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 745 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `exp2` — DD 4 pts (log sweep, |x| = 10^e); QF 647 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 628 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 608 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `exp10` — DD 26 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 903 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 898 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 888 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `expm1` — DD 55 pts (log sweep, |x| = 10^e); QF 369 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 369 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 421 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log` — DD 2 pts (linear sweep over [-8, 8]); QF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log2` — DD 2 pts (linear sweep over [-8, 8]); QF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log10` — DD 2 pts (linear sweep over [-8, 8]); QF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log1p` — DD 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `pow` — DD 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 147 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 7 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `sqrt` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Trigonometric and inverse trigonometric (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `sin` | DD | 31.00 | 21.38 | 0 .. 3.1 | 33% | above 3.142: 15.42 | 72 | CONDITIONING |
| `sin` | QF | 29.00 | 19.00 | 3e-28 .. 3.1 | 32% | below 3e-30: 14.31; above 3.142: 13.46 | 520 | CONDITIONING (**4 unexplained**) |
| `sin` | TF | 21.70 | 11.84 | 3e-28 .. 3.1 | 32% | below 1e-323: 0.00; above 3.142: 7.15 | 1078 | CONDITIONING |
| `sin` | FF | 14.00 | 4.66 | 1e-30 .. 3.05 | 32% | below 1e-323: 0.00; above 3.142: 0.00 | 1092 | CONDITIONING |
| `cos` | DD | 31.00 | 29.75 | 1.6 .. 1e+05 | 93% | below 1.571: 15.24; above 1e+17: 0.00 | 54 | CONDITIONING |
| `cos` | QF | 29.00 | 27.66 | 1.6 .. 1000 | 93% | below 1.571: 13.39; above 1e+15: 14.35 | 72 | CONDITIONING |
| `cos` | TF | 21.70 | 20.62 | 1.6 .. 1000 | 93% | below 1.571: 7.35; above 3e+12: 10.64 | 82 | CONDITIONING |
| `cos` | FF | 14.00 | 13.00 | 7.9 .. 316.2 | 92% | below 1.571: 0.00; above 3e+07: 6.72 | 98 | CONDITIONING |
| `tan` | DD | 31.00 | 21.35 | 0 .. 1.55 | 33% | above 1.571: 15.24 | 76 | CONDITIONING |
| `tan` | QF | 29.00 | 18.89 | 3e-28 .. 1.55 | 32% | below 3e-30: 14.31; above 1.571: 13.90 | 532 | CONDITIONING (**4 unexplained**) |
| `tan` | TF | 21.70 | 11.75 | 3e-28 .. 1.55 | 31% | below 1e-323: 0.00; above 1.571: 6.84 | 1090 | CONDITIONING |
| `tan` | FF | 14.00 | 4.55 | 1e-30 .. 1.55 | 31% | below 1e-323: 0.00; above 1.571: 0.00 | 1104 | CONDITIONING (**4 unexplained**) |
| `asin` | DD | 31.00 | 30.90 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `asin` | QF | 29.00 | 28.55 | 3e-28 .. 1e+28 | 99% | below 3e-30: 14.35; above 3e+29: 14.48 | 12 | UNEXPLAINED (**6 unexplained**) |
| `asin` | TF | 21.70 | 21.50 | 3e-28 .. 1e+28 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asin` | FF | 14.00 | 13.83 | 1 .. 1e+30 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `acos` | DD | 31.00 | 30.99 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `acos` | QF | 29.00 | 28.94 | 1.001 .. 1e+30 | 100% | -- | 0 | -- |
| `acos` | TF | 21.70 | 21.66 | 1 .. 1e+30 | 99% | -- | 0 | -- |
| `acos` | FF | 14.00 | 13.95 | 1.01 .. 1e+30 | 99% | -- | 0 | -- |
| `atan` | DD | 31.00 | 30.98 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `atan` | QF | 29.00 | 28.79 | 3e-28 .. 1e+30 | 99% | below 3e-30: 14.35 | 8 | UNEXPLAINED (**4 unexplained**) |
| `atan` | TF | 21.70 | 21.59 | 3e-28 .. 1e+30 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `atan` | FF | 14.00 | 13.94 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sin` — DD 72 pts (log sweep, |x| = 10^e); QF 520 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 1078 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 1092 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `cos` — DD 54 pts (log sweep, |x| = 10^e); QF 72 pts (log sweep, |x| = 10^e); TF 82 pts (log sweep, |x| = 10^e); FF 98 pts (log sweep, |x| = 10^e)
- `tan` — DD 76 pts (log sweep, |x| = 10^e); QF 532 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 1090 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 1104 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `asin` — QF 12 pts (log sweep, |x| = 10^e); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `atan` — QF 8 pts (log sweep, |x| = 10^e); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Hyperbolic and inverse hyperbolic (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `sinh` | DD | 31.00 | 30.62 | 0 .. 316.2 | 100% | above 1000: 0.00 | 6 | OVERFLOW |
| `sinh` | QF | 29.00 | 16.09 | 1e-30 .. 87.96 | 56% | below 1e-323: 0.00; above 91.11: 0.00 | 734 | OVERFLOW |
| `sinh` | TF | 21.70 | 12.00 | 1e-30 .. 87.96 | 56% | below 1e-323: 0.00; above 91.11: 0.00 | 734 | OVERFLOW |
| `sinh` | FF | 14.00 | 7.69 | 1e-30 .. 62.83 | 55% | below 1e-323: 0.00; above 91.11: 0.00 | 734 | OVERFLOW |
| `cosh` | DD | 31.00 | 30.62 | 0 .. 316.2 | 100% | above 1000: 0.00 | 6 | OVERFLOW |
| `cosh` | QF | 29.00 | 16.16 | 0 .. 87.96 | 56% | above 91.11: 0.00 | 730 | OVERFLOW |
| `cosh` | TF | 21.70 | 12.05 | 0 .. 87.96 | 56% | above 91.11: 0.00 | 730 | OVERFLOW |
| `cosh` | FF | 14.00 | 7.73 | 0 .. 62.83 | 55% | above 91.11: 0.00 | 730 | OVERFLOW |
| `tanh` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `tanh` | QF | 29.00 | 28.93 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `tanh` | TF | 21.70 | 21.64 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `tanh` | FF | 14.00 | 13.97 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asinh` | DD | 31.00 | 30.96 | 1e-323 .. 1e+30 | 100% | below 5e-324: 0.00 | 2 | ARG_RANGE |
| `asinh` | QF | 29.00 | 28.89 | 3e-29 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asinh` | TF | 21.70 | 21.63 | 3e-29 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asinh` | FF | 14.00 | 13.97 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `acosh` | DD | 31.00 | 30.89 | 1 .. 1e+30 | 99% | -- | 0 | -- |
| `acosh` | QF | 29.00 | 28.82 | 1 .. 1e+30 | 97% | -- | 0 | -- |
| `acosh` | TF | 21.70 | 21.53 | 1 .. 1e+30 | 97% | -- | 0 | -- |
| `acosh` | FF | 14.00 | 13.83 | 1.001 .. 1e+30 | 96% | -- | 0 | -- |
| `atanh` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `atanh` | QF | 29.00 | 28.90 | 1e-30 .. 1e+29 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `atanh` | TF | 21.70 | 21.63 | 1e-30 .. 1e+29 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `atanh` | FF | 14.00 | 13.92 | 1.001 .. 1e+30 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sinh` — DD 6 pts (log sweep, |x| = 10^e); QF 734 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 734 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 734 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `cosh` — DD 6 pts (log sweep, |x| = 10^e); QF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `tanh` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `asinh` — DD 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `atanh` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Complex arithmetic and construction (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `add` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `add` | QF | 29.00 | 28.92 | 1e-29 .. 1 | 100% | below 1e-30: 13.17; above 1: 4.32 | 6 | UNDERFLOW |
| `add` | TF | 21.70 | 21.65 | 1e-29 .. 1 | 100% | above 1: 4.32 | 4 | UNDERFLOW |
| `add` | FF | 14.00 | 13.27 | 1 .. 1 | 90% | below 0.99: 5.16; above 1: 6.33 | 84 | CONDITIONING |
| `sub` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `sub` | QF | 29.00 | 28.91 | 1e-29 .. 1 | 100% | below 1e-30: 5.52; above 1: 8.53 | 6 | UNDERFLOW |
| `sub` | TF | 21.70 | 21.64 | 1e-29 .. 1 | 100% | below 1e-30: 5.52; above 1: 8.53 | 6 | UNDERFLOW |
| `sub` | FF | 14.00 | 13.23 | 0.99 .. 1 | 91% | below 0.99: 4.48; above 1: 0.00 | 96 | CONDITIONING |
| `mul` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `mul` | QF | 29.00 | 27.02 | 100 .. 1e+08 | 84% | below 10: 13.55 | 76 | UNDERFLOW |
| `mul` | TF | 21.70 | 20.83 | 100 .. 1e+08 | 92% | below 10: 10.62; above 1e+08: 7.52 | 63 | UNDERFLOW (**6 unexplained**) |
| `mul` | FF | 14.00 | 13.63 | 10 .. 1e+08 | 96% | below 10: 0.36; above 1e+08: 0.00 | 45 | UNDERFLOW |
| `div` | DD | 31.00 | 29.59 | 0 .. 1e-09 | 89% | above 0.99: 13.14 | 7 | UNEXPLAINED (**5 unexplained**) |
| `div` | QF | 29.00 | 25.98 | 1e-13 .. 1e-10 | 76% | below 1e-15: 10.44; above 1e-08: 11.73 | 123 | UNDERFLOW (**13 unexplained**) |
| `div` | TF | 21.70 | 20.23 | 1e-14 .. 1e-09 | 84% | below 1e-15: 10.44; above 1e-08: 0.00 | 83 | UNDERFLOW (**9 unexplained**) |
| `div` | FF | 14.00 | 12.49 | 0.99 .. 1 | 88% | below 0.99: 0.00; above 1: 0.00 | 189 | CONDITIONING |
| `abs` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `abs` | QF | 29.00 | 29.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `abs` | TF | 21.70 | 21.70 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `abs` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | QF | 29.00 | 29.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | TF | 21.70 | 21.70 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `polar` | DD | 31.00 | 30.74 | 0 .. 1e+04 | 98% | -- | 0 | -- |
| `polar` | QF | 29.00 | 27.44 | 1 .. 2 | 86% | below 1: 13.77; above 2: 13.77 | 21 | UNEXPLAINED (**20 unexplained**) |
| `polar` | TF | 21.70 | 21.08 | 1 .. 2 | 90% | -- | 0 | -- |
| `polar` | FF | 14.00 | 13.75 | 0 .. 1000 | 97% | above 1e+08: 6.53 | 12 | CONDITIONING |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `add` — QF 6 pts (perpendicular approach to the real axis); TF 4 pts (perpendicular approach to the real axis); FF 84 pts (perpendicular approach to the real axis)
- `sub` — QF 6 pts (perpendicular approach to the real axis); TF 6 pts (perpendicular approach to the real axis); FF 96 pts (perpendicular approach to the real axis)
- `mul` — QF 76 pts (perpendicular approach to the real axis); TF 63 pts (perpendicular approach to the real axis); FF 45 pts (perpendicular approach to the real axis)
- `div` — DD 7 pts (polar shells); QF 123 pts (perpendicular approach to the real axis); TF 83 pts (perpendicular approach to the real axis); FF 189 pts (perpendicular approach to the real axis)
- `polar` — QF 21 pts (perpendicular approach to the real axis); FF 12 pts (polar shells)

---

## Complex exponential, logarithmic, power and root (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `exp` | DD | 31.00 | 29.81 | 0 .. 100 | 96% | above 1000: 0.00 | 61 | OVERFLOW |
| `exp` | QF | 29.00 | 24.60 | 1 .. 2 | 79% | below 1: 13.77; above 2: 13.82 | 219 | OVERFLOW (**14 unexplained**) |
| `exp` | TF | 21.70 | 19.00 | 1e-27 .. 0.1 | 84% | above 100: 1.77 | 193 | OVERFLOW |
| `exp` | FF | 14.00 | 12.40 | 0 .. 2.236 | 89% | above 100: 1.77 | 193 | OVERFLOW |
| `log` | DD | 31.00 | 27.41 | 2 .. 2.236 | 79% | below 2: 0.00; above 10: 0.00 | 84 | CONDITIONING |
| `log` | QF | 29.00 | 24.55 | 2 .. 2.236 | 75% | below 2: 0.00; above 10: 0.00 | 249 | UNDERFLOW (**8 unexplained**) |
| `log` | TF | 21.70 | 18.49 | 2 .. 2.236 | 76% | below 2: 0.00; above 10: 0.00 | 220 | UNDERFLOW |
| `log` | FF | 14.00 | 11.74 | 2 .. 2.236 | 75% | below 2: 0.00; above 10: 0.00 | 208 | CONDITIONING |
| `log10` | DD | 31.00 | 27.41 | 2 .. 2.236 | 79% | below 2: 0.00; above 10: 0.00 | 84 | CONDITIONING |
| `log10` | QF | 29.00 | 24.34 | 2 .. 2.236 | 73% | below 2: 0.00; above 10: 0.00 | 249 | UNDERFLOW (**2 unexplained**) |
| `log10` | TF | 21.70 | 18.46 | 2 .. 2.236 | 75% | below 2: 0.00; above 10: 0.00 | 220 | UNDERFLOW |
| `log10` | FF | 14.00 | 11.74 | 2 .. 2.236 | 75% | below 2: 0.00; above 10: 0.00 | 208 | CONDITIONING |
| `pow` | DD | 31.00 | 28.95 | 2 .. 2.236 | 94% | below 2: 0.00; above 10: 0.00 | 83 | CONDITIONING |
| `pow` | QF | 29.00 | 26.09 | 2 .. 2.002 | 87% | below 2: 14.11; above 2.236: 8.92 | 124 | CONDITIONING |
| `pow` | TF | 21.70 | 19.80 | 2 .. 2.002 | 89% | below 2: 0.00; above 2.236: 8.92 | 113 | CONDITIONING |
| `pow` | FF | 14.00 | 12.49 | 0.5 .. 0.5 | 75% | below 0.5: 0.08; above 0.9: 0.00 | 90 | CONDITIONING |
| `sqrt` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `sqrt` | QF | 29.00 | 27.74 | 1 .. 2 | 86% | -- | 0 | -- |
| `sqrt` | TF | 21.70 | 21.39 | 1e-25 .. 0.1 | 94% | -- | 0 | -- |
| `sqrt` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `exp` — DD 61 pts (polar shells); QF 219 pts (perpendicular approach to the real axis); TF 193 pts (perpendicular approach to the real axis); FF 193 pts (perpendicular approach to the real axis)
- `log` — DD 84 pts (the real axis on a geometric ladder); QF 249 pts (perpendicular approach to the real axis); TF 220 pts (the real axis on a geometric ladder); FF 208 pts (the real axis on a geometric ladder)
- `log10` — DD 84 pts (the real axis on a geometric ladder); QF 249 pts (perpendicular approach to the real axis); TF 220 pts (the real axis on a geometric ladder); FF 208 pts (the real axis on a geometric ladder)
- `pow` — DD 83 pts (the real axis on a geometric ladder); QF 124 pts (the real axis on a geometric ladder); TF 113 pts (the real axis on a geometric ladder); FF 90 pts (the real axis on a geometric ladder)

---

## Complex trigonometric and hyperbolic (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `sin` | DD | 31.00 | 30.08 | 0 .. 1000 | 96% | above 1e+04: 0.00 | 30 | OVERFLOW |
| `sin` | QF | 29.00 | 26.13 | 1 .. 2 | 75% | below 1: 13.80; above 2: 13.77 | 46 | OVERFLOW (**16 unexplained**) |
| `sin` | TF | 21.70 | 20.49 | 1e-27 .. 0.1 | 87% | above 1e+04: 0.00 | 42 | OVERFLOW |
| `sin` | FF | 14.00 | 13.31 | 0 .. 100 | 95% | above 1e+04: 0.00 | 58 | OVERFLOW |
| `cos` | DD | 31.00 | 30.09 | 0 .. 1000 | 96% | above 1e+04: 0.00 | 30 | OVERFLOW |
| `cos` | QF | 29.00 | 26.21 | 1 .. 2 | 76% | below 1: 13.77; above 2: 13.77 | 47 | OVERFLOW (**12 unexplained**) |
| `cos` | TF | 21.70 | 20.54 | 1 .. 2 | 87% | above 1e+04: 0.00 | 42 | OVERFLOW |
| `cos` | FF | 14.00 | 13.32 | 0 .. 100 | 95% | above 1e+04: 0.00 | 58 | OVERFLOW |
| `tan` | DD | 31.00 | 30.51 | 0 .. 1e+04 | 97% | above 1e+04: 0.00 | 12 | UNDERFLOW |
| `tan` | QF | 29.00 | 26.01 | 1 .. 2 | 72% | below 1: 13.84; above 10: 13.31 | 56 | UNDERFLOW (**4 unexplained**) |
| `tan` | TF | 21.70 | 20.49 | 1e-27 .. 0.1 | 85% | above 10: 10.52 | 48 | UNDERFLOW |
| `tan` | FF | 14.00 | 13.43 | 0 .. 2.236 | 95% | above 1e+04: 0.00 | 40 | OVERFLOW |
| `sinh` | DD | 31.00 | 29.43 | 0 .. 100 | 95% | above 1000: 0.00 | 83 | OVERFLOW |
| `sinh` | QF | 29.00 | 23.70 | 1 .. 2 | 70% | below 1: 13.80; above 2: 13.77 | 233 | OVERFLOW (**18 unexplained**) |
| `sinh` | TF | 21.70 | 18.60 | 1e-27 .. 0.1 | 80% | above 100: 0.00 | 215 | OVERFLOW |
| `sinh` | FF | 14.00 | 12.22 | 0 .. 10.05 | 88% | above 100: 0.00 | 215 | OVERFLOW |
| `cosh` | DD | 31.00 | 29.43 | 0 .. 100 | 95% | above 1000: 0.00 | 83 | OVERFLOW |
| `cosh` | QF | 29.00 | 23.73 | 1 .. 2 | 70% | below 1: 13.77; above 2: 13.77 | 232 | OVERFLOW (**16 unexplained**) |
| `cosh` | TF | 21.70 | 18.62 | 1 .. 2 | 80% | above 100: 0.00 | 215 | OVERFLOW |
| `cosh` | FF | 14.00 | 12.22 | 0 .. 10.05 | 88% | above 100: 0.00 | 215 | OVERFLOW |
| `tanh` | DD | 31.00 | 30.55 | 0 .. 1000 | 99% | above 1e+04: 0.00 | 12 | UNDERFLOW |
| `tanh` | QF | 29.00 | 24.47 | 1 .. 2 | 72% | below 1: 13.84; above 10: 13.31 | 178 | UNDERFLOW (**6 unexplained**) |
| `tanh` | TF | 21.70 | 19.27 | 1e-27 .. 0.1 | 82% | above 10: 10.52 | 156 | UNDERFLOW |
| `tanh` | FF | 14.00 | 12.71 | 0 .. 2.236 | 91% | above 100: 0.00 | 136 | UNDERFLOW |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sin` — DD 30 pts (polar shells); QF 46 pts (polar shells); TF 42 pts (polar shells); FF 58 pts (polar shells)
- `cos` — DD 30 pts (polar shells); QF 47 pts (polar shells); TF 42 pts (polar shells); FF 58 pts (polar shells)
- `tan` — DD 12 pts (polar shells); QF 56 pts (perpendicular approach to the imaginary axis); TF 48 pts (perpendicular approach to the imaginary axis); FF 40 pts (the real axis on a geometric ladder)
- `sinh` — DD 83 pts (the real axis on a geometric ladder); QF 233 pts (perpendicular approach to the real axis); TF 215 pts (perpendicular approach to the real axis); FF 215 pts (perpendicular approach to the real axis)
- `cosh` — DD 83 pts (the real axis on a geometric ladder); QF 232 pts (perpendicular approach to the real axis); TF 215 pts (perpendicular approach to the real axis); FF 215 pts (perpendicular approach to the real axis)
- `tanh` — DD 12 pts (polar shells); QF 178 pts (perpendicular approach to the real axis); TF 156 pts (perpendicular approach to the real axis); FF 136 pts (perpendicular approach to the real axis)

---

## Complex inverse functions (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `asin` | DD | 31.00 | 30.97 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `asin` | QF | 29.00 | 27.64 | 10 .. 1e+15 | 87% | below 10: 13.77 | 20 | UNEXPLAINED (**16 unexplained**) |
| `asin` | TF | 21.70 | 21.30 | 10 .. 1e+15 | 94% | -- | 0 | -- |
| `asin` | FF | 14.00 | 13.90 | 1.005 .. 1e+15 | 98% | -- | 0 | -- |
| `acos` | DD | 31.00 | 30.81 | 0 .. 1e+04 | 99% | -- | 0 | -- |
| `acos` | QF | 29.00 | 27.86 | 1 .. 2 | 90% | below 1: 14.49; above 2: 14.12 | 14 | UNEXPLAINED (**6 unexplained**) |
| `acos` | TF | 21.70 | 21.28 | 1e-28 .. 0.1 | 94% | -- | 0 | -- |
| `acos` | FF | 14.00 | 13.73 | 1.005 .. 10.05 | 96% | above 1e+08: 6.93 | 5 | UNEXPLAINED (**5 unexplained**) |
| `atan` | DD | 31.00 | 30.99 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `atan` | QF | 29.00 | 27.80 | 1 .. 2 | 88% | below 1e-30: 13.95; above 10: 13.51 | 20 | UNDERFLOW (**4 unexplained**) |
| `atan` | TF | 21.70 | 21.35 | 1e-28 .. 0.1 | 95% | -- | 0 | -- |
| `atan` | FF | 14.00 | 13.96 | 0 .. 10.05 | 100% | -- | 0 | -- |
| `asinh` | DD | 31.00 | 30.96 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `asinh` | QF | 29.00 | 27.32 | 1 .. 2 | 84% | below 1: 14.10; above 2: 14.46 | 36 | UNEXPLAINED (**24 unexplained**) |
| `asinh` | TF | 21.70 | 21.21 | 1e-27 .. 0.1 | 92% | -- | 0 | -- |
| `asinh` | FF | 14.00 | 13.94 | 0 .. 10.05 | 100% | -- | 0 | -- |
| `acosh` | DD | 31.00 | 30.89 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `acosh` | QF | 29.00 | 27.94 | 1 .. 2 | 91% | above 2: 14.12 | 12 | UNEXPLAINED (**6 unexplained**) |
| `acosh` | TF | 21.70 | 21.40 | 1e-28 .. 0.1 | 96% | -- | 0 | -- |
| `acosh` | FF | 14.00 | 13.82 | 1.005 .. 10.05 | 98% | -- | 0 | -- |
| `atanh` | DD | 31.00 | 30.99 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `atanh` | QF | 29.00 | 27.86 | 10 .. 1e+04 | 90% | below 10: 13.51 | 26 | CONDITIONING (**6 unexplained**) |
| `atanh` | TF | 21.70 | 21.28 | 10 .. 1e+15 | 96% | below 1: 0.00 | 16 | CONDITIONING |
| `atanh` | FF | 14.00 | 13.80 | 1 .. 1e+15 | 98% | below 1: 0.00 | 16 | CONDITIONING |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `asin` — QF 20 pts (perpendicular approach to the imaginary axis)
- `acos` — QF 14 pts (perpendicular approach to the real axis); FF 5 pts (polar shells)
- `atan` — QF 20 pts (perpendicular approach to the real axis)
- `asinh` — QF 36 pts (perpendicular approach to the real axis)
- `acosh` — QF 12 pts (perpendicular approach to the real axis)
- `atanh` — QF 26 pts (the real axis on a geometric ladder); TF 16 pts (the real axis on a geometric ladder); FF 16 pts (the real axis on a geometric ladder)

---

## Totals across all 252 cells

| classification | points below 50% of cap |
|---|---:|
| UNDERFLOW | 5849 |
| OVERFLOW | 11169 |
| ARG_RANGE | 332 |
| CONDITIONING | 8918 |
| UNEXPLAINED | 238 |
| **total triaged** | **26506** |

Out of 428,592 scored points.

---

## Note for CI (S7)

`validation/check_domains_fresh.sh` regenerates this file and diffs it against
the committed copy, exiting nonzero on any difference. It needs only Python 3
and the three committed CSVs — no build, no Kokkos, no libquadmath — and runs
in well under a second. It is deliberately **not** registered as a ctest test,
because the rest of the suite tests compiled numerics and a documentation
freshness check does not belong in the same gate. Wire it into CI directly.

Note that the check only proves the markdown matches the CSVs. If a numeric fix
lands, the CSVs must be regenerated first — `scripts/sweep_accuracy` to
re-baseline, then `scripts/sweep_accuracy --classify
validation/sweep/sweep_classified.csv` — and only then this file.
