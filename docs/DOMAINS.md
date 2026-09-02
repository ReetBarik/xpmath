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
| `fma` | QF | 29.00 | 27.85 | 1 .. 1 | 90% | below 1: 14.31; above 1.15: 14.25 | 31 | ARG_RANGE (**12 unexplained**) |
| `fma` | TF | 21.70 | 20.65 | 157.1 .. 169.6 | 89% | below 144.5: 7.74; above 172.8: 10.27 | 63 | UNEXPLAINED (**44 unexplained**) |
| `fma` | FF | 14.00 | 12.93 | 1 .. 1 | 88% | below 1: 4.16; above 1: 6.67 | 129 | CONDITIONING (**4 unexplained**) |
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
| `hypot` | QF | 29.00 | 28.82 | 3e-10 .. 1e+30 | 98% | below 1e-16: 13.71 | 4 | UNEXPLAINED (**4 unexplained**) |
| `hypot` | TF | 21.70 | 21.64 | 1e-13 .. 1e+30 | 99% | below 3e-18: 10.54 | 1 | UNEXPLAINED (**1 unexplained**) |
| `hypot` | FF | 14.00 | 14.00 | 1e-17 .. 1e+30 | 100% | -- | 0 | -- |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `add` — QF 1 pts (log sweep, |x| = 10^e); TF 1 pts (log sweep, |x| = 10^e); FF 100 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `sub` — QF 1 pts (log sweep, |x| = 10^e); TF 1 pts (log sweep, |x| = 10^e); FF 87 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `mul` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 35 pts (log sweep, |x| = 10^e); TF 30 pts (log sweep, |x| = 10^e); FF 25 pts (log sweep, |x| = 10^e)
- `div` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 17 pts (log sweep, |x| = 10^e); TF 12 pts (log sweep, |x| = 10^e); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fma` — QF 31 pts (log sweep, |x| = 10^e); TF 63 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 129 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `abs` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `copysign` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fmax` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fmin` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fdim` — FF 35 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `hypot` — QF 4 pts (log sweep, |x| = 10^e); TF 1 pts (log sweep, |x| = 10^e)

---

## Rounding and remainder (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `ceil` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `ceil` | QF | 29.00 | 28.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `ceil` | TF | 21.70 | 21.67 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `ceil` | FF | 14.00 | 13.44 | 1e-30 .. 1e+14 | 96% | below 1e-323: 0.00; above 3e+14: 0.00 | 66 | UNEXPLAINED (**64 unexplained**) |
| `floor` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `floor` | QF | 29.00 | 28.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `floor` | TF | 21.70 | 21.67 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 2 | ARG_RANGE |
| `floor` | FF | 14.00 | 13.44 | 1e-30 .. 1e+14 | 96% | below 1e-323: 0.00; above 3e+14: 0.00 | 66 | UNEXPLAINED (**64 unexplained**) |
| `round` | DD | 31.00 | 30.85 | 6.55 .. 1e+30 | 100% | below 6.5: 0.85 | 8 | UNEXPLAINED (**8 unexplained**) |
| `round` | QF | 29.00 | 28.85 | 7.55 .. 1e+15 | 99% | below 7.5: 0.90 | 8 | UNEXPLAINED (**8 unexplained**) |
| `round` | TF | 21.70 | 21.59 | 7.55 .. 1e+15 | 99% | below 7.5: 0.90 | 8 | UNEXPLAINED (**8 unexplained**) |
| `round` | FF | 14.00 | 13.39 | 6.55 .. 1e+14 | 96% | below 6.5: 0.85; above 3e+14: 0.00 | 72 | UNEXPLAINED (**72 unexplained**) |
| `trunc` | DD | 31.00 | 31.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | QF | 29.00 | 29.00 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | TF | 21.70 | 21.70 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `trunc` | FF | 14.00 | 13.46 | 0 .. 1e+14 | 96% | above 3e+14: 0.00 | 64 | UNEXPLAINED (**64 unexplained**) |
| `fmod` | DD | 31.00 | 26.20 | 0 .. 1e-07 | 81% | above 3e-07: 12.03 | 317 | UNEXPLAINED (**317 unexplained**) |
| `fmod` | QF | 29.00 | 24.00 | 3e-30 .. 1e-15 | 78% | below 1e-30: 8.23; above 3e-15: 0.55 | 353 | UNEXPLAINED (**341 unexplained**) |
| `fmod` | TF | 21.70 | 16.43 | 3e-30 .. 3e-17 | 68% | below 1e-30: 8.23; above 3e-16: 8.07 | 412 | UNEXPLAINED (**401 unexplained**) |
| `fmod` | FF | 14.00 | 9.07 | 3e-30 .. 1e-25 | 55% | below 1e-323: 0.00; above 3e-16: 0.00 | 586 | CONDITIONING (**26 unexplained**) |
| `remainder` | DD | 31.00 | 25.27 | 0 .. 1e-14 | 78% | above 3e-14: 0.00 | 356 | UNEXPLAINED (**356 unexplained**) |
| `remainder` | QF | 29.00 | 23.43 | 1e-28 .. 1e-14 | 76% | below 3e-29: 13.84; above 3e-14: 0.18 | 393 | UNEXPLAINED (**372 unexplained**) |
| `remainder` | TF | 21.70 | 16.07 | 1e-28 .. 1e-17 | 67% | below 1e-30: 2.81; above 3e-14: 0.18 | 451 | UNEXPLAINED (**437 unexplained**) |
| `remainder` | FF | 14.00 | 8.54 | 3e-30 .. 1e-27 | 49% | below 1e-30: 2.81; above 3e-25: 1.69 | 634 | CONDITIONING (**35 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `ceil` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 66 pts (log sweep, |x| = 10^e)
- `floor` — QF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 66 pts (log sweep, |x| = 10^e)
- `round` — DD 8 pts (linear sweep over [-8, 8]); QF 8 pts (linear sweep over [-8, 8]); TF 8 pts (linear sweep over [-8, 8]); FF 72 pts (log sweep, |x| = 10^e)
- `trunc` — FF 64 pts (log sweep, |x| = 10^e)
- `fmod` — DD 317 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 353 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 412 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 586 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `remainder` — DD 356 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 393 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 451 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 634 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Exponential, logarithmic and power (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `exp` | DD | 31.00 | 30.32 | 0 .. 316.2 | 100% | above 1000: 0.00 | 6 | OVERFLOW |
| `exp` | QF | 29.00 | 15.37 | 0 .. 43.98 | 52% | above 72.26: 14.21 | 759 | UNDERFLOW |
| `exp` | TF | 21.70 | 11.67 | 0 .. 59.69 | 53% | above 81.68: 10.03 | 745 | UNDERFLOW |
| `exp` | FF | 14.00 | 7.63 | 0 .. 62.83 | 54% | above 91.11: 0.00 | 730 | OVERFLOW |
| `exp2` | DD | 31.00 | 30.37 | 0 .. 1000 | 100% | above 3162: 0.00 | 4 | OVERFLOW |
| `exp2` | QF | 29.00 | 17.36 | 0 .. 62.83 | 57% | above 103.7: 13.94 | 647 | UNDERFLOW |
| `exp2` | TF | 21.70 | 13.19 | 0 .. 84.82 | 59% | above 116.2: 10.55 | 628 | UNDERFLOW |
| `exp2` | FF | 14.00 | 8.65 | 0 .. 94.25 | 61% | above 128.8: 0.00 | 608 | OVERFLOW |
| `exp10` | DD | 31.00 | 29.59 | 0 .. 295.3 | 97% | above 311: 0.00 | 26 | OVERFLOW |
| `exp10` | QF | 29.00 | 12.91 | 0 .. 18.85 | 44% | above 31.42: 13.96 | 903 | UNDERFLOW |
| `exp10` | TF | 21.70 | 9.74 | 0 .. 25.13 | 45% | above 34.56: 10.76 | 898 | UNDERFLOW |
| `exp10` | FF | 14.00 | 6.33 | 0 .. 31.62 | 46% | above 40.84: 0.00 | 888 | OVERFLOW |
| `expm1` | DD | 31.00 | 29.67 | 0 .. 316.2 | 97% | above 1000: 0.00 | 55 | OVERFLOW |
| `expm1` | QF | 29.00 | 22.24 | 3e-20 .. 87.96 | 76% | below 1e-323: 0.00; above 91.11: 0.00 | 369 | OVERFLOW |
| `expm1` | TF | 21.70 | 16.27 | 0.03162 .. 87.96 | 71% | below 1e-09: 10.81; above 91.11: 0.00 | 371 | OVERFLOW (**2 unexplained**) |
| `expm1` | FF | 14.00 | 10.33 | 1e-30 .. 62.83 | 74% | below 1e-323: 0.00; above 91.11: 0.00 | 421 | OVERFLOW |
| `log` | DD | 31.00 | 30.36 | 1.01 .. 1e+30 | 96% | below 0: 0.00 | 2 | OVERFLOW |
| `log` | QF | 29.00 | 28.27 | 1.001 .. 3e+20 | 95% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log` | TF | 21.70 | 21.18 | 1.1 .. 1e+30 | 95% | below 1: 10.81 | 10 | ARG_RANGE (**4 unexplained**) |
| `log` | FF | 14.00 | 13.76 | 1.1 .. 1e+30 | 97% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log2` | DD | 31.00 | 30.36 | 1.01 .. 1e+30 | 96% | below 0: 0.00 | 2 | OVERFLOW |
| `log2` | QF | 29.00 | 28.27 | 1.001 .. 3e+20 | 95% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log2` | TF | 21.70 | 21.18 | 1.1 .. 1e+30 | 95% | below 1: 10.81 | 10 | ARG_RANGE (**4 unexplained**) |
| `log2` | FF | 14.00 | 13.76 | 1.1 .. 1e+30 | 97% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log10` | DD | 31.00 | 30.36 | 1.01 .. 1e+30 | 96% | below 0: 0.00 | 2 | OVERFLOW |
| `log10` | QF | 29.00 | 28.27 | 1.001 .. 3e+20 | 95% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log10` | TF | 21.70 | 21.17 | 1.1 .. 1e+30 | 95% | below 1: 10.81 | 10 | ARG_RANGE (**4 unexplained**) |
| `log10` | FF | 14.00 | 13.76 | 1.1 .. 1e+30 | 97% | below 1e-323: 0.00 | 6 | ARG_RANGE |
| `log1p` | DD | 31.00 | 30.76 | 1 .. 1e+30 | 100% | below 1: 0.00 | 6 | OVERFLOW |
| `log1p` | QF | 29.00 | 28.37 | 1 .. 3e+18 | 96% | below 1: 0.00 | 8 | OVERFLOW |
| `log1p` | TF | 21.70 | 21.44 | 1 .. 1e+29 | 99% | below 1: 0.00 | 8 | OVERFLOW |
| `log1p` | FF | 14.00 | 13.86 | 1.001 .. 1e+30 | 99% | below 1: 0.00 | 8 | OVERFLOW |
| `pow` | DD | 31.00 | 30.02 | 1e-30 .. 1e+30 | 100% | below 5e-324: 15.19 | 2 | ARG_RANGE |
| `pow` | QF | 29.00 | 25.99 | 0.3162 .. 3.2 | 84% | below 3e-05: 14.48; above 3.65: 13.96 | 147 | UNDERFLOW |
| `pow` | TF | 21.70 | 19.96 | 1 .. 1.001 | 84% | below 1e-323: 0.00; above 28.27: 9.78 | 7 | ARG_RANGE |
| `pow` | FF | 14.00 | 13.40 | 1e-30 .. 1e-08 | 90% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `sqrt` | DD | 31.00 | 30.98 | 1e-30 .. 1e+30 | 100% | -- | 0 | -- |
| `sqrt` | QF | 29.00 | 28.70 | 3e-20 .. 1e+30 | 97% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `sqrt` | TF | 21.70 | 21.59 | 3e-26 .. 1e+30 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `sqrt` | FF | 14.00 | 13.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `exp` — DD 6 pts (log sweep, |x| = 10^e); QF 759 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 745 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `exp2` — DD 4 pts (log sweep, |x| = 10^e); QF 647 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 628 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 608 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `exp10` — DD 26 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 903 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 898 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 888 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `expm1` — DD 55 pts (log sweep, |x| = 10^e); QF 369 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 371 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 421 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log` — DD 2 pts (linear sweep over [-8, 8]); QF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 10 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log2` — DD 2 pts (linear sweep over [-8, 8]); QF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 10 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log10` — DD 2 pts (linear sweep over [-8, 8]); QF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 10 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log1p` — DD 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `pow` — DD 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 147 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 7 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `sqrt` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Trigonometric and inverse trigonometric (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `sin` | DD | 31.00 | 21.38 | 0 .. 3.1 | 33% | above 3.142: 15.42 | 72 | CONDITIONING |
| `sin` | QF | 29.00 | 18.99 | 3e-28 .. 3.1 | 32% | below 3e-30: 14.31; above 3.142: 13.46 | 520 | CONDITIONING (**4 unexplained**) |
| `sin` | TF | 21.70 | 11.84 | 3e-28 .. 3.1 | 32% | below 1e-323: 0.00; above 3.142: 7.15 | 1078 | CONDITIONING (**14 unexplained**) |
| `sin` | FF | 14.00 | 4.60 | 3e-29 .. 3.05 | 32% | below 1e-29: 0.00; above 3.142: 0.00 | 1098 | CONDITIONING (**6 unexplained**) |
| `cos` | DD | 31.00 | 29.75 | 1.6 .. 1e+05 | 93% | below 1.571: 15.24; above 1e+17: 0.00 | 54 | CONDITIONING |
| `cos` | QF | 29.00 | 27.65 | 1.6 .. 1000 | 93% | below 1.571: 13.39; above 1e+15: 14.35 | 72 | CONDITIONING |
| `cos` | TF | 21.70 | 20.55 | 1.6 .. 1000 | 93% | below 1.571: 7.35; above 3e+12: 10.64 | 82 | CONDITIONING (**14 unexplained**) |
| `cos` | FF | 14.00 | 12.94 | 7.9 .. 316.2 | 92% | below 1.571: 0.00; above 3e+07: 6.72 | 104 | CONDITIONING (**6 unexplained**) |
| `tan` | DD | 31.00 | 21.35 | 0 .. 1.55 | 33% | above 1.571: 15.24 | 76 | CONDITIONING |
| `tan` | QF | 29.00 | 18.89 | 3e-28 .. 1.55 | 32% | below 3e-30: 14.31; above 1.571: 13.90 | 532 | CONDITIONING (**6 unexplained**) |
| `tan` | TF | 21.70 | 11.75 | 3e-28 .. 1.55 | 31% | below 1e-323: 0.00; above 1.571: 6.84 | 1090 | CONDITIONING (**14 unexplained**) |
| `tan` | FF | 14.00 | 4.50 | 3e-29 .. 1.55 | 31% | below 1e-29: 0.00; above 1.571: 0.00 | 1110 | CONDITIONING (**72 unexplained**) |
| `asin` | DD | 31.00 | 30.90 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `asin` | QF | 29.00 | 28.55 | 3e-28 .. 1e+28 | 99% | below 3e-30: 14.35; above 3e+29: 14.48 | 12 | UNEXPLAINED (**6 unexplained**) |
| `asin` | TF | 21.70 | 21.50 | 3e-28 .. 1e+28 | 99% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asin` | FF | 14.00 | 13.73 | 1 .. 3e+28 | 99% | below 1e-29: 0.00; above 1e+29: 0.00 | 16 | UNEXPLAINED (**12 unexplained**) |
| `acos` | DD | 31.00 | 30.99 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `acos` | QF | 29.00 | 28.94 | 1.001 .. 1e+30 | 100% | -- | 0 | -- |
| `acos` | TF | 21.70 | 21.66 | 1 .. 1e+30 | 99% | -- | 0 | -- |
| `acos` | FF | 14.00 | 13.95 | 1.01 .. 1e+30 | 99% | -- | 0 | -- |
| `atan` | DD | 31.00 | 30.98 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `atan` | QF | 29.00 | 28.01 | 3e-28 .. 1e+19 | 96% | below 3e-30: 14.35; above 3e+19: 0.00 | 52 | UNEXPLAINED (**48 unexplained**) |
| `atan` | TF | 21.70 | 21.02 | 3e-28 .. 1e+19 | 96% | below 1e-323: 0.00; above 3e+19: 0.00 | 48 | UNEXPLAINED (**44 unexplained**) |
| `atan` | FF | 14.00 | 13.52 | 3e-29 .. 1e+19 | 97% | below 1e-29: 0.00; above 3e+19: 0.00 | 54 | UNEXPLAINED (**50 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sin` — DD 72 pts (log sweep, |x| = 10^e); QF 520 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 1078 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 1098 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `cos` — DD 54 pts (log sweep, |x| = 10^e); QF 72 pts (log sweep, |x| = 10^e); TF 82 pts (log sweep, |x| = 10^e); FF 104 pts (log sweep, |x| = 10^e)
- `tan` — DD 76 pts (log sweep, |x| = 10^e); QF 532 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 1090 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 1110 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `asin` — QF 12 pts (log sweep, |x| = 10^e); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 16 pts (log sweep, |x| = 10^e)
- `atan` — QF 52 pts (log sweep, |x| = 10^e); TF 48 pts (log sweep, |x| = 10^e); FF 54 pts (log sweep, |x| = 10^e)

---

## Hyperbolic and inverse hyperbolic (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `sinh` | DD | 31.00 | 29.95 | 0.001 .. 316.2 | 96% | below 1e-323: 0.00; above 1000: 0.00 | 10 | OVERFLOW |
| `sinh` | QF | 29.00 | 15.78 | 1e-30 .. 87.96 | 56% | below 1e-323: 0.00; above 91.11: 0.00 | 734 | OVERFLOW |
| `sinh` | TF | 21.70 | 11.80 | 1e-30 .. 87.96 | 56% | below 1e-323: 0.00; above 91.11: 0.00 | 734 | OVERFLOW |
| `sinh` | FF | 14.00 | 7.54 | 1e-30 .. 62.83 | 54% | below 1e-323: 0.00; above 91.11: 0.00 | 734 | OVERFLOW |
| `cosh` | DD | 31.00 | 30.35 | 0 .. 316.2 | 100% | above 1000: 0.00 | 6 | OVERFLOW |
| `cosh` | QF | 29.00 | 15.87 | 0 .. 87.96 | 56% | above 91.11: 0.00 | 730 | OVERFLOW |
| `cosh` | TF | 21.70 | 11.87 | 0 .. 87.96 | 56% | above 91.11: 0.00 | 730 | OVERFLOW |
| `cosh` | FF | 14.00 | 7.59 | 0 .. 62.83 | 54% | above 91.11: 0.00 | 730 | OVERFLOW |
| `tanh` | DD | 31.00 | 30.99 | 0 .. 1e+30 | 100% | -- | 0 | -- |
| `tanh` | QF | 29.00 | 28.90 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `tanh` | TF | 21.70 | 21.62 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `tanh` | FF | 14.00 | 13.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asinh` | DD | 31.00 | 30.45 | 0.03162 .. 1e+30 | 96% | below 1e-323: 0.00 | 4 | ARG_RANGE |
| `asinh` | QF | 29.00 | 27.48 | 0.01 .. 1e+19 | 92% | below 3e-30: 14.01; above 3e+19: 0.00 | 52 | UNEXPLAINED (**48 unexplained**) |
| `asinh` | TF | 21.70 | 19.38 | 0.05 .. 21.99 | 67% | below 1e-09: 10.81; above 3e+06: 10.27 | 72 | UNEXPLAINED (**68 unexplained**) |
| `asinh` | FF | 14.00 | 13.11 | 0.35 .. 1e+19 | 90% | below 1e-323: 0.00; above 3e+19: 0.00 | 48 | UNEXPLAINED (**44 unexplained**) |
| `acosh` | DD | 31.00 | 30.67 | 1 .. 1e+30 | 97% | -- | 0 | -- |
| `acosh` | QF | 29.00 | 27.70 | 1 .. 1e+19 | 94% | above 3e+19: 0.00 | 44 | UNEXPLAINED (**44 unexplained**) |
| `acosh` | TF | 21.70 | 20.73 | 1 .. 1e+19 | 94% | above 3e+19: 0.00 | 44 | UNEXPLAINED (**44 unexplained**) |
| `acosh` | FF | 14.00 | 13.36 | 1.01 .. 1e+19 | 93% | above 3e+19: 0.00 | 44 | UNEXPLAINED (**44 unexplained**) |
| `atanh` | DD | 31.00 | 28.67 | 1 .. 138.2 | 90% | below 1: 0.00 | 12 | OVERFLOW |
| `atanh` | QF | 29.00 | 27.99 | 1 .. 1e+29 | 96% | below 1: 0.00 | 54 | UNEXPLAINED (**42 unexplained**) |
| `atanh` | TF | 21.70 | 21.56 | 1 .. 1e+29 | 99% | below 1: 0.00 | 8 | OVERFLOW |
| `atanh` | FF | 14.00 | 13.51 | 1.001 .. 1e+30 | 96% | below 1: 0.00 | 54 | CONDITIONING (**4 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sinh` — DD 10 pts (log sweep, |x| = 10^e); QF 734 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 734 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 734 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `cosh` — DD 6 pts (log sweep, |x| = 10^e); QF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 730 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `tanh` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `asinh` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 52 pts (log sweep, |x| = 10^e); TF 72 pts (log sweep, |x| = 10^e); FF 48 pts (log sweep, |x| = 10^e)
- `acosh` — QF 44 pts (log sweep, |x| = 10^e); TF 44 pts (log sweep, |x| = 10^e); FF 44 pts (log sweep, |x| = 10^e)
- `atanh` — DD 12 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 54 pts (geometric approach to +-1); TF 8 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 54 pts (geometric approach to +-1)

---

## Complex arithmetic and construction (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `add` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `add` | QF | 29.00 | 28.92 | 1e-29 .. 1 | 100% | below 1e-30: 13.17; above 1: 4.32 | 6 | CONDITIONING |
| `add` | TF | 21.70 | 21.65 | 1e-29 .. 1 | 100% | above 1: 4.32 | 4 | CONDITIONING |
| `add` | FF | 14.00 | 13.27 | 1 .. 1 | 90% | below 0.99: 5.16; above 1: 6.33 | 84 | CONDITIONING |
| `sub` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `sub` | QF | 29.00 | 28.91 | 1e-29 .. 1 | 100% | below 1e-30: 5.52; above 1: 8.53 | 6 | CONDITIONING |
| `sub` | TF | 21.70 | 21.64 | 1e-29 .. 1 | 100% | below 1e-30: 5.52; above 1: 8.53 | 6 | CONDITIONING |
| `sub` | FF | 14.00 | 13.23 | 0.99 .. 1 | 91% | below 0.99: 4.48; above 1: 0.00 | 96 | CONDITIONING |
| `mul` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `mul` | QF | 29.00 | 27.02 | 100 .. 1e+08 | 84% | below 10: 13.55 | 76 | UNEXPLAINED (**49 unexplained**) |
| `mul` | TF | 21.70 | 20.83 | 100 .. 1e+08 | 92% | below 10: 10.62; above 1e+08: 7.52 | 63 | UNEXPLAINED (**41 unexplained**) |
| `mul` | FF | 14.00 | 13.63 | 10 .. 1e+08 | 96% | below 10: 0.36; above 1e+08: 0.00 | 45 | UNEXPLAINED (**24 unexplained**) |
| `div` | DD | 31.00 | 29.59 | 0 .. 1e-09 | 89% | above 0.99: 13.14 | 7 | UNEXPLAINED (**5 unexplained**) |
| `div` | QF | 29.00 | 25.46 | 1e+04 .. 1e+08 | 73% | below 1e+04: 13.22; above 1e+08: 13.51 | 181 | UNEXPLAINED (**163 unexplained**) |
| `div` | TF | 21.70 | 19.65 | 1e-22 .. 1e-20 | 82% | below 1e-23: 5.88; above 1e-19: 10.49 | 171 | UNEXPLAINED (**161 unexplained**) |
| `div` | FF | 14.00 | 12.38 | 0.99 .. 1 | 87% | below 0.99: 0.00; above 1: 0.00 | 197 | CONDITIONING (**17 unexplained**) |
| `abs` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `abs` | QF | 29.00 | 28.67 | 1e-09 .. 1e+15 | 97% | below 1e-17: 11.79 | 12 | UNEXPLAINED (**12 unexplained**) |
| `abs` | TF | 21.70 | 21.57 | 1e-12 .. 1e+15 | 98% | below 1e-18: 9.88 | 6 | UNEXPLAINED (**6 unexplained**) |
| `abs` | FF | 14.00 | 13.97 | 1e-16 .. 1e+15 | 99% | -- | 0 | -- |
| `conj` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | QF | 29.00 | 29.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | TF | 21.70 | 21.70 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `conj` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | -- |
| `polar` | DD | 31.00 | 30.74 | 0 .. 1e+04 | 98% | -- | 0 | -- |
| `polar` | QF | 29.00 | 27.44 | 1 .. 2 | 86% | below 1: 13.77; above 2: 13.77 | 21 | UNEXPLAINED (**21 unexplained**) |
| `polar` | TF | 21.70 | 21.08 | 1 .. 2 | 90% | -- | 0 | -- |
| `polar` | FF | 14.00 | 13.44 | 0 .. 0.1 | 94% | above 0.5: 0.00 | 52 | UNEXPLAINED (**40 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `add` — QF 6 pts (perpendicular approach to the real axis); TF 4 pts (perpendicular approach to the real axis); FF 84 pts (perpendicular approach to the real axis)
- `sub` — QF 6 pts (perpendicular approach to the real axis); TF 6 pts (perpendicular approach to the real axis); FF 96 pts (perpendicular approach to the real axis)
- `mul` — QF 76 pts (perpendicular approach to the real axis); TF 63 pts (perpendicular approach to the real axis); FF 45 pts (perpendicular approach to the real axis)
- `div` — DD 7 pts (polar shells); QF 181 pts (perpendicular approach to the real axis); TF 171 pts (perpendicular approach to the real axis); FF 197 pts (perpendicular approach to the real axis)
- `abs` — QF 12 pts (the real axis on a geometric ladder); TF 6 pts (the real axis on a geometric ladder)
- `polar` — QF 21 pts (perpendicular approach to the real axis); FF 52 pts (perpendicular approach to the real axis)

---

## Complex exponential, logarithmic, power and root (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `exp` | DD | 31.00 | 29.24 | 0 .. 100 | 96% | above 1000: 0.00 | 75 | OVERFLOW |
| `exp` | QF | 29.00 | 24.32 | 1 .. 2 | 79% | below 1: 13.77; above 2: 13.82 | 219 | OVERFLOW (**26 unexplained**) |
| `exp` | TF | 21.70 | 18.66 | 1e-27 .. 0.1 | 84% | above 100: 1.77 | 193 | OVERFLOW |
| `exp` | FF | 14.00 | 11.86 | 1e-28 .. 0.1 | 86% | below 1e-29: 0.00; above 0.5: 0.00 | 243 | OVERFLOW (**36 unexplained**) |
| `log` | DD | 31.00 | 26.77 | 2 .. 2.236 | 76% | below 2: 0.00; above 10: 0.00 | 84 | CONDITIONING |
| `log` | QF | 29.00 | 23.97 | 2 .. 2.236 | 71% | below 2: 0.00; above 10: 0.00 | 259 | CONDITIONING (**24 unexplained**) |
| `log` | TF | 21.70 | 17.99 | 2 .. 2.236 | 69% | below 2: 0.00; above 10: 0.00 | 226 | CONDITIONING (**6 unexplained**) |
| `log` | FF | 14.00 | 11.45 | 2 .. 2.236 | 68% | below 2: 0.00; above 10: 0.00 | 228 | CONDITIONING (**24 unexplained**) |
| `log10` | DD | 31.00 | 26.77 | 2 .. 2.236 | 76% | below 2: 0.00; above 10: 0.00 | 84 | CONDITIONING |
| `log10` | QF | 29.00 | 23.77 | 2 .. 2.236 | 68% | below 2: 0.00; above 10: 0.00 | 259 | CONDITIONING (**24 unexplained**) |
| `log10` | TF | 21.70 | 17.95 | 2 .. 2.236 | 69% | below 2: 0.00; above 10: 0.00 | 226 | CONDITIONING (**6 unexplained**) |
| `log10` | FF | 14.00 | 11.45 | 2 .. 2.236 | 68% | below 2: 0.00; above 10: 0.00 | 228 | CONDITIONING (**24 unexplained**) |
| `pow` | DD | 31.00 | 28.21 | 2 .. 2.236 | 90% | below 2: 0.00; above 10: 0.00 | 83 | CONDITIONING |
| `pow` | QF | 29.00 | 25.38 | 2 .. 2.002 | 83% | below 2: 14.11; above 2.236: 8.92 | 135 | CONDITIONING (**11 unexplained**) |
| `pow` | TF | 21.70 | 19.08 | 2 .. 2.002 | 80% | below 2: 0.00; above 2.236: 8.92 | 122 | CONDITIONING (**11 unexplained**) |
| `pow` | FF | 14.00 | 12.13 | 0.5 .. 0.5 | 70% | below 0.5: 0.08; above 0.9: 0.00 | 114 | CONDITIONING (**24 unexplained**) |
| `sqrt` | DD | 31.00 | 29.57 | 2 .. 2.236 | 95% | below 2: 0.00; above 10: 0.00 | 82 | UNEXPLAINED (**82 unexplained**) |
| `sqrt` | QF | 29.00 | 26.21 | 2 .. 2.236 | 80% | below 2: 0.00; above 10: 0.00 | 92 | UNEXPLAINED (**92 unexplained**) |
| `sqrt` | TF | 21.70 | 20.30 | 2 .. 2.236 | 88% | below 2: 0.00; above 10: 0.00 | 87 | UNEXPLAINED (**87 unexplained**) |
| `sqrt` | FF | 14.00 | 13.34 | 2 .. 2.236 | 95% | below 2: 0.00; above 10: 0.00 | 82 | UNEXPLAINED (**43 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `exp` — DD 75 pts (polar shells); QF 219 pts (perpendicular approach to the real axis); TF 193 pts (perpendicular approach to the real axis); FF 243 pts (perpendicular approach to the real axis)
- `log` — DD 84 pts (the real axis on a geometric ladder); QF 259 pts (perpendicular approach to the real axis); TF 226 pts (the real axis on a geometric ladder); FF 228 pts (the real axis on a geometric ladder)
- `log10` — DD 84 pts (the real axis on a geometric ladder); QF 259 pts (perpendicular approach to the real axis); TF 226 pts (the real axis on a geometric ladder); FF 228 pts (the real axis on a geometric ladder)
- `pow` — DD 83 pts (the real axis on a geometric ladder); QF 135 pts (the real axis on a geometric ladder); TF 122 pts (the real axis on a geometric ladder); FF 114 pts (the real axis on a geometric ladder)
- `sqrt` — DD 82 pts (the real axis on a geometric ladder); QF 92 pts (the real axis on a geometric ladder); TF 87 pts (the real axis on a geometric ladder); FF 82 pts (the real axis on a geometric ladder)

---

## Complex trigonometric and hyperbolic (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `sin` | DD | 31.00 | 27.64 | 0.5 .. 1 | 76% | above 1e+04: 0.00 | 60 | OVERFLOW |
| `sin` | QF | 29.00 | 25.96 | 1 .. 2 | 75% | below 1: 13.80; above 2: 13.77 | 46 | OVERFLOW (**16 unexplained**) |
| `sin` | TF | 21.70 | 20.25 | 1e-27 .. 0.1 | 87% | above 1e+04: 0.00 | 42 | OVERFLOW |
| `sin` | FF | 14.00 | 12.71 | 1e-28 .. 1 | 92% | below 1e-29: 0.00; above 1: 0.00 | 120 | OVERFLOW (**32 unexplained**) |
| `cos` | DD | 31.00 | 27.79 | 0.5 .. 1 | 77% | above 1e+04: 0.00 | 60 | OVERFLOW |
| `cos` | QF | 29.00 | 26.04 | 1 .. 2 | 76% | below 1: 13.77; above 2: 13.77 | 47 | OVERFLOW (**13 unexplained**) |
| `cos` | TF | 21.70 | 20.30 | 1 .. 2 | 87% | above 1e+04: 0.00 | 42 | OVERFLOW |
| `cos` | FF | 14.00 | 12.71 | 1e-28 .. 1 | 92% | below 1e-29: 0.00; above 1: 0.00 | 120 | OVERFLOW (**32 unexplained**) |
| `tan` | DD | 31.00 | 27.16 | 0.5 .. 1 | 68% | above 1e+04: 0.00 | 60 | UNEXPLAINED (**32 unexplained**) |
| `tan` | QF | 29.00 | 24.87 | 1 .. 2 | 66% | below 1: 13.84; above 2: 13.68 | 104 | UNEXPLAINED (**72 unexplained**) |
| `tan` | TF | 21.70 | 19.36 | 1e-27 .. 0.1 | 77% | above 10: 10.19 | 92 | UNEXPLAINED (**48 unexplained**) |
| `tan` | FF | 14.00 | 12.21 | 1e-28 .. 1 | 84% | below 1e-29: 0.00; above 1: 0.00 | 229 | UNEXPLAINED (**175 unexplained**) |
| `sinh` | DD | 31.00 | 27.21 | 0.001 .. 0.9 | 79% | above 1000: 0.00 | 112 | OVERFLOW |
| `sinh` | QF | 29.00 | 23.43 | 1 .. 2 | 70% | below 1: 13.80; above 2: 13.77 | 233 | OVERFLOW (**18 unexplained**) |
| `sinh` | TF | 21.70 | 18.27 | 1e-27 .. 0.1 | 80% | above 100: 0.00 | 215 | OVERFLOW |
| `sinh` | FF | 14.00 | 11.55 | 1e-28 .. 0.1 | 84% | below 1e-29: 0.00; above 0.5: 0.00 | 280 | OVERFLOW (**36 unexplained**) |
| `cosh` | DD | 31.00 | 27.51 | 0.001 .. 0.9 | 82% | above 1000: 0.00 | 112 | OVERFLOW |
| `cosh` | QF | 29.00 | 23.48 | 1 .. 2 | 70% | below 1: 13.77; above 2: 13.77 | 232 | OVERFLOW (**17 unexplained**) |
| `cosh` | TF | 21.70 | 18.31 | 1 .. 2 | 80% | above 100: 0.00 | 215 | OVERFLOW |
| `cosh` | FF | 14.00 | 11.56 | 1e-28 .. 0.1 | 84% | below 1e-29: 0.00; above 0.5: 0.00 | 280 | OVERFLOW (**36 unexplained**) |
| `tanh` | DD | 31.00 | 27.87 | 0 .. 2.236 | 84% | above 100: 0.00 | 136 | UNEXPLAINED (**136 unexplained**) |
| `tanh` | QF | 29.00 | 24.12 | 1 .. 2 | 68% | below 1: 13.84; above 2: 13.68 | 178 | UNEXPLAINED (**178 unexplained**) |
| `tanh` | TF | 21.70 | 18.66 | 1e-27 .. 0.1 | 77% | above 10: 10.19 | 152 | UNEXPLAINED (**152 unexplained**) |
| `tanh` | FF | 14.00 | 11.98 | 1e-28 .. 0.1 | 82% | below 1e-29: 0.00; above 0.5: 0.00 | 172 | UNEXPLAINED (**172 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sin` — DD 60 pts (polar shells); QF 46 pts (polar shells); TF 42 pts (polar shells); FF 120 pts (polar shells)
- `cos` — DD 60 pts (polar shells); QF 47 pts (polar shells); TF 42 pts (polar shells); FF 120 pts (polar shells)
- `tan` — DD 60 pts (polar shells); QF 104 pts (polar shells); TF 92 pts (polar shells); FF 229 pts (perpendicular approach to the imaginary axis)
- `sinh` — DD 112 pts (polar shells); QF 233 pts (perpendicular approach to the real axis); TF 215 pts (perpendicular approach to the real axis); FF 280 pts (perpendicular approach to the real axis)
- `cosh` — DD 112 pts (polar shells); QF 232 pts (perpendicular approach to the real axis); TF 215 pts (perpendicular approach to the real axis); FF 280 pts (perpendicular approach to the real axis)
- `tanh` — DD 136 pts (perpendicular approach to the real axis); QF 178 pts (perpendicular approach to the real axis); TF 152 pts (perpendicular approach to the real axis); FF 172 pts (perpendicular approach to the real axis)

---

## Complex inverse functions (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | dominant class |
|---|---|---:|---:|---|---:|---|---:|---|
| `asin` | DD | 31.00 | 27.96 | 1 .. 100 | 76% | below 0.5: 15.23; above 1e+09: 14.98 | 77 | UNEXPLAINED (**77 unexplained**) |
| `asin` | QF | 29.00 | 24.92 | 10 .. 100 | 65% | below 10: 13.77; above 1e+08: 13.68 | 128 | UNEXPLAINED (**112 unexplained**) |
| `asin` | TF | 21.70 | 18.51 | 1.005 .. 2 | 65% | below 1: 0.00; above 1e+07: 10.18 | 132 | UNEXPLAINED (**116 unexplained**) |
| `asin` | FF | 14.00 | 11.27 | 2 .. 2.236 | 59% | below 2: 0.00; above 10: 0.00 | 195 | UNEXPLAINED (**175 unexplained**) |
| `acos` | DD | 31.00 | 27.98 | 1 .. 100 | 77% | below 0.5: 15.23; above 1e+09: 14.98 | 77 | UNEXPLAINED (**77 unexplained**) |
| `acos` | QF | 29.00 | 25.31 | 1 .. 2 | 69% | below 1: 0.00; above 2: 14.12 | 112 | UNEXPLAINED (**90 unexplained**) |
| `acos` | TF | 21.70 | 18.61 | 1.005 .. 2 | 67% | below 1: 0.00; above 1e+07: 10.18 | 131 | UNEXPLAINED (**115 unexplained**) |
| `acos` | FF | 14.00 | 11.32 | 2 .. 2.236 | 59% | below 2: 0.00; above 10: 0.00 | 177 | UNEXPLAINED (**157 unexplained**) |
| `atan` | DD | 31.00 | 25.19 | 0 .. 1e-17 | 60% | above 1e-08: 8.13 | 256 | UNEXPLAINED (**250 unexplained**) |
| `atan` | QF | 29.00 | 22.33 | 0.99 .. 1 | 53% | below 0.99: 13.73; above 1: 14.48 | 330 | UNEXPLAINED (**314 unexplained**) |
| `atan` | TF | 21.70 | 15.83 | 0.99 .. 1 | 52% | below 0.99: 7.26; above 1: 10.81 | 365 | UNEXPLAINED (**357 unexplained**) |
| `atan` | FF | 14.00 | 9.51 | 0.99 .. 1 | 52% | below 0.99: 0.00; above 1: 0.00 | 454 | UNEXPLAINED (**448 unexplained**) |
| `asinh` | DD | 31.00 | 29.34 | 10 .. 1e+15 | 86% | below 10: 0.00 | 10 | UNEXPLAINED (**8 unexplained**) |
| `asinh` | QF | 29.00 | 25.86 | 100 .. 1e+15 | 70% | below 100: 14.49 | 58 | UNEXPLAINED (**47 unexplained**) |
| `asinh` | TF | 21.70 | 19.75 | 100 .. 1e+15 | 76% | below 10: 0.00 | 22 | UNEXPLAINED (**16 unexplained**) |
| `asinh` | FF | 14.00 | 12.33 | 100 .. 1e+15 | 79% | below 100: 0.00 | 72 | UNEXPLAINED (**68 unexplained**) |
| `acosh` | DD | 31.00 | 25.79 | 2 .. 2.236 | 72% | below 2: 0.00; above 10: 0.00 | 223 | CONDITIONING (**94 unexplained**) |
| `acosh` | QF | 29.00 | 23.50 | 2 .. 2.236 | 67% | below 2: 0.00; above 10: 0.00 | 247 | CONDITIONING (**107 unexplained**) |
| `acosh` | TF | 21.70 | 17.38 | 2 .. 2.236 | 68% | below 2: 0.00; above 10: 0.00 | 251 | CONDITIONING (**122 unexplained**) |
| `acosh` | FF | 14.00 | 10.77 | 2 .. 2.236 | 68% | below 2: 0.00; above 10: 0.00 | 311 | UNEXPLAINED (**182 unexplained**) |
| `atanh` | DD | 31.00 | 27.35 | 0 .. 0.1 | 76% | above 1: 0.00 | 135 | UNEXPLAINED (**126 unexplained**) |
| `atanh` | QF | 29.00 | 23.84 | 1e-28 .. 1e-08 | 65% | below 1e-30: 13.95; above 0.5: 13.10 | 237 | UNEXPLAINED (**193 unexplained**) |
| `atanh` | TF | 21.70 | 17.42 | 1e-28 .. 0.1 | 60% | above 0.5: 7.36 | 245 | UNEXPLAINED (**210 unexplained**) |
| `atanh` | FF | 14.00 | 10.76 | 1e-29 .. 0.1 | 60% | below 1e-30: 0.00; above 0.5: 0.00 | 305 | UNEXPLAINED (**267 unexplained**) |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `asin` — DD 77 pts (perpendicular approach to the real axis); QF 128 pts (perpendicular approach to the real axis); TF 132 pts (perpendicular approach to the real axis); FF 195 pts (perpendicular approach to the real axis)
- `acos` — DD 77 pts (perpendicular approach to the real axis); QF 112 pts (perpendicular approach to the real axis); TF 131 pts (perpendicular approach to the real axis); FF 177 pts (perpendicular approach to the real axis)
- `atan` — DD 256 pts (perpendicular approach to the real axis); QF 330 pts (perpendicular approach to the real axis); TF 365 pts (perpendicular approach to the real axis); FF 454 pts (perpendicular approach to the real axis)
- `asinh` — DD 10 pts (polar shells); QF 58 pts (perpendicular approach to the real axis); TF 22 pts (polar shells); FF 72 pts (perpendicular approach to the real axis)
- `acosh` — DD 223 pts (the real axis on a geometric ladder); QF 247 pts (the real axis on a geometric ladder); TF 251 pts (the real axis on a geometric ladder); FF 311 pts (perpendicular approach to the real axis)
- `atanh` — DD 135 pts (perpendicular approach to the imaginary axis); QF 237 pts (perpendicular approach to the imaginary axis); TF 245 pts (perpendicular approach to the imaginary axis); FF 305 pts (perpendicular approach to the imaginary axis)

---

## Totals across all 252 cells

| classification | points below 50% of cap |
|---|---:|
| UNDERFLOW | 4167 |
| OVERFLOW | 11633 |
| ARG_RANGE | 342 |
| CONDITIONING | 10360 |
| UNEXPLAINED | 9238 |
| **total triaged** | **35740** |

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
