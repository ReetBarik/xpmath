# Domain limits — where each operation can and cannot be trusted

**GENERATED FILE — do not edit.** Produced by `scripts/gen_domains.py` from
`validation/sweep/sweep_baseline.csv`, `validation/sweep/sweep_grid.csv` and
`validation/sweep/open_defects.txt`. Regenerate with:

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
  single verdict for the cell: `at or below bound` when every point is
  explained by the format and the conditioning, `UNRESOLVED` when the format
  cannot carry some point at all, `OPEN DEFECT` when some point exceeds its
  derived bound and is carried in `validation/sweep/open_defects.txt`.

**Caveat for the binary ops** (`add sub mul div pow hypot fmod remainder`
`copysign fmax fmin fdim fma`, and the complex `add sub mul div pow`): the
magnitude axis is the **first operand only**. The second is drawn
log-uniformly over a per-op window, with one point in seven a deliberately
cancelling pair, so a failure at a given |x| may be caused by the operand
paired with it rather than by x. For those rows read the `fails` count and the
verdict, and treat the band as indicative. The unary rows have no such
ambiguity.

Verdicts, from `scripts/sweep_accuracy --ulp`:

- **at or below bound** — every point is at or under the bound the format and the conditioning derive for it -- nothing left to explain
- **UNRESOLVED** — no verdict issuable: the answer, an operand or an intermediate does not fit the format, or the derived bound already exceeds 2^p ulps
- **OPEN DEFECT** — at least one point exceeds its derived bound and is carried in validation/sweep/open_defects.txt

There is one measurement behind all three: the error in ulps against the
`__float128` / `__complex128` oracle, compared against a bound derived from the
format (word count, exponent range, intermediate width) and the condition
number — never from what the implementation currently scores. The derivation is
in `docs/CORRECTNESS.md`. `UNRESOLVED` and `at or below bound` are both healthy
outcomes; `OPEN DEFECT` is the only one that names a bug, and every such point
is listed by name in the register.

### Why grouped by operation family rather than by backend

The four backends share one op inventory and one grid, so their cells are
directly comparable — and the question a caller actually has is *"can this op
hold my range, and if not which backend can?"*. Putting the four backends on
adjacent rows answers that by eye. A backend-major layout would answer the
rarer question ("what does DD do across all 63 ops?") while forcing a reader
comparing FF against DD to page between two distant sections.

---

## Arithmetic and selection (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `add` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `add` | QF | 29.00 | 28.89 | 3e-30 .. 1e+30 | 100% | below 1e-30: 9.56; above 4e+151: 0.00 | 7 | UNRESOLVED |
| `add` | TF | 21.70 | 21.62 | 3e-30 .. 1e+30 | 100% | below 1e-30: 9.56; above 4e+151: 0.00 | 7 | UNRESOLVED |
| `add` | FF | 14.00 | 13.15 | 0.9999 .. 1 | 90% | below 0.999: 6.22; above 1.1: 6.36 | 109 | UNRESOLVED |
| `sub` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `sub` | QF | 29.00 | 28.88 | 3e-30 .. 1e+30 | 100% | below 1e-30: 1.91; above 4e+151: 0.00 | 7 | UNRESOLVED |
| `sub` | TF | 21.70 | 21.61 | 3e-30 .. 1e+30 | 100% | below 1e-30: 1.91; above 4e+151: 0.00 | 7 | UNRESOLVED |
| `sub` | FF | 14.00 | 13.17 | 1 .. 1.05 | 89% | below 0.999: 3.01; above 1.571: 5.61 | 94 | UNRESOLVED |
| `mul` | DD | 31.00 | 30.95 | 1e-30 .. 5e+255 | 100% | below 1e-323: 4.89 | 4 | UNRESOLVED |
| `mul` | QF | 29.00 | 28.28 | 1e-10 .. 3e+20 | 96% | below 1e-16: 13.67; above 1e+21: 0.00 | 41 | UNRESOLVED |
| `mul` | TF | 21.70 | 21.24 | 3e-14 .. 3e+20 | 97% | below 3e-18: 10.59; above 1e+21: 0.00 | 36 | UNRESOLVED |
| `mul` | FF | 14.00 | 13.74 | 1e-17 .. 3e+20 | 98% | below 3e-20: 6.70; above 1e+21: 0.00 | 31 | UNRESOLVED |
| `div` | DD | 31.00 | 30.95 | 1e-30 .. 5e+255 | 100% | below 1e-323: 14.55 | 4 | UNRESOLVED |
| `div` | QF | 29.00 | 28.41 | 1e-06 .. 1e+24 | 96% | below 1e-21: 11.81; above 3e+24: 0.00 | 23 | UNRESOLVED |
| `div` | TF | 21.70 | 21.40 | 1e-12 .. 1e+24 | 97% | below 1e-22: 9.81; above 3e+24: 0.00 | 18 | UNRESOLVED |
| `div` | FF | 14.00 | 13.88 | 3e-21 .. 1e+24 | 99% | below 1e-323: 0.00; above 3e+24: 0.00 | 12 | UNRESOLVED |
| `fma` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `fma` | QF | 29.00 | 28.54 | 3e-09 .. 3e+20 | 98% | below 3e-13: 6.94; above 1e+21: 0.00 | 25 | UNRESOLVED |
| `fma` | TF | 21.70 | 21.39 | 1e-12 .. 3e+20 | 98% | below 3e-13: 6.94; above 1e+21: 0.00 | 25 | UNRESOLVED |
| `fma` | FF | 14.00 | 12.94 | 1 .. 1 | 88% | below 1: 0.00; above 1: 6.83 | 135 | UNRESOLVED |
| `abs` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `abs` | QF | 29.00 | 28.83 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `abs` | TF | 21.70 | 21.57 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `abs` | FF | 14.00 | 13.92 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `copysign` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `copysign` | QF | 29.00 | 28.83 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `copysign` | TF | 21.70 | 21.57 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `copysign` | FF | 14.00 | 13.92 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `fmax` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `fmax` | QF | 29.00 | 28.84 | 191.6 .. 3e+05 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 6 | UNRESOLVED |
| `fmax` | TF | 21.70 | 21.60 | 191.6 .. 3e+05 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 6 | UNRESOLVED |
| `fmax` | FF | 14.00 | 13.95 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 6 | UNRESOLVED |
| `fmin` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `fmin` | QF | 29.00 | 28.88 | 6.15 .. 216.8 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 5 | UNRESOLVED |
| `fmin` | TF | 21.70 | 21.62 | 6.15 .. 216.8 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 5 | UNRESOLVED |
| `fmin` | FF | 14.00 | 13.96 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 5 | UNRESOLVED |
| `fdim` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `fdim` | QF | 29.00 | 28.95 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 3 | UNRESOLVED |
| `fdim` | TF | 21.70 | 21.66 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 3 | UNRESOLVED |
| `fdim` | FF | 14.00 | 13.67 | 5e+08 .. 1e+29 | 96% | below 314.2: 6.41; above 4e+151: 0.00 | 39 | UNRESOLVED |
| `hypot` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `hypot` | QF | 29.00 | 28.85 | 1e-19 .. 1e+30 | 99% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `hypot` | TF | 21.70 | 21.61 | 1e-26 .. 1e+30 | 99% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `hypot` | FF | 14.00 | 13.95 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `add` — FF 109 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `sub` — FF 94 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `mul` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 41 pts (log sweep, |x| = 10^e); TF 36 pts (log sweep, |x| = 10^e); FF 31 pts (log sweep, |x| = 10^e)
- `div` — DD 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 23 pts (log sweep, |x| = 10^e); TF 18 pts (log sweep, |x| = 10^e)
- `fma` — QF 25 pts (log sweep, |x| = 10^e); TF 25 pts (log sweep, |x| = 10^e); FF 135 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `fdim` — FF 39 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Rounding and remainder (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `ceil` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `ceil` | QF | 29.00 | 28.86 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 8 | UNRESOLVED |
| `ceil` | TF | 21.70 | 21.60 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 8 | UNRESOLVED |
| `ceil` | FF | 14.00 | 13.93 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 8 | UNRESOLVED |
| `floor` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `floor` | QF | 29.00 | 28.86 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 8 | UNRESOLVED |
| `floor` | TF | 21.70 | 21.60 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 8 | UNRESOLVED |
| `floor` | FF | 14.00 | 13.93 | 1e-30 .. 1e+30 | 100% | below 1e-323: 0.00; above 4e+151: 0.00 | 8 | UNRESOLVED |
| `round` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `round` | QF | 29.00 | 28.90 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `round` | TF | 21.70 | 21.62 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `round` | FF | 14.00 | 13.95 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `trunc` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `trunc` | QF | 29.00 | 28.90 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `trunc` | TF | 21.70 | 21.62 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `trunc` | FF | 14.00 | 13.95 | 0 .. 1e+30 | 100% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `fmod` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `fmod` | QF | 29.00 | 28.70 | 40.84 .. 219.9 | 99% | below 40.84: 0.36; above 219.9: 0.15 | 18 | UNRESOLVED |
| `fmod` | TF | 21.70 | 21.48 | 40.84 .. 219.9 | 99% | below 40.84: 0.36; above 219.9: 0.15 | 18 | UNRESOLVED |
| `fmod` | FF | 14.00 | 9.34 | 3e-30 .. 1e-25 | 57% | below 1e-323: 0.00; above 3e-16: 0.00 | 574 | UNRESOLVED |
| `remainder` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `remainder` | QF | 29.00 | 28.65 | 6.283 .. 78.54 | 99% | below 6.283: 0.00; above 78.54: 0.76 | 21 | UNRESOLVED |
| `remainder` | TF | 21.70 | 21.44 | 6.283 .. 78.54 | 99% | below 6.283: 0.00; above 78.54: 0.76 | 20 | UNRESOLVED |
| `remainder` | FF | 14.00 | 8.85 | 3e-30 .. 1e-27 | 51% | below 1e-30: 2.81; above 3e-25: 1.69 | 623 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `fmod` — QF 18 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 18 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 574 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `remainder` — QF 21 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 20 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 623 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Exponential, logarithmic and power (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `exp` | DD | 31.00 | 30.89 | 0 .. 344 | 100% | above 1000: 0.00 | 6 | UNRESOLVED |
| `exp` | QF | 29.00 | 16.02 | 0 .. 43.98 | 53% | above 72.26: 14.21 | 761 | UNRESOLVED |
| `exp` | TF | 21.70 | 12.16 | 0 .. 59.69 | 54% | above 81.68: 10.03 | 747 | UNRESOLVED |
| `exp` | FF | 14.00 | 7.92 | 0 .. 75.4 | 56% | above 91.11: 0.00 | 732 | UNRESOLVED |
| `exp2` | DD | 31.00 | 29.92 | 0 .. 1000 | 97% | above 3162: 0.00 | 59 | UNRESOLVED |
| `exp2` | QF | 29.00 | 16.95 | 0 .. 62.83 | 55% | above 103.7: 13.94 | 707 | UNRESOLVED |
| `exp2` | TF | 21.70 | 12.93 | 0 .. 84.82 | 57% | above 116.2: 10.55 | 688 | UNRESOLVED |
| `exp2` | FF | 14.00 | 8.46 | 0 .. 110 | 59% | above 128.8: 0.00 | 668 | UNRESOLVED |
| `exp10` | DD | 31.00 | 30.44 | 0 .. 295.3 | 97% | above 311: 0.00 | 28 | UNRESOLVED |
| `exp10` | QF | 29.00 | 13.58 | 0 .. 18.85 | 46% | above 31.42: 13.96 | 905 | UNRESOLVED |
| `exp10` | TF | 21.70 | 10.24 | 0 .. 25.13 | 46% | above 34.56: 10.76 | 900 | UNRESOLVED |
| `exp10` | FF | 14.00 | 6.64 | 0 .. 31.62 | 47% | above 40.84: 0.00 | 890 | UNRESOLVED |
| `expm1` | DD | 31.00 | 29.63 | 0 .. 344 | 96% | above 1000: 0.00 | 75 | UNRESOLVED |
| `expm1` | QF | 29.00 | 22.60 | 3e-20 .. 87.96 | 77% | below 1e-323: 0.00; above 91.11: 0.00 | 370 | UNRESOLVED |
| `expm1` | TF | 21.70 | 16.98 | 1e-30 .. 87.96 | 78% | below 1e-323: 0.00; above 91.11: 0.00 | 370 | UNRESOLVED |
| `expm1` | FF | 14.00 | 10.34 | 1e-30 .. 87.96 | 74% | below 1e-323: 0.00; above 91.11: 0.00 | 442 | UNRESOLVED |
| `log` | DD | 31.00 | 30.93 | 1e-30 .. 5e+255 | 100% | below 0: 0.00 | 2 | UNRESOLVED |
| `log` | QF | 29.00 | 28.71 | 1 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log` | TF | 21.70 | 21.50 | 1.001 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log` | FF | 14.00 | 13.81 | 1.01 .. 1e+30 | 97% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log2` | DD | 31.00 | 30.93 | 1e-30 .. 5e+255 | 100% | below 0: 0.00 | 2 | UNRESOLVED |
| `log2` | QF | 29.00 | 28.71 | 1 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log2` | TF | 21.70 | 21.50 | 1.001 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log2` | FF | 14.00 | 13.81 | 1.01 .. 1e+30 | 97% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log10` | DD | 31.00 | 30.93 | 1e-30 .. 5e+255 | 100% | below 0: 0.00 | 2 | UNRESOLVED |
| `log10` | QF | 29.00 | 28.71 | 1 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log10` | TF | 21.70 | 21.50 | 1.001 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log10` | FF | 14.00 | 13.81 | 1.01 .. 1e+30 | 97% | below 1e-323: 0.00; above 4e+151: 0.00 | 12 | UNRESOLVED |
| `log1p` | DD | 31.00 | 30.89 | 1 .. 5e+255 | 100% | below 1: 0.00 | 6 | UNRESOLVED |
| `log1p` | QF | 29.00 | 28.57 | 1 .. 3e+18 | 97% | below 1: 0.00; above 4e+151: 0.00 | 14 | UNRESOLVED |
| `log1p` | TF | 21.70 | 21.50 | 1 .. 1e+29 | 99% | below 1: 0.00; above 4e+151: 0.00 | 14 | UNRESOLVED |
| `log1p` | FF | 14.00 | 13.86 | 1.001 .. 1e+30 | 99% | below 1: 0.00; above 4e+151: 0.00 | 14 | UNRESOLVED |
| `pow` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `pow` | QF | 29.00 | 26.76 | 0.3162 .. 3.45 | 86% | below 3e-05: 14.48; above 3.65: 13.96 | 155 | UNRESOLVED |
| `pow` | TF | 21.70 | 20.64 | 0.0001 .. 3.6 | 87% | below 1e-323: 0.00; above 28.27: 9.78 | 13 | UNRESOLVED |
| `pow` | FF | 14.00 | 13.82 | 40.84 .. 188.5 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `sqrt` | DD | 31.00 | 30.98 | 1e-30 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `sqrt` | QF | 29.00 | 28.60 | 3e-20 .. 1e+30 | 97% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `sqrt` | TF | 21.70 | 21.52 | 3e-26 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `sqrt` | FF | 14.00 | 13.91 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `exp` — DD 6 pts (log sweep, |x| = 10^e); QF 761 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 747 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 732 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `exp2` — DD 59 pts (log sweep, |x| = 10^e); QF 707 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 688 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 668 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `exp10` — DD 28 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); QF 905 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 900 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 890 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `expm1` — DD 75 pts (log sweep, |x| = 10^e); QF 370 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 370 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 442 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `log` — DD 2 pts (linear sweep over [-8, 8])
- `log2` — DD 2 pts (linear sweep over [-8, 8])
- `log10` — DD 2 pts (linear sweep over [-8, 8])
- `log1p` — DD 6 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `pow` — QF 155 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 13 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Trigonometric and inverse trigonometric (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `sin` | DD | 31.00 | 21.38 | 0 .. 3.1 | 34% | above 3.142: 15.42 | 88 | UNRESOLVED |
| `sin` | QF | 29.00 | 19.01 | 3e-28 .. 3.1 | 33% | below 3e-30: 14.31; above 3.142: 13.46 | 538 | UNRESOLVED |
| `sin` | TF | 21.70 | 11.89 | 3e-28 .. 3.1 | 33% | below 1e-323: 0.00; above 3.142: 7.15 | 1096 | UNRESOLVED |
| `sin` | FF | 14.00 | 4.70 | 1e-30 .. 3.05 | 32% | below 1e-323: 0.00; above 3.142: 0.00 | 1120 | UNRESOLVED |
| `cos` | DD | 31.00 | 29.36 | 12.57 .. 316.2 | 91% | below 1.571: 15.24; above 5e+08: 15.33 | 76 | UNRESOLVED |
| `cos` | QF | 29.00 | 27.27 | 12.57 .. 316.2 | 91% | below 11: 13.79; above 344: 14.27 | 104 | UNRESOLVED |
| `cos` | TF | 21.70 | 20.26 | 12.57 .. 316.2 | 91% | below 11: 6.98; above 344: 7.08 | 118 | UNRESOLVED |
| `cos` | FF | 14.00 | 12.71 | 12.57 .. 316.2 | 90% | below 11: 0.00; above 344: 0.00 | 138 | UNRESOLVED |
| `tan` | DD | 31.00 | 21.03 | 0 .. 1.55 | 32% | above 1.571: 15.24 | 108 | UNRESOLVED |
| `tan` | QF | 29.00 | 18.58 | 3e-28 .. 1.55 | 31% | below 3e-30: 14.31; above 1.571: 13.90 | 576 | UNRESOLVED |
| `tan` | TF | 21.70 | 11.50 | 3e-28 .. 1.55 | 30% | below 1e-323: 0.00; above 1.571: 6.84 | 1138 | UNRESOLVED |
| `tan` | FF | 14.00 | 4.42 | 1e-30 .. 1.55 | 30% | below 1e-323: 0.00; above 1.571: 0.00 | 1152 | UNRESOLVED |
| `asin` | DD | 31.00 | 30.90 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `asin` | QF | 29.00 | 28.46 | 3e-28 .. 1e+28 | 98% | below 3e-30: 14.35; above 3e+29: 14.48 | 18 | UNRESOLVED |
| `asin` | TF | 21.70 | 21.43 | 3e-28 .. 1e+28 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `asin` | FF | 14.00 | 13.79 | 1 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `acos` | DD | 31.00 | 30.99 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `acos` | QF | 29.00 | 28.94 | 1.001 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `acos` | TF | 21.70 | 21.66 | 1 .. 5e+255 | 99% | -- | 0 | UNRESOLVED |
| `acos` | FF | 14.00 | 13.95 | 1.01 .. 5e+255 | 99% | -- | 0 | UNRESOLVED |
| `atan` | DD | 31.00 | 30.98 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `atan` | QF | 29.00 | 28.69 | 3e-28 .. 1e+30 | 99% | below 3e-30: 14.35; above 4e+151: 0.00 | 14 | UNRESOLVED |
| `atan` | TF | 21.70 | 21.52 | 3e-28 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `atan` | FF | 14.00 | 13.90 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sin` — DD 88 pts (log sweep, |x| = 10^e); QF 538 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 1096 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 1120 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `cos` — DD 76 pts (log sweep, |x| = 10^e); QF 104 pts (log sweep, |x| = 10^e); TF 118 pts (log sweep, |x| = 10^e); FF 138 pts (log sweep, |x| = 10^e)
- `tan` — DD 108 pts (log sweep, |x| = 10^e); QF 576 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 1138 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 1152 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `asin` — QF 18 pts (log sweep, |x| = 10^e)

---

## Hyperbolic and inverse hyperbolic (real)

| op | backend | cap | mean | trusted \|x\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `sinh` | DD | 31.00 | 30.89 | 0 .. 344 | 100% | above 1000: 0.00 | 6 | UNRESOLVED |
| `sinh` | QF | 29.00 | 16.44 | 1e-30 .. 87.96 | 57% | below 1e-323: 0.00; above 91.11: 0.00 | 736 | UNRESOLVED |
| `sinh` | TF | 21.70 | 12.29 | 1e-30 .. 87.96 | 57% | below 1e-323: 0.00; above 91.11: 0.00 | 736 | UNRESOLVED |
| `sinh` | FF | 14.00 | 7.84 | 1e-30 .. 87.96 | 56% | below 1e-323: 0.00; above 91.11: 0.00 | 742 | UNRESOLVED |
| `cosh` | DD | 31.00 | 30.89 | 0 .. 344 | 100% | above 1000: 0.00 | 6 | UNRESOLVED |
| `cosh` | QF | 29.00 | 16.50 | 0 .. 87.96 | 57% | above 91.11: 0.00 | 732 | UNRESOLVED |
| `cosh` | TF | 21.70 | 12.34 | 0 .. 87.96 | 57% | above 91.11: 0.00 | 732 | UNRESOLVED |
| `cosh` | FF | 14.00 | 7.88 | 0 .. 87.96 | 57% | above 91.11: 0.00 | 738 | UNRESOLVED |
| `tanh` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `tanh` | QF | 29.00 | 28.93 | 1e-30 .. 5e+255 | 100% | below 1e-323: 0.00 | 4 | UNRESOLVED |
| `tanh` | TF | 21.70 | 21.65 | 1e-30 .. 5e+255 | 100% | below 1e-323: 0.00 | 4 | UNRESOLVED |
| `tanh` | FF | 14.00 | 13.97 | 1e-30 .. 5e+255 | 100% | below 1e-323: 0.00 | 4 | UNRESOLVED |
| `asinh` | DD | 31.00 | 30.96 | 1e-323 .. 5e+255 | 100% | below 5e-324: 0.00 | 2 | UNRESOLVED |
| `asinh` | QF | 29.00 | 28.79 | 3e-29 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `asinh` | TF | 21.70 | 21.56 | 3e-29 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `asinh` | FF | 14.00 | 13.92 | 1e-30 .. 1e+30 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `acosh` | DD | 31.00 | 30.89 | 1 .. 5e+255 | 99% | -- | 0 | UNRESOLVED |
| `acosh` | QF | 29.00 | 28.73 | 1 .. 1e+30 | 96% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `acosh` | TF | 21.70 | 21.45 | 1 .. 1e+30 | 96% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `acosh` | FF | 14.00 | 13.78 | 1.001 .. 1e+30 | 96% | above 4e+151: 0.00 | 6 | UNRESOLVED |
| `atanh` | DD | 31.00 | 31.00 | 0 .. 5e+255 | 100% | -- | 0 | UNRESOLVED |
| `atanh` | QF | 29.00 | 28.80 | 1e-30 .. 1e+29 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `atanh` | TF | 21.70 | 21.56 | 1e-30 .. 1e+29 | 99% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |
| `atanh` | FF | 14.00 | 13.88 | 1.001 .. 1e+30 | 98% | below 1e-323: 0.00; above 4e+151: 0.00 | 10 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `sinh` — DD 6 pts (log sweep, |x| = 10^e); QF 736 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 736 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 742 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `cosh` — DD 6 pts (log sweep, |x| = 10^e); QF 732 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 732 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 738 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `tanh` — QF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); TF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi); FF 4 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)
- `asinh` — DD 2 pts (within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi)

---

## Complex arithmetic and construction (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `add` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `add` | QF | 29.00 | 28.92 | 1e-29 .. 1 | 100% | below 1e-30: 13.17; above 1: 4.32 | 6 | UNRESOLVED |
| `add` | TF | 21.70 | 21.65 | 1e-29 .. 1 | 100% | above 1: 4.32 | 4 | UNRESOLVED |
| `add` | FF | 14.00 | 13.27 | 1 .. 1 | 90% | below 0.99: 5.16; above 1: 6.33 | 84 | UNRESOLVED |
| `sub` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | at or below bound |
| `sub` | QF | 29.00 | 28.91 | 1e-29 .. 1 | 100% | below 1e-30: 5.52; above 1: 8.53 | 6 | UNRESOLVED |
| `sub` | TF | 21.70 | 21.64 | 1e-29 .. 1 | 100% | below 1e-30: 5.52; above 1: 8.53 | 6 | UNRESOLVED |
| `sub` | FF | 14.00 | 13.23 | 0.99 .. 1 | 91% | below 0.99: 4.48; above 1: 0.00 | 96 | UNRESOLVED |
| `mul` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `mul` | QF | 29.00 | 27.02 | 100 .. 1e+08 | 84% | below 10: 13.55 | 76 | UNRESOLVED |
| `mul` | TF | 21.70 | 20.83 | 100 .. 1e+08 | 92% | below 10: 10.62; above 1e+08: 7.52 | 63 | UNRESOLVED |
| `mul` | FF | 14.00 | 13.63 | 10 .. 1e+08 | 96% | below 10: 0.36; above 1e+08: 0.00 | 45 | UNRESOLVED |
| `div` | DD | 31.00 | 29.59 | 0 .. 1e-09 | 89% | above 0.99: 13.14 | 7 | UNRESOLVED |
| `div` | QF | 29.00 | 26.05 | 1e-14 .. 1e-10 | 77% | below 1e-19: 11.38; above 1e-08: 3.44 | 113 | UNRESOLVED |
| `div` | TF | 21.70 | 20.26 | 1e-14 .. 1e-09 | 84% | below 1e-15: 10.44; above 1e-08: 0.00 | 80 | UNRESOLVED |
| `div` | FF | 14.00 | 12.50 | 0.99 .. 1 | 89% | below 0.99: 0.00; above 1: 0.00 | 189 | UNRESOLVED |
| `abs` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `abs` | QF | 29.00 | 29.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `abs` | TF | 21.70 | 21.70 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `abs` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `conj` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `conj` | QF | 29.00 | 29.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `conj` | TF | 21.70 | 21.70 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `conj` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `polar` | DD | 31.00 | 30.74 | 0 .. 1e+04 | 98% | -- | 0 | UNRESOLVED |
| `polar` | QF | 29.00 | 27.44 | 1 .. 2 | 86% | below 1: 13.77; above 2: 13.77 | 21 | UNRESOLVED |
| `polar` | TF | 21.70 | 21.08 | 1 .. 2 | 90% | -- | 0 | UNRESOLVED |
| `polar` | FF | 14.00 | 13.75 | 0 .. 1000 | 97% | above 1e+08: 6.53 | 12 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `add` — QF 6 pts (perpendicular approach to the real axis); TF 4 pts (perpendicular approach to the real axis); FF 84 pts (perpendicular approach to the real axis)
- `sub` — QF 6 pts (perpendicular approach to the real axis); TF 6 pts (perpendicular approach to the real axis); FF 96 pts (perpendicular approach to the real axis)
- `mul` — QF 76 pts (perpendicular approach to the real axis); TF 63 pts (perpendicular approach to the real axis); FF 45 pts (perpendicular approach to the real axis)
- `div` — DD 7 pts (polar shells); QF 113 pts (perpendicular approach to the real axis); TF 80 pts (perpendicular approach to the real axis); FF 189 pts (perpendicular approach to the real axis)
- `polar` — QF 21 pts (perpendicular approach to the real axis); FF 12 pts (polar shells)

---

## Complex exponential, logarithmic, power and root (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `exp` | DD | 31.00 | 29.81 | 0 .. 100 | 96% | above 1000: 0.00 | 61 | UNRESOLVED |
| `exp` | QF | 29.00 | 24.60 | 1 .. 2 | 79% | below 1: 13.77; above 2: 13.82 | 219 | UNRESOLVED |
| `exp` | TF | 21.70 | 19.00 | 1e-27 .. 0.1 | 84% | above 100: 1.77 | 193 | UNRESOLVED |
| `exp` | FF | 14.00 | 12.40 | 0 .. 2.236 | 89% | above 100: 1.77 | 193 | UNRESOLVED |
| `log` | DD | 31.00 | 28.80 | 1 .. 1e+15 | 84% | below 0: 0.00 | 2 | UNRESOLVED |
| `log` | QF | 29.00 | 25.85 | 1e-30 .. 0.1 | 79% | below 0: 0.00; above 0.5: 13.95 | 167 | UNRESOLVED |
| `log` | TF | 21.70 | 19.46 | 1e-30 .. 0.1 | 80% | below 0: 0.00; above 1: 6.91 | 138 | UNRESOLVED |
| `log` | FF | 14.00 | 12.33 | 1.01 .. 10.05 | 78% | below 1: 0.00 | 126 | UNRESOLVED |
| `log10` | DD | 31.00 | 28.80 | 1 .. 1e+15 | 84% | below 0: 0.00 | 2 | UNRESOLVED |
| `log10` | QF | 29.00 | 25.63 | 1e-30 .. 0.1 | 77% | below 0: 0.00; above 0.5: 13.92 | 167 | UNRESOLVED |
| `log10` | TF | 21.70 | 19.42 | 1e-30 .. 0.1 | 79% | below 0: 0.00; above 1: 6.91 | 138 | UNRESOLVED |
| `log10` | FF | 14.00 | 12.34 | 1.01 .. 10.05 | 78% | below 1: 0.00 | 126 | UNRESOLVED |
| `pow` | DD | 31.00 | 30.37 | 1e-30 .. 1 | 98% | below 0: 0.00 | 1 | UNRESOLVED |
| `pow` | QF | 29.00 | 27.32 | 2 .. 2.002 | 92% | below 2: 14.11; above 2.236: 8.92 | 47 | UNRESOLVED |
| `pow` | TF | 21.70 | 20.75 | 2 .. 2.002 | 93% | below 1.1: 7.92; above 2.236: 8.92 | 35 | UNRESOLVED |
| `pow` | FF | 14.00 | 13.10 | 2 .. 2 | 79% | below 1: 6.89; above 10: 3.48 | 8 | UNRESOLVED |
| `sqrt` | DD | 31.00 | 31.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `sqrt` | QF | 29.00 | 27.74 | 1 .. 2 | 86% | -- | 0 | UNRESOLVED |
| `sqrt` | TF | 21.70 | 21.39 | 1e-25 .. 0.1 | 94% | -- | 0 | UNRESOLVED |
| `sqrt` | FF | 14.00 | 14.00 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `exp` — DD 61 pts (polar shells); QF 219 pts (perpendicular approach to the real axis); TF 193 pts (perpendicular approach to the real axis); FF 193 pts (perpendicular approach to the real axis)
- `log` — DD 2 pts (perpendicular approach to the real axis); QF 167 pts (perpendicular approach to the real axis); TF 138 pts (perpendicular approach to the real axis); FF 126 pts (perpendicular approach to the real axis)
- `log10` — DD 2 pts (perpendicular approach to the real axis); QF 167 pts (perpendicular approach to the real axis); TF 138 pts (perpendicular approach to the real axis); FF 126 pts (perpendicular approach to the real axis)
- `pow` — DD 1 pts (perpendicular approach to the real axis); QF 47 pts (perpendicular approach to the real axis); TF 35 pts (perpendicular approach to the real axis); FF 8 pts (perpendicular approach to the real axis)

---

## Complex trigonometric and hyperbolic (complex)

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `sin` | DD | 31.00 | 30.08 | 0 .. 1000 | 96% | above 1e+04: 0.00 | 30 | UNRESOLVED |
| `sin` | QF | 29.00 | 26.12 | 1 .. 2 | 75% | below 1: 13.80; above 2: 13.77 | 46 | UNRESOLVED |
| `sin` | TF | 21.70 | 20.49 | 1e-27 .. 0.1 | 87% | above 1e+04: 0.00 | 42 | UNRESOLVED |
| `sin` | FF | 14.00 | 13.31 | 0 .. 100 | 95% | above 1e+04: 0.00 | 58 | UNRESOLVED |
| `cos` | DD | 31.00 | 30.09 | 0 .. 1000 | 96% | above 1e+04: 0.00 | 30 | UNRESOLVED |
| `cos` | QF | 29.00 | 26.21 | 1 .. 2 | 76% | below 1: 13.77; above 2: 13.77 | 47 | UNRESOLVED |
| `cos` | TF | 21.70 | 20.54 | 1 .. 2 | 87% | above 1e+04: 0.00 | 42 | UNRESOLVED |
| `cos` | FF | 14.00 | 13.32 | 0 .. 100 | 95% | above 1e+04: 0.00 | 58 | UNRESOLVED |
| `tan` | DD | 31.00 | 30.51 | 0 .. 1e+04 | 97% | above 1e+04: 0.00 | 12 | UNRESOLVED |
| `tan` | QF | 29.00 | 26.00 | 1 .. 2 | 72% | below 1: 13.84; above 10: 13.31 | 56 | UNRESOLVED |
| `tan` | TF | 21.70 | 20.49 | 1e-27 .. 0.1 | 85% | above 10: 10.52 | 48 | UNRESOLVED |
| `tan` | FF | 14.00 | 13.43 | 0 .. 2.236 | 95% | above 1e+04: 0.00 | 40 | OPEN DEFECT (**4 above bound**) |
| `sinh` | DD | 31.00 | 29.43 | 0 .. 100 | 95% | above 1000: 0.00 | 83 | UNRESOLVED |
| `sinh` | QF | 29.00 | 23.70 | 1 .. 2 | 70% | below 1: 13.80; above 2: 13.77 | 233 | UNRESOLVED |
| `sinh` | TF | 21.70 | 18.60 | 1e-27 .. 0.1 | 80% | above 100: 0.00 | 215 | UNRESOLVED |
| `sinh` | FF | 14.00 | 12.22 | 0 .. 10.05 | 88% | above 100: 0.00 | 215 | UNRESOLVED |
| `cosh` | DD | 31.00 | 29.43 | 0 .. 100 | 95% | above 1000: 0.00 | 83 | UNRESOLVED |
| `cosh` | QF | 29.00 | 23.73 | 1 .. 2 | 70% | below 1: 13.77; above 2: 13.77 | 232 | UNRESOLVED |
| `cosh` | TF | 21.70 | 18.62 | 1 .. 2 | 80% | above 100: 0.00 | 215 | UNRESOLVED |
| `cosh` | FF | 14.00 | 12.22 | 0 .. 10.05 | 88% | above 100: 0.00 | 215 | UNRESOLVED |
| `tanh` | DD | 31.00 | 30.58 | 0 .. 1000 | 99% | above 1e+04: 0.00 | 12 | UNRESOLVED |
| `tanh` | QF | 29.00 | 24.47 | 1 .. 2 | 72% | below 1: 13.84; above 10: 13.31 | 178 | UNRESOLVED |
| `tanh` | TF | 21.70 | 19.27 | 1e-27 .. 0.1 | 82% | above 10: 10.52 | 156 | UNRESOLVED |
| `tanh` | FF | 14.00 | 12.71 | 0 .. 2.236 | 91% | above 100: 0.00 | 136 | UNRESOLVED |

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

| op | backend | cap | mean | trusted \|z\| | at cap | boundary (digits) | fails | verdict |
|---|---|---:|---:|---|---:|---|---:|---|
| `asin` | DD | 31.00 | 30.97 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `asin` | QF | 29.00 | 27.64 | 10 .. 1e+15 | 87% | below 10: 13.77 | 20 | UNRESOLVED |
| `asin` | TF | 21.70 | 21.30 | 10 .. 1e+15 | 94% | -- | 0 | UNRESOLVED |
| `asin` | FF | 14.00 | 13.90 | 1.005 .. 1e+15 | 98% | -- | 0 | UNRESOLVED |
| `acos` | DD | 31.00 | 30.81 | 0 .. 1e+04 | 99% | -- | 0 | UNRESOLVED |
| `acos` | QF | 29.00 | 27.86 | 1 .. 2 | 90% | below 1: 14.49; above 2: 14.12 | 14 | UNRESOLVED |
| `acos` | TF | 21.70 | 21.28 | 1e-28 .. 0.1 | 94% | -- | 0 | UNRESOLVED |
| `acos` | FF | 14.00 | 13.73 | 1.005 .. 10.05 | 96% | above 1e+08: 6.93 | 5 | UNRESOLVED |
| `atan` | DD | 31.00 | 30.99 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `atan` | QF | 29.00 | 27.80 | 1 .. 2 | 88% | below 1e-30: 13.95; above 10: 13.51 | 20 | UNRESOLVED |
| `atan` | TF | 21.70 | 21.35 | 1e-28 .. 0.1 | 95% | -- | 0 | UNRESOLVED |
| `atan` | FF | 14.00 | 13.96 | 0 .. 10.05 | 100% | -- | 0 | UNRESOLVED |
| `asinh` | DD | 31.00 | 30.96 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `asinh` | QF | 29.00 | 27.32 | 1 .. 2 | 84% | below 1: 14.10; above 2: 14.46 | 36 | UNRESOLVED |
| `asinh` | TF | 21.70 | 21.21 | 1e-27 .. 0.1 | 92% | -- | 0 | UNRESOLVED |
| `asinh` | FF | 14.00 | 13.94 | 0 .. 10.05 | 100% | -- | 0 | UNRESOLVED |
| `acosh` | DD | 31.00 | 30.89 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `acosh` | QF | 29.00 | 27.94 | 1 .. 2 | 91% | above 2: 14.12 | 12 | UNRESOLVED |
| `acosh` | TF | 21.70 | 21.40 | 1e-28 .. 0.1 | 96% | -- | 0 | UNRESOLVED |
| `acosh` | FF | 14.00 | 13.82 | 1.005 .. 10.05 | 98% | -- | 0 | UNRESOLVED |
| `atanh` | DD | 31.00 | 30.99 | 0 .. 1e+15 | 100% | -- | 0 | UNRESOLVED |
| `atanh` | QF | 29.00 | 28.12 | 1 .. 2 | 91% | below 0.5: 14.25; above 10: 13.51 | 10 | UNRESOLVED |
| `atanh` | TF | 21.70 | 21.48 | 10 .. 1e+15 | 96% | -- | 0 | UNRESOLVED |
| `atanh` | FF | 14.00 | 13.92 | 1 .. 1e+15 | 98% | -- | 0 | UNRESOLVED |

**Where the failures sit.** The grid family carrying the most failures
for each cell that has any:

- `asin` — QF 20 pts (perpendicular approach to the imaginary axis)
- `acos` — QF 14 pts (perpendicular approach to the real axis); FF 5 pts (polar shells)
- `atan` — QF 20 pts (perpendicular approach to the real axis)
- `asinh` — QF 36 pts (perpendicular approach to the real axis)
- `acosh` — QF 12 pts (perpendicular approach to the real axis)
- `atanh` — QF 10 pts (perpendicular approach to the real axis)

---

## Totals across all 252 cells

| classification | points below 50% of cap |
|---|---:|
| UNDERFLOW | 0 |
| OVERFLOW | 0 |
| ARG_RANGE | 0 |
| CONDITIONING | 0 |
| UNEXPLAINED | 0 |
| **total triaged** | **252** |

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
lands, the baseline must be regenerated first — `scripts/sweep_accuracy --ulp
--register validation/sweep/open_defects.txt --out <tmp>`, gzipped into
`validation/sweep/sweep_baseline.csv.gz` — and only then this file.
