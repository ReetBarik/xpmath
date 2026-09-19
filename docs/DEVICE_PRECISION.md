# Device precision — host vs A100 vs MI250X

**GENERATED FILE — do not edit.** Produced by
`scripts/gen_domains.py --device-precision` from the three committed sweep
baselines and `validation/sweep/sweep_grid.csv`. Regenerate with:

```bash
scripts/gen_domains.py --device-precision > docs/DEVICE_PRECISION.md
```

`validation/check_device_domains_fresh.sh` fails if this file is not what the
generator currently emits. Registered as the `device_domains_fresh` ctest
target.

One measurement, one scorer: every ulp here is the host `sweep_accuracy`
MPFR/MPC oracle applied to raw limbs produced on each arch. There is no
device-side ulp computation. See `docs/CORRECTNESS.md`.

**Coverage is complete for the two device arches this arc measures.**
Both A100 and MI250X baselines are present; every table below has a
measured column for each.

## Arch coverage

| arch | baseline | status |
|---|---|---|
| host | `validation/sweep/sweep_baseline.csv.gz` | PRESENT — 436080 rows, where=`host` |
| a100 | `sweep_baseline_a100.csv.gz` | PRESENT — 436080 rows, where=`a100`, Cobalt 1001685 (docs/CORE_PLAN_STATUS.md §C7) |
| mi250 | `sweep_baseline_mi250.csv.gz` | PRESENT — 436080 rows, where=`mi250`, Cobalt 1001915 (docs/CORE_PLAN_STATUS.md §C8) |

Absolute gate (`ulps ≤ 8 × derived bound`), scored points above it:

| arch | above bound |
|---|---:|
| host | 0 |
| a100 | 0 |
| mi250 | 0 |

`validation/sweep/open_defects.txt` carries a point only when it exceeds
that bound. A zero in every present column means the register stays empty.

## Host vs device — global deltas

Same identity key, same 0.1-digit noise floor the monotone gate uses
(`10^0.1`). "Worse" / "better" count only points that move *beyond*
that floor; "differ at all" is any ulps/digits/state change.

| arch | identical | differ at all | worse beyond noise | better beyond noise | state moved | above bound |
|---|---:|---:|---:|---:|---:|---:|
| a100 | 428956 | 7124 | 1221 | 1130 | 3507 | 0 |
| mi250 | 423435 | 12645 | 4784 | 3941 | 0 | 0 |

A100 state transitions (host → a100):

- `S → N`: 3507

MI250: **0** state transitions. Every point keeps the host verdict
letter; the 12k+ ulp moves below stay inside `S`/`U`/`N` as scored
on the host.

---

## The four mechanisms, measured

The repository already knew these in pieces. The numbers below are from
the three baselines, not from reasoning about flags.

### 1. FTZ / DAZ on the trailing limbs

The "full-precision floor" row of `docs/DOMAINS.md` (DD 2.0e-292, QF
5.6e-17, TF 3.3e-24, FF 2.0e-31) is the magnitude below which trailing
limbs go subnormal. A GPU flushing subnormals would raise that floor.
Measured floor here is the smallest |x| at which real `abs` still reaches
90% of cap on every grid point at that magnitude — the same definition
`docs/DOMAINS.md` uses for its trusted band.

| backend | format floor | host measured | a100 measured | mi250 measured |
|---|---|---|---|---|
| DD | 2.0e-292 | 5e-324 | 5e-324 | 5e-324 |
| QF | 5.6e-17 | 1e-30 | 1e-30 | 1e-30 |
| TF | 3.3e-24 | 1e-30 | 1e-30 | 1e-30 |
| FF | 2.0e-31 | 1e-30 | 1e-30 | 1e-30 |

On every present arch the measured `abs` floor matches the host. FTZ did
**not** raise the trusted band for `abs`. The FTZ signal that *does*
appear is scorability, not the floor:

- A100: **3507** points move `S → N` (host scored them; A100 marks them
  unscorable). They concentrate in DD complex `atan` and `atanh` — see
  the C7 STATUS block. That is why A100 has its own baseline rather than
  sharing the host monotone gate.
- MI250: **0** `S → N` moves.

### 2. FMA contraction

Every arch builds with `-ffp-contract=off` / `--fmad=false`, so Dekker
TwoSum / TwoProd sequences must not collapse into an FMA. Confirmation is
from the results, not from the flags: count of bit-identical points in the
arithmetic / selection / rounding family
(`add sub mul div fma abs copysign fmax fmin fdim hypot ceil floor round
trunc fmod remainder conj polar`), host vs each device.

| arch | bit-identical arithmetic points | of |
|---|---:|---:|
| a100 | 165440 | 165440 |
| mi250 | 165440 | 165440 |

A count below the total would be an FMA regression. Both present arches
are exact.

### 3. Vendor libm
`xp::detail::` scalar dispatch resolves `sin` / `exp` / `sqrt` / … to a
different implementation on each device. Every host↔device ulp move that
is not FTZ and not arithmetic lands here.

| arch | libm-family points that differ | of | worse beyond noise | better beyond noise |
|---|---:|---:|---:|---:|
| a100 | 7124 | 270640 | 1221 | 1130 |
| mi250 | 12645 | 270640 | 4784 | 3941 |

The per-op table below names the cells. Inverse trig, inverse hyperbolic,
and `log` / `pow` dominate; pure arithmetic does not appear.

### 4. The gfx90a mitigations

`XPMATH_NOINLINE_FUNCTION` (TD-1 / BranchRelaxation) and the de-recursed
device self-calls (TD-2 / dynamic stack) exist to change codegen on
MI250X. Their success is not an ulp delta — it is that the producer
finished:

- MI250 producer: `last_error() == 0` over all 436,080 points
  (Cobalt 1001915).
- Absolute gate on the scored baseline: **0** above bound.
- Arithmetic family: bit-identical to host (table above) — the
  volatile EFT wrappers under `__HIP_DEVICE_COMPILE__` did not
  perturb TwoSum/TwoProd.
- `scripts/xpm_lint_device_asm.sh` was **not** re-run on the C8 binary
  (no `--save-temps` objects kept); completion is not a substitute for
  tiers 1–4. Recorded in the C8 STATUS block.

---

## Per-op mean ulps — host / a100 / mi250

Mean ulps over **scored** (`state=S`) points only — `N` carries the
sentinel `ulps=-1` and must not enter the average. The identical-point
counts cover every grid point and are the sharper signal. `cause` is one
label for the cell across both device arches; see the class list at the
top of `scripts/gen_domains.py`.

### Arithmetic and selection (real)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `add` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `add` | 0.003296 | 0.003296 | 0.003296 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `add` | 0.01779 | 0.01779 | 0.01779 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `add` | 5.911e+04 | 5.911e+04 | 5.911e+04 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `sub` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `sub` | 0.002893 | 0.002893 | 0.002893 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `sub` | 0.0176 | 0.0176 | 0.0176 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `sub` | 1.296e+05 | 1.296e+05 | 1.296e+05 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `mul` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `mul` | 4.661e+24 | 4.661e+24 | 4.661e+24 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `mul` | 2.778e+17 | 2.778e+17 | 2.778e+17 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `mul` | 1.656e+10 | 1.656e+10 | 1.656e+10 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `div` | 0.127 | 0.127 | 0.127 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `div` | 9.526e+18 | 9.526e+18 | 9.526e+18 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `div` | 5.525e+11 | 5.525e+11 | 5.525e+11 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `div` | 3.293e+04 | 3.293e+04 | 3.293e+04 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `fma` | 0.04192 | 0.04192 | 0.04192 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `fma` | 4.480e+23 | 4.480e+23 | 4.480e+23 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `fma` | 2.670e+16 | 2.670e+16 | 2.670e+16 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `fma` | 1.652e+09 | 1.652e+09 | 1.652e+09 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `abs` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `abs` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `abs` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `abs` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `copysign` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `copysign` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `copysign` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `copysign` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `fmax` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `fmax` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `fmax` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `fmax` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `fmin` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `fmin` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `fmin` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `fmin` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `fdim` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `fdim` | 0.003109 | 0.003109 | 0.003109 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `fdim` | 0.01728 | 0.01728 | 0.01728 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `fdim` | 6.821e+04 | 6.821e+04 | 6.821e+04 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `hypot` | 0.08771 | 0.08771 | 0.08771 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `hypot` | 52.37 | 52.37 | 52.37 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `hypot` | 0.1199 | 0.1199 | 0.1199 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `hypot` | 0.1792 | 0.1792 | 0.1792 | 0 | 0 | 1700/1700 | 1700/1700 | match |

### Rounding and remainder (real)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `ceil` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `ceil` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `ceil` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `ceil` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `floor` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `floor` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `floor` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `floor` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `round` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `round` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `round` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `round` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `trunc` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `trunc` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `trunc` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `trunc` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `fmod` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `fmod` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `fmod` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `fmod` | 0.002168 | 0.002168 | 0.002168 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `remainder` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `remainder` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `remainder` | 0 | 0 | 0 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `remainder` | 0.001475 | 0.001475 | 0.001475 | 0 | 0 | 1700/1700 | 1700/1700 | match |

### Exponential, logarithmic and power (real)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `exp` | 0.2579 | 0.2579 | 0.2579 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `exp` | 3.492e+23 | 3.492e+23 | 3.492e+23 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `exp` | 2.082e+16 | 2.082e+16 | 2.082e+16 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `exp` | 6.980e+09 | 6.980e+09 | 6.980e+09 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `exp2` | 0.2358 | 0.2358 | 0.2358 | 0 | 0 | 1645/1700 | 1645/1700 | VENDOR_LIBM |
| QF | `exp2` | 3.246e+23 | 3.246e+23 | 3.246e+23 | 0 | 0 | 1648/1700 | 1648/1700 | VENDOR_LIBM |
| TF | `exp2` | 1.935e+16 | 1.935e+16 | 1.935e+16 | 0 | 0 | 1648/1700 | 1648/1700 | VENDOR_LIBM |
| FF | `exp2` | 2.829e+10 | 2.829e+10 | 2.829e+10 | 0 | 0 | 1648/1700 | 1648/1700 | VENDOR_LIBM |
| DD | `exp10` | 1.555e+21 | 1.555e+21 | 1.555e+21 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `exp10` | 1.643e+22 | 1.643e+22 | 1.643e+22 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `exp10` | 9.795e+14 | 9.795e+14 | 9.795e+14 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `exp10` | 5.838e+07 | 5.838e+07 | 5.838e+07 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `expm1` | 0.1705 | 0.1705 | 0.1705 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `expm1` | 3913 | 3913 | 3913 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `expm1` | 0.03553 | 0.03553 | 0.03553 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `expm1` | 0.126 | 0.126 | 0.126 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `log` | 7.092e+12 | 7.092e+12 | 7.092e+12 | 0.000977 | 0.001953 | 1696/1700 | 1698/1700 | VENDOR_LIBM |
| QF | `log` | 2379 | 1635 | 1634 | -744 | -744 | 1652/1700 | 1466/1700 | VENDOR_LIBM |
| TF | `log` | 2601 | 2626 | 3332 | 25.38 | 731.3 | 1672/1700 | 1588/1700 | VENDOR_LIBM |
| FF | `log` | 5448 | 5448 | 5444 | 0.008156 | -4.52 | 1670/1700 | 1504/1700 | VENDOR_LIBM |
| DD | `log2` | 7.092e+12 | 7.092e+12 | 7.092e+12 | 0 | 0 | 1698/1700 | 1698/1700 | VENDOR_LIBM |
| QF | `log2` | 2379 | 1635 | 1634 | -744 | -744 | 1674/1700 | 1568/1700 | VENDOR_LIBM |
| TF | `log2` | 2601 | 2626 | 3332 | 25.38 | 731.3 | 1694/1700 | 1652/1700 | VENDOR_LIBM |
| FF | `log2` | 5449 | 5449 | 5444 | 0.0077 | -4.52 | 1678/1700 | 1606/1700 | VENDOR_LIBM |
| DD | `log10` | 7.092e+12 | 7.092e+12 | 7.092e+12 | 0 | 0.001953 | 1696/1700 | 1698/1700 | VENDOR_LIBM |
| QF | `log10` | 2379 | 1635 | 1634 | -744 | -744 | 1682/1700 | 1534/1700 | VENDOR_LIBM |
| TF | `log10` | 2601 | 2626 | 3332 | 25.38 | 731.3 | 1696/1700 | 1640/1700 | VENDOR_LIBM |
| FF | `log10` | 5449 | 5449 | 5444 | 0.008805 | -4.52 | 1686/1700 | 1572/1700 | VENDOR_LIBM |
| DD | `log1p` | 0.341 | 0.3406 | 0.3413 | -0.000468 | 0.000218 | 1699/1700 | 1697/1700 | VENDOR_LIBM |
| QF | `log1p` | 4552 | 4552 | 4552 | 2.45e-05 | -0.000932 | 1680/1700 | 1547/1700 | VENDOR_LIBM |
| TF | `log1p` | 0.1303 | 0.1303 | 0.1304 | 1.18e-06 | 0.000189 | 1686/1700 | 1631/1700 | VENDOR_LIBM |
| FF | `log1p` | 0.3118 | 0.3157 | 0.3119 | 0.003816 | 6.31e-05 | 1686/1700 | 1631/1700 | VENDOR_LIBM |
| DD | `pow` | 1.16 | 1.155 | 1.155 | -0.00558 | -0.00496 | 1695/1700 | 1693/1700 | VENDOR_LIBM |
| QF | `pow` | 1.251e+16 | 1.251e+16 | 1.251e+16 | 0 | 0 | 1687/1700 | 1647/1700 | VENDOR_LIBM |
| TF | `pow` | 7.458e+08 | 7.458e+08 | 7.458e+08 | 0.005837 | 0.005034 | 1666/1700 | 1596/1700 | VENDOR_LIBM |
| FF | `pow` | 46.53 | 46.54 | 46.55 | 0.01155 | 0.02168 | 1645/1700 | 1598/1700 | VENDOR_LIBM |
| DD | `sqrt` | 6.532e+12 | 6.532e+12 | 6.532e+12 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `sqrt` | 0.07145 | 0.07145 | 0.07145 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `sqrt` | 0.1271 | 0.1271 | 0.1271 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `sqrt` | 0.1674 | 0.1674 | 0.1674 | 0 | 0 | 1700/1700 | 1700/1700 | match |

### Trigonometric and inverse trigonometric (real)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `sin` | 0.4503 | 0.4503 | 0.4503 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `sin` | 0.2208 | 0.2208 | 0.2208 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `sin` | 0.06619 | 0.06619 | 0.06619 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `sin` | 0.3272 | 0.3272 | 0.3272 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `cos` | 0.1701 | 0.1701 | 0.1701 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `cos` | 0.03536 | 0.03536 | 0.03536 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `cos` | 0.03566 | 0.03566 | 0.03566 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `cos` | 0.1506 | 0.1506 | 0.1506 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `tan` | 0.6439 | 0.6439 | 0.6439 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `tan` | 0.2761 | 0.2761 | 0.2761 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `tan` | 0.129 | 0.129 | 0.129 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `tan` | 0.8538 | 0.8538 | 0.8538 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `asin` | 0.7027 | 0.686 | 0.685 | -0.0166 | -0.0177 | 1426/1700 | 1418/1700 | VENDOR_LIBM |
| QF | `asin` | 0.4193 | 0.4201 | 0.4216 | 0.000767 | 0.002318 | 1684/1700 | 1460/1700 | VENDOR_LIBM |
| TF | `asin` | 0.2917 | 0.2919 | 0.2969 | 0.000224 | 0.005194 | 1688/1700 | 1468/1700 | VENDOR_LIBM |
| FF | `asin` | 0.8981 | 0.8982 | 0.8999 | 0.000122 | 0.001788 | 1688/1700 | 1484/1700 | VENDOR_LIBM |
| DD | `acos` | 0.1552 | 0.1529 | 0.1538 | -0.00228 | -0.00138 | 1680/1700 | 1679/1700 | VENDOR_LIBM |
| QF | `acos` | 876.1 | 876.1 | 876.1 | -0.000285 | 0.000341 | 1601/1700 | 1591/1700 | VENDOR_LIBM |
| TF | `acos` | 626 | 626 | 626.1 | 0.000479 | 0.000644 | 1637/1700 | 1626/1700 | VENDOR_LIBM |
| FF | `acos` | 1498 | 1498 | 1498 | 0.001073 | -0.000527 | 1679/1700 | 1666/1700 | VENDOR_LIBM |
| DD | `atan` | 0.2258 | 0.227 | 0.2295 | 0.001189 | 0.003697 | 1658/1700 | 1652/1700 | VENDOR_LIBM |
| QF | `atan` | 0.04278 | 0.04262 | 0.04088 | -0.000162 | -0.0019 | 1604/1700 | 1538/1700 | VENDOR_LIBM |
| TF | `atan` | 0.07288 | 0.07379 | 0.07365 | 0.000903 | 0.000771 | 1628/1700 | 1618/1700 | VENDOR_LIBM |
| FF | `atan` | 0.1972 | 0.1887 | 0.1909 | -0.00844 | -0.00624 | 1656/1700 | 1600/1700 | VENDOR_LIBM |

### Hyperbolic and inverse hyperbolic (real)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `sinh` | 0.3924 | 0.3924 | 0.3924 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `sinh` | 0.226 | 0.226 | 0.226 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `sinh` | 0.4062 | 0.4062 | 0.4062 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `sinh` | 0.3873 | 0.3873 | 0.3873 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `cosh` | 0.3766 | 0.3766 | 0.3766 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `cosh` | 0.2192 | 0.2192 | 0.2192 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `cosh` | 0.3945 | 0.3945 | 0.3945 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `cosh` | 0.3787 | 0.3787 | 0.3787 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `tanh` | 0.1197 | 0.1197 | 0.1197 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `tanh` | 0.03503 | 0.03503 | 0.03503 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| TF | `tanh` | 0.0797 | 0.0797 | 0.0797 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| FF | `tanh` | 0.134 | 0.134 | 0.134 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| DD | `asinh` | 0.1653 | 0.165 | 0.1653 | -0.000349 | 0 | 1698/1700 | 1700/1700 | VENDOR_LIBM |
| QF | `asinh` | 5.475e+10 | 5.475e+10 | 5.475e+10 | 7.63e-05 | 0.000557 | 1682/1700 | 1468/1700 | VENDOR_LIBM |
| TF | `asinh` | 3264 | 3264 | 3264 | -4.69e-05 | -0.000349 | 1694/1700 | 1624/1700 | VENDOR_LIBM |
| FF | `asinh` | 0.1802 | 0.1799 | 0.1812 | -0.000301 | 0.000941 | 1696/1700 | 1570/1700 | VENDOR_LIBM |
| DD | `acosh` | 7.8e+04 | 7.8e+04 | 7.8e+04 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `acosh` | 2.115e+04 | 2.115e+04 | 1.712e+04 | -3.64e-06 | -4.03e+03 | 1671/1700 | 1387/1700 | VENDOR_LIBM |
| TF | `acosh` | 2.331e+04 | 2.331e+04 | 2.113e+04 | -0.000241 | -2.18e+03 | 1693/1700 | 1559/1700 | VENDOR_LIBM |
| FF | `acosh` | 2.338e+04 | 2.338e+04 | 2.338e+04 | 0.003733 | -3.05 | 1688/1700 | 1568/1700 | VENDOR_LIBM |
| DD | `atanh` | 0.4637 | 0.4637 | 0.4637 | 0 | 0 | 1700/1700 | 1700/1700 | match |
| QF | `atanh` | 0.06616 | 0.06626 | 0.06573 | 0.000108 | -0.00043 | 1697/1700 | 1674/1700 | VENDOR_LIBM |
| TF | `atanh` | 0.09534 | 0.09535 | 0.0954 | 1.68e-05 | 6.28e-05 | 1699/1700 | 1689/1700 | VENDOR_LIBM |
| FF | `atanh` | 0.3363 | 0.3363 | 0.336 | 0 | -0.000319 | 1700/1700 | 1685/1700 | VENDOR_LIBM |

### Complex arithmetic and construction (complex)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `add` | 0 | 0 | 0 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `add` | 3.012e+12 | 3.012e+12 | 3.012e+12 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `add` | 1.796e+05 | 1.796e+05 | 1.796e+05 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `add` | 1.835e+10 | 1.835e+10 | 1.835e+10 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `sub` | 0 | 0 | 0 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `sub` | 1.344e+20 | 1.344e+20 | 1.344e+20 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `sub` | 8.011e+12 | 8.011e+12 | 8.011e+12 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `sub` | 1.206e+10 | 1.206e+10 | 1.206e+10 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `mul` | 0.03324 | 0.03324 | 0.03324 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `mul` | 1.918e+24 | 1.918e+24 | 1.918e+24 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `mul` | 1.143e+17 | 1.143e+17 | 1.143e+17 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `mul` | 6.815e+09 | 6.815e+09 | 6.815e+09 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `div` | 0.3051 | 0.3051 | 0.3051 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `div` | 4.119e+08 | 4.119e+08 | 4.119e+08 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `div` | 0.08634 | 0.08634 | 0.08634 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `div` | 0.3179 | 0.3179 | 0.3179 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `abs` | 0.06581 | 0.06581 | 0.06581 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `abs` | 0.03925 | 0.03925 | 0.03925 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `abs` | 0.04281 | 0.04281 | 0.04281 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `abs` | 0.08212 | 0.08212 | 0.08212 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `conj` | 0 | 0 | 0 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `conj` | 0 | 0 | 0 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `conj` | 0 | 0 | 0 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `conj` | 0.0368 | 0.0368 | 0.0368 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `polar` | 0.2527 | 0.2527 | 0.2527 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `polar` | 6.798e+11 | 6.798e+11 | 6.798e+11 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `polar` | 4.052e+04 | 4.052e+04 | 4.052e+04 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `polar` | 4.925e+04 | 4.925e+04 | 4.925e+04 | 0 | 0 | 1780/1780 | 1780/1780 | match |

### Complex exponential, logarithmic, power and root (complex)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `exp` | 0.3884 | 0.3884 | 0.3884 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `exp` | 0.09597 | 0.09597 | 0.09597 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `exp` | 0.09247 | 0.09247 | 0.09247 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `exp` | 2.081e+11 | 2.081e+11 | 2.081e+11 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `log` | 6.263e+04 | 6.263e+04 | 6.263e+04 | 0.002246 | 0.004071 | 1725/1780 | 1733/1780 | VENDOR_LIBM |
| QF | `log` | 6488 | 5776 | 5491 | -712 | -998 | 1720/1780 | 1617/1780 | VENDOR_LIBM |
| TF | `log` | 3341 | 3365 | 4042 | 24.32 | 700.9 | 1738/1780 | 1624/1780 | VENDOR_LIBM |
| FF | `log` | 1.198e+04 | 1.198e+04 | 1.198e+04 | -0.00458 | -3.98 | 1743/1780 | 1626/1780 | VENDOR_LIBM |
| DD | `log10` | 6.263e+04 | 6.263e+04 | 6.263e+04 | 0.000872 | 0.001217 | 1755/1780 | 1757/1780 | VENDOR_LIBM |
| QF | `log10` | 6381 | 5668 | 5383 | -712 | -998 | 1722/1780 | 1621/1780 | VENDOR_LIBM |
| TF | `log10` | 3341 | 3365 | 4042 | 24.32 | 700.9 | 1740/1780 | 1634/1780 | VENDOR_LIBM |
| FF | `log10` | 1.198e+04 | 1.198e+04 | 1.198e+04 | -0.00378 | -3.98 | 1741/1780 | 1622/1780 | VENDOR_LIBM |
| DD | `pow` | 7.809 | 7.814 | 7.834 | 0.005153 | 0.02552 | 1762/1780 | 1767/1780 | VENDOR_LIBM |
| QF | `pow` | 4.900e+22 | 4.900e+22 | 4.900e+22 | 0 | 0 | 1759/1780 | 1697/1780 | VENDOR_LIBM |
| TF | `pow` | 2.920e+15 | 2.920e+15 | 2.920e+15 | 0 | 0 | 1759/1780 | 1690/1780 | VENDOR_LIBM |
| FF | `pow` | 1.741e+08 | 1.741e+08 | 1.741e+08 | -0.0217 | 0.01515 | 1759/1780 | 1678/1780 | VENDOR_LIBM |
| DD | `sqrt` | 0.2442 | 0.2442 | 0.2442 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `sqrt` | 6.244e+09 | 6.244e+09 | 6.244e+09 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `sqrt` | 372.3 | 372.3 | 372.3 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `sqrt` | 0.1887 | 0.1887 | 0.1887 | 0 | 0 | 1780/1780 | 1780/1780 | match |

### Complex trigonometric and hyperbolic (complex)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `sin` | 0.6148 | 0.6148 | 0.6148 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `sin` | 0.1027 | 0.1027 | 0.1027 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `sin` | 0.1552 | 0.1552 | 0.1552 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `sin` | 0.5196 | 0.5196 | 0.5196 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `cos` | 0.4854 | 0.4854 | 0.4854 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `cos` | 0.09112 | 0.09112 | 0.09112 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `cos` | 0.1505 | 0.1505 | 0.1505 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `cos` | 0.5963 | 0.5963 | 0.5963 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `tan` | 1.029 | 1.029 | 1.029 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `tan` | 0.2897 | 0.2897 | 0.2897 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `tan` | 0.1855 | 0.1855 | 0.1855 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `tan` | 0.9148 | 0.9148 | 0.9148 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `sinh` | 0.5473 | 0.5473 | 0.5473 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `sinh` | 0.1096 | 0.1096 | 0.1096 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `sinh` | 0.1426 | 0.1426 | 0.1426 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `sinh` | 0.4206 | 0.4206 | 0.4206 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `cosh` | 0.4847 | 0.4847 | 0.4847 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `cosh` | 0.0991 | 0.0991 | 0.0991 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `cosh` | 0.1562 | 0.1562 | 0.1562 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `cosh` | 0.5915 | 0.5915 | 0.5915 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| DD | `tanh` | 0.5753 | 0.5753 | 0.5753 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| QF | `tanh` | 0.1994 | 0.1994 | 0.1994 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| TF | `tanh` | 0.2355 | 0.2355 | 0.2355 | 0 | 0 | 1780/1780 | 1780/1780 | match |
| FF | `tanh` | 0.6169 | 0.6169 | 0.6169 | 0 | 0 | 1780/1780 | 1780/1780 | match |

### Complex inverse functions (complex)

| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DD | `asin` | 0.4736 | 0.4747 | 0.4742 | 0.001144 | 0.000636 | 1712/1780 | 1711/1780 | VENDOR_LIBM |
| QF | `asin` | 283.7 | 283.7 | 283.7 | -0.00124 | -0.00805 | 1717/1780 | 1354/1780 | VENDOR_LIBM |
| TF | `asin` | 0.1415 | 0.1413 | 0.1444 | -0.00021 | 0.002897 | 1737/1780 | 1425/1780 | VENDOR_LIBM |
| FF | `asin` | 2.478 | 2.476 | 2.508 | -0.00227 | 0.02993 | 1729/1780 | 1402/1780 | VENDOR_LIBM |
| DD | `acos` | 0.3215 | 0.3219 | 0.3223 | 0.000366 | 0.000726 | 1733/1780 | 1723/1780 | VENDOR_LIBM |
| QF | `acos` | 0.1107 | 0.1102 | 0.1069 | -0.000537 | -0.00379 | 1710/1780 | 1370/1780 | VENDOR_LIBM |
| TF | `acos` | 0.1311 | 0.1316 | 0.1319 | 0.00045 | 0.000721 | 1720/1780 | 1437/1780 | VENDOR_LIBM |
| FF | `acos` | 4265 | 4265 | 4265 | 0.003153 | 0.01749 | 1741/1780 | 1431/1780 | VENDOR_LIBM |
| DD | `atan` | 0.2581 | 0 | 0.262 | -0.258 | 0.003955 | 6/1780 | 1728/1780 | FTZ |
| QF | `atan` | 0.0631 | 0.06473 | 0.06438 | 0.001632 | 0.001286 | 1598/1780 | 1536/1780 | VENDOR_LIBM |
| TF | `atan` | 0.09385 | 0.09026 | 0.09023 | -0.00359 | -0.00362 | 1626/1780 | 1587/1780 | VENDOR_LIBM |
| FF | `atan` | 0.2847 | 0.301 | 0.3009 | 0.01631 | 0.0162 | 1652/1780 | 1593/1780 | VENDOR_LIBM |
| DD | `asinh` | 0.5912 | 0.5914 | 0.5884 | 0.000192 | -0.00281 | 1718/1780 | 1704/1780 | VENDOR_LIBM |
| QF | `asinh` | 567.3 | 567.3 | 567.3 | -0.000486 | -0.00737 | 1696/1780 | 1312/1780 | VENDOR_LIBM |
| TF | `asinh` | 0.1771 | 0.1758 | 0.1783 | -0.00121 | 0.001255 | 1720/1780 | 1349/1780 | VENDOR_LIBM |
| FF | `asinh` | 0.4613 | 0.4618 | 0.4943 | 0.000554 | 0.03304 | 1716/1780 | 1332/1780 | VENDOR_LIBM |
| DD | `acosh` | 0.6782 | 0.6759 | 0.6778 | -0.00227 | -0.00039 | 1624/1780 | 1625/1780 | VENDOR_LIBM |
| QF | `acosh` | 0.2048 | 0.2055 | 0.2017 | 0.000763 | -0.00304 | 1672/1780 | 1043/1780 | VENDOR_LIBM |
| TF | `acosh` | 0.2211 | 0.2201 | 0.2219 | -0.00102 | 0.0008 | 1685/1780 | 1309/1780 | VENDOR_LIBM |
| FF | `acosh` | 4266 | 4266 | 4266 | 0.00647 | 0.01184 | 1689/1780 | 1214/1780 | VENDOR_LIBM |
| DD | `atanh` | 0.2924 | 1.687 | 0.2964 | 1.394 | 0.004029 | 35/1780 | 1751/1780 | FTZ |
| QF | `atanh` | 0.06688 | 0.06707 | 0.06687 | 0.000196 | -6.98e-06 | 1729/1780 | 1658/1780 | VENDOR_LIBM |
| TF | `atanh` | 0.09253 | 0.09284 | 0.09283 | 0.000302 | 0.000299 | 1742/1780 | 1699/1780 | VENDOR_LIBM |
| FF | `atanh` | 943.6 | 943.6 | 943.6 | 0.001454 | 8.74e-05 | 1752/1780 | 1699/1780 | VENDOR_LIBM |

## Cause totals across 252 cells

| cause | cells |
|---|---:|
| match | 170 |
| FMA | 0 |
| FTZ | 2 |
| VENDOR_LIBM | 80 |
| UNATTRIBUTED | 0 |
| NO BASELINE | 0 |
| **total** | **252** |

**0 cells are UNATTRIBUTED.** Every differing cell landed in FTZ or
VENDOR_LIBM; every arithmetic cell matched.

---

## Freshness

`validation/check_device_domains_fresh.sh` regenerates this file and diffs
it against the committed copy. Re-score a device arch only through the
recipes in `docs/CORRECTNESS.md` and the C7/C8 STATUS blocks; then rerun
the generator.
