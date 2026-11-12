# Perf-vs-accuracy policy for Phase 3 (P3.0)

**Decided: (c) two tiers — `strict` (default) and `fast` — 2026-11-11, by the user.**
Options considered: (a) never accept a monotone regression; (b) accept anything
the absolute gate passes, then re-baseline; (c) strict default + opt-in fast
tier; (d) (b) with caps. (c) was chosen; the fast tier uses (d)'s caps.

## Facts this rests on
- CORRECTNESS.md §3 says no point above its bound, but the absolute gate fails
  only above **8x** bound (`scripts/sweep_accuracy.cpp:2097` `kUlpAllowance = 8.0`,
  tested at :3196 real, :3402 complex). The host baseline already has 109 scored
  rows above 1x (e.g. FF r div 3.73 vs 3; QF r mul 9.0e12 vs 7.8e12). **This
  doc/code gap should be fixed in CORRECTNESS.md separately; P3 does not touch it.**
- Headroom (`validation/sweep/sweep_baseline.csv.gz`): DD real exp/log/sin/cos
  median ulps 0.12-0.33 vs median bound 90-8e15; mul/div median bound 3, median
  ulps 0-0.25. So the bound limits mul/div, the bit cap limits transcendentals.
- Unverified (from memory): the JMP FMA multiply (P3.4) may reach ~4-5 ulps,
  above the mul bound of 3. If so it fails both tiers; P3.4 must measure it.

## Definitions (all per backend, per point, scored points only)
- **Reference** = the baseline .gz files (host + a100, b200, mi250, mi300) at
  xpmath `4d597fd`, frozen by P3.0b. Never regenerated.
- **Current** = the tier's latest committed baseline.
- **Worse(x, y)** = `x > y * 1.2589254` and `x > 1 ulp` (the monotone rule).
- **Bound rule**: fresh ulps <= 1x bound, or (reference ulps > 1x bound and not
  Worse(fresh, reference)).
- **Speed** = CHECK-PERF ratio per op for that backend, same GPU before/after.

| Rule | strict | fast |
|---|---|---|
| Absolute gate (scorer exit code) + register unchanged | must pass | must pass |
| Bound rule (1x, not 8x) | must pass | must pass |
| vs Current | not Worse | not Worse vs fast Current |
| vs Reference | not Worse | fresh <= 4x reference (2 bits) or fresh <= 1 ulp |
| Speed: changed op | >= 3% faster on >= 1 gated GPU | >= 10% faster on >= 1 gated GPU |
| Speed: every op of that backend | no more than 3% slower on any gated GPU | same |

GB300 is scored and timed, reported only, never gated.

## Gate procedure for every P3.1-P3.5[b] task (mechanical)
0. **Prerequisite:** P3.0b (switch, fast baselines, frozen reference, the
   compare script, xpmath-kokkos two-tier bit-identity tests) is done. If not: stop.
1. Never edit the bound, its derivation, `kUlpAllowance`, the register, or the
   Reference. If the change seems to need it: **stop and ask the user**.
2. Absolute gate fails or the register changes on host or any GPU -> **reject**.
3. All strict rows pass on host + 4 gated GPUs -> **accept into strict**
   (the default path; fast inherits it). Re-baseline strict and fast. If fast
   already has its own version of this function for this backend: stop and ask.
4. Else, all fast rows pass -> **accept into fast only**; strict keeps the old
   code for that backend. Re-baseline fast.
5. Else -> **reject**. Record the numbers in the status block; keep the branch.
6. Tool error, missing data, or a case not covered above -> **stop and ask**.

Per backend: each [b] is judged alone; a pass for dd says nothing about ff.

**Re-baseline** (CORRECTNESS.md §5), its own commit, after the code commit:
the tier's host .gz, 4 device .gz, `docs/DOMAINS.md`, `docs/DEVICE_PRECISION.md`.
Subject: `P3.<n>[<b>]: re-baseline <strict|fast> after <function>`. Body:
the numbers below.

**Report** (status block and re-baseline body), per op touched: max/median/p99
ulps before -> after; points Worse vs Current and vs Reference; worst
ulps/bound; worst bits lost vs Reference (log2 of the ratio); speedup per GPU
(all 5, GB300 marked reporting-only); tier accepted into.

**Deferred GPU (C-<arch> step 3):** run the device sweep at the row's after-SHA,
score it with the same compare script, tier and Reference the task used.
Any failing row -> stop (C-<arch> "Stop if"). Speed for a deferred GPU comes from
C-<arch> step 5 (all rows together); any op more than 3% slower -> stop and ask.

Phase 2 context: tf/qf inlining was bit-identical but NO-GO (AMD 17-30% faster,
NVIDIA 35-65% slower); parked for P3.4[tf]/[qf]. It does not affect this policy.
