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
- **Precision ladder (P3.0c):** the backends fill the ladder between native
  types — float 24 bits, FF ~46, double 53, TF ~72, QF ~96, DD ~103–106,
  fp128 113 on host. A fast tier is its own rung: close to strict (at most
  2 bits lost, i.e. K = 4 ulps for every backend and op), well above the next
  rung down, and cheaper. The bound rule keeps multiply/divide at their bound
  of 3.

## Definitions (all per backend, per point, scored points only)
- **Reference** = the baseline .gz files (host + a100, b200, mi250, mi300) at
  xpmath `4d597fd`, frozen by P3.0b. Never regenerated.
- **Current** = the tier's latest committed baseline (strict Current `c` for the
  strict rule; fast has no Current check).
- **Constants:** `STRICT_SLACK = 0.5`, `FAST_CAP = 4.0`, `FAST_SLACK = 1.0`.
- **Bound rule (both tiers):** `f <= b` or `r > b`, where `f` fresh ulps,
  `r` Reference ulps, `b` bound. If the Reference already exceeds its bound,
  the bound does not block the candidate.
- **Strict vs Reference:** `f <= max(1, r + 0.5)`.
- **Strict vs Current:** `f <= max(1, c + 0.5)`.
- **Fast vs Reference:** `f <= max(4, r + 1.0)`. No check against fast Current;
  the Reference cap is absolute so chained overrides cannot drift past it.
- **Speed** = CHECK-PERF ratio per op for that backend, same GPU before/after.
- **Gate script** = `python3 -I scripts/p3_tier_gate.py` (stdlib, Python 3.6+).
  It reads the candidate's FAST-tier scored sweeps (`<loc>.csv[.gz]` and
  `<loc>.exits`) plus `ratios_<arch>.csv`, and judges them against the table
  below. Exit codes: **0** every strict rule passed (accept into strict);
  **1** strict failed but every fast rule passed (accept into fast only);
  **2** reject; **3** tool error, missing data or drift (stop and ask).
  Host-only pre-screen: `python3 -I scripts/p3_tier_gate.py --host-only
  --backend DD --ops log --base <sha> --fresh <dir>` (needs only host.csv[.gz]
  + host.exits, no --perf); exit 0 = strict accuracy passes at host,
  1 = fast only, 2 = neither, 3 = stop.
- **Worker flow**: make the change in the fast tier first and score it with
  `TIER=fast`. On gate exit 0, promote the change to strict and re-run
  `TIER=both`. On exit 1 keep it fast-only. On 2 reject; on 3 stop.
- **Device binaries**: the fast device binaries (`sweep_device_fast`) are first
  run on GPUs in P3.1[dd]. A fast-device fault there is a P3.0b defect, not a
  P3.1[dd] defect: stop and report it.

| Rule | strict | fast |
|---|---|---|
| Absolute gate (scorer exit code) + register unchanged | must pass | must pass |
| Bound rule (1x, not 8x) | `f <= b` or `r > b` | `f <= b` or `r > b` |
| vs Reference | `f <= max(1, r + 0.5)` | `f <= max(4, r + 1)` |
| vs Current | `f <= max(1, c + 0.5)` | none (Reference cap is absolute) |
| Speed: changed op | >= 3% faster on >= 1 gated GPU | >= 10% faster on >= 1 gated GPU |
| Speed: every op of that backend | no more than 3% slower on any gated GPU | same |

GB300 is scored and timed, reported only, never gated.

## Gate procedure for every P3.1-P3.5[b] task (mechanical)
0. **Prerequisite:** P3.0b is done: `include/xp/tier.hpp` and
   `include/xp/{dd,ff,tf,qf}_fast.hpp` (the tier switch), the fast baselines
   `validation/sweep/sweep_baseline_fast.csv.gz` and
   `sweep_baseline_fast_{a100,b200,mi250,mi300}.csv.gz`, the frozen Reference
   `validation/sweep/reference_4d597fd/` (guarded by
   `validation/check_reference_frozen.sh`), `scripts/p3_tier_gate.py`,
   `perf/perf_ratios.py` in xpmath-kokkos, and the tier-aware
   `validation/<arch>/run_*_sweep.sh` (`TIER=strict|fast|both`). If any is
   missing: stop.
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
ulps before -> after; counts of points failing each accuracy rule (bound,
strict-Ref, strict-Cur, fast-Ref); worst ulps/bound; worst bits lost vs
Reference (log2 of the ratio); speedup per GPU (all 5, GB300 marked
reporting-only); tier accepted into.

**Pre-screen (host-only):** before requesting GPUs, run the gate in
`--host-only` mode on the host sweep alone; it applies the accuracy rules
at host only and prints `HOST-ONLY: accuracy pre-screen, no speed verdict`
on its RESULT line. Exit 0 = strict accuracy passes, 1 = fast only, 2 =
neither, 3 = stop.

**Deferred GPU (C-<arch> step 3):** run the device sweep at the row's after-SHA,
score it with the same compare script, tier and Reference the task used.
Any failing row -> stop (C-<arch> "Stop if"). Speed for a deferred GPU comes from
C-<arch> step 5 (all rows together); any op more than 3% slower -> stop and ask.

Phase 2 context: tf/qf inlining was bit-identical but NO-GO (AMD 17-30% faster,
NVIDIA 35-65% slower); parked for P3.4[tf]/[qf]. It does not affect this policy.
