# Performance measurement plan — kokkos-extended-precision-demo

**PARKED.** This arc is paused pending completion of `docs/UPSTREAM_PLAN.md`. The standalone library extraction will restructure the repository layout (namespace moves, header reorganization, build system changes), invalidating the assumptions in this plan. Phase 1 must not be started until the upstream arc completes and this plan is updated to reflect the new structure.

---

This document is the authoritative spec for the perf-measurement arc
that produces the README performance tables. Each phase depends on
the previous one landing on `main`. A fresh Claude session should be
able to pick up any single phase by reading this document and the
phase's own section — no other context required.

## Goal

Publish, in the top-level `README.md`, three tables (one per backend
— DD, QF, FF) that quantify the runtime cost of using each backend
against native FP64 on the two GPUs of interest.

Table shape (per backend):

```
op       GB200         MI300X
add      <slowdown>x   <slowdown>x
mul      ...
div      ...
sqrt     ...
fma      ...
exp      ...
log      ...
sin      ...
atan     ...
```

Cell contents: **slowdown factor vs FP64 on the same GPU**,
dimensionless (e.g. `9.2x`). Higher means the backend is more
expensive relative to native FP64.

No preamble hand-holding. No FF caveats. Readers cross-reference the
existing accuracy tables and draw their own conclusions.

Ops list is fixed: **add, mul, div, sqrt, fma, exp, log, sin, atan**.
Real-valued only. No complex overloads.

## Methodology (applies to every phase)

The measurement harness is a single-authority replacement for the
current `bench_cost` tool. The demos directory is retired as a
benchmark and (eventually) renamed to usage examples.

Non-negotiable methodology:

- **Kernel-level cost.** One `Kokkos::parallel_for` over a
  saturating batch, one op per element (or the minimum arithmetic
  work that expresses the op — e.g. `fma` is one FMA per element,
  `add` is one add per element). No inner op-loop that turns the
  kernel compute-bound artificially; the number reflects what a
  user's real kernel feels.

- **Same code path, swap the type.** The FP64 measurement and the
  backend measurement must be the same Kokkos kernel with only the
  scalar type differing. No conditional branches, no separate
  kernel bodies. This is the whole point — the ratio measures the
  emulation cost, not two different kernels.

- **Saturating batch size, measured plateau.** Sweep batch size and
  publish the number at the batch size where per-op cost stops
  falling. Do NOT hardcode a size; the plateau point differs
  between GB200 and MI300X, between FP64 and QF (compute-bound
  vs memory-bound regimes differ), and possibly between ops. The
  harness has to find it, record it, and use it.

- **Warmup + median.** Discard first N warmup runs (N ≥ 3), then
  take the median of at least 5 timed runs. Not the mean. Median
  is robust to the one-off outlier that always happens.

- **Event-based timing.** CUDA events on Nvidia, HIP events on AMD.
  No host-side wall-clock timing of GPU kernels — the launch/sync
  overhead pollutes the measurement.

- **Clock/power state disclosed.** Record whether the GPU is in
  its default P-state or pinned. Do not silently pin — if you
  pin, say so, and say what you pinned to.

- **Machine-readable output.** The harness emits a table (CSV or
  JSON, your call) that the README can lift from directly. No
  hand-transcription of numbers from run logs into Markdown; that
  is where drift starts.

## Repository conventions

- All work lands on `main`. No task branches unless a phase says
  otherwise.
- Commits: one commit per logical unit, message pattern
  `perf: <phase> — <one-line summary>`.
- Each phase closes with a small STATUS block in
  `docs/PERF_PLAN_STATUS.md` (create on first phase; extend
  thereafter). Format mirrors `TEST_SUITE_PLAN.md`'s STATUS
  blocks — commit SHA, one-paragraph outcome, any deviations
  or surprises up front.
- Rule 4 still applies: library headers
  (`third_party/include/*.hpp`) are OFF-LIMITS for this arc. All
  measurement code lives under `bench/` (create if needed) and
  test-adjacent utilities under `tests/`.

---

## Phase 1 — Methodology, one cell

**Goal:** Get the harness right on DD `add` on GB200. One data
point. Establish the shape everything else copies.

**Deliverables:**

1. `bench/bench_cost/` directory (or wherever the harness lives —
   your call, but justify in the commit if you diverge from
   `bench/bench_cost/`). Contains:
   - The kernel body, templated on scalar type
   - The saturation-sweep driver
   - Warmup + median timing wrapper
   - CUDA-event-based timer
   - Machine-readable output emitter (one line per (op, backend,
     GPU, batch_size) sample; final "chosen plateau" row per (op,
     backend, GPU) clearly marked)
2. A single run on GB200 producing:
   - FP64 `add` — saturation sweep + chosen plateau
   - DD `add` — saturation sweep + chosen plateau
   - Slowdown ratio (DD/FP64)
3. Commit the harness AND the run's output artifact (CSV/JSON) under
   `bench/results/gb200/` (or similar). We want the numbers in the
   repo, not just in a report.
4. `docs/PERF_PLAN_STATUS.md` STATUS block for Phase 1: SHA, chosen
   plateau batch size for FP64 and DD, the ratio, any harness
   deviations from the spec, GPU P-state / clock notes.

**Explicitly NOT in scope for Phase 1:**

- MI300X (Phase 4)
- Other ops (Phase 2)
- Other backends (Phase 3)
- README wiring (Phase 5)
- Demos rename/retirement (Phase 6)
- Any library header change

**Verification bar:** the plateau is real (per-op cost stops falling
past the chosen batch size — show the sweep in the artifact, not
just the plateau point). The median-of-N is stable across two
independent runs (run it twice; ratios should agree to within a few
percent, or the harness is broken).

**Success criterion:** one ratio number for DD `add` on GB200,
reproducible, in the repo as an artifact, and a harness that
`grep -A` on the phase-2 prompt will treat as a template.

---

## Phase 2 — Expand ops on DD + GB200

**Prerequisites:** Phase 1 landed on `main`.

**Goal:** Fill out the full 9-row DD column for GB200.

**Deliverables:**

1. Extend the harness (from Phase 1) to sweep all 9 ops: add, mul,
   div, sqrt, fma, exp, log, sin, atan. One kernel per op,
   templated on scalar type. Each op gets its own saturation
   sweep — do NOT reuse `add`'s plateau, because compute intensity
   varies wildly across the 9 (transcendentals are compute-bound
   at much smaller batch sizes than add).
2. Run all 9 ops for FP64 and DD on GB200. Update the artifact
   under `bench/results/gb200/`.
3. STATUS block for Phase 2: SHA, per-op plateau + ratio, any op
   that surprised (e.g. a plateau that behaves nonmonotonically,
   an op where DD is faster than FP64), GPU P-state notes.

**Not in scope:** QF, FF, MI300X, README wiring.

**Verification bar:** the harness output for `add` reproduces the
Phase 1 ratio (within noise). If it doesn't, the harness broke
during expansion and the whole run is suspect.

---

## Phase 3 — Add QF and FF on GB200

**Prerequisites:** Phase 2 landed on `main`.

**Goal:** Fill the GB200 column for all three backends.

**Deliverables:**

1. Extend the harness's type-parameter set to include QF and FF.
   Same kernel bodies, only the scalar type differs.
2. Run all 9 ops for QF and FF on GB200. Update the artifact.
3. STATUS block for Phase 3: SHAs, per-(backend, op) plateau +
   ratio, note any FF ratio below 1.0x (expected on some ops given
   FP32 substrate), note any QF ratio that jumps unexpectedly
   between ops.

**Not in scope:** MI300X, README wiring.

**Verification bar:** DD ratios reproduce Phase 2's numbers within
noise. FF's add/mul ratios are sanity-checked against back-of-
envelope (FF is 2 FP32 words, ~10 FLOPs per add, so ratio vs FP64
should be O(1–10x), not O(100x); if it's off by orders of
magnitude the FF path is not being compiled correctly).

---

## Phase 4 — Port to MI300X

**Prerequisites:** Phase 3 landed on `main`.

**Goal:** Fill the MI300X column for all three backends, all 9 ops.

**Deliverables:**

1. Verify the harness builds and runs under HIP on MI300X. Fix any
   Nvidia-isms (CUDA events → HIP events; any `cudaDeviceSynchronize`
   → `hipDeviceSynchronize`; etc.).
2. Run FP64 + DD + QF + FF × 9 ops on MI300X. Save artifact under
   `bench/results/mi300x/`.
3. STATUS block for Phase 4: SHAs, cross-GPU comparison notes
   (any ratio that differs by more than ~2x between GPUs on the
   same (backend, op) — those are the interesting rows for the
   eventual README).

**Not in scope:** README wiring (Phase 5).

**Verification bar:** the harness on Nvidia produces the same DD
`add` ratio as Phase 2 after the HIP port lands (i.e., the HIP
port didn't accidentally regress the CUDA path). Rebuild + rerun
Phase 2's single sanity cell on GB200 to confirm.

---

## Phase 5 — README wiring

**Prerequisites:** Phase 4 landed on `main`.

**Goal:** Publish the three tables in the top-level `README.md`.
This is what the whole arc was for.

**Deliverables:**

1. New section in `README.md` — placement TBD by whoever runs
   Phase 5, but likely between the existing accuracy discussion
   and the licensing/build sections. Title suggestion:
   "Performance overhead vs FP64" (adjust if a better fit exists).
2. Three tables (DD, QF, FF), 9 rows × 2 columns (GB200, MI300X),
   cells lifted directly from `bench/results/*/` artifacts.
3. One-line pointer to `bench/results/` under each table so
   readers can inspect the raw sweep data if they want to.
4. No preamble hand-holding. The tables sit alongside the
   existing accuracy tables. Cross-reference between them is the
   reader's job.
5. Move `README_PARKED.md` contents (the parked cost/demos
   material from commit `54bce8d`) into `bench/README.md` or
   similar as historical record if any of it is still relevant
   — or delete outright if the phase-1-through-4 harness has
   superseded all of it. Justify the call in the commit.
6. STATUS block for Phase 5.

**Not in scope:** the demos rename (Phase 6).

**Verification bar:** every number in the README tables is
traceable to a specific row in `bench/results/*/`. No
hand-transcription errors. Numbers agree with the artifact to
the printed precision.

---

## Phase 6 — Retire demos as benchmark, keep as usage examples

**Prerequisites:** Phase 5 landed on `main`. Independent of
Phase 5's exact wording — this phase does not touch the README
tables, only the demos directory.

**Goal:** Eliminate the "which measurement tool do I trust"
liability by making it structurally impossible to mistake the
demos for a benchmark.

**Deliverables:**

1. Rename `demos/` to `examples/` (or `usage/` — choose based on
   what fits repo conventions; check other Kokkos-org repos for
   precedent).
2. Update every reference to `demos/` in the codebase: README,
   docs, comments, CMakeLists, tests. `git grep -l demos` before
   and after — after should return only historical mentions in
   `PORT_NOTES.md` / `TEST_SUITE_PLAN.md` STATUS blocks (those
   are the record, do not rewrite history).
3. Add a short header comment to each renamed file clarifying
   "this is a usage example, not a benchmark; see `bench/` for
   performance measurement" — one sentence, not a lecture.
4. STATUS block for Phase 6.

**Not in scope:** any change to what the examples do; only their
location and framing.

**Verification bar:** `git grep demos` after the phase returns
only historical references. Build + ctest still green (the
examples were part of the build; the renamed examples still are).

---

## Out of scope for the entire arc

- Any change to `third_party/include/*.hpp` (Rule 4)
- Complex overloads (real ops only in the tables)
- Consumer-GPU measurement (RTX cards, etc.) — server GPUs only
- Cross-arch comparison (CPU `__float128` etc.) — GPU-only, same
  GPU, same code path
- The precision-efficiency composite metric (discussed and dropped
  during design; slowdown-vs-FP64 is simpler and sufficient)
- The FF-vs-FP32 baseline framing (discussed and dropped; readers
  cross-reference the accuracy tables)

## Design decisions locked in (do not re-litigate)

- Slowdown vs FP64 as the sole metric
- FP64 baseline on the SAME GPU (no cross-GPU or cross-arch baselines)
- Kernel-level cost (not per-op-in-a-loop)
- Same-code-path type swap (not separate kernel bodies)
- 9 ops, real only
- GB200 and MI300X only
- Phased sequencing 1 → 2 → 3 → 4 → 5 → 6, each depending on the
  previous
