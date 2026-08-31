# UPSTREAM_PLAN_STATUS.md

This file records the outcome of each sub-plan in `docs/UPSTREAM_PLAN.md`. Append a STATUS block after each sub-plan completes.

---

## S0 — Freeze tag + doc hygiene

**Commits:** (none — docs-only changes not yet committed)

**Tag:** `kokkos-native-freeze` (annotated, local only, not pushed) at commit `7f32f4e`

**Outcome:** Freeze point established, documentation updated to match current repository state. Annotated tag `kokkos-native-freeze` created on current `main` (commit 7f32f4e) as immutable fallback for any future Kokkos-core-native contribution work. Tag message documents this is the last fully Kokkos-native state before standalone restructure (`docs/UPSTREAM_PLAN.md`). Tag created locally only — not pushed.

`CLAUDE.md` rewritten to describe the actual repository: three portable backends (DD, FF, QF), seven executables (6 demos + 1 bench), two branches (`main` with all three backends, `CUDAFP128Kokkos` with sm_100-only CUDA FP128), 23 tests in the ctest suite. Old content described a stale CUDA-FP128+DD configuration with four executables and a three-branch structure that never existed on this branch. New version is short, points at README/TEST_SUITE_PLAN/PERF_PLAN/UPSTREAM_PLAN instead of duplicating their content.

`docs/PERF_PLAN.md` marked PARKED with a banner stating the arc is paused pending `docs/UPSTREAM_PLAN.md` completion (repo layout will change underneath it) and that Phase 1 must not be started. No other edits to that file.

**Deviations and surprises:** None. All verification passed: CMakeLists.txt shows 7 `add_executable` calls, tests/CMakeLists.txt shows 23 test registrations (14 plain + 6 EFT + 3 EFT-contract-on), two branches confirmed via `git branch -a`. Tag created exactly as specified, local only, no push attempted. Commit convention note: S0 is docs-only and the plan's commit-per-logical-unit rule applies at the sub-plan level — the next commit will bundle the CLAUDE.md + PERF_PLAN.md + STATUS.md changes as a single `upstream: S0 — freeze tag + doc hygiene` commit.

**Next sub-plan (S1/S2) must know:** The freeze tag is local only and must be pushed when the human reviews. `CLAUDE.md` now accurately describes the repository state as of commit 7f32f4e (HEAD of `main`). The 23-test count verified from tests/CMakeLists.txt matches what the plan states. PERF_PLAN.md is parked and should not be referenced until the upstream arc completes.
