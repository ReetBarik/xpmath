# Toolchain defects found by this project

Defects in **other people's** toolchains — compilers, backends, vendor
libraries — that this project measured, worked around, and has not yet filed.

This is deliberately not `validation/sweep/open_defects.txt`. That file is the
authoritative list of defects **in this library**, enforced in both directions
by the `sweep_absolute_gate` ctest target, and it is currently empty. The
entries here are the opposite: our code is correct and something underneath it
is not. Nothing automated checks this file, so it carries a filing status per
entry and is otherwise a place things can be retrieved from rather than
rediscovered.

**Each entry's report text is already written as drop-in issue text.** The work
left is submitting it, not composing it.

---

## Open — not filed

### TD-1 · gfx90a: `BranchRelaxation` scavenges the live return-address pair

| | |
|---|---|
| **Report** | [`ROCM_BRANCH_RELAXATION_BUG.md`](ROCM_BRANCH_RELAXATION_BUG.md) |
| **File against** | `ROCm/llvm-project` |
| **Found** | 2026-09, MI250X bringup (S8) |
| **Affects** | ROCm 7.0.2 / AMD clang 20.0.0git (roc-7.0.2 25385), gfx90a, `-O2` and `-O3` |
| **Symptom** | A device callee larger than the ±131,068 B `S_BRANCH` reach gets its out-of-range branch expanded through `s_getpc_b64`/`s_setpc_b64`; `SIInstrInfo::insertIndirectBranch` scavenges the scratch pair without modelling the liveness of `s[30:31]`, so the callee's own return jumps back into its body. Infinite loop (exit 124), or a wild `flat_store` (exit 134). |
| **Our mitigation** | `XPMATH_NOINLINE_FUNCTION` in `include/xp/config.hpp` keeps every callee under the reach. Commit `48df439`. |
| **Our guard** | `scripts/xpm_lint_device_asm.sh` tiers 1–3, in the `device-hip` CI lane |

### TD-2 · gfx90a: a recursive device call provisions no stack

| | |
|---|---|
| **Report** | [`ROCM_RECURSIVE_DEVICE_STACK.md`](ROCM_RECURSIVE_DEVICE_STACK.md) |
| **File against** | `ROCm/llvm-project` |
| **Found** | 2026-09, MI250X bringup (S8), after TD-1 |
| **Affects** | same toolchain as TD-1 |
| **Symptom** | A device function that calls itself makes the backend set `uses_dynamic_stack` and provision nothing, charging the whole non-inlined chain below it to the HIP runtime's per-thread allowance — measured default **1024 B**. A provably depth-1 sign reflection overran it: FF and TF took `hipErrorIllegalAddress`, DD and QF **silently returned wrong values**. There is no diagnostic at compile time, from `hipGetLastError`, or in the kernel's return code. |
| **Our mitigation** | de-recursed every device self-call — `asinh` (DD/FF/QF/TF) and `tanh` (DD/FF/QF) fold on sign, `tgamma` (DD/FF) calls a split-out Lanczos core. Commit `33e4767`. |
| **Our guard** | `scripts/xpm_lint_device_asm.sh` tier 4, in the `device-hip` CI lane |
| **Verified fixed** | runtime on MI250X, 17/17 arms at the default stack — job 1000936, commit `c14a72e` |

> **TD-1 and TD-2 interact.** The `noinline` that mitigates TD-1 turns each link
> of the chain into a real frame, which measurably deepened TD-2's overrun. Any
> change to either mitigation must be re-measured against both.

### TD-3 · libquadmath: trig argument reduction fails above ~1e40

| | |
|---|---|
| **Report** | not yet written up separately; the measurement is in `scripts/sweep_accuracy.cpp`'s `--oracle` help text and `docs/ULP_METRIC.md` |
| **File against** | GCC (libquadmath) |
| **Found** | during the ulp-metric work |
| **Symptom** | `sinq`/`cosq`/`tanq` argument reduction degrades with magnitude: clean at 1e40, **~1e34 ulps wrong at 1e60+**. Large-\|x\| trig rows were being scored against a broken reference. |
| **Our mitigation** | switched to MPFR/MPC at 400 bits as the ONLY oracle. libquadmath is gone: `--oracle=mpfr` is an accepted no-op and `--oracle=quadmath` is REFUSED (see `scripts/sweep_accuracy.cpp` around line 4342). |
| **Note** | This one is a *reference* defect, not a codegen defect: it made our measurements wrong rather than our results wrong. Filing it needs the write-up TD-1 and TD-2 already have. |

### TD-4 · nvcc ptxas underestimates `__local__` spill frames

| | |
|---|---|
| **Report** | not yet written up separately; the measurement is job 1001685 / C7 |
| **File against** | NVIDIA CUDA (nvcc / ptxas) |
| **Found** | 2026-09-18, A100/sm_80 C7 bringup |
| **Affects** | CUDA 12.9.1, sm_80, `nvcc_wrapper`, `-O3` |
| **Symptom** | A large `XPMATH_NOINLINE_FUNCTION` device callee (exp(QF) inlined into exp(QFC); sincos inlined into atan2(QF)) is compiled with a `__local__` spill frame smaller than the stores ptxas emits. `compute-sanitizer --tool memcheck` reports `Invalid __local__ write of size 4 bytes` at a fixed offset in the callee (`exp(QF)+0x151b0`, then `atan2(QF)+0x2ad30` after the exp extract). The kernel's CUDA last-error stays 0; the corruption is silent without the sanitizer. `cudaDeviceSetLimit(cudaLimitStackSize, 16384)` does **not** help: ptxas sizes the frame statically. `#pragma unroll 1` on the Taylor/squarings loops did not move the offset. |
| **Our mitigation** | Extract the oversized inline bodies into NOINLINE helpers (`detail::qf_exp_cw_reduce` / `qf_exp_taylor` / `qf_exp_squarings`); mark `sin`/`cos` NOINLINE and call them instead of inlining `sincos` into `angle()` / `exp(QFC)` / `exp(TFC)`. Same pattern on the Payne-Hanek helpers. `XPMATH_NOINLINE_FUNCTION` under CUDA is `__host__ __device__ __noinline__ inline` (COMDAT + noinline); `-Wattributes` on that token is suppressed by a CUDA-only `#pragma GCC diagnostic ignored "-Wattributes"` in `include/xp/config.hpp`. |
| **Verified fixed** | `compute-sanitizer --tool memcheck` on `$HOME/sweep_device_sm80`: `ERROR SUMMARY: 0 errors`. Full A100 sweep, 436,080 rows, last_error()==0, Cobalt job 1001685, gpu06. |

---

## Related, but not defects

- **Kokkos has no `__complex128` math wrapper, and this repo no longer cares.** `impl/Kokkos_ComplexQuadPrecisionMath.hpp` was a local extension carried in `patches/`, not upstream, and it is why the four complex demos used to sit behind a CMake probe. Both are deleted: the demos stopped printing accuracy columns, which is all the wrapper ever served. See `patches/README.md`.
- **`nvcc` rejects `__float128` in device code.** Documented behaviour, not a defect. It is a real constraint on this repo (it forces the host-oracle/device-code TU split scheduled for S6), recorded in the S1 STATUS block.

---

## Filed

*Nothing yet.*
