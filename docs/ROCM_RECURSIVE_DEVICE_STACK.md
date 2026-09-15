# gfx90a: a one-level recursive device call costs the kernel its stack bound

Second upstream defect found while bringing `include/xp/` up on MI250X, and
**unrelated to [ROCM_BRANCH_RELAXATION_BUG.md](ROCM_BRANCH_RELAXATION_BUG.md)**
except that the mitigation for that one made this one worse (see
[Interaction](#interaction-with-the-branch-reach-mitigation)). This file is
drop-in issue text plus the repo-facing notes.

- **Local fix:** de-recurse. Every device self-call in `include/xp/` is gone —
  `asinh` (DD/FF/QF/TF) and `tanh` (DD/FF/QF) reflect by a sign *fold*,
  `tgamma` (DD/FF) reflects by calling a split-out Lanczos core. Proven
  bit-identical on the host over 90,453 points × 15 functions.
- **Local regression guard:** tier 4 of `scripts/xpm_lint_device_asm.sh`, wired
  into the `device-hip` lane of `.github/workflows/ci.yml`.
- **Status:** not yet filed upstream.

Every number below was measured on MI250X across three batch jobs on
`gpu_amd_mi250` (JLSE), 2026-09-13 … 2026-09-15, jobs 1000794 / 1000851 /
1000909. Nothing here is inferred from the host.

---

## Report

**Title:** [AMDGPU] A recursive device call sets `uses_dynamic_stack` and
provisions nothing, so a one-level reflection overruns the 1024 B default HIP
stack (gfx90a, ROCm 7.0.2)

### Environment

```
AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project
                             roc-7.0.2 25385 0dda3adf56766e0aac0d03173ced3759e1ffecbc)
HIP 7.0.51831-7c9236b16, ROCm 7.0.2
target amdgcn-amd-amdhsa, gfx90a:sramecc+:xnack-, MI250X
-O3, -ffp-contract=off
```

### Summary

A device function that calls itself makes the backend stop bounding the kernel's
stack. The kernel descriptor gets

```
.amdhsa_private_segment_fixed_size <covers only the statically bounded part>
.amdhsa_uses_dynamic_stack 1
```

and **nothing is provisioned for the recursive frame**. The entire non-inlined
call chain below the recursive entry is then charged at run time to the HIP
runtime's per-thread allowance, `hipDeviceSetLimit(hipLimitStackSize, N)`, whose
default we measured at **1024 bytes**.

The recursion here is *one level and statically so* — an odd-function sign
reflection, `asinh(-a) = -asinh(a)`. The recursive call is made only when
`a < 0`, and it passes `-a`, which cannot take that branch again. Depth is
provably 1. The backend does not need to prove that; it needs only to not
silently hand the whole frame chain to a 1 KB allowance.

### Two symptoms, one cause

The same defect presents two different ways depending on how much stack the
chain actually wants:

| backend | `pss` (posture L) | symptom |
|---|---|---|
| `FloatFloat` | 1328 | `hipErrorIllegalAddress`, rc=134 |
| `TripleFloat` | 1568 | `hipErrorIllegalAddress`, rc=134 |
| `DoubleDouble` | 2416 | **silently wrong results** on the reflected arm, rc=0 |
| `QuadFloat` | 2208 | **silently wrong results** on the reflected arm, rc=0 |

The silent arm is the dangerous one. `xp::asinh` on `DoubleDouble` returned
grossly wrong values — not last-place error, wrong by many orders of magnitude —
for 21 of 849 negative grid points, with `hipGetLastError()` clean and the
kernel reporting success. `QuadFloat`: 135 of 849.

Only the reflected arm is affected. Positive arguments, which never make the
recursive call, are correct at every input in all four backends.

### The knob proves it, on an unmodified binary

`hipDeviceSetLimit(hipLimitStackSize, N)` before launch, same binary, same
inputs. `ABORT` = `hipErrorIllegalAddress`; the number is the count of grossly
wrong results out of 1700 grid points:

| N (bytes) | DD | FF | QF | TF |
|---|---|---|---|---|
| default (1024) | 21 | ABORT | 135 | ABORT |
| 2048 | 21 | 6 | 135 | ABORT |
| 4096 | **0** | 6 | **6** | 6 |
| 8192 | 0 | 6 | 6 | 6 |
| 16384 | 0 | 6 | 6 | 6 |
| 65536 | 0 | 6 | 6 | 6 |

Both symptoms clear at the same place. The residual 6 are a genuine numerical
issue in the `|a| > 1/sqrt(eps)` branch, unrelated: they are present at every
rung, and they are present in the de-recursed build too.

A build with the recursion ablated at source (reflection rewritten as a fold) is
**completely insensitive to the knob** — identical results at the default and at
65536, rc=0 throughout. That closes the loop: the knob and the source change are
two ways of removing the same thing.

### A self-call is necessary but not sufficient

Worth stating because it is the first thing a reader will try to falsify.
`xp::tanh` carries the identical construct in DD/FF/QF — same one-level sign
reflection, same self-call, `uses_dynamic_stack 1` in every kernel that reaches
it — and it never aborts and never returns a wrong value, at any rung.

The self-call is what removes the *bound*. What decides whether removing the
bound hurts is the peak depth of the non-inlined chain below the recursive
entry. `asinh`'s is `asinh → asinh → log1p → log → exp → multiply`, six real
frames. `tanh`'s is `tanh → tanh → expm1 → …`, and it fits in 1024 B.

Direct evidence that this is margin and not construction: ablating `asinh` alone
drops the `TripleFloat` kernel to `uses_dynamic_stack 0` — TF has no `tanh`
reflection — while DD, FF and QF **stay at 1** on `tanh`'s self-call alone. Three
of four kernels remain on an unbounded stack after the symptomatic function is
fixed. Anything that later deepens `tanh`'s chain by three frames converts it
into `asinh` with no source change to `tanh` at all.

### Reproducer

Compile-only is enough to see the cause:

```cpp
#include <hip/hip_runtime.h>
__device__ double f(double a) { return a < 0.0 ? -f(-a) : a * a + 1.0; }
__global__ void k(const double* in, double* out) { out[0] = f(in[0]); }
```

```
hipcc -std=c++17 --offload-arch=gfx90a -O3 --save-temps=obj -c repro.hip
grep -E 'uses_dynamic_stack|private_segment_fixed_size' repro-*gfx90a.s
```

Removing the recursion (`double s = a < 0.0 ? -1.0 : 1.0; a *= s; return s * (a*a+1.0);`)
flips `uses_dynamic_stack` back to 0.

To see the *effect* you need a chain deep enough to exceed 1024 B; the four
`include/xp/` backends at commit `6ddcb2d` are one, and the table above is that
measurement.

### Control: NVIDIA

The same sources are correct on an A100 at every optimisation level. `nvcc`
handles the same recursion without a per-thread stack surprise (CUDA's default
`cudaLimitStackSize` is 1 KB too, but the frames land differently).

### Suggested fix direction

Ranked by how much we would trust them:

1. **Provision something.** When `uses_dynamic_stack` is set because of
   recursion whose depth the backend cannot bound, it still knows the *static*
   frame sizes of the chain. Emitting a `private_segment_fixed_size` that covers
   one recursive frame plus the known chain would turn an illegal address into a
   correct result for the overwhelmingly common one-level case.
2. **Bound the provable case.** A self-call under a condition that the recursive
   argument demonstrably cannot re-satisfy is depth-1. That is an analysis, not
   a heuristic, and SCCP-style reasoning already in the pipeline is close to it.
3. **At minimum, diagnose.** `-Wframe-larger-than`-style: warn when a device
   function is recursive *and* the statically known chain below it exceeds the
   runtime default, because the failure mode is otherwise a wrong number with a
   clean error code.

(3) alone would have saved this campaign three batch jobs. The silent-wrong-
answer arm is the real complaint: there is no diagnostic anywhere — not at
compile time, not from `hipGetLastError`, not from the kernel's return code.

---

## What we did locally

### The fix

De-recurse. Ship no device self-call at all, rather than ship a self-call with
a stack knob next to it.

The reason to fix the library rather than to call
`hipDeviceSetLimit(hipLimitStackSize, 4096)` in the harness is that the knob is
a property of the *process*, and `include/xp/` is a header-only library with no
process of its own. Setting it in the harness fixes the harness; every other
consumer — a Kokkos application, a downstream `#include`, the next validation
tool — inherits the defect and gets no warning. The knob was the diagnostic
instrument. It is not the deliverable.

Two shapes of de-recursion, both value-preserving by construction:

**Sign fold**, for the odd-function reflections (`asinh` ×4, `tanh` ×3):

```cpp
// was: if (a.hi < 0.0) return negate(asinh(negate(a)));
const bool neg = (a.hi < 0.0);
if (neg) a = negate(a);
DoubleDouble r;
... existing branches, assigning r ...
return neg ? negate(r) : r;
```

`negate()` is exact word-wise negation, so evaluating on `|a|` and negating the
result returns the identical bits the recursive form returned. **This is not a
numerical change and must not be scored as one.**

**Core split**, for the argument reflection (`tgamma` ×2), where the reflected
call passes `1-a` rather than `-a`:

```cpp
XPMATH_INLINE_FUNCTION DoubleDouble dd_tgamma_lanczos(DoubleDouble a) { ...core... }

XPMATH_INLINE_FUNCTION DoubleDouble tgamma(DoubleDouble a) {
    if (a.hi < 0.5) return divide(pi, multiply(sin(multiply(pi, a)),
                                  dd_tgamma_lanczos(subtract(DoubleDouble(1.0), a))));
    return dd_tgamma_lanczos(a);
}
```

Value-preserving by inspection: the reflection fires only for `a.hi < 0.5` and
passes `1-a`, whose leading word is then `>= 0.5`, so the recursive form could
only ever have reached the Lanczos branch anyway.

`tgamma` is included even though it was never measured failing: it is not in the
device sweep's op list and does not appear in the posture-L device asm, so it has
never executed on a GPU. It is the same construct, it is latent, and leaving it
in would mean the tier-4 gate below had to ship with a permanent exception.

### Verification of the fix

| check | result |
|---|---|
| host bit-identity, pre- vs post-fix | **identical**, 90,453 points × 15 functions, 18,090,600 B, same SHA-256 |
| `scripts/check_standalone_no_kokkos.sh` | PASS |
| device self-calls, `hipcc --offload-arch=gfx90a -O3` | **9 → 0** (asinh ×4, tanh ×3, tgamma ×2) |
| … same, posture H (blanket `noinline`) | **9 → 0** |
| `uses_dynamic_stack` on the probe kernel | **1 → 0** (posture L and posture H) |
| `private_segment_fixed_size` on the probe kernel | 2560 → 1936 B |
| `scripts/xpm_lint_device_asm.sh` | exit 1 pre-fix, exit 0 post-fix |

The host grid is deliberately symmetric and straddles every internal branch
point (0.5, `kSqHi`, the `tanh` saturation knee, the `tgamma` poles), because the
fold changes only where the sign is applied — a grid that does not test both
signs at the same magnitude cannot see a sign-fold bug at all.

### The guard

`scripts/xpm_lint_device_asm.sh` grew a fourth tier over the same census:

| tier | severity | what |
|---|---|---|
| 4 | FATAL | any function that calls itself |
| 4 | WARN | a kernel descriptor with `uses_dynamic_stack=1` |

Three rules it encodes:

- **Zero tolerance, not a threshold.** Unlike tier 3, there is no safe amount of
  recursion: one self-call disables the static bound for the entire chain. The
  depth is not what hurts.

- **Anchor on the relocation operand, and require the symbol to match
  *exactly*.** The backend spells a direct call as
  `s_add_u32 sN, sN, <callee>@rel32@lo+4`, so a self-call is exactly a
  `<sym>@rel32@lo` inside the body of `<sym>`. Both halves of that sentence are
  load-bearing:

  *Operand, not name.* A source-level grep for `name(` inside `name` reports
  ~76 hits on these headers, nearly all of them `detail::` scalar overloads and
  comments. And because the mangled name carries the signature, a same-named
  *overload* — `abs(complex)` calling `abs(real)`, `sincos(a,s,c)` called from
  `sin(a)` — is correctly not a self-call.

  *Exactly, not as a prefix.* A function that reads a `constexpr` table takes
  the PC-relative address of its own `__const.<mangled-name>.t`, which contains
  the function's mangled name as a substring. The four Payne–Hanek table
  helpers (`xp_ph_ipio2_{d,f}`, `xp_ph_pio2_{d,f}`, each a bare
  `return t[k];`) are outlined at posture H and are exactly this shape. A
  prefix-matching detector reports them, which is why the step-3 campaign's
  in-job scan reported **11** self-calls at posture H where the true count is
  **9**. Verified against that case: on a posture-H TU that outlines all four
  helpers, tier 4 reports 9 pre-fix and 0 post-fix.

  This is the same lesson `.Lpost_getpc` teaches in tiers 1–3, in a new costume:
  *on AMDGPU, taking an address and making a call look alike, and the library's
  constant tables make the difference matter.*

  Two intermediate versions of this detector were wrong in the other direction —
  one missed all 9 hits because AMDGPU label lines carry a trailing `; @sym`
  comment, another because `.LBB*` local labels match a bare "identifier
  followed by colon" and reset the current-function tracker.

- **Tier 4 is the only tier where kernels are not exempt.** In tiers 1–3 a
  kernel is immune because it ends in `s_endpgm` and holds no return address.
  Here the kernel is the victim: the self-call is in a callee, and the kernel is
  what carries `uses_dynamic_stack` and overruns the allowance.

The `uses_dynamic_stack` WARN is reported rather than fatal on its own: a
legitimate dynamic `alloca` would land there too. The library has none, so a hit
with tier 4 clean means come and read this file.

### Interaction with the branch-reach mitigation

These two defects pull against each other and **any future change to either must
be re-measured against both.**

`XPMATH_NOINLINE_FUNCTION` (commit `48df439`, shipped in `6ddcb2d`) is the
mitigation for the branch-relaxation bug: it keeps every device callee under the
±131,068 B `S_BRANCH` reach by refusing to inline. That is exactly what turns
each link of `asinh`'s chain into a real stack frame. Measured: at the posture
where *every* `xp` function is forced `noinline`, `QuadFloat` asinh's gross-error
count rises from 73 to 135.

So `6ddcb2d` did not create this defect — the recursion predates it and the
default allowance is the same either way — but it deepened the chain that
overruns the allowance, and `ROCM_BRANCH_RELAXATION_BUG.md`'s claim that the
mitigation's cost is nil ("Numerics are untouched") is true only in the sense it
was written in: `noinline` cannot change an FP computation. It can and did change
whether that computation runs at all.

### Where it runs

The `device-hip` lane of `.github/workflows/ci.yml`, over the `.s` that
`--save-temps=obj` leaves next to the object, alongside tiers 1–3. The lane's TU
was extended to call `asinh`/`tanh`/`tgamma` in all four backends: without those
calls the TU emits no recursive callee and tier 4 is vacuously clean. The lane
remains `continue-on-error: true` — that reflects the unverified ROCm container,
not the guard, which was run locally against ROCm 7.0.2 both ways.

### What this does not fix

- **The device accuracy corpus.** `amd_fixed_real_O3.csv`'s `asinh` columns were
  measured under the defect and are not usable; see the note in that file's
  vicinity. Note that it records device-vs-host *agreement*, not accuracy — the
  host record in `validation/sweep/` is untouched by any of this.
- **The complex kernels.** The `ComplexFunctor` kernels for DD, FF and QF also
  carry `uses_dynamic_stack=1` in the pre-fix asm, by reaching the real `asinh`
  and `tanh` through their complex counterparts. The de-recursion removes the
  cause for them too, but no complex arm was ever *measured* on device in this
  campaign, so their pre-fix behaviour is unknown and their post-fix behaviour
  is asserted only structurally.
