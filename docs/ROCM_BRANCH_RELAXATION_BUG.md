# gfx90a: BranchRelaxation clobbers the return address

Upstream defect found while bringing `include/xp/` up on MI250X. This file is
**drop-in issue text** for `ROCm/llvm-project`, plus the repo-facing notes on
what we did about it locally.

- **Local mitigation:** commit `48df439` — `XPMATH_NOINLINE_FUNCTION` in
  `include/xp/config.hpp` keeps every device callee under the branch reach.
- **Local regression guard:** `scripts/xpm_lint_device_asm.sh`, wired into the
  `device-hip` lane of `.github/workflows/ci.yml`.
- **Status:** not yet filed upstream.

Everything in the "Report" section below was re-measured on the round-0 (pre-fix)
device assembly on 2026-09-15, except the two items explicitly marked as carried
from the device-validation campaign (the runtime hangs and the A100 control),
which were measured there and not re-run here.

---

## Report

**Title:** [AMDGPU] BranchRelaxation scavenges the live return-address pair
`s[30:31]` in a non-entry function, producing an infinite loop (gfx90a, ROCm 7.0.2)

### Environment

```
AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project
                             roc-7.0.2 25385 0dda3adf56766e0aac0d03173ced3759e1ffecbc)
HIP 7.0.51831-7c9236b16, ROCm 7.0.2
target amdgcn-amd-amdhsa, gfx90a:sramecc+:xnack-, MI250X
```

`-O2` and `-O3`. `-O0`/`-O1` are unaffected, but only incidentally: at those
levels the functions stay small enough that no branch needs relaxing.

### Summary

In a device function large enough to require branch relaxation,
`SIInstrInfo::insertIndirectBranch` obtains its scratch SGPR pair from
`RegScavenger::scavengeRegisterBackwards` and is handed `SGPR30_SGPR31` — the
pair holding the incoming return address — with no spill and no restore
(`.private_seg_size` is `0` for all three functions below). The function later
returns with `S_SETPC_B64_return s[30:31]`, which now branches back into its own
body instead of to the caller.

Observed effects: the kernel never completes and `hipDeviceSynchronize` never
returns (infinite loop), or a wild `flat_store` and `hipErrorIllegalAddress`
once the diverted path reaches a store with stale address registers.

Kernels are unaffected. They end in `s_endpgm` and never hold a return address,
so `s[30:31]` is genuinely dead there and scavenging it is legal.

### Cleanest instance

`xp::QuadFloatComplex::operator/(QuadFloatComplex) const`
(`_ZNK2xp16QuadFloatComplexdvES0_`), 78,778 asm lines, 316,720 bytes by ELF
`st_size`. The relaxed edge comes straight off an ordinary source-level guard,
`if (b.re == 0 && b.im == 0)`:

```asm
        s_waitcnt vmcnt(0)
        v_cmp_neq_f32_e64 s[4:5], 0, v24         ; b.im != 0
        s_or_b64 s[4:5], vcc, s[4:5]
        s_and_saveexec_b64 s[12:13], s[4:5]
        s_cbranch_execnz .LBB46_1                ; inverted by BranchRelaxation
; %bb.10347:                                     ; all active lanes divide by zero
        s_getpc_b64 s[30:31]                     ; <-- clobbers the return address
.Lpost_getpc156:
        s_add_u32  s30, s30, (.LBB46_10344-.Lpost_getpc156)&4294967295
        s_addc_u32 s31, s31, (.LBB46_10344-.Lpost_getpc156)>>32
        s_setpc_b64 s[30:31]
.LBB46_1:                               ; %._crit_edge
```

and `.LBB46_10344` is the function's **return block**:

```asm
.LBB46_10344:                           ; %_ZN2xp6divideENS_9QuadFloatES0_.exit902
        s_or_b64 exec, exec, s[12:13]
        v_accvgpr_read_b32 v0, a0
        v_mov_b32_e32 v12, v19
        v_accvgpr_read_b32 v1, a1
        v_mov_b32_e32 v6, v5
        flat_store_dwordx4 v[0:1], v[12:15]
        flat_store_dwordx4 v[0:1], v[6:9] offset:16
        s_waitcnt vmcnt(0) lgkmcnt(0)
        s_setpc_b64 s[30:31]            ; s[30:31] == &.LBB46_10344
.Lfunc_end46:
```

so the return branches to itself: a single-block infinite loop that re-issues
both stores forever.

This function contains three relaxed edges, and **all three** land in or
immediately above that return block — `.LBB46_10342` and `.LBB46_10343` are
one-instruction `s_or_b64 exec, exec, ...` blocks that fall straight through
into `.LBB46_10344`:

```asm
.LBB46_10342:                           ; %Flow19670
        s_or_b64 exec, exec, s[8:9]
.LBB46_10343:                           ; %Flow19672
        s_or_b64 exec, exec, s[6:7]
.LBB46_10344:                           ; %_ZN2xp6divideENS_9QuadFloatES0_.exit902
```

Across the whole 78,778-line function, `s[30:31]` / `s30` / `s31` appear on
exactly thirteen lines: the twelve that make up the three relaxation sequences,
and the final `s_setpc_b64 s[30:31]` return. The pair is never saved, never
restored, and never otherwise written.

### Corroborating instances in the same translation unit

| function | ELF `st_size` | sites on `s[30:31]` | targets |
|---|---|---|---|
| `xp::QuadFloatComplex::operator/` | 316,720 B | 3 | `.LBB46_10342` / `_10343` / `_10344` — all reaching the return block |
| `xp::sinhcosh(QuadFloat, QuadFloat&, QuadFloat&)` | 251,992 B | 3 | `.LBB27_8542`, `.LBB27_8541`, `.LBB27_729` |
| `xp::TripleFloatComplex::operator/` | 172,924 B | 1 | `.LBB53_5488`, the return block |

`sinhcosh` is worth calling out, because only one of its three relaxed edges
targets the return block directly:

- `.LBB27_8542` **is** the return block — same self-loop as above.
- `.LBB27_8541` is `s_or_b64 exec, exec, s[10:11]` + an
  `s_andn2_saveexec_b64`, a merge block that reaches `.LBB27_8542`.
- `.LBB27_729` is an ordinary mid-function block full of
  `buffer_load_dword`/`v_mul_f32`. Returning into it re-executes a large span of
  the function with a corrupted return address still in `s[30:31]`, so it loops
  too — but it is nowhere near the return.

Any predicate of the form "only fatal if the relaxed branch targets the return
block" would clear `.LBB27_729` and be wrong.

### The correlation: SGPR pressure, inversely

Of the 13 functions that needed relaxation in this one translation unit, the
scavenger's choice tracks SGPR pressure *inversely* — which is what one would
expect if the return-address pair's liveness is unmodelled at a mid-function
block:

| `.numbered_sgpr` | pair chosen | function |
|---|---|---|
| 76 | **`s[30:31]`** | `QuadFloatComplex::operator/` (3 sites) |
| 76 | **`s[30:31]`** | `TripleFloatComplex::operator/` (1 site) |
| 79 | **`s[30:31]`** | `sinhcosh(QuadFloat, …)` (3 sites) |
| `max(92, …)` | `s[70:71]` | `asinh(QuadFloat)` (1 site) |
| 100 | `s[98:99]` | `angle(QuadFloat, QuadFloat)` (2 sites) |

Under pressure the allocator has a real value in `s30`/`s31`, so the scavenger
sees it live and skips it. In a lower-pressure function the pair holds only the
incoming return address, whose liveness at the relaxation point appears not to
be represented, so it looks free and is taken.

The practical consequence is that this is a lottery, not a deterministic
miscompile: the same source, recompiled after any change that moves SGPR
pressure, may quietly stop reproducing. Four conditions must coincide — the code
is in a callee, some branch exceeds the reach, pressure is low enough that the
scavenger picks `s[30:31]`, and the relaxed edge is actually executed.

### Reproducer

Attached `qf_sat.hip` (self-contained, one header):

```bash
hipcc -std=c++17 -O3 -ffp-contract=off --offload-arch=gfx90a \
      -mllvm -amdgpu-s-branch-bits=14 qf_sat.hip -o /tmp/a1
XPM_RUNG=0 timeout 30 /tmp/a1 $(seq 89)     # argc=90 -> hangs, exit 124
XPM_RUNG=0 timeout 30 /tmp/a1               # argc=1  -> ok (edge not taken)
```

`-amdgpu-s-branch-bits=14` shrinks the reach from ±131,068 B to ±32 KB so that
an 88 KB function reproduces. Without it the same defect needs a ~250 KB
function; `qf_repro.hip` (also attached) hangs at stock `-O3` with no `-mllvm`
flags at all.

Note that *both* long branches *and* low SGPR pressure are required. A 20-line
function built with `-amdgpu-s-branch-bits=6` relaxes cleanly, because there the
scavenger finds a genuinely free pair.

*(Carried from the device-validation campaign; the runtime exit codes above were
measured there, on MI250X. Everything else in this report was re-measured from
the pre-fix device assembly.)*

### Control: NVIDIA

The same sources are correct on an A100 at `-O0`/`-O1`/`-O2`/`-O3`, 104/104
probes, `compute-sanitizer` clean. *(Also carried from the device-validation
campaign, not re-run here.)*

### Suggested fix direction

Any one of:

1. Mark `SGPR30_SGPR31` live-in to every non-entry function for scavenging
   purposes, so `RegScavenger` can see it.
2. Have `insertIndirectBranch` explicitly exclude the return-address pair when
   `MF.getFunction()` is not an entry point (`CallingConv::AMDGPU_KERNEL` et al.).
3. Spill and restore whatever pair it scavenges, as the generic path does. This
   is the most conservative but costs a scratch slot in a function that
   currently declares `.private_seg_size = 0`.

(1) or (2) seem preferable; (3) changes the ABI footprint of the function.

Related: [D114652](https://reviews.llvm.org/D114652), which moved the
call-clobbered return-address registers into the callee-saved range.

---

## What we did locally

### The mitigation

Size is the only lever a library has, and it is a sufficient one: **a callee
smaller than the branch reach provably cannot contain an out-of-range branch.**
So `include/xp/config.hpp` grew `XPMATH_NOINLINE_FUNCTION`, which under `hipcc`
expands to `__host__ __device__ inline __attribute__((noinline))`, and the
primitives that dominate inlined bulk (`multiply`, `divide`, `sqrt`, `exp`,
`log`, `atan2`, and a few others) carry it. `QuadFloatComplex::operator/` and
`TripleFloatComplex::operator/` were additionally split into per-branch helpers,
because their legs are disjoint — exactly one runs per call — but fused into one
body no amount of `noinline` on the primitives could shrink them.

`-DXPMATH_DISABLE_AMDGPU_SIZE_GUARD` opts out.

This removes the *trigger*, not the bug, and the trigger is a byte count that
regrows: a new op, a higher inline threshold in a future LLVM, a different
`--offload-arch`, `-ffast-math` changing unroll decisions. Hence the guard.

Numerics are untouched — `noinline` cannot change FP results, and the build uses
`-ffp-contract=off`, so no contraction decision crosses a call boundary.

### The guard

`scripts/xpm_lint_device_asm.sh` (+ `scripts/xpm_device_guard.awk`) reads the
device `.s` and reports three tiers:

| tier | severity | what |
|---|---|---|
| 1 | FATAL | relaxation scavenged `s[30:31]` in a callee — the live bug |
| 2 | WARN | any relaxation site in a callee — one allocation decision from tier 1 |
| 3 | FATAL / WARN | a callee over 131,068 B / over 98,304 B |

Three rules it encodes, all learned the hard way:

- **Anchor tier 1 on the `.Lpost_getpc` label, never on a bare
  `s_getpc_b64 s[30:31]`.** That instruction is also how the backend spells an
  ordinary call (`… s_swappc_b64`) and a PC-relative constant address
  (`sym@rel32@lo`). A naive grep reports 27 "poisoned" sites on the *fixed*
  build, where the correct count is zero; the 27 are calls and the address of
  the Payne–Hanek 2/π table. Only relaxation plants the label.

- **Never size from ELF symbols on this target.** On the pre-fix linked device
  image, 27 of 86 `FUNC` symbols report `st_size == 0`, and they are
  disproportionately the large ones. Sizes come from instruction-line counts
  instead.

- **Do not classify the jump target, and do not clear a hit because a test
  passed.** `target_is_return` is neither necessary nor sufficient — see
  `sinhcosh` above. And reachability depends on the exec mask, so a `<<<1,1>>>`
  run cannot observe divergent-merge edges at all: "provably dead at one lane"
  is not "dead".

**The size estimate is deliberately conservative in the wrong direction, so read
the WARN, not the FATAL.** Sizes are instruction-lines × 4.71 B/instr. Function
bodies here are `.p2align 2` and `.size` is `.Lfunc_end - start`, so wherever a
`FUNC` symbol carries a nonzero `st_size` that value is exact and can be used to
calibrate. Measured that way over three gfx90a builds:

| | all functions | functions > 50 KB |
|---|---|---|
| mean estimate / `st_size` | 0.865 … 0.919 | 0.908 … 0.940 |
| min | 0.678 | 0.865 |

So 4.71 under-reports by roughly 6–13 % on exactly the large functions tier 3
judges. A callee estimated at the 131,068 B reach could really be ~151 KB. The
98,304 B (0.75 × reach) WARN is what absorbs that: 98,304 / 0.865 = 113,646 B,
still inside the reach at the worst ratio observed. Treat WARN as the
operational gate and FATAL as the backstop. `XPM_BYTES_PER_INSTR` overrides the
constant if a future toolchain shifts the encoding mix.

Tier 3 is also the tier that generalises. It is a *provable* property of the
binary, whereas tiers 1 and 2 only describe the register-allocation outcome you
happened to compile. Keep tier 1 as the backstop for a future LLVM that relaxes
something tier 3 did not predict.

### Where it runs

The `device-hip` lane of `.github/workflows/ci.yml`, over the `.s` that
`--save-temps=obj` leaves next to the object. That lane is
`continue-on-error: true` and stays that way for now — see the comment above the
job for why.
