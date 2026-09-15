#!/usr/bin/env bash
# ============================================================================
# gfx90a device-codegen guard for xpmath.  Two unrelated backend defects, one
# pass over the emitted device asm: branch relaxation (tiers 1-3) and recursive
# device calls (tier 4).
#
#   Usage: scripts/xpm_lint_device_asm.sh <dir-or-.s> [<dir-or-.s> ...]
#
# Why this exists.  gfx9 S_BRANCH carries a signed 16-bit DWORD displacement:
# +/-131,068 bytes.  A device CALLEE larger than that makes LLVM's
# BranchRelaxation pass expand an out-of-range branch through
# s_getpc_b64/s_add_u32/s_addc_u32/s_setpc_b64, and SIInstrInfo::
# insertIndirectBranch scavenges that scratch pair with RegScavenger without
# modelling the liveness of s[30:31] -- the ABI return-address pair.  When the
# scavenger picks s[30:31] the function's own `s_setpc_b64 s[30:31]` return
# jumps back into its own body: an infinite loop on the GPU (exit 124), or a
# wild flat_store once the corrupted pointer reaches a store (exit 134).
# Confirmed on ROCm 7.0.2 / AMD clang 20.0.0git (roc-7.0.2 25385), gfx90a.
# See docs/ROCM_BRANCH_RELAXATION_BUG.md.
#
# Kernels are exempt: they end in s_endpgm and never hold a return address, so
# s[30:31] is genuinely dead there and scavenging it is legal.
#
# Four tiers:
#   tier 1  FATAL  relaxation scavenged s[30:31] in a callee   -- the live bug
#   tier 2  WARN   any relaxation site in a callee             -- one register-
#                  allocation decision away from tier 1
#   tier 3  FATAL  a callee over 131,068 B (unprovable)
#           WARN   a callee over 98,304 B (0.75x, drift margin)
#   tier 4  FATAL  any function that calls itself              -- the second bug
#           WARN   a kernel descriptor with uses_dynamic_stack=1
#
# TIER 4 IS A DIFFERENT DEFECT WITH A DIFFERENT MECHANISM.  A recursive device
# call makes the backend give up on bounding the kernel's stack: it sets
# `uses_dynamic_stack` and provisions nothing, so the whole non-inlined chain
# below the recursive entry is charged to the HIP runtime's per-thread
# allowance, which defaults to 1024 B.  xp::asinh's one-level sign reflection
# overran it on MI250X in all four backends -- FF and TF took
# hipErrorIllegalAddress, DD and QF silently returned wrong values on the
# reflected arm.  Note the interaction with the tiers above: the
# XPMATH_NOINLINE_FUNCTION that defends against branch relaxation is what turns
# each link of that chain into a real frame, so the tier-1..3 mitigation made
# the tier-4 defect WORSE.  See docs/ROCM_RECURSIVE_DEVICE_STACK.md.
#
# THE 98,304 WARN IS THE OPERATIONAL GATE, NOT DECORATION.  Sizes are estimated
# as instruction-lines x 4.71 B/instr, which measures 6-13 % LOW against ELF
# st_size on the large functions tier 3 judges (see xpm_device_guard.awk for the
# three-build calibration).  A callee estimated at 131,068 B could really be
# ~151 KB; one estimated at 98,304 B is at most ~114 KB even at the worst ratio
# observed.  So: FATAL is the backstop, WARN is the line you actually hold.
#
# Tier 3 is the one that generalises: a function shorter than the reach
# *provably* cannot hold an out-of-range branch, whereas tiers 1 and 2 only
# describe the configuration you happened to compile.  Keep tier 1 as the
# backstop for a future LLVM that relaxes something tier 3 did not predict.
#
# Requires the device .s, i.e. --save-temps (or --save-temps=obj) on the device
# compile.  Nothing here reads ELF symbol sizes; see xpm_device_guard.awk.
#
# Env:
#   XPM_SIZE_GATE        override the tier-3 WARN threshold (default 98304)
#   XPM_REACH            override the tier-3 FATAL threshold (default 131068;
#                        only sensible with -mllvm -amdgpu-s-branch-bits=N)
#   XPM_BYTES_PER_INSTR  override the size estimator (default 4.71)
#
# Exit: 0 clean (warnings allowed), 1 tier-1/3/4 FATAL, 2 bad invocation.
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWK="$HERE/xpm_device_guard.awk"
[[ -r $AWK ]] || { echo "xpm_lint_device_asm.sh: missing $AWK" >&2; exit 2; }

REACH=${XPM_REACH:-131068}      # hard: S_BRANCH reach on gfx9.  Over this == unprovable.
GATE=${XPM_SIZE_GATE:-98304}    # 0.75 x reach: CI margin for compiler drift AND
                                # for the estimator's ~10% downward bias.
BPI=${XPM_BYTES_PER_INSTR:-4.71}

[[ $# -gt 0 ]] || { echo "usage: $0 <dir-or-.s> [...]" >&2; exit 2; }

shopt -s nullglob
asms=()
for a in "$@"; do
  if [[ -d $a ]]; then asms+=("$a"/*-amdgcn-amd-amdhsa-*.s); else asms+=("$a"); fi
done
[[ ${#asms[@]} -gt 0 ]] || {
  echo "xpm_lint_device_asm.sh: no device .s found (need --save-temps=obj)" >&2; exit 2; }

rc=0
for s in "${asms[@]}"; do
  echo "== $s"
  [[ -r $s ]] || { echo "  ERROR        unreadable" >&2; rc=2; continue; }

  rep=$(awk -v BPI="$BPI" -f "$AWK" "$s" | sort -k2,2rn)
  if [[ -z $rep ]]; then
    # An empty census means the input is not a device .s (or the asm-printer
    # format moved).  Silence here would be indistinguishable from a clean run.
    echo "  ERROR        no functions recognised -- is this an amdgcn .s?" >&2
    rc=2; continue
  fi

  # ---- tier 1  FATAL: relaxation scavenged s[30:31] in a callee -------------
  t1=$(awk '$1=="callee" && $4+0>0 {printf "    %4d site(s)  %s\n", $4, $6}' <<<"$rep")
  if [[ -n $t1 ]]; then
    echo "  TIER1 FATAL  return-address register scavenged by branch relaxation:"
    echo "$t1"
    rc=1
  else
    echo "  TIER1 ok     no callee scavenges s[30:31] for a relaxed branch"
  fi
  # A kernel may legitimately use s[30:31] as scratch -- report, never fail.
  awk '$1=="KERNEL" && $4+0>0 {printf "  tier1 note   %s uses s[30:31] in a kernel (benign)\n",$6}' <<<"$rep"

  # ---- tier 2  WARN: any relaxation in a callee -----------------------------
  awk '$1=="callee" && $3+0>0 {s+=$3; n++}
       END{ if (n) printf "  TIER2 WARN   %d site(s) in %d callee(s) -- one register-allocation decision from tier 1\n", s, n;
            else   printf "  TIER2 ok     no branch relaxation in any callee\n" }' <<<"$rep"

  # ---- tier 3  FATAL: callee size -------------------------------------------
  over=$(awk -v R="$REACH" '$1=="callee" && $2+0>R  {printf "    %9d B  %s\n",$2,$6}' <<<"$rep")
  near=$(awk -v R="$REACH" -v G="$GATE" '$1=="callee" && $2+0>G && $2+0<=R {printf "    %9d B  %s\n",$2,$6}' <<<"$rep")
  if [[ -n $over ]]; then
    echo "  TIER3 FATAL  callee(s) exceed the ${REACH} B branch reach:"
    echo "$over"
    rc=1
  else
    echo "  TIER3 ok     every callee is inside the ${REACH} B branch reach"
  fi
  if [[ -n $near ]]; then
    echo "  TIER3 WARN   callee(s) over the ${GATE} B drift gate (the estimator runs"
    echo "               ~10% low, so this gate -- not the reach -- is the real line):"
    echo "$near"
  fi

  # ---- tier 4  FATAL: device self-recursion ---------------------------------
  # Zero tolerance, and it applies to KERNELS TOO -- unlike tiers 1-3, where a
  # kernel is exempt because it holds no return address.  Here the kernel is the
  # victim: one self-call anywhere in its call graph is what flips
  # uses_dynamic_stack, and the kernel is what then overruns the 1024 B default.
  t4=$(awk '$5+0>0 {printf "    %4d site(s)  %s\n", $5, $6}' <<<"$rep")
  if [[ -n $t4 ]]; then
    echo "  TIER4 FATAL  device SELF-CALL -- this disables the static stack bound"
    echo "               for every kernel that reaches it (uses_dynamic_stack=1,"
    echo "               nothing provisioned, 1024 B default allowance):"
    echo "$t4"
    echo "               De-recurse it.  A sign reflection becomes a fold; an"
    echo "               argument reflection becomes a call to a split-out core."
    echo "               See docs/ROCM_RECURSIVE_DEVICE_STACK.md."
    rc=1
  else
    echo "  TIER4 ok     no function calls itself"
  fi

  # The descriptor flag itself, read straight from the kernel metadata.  This is
  # the ground truth tier 4 is a proxy for: tier 4 finds the cause in the .text,
  # this finds the consequence in the .amdhsa_kernel block.  Reported, not fatal
  # on its own -- a legitimately dynamic alloca would land here too, and the
  # library has none, so a hit with tier 4 clean is a message to come read this.
  awk '/\.amdhsa_kernel /{n=$2}
       /\.amdhsa_private_segment_fixed_size/{p=$2}
       /\.amdhsa_uses_dynamic_stack/{ if ($2+0>0) printf "  TIER4 WARN   %s: uses_dynamic_stack=1 (pss=%s)\n", n, p; else k++ }
       END{ if (k) printf "  tier4 note   %d kernel descriptor(s) with a bounded stack\n", k }' "$s"

  awk '$1=="callee"{c++; if($2+0>m){m=$2; mn=$6}} $1=="KERNEL"{k++; if($2+0>km)km=$2}
       END{printf "  summary      %d callees (largest %d B, %s), %d kernels (exempt, largest %d B)\n",
                  c+0, m+0, (mn==""?"-":mn), k+0, km+0}' <<<"$rep"
done
exit $rc
