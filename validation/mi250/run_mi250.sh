#!/usr/bin/env bash
# ===========================================================================
# validation/mi250/run_mi250.sh — S8: runtime verification of the gfx90a
#                                 de-recursion fix, on MI250X
# ===========================================================================
#
# Closes the one loop docs/ROCM_RECURSIVE_DEVICE_STACK.md leaves open: the fix
# was verified in the emitted assembly and on the host, never by running the
# reflected arms on the GPU.
#
# Compiles on the COMPUTE NODE -- hipcc is not on the JLSE login node.
#
# Does three things, cheapest first, so a failure early still leaves evidence:
#   1. provenance  : hostname, rocm-smi, hipcc version, arch
#   2. static guard: scripts/xpm_lint_device_asm.sh over the emitted device asm
#                    (tiers 1-3 branch reach, tier 4 self-calls) -- the same
#                    check CI runs non-gating, here against a real ROCm 7.0.2
#   3. RUNTIME     : validation/mi250/derecursion_runtime.hip at the DEFAULT
#                    per-thread stack, which is the only place the defect shows
#
# SUBMIT — script mode:
#     qsub -A pepper_hep -n 1 -t 30 -q gpu_amd_mi250 --mode script \
#          validation/mi250/run_mi250.sh
#
# SUBMIT — interactive:
#     qsub -A pepper_hep -I -n 1 -t 30 -q gpu_amd_mi250
#     # then: bash validation/mi250/run_mi250.sh
#
# Knobs:
#     REPO_ROOT    repo checkout (default: resolved from this script's path)
#     ROCM_MODULE  module to load (default rocm/7.0.2 -- the version of record)
#     ROCM_ROOT    direct install path used if the module system is absent
#                  (default /soft/compilers/rocm/rocm-7.0.2)
#     OFFLOAD_ARCH default gfx90a
# ===========================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 30
#COBALT -q gpu_amd_mi250

# NOT -e: a failing check is a FINDING to be recorded, not a reason to abort
# the job and lose the rest of the evidence.
set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
ROCM_MODULE=${ROCM_MODULE:-rocm/7.0.2}
# Direct install path, used when the module system is not reachable (batch).
ROCM_ROOT=${ROCM_ROOT:-/soft/compilers/rocm/rocm-7.0.2}
OFFLOAD_ARCH=${OFFLOAD_ARCH:-gfx90a}

if [ ! -f "$REPO_ROOT/include/xp/dd_math.hpp" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/mi250/logs"
WORK="$REPO_ROOT/validation/mi250/work"
mkdir -p "$LOGDIR" "$WORK"
LOG="$LOGDIR/derecursion_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo " gfx90a de-recursion runtime verification"
echo " date      : $(date -Is)"
echo " host      : $(hostname)"
echo " repo      : $REPO_ROOT"
echo " commit    : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " log       : $LOG"
echo "==========================================================="

# ---------------------------------------------------- 0. toolchain + device
# Cobalt's batch shell is NON-INTERACTIVE, so the `module` bash function does
# not exist -- job 1000931 died here, and ~/.bashrc line 13 hit the same wall.
# The login shell gets the function from /etc/profile.d/modules.sh, so source
# it if it is missing.
if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles 2>/dev/null
  module load "$ROCM_MODULE" 2>&1 | sed 's/^/  /'
else
  echo "  (no module system in this shell -- using the direct path)"
fi

# Belt-and-braces. The rocm/7.0.2 modulefile does exactly three things, so if
# the module system is unavailable or loaded nothing, do them by hand rather
# than fail. Verified against `module show rocm/7.0.2`:
#     ROCM_PATH=/soft/compilers/rocm/rocm-7.0.2
#     PATH += $ROCM_PATH/bin        LD_LIBRARY_PATH += $ROCM_PATH/lib
if ! command -v hipcc >/dev/null 2>&1; then
  export ROCM_PATH="$ROCM_ROOT"
  export PATH="$ROCM_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}"
  echo "  fell back to ROCM_ROOT=$ROCM_ROOT"
fi

echo; echo "--- [0/3] provenance ---"
hipcc --version 2>&1 | head -5 | sed 's/^/  /'
rocm-smi --showproductname 2>&1 | head -12 | sed 's/^/  /'

if ! command -v hipcc >/dev/null 2>&1; then
  echo "FATAL: hipcc not on PATH after loading $ROCM_MODULE" >&2
  exit 2
fi

# ------------------------------------------------------------- 1. compile
# -O3 -ffp-contract=off matches the environment in the upstream report, and
# --save-temps=obj is what leaves the device .s for the static guard.
echo; echo "--- [1/3] compile (arch=$OFFLOAD_ARCH) ---"
rm -f "$WORK"/*.s "$WORK"/*.o "$WORK"/derecursion_runtime
set -x
hipcc -std=c++17 --offload-arch="$OFFLOAD_ARCH" -O3 -ffp-contract=off \
      --save-temps=obj -I"$REPO_ROOT/include" \
      "$REPO_ROOT/validation/mi250/derecursion_runtime.hip" \
      -o "$WORK/derecursion_runtime"
rc_compile=$?
set +x
if [ $rc_compile -ne 0 ]; then
  echo "FATAL: compile failed (rc=$rc_compile) -- nothing else can run." >&2
  exit 2
fi
echo "  ok; device asm:"
ls -la "$WORK"/*amdgcn-amd-amdhsa*.s 2>/dev/null | sed 's/^/    /'

# --------------------------------------------------------- 2. static guard
echo; echo "--- [2/3] static device-codegen guard (tiers 1-4) ---"
"$REPO_ROOT/scripts/xpm_lint_device_asm.sh" "$WORK"
rc_lint=$?
echo "  lint exit: $rc_lint  (0 = clean; tier 4 fires on any self-call)"

# -------------------------------------------------------------- 3. runtime
echo; echo "--- [3/3] RUNTIME on device, default stack ---"
"$WORK/derecursion_runtime"
rc_run=$?

# ----------------------------------------------------------------- verdict
echo
echo "==========================================================="
echo " compile : $rc_compile   static guard : $rc_lint   runtime : $rc_run"
if [ $rc_lint -eq 0 ] && [ $rc_run -eq 0 ]; then
  echo " VERDICT : PASS — the de-recursion fix holds at runtime on MI250X."
  echo "           This is the evidence the report's verification table lacks."
else
  echo " VERDICT : FAIL — see above. Report, do not fix in place (Rule 4)."
fi
echo " log     : $LOG"
echo "==========================================================="
[ $rc_lint -eq 0 ] && [ $rc_run -eq 0 ]
