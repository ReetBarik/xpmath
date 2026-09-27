#!/usr/bin/env bash
# ===========================================================================
# validation/mi300/run_mi300_sweep.sh — C12: produce and score the MI300
#                                       device sweep (CORE_PLAN C12)
# ===========================================================================
#
# WHAT IT DOES
#   1. Build both trees through scripts/xpm_build.sh --arch mi300 --no-test.
#      Host tree: g++, sweep_accuracy. Device tree: hipcc, -ffp-contract=off,
#      --offload-arch=gfx942 (confirmed via rocminfo on amdgpu00, 2026-09-26;
#      Marketing Name reports "AMD Radeon Graphics", ISA Name gfx942).
#   2. Run <build>/device/tests/sweep_device --out <raw.csv> on the GPU.
#      last_error() != 0 is a hard fail; the CSV must not be scored.
#   3. Score on the host binary:
#        sweep_accuracy --score-results <raw> --where mi300 --out <scored>
#      and, if the committed baseline exists, monotone-compare against it.
#
# Modules / queue (confirmed on JLSE, 2026-09-26):
#   modules : gcc/13.3.0 cmake/3.28.3 rocm/7.0.2
#   queue   : gpu_amd_mi300x  (nodes amdgpu00 / amdgpu01)
#
# Asm lint (TD-1 / TD-2 re-verify) is run on the login node BEFORE qsub —
# see the C12 STATUS block. This script does not invent a new mitigation.
#
# SUBMIT — script mode:
#     qsub -A pepper_hep -n 1 -t 360 -q gpu_amd_mi300x --mode script \
#          validation/mi300/run_mi300_sweep.sh
#
# Knobs:
#     REPO_ROOT      repo checkout (default: resolved from this script's path)
#     BUILD_DIR      build tree     (default: $REPO_ROOT/build-mi300-sweep)
#     RAW_OUT        raw limbs CSV  (default: $REPO_ROOT/validation/mi300/logs/sweep_mi300_raw.csv)
#     SCORED_OUT     scored CSV     (default: $REPO_ROOT/validation/mi300/logs/sweep_mi300_scored.csv)
# ===========================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 360
#COBALT -q gpu_amd_mi300x

set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build-mi300-sweep}
RAW_OUT=${RAW_OUT:-$REPO_ROOT/validation/mi300/logs/sweep_mi300_raw.csv}
SCORED_OUT=${SCORED_OUT:-$REPO_ROOT/validation/mi300/logs/sweep_mi300_scored.csv}
BASELINE=$REPO_ROOT/validation/sweep/sweep_baseline_mi300.csv.gz

if [ ! -f "$REPO_ROOT/include/xp/dd_math.hpp" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/mi300/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/mi300_sweep_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo " MI300 device sweep (C12)"
echo " date      : $(date -Is)"
echo " host      : $(hostname)"
echo " repo      : $REPO_ROOT"
echo " commit    : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " build dir : $BUILD_DIR"
echo " raw out   : $RAW_OUT"
echo " scored out: $SCORED_OUT"
echo " log       : $LOG"
echo " cobalt    : ${COBALT_JOBID:-none}"
echo "==========================================================="

# Cobalt's batch shell is non-interactive; the module function is not there.
if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles 2>/dev/null
  for m in gcc/13.3.0 cmake/3.28.3 rocm/7.0.2; do
    echo "  module load $m"
    module load "$m" || echo "  WARNING: module load $m failed"
  done
else
  echo "  WARNING: no module system in this shell -- using whatever is on PATH"
fi

echo
echo "--- provenance ---"
echo "  hostname : $(hostname)"
rocminfo 2>/dev/null | awk '/Marketing Name:|Name: +gfx/{print}' | head -12 | sed 's/^/    /' || true
hipcc --version 2>&1 | head -5 | sed 's/^/    /' || true
g++ --version 2>&1 | head -1 | sed 's/^/    /'

echo
echo "--- [1/3] build (xpm_build.sh --arch mi300 --no-test) ---"
bash "$REPO_ROOT/scripts/xpm_build.sh" --arch mi300 --build-dir "$BUILD_DIR" --no-test
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FATAL: xpm_build.sh exited $rc" >&2
  exit "$rc"
fi

DEV_BIN=$BUILD_DIR/device/tests/sweep_device
HOST_BIN=$BUILD_DIR/host/tests/sweep_accuracy
if [ ! -x "$DEV_BIN" ]; then
  echo "FATAL: device producer missing: $DEV_BIN" >&2
  exit 2
fi
if [ ! -x "$HOST_BIN" ]; then
  echo "FATAL: host scorer missing: $HOST_BIN" >&2
  exit 2
fi

echo
echo "--- [2/3] sweep_device on this GPU ---"
"$DEV_BIN" --out "$RAW_OUT"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FATAL: sweep_device exited $rc -- last_error() was nonzero; do not score $RAW_OUT" >&2
  exit "$rc"
fi

echo
echo "--- [3/3] host score --where mi300 ---"
score_args=(--ulp --score-results "$RAW_OUT" --where mi300 --out "$SCORED_OUT")
if [ -f "$BASELINE" ]; then
  score_args+=(--baseline "$BASELINE")
  echo "  monotone-comparing against $BASELINE"
else
  echo "  no committed mi300 baseline yet; scoring only"
fi
"$HOST_BIN" "${score_args[@]}"
rc=$?
echo
echo "scorer exit: $rc"
echo "raw    : $RAW_OUT"
echo "scored : $SCORED_OUT"
echo "log    : $LOG"
exit "$rc"
