#!/usr/bin/env bash
# ===========================================================================
# validation/b200/run_b200_sweep.sh — C11: produce and score the B200 device
#                                     sweep (CORE_PLAN C11)
# ===========================================================================
#
# WHAT IT DOES
#   1. Build both trees through scripts/xpm_build.sh --arch b200 --no-test.
#      Host tree: g++, sweep_accuracy. Device tree: nvcc_wrapper, -arch=sm_100.
#   2. Run <build>/device/tests/sweep_device --out <raw.csv> on the GPU.
#      last_error() != 0 is a hard fail; the CSV must not be scored.
#   3. Score on the host binary:
#        sweep_accuracy --score-results <raw> --where b200 --out <scored>
#      and, if the committed baseline exists, monotone-compare against it.
#
# Modules / queue (confirmed on JLSE login, 2026-09-26):
#   modules : gcc/13.3.0 cmake/3.28.3 cuda/12.9.1
#             (cuda/12.9.1 nvcc 12.9.86 advertises sm_100; do not bump to
#              cuda/13.x unless this node rejects 12.9.1 — record that in
#              the STATUS block if it happens)
#   queue   : gpu_b200  (node blackwell00)
#
# This is NOT the CUDAFP128Kokkos branch. sm_100 is only the architecture
# flag for the portable backends.
#
# SUBMIT — script mode:
#     qsub -A pepper_hep -n 1 -t 360 -q gpu_b200 --mode script \
#          validation/b200/run_b200_sweep.sh
#
# Knobs:
#     REPO_ROOT      repo checkout (default: resolved from this script's path)
#     BUILD_DIR      build tree     (default: $REPO_ROOT/build-b200-sweep)
#     RAW_OUT        raw limbs CSV  (default: $REPO_ROOT/validation/b200/logs/sweep_b200_raw.csv)
#     SCORED_OUT     scored CSV     (default: $REPO_ROOT/validation/b200/logs/sweep_b200_scored.csv)
# ===========================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 360
#COBALT -q gpu_b200

set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build-b200-sweep}
RAW_OUT=${RAW_OUT:-$REPO_ROOT/validation/b200/logs/sweep_b200_raw.csv}
SCORED_OUT=${SCORED_OUT:-$REPO_ROOT/validation/b200/logs/sweep_b200_scored.csv}
BASELINE=$REPO_ROOT/validation/sweep/sweep_baseline_b200.csv.gz

if [ ! -f "$REPO_ROOT/include/xp/dd_math.hpp" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/b200/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/b200_sweep_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo " B200 device sweep (C11)"
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
  for m in gcc/13.3.0 cmake/3.28.3 cuda/12.9.1; do
    echo "  module load $m"
    module load "$m" || echo "  WARNING: module load $m failed"
  done
else
  echo "  WARNING: no module system in this shell -- using whatever is on PATH"
fi

echo
echo "--- provenance ---"
echo "  hostname : $(hostname)"
nvidia-smi 2>&1 | head -20 | sed 's/^/    /' || true
nvcc --version 2>&1 | sed 's/^/    /' || true
g++ --version 2>&1 | head -1 | sed 's/^/    /'

echo
echo "--- [1/3] build (xpm_build.sh --arch b200 --no-test) ---"
bash "$REPO_ROOT/scripts/xpm_build.sh" --arch b200 --build-dir "$BUILD_DIR" --no-test
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
echo "--- [3/3] host score --where b200 ---"
score_args=(--ulp --score-results "$RAW_OUT" --where b200 --out "$SCORED_OUT")
if [ -f "$BASELINE" ]; then
  score_args+=(--baseline "$BASELINE")
  echo "  monotone-comparing against $BASELINE"
else
  echo "  no committed b200 baseline yet; scoring only"
fi
"$HOST_BIN" "${score_args[@]}"
rc=$?
echo
echo "scorer exit: $rc"
echo "raw    : $RAW_OUT"
echo "scored : $SCORED_OUT"
echo "log    : $LOG"
exit "$rc"
