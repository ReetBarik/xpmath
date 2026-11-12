#!/usr/bin/env bash
# ===========================================================================
# validation/a100/run_a100_sweep.sh — C7: produce and score the A100 device
#                                     sweep (CORE_PLAN C7)
# ===========================================================================
#
# Do NOT edit validation/a100/run_a100.sh for this. That script is the S8b
# wrapper-validation job (ctest of the then-suite on gpu_a100). This one is
# the accuracy record: sweep_device on the card, host MPFR/MPC scoring, then
# a monotone comparison against validation/sweep/sweep_baseline_a100.csv.gz.
#
# WHAT IT DOES
#   1. Build both trees through scripts/xpm_build.sh --arch a100 --no-test.
#      Host tree: g++, sweep_accuracy. Device tree: nvcc_wrapper, sweep_device.
#   2. Run <build>/device/tests/sweep_device --out <raw.csv> on the GPU.
#      last_error() != 0 is a hard fail; the CSV must not be scored.
#   3. Score on the host binary:
#        sweep_accuracy --score-results <raw> --where a100 --out <scored>
#      and, if the committed baseline exists, monotone-compare against it.
#
# Job 1004542 (2026-09-27, gpu07) is the run of record. It replaced 1001685
# after DD complex atan/atanh stopped returning NaN on sm_80. 1001685 remains
# the pre-fix raw. This script is the submit-able form of that recipe.
#
# SUBMIT — script mode:
#     qsub -A pepper_hep -n 1 -t 360 -q gpu_a100 --mode script \
#          validation/a100/run_a100_sweep.sh
#
# SUBMIT — interactive (how 1001685 actually ran):
#     qsub -A pepper_hep -I -n 1 -t 360 -q gpu_a100
#     # then: bash validation/a100/run_a100_sweep.sh
#
# Knobs:
#     REPO_ROOT      repo checkout (default: resolved from this script's path)
#     BUILD_DIR      build tree     (default: $REPO_ROOT/build-a100-sweep)
#     RAW_OUT        raw limbs CSV  (default: $REPO_ROOT/validation/a100/logs/sweep_a100_raw.csv)
#     SCORED_OUT     scored CSV     (default: $REPO_ROOT/validation/a100/logs/sweep_a100_scored.csv)
# ===========================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 360
#COBALT -q gpu_a100

set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build-a100-sweep}
RAW_OUT=${RAW_OUT:-$REPO_ROOT/validation/a100/logs/sweep_a100_raw.csv}
SCORED_OUT=${SCORED_OUT:-$REPO_ROOT/validation/a100/logs/sweep_a100_scored.csv}
BASELINE=$REPO_ROOT/validation/sweep/sweep_baseline_a100.csv.gz

# TIER=strict|fast|both (default strict, so a plain run behaves as before).
# "both" runs the strict pass, then the fast pass, and exits with the first
# nonzero exit. The fast pass builds sweep_device_fast, writes RAW_OUT and
# SCORED_OUT with a _fast suffix and scores against sweep_baseline_fast_a100.
# Each pass also writes <raw>.exits ("absolute=<n>" from --register and
# "monotone=<n>"), the file scripts/p3_tier_gate.py reads.
TIER=${TIER:-strict}
case "$TIER" in
  strict|fast) ;;
  both)
    rc_both=0
    for _t in strict fast; do
      TIER=$_t bash "${BASH_SOURCE[0]:-$0}"
      _rc=$?
      if [ "$rc_both" -eq 0 ]; then rc_both=$_rc; fi
    done
    exit "$rc_both"
    ;;
  *) echo "FATAL: TIER must be strict, fast or both (got '$TIER')" >&2; exit 2 ;;
esac

if [ ! -f "$REPO_ROOT/include/xp/dd_math.hpp" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

TIER_SUF=""
REGISTER=$REPO_ROOT/validation/sweep/open_defects.txt
if [ "$TIER" = fast ]; then
  TIER_SUF="_fast"
  RAW_OUT=${RAW_OUT%.csv}_fast.csv
  SCORED_OUT=${SCORED_OUT%.csv}_fast.csv
  BASELINE=$REPO_ROOT/validation/sweep/sweep_baseline_fast_a100.csv.gz
fi

LOGDIR="$REPO_ROOT/validation/a100/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/a100_sweep${TIER_SUF}_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo " A100 device sweep (C7)"
echo " date      : $(date -Is)"
echo " host      : $(hostname)"
echo " repo      : $REPO_ROOT"
echo " commit    : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " build dir : $BUILD_DIR"
echo " raw out   : $RAW_OUT"
echo " scored out: $SCORED_OUT"
echo " log       : $LOG"
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
nvidia-smi 2>&1 | head -15 | sed 's/^/    /' || true
nvcc --version 2>&1 | sed 's/^/    /' || true
g++ --version 2>&1 | head -1 | sed 's/^/    /'

echo
echo "--- [1/3] build (xpm_build.sh --arch a100 --no-test) ---"
bash "$REPO_ROOT/scripts/xpm_build.sh" --arch a100 --build-dir "$BUILD_DIR" --no-test
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FATAL: xpm_build.sh exited $rc" >&2
  exit "$rc"
fi

DEV_BIN=$BUILD_DIR/device/tests/sweep_device
HOST_BIN=$BUILD_DIR/host/tests/sweep_accuracy
if [ "$TIER" = fast ]; then
  echo "--- building sweep_device_fast (EXCLUDE_FROM_ALL target) ---"
  nice -n 10 cmake --build "$BUILD_DIR/device" --target sweep_device_fast -j"${BUILD_JOBS:-8}"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: building sweep_device_fast exited $rc" >&2
    exit "$rc"
  fi
  DEV_BIN=$BUILD_DIR/device/tests/sweep_device_fast
fi
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
echo "--- [3/3] host score --where a100 ---"
score_args=(--tier "$TIER" --ulp --score-results "$RAW_OUT" --where a100 --out "$SCORED_OUT")
if [ -f "$BASELINE" ]; then
  score_args+=(--baseline "$BASELINE")
  echo "  monotone-comparing against $BASELINE"
else
  echo "  no committed a100 baseline yet; scoring only"
fi
"$HOST_BIN" "${score_args[@]}"
rc=$?
scorer_rc=$rc
echo
echo "--- gate exits (TIER=$TIER) -> ${RAW_OUT}.exits ---"
"$HOST_BIN" --tier "$TIER" --ulp --register "$REGISTER" \
    --score-results "$RAW_OUT" --where a100 > "$LOGDIR/gate_absolute_${TIER}_${STAMP}.log" 2>&1
abs_rc=$?
mono_rc=-1   # -1 = monotone gate skipped (no baseline file)
if [ -f "$BASELINE" ]; then
  "$HOST_BIN" --tier "$TIER" --score-results "$RAW_OUT" --where a100 \
      --baseline "$BASELINE" > "$LOGDIR/gate_monotone_${TIER}_${STAMP}.log" 2>&1
  mono_rc=$?
fi
printf 'absolute=%d\nmonotone=%d\n' "$abs_rc" "$mono_rc" > "${RAW_OUT}.exits"
echo "  absolute (--register) exit: $abs_rc"
echo "  monotone (--baseline) exit: $mono_rc"
rc=$scorer_rc
if [ "$rc" -eq 0 ]; then rc=$abs_rc; fi
if [ "$rc" -eq 0 ] && [ "$mono_rc" -gt 0 ]; then rc=$mono_rc; fi
echo
echo "scorer exit: $rc"
echo "raw    : $RAW_OUT"
echo "scored : $SCORED_OUT"
echo "log    : $LOG"
exit "$rc"
