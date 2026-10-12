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
SWEEP_LOG="$LOGDIR/sweep_device_${STAMP}.out"
"$DEV_BIN" --out "$RAW_OUT" >"$SWEEP_LOG" 2>&1
rc=$?
cat "$SWEEP_LOG"
if [ "$rc" -ne 0 ]; then
  echo "sweep_device exited $rc -- last_error() was nonzero; do not score $RAW_OUT" >&2
  echo
  echo "--- isolating each remaining kernel in its own process ---"
  echo "    one illegal access poisons every later launch in the same process."
  echo "    a kernel that exits 0 here is not scored unless every kernel from"
  echo "    the first fault onward also exits 0."

  # Same order as scripts/sweep_device.cpp: op-major, backend-minor, real then complex.
  real_ops=(add sub mul div sqrt abs exp log exp2 exp10 expm1 log2 log10 log1p \
            sin cos tan asin acos atan sinh cosh tanh acosh asinh atanh \
            pow hypot fmod remainder copysign fmax fmin fdim fma \
            ceil floor round trunc)
  complex_ops=(add sub mul div abs conj sqrt exp log log10 \
               sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh \
               pow polar)
  backends=(DD FF QF TF)
  keys=()
  for op in "${real_ops[@]}"; do
    for be in "${backends[@]}"; do keys+=("${be}:r:${op}"); done
  done
  for op in "${complex_ops[@]}"; do
    for be in "${backends[@]}"; do keys+=("${be}:c:${op}"); done
  done

  fault=$(grep -E 'sweep_device: first fault backend=' "$SWEEP_LOG" | tail -1 || true)
  start=0
  if [ -n "$fault" ]; then
    be=$(sed -n 's/.*backend=\([^ ]*\).*/\1/p' <<<"$fault")
    kind=$(sed -n 's/.*kind=\([^ ]*\).*/\1/p' <<<"$fault")
    op=$(sed -n 's/.*op=\([^ ]*\).*/\1/p' <<<"$fault")
    needle="${be}:${kind}:${op}"
    for i in "${!keys[@]}"; do
      if [ "${keys[$i]}" = "$needle" ]; then start=$i; break; fi
    done
    echo "  first fault: $needle  (resume index $start / ${#keys[@]})"
  else
    echo "  no 'first fault' line; re-running every kernel"
  fi

  ISO="$LOGDIR/isolate_${STAMP}"
  mkdir -p "$ISO"
  faults=()
  for ((i = start; i < ${#keys[@]}; i++)); do
    key=${keys[$i]}
    one="$ISO/${key//:/_}.csv"
    "$DEV_BIN" --only "$key" --out "$one"
    one_rc=$?
    if [ "$one_rc" -ne 0 ]; then
      echo "  FAULT $key exit $one_rc"
      faults+=("$key")
    else
      echo "  ok    $key"
    fi
  done

  if [ "${#faults[@]}" -ne 0 ]; then
    echo
    echo "kernels that faulted in their own process (${#faults[@]}):"
    printf '  %s\n' "${faults[@]}"
    echo "not scoring. the full-sweep file $RAW_OUT stops before the first fault"
    echo "and is not a result."
    exit 3
  fi

  STITCHED="$LOGDIR/sweep_b200_stitched.csv"
  {
    awk 'BEGIN{p=1} /^[^#]/ && !/^backend,/ {p=0} p' "$RAW_OUT"
    if [ "$start" -gt 0 ]; then
      awk '!/^#/ && !/^backend,/' "$RAW_OUT"
    fi
    for ((i = start; i < ${#keys[@]}; i++)); do
      key=${keys[$i]}
      awk '!/^#/ && !/^backend,/' "$ISO/${key//:/_}.csv"
    done
  } > "$STITCHED"
  echo
  echo "every kernel from the fault onward exited 0; stitched $STITCHED"
  RAW_OUT=$STITCHED
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
