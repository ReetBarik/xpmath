#!/usr/bin/env bash
# ===========================================================================
# validation/gb300/run_gb300_sweep.sh — produce the GB300 raw device sweep
# ===========================================================================
#
# WHAT IT DOES (on the GB300 node, aarch64)
#   1. Build the DEVICE tree only: scripts/xpm_build.sh --arch gb300 --no-test
#      (nvcc_wrapper, -arch=sm_103). aarch64 has no __float128, so the host
#      oracle tree cannot be built here.
#   2. Run <build>/device/tests/sweep_device --out <raw.csv> on the GPU.
#      last_error() != 0 is a hard fail; the CSV must not be scored.
#
# IT DOES NOT SCORE. Scoring needs sweep_accuracy (MPFR/MPC, __float128), so it
# runs on an x86 login node:
#     <host-build>/host/tests/sweep_accuracy --ulp --score-results <raw> \
#         --where gb300 --out <scored>
# There is deliberately no committed gb300 baseline and no sweep_device_gate_gb300
# ctest target; whether GB300 becomes a gate is the user's decision.
#
# Modules (scripts/xpm_build.sh ARCH_MODULES[gb300]): gcc/11.2.0 cmake/3.22.1
# cuda/13.3.1.
#
# RUN (on the node):
#     ssh <gb300-host> 'bash -lc "validation/gb300/run_gb300_sweep.sh"'
#
# Knobs:
#     REPO_ROOT   repo checkout (default: resolved from this script's path)
#     BUILD_DIR   build tree    (default: $REPO_ROOT/build-gb300-sweep)
#     RAW_OUT     raw limbs CSV (default: $REPO_ROOT/validation/gb300/logs/sweep_gb300_raw.csv)
# ===========================================================================

set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build-gb300-sweep}
RAW_OUT=${RAW_OUT:-$REPO_ROOT/validation/gb300/logs/sweep_gb300_raw.csv}

# TIER=strict|fast|both (default strict). "both" runs the strict pass, then the
# fast pass, and exits with the first nonzero exit. The fast pass builds
# sweep_device_fast and writes RAW_OUT with a _fast suffix. GB300 has no
# committed baseline and cannot score here, so the printed scoring command (run
# on the x86 login node) is score-only for both tiers.
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
TIER_SUF=""
if [ "$TIER" = fast ]; then
  TIER_SUF="_fast"
  RAW_OUT=${RAW_OUT%.csv}_fast.csv
fi

if [ ! -f "$REPO_ROOT/include/xp/dd_math.hpp" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/gb300/logs"
mkdir -p "$LOGDIR" "$(dirname "$RAW_OUT")"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/gb300_sweep${TIER_SUF}_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo " GB300 device sweep (raw only; score on the login node)"
echo " date      : $(date -Is)"
echo " host      : $(hostname) ($(uname -m))"
echo " repo      : $REPO_ROOT"
echo " commit    : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " build dir : $BUILD_DIR"
echo " raw out   : $RAW_OUT"
echo " log       : $LOG"
echo "==========================================================="

if [ "$(uname -m)" != aarch64 ]; then
  echo "FATAL: GB300 builds on the node only (aarch64); this is $(uname -m)." >&2
  exit 2
fi

if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles 2>/dev/null
  for m in gcc/11.2.0 cmake/3.22.1 cuda/13.3.1; do
    echo "  module load $m"
    module load "$m" || echo "  WARNING: module load $m failed"
  done
else
  echo "  WARNING: no module system in this shell -- using whatever is on PATH"
fi

echo
echo "--- provenance ---"
nvidia-smi 2>&1 | head -20 | sed 's/^/    /' || true
nvcc --version 2>&1 | sed 's/^/    /' || true
g++ --version 2>&1 | head -1 | sed 's/^/    /'

echo
echo "--- [1/2] build (xpm_build.sh --arch gb300 --no-test) ---"
bash "$REPO_ROOT/scripts/xpm_build.sh" --arch gb300 --build-dir "$BUILD_DIR" --no-test
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FATAL: xpm_build.sh exited $rc" >&2
  exit "$rc"
fi

DEV_BIN=$BUILD_DIR/device/tests/sweep_device
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

echo
echo "--- [2/2] sweep_device on this GPU ---"
SWEEP_LOG="$LOGDIR/sweep_device${TIER_SUF}_${STAMP}.out"
"$DEV_BIN" --out "$RAW_OUT" >"$SWEEP_LOG" 2>&1
rc=$?
cat "$SWEEP_LOG"
if [ "$rc" -ne 0 ]; then
  echo "sweep_device exited $rc -- last_error() was nonzero; do not score $RAW_OUT" >&2
  echo
  echo "--- isolating each remaining kernel in its own process ---"
  echo "    one illegal access poisons every later launch in the same process."

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

  ISO="$LOGDIR/isolate${TIER_SUF}_${STAMP}"
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
    echo "not a result: the full-sweep file $RAW_OUT stops before the first fault."
    exit 3
  fi

  STITCHED="$LOGDIR/sweep_gb300_stitched${TIER_SUF}.csv"
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
echo "sweep_device exit 0 -- raw device output written (NOT scored)."
echo "raw : $RAW_OUT"
echo "log : $LOG"
echo "score on an x86 login node with:"
echo "  <host-build>/host/tests/sweep_accuracy --tier $TIER --ulp --score-results $RAW_OUT --where gb300 --out /tmp/gb300_scored${TIER_SUF}.csv"
exit 0
