#!/usr/bin/env bash
# =============================================================================
# validation/a100/run_a100.sh — S1 step 2: RUN on the A100 (Cobalt)
# =============================================================================
#
# Compiles nothing. validation/a100/build_login.sh must have completed on the
# login node first. This script only executes the already-built binaries, so the
# GPU reservation is spent running.
#
# Runs, in order (ctest first — it is the primary S1 evidence):
#   0. environment + device provenance (hostname, nvidia-smi, Kokkos config)
#   1. the full ctest suite, serially (-j1: one GPU, no launch contention)
#   2. all six demos at the byte-identical-gate arguments
#      (--batch 1000000 --repeats 5 --seed 12345)
#
# Everything is tee-d into validation/a100/logs/run/.
#
# SUBMIT — script mode (automatable):
#     qsub -A pepper_hep -t 120 -n 1 -q gpu_a100 --mode script \
#          validation/a100/run_a100.sh
#
# SUBMIT — interactive (the usual habit):
#     qsub -A pepper_hep -I -t 120 -n 1 -q gpu_a100
#     # then, in the interactive shell:
#     bash validation/a100/run_a100.sh
#
# Knobs (environment variables; in script mode pass them with
#  `qsub --env REPO_BUILD=/path:REPO_ROOT=/path ...`):
#     REPO_ROOT     repo checkout    (default: resolved from this script's path)
#     REPO_BUILD    build dir        (default ~/kokkos-ep-build-a100)
#     DEMO_BATCH    demo --batch     (default 1000000)
#     DEMO_REPEATS  demo --repeats   (default 5)
#     DEMO_SEED     demo --seed      (default 12345)
#     CTEST_TIMEOUT per-test seconds (default 1800)
# =============================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 120
#COBALT -q gpu_a100

set -uo pipefail   # NOT -e: a failing test is a FINDING, not a reason to abort

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
REPO_BUILD=${REPO_BUILD:-$HOME/kokkos-ep-build-a100}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-$HOME/kokkos-install-cuda-sm80-quadmath}
DEMO_BATCH=${DEMO_BATCH:-1000000}
DEMO_REPEATS=${DEMO_REPEATS:-5}
DEMO_SEED=${DEMO_SEED:-12345}
CTEST_TIMEOUT=${CTEST_TIMEOUT:-1800}

if [ ! -f "$REPO_ROOT/CMakeLists.txt" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/a100/logs/run"
mkdir -p "$LOGDIR"

say() { printf '\n=== %s\n' "$*"; }

# --- 0. modules + provenance -----------------------------------------------
module use /soft/modulefiles   >/dev/null 2>&1
module load gcc/13.3.0         >/dev/null 2>&1
module load cmake/3.28.3       >/dev/null 2>&1
module load cuda/12.9.1        >/dev/null 2>&1

ulimit -s 131072 2>/dev/null || true
export LD_LIBRARY_PATH="$KOKKOS_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

{
  say "provenance"
  echo "date       : $(date -Is)"
  echo "hostname   : $(hostname)"
  echo "COBALT_JOBID  : ${COBALT_JOBID:-<none / interactive>}"
  echo "COBALT_PARTNAME: ${COBALT_PARTNAME:-<unset>}"
  echo "REPO_ROOT  : $REPO_ROOT"
  echo "REPO_BUILD : $REPO_BUILD"
  echo "repo HEAD  : $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  say "nvidia-smi"
  nvidia-smi 2>&1
  say "installed Kokkos config"
  grep -E '^#define KOKKOS_(ENABLE|ARCH|VERSION)' \
    "$KOKKOS_PREFIX/include/KokkosCore_config.h" 2>&1
} 2>&1 | tee "$LOGDIR/00_env.log"

# Hard gate: if this landed somewhere without a GPU (e.g. Cobalt script mode
# executed the script on a service node rather than the compute node), stop now
# and say so — a Serial-fallback run would be worthless S1 evidence.
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "FATAL: no NVIDIA device visible on $(hostname)." | tee -a "$LOGDIR/00_env.log"
  echo "       Cobalt script mode may have run this off the compute node." \
    | tee -a "$LOGDIR/00_env.log"
  echo "       Re-run via the interactive path (qsub -I ...) and report this." \
    | tee -a "$LOGDIR/00_env.log"
  exit 3
fi

# --- 1. ctest ---------------------------------------------------------------
# -V so each test's "Execution space: <name>" banner is captured; that banner is
# how the analysis half confirms DefaultExecutionSpace == Cuda rather than Serial.
# -j1 because there is one GPU.
say "ctest (serial, verbose)"
ctest --test-dir "$REPO_BUILD" -V -j1 --timeout "$CTEST_TIMEOUT" \
  2>&1 | tee "$LOGDIR/01_ctest_verbose.log"
CTEST_RC=${PIPESTATUS[0]}
echo "ctest exit code: $CTEST_RC" | tee "$LOGDIR/01_ctest_rc.log"

# Compact pass/fail table for the S1 STATUS block.
say "ctest summary"
ctest --test-dir "$REPO_BUILD" -N 2>&1 | tee "$LOGDIR/01_ctest_list.log"
grep -E '^\s*[0-9]+/[0-9]+ Test' "$LOGDIR/01_ctest_verbose.log" \
  2>&1 | tee "$LOGDIR/01_ctest_summary.log"

# --- 2. demos ---------------------------------------------------------------
# Byte-identical-gate arguments (UPSTREAM_PLAN "Common context"): the accuracy
# columns here are what gets diffed against README Section 2's Serial numbers.
say "demos (--batch $DEMO_BATCH --repeats $DEMO_REPEATS --seed $DEMO_SEED)"
DEMOS="kokkos_ep_demo kokkos_ep_demo_complex \
       kokkos_ep_demo_ff kokkos_ep_demo_ff_complex \
       kokkos_ep_demo_qf kokkos_ep_demo_qf_complex"

for d in $DEMOS; do
  if [ ! -x "$REPO_BUILD/$d" ]; then
    echo "MISSING BINARY: $REPO_BUILD/$d (did it fail to compile?)" \
      | tee "$LOGDIR/02_${d}.log"
    continue
  fi
  say "$d"
  "$REPO_BUILD/$d" --batch "$DEMO_BATCH" --repeats "$DEMO_REPEATS" \
                   --seed "$DEMO_SEED" 2>&1 | tee "$LOGDIR/02_${d}.log"
  echo "exit code: ${PIPESTATUS[0]}" | tee -a "$LOGDIR/02_${d}.log"
done

# src/bench_cost.cpp is deliberately NOT run: it has no Views and no
# parallel_for (pure host timing loops), so it produces no device evidence.

say "done"
cat <<EOF

=============================================================================
RUN COMPLETE.
  ctest exit code : $CTEST_RC   (non-zero is a FINDING for S1 STATUS, not a bug to fix)
  logs            : $LOGDIR

Hand back to the analysis session: commit validation/a100/logs/ (or paste the
contents of 00_env.log, 01_ctest_verbose.log, 01_ctest_summary.log and the six
02_*.log files), plus validation/a100/logs/build/contraction_flags.log and
kokkos_config_defines.log from the login-node build.
=============================================================================
EOF
