#!/usr/bin/env bash
# ===========================================================================
# Build scripts/probe_b1_sqrt_sq.cpp against the same INSTRUMENTED copy of
# include/xp/ that probe_div_downstream.sh produces -- qf_math.hpp's `divide`
# definition renamed to qf_divide_shipped, so the probe can supply `divide`
# and switch the real divide from inside qf::sqrt's Heron iteration.
#
# The shipped headers are never modified.  -I points only at the instrumented
# tree, so a missed header is a hard error rather than a silent fallback.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INST="${1:-/tmp/xp_probe_inst_b1}"
OUT="${2:-/tmp/probe_b1_sqrt_sq}"

bash "$ROOT/scripts/probe_div_downstream.sh" "$INST" /dev/null >/dev/null 2>&1 || true
if [ ! -f "$INST/xp/qf_math.hpp" ]; then
  echo "probe_b1_sqrt_sq.sh: instrumented tree $INST missing" >&2
  exit 1
fi

module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0        >/dev/null 2>&1 || true

g++ -O2 -std=c++17 -fext-numeric-literals -I "$INST" \
    "$ROOT/scripts/probe_b1_sqrt_sq.cpp" -o "$OUT" -lmpfr -lgmp

echo "built $OUT against instrumented headers in $INST"
