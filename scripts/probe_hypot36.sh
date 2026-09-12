#!/usr/bin/env bash
# ===========================================================================
# Build and run scripts/probe_hypot36.cpp.
#
# It #includes scripts/sweep_accuracy.cpp, so it needs exactly what the sweep
# needs: the shipped include/xp/ headers, libquadmath, and (for the exact
# quotient and the format floor) MPFR.  No instrumented header copy is used --
# this probe measures the shipped code as built.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/probe_hypot36}"

module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0        >/dev/null 2>&1 || true

g++ -O2 -std=c++17 -fext-numeric-literals \
    -I "$ROOT/include" -I "$ROOT/scripts" \
    -DXPMATH_HAVE_MPFR=1 \
    "$ROOT/scripts/probe_hypot36.cpp" -o "$OUT" \
    -lquadmath -lmpfr -lgmp

"$OUT"
