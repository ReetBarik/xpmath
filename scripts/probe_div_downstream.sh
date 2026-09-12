#!/usr/bin/env bash
# ===========================================================================
# Build scripts/probe_div_downstream.cpp against an INSTRUMENTED COPY of
# include/xp/.
#
# WHY A COPY.  The probe has to see three different real divides -- the
# pre-lift one, the shipped lifted one, and an exactly-rounded one -- from
# INSIDE the complex headers, which call `divide` by name.  There is no way to
# substitute an overload from outside a header that already resolved the call.
# So the four *_math.hpp copies get their `divide` DEFINITION renamed to
# `<be>_divide_shipped`, which leaves the forward declaration at the top of
# each header (dd:84, ff:96, tf:84, qf:95) undefined -- and the probe .cpp
# supplies the definition.  Every internal call in the complex headers then
# lands in the probe's hook.
#
# The shipped headers are NEVER modified: `git status` stays clean across a
# probe run.  The copy is regenerated from include/xp/ on every invocation, so
# it cannot drift from what ships.
#
# The rename is asserted to have applied EXACTLY ONCE per header.  A sed that
# silently matched nothing would leave `divide` defined twice (shipped +
# probe), which is an ODR violation the compiler is not required to diagnose.
#
# -I points ONLY at the instrumented tree, never at include/, so a header the
# copy missed is a hard error rather than a silent fallback to the shipped one.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INST="${1:-/tmp/xp_probe_inst}"
OUT="${2:-/tmp/probe_div_downstream}"

rm -rf "$INST"
mkdir -p "$INST/xp"
cp "$ROOT"/include/xp/*.hpp "$INST/xp/"

rename() {   # rename <header> <type> <newname>
  local f="$INST/xp/$1" t="$2" n="$3"
  local pat="^XPMATH_INLINE_FUNCTION $t divide($t a, $t b) {"
  local hits
  hits=$(grep -c "$pat" "$f" || true)
  if [ "$hits" != "1" ]; then
    echo "probe_div_downstream.sh: expected exactly 1 divide definition in $1, found $hits" >&2
    exit 1
  fi
  sed -i "s|$pat|XPMATH_INLINE_FUNCTION $t $n($t a, $t b) {|" "$f"
}

rename dd_math.hpp DoubleDouble dd_divide_shipped
rename ff_math.hpp FloatFloat   ff_divide_shipped
rename tf_math.hpp TripleFloat  tf_divide_shipped
rename qf_math.hpp QuadFloat    qf_divide_shipped

module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0        >/dev/null 2>&1 || true

g++ -O2 -std=c++17 -fext-numeric-literals \
    -I "$INST" \
    "$ROOT/scripts/probe_div_downstream.cpp" -o "$OUT" \
    -lmpc -lmpfr -lgmp

echo "built $OUT against instrumented headers in $INST"
