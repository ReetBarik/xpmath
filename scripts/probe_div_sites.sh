#!/usr/bin/env bash
# ===========================================================================
# Call-site attribution for the 28 regressed rows.
#
# Builds a SECOND probe binary with -DPROBE_SITES -g -no-pie, in which the
# four `divide` hooks are noinline and record __builtin_return_address(0).
# Runs --sites, then resolves every address with `addr2line -i`, which unwinds
# the inline chain so the answer names the qf_complex.hpp line that issued the
# divide, not the hook that intercepted it.
#
# WHY A SECOND BINARY.  Forbidding the inline of `divide` also forbids GCC
# from contracting a multiply inside divide with an add in the caller.  That
# could move the last bit, so attribution is kept out of the binary that
# produced the measurements.  This script then runs --verify on the sites
# binary and requires 0 mismatches: if the noinline build did not reproduce
# the same 28 before/after ulps, its attribution is not about the same code
# and the script fails rather than reporting.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INST="${1:-/tmp/xp_probe_inst_sites}"
OUT="${2:-/tmp/probe_div_sites}"

bash "$ROOT/scripts/probe_div_downstream.sh" "$INST" /dev/null >/dev/null 2>&1 || true

# probe_div_downstream.sh already produced the instrumented tree; rebuild the
# binary from it with the sites flags.
if [ ! -f "$INST/xp/qf_math.hpp" ]; then
  echo "probe_div_sites.sh: instrumented tree $INST missing" >&2
  exit 1
fi

module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0        >/dev/null 2>&1 || true

g++ -O2 -g -no-pie -std=c++17 -fext-numeric-literals -DPROBE_SITES \
    -I "$INST" \
    "$ROOT/scripts/probe_div_downstream.cpp" -o "$OUT" \
    -lmpc -lmpfr -lgmp

echo "== the sites binary must reproduce the measurement binary's numbers =="
if ! "$OUT" --verify | tail -1 | grep -q ' 0 mismatch'; then
  echo "probe_div_sites.sh: the PROBE_SITES build does NOT reproduce the 28 rows;" >&2
  echo "  its call-site attribution would not be about the measured code." >&2
  "$OUT" --verify | tail -3 >&2
  exit 1
fi
"$OUT" --verify | tail -1

echo
echo "== call sites, inline chain unwound =="
"$OUT" --sites | grep '^SITE ' | while read -r _ be _ op pt addr fired count; do
  chain=$(addr2line -i -f -C -e "$OUT" "$addr" 2>/dev/null \
          | paste - - | sed 's|.*/||' | tr '\n' '|' | sed 's/|$//')
  printf '%-3s c %-6s %-5s %-7s %-9s  %s\n' "$be" "$op" "$pt" "$fired" "$count" "$chain"
done
