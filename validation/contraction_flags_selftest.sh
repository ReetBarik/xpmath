#!/usr/bin/env bash
# ===========================================================================
# validation/contraction_flags_selftest.sh — prove the contraction guard fails
# ===========================================================================
#
# Same contract as validation/gate_selftest.sh: a negative assertion that has
# only ever been observed passing is indistinguishable from one that cannot
# fail. So the guard is run against deliberately poisoned manifests, built in
# the BUILD directory and never in validation/, and is required to fail on each.
#
# Usage: contraction_flags_selftest.sh <build-tests-dir> <source-tests-dir>
# ===========================================================================
set -uo pipefail

BUILD_TESTS="${1:?usage: $0 <build-tests-dir> <source-tests-dir>}"
SRC_TESTS="${2:?usage: $0 <build-tests-dir> <source-tests-dir>}"
CHECK="${SRC_TESTS}/check_contraction_flags.cmake"
REAL="${BUILD_TESTS}/contraction_manifest.txt"
WORK="${BUILD_TESTS}/contraction_selftest"

[ -r "$REAL" ]  || { echo "FATAL: no manifest at $REAL"; exit 2; }
[ -r "$CHECK" ] || { echo "FATAL: no checker at $CHECK"; exit 2; }

rm -rf "$WORK"; mkdir -p "$WORK"
n_off=$(grep -c '^off|' "$REAL"); n_on=$(grep -c '^on|' "$REAL")
rc=0

run() {  # run <manifest> <expect-off> <expect-on>  -> echoes exit code
  cmake -DMANIFEST="$1" -DEXPECT_OFF="$2" -DEXPECT_ON="$3" -P "$CHECK" >/dev/null 2>&1
  echo $?
}
expect() {  # expect <label> <got> <want>
  if [ "$2" = "$3" ]; then echo "  ok    $1 (exit $2)"
  else echo "  FAIL  $1: exit $2, expected $3"; rc=1; fi
}

echo "=== contraction flag guard self-test ==="
echo "manifest: $n_off off, $n_on on"

# 0. the real thing must PASS -- a guard wired to fail unconditionally does not
#    satisfy this self-test either.
expect "clean manifest passes" "$(run "$REAL" "$n_off" "$n_on")" 0

# 1. the flag stripped from an OFF target -- the actual regression this exists
#    for, and the one that fails nothing else in the suite.
sed 's/-ffp-contract=off//g' "$REAL" > "$WORK/stripped.txt"
expect "OFF flag stripped fails" "$(run "$WORK/stripped.txt" "$n_off" "$n_on")" 1

# 2. an OFF target given the ON flag -- contraction silently enabled.
sed 's/-ffp-contract=off/-ffp-contract=fast/g' "$REAL" > "$WORK/flipped.txt"
expect "OFF flipped to fast fails" "$(run "$WORK/flipped.txt" "$n_off" "$n_on")" 1

# 3. a guarded target DELETED. Without the count assertion the guard would pass
#    on a shrinking manifest, which is how this kind of check rots.
grep -v '^off|' "$REAL" | cat > "$WORK/dropped.txt"
head -c0 /dev/null
grep '^off|' "$REAL" | tail -n +2 >> "$WORK/dropped.txt"
expect "a dropped OFF target fails" "$(run "$WORK/dropped.txt" "$n_off" "$n_on")" 1

# 4. empty manifest -- the degenerate case of 3.
: > "$WORK/empty.txt"
expect "empty manifest fails" "$(run "$WORK/empty.txt" "$n_off" "$n_on")" 1

# 5. NEGATIVE CONTROL. Reordering rows changes nothing semantically and must
#    stay silent; without this a guard hard-wired to fail would pass 1-4.
sort -r "$REAL" > "$WORK/reordered.txt"
expect "reordered manifest still passes" "$(run "$WORK/reordered.txt" "$n_off" "$n_on")" 0

echo "=== $([ $rc -eq 0 ] && echo PASS || echo FAIL) ==="
exit $rc
