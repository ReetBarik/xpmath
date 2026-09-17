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
# TWO ARMS, BECAUSE THE REAL MANIFEST CAN LEGITIMATELY BE EMPTY.
# CORE_PLAN C4 moved this test outside the XPMATH_WITH_KOKKOS guard so it runs
# in both configurations. In the Kokkos-free one every contraction-guarded
# target is (for now) absent, so the real manifest has zero rows -- and a poison
# matrix over zero rows poisons nothing. `sed` on an empty file yields an empty
# file, the checker is handed the same 0/0 expectation, and all four "must fail"
# cases would PASS. That is precisely the shape of vacuity this file exists to
# prevent, so it would be self-defeating to let the run go quiet there.
#
#   ARM 1 (always) -- the matrix over a SYNTHETIC manifest written here. It
#                     exercises the checker's logic unconditionally, so "can
#                     this guard fail?" is answered in every configuration.
#   ARM 2 (when the real manifest has rows) -- the same matrix over the REAL
#                     manifest. This is the stronger check and the original
#                     one: it poisons the actual configure output, so a real
#                     row whose format the poison cannot touch is caught.
#
# Both arms end with a NEGATIVE CONTROL, and the real manifest is always
# required to pass clean, so a guard hard-wired to fail satisfies neither arm.
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
rc=0

run() {  # run <manifest> <expect-off> <expect-on>  -> echoes exit code
  cmake -DMANIFEST="$1" -DEXPECT_OFF="$2" -DEXPECT_ON="$3" -P "$CHECK" >/dev/null 2>&1
  echo $?
}
expect() {  # expect <label> <got> <want>
  if [ "$2" = "$3" ]; then echo "  ok    $1 (exit $2)"
  else echo "  FAIL  $1: exit $2, expected $3"; rc=1; fi
}

# The four poisons plus the negative control, over whatever manifest it is
# handed. <tag> only labels the output lines.
matrix() {  # matrix <tag> <manifest>
  local tag="$1" src="$2"
  local n_off n_on
  n_off=$(grep -c '^off|' "$src"); n_on=$(grep -c '^on|' "$src")
  echo "  [$tag] $n_off off, $n_on on"

  # 1. the flag stripped from an OFF target -- the actual regression this exists
  #    for, and the one that fails nothing else in the suite.
  sed 's/-ffp-contract=off//g' "$src" > "$WORK/$tag.stripped.txt"
  expect "[$tag] OFF flag stripped fails" \
         "$(run "$WORK/$tag.stripped.txt" "$n_off" "$n_on")" 1

  # 2. an OFF target given the ON flag -- contraction silently enabled.
  sed 's/-ffp-contract=off/-ffp-contract=fast/g' "$src" > "$WORK/$tag.flipped.txt"
  expect "[$tag] OFF flipped to fast fails" \
         "$(run "$WORK/$tag.flipped.txt" "$n_off" "$n_on")" 1

  # 3. a guarded target DELETED. Without the count assertion the guard would
  #    pass on a shrinking manifest, which is how this kind of check rots.
  grep -v '^off|' "$src" > "$WORK/$tag.dropped.txt"
  grep '^off|' "$src" | tail -n +2 >> "$WORK/$tag.dropped.txt"
  expect "[$tag] a dropped OFF target fails" \
         "$(run "$WORK/$tag.dropped.txt" "$n_off" "$n_on")" 1

  # 4. empty manifest -- the degenerate case of 3.
  : > "$WORK/$tag.empty.txt"
  expect "[$tag] empty manifest fails" \
         "$(run "$WORK/$tag.empty.txt" "$n_off" "$n_on")" 1

  # 5. NEGATIVE CONTROL. Reordering rows changes nothing semantically and must
  #    stay silent; without this a guard hard-wired to fail would pass 1-4.
  sort -r "$src" > "$WORK/$tag.reordered.txt"
  expect "[$tag] reordered manifest still passes" \
         "$(run "$WORK/$tag.reordered.txt" "$n_off" "$n_on")" 0
}

echo "=== contraction flag guard self-test ==="

real_off=$(grep -c '^off|' "$REAL"); real_on=$(grep -c '^on|' "$REAL")
echo "real manifest: $real_off off, $real_on on"

# 0. the real thing must PASS -- a guard wired to fail unconditionally does not
#    satisfy this self-test either. Runs in BOTH configurations; when the real
#    manifest is empty this is the assertion that 0/0 is what the configure
#    actually produced.
expect "clean real manifest passes" "$(run "$REAL" "$real_off" "$real_on")" 0

# --- ARM 1: synthetic fixture, unconditional ------------------------------
# Row shape copied from the configure-time writer in tests/CMakeLists.txt:
#   <posture>|<target>|<compile options>
# with the options left as the unevaluated genex text the real manifest carries,
# so the checker's substring test is exercised on the same input shape.
cat > "$WORK/synthetic.txt" <<'EOF'
off|synthetic_eft_test|$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
off|synthetic_guard_test|$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
on|synthetic_guard_test_contract_on|$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>
EOF
expect "clean synthetic manifest passes" "$(run "$WORK/synthetic.txt" 2 1)" 0
matrix synthetic "$WORK/synthetic.txt"

# --- ARM 2: the real manifest, when there is anything to poison -----------
if [ "$real_off" -gt 0 ] || [ "$real_on" -gt 0 ]; then
  matrix real "$REAL"
else
  echo "  note  real manifest has no rows in this configuration"
  echo "        (XPMATH_WITH_KOKKOS=OFF: every contraction-guarded target is"
  echo "         registered by an EFT helper inside the Kokkos block). ARM 1"
  echo "         above still proved the guard can fail. SKIPPING ARM 2 is not"
  echo "         the same as it passing, and it is not counted as a pass."
fi

echo "=== $([ $rc -eq 0 ] && echo PASS || echo FAIL) ==="
exit $rc
