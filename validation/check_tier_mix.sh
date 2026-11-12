#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ctest tier_mix_compile_fail: strict and fast tiers must not mix implicitly.
#   usage: check_tier_mix.sh <cxx> <include-dir> <tests/tier_mix_compile_fail.cpp>
# CASE 0 (explicit conversions, same-tier arithmetic) must compile. Cases 1..10
# each mix the tiers and must FAIL with a compiler error that names a *Fast type
# (so a missing header or a typo in the probe cannot satisfy the test).
set -u
cxx=$1; inc=$2; src=$3
log=$(mktemp)
trap 'rm -f "$log"' EXIT
if ! "$cxx" -std=c++17 -fsyntax-only -I "$inc" -DCASE=0 "$src" >"$log" 2>&1; then
  echo "FAIL: positive control (CASE 0) does not compile"; tail -20 "$log"; exit 1
fi
rc=0
for n in 1 2 3 4 5 6 7 8 9 10; do
  if "$cxx" -std=c++17 -fsyntax-only -I "$inc" -DCASE=$n "$src" >"$log" 2>&1; then
    echo "FAIL: CASE $n compiled but mixes tiers"; rc=1; continue
  fi
  if ! grep -q 'error' "$log" || ! grep -q 'Fast' "$log"; then
    echo "FAIL: CASE $n failed, but not with a tier-mixing diagnostic"; head -5 "$log"; rc=1; continue
  fi
  echo "ok: CASE $n rejected"
done
exit $rc
