#!/usr/bin/env bash
# A STRUCTURAL TEST NOBODY HAS SEEN FAIL IS A TEST NOBODY HAS TESTED.
#
# Companion to validation/gate_selftest.sh, same contract: poison the input,
# demand FAILURE; run clean, demand SUCCESS. A test wired to fail
# unconditionally would satisfy every poison case here and be just as useless,
# so the clean run matters as much as the poisons.
#
# WHY THIS EXISTS. exp_reduction_test asserts that every Cody-Waite piece of
# ln2 is narrow enough that k*c_i is exact in one machine word. That property is
# invisible at the call site -- the constants look like ordinary hex floats --
# so the failure mode is a well-meaning edit that "improves precision" by
# widening a piece, silently reintroducing the |a|*2^-p reduction error the
# whole KI-42 change removed.
#
# THE TRAP THIS IS BUILT AROUND. During development, poisoning by re-deriving a
# constant at a WIDER nominal width tested nothing: round-to-nearest of ln2 at
# 17 and 18 significant bits yields a float BIT-IDENTICAL to the 16-bit one
# (the 16-bit value has 9 trailing zero mantissa bits), and the same happens on
# DD at 43 bits. Two poisons in a row were blind. So poison 1 sets mantissa bit
# 0, which always changes the value and always makes k*c_i inexact for k >= 2,
# with no dependence on where rounding happened to land.
#
#   validation/exp_reduction_selftest.sh <build dir>
#
# Each case is a compile plus a sub-second run; budget ~30s total.
set -u

builddir="${1:?usage: exp_reduction_selftest.sh <build dir>}"
src="$(cd "$(dirname "$0")/.." && pwd)/tests/exp_reduction_test.cpp"
inc="$(cd "$(dirname "$0")/.." && pwd)/include"
work="${builddir}/exp_reduction_selftest"
mkdir -p "${work}"

CXX="${CXX:-g++}"
fail=0

build_and_run() {   # <tag> <extra-defines...>
  local tag="$1"; shift
  local exe="${work}/ert_${tag}"
  if ! "${CXX}" -O2 -std=c++17 -I "${inc}" "$@" "${src}" -o "${exe}" -lquadmath \
       > "${work}/${tag}.build.log" 2>&1; then
    echo "  ${tag}: COMPILE FAILED (see ${work}/${tag}.build.log)"
    return 2
  fi
  "${exe}" > "${work}/${tag}.log" 2>&1
  return $?
}

echo "=== clean build must PASS ==="
if build_and_run clean; then
  echo "  clean: PASS"
else
  echo "  clean: FAIL  <-- the test does not pass on the shipped constants"
  sed -n '1,40p' "${work}/clean.log"
  fail=1
fi

# Poison 1 is the only one the test can currently self-inflict via a define.
# Cases 2-4 (drop a piece / reverse the order / restore the rounded product)
# are edits to the exp() bodies themselves, not to the test's constant table,
# so they are NOT exercised here -- stated plainly rather than implied by
# omission. Adding them means teaching the test to reimplement the reduction.
for p in 1; do
  echo "=== poison ${p} must be DETECTED ==="
  if build_and_run "poison${p}" "-DXPMATH_POISON_EXP_REDUCTION=${p}"; then
    # exit 0 under poison means the test recognised the poison (see the test's
    # own epilogue: it returns 0 when poisoned AND failures were observed).
    if grep -q "poison ${p} correctly detected" "${work}/poison${p}.log"; then
      echo "  poison ${p}: DETECTED"
    else
      echo "  poison ${p}: NOT DETECTED  <-- the test is blind to a widened piece"
      fail=1
    fi
  else
    echo "  poison ${p}: harness error"
    sed -n '1,20p' "${work}/poison${p}.log"
    fail=1
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: PASS"
exit 0
