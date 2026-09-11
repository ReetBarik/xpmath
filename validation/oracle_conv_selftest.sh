#!/usr/bin/env bash
# A STRUCTURAL TEST NOBODY HAS SEEN FAIL IS A TEST NOBODY HAS TESTED.
#
# Companion to validation/trig_reduction_selftest.sh, same contract: poison the
# thing under test, demand FAILURE; run clean, demand SUCCESS.
#
# WHY THIS EXISTS. `sweep_accuracy --oracle-selftest` checks that the
# __float128 <-> mpfr conversions feeding the MPFR oracle are EXACT. Nothing
# else in the repo can check that: the sweep has no oracle-error term anywhere,
# so a wrong reference is scored as a wrong library, and the wrongness lands on
# exactly the rows the MPFR oracle was added to measure.
#
# THE TRAP THIS IS BUILT AROUND. The conversion it replaced was not sloppy —
# it round-tripped binary128 exactly, which is the property it was written and
# commented against. It was wrong about a DIFFERENT property: the oracle never
# converts back, it EVALUATES AT the 400-bit number, and 41 decimal digits give
# a 400-bit number that is only correct to 2^-136. At a zero of sin the
# condition number is unbounded, so 2^-136 in the argument came out as 1.28e11
# ulps in the reference.
#
# So poison 1 — the conversion that actually shipped — is invisible to a round
# trip check, and IS invisible to check B below. It is caught only by A (equality
# at 400 bits against a route sharing no mechanism) and C (the oracle's own
# answer at the arguments where it went wrong). A selftest built only on round
# trips would have passed the defective code.
#
#   validation/oracle_conv_selftest.sh <build dir>
#
# Five compiles of one ~3k-line translation unit; budget ~90s.
set -u

builddir="${1:?usage: oracle_conv_selftest.sh <build dir>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
src="${root}/scripts/sweep_accuracy.cpp"
inc="${root}/include"
work="${builddir}/oracle_conv_selftest"
mkdir -p "${work}"

CXX="${CXX:-g++}"
fail=0

build_and_run() {   # <tag> <extra-defines...>
  local tag="$1"; shift
  local exe="${work}/sw_${tag}"
  if ! "${CXX}" -O2 -std=c++17 -DNDEBUG -fext-numeric-literals -I "${inc}" \
       "$@" "${src}" -o "${exe}" -lquadmath -lmpfr -lgmp \
       > "${work}/${tag}.build.log" 2>&1; then
    echo "  ${tag}: COMPILE FAILED (see ${work}/${tag}.build.log)"
    return 2
  fi
  "${exe}" --oracle-selftest > "${work}/${tag}.log" 2>&1
  return $?
}

echo "=== clean build must PASS ==="
if build_and_run clean; then
  echo "  clean: PASS"
else
  echo "  clean: FAIL  <-- the conversions shipped in sweep_accuracy.cpp are not exact"
  sed -n '1,40p' "${work}/clean.log"
  fail=1
fi

# 1  q_to_mpfr goes back to the 41-digit decimal round trip -- the conversion
#    that actually shipped, and the whole reason this file exists
# 2  q_to_mpfr drops the LAST significand bit -- the smallest corruption
#    binary128 admits, 2^25 finer than poison 1
# 3  mpfr_to_q goes back to the 45-digit decimal round trip -- correctly
#    rounded for almost every value, which is why the check for it has to be
#    built out of values that sit a fraction of an ulp inside the interval
# 4  mpfr_to_q rounds to 112 bits instead of 113
#
# The expected-checks column below is asserted, not decorative, and it has
# already earned its keep: it caught two wrong beliefs about what the checks
# cover. C originally drove bare doubles, whose binary128 image has 60 trailing
# zero significand bits, so it could not see poison 2 at all -- C now drives DD
# pairs, which is what the sweep actually feeds the oracle. And C can never see
# poison 4, because both of its chains end in the same mpfr_to_q; B and D are
# what cover the output direction.
declare -A why=(
  [1]="A C"      # NOT B: the decimal route round-trips binary128 perfectly
  [2]="A B C"
  [3]="D"
  [4]="B D"      # NOT C: see above
)
for p in 1 2 3 4; do
  echo "=== poison ${p} must be DETECTED (expect checks: ${why[$p]}) ==="
  if build_and_run "poison${p}" "-DXPMATH_POISON_ORACLE_CONV=${p}"; then
    echo "  poison ${p}: NOT DETECTED  <-- the selftest is blind to it"
    fail=1
  elif ! grep -q "^oracle conversion selftest: FAIL" "${work}/poison${p}.log"; then
    echo "  poison ${p}: harness error (nonzero exit but no verdict line)"
    sed -n '1,20p' "${work}/poison${p}.log"
    fail=1
  else
    # The poison must be caught by the checks it is aimed at, not by some
    # unrelated one that happens to also go red.
    got=""
    for c in A B C D E; do
      grep -q "^  FAIL  ${c} " "${work}/poison${p}.log" && got="${got}${got:+ }${c}"
    done
    if [ "${got}" = "${why[$p]}" ]; then
      echo "  poison ${p}: DETECTED by ${got}"
    else
      echo "  poison ${p}: DETECTED, but by ${got:-nothing} and not ${why[$p]}"
      echo "               the poison and the check have drifted apart"
      fail=1
    fi
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: PASS"
exit 0
