#!/usr/bin/env bash
# A STRUCTURAL TEST NOBODY HAS SEEN FAIL IS A TEST NOBODY HAS TESTED.
#
# Companion to validation/exp_reduction_selftest.sh, same contract: poison the
# input, demand FAILURE; run clean, demand SUCCESS. A test wired to fail
# unconditionally would satisfy every poison case here and be just as useless,
# so the clean run matters as much as the poisons.
#
# WHY THIS EXISTS. tests/trig_reduction_test.cpp asserts that the Payne-Hanek
# table and guard in include/xp/trig_reduction_data.hpp are DEEP ENOUGH, at the
# specific inputs that make them work hardest. Both quantities look like
# arbitrary integers at the call site, and both fail silently: a table one chunk
# short is still right for almost every input, and a guard sized p+4 instead of
# p+C+4 is right everywhere except at the cancellations nobody samples.
#
# THE TRAP THIS IS BUILT AROUND. The obvious poison -- "make the table one
# chunk shorter" -- proves nothing if the table was sized with slack, because a
# short table is then still correct. That is not hypothetical: sizing the depth
# as max(log2|x|) + max(C), which is the natural-looking combination of the two
# measured maxima, over-sizes FF by 8 chunks and QF by 4, and the N-1 arm goes
# quiet. The shipped depths are sized from max(log2|x| + C(x)) with NO slack
# added, and P7 in the test asserts every run that one chunk fewer MISSES.
#
# POISON 3 IS ALSO A MEASUREMENT. It shortens the table for BOTH input families
# the test drives. The depth-pinning family fails; the deepest-cancellation
# family still passes. That is the two-maxima claim in the data header,
# demonstrated rather than asserted -- the cancellation inputs do not exercise
# the depth, so sizing the table from them would be sizing it from the wrong
# measurement.
#
#   validation/trig_reduction_selftest.sh <build dir>
#
# Each case is a compile plus a sub-second run; budget ~40s total.
set -u

builddir="${1:?usage: trig_reduction_selftest.sh <build dir>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
src="${root}/tests/trig_reduction_test.cpp"
inc="${root}/include"
work="${builddir}/trig_reduction_selftest"
mkdir -p "${work}"

CXX="${CXX:-g++}"
fail=0

build_and_run() {   # <tag> <extra-defines...>
  local tag="$1"; shift
  local exe="${work}/trt_${tag}"
  if ! "${CXX}" -O2 -std=c++17 -I "${inc}" "$@" "${src}" -o "${exe}" -lmpfr -lgmp \
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
  echo "  clean: FAIL  <-- the test does not pass on the shipped table"
  sed -n '1,40p' "${work}/clean.log"
  fail=1
fi

# 1  a single corrupted 2/pi chunk, at the DEEPEST position (the hardest to
#    see) -- must be caught by the exact-truncation check, P1
# 2  a single corrupted pi/2 word, at the SECOND position (the last one has
#    53 bits of slack over the requirement, so corrupting it is invisible and
#    would be a blind poison) -- must be caught by P3
# 3  the shipped arm reduces with one chunk fewer -- must be caught by P6
# 4  the shipped arm reduces with the naive guard p+4 -- must be caught by P6
for p in 1 2 3 4; do
  echo "=== poison ${p} must be DETECTED ==="
  if build_and_run "poison${p}" "-DXPMATH_POISON_TRIG_REDUCTION=${p}"; then
    # exit 0 under poison means the test recognised the poison (see the test's
    # own epilogue: it returns 0 when poisoned AND failures were observed).
    if grep -q "poison ${p} correctly detected" "${work}/poison${p}.log"; then
      echo "  poison ${p}: DETECTED"
    else
      echo "  poison ${p}: NOT DETECTED  <-- the test is blind to it"
      fail=1
    fi
  else
    echo "  poison ${p}: harness error"
    sed -n '1,20p' "${work}/poison${p}.log"
    fail=1
  fi
done

# The generator that produced the table must still reproduce its own two
# validations: the published IEEE-double worst case, and the brute-force check
# on its candidate enumeration. --emit refuses to run if either regresses, so
# a successful emit IS the check. Skipped, loudly, when mpfr headers are absent.
echo "=== the generator must still validate its own sizing ==="
gen="${work}/gen"
if "${CXX}" -O2 -std=c++17 -o "${gen}" "${root}/scripts/gen_trig_reduction_constants.cpp" \
     -lmpfr -lgmp > "${work}/gen.build.log" 2>&1; then
  if "${gen}" --emit > "${work}/regen.hpp" 2> "${work}/gen.err"; then
    if diff -q "${work}/regen.hpp" "${inc}/xp/trig_reduction_data.hpp" > /dev/null; then
      echo "  generator: PASS (regenerated header is byte-identical)"
    else
      echo "  generator: FAIL  <-- include/xp/trig_reduction_data.hpp is not what the generator emits"
      diff "${inc}/xp/trig_reduction_data.hpp" "${work}/regen.hpp" | sed -n '1,20p'
      fail=1
    fi
  else
    echo "  generator: FAIL  <-- --emit refused to emit"
    sed -n '1,20p' "${work}/gen.err"
    fail=1
  fi
else
  echo "  generator: SKIPPED (will not compile here; see ${work}/gen.build.log)"
fi

if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: PASS"
exit 0
