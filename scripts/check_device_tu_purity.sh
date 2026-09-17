#!/usr/bin/env bash
# ===========================================================================
# scripts/check_device_tu_purity.sh — the test TUs stay on their own side
# ===========================================================================
#
#   scripts/check_device_tu_purity.sh [extra -I dir]...
#
# TWO CHECKS, POINTING IN OPPOSITE DIRECTIONS. C2 wrote the first; CORE_PLAN C4
# step 4 (chunk E) added the second, and the pair is the whole enforcement:
#
#   FORWARD (part 1)  no DEVICE TU may carry __float128.
#   MIRROR  (part 2)  no HOST TU may carry a kernel launch.
#
# Neither implies the other, and the mirror is not decoration. MEASURED on the
# run that introduced it: tests/hello_test.cpp had been taken off Kokkos by C4
# chunk A and re-pointed at tests/device_harness.hpp, passed the forward check
# trivially (it is not in FILES, so the forward check never looked at it), built
# clean, and ran green -- while being a TU that includes binary128 AND launches
# a kernel, which is the exact shape nvcc rejects (S6) and the exact shape C4
# exists to eliminate. The mirror went red on it immediately. It was split.
#
# WHAT THIS ENFORCES
#   tests/test_utils_device.hpp, and every file added to FILES below, must be
#   compilable into a DEVICE translation unit. The one thing that makes that
#   impossible is __float128: nvcc walks STL member signatures during the device
#   pass and rejects the 128-bit float. std::vector<__float128> in the unsplit
#   tests/test_utils.hpp put that type into all 21 test TUs that included the
#   header, whether or not they used the oracle, and A100 (S8b) could not build
#   9 of 49 targets.
#
#   Do NOT read that as "the split makes those 9 build". MEASURED in C2: every
#   host-classified TU uses an oracle symbol of its own, hello_test included.
#   This gate protects the BOUNDARY, so C4 can add device TUs that stay clean.
#
# WHY THIS IS NOT `grep __float128`, WHICH IS WHAT YOU WOULD WRITE FIRST
#   MEASURED, gcc 13.3.0, this tree: preprocessing a TU that includes only the
#   xp core plus <cmath>/<cstdint>/<random> yields exactly ONE occurrence of
#   __float128, and it comes from /usr/include/bits/floatn.h -- a SYSTEM header
#   that typedefs the type on x86_64 whether or not anyone uses it. A bare grep
#   therefore fails on provably clean code, and the gate would have been red
#   from the day it was written.
#
#   The type EXISTING is harmless. The type being USED BY OUR CODE is the
#   defect. So this walks the preprocessor's `# <line> "<file>"` line markers,
#   attributes every occurrence to the file it came from, and fails only when
#   that file is inside this repository. System headers are ignored by
#   provenance, not by an allowlist of names that would rot.
#
# WHY IT ASSERTS THAT IT RAN
#   A gate whose pass condition is "no output" is indistinguishable from a gate
#   that never executed -- a mistyped path, a missing include dir, a compiler
#   that is not on PATH all produce silence that reads as success. So every step
#   below checks its own exit code and requires the preprocessed output to be
#   non-empty before it will report a pass.
#
# Comments are NOT a false-positive source: the preprocessor strips them, so the
# explanatory `__float128` mentions in the device header's own prose are gone by
# the time this looks. That is deliberate -- those comments explain where the
# digit caps come from and should stay readable.
#
# KOKKOS IS NOT REQUIRED, AND THAT IS DELIBERATE
#   tests/test_utils_device.hpp names the xp CORE (<xp/dd_math.hpp>), not the
#   Kokkos compat wrappers in third_party/include, so this runs with -Iinclude
#   alone and the ctest target is registered in the KOKKOS-FREE set -- it gates
#   the no-kokkos lane, where a device-purity regression is most likely to be
#   introduced by someone who never builds with Kokkos at all.
#
#   That cost no type churn: third_party/include/dd_math.hpp says
#   `using DoubleDouble = xp::DoubleDouble`, a true alias rather than a distinct
#   wrapper, so the core spelling names the same type the Kokkos spelling does.
#   MEASURED, this tree: dropping Kokkos from the device header took the
#   preprocessed output from 130341 lines to 56339.
#
#   The optional trailing -I arguments are still accepted and still honoured.
#   C4 adds *_test_device.cpp entries to FILES, and those may need include paths
#   of their own.
# ===========================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"

# The maintained list. C4 extends this with every *_test_device.cpp.
FILES=(
  "tests/test_utils_device.hpp"
  "tests/device_harness.hpp"
  # CORE_PLAN C4 step 3: the tests that were already device-only and have now been
  # moved off Kokkos onto the harness above. Each is a whole translation unit, not
  # a header -- being listed here is the standing claim that no binary128 can reach
  # a device compile of it, which is exactly the claim S6 showed the mixed TUs
  # cannot make (nvcc rejects std::vector<__float128> in a TU it gives a device
  # pass).
  #
  #
  # tests/qf_eft_test.cpp used to be DELIBERATELY ABSENT here: it linked no Kokkos
  # and ran its device parity through the same harness, but its Test C carried a
  # binary128 wide-spread truncation check (test_renorm_4_wide), making it a MIXED
  # TU -- a ninth beyond the eight the plan names. C4 step 2 (chunk D) split it.
  # The host half keeps the target name and Test C; the device half is
  # tests/qf_eft_test_device.cpp, listed below. The absence is resolved, not
  # carried forward, and the gate was never weakened to accommodate it.
  "tests/dd_invariant_test.cpp"
  "tests/ff_invariant_test.cpp"
  "tests/ff_eft_test.cpp"
  "tests/tf_eft_test.cpp"
  "tests/tf_fma_guard_test.cpp"
  # CORE_PLAN C4 step 2 (chunk C): the DEVICE HALVES of the split DD/FF mixed
  # TUs. Each one's host half kept the target name, the oracle and
  # tests/test_utils_host.hpp; these carry the kernel and nothing that needs a
  # type wider than a machine word. Listing them here is the point of the split
  # -- an unlisted device half would be a device TU nobody is checking, which is
  # the state the eight mixed TUs were already in.
  "tests/dd_eft_test_device.cpp"
  "tests/dd_fma_guard_test_device.cpp"
  "tests/dd_property_test_device.cpp"
  "tests/ff_property_test_device.cpp"
  # ff_fma_guard_test.cpp is here WITHOUT a _device sibling, on purpose. Chunk C
  # migrated it WHOLE rather than splitting it: post-C2 it includes
  # test_utils_device.hpp only, and its oracle is an exact FP64 product (a
  # 48-bit result in a 53-bit mantissa), not a 128-bit one. MEASURED on this
  # tree before adding it: the only mentions of the wide type's name in that
  # file are prose, which the preprocessor strips. Note that a STRING LITERAL
  # would not be stripped -- one of its printf lines was reworded for exactly
  # that reason.
  "tests/ff_fma_guard_test.cpp"
  # CORE_PLAN C4 step 2 (chunk D): the DEVICE HALVES of the split QF mixed TUs,
  # same shape as chunk C's DD/FF entries above. qf_eft_test_device.cpp is the one
  # that resolves the deliberate absence noted at the top of this array.
  "tests/qf_eft_test_device.cpp"
  "tests/qf_nonoverlap_test_device.cpp"
  "tests/qf_property_test_device.cpp"
  # qf_fma_guard_test.cpp is here WITHOUT a _device sibling, for the same reason
  # ff_fma_guard_test.cpp is: chunk D migrated it WHOLE rather than splitting it.
  # Its oracle is an exact FP64 product (48 bits in a 53-bit mantissa), not a
  # 128-bit one, and post-C2 it includes test_utils_device.hpp only. MEASURED on
  # this tree before adding it: one of its printf STRING LITERALS named the wide
  # type, which the preprocessor does NOT strip, and was reworded.
  "tests/qf_fma_guard_test.cpp"
  # CORE_PLAN C4 step 4 (chunk E). Two device TUs that were NOT listed before and
  # should have been:
  #   device_harness_test.cpp -- C3's self-test for the launch layer. It is the
  #     most device-shaped TU in the tree and was simply never added.
  #   hello_test_device.cpp   -- the device half of the split the MIRROR CHECK
  #     below forced. See this file's header.
  # Both matter beyond bookkeeping now: FILES is what partitions tests/*.cpp into
  # the two halves, so an unlisted launching TU is not merely unchecked by part 1,
  # it is CLAIMED AS HOST by part 2 and goes red there.
  "tests/device_harness_test.cpp"
  "tests/hello_test_device.cpp"
)

# ===========================================================================
# PART 2's INPUT: the tokens that mean "this TU launches a kernel".
# ===========================================================================
# Kokkos's two spellings plus the C3 harness's one. KOKKOS_LAMBDA is kept even
# though no test links Kokkos after C4 -- the check is against a TU REACQUIRING
# a launch, and the Kokkos spelling is the one a reader porting old code would
# reach for first.
LAUNCH_TOKENS='Kokkos::parallel_for|parallel_for_n|KOKKOS_LAMBDA'

INCS=(-I"${REPO_ROOT}/include" -I"${REPO_ROOT}/tests" -I"${REPO_ROOT}/third_party/include")
for extra in "$@"; do
  if [ -n "${extra}" ]; then
    INCS+=(-I"${extra}")
  fi
done

if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "FAIL: compiler '${CXX}' is not on PATH -- the check did not run"
  exit 1
fi
CXXVER="$("${CXX}" -dumpfullversion 2>/dev/null)"

rc=0

# ---------------------------------------------------------------------------
# scan <relative-path> <egrep-pattern> <what>
#
# Preprocesses one TU and prints, one per line, every in-repo file the pattern
# was seen in. Exits nonzero -- and says why -- when it could not actually look:
# missing file, failed preprocess, empty output. Both parts below share it, so
# "the check ran" means the same thing in each.
# ---------------------------------------------------------------------------
# $3 is the SINGULAR subject, used in the failure line ("<what> reaches this
# TU"); $4 is its plural, used in the success line ("0 in-repo <plural>"). Two
# strings rather than one because the same noun cannot serve both -- "0 in-repo
# a kernel launch" is what one string produces.
scan() {
  local rel="$1" pat="$2" what="$3" plural="$4"
  local src="${REPO_ROOT}/${rel}"

  if [ ! -f "${src}" ]; then
    echo "FAIL: ${rel} does not exist"
    return 1
  fi

  local tmp err
  tmp="$(mktemp)"
  err="${tmp}.err"
  if ! "${CXX}" -std=c++17 -E -x c++ "${INCS[@]}" "${src}" -o "${tmp}" 2>"${err}"; then
    echo "FAIL: ${rel} did not preprocess -- the check could not run"
    sed -n '1,5p' "${err}"
    rm -f "${tmp}" "${err}"
    return 1
  fi
  if [ ! -s "${tmp}" ]; then
    echo "FAIL: ${rel} preprocessed to an EMPTY file -- the check did not really run"
    rm -f "${tmp}" "${err}"
    return 1
  fi

  # Attribute each occurrence to the file the preprocessor says it came from,
  # and keep only those originating inside this repository. System headers are
  # ignored by PROVENANCE, not by a name allowlist that would rot.
  local hits lines
  hits="$(awk -v root="${REPO_ROOT}" -v pat="${pat}" '
      /^# [0-9]+ "/ { f = $3; gsub(/"/, "", f); next }
      $0 ~ pat      { if (index(f, root) == 1) print f }
    ' "${tmp}" | sort | uniq -c | sort -rn)"
  lines="$(wc -l < "${tmp}")"
  rm -f "${tmp}" "${err}"

  if [ -n "${hits}" ]; then
    echo "FAIL: ${rel} -- ${what} reaches this TU from files in this repo:"
    echo "${hits}" | sed 's/^/    /'
    return 1
  fi
  echo "  ok   ${rel}  (${lines} preprocessed lines, 0 in-repo ${plural})"
  return 0
}

# ===========================================================================
# PART 1 -- FORWARD: no __float128 in a device TU.
# ===========================================================================
echo "=== part 1: device TUs carry no __float128 ==="
echo "device-TU purity: ${CXX} ${CXXVER}, ${#FILES[@]} file(s)"
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "FAIL: FILES is EMPTY -- part 1 inspected nothing and its silence means nothing"
  rc=1
fi
for rel in "${FILES[@]}"; do
  scan "${rel}" '__float128' '__float128' '__float128' || rc=1
done

# ===========================================================================
# PART 1b -- COVERAGE: every *_test_device.cpp on the branch is in FILES.
# ===========================================================================
# A device half nobody listed is a device TU nobody is checking, which is the
# state the eight mixed TUs were already in before C4. The naming convention is
# load-bearing, so it is cheap to assert against.
echo "=== part 1b: every tests/*_test_device.cpp is listed in FILES ==="
shopt -s nullglob
device_named=("${REPO_ROOT}"/tests/*_test_device.cpp)
shopt -u nullglob
if [ "${#device_named[@]}" -eq 0 ]; then
  echo "FAIL: found NO tests/*_test_device.cpp -- either the glob is wrong or the"
  echo "      C4 splits are gone. Either way this assertion proved nothing."
  rc=1
else
  echo "  ${#device_named[@]} file(s) match tests/*_test_device.cpp"
  for abs in "${device_named[@]}"; do
    rel="tests/$(basename "${abs}")"
    found=0
    for listed in "${FILES[@]}"; do
      if [ "${listed}" = "${rel}" ]; then found=1; break; fi
    done
    if [ "${found}" -eq 1 ]; then
      echo "  ok   ${rel} is in FILES"
    else
      echo "FAIL: ${rel} exists but is NOT in this script's FILES list"
      rc=1
    fi
  done
fi

# ===========================================================================
# PART 2 -- MIRROR: no host TU launches a kernel.
# ===========================================================================
# CORE_PLAN C4 step 4. THE HOST SET IS DERIVED, NOT MAINTAINED: it is every
# top-level tests/*.cpp that FILES does not claim as a device TU. That closes
# the loop in both directions with one list --
#
#   * a NEW test TU is host by default, so it gets the mirror check for free;
#   * giving it a launch turns the mirror red until someone adds it to FILES;
#   * adding it to FILES subjects it to part 1.
#
# There is no third state and no way to be in neither, which is what a second
# hand-maintained list would have allowed.
#
# Scope is top-level tests/*.cpp on purpose. tests/standalone/*.cpp are
# compile-and-run smokes for the core with no launch surface at all, and
# tests/consumer/ is a nested CMake project, not a TU.
#
# WHY IT PREPROCESSES INSTEAD OF GREPPING THE SOURCE, which is what you would
# write first and which would be red right now: eight host halves explain in
# PROSE where their launch went ("... used to run here inside a
# Kokkos::parallel_for"), and a raw grep cannot tell that from a call. The
# preprocessor strips comments, so those explanations stay readable. Note the
# limit, which is the same one part 1 has: a STRING LITERAL naming a token is
# NOT stripped and will fail this check. That is the right default -- reword the
# literal.
echo "=== part 2 (mirror): host TUs contain no kernel launch ==="
shopt -s nullglob
all_tus=("${REPO_ROOT}"/tests/*.cpp)
shopt -u nullglob
if [ "${#all_tus[@]}" -eq 0 ]; then
  echo "FAIL: found NO tests/*.cpp -- the host set is empty and part 2's silence"
  echo "      would be indistinguishable from a wrong path. Refusing to pass."
  rc=1
fi

host_tus=()
for abs in "${all_tus[@]}"; do
  rel="tests/$(basename "${abs}")"
  is_device=0
  for listed in "${FILES[@]}"; do
    if [ "${listed}" = "${rel}" ]; then is_device=1; break; fi
  done
  if [ "${is_device}" -eq 0 ]; then
    host_tus+=("${rel}")
  fi
done

echo "host-TU mirror: ${#all_tus[@]} tests/*.cpp total, ${#host_tus[@]} classified host"
echo "                (launch tokens: ${LAUNCH_TOKENS})"
if [ "${#host_tus[@]}" -eq 0 ]; then
  echo "FAIL: the HOST set is EMPTY -- part 2 inspected nothing. Either FILES"
  echo "      claims every TU or the glob is wrong; a pass here would be a lie."
  rc=1
fi
for rel in "${host_tus[@]}"; do
  scan "${rel}" "${LAUNCH_TOKENS}" 'a kernel launch' 'kernel launches' || rc=1
done

if [ "${rc}" -eq 0 ]; then
  echo "device-TU purity: PASS"
else
  echo "device-TU purity: FAIL"
fi
exit "${rc}"
