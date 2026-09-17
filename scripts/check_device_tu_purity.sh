#!/usr/bin/env bash
# ===========================================================================
# scripts/check_device_tu_purity.sh — no device TU may carry the 128-bit type
# ===========================================================================
#
#   scripts/check_device_tu_purity.sh [extra -I dir]...
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
)

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
echo "device-TU purity: ${CXX} $("${CXX}" -dumpfullversion 2>/dev/null), ${#FILES[@]} file(s)"

rc=0
for rel in "${FILES[@]}"; do
  src="${REPO_ROOT}/${rel}"
  if [ ! -f "${src}" ]; then
    echo "FAIL: ${rel} does not exist"
    rc=1
    continue
  fi

  tmp="$(mktemp)"
  err="${tmp}.err"
  if ! "${CXX}" -std=c++17 -E -x c++ "${INCS[@]}" "${src}" -o "${tmp}" 2>"${err}"; then
    echo "FAIL: ${rel} did not preprocess -- the check could not run"
    sed -n '1,5p' "${err}"
    rm -f "${tmp}" "${err}"
    rc=1
    continue
  fi
  if [ ! -s "${tmp}" ]; then
    echo "FAIL: ${rel} preprocessed to an EMPTY file -- the check did not really run"
    rm -f "${tmp}" "${err}"
    rc=1
    continue
  fi

  # Attribute each __float128 occurrence to the file the preprocessor says it
  # came from, and keep only those originating inside this repository.
  hits="$(awk -v root="${REPO_ROOT}" '
      /^# [0-9]+ "/ { f = $3; gsub(/"/, "", f); next }
      /__float128/  { if (index(f, root) == 1) print f }
    ' "${tmp}" | sort | uniq -c | sort -rn)"

  lines="$(wc -l < "${tmp}")"
  if [ -n "${hits}" ]; then
    echo "FAIL: ${rel} -- __float128 reaches a device TU from files in this repo:"
    echo "${hits}" | sed 's/^/    /'
    rc=1
  else
    echo "  ok   ${rel}  (${lines} preprocessed lines, 0 in-repo __float128)"
  fi
  rm -f "${tmp}" "${err}"
done

if [ "${rc}" -eq 0 ]; then
  echo "device-TU purity: PASS"
else
  echo "device-TU purity: FAIL"
fi
exit "${rc}"
