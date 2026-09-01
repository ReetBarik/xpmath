#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-DHB-License
# SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
#
# S2 / S5 deliverable — prove the standalone extended-precision core stands alone.
#
# Compiles tests/standalone/{dd,dd_complex,ff,qf}_no_kokkos_smoke.cpp with plain
# g++ -std=c++17 and an include path containing ONLY the repo's include/ directory:
#   * no Kokkos install on the include path,
#   * not even third_party/include/, so the Kokkos compat wrapper is
#     unreachable and cannot mask a missing dependency,
#   * no libquadmath, no -fext-numeric-literals, nothing x86_64-specific
#     (the standalone core is portable; only the demos and the oracle-scored
#     tests are not).
#
# Then it does the check that the compile alone does not make: it preprocesses
# the same TU and greps for the token "Kokkos". An empty grep is the actual
# evidence for "zero Kokkos includes" — the compile succeeding only proves
# nothing is MISSING, not that nothing extra was pulled in.
#
# Usage:  scripts/check_standalone_no_kokkos.sh [compiler]
# Exit 0 = standalone core compiles, links, runs clean, and is Kokkos-free.

set -euo pipefail

CXX="${1:-${CXX:-g++}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DD="${REPO_ROOT}/tests/standalone/dd_no_kokkos_smoke.cpp"
SRC_DD_COMPLEX="${REPO_ROOT}/tests/standalone/dd_complex_no_kokkos_smoke.cpp"
SRC_FF="${REPO_ROOT}/tests/standalone/ff_no_kokkos_smoke.cpp"
SRC_FF_COMPLEX="${REPO_ROOT}/tests/standalone/ff_complex_no_kokkos_smoke.cpp"
SRC_QF="${REPO_ROOT}/tests/standalone/qf_no_kokkos_smoke.cpp"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# ONLY include/. Anything the header needs beyond the C++17 standard library
# will fail here, which is the point.
INCLUDES=(-I "${REPO_ROOT}/include")
STD=(-std=c++17)

echo "=== standalone no-Kokkos compile smoke (DD + DD complex + FF + FF complex + QF) ==="
echo "compiler : $("${CXX}" --version | head -1)"
echo "TUs      : tests/standalone/{dd,dd_complex,ff,ff_complex,qf}_no_kokkos_smoke.cpp"
echo "includes : ${REPO_ROOT#"${HOME}/"}/include   (and nothing else)"
echo

# ---------------------------------------------------------------- 1. compile DD
echo "[1/11] compile + link DD ..."
"${CXX}" "${STD[@]}" -Wall -Wextra -O2 "${INCLUDES[@]}" \
  "${SRC_DD}" -o "${WORK}/dd_no_kokkos_smoke"
echo "      ok"

# ------------------------------------------------------------------- 2. run DD
echo "[2/11] run DD ..."
"${WORK}/dd_no_kokkos_smoke"

# -------------------------------------------------------- 3. compile DD complex
echo "[3/11] compile + link DD complex ..."
"${CXX}" "${STD[@]}" -Wall -Wextra -O2 "${INCLUDES[@]}" \
  "${SRC_DD_COMPLEX}" -o "${WORK}/dd_complex_no_kokkos_smoke"
echo "      ok"

# ----------------------------------------------------------- 4. run DD complex
echo "[4/11] run DD complex ..."
"${WORK}/dd_complex_no_kokkos_smoke"

# ---------------------------------------------------------------- 5. compile FF
echo "[5/11] compile + link FF ..."
"${CXX}" "${STD[@]}" -Wall -Wextra -O2 "${INCLUDES[@]}" \
  "${SRC_FF}" -o "${WORK}/ff_no_kokkos_smoke"
echo "      ok"

# ------------------------------------------------------------------- 6. run FF
echo "[6/11] run FF ..."
"${WORK}/ff_no_kokkos_smoke"

# -------------------------------------------------------- 7. compile FF complex
echo "[7/11] compile + link FF complex ..."
"${CXX}" "${STD[@]}" -Wall -Wextra -O2 "${INCLUDES[@]}" \
  "${SRC_FF_COMPLEX}" -o "${WORK}/ff_complex_no_kokkos_smoke"
echo "      ok"

# ----------------------------------------------------------- 8. run FF complex
echo "[8/11] run FF complex ..."
"${WORK}/ff_complex_no_kokkos_smoke"

# ---------------------------------------------------------------- 9. compile QF
echo "[9/11] compile + link QF ..."
"${CXX}" "${STD[@]}" -Wall -Wextra -O2 "${INCLUDES[@]}" \
  "${SRC_QF}" -o "${WORK}/qf_no_kokkos_smoke"
echo "      ok"

# ------------------------------------------------------------------ 10. run QF
echo "[10/11] run QF ..."
"${WORK}/qf_no_kokkos_smoke"

# --------------------------------------------- 11. no Kokkos in the preprocess
# Preprocess a MINIMAL TU rather than the smoke test itself: the smoke test's
# own diagnostic strings say the word "Kokkos", which would trip the grep for
# the wrong reason. This TU contains nothing but all five includes.
echo "[11/11] preprocessed output contains no Kokkos ..."
printf '#include <xp/dd_math.hpp>\n#include <xp/dd_complex.hpp>\n#include <xp/ff_math.hpp>\n#include <xp/ff_complex.hpp>\n#include <xp/qf_math.hpp>\nint main() { return 0; }\n' > "${WORK}/only_include.cpp"
"${CXX}" "${STD[@]}" -E "${INCLUDES[@]}" "${WORK}/only_include.cpp" > "${WORK}/pp.ii"
if grep -n "Kokkos\|KOKKOS_" "${WORK}/pp.ii" > "${WORK}/hits.txt"; then
  echo "      FAIL: Kokkos reached the preprocessed TU:"
  head -20 "${WORK}/hits.txt"
  exit 1
fi
echo "      ok ($(wc -l < "${WORK}/pp.ii") preprocessed lines, 0 Kokkos hits)"

# ------------------------------------------ 12. no Kokkos in the header source
# Belt and braces: the standalone headers themselves must not name Kokkos in
# code. Comments may mention it (they explain the relationship to the wrapper),
# so strip // comments before grepping. This check runs on ALL xp/*.hpp so it
# automatically covers dd/dd_complex/ff/ff_complex/qf_math.hpp.
echo "[11/11] include/xp/*.hpp name Kokkos only in comments ..."
bad=0
for h in "${REPO_ROOT}"/include/xp/*.hpp; do
  if sed 's://.*::' "${h}" | grep -n "Kokkos\|KOKKOS_" > "${WORK}/h.txt"; then
    echo "      FAIL: $(basename "${h}") references Kokkos in code:"
    cat "${WORK}/h.txt"
    bad=1
  fi
done
[ "${bad}" -eq 0 ] || exit 1
echo "      ok"

echo
echo "=== PASS: the standalone core stands alone ==="
