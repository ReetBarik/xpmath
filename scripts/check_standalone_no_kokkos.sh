#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-DHB-License
# SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
#
# S2 deliverable 4 — prove the standalone extended-precision core stands alone.
#
# Compiles tests/standalone/dd_no_kokkos_smoke.cpp with plain g++ -std=c++17
# and an include path containing ONLY the repo's include/ directory:
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
SRC="${REPO_ROOT}/tests/standalone/dd_no_kokkos_smoke.cpp"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# ONLY include/. Anything the header needs beyond the C++17 standard library
# will fail here, which is the point.
INCLUDES=(-I "${REPO_ROOT}/include")
STD=(-std=c++17)

echo "=== S2 no-Kokkos compile smoke ==="
echo "compiler : $("${CXX}" --version | head -1)"
echo "TU       : ${SRC#"${REPO_ROOT}/"}"
echo "includes : ${REPO_ROOT#"${HOME}/"}/include   (and nothing else)"
echo

# ---------------------------------------------------------------- 1. compile
echo "[1/4] compile + link ..."
"${CXX}" "${STD[@]}" -Wall -Wextra -O2 "${INCLUDES[@]}" \
  "${SRC}" -o "${WORK}/dd_no_kokkos_smoke"
echo "      ok"

# ------------------------------------------------------------------- 2. run
echo "[2/4] run ..."
"${WORK}/dd_no_kokkos_smoke"

# --------------------------------------------- 3. no Kokkos in the preprocess
# Preprocess a MINIMAL TU rather than the smoke test itself: the smoke test's
# own diagnostic strings say the word "Kokkos", which would trip the grep for
# the wrong reason. This TU contains nothing but the include.
echo "[3/4] preprocessed output contains no Kokkos ..."
printf '#include <EPLIB/dd_math.hpp>\nint main() { return 0; }\n' > "${WORK}/only_include.cpp"
"${CXX}" "${STD[@]}" -E "${INCLUDES[@]}" "${WORK}/only_include.cpp" > "${WORK}/pp.ii"
if grep -n "Kokkos\|KOKKOS_" "${WORK}/pp.ii" > "${WORK}/hits.txt"; then
  echo "      FAIL: Kokkos reached the preprocessed TU:"
  head -20 "${WORK}/hits.txt"
  exit 1
fi
echo "      ok ($(wc -l < "${WORK}/pp.ii") preprocessed lines, 0 Kokkos hits)"

# ------------------------------------------ 4. no Kokkos in the header source
# Belt and braces: the standalone headers themselves must not name Kokkos in
# code. Comments may mention it (they explain the relationship to the wrapper),
# so strip // comments before grepping.
echo "[4/4] include/EPLIB/*.hpp name Kokkos only in comments ..."
bad=0
for h in "${REPO_ROOT}"/include/EPLIB/*.hpp; do
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
