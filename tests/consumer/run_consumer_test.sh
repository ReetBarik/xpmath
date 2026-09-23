#!/usr/bin/env bash
# Consumer smoke test driver: install xpmath, then build SEPARATE projects
# against the installed package.
#
# A PACKAGING TEST NOBODY HAS SEEN FAIL IS A PACKAGING TEST NOBODY HAS TESTED.
# Same contract as validation/*_selftest.sh: the clean cases must PASS, and the
# poisoned cases must FAIL. Here the "poisons" are the two ways an export can
# be wrong while looking right:
#
#   1. A version file that answers every query. If find_package(xpmath 99.0)
#      succeeds, the version file is decorative and a consumer can silently
#      link an incompatible xpmath.
#   2. Headers that are not actually installed. If the consumer compiles after
#      the include directory is removed from the install prefix, then it is
#      reading the SOURCE tree through some leaked absolute path, and the
#      install proves nothing.
#
# Two clean consumers share the same install prefix:
#   - tests/consumer/          — packaging smoke (all installed headers)
#   - examples/standalone/     — the published usage example (host only)
# Both must configure, build, and run. The example is not in the main build, so
# without this step it can rot while the suite stays green.
#
#   tests/consumer/run_consumer_test.sh <install-prefix> [build-dir]
#
# Exits 0 only if both clean builds pass AND both poisons fail.
set -u

prefix="${1:?usage: run_consumer_test.sh <install-prefix> [build-dir]}"
work="${2:-${TMPDIR:-/tmp}/xpmath_consumer.$$}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# examples/standalone lives next to tests/, not under tests/consumer/.
example_src="$(cd "${here}/../../examples/standalone" && pwd)"

fail=0
mkdir -p "${work}"

banner() { printf '\n=== %s ===\n' "$1"; }

# --- 1. clean: find_package + compile + run must all succeed ----------------
banner "clean consumer build must PASS"
if cmake -S "${here}" -B "${work}/clean" \
        -DCMAKE_PREFIX_PATH="${prefix}" \
        -DCMAKE_BUILD_TYPE=Release > "${work}/clean.cfg.log" 2>&1 \
   && cmake --build "${work}/clean" > "${work}/clean.build.log" 2>&1 \
   && "${work}/clean/consumer_smoke" > "${work}/clean.run.log" 2>&1; then
  sed 's/^/    /' "${work}/clean.run.log"
  echo "  clean: PASS"
else
  echo "  clean: FAIL  <-- the installed package is not consumable"
  tail -30 "${work}/clean.cfg.log" "${work}/clean.build.log" \
           "${work}/clean.run.log" 2>/dev/null | sed 's/^/    /'
  fail=1
fi

# --- 1b. clean: examples/standalone against the same install ----------------
# CORE_PLAN C10: the published example is outside the main build. Building it
# here is what keeps its CMakeLists and source from drifting off find_package.
banner "clean standalone example must PASS"
if cmake -S "${example_src}" -B "${work}/example" \
        -DCMAKE_PREFIX_PATH="${prefix}" \
        -DCMAKE_BUILD_TYPE=Release > "${work}/example.cfg.log" 2>&1 \
   && cmake --build "${work}/example" > "${work}/example.build.log" 2>&1 \
   && "${work}/example/compensated_reduction" > "${work}/example.run.log" 2>&1; then
  sed 's/^/    /' "${work}/example.run.log"
  echo "  standalone example: PASS"
else
  echo "  standalone example: FAIL  <-- examples/standalone does not consume the install"
  tail -30 "${work}/example.cfg.log" "${work}/example.build.log" \
           "${work}/example.run.log" 2>/dev/null | sed 's/^/    /'
  fail=1
fi

# --- 2. poison: an impossible version must be REFUSED -----------------------
banner "poison 1: find_package(xpmath 99.0 REQUIRED) must FAIL"
cat > "${work}/ver_check.cmake" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(xpmath_version_probe LANGUAGES CXX)
find_package(xpmath 99.0 REQUIRED)
EOF
mkdir -p "${work}/verproj"
cp "${work}/ver_check.cmake" "${work}/verproj/CMakeLists.txt"
if cmake -S "${work}/verproj" -B "${work}/ver" \
        -DCMAKE_PREFIX_PATH="${prefix}" > "${work}/ver.log" 2>&1; then
  echo "  poison 1: NOT DETECTED  <-- the version file accepts anything"
  fail=1
else
  echo "  poison 1: DETECTED (version 99.0 correctly refused)"
fi

# --- 3. poison: headers absent from the prefix must break the build ---------
# Copy the prefix, delete the installed headers from the COPY, and require the
# consumer to fail against it. If it still builds, the include path being used
# is not the installed one.
banner "poison 2: consumer must FAIL when installed headers are removed"
cp -a "${prefix}" "${work}/prefix_noheaders"
rm -rf "${work}/prefix_noheaders/include/xp"
if cmake -S "${here}" -B "${work}/nohdr" \
        -DCMAKE_PREFIX_PATH="${work}/prefix_noheaders" \
        -DCMAKE_BUILD_TYPE=Release > "${work}/nohdr.cfg.log" 2>&1 \
   && cmake --build "${work}/nohdr" > "${work}/nohdr.build.log" 2>&1; then
  echo "  poison 2: NOT DETECTED  <-- consumer is not reading the install tree"
  fail=1
else
  echo "  poison 2: DETECTED (no installed headers, no build)"
fi

banner "result"
if [ "${fail}" -ne 0 ]; then
  echo "RESULT: FAIL   (logs under ${work})"
  exit 1
fi
echo "RESULT: PASS"
rm -rf "${work}"
exit 0
