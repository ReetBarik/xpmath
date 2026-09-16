#!/usr/bin/env bash
# ===========================================================================
# validation/a100/run_a100.sh — S8b: validate the a100 arch block of
#                               scripts/xpm_build.sh on real A100 hardware
# ===========================================================================
#
# scripts/xpm_build.sh gained `--arch a100` in step2 (`0230239`). At that point
# the row had been shown to CONFIGURE and nothing more — no A100 had run it.
# This script closes that: it builds through the wrapper on a gpu_a100 node and
# records what the suite actually does there.
#
# It also re-runs the S1 device regression gate. S1 (Cobalt 996777, gpu07,
# 2026-08-31, commit 029180c) established that five tests are genuine A100
# device evidence and all five passed:
#     dd_invariant_test  ff_invariant_test  dd_property_test
#     dd_eft_test        ff_eft_test
# Those five predate S2/S3/S5/S10. If any of them is RED now, the restructure
# moved a device result and that is the headline.
#
# WHAT IS EXPECTED TO FAIL, AND IS NOT A REGRESSION
# -------------------------------------------------------------------------
# Kokkos exports `-arch=sm_80` in its INTERFACE_COMPILE_OPTIONS, so every TU
# that links Kokkos::kokkos gets an nvcc DEVICE pass. A TU that also declares a
# `std::vector<__float128>` makes nvcc instantiate
# `std::initializer_list<__float128>` and reject the 128-bit float as "not
# supported in device code". No compiler flag avoids this; the fix is a
# translation-unit split and it belongs to sub-plan S6. S1 lost 16 of 23
# targets to it, first failing site `src/demo_ff_complex.cpp:387`.
#
# So MOST TARGETS ARE EXPECTED NOT TO BUILD. A target that fails to BUILD and a
# target that builds and then FAILS are completely different findings, and this
# script keeps them apart everywhere it counts.
#
# ORDERING — cheapest evidence first, and the headline banked early
# -------------------------------------------------------------------------
# S1 ran `make -j` with no `-k`, which aborts at the first error: fourteen of
# its sixteen "failures" were never even attempted, leaving them UNKNOWN rather
# than known-broken. This script therefore builds three times, on purpose:
#
#   [2] scripts/xpm_build.sh --arch a100   — the thing under test. Configures,
#       then builds with plain `cmake --build -j`, so it may abort early. That
#       is the wrapper's behaviour and it is being measured, not worked around.
#   [4] the five gate targets, explicitly  — so the regression gate exists even
#       if the job hits its walltime during the long tail.
#   [6] everything, KEEP-GOING            — one pass enumerating every target
#       that cannot compile, which is what S1 could not produce.
#
# The gate ctest [5] runs between them, so the headline is on disk before the
# expensive full build starts.
#
# SUBMIT — script mode:
#     qsub -A pepper_hep -n 1 -t 60 -q gpu_a100 --mode script \
#          validation/a100/run_a100.sh
#
# SUBMIT — interactive:
#     qsub -A pepper_hep -I -n 1 -t 60 -q gpu_a100
#     # then: bash validation/a100/run_a100.sh
#
# Knobs:
#     REPO_ROOT      repo checkout (default: resolved from this script's path)
#     BUILD_DIR      build tree     (default: $REPO_ROOT/build-a100)
#     KOKKOS_PREFIX  Kokkos install to read config macros from; must match the
#                    ARCH_KOKKOS_PREFIX[a100] row of scripts/xpm_build.sh, which
#                    is the source of truth. Mismatch is WARNed about, not
#                    silently accepted.
# ===========================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 60
#COBALT -q gpu_a100

# NOT -e: a failing check is a FINDING to be recorded, not a reason to abort
# the job and lose the rest of the evidence.
set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build-a100}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-$HOME/kokkos-install-cuda-sm80-quadmath}

# The five tests S1 established as the DD/FF device regression gate.
GATE_TESTS=(dd_invariant_test ff_invariant_test dd_property_test
            dd_eft_test ff_eft_test)

if [ ! -f "$REPO_ROOT/include/xp/dd_math.hpp" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/a100/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/a100_${STAMP}.log"
BUILD_KEEPGOING_LOG="$LOGDIR/build_keepgoing_${STAMP}.log"
CTEST_LOG="$LOGDIR/ctest_${STAMP}.log"
GATE_LOG="$LOGDIR/ctest_gate_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo " A100 validation of scripts/xpm_build.sh --arch a100"
echo " date      : $(date -Is)"
echo " host      : $(hostname)"
echo " repo      : $REPO_ROOT"
echo " commit    : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " build dir : $BUILD_DIR"
echo " log       : $LOG"
echo "==========================================================="

# --------------------------------------------------------- 0. toolchain
# Cobalt's batch shell is NON-INTERACTIVE, so the `module` bash function does
# not exist. The login shell gets it from /etc/profile.d/modules.sh.
#
# scripts/xpm_build.sh loads its own modules, but it runs as a CHILD PROCESS:
# its PATH and LD_LIBRARY_PATH edits die with it. ctest is launched from THIS
# shell, and sweep_accuracy carries no RPATH for libquadmath -- it resolves
# libquadmath.so.0 through LD_LIBRARY_PATH at every launch. Run it without
# gcc/13.3.0 and it picks up /usr/lib64/libquadmath.so.0, the oracle
# fingerprint reads 54901e8104607a77 instead of 578322f998a329c8, and ~390
# sweep rows read as regressions that are not there. The binary does not warn.
# So load the same modules here, independently.
if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles 2>/dev/null
  for m in gcc/13.3.0 cmake/3.28.3 cuda/12.9.1; do
    echo "  module load $m"
    module load "$m" || echo "  WARNING: module load $m failed"
  done
else
  echo "  WARNING: no module system in this shell -- using whatever is on PATH"
fi

# ------------------------------------------------------- 1. provenance
echo; echo "--- [1/8] provenance ---"
echo "  hostname : $(hostname)"
echo
echo "  nvidia-smi:"
nvidia-smi 2>&1 | head -15 | sed 's/^/    /'
echo
echo "  nvcc --version:"
nvcc --version 2>&1 | sed 's/^/    /'
echo
echo "  g++ --version:"
g++ --version 2>&1 | head -1 | sed 's/^/    /'
echo "  cmake --version:"
cmake --version 2>&1 | head -1 | sed 's/^/    /'

echo
echo "  Kokkos install: $KOKKOS_PREFIX"
# The wrapper's arch table is the source of truth for this path. Do not trust a
# default that has quietly diverged from it.
if ! grep -q "\[a100\]=\"[^\"]*$(basename "$KOKKOS_PREFIX")\"" "$REPO_ROOT/scripts/xpm_build.sh"; then
  echo "  WARNING: $(basename "$KOKKOS_PREFIX") is not the ARCH_KOKKOS_PREFIX[a100] row"
  echo "           of scripts/xpm_build.sh -- these config macros may describe a"
  echo "           DIFFERENT install than the one the build will link."
  grep -n "ARCH_KOKKOS_PREFIX" -A 5 "$REPO_ROOT/scripts/xpm_build.sh" | sed 's/^/           /'
fi
KCFG="$KOKKOS_PREFIX/include/KokkosCore_config.h"
if [ -r "$KCFG" ]; then
  echo "  KokkosCore_config.h macros (the ones that decide whether this is a"
  echo "  CUDA build at all). Only the DEFINED ones: the file carries ~90"
  echo "  '#undef KOKKOS_ARCH_*' lines and printing them buries the answer."
  grep -E "^#define +KOKKOS_(VERSION|ENABLE_|ARCH_)" "$KCFG" 2>&1 | sed 's/^/    /'
else
  echo "  WARNING: no readable $KCFG"
fi

# --------------------------------------------- 2. the thing under test
echo; echo "--- [2/8] scripts/xpm_build.sh --arch a100 ---"
echo "  (plain 'cmake --build -j', so this may abort at the first failing"
echo "   target; step 6 re-runs it keep-going to enumerate the rest)"
# Start from nothing: a stale CMakeCache from another arch would silently keep
# the old compiler. But BUILD_DIR is a knob, so refuse to rm -rf anything that
# is not recognisably a build tree.
if [ -e "$BUILD_DIR" ] && [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "FATAL: BUILD_DIR=$BUILD_DIR exists and holds no CMakeCache.txt." >&2
  echo "       Refusing to delete it. Point BUILD_DIR somewhere else." >&2
  exit 2
fi
rm -rf "$BUILD_DIR"
bash "$REPO_ROOT/scripts/xpm_build.sh" --arch a100 --build-dir "$BUILD_DIR"
rc_wrapper=$?
echo "  xpm_build.sh exit: $rc_wrapper  (2 = configure/precondition failure;"
echo "   nonzero-other = configure ok, build failed; 0 = everything built)"

if [ ! -d "$BUILD_DIR" ]; then
  echo "FATAL: no build directory -- the configure never ran. Nothing else is" >&2
  echo "       measurable; ctest on a missing directory exits 0 and proves" >&2
  echo "       nothing." >&2
  exit 2
fi

# --------------------------------------------------- 3. build-info.txt
echo; echo "--- [3/8] build-info.txt (written by the CONFIGURE, not the wrapper) ---"
if [ -f "$BUILD_DIR/build-info.txt" ]; then
  sed 's/^/  /' "$BUILD_DIR/build-info.txt"
else
  echo "  MISSING -- the stamp block in CMakeLists.txt did not run."
fi

# --------------------------------------------- 4. gate targets, explicitly
echo; echo "--- [4/8] build the five S1 device-gate targets explicitly ---"
echo "  ${GATE_TESTS[*]}"
cmake --build "$BUILD_DIR" -j"$(nproc)" --target "${GATE_TESTS[@]}" 2>&1 \
  | tail -40 | sed 's/^/  /'
rc_gate_build=${PIPESTATUS[0]}
echo "  gate build exit: $rc_gate_build"
echo "  binaries present:"
for t in "${GATE_TESTS[@]}"; do
  if [ -x "$BUILD_DIR/tests/$t" ]; then echo "    yes  $t"; else echo "    NO   $t"; fi
done

# ------------------------------------------------------- 5. gate ctest
# -V, not --output-on-failure: the execution-space banner is printed by tests
# that PASS, and a silent Serial fallback -- which would make every number in
# this run meaningless -- is exactly the thing that passes quietly.
echo; echo "--- [5/8] ctest, the five gate tests, VERBOSE ---"
ctest --test-dir "$BUILD_DIR" -V -R "^($(IFS='|'; echo "${GATE_TESTS[*]}"))\$" \
  > "$GATE_LOG" 2>&1
rc_gate_ctest=$?
echo "  full output: $GATE_LOG"
grep -E "^ *[0-9]+/[0-9]+ +Test +#|Execution space|execution space|tests passed|tests failed" \
     "$GATE_LOG" | sed 's/^/  /'
echo
echo "  EXECUTION SPACE, quoted verbatim from the gate log:"
if grep -hoE "Execution space: [A-Za-z0-9_:]+" "$GATE_LOG" | sort -u | sed 's/^/    /'; then :; fi
grep -hoE "\((Cuda|Serial|OpenMP|HIP)\)" "$GATE_LOG" | sort -u | sed 's/^/    also seen: /'
if grep -q "Execution space: Cuda" "$GATE_LOG"; then
  echo "    -> CUDA CONFIRMED."
  space_ok=1
else
  echo "    -> NOT CONFIRMED. If no line above says Cuda, this run proved"
  echo "       NOTHING about device behaviour: a silent Serial fallback"
  echo "       produces green tests that measure the host."
  space_ok=0
fi

# ------------------------------------------- 6. full build, keep-going
echo; echo "--- [6/8] full build, KEEP-GOING (enumerates every unbuildable target) ---"
_gen=$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)
case "$_gen" in
  Ninja*) _keep=(-- -k 0) ;;
  *)      _keep=(-- -k)   ;;   # Unix Makefiles
esac
echo "  generator: ${_gen:-unknown}; keep-going flag: ${_keep[*]}"
cmake --build "$BUILD_DIR" -j"$(nproc)" "${_keep[@]}" > "$BUILD_KEEPGOING_LOG" 2>&1
rc_build_all=$?
echo "  exit: $rc_build_all   full log: $BUILD_KEEPGOING_LOG"
echo "  targets whose link/compile FAILED:"
grep -oE "CMakeFiles/[A-Za-z0-9_.-]+\.dir/(all|build)\] Error" "$BUILD_KEEPGOING_LOG" \
  | sed -E 's#CMakeFiles/([A-Za-z0-9_.-]+)\.dir/(all|build)\] Error#\1#' \
  | sort -u | sed 's/^/    /'
echo "  first nvcc 128-bit-float rejection, if any (the S6 blocker):"
grep -n -m 3 -E "128-bit floating|__float128.*not supported|initializer_list<_E>" \
     "$BUILD_KEEPGOING_LOG" | cut -c1-200 | sed 's/^/    /'

# ------------------------------------------------------- 7. libquadmath
# CLAUDE.md's standing check. A correctly BUILT binary still mis-scores if it
# is RUN against the system libquadmath.
echo; echo "--- [7/8] libquadmath resolution for the oracle binary ---"
if [ -x "$BUILD_DIR/tests/sweep_accuracy" ]; then
  ldd "$BUILD_DIR/tests/sweep_accuracy" 2>&1 | grep -i quadmath | sed 's/^/  /'
  if ldd "$BUILD_DIR/tests/sweep_accuracy" 2>/dev/null | grep -q "/soft/compilers/gcc/13.3.0"; then
    echo "  -> resolves to the gcc/13.3.0 libquadmath. Oracle numbers are trustworthy."
  else
    echo "  -> NOT the gcc/13.3.0 libquadmath. Any sweep verdict below is SUSPECT."
  fi
else
  echo "  sweep_accuracy was not built -- no oracle binary to check."
fi

# ------------------------------------------------------------ 8. ctest
echo; echo "--- [8/8] ctest, full suite ---"
n_registered=$(ctest --test-dir "$BUILD_DIR" -N 2>/dev/null | sed -n 's/^Total Tests: //p')
echo "  registered tests: ${n_registered:-0}"
if [ -z "${n_registered:-}" ] || [ "${n_registered:-0}" -eq 0 ]; then
  # ctest --test-dir on a MISSING or unconfigured directory exits 0. A zero
  # test count is the tell, and without this check the run proves nothing.
  echo "  FATAL: zero tests registered. A ctest exit of 0 here would be vacuous." >&2
  exit 2
fi
ctest --test-dir "$BUILD_DIR" --output-on-failure > "$CTEST_LOG" 2>&1
rc_ctest=$?
echo "  exit: $rc_ctest   full log: $CTEST_LOG"

# ctest counts a missing executable as a failure. It is not one: nothing ran.
# S1's headline number, "30% tests passed, 16 tests failed out of 23", was
# entirely this artifact and must never be quoted on its own.
n_pass=$(grep -cE "^ *[0-9]+/[0-9]+ +Test +#[0-9]+: .* +Passed" "$CTEST_LOG")
n_notrun=$(grep -cE "\*\*\*Not Run" "$CTEST_LOG")
n_failed=$(grep -cE "\*\*\*(Failed|Exception|Timeout)" "$CTEST_LOG")
n_skipped=$(grep -cE "\*\*\*Skipped" "$CTEST_LOG")

echo
echo "  PASSED     : $n_pass"
echo "  FAILED     : $n_failed   (built, ran, and gave a wrong answer)"
echo "  NOT BUILT  : $n_notrun   (ctest calls these failures; nothing ran)"
echo "  SKIPPED    : $n_skipped"
echo
echo "  not built (ctest '***Not Run'):"
grep -E "\*\*\*Not Run" "$CTEST_LOG" \
  | sed -E 's/^ *[0-9]+\/[0-9]+ +Test +#[0-9]+: +([^ ]+).*/    \1/' | sort -u
echo "  built and FAILED:"
grep -E "\*\*\*(Failed|Exception|Timeout)" "$CTEST_LOG" \
  | sed -E 's/^ *[0-9]+\/[0-9]+ +Test +#[0-9]+: +([^ ]+).*/    \1/' | sort -u

# --------------------------------------------------- the five-test gate
echo
echo "--- S1 DEVICE REGRESSION GATE (the headline) ---"
gate_fail=0
gate_notbuilt=0
for t in "${GATE_TESTS[@]}"; do
  line=$(grep -E "Test +#[0-9]+: +$t\b" "$GATE_LOG" | grep -E "Passed|\*\*\*" | tail -1)
  if echo "$line" | grep -q "Passed"; then
    verdict="PASS"
  elif echo "$line" | grep -q "Not Run"; then
    verdict="NOT BUILT"; gate_notbuilt=$((gate_notbuilt + 1))
  elif [ -z "$line" ]; then
    verdict="NO RESULT"; gate_notbuilt=$((gate_notbuilt + 1))
  else
    verdict="FAIL"; gate_fail=$((gate_fail + 1))
  fi
  printf "  %-22s %s\n" "$t" "$verdict"
done
echo "  S1 recorded all five PASSING on A100 at 029180c."
if [ "$gate_fail" -gt 0 ]; then
  echo "  -> $gate_fail of 5 REGRESSED. This is a real device regression and is"
  echo "     the headline of the report. Report it; do not fix it here."
elif [ "$gate_notbuilt" -gt 0 ]; then
  echo "  -> $gate_notbuilt of 5 produced no result. The gate is INCONCLUSIVE, which"
  echo "     is not the same as green."
else
  echo "  -> all five hold. No device regression from the restructure."
fi

# ----------------------------------------------------------------- verdict
echo
echo "==========================================================="
echo " xpm_build --arch a100 : $rc_wrapper"
echo " gate build            : $rc_gate_build"
echo " gate ctest            : $rc_gate_ctest"
echo " full build (keep-going): $rc_build_all"
echo " full ctest            : $rc_ctest"
echo " passed/failed/not-built: $n_pass / $n_failed / $n_notrun"
echo " execution space is Cuda: $([ "$space_ok" -eq 1 ] && echo yes || echo NO)"
if [ "$space_ok" -eq 1 ] && [ "$gate_fail" -eq 0 ] && [ "$gate_notbuilt" -eq 0 ]; then
  echo " VERDICT : the a100 arch block builds and runs on A100, and the five-test"
  echo "           S1 device gate still holds. Targets lost to the S6 __float128"
  echo "           TU coupling are the known state, not a regression."
else
  echo " VERDICT : see above. Report, do not fix in place (Rule 4)."
fi
echo " logs    : $LOG"
echo "           $BUILD_KEEPGOING_LOG"
echo "           $GATE_LOG"
echo "           $CTEST_LOG"
echo "==========================================================="
[ "$space_ok" -eq 1 ] && [ "$gate_fail" -eq 0 ] && [ "$gate_notbuilt" -eq 0 ]
