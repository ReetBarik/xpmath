#!/usr/bin/env bash
# ===========================================================================
# validation/mi250/run_mi250_build.sh — S8c: first execution of the `mi250`
#                                       arch block of scripts/xpm_build.sh
# ===========================================================================
#
# SIBLING OF run_mi250.sh, NOT A REPLACEMENT. That script verifies a LIBRARY
# claim -- the gfx90a de-recursion fix, at runtime, on the default stack -- by
# driving hipcc directly and never touching CMake or Kokkos. This one verifies
# a BUILD RECIPE claim: that `scripts/xpm_build.sh --arch mi250` configures,
# builds and tests this repo through a real gfx90a Kokkos. Neither subsumes
# the other and run_mi250.sh is not modified.
#
# WHY IT EXISTS. The `mi250` row of the arch table in scripts/xpm_build.sh had
# never run. S8's deliverable 1 says so in as many words: "--arch mi250 and
# --kokkos build are still unexecuted -- no hipcc on the login node". The row
# was written from the recorded configuration of the install at
# ~/xpm_device/kokkos-hip-gfx90a, which is inference, not measurement. This
# script is the measurement. Anything it finds wrong is REPORTED, not patched
# around here -- a job script carrying a fix the recipe lacks leaves the recipe
# broken and the log saying it works.
#
# Compiles on the COMPUTE NODE -- hipcc is not on the JLSE login node. That is
# the whole reason the row went unexecuted.
#
# ---------------------------------------------------------------------------
# THE SHAPE OF THE ANSWER, AND WHY THERE ARE TWO SETS OF COUNTS
# ---------------------------------------------------------------------------
# Step [3/6] runs ctest on the tree exactly as the recipe left it. Those are
# THE counts -- what a user who types `scripts/xpm_build.sh --arch mi250`
# actually gets. But `make` abandons a directory at its first error, so when
# the build fails those counts collapse to "almost everything Not Run" and say
# nothing about WHICH targets are broken and which were merely downwind of a
# broken one.
#
# So steps [4/6] and [6/6] are a DIAGNOSTIC pass, run only if the recipe build
# failed and labelled as such everywhere they print: a keep-going rebuild of
# the ALREADY-CONFIGURED tree, then ctest again. Nothing in the
# diagnostic pass changes the configure, the flags or the sources -- it only
# stops make from giving up early. Its counts are reported separately and are
# never the headline.
#
# ---------------------------------------------------------------------------
# WHAT A GREEN RUN HERE WOULD NOT MEAN
# ---------------------------------------------------------------------------
# The Kokkos at ~/xpm_device/kokkos-hip-gfx90a was built with LIBQUADMATH OFF,
# while the host and a100 installs have it ON:
#
#     $ grep LIBQUADMATH ~/xpm_device/kokkos-hip-gfx90a/include/KokkosCore_config.h
#     /* #undef KOKKOS_ENABLE_LIBQUADMATH */
#
# tests/CMakeLists.txt's stated posture for that case is graceful degradation:
# the oracle-scored tests still BUILD and return 77 at runtime, so CTest calls
# them Skipped. A skipped test asserts nothing, so SKIPPED is printed as a
# first-class number here and never folded into a pass rate. Do not read
# "0 failed" as "the suite is green", and do not rebuild Kokkos to make the
# number go away -- removing the asymmetry is Arc B's job, not this script's.
#
# And watch the NOT BUILT column beside it: a target that fails to COMPILE
# without the oracle is not degrading gracefully, it is a different and worse
# outcome than a skip, and only these two columns side by side tell them apart.
#
# SUBMIT — script mode:
#     qsub -A pepper_hep -n 1 -t 60 -q gpu_amd_mi250 --mode script \
#          validation/mi250/run_mi250_build.sh
#
# SUBMIT — interactive:
#     qsub -A pepper_hep -I -n 1 -t 60 -q gpu_amd_mi250
#     # then: bash validation/mi250/run_mi250_build.sh
#
# Knobs:
#     REPO_ROOT      repo checkout (default: resolved from this script's path)
#     BUILD_DIR      configure into here (default $REPO_ROOT/build-mi250).
#                    Deliberately NOT $REPO_ROOT/build: that is xpm_build.sh's
#                    default and holds host builds, and a half-overwritten
#                    CMakeCache.txt from another compiler is a worse failure
#                    than no build at all.
#     KOKKOS_PREFIX  the gfx90a Kokkos to sanity-check the macros of. Must
#                    match the mi250 row of ARCH_KOKKOS_PREFIX in
#                    scripts/xpm_build.sh; checked, not assumed.
#     ROCM_ROOT      direct ROCm path if the module system is absent
#     CTEST_TIMEOUT  per-test timeout in seconds (default 900)
#     TRIAGE_BUDGET  seconds allowed for the per-target first-error triage
#                    (default 900; 0 disables it)
#     FRESH          1 = rm -rf BUILD_DIR first (default 1; a stale cache from
#                    a different compiler is the classic false result)
# ===========================================================================

#COBALT -A pepper_hep
#COBALT -n 1
#COBALT -t 60
#COBALT -q gpu_amd_mi250

# NOT -e. Every step here is evidence: a failed build is a FINDING about the
# arch block, and aborting on it throws away the ctest counts that say how far
# the recipe got. Each step records its own exit code and the verdict block at
# the bottom is the only place they are judged.
set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_self/../.." && pwd)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build-mi250}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-$HOME/xpm_device/kokkos-hip-gfx90a}
ROCM_ROOT=${ROCM_ROOT:-/soft/compilers/rocm/rocm-7.0.2}
CTEST_TIMEOUT=${CTEST_TIMEOUT:-900}
TRIAGE_BUDGET=${TRIAGE_BUDGET:-900}
FRESH=${FRESH:-1}

if [ ! -f "$REPO_ROOT/scripts/xpm_build.sh" ]; then
  echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the repo. Set REPO_ROOT." >&2
  exit 2
fi

LOGDIR="$REPO_ROOT/validation/mi250/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/build_$STAMP.log"
CTLOG="$LOGDIR/ctest_$STAMP.log"
CTLOG2="$LOGDIR/ctest_keepgoing_$STAMP.log"
KGLOG="$LOGDIR/keepgoing_build_$STAMP.log"
WORK="$REPO_ROOT/validation/mi250/work"
PROBEDIR="$WORK/space_probe"
mkdir -p "$WORK"

# Everything below is teed into $LOG as well as the job's stdout. Keep TEE_PID:
# the batch scheduler tears the process group down the moment this script
# returns, and a tee it never waited for can lose whatever is still in the pipe
# -- which is the verdict block, the one part nobody can reconstruct. Job
# 1000940 lost exactly that, and reported exit 0 for a run whose build failed.
# The tail of this script closes the pipe and waits before exiting.
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!

echo "==========================================================="
echo " S8c — xpm_build.sh --arch mi250, first execution"
echo " date      : $(date -Is)"
echo " host      : $(hostname)"
echo " repo      : $REPO_ROOT"
echo " commit    : $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " branch    : $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo " build dir : $BUILD_DIR"
echo " log       : $LOG"
echo "==========================================================="

# ------------------------------------------------------- 0. provenance
# The batch shell is NON-INTERACTIVE, so the `module` bash function does not
# exist -- job 1000931 died on exactly this. /etc/profile.d/modules.sh defines
# it. xpm_build.sh does the same dance internally; we do it here too because
# steps [0/6] and [5/6] need hipcc and rocm-smi on PATH whether or not the
# build step ever runs.
if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles 2>/dev/null
  # xpm_build.sh loads the arch row's modules, but it runs as a CHILD process,
  # so those loads die with it. THIS shell is the one that then runs ctest and
  # every binary the build produced -- and the host oracle carries no RPATH for
  # libquadmath, so whichever libquadmath is on LD_LIBRARY_PATH at launch is the
  # one that decides the fingerprint (578322f998a329c8 with the module,
  # 54901e8104607a77 with /usr/lib64, and ~390 rows read as false regressions).
  # Load it here too, so the run-time shell is the toolchain of record.
  #
  # NOT piped, and not inside $( ): `module` is a shell FUNCTION, so anything
  # that puts it in a subshell (a pipe to sed, a command substitution) loads the
  # module into a process that then exits and takes the environment with it.
  # Job 1000942 did exactly that and ran ctest against /usr/bin/g++ 7 with no
  # gcc lib64 on LD_LIBRARY_PATH, while printing "loaded".
  module load gcc/13.3.0 cmake/3.28.3
  echo "  module system: present; gcc/13.3.0 + cmake/3.28.3 loaded in THIS shell"
else
  echo "  module system: ABSENT in this shell -- xpm_build.sh will warn and use PATH"
fi
if ! command -v hipcc >/dev/null 2>&1; then
  export ROCM_PATH="$ROCM_ROOT"
  export PATH="$ROCM_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}"
  echo "  hipcc not on PATH -- fell back to ROCM_ROOT=$ROCM_ROOT"
fi

echo; echo "--- [0/6] provenance ---"

echo "  -- hostname / kernel --"
{ hostname; uname -srm; } 2>&1 | sed 's/^/    /'

echo "  -- rocm-smi --showproductname --"
rocm-smi --showproductname 2>&1 | sed 's/^/    /'

echo "  -- hipcc --version --"
hipcc --version 2>&1 | sed 's/^/    /'

# The run-time toolchain of THIS shell, recorded because ctest and everything
# it launches inherit it -- not the build's. A g++ or a libquadmath from
# /usr/lib64 here is a wrong-oracle hazard even when the build was perfect.
echo "  -- run-time toolchain of this shell (ctest inherits it) --"
{
  echo "g++    : $(command -v g++ || echo '<none>')  $(g++ -dumpversion 2>/dev/null)"
  echo "cmake  : $(command -v cmake || echo '<none>')  $(cmake --version 2>/dev/null | head -1)"
  echo "libquadmath on LD_LIBRARY_PATH:"
  printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | while read -r d; do
    [ -n "$d" ] && [ -e "$d/libquadmath.so.0" ] && echo "  $d/libquadmath.so.0"
  done
} 2>&1 | sed 's/^/    /'

echo "  -- Kokkos install: $KOKKOS_PREFIX --"
KCFG="$KOKKOS_PREFIX/include/KokkosCore_config.h"
if [ -r "$KCFG" ]; then
  # The ones that decide what this run can and cannot assert:
  #   HIP / SERIAL       which execution spaces exist at all
  #   ARCH_AMD_GFX90A    that it was built for THIS card
  #   LIBQUADMATH        whether the oracle-scored tests can score anything
  grep -E '#define KOKKOS_VERSION |KOKKOS_ENABLE_SERIAL|KOKKOS_ENABLE_OPENMP|KOKKOS_ENABLE_CUDA|KOKKOS_ENABLE_HIP\b|KOKKOS_ENABLE_LIBQUADMATH|KOKKOS_ARCH_AMD_GFX90A|KOKKOS_ARCH_AMD_GPU' \
       "$KCFG" 2>&1 | sed 's/^/    /'
else
  echo "    MISSING: $KCFG -- the mi250 row of ARCH_KOKKOS_PREFIX points somewhere that is not a Kokkos install"
fi

# State the expected consequence BEFORE the run, so the log cannot be read as
# a post-hoc excuse for whatever the skip count turns out to be.
if [ -r "$KCFG" ] && grep -q '^#define KOKKOS_ENABLE_LIBQUADMATH' "$KCFG"; then
  QUADMATH_IN_KOKKOS=yes
  echo "    => LIBQUADMATH ON: oracle-scored tests are expected to RUN"
else
  QUADMATH_IN_KOKKOS=no
  echo "    => LIBQUADMATH OFF: per tests/CMakeLists.txt:12 the oracle-scored tests"
  echo "       are expected to BUILD and runtime-SKIP (exit 77). Counted separately"
  echo "       below. A skip is not a pass -- and a target that does not BUILD is"
  echo "       not a skip either, so read NOT BUILT beside it."
fi

echo "  -- mi250 row of scripts/xpm_build.sh, as written --"
sed -n '/^declare -A ARCH_/,/^)/p' "$REPO_ROOT/scripts/xpm_build.sh" \
  | grep -E '^\s*\[mi250\]|^declare' | sed 's/^/    /'

# --------------------------------------------- 1. configure + build (THE TEST)
# This is the subject under test. It is invoked exactly as a user would, with
# no extra flags, no environment fixups and no fallbacks: whatever the mi250
# row does is what gets recorded.
echo; echo "--- [1/6] scripts/xpm_build.sh --arch mi250   (THE RECIPE, VERBATIM) ---"
if [ "$FRESH" = 1 ] && [ -d "$BUILD_DIR" ]; then
  echo "  FRESH=1: removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi
t_build_start=$SECONDS
set -x
bash "$REPO_ROOT/scripts/xpm_build.sh" --arch mi250 --build-dir "$BUILD_DIR"
rc_build=$?
set +x
t_build=$(( SECONDS - t_build_start ))
echo "  xpm_build exit: $rc_build   (${t_build}s)"

# ------------------------------------------------------- 2. build-info.txt
# Written by the CONFIGURE, from the top-level CMakeLists.txt, not by the
# wrapper -- so it exists even for builds nobody can otherwise trace. If the
# configure got that far it is here regardless of whether the compile did.
echo; echo "--- [2/6] build-info.txt ---"
if [ -f "$BUILD_DIR/build-info.txt" ]; then
  sed 's/^/    /' "$BUILD_DIR/build-info.txt"
else
  echo "    ABSENT -- configure did not reach the provenance stamp block"
fi

# ---------------------------------------------------------------- counting
# Counted off the per-test result lines rather than off ctest's summary: the
# summary does not break skipped out, and a target that never linked appears
# only here, as ***Not Run.
count_ctest() {   # $1 = ctest log; sets n_pass / n_fail / n_skip / n_notrun
  local results
  results=$(grep -E 'Test +#[0-9]+:' "$1")
  n_pass=$(  printf '%s\n' "$results" | grep -c   'Passed')
  n_fail=$(  printf '%s\n' "$results" | grep -cE '\*\*\*Failed|\*\*\*Exception|\*\*\*Timeout')
  n_skip=$(  printf '%s\n' "$results" | grep -c  '\*\*\*Skipped')
  n_notrun=$(printf '%s\n' "$results" | grep -c  '\*\*\*Not Run')
}
name_ctest() {    # $1 = ctest log, $2 = grep pattern -- prints the target names
  grep -E 'Test +#[0-9]+:' "$1" | grep -E "$2" \
    | sed -E 's/^.*Test +#[0-9]+: +([^ .]+).*$/    \1/'
}
print_counts() {  # $1 = label
  echo
  echo "  ============ FOUR COUNTS — $1"
  printf '   registered : %s\n' "$n_total"
  printf '   PASSED     : %s\n' "$n_pass"
  printf '   FAILED     : %s\n' "$n_fail"
  printf '   SKIPPED    : %s   <- asserts nothing\n' "$n_skip"
  printf '   NOT BUILT  : %s   (ctest ***Not Run: the target never linked)\n' "$n_notrun"
  echo "  ==========================================================="
  local acct=$(( n_pass + n_fail + n_skip + n_notrun ))
  if [ "$acct" -ne "${n_total:-0}" ]; then
    echo "   WARNING: $acct accounted for, $n_total registered -- the counts do not close."
  fi
}

# -------------------------------------------- 3. ctest, AS THE RECIPE LEFT IT
echo; echo "--- [3/6] ctest on the tree AS THE RECIPE LEFT IT   (THE HEADLINE COUNTS) ---"
n_total=0; n_pass=0; n_fail=0; n_skip=0; n_notrun=0; rc_ctest=2
if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
  # `ctest --test-dir` on a directory with no test file exits 0. An exit code
  # from a run that never happened is the cheapest way to report a pass that
  # did not occur, so refuse to run rather than report it.
  echo "  NO $BUILD_DIR/CTestTestfile.cmake -- refusing to run ctest."
  echo "  (ctest --test-dir on a missing/empty dir exits 0 and proves nothing.)"
else
  n_total=$(ctest --test-dir "$BUILD_DIR" -N 2>/dev/null | sed -n 's/^Total Tests: *//p' | tail -1)
  n_total=${n_total:-0}
  echo "  registered targets: $n_total"
  ctest --test-dir "$BUILD_DIR" --timeout "$CTEST_TIMEOUT" > "$CTLOG" 2>&1
  rc_ctest=$?
  tail -12 "$CTLOG" | sed 's/^/    /'
  echo "  (full ctest output: $CTLOG)"
  count_ctest "$CTLOG"
fi
print_counts "RECIPE AS SHIPPED"
r_pass=$n_pass; r_fail=$n_fail; r_skip=$n_skip; r_notrun=$n_notrun; r_total=$n_total

if [ "${r_skip:-0}" -gt 0 ]; then
  echo; echo "  -- SKIPPED targets, named (gfx90a Kokkos LIBQUADMATH=$QUADMATH_IN_KOKKOS) --"
  name_ctest "$CTLOG" '\*\*\*Skipped'
fi
if [ "${r_fail:-0}" -gt 0 ]; then
  echo; echo "  -- FAILED targets, named --"
  name_ctest "$CTLOG" '\*\*\*Failed|\*\*\*Exception|\*\*\*Timeout'
fi
if [ "${r_notrun:-0}" -gt 0 ]; then
  echo; echo "  -- NOT BUILT targets, named --"
  name_ctest "$CTLOG" '\*\*\*Not Run'
fi

# ------------------------------------------------- 4. DIAGNOSTIC keep-going
# Not the recipe. `make` abandons a directory at its first error, so the counts
# above cannot distinguish "this target is broken" from "this target sat behind
# a broken one". Keeping going lets every target be attempted on its own
# merits. Same configure, same flags, same sources.
#
# The keep-going flag is SPELLED DIFFERENTLY per generator and the wrong
# spelling fails silently-ish: ninja wants `-k 0` (0 = unlimited), gmake wants
# a bare `-k` and reads `-k 0` as "keep going, and build a target named 0",
# which dies with "No rule to make target '0'" in zero seconds and diagnoses
# nothing. Read the generator out of the cache rather than assuming.
rc_keep=0
did_keepgoing=no
if [ "$rc_build" -ne 0 ] && [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  did_keepgoing=yes
  generator=$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt")
  case "$generator" in
    Ninja*) keep_flag=(-k 0) ;;
    *)      keep_flag=(-k)   ;;
  esac
  echo; echo "--- [4/6] DIAGNOSTIC keep-going rebuild (NOT THE RECIPE) ---"
  echo "  generator: ${generator:-unknown}"
  echo "  cmake --build $BUILD_DIR -j$(nproc) -- ${keep_flag[*]}"
  t_keep_start=$SECONDS
  cmake --build "$BUILD_DIR" -j"$(nproc)" -- "${keep_flag[@]}" > "$KGLOG" 2>&1
  rc_keep=$?
  echo "  keep-going build exit: $rc_keep   ($(( SECONDS - t_keep_start ))s)   log: $KGLOG"
  echo "  distinct compiler diagnostics (top 15 by count):"
  grep -hoE "error: [^/]{0,90}" "$KGLOG" \
    | sed -E 's/^error: //; s/[[:space:]]+$//' \
    | grep -vE '^$' | sort | uniq -c | sort -rn | head -15 | sed 's/^/    /'
else
  echo; echo "--- [4/6] DIAGNOSTIC keep-going rebuild — SKIPPED (recipe build succeeded) ---"
fi

# ---------------------------------------------- 5. execution space, resolved
# The one thing a HIP build can silently get wrong: a Kokkos with both SERIAL
# and HIP enabled will happily fall back to Serial as the default device, and
# every "device" test then passes on the CPU. Ask the RUNTIME -- the config
# header says what is ENABLED, only a running program says what was CHOSEN.
#
# Built as a separate throwaway CMake project rather than by running one of the
# repo's own test binaries, because on this target there may not BE one; see
# the header of kokkos_space_probe.cpp. Same Kokkos prefix, same compiler, same
# flags as the arch row -- read back out of the recipe's own CMakeCache.txt so
# they cannot drift from it.
echo; echo "--- [5/6] execution space actually resolved by the runtime ---"
rc_exec=1
EXEC_LINE="(not determined)"
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  probe_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p'        "$BUILD_DIR/CMakeCache.txt" | head -1)
  probe_pfx=$(sed -n 's/^CMAKE_PREFIX_PATH:[A-Z]*=//p'         "$BUILD_DIR/CMakeCache.txt" | head -1)
  probe_flg=$(sed -n 's/^CMAKE_CXX_FLAGS:[A-Z]*=//p'           "$BUILD_DIR/CMakeCache.txt" | head -1)
  probe_rel=$(sed -n 's/^CMAKE_CXX_FLAGS_RELEASE:[A-Z]*=//p'   "$BUILD_DIR/CMakeCache.txt" | head -1)
  echo "  toolchain taken from the recipe's own CMakeCache.txt:"
  echo "    CMAKE_CXX_COMPILER      = $probe_cxx"
  echo "    CMAKE_PREFIX_PATH       = $probe_pfx"
  echo "    CMAKE_CXX_FLAGS         = $probe_flg"
  echo "    CMAKE_CXX_FLAGS_RELEASE = $probe_rel"

  rm -rf "$PROBEDIR"; mkdir -p "$PROBEDIR/src"
  cp "$REPO_ROOT/validation/mi250/kokkos_space_probe.cpp" "$PROBEDIR/src/"
  cat > "$PROBEDIR/src/CMakeLists.txt" <<'PROBE_CMAKE'
cmake_minimum_required(VERSION 3.16)
project(kokkos_space_probe LANGUAGES CXX)
find_package(Kokkos REQUIRED)
add_executable(kokkos_space_probe kokkos_space_probe.cpp)
target_link_libraries(kokkos_space_probe PRIVATE Kokkos::kokkos)
PROBE_CMAKE
  cmake -S "$PROBEDIR/src" -B "$PROBEDIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$probe_cxx" \
        -DCMAKE_PREFIX_PATH="$probe_pfx" \
        -DCMAKE_CXX_FLAGS="$probe_flg" \
        -DCMAKE_CXX_FLAGS_RELEASE="$probe_rel" > "$PROBEDIR/configure.log" 2>&1
  rc_pc=$?
  cmake --build "$PROBEDIR/build" -j4 >> "$PROBEDIR/configure.log" 2>&1
  rc_pb=$?
  echo "  probe configure: $rc_pc   probe build: $rc_pb"
  if [ "$rc_pb" -ne 0 ]; then
    echo "  probe did not build -- last 25 lines of $PROBEDIR/configure.log:"
    tail -25 "$PROBEDIR/configure.log" | sed 's/^/    /'
  else
    PROBE_OUT=$("$PROBEDIR/build/kokkos_space_probe" 2>&1)
    echo "$PROBE_OUT" | sed 's/^/    /'
    echo
    echo "  -- the lines that decide it --"
    echo "$PROBE_OUT" | grep -E '^resolved-|^probe-kernel|^probe-verdict' | sed 's/^/    /'
    EXEC_LINE=$(echo "$PROBE_OUT" | grep -m1 '^resolved-default-execution-space:')
    if echo "$PROBE_OUT" | grep -q '^probe-verdict: HIP-EXECUTED'; then
      echo "  => DEFAULT EXECUTION SPACE IS HIP, and a kernel really ran in it."
      rc_exec=0
    else
      echo "  => NOT HIP, or the kernel returned wrong values. NOT device evidence."
    fi
  fi
else
  echo "  no CMakeCache.txt -- configure never completed; nothing to probe."
fi

# The oracle's libquadmath is resolved through LD_LIBRARY_PATH with no RPATH,
# so which one a binary picks up is a property of THIS shell, not of the build.
echo
echo "  -- libquadmath actually linked by sweep_accuracy (no RPATH; LD_LIBRARY_PATH decides) --"
if [ -x "$BUILD_DIR/tests/sweep_accuracy" ]; then
  ldd "$BUILD_DIR/tests/sweep_accuracy" 2>&1 | grep -i quadmath | sed 's/^/    /' \
    || echo "    (no quadmath line in ldd output)"
else
  echo "    sweep_accuracy not built -- cannot check"
fi

# --------------------------------- 6. ctest after the diagnostic, + triage
echo; echo "--- [6/6] DIAGNOSTIC ctest + per-target first error ---"
k_pass=-1; k_fail=-1; k_skip=-1; k_notrun=-1
if [ "$did_keepgoing" = yes ]; then
  ctest --test-dir "$BUILD_DIR" --timeout "$CTEST_TIMEOUT" > "$CTLOG2" 2>&1
  rc_ctest2=$?
  echo "  ctest exit: $rc_ctest2   log: $CTLOG2"
  count_ctest "$CTLOG2"
  print_counts "AFTER DIAGNOSTIC KEEP-GOING BUILD (not the recipe)"
  k_pass=$n_pass; k_fail=$n_fail; k_skip=$n_skip; k_notrun=$n_notrun
  if [ "$n_skip" -gt 0 ]; then
    echo; echo "  -- SKIPPED targets, named --"; name_ctest "$CTLOG2" '\*\*\*Skipped'
  fi
  if [ "$n_fail" -gt 0 ]; then
    echo; echo "  -- FAILED targets, named --"
    name_ctest "$CTLOG2" '\*\*\*Failed|\*\*\*Exception|\*\*\*Timeout'
  fi
  if [ "$n_notrun" -gt 0 ]; then
    echo; echo "  -- NOT BUILT targets, named --"; name_ctest "$CTLOG2" '\*\*\*Not Run'
  fi

  # First error per still-missing target. Rebuilt one at a time with -j1 so the
  # diagnostic belongs to the target that produced it -- with -j the messages
  # from forty parallel clangs interleave mid-line and cannot be attributed
  # (the [3/6] log of job 1000939 has the shredded proof of that).
  # These compiles all die at parse time, so they are cheap; the budget exists
  # only so a surprise cannot eat the job's wall clock.
  if [ "$TRIAGE_BUDGET" -gt 0 ] && [ "$n_notrun" -gt 0 ]; then
    echo
    echo "  -- first compiler error per unbuilt target (serial, budget ${TRIAGE_BUDGET}s) --"
    t_triage_end=$(( SECONDS + TRIAGE_BUDGET ))
    name_ctest "$CTLOG2" '\*\*\*Not Run' | tr -d ' ' | sort -u > "$PROBEDIR.notbuilt.txt"
    while read -r tgt; do
      [ -z "$tgt" ] && continue
      if [ "$SECONDS" -ge "$t_triage_end" ]; then
        echo "    (triage budget exhausted -- remaining targets not triaged)"
        break
      fi
      first=$(cmake --build "$BUILD_DIR" --target "$tgt" -j1 2>&1 \
              | grep -m1 -E 'error:|No rule to make target' \
              | sed -E 's#^.*/([A-Za-z0-9_.]+\.(hpp|cpp|h):[0-9]+:[0-9]+: error:)#\1#' \
              | cut -c1-150)
      printf '    %-34s %s\n' "$tgt" "${first:-<no error line captured>}"
    done < "$PROBEDIR.notbuilt.txt"
  fi
else
  echo "  (no diagnostic pass -- the recipe build succeeded)"
fi

# ----------------------------------------------------------------- verdict
echo
echo "==========================================================="
echo " S8c RESULT"
echo "   recipe build    : exit $rc_build   (${t_build}s)"
echo "   execution space : $EXEC_LINE"
echo "   FOUR COUNTS (recipe as shipped, of $r_total registered):"
echo "       passed=$r_pass  failed=$r_fail  skipped=$r_skip  not-built=$r_notrun"
if [ "$did_keepgoing" = yes ]; then
  echo "   FOUR COUNTS (diagnostic keep-going build, NOT the recipe):"
  echo "       passed=$k_pass  failed=$k_fail  skipped=$k_skip  not-built=$k_notrun"
fi
if [ "$rc_build" -eq 0 ] && [ "$rc_exec" -eq 0 ] && [ "${r_fail:-1}" -eq 0 ] && [ "${r_notrun:-1}" -eq 0 ]; then
  echo " VERDICT : the mi250 arch block BUILDS AND TESTS on real gfx90a."
  echo "           NOT a green suite: $r_skip targets skipped for want of the"
  echo "           host oracle and asserted nothing."
else
  echo " VERDICT : the mi250 arch block did NOT come through clean."
  echo "           Report what broke; do not patch it in this script (Rule 4)."
fi
echo " logs    : $LOG"
echo "           $CTLOG"
if [ "$did_keepgoing" = yes ]; then
  echo "           $KGLOG"
  echo "           $CTLOG2"
fi
echo "==========================================================="

# Exit code carries the RECIPE's verdict -- not the diagnostic's, not the
# suite's. Skips are expected here and must not colour it; failures, unlinked
# targets and a non-HIP execution space must.
if [ "$rc_build" -eq 0 ] && [ "$rc_exec" -eq 0 ] \
   && [ "${r_fail:-1}" -eq 0 ] && [ "${r_notrun:-1}" -eq 0 ]; then
  rc_final=0
else
  rc_final=1
fi

# Close the pipe so tee sees EOF, then wait for it: without this the scheduler
# can kill tee with the verdict block still unwritten (see TEE_PID above).
exec 1>&- 2>&-
wait "$TEE_PID" 2>/dev/null || true
exit "$rc_final"
