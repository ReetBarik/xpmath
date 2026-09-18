#!/usr/bin/env bash
# ===========================================================================
# scripts/xpm_build.sh — configure and build xpmath for a NAMED ARCHITECTURE,
#                        as TWO TREES, and return ONE verdict over both
# ===========================================================================
#
#   scripts/xpm_build.sh --arch {host|a100|mi250} [--build-dir DIR]
#                        [--only {host|device|both}]
#                        [--kokkos {use-existing|build}] [--opt LEVEL]
#                        [--no-kokkos] [--no-test]
#
# Selecting a target is an ARGUMENT here, not an edit. Before this script the
# only recipe was scripts/build_with_kokkos.sh, whose GPU was a literal on line
# 45 and whose HIP branch was unreachable; retargeting meant editing two lines
# and remembering which two.
#
# ---------------------------------------------------------------------------
# WHY TWO TREES -- CORE_PLAN C5
# ---------------------------------------------------------------------------
# CMake supports exactly ONE CXX compiler per project and offers no per-target
# override. So "host tools with g++, device tests with nvcc/hipcc" is not a
# property this build can express in one tree: it is two configures, and this
# script drives both.
#
#   <build-dir>/host    always g++, always -ffp-contract=off, always
#                       XPMATH_WITH_KOKKOS=OFF, XPMATH_BUILD_DEVICE_TARGETS=OFF.
#                       IDENTICAL for host, a100 and mi250.
#   <build-dir>/device  the arch's compiler, contraction spelling and Kokkos
#                       prefix from the data table below, with
#                       XPMATH_BUILD_HOST_TARGETS=OFF. For --arch host this is
#                       still g++ with the serial harness backend, which is what
#                       makes the device-side tests runnable without a GPU.
#
# THIS IS NOT A TIDINESS EXERCISE. It is the whole of what went wrong on A100 in
# Cobalt job 1000938, where `--arch a100` set nvcc_wrapper project-wide and ALL
# ELEVEN red results were host-only translation units handed a device pass:
#
#   (a) sweep_accuracy         std::initializer_list<__float128> "not supported
#                              in device code", taking sweep_absolute_gate,
#                              sweep_monotone_gate, both gate selftests and
#                              oracle_conv_test down with it ............... 5
#   (b) hello_test             the same, reached through test_utils.hpp ..... 1
#   (c) five standalone smokes `no operator "<<"`, because the device pass
#                              defines XPMATH_ON_DEVICE and #if-guards away
#                              every ostream overload ..................... 5
#
# All eleven are HOST-tagged targets (tests/CMakeLists.txt), so all eleven are
# built by the host tree's g++ and none of them is ever shown to nvcc again.
# The count of record was five; five was the sweep_accuracy cluster alone.
#
# THE HOST TREE IS KOKKOS-FREE, AND THAT IS WHAT MAKES IT IDENTICAL ACROSS
# ARCHES. After CORE_PLAN C4 no test links Kokkos, and the eight demos -- the
# only remaining Kokkos consumers, and the only consumers of
# third_party/include/ -- are device-tagged. A host tree configured with the
# a100 Kokkos prefix would therefore FIND a Kokkos that nothing in it links,
# and would differ from the mi250 host tree in a way no compiled byte reflects.
# Worse, it would make `--only host` impossible on a node without that arch's
# Kokkos install. So the host tree passes XPMATH_WITH_KOKKOS=OFF on every arch
# and the arch's prefix is only ever handed to the device tree.
#
# WHAT --no-kokkos NOW MEANS: it drops Kokkos from the DEVICE tree, which is the
# only tree that had it. The eight demos are then not built and the device tree
# is the Kokkos-free device harness alone. The host tree is unaffected because
# it never had Kokkos to drop.
#
# ---------------------------------------------------------------------------
# APPLES-TO-APPLES: WHAT IS HELD IDENTICAL, AND WHAT IS NOT
# ---------------------------------------------------------------------------
# Numbers produced under two of these arches are only comparable to the extent
# that the things which move a floating-point result are the same. State them,
# do not assume them.
#
# HELD IDENTICAL across host, a100 and mi250:
#
#   -O level        --opt, default O3, reaching the compiler as
#                   CMAKE_BUILD_TYPE=Release plus CMAKE_CXX_FLAGS_RELEASE
#                   ="-O<level> -DNDEBUG". It is NOT appended as a bare flag,
#                   so `cmake --build` and any nested configure see the same
#                   level, and CMakeCache.txt records it where build-info.txt
#                   can read it back. BOTH TREES GET THE SAME LEVEL.
#
#   C++ standard    C++17 for THIS project, on every arch, in both trees. The
#                   three Kokkos installs are all C++20 because Kokkos 5.1.0
#                   hard-errors on 17 (KI-3), and Kokkos's
#                   INTERFACE_COMPILE_FEATURES cxx_std_20 wins on any target
#                   that links it -- which after C4 is the eight demos and
#                   nothing else. The two numbers are deliberately different
#                   and must stay different; see .github/workflows/ci.yml.
#
#   FP contraction  OFF, on every arch and in both trees -- but spelled
#                   differently, and with a different reach, per toolchain. See
#                   the next block.
#
#   THE HOST TREE   held identical across all three arches, by construction:
#                   same compiler, same flags, no Kokkos. That is the point of
#                   splitting it out. An accuracy number measured under
#                   `--arch mi250` comes out of the same host tree as one
#                   measured under `--arch a100`.
#
# NOT held identical, by construction:
#
#   compiler        DEVICE TREE ONLY: g++ (host), nvcc_wrapper wrapping
#                   nvcc+g++ (a100), hipcc (mi250). There is no common compiler
#                   for three vendors. The host tree is g++ everywhere.
#   execution space Serial, Cuda, HIP.
#   host oracle     NO LONGER A DIFFERENCE, and the row is kept to say so. It
#                   used to read: libquadmath is present in the host and a100
#                   Kokkos installs and ABSENT from the mi250 one, so the
#                   oracle-scored tests runtime-SKIP on mi250. The B-arc removed
#                   the last libquadmath call site from tests/ and the skip that
#                   hid it; sweep_accuracy's oracle is MPFR/MPC and links no
#                   libquadmath on any arch. Whether the Kokkos install has
#                   libquadmath no longer changes which tests score -- and the
#                   oracle-scored targets now live in a tree that has no Kokkos
#                   install at all.
#   device build    a100 and mi250 additionally compile every device-tagged TU
#                   through a device pass; host does not.
#
# ---------------------------------------------------------------------------
# FP CONTRACTION, SPELLED OUT
# ---------------------------------------------------------------------------
# The Dekker twoProduct needs `a1*b1 - c11` to be two rounded operations. Fuse
# them and the error term collapses to zero and the error-free transform stops
# being error-free -- silently, because nothing fails.
#
#   host tree           -ffp-contract=off given to g++, on every arch.
#   device tree, host   -ffp-contract=off given to g++   as a CMAKE_CXX_FLAGS entry.
#   device tree, mi250  -ffp-contract=off given to hipcc as a CMAKE_CXX_FLAGS entry.
#          hipcc is clang, so the one spelling covers its host AND device pass.
#   device tree, a100   --fmad=false given to nvcc_wrapper as a CMAKE_CXX_FLAGS
#          entry, which forwards it to nvcc and so reaches the DEVICE pass. The
#          host pass of the same TU is g++ and is covered by
#          tests/CMakeLists.txt's per-target -ffp-contract=off on the EFT targets.
#
# WHICH TARGETS GET IT: all of them, in both trees. This is a whole-build
# CMAKE_CXX_FLAGS entry, not a per-target option, because the apples-to-apples
# claim is about the build, not about eight test targets. tests/CMakeLists.txt
# keeps its per-target flags unchanged; per-target options are appended AFTER
# CMAKE_CXX_FLAGS, so the `*_contract_on` reporters still override the host-side
# flag and still report what contraction collapses.
#
# RECORDED, NOT FIXED HERE -- the nvcc per-target guard has never engaged.
# tests/CMakeLists.txt guards its `--fmad=false` / `--fmad=true` with
# $<COMPILE_LANGUAGE:CUDA>. The top-level CMakeLists.txt declares
# `project(xpmath ... LANGUAGES CXX)`, so under nvcc_wrapper-as-CXX every TU is
# CXX and that genex never evaluates true. Measured, not inferred: the S1
# STATUS block (docs/UPSTREAM_PLAN_STATUS.md, finding (e)) counted ZERO
# occurrences of `--fmad` across the whole A100 build, against a prediction of
# zero. Two consequences on a100, both of which follow from that measurement:
#   * the `--fmad=false` in the a100 block below is the ONLY thing switching
#     device contraction off, and
#   * the `_contract_on` reporters do NOT get `--fmad=true` on their device
#     pass, so on a100 they are host-contracted and device-uncontracted.
# Fixing that means enabling CUDA as a project language, which changes how
# every TU is compiled. It is out of scope here and is deliberately left as a
# recorded defect rather than a silent one.
#
# ---------------------------------------------------------------------------
# THE MERGED VERDICT, AND WHY IT IS WRITTEN THE WAY IT IS
# ---------------------------------------------------------------------------
# This script now runs ctest in each tree it built and prints ONE summary with a
# per-tree breakdown. A merged summary that hides one tree's failures is worse
# than no summary, so:
#
#   * it EXITS 1 if either tree fails -- 1 and not "the number of bad trees",
#     because 2 is already this script's usage/environment error code (die) and
#     a caller should not have to tell "both trees red" from "you spelled --arch
#     wrong";
#   * it asserts the tree DIRECTORY exists and carries a CTestTestfile.cmake
#     before invoking ctest, because `ctest --test-dir` on a missing directory
#     exits 0 and a green run over nothing is the trap this repository has been
#     bitten by before (CLAUDE.md, "Working on accuracy");
#   * it requires ctest's own "N tests failed out of M" line to be present and M
#     to be NONZERO. A tree that ran zero tests is a FAILED tree here, not a
#     passing one;
#   * a tree that failed to CONFIGURE or BUILD gets a row saying so rather than
#     being quietly absent, and its ctest is not attempted;
#   * a tree that was not requested (--only) gets a SKIPPED row, which is
#     visibly different from a failed one.
#
# `--no-test` builds without testing. Use it when a separate ctest invocation
# with its own -R/-E is going to follow; do not use it to make a red build look
# quiet.
#
# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------
# build-info.txt is written into EACH build directory by ITS configure, from
# the top-level CMakeLists.txt -- not by this script. That is on purpose: a
# stamp only this wrapper wrote would be absent from exactly the builds nobody
# can trace, which are the ones made by calling cmake directly. It carries a
# `tree: host|device|both` field derived from the two options, so the two
# stamps this script produces are told apart by content and not by path, and
# the ctest target `build_provenance` -- which is registered in EVERY tree --
# fails when the stamp is missing or short a field.
#
# `arch:` records the arch that was ASKED FOR, in both trees. The host tree of
# an `--arch a100` run is byte-for-byte the host tree of an `--arch mi250` run,
# but it was produced as part of an a100 campaign and that is what makes its
# artifacts traceable. `tree:` is what says which half it is.
# ===========================================================================

set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "$_self/.." && pwd)

die() { echo "xpm_build: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
scripts/xpm_build.sh — configure and build xpmath for a NAMED ARCHITECTURE,
                       as TWO TREES, and return ONE verdict over both

  scripts/xpm_build.sh --arch {host|a100|mi250} [--build-dir DIR]
                       [--only {host|device|both}]
                       [--kokkos {use-existing|build}] [--opt LEVEL]
                       [--no-kokkos] [--no-test]

  --arch        host | a100 | mi250          (required)
  --build-dir   parent directory; the two trees are <dir>/host and <dir>/device
                                             (default: $REPO_ROOT/build)
  --only        host | device | both         (default: both)
  --kokkos      use-existing | build         (default: use-existing)
  --opt         O0 O1 O2 O3 Os Og Ofast      (default: O3)
  --no-kokkos   configure the DEVICE tree -DXPMATH_WITH_KOKKOS=OFF; the host
                tree is Kokkos-free on every arch already
  --no-test     build only; do not run ctest and do not print a verdict
EOF
}

# ---------------------------------------------------------------------------
# THE ARCH TABLE. Pure data, one row per key. A row that is not selected is
# never expanded and never executed -- it is a value in an array, not a
# commented-out line somebody has to remember to uncomment.
#
# EVERY ROW HERE DESCRIBES THE DEVICE TREE. The host tree takes none of them:
# it is g++ / -ffp-contract=off / no Kokkos on all three arches, which is the
# constant the HOST_* values below spell out.
# ---------------------------------------------------------------------------
declare -A ARCH_DESC=(
  [host]="x86_64 login/compute node, Kokkos Serial"
  [a100]="NVIDIA A100 / sm_80, Kokkos CUDA"
  [mi250]="AMD MI250X / gfx90a, Kokkos HIP"
)
declare -A ARCH_MODULES=(
  [host]="gcc/13.3.0 cmake/3.28.3"
  [a100]="gcc/13.3.0 cmake/3.28.3 cuda/12.9.1"
  [mi250]="gcc/13.3.0 cmake/3.28.3 rocm/7.0.2"
)
declare -A ARCH_CC=(
  [host]="gcc"
  [a100]="gcc"
  [mi250]="gcc"
)
declare -A ARCH_CXX=(
  [host]="g++"
  [a100]="/home/rbarik/kokkos-src-5.1.0-cuda-sm80/bin/nvcc_wrapper"
  [mi250]="/soft/compilers/rocm/rocm-7.0.2/bin/hipcc"
)
declare -A ARCH_KOKKOS_PREFIX=(
  [host]="$HOME/kokkos-install-quadmath"
  [a100]="$HOME/kokkos-install-cuda-sm80-quadmath"
  [mi250]="$HOME/xpm_device/kokkos-hip-gfx90a"
)
# The Kokkos_ARCH_* macro and backend, used only by --kokkos build. They are
# stated for every arch anyway so the stamp can name the arch the prefix was
# supposed to have been built for.
declare -A ARCH_KOKKOS_ARCH=(
  [host]="NONE"
  [a100]="Kokkos_ARCH_AMPERE80"
  [mi250]="Kokkos_ARCH_AMD_GFX90A"
)
declare -A ARCH_KOKKOS_BACKEND=(
  [host]="serial"
  [a100]="cuda"
  [mi250]="hip"
)
declare -A ARCH_CUDA_ARCHITECTURES=(
  [host]=""
  [a100]="80"
  [mi250]=""
)
declare -A ARCH_CONTRACT=(
  [host]="-ffp-contract=off"
  [a100]="--fmad=false"
  # --offload-arch is load-bearing for C8. After C4, sweep_device and the
  # other device TUs do not link Kokkos, so they do not inherit
  # Kokkos_HIP_ARCHITECTURES=gfx90a. Login-node hipcc then defaults to
  # gfx906 (MI50). MEASURED on the first 1001915 build: strings showed
  # only gfx906. Pin it here, on the device-tree CMAKE_CXX_FLAGS, so a
  # login-node --arch mi250 build cannot silently target the wrong ISA.
  # Do NOT add -Xarch_device: that is a CUDA-clang spelling. hipcc
  # accepts it as one unused argument and warns at every link
  # (MEASURED, c9_hip_eft). The HIP device pass is pinned by the
  # volatile EFT wrappers in include/xp/config.hpp, not by this flag.
  [mi250]="-ffp-contract=off --offload-arch=gfx90a"
)

# The host tree, as constants rather than as a fourth table row -- there is no
# per-arch choice to make here and a row would invite one.
HOST_CC="gcc"
HOST_CXX="g++"
HOST_CONTRACT="-ffp-contract=off"

# ctest invocation, identical in both trees. -j8 and --timeout 1800 are the
# numbers every gate recipe in docs/ and in CORE_PLAN uses; matching them keeps
# a wrapper run and a hand-run comparable. -j8 does not shorten the suite much:
# it is dominated by sweep_monotone_gate_selftest at ~11 min, which is one
# serialized test.
CTEST_JOBS=8
CTEST_TIMEOUT=1800

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ARCH=""
BUILD_DIR=""
ONLY="both"
KOKKOS_MODE="use-existing"
OPT="O3"
WITH_KOKKOS=ON
RUN_TESTS=1

while [ $# -gt 0 ]; do
  case "$1" in
    --arch)       ARCH=${2:-};        shift 2 || die "--arch needs a value" ;;
    --build-dir)  BUILD_DIR=${2:-};   shift 2 || die "--build-dir needs a value" ;;
    --only)       ONLY=${2:-};        shift 2 || die "--only needs a value" ;;
    --kokkos)     KOKKOS_MODE=${2:-}; shift 2 || die "--kokkos needs a value" ;;
    --opt)        OPT=${2:-};         shift 2 || die "--opt needs a value" ;;
    --no-kokkos)  WITH_KOKKOS=OFF;    shift ;;
    --no-test)    RUN_TESTS=0;        shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -n "$ARCH" ] || { usage >&2; die "--arch is required"; }
[ -n "${ARCH_DESC[$ARCH]+set}" ] || die "unknown --arch '$ARCH' (host|a100|mi250)"
case "$ONLY" in
  host|device|both) ;;
  *) die "unknown --only '$ONLY' (host|device|both)" ;;
esac
case "$KOKKOS_MODE" in
  use-existing|build) ;;
  *) die "unknown --kokkos '$KOKKOS_MODE' (use-existing|build)" ;;
esac
OPT=${OPT#-}
case "$OPT" in
  O0|O1|O2|O3|Os|Og|Ofast) ;;
  *) die "unknown --opt '$OPT' (O0 O1 O2 O3 Os Og Ofast)" ;;
esac
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build}

# Which trees this invocation is responsible for.
BUILD_HOST_TREE=0
BUILD_DEVICE_TREE=0
case "$ONLY" in
  host)   BUILD_HOST_TREE=1 ;;
  device) BUILD_DEVICE_TREE=1 ;;
  both)   BUILD_HOST_TREE=1; BUILD_DEVICE_TREE=1 ;;
esac

DEV_CXX=${ARCH_CXX[$ARCH]}
DEV_CC=${ARCH_CC[$ARCH]}
DEV_CONTRACT=${ARCH_CONTRACT[$ARCH]}
KOKKOS_PREFIX=${ARCH_KOKKOS_PREFIX[$ARCH]}

# ---------------------------------------------------------------------------
# Modules. This used to be the libquadmath-fingerprint trap: a binary RUN
# without gcc/13.3.0 resolved a different libquadmath, moved the oracle
# fingerprint off 578322f998a329c8, and made ~390 sweep rows read as
# regressions that were not there. That is RETIRED -- sweep_accuracy links no
# libquadmath and MPFR/MPC are correctly rounded, so the answer no longer
# depends on which build is found (CLAUDE.md, "Working on accuracy"). Load them
# anyway: the rest of the suite is not sweep_accuracy, and this is the compiler
# of record for every number in validation/.
#
# The arch's whole module set is loaded even for --only host. gcc and cmake are
# what the host tree needs and they are in every row; the vendor module is
# inert in a tree that does not use its compiler, and dropping it would make
# `--only host` and `--only both` load different environments for the same tree.
# ---------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles 2>/dev/null
  for m in ${ARCH_MODULES[$ARCH]}; do
    # NOT piped anywhere. `module` is a shell function that edits PATH; put it
    # on the left of a pipe and it runs in a subshell, the environment change
    # is discarded, and the very next line reports cmake as not installed.
    echo "  module load $m"
    module load "$m" || die "module load $m failed"
  done
else
  echo "xpm_build: WARNING no module system in this shell; using whatever is on PATH" >&2
fi

command -v cmake >/dev/null 2>&1 || die "cmake not on PATH after loading modules"
if [ "$BUILD_HOST_TREE" = 1 ]; then
  command -v "$HOST_CXX" >/dev/null 2>&1 || [ -x "$HOST_CXX" ] \
    || die "host-tree C++ compiler '$HOST_CXX' not found"
fi
if [ "$BUILD_DEVICE_TREE" = 1 ]; then
  command -v "$DEV_CXX" >/dev/null 2>&1 || [ -x "$DEV_CXX" ] \
    || die "device-tree C++ compiler '$DEV_CXX' not found (arch $ARCH)"
fi

# ---------------------------------------------------------------------------
# Kokkos -- a DEVICE-TREE concern only, and resolved only when that tree is
# being built. `--only host` on a node with no Kokkos install for this arch is
# a supported and useful thing to do, and demanding the prefix anyway would
# make it fail over a directory nothing in that tree reads.
# ---------------------------------------------------------------------------
if [ "$BUILD_DEVICE_TREE" != 1 ]; then
  echo "xpm_build: --only $ONLY -- not resolving the ${ARCH} Kokkos prefix (device tree not built)"
  KOKKOS_PREFIX=""
elif [ "$WITH_KOKKOS" = OFF ]; then
  # --no-kokkos drops the arch's Kokkos settings -- prefix, arch macro, backend
  # -- and keeps its toolchain: the modules, the compiler and the contraction
  # spelling are properties of the machine, not of whether Kokkos is linked.
  # The arch name is still recorded in the stamp.
  echo "xpm_build: XPMATH_WITH_KOKKOS=OFF -- ignoring the ${ARCH} Kokkos prefix"
  KOKKOS_PREFIX=""
elif [ "$KOKKOS_MODE" = build ]; then
  # Delegate to the existing recipe, parameterized through its environment.
  # KOKKOS_ONLY=1 stops it configuring the repo itself: this script owns the
  # repo configure so that --opt, --build-dir and the stamp behave the same
  # way in both --kokkos modes.
  KOKKOS_INSTALL_ROOT=${KOKKOS_INSTALL_ROOT:-$HOME/kokkos-xpm-$ARCH}
  echo "xpm_build: building Kokkos for $ARCH into $KOKKOS_INSTALL_ROOT"
  KOKKOS_ARCH_FLAG=${ARCH_KOKKOS_ARCH[$ARCH]} \
  KOKKOS_BACKEND=${ARCH_KOKKOS_BACKEND[$ARCH]} \
  KOKKOS_CUDA_ARCHITECTURES=${ARCH_CUDA_ARCHITECTURES[$ARCH]} \
  KOKKOS_CXX=${DEV_CXX} \
  KOKKOS_ONLY=1 \
    bash "$_self/build_with_kokkos.sh" "$KOKKOS_INSTALL_ROOT" \
      || die "scripts/build_with_kokkos.sh failed for arch $ARCH"
  KOKKOS_PREFIX=$KOKKOS_INSTALL_ROOT/kokkos/install/kokkos-5.1.0/Release
  [ -d "$KOKKOS_PREFIX" ] || die "Kokkos build left no install at $KOKKOS_PREFIX"
else
  [ -d "$KOKKOS_PREFIX" ] \
    || die "no Kokkos install at $KOKKOS_PREFIX (arch $ARCH). Pass --kokkos build, or --no-kokkos."
fi

# ---------------------------------------------------------------------------
# Per-tree state. Every tree this script knows about gets a row in the summary,
# including the ones it did not build, so "skipped" and "absent" never look the
# same.
# ---------------------------------------------------------------------------
TREES=(host device)
# SKIPPED | CONFIGURE-FAILED | BUILD-FAILED | BUILT | PASS | FAIL
#   SKIPPED  not requested by --only
#   BUILT    built, and no verdict taken because --no-test
declare -A TREE_STATE=([host]=SKIPPED [device]=SKIPPED)
declare -A TREE_NOTE=([host]="--only $ONLY" [device]="--only $ONLY")
declare -A TREE_TOTAL=([host]=0 [device]=0)
declare -A TREE_PASSED=([host]=0 [device]=0)
declare -A TREE_FAILED=([host]=0 [device]=0)

tree_dir() { echo "$BUILD_DIR/$1"; }

# --- configure + build one tree -------------------------------------------
build_tree() {
  local tree=$1
  local dir; dir=$(tree_dir "$tree")
  local cxx cc contract prefix host_targets device_targets kokkos

  if [ "$tree" = host ]; then
    cxx=$HOST_CXX; cc=$HOST_CC; contract=$HOST_CONTRACT
    prefix=""; host_targets=ON; device_targets=OFF; kokkos=OFF
  else
    cxx=$DEV_CXX; cc=$DEV_CC; contract=$DEV_CONTRACT
    prefix=$KOKKOS_PREFIX; host_targets=OFF; device_targets=ON; kokkos=$WITH_KOKKOS
  fi

  local args=(
    -S "$REPO_ROOT"
    -B "$dir"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_C_COMPILER="$cc"
    -DCMAKE_CXX_COMPILER="$cxx"
    -DCMAKE_CXX_FLAGS="$contract"
    -DCMAKE_CXX_FLAGS_RELEASE="-$OPT -DNDEBUG"
    -DXPMATH_WITH_KOKKOS="$kokkos"
    -DXPMATH_BUILD_HOST_TARGETS="$host_targets"
    -DXPMATH_BUILD_DEVICE_TARGETS="$device_targets"
    -DXPMATH_ARCH="$ARCH"
    -DXPMATH_OPT="$OPT"
  )
  [ -n "$prefix" ] && args+=(-DCMAKE_PREFIX_PATH="$prefix")

  echo
  echo "==========================================================="
  echo " xpm_build : ${tree^^} TREE"
  echo "   arch        : $ARCH  (${ARCH_DESC[$ARCH]})"
  echo "   build dir   : $dir"
  echo "   compiler    : $cxx"
  echo "   optimization: -$OPT"
  echo "   contraction : $contract"
  echo "   kokkos      : $kokkos  ${prefix:-(none)}  [$KOKKOS_MODE]"
  echo "   targets     : HOST=$host_targets DEVICE=$device_targets"
  echo "==========================================================="

  if ! cmake "${args[@]}"; then
    TREE_STATE[$tree]=CONFIGURE-FAILED
    TREE_NOTE[$tree]="cmake configure returned nonzero"
    return 1
  fi

  if [ ! -f "$dir/build-info.txt" ]; then
    TREE_STATE[$tree]=CONFIGURE-FAILED
    TREE_NOTE[$tree]="configure wrote no build-info.txt (stamp block in CMakeLists.txt did not run)"
    return 1
  fi

  cmake --build "$dir" -j"$(nproc)"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    TREE_STATE[$tree]=BUILD-FAILED
    TREE_NOTE[$tree]="cmake --build exit $rc"
    if [ "$tree" = device ] && [ "$ARCH" != host ]; then
      echo "xpm_build: device tree failed to build on arch $ARCH." >&2
      echo "xpm_build: this tree is the one that gets nvcc/hipcc. A host-only TU" >&2
      echo "xpm_build: reaching it is a tagging bug in tests/CMakeLists.txt, not a" >&2
      echo "xpm_build: toolchain bug -- check the C5 side tag on the failing target." >&2
    fi
    return 1
  fi

  TREE_STATE[$tree]=BUILT
  TREE_NOTE[$tree]="built"
  return 0
}

# --- run ctest in one tree -------------------------------------------------
# Sets TREE_STATE/TOTAL/PASSED/FAILED. Returns 0 only for a clean tree.
test_tree() {
  local tree=$1
  local dir; dir=$(tree_dir "$tree")
  local log="$BUILD_DIR/ctest-$tree.log"

  # `ctest --test-dir` on a MISSING directory exits 0. Assert the subject
  # exists and is a configured build tree before believing any exit code.
  if [ ! -d "$dir" ]; then
    TREE_STATE[$tree]=FAIL
    TREE_NOTE[$tree]="no such directory: $dir"
    return 1
  fi
  if [ ! -f "$dir/CTestTestfile.cmake" ]; then
    TREE_STATE[$tree]=FAIL
    TREE_NOTE[$tree]="$dir is not a configured build tree (no CTestTestfile.cmake)"
    return 1
  fi

  echo
  echo "--- ctest: ${tree} tree ($dir) ---"
  ctest --test-dir "$dir" -j"$CTEST_JOBS" --timeout "$CTEST_TIMEOUT" 2>&1 | tee "$log"

  # ctest's own verdict line. Absent means ctest did not get far enough to have
  # an opinion -- do not infer one from the exit code.
  local line
  line=$(grep -E '^[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+$' "$log" | tail -1)
  if [ -z "$line" ]; then
    TREE_STATE[$tree]=FAIL
    TREE_NOTE[$tree]="ctest printed no verdict line; see $log"
    return 1
  fi

  local failed total
  read -r failed total < <(sed -E \
    's/^[0-9]+% tests passed, ([0-9]+) tests failed out of ([0-9]+)$/\1 \2/' <<<"$line")
  TREE_TOTAL[$tree]=$total
  TREE_FAILED[$tree]=$failed
  TREE_PASSED[$tree]=$(( total - failed ))

  if [ "$total" -eq 0 ]; then
    TREE_STATE[$tree]=FAIL
    TREE_NOTE[$tree]="ran ZERO tests -- a green run over nothing is not a green run"
    return 1
  fi
  if [ "$failed" -ne 0 ]; then
    TREE_STATE[$tree]=FAIL
    TREE_NOTE[$tree]="$failed failing; see $log"
    return 1
  fi

  TREE_STATE[$tree]=PASS
  TREE_NOTE[$tree]="clean"
  return 0
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
[ "$BUILD_HOST_TREE"   = 1 ] && build_tree host
[ "$BUILD_DEVICE_TREE" = 1 ] && build_tree device

for t in "${TREES[@]}"; do
  d=$(tree_dir "$t")
  if [ -f "$d/build-info.txt" ] && [ "${TREE_STATE[$t]}" != SKIPPED ]; then
    echo
    echo "--- build-info.txt (${t} tree) ---"
    cat "$d/build-info.txt"
  fi
done

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if [ "$RUN_TESTS" = 1 ]; then
  for t in "${TREES[@]}"; do
    # Only a tree that BUILT gets tested. Everything else already carries a row
    # saying why, and running ctest over it would replace that reason with a
    # less specific one.
    [ "${TREE_STATE[$t]}" = BUILT ] && test_tree "$t"
  done
fi

# ---------------------------------------------------------------------------
# THE MERGED VERDICT
# ---------------------------------------------------------------------------
echo
echo "==========================================================="
echo " xpm_build MERGED VERDICT -- arch $ARCH, --only $ONLY"
echo "   build dir : $BUILD_DIR"
echo "-----------------------------------------------------------"
printf "   %-7s %6s %7s %7s  %-16s %s\n" tree tests passed failed result note
bad=0
attempted=0
tot_t=0; tot_p=0; tot_f=0
for t in "${TREES[@]}"; do
  state=${TREE_STATE[$t]}
  [ "$state" = SKIPPED ] || attempted=$(( attempted + 1 ))
  case "$state" in
    PASS)    ;;
    SKIPPED) ;;
    # BUILT is clean only under --no-test. With testing on it means the tree
    # built and then was never tested, which is a bug in this script and must
    # not read as a pass.
    BUILT)   [ "$RUN_TESTS" = 1 ] && bad=$(( bad + 1 )) ;;
    *)       bad=$(( bad + 1 )) ;;
  esac
  if [ "$state" = PASS ] || [ "$state" = FAIL ]; then
    tot_t=$(( tot_t + TREE_TOTAL[$t]  ))
    tot_p=$(( tot_p + TREE_PASSED[$t] ))
    tot_f=$(( tot_f + TREE_FAILED[$t] ))
  fi
  printf "   %-7s %6s %7s %7s  %-16s %s\n" \
         "$t" "${TREE_TOTAL[$t]}" "${TREE_PASSED[$t]}" "${TREE_FAILED[$t]}" \
         "$state" "${TREE_NOTE[$t]}"
done
echo "-----------------------------------------------------------"
printf "   %-7s %6s %7s %7s\n" TOTAL "$tot_t" "$tot_p" "$tot_f"
echo "-----------------------------------------------------------"

if [ "$RUN_TESTS" != 1 ]; then
  echo " --no-test: NO VERDICT WAS TAKEN. The table above reports the build only."
fi

if [ "$bad" -ne 0 ]; then
  echo " RESULT: FAIL -- $bad of $attempted requested tree(s) did not come out clean."
  echo "==========================================================="
  exit 1
fi

if [ "$RUN_TESTS" = 1 ]; then
  echo " RESULT: PASS -- every tree built and every registered test passed."
else
  echo " RESULT: BUILT -- every requested tree configured and built."
  for t in "${TREES[@]}"; do
    [ "${TREE_STATE[$t]}" = SKIPPED ] && continue
    echo "         ctest --test-dir $(tree_dir "$t")"
  done
fi
echo "==========================================================="
exit 0
