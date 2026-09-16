#!/usr/bin/env bash
# ===========================================================================
# scripts/xpm_build.sh — configure and build xpmath for a NAMED ARCHITECTURE
# ===========================================================================
#
#   scripts/xpm_build.sh --arch {host|a100|mi250} [--build-dir DIR]
#                        [--kokkos {use-existing|build}] [--opt LEVEL]
#                        [--no-kokkos]
#
# Selecting a target is an ARGUMENT here, not an edit. Before this script the
# only recipe was scripts/build_with_kokkos.sh, whose GPU was a literal on line
# 45 and whose HIP branch was unreachable; retargeting meant editing two lines
# and remembering which two.
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
#                   can read it back.
#
#   C++ standard    C++17 for THIS project, on every arch. The three Kokkos
#                   installs are all C++20 because Kokkos 5.1.0 hard-errors on
#                   17 (KI-3). The two numbers are deliberately different and
#                   must stay different; see .github/workflows/ci.yml.
#
#   FP contraction  OFF, on every arch -- but spelled differently, and with a
#                   different reach, per toolchain. See the next block.
#
# NOT held identical, by construction:
#
#   compiler        g++ (host), nvcc_wrapper wrapping nvcc+g++ (a100), hipcc
#                   (mi250). There is no common compiler for three vendors.
#   execution space Serial, Cuda, HIP.
#   host oracle     NO LONGER A DIFFERENCE, and the row is kept to say so. It
#                   used to read: libquadmath is present in the host and a100
#                   Kokkos installs and ABSENT from the mi250 one, so the
#                   oracle-scored tests runtime-SKIP on mi250. The B-arc removed
#                   the last libquadmath call site from tests/ and the skip that
#                   hid it; sweep_accuracy's oracle is MPFR/MPC and links no
#                   libquadmath on any arch. Whether the Kokkos install has
#                   libquadmath no longer changes which tests score.
#   device build    a100 and mi250 additionally compile every Kokkos-linked TU
#                   through a device pass; host does not.
#
# ---------------------------------------------------------------------------
# FP CONTRACTION, SPELLED OUT
# ---------------------------------------------------------------------------
# The Dekker twoProduct needs `a1*b1 - c11` to be two rounded operations. Fuse
# them and the error term collapses to zero and the error-free transform stops
# being error-free -- silently, because nothing fails.
#
#   host   -ffp-contract=off   given to g++   as a CMAKE_CXX_FLAGS entry.
#   mi250  -ffp-contract=off   given to hipcc as a CMAKE_CXX_FLAGS entry.
#          hipcc is clang, so the one spelling covers its host AND device pass.
#   a100   --fmad=false        given to nvcc_wrapper as a CMAKE_CXX_FLAGS entry,
#          which forwards it to nvcc and so reaches the DEVICE pass. The host
#          pass of the same TU is g++ and is covered by tests/CMakeLists.txt's
#          per-target -ffp-contract=off on the EFT targets.
#
# WHICH TARGETS GET IT: all of them. This is a whole-build CMAKE_CXX_FLAGS
# entry, not a per-target option, because the apples-to-apples claim is about
# the build, not about eight test targets. tests/CMakeLists.txt keeps its
# per-target flags unchanged; per-target options are appended AFTER
# CMAKE_CXX_FLAGS, so the five `*_contract_on` reporters still override the
# host-side flag and still report what contraction collapses.
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
# PROVENANCE
# ---------------------------------------------------------------------------
# build-info.txt is written into the build directory by the CONFIGURE, from
# the top-level CMakeLists.txt -- not by this script. That is on purpose: a
# stamp only this wrapper wrote would be absent from exactly the builds nobody
# can trace, which are the ones made by calling cmake directly. The ctest
# target `build_provenance` fails when it is missing or short a field.
# ===========================================================================

set -uo pipefail

_self=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "$_self/.." && pwd)

die() { echo "xpm_build: $*" >&2; exit 2; }

usage() {
  sed -n '3,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "  --arch        host | a100 | mi250          (required)"
  echo "  --build-dir   directory to configure into  (default: \$REPO_ROOT/build)"
  echo "  --kokkos      use-existing | build         (default: use-existing)"
  echo "  --opt         O0 O1 O2 O3 Os Og Ofast      (default: O3)"
  echo "  --no-kokkos   configure -DXPMATH_WITH_KOKKOS=OFF"
}

# ---------------------------------------------------------------------------
# THE ARCH TABLE. Pure data, one row per key. A row that is not selected is
# never expanded and never executed -- it is a value in an array, not a
# commented-out line somebody has to remember to uncomment.
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
  [mi250]="-ffp-contract=off"
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ARCH=""
BUILD_DIR=""
KOKKOS_MODE="use-existing"
OPT="O3"
WITH_KOKKOS=ON

while [ $# -gt 0 ]; do
  case "$1" in
    --arch)       ARCH=${2:-};        shift 2 || die "--arch needs a value" ;;
    --build-dir)  BUILD_DIR=${2:-};   shift 2 || die "--build-dir needs a value" ;;
    --kokkos)     KOKKOS_MODE=${2:-}; shift 2 || die "--kokkos needs a value" ;;
    --opt)        OPT=${2:-};         shift 2 || die "--opt needs a value" ;;
    --no-kokkos)  WITH_KOKKOS=OFF;    shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -n "$ARCH" ] || { usage >&2; die "--arch is required"; }
[ -n "${ARCH_DESC[$ARCH]+set}" ] || die "unknown --arch '$ARCH' (host|a100|mi250)"
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

CXX_NAME=${ARCH_CXX[$ARCH]}
CC_NAME=${ARCH_CC[$ARCH]}
CONTRACT=${ARCH_CONTRACT[$ARCH]}
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
command -v "$CXX_NAME" >/dev/null 2>&1 || [ -x "$CXX_NAME" ] \
  || die "C++ compiler '$CXX_NAME' not found (arch $ARCH)"

# ---------------------------------------------------------------------------
# Kokkos
# ---------------------------------------------------------------------------
if [ "$WITH_KOKKOS" = OFF ]; then
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
  KOKKOS_CXX=${CXX_NAME} \
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
# Configure
# ---------------------------------------------------------------------------
CMAKE_ARGS=(
  -S "$REPO_ROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=17
  -DCMAKE_C_COMPILER="$CC_NAME"
  -DCMAKE_CXX_COMPILER="$CXX_NAME"
  -DCMAKE_CXX_FLAGS="$CONTRACT"
  -DCMAKE_CXX_FLAGS_RELEASE="-$OPT -DNDEBUG"
  -DXPMATH_WITH_KOKKOS="$WITH_KOKKOS"
  -DXPMATH_ARCH="$ARCH"
  -DXPMATH_OPT="$OPT"
)
if [ -n "$KOKKOS_PREFIX" ]; then
  CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$KOKKOS_PREFIX")
fi

echo "==========================================================="
echo " xpm_build"
echo "   arch        : $ARCH  (${ARCH_DESC[$ARCH]})"
echo "   build dir   : $BUILD_DIR"
echo "   compiler    : $CXX_NAME"
echo "   optimization: -$OPT"
echo "   contraction : $CONTRACT"
echo "   kokkos      : $WITH_KOKKOS  ${KOKKOS_PREFIX:-(none)}  [$KOKKOS_MODE]"
echo "==========================================================="

cmake "${CMAKE_ARGS[@]}" || die "configure failed (arch $ARCH)"

[ -f "$BUILD_DIR/build-info.txt" ] \
  || die "configure wrote no $BUILD_DIR/build-info.txt -- the stamp block in CMakeLists.txt did not run"

cmake --build "$BUILD_DIR" -j"$(nproc)"
rc=$?

echo
echo "--- build-info.txt ---"
cat "$BUILD_DIR/build-info.txt"

if [ "$rc" -ne 0 ]; then
  echo
  echo "xpm_build: CONFIGURE SUCCEEDED, BUILD FAILED (exit $rc) for arch $ARCH." >&2
  if [ "$ARCH" != host ] && [ "$WITH_KOKKOS" = ON ]; then
    echo "xpm_build: on a device arch the known cause is the host __float128 oracle" >&2
    echo "xpm_build: sharing a translation unit with device code -- see the S1 STATUS" >&2
    echo "xpm_build: block; the TU split is S6's work, not this script's." >&2
  fi
  exit "$rc"
fi
echo "xpm_build: OK -- ctest --test-dir $BUILD_DIR"
