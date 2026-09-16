#!/usr/bin/env bash
################################################################
# Usage: source scripts/build_with_kokkos.sh <install-dir>     #
#      Ensure your environment has compilers and CUDA drivers. #
################################################################
# Builds Kokkos into <install-dir>, then configures and builds #
# this repository under <repo>/build/.                         #
################################################################
# EVERY TARGET-SPECIFIC VALUE BELOW COMES FROM THE ENVIRONMENT. #
# The Blackwell/sm_100 values are the defaults and nothing more #
# -- they used to be literals on two lines, which made          #
# retargeting an edit.  scripts/xpm_build.sh is the wrapper that#
# supplies them per arch; this script is still usable on its own#
# and, called with no environment, does exactly what it did.    #
#                                                               #
#   KOKKOS_BACKEND              cuda | hip | serial   (cuda)    #
#   KOKKOS_ARCH_FLAG            Kokkos_ARCH_* or NONE           #
#                               (Kokkos_ARCH_BLACKWELL100)      #
#   KOKKOS_CUDA_ARCHITECTURES   CMAKE_CUDA_ARCHITECTURES (100)  #
#                               ignored unless BACKEND=cuda     #
#   KOKKOS_CXX                  C++ compiler for the Kokkos     #
#                               build (hipcc when BACKEND=hip)  #
#   KOKKOS_LIBQUADMATH          ON | OFF            (ON)        #
#   KOKKOS_GCC_TOOLCHAIN        --gcc-toolchain= for hipcc      #
#   KOKKOS_ONLY                 1 = stop after installing       #
#                               Kokkos, do not configure the    #
#                               repo (xpm_build.sh owns that)   #
#   REPO_BUILD_DIR              where to configure this repo    #
#                               (<repo>/build)                  #
################################################################

export TARGET_DIR=$1
if [ "$#" -ne 1 ]; then
  export TARGET_DIR=$(pwd -LP)
fi
START_DIR=$(pwd -LP)

# Resolve repo root while cwd is still the caller's directory. After `cd "$TARGET_DIR"`
# below, a relative ${BASH_SOURCE[0]} would break (e.g. dirname becomes build/scripts).
_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "${_SCRIPT_DIR}/.." && pwd)

echo "Installing Kokkos into: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR" || exit 1

export LOGDIR=$TARGET_DIR/build_logs
mkdir -p "$LOGDIR"

##############
## SETTINGS ##
##############

CC=$(which gcc)
CXX=$(which g++)

# KOKKOS Related settings
KOKKOS_TAG=5.1.0
KOKKOS_BUILD=Release
KOKKOS_URL=https://github.com/kokkos/kokkos.git

# Target selection. Defaults reproduce the pre-parameterization script exactly:
# CUDA, Blackwell100, CMAKE_CUDA_ARCHITECTURES=100.
KOKKOS_BACKEND=${KOKKOS_BACKEND:-cuda}
KOKKOS_ARCH_FLAG=${KOKKOS_ARCH_FLAG:-Kokkos_ARCH_BLACKWELL100}
KOKKOS_CUDA_ARCHITECTURES=${KOKKOS_CUDA_ARCHITECTURES:-100}
KOKKOS_LIBQUADMATH=${KOKKOS_LIBQUADMATH:-ON}
KOKKOS_GCC_TOOLCHAIN=${KOKKOS_GCC_TOOLCHAIN:-/soft/compilers/gcc/13.3.0/x86_64-suse-linux}
KOKKOS_ONLY=${KOKKOS_ONLY:-0}

case "$KOKKOS_BACKEND" in
  cuda)   KOKKOS_ENABLED=Kokkos_ENABLE_CUDA ;;
  hip)    KOKKOS_ENABLED=Kokkos_ENABLE_HIP ;;
  serial) KOKKOS_ENABLED=Kokkos_ENABLE_SERIAL ;;
  *) echo "build_with_kokkos: unknown KOKKOS_BACKEND=$KOKKOS_BACKEND" >&2; return 2 2>/dev/null || exit 2 ;;
esac

# The Kokkos build's own C++ compiler. CUDA goes through nvcc_wrapper and HIP
# through hipcc; both are supplied by the caller, because neither is $(which g++).
KOKKOS_CXX=${KOKKOS_CXX:-$CXX}

# Per-backend cmake arguments. ARRAYS, not strings: the HIP row carries a flag
# with an embedded space, and the old string form quoted it in a way that only
# survived because the row was never expanded -- line 53 assigned the CUDA row
# unconditionally, so the HIP path had never once executed.
NO_EXTRA_FLAGS=()
CUDA_EXTRA_FLAGS=(-DKokkos_ENABLE_CUDA_LAMBDA=On
                  -DKokkos_ENABLE_CUDA_CONSTEXPR=On
                  -DKokkos_ENABLE_CUDA_FASTMATH=Off)
HIP_EXTRA_FLAGS=(-DCMAKE_CXX_COMPILER="$KOKKOS_CXX"
                 -DCMAKE_CXX_FLAGS="--gcc-toolchain=$KOKKOS_GCC_TOOLCHAIN")

case "$KOKKOS_BACKEND" in
  cuda)   EXTRA_FLAGS=("${CUDA_EXTRA_FLAGS[@]}" -DCMAKE_CXX_COMPILER="$KOKKOS_CXX") ;;
  hip)    EXTRA_FLAGS=("${HIP_EXTRA_FLAGS[@]}") ;;
  serial) EXTRA_FLAGS=("${NO_EXTRA_FLAGS[@]}" -DCMAKE_CXX_COMPILER="$KOKKOS_CXX") ;;
esac

# The oracle. The three installs of record are libquadmath ON for host and
# CUDA and OFF for HIP; the old script passed nothing at all, which left it
# OFF and produced a Kokkos this repository's demos cannot compile against.
EXTRA_FLAGS+=(-DKokkos_ENABLE_LIBQUADMATH="$KOKKOS_LIBQUADMATH")

# CMAKE_CUDA_ARCHITECTURES means nothing to a HIP or Serial build.
CUDA_ARCH_FLAGS=()
if [ "$KOKKOS_BACKEND" = cuda ] && [ -n "$KOKKOS_CUDA_ARCHITECTURES" ]; then
  CUDA_ARCH_FLAGS=(-DCMAKE_CUDA_ARCHITECTURES="$KOKKOS_CUDA_ARCHITECTURES")
fi

####################
## install Kokkos ##
####################
echo "Installing Kokkos BACKEND=$KOKKOS_BACKEND ARCH=$KOKKOS_ARCH_FLAG CXX=$KOKKOS_CXX"
{
  git clone "$KOKKOS_URL" -b "$KOKKOS_TAG"
  cd kokkos

  if [ "$KOKKOS_ARCH_FLAG" = "NONE" ]; then
    cmake -S . -B "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
      -DCMAKE_CXX_STANDARD=20 \
      -D$KOKKOS_ENABLED=ON \
      "${EXTRA_FLAGS[@]}"
  else
    cmake -S . -B "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
      -DCMAKE_CXX_STANDARD=20 \
      -D$KOKKOS_ARCH_FLAG=ON \
      -D$KOKKOS_ENABLED=ON \
      "${CUDA_ARCH_FLAGS[@]}" \
      "${EXTRA_FLAGS[@]}"
  fi

  make -C "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" -j"$(nproc)" install

  echo "export KOKKOS_HOME=$PWD/install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" >setup.sh
  echo "export CMAKE_PREFIX_PATH=\$KOKKOS_HOME/lib64/cmake/Kokkos:\$CMAKE_PREFIX_PATH" >>setup.sh
  echo "export CPATH=\$KOKKOS_HOME/include:\$CPATH" >>setup.sh
  echo "export PATH=\$KOKKOS_HOME/bin:\$PATH" >>setup.sh
  echo "export LD_LIBRARY_PATH=\$KOKKOS_HOME/lib64:\$LD_LIBRARY_PATH" >>setup.sh
  # shellcheck disable=SC1091
  source setup.sh
} 1>"$LOGDIR/kokkos.stdout.txt" 2>"$LOGDIR/kokkos.stderr.txt"

cd "$TARGET_DIR" || exit 1

ulimit -s 131072

# shellcheck disable=SC1091
source "$TARGET_DIR/kokkos/setup.sh"

# KOKKOS_ONLY stops here, with the install tree and setup.sh in place. It is
# what scripts/xpm_build.sh --kokkos build uses: that wrapper owns the repo
# configure, so that --opt, --build-dir and build-info.txt behave identically
# whether Kokkos was just built or already existed.
if [ "$KOKKOS_ONLY" = "1" ]; then
  echo "KOKKOS_ONLY=1: Kokkos installed under $TARGET_DIR/kokkos/install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD"
  cd "$START_DIR" || exit 1
  return 0 2>/dev/null || exit 0
fi

REPO_BUILD_DIR=${REPO_BUILD_DIR:-$REPO_ROOT/build}
mkdir -p "$REPO_BUILD_DIR"
cd "$REPO_BUILD_DIR" || exit 1

# KOKKOS_CXX, not $CXX: a CUDA or HIP Kokkos install requires its consumers to
# use the same compiler it was built with, and $CXX is plain g++.
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$KOKKOS_CXX" \
  -DCMAKE_CXX_FLAGS="-g" \
  -DXPMATH_ARCH="build_with_kokkos:$KOKKOS_BACKEND/$KOKKOS_ARCH_FLAG" \
  "$REPO_ROOT"

make -j"$(nproc)"

echo "Executables:"
echo "  $REPO_BUILD_DIR/kokkos_ep_demo"
echo "  $REPO_BUILD_DIR/kokkos_ep_demo_complex"
cd ..
