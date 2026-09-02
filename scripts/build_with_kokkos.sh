#!/usr/bin/env bash
################################################################
# Usage: source scripts/build_with_kokkos.sh <install-dir>     #
#      Ensure your environment has compilers and CUDA drivers. #
################################################################
# Builds Kokkos into <install-dir>, then configures and builds #
# this repository under <repo>/build/.                         #
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

KOKKOS_ENABLED=Kokkos_ENABLE_CUDA

# Match reference script (edit here for your GPU arch)
KOKKOS_ARCH_FLAG=Kokkos_ARCH_BLACKWELL100

NO_EXTRA_FLAGS=
CUDA_EXTRA_FLAGS="-DKokkos_ENABLE_CUDA_LAMBDA=On \
                  -DKokkos_ENABLE_CUDA_CONSTEXPR=On \
                  -DKokkos_ENABLE_CUDA_FASTMATH=Off"
HIP_EXTRA_FLAGS="-DCMAKE_CXX_COMPILER=$CXX \
                 -DCMAKE_CXX_FLAGS=\"--gcc-toolchain=/soft/compilers/gcc/13.3.0/x86_64-suse-linux\""
EXTRA_FLAGS=$CUDA_EXTRA_FLAGS

####################
## install Kokkos ##
####################
echo "Installing Kokkos ARCH=$KOKKOS_ARCH_FLAG"
{
  git clone "$KOKKOS_URL" -b "$KOKKOS_TAG"
  cd kokkos

  if [ "$KOKKOS_ARCH_FLAG" = "NONE" ]; then
    cmake -S . -B "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
      -DCMAKE_CXX_STANDARD=20 \
      -D$KOKKOS_ENABLED=ON \
      $EXTRA_FLAGS
  else
    cmake -S . -B "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
      -DCMAKE_CXX_STANDARD=20 \
      -D$KOKKOS_ARCH_FLAG=ON \
      -D$KOKKOS_ENABLED=ON \
      -DCMAKE_CUDA_ARCHITECTURES=100 \
      $EXTRA_FLAGS
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

mkdir -p "$REPO_ROOT/build"
cd "$REPO_ROOT/build" || exit 1

cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="-g" \
  "$REPO_ROOT"

make -j"$(nproc)"

echo "Executables:"
echo "  $REPO_ROOT/build/kokkos_ep_demo"
echo "  $REPO_ROOT/build/kokkos_ep_demo_complex"
cd ..
