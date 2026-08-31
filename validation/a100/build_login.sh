#!/usr/bin/env bash
# =============================================================================
# validation/a100/build_login.sh — S1 step 1: build everything on the LOGIN node
# =============================================================================
#
# WHY LOGIN NODE: nvcc cross-compiles for sm_80 with no A100 present. Compiling
# Kokkos + this repo takes far longer than running the suite, so the GPU
# reservation must be spent RUNNING, not COMPILING. This script does all the
# compiling; validation/a100/run_a100.sh does all the running.
#
# WHAT IT DOES
#   1. loads the JLSE modules (same set as scripts/prepare.sh)
#   2. clones Kokkos 5.1.0 into $KOKKOS_SRC
#   3. applies patches/kokkos_complex_quad_math.hpp to the Kokkos SOURCE tree
#      (the __complex128 oracle the three *_complex demos need; see
#      patches/README.md — it is NOT upstream)
#   4. configures/builds/installs Kokkos with CUDA + AMPERE80 + LIBQUADMATH
#      into $KOKKOS_PREFIX
#   5. configures/builds this repo against that install into $REPO_BUILD
#   6. records the evidence the analysis half needs (KokkosCore_config.h
#      defines, compiler id, compile_commands.json fmad/fp-contract grep)
#
# NOTE ON scripts/build_with_kokkos.sh: that script hardcodes
# Kokkos_ARCH_BLACKWELL100 / CMAKE_CUDA_ARCHITECTURES=100 and builds the repo
# with plain g++ (which cannot compile a CUDA-backed Kokkos). This script is the
# explicit sm_80 counterpart, per UPSTREAM_PLAN S1 deliverable 1 — one arch,
# no multi-arch abstraction.
#
# USAGE (from the repo root, on a JLSE login node):
#     bash validation/a100/build_login.sh
#
# Knobs (environment variables, all optional):
#     JOBS           parallel compile jobs            (default 16)
#     KOKKOS_SRC     Kokkos checkout                  (default ~/kokkos-src-5.1.0-cuda-sm80)
#     KOKKOS_PREFIX  Kokkos install prefix            (default ~/kokkos-install-cuda-sm80-quadmath)
#     REPO_BUILD     repo build dir                   (default ~/kokkos-ep-build-a100)
#
# REPO_BUILD deliberately lives OUTSIDE the repo: the repo's .gitignore covers
# /build/ only, and S1 must not modify anything outside validation/a100/.
# =============================================================================

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VAL_DIR="$REPO_ROOT/validation/a100"
LOGDIR="$VAL_DIR/logs/build"
mkdir -p "$LOGDIR"

JOBS=${JOBS:-16}
# Same tag string scripts/build_with_kokkos.sh uses (that build produced
# KOKKOS_VERSION 50100 per patches/README.md). A "5.1.00"-style tag is tried as
# a fallback in case the remote spelling differs.
KOKKOS_TAG=${KOKKOS_TAG:-5.1.0}
KOKKOS_TAG_ALT=${KOKKOS_TAG_ALT:-5.1.00}
KOKKOS_URL=https://github.com/kokkos/kokkos.git
KOKKOS_SRC=${KOKKOS_SRC:-$HOME/kokkos-src-5.1.0-cuda-sm80}
KOKKOS_BUILD=${KOKKOS_BUILD:-$HOME/kokkos-build-cuda-sm80-quadmath}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-$HOME/kokkos-install-cuda-sm80-quadmath}
REPO_BUILD=${REPO_BUILD:-$HOME/kokkos-ep-build-a100}

say() { printf '\n=== %s\n' "$*"; }

# --- 1. modules -------------------------------------------------------------
say "modules"
module use /soft/modulefiles
module load gcc/13.3.0
module load cmake/3.28.3
module load cuda/12.9.1
module list 2>&1 | tee "$LOGDIR/modules.log"

{
  echo "date        : $(date -Is)"
  echo "host        : $(hostname)"
  echo "gcc         : $(gcc --version | head -1)"
  echo "g++         : $(g++ --version | head -1)"
  echo "nvcc        : $(nvcc --version | tail -2 | head -1)"
  echo "cmake       : $(cmake --version | head -1)"
  echo "repo HEAD   : $(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "repo branch : $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  echo "repo dirty  : $(git -C "$REPO_ROOT" status --porcelain | wc -l) modified/untracked entries"
  echo "KOKKOS_SRC  : $KOKKOS_SRC"
  echo "KOKKOS_PREFIX: $KOKKOS_PREFIX"
  echo "REPO_BUILD  : $REPO_BUILD"
  echo "JOBS        : $JOBS"
} 2>&1 | tee "$LOGDIR/env.log"

ulimit -s 131072 || true

# --- 2. Kokkos source -------------------------------------------------------
say "Kokkos source ($KOKKOS_TAG)"
if [ ! -d "$KOKKOS_SRC/.git" ]; then
  {
    git clone --depth 1 --branch "$KOKKOS_TAG" "$KOKKOS_URL" "$KOKKOS_SRC" ||
    git clone --depth 1 --branch "$KOKKOS_TAG_ALT" "$KOKKOS_URL" "$KOKKOS_SRC"
  } 2>&1 | tee "$LOGDIR/kokkos_clone.log"
else
  echo "reusing existing checkout at $KOKKOS_SRC" | tee "$LOGDIR/kokkos_clone.log"
fi
git -C "$KOKKOS_SRC" log -1 --oneline 2>&1 | tee -a "$LOGDIR/kokkos_clone.log"

# --- 3. complex-quadmath oracle patch (patches/README.md) -------------------
say "apply local __complex128 oracle header to the Kokkos source tree"
cp -v "$REPO_ROOT/patches/kokkos_complex_quad_math.hpp" \
      "$KOKKOS_SRC/core/src/impl/Kokkos_ComplexQuadPrecisionMath.hpp" \
      2>&1 | tee "$LOGDIR/patch.log"

# --- 4. build + install Kokkos ---------------------------------------------
# CUDA-backed Kokkos must be compiled with nvcc_wrapper as the C++ compiler —
# this applies to Kokkos itself AND to every consumer (step 5).
export NVCC_WRAPPER_DEFAULT_COMPILER=$(command -v g++)
NVCC_WRAPPER="$KOKKOS_SRC/bin/nvcc_wrapper"

say "configure Kokkos (CUDA / AMPERE80 / sm_80 / LIBQUADMATH)"
# Kokkos 5.1.0 REQUIRES C++20 (cmake/kokkos_test_cxx_std.cmake rejects 17).
# The existing ~/kokkos-install* trees confirm it: Kokkos_CXX_STANDARD 20.
# The consuming repo stays at C++17 -- that direction is fine, a C++17 TU can
# link a C++20-built Kokkos, and this mirrors the working Serial installs.
cmake -S "$KOKKOS_SRC" -B "$KOKKOS_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
  -DKokkos_ENABLE_CUDA_FASTMATH=OFF \
  -DKokkos_ENABLE_LIBQUADMATH=ON \
  2>&1 | tee "$LOGDIR/kokkos_configure.log"

say "build + install Kokkos (-j$JOBS)"
cmake --build "$KOKKOS_BUILD" -j"$JOBS" --target install \
  2>&1 | tee "$LOGDIR/kokkos_build.log"

# The complex oracle header is a local addition and may not be picked up by
# Kokkos's install rules; copy it into the install tree if it is missing
# (patches/README.md documents this exact fallback).
if [ ! -f "$KOKKOS_PREFIX/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp" ]; then
  echo "install tree missing the complex oracle header; copying directly"
  cp -v "$REPO_ROOT/patches/kokkos_complex_quad_math.hpp" \
        "$KOKKOS_PREFIX/include/impl/Kokkos_ComplexQuadPrecisionMath.hpp"
fi | tee -a "$LOGDIR/patch.log"

say "record the installed Kokkos configuration (EVIDENCE)"
grep -E '^#define KOKKOS_(ENABLE|ARCH|VERSION)' \
  "$KOKKOS_PREFIX/include/KokkosCore_config.h" \
  2>&1 | tee "$LOGDIR/kokkos_config_defines.log"

# --- 5. build this repo against the CUDA Kokkos ----------------------------
# -fext-numeric-literals is passed explicitly rather than relying on the
# top-level CMakeLists.txt's `if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")` branch:
# under nvcc_wrapper the detected compiler id is a checkpoint, not a given, and
# the __float128 `Q` literals in the oracle will not parse without the flag.
# The top-level file string(FIND)s the flag before appending, so this cannot
# duplicate it. This is a build-line choice, not a repo change.
say "configure this repo against $KOKKOS_PREFIX"
cmake -S "$REPO_ROOT" -B "$REPO_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
  -DCMAKE_PREFIX_PATH="$KOKKOS_PREFIX" \
  -DCMAKE_CXX_FLAGS="-g -fext-numeric-literals" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKOKKOS_EP_BUILD_TESTS=ON \
  2>&1 | tee "$LOGDIR/repo_configure.log"

say "build this repo (-j$JOBS)"
cmake --build "$REPO_BUILD" -j"$JOBS" \
  2>&1 | tee "$LOGDIR/repo_build.log"

# --- 6. contraction-flag evidence (EVIDENCE for S1 deliverable 3) ----------
# tests/CMakeLists.txt attaches --fmad=false / --fmad=true behind
# $<COMPILE_LANGUAGE:CUDA>, and -ffp-contract=off/fast behind
# $<COMPILE_LANGUAGE:CXX>. Which of those actually reach the compile line under
# a CUDA build is exactly what the analysis half must determine — so dump the
# real command lines rather than assuming.
say "contraction-flag evidence"
{
  echo "--- detected C++ compiler id / path (from CMakeCache) ---"
  grep -E '^CMAKE_CXX_COMPILER(_ID)?(:|=)' "$REPO_BUILD/CMakeCache.txt" || true
  grep -E '^CMAKE_CXX_FLAGS:' "$REPO_BUILD/CMakeCache.txt" || true
  echo
  echo "--- every compile line for the EFT / FMA-guard targets ---"
  for t in dd_eft_test ff_eft_test qf_eft_test \
           dd_fma_guard_test ff_fma_guard_test qf_fma_guard_test; do
    echo "### $t"
    python3 - "$REPO_BUILD/compile_commands.json" "$t" <<'PY' || true
import json,sys
db,tgt=json.load(open(sys.argv[1])),sys.argv[2]
for e in db:
    if tgt in e.get("output","") or tgt in e.get("command",""):
        print(e.get("command") or " ".join(e.get("arguments",[])))
PY
    echo
  done
  echo "--- grep summary: does --fmad appear anywhere? ---"
  grep -c -- '--fmad' "$REPO_BUILD/compile_commands.json" || echo "0 occurrences of --fmad"
  echo "--- grep summary: does -ffp-contract appear anywhere? ---"
  grep -c -- '-ffp-contract' "$REPO_BUILD/compile_commands.json" || echo "0 occurrences of -ffp-contract"
} 2>&1 | tee "$LOGDIR/contraction_flags.log"

say "built targets"
ls -1 "$REPO_BUILD"/kokkos_ep_* "$REPO_BUILD"/tests/*_test* 2>/dev/null \
  | grep -v '\.' | tee "$LOGDIR/targets.log" || true

cat <<EOF

=============================================================================
BUILD COMPLETE (login node).
  Kokkos install : $KOKKOS_PREFIX
  Repo build     : $REPO_BUILD
  Build logs     : $LOGDIR

Next: submit the run to the A100 queue. From the repo root:

  # script mode (automatable, preferred)
  qsub -A pepper_hep -t 120 -n 1 -q gpu_a100 --mode script \\
       validation/a100/run_a100.sh

  # or interactive, then inside the shell:
  qsub -A pepper_hep -I -t 120 -n 1 -q gpu_a100
  bash validation/a100/run_a100.sh

If you override REPO_BUILD/KOKKOS_PREFIX here, export the same values before
run_a100.sh (script mode does not inherit this shell).
=============================================================================
EOF
