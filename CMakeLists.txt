cmake_minimum_required(VERSION 3.16)

# The project is named xpmath because that is the name a consumer writes in
# find_package(). The repository directory keeps its historical name; nothing
# in the build depended on the old project() name (verified: no reference to
# PROJECT_NAME or CMAKE_PROJECT_NAME anywhere in the tree).
project(xpmath VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# THE C++ STANDARD IS MEASURED, NOT ASSERTED BY THIS LINE -- CORE_PLAN C5 step 6
# ---------------------------------------------------------------------------
# `cxx-standard: 17` has been in build-info.txt since the stamp existed, and it
# was NOT true of every translation unit: Kokkos 5.1.0 carries
# INTERFACE_COMPILE_FEATURES cxx_std_20, which raises every target that links
# it, and the stamp -- which only ever echoed CMAKE_CXX_STANDARD back -- said
# 17 anyway. S8c noticed the discrepancy on gfx90a and nothing in the build
# could confirm or deny it.
#
# compile_commands.json is the one artifact that records the flags each TU
# ACTUALLY got, so this turns it on and `build_provenance` reads it back. Only
# for the generators that can produce it; asking a generator that cannot would
# make the check fail on a missing file that was never going to be written.
# Not FORCEd -- a caller who turns it off gets a loud NOT MEASURED line from
# the check rather than a failure.
#
# SET AS A NORMAL VARIABLE, AND ONLY WHEN THE CACHE ENTRY IS EMPTY. CMake
# pre-declares CMAKE_EXPORT_COMPILE_COMMANDS as an empty cache entry during
# compiler detection, so `set(... CACHE BOOL ...)` without FORCE is a no-op --
# measured: the entry stayed empty, no database was written, and the check
# correctly reported NOT MEASURED. FORCE would instead overrule a caller who
# passed -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF on purpose. The empty-cache test
# distinguishes "nobody asked" from "somebody said no".
if(CMAKE_GENERATOR MATCHES "Makefile|Ninja")
  if("$CACHE{CMAKE_EXPORT_COMPILE_COMMANDS}" STREQUAL "")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  endif()
endif()

# CLOSED SET of targets allowed to compile above CMAKE_CXX_STANDARD. After
# CORE_PLAN C10 this repository links no Kokkos, so the set is permanently
# empty: every TU is at CMAKE_CXX_STANDARD. Kept as a named variable so
# check_build_provenance.cmake still has something to measure against.
set(XPMATH_KOKKOS_LINKED_TARGETS "")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(FIND "${CMAKE_CXX_FLAGS}" "-fext-numeric-literals" _extlit)
  if(_extlit EQUAL -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals" CACHE STRING "" FORCE)
  endif()
endif()

# Suppress -Wattributes in CUDA host-pass compilation.
# XPMATH_NOINLINE_FUNCTION expands to '__host__ __device__ __noinline__ inline'
# under nvcc.  The combination of the C++ 'inline' keyword (needed for
# header-only COMDAT linkage) and the __noinline__ attribute (needed to prevent
# call-site inlining for the CUDA frame-size fix) is self-contradictory to GCC,
# which warns regardless of how the forward declaration is annotated.  There is
# no source-level combination that satisfies both requirements without this
# warning; suppressing it is the standard practice for CUDA header-only
# libraries that need the same pattern.
if(CMAKE_CUDA_COMPILER)
  string(FIND "${CMAKE_CUDA_FLAGS}" "-Xcompiler=-Wno-attributes" _wattr)
  if(_wattr EQUAL -1)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-attributes"
        CACHE STRING "" FORCE)
  endif()
endif()

# ---------------------------------------------------------------------------
# WHICH SIDE OF THE BUILD TO PRODUCE -- CORE_PLAN C5 step 1
# ---------------------------------------------------------------------------
# CMake supports exactly ONE CXX compiler per project and has no per-target
# override, so "host tools with g++, device tests with nvcc/hipcc" can only be
# two build trees. These two options are what lets one source tree produce
# either half. scripts/xpm_build.sh drives both.
#
# BOTH DEFAULT ON, AND THAT IS LOAD-BEARING. With both ON the build is one
# tree, one compiler, the full ctest suite -- the bare `cmake -B build` path
# that CI runs. A default that changed the target set would be a silent CI
# change wearing a refactor's clothes.
#
# THE FAILURE THIS PREVENTS is a host-only tool compiled by a device compiler.
# On A100 (job 1000938) `xpm_build.sh --arch a100` set nvcc_wrapper
# project-wide, so sweep_accuracy -- which links no device code at all -- went
# through nvcc and failed, taking both gates, oracle_conv_test and both gate
# selftests down with it. Five of that job's eleven red results were that one
# mistake.
#
# EVERY registered test is tagged host or device, and the assertion at the
# bottom of tests/CMakeLists.txt fails the CONFIGURE if one is not.
#
# CORE_PLAN C10: this repository no longer finds or links Kokkos. The eight
# demos and third_party/include/ compat wrappers moved to xpmath-kokkos (K6);
# they last existed here at commit 158d618b4d02e9e1ed542438514290e27123079f.
# Device execution in this repo is tests/device_harness.hpp (CUDA/HIP/serial).
option(XPMATH_BUILD_HOST_TARGETS
       "Build the host-side targets: the oracle-scored tools, the gates and the host test halves" ON)
option(XPMATH_BUILD_DEVICE_TARGETS
       "Build the device-side targets: the *_test_device halves, the harness self-test and the standalone smokes" ON)

if(NOT XPMATH_BUILD_HOST_TARGETS AND NOT XPMATH_BUILD_DEVICE_TARGETS)
  # Not a philosophical objection -- this configuration registers zero tests,
  # and `ctest` over zero tests exits 0. That is the same trap as
  # `ctest --test-dir` on a missing directory, and it would report a green run
  # having measured nothing.
  message(FATAL_ERROR
          "XPMATH_BUILD_HOST_TARGETS and XPMATH_BUILD_DEVICE_TARGETS are both "
          "OFF: this build would contain no targets and `ctest` over it would "
          "exit 0 having measured nothing. Turn at least one back ON.")
endif()

# ---- Installable package (header-only core) ---------------------------------
#
# xpmath the LIBRARY is the headers in include/xp/. They are header-only and
# they do not include or require Kokkos -- every mention of Kokkos in them is a
# comment explaining that the Kokkos layer lives in the xpmath-kokkos
# repository. tests/standalone/*_no_kokkos_smoke.cpp compiles them with a bare
# g++ and an include path containing only include/.

add_library(xpmath INTERFACE)
add_library(xpmath::xpmath ALIAS xpmath)

target_include_directories(xpmath INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# C++17 is a real requirement of the headers (if constexpr, inline variables),
# not a project style choice, so it travels with the exported target.
target_compile_features(xpmath INTERFACE cxx_std_17)

install(TARGETS xpmath EXPORT xpmathTargets)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/xp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp")

install(EXPORT xpmathTargets
        FILE xpmathTargets.cmake
        NAMESPACE xpmath::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/xpmath")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xpmathConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/xpmathConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/xpmath")

# SameMajorVersion with a major version of 0: under CMake's rules every 0.x is
# then incompatible with every other 0.x. That is the honest setting for a
# library whose API has never been consumed by anyone -- find_package(xpmath 0.1)
# passes, find_package(xpmath 0.2) fails, and no compatibility is promised that
# this project has not earned.
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/xpmathConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/xpmathConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/xpmathConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/xpmath")

# The licence ships WITH the headers. A consumer who vendors an install prefix
# into their own tree gets the terms alongside the code rather than having to
# go back to a repository they may never have cloned.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/licenses/xpmath")

# ---------------------------------------------------------------------------
# BUILD PROVENANCE -- build-info.txt
# ---------------------------------------------------------------------------
# Three campaign artifacts produced this month could not be traced back to the
# tree that produced them. This is the fix: every configure stamps the build
# directory with what it was, and the ctest target `build_provenance` fails if
# the stamp is missing or short a field.
#
# WRITTEN HERE, NOT BY scripts/xpm_build.sh. A stamp only the wrapper wrote
# would be absent from exactly the builds nobody can trace -- the ones made by
# calling cmake directly, which is what CI does and what anybody debugging
# does. CMake is the one thing every build goes through, and it is also the
# only place that knows the RESOLVED compiler version and the FINAL flags
# (this file mutates CMAKE_CXX_FLAGS above).
#
# XPMATH_ARCH and XPMATH_OPT are supplied by the wrapper. A direct cmake call
# leaves them at their defaults below, which says so rather than guessing.
set(XPMATH_ARCH "unspecified" CACHE STRING
    "Name of the target-hardware configuration this build was made for")
set(XPMATH_OPT "from-CMAKE_CXX_FLAGS_RELEASE" CACHE STRING
    "Optimization level requested by scripts/xpm_build.sh --opt")

# git HEAD, with -dirty when the tree carries uncommitted changes to tracked
# files. `git describe --dirty` is not used: this repo has tags, so it would
# report a tag name where a SHA is wanted.
set(_xp_git_head "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                  OUTPUT_VARIABLE _xp_sha OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _xp_rc)
  if(_xp_rc EQUAL 0 AND _xp_sha)
    set(_xp_git_head "${_xp_sha}")
    # --quiet exits 1 when there is a difference. Tracked files only, which is
    # the conventional meaning of "dirty"; an untracked scratch file is not a
    # change to the tree that produced the binary.
    execute_process(COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD --
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    RESULT_VARIABLE _xp_dirty ERROR_QUIET)
    if(NOT _xp_dirty EQUAL 0)
      set(_xp_git_head "${_xp_sha}-dirty")
    endif()
  endif()
endif()

# The flags the compiler ACTUALLY gets: the global set plus the per-config set
# for the configuration being built. Single-config generators only; a
# multi-config generator has no one answer at configure time and says so.
if(CMAKE_BUILD_TYPE)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _xp_cfg)
  set(_xp_all_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_xp_cfg}}")
else()
  set(_xp_all_flags "${CMAKE_CXX_FLAGS} (multi-config generator: per-config flags vary)")
endif()
string(STRIP "${_xp_all_flags}" _xp_all_flags)

# CORE_PLAN C10: this repository does not find or link Kokkos. The stamp fields
# stay so build_provenance's required-key list does not change; the values say
# so rather than echoing a deleted option.
set(_xp_kokkos_prefix "(none: Kokkos lives in xpmath-kokkos)")

string(TIMESTAMP _xp_stamp_utc "%Y-%m-%dT%H:%M:%SZ" UTC)

# THE STANDARD THIS BUILD COMPILES AT -- CORE_PLAN C5 step 6 / C10
#
# `cxx-standard` above is what the project asks for. `cxx-standard-raised` names
# any exception. After C10 there are none: no target links Kokkos, so nothing
# inherits INTERFACE_COMPILE_FEATURES cxx_std_20. check_build_provenance.cmake
# MEASURES the same claim out of compile_commands.json.
set(_xp_raised "(none: no target in this tree links Kokkos)")

# WHICH SIDE OF THE C5 SPLIT THIS BUILD DIRECTORY IS -- CORE_PLAN C5 step 5.
#
# DERIVED FROM THE TWO OPTIONS, NOT PASSED IN AS A -D. scripts/xpm_build.sh
# produces <dir>/host and <dir>/device and could have stamped the answer it
# already knows, but then a bare `cmake -B build` -- which is what CI runs and
# what anybody debugging runs -- would carry no tree field at all, or a wrong
# one, and this file is the ONE thing every build goes through. Deriving it
# also makes the field impossible to disagree with the build it describes.
#
# `both` is a real and correct value, not a fallback: it is the one-tree,
# one-compiler configuration the bare cmake path produces.
if(XPMATH_BUILD_HOST_TARGETS AND XPMATH_BUILD_DEVICE_TARGETS)
  set(_xp_tree "both")
elseif(XPMATH_BUILD_HOST_TARGETS)
  set(_xp_tree "host")
else()
  set(_xp_tree "device")
endif()

# check_build_provenance.cmake parses `key: value`, so every field is one line
# and no value may be empty.
file(WRITE "${CMAKE_BINARY_DIR}/build-info.txt"
"# xpmath build provenance -- written by CMakeLists.txt at configure time.
# Regenerated on every configure. Do not edit; do not commit.
arch: ${XPMATH_ARCH}
tree: ${_xp_tree}
git-head: ${_xp_git_head}
compiler: ${CMAKE_CXX_COMPILER}
compiler-id: ${CMAKE_CXX_COMPILER_ID}
compiler-version: ${CMAKE_CXX_COMPILER_VERSION}
kokkos: OFF
kokkos-prefix: ${_xp_kokkos_prefix}
cxx-standard: ${CMAKE_CXX_STANDARD}
cxx-standard-raised: ${_xp_raised}
build-type: ${CMAKE_BUILD_TYPE}
opt-level: ${XPMATH_OPT}
cxx-flags: ${_xp_all_flags}
timestamp-utc: ${_xp_stamp_utc}
")

if(_xp_git_head MATCHES "-dirty$")
  message(STATUS "build provenance: tree is DIRTY -- build-info.txt records ${_xp_git_head}")
endif()
message(STATUS "build provenance: ${CMAKE_BINARY_DIR}/build-info.txt (arch=${XPMATH_ARCH}, tree=${_xp_tree})")

# ---- Tests -----------------------------------------------------------------
# XPMATH_BUILD_TESTS only. The deprecated KOKKOS_EP_BUILD_TESTS alias that C4
# kept for one cycle was removed in CORE_PLAN C10 step 4.
option(XPMATH_BUILD_TESTS "Build the extended-precision test suite" ON)

if(XPMATH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

