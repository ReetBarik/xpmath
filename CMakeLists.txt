cmake_minimum_required(VERSION 3.16)

# The project is named xpmath because that is the name a consumer writes in
# find_package(). The repository directory keeps its historical name; nothing
# in the build depended on the old project() name (verified: no reference to
# PROJECT_NAME or CMAKE_PROJECT_NAME anywhere in the tree).
project(xpmath VERSION 0.1.0 LANGUAGES CXX)

# Included here, not next to the install rules, because CMAKE_INSTALL_BINDIR is
# used by the demo install() further down -- before the package block. An
# undefined CMAKE_INSTALL_* expands to the empty string and silently installs
# to the prefix root.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# ---------------------------------------------------------------------------
# KOKKOS IS OPTIONAL, AND THAT IS THE WHOLE POINT OF include/xp/.
#
# The library is the headers in include/xp/. They do not include or require
# Kokkos -- every mention of it in them is a comment. Until this option existed
# the BUILD did not believe that: find_package(Kokkos REQUIRED) sat at the top
# of this file and gated everything downstream of it, including the header-only
# export, its install rules, and the entire test suite. So you could not build,
# install or TEST the Kokkos-free core without Kokkos, and `sweep_accuracy` --
# the binary behind both correctness gates, which links no Kokkos and includes
# only include/ -- was unreachable on a machine that had none.
#
# With XPMATH_WITH_KOKKOS=OFF you get: the installable package, both gates, and
# every test that does not link Kokkos. What you lose is exactly what genuinely
# needs it -- the nine demos and the tests that exercise the compat wrappers in
# third_party/include/.
# ---------------------------------------------------------------------------
option(XPMATH_WITH_KOKKOS
       "Build the Kokkos demos and the Kokkos-linked tests (needs a Kokkos install)" ON)

# ---------------------------------------------------------------------------
# WHICH SIDE OF THE BUILD TO PRODUCE -- CORE_PLAN C5 step 1
# ---------------------------------------------------------------------------
# CMake supports exactly ONE CXX compiler per project and has no per-target
# override, so "host tools with g++, device tests with nvcc/hipcc" can only be
# two build trees. These two options are what lets one source tree produce
# either half. C5's remaining steps (the wrapper driving both trees) are a
# separate change; this one only makes the halves addressable.
#
# BOTH DEFAULT ON, AND THAT IS LOAD-BEARING. With both ON the build is exactly
# what it was before this option existed -- one tree, one compiler, all 61
# ctest targets. That is the `cmake -B build` path documented in CLAUDE.md and
# it is what CI runs; a default that changed the target set would be a silent
# CI change wearing a refactor's clothes.
#
# THE FAILURE THIS PREVENTS is a host-only tool compiled by a device compiler.
# On A100 (job 1000938) `xpm_build.sh --arch a100` set nvcc_wrapper
# project-wide, so sweep_accuracy -- which links no Kokkos and no device code
# at all -- went through nvcc and failed, taking both gates, oracle_conv_test
# and both gate selftests down with it. Five of that job's eleven red results
# were that one mistake.
#
# EVERY registered test is tagged host or device, and the assertion at the
# bottom of tests/CMakeLists.txt fails the CONFIGURE if one is not. A target
# that is silently built in both trees is the other half of the same defect:
# it gets compiled twice, and the tree it does not belong to is the one that
# will eventually break it.
option(XPMATH_BUILD_HOST_TARGETS
       "Build the host-side targets: the oracle-scored tools, the gates and the host test halves" ON)
option(XPMATH_BUILD_DEVICE_TARGETS
       "Build the device-side targets: the *_test_device halves, the harness self-test and the demos" ON)

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

if(XPMATH_WITH_KOKKOS)
find_package(Kokkos REQUIRED)

# There is deliberately no libquadmath probe here any more, and no
# Kokkos_ENABLE_LIBQUADMATH requirement. Until B3 this file warned when the
# Kokkos install lacked the LIBQUADMATH TPL, and then ran a
# check_cxx_source_compiles probe for impl/Kokkos_ComplexQuadPrecisionMath.hpp
# (a header that is NOT upstream in Kokkos — this repo carried it as a patch)
# to decide whether the four complex demos could be added at all. Both existed
# only to serve the accuracy columns the demos used to print. Those columns are
# gone: the demos are timing and smoke only, and the accuracy record is
# validation/sweep/, scored in ulps against an MPFR/MPC oracle with one verdict
# per point (docs/CORRECTNESS.md). With them went the patch, the probe, and the
# requirement — a Kokkos built WITHOUT libquadmath, such as the MI250
# (hip/gfx90a) install, is a first-class supported target.
#
# No target here uses libquadmath. The last consumer was the cost benchmark
# (src/bench_cost.cpp, which called ::expq/::powq from libquadmath as the
# incumbent being timed against), and it is deleted.

set(THIRD_PARTY_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/include")

# S2 (renamed in S3): the standalone, Kokkos-free extended-precision core
# lives in include/xp/. The compat wrappers in third_party/include/ resolve
# `#include <xp/...>`
# through this directory, so every target that already gets THIRD_PARTY_INCLUDE
# needs it as well. Appended to the same variable (which tests/CMakeLists.txt
# also consumes) so no individual target's include list has to change.
list(APPEND THIRD_PARTY_INCLUDE "${CMAKE_SOURCE_DIR}/include")

# --- THE EIGHT DEMOS ARE DEVICE TARGETS -- CORE_PLAN C5 step 1 --------------
# Every one of them launches Kokkos kernels, so they belong on the side of the
# split that gets the arch's compiler and the arch's Kokkos. Left ungated they
# would be compiled by BOTH trees: once by the host tree's g++, where nothing
# about them is being exercised that the device tree does not exercise better,
# and once by the device tree's nvcc/hipcc, where they are the point. Building
# them twice is the "silently built in both configurations" failure named above,
# spelled with executables instead of tests.
#
# The demos are not ctest targets, so they do not appear in the tagged-test
# assertion at the bottom of tests/CMakeLists.txt. This comment is where their
# side is recorded.
#
# The install() list goes inside the same guard: naming a target that was never
# added is a hard configure error, which is how KI-21 used to fire.
#
# CONSEQUENCE, WRITTEN DOWN RATHER THAN DISCOVERED LATER: third_party/include/
# is exercised ONLY by these eight (tests/CMakeLists.txt's C4 block says so), so
# a host-only tree does not compile the Kokkos compat wrappers at all. A build
# with both options ON -- the default, the bare `cmake -B build`, and CI -- does.
if(XPMATH_BUILD_DEVICE_TARGETS)

add_executable(kokkos_ep_demo src/demo_real.cpp)
target_include_directories(kokkos_ep_demo PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo PRIVATE Kokkos::kokkos)

add_executable(kokkos_ep_demo_complex src/demo_complex.cpp)
target_include_directories(kokkos_ep_demo_complex PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_complex PRIVATE Kokkos::kokkos)

# ---- FF (float-float) demos (T2.0) ------------------------------------------
# Timing and smoke only, like the DD demos: no oracle, no libquadmath.
# -fext-numeric-literals is already set globally for GCC.
add_executable(kokkos_ep_demo_ff src/demo_ff_real.cpp)
target_include_directories(kokkos_ep_demo_ff PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_ff PRIVATE Kokkos::kokkos)

add_executable(kokkos_ep_demo_ff_complex src/demo_ff_complex.cpp)
target_include_directories(kokkos_ep_demo_ff_complex PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_ff_complex PRIVATE Kokkos::kokkos)

# ---- QF (quad-float, 4xFP32) demos (T3.0b real, T3.0c complex) ---------------
# Timing and smoke only. These two used to carry the RC-0/RC-1 accuracy verdict
# that B3 removed -- the correctness contract allows one verdict per point and
# the sweep gates issue it (docs/CORRECTNESS.md).
add_executable(kokkos_ep_demo_qf src/demo_qf_real.cpp)
target_include_directories(kokkos_ep_demo_qf PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_qf PRIVATE Kokkos::kokkos)

add_executable(kokkos_ep_demo_qf_complex src/demo_qf_complex.cpp)
target_include_directories(kokkos_ep_demo_qf_complex PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_qf_complex PRIVATE Kokkos::kokkos)

# ---- TF (triple-float, 3xFP32) demos (S10 Phase 5 real + complex) ----------
# Timing and smoke only, same shape as the QF pair above.
add_executable(kokkos_ep_demo_tf src/demo_tf_real.cpp)
target_include_directories(kokkos_ep_demo_tf PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_tf PRIVATE Kokkos::kokkos)

add_executable(kokkos_ep_demo_tf_complex src/demo_tf_complex.cpp)
target_include_directories(kokkos_ep_demo_tf_complex PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_tf_complex PRIVATE Kokkos::kokkos)


# KI-21 is closed by deletion. The four complex demos used to be named through
# XP_COMPLEX_DEMO_TARGETS because the probe could leave them unbuilt, and naming
# a target that was never added is a hard configure error. Every demo is now
# added unconditionally, so the list can be spelled out.
install(TARGETS kokkos_ep_demo
                kokkos_ep_demo_complex
                kokkos_ep_demo_ff
                kokkos_ep_demo_ff_complex
                kokkos_ep_demo_qf
                kokkos_ep_demo_qf_complex
                kokkos_ep_demo_tf
                kokkos_ep_demo_tf_complex
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

else()
  message(STATUS
    "XPMATH_BUILD_DEVICE_TARGETS=OFF: the eight Kokkos demos are not built in "
    "this tree, and with them third_party/include/ is not compiled here.")
endif()  # XPMATH_BUILD_DEVICE_TARGETS

else()
  message(STATUS
    "XPMATH_WITH_KOKKOS=OFF: building the header-only package, both accuracy "
    "gates and every Kokkos-free test. The nine demos and the wrapper-linked "
    "tests are skipped -- they are the only things that need Kokkos.")
endif()

# ---- Installable package (header-only core) ---------------------------------
#
# WHAT IS INSTALLED, AND WHY IT IS NOT THE WHOLE REPO.
#
# xpmath the LIBRARY is the 11 headers in include/xp/. They are header-only and
# they do not include or require Kokkos -- every mention of Kokkos in them is a
# comment (verified: `grep -n Kokkos include/xp/*.hpp` matches only comment
# lines). tests/standalone/*_no_kokkos_smoke.cpp already compiles them with a
# bare g++ and an include path containing only include/.
#
# So the exported package deliberately does NOT find_package(Kokkos). A consumer
# who wants the numeric core gets it with zero TPLs; a consumer who wants device
# execution brings their own Kokkos and uses it alongside. Making Kokkos a hard
# requirement of the export would impose a dependency the headers do not have.
#
# third_party/include/ is the Kokkos COMPAT WRAPPER, not the library. It is not
# installed: it exists so this repo's own Kokkos demos can include <xp/...>
# through the old Kokkos-facing names.

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
# (this file mutates CMAKE_CXX_FLAGS above, and Kokkos adds to the picture
# through its interface).
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

if(XPMATH_WITH_KOKKOS)
  set(_xp_kokkos_prefix "${Kokkos_DIR}")
  if(NOT _xp_kokkos_prefix)
    set(_xp_kokkos_prefix "${CMAKE_PREFIX_PATH}")
  endif()
else()
  set(_xp_kokkos_prefix "(none: XPMATH_WITH_KOKKOS=OFF)")
endif()

string(TIMESTAMP _xp_stamp_utc "%Y-%m-%dT%H:%M:%SZ" UTC)

# check_build_provenance.cmake parses `key: value`, so every field is one line
# and no value may be empty.
file(WRITE "${CMAKE_BINARY_DIR}/build-info.txt"
"# xpmath build provenance -- written by CMakeLists.txt at configure time.
# Regenerated on every configure. Do not edit; do not commit.
arch: ${XPMATH_ARCH}
git-head: ${_xp_git_head}
compiler: ${CMAKE_CXX_COMPILER}
compiler-id: ${CMAKE_CXX_COMPILER_ID}
compiler-version: ${CMAKE_CXX_COMPILER_VERSION}
kokkos: ${XPMATH_WITH_KOKKOS}
kokkos-prefix: ${_xp_kokkos_prefix}
cxx-standard: ${CMAKE_CXX_STANDARD}
build-type: ${CMAKE_BUILD_TYPE}
opt-level: ${XPMATH_OPT}
cxx-flags: ${_xp_all_flags}
timestamp-utc: ${_xp_stamp_utc}
")

if(_xp_git_head MATCHES "-dirty$")
  message(STATUS "build provenance: tree is DIRTY -- build-info.txt records ${_xp_git_head}")
endif()
message(STATUS "build provenance: ${CMAKE_BINARY_DIR}/build-info.txt (arch=${XPMATH_ARCH})")

# ---- Tests (T0.1: backend-parameterized harness) ----------------------------
# THIRD_PARTY_INCLUDE (set above) is NO LONGER consumed by tests/CMakeLists.txt.
# CORE_PLAN C4 step 5 deleted the last two test-side users of it along with the
# three kokkos_ep_add_* helpers. It stays set here because the eight demo
# targets in src/ still include the Kokkos compat wrappers through it, and they
# are what keeps that layer exercised until C10 moves them out.
#
# RENAMED BY CORE_PLAN C4: KOKKOS_EP_BUILD_TESTS -> XPMATH_BUILD_TESTS. The
# project is xpmath, and after C4 the suite is not Kokkos-specific -- most of it
# does not link Kokkos at all. The old spelling survives as a deprecated alias
# because dropping it outright would be the worst possible failure mode: an
# existing `-DKOKKOS_EP_BUILD_TESTS=OFF` would be ignored and the caller would
# get the whole suite it asked not to build, silently. C10 removes the alias.
option(XPMATH_BUILD_TESTS "Build the extended-precision test suite" ON)

# NOT an option(): the alias must distinguish "not set" from "set to OFF", and
# option() would default it to a value and make every ordinary build look like a
# deprecated invocation. An unset CACHE STRING is the empty string, so only an
# explicit -D -- or a cache carried over from before the rename -- is non-empty.
set(KOKKOS_EP_BUILD_TESTS "" CACHE STRING
    "DEPRECATED alias for XPMATH_BUILD_TESTS. Removed in CORE_PLAN C10.")
mark_as_advanced(KOKKOS_EP_BUILD_TESTS)
if(NOT KOKKOS_EP_BUILD_TESTS STREQUAL "")
  # ONCE, not once per configure. The sentinel is CACHE INTERNAL so a rebuild of
  # an existing tree does not re-emit it; a fresh tree, or one where the operator
  # changed the value, gets told.
  if(NOT DEFINED _xpm_warned_build_tests_alias
     OR NOT _xpm_warned_build_tests_alias STREQUAL "${KOKKOS_EP_BUILD_TESTS}")
    message(DEPRECATION
            "KOKKOS_EP_BUILD_TESTS is deprecated -- use XPMATH_BUILD_TESTS. "
            "Honouring KOKKOS_EP_BUILD_TESTS=${KOKKOS_EP_BUILD_TESTS} and "
            "ignoring XPMATH_BUILD_TESTS=${XPMATH_BUILD_TESTS} for this build "
            "tree.")
    set(_xpm_warned_build_tests_alias "${KOKKOS_EP_BUILD_TESTS}"
        CACHE INTERNAL "deprecation warning already emitted for this value")
  endif()
  # The old name WINS when both are given. A caller who still spells it the old
  # way is, by construction, one whose tooling has not been updated; silently
  # preferring the new default would defeat the point of keeping the alias.
  set(XPMATH_BUILD_TESTS ${KOKKOS_EP_BUILD_TESTS})
endif()

if(XPMATH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

