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
# src/bench_cost.cpp is the one file left under src/ that includes <quadmath.h>.
# There libquadmath is the measurement SUBJECT — the incumbent that DD and QF are
# timed against — not an oracle, so it stays, and it still resolves ::expq/::powq
# through whatever supplies them (today, Kokkos's TPL when it is enabled).

set(THIRD_PARTY_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/include")

# S2 (renamed in S3): the standalone, Kokkos-free extended-precision core
# lives in include/xp/. The compat wrappers in third_party/include/ resolve
# `#include <xp/...>`
# through this directory, so every target that already gets THIRD_PARTY_INCLUDE
# needs it as well. Appended to the same variable (which tests/CMakeLists.txt
# also consumes) so no individual target's include list has to change.
list(APPEND THIRD_PARTY_INCLUDE "${CMAKE_SOURCE_DIR}/include")

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

# ---- Cost benchmark ----------------------------------------------------------
# Times each backend against the incumbent at comparable precision (FF vs FP64,
# DD/QF vs __float128) rather than against FP64 across the board. This is the ONE
# target here that still wants libquadmath, and not as an oracle: __float128 is
# the incumbent being benchmarked, so src/bench_cost.cpp includes <quadmath.h>
# and calls ::expq/::powq directly. It resolves them through Kokkos::kokkos when
# Kokkos carries the LIBQUADMATH TPL; against a Kokkos built without it, this one
# target does not link. Every other target here is libquadmath-free.
add_executable(kokkos_ep_bench_cost src/bench_cost.cpp)
target_include_directories(kokkos_ep_bench_cost PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_bench_cost PRIVATE Kokkos::kokkos)

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
                kokkos_ep_bench_cost
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

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

# ---- Tests (T0.1: backend-parameterized harness) ----------------------------
# THIRD_PARTY_INCLUDE (set above) is consumed by tests/CMakeLists.txt.
option(KOKKOS_EP_BUILD_TESTS "Build the extended-precision test suite" ON)
if(KOKKOS_EP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

