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

# The host __float128 oracle is provided by Kokkos's quadmath overloads
# (impl/Kokkos_QuadPrecisionMath.hpp), which are compiled and linked when Kokkos
# itself is built with Kokkos_ENABLE_LIBQUADMATH=ON. No separate find_library or
# x86_64 gate is needed here — linking Kokkos::kokkos brings in libquadmath.
# Kokkos exposes its enabled TPLs via the Kokkos_TPLS list (there is no
# Kokkos_ENABLE_LIBQUADMATH variable for TPLs).
if("LIBQUADMATH" IN_LIST Kokkos_TPLS)
  message(STATUS "Kokkos built with LIBQUADMATH: host __float128 oracle enabled")
else()
  message(WARNING
    "Kokkos was NOT built with Kokkos_ENABLE_LIBQUADMATH=ON; the host "
    "__float128 oracle in the demos will not compile. Rebuild Kokkos with "
    "-DKokkos_ENABLE_LIBQUADMATH=ON.")
endif()

# The complex __complex128 oracle in src/demo_complex.cpp routes through a LOCAL
# Kokkos extension header, impl/Kokkos_ComplexQuadPrecisionMath.hpp, which is NOT
# upstream in Kokkos (see patches/README.md). Probe the Kokkos install for it and,
# if absent, warn and continue — same graceful-degradation posture used above for
# LIBQUADMATH itself. Only kokkos_ep_demo_complex depends on this header.
if("LIBQUADMATH" IN_LIST Kokkos_TPLS)
  include(CheckCXXSourceCompiles)
  get_target_property(_kokkos_inc Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  set(CMAKE_REQUIRED_INCLUDES ${_kokkos_inc})
  set(CMAKE_REQUIRED_FLAGS "-fext-numeric-literals")
  # Link against Kokkos so libquadmath (the ::c<fn>q symbols the wrapper forwards
  # to) resolves, exactly as kokkos_ep_demo_complex does. Without this the probe
  # compiles but fails to link (undefined reference to cexpq), a false negative.
  set(CMAKE_REQUIRED_LIBRARIES Kokkos::kokkos)
  check_cxx_source_compiles("
    #include <impl/Kokkos_ComplexQuadPrecisionMath.hpp>
    int main() {
      __complex128 z; __real__ z = 1.0Q; __imag__ z = 2.0Q;
      __complex128 w = Kokkos::exp(z);
      return (int)Kokkos::real(w) * 0;
    }" KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
    message(STATUS
      "Kokkos install has impl/Kokkos_ComplexQuadPrecisionMath.hpp: "
      "complex __complex128 oracle enabled")
  else()
    message(WARNING
      "Kokkos install is MISSING impl/Kokkos_ComplexQuadPrecisionMath.hpp; "
      "the four complex demos (kokkos_ep_demo_complex, _ff_complex, "
      "_qf_complex, _tf_complex) are SKIPPED — they are the only targets that "
      "need it. Everything else, including the whole ctest suite, still "
      "configures and builds. This header is a local extension carried in this "
      "repo — apply it per patches/README.md, then rebuild Kokkos with "
      "-DKokkos_ENABLE_LIBQUADMATH=ON to get the complex demos back.")
  endif()
endif()

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

# KI-21. Defined only when the probe above found the header. Without the
# guard the WARNING promised degradation and then the build failed anyway:
# src/demo_complex.cpp includes the oracle unconditionally, so `make` stopped
# here and 27 of the 34 ctest targets — none of which need the header —
# never linked and read as `***Not Run`.
if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
  add_executable(kokkos_ep_demo_complex src/demo_complex.cpp)
  target_include_directories(kokkos_ep_demo_complex PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(kokkos_ep_demo_complex PRIVATE Kokkos::kokkos)
  list(APPEND XP_COMPLEX_DEMO_TARGETS kokkos_ep_demo_complex)
endif()

# ---- FF (float-float) demos (T2.0) ------------------------------------------
# Same oracle plumbing as the DD demos: the __float128 real oracle comes through
# Kokkos::kokkos (LIBQUADMATH TPL, T0.0); kokkos_ep_demo_ff_complex additionally
# needs the local impl/Kokkos_ComplexQuadPrecisionMath.hpp wrapper (T0.3, probed
# above). -fext-numeric-literals is already set globally for GCC.
add_executable(kokkos_ep_demo_ff src/demo_ff_real.cpp)
target_include_directories(kokkos_ep_demo_ff PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_ff PRIVATE Kokkos::kokkos)

# KI-21. Defined only when the probe above found the header. Without the
# guard the WARNING promised degradation and then the build failed anyway:
# src/demo_ff_complex.cpp includes the oracle unconditionally, so `make` stopped
# here and 27 of the 34 ctest targets — none of which need the header —
# never linked and read as `***Not Run`.
if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
  add_executable(kokkos_ep_demo_ff_complex src/demo_ff_complex.cpp)
  target_include_directories(kokkos_ep_demo_ff_complex PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(kokkos_ep_demo_ff_complex PRIVATE Kokkos::kokkos)
  list(APPEND XP_COMPLEX_DEMO_TARGETS kokkos_ep_demo_ff_complex)
endif()

# ---- QF (quad-float, 4xFP32) demos (T3.0b real, T3.0c complex) ---------------
# Real transcendentals exercised against the __float128 oracle (T3.0b) and
# complex ops against the __complex128 oracle (T3.0c), same plumbing as the
# DD/FF demos. kokkos_ep_demo_qf_complex additionally needs the local
# impl/Kokkos_ComplexQuadPrecisionMath.hpp wrapper (T0.3, probed above), exactly
# as kokkos_ep_demo_ff_complex does.
add_executable(kokkos_ep_demo_qf src/demo_qf_real.cpp)
target_include_directories(kokkos_ep_demo_qf PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_qf PRIVATE Kokkos::kokkos)

# KI-21. Defined only when the probe above found the header. Without the
# guard the WARNING promised degradation and then the build failed anyway:
# src/demo_qf_complex.cpp includes the oracle unconditionally, so `make` stopped
# here and 27 of the 34 ctest targets — none of which need the header —
# never linked and read as `***Not Run`.
if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
  add_executable(kokkos_ep_demo_qf_complex src/demo_qf_complex.cpp)
  target_include_directories(kokkos_ep_demo_qf_complex PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(kokkos_ep_demo_qf_complex PRIVATE Kokkos::kokkos)
  list(APPEND XP_COMPLEX_DEMO_TARGETS kokkos_ep_demo_qf_complex)
endif()

# ---- TF (triple-float, 3xFP32) demos (S10 Phase 5 real + complex) ----------
# TF real transcendentals exercised against the __float128 oracle (S10 P5) and
# complex ops against the __complex128 oracle, same plumbing as the DD/FF/QF
# demos. kokkos_ep_demo_tf_complex additionally needs the local
# impl/Kokkos_ComplexQuadPrecisionMath.hpp wrapper (T0.3, probed in CMakeLists
# lines 43-73), exactly as kokkos_ep_demo_ff_complex and kokkos_ep_demo_qf_complex do.
add_executable(kokkos_ep_demo_tf src/demo_tf_real.cpp)
target_include_directories(kokkos_ep_demo_tf PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_tf PRIVATE Kokkos::kokkos)

# KI-21. Defined only when the probe above found the header. Without the
# guard the WARNING promised degradation and then the build failed anyway:
# src/demo_tf_complex.cpp includes the oracle unconditionally, so `make` stopped
# here and 27 of the 34 ctest targets — none of which need the header —
# never linked and read as `***Not Run`.
if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
  add_executable(kokkos_ep_demo_tf_complex src/demo_tf_complex.cpp)
  target_include_directories(kokkos_ep_demo_tf_complex PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(kokkos_ep_demo_tf_complex PRIVATE Kokkos::kokkos)
  list(APPEND XP_COMPLEX_DEMO_TARGETS kokkos_ep_demo_tf_complex)
endif()

# ---- Cost benchmark ----------------------------------------------------------
# Times each backend against the incumbent at comparable precision (FF vs FP64,
# DD/QF vs __float128) rather than against FP64 across the board. Same oracle
# plumbing as the demos: __float128 arrives through Kokkos::kokkos (LIBQUADMATH
# TPL, T0.0).
add_executable(kokkos_ep_bench_cost src/bench_cost.cpp)
target_include_directories(kokkos_ep_bench_cost PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_bench_cost PRIVATE Kokkos::kokkos)

# KI-21: the four complex demos are named through XP_COMPLEX_DEMO_TARGETS, which
# is empty when the probe did not find impl/Kokkos_ComplexQuadPrecisionMath.hpp.
# Naming a target that was never added is a hard configure error, so this list
# cannot be spelled out literally.
install(TARGETS kokkos_ep_demo
                kokkos_ep_demo_ff
                kokkos_ep_demo_qf
                kokkos_ep_demo_tf
                kokkos_ep_bench_cost
                ${XP_COMPLEX_DEMO_TARGETS}
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
# THIRD_PARTY_INCLUDE (set above) is consumed by tests/CMakeLists.txt.
option(KOKKOS_EP_BUILD_TESTS "Build the extended-precision test suite" ON)
if(KOKKOS_EP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

