cmake_minimum_required(VERSION 3.16)

project(kokkos-extended-precision-demo LANGUAGES CXX)

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
        RUNTIME DESTINATION bin)

# ---- Tests (T0.1: backend-parameterized harness) ----------------------------
# THIRD_PARTY_INCLUDE (set above) is consumed by tests/CMakeLists.txt.
option(KOKKOS_EP_BUILD_TESTS "Build the extended-precision test suite" ON)
if(KOKKOS_EP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

