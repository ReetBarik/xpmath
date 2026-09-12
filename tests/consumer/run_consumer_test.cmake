# Driver that lets ctest run the consumer packaging test.
#
# WHY THIS FILE EXISTS AND WHAT IT DOES NOT DO.
#
# The poison logic lives in ONE place: run_consumer_test.sh. This script does
# not reimplement it. All it does is the part ctest cannot do for itself --
# produce a completed `cmake --install` of this project into a scratch prefix,
# at test RUN time, because no such install exists during the configure that
# registers the test.
#
# So: configure + build + install xpmath into a throwaway prefix, then hand
# that prefix to the shell script, which owns the clean case and both poisons.
# Two implementations of the same contract would be exactly the duplication the
# rest of this suite is trying to get rid of.
#
# Invoked by tests/CMakeLists.txt, never by hand. For a manual run, call
# run_consumer_test.sh directly against an install prefix you already have.
#
# Required -D arguments: XPMATH_SOURCE_DIR, XPMATH_SCRATCH, XPMATH_GENERATOR,
# XPMATH_CXX_COMPILER.

cmake_minimum_required(VERSION 3.16)

foreach(v XPMATH_SOURCE_DIR XPMATH_SCRATCH XPMATH_GENERATOR XPMATH_CXX_COMPILER)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "run_consumer_test.cmake: -D${v} is required")
  endif()
endforeach()

# The nested configure builds the WHOLE project, and the top-level CMakeLists
# does find_package(Kokkos REQUIRED) unconditionally -- so it needs the same
# prefix path the outer configure was given. Forwarded rather than rediscovered:
# a nested build that found a DIFFERENT Kokkos would be testing a different
# package than the one just built.
if(NOT DEFINED XPMATH_PREFIX_PATH)
  set(XPMATH_PREFIX_PATH "")
endif()
if(NOT DEFINED XPMATH_KOKKOS_DIR)
  set(XPMATH_KOKKOS_DIR "")
endif()

# Multi-config generators would need --config threaded through every nested
# step. Rather than half-support them, skip with ctest's SKIP_RETURN_CODE
# convention so the result reads as "not run here", never as a pass.
if(NOT XPMATH_GENERATOR MATCHES "Makefiles|Ninja")
  message(STATUS "consumer test: generator '${XPMATH_GENERATOR}' not supported; skipping")
  return()
endif()

find_program(BASH_EXE bash)
if(NOT BASH_EXE)
  message(FATAL_ERROR "consumer test: bash not found")
endif()

set(build_dir  "${XPMATH_SCRATCH}/build")
set(prefix_dir "${XPMATH_SCRATCH}/prefix")
set(work_dir   "${XPMATH_SCRATCH}/work")

# A stale scratch tree from a previous run would let a DELETED install rule
# still look present, so the prefix is rebuilt from nothing every time.
file(REMOVE_RECURSE "${XPMATH_SCRATCH}")
file(MAKE_DIRECTORY "${XPMATH_SCRATCH}")

# KOKKOS_EP_BUILD_TESTS=OFF is what keeps this from recursing: with tests on,
# the nested configure would register consumer_package again inside itself.
# It also means the nested build compiles only what install() ships, which is
# the point -- if the package needs a test target to be installable, that is a
# defect in the package.
message(STATUS "consumer test: configuring xpmath into ${prefix_dir}")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${XPMATH_SOURCE_DIR}" -B "${build_dir}"
          -G "${XPMATH_GENERATOR}"
          -DCMAKE_CXX_COMPILER=${XPMATH_CXX_COMPILER}
          -DCMAKE_BUILD_TYPE=Release
          -DKOKKOS_EP_BUILD_TESTS=OFF
          -DCMAKE_INSTALL_PREFIX=${prefix_dir}
          "-DCMAKE_PREFIX_PATH=${XPMATH_PREFIX_PATH}"
          -DKokkos_DIR=${XPMATH_KOKKOS_DIR}
  RESULT_VARIABLE rc
  OUTPUT_FILE "${XPMATH_SCRATCH}/configure.log"
  ERROR_FILE  "${XPMATH_SCRATCH}/configure.log")
if(NOT rc EQUAL 0)
  file(READ "${XPMATH_SCRATCH}/configure.log" log)
  message(FATAL_ERROR "consumer test: nested configure failed (${rc})\n${log}")
endif()

message(STATUS "consumer test: building")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${build_dir}" --parallel
  RESULT_VARIABLE rc
  OUTPUT_FILE "${XPMATH_SCRATCH}/build.log"
  ERROR_FILE  "${XPMATH_SCRATCH}/build.log")
if(NOT rc EQUAL 0)
  file(READ "${XPMATH_SCRATCH}/build.log" log)
  message(FATAL_ERROR "consumer test: nested build failed (${rc})\n${log}")
endif()

message(STATUS "consumer test: installing")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
  RESULT_VARIABLE rc
  OUTPUT_FILE "${XPMATH_SCRATCH}/install.log"
  ERROR_FILE  "${XPMATH_SCRATCH}/install.log")
if(NOT rc EQUAL 0)
  file(READ "${XPMATH_SCRATCH}/install.log" log)
  message(FATAL_ERROR "consumer test: install failed (${rc})\n${log}")
endif()

# The script owns the verdict: clean must pass, both poisons must fail.
message(STATUS "consumer test: running run_consumer_test.sh")
execute_process(
  COMMAND "${BASH_EXE}" "${XPMATH_SOURCE_DIR}/tests/consumer/run_consumer_test.sh"
          "${prefix_dir}" "${work_dir}"
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "consumer test: FAILED (${rc}); logs under ${XPMATH_SCRATCH}")
endif()

file(REMOVE_RECURSE "${XPMATH_SCRATCH}")
message(STATUS "consumer test: PASS")
