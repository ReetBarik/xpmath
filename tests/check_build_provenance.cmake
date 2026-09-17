# ===========================================================================
# check_build_provenance.cmake — the build directory must say what it is
# ===========================================================================
#
# WHY THIS EXISTS. Three separate campaign artifacts produced this month could
# not be traced back to the tree that produced them: a results table, a log,
# and an assembly dump, none of which recorded the commit, the compiler or the
# flags. The cost is not that the numbers are wrong; it is that nobody can say
# whether they are, so they cannot be used as a baseline and the run has to be
# repeated on hardware that is queued for days.
#
# The stamp is written by the top-level CMakeLists.txt on every configure, so
# it exists for a plain `cmake -B build` as much as for
# scripts/xpm_build.sh. This target is what stops it rotting: a field that
# stops being written fails here rather than going unnoticed until the next
# campaign needs it.
#
# WHAT IT CHECKS. Presence of the file, presence of every required key, and a
# NON-EMPTY value for each. It deliberately does not check the values
# themselves: "arch: mi250" cannot be validated from inside the build, and a
# check that pretends otherwise would only be asserting that CMake can echo a
# -D back. There are two exceptions, and both earn it:
#
#   git-head  the -dirty suffix is REPORTED (not failed on) so a dirty build is
#             visible in the ctest output rather than only in a file nobody
#             opens.
#   tree      the value IS validated, against the closed set host/device/both.
#             Unlike arch, this field is not echoed from a -D -- CMakeLists.txt
#             DERIVES it from XPMATH_BUILD_HOST_TARGETS and
#             XPMATH_BUILD_DEVICE_TARGETS, so there is a right answer and this
#             file knows all three of them. A fourth value means the derivation
#             was changed without this check being told, which is exactly the
#             rot this target exists to catch.
#
# WHY `tree` IS REQUIRED AT ALL (CORE_PLAN C5 step 5). scripts/xpm_build.sh
# produces TWO build directories from one command, and the artifacts that come
# out of them -- logs, raw device results, assembly dumps -- are what later
# sections commit. Told apart only by path, two stamps become one stamp the
# moment either is copied off a compute node. `tree:` is what survives that.
#
# Invoked as:
#   cmake -DINFO=<path to build-info.txt> -P this
# ===========================================================================

if(NOT DEFINED INFO)
  message(FATAL_ERROR "build provenance: INFO not set")
endif()

if(NOT EXISTS "${INFO}")
  message(FATAL_ERROR
    "build provenance: no build-info.txt at ${INFO}.\n"
    "The configure that should have written it did not run, or the stamp block "
    "in the top-level CMakeLists.txt was removed. An unstamped build directory "
    "produces artifacts nobody can trace to a tree.")
endif()

set(REQUIRED
    arch
    tree
    git-head
    compiler
    compiler-id
    compiler-version
    kokkos
    kokkos-prefix
    build-type
    opt-level
    cxx-flags
    timestamp-utc)

file(STRINGS "${INFO}" lines)
set(failures "")
set(dirty FALSE)
set(tree_value "")

foreach(key IN LISTS REQUIRED)
  set(found FALSE)
  foreach(line IN LISTS lines)
    if(line MATCHES "^${key}: (.+)$")
      set(found TRUE)
      set(value "${CMAKE_MATCH_1}")
      string(STRIP "${value}" value)
      if(value STREQUAL "")
        string(APPEND failures "  ${key}: present but empty\n")
      else()
        message("  ${key}: ${value}")
      endif()
      if(key STREQUAL "git-head" AND value MATCHES "-dirty$")
        set(dirty TRUE)
      endif()
      if(key STREQUAL "tree")
        set(tree_value "${value}")
      endif()
      break()
    endif()
  endforeach()
  if(NOT found)
    string(APPEND failures "  ${key}: MISSING from ${INFO}\n")
  endif()
endforeach()

# The one value check. Deliberately AFTER the presence loop, so a MISSING tree
# field is reported as missing rather than as "not one of host/device/both",
# which would send the reader looking for a typo in a line that is not there.
if(NOT tree_value STREQUAL ""
   AND NOT tree_value STREQUAL "host"
   AND NOT tree_value STREQUAL "device"
   AND NOT tree_value STREQUAL "both")
  string(APPEND failures
         "  tree: '${tree_value}' is not one of host, device, both\n")
endif()

if(NOT failures STREQUAL "")
  message("build provenance: FAIL")
  message("${failures}")
  message(FATAL_ERROR
    "build-info.txt is short a field. Every campaign artifact this build "
    "produces is traceable only through that file; see the header of "
    "tests/check_build_provenance.cmake.")
endif()

if(dirty)
  # Not a failure. A dirty tree is a normal state to build in; an UNRECORDED
  # dirty tree is the defect, and the stamp recorded it.
  message("build provenance: PASS -- note, this build was made from a DIRTY tree")
else()
  message("build provenance: PASS")
endif()
