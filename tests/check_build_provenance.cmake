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
# -D back. The one exception is `git-head`, where the -dirty suffix is
# reported so a dirty build is visible in the ctest output rather than only in
# a file nobody opens.
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
      break()
    endif()
  endforeach()
  if(NOT found)
    string(APPEND failures "  ${key}: MISSING from ${INFO}\n")
  endif()
endforeach()

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
