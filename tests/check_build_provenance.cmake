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
#   cxx-standard  the value is MEASURED, against compile_commands.json. See the
#             next block; this is the one field that was known to be wrong.
#
# ---------------------------------------------------------------------------
# THE C++ STANDARD IS THE FIELD THAT LIED -- CORE_PLAN C5 step 6
# ---------------------------------------------------------------------------
# `cxx-standard: 17` was CMAKE_CXX_STANDARD echoed back, and it was not true of
# every translation unit: Kokkos 5.1.0's INTERFACE_COMPILE_FEATURES cxx_std_20
# raises every target that links it, so the demos compiled at -std=c++20 under
# a stamp that said 17. S8c hit the discrepancy on gfx90a and nothing in the
# build could settle it, because nothing in the build looked at a compile line.
#
# So this does. Given -DCOMPILE_COMMANDS=<compile_commands.json>, it reads the
# `-std=` off every recorded compile line and requires:
#
#   * at least one line to carry a -std= at all. A parse that matches nothing
#     is indistinguishable from a healthy build if it passes on silence, and
#     this repository has been bitten by that shape before (`ctest --test-dir`
#     on a missing directory exits 0). It FAILS instead.
#   * every TU to be at the stamped standard, EXCEPT ones belonging to a target
#     named in -DRAISED_TARGETS -- the closed set of Kokkos-linked targets,
#     which the top-level CMakeLists.txt spells out next to the add_executable
#     calls that create them. An unlisted target at a different standard is a
#     failure with its name and its standard printed.
#
# What it deliberately does NOT do is fail when a listed target turns out to be
# at the stamped standard after all. That is Kokkos relaxing its requirement,
# which is good news and not a defect in this tree; the counts are printed so
# the reader sees it.
#
# The target attribution comes from the object path (`CMakeFiles/<t>.dir/`),
# which is a Makefile/Ninja layout. That is the same condition under which
# CMakeLists.txt turns the export on, so the two agree by construction.
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
    cxx-standard
    cxx-standard-raised
    build-type
    opt-level
    cxx-flags
    timestamp-utc)

file(STRINGS "${INFO}" lines)
set(failures "")
set(dirty FALSE)
set(tree_value "")
set(std_value "")

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
      if(key STREQUAL "cxx-standard")
        set(std_value "${value}")
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

# ---------------------------------------------------------------------------
# THE MEASUREMENT: what -std= did each translation unit actually get?
# ---------------------------------------------------------------------------
if(NOT DEFINED COMPILE_COMMANDS)
  # Said out loud. A hand invocation of this script (the header documents one)
  # legitimately has no compile database, but "did not measure" must never
  # render the same as "measured and agreed".
  message("  cxx-standard: NOT MEASURED -- no -DCOMPILE_COMMANDS given")
elseif(NOT EXISTS "${COMPILE_COMMANDS}")
  string(APPEND failures
         "  cxx-standard: the build said it would export a compile database and "
         "there is none at ${COMPILE_COMMANDS}\n")
elseif(std_value STREQUAL "")
  # The presence loop already recorded the missing/empty field; do not also
  # report a measurement against nothing.
  message("  cxx-standard: NOT MEASURED -- the stamp carries no standard to compare against")
else()
  # RAISED_TARGETS arrives comma-separated. A semicolon would have been split
  # by add_test() into one argument per target and only the first would have
  # survived into this variable; see the registration site.
  set(raised_list "")
  if(DEFINED RAISED_TARGETS AND NOT RAISED_TARGETS STREQUAL "")
    string(REPLACE "," ";" raised_list "${RAISED_TARGETS}")
  endif()

  file(STRINGS "${COMPILE_COMMANDS}" cc_lines REGEX "\"command\"")
  set(n_at 0)
  set(n_raised 0)
  set(n_nostd 0)
  set(seen_raised "")
  set(bad "")
  foreach(cl IN LISTS cc_lines)
    if(NOT cl MATCHES "-std=c\\+\\+([0-9a-z]+)")
      math(EXPR n_nostd "${n_nostd} + 1")
      continue()
    endif()
    set(this_std "${CMAKE_MATCH_1}")
    set(this_tgt "(unattributed)")
    if(cl MATCHES "CMakeFiles/([^/]+)\\.dir/")
      set(this_tgt "${CMAKE_MATCH_1}")
    endif()
    # list(FIND) and not `IN_LIST`: this file runs in `cmake -P` script mode,
    # where no project policies are in effect and CMP0057 is therefore unset --
    # so the operator is taken as three bare arguments and the elseif() is a
    # hard error. Measured, on the device tree, which is the only one with a
    # non-empty RAISED_TARGETS to reach it.
    list(FIND raised_list "${this_tgt}" raised_idx)
    if(this_std STREQUAL "${std_value}")
      math(EXPR n_at "${n_at} + 1")
    elseif(NOT raised_idx EQUAL -1)
      math(EXPR n_raised "${n_raised} + 1")
      list(APPEND seen_raised "${this_tgt}@c++${this_std}")
    else()
      list(APPEND bad "${this_tgt} at -std=c++${this_std}")
    endif()
  endforeach()

  math(EXPR n_total "${n_at} + ${n_raised} + ${n_nostd}")
  list(LENGTH bad n_bad)
  math(EXPR n_total "${n_total} + ${n_bad}")

  if(n_total EQUAL 0)
    string(APPEND failures
           "  cxx-standard: ${COMPILE_COMMANDS} recorded no compile lines at all; "
           "the measurement did not happen and must not read as agreement\n")
  elseif(n_at EQUAL 0 AND n_raised EQUAL 0 AND n_bad EQUAL 0)
    string(APPEND failures
           "  cxx-standard: ${n_nostd} compile line(s) and not one -std= among them; "
           "the measurement did not happen and must not read as agreement\n")
  else()
    list(REMOVE_DUPLICATES seen_raised)
    string(REPLACE ";" " " seen_raised_s "${seen_raised}")
    message("  cxx-standard MEASURED: ${n_at} TU(s) at c++${std_value}, "
            "${n_raised} raised, ${n_bad} disagreeing, ${n_nostd} with no -std=")
    if(n_raised GREATER 0)
      message("  cxx-standard raised on: ${seen_raised_s}")
    endif()
  endif()

  if(NOT n_bad EQUAL 0)
    string(REPLACE ";" "\n           " bad_s "${bad}")
    string(APPEND failures
           "  cxx-standard: stamp says c++${std_value}, but ${n_bad} compile "
           "line(s) disagree and belong to no Kokkos-linked target:\n"
           "           ${bad_s}\n")
  endif()
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
