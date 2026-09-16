# ===========================================================================
# check_contraction_flags.cmake — the FMA-contraction flags are load-bearing
# ===========================================================================
#
# WHY THIS EXISTS. The Dekker twoProduct in every backend relies on `a1*b1 - c11`
# being two distinct rounded operations. If the compiler contracts that into a
# single FMA the error term collapses to zero and the error-free transform
# silently stops being error-free. The EFT tests would then keep passing while
# validating a transform the shipped binary does not perform.
#
# So `-ffp-contract=off` on those targets is not a style choice, and losing it
# does not fail anything: it changes what the tests mean. That is the worst
# shape a requirement can have, and it is why this is a permanent gate rather
# than a one-time check performed during a refactor.
#
# WHAT IT READS. A manifest written at configure time from each target's ACTUAL
# `COMPILE_OPTIONS` property -- not from a parallel list of what the helper
# believes it applied, which would pass by agreeing with itself.
#
#   <posture>|<target>|<compile options>
#
# Options are captured unevaluated, so a genex-wrapped
# `$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>` reads as the literal text and
# the substring test still sees the flag. That is deliberate: `file(GENERATE)`
# cannot evaluate COMPILE_LANGUAGE, and a check that cannot run is worse than a
# textual one that can.
#
# Invoked as:
#   cmake -DMANIFEST=<path> -DEXPECT_OFF=<n> -DEXPECT_ON=<n> -P this
# ===========================================================================

if(NOT DEFINED MANIFEST)
  message(FATAL_ERROR "contraction guard: MANIFEST not set")
endif()
if(NOT EXISTS "${MANIFEST}")
  message(FATAL_ERROR "contraction guard: no manifest at ${MANIFEST} -- the "
                      "configure step that writes it did not run")
endif()

file(STRINGS "${MANIFEST}" lines)
set(n_off 0)
set(n_on 0)
set(failures "")

foreach(line IN LISTS lines)
  if(line STREQUAL "")
    continue()
  endif()
  string(REPLACE "|" ";" parts "${line}")
  list(GET parts 0 posture)
  list(GET parts 1 target)
  list(LENGTH parts nparts)
  set(opts "")
  if(nparts GREATER 2)
    list(GET parts 2 opts)
  endif()

  if(posture STREQUAL "off")
    math(EXPR n_off "${n_off} + 1")
    if(NOT opts MATCHES "-ffp-contract=off" AND NOT opts MATCHES "-fp-model=precise")
      string(APPEND failures
        "  ${target}: expected -ffp-contract=off (or -fp-model=precise), got: ${opts}\n")
    endif()
    if(opts MATCHES "-ffp-contract=fast")
      string(APPEND failures
        "  ${target}: has -ffp-contract=fast on a contraction-OFF target\n")
    endif()
  elseif(posture STREQUAL "on")
    math(EXPR n_on "${n_on} + 1")
    if(NOT opts MATCHES "-ffp-contract=fast" AND NOT opts MATCHES "-fp-model=fast")
      string(APPEND failures
        "  ${target}: expected -ffp-contract=fast (or -fp-model=fast), got: ${opts}\n")
    endif()
  else()
    string(APPEND failures "  malformed manifest row: ${line}\n")
  endif()
endforeach()

# Count assertions, so DELETING a guarded target fails too. Without these the
# guard would pass on an empty manifest, which is the classic way a negative
# assertion stops meaning anything.
if(DEFINED EXPECT_OFF AND NOT n_off EQUAL EXPECT_OFF)
  string(APPEND failures
    "  contraction-OFF target count is ${n_off}, expected ${EXPECT_OFF}\n")
endif()
if(DEFINED EXPECT_ON AND NOT n_on EQUAL EXPECT_ON)
  string(APPEND failures
    "  contraction-ON target count is ${n_on}, expected ${EXPECT_ON}\n")
endif()

if(NOT failures STREQUAL "")
  message("contraction flag guard: FAIL")
  message("${failures}")
  message(FATAL_ERROR
    "The FMA-contraction flags are what make the Dekker twoProduct an "
    "error-free transform. Losing one does not fail a test -- it changes what "
    "the EFT tests mean. See tests/check_contraction_flags.cmake.")
endif()

message("contraction flag guard: PASS (${n_off} off, ${n_on} on)")
