# ============================================================================
# tests/CMakeLists.txt — extended-precision test suite (T0.1 harness)
# ============================================================================
# Registered from the top-level CMakeLists.txt via add_subdirectory(tests),
# guarded by option XPMATH_BUILD_TESTS (default ON) and enable_testing().
# (KOKKOS_EP_BUILD_TESTS is still accepted as a deprecated alias -- CORE_PLAN C4.)

# --- Oracle availability: THERE IS NOTHING LEFT TO GATE ---------------------
# This file used to open with a LIBQUADMATH probe. The __float128 oracle reached
# the tests through impl/Kokkos_QuadPrecisionMath.hpp, which exists only when
# Kokkos was built with Kokkos_ENABLE_LIBQUADMATH=ON, so the probe set
# KOKKOS_EP_HAVE_QUADMATH iff LIBQUADMATH was in Kokkos_TPLS. Without it the
# oracle-dependent regions were #ifdef'd out and main() returned 77, which
# SKIP_RETURN_CODE turned into CTest "Skipped".
#
# The gate is gone because its premise is gone: no test calls a `*q` function
# any more. __float128 remains as the CARRIER -- it is a compiler type, and its
# arithmetic is libgcc's -- while the handful of elementary functions the tests
# needed moved to __builtin_ (bit-exact and IEEE-mandated ones) or to MPFR (the
# one genuine reference, exp_reduction_test's ln2). See the q_* block in
# tests/test_utils_host.hpp.
#
# REMOVING IT IS THE POINT, NOT A TIDY-UP. Left in place it would now be
# ACTIVELY HARMFUL: it would keep reporting Skipped on a machine where the test
# can run, and a skip reads as a pass in a summary line. The MI250 install is
# exactly that machine -- its Kokkos has LIBQUADMATH OFF, so every oracle-scored
# target there skipped while the suite reported green.
#
# SKIP_RETURN_CODE 77 goes with it, for the same reason: it existed solely to
# serve this gate, and a live SKIP_RETURN_CODE on every target means any test
# that ever exits 77 for an unrelated reason is silently absolved. The SKIP-77
# uses elsewhere in this file that answer to MPFR rather than to quadmath are
# untouched -- but there are none left to keep, because MPFR is a hard
# FATAL_ERROR requirement further down rather than a runtime condition.
#
# TWO THINGS THE TEXT ABOVE WOULD LET YOU BELIEVE, BOTH WRONG, BOTH MEASURED.
#
# 1. THE SKIP PATH WAS ALREADY DEAD -- it did not skip, it failed to build.
#    "tests still BUILD but runtime-skip" describes an intent, not this tree.
#    tests/test_utils.hpp declared kUlpUnscorable INSIDE the
#    KOKKOS_EP_HAVE_QUADMATH guard and used it from accumulate_ulps() OUTSIDE
#    it, so with the define absent the header did not compile at all. MEASURED
#    on this commit's parent, configured against a Kokkos built with
#    LIBQUADMATH=OFF: 26 errors, every one of them
#    "kUlpUnscorable was not declared in this scope", and all 26 Kokkos-linked
#    test targets failed to compile. The graceful degradation had rotted into a
#    build break wearing a skip's clothes. Deleting the guard is what repairs
#    it: the SAME configuration after this commit builds 48 targets, runs 48,
#    and skips 0.
#
# 2. `ldd` IS THE WRONG CHECK HERE, AND WILL LOOK LIKE A FAILURE.
#    scripts/sweep_accuracy.cpp asserts an empty `ldd | grep -i quadmath`,
#    because it controls its own link line. These targets do not: a Kokkos built
#    with the TPL ships Kokkos::kokkoscore with
#    INTERFACE_LINK_LIBRARIES "quadmath" (KokkosTargets.cmake), so -lquadmath
#    lands on every Kokkos-linked target and leaves a DT_NEEDED
#    libquadmath.so.0 behind with NOT ONE SYMBOL BOUND TO IT. That is Kokkos's
#    property, not this suite's, and it is absent in the other two
#    configurations. The check that means something is the symbol one:
#        nm -D --undefined-only <binary> | grep -E 'q$'   -> empty
#    MEASURED over all 37 tests/ binaries in all three configurations
#    (Kokkos+TPL, Kokkos without the TPL, XPMATH_WITH_KOKKOS=OFF): empty in
#    every one. The src/ demos still carry `*q` undefineds -- that is B3's
#    scope, not this commit's.

# ===========================================================================
# WHERE THE THREE kokkos_ep_add_* HELPERS WENT -- CORE_PLAN C4 step 5
# ===========================================================================
# kokkos_ep_add_test, kokkos_ep_add_eft_test and
# kokkos_ep_add_eft_test_contract_on stood here. They linked Kokkos::kokkos,
# added ${THIRD_PARTY_INCLUDE} (the compat-wrapper directory), and defined
# XPMATH_TEST_HAVE_KOKKOS=1 to open the Kokkos half of test_utils_host.hpp.
# Steps 1, 2 and 3 of this section moved every one of their 26 call sites onto
# the six xpm_add_{host,device}[_eft]_test[_contract_on] helpers below, leaving
# all three defined and called zero times. They are deleted rather than kept
# "in case": an unused helper that still links Kokkos is an invitation to
# reintroduce the dependency this section exists to remove, and a reader would
# have no way to tell it was dead.
#
# ${THIRD_PARTY_INCLUDE} goes with them. It is still SET by the top-level
# CMakeLists.txt, which is where the demos consume it; nothing under tests/
# references it any more.
#
# THE WRAPPER LAYER IS STILL EXERCISED, AND THAT IS WHY THIS IS SAFE.
# third_party/include/ is what lets existing Kokkos code compile unchanged, so
# losing coverage of it would be a real loss rather than a bookkeeping one. It
# is covered by the eight src/demo_*.cpp targets, every one of which includes
# the wrappers and links Kokkos, and every one of which still builds in the
# --arch host configuration. C10 moves those demos to the xpmath-kokkos
# repository, and from then on the wrapper layer is that repository's to test --
# which is the right home for it, because that is where the Kokkos dependency
# will live.
#
# What IS gone as of C4, and is written down here rather than inferred: the
# wrappers no longer get incidental coverage from the test suite. Before this
# section, 26 test targets compiled third_party/include/*_math.hpp as a side
# effect of linking Kokkos. None do now. The wrappers are true aliases
# (`using DoubleDouble = xp::DoubleDouble`), so what that coverage actually
# proved was that the alias header parses -- but it proved it on every build,
# and now only the demos do.
# ===========================================================================

# ===========================================================================
# SIDE TAGGING -- CORE_PLAN C5 step 1
# ===========================================================================
# Two options in the top-level CMakeLists.txt, XPMATH_BUILD_HOST_TARGETS and
# XPMATH_BUILD_DEVICE_TARGETS, select which half of the suite this tree
# contains. Every registered test declares its side through xpm_tag_test, and
# the assertion at the BOTTOM of this file fails the configure when one does
# not. The two facts that assertion establishes are the ones the split depends
# on: nothing is registered without a side, and nothing is registered on both.
#
# WHY A TAG AND NOT JUST AN if(). An if() around a registration makes the
# target disappear from the wrong tree, which is most of what is wanted -- but
# it leaves no record that anyone decided. A test added later inside the wrong
# if(), or outside both, is then indistinguishable from one that was classified.
# The GLOBAL lists below are that record, and the count assertion is what makes
# forgetting to write to them an error rather than a default.
#
# THE ASSERTION MUST PROVE IT LOOKED AT SOMETHING. Every check here is written
# so an empty input FAILS: zero registered tests is a FATAL_ERROR, zero tagged
# tests is a FATAL_ERROR, and the two counts must be EQUAL rather than merely
# consistent. A tagging gate that passes because it found nothing to tag is the
# same failure as `ctest --test-dir` on a directory that does not exist.
#
# WHICH SIDE IS WHICH, for the cases where it is not obvious from the name:
#
#   host    anything that touches the binary128 oracle or MPFR/MPC
#           (sweep_accuracy and everything downstream of it, exp_reduction,
#           trig_reduction, oracle_conv), anything that inspects THIS tree or
#           the source tree with host tools (domains_fresh, device_tu_purity,
#           consumer_package), and every *_test.cpp host half.
#
#   device  every *_test_device.cpp, the tests migrated whole onto
#           tests/device_harness.hpp, and device_harness_test itself.
#
#   both    build_provenance ALONE, and it is the one test whose tag is
#           computed rather than written. It judges the build DIRECTORY, and
#           every tree has one -- including the `--only device` tree C7 and C8
#           configure on a compute node, whose logs are the artifacts that most
#           need to be traceable. It is registered once in every configuration
#           and tagged host when the host side is present, device otherwise.
#           See its block near the bottom of this file.
#
# THE STANDALONE SMOKES ARE HOST, AND THE C5 PLAN TEXT SAYS DEVICE. The plan is
# wrong and this is a deliberate departure. Six of the eight -- dd, dd_complex,
# ff, ff_complex, qf, tf -- stream an xp:: value into a std::ostringstream, and
# operator<< for these types exists only inside
# `#if !defined(XPMATH_ON_DEVICE)` in include/xp/config.hpp. Compiling them
# with a device compiler recreates verbatim the five `no operator "<<"`
# failures A100 job 1000938 hit. What they actually assert is that the core
# headers compile and run against an include path containing ONLY include/ --
# a host-toolchain question -- and scripts/check_standalone_no_kokkos.sh, which
# is their sibling, drives the same sources through plain g++ for the same
# reason.
set_property(GLOBAL PROPERTY XPMATH_HOST_TESTS "")
set_property(GLOBAL PROPERTY XPMATH_DEVICE_TESTS "")

# xpm_tag_test(<host|device> <test-name>...)
# Records the side of one or more ALREADY-REGISTERED ctest tests. Call it at
# the registration site, not in a list somewhere else: the point is that a
# reader adding a target sees the question being asked.
function(xpm_tag_test side)
  if(side STREQUAL "host")
    set(_prop XPMATH_HOST_TESTS)
  elseif(side STREQUAL "device")
    set(_prop XPMATH_DEVICE_TESTS)
  else()
    message(FATAL_ERROR "xpm_tag_test: side must be host or device, got '${side}'")
  endif()
  if(NOT ARGN)
    message(FATAL_ERROR "xpm_tag_test(${side}): no test names given")
  endif()
  foreach(_n IN LISTS ARGN)
    set_property(GLOBAL APPEND PROPERTY ${_prop} ${_n})
  endforeach()
endfunction()

# --- Helper: define + register one test that links NO Kokkos ---------------
# Usage: xpm_add_host_test(<name>)  expects tests/<name>.cpp
#
# CORE_PLAN C4 step 1. Same shape as the deleted kokkos_ep_add_test, minus the
# two lines that made a target Kokkos's: no Kokkos::kokkos link, no
# THIRD_PARTY_INCLUDE, and no XPMATH_TEST_HAVE_KOKKOS. The include path is the
# xp CORE plus this directory, which is what a host-oracle test that names
# xp:: types and tests/test_utils_host.hpp actually needs.
#
# THE SEVEN TARGETS THIS EXISTS FOR were linking Kokkos to call
# Kokkos::initialize/finalize around code that never launched a kernel --
# corpus_test referenced ZERO Kokkos APIs and still could not build without an
# install. That is not a small tidy-up: it is why a Kokkos build failure and a
# library defect were indistinguishable from outside, and why the Kokkos-free
# suite was 24 targets when 33 of them need nothing from Kokkos.
#
# It takes no contraction flags: nothing registered through it is an EFT test.
# If that ever changes, the target must go through an EFT helper instead, or the
# contraction guard below will not know about it.
#
# C5 step 1: it returns without defining anything when this tree is not
# building the host side, and tags what it does define. The early return is
# what keeps the arch's device compiler away from a TU full of binary128.
function(xpm_add_host_test name)
  if(NOT XPMATH_BUILD_HOST_TARGETS)
    return()
  endif()
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE
                             ${CMAKE_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(${name} PRIVATE cxx_std_17)
  add_test(NAME ${name} COMMAND ${name})
  xpm_tag_test(host ${name})
endfunction()

# --- Helper: define + register one DEVICE test that links NO Kokkos --------
# Usage: xpm_add_device_test(<name>)  expects tests/<name>.cpp
#
# CORE_PLAN C4 step 3. Structurally identical to xpm_add_host_test above --
# there is no third include path and no extra link -- because
# tests/device_harness.hpp needs exactly two things: the xp core (it includes
# <xp/config.hpp>, and the tests it serves include <xp/*_math.hpp>) and this
# directory (the harness header and test_utils_device.hpp live here). The
# vendor runtime arrives with the compiler: nvcc force-includes cuda_runtime.h,
# and the harness includes <hip/hip_runtime.h> itself under __HIPCC__.
#
# IT IS A SEPARATE FUNCTION FROM xpm_add_host_test ANYWAY, and deliberately so.
# The two say different things about their targets -- "host oracle, never
# launches" versus "launches a kernel through the C3 harness" -- and the second
# is the one that will grow a CUDA/HIP source-language property, a per-arch
# flag, or a device-lane label. Collapsing them today would mean splitting them
# again at C5, and in between there would be no way to ask CMake which targets
# are the device ones.
#
# C5 step 1 gave it the mirror of xpm_add_host_test's early return, and this is
# the function the comment above predicted would grow a per-arch property: the
# side it tags is the side whose tree gets nvcc_wrapper or hipcc.
function(xpm_add_device_test name)
  if(NOT XPMATH_BUILD_DEVICE_TARGETS)
    return()
  endif()
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE
                             ${CMAKE_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(${name} PRIVATE cxx_std_17)
  add_test(NAME ${name} COMMAND ${name})
  xpm_tag_test(device ${name})
endfunction()

# --- WHY EVERY EFT TEST BELOW IS BUILT WITH CONTRACTION OFF ----------------
# Error-free-transform tests (Layer 1: T1.1 DD, T2.1 FF, T3.1 QF) validate that
# twoSum/twoProd produce an EXACT (result, error) pair. The Dekker twoProduct in
# dd_math.hpp's multiply relies on `a1*b1 - c11` being two distinct rounded
# operations; if the compiler contracts that into a single FMA the error term
# collapses to zero and the EFT silently breaks — and the test would validate a
# transform the shipped binary does not perform. So EFT test targets MUST be
# compiled with contraction OFF.
#
# This is applied PER TARGET (target_compile_options), not as a global build flag:
# only the EFT tests need it; the demos and other test layers should build with
# the project's normal flags. Per-target keeps the requirement local and explicit.
#
# Compiler coverage in the four helpers below matches what this project actually
# builds under (GCC host today; nvcc/hipcc once C5 gives the device targets their
# own tree). IntelLLVM/Intel and Clang lines are included because they are
# one-liners and cost nothing, but are not the tested path.
#
# The ON-posture companions build the SAME source into a SEPARATE executable
# (<name>_contract_on) with contraction forced on, so the Dekker sequence is
# exercised under the posture the OFF helper protects against. Whether that
# produces the same (compiler didn't contract) or different (it did) results, the
# outcome is RECORDED rather than gated — the ON test is a reporter. See the
# dd_fma_guard_test.cpp header.
#
# CORE_PLAN C4 step 5: the Kokkos-linked pair that used to stand here
# (kokkos_ep_add_eft_test / kokkos_ep_add_eft_test_contract_on) is deleted with
# the rest of the kokkos_ep_add_* family — see the block at the top of this file.
# The four Kokkos-free helpers below carry the whole discipline unchanged: same
# flags, same KOKKOS_EP_CONTRACTION_MODE, same <base>_baseline.txt derivation,
# same GLOBAL manifest registration that the contraction guard reads.

# --- Helpers: the two contraction postures, DEVICE lane --------------------
# CORE_PLAN C4 step 3. THE CONTRACTION DISCIPLINE DOES NOT CHANGE WHEN THE
# LAUNCH MECHANISM DOES. An EFT test migrated onto the C3 harness is still a
# test of whether `a1*b1 - c11` stayed two rounded operations, so it is still
# built with contraction OFF, still sets KOKKOS_EP_CONTRACTION_MODE, still gets
# its _contract_on companion where it had one, and is still registered in the
# two GLOBAL manifests the contraction guard reads. The only thing these drop
# relative to the deleted kokkos_ep_add_eft_test is Kokkos.
#
# They delegate the plain part to xpm_add_device_test / to an inline mirror of
# it, exactly as the deleted Kokkos pair delegated to kokkos_ep_add_test, so a
# change to the include paths happens in one place per posture rather than four.
#
# C5 step 1: EVERY ONE OF THESE FOUR NEEDS ITS OWN EARLY RETURN, not just the
# base helper it delegates to. Without one the delegate would return having
# defined nothing and the caller would go on to set COMPILE_OPTIONS on a target
# that does not exist -- a hard configure error, in the tree where the side is
# switched off.
function(xpm_add_device_eft_test name)
  if(NOT XPMATH_BUILD_DEVICE_TARGETS)
    return()
  endif()
  xpm_add_device_test(${name})   # tags device
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>)
  endif()
  # Keyed on CMAKE_CUDA_COMPILER rather than Kokkos_ENABLE_CUDA: these targets
  # do not link Kokkos, so whether Kokkos was built with CUDA says nothing about
  # how they compile. A no-op today (no source here is CUDA-language yet) and
  # the line C5 needs when one is.
  if(CMAKE_CUDA_COMPILER)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()
  target_compile_definitions(${name} PRIVATE KOKKOS_EP_CONTRACTION_MODE=0)
  set_property(GLOBAL APPEND PROPERTY XPMATH_EFT_OFF_TARGETS ${name})
endfunction()

function(xpm_add_device_eft_test_contract_on name)
  if(NOT XPMATH_BUILD_DEVICE_TARGETS)
    return()
  endif()
  set(tgt ${name}_contract_on)
  add_executable(${tgt} ${name}.cpp)
  target_include_directories(${tgt} PRIVATE
                             ${CMAKE_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(${tgt} PRIVATE cxx_std_17)
  add_test(NAME ${tgt} COMMAND ${tgt})
  # Tagged here rather than by a delegate: this helper builds its own target,
  # so no base helper ever sees ${tgt}.
  xpm_tag_test(device ${tgt})

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=fast>)
  endif()
  if(CMAKE_CUDA_COMPILER)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  endif()

  # Same baseline-path derivation as the Kokkos helper, so a migrated test keeps
  # reading the same <base>_baseline.txt it read before and drift detection is
  # continuous across the migration rather than restarting from "no file".
  string(REGEX REPLACE "_test$" "" _base "${name}")
  target_compile_definitions(${tgt} PRIVATE
    KOKKOS_EP_CONTRACTION_MODE=1
    KOKKOS_EP_BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/${_base}_baseline.txt")
  set_property(GLOBAL APPEND PROPERTY XPMATH_EFT_ON_TARGETS ${tgt})
endfunction()

# --- Helpers: the two contraction postures for a HOST-ONLY EFT test --------
# CORE_PLAN C4 step 2. When a mixed TU splits, its HOST half keeps the oracle
# and stops launching -- but it does NOT stop being an EFT test. dd_eft_test's
# host half still calls the Dekker sequence `a1*b1 - c11` directly, and a
# compiler that contracts it there collapses the error term exactly as it would
# in a kernel. So the host half needs contraction OFF just as much as the device
# half does, and its _contract_on companion has to survive the split too.
#
# Neither alternative fits: the Kokkos EFT pair linked Kokkos (the whole point
# of the split is that these halves do not, and C4 step 5 has since deleted it),
# and xpm_add_device_eft_test
# is the device-lane function that C5 will grow a source-language property on.
# xpm_add_host_test cannot be widened either -- it documents, correctly, that it
# takes no contraction flags, which is what lets the contraction guard below
# trust that nothing registered through it is an EFT test.
#
# These two therefore delegate to xpm_add_host_test / mirror it, and are
# otherwise byte-for-byte the device pair's logic: same flags, same
# KOKKOS_EP_CONTRACTION_MODE, same <base>_baseline.txt derivation (so a split
# host half keeps reading the baseline it read before the split), same GLOBAL
# manifest registration.
function(xpm_add_host_eft_test name)
  if(NOT XPMATH_BUILD_HOST_TARGETS)
    return()
  endif()
  xpm_add_host_test(${name})   # tags host
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>)
  endif()
  target_compile_definitions(${name} PRIVATE KOKKOS_EP_CONTRACTION_MODE=0)
  set_property(GLOBAL APPEND PROPERTY XPMATH_EFT_OFF_TARGETS ${name})
endfunction()

function(xpm_add_host_eft_test_contract_on name)
  if(NOT XPMATH_BUILD_HOST_TARGETS)
    return()
  endif()
  set(tgt ${name}_contract_on)
  add_executable(${tgt} ${name}.cpp)
  target_include_directories(${tgt} PRIVATE
                             ${CMAKE_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(${tgt} PRIVATE cxx_std_17)
  add_test(NAME ${tgt} COMMAND ${tgt})
  xpm_tag_test(host ${tgt})

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=fast>)
  endif()

  string(REGEX REPLACE "_test$" "" _base "${name}")
  target_compile_definitions(${tgt} PRIVATE
    KOKKOS_EP_CONTRACTION_MODE=1
    KOKKOS_EP_BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/${_base}_baseline.txt")
  set_property(GLOBAL APPEND PROPERTY XPMATH_EFT_ON_TARGETS ${tgt})
endfunction()

# --- HOST-ORACLE TESTS THAT LINK NO KOKKOS ---------------------------------
# CORE_PLAN C4 step 1. These seven were registered inside the Kokkos block
# below until C4, and every one of them was there for the same non-reason: a
# Kokkos::initialize/finalize pair bracketing host-side code that launches
# nothing. corpus_test did not even have that -- it referenced zero Kokkos APIs
# and linked the library anyway.
#
# Three of them (dd_e2e, ff_cancellation, qf_cancellation) additionally reached
# their types through third_party/include/<x>_math.hpp, whose only material
# content beyond `using T = xp::T` is #include <Kokkos_Core.hpp>. They now
# include <xp/..._math.hpp> directly and alias `xp` instead of
# Kokkos::Experimental. The types are IDENTICAL -- the wrappers are true
# aliases, not distinct wrapper types -- so this changes what is compiled, not
# what is measured.
#
# WHAT THIS COSTS: the wrapper layer loses these seven as incidental coverage.
# It is still exercised, by the demos and by the Kokkos-linked tests below, and
# after C10 by the xpmath-kokkos repository.
xpm_add_host_test(hello_test)
xpm_add_host_test(corpus_test)
# hello_test WAS the whole of the T0.1 smoke and is now only its host half.
# Chunk A took it off Kokkos but left it MIXED -- binary128 via
# test_utils_host.hpp plus a live xpt::parallel_for_n -- and nothing noticed
# until chunk E's mirror check in scripts/check_device_tu_purity.sh went red on
# it. Its device half is registered in the device block below. See step 4 and
# the two files' headers.
xpm_add_host_test(dd_e2e_test)
xpm_add_host_test(ff_cancellation_test)
xpm_add_host_test(qf_cancellation_test)
xpm_add_host_test(tf_cancellation_test)
xpm_add_host_test(tf_property_test)

# --- HOST HALVES OF THE C4 STEP-2 SPLITS (DD and FF) -----------------------
# CORE_PLAN C4 step 2, chunk C. Each of these four kept its target name, its
# tests/test_utils_host.hpp include and its oracle scoring; what it lost is the
# kernel launch, which moved to a sibling *_test_device.cpp registered in the
# device block below. They link no Kokkos for the same reason the seven above
# do not: with the launch gone, there is nothing left in them that Kokkos was
# providing.
#
# dd_property_test and ff_property_test use the plain host helper -- algebraic
# identities are FMA-contraction agnostic, exactly as their pre-split
# registrations said.
xpm_add_host_test(dd_property_test)
xpm_add_host_test(ff_property_test)
# dd_eft_test and dd_fma_guard_test use the EFT pair: both call the Dekker
# sequence `a1*b1 - c11` directly on the host, so contraction OFF is as load-
# bearing here as it was before the split. dd_fma_guard_test keeps its
# _contract_on companion and, via the <base>_baseline.txt derivation, keeps
# reading tests/dd_fma_guard_baseline.txt -- drift detection is continuous
# across the split rather than restarting from "no file".
xpm_add_host_eft_test(dd_eft_test)
xpm_add_host_eft_test(dd_fma_guard_test)              # -> dd_fma_guard_test              (OFF, gates)
xpm_add_host_eft_test_contract_on(dd_fma_guard_test)  # -> dd_fma_guard_test_contract_on  (ON, reports)

# --- HOST HALVES OF THE C4 STEP-2 SPLITS (QF) ------------------------------
# CORE_PLAN C4 step 2, chunk D. Same shape as chunk C above: each kept its
# target name, its tests/test_utils_host.hpp include and its oracle scoring, and
# lost only the kernel launch to a sibling *_test_device.cpp in the device block.
#
# qf_property_test uses the plain helper (algebraic identities are contraction
# agnostic) and qf_nonoverlap_test likewise -- a per-word add/compare has nothing
# to contract, which is what its pre-split plain-helper registration said.
xpm_add_host_test(qf_property_test)
xpm_add_host_test(qf_nonoverlap_test)
# qf_eft_test is the NINTH mixed TU -- the one C4's eight-file list does not name
# and chunk B could only half-migrate, because Test C's wide-spread renorm_4
# truncation check is a genuine binary128 host assertion. It is split now: this
# half keeps Tests A-D including that check, and moves from
# xpm_add_device_eft_test to the HOST EFT helper. The contraction-OFF posture
# travels with it -- Tests A/B call the Dekker sequence `a1*b1 - c11` directly on
# the host, so the flag is as load-bearing here as before the split.
xpm_add_host_eft_test(qf_eft_test)

# --- DEVICE TESTS THAT LINK NO KOKKOS --------------------------------------
# CORE_PLAN C4 step 3. These six sources were already device-only in substance
# -- every one of them launches a kernel and checks the result on host, and none
# of them scores anything against the oracle. What made them Kokkos's was the
# LAUNCH: Kokkos::initialize, a View pair per array, a KOKKOS_LAMBDA and a
# fence. All of that is now tests/device_harness.hpp (C3), which does the same
# job in ~40 lines of vendor API and works identically on CUDA, HIP and a serial
# host loop.
#
# WHAT THIS BUYS. Until now a GPU could not be asked "does the standalone core
# run here" without a working Kokkos install on it, so a Kokkos build failure
# and an xp defect were the same observation from outside. S8c is the measured
# form of that: on MI250X, 39 of 49 targets did not build and not one of the 8
# that ran compiled a line of gfx90a code. These eight targets are the first
# device evidence that does not route through Kokkos.
#
# WHAT IT DOES NOT COST. The functors are the same expressions on the same
# inputs with the same seeds, and the harness's serial backend keeps two real
# allocations with a memcpy between them, so a forgotten to_device() fails on
# the host build too. Each migrated main() additionally asserts
# xpt::last_error() == 0, which the Kokkos versions had no analogue for -- a
# launch that never ran used to be indistinguishable from one that ran clean.
#
# ALL OF THEM ARE IN scripts/check_device_tu_purity.sh's FILES list. Five were
# when chunk B wrote this; qf_eft_test was the exception, and chunk D removed the
# exception by splitting it -- see its registration below.
xpm_add_device_test(dd_invariant_test)
xpm_add_device_test(ff_invariant_test)
# CORE_PLAN C4 step 4 (chunk E): the device half of hello_test, the NINTH mixed
# TU and the one the plan's list does not name -- because it was on the "links
# Kokkos and barely uses it" list instead, and lifting the launch onto the C3
# harness (chunk A) left it mixed rather than making it host. The mirror check
# found it. Plain device helper: the op is the identity, which has nothing to
# contract. It asserts BIT-EQUALITY of output to input, which is strictly
# stronger than the digits-vs-oracle assertion it replaces, and needs no oracle.
xpm_add_device_test(hello_test_device)
# ff/qf/tf EFT + the TF guard keep the contraction posture they had; see the
# xpm_add_device_eft_test comment. tf_eft_test and tf_fma_guard_test keep their
# _contract_on companions, which are the only two ON-posture targets outside the
# Kokkos block.
xpm_add_device_eft_test(ff_eft_test)
# qf_eft_test WAS registered here, still including test_utils_host.hpp on
# purpose, because Test C's test_renorm_4_wide is a genuine binary128 host
# assertion -- making it a NINTH mixed TU that the C4 step-2 list does not name.
# Chunk D split it: the host half moved to xpm_add_host_eft_test above with Test
# C intact, and the device half is qf_eft_test_device, registered below and in
# the purity gate's FILES list.
xpm_add_device_eft_test(tf_eft_test)                    # -> tf_eft_test              (OFF, gates)
xpm_add_device_eft_test_contract_on(tf_eft_test)        # -> tf_eft_test_contract_on  (ON, reports)
xpm_add_device_eft_test(tf_fma_guard_test)              # -> tf_fma_guard_test              (OFF, gates)
xpm_add_device_eft_test_contract_on(tf_fma_guard_test)  # -> tf_fma_guard_test_contract_on  (ON, reports)

# --- DEVICE HALVES OF THE C4 STEP-2 SPLITS (DD and FF) ---------------------
# CORE_PLAN C4 step 2, chunk C. Four new TUs, each the device pass of a mixed
# test whose host half stayed behind with the oracle (registered in the host
# block above). Every one of them is in scripts/check_device_tu_purity.sh's
# FILES list -- that is the gate that keeps "no 128-bit type in a device TU"
# true rather than merely intended.
#
# The property pair uses the plain device helper (identities are contraction
# agnostic). The EFT pair uses the contraction-OFF helper, and
# dd_fma_guard_test_device additionally gets its _contract_on companion: the
# question that test asks -- "did the compiler fuse the Dekker sequence" -- is
# meaningless without both postures, and it is a per-compilation question, so
# the device half needs its own pair rather than inheriting the host half's.
xpm_add_device_test(dd_property_test_device)
xpm_add_device_test(ff_property_test_device)
xpm_add_device_eft_test(dd_eft_test_device)
xpm_add_device_eft_test(dd_fma_guard_test_device)              # -> ...                    (OFF, gates)
xpm_add_device_eft_test_contract_on(dd_fma_guard_test_device)  # -> ..._contract_on        (ON, reports)
# ff_fma_guard_test is here WHOLE, not split, and that is a deliberate deviation
# from C4 step 2's eight-TU list -- which was drawn up before C2 split
# test_utils.hpp. Post-C2 this file includes test_utils_device.hpp only, and its
# oracle is an exact FP64 product (48 <= 53 bits), not a 128-bit one: the only
# occurrences of the wide type's name in it are prose, which the preprocessor
# strips, so it passes the purity gate as it stands. Splitting it would duplicate
# ~250 lines of apparatus and make four targets where two suffice, to separate a
# host oracle that is not the kind of oracle the split exists to isolate. Chunk
# B left its sibling ff_eft_test whole for the same reason. What changed here is
# only the launch surface: Kokkos -> tests/device_harness.hpp.
xpm_add_device_eft_test(ff_fma_guard_test)              # -> ff_fma_guard_test              (OFF, gates)
xpm_add_device_eft_test_contract_on(ff_fma_guard_test)  # -> ff_fma_guard_test_contract_on  (ON, reports)

# --- DEVICE HALVES OF THE C4 STEP-2 SPLITS (QF) ----------------------------
# CORE_PLAN C4 step 2, chunk D. Three new TUs plus one whole migration, mirroring
# chunk C's block above. All four are in scripts/check_device_tu_purity.sh's
# FILES list.
#
# The property and non-overlap halves use the plain device helper -- neither the
# bit-exact identities nor a per-word add/compare has anything to contract, which
# is what their pre-split plain registrations said. qf_eft_test_device uses the
# contraction-OFF helper: it calls qf_two_prod's Dekker sequence in a kernel, so
# the flag is exactly as load-bearing on this half as on the host one.
xpm_add_device_test(qf_property_test_device)
xpm_add_device_test(qf_nonoverlap_test_device)
xpm_add_device_eft_test(qf_eft_test_device)
# qf_fma_guard_test is here WHOLE, not split -- the same deliberate deviation
# from C4 step 2's eight-TU list that chunk C made for ff_fma_guard_test, and for
# the same reason: that list predates C2, and post-C2 this file includes
# test_utils_device.hpp only and scores against an exact FP64 product (48 <= 53
# bits), not a 128-bit one. Splitting it would duplicate ~260 lines of apparatus
# and make four targets where two suffice. MEASURED before it joined the purity
# gate: its prose mentions of the wide type are stripped by the preprocessor, but
# one printf STRING LITERAL was not, and was reworded. What changed here is the
# launch surface (Kokkos -> tests/device_harness.hpp) and one more thing: the ON
# reporter's exit code now folds in ep_exit_code(), because the migration puts a
# KOKKOS_EP_ASSERT on the path BOTH postures take and the old expression would
# have printed ASSERT FAILED and still exited 0.
xpm_add_device_eft_test(qf_fma_guard_test)              # -> qf_fma_guard_test              (OFF, gates)
xpm_add_device_eft_test_contract_on(qf_fma_guard_test)  # -> qf_fma_guard_test_contract_on  (ON, reports)

# --- THE DEVICE PRODUCER FOR THE ACCURACY SWEEP (CORE_PLAN C6, chunk A) -----
# scripts/sweep_device.cpp evaluates every backend x op x grid point through
# tests/device_harness.hpp and writes the RAW RESULT LIMBS as hex bit patterns.
# It is the second PRODUCER in C6's one-scorer/two-producer design; it issues no
# verdict and contains no ulp arithmetic, because docs/CORRECTNESS.md permits
# exactly one scorer and that is sweep_accuracy.
#
# IT IS REGISTERED BY HAND rather than through xpm_add_device_test, for two
# reasons the helper cannot express:
#   * the source is scripts/sweep_device.cpp, not tests/<name>.cpp -- the same
#     arrangement sweep_accuracy has, and for the same reason it was given:
#     a target built by the build system can never be the stale hand-compiled
#     binary somebody left in the source tree. The binary lands at
#     <build>/tests/sweep_device. (CORE_PLAN's C6 text says
#     <build>/scripts/sweep_device; there is no scripts/ directory in the build
#     tree, because `tests` is the only add_subdirectory() in the project.)
#   * it needs ${CMAKE_SOURCE_DIR}/tests on the include path from a source that
#     does not live there, to reach device_harness.hpp.
#
# NO add_test(). C6 step 5 is where the device GATES are specified, and that is
# a later section's work; a placeholder test here would move the pinned suite
# counts (device 24, union 63, in three places in .github/workflows/ci.yml)
# once now and again then. The counts are therefore UNCHANGED by this chunk,
# and the tag-completeness assertion at the bottom of this file is satisfied
# trivially -- it constrains registered TESTS, and this registers none.
#
# The grid path is compiled in rather than defaulted in the source, so the
# binary finds the committed manifest from any working directory and a reader
# can see which file that is without grepping a #define.
if(XPMATH_BUILD_DEVICE_TARGETS)
  add_executable(sweep_device ${CMAKE_SOURCE_DIR}/scripts/sweep_device.cpp)
  target_include_directories(sweep_device PRIVATE
                             ${CMAKE_SOURCE_DIR}/include
                             ${CMAKE_SOURCE_DIR}/tests)
  target_compile_features(sweep_device PRIVATE cxx_std_17)
  target_compile_definitions(sweep_device PRIVATE
    XPMATH_SWEEP_GRID_DEFAULT="${CMAKE_SOURCE_DIR}/validation/sweep/sweep_grid.csv"
    # The ~40 domain diagnostics in the numeric headers ("DDEXP: overflow") are
    # compiled OUT here. This is not cosmetic. The sweep deliberately evaluates
    # every op at overflow, at poles and outside its domain, so the diagnostics
    # fire in bulk: MEASURED on this tree, a diagnostics-ON host run emitted
    # ~269 KB of them to stdout. On a GPU that is a device-side printf per
    # offending thread, which costs registers in every kernel, serialises on the
    # host-side buffer and can be DROPPED under load -- a producer whose
    # correctness story is "the limbs transfer losslessly" should not also be
    # pushing hundreds of thousands of format strings through that path.
    #
    # It changes no result, and that is MEASURED rather than taken from
    # include/xp/config.hpp's claim that every diagnostic site returns the same
    # value either way: the two builds' 436,080-row outputs were compared with
    # `cmp` and are BYTE-IDENTICAL.
    XPMATH_ENABLE_DIAGNOSTICS=0)
endif()

# --- WHERE THE KOKKOS-LINKED TESTS WENT ------------------------------------
# There is no if(XPMATH_WITH_KOKKOS) here any more. It used to wrap everything
# that linked Kokkos::kokkos to exercise the compat wrappers in
# third_party/include/; C4 steps 1, 2 and 3 moved the last of it out, chunk D
# left the shell standing over an empty body, and step 5 (chunk E) removed the
# shell. What follows is the per-test commentary ONLY -- no registrations --
# kept where a reader looking for a QF or TF target will find the note saying
# where it went.
#
# That emptiness is the point of C4: with no test conditional on
# XPMATH_WITH_KOKKOS, the with-Kokkos and no-Kokkos ctest counts are EQUAL, and a
# GPU can be asked "does the standalone core run here" without a working Kokkos
# install on it.
# T1.2 dd_invariant_test and T2.2 ff_invariant_test moved to the Kokkos-free
# device block above in C4 step 3. Both were oracle-independent already; their
# device tripwires now launch through tests/device_harness.hpp.
# T1.3 property/identity tests: Group A bit-exact (no oracle), Group B/Test C use
# the __float128 oracle.
# (dd_property_test split in C4 step 2: the host half is registered in the
# Kokkos-free host block above, its device pass is dd_property_test_device.)
# T1.4 differential accuracy: per-op digits-of-accuracy against the __float128
# oracle. Plain helper — NOT an EFT test, so no contraction flags.
# T1.6 end-to-end cancellation kernels (dd_e2e_test) moved to the Kokkos-free
# block above in C4.
# (dd_eft_test split in C4 step 2 as well: host half above via
# xpm_add_host_eft_test, device half as dd_eft_test_device. The contraction-OFF
# posture travelled with BOTH halves.)
# T2.1 EFT unit tests for FF: twoSum + Dekker twoProduct, FP64 oracle (exact —
# 25-bit sum / 48-bit product fit in FP64's 53-bit mantissa, so it needs no type
# wider than a machine word). Same contraction-OFF posture as dd_eft_test (Dekker
# twoProduct is the contraction hazard). Contraction-ON reporter mirror is T2.5.
# (ff_eft_test moved to the Kokkos-free device block above in C4 step 3; the
# contraction-OFF posture travelled with it, via xpm_add_device_eft_test.)
# T2.3 property/identity tests for FF: Group A bit-exact (no oracle), Group B/Test C
# use the __float128 oracle. FF analogue of dd_property_test.
# (ff_property_test split in C4 step 2, exactly as its DD analogue: host half in
# the Kokkos-free host block above, device pass as ff_property_test_device.)
# T2.4 differential accuracy for FF: per-op digits-of-accuracy against the
# __float128 oracle. FF analogue of
# Plain helper — NOT an EFT test, so no contraction flags.
# T2.5 end-to-end cancellation kernels for FF: __float128-oracle-scored, host-side.
# FF analogue of dd_e2e_test (T1.6). Plain helper — NOT an EFT test, so no
# contraction flags. NOTE ON CONTRACTION: unlike T1.6's dd_e2e_test (which also
# used the plain helper), K1's naive form sqrt(x²+1)-x contains a mul-then-sub
# adjacency an FMA *could* contract. That is intentionally left to the project's
# normal flags here for two reasons: (1) K1_naive is REPORTED, never gated, so a
# contraction-induced shift in its (already ~0-digit) result cannot flip a
# pass/fail; (2) the GATED path is K1_stable = 1/(sqrt(x²+1)+x), which has no
# subtractive-cancellation adjacency to contract. So this test mirrors
# dd_e2e_test's registration exactly (no EFT helper) — the contraction hazard the
# EFT helper guards (Dekker twoProduct's a1*b1-c11) is not on any gated path here.
# (ff_cancellation_test itself moved to the Kokkos-free block above in C4; the
# contraction reasoning is unchanged and xpm_add_host_test applies no flags
# either.)
# T1.5 FMA-contraction guard: same source, two postures (see dd_fma_guard_test.cpp).
# (dd_fma_guard_test split in C4 step 2. BOTH postures exist on BOTH halves now:
# the host pair is registered above through xpm_add_host_eft_test[_contract_on],
# the device pair is dd_fma_guard_test_device[_contract_on]. Four targets where
# there were two, because "did the compiler fuse this" is a question about one
# compilation and the two halves are compiled separately.)
# T2.5 FMA-contraction guard for FF: FF analogue of T1.5, same source/two-postures
# dual-registration pattern (see ff_fma_guard_test.cpp). Unlike the DD guard, the
# FF source uses an FP64 oracle (exact, no type wider than a machine word), so it
# runs on any target rather than x86_64 only — and that is also why C4 migrated it
# WHOLE onto the C3 harness instead of splitting it. Both postures are registered
# in the Kokkos-free device block above; the reasoning is written there.
# T3.1 EFT unit tests for QF: qf_two_sum / qf_quick_two_sum / qf_two_prod / qf_two_sqr
# (FP64 oracle, exact — 25-bit sum / 48-bit product fit in FP64's 53-bit mantissa)
# PLUS renorm_4 (len 5->4) + renorm (len 4->4), the QF-unique renormalization
# primitives (Priest non-overlap invariant + exact FP64 value-preservation on
# bounded-spread inputs, plus a __float128 wide-spread truncation check).
# Calls the SHIPPED qf_math.hpp free functions directly (no mirror-and-comment —
# unlike ff_eft_test, QF exposes its primitives as free functions). Same
# contraction-OFF posture as ff_eft_test (Dekker twoProduct is the contraction
# hazard). Contraction-ON reporter mirror is T3.5.
# (qf_eft_test moved to the Kokkos-free device block above in C4 step 3, and was
# SPLIT there in C4 step 2 chunk D: host half via xpm_add_host_eft_test with the
# wide-spread renorm_4 check intact, device half as qf_eft_test_device. Both
# postures of the split are Kokkos-free; the purity-gate exception it used to
# carry is gone.)

# T3.2 non-overlap invariant checks for QF: the Priest length-4 invariant
# |f_{i+1}| <= 1/2 ulp(f_i) (i=0,1,2) on the OUTPUT of every QF op that returns a
# QuadFloat (arithmetic, sqrt, transcendentals, joint sincos/sinhcosh, scalar
# multiply_scalar/mul_pwr2, fma, pow_int, round family, utilities). Oracle-
# independent (mathematical 1/2-ulp form via frexp — no reference value), so it
# runs unconditionally; __float128 only enriches inputs to ~96-bit width. Plain
# helper (NOT the EFT helper): a per-word add/compare has nothing to contract,
# so no -ffp-contract=off posture is required. QF analogue of
# ff_invariant_test (T2.2); shares half_ulp / nonoverlap_holds with qf_eft_test.
# (SPLIT in C4 step 2, chunk D. Test B, the device tripwire, is now
# qf_nonoverlap_test_device; Tests A and C stayed with the host half, which keeps
# the target name. Both are registered in the Kokkos-free blocks above. The
# device half's inputs are enriched through a DoubleDouble carrier rather than
# binary128 -- same construction, a type the device pass can actually compile;
# the reason is written in that file's header.)

# T3.3 property/identity tests for QF: Group A bit-exact identities (additive
# inverse/zero, multiply by 0/+-1, negation, abs sign branches, add commutativity
# — bit-exact even on WIDE operands, a QF strengthening over FF/DD — and the
# mul_pwr2 power-of-2 round-trip) + Group B tolerance identities (demoted mul
# commutativity, sqrt/square, exp/log round-trips, Pythagorean, addition formulas,
# hyperbolic, inverse pairs, pow/hypot, and the small-arg exp(a+eps) sensitivity —
# the T2.3 B4 pattern, which does NOT recur in QF). __float128-ordered corpus;
# QF quad-word tolerance = ulp of U=2^-96 (10 ulp = 27.90 digits default; 30 ulp
# for exp-denormal-tail-limited round-trips per PORT_NOTES_QF §10). Group A runs
# unconditionally; Group B / Test C score against the __float128 oracle. Plain
# helper (NOT the EFT helper): algebraic identities are FMA-contraction agnostic.
# QF analogue of ff_property_test (T2.3).
# (SPLIT in C4 step 2, chunk D, exactly as its DD/FF analogues were in chunk C:
# host half keeps the target name and the Group B / Test C oracle scoring, device
# pass is qf_property_test_device. The device half GREW -- it runs all twelve
# Group A identities, where the old device block ran three.)

# T3.4 differential accuracy for QF: per-op digits-of-accuracy against the
# __float128 oracle. QF analogue of the retired per-backend accuracy tests
# (T1.4 / T2.4). Scores every QF op returning a
# QuadFloat with a binary128 counterpart across narrow(Route-A) + broad(~96-bit) +
# corpus passes; gates on the MEAN against the QF quad-word absolute ulp-of-U floor
# (10 ulp = 27.90 digits default; 30 ulp = 27.42 for the exp-family output-denormal
# tail, PORT_NOTES_QF §10), with low mins for §5-registry ops reported as
# EXPECTED-MIN-DROP. Plain helper — NOT an EFT test, so no contraction flags
# (plain helper; no contraction flags).

# T3.5 FMA-contraction guard for QF: QF analogue of T2.5's ff_fma_guard_test, same
# source/two-postures dual-registration pattern (see qf_fma_guard_test.cpp). Reuses
# the SAME helpers as the DD/FF guards — the only per-variant knobs are the compile
# definitions the helpers set; the ON helper derives qf_fma_guard_baseline.txt from
# the target name. Unlike FF (which mirrors ff_math.hpp's embedded Dekker), this test
# calls the SHIPPED qf_math.hpp primitives (qf_two_prod / qf_two_sqr) directly, so it
# characterizes GCC's contraction of the shipped source itself (cf. T3.1 qf_eft_test,
# which established the same direct-call posture under contraction-OFF). Also guards
# qf_two_sqr (a second Dekker sequence, no FF analogue). FP64 oracle (exact, no
# type wider than a machine word), so it runs on any target, not x86_64 only --
# and that is also why C4 chunk D migrated it WHOLE onto the C3 harness instead
# of splitting it, exactly as chunk C did with the FF guard. Both postures are
# registered in the Kokkos-free device block above; the reasoning is written
# there and in the file's own header.

# T3.6 end-to-end cancellation kernels for QF: __float128-oracle-scored, host-side.
# QF analogue of dd_e2e_test (T1.6) / ff_cancellation_test (T2.6). Plain helper —
# NOT an EFT test, so no contraction flags. Same K1 rationale as T2.6: the GATED
# path is K1_stable = 1/(sqrt(x²+1)+x), which has no subtractive-cancellation
# adjacency to contract; K1's naive form sqrt(x²+1)-x has a mul-then-sub adjacency
# an FMA *could* contract, but K1_naive is REPORTED, never gated, so a
# contraction-induced shift in its (already low-digit) result cannot flip a
# pass/fail. The Dekker-twoProduct contraction hazard the EFT helper guards is on no
# gated path here (it is covered by qf_fma_guard_test, T3.5). So this test mirrors
# dd_e2e_test / ff_cancellation_test registration exactly (no EFT helper).
# (qf_cancellation_test itself moved to the Kokkos-free block above in C4.)

# S10 Phase 2: TF (TripleFloat, 3×FP32) test suite. TF analogue of T3.1-T3.6 (QF),
# T2.1-T2.6 (FF), T1.1-T1.6 (DD). Same registration pattern as QF: EFT test with
# contraction-OFF + contraction-ON dual; property/accuracy/cancellation use plain
# helper (no contraction flags).
# tf_eft_test + tf_eft_test_contract_on and tf_fma_guard_test +
# tf_fma_guard_test_contract_on moved to the Kokkos-free device block above in
# C4 step 3, both postures intact. tf_property_test (algebraic identities,
# Group A/B/C) and tf_cancellation_test (e2e cancellation kernels K1-K4) moved
# to the Kokkos-free host block above in C4 step 1.
# With that, TF has no Kokkos-linked target left at all.
# =================== END OF THE FORMER KOKKOS-LINKED BLOCK ==================

# ---------------------------------------------------------------------------
# THE CONTRACTION FLAG GUARD
# ---------------------------------------------------------------------------
# OUTSIDE the XPMATH_WITH_KOKKOS guard as of CORE_PLAN C4 step 6. Both targets
# are script-driven -- one runs a .cmake checker, the other a shell script --
# and neither compiles or links anything, so being Kokkos-CONDITIONAL was never
# a property of what they check. It was, however, the reason the two suite sizes
# differed by 28 while only 26 targets actually linked Kokkos.
#
# -ffp-contract=off on the EFT targets is what keeps the Dekker twoProduct an
# error-free transform. Drop it and nothing fails -- the error term collapses
# to zero and the EFT tests go on passing while validating a transform the
# binary no longer performs. A requirement that fails silently needs a gate,
# not a code comment, and especially so while the build is being restructured
# around it.
#
# The manifest is built from each target's ACTUAL COMPILE_OPTIONS property, read
# here where every target is already defined. It is deliberately not built from
# a list of what the helpers intended to apply: that would pass by agreeing with
# itself. Options are captured unevaluated -- file(GENERATE) cannot evaluate
# COMPILE_LANGUAGE -- so the checker does a substring test on the genex text.
get_property(_xp_eft_off GLOBAL PROPERTY XPMATH_EFT_OFF_TARGETS)
get_property(_xp_eft_on  GLOBAL PROPERTY XPMATH_EFT_ON_TARGETS)
set(_xp_manifest "")
foreach(_t IN LISTS _xp_eft_off)
  get_target_property(_o ${_t} COMPILE_OPTIONS)
  if(NOT _o)
    set(_o "")
  endif()
  string(REPLACE ";" " " _o "${_o}")
  string(APPEND _xp_manifest "off|${_t}|${_o}\n")
endforeach()
foreach(_t IN LISTS _xp_eft_on)
  get_target_property(_o ${_t} COMPILE_OPTIONS)
  if(NOT _o)
    set(_o "")
  endif()
  string(REPLACE ";" " " _o "${_o}")
  string(APPEND _xp_manifest "on|${_t}|${_o}\n")
endforeach()
list(LENGTH _xp_eft_off _xp_n_off)
list(LENGTH _xp_eft_on  _xp_n_on)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/contraction_manifest.txt" "${_xp_manifest}")

# THE EXPECTED COUNTS ARE COMMITTED CONSTANTS, NOT ${_xp_n_off}/${_xp_n_on}.
# Deriving them from the same list that builds the manifest would make the
# check circular: delete an EFT-helper call and both the manifest
# and the expectation shrink together, so the guard passes while a target has
# silently lost its contraction flag by ceasing to exist. Same contract as
# ci.yml's `expected=49` -- adding or removing a guarded target is a deliberate
# act that updates a number here.
#
# THEY WERE BRIEFLY NOT PER-CONFIGURATION, AND ARE AGAIN -- ALONG A DIFFERENT
# AXIS. The C4 version of this comment recorded the collapse of the old
# Kokkos/no-Kokkos split ("Chunk D did that and the if() is gone"). C5 step 1
# reintroduces a split, but on host-vs-device rather than Kokkos-vs-not, and the
# manifest above is built from the targets that EXIST in this tree -- so in a
# host-only or device-only tree it legitimately has fewer rows. A single 11/6
# expectation would make the guard fail on a correct build.
#
# THE FIX IS TWO COMMITTED PAIRS PLUS ARITHMETIC, NOT A DERIVED NUMBER. The
# per-side constants below are still hand-written, so deleting an EFT-helper
# call still makes the guard red; what is computed is only the SUM over the
# sides this tree builds. 3+8 = 11 and 1+5 = 6 reproduce C4's figures exactly
# in the default both-ON configuration, which is the bare `cmake -B build` path
# and what CI runs.
#
# The predicted figure at C4 was 10/6; the measured one was 11/6, because the
# prediction counted qf_fma_guard_test moving but not qf_eft_test SPLITTING --
# the same arithmetic slip an earlier revision made about chunk C, in the same
# direction: splitting a guarded TU adds a guarded target, it does not conserve
# the count.
#
# THE TOTALS GREW, from 10/6 to 11/6, and no target quietly acquired a flag. The
# one new OFF entry is qf_eft_test_device, the device half of the ninth mixed TU.
# Both halves call qf_two_prod's Dekker sequence -- the host one on the host, the
# device one in a kernel -- so both are still EFT tests and both must stay
# guarded; a split that guarded only one half would be exactly the silent failure
# this gate exists to catch. qf_fma_guard_test was NOT split (it migrated whole;
# see its registration) so it still contributes 1 OFF + 1 ON, from the
# Kokkos-free branch now.
#
# The 11 OFF: ff_eft, tf_eft, tf_fma_guard (step 3) + dd_eft_test,
# dd_eft_test_device, dd_fma_guard_test, dd_fma_guard_test_device,
# ff_fma_guard_test (chunk C) + qf_eft_test, qf_eft_test_device,
# qf_fma_guard_test (chunk D).
# The 6 ON: tf_eft_test_contract_on, tf_fma_guard_test_contract_on,
# dd_fma_guard_test_contract_on, dd_fma_guard_test_device_contract_on,
# ff_fma_guard_test_contract_on (chunk C) + qf_fma_guard_test_contract_on
# (chunk D).
#
# SPLIT BY SIDE FOR C5. The 3 host OFF are dd_eft_test, dd_fma_guard_test and
# qf_eft_test; the 1 host ON is dd_fma_guard_test_contract_on. Everything else
# in the two lists above is device: 8 OFF and 5 ON.
set(XPMATH_EFT_OFF_HOST_EXPECTED    3)
set(XPMATH_EFT_ON_HOST_EXPECTED     1)
set(XPMATH_EFT_OFF_DEVICE_EXPECTED  8)
set(XPMATH_EFT_ON_DEVICE_EXPECTED   5)
set(XPMATH_EFT_OFF_EXPECTED 0)
set(XPMATH_EFT_ON_EXPECTED  0)
if(XPMATH_BUILD_HOST_TARGETS)
  math(EXPR XPMATH_EFT_OFF_EXPECTED
       "${XPMATH_EFT_OFF_EXPECTED} + ${XPMATH_EFT_OFF_HOST_EXPECTED}")
  math(EXPR XPMATH_EFT_ON_EXPECTED
       "${XPMATH_EFT_ON_EXPECTED} + ${XPMATH_EFT_ON_HOST_EXPECTED}")
endif()
if(XPMATH_BUILD_DEVICE_TARGETS)
  math(EXPR XPMATH_EFT_OFF_EXPECTED
       "${XPMATH_EFT_OFF_EXPECTED} + ${XPMATH_EFT_OFF_DEVICE_EXPECTED}")
  math(EXPR XPMATH_EFT_ON_EXPECTED
       "${XPMATH_EFT_ON_EXPECTED} + ${XPMATH_EFT_ON_DEVICE_EXPECTED}")
endif()
message(STATUS "contraction guard: ${_xp_n_off} OFF targets, ${_xp_n_on} ON targets "
               "(expected ${XPMATH_EFT_OFF_EXPECTED}/${XPMATH_EFT_ON_EXPECTED})")

# THE PAIR IS TAGGED DEVICE, AND THE CHOICE IS NOT ARBITRARY -- IT IS A LOSS
# TAKEN ON PURPOSE. Neither target compiles or links anything, so on substance
# they belong to neither side; but the C5 arithmetic requires every registered
# test to sit on exactly one, so one tree has to go without a contraction check.
# Device is where the smaller gap lands: 8 of the 11 OFF targets and 5 of the 6
# ON targets live there, and it is the tree whose compiler CHANGES per arch
# (g++ for --arch host, nvcc_wrapper for a100, hipcc for mi250) -- which is
# exactly when a contraction flag is most likely to be dropped or renamed. A
# host-only tree runs g++ with the same flags it has always run.
#
# WHAT IS LOST: in a host-only tree the 3 host OFF and 1 host ON targets carry
# their flags unverified. In the DEFAULT both-ON tree -- the bare
# `cmake -B build`, and CI -- nothing is lost: the manifest holds all 11/6 and
# the guard reads all of it.
if(XPMATH_BUILD_DEVICE_TARGETS)
add_test(NAME contraction_flags_guard
         COMMAND ${CMAKE_COMMAND}
                 -DMANIFEST=${CMAKE_CURRENT_BINARY_DIR}/contraction_manifest.txt
                 -DEXPECT_OFF=${XPMATH_EFT_OFF_EXPECTED}
                 -DEXPECT_ON=${XPMATH_EFT_ON_EXPECTED}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_contraction_flags.cmake)

# ... and the poison test that proves the guard can fail, including a negative
# control so a guard hard-wired to fail cannot satisfy it either. It derives its
# own expectations from the manifest it finds, so it needs no per-side constant.
add_test(NAME contraction_flags_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/contraction_flags_selftest.sh
                 ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

xpm_tag_test(device contraction_flags_guard contraction_flags_selftest)
endif()

# ---------------------------------------------------------------------------
# STANDALONE SMOKES -- the Kokkos-free core, as real ctest targets
# ---------------------------------------------------------------------------
# These compile include/xp/ against an include path containing ONLY include/,
# so third_party/include/ cannot mask a missing dependency. They existed as
# sources driven solely by scripts/check_standalone_no_kokkos.sh, which CI runs
# and `ctest` did not -- so with Kokkos off the suite had nothing to say about
# the very code that does not need Kokkos.
#
# This does NOT replace that script: it still does the part CMake cannot, which
# is preprocessing each TU and grepping the output for the token Kokkos. A
# successful compile only proves nothing is MISSING, not that nothing extra was
# pulled in.
#
# ALL EIGHT ARE HOST -- C5 step 1, and a deliberate departure from the C5 plan
# text, which classifies them device. The reasoning is in the SIDE TAGGING block
# at the top of this file: six of the eight stream an xp:: value through
# operator<<, which include/xp/config.hpp defines only under
# `#if !defined(XPMATH_ON_DEVICE)`, so a device compiler reproduces the five
# `no operator "<<"` errors of A100 job 1000938 exactly. What they assert -- that
# the core compiles and runs against an include path containing only include/ --
# is a host-toolchain question, and their sibling
# scripts/check_standalone_no_kokkos.sh drives the same sources through g++.
if(XPMATH_BUILD_HOST_TARGETS)
foreach(_sm dd dd_complex ff ff_complex qf qf_complex tf tf_complex)
  add_executable(${_sm}_no_kokkos_smoke standalone/${_sm}_no_kokkos_smoke.cpp)
  target_include_directories(${_sm}_no_kokkos_smoke PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_features(${_sm}_no_kokkos_smoke PRIVATE cxx_std_17)
  add_test(NAME ${_sm}_no_kokkos_smoke COMMAND ${_sm}_no_kokkos_smoke)
  xpm_tag_test(host ${_sm}_no_kokkos_smoke)
endforeach()
endif()

# ---------------------------------------------------------------------------
# device_harness_test -- the self-test for tests/device_harness.hpp
# ---------------------------------------------------------------------------
# The harness (CORE_PLAN C3) is the Kokkos-free CUDA/HIP/serial launch layer
# that C4-C8 measure the core through. Here it compiles with plain g++, so what
# runs is its SERIAL fallback -- deliberately, because that fallback is the
# thing most likely to rot: nobody exercises it on a GPU node, and every device
# test will depend on it behaving like the vendor paths.
#
# In the KOKKOS-FREE set, next to the standalone smokes, for the same reason
# device_tu_purity is: the harness includes no Kokkos and must never need one.
# It counts in both configurations. Together with the tf smoke added to the
# foreach above, C3 takes the totals from 50/22 to 52 with Kokkos and 24
# without.
#
# The test cannot pass on silence. It poisons both sides of the output buffer
# before the launch and requires every element to differ from the poison AND to
# equal an exact function of its index, so a kernel that never ran, or a
# from_device() that copied nothing, is RED rather than quiet. See the file
# header.
#
# C5 step 1 tags it DEVICE and registers it only in the device tree, which is
# the one change that makes the paragraph above stop being a compromise: in a
# device tree built with --arch a100 or --arch mi250 this compiles through
# nvcc/hipcc and exercises the REAL backend rather than the serial fallback.
# With both options ON -- the default -- it is still the g++ serial build
# described above.
if(XPMATH_BUILD_DEVICE_TARGETS)
add_executable(device_harness_test device_harness_test.cpp)
target_include_directories(device_harness_test PRIVATE
                           ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(device_harness_test PRIVATE cxx_std_17)
add_test(NAME device_harness_test COMMAND device_harness_test)
xpm_tag_test(device device_harness_test)
endif()

# COMPLEX accuracy, all four backends. Until these landed, every target above
# scored REAL arithmetic only — `grep -c complex tests/CMakeLists.txt` was 0 —
# and the complex headers were covered solely by the demos, which print a table
# nobody diffs per-op against an oracle. That is how KI-1 (complex acosh returns
# the wrong branch for Re(z) < 0, in ALL FOUR backends) survived from the
# original DD port through FF, QF and TF. See docs/history/KNOWN_ISSUES.md.
#
# Standalone registration, NOT one of the test helpers: these read precomputed
# binary128
# references from a precomputed binary corpus and therefore ran on
# toolchains with no libquadmath, back when the oracle-scored tests above skipped
# there. Those tests no longer skip anywhere; this one still needs no oracle.
# 24 ops each (add sub mul div abs conj sqrt exp log log10 sin cos tan asin acos
# atan sinh cosh tanh asinh acosh atanh pow polar), per-op tolerance table set
# from measurement. NOT EFT tests, so no contraction flags — verified
# insensitive to -O0/-O2/-O3 and -ffp-contract=off/fast.

# ---------------------------------------------------------------------------
# WHY THERE IS NO PER-BACKEND ACCURACY TEST
# ---------------------------------------------------------------------------
# There used to be eight: dd/ff/qf/tf_accuracy_test and their complex
# counterparts. They were retired in 334cc9b and deleted outright once nothing
# referenced them.
#
# Each scored a MEAN digit count per op against a per-op tolerance table, and
# every entry in those tables was set by measuring the implementation and
# rounding down. That makes the current behaviour the specification: a defect
# present when the table was written is, by construction, a pass. Not one of the
# forty entries in the known-issues history was ever caught by them. Averaging
# compounds it -- one point at 0 digits inside a thousand at 31 moves the mean
# by 0.03.
#
# They are replaced by the two gates below, over ONE measurement: ulps against
# the binary128 oracle, per point, versus a bound derived from the format and
# the condition number. The verdict is per point rather than per mean, and the
# bound comes from the format rather than from the implementation. See
# docs/CORRECTNESS.md.
#
# Nothing STRUCTURAL was retired: the bit-exact EFT tests, the FMA-contraction
# guards in both postures, the representation invariants, the QF/TF non-overlap
# tests, the property/identity tests and the cancellation kernels all remain
# registered above.

# ---------------------------------------------------------------------------
# THE TWO GATES
# ---------------------------------------------------------------------------
# One measurement, one verdict per point, two gates over that single number:
#
#   sweep_absolute_gate   no point above its derived bound, except the points
#                         named in validation/sweep/open_defects.txt. The
#                         register is checked in BOTH directions, so a new
#                         defect and a stale entry both fail.
#   sweep_monotone_gate   no point worse than the committed baseline, past a
#                         stated 0.1-digit noise floor.
#
# Both run the same binary over the same 436,080 points in about 35 seconds, so
# `ctest` alone is the whole correctness answer. (It was eight seconds and
# 428,592 points; `5f2fc90` widened the grid, and the oracle moving from
# libquadmath to correctly-rounded MPFR/MPC at 400 bits accounts for the rest.
# The TIMEOUTs at the bottom of this file were set against the old number and
# still have room, but they are no longer 75x headroom -- check them before
# adding work per point.)
#
# MPFR IS REQUIRED. It used to be optional -- "a measurement dependency, not a
# library one", found when present and degraded around when absent -- because
# the sweep's oracle was libquadmath and MPFR was an ALTERNATE reachable with
# --oracle=mpfr. There is no alternate any more: sweep_accuracy scores every
# point against MPFR/MPC and refuses --oracle=quadmath, so a build without MPFR
# is not a build with less coverage, it is a build with no oracle. The source
# says the same thing with an #error; this says it at configure time, where the
# operator can still act on it, instead of after a full compile.
#
# This is still not a dependency of the LIBRARY. include/xp/ uses neither MPFR
# nor libquadmath and compiles anywhere; what needs it is the apparatus that
# judges the library.
#
# The optional-MPFR posture also cost something measurable while it lasted: four
# ctest targets sat behind if(XPMATH_MPFR_FOUND), so a runner missing one -dev
# package silently dropped them and stayed green. That is why the CI lanes
# assert an exact target count. With MPFR required there is one configuration
# instead of two and the count cannot drift for this reason.
#
# MPC (complex MPFR) is required in the SAME conjunction rather than behind its
# own XPMATH_HAVE_MPC. A build where the real ops score against MPFR but the
# complex ops fall back somewhere else is exactly the split this arc removed; a
# second guard would make that half-state reachable by configuration.
# One flag, one meaning: the oracle is whole-library or the build stops.
#
# THE REQUIREMENT IS NOW SCOPED TO THE HOST SIDE -- CORE_PLAN C5 step 1, and it
# is a narrowing rather than a softening. Everything that needs MPFR/MPC is
# host: sweep_accuracy and the four targets downstream of it, exp_reduction_test
# and trig_reduction_test. A DEVICE-only tree contains none of them, and that
# tree is the one configured on a compute node -- where demanding libmpfr-dev
# would be a FATAL_ERROR over a library no target in it links. With
# XPMATH_BUILD_HOST_TARGETS=ON, which is the default, the bare `cmake -B build`
# path and every CI lane, the hard error below is exactly as hard as it was.
#
# ============================== HOST-SIDE REGION ==============================
# Everything from here to the matching
# `endif()  # ===== END OF THE HOST-SIDE REGION =====` is host, and is
# registered only when XPMATH_BUILD_HOST_TARGETS is ON. The body is deliberately
# NOT re-indented: reflowing ~260 lines, most of them the explanatory comments
# this file is built out of, would bury a one-line change in whitespace.
#
# TWO THINGS SIT BELOW THE endif AND ARE DELIBERATELY OUTSIDE THE REGION, because
# both run in EVERY configuration: the BUILD PROVENANCE target, which judges the
# build directory and so belongs to whichever tree that directory is, and the TAG
# COMPLETENESS ASSERTION that closes the file.
# ==============================================================================
if(XPMATH_BUILD_HOST_TARGETS)

find_path(MPFR_INCLUDE_DIR mpfr.h)
find_library(MPFR_LIBRARY NAMES mpfr)
find_library(GMP_LIBRARY NAMES gmp)
find_path(MPC_INCLUDE_DIR mpc.h)
find_library(MPC_LIBRARY NAMES mpc)
if(MPFR_INCLUDE_DIR AND MPFR_LIBRARY AND GMP_LIBRARY
   AND MPC_INCLUDE_DIR AND MPC_LIBRARY)
  message(STATUS "MPFR found: ${MPFR_LIBRARY}")
  message(STATUS "MPC  found: ${MPC_LIBRARY} -- the oracle is available for BOTH "
                 "real and complex ops")
else()
  # Name WHICH one is missing. The failure mode this replaces is a bare "NOT
  # found" followed ten minutes later by a link error naming a library the
  # operator was never told to install.
  set(_xp_missing "")
  if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY)
    string(APPEND _xp_missing " libmpfr-dev")
  endif()
  if(NOT GMP_LIBRARY)
    string(APPEND _xp_missing " libgmp-dev")
  endif()
  if(NOT MPC_INCLUDE_DIR OR NOT MPC_LIBRARY)
    string(APPEND _xp_missing " libmpc-dev")
  endif()
  message(FATAL_ERROR
          "MPFR/MPC NOT found (missing:${_xp_missing}).\n"
          "  These are REQUIRED: sweep_accuracy scores every point against "
          "MPFR at 400 bits\n"
          "  and MPC, and both ctest gates run that binary. There is no "
          "libquadmath fallback\n"
          "  to degrade to -- it was removed because it mis-reduces trig above "
          "~1e40 and its\n"
          "  complex divide is round-then-cancel, so it scored correct results "
          "as regressions.\n"
          "  Install the packages above and reconfigure.")
endif()

# sweep_accuracy is a host-only tool: it has no Kokkos in it. It is built here
# rather than in scripts/ so that a stale hand-compiled binary in the source
# tree can never be what the gate runs.
add_executable(sweep_accuracy ${CMAKE_SOURCE_DIR}/scripts/sweep_accuracy.cpp)
target_include_directories(sweep_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(sweep_accuracy PRIVATE cxx_std_17)
# GNU ONLY. -fext-numeric-literals is a gcc spelling; clang does not know it and
# hard-errors on an unknown argument rather than warning, which would take out
# sweep_accuracy and with it both gates, oracle_conv_test and both gate
# selftests. The top-level CMakeLists.txt site is guarded the same way.
target_compile_options(sweep_accuracy PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-fext-numeric-literals>)
# NO quadmath. __float128 is still the sweep's carrier -- ulp arithmetic, the
# to_q() widenings, the CSV formatting -- but a __float128 is a COMPILER type
# whose arithmetic lives in libgcc, and the `*q` elementary functions that DID
# come from libquadmath are gone. Adding `quadmath` back here would relink a
# library nothing calls and break the two acceptance checks in the header of
# scripts/sweep_accuracy.cpp.
target_include_directories(sweep_accuracy PRIVATE ${MPFR_INCLUDE_DIR} ${MPC_INCLUDE_DIR})
# MPC before MPFR before GMP: mpc depends on mpfr depends on gmp, and static
# link lines are order-sensitive.
target_link_libraries(sweep_accuracy PRIVATE ${MPC_LIBRARY} ${MPFR_LIBRARY} ${GMP_LIBRARY})
target_compile_definitions(sweep_accuracy PRIVATE XPMATH_HAVE_MPFR=1)

add_test(NAME sweep_absolute_gate
         COMMAND sweep_accuracy --ulp
                 --register ${CMAKE_SOURCE_DIR}/validation/sweep/open_defects.txt)

add_test(NAME sweep_monotone_gate
         COMMAND sweep_accuracy
                 --baseline ${CMAKE_SOURCE_DIR}/validation/sweep/sweep_baseline.csv.gz)

xpm_tag_test(host sweep_absolute_gate sweep_monotone_gate)

# ---------------------------------------------------------------------------
# THE GATES' OWN SELF-TESTS
# ---------------------------------------------------------------------------
# Both gates are negative assertions, and a negative assertion observed only
# passing is indistinguishable from one that cannot fail. The monotone gate WAS
# one of those: `digits` and `bound`, two of the four value columns of every
# baseline row, were parsed and discarded, so a baseline with its last numeric
# column rewritten to zero still reported all 428,592 points unchanged and
# exited 0. These two tests poison the gates' inputs in the build directory and
# require a NONZERO exit -- and require a clean PASS on the real inputs, so a
# gate wired to fail unconditionally does not satisfy them either.
add_test(NAME sweep_monotone_gate_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/gate_selftest.sh
                 $<TARGET_FILE:sweep_accuracy> monotone ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME sweep_absolute_gate_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/gate_selftest.sh
                 $<TARGET_FILE:sweep_accuracy> absolute ${CMAKE_CURRENT_BINARY_DIR})

xpm_tag_test(host sweep_monotone_gate_selftest sweep_absolute_gate_selftest)

# ---------------------------------------------------------------------------
# C6 DEVICE GATES -- registered ONLY when a per-arch baseline exists
# ---------------------------------------------------------------------------
# CORE_PLAN C6 step 5, filled in by C7 (a100). sweep_device_gate_<arch> is a
# HOST test: it re-scores the committed raw device limbs with the host
# MPFR/MPC oracle and monotone-compares against
# validation/sweep/sweep_baseline_<arch>.csv.gz. It does not launch on a GPU,
# so it IS in CI -- the thing GitHub cannot do is re-produce the limbs, and
# those are the committed artifact.
#
# C7 landed sweep_baseline_a100.csv.gz and wired the a100 arm. mi250 still
# FATAL_ERRORs if its baseline is committed without an add_test() body: that
# is C8's to fill in, not a silent "zero tests" hole.
#
# Count tripwires move with the a100 arm: host 38 -> 40 (gate + selftest),
# device 24 unchanged, union 61 -> 63.
foreach(_xp_arch IN ITEMS a100 mi250)
  set(_xp_dev_base
      "${CMAKE_SOURCE_DIR}/validation/sweep/sweep_baseline_${_xp_arch}.csv.gz")
  if(EXISTS "${_xp_dev_base}")
    if(_xp_arch STREQUAL "a100")
      set(_xp_dev_raw
          "${CMAKE_SOURCE_DIR}/validation/a100/logs/1001685_raw.csv.gz")
      if(NOT EXISTS "${_xp_dev_raw}")
        message(FATAL_ERROR
          "C7: ${_xp_dev_base} exists, but the raw limbs it was scored from "
          "(${_xp_dev_raw}) do not. The gate re-scores those limbs; without "
          "them it cannot run.")
      endif()
      add_test(NAME sweep_device_gate_${_xp_arch}
               COMMAND sweep_accuracy
                       --score-results ${_xp_dev_raw}
                       --where ${_xp_arch}
                       --baseline ${_xp_dev_base})
      xpm_tag_test(host sweep_device_gate_${_xp_arch})
    else()
      message(FATAL_ERROR
        "C6: ${_xp_dev_base} exists, but sweep_device_gate_${_xp_arch} is not "
        "wired yet. C8 owns the add_test() body (produce via sweep_device, "
        "score via sweep_accuracy --score-results --where ${_xp_arch}, monotone "
        "compare against the arch baseline). Fill that in before committing "
        "the baseline.")
    endif()
  endif()
endforeach()

if(EXISTS "${CMAKE_SOURCE_DIR}/validation/sweep/sweep_baseline_a100.csv.gz")
  add_test(NAME sweep_device_gate_selftest
           COMMAND ${CMAKE_SOURCE_DIR}/validation/gate_selftest.sh
                   $<TARGET_FILE:sweep_accuracy> device ${CMAKE_CURRENT_BINARY_DIR})
  xpm_tag_test(host sweep_device_gate_selftest)
endif()

# ---------------------------------------------------------------------------
# KI-42/43 STRUCTURAL TEST FOR exp()'s ARGUMENT REDUCTION
# ---------------------------------------------------------------------------
# exp/exp2/exp10 reduce their argument by subtracting k*ln2 (resp. k*log10(2))
# as a sum of Cody-Waite pieces, each narrow enough that k*c_i is EXACT in one
# machine word. That property is invisible at the call site -- the pieces look
# like ordinary hex float literals -- so the failure mode is an edit that
# "improves precision" by widening one, silently restoring the |a|*2^-p
# reduction error the change removed.
#
# This test asserts the property structurally: piece widths read from the
# shipped mantissas (never re-derived at a nominal width -- rounding ln2 to 17
# or 18 bits yields a BIT-IDENTICAL float, so a widen-by-one poison tests
# nothing), exactness of k*c_i over the FULL k range the guards admit
# (DD [-1075, 1024], not the [-1024, 1024] the overflow limit suggests), and a
# tail that pins the piece COUNT so the test cannot be satisfied by deleting a
# piece.
#
# It issues NO accuracy verdict on exp -- docs/CORRECTNESS.md:5-6 reserves that
# for the sweep. Bit-equalities only, no oracle, no tolerance.
#
# Registered by hand rather than through a helper: it needs
# no corpus define, and no Kokkos link (no device code). It links MPFR, not
# libquadmath: the tail arithmetic is carried in __float128 (a compiler type,
# libgcc's arithmetic) and the ONE value it needs from a library is ln2, which
# is a reference and therefore comes correctly rounded from MPFR. See the "WHY
# THIS LINKS MPFR AND NOT LIBQUADMATH" block in the source.
# pow()'s non-positive-base contract, all four backends. Registered separately
# from the sweep because the sweep CANNOT see this: repair_real's R_Pow case
# (scripts/sweep_accuracy.cpp:529-537) sets a = fabs(a) and maps a == 0 to 1, so
# no grid point ever reaches pow with a zero or negative base. TF shipped
# without the guard the other three carry, returning 1 for pow(-0.5, 2), and
# nothing in the apparatus could notice. Contract assertions only -- exact
# comparisons against specified values, no accuracy verdict.
add_executable(pow_domain_test pow_domain_test.cpp)
target_include_directories(pow_domain_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME pow_domain_test COMMAND pow_domain_test)
# HOST: a plain host executable that launches nothing. Asking the same contract
# questions in a device functor would be worth having and is not this chunk's
# to add -- it would be a new target, and the count must not move here.
xpm_tag_test(host pow_domain_test)

add_executable(exp_reduction_test exp_reduction_test.cpp)
target_include_directories(exp_reduction_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(exp_reduction_test PRIVATE ${MPFR_INCLUDE_DIR})
target_link_libraries(exp_reduction_test PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})
add_test(NAME exp_reduction_test COMMAND exp_reduction_test)

# ... and the self-test that proves it can fail. Same contract as the gate
# self-tests above: poison must FAIL, clean must PASS.
add_test(NAME exp_reduction_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/exp_reduction_selftest.sh
                 ${CMAKE_CURRENT_BINARY_DIR})

# HOST, and not by convention: both link MPFR and carry their tail arithmetic in
# __float128, which is the one thing a device TU may not contain.
xpm_tag_test(host exp_reduction_test exp_reduction_selftest)

# Payne-Hanek argument reduction: does the engine in include/xp/trig_reduction.hpp
# return the exact quadrant and a fraction within one ulp at the inputs that
# make the shipped table and guard work hardest? Links MPFR because it computes
# its own 4000-bit ground truth rather than citing a number it cannot check --
# see the header comment for why that distinction matters here. No accuracy
# verdict on sin/cos/tan: this scores the reduction, which the sweep cannot see.
# This test recomputes the reduction tables at high precision, so there is
# nothing it can check without MPFR. It used to be guarded at the TARGET level;
# MPFR is required above now, so the guard is gone with it.
add_executable(trig_reduction_test trig_reduction_test.cpp)
target_include_directories(trig_reduction_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(trig_reduction_test PRIVATE ${MPFR_INCLUDE_DIR})
target_link_libraries(trig_reduction_test PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})
add_test(NAME trig_reduction_test COMMAND trig_reduction_test)

# ... and the self-test that proves it can fail: four poisons, plus a check
# that the committed table is byte-identical to what the generator emits.
add_test(NAME trig_reduction_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/trig_reduction_selftest.sh
                 ${CMAKE_CURRENT_BINARY_DIR})

# HOST: it computes 4000-bit ground truth with MPFR.
xpm_tag_test(host trig_reduction_test trig_reduction_selftest)

# The oracle's __float128 <-> mpfr conversions must be EXACT, not merely
# round-trippable. The sweep has no oracle-error term, so a reference that is
# wrong reads as a library that is wrong -- and the old conversion was wrong by
# 1.28e11 ulps on precisely the large-|x| trig rows this oracle exists to
# measure. Runs the check, then poisons it eight ways and requires each to fail.
# No if(XPMATH_MPFR_FOUND): these now score the only oracle there is.
add_test(NAME oracle_conv_test COMMAND sweep_accuracy --oracle-selftest)
add_test(NAME oracle_conv_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/oracle_conv_selftest.sh
                 ${CMAKE_CURRENT_BINARY_DIR})

# HOST: the subject is the oracle's own __float128 <-> mpfr conversion.
xpm_tag_test(host oracle_conv_test oracle_conv_selftest)

# The generated domain document must be what the generator currently emits from
# the committed baseline. Pure Python, well under a second, no build products.
# ---------------------------------------------------------------------------
# THE CONSUMER TEST (packaging)
# ---------------------------------------------------------------------------
# Everything else in this file tests the LIBRARY. This one tests the PACKAGE:
# it installs xpmath to a scratch prefix and builds a separate project against
# it with find_package(). Nothing else in the build exercises the exported
# target, the installed header layout or the version file, so all three can be
# broken while the other 38 targets stay green.
#
# It is registered as a ctest target rather than left as a script somebody
# remembers to run, because validation/strip_timing.sh is already an object
# lesson in what happens to a check nothing invokes.
#
# WHY IT SHELLS OUT TO CMAKE INSTEAD OF BEING A NORMAL TEST. The check requires
# a completed `cmake --install` of THIS project, which cannot happen during the
# configure that is registering the test. So the test drives a nested
# configure/build/install into a scratch prefix at RUN time. That makes it the
# most expensive target here by wall clock and the only one that recurses into
# CMake; both are inherent to what it verifies.
#
# Skipped (not failed) when the generator is not Makefiles/Ninja, since the
# nested build inherits the generator and multi-config generators would need a
# --config argument threaded through.
add_test(NAME consumer_package
         COMMAND "${CMAKE_COMMAND}"
                 -DXPMATH_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                 -DXPMATH_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/consumer_scratch
                 -DXPMATH_GENERATOR=${CMAKE_GENERATOR}
                 -DXPMATH_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 "-DXPMATH_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                 -DXPMATH_KOKKOS_DIR=${Kokkos_DIR}
                 -P "${CMAKE_SOURCE_DIR}/tests/consumer/run_consumer_test.cmake")
set_tests_properties(consumer_package PROPERTIES
                     LABELS packaging
                     TIMEOUT 600)
# HOST: the nested configure it drives is an ordinary host CMake build
# (XPMATH_WITH_KOKKOS=OFF, XPMATH_BUILD_TESTS=OFF), and the package it checks is
# the header-only export, which has no side.
xpm_tag_test(host consumer_package)

add_test(NAME domains_fresh
         COMMAND ${CMAKE_SOURCE_DIR}/validation/check_domains_fresh.sh)
# HOST: docs/DOMAINS.md is generated from the HOST baseline. C9 adds the device
# counterpart (device_domains_fresh) once there is a device baseline to read.
xpm_tag_test(host domains_fresh)

# ---------------------------------------------------------------------------
# DEVICE-TU PURITY
# ---------------------------------------------------------------------------
# tests/test_utils_device.hpp is the half of the test harness a device
# translation unit may include, and the one thing that would make it unusable
# there is __float128: nvcc walks STL member signatures during the device pass
# and rejects the 128-bit float. That is not hypothetical -- std::vector<__float128>
# in the unsplit tests/test_utils.hpp put the type into all 21 test TUs that
# included the header, and A100 (S8b) could not build 9 of 49 targets.
#
# Registered OUTSIDE the XPMATH_WITH_KOKKOS guard, like build_provenance above,
# and for a stronger reason than convenience: the device header names the xp
# core rather than the third_party Kokkos wrappers, so this gate needs no
# Kokkos install to run. It counts in both configurations: 49 -> 50 with
# Kokkos, 21 -> 22 without.
#
# NOT a `grep __float128`, which is what you would write first and which would
# have been red from the day it was written -- /usr/include/bits/floatn.h
# typedefs the type on x86_64 whether or not anyone uses it. The script
# attributes each occurrence to its originating file via the preprocessor's
# line markers and fails only on hits from inside this repository.
#
# It takes no contraction flags and adds no target, so
# XPMATH_EFT_OFF_EXPECTED / XPMATH_EFT_ON_EXPECTED above are unchanged by it.
add_test(NAME device_tu_purity
         COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_device_tu_purity.sh)
# HOST DESPITE THE NAME, and it is worth being explicit about why. What it
# checks is device TUs; what it RUNS is `${CXX:-g++} -E` over source FILES from
# a hand-maintained list, so it needs a host preprocessor and needs nothing to
# have been built. Putting it in the device tree would hand it to nvcc/hipcc,
# whose preprocessed output carries different line markers and different system
# headers, for no gain -- and would lose the mirror half of the check (no host
# TU may launch a kernel), whose subjects are the host targets.
xpm_tag_test(host device_tu_purity)

set_tests_properties(sweep_absolute_gate sweep_monotone_gate
                     PROPERTIES TIMEOUT 600)
set_tests_properties(sweep_absolute_gate_selftest sweep_monotone_gate_selftest
                     PROPERTIES TIMEOUT 1200)
if(EXISTS "${CMAKE_SOURCE_DIR}/validation/sweep/sweep_baseline_a100.csv.gz")
  set_tests_properties(sweep_device_gate_a100 PROPERTIES TIMEOUT 600)
  set_tests_properties(sweep_device_gate_selftest PROPERTIES TIMEOUT 1200)
endif()
# Nine compiles of sweep_accuracy.cpp, four at a time.
set_tests_properties(oracle_conv_selftest PROPERTIES TIMEOUT 600)

endif()  # ===== END OF THE HOST-SIDE REGION =====

# ===========================================================================
# BUILD PROVENANCE -- REGISTERED IN EVERY TREE, CORE_PLAN C5 step 5
# ===========================================================================
# Every other target in this file judges the LIBRARY. This one judges the BUILD
# DIRECTORY: it fails when ${CMAKE_BINARY_DIR}/build-info.txt is absent, short a
# field, or carries a `tree:` value outside host/device/both, so a build whose
# artifacts could not be traced back to a tree is caught by ctest rather than by
# the next person who needs the baseline.
#
# IT USED TO BE HOST-ONLY, AND THAT WAS A HOLE THIS SECTION HAD TO CLOSE.
# C5 chunk A tagged it `host` and wrote down what that cost: a device-only tree
# stamps a build-info.txt that nothing then checks. That is not a hypothetical
# tree -- it is precisely what C7 and C8 configure on a compute node with
# `--only device`, and their GPU logs are the artifacts whose provenance is
# hardest to reconstruct and most expensive to re-measure. The one tree with no
# provenance check would have been the one that needed it most.
#
# SO IT IS REGISTERED OUTSIDE BOTH REGIONS -- outside XPMATH_WITH_KOKKOS, as it
# always was, and now outside XPMATH_BUILD_HOST_TARGETS as well. Provenance is a
# property of the build directory, and every build directory has one.
#
# THE TAG FOLLOWS THE TREE rather than being a constant, because the tag
# assertion below demands exactly one side per registered test and this test is
# in every side. In a host-only tree it is host; in a device-only tree it is
# device; in the default both-ON tree it is host, where it has always counted.
# The name is registered once in every configuration, so no tree builds it twice
# and the bare `cmake -B build` count is unchanged.
#
# It takes NO contraction flags and adds no target, so
# XPMATH_EFT_OFF_EXPECTED / XPMATH_EFT_ON_EXPECTED above are unchanged by it.
#
# IT ALSO MEASURES THE C++ STANDARD -- CORE_PLAN C5 step 6, and the reason this
# is one test rather than two. `cxx-standard: 17` was the one stamped field
# known to be false (Kokkos raises its linked targets to c++20 and the stamp
# echoed CMAKE_CXX_STANDARD regardless), so the check reads the -std= back off
# compile_commands.json. RAISED_TARGETS is the closed set allowed to differ,
# spelled out in the top-level CMakeLists.txt beside the add_executable calls.
# A separate ctest target would have changed all three suite counts to assert a
# property of the same file this one already judges.
#
# COMPILE_COMMANDS is passed only when the database will exist. Handing the
# check a path to a file the generator was never going to write would fail on
# the generator rather than on the build.
#
# COMMA-SEPARATED, NOT SEMICOLON. A `;` inside an add_test() argument is a list
# separator to the generator, which splits it into one argument per target:
# measured, the check then saw RAISED_TARGETS=kokkos_ep_demo alone and reported
# the other seven demos as disagreeing. The script splits the commas back.
set(_xp_prov_args -DINFO=${CMAKE_BINARY_DIR}/build-info.txt)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  string(REPLACE ";" "," _xp_raised_csv "${XPMATH_KOKKOS_LINKED_TARGETS}")
  list(APPEND _xp_prov_args
       -DCOMPILE_COMMANDS=${CMAKE_BINARY_DIR}/compile_commands.json
       -DRAISED_TARGETS=${_xp_raised_csv})
endif()
add_test(NAME build_provenance
         COMMAND ${CMAKE_COMMAND}
                 ${_xp_prov_args}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_build_provenance.cmake)
if(XPMATH_BUILD_HOST_TARGETS)
  xpm_tag_test(host build_provenance)
else()
  xpm_tag_test(device build_provenance)
endif()

# ===========================================================================
# TAG COMPLETENESS ASSERTION -- CORE_PLAN C5 step 1
# ===========================================================================
# Every test registered in this directory must carry exactly one side tag, and
# this fails the CONFIGURE when one does not. A target that reaches a tree it
# does not belong to is the failure C5 exists to prevent -- the A100 job that
# put sweep_accuracy through nvcc -- and the way that happens is not malice but
# a new add_test() written without a thought about sides. Catching it here means
# the author sees it, rather than a GPU queue seeing it a week later.
#
# IT CANNOT PASS ON SILENCE. Four separate failures, and two of them fire on
# emptiness rather than on content:
#   - zero registered tests            -> FATAL. An empty suite is what a
#                                         misplaced if() produces, and `ctest`
#                                         over it exits 0.
#   - zero tagged tests                -> FATAL. This is the check checking
#                                         itself: if the GLOBAL lists were never
#                                         written to, "no untagged tests found"
#                                         would otherwise be vacuously true.
#   - registered but untagged          -> FATAL, naming them.
#   - tagged but not registered, or
#     tagged twice, or tagged on both
#     sides                            -> FATAL, naming them.
# The counts must be EQUAL, not merely compatible, which is what closes the last
# case: a test tagged on both sides makes the tagged count exceed the registered
# one even though every name appears in both lists.
get_property(_xp_registered DIRECTORY PROPERTY TESTS)
get_property(_xp_host_tests GLOBAL PROPERTY XPMATH_HOST_TESTS)
get_property(_xp_dev_tests  GLOBAL PROPERTY XPMATH_DEVICE_TESTS)

list(LENGTH _xp_registered _xp_n_registered)
list(LENGTH _xp_host_tests _xp_n_host)
list(LENGTH _xp_dev_tests  _xp_n_dev)
math(EXPR _xp_n_tagged "${_xp_n_host} + ${_xp_n_dev}")

if(_xp_n_registered EQUAL 0)
  message(FATAL_ERROR
          "C5 tag assertion: this configuration registered ZERO ctest targets. "
          "`ctest` over it would exit 0 having measured nothing.")
endif()
if(_xp_n_tagged EQUAL 0)
  message(FATAL_ERROR
          "C5 tag assertion: ${_xp_n_registered} tests are registered and NONE "
          "is tagged. The tagging mechanism itself is not running.")
endif()

set(_xp_untagged ${_xp_registered})
if(_xp_host_tests)
  list(REMOVE_ITEM _xp_untagged ${_xp_host_tests})
endif()
if(_xp_dev_tests)
  list(REMOVE_ITEM _xp_untagged ${_xp_dev_tests})
endif()
if(_xp_untagged)
  list(JOIN _xp_untagged ", " _xp_untagged_s)
  message(FATAL_ERROR
          "C5 tag assertion: these registered tests carry no side tag -- call "
          "xpm_tag_test(host|device <name>) at their registration site, inside "
          "the matching XPMATH_BUILD_*_TARGETS guard:\n  ${_xp_untagged_s}")
endif()

set(_xp_ghost ${_xp_host_tests} ${_xp_dev_tests})
if(_xp_registered)
  list(REMOVE_ITEM _xp_ghost ${_xp_registered})
endif()
if(_xp_ghost)
  list(JOIN _xp_ghost ", " _xp_ghost_s)
  message(FATAL_ERROR
          "C5 tag assertion: these names are tagged but were never registered "
          "with add_test():\n  ${_xp_ghost_s}")
endif()

if(NOT _xp_n_tagged EQUAL _xp_n_registered)
  message(FATAL_ERROR
          "C5 tag assertion: ${_xp_n_registered} tests registered but "
          "${_xp_n_tagged} tags recorded (${_xp_n_host} host + ${_xp_n_dev} "
          "device). Every name is accounted for in both directions, so the "
          "difference is a DUPLICATE tag -- most likely one test tagged on both "
          "sides, which would build it in both trees.")
endif()

message(STATUS
        "C5 side tags: ${_xp_n_registered} tests registered = "
        "${_xp_n_host} host + ${_xp_n_dev} device "
        "(HOST_TARGETS=${XPMATH_BUILD_HOST_TARGETS}, "
        "DEVICE_TARGETS=${XPMATH_BUILD_DEVICE_TARGETS})")
