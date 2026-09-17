# ============================================================================
# tests/CMakeLists.txt — extended-precision test suite (T0.1 harness)
# ============================================================================
# Registered from the top-level CMakeLists.txt via add_subdirectory(tests),
# guarded by option KOKKOS_EP_BUILD_TESTS (default ON) and enable_testing().

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

# --- Helper: define + register one test -----------------------------------
# Usage: kokkos_ep_add_test(<name>)  expects tests/<name>.cpp
#
# Creates target <name> from tests/<name>.cpp, links Kokkos::kokkos and the
# third-party include dir (dd_math.hpp etc.), and registers it with CTest.
#
# To add a new test:  create tests/foo_test.cpp, then add
#   kokkos_ep_add_test(foo_test)
# below. That's the whole recipe (see tests/README.md).
function(kokkos_ep_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(${name} PRIVATE Kokkos::kokkos)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- Helper: EFT test with FMA contraction disabled -----------------------
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
# Compiler coverage below matches what this project actually builds under (GCC
# host today; nvcc when Kokkos is CUDA-backed). IntelLLVM/Intel and Clang lines
# are included because they are one-liners and cost nothing, but are not the
# tested path — extend as needed for other toolchains.
function(kokkos_ep_add_eft_test name)
  kokkos_ep_add_test(${name})
  # Host C++ (GNU/Clang): -ffp-contract=off. GCC's default at -O2 is already
  # -ffp-contract=off for standard C++, but we set it explicitly so the guarantee
  # does not depend on the default or on -Ofast/-ffast-math leaking in.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
  # Intel compilers: -fp-model=precise disables contraction/reassociation.
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>)
  endif()
  # When Kokkos is CUDA-backed, source compiles as CUDA (nvcc). --fmad=false stops
  # nvcc fusing the Dekker sequence. Guarded on COMPILE_LANGUAGE:CUDA so it is a
  # no-op for the Serial/host build; harmless when CUDA is not enabled.
  if(Kokkos_ENABLE_CUDA)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()
  # Tell the test source it is the contraction-OFF variant (gate on mismatches).
  # dd_fma_guard_test.cpp keys its gate-vs-report behavior on this; other EFT
  # tests ignore it. Harmless where unused.
  target_compile_definitions(${name} PRIVATE KOKKOS_EP_CONTRACTION_MODE=0)
  # For the contraction guard at the bottom of this file.
  set_property(GLOBAL APPEND PROPERTY XPMATH_EFT_OFF_TARGETS ${name})
endfunction()

# --- Helper: companion EFT test with FMA contraction forced ON ------------
# T1.5's positive counterpart to kokkos_ep_add_eft_test. It builds the SAME test
# source into a SEPARATE executable (<name>_contract_on) compiled with contraction
# forced ON, so the Dekker twoProduct is exercised under the posture the OFF helper
# protects against. Whether that produces the same (compiler didn't contract) or
# different (compiler did contract) results, the outcome is recorded rather than
# gated — the ON test is a REPORTER (see dd_fma_guard_test.cpp header). Both
# variants coexist because the target name gets the _contract_on suffix.
#
# We do not call kokkos_ep_add_test here (it would collide on the base target name
# and apply no contraction flags); instead we build the suffixed target directly,
# mirroring kokkos_ep_add_test's body plus the ON flags/definitions.
function(kokkos_ep_add_eft_test_contract_on name)
  set(tgt ${name}_contract_on)
  add_executable(${tgt} ${name}.cpp)
  target_include_directories(${tgt} PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(${tgt} PRIVATE Kokkos::kokkos)
  add_test(NAME ${tgt} COMMAND ${tgt})

  # Force contraction ON (the mirror image of kokkos_ep_add_eft_test).
  # GCC/Clang: -ffp-contract=fast is the portable spelling both accept (GCC also
  # accepts =on, but =fast is the one clang honors identically, so we use =fast for
  # both). This permits — does not force — the compiler to fuse mul-then-add; on a
  # non-FMA ISA target GCC still emits mul+sub, which is exactly the "belt+
  # suspenders" outcome the ON reporter records.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>)
  endif()
  # Intel: -fp-model=fast enables contraction/reassociation.
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fp-model=fast>)
  endif()
  # CUDA: --fmad=true is nvcc's DEFAULT; stated explicitly for clarity/symmetry.
  if(Kokkos_ENABLE_CUDA)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  endif()

  # Contraction-ON mode (report, don't gate) + baseline path for drift detection.
  # Baseline file is per-test: <base>_baseline.txt where <base> = name with the
  # trailing "_test" stripped (dd_fma_guard_test -> dd_fma_guard_baseline.txt,
  # ff_fma_guard_test -> ff_fma_guard_baseline.txt). Generalized from the original
  # hardcoded dd_ path so T2.5's FF guard reuses this helper unchanged.
  string(REGEX REPLACE "_test$" "" _base "${name}")
  target_compile_definitions(${tgt} PRIVATE
    KOKKOS_EP_CONTRACTION_MODE=1
    KOKKOS_EP_BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/${_base}_baseline.txt")
  set_property(GLOBAL APPEND PROPERTY XPMATH_EFT_ON_TARGETS ${tgt})
endfunction()

# --- Registered tests -----------------------------------------------------
# Everything from here to the "END KOKKOS-LINKED TESTS" marker links
# Kokkos::kokkos, because it exercises the compat wrappers in
# third_party/include/. The Kokkos-free tests -- both gates, pow_domain,
# exp_reduction, trig_reduction, the standalone smokes, the consumer package --
# are registered after that marker and build with XPMATH_WITH_KOKKOS=OFF.
if(XPMATH_WITH_KOKKOS)
kokkos_ep_add_test(hello_test)
kokkos_ep_add_test(corpus_test)
# T1.2 non-overlap invariant: oracle-independent, no contraction flags needed.
kokkos_ep_add_test(dd_invariant_test)
# T2.2 non-overlap invariant for FF: FP32-typed fl(hi+lo)==hi over every FF op.
# Oracle-independent (no __float128 at all). No contraction flags — the
# invariant holds regardless of FMA contraction (not an EFT test).
kokkos_ep_add_test(ff_invariant_test)
# T1.3 property/identity tests: Group A bit-exact (no oracle), Group B/Test C use
# the __float128 oracle. Plain helper — NOT an EFT test, so no contraction flags.
kokkos_ep_add_test(dd_property_test)
# T1.4 differential accuracy: per-op digits-of-accuracy against the __float128
# oracle. Plain helper — NOT an EFT test, so no contraction flags.
# T1.6 end-to-end cancellation kernels: __float128-oracle-scored, host-side. Plain
# helper — NOT an EFT test, so no contraction flags (kernels are not EFT atoms).
kokkos_ep_add_test(dd_e2e_test)
kokkos_ep_add_eft_test(dd_eft_test)
# T2.1 EFT unit tests for FF: twoSum + Dekker twoProduct, FP64 oracle (exact —
# 25-bit sum / 48-bit product fit in FP64's 53-bit mantissa, so it needs no type
# wider than a machine word). Same contraction-OFF posture as dd_eft_test (Dekker
# twoProduct is the contraction hazard). Contraction-ON reporter mirror is T2.5.
kokkos_ep_add_eft_test(ff_eft_test)
# T2.3 property/identity tests for FF: Group A bit-exact (no oracle), Group B/Test C
# use the __float128 oracle. FF analogue of dd_property_test. Plain helper —
# NOT an EFT test, so no contraction flags
# (identities are FMA-contraction agnostic).
kokkos_ep_add_test(ff_property_test)
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
kokkos_ep_add_test(ff_cancellation_test)
# T1.5 FMA-contraction guard: same source, two postures (see dd_fma_guard_test.cpp).
kokkos_ep_add_eft_test(dd_fma_guard_test)              # -> dd_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(dd_fma_guard_test)  # -> dd_fma_guard_test_contract_on  (ON, reports)
# T2.5 FMA-contraction guard for FF: FF analogue of T1.5, same source/two-postures
# dual-registration pattern (see ff_fma_guard_test.cpp). Reuses the SAME helpers as
# dd_fma_guard_test — the only per-variant knobs are the compile definitions the
# helpers set; the ON helper derives ff_fma_guard_baseline.txt from the target name.
# Unlike the DD guard, the FF source uses an FP64 oracle (exact, no type wider
# than a machine word), so it runs on any target rather than x86_64 only.
kokkos_ep_add_eft_test(ff_fma_guard_test)              # -> ff_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(ff_fma_guard_test)  # -> ff_fma_guard_test_contract_on  (ON, reports)
# T3.1 EFT unit tests for QF: qf_two_sum / qf_quick_two_sum / qf_two_prod / qf_two_sqr
# (FP64 oracle, exact — 25-bit sum / 48-bit product fit in FP64's 53-bit mantissa)
# PLUS renorm_4 (len 5->4) + renorm (len 4->4), the QF-unique renormalization
# primitives (Priest non-overlap invariant + exact FP64 value-preservation on
# bounded-spread inputs, plus a __float128 wide-spread truncation check).
# Calls the SHIPPED qf_math.hpp free functions directly (no mirror-and-comment —
# unlike ff_eft_test, QF exposes its primitives as free functions). Same
# contraction-OFF posture as ff_eft_test (Dekker twoProduct is the contraction
# hazard). Contraction-ON reporter mirror is T3.5.
kokkos_ep_add_eft_test(qf_eft_test)

# T3.2 non-overlap invariant checks for QF: the Priest length-4 invariant
# |f_{i+1}| <= 1/2 ulp(f_i) (i=0,1,2) on the OUTPUT of every QF op that returns a
# QuadFloat (arithmetic, sqrt, transcendentals, joint sincos/sinhcosh, scalar
# multiply_scalar/mul_pwr2, fma, pow_int, round family, utilities). Oracle-
# independent (mathematical 1/2-ulp form via frexp — no reference value), so it
# runs unconditionally; __float128 only enriches inputs to ~96-bit width. Plain
# helper (NOT the EFT helper): a per-word add/compare has nothing to contract,
# so no -ffp-contract=off posture is required. QF analogue of
# ff_invariant_test (T2.2); shares half_ulp / nonoverlap_holds with qf_eft_test.
kokkos_ep_add_test(qf_nonoverlap_test)

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
kokkos_ep_add_test(qf_property_test)

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
# type wider than a machine word), so it runs on any target, not x86_64 only.
kokkos_ep_add_eft_test(qf_fma_guard_test)              # -> qf_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(qf_fma_guard_test)  # -> qf_fma_guard_test_contract_on  (ON, reports)

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
kokkos_ep_add_test(qf_cancellation_test)

# S10 Phase 2: TF (TripleFloat, 3×FP32) test suite. TF analogue of T3.1-T3.6 (QF),
# T2.1-T2.6 (FF), T1.1-T1.6 (DD). Same registration pattern as QF: EFT test with
# contraction-OFF + contraction-ON dual; property/accuracy/cancellation use plain
# helper (no contraction flags).
kokkos_ep_add_eft_test(tf_eft_test)                    # -> tf_eft_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(tf_eft_test)        # -> tf_eft_test_contract_on  (ON, reports)
kokkos_ep_add_test(tf_property_test)                   # algebraic identities (Group A/B/C)
kokkos_ep_add_test(tf_cancellation_test)               # e2e cancellation kernels (K1-K4)

kokkos_ep_add_eft_test(tf_fma_guard_test)              # -> tf_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(tf_fma_guard_test)  # -> tf_fma_guard_test_contract_on  (ON, reports)

# ---------------------------------------------------------------------------
# THE CONTRACTION FLAG GUARD
# ---------------------------------------------------------------------------
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
# check circular: delete a kokkos_ep_add_eft_test call and both the manifest
# and the expectation shrink together, so the guard passes while a target has
# silently lost its contraction flag by ceasing to exist. Same contract as
# ci.yml's `expected=49` -- adding or removing a guarded target is a deliberate
# act that updates a number here.
set(XPMATH_EFT_OFF_EXPECTED 8)   # dd/ff/qf/tf _eft_test + dd/ff/qf/tf _fma_guard_test
set(XPMATH_EFT_ON_EXPECTED  5)   # the four *_fma_guard_test_contract_on + tf_eft_test_contract_on
message(STATUS "contraction guard: ${_xp_n_off} OFF targets, ${_xp_n_on} ON targets "
               "(expected ${XPMATH_EFT_OFF_EXPECTED}/${XPMATH_EFT_ON_EXPECTED})")

add_test(NAME contraction_flags_guard
         COMMAND ${CMAKE_COMMAND}
                 -DMANIFEST=${CMAKE_CURRENT_BINARY_DIR}/contraction_manifest.txt
                 -DEXPECT_OFF=${XPMATH_EFT_OFF_EXPECTED}
                 -DEXPECT_ON=${XPMATH_EFT_ON_EXPECTED}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_contraction_flags.cmake)

# ... and the poison test that proves the guard can fail, including a negative
# control so a guard hard-wired to fail cannot satisfy it either.
add_test(NAME contraction_flags_selftest
         COMMAND ${CMAKE_SOURCE_DIR}/validation/contraction_flags_selftest.sh
                 ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

endif()  # ===================== END KOKKOS-LINKED TESTS =====================

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
foreach(_sm dd dd_complex ff ff_complex qf qf_complex tf tf_complex)
  add_executable(${_sm}_no_kokkos_smoke standalone/${_sm}_no_kokkos_smoke.cpp)
  target_include_directories(${_sm}_no_kokkos_smoke PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_features(${_sm}_no_kokkos_smoke PRIVATE cxx_std_17)
  add_test(NAME ${_sm}_no_kokkos_smoke COMMAND ${_sm}_no_kokkos_smoke)
endforeach()

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
add_executable(device_harness_test device_harness_test.cpp)
target_include_directories(device_harness_test PRIVATE
                           ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(device_harness_test PRIVATE cxx_std_17)
add_test(NAME device_harness_test COMMAND device_harness_test)

# COMPLEX accuracy, all four backends. Until these landed, every target above
# scored REAL arithmetic only — `grep -c complex tests/CMakeLists.txt` was 0 —
# and the complex headers were covered solely by the demos, which print a table
# nobody diffs per-op against an oracle. That is how KI-1 (complex acosh returns
# the wrong branch for Re(z) < 0, in ALL FOUR backends) survived from the
# original DD port through FF, QF and TF. See docs/history/KNOWN_ISSUES.md.
#
# Standalone helper, NOT kokkos_ep_add_test: these read precomputed binary128
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

add_test(NAME domains_fresh
         COMMAND ${CMAKE_SOURCE_DIR}/validation/check_domains_fresh.sh)

# ---------------------------------------------------------------------------
# BUILD PROVENANCE
# ---------------------------------------------------------------------------
# Every other target here judges the LIBRARY. This one judges the BUILD
# DIRECTORY: it fails when ${CMAKE_BINARY_DIR}/build-info.txt is absent or
# short a field, so a build whose artifacts could not be traced back to a tree
# is caught by ctest rather than by the next person who needs the baseline.
#
# Registered OUTSIDE the XPMATH_WITH_KOKKOS guard -- provenance is a property
# of the build, not of whether Kokkos is in it -- so it counts in both
# configurations: 48 -> 49 with Kokkos, 20 -> 21 without.
#
# It takes NO contraction flags and adds no target, so
# XPMATH_EFT_OFF_EXPECTED / XPMATH_EFT_ON_EXPECTED above stay at 8 / 5.
add_test(NAME build_provenance
         COMMAND ${CMAKE_COMMAND}
                 -DINFO=${CMAKE_BINARY_DIR}/build-info.txt
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_build_provenance.cmake)

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
# XPMATH_EFT_OFF_EXPECTED / XPMATH_EFT_ON_EXPECTED above stay at 8 / 5.
add_test(NAME device_tu_purity
         COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_device_tu_purity.sh)

set_tests_properties(sweep_absolute_gate sweep_monotone_gate
                     PROPERTIES TIMEOUT 600)
set_tests_properties(sweep_absolute_gate_selftest sweep_monotone_gate_selftest
                     PROPERTIES TIMEOUT 1200)
# Nine compiles of sweep_accuracy.cpp, four at a time.
set_tests_properties(oracle_conv_selftest PROPERTIES TIMEOUT 600)
