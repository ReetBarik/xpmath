# ============================================================================
# tests/CMakeLists.txt — extended-precision test suite (T0.1 harness)
# ============================================================================
# Registered from the top-level CMakeLists.txt via add_subdirectory(tests),
# guarded by option KOKKOS_EP_BUILD_TESTS (default ON) and enable_testing().

# --- Oracle availability --------------------------------------------------
# The __float128 oracle (impl/Kokkos_QuadPrecisionMath.hpp) exists only when
# Kokkos was built with Kokkos_ENABLE_LIBQUADMATH=ON. Kokkos advertises enabled
# TPLs via the Kokkos_TPLS list (checked the same way in the top-level file).
# When present, we define KOKKOS_EP_HAVE_QUADMATH so test_utils.hpp compiles its
# oracle path; when absent, tests still BUILD but runtime-skip (exit code 77 ->
# CTest "Skipped"). Graceful degradation = skip, not configure error, matching
# the T0.0/T0.3 posture for the demos.
if("LIBQUADMATH" IN_LIST Kokkos_TPLS)
  set(KOKKOS_EP_QUADMATH_DEFINE KOKKOS_EP_HAVE_QUADMATH)
  message(STATUS "tests: __float128 oracle enabled (LIBQUADMATH present)")
else()
  set(KOKKOS_EP_QUADMATH_DEFINE "")
  message(WARNING
    "tests: Kokkos has no LIBQUADMATH; oracle-dependent tests will runtime-SKIP "
    "(CTest reports Skipped). Rebuild Kokkos with -DKokkos_ENABLE_LIBQUADMATH=ON "
    "to enable them.")
endif()

# --- Helper: define + register one test -----------------------------------
# Usage: kokkos_ep_add_test(<name>)  expects tests/<name>.cpp
#
# Creates target <name> from tests/<name>.cpp, links Kokkos::kokkos and the
# third-party include dir (dd_math.hpp etc.), applies the quadmath define, and
# registers it with CTest. SKIP_RETURN_CODE=77 lets a test signal "Skipped" at
# runtime (returned by main() when KOKKOS_EP_HAVE_QUADMATH is undefined).
#
# To add a new test:  create tests/foo_test.cpp, then add
#   kokkos_ep_add_test(foo_test)
# below. That's the whole recipe (see tests/README.md).
function(kokkos_ep_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${THIRD_PARTY_INCLUDE})
  target_link_libraries(${name} PRIVATE Kokkos::kokkos)
  if(KOKKOS_EP_QUADMATH_DEFINE)
    target_compile_definitions(${name} PRIVATE ${KOKKOS_EP_QUADMATH_DEFINE})
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
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
  if(KOKKOS_EP_QUADMATH_DEFINE)
    target_compile_definitions(${tgt} PRIVATE ${KOKKOS_EP_QUADMATH_DEFINE})
  endif()
  add_test(NAME ${tgt} COMMAND ${tgt})
  set_tests_properties(${tgt} PROPERTIES SKIP_RETURN_CODE 77)

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
endfunction()

# --- Helper: standalone test, no Kokkos and no libquadmath ----------------
# For tests that must keep running on toolchains where the GCC-flavoured
# __float128 oracle does not exist. hipcc and icpx are clang-based; there,
# KOKKOS_EP_QUADMATH_DEFINE is unset and every oracle-scored test above
# runtime-SKIPs with exit 77, so an AMD or Intel run reports no accuracy signal
# at all. The *_complex_accuracy_test targets close that hole by reading
# precomputed binary128 references from a file instead of computing them:
#
#   tests/data/xp_corpus_complex.bin   generated ONCE on an x86_64 GCC host by
#                                      scripts/gen_corpus.cpp (which does link
#                                      libquadmath), committed, consumed anywhere.
#
# So this helper deliberately does NOT do three things kokkos_ep_add_test does:
#   * no Kokkos::kokkos link      — the sources include only include/xp/, and a
#                                   Kokkos link would re-import -arch=sm_XX and
#                                   the rest of the interface flags into a TU
#                                   that has no device code;
#   * no KOKKOS_EP_QUADMATH_DEFINE — the consumer side must not compile a
#                                   quadmath path even when one is available,
#                                   or the decoupling would be untested here;
#   * no SKIP_RETURN_CODE 77       — there is nothing to skip. A missing or stale
#                                   corpus is a hard FAIL with instructions,
#                                   not a silent pass.
#
# Include path is include/ (the standalone core) plus this directory (for
# corpus_binary.hpp); third_party/include is intentionally absent, mirroring
# scripts/check_standalone_no_kokkos.sh, so a Kokkos compat wrapper cannot
# accidentally satisfy an include.
function(kokkos_ep_add_standalone_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE
    KOKKOS_EP_COMPLEX_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/data/xp_corpus_complex.bin")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- Registered tests -----------------------------------------------------
kokkos_ep_add_test(hello_test)
kokkos_ep_add_test(corpus_test)
# T1.2 non-overlap invariant: oracle-independent, no contraction flags needed.
kokkos_ep_add_test(dd_invariant_test)
# T2.2 non-overlap invariant for FF: FP32-typed fl(hi+lo)==hi over every FF op.
# Oracle-independent (no __float128, runs without LIBQUADMATH). No contraction
# flags — the invariant holds regardless of FMA contraction (not an EFT test).
kokkos_ep_add_test(ff_invariant_test)
# T1.3 property/identity tests: Group A bit-exact (no oracle), Group B/Test C use
# the __float128 oracle. Plain helper — NOT an EFT test, so no contraction flags.
kokkos_ep_add_test(dd_property_test)
# T1.4 differential accuracy vs quadmath: per-op digits-of-accuracy against the
# __float128 oracle. Quadmath-gated (runtime-SKIPs without LIBQUADMATH). Plain
# helper — NOT an EFT test, so no contraction flags.
kokkos_ep_add_test(dd_accuracy_test)
# T1.6 end-to-end cancellation kernels: quadmath-oracle-gated, host-side. Plain
# helper — NOT an EFT test, so no contraction flags (kernels are not EFT atoms).
kokkos_ep_add_test(dd_e2e_test)
kokkos_ep_add_eft_test(dd_eft_test)
# T2.1 EFT unit tests for FF: twoSum + Dekker twoProduct, FP64 oracle (exact —
# 25-bit sum / 48-bit product fit in FP64's 53-bit mantissa; runs unconditionally,
# no LIBQUADMATH gate). Same contraction-OFF posture as dd_eft_test (Dekker
# twoProduct is the contraction hazard). Contraction-ON reporter mirror is T2.5.
kokkos_ep_add_eft_test(ff_eft_test)
# T2.3 property/identity tests for FF: Group A bit-exact (no oracle), Group B/Test C
# use the __float128 oracle (runtime-SKIPs without LIBQUADMATH). FF analogue of
# dd_property_test. Plain helper — NOT an EFT test, so no contraction flags
# (identities are FMA-contraction agnostic).
kokkos_ep_add_test(ff_property_test)
# T2.4 differential accuracy vs quadmath for FF: per-op digits-of-accuracy against
# the __float128 oracle (runtime-SKIPs without LIBQUADMATH). FF analogue of
# dd_accuracy_test. Plain helper — NOT an EFT test, so no contraction flags
# (mirrors dd_accuracy_test's registration).
kokkos_ep_add_test(ff_accuracy_test)
# T2.5 end-to-end cancellation kernels for FF: quadmath-oracle-gated, host-side.
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
# Unlike the DD guard, the FF source uses an FP64 oracle (exact, no LIBQUADMATH), so
# both variants run unconditionally (no quadmath gate / SKIP-77).
kokkos_ep_add_eft_test(ff_fma_guard_test)              # -> ff_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(ff_fma_guard_test)  # -> ff_fma_guard_test_contract_on  (ON, reports)
# T3.1 EFT unit tests for QF: qf_two_sum / qf_quick_two_sum / qf_two_prod / qf_two_sqr
# (FP64 oracle, exact — 25-bit sum / 48-bit product fit in FP64's 53-bit mantissa)
# PLUS renorm_4 (len 5->4) + renorm (len 4->4), the QF-unique renormalization
# primitives (Priest non-overlap invariant + exact FP64 value-preservation on
# bounded-spread inputs, quadmath wide-spread truncation check behind the guard).
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
# runs unconditionally; __float128 only enriches inputs to ~96-bit width behind
# the quadmath gate. Plain helper (NOT the EFT helper): a per-word add/compare has
# nothing to contract, so no -ffp-contract=off posture is required. QF analogue of
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
# unconditionally; Group B / Test C are quadmath-gated (SKIP 77 otherwise). Plain
# helper (NOT the EFT helper): algebraic identities are FMA-contraction agnostic.
# QF analogue of ff_property_test (T2.3).
kokkos_ep_add_test(qf_property_test)

# T3.4 differential accuracy vs quadmath for QF: per-op digits-of-accuracy against
# the __float128 oracle (runtime-SKIPs without LIBQUADMATH). QF analogue of
# ff_accuracy_test (T2.4) / dd_accuracy_test (T1.4). Scores every QF op returning a
# QuadFloat with a quadmath counterpart across narrow(Route-A) + broad(~96-bit) +
# corpus passes; gates on the MEAN against the QF quad-word absolute ulp-of-U floor
# (10 ulp = 27.90 digits default; 30 ulp = 27.42 for the exp-family output-denormal
# tail, PORT_NOTES_QF §10), with low mins for §5-registry ops reported as
# EXPECTED-MIN-DROP. Plain helper — NOT an EFT test, so no contraction flags
# (mirrors dd_accuracy_test / ff_accuracy_test registration).
kokkos_ep_add_test(qf_accuracy_test)

# T3.5 FMA-contraction guard for QF: QF analogue of T2.5's ff_fma_guard_test, same
# source/two-postures dual-registration pattern (see qf_fma_guard_test.cpp). Reuses
# the SAME helpers as the DD/FF guards — the only per-variant knobs are the compile
# definitions the helpers set; the ON helper derives qf_fma_guard_baseline.txt from
# the target name. Unlike FF (which mirrors ff_math.hpp's embedded Dekker), this test
# calls the SHIPPED qf_math.hpp primitives (qf_two_prod / qf_two_sqr) directly, so it
# characterizes GCC's contraction of the shipped source itself (cf. T3.1 qf_eft_test,
# which established the same direct-call posture under contraction-OFF). Also guards
# qf_two_sqr (a second Dekker sequence, no FF analogue). FP64 oracle (exact, no
# LIBQUADMATH), so both variants run unconditionally (no quadmath gate / SKIP-77).
kokkos_ep_add_eft_test(qf_fma_guard_test)              # -> qf_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(qf_fma_guard_test)  # -> qf_fma_guard_test_contract_on  (ON, reports)

# T3.6 end-to-end cancellation kernels for QF: quadmath-oracle-gated, host-side.
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
kokkos_ep_add_test(tf_accuracy_test)                   # per-op digits vs quadmath oracle
kokkos_ep_add_test(tf_cancellation_test)               # e2e cancellation kernels (K1-K4)
kokkos_ep_add_eft_test(tf_fma_guard_test)              # -> tf_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(tf_fma_guard_test)  # -> tf_fma_guard_test_contract_on  (ON, reports)

# COMPLEX accuracy, all four backends. Until these landed, every target above
# scored REAL arithmetic only — `grep -c complex tests/CMakeLists.txt` was 0 —
# and the complex headers were covered solely by the demos, which print a table
# nobody diffs per-op against an oracle. That is how KI-1 (complex acosh returns
# the wrong branch for Re(z) < 0, in ALL FOUR backends) survived from the
# original DD port through FF, QF and TF. See docs/KNOWN_ISSUES.md.
#
# Standalone helper, NOT kokkos_ep_add_test: these read precomputed binary128
# references from tests/data/xp_corpus_complex.bin and therefore run on
# toolchains with no libquadmath, where the oracle-scored tests above only skip.
# 24 ops each (add sub mul div abs conj sqrt exp log log10 sin cos tan asin acos
# atan sinh cosh tanh asinh acosh atanh pow polar), per-op tolerance table set
# from measurement. NOT EFT tests, so no contraction flags — verified
# insensitive to -O0/-O2/-O3 and -ffp-contract=off/fast.
kokkos_ep_add_standalone_test(dd_complex_accuracy_test)
kokkos_ep_add_standalone_test(ff_complex_accuracy_test)
kokkos_ep_add_standalone_test(qf_complex_accuracy_test)
kokkos_ep_add_standalone_test(tf_complex_accuracy_test)
