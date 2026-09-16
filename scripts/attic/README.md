# scripts/attic/

Unbuilt historical tools kept for provenance. Several score against libquadmath,
and **their numbers are not comparable point-for-point with the sweep's**: the
sweep scores in ulps against an MPFR/MPC oracle with one verdict per point (see
`docs/CORRECTNESS.md`).

## Files

- **gen_corpus.cpp** — Shared corpus generator (39 real + 24 complex ops). Emits
  one set of inputs plus `__float128` reference per (op, element) so backends
  could be scored on identical data instead of each demo generating its own.
  Never integrated: no `CMakeLists.txt` references it, nothing consumes it, and
  the loader that paired with it (`tests/corpus_binary.hpp`) is gone. Not to be
  confused with `tests/corpus.hpp`, the T0.2 corner-case corpus consumed by
  `corpus_test` and the invariant tests.

- **gen_qf_constants.cpp** — Constant GENERATOR for QF. Its output is committed
  in the headers (`include/xp/qf_math.hpp`, `include/xp/qf_complex.hpp`), so it
  is the provenance of those constants and must not be deleted. Scores against
  libquadmath.

- **probe_complex_oracle.cpp** — Early probe tool for testing complex oracle
  behavior under libquadmath. Used during development to understand the
  `__complex128` oracle that the demos used to print. Scores against
  libquadmath.

- **probe_acos_branch.cpp** — Branch-selection probe for `acos` implementation.
  Used to verify which branch of the complex `acos` algorithm is taken for
  different inputs. Scores against libquadmath.

- **probe_cpow.cpp** — Complex power probe tool. Used to explore `cpow` behavior
  and edge cases under libquadmath during the complex transcendental
  implementation. Scores against libquadmath.

- **test_qfmul.cpp** — Early QF multiplication test harness. Used during initial
  QF development to verify the multiply algorithm in isolation. Scores against
  libquadmath.
