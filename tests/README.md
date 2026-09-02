# `tests/` — extended-precision test suite

This directory holds the backend-parameterized test harness for the portable
extended-precision backends benchmarked in this repo. The authoritative spec is
[`docs/TEST_SUITE_PLAN.md`](../docs/TEST_SUITE_PLAN.md) — this README is the
operational quick-reference for the harness created in **T0.1**.

## Purpose

Each backend (DD today; FF, QF in later phases) is validated across six test
layers: EFT unit tests, non-overlap invariants, property/identity tests,
differential accuracy vs a `__float128` oracle, FMA-contraction guards, and
end-to-end cancellation kernels. The harness in `test_utils.hpp` provides the
shared plumbing so a single test file can target any backend without
duplication.

Two scaffolding tests exercise the harness itself: `hello_test.cpp`, a smoke test
that exercises the harness end-to-end (Kokkos init, input generation, host↔device
copy, oracle comparison, reporting) on a trivial DD round-trip identity; and
`corpus_test.cpp`, which validates the corner-case corpus (`corpus.hpp`) itself.
Neither runs a real DD math op.

Real DD correctness coverage begins in Phase 1 with `dd_eft_test.cpp` (**T1.1**),
the Layer-1 EFT unit test (see [EFT tests](#eft-tests-layer-1) below).

## Registered tests

| Test              | Layer / task | What it covers                                        |
|-------------------|--------------|-------------------------------------------------------|
| `hello_test`      | T0.1         | Harness plumbing on a trivial DD round-trip identity  |
| `corpus_test`     | T0.2         | Corner-case corpus (`corpus.hpp`) scaffolding         |
| `dd_eft_test`     | T1.1         | EFT bit-exactness: DD `twoSum` + Dekker `twoProduct`  |
| `ff_eft_test`     | T2.1         | EFT bit-exactness: FF `twoSum` + Dekker `twoProduct` (splitter `8193.0f` = 2¹³+1); **FP64 oracle** — exact, no LIBQUADMATH needed, runs unconditionally |
| `dd_invariant_test` | T1.2       | Non-overlap invariant `fl(hi+lo)==hi` for **every** DD op (unary/binary/ternary/two-output); oracle-independent (no `__float128`, runs without LIBQUADMATH) |
| `ff_invariant_test` | T2.2       | Non-overlap invariant `fl(hi+lo)==hi` for **every** FF op, evaluated in **raw FP32**; FP32-narrower domain predicates derived from `ff_math.hpp`; oracle-independent (no `__float128`, runs without LIBQUADMATH) |
| `dd_property_test` | T1.3        | Algebraic identities: **Group A** bit-exact (no oracle, e.g. `a·1==a`, `a-a==0`), **Group B** tolerance vs `__float128` (e.g. `sqrt(a)²≈a`, `sin²+cos²≈1`), **Test C** named-constant regressions (`sin(π)≈0`, …) |
| `ff_property_test` | T2.3        | FF analogue of `dd_property_test`: **Group A** bit-exact (7 identities, no oracle), **Group B** tolerance vs `__float128` (13 identities, **mean-gated** at −log10(N·u²) ≈ 8.45 for u=2⁻²⁴), **Test C** named-constant regressions (target ≥12 of FF's 14 digits); exp-guard-narrowed domains for the exp round-trips; runtime-SKIPs without LIBQUADMATH |
| `dd_accuracy_test` | T1.4        | Differential accuracy vs `__float128`: per-op digits of accuracy over 10⁶ random + corpus; **fail-gates on MEAN** ≥ −log10(N·u²) ≈ 25.91; PORT_NOTES §5 conditioning-limited ops report **EXPECTED-MIN-DROP** (gated on mean, not min); runtime-SKIPs without LIBQUADMATH |
| `ff_accuracy_test` | T2.4        | FF analogue of `dd_accuracy_test`: per-op digits of accuracy over 10⁶ random + corpus vs `__float128`; **fail-gates on MEAN** ≥ −log10(N·u²) ≈ 8.45 (u=2⁻²⁴, cap 14); FP32-narrower op domains from T2.2; shared PORT_NOTES §5 registry → **EXPECTED-MIN-DROP**; runtime-SKIPs without LIBQUADMATH |
| `dd_e2e_test`     | T1.6         | End-to-end cancellation kernels: √(x²+1)−x, Σ1/k², Machin's π, alternating harmonic — all quadmath-oracle-gated |
| `ff_cancellation_test` | T2.5    | FF analogue of `dd_e2e_test`: same four cancellation kernels (√(x²+1)−x, Σ1/k², Machin's π, alternating harmonic) scored in digits vs `__float128`/closed-form oracles; **mean-gated at 14−3 = 11.0** (FF's cap minus headroom); K1 naive-vs-stable compares FF against **FP32** (FF's base scalar) at x∈{1e2,1e4,1e6}; runtime-SKIPs without LIBQUADMATH |
| `dd_fma_guard_test` | T1.5       | FMA-contraction guard, **contraction OFF** — same Dekker `twoProduct` built `-ffp-contract=off`; **fail-gates** on any mismatch (stronger form of T1.1) |
| `dd_fma_guard_test_contract_on` | T1.5 | FMA-contraction guard, **contraction ON** — the *same source* built `-ffp-contract=fast`; **reports only** (always exits 0), prints the mismatch count and warns on drift vs `dd_fma_guard_baseline.txt` |
| `ff_fma_guard_test` | T2.5 | FF analogue of `dd_fma_guard_test`, **contraction OFF** — same Dekker `twoProduct` (splitter `8193.0f` = 2¹³+1) built `-ffp-contract=off`; **fail-gates** on any mismatch. **FP64 oracle** (exact — 48-bit product fits FP64's 53-bit mantissa), so runs **unconditionally** (no LIBQUADMATH gate, no SKIP-77) |
| `ff_fma_guard_test_contract_on` | T2.5 | FF analogue, **contraction ON** — the *same source* built `-ffp-contract=fast`; **reports only** (always exits 0), prints the mismatch count and warns on drift vs `ff_fma_guard_baseline.txt` |
| `qf_eft_test`     | T3.1         | EFT bit-exactness for QF: `qf_two_sum` / `qf_quick_two_sum` / `qf_two_prod` / `qf_two_sqr` (splitter `8193.0f` = 2¹³+1; **FP64 oracle**, exact, no LIBQUADMATH) **plus** the QF-unique `renorm_4` (len 5→4) and `renorm` (len 4→4): Priest non-overlap invariant + **exact FP64 value-preservation** on bounded-spread inputs, and a wide-spread `__float128` truncation check (rel ≤ 2⁻⁸⁸) behind the quadmath guard. Calls the **shipped** `qf_math.hpp` free functions directly (no mirror-and-comment). Contraction OFF |
| `qf_nonoverlap_test` | T3.2      | Priest **length-4** non-overlap invariant `\|f_{i+1}\| ≤ ½ ulp(f_i)` (i=0,1,2, mathematical ½-ulp form via `frexp`) on the **output** of every QF op returning a `QuadFloat` (arithmetic incl. `ieee_add`/`sloppy_add`/`divide_accurate`, `sqr`/`sqrt`, all transcendentals, joint `sincos`/`sinhcosh` components, `angle`, `multiply_scalar`, `mul_pwr2` **(±2ᵏ only)**, `fma`, `pow_int`, round family, utilities); oracle-independent (runs without LIBQUADMATH), `__float128` only **enriches inputs** to ~96-bit ordered width behind the quadmath gate. QF analogue of `ff_invariant_test`; trig domain has **no tiny-arg lower bound** (QF `sincos` has no FFCSSNR stall). Two-tier classifier (`NOVL_OK`/`NOVL_WEAK`/`NOVL_FAIL`); **ships RED under the default strict Priest gate** because QD `renorm` gives only Shewchuk-weak `≤ ulp` non-overlap — flip `kStrictPriestGate` for the weak gate (see §QF non-overlap). Plain helper (no contraction posture) |
| `qf_property_test` | T3.3         | QF analogue of `ff_property_test`: **Group A** 12 bit-exact identities (no oracle — additive inverse/zero, `a·0`/`a·1`/`a·(-1)`, double negation, `abs` sign branches/`abs(-a)`, **add commutativity bit-exact on WIDE 4-word operands** — a QF strengthening over FF/DD's single-word restriction — and the `mul_pwr2` ±2ᵏ round-trip), **Group B** 15 tolerance identities vs `__float128` (**mean-gated** at an absolute **ulp of U=2⁻⁹⁶** floor — 10 ulp = 27.90 digits default, 30 ulp = 27.42 for the exp-denormal-tail-limited `exp(log)`/`pow(x,2)` per PORT_NOTES_QF §10 — **NOT** the DD/FF `−log10(N·u²)` statistical floor, since QF is a quad-word whose resolution IS U), **Test C** named-constant regressions (target ≥27 of QF's 29 digits). Includes the T2.3 **B4 exp-eps** pattern (`exp(a+eps)≈exp(a)(1+eps)`) — which does **NOT** recur in QF (`qf_math.hpp` exp uses `eps=1e-28f` coarser than U). Group A unconditional; Group B / Test C runtime-SKIP without LIBQUADMATH. Plain helper (no contraction posture) |
| `qf_accuracy_test` | T3.4         | QF analogue of `ff_accuracy_test` (T2.4) / `dd_accuracy_test` (T1.4): per-op digits of accuracy vs `__float128` for **every** QF op returning a `QuadFloat` with a quadmath counterpart (49 scored + 5 skipped = T3.2's 54). Three passes per op — **narrow** (Route-A over the §11 demo domains) + **broad** (`make_wide_input` ~96-bit ordered) + **corpus** (named §3/§4 accessor where one exists) — combined into one (min, mean, n). **Fail-gates on the MEAN** against the QF quad-word **absolute ulp-of-U** floor (10 ulp = **27.90** default; 30 ulp = **27.42** for the exp-family output-denormal tail, PORT_NOTES_QF §10) — **NOT** the DD/FF `−log10(N·u²)` statistical floor (QF is a quad-word whose resolution IS U=2⁻⁹⁶); low mins for the shared PORT_NOTES §5 registry ops → **EXPECTED-MIN-DROP**. Oracle evaluated at the **exact widened input** `qf_to_q(input)` (QF's ~29 digits are finer than a double, so the broad regime's sub-double bits matter — a QF-specific twist over DD/FF's `(float128)x` oracle). exp/exp2/exp10 domains narrowed to keep the quad-word result in FP32 normal range (§10). Runtime-SKIPs without LIBQUADMATH. Plain helper (no contraction flags) |
| `qf_fma_guard_test` | T3.5 | QF analogue of `ff_fma_guard_test`, **contraction OFF** — the **shipped** Dekker EFTs `qf_two_prod` **and** `qf_two_sqr` (splitter `8193.0f` = 2¹³+1) built `-ffp-contract=off`; **fail-gates** on any collapsed/wrong error term (a stronger form of T3.1). Calls the shipped `qf_math.hpp` primitives **directly** (no mirror-and-comment, cf. `qf_eft_test`). **FP64 oracle** (exact — 48-bit product fits FP64's 53-bit mantissa), so runs **unconditionally** (no LIBQUADMATH gate, no SKIP-77). `qf_two_sum` included as a contraction-immune **control** |
| `qf_fma_guard_test_contract_on` | T3.5 | QF analogue, **contraction ON** — the *same source* built `-ffp-contract=fast`; **reports** per-op three-way `{ERR_ZERO / ERR_NONZERO_CORRECT / ERR_NONZERO_WRONG}` counts and warns on drift vs `qf_fma_guard_baseline.txt`. **PASS iff `ERR_NONZERO_WRONG == 0`** (any `ERR_ZERO`/`CORRECT` mix is acceptable — collapse under contraction is *informative*, not a fault of `qf_math.hpp`) |
| `qf_cancellation_test` | T3.6    | QF analogue of `dd_e2e_test` / `ff_cancellation_test`: same four cancellation kernels (√(x²+1)−x, Σ1/k², Machin's π, alternating harmonic) scored in digits vs `__float128`/closed-form oracles; **mean-gated at 29−3 = 26.0** (QF's cap minus headroom); K1 naive-vs-stable compares QF against **FP32** (QF's base scalar) at x∈{1e2,1e4,1e6}; runtime-SKIPs without LIBQUADMATH |
| `dd_complex_accuracy_test` | — | Per-op accuracy gate for the **24 complex ops** of `dd_complex.hpp` (add sub mul div abs conj sqrt exp log log10 sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow polar). **Corpus-scored, not oracle-linked**: reads binary128 references from `data/xp_corpus_complex.bin` and needs **neither `-lquadmath` nor `-fext-numeric-literals`** — see [Decoupled complex accuracy tests](#decoupled-complex-accuracy-tests) below. Per-element score is `min(d_re, d_im)`; **fail-gates on the MEAN** against a per-op table set from measurement with ~0.3 digits of margin. `acosh` excludes `Re(z) < 0` (**KI-1**, wrong branch in all four backends) by domain predicate. Runs unconditionally on every toolchain |
| `ff_complex_accuracy_test` | — | FF analogue, cap 14. `FloatFloat(double)` keeps only ~48 bits, so FF alone is scored on a slightly different input than the oracle saw — which is why its `add`/`sub` gates sit at 13.13/13.14 where DD/QF/TF hold those bit-exact |
| `qf_complex_accuracy_test` | — | QF analogue, cap 29 |
| `tf_complex_accuracy_test` | — | TF analogue, cap 21.7 |

## Decoupled complex accuracy tests

The four `*_complex_accuracy_test` targets are the only tests in the suite that
score against a `__float128` oracle **without linking libquadmath**. The split:

- **Generator** (`scripts/gen_corpus.cpp`, host-only, links `-lquadmath`) emits
  inputs plus a `__complex128` reference per (op, element) into a versioned binary.
  `tests/data/xp_corpus_complex.bin` is that file, committed: 24 ops × 2000
  elements, seed 12345, checksum `0xdec61578158e39e7`.
- **Consumer** (`tests/corpus_binary.hpp`) decodes each 16-byte binary128 into an
  exact 3-double expansion plus a separately-held binary exponent (159 bits ≥
  binary128's 113), and scores in plain `double`. No `<quadmath.h>`, no
  `__float128` required — the header's `ref_to_float128()` convenience is guarded
  behind `KOKKOS_EP_CORPUS_HAVE_FLOAT128` and nothing depends on it. The decoder
  was validated bit-exactly against `__float128` over 220k random binary128
  patterns and 200k perturbation trials.

Why it matters: hipcc and icpx are clang-based and have no GCC-flavoured quadmath,
so every other oracle-scored test runtime-SKIPs (exit 77) there and an AMD or Intel
run reports **no accuracy signal at all**. These four run everywhere. They are
registered through `kokkos_ep_add_standalone_test()`, which deliberately does not
link `Kokkos::kokkos`, does not apply `KOKKOS_EP_QUADMATH_DEFINE`, and sets no
`SKIP_RETURN_CODE`.

Regenerating the corpus (only needed if the op inventory or sampling changes):

```bash
g++ -std=c++17 -O2 -fext-numeric-literals -I include scripts/gen_corpus.cpp \
    -lquadmath -o gen_corpus
./gen_corpus --complex-only --n 2000 --out tests/data/xp_corpus_complex.bin
```

The loader hard-FAILS (it does not skip) on a missing file, a format-version
mismatch, a generator-version mismatch or a checksum mismatch, so a stale corpus
cannot quietly turn into a green test.

## How to run

Configure and build (from the repo root), then run the tests:

```bash
# Configure + build (Kokkos must be installed; see top-level README/CLAUDE.md)
cmake -B build -DCMAKE_PREFIX_PATH=<kokkos-install-dir>
cmake --build build -j

# Run the test suite
ctest --test-dir build -V
```

A passing run shows:

```
    Start 1: hello_test
1/1 Test #1: hello_test .......................   Passed
```

Tests build under `build/tests/`; you can also run a binary directly, e.g.
`./build/tests/hello_test`.

## How to add a new test

Five-line recipe:

1. Create `tests/foo_test.cpp`.
2. `#include "test_utils.hpp"` and write `main()` (init Kokkos, call the runners,
   use `KOKKOS_EP_ASSERT`, `return kokkos_ep::ep_exit_code();`).
3. Add one line to `tests/CMakeLists.txt`:  `kokkos_ep_add_test(foo_test)`.
4. Rebuild: `cmake --build build -j`.
5. Run: `ctest --test-dir build -V` — `foo_test` now appears.

`kokkos_ep_add_test(<name>)` compiles `tests/<name>.cpp`, links
`Kokkos::kokkos` + the third-party include dir, applies the quadmath define, and
registers the test with CTest (with `SKIP_RETURN_CODE=77`).

**Coverage recipe (Phase 1+):** a per-op test should run **two passes**. First a
random-generator pass via `run_unary_op` / `run_binary_op` (breadth). Then a
**corpus pass** via `run_unary_op_on_corpus` / `run_binary_op_on_corpus` fed from
`corpus.hpp` — prefer the **named accessor** for the op's known failure family
(so a failure cites a PORT_NOTES bug) and fall back to the `unary<T>()` /
`binary<T>()` bundler for broad invariant sweeps. When reporting the min digit
count, consult `lookup_expected_min_drop(op_name)` so PORT_NOTES §5
conditioning-limited ops report "expected-min-drop: OK" instead of failing.

## Backends and how tags map to types

The harness is templated over a **backend tag** via `BackendTraits<Tag>`:

| Tag  | Type            | Digits | Status                                  |
|------|-----------------|--------|-----------------------------------------|
| `DD` | `dd::DoubleDouble` | ~31 | supported (this branch)                 |
| `FF` | `ff::FloatFloat`   | ~14 | Phase 2 — adds `BackendTraits<FF>`      |
| `QF` | `qf::QuadFloat`    | ~29 | Phase 3 — adds `BackendTraits<QF>`      |

`DD` etc. are empty tag types, **not** the arithmetic types. `BackendTraits<DD>`
exposes `type` (`dd::DoubleDouble`), `u_squared`, `max_digits`, `name()`, and
`to_quad()`. A test written against `BackendTraits<Backend>` instantiates across
backends with no source duplication; Phase 2/3 add the FF/QF specializations.

## Corpus (`corpus.hpp`)

Uniform random inputs miss the pathological cases that actually break
extended-precision code — `PORT_NOTES.md` on branch `fffunKokkos` documents two
FF bugs (§4a `exp` NaN above input 79.4, §4b `ffnint` off-by-one at 19.4999…)
that slipped through the demo's own accuracy table precisely because random
almost never lands on them. `corpus.hpp` (T0.2) supplies a deterministic
corner-case corpus so, from Phase 1 onward, **every** test layer runs a random
pass *and* a corpus pass and those inputs are always exercised.

`corpus.hpp` is pure **data** plus a tiny API to iterate it — it is not tests. It
is downstream-only (no SPDX header, unlike `dd_math.hpp`). Entries are
precision-parametric: templated on the scalar type (`double` for DD today,
`float` for FF/QF later). Returns are materialized `std::vector<T>` (unary) /
`std::vector<std::pair<T,T>>` (binary), not `InputDist` generators — corpus
entries are fixed constants, so one vector element == one deterministic test
input.

**Categories:** subnormals, ±0, ±inf, quiet NaN (opt-in), powers of two,
`nextafter` neighbors, near-cancellation pairs, huge/tiny magnitude mixes,
half-integer boundaries. Plus the explicit PORT_NOTES §3/§4 regression families.

### Two API styles — pick per test

- **Bundlers** — `corpus::unary<T>(flags)` / `corpus::binary<T>(flags)`: "throw
  the whole corpus at this op." Use for the **T\*.2 invariant tests**, which just
  want broad coverage and don't care which category a failure came from. The
  `CorpusFlags` struct gates whole classes (`include_nan` defaults false;
  `include_inf`/`include_zero`/`include_subnormals` default true) and is
  authoritative over the *entire* assembled bundle, including members other
  categories emit incidentally (e.g. `nextafter(0,+inf)` = `denorm_min`).

- **Named accessors** — `corpus::exp_overflow<T>()`,
  `corpus::nint_half_integer<T>()`, `corpus::remainder_regression<T>()`,
  `corpus::atanh_small<T>()`, `corpus::sinh_cosh_small<T>()`,
  `corpus::trig_near_pi<T>()`, and the category accessors (`subnormals<T>()`,
  `powers_of_two<T>()`, …): grab exactly the family an op needs. Use for the
  **T\*.4 accuracy tests** so a failure cites a specific PORT_NOTES bug
  ("`exp_overflow` item 3") rather than "corpus item 47".

The corpus-pass **runners** live in `test_utils.hpp`:
`run_unary_op_on_corpus(inputs, host_oracle, device_op)` and
`run_binary_op_on_corpus(pairs, host_oracle, device_op)` — same
host→device→host→oracle pipeline as the random-pass runners, but driven by a
corpus vector instead of `(seed, n)` + generator.

### `lookup_expected_min_drop` — conditioning limits vs real regressions

Some ops legitimately show a low **min** digit count that is **not** a
regression: the operation is conditioning-limited and no fixed-precision
algorithm can do better (PORT_NOTES §5 — e.g. `sub`/`fdim`/`fma` under exact
cancellation, `asin`/`acos`/`atanh` near `|a|=1`, `remainder` near a multiple of
`b`, `exp` in the output-denormal range, `sin`/`cos`/`tan` near ±π). Tests
**fail-gate on the mean** column but must **not** fail on the min for these ops.

`test_utils.hpp` provides a registry:

```cpp
const ExpectedMinDropAnnotation* ann = lookup_expected_min_drop(op_name);
if (ann && stats.min >= ann->min_digits_allowed) {
  // "expected-min-drop: OK" — cite ann->reason
} else {
  // fail-gate on mean as usual
}
```

`lookup_expected_min_drop("sub")` returns non-null (with a `reason` string);
`lookup_expected_min_drop("add")` returns null (add is not conditioning-limited).

**Source of truth:** `PORT_NOTES.md` on branch `fffunKokkos` — §4 (two outright
bugs → the named regression accessors) and §5 (conditioning limits → the
expected-min-drop registry). The FF-side corpus specialization happens in Phase 2.

## EFT tests (Layer 1)

Layer 1 of the six-layer suite validates the two **error-free transforms** that
every double-word operation is built on:

- `twoSum(a, b) -> (s, e)`: `s = fl(a+b)` and `e = (a+b) - s` **exactly**.
- `twoProd_Dekker(a, b) -> (p, e)`: `p = fl(a*b)` and `e = a*b - p` **exactly**.

`dd_eft_test.cpp` (T1.1) tests these at the raw-`double` level — the twoSum
embedded in `DoubleDouble` `add` and the Dekker twoProduct embedded in `multiply`
(mirrored into the test file for RAW doubles; `dd_math.hpp` is not modified). If
either EFT is not bit-exact, nothing downstream (sqrt/exp/log/…) is trustworthy,
so this layer runs first. Ground truth is `__float128`, which is **provable, not
approximate**: the exact FP64 sum needs ≤54 bits and the exact FP64 product ≤106
bits, both of which fit in binary128's 113-bit mantissa, so widening the operands
and summing/multiplying in `__float128` is exact.

Inputs outside each transform's proven domain are **skipped, not failed**: twoSum
skips only non-finite pairs and sums that overflow; Dekker twoProduct additionally
skips subnormal operands, splitter-overflow magnitudes (`|x| ≥ 2^996`), and
products that overflow or gradually underflow (error term would fall subnormal) —
these are documented limits of Dekker's method, not defects in `multiply`.

### Contraction-off requirement

EFT tests **must** compile with FMA contraction disabled. Dekker's twoProduct
depends on `a1*b1 - c11` being two distinct rounded operations; if the compiler
fuses them into a single FMA, the error term collapses to zero and the transform
silently breaks — the test would then validate a transform the shipped binary does
not perform. `tests/CMakeLists.txt` provides a helper for this:

```cmake
kokkos_ep_add_eft_test(dd_eft_test)
```

`kokkos_ep_add_eft_test(<name>)` is `kokkos_ep_add_test(<name>)` plus per-target
contraction-off flags: `-ffp-contract=off` (GNU/Clang, on `COMPILE_LANGUAGE:CXX`),
`-fp-model=precise` (Intel), and `--fmad=false` (nvcc, on `COMPILE_LANGUAGE:CUDA`,
applied only when `Kokkos_ENABLE_CUDA`). Applied per target, not globally, so the
demos and other test layers keep the project's normal flags. Reuse this helper for
the future FF (T2.1) and QF (T3.1) EFT tests. (T1.5 later builds the full
contraction on/off regression matrix; T1.1 only needs the posture in place so its
own results are meaningful.)

### FF EFT (Layer 1, Phase 2)

`ff_eft_test.cpp` (T2.1) is the FF analogue of `dd_eft_test.cpp`. It tests the
same two transforms at the raw-`float` level — the twoSum embedded in `FloatFloat`
`add` and the Dekker twoProduct embedded in `multiply`
(≡ the standalone `two_prod`), mirrored into the test
file for RAW floats; `ff_math.hpp` is not modified. It reuses the same four corpora
(broad random `[-1e30f,1e30f]`, narrow random `[-1,1]`, `|a|≫|b|` with `k∈[1,20]`,
full `corpus::unary<float>()` cross-product), named hard cases, and device-parity
pass.

The one material difference from T1.1 is the **oracle**. For FF the ground truth is
plain **FP64**, not `__float128`: the exact FP32 sum needs ≤25 bits and the exact
FP32 product ≤48 bits, both of which fit in FP64's 53-bit mantissa, so widening the
operands and summing/multiplying in `double` is **algebraically exact** — a
*stronger* oracle than DD's quadmath (exact, not merely higher-precision), and one
that needs no external library. So `ff_eft_test` carries **no `KOKKOS_EP_HAVE_QUADMATH`
gate and no runtime SKIP-77** — it runs on every build.

FP32-specific domain skips (out-of-domain, **not** failures): twoSum skips only
non-finite pairs and sums that overflow FP32; Dekker twoProduct additionally skips
subnormal operands, splitter-overflow magnitudes (`|x| ≥ FLT_MAX / (2¹³+1) ≈
2^115`, derived from the `a * 8193.0f` split — this is PORT_NOTES §4a's `exp`
splitter-overflow mechanism), and products that overflow or gradually underflow
(error term `< 2⁻¹⁰²` would fall into FP32 subnormals). Registered with the same
`kokkos_ep_add_eft_test(ff_eft_test)` (contraction OFF); the contraction-ON reporter
mirror is T2.5.

> **Splitter naming.** The shipped FF splitter is `8193.0f`, which is **2¹³ + 1**
> (not 2¹² + 1 = 4097). The comment above `multiply`'s splitter states this
> correctly; a stale
> "2^12+1" typo in the `ff_math.hpp` license header and in the T2.1 task text is
> noted but not fixed here (T2.1 does not modify `ff_math.hpp`).

### QF EFT (Layer 1, Phase 3)

`qf_eft_test.cpp` (T3.1) is the QF analogue of `ff_eft_test.cpp`. It tests the FP32
error-free transforms QuadFloat composes on, plus the QF-unique renormalization
primitives. **Two structural differences from T2.1:**

1. **No mirror-and-comment.** `ff_eft_test` had to *duplicate* FF's twoSum / Dekker
   twoProduct into the test because `ff_math.hpp` embeds them inside longer
   `add`/`multiply` sequences. `qf_math.hpp` instead **exposes** the shipped
   primitives as free functions in `Kokkos::Experimental` — `qf_two_sum`,
   `qf_quick_two_sum`, `qf_two_prod`, `qf_two_sqr` and
   `renorm` / `renorm_4` — so this test calls the **actual
   shipped code**, a strictly stronger check (a mirror can drift; a direct call
   cannot). `qf_math.hpp` is not modified (rule 4).
2. **`renorm_4` has no FF analogue.** FF's two-word type never renormalizes a wide
   expansion; QF's `renorm_4` (len 5→4) and `renorm` (len 4→4) are genuinely
   QF-unique surface, and their oracle strategy is T3.1-original.

**twoSum/twoProd oracle** is the same provable **FP64** as T2.1 (25-bit sum / 48-bit
product both fit FP64's 53-bit mantissa), so that half runs unconditionally with no
LIBQUADMATH gate. FP32 domain skips (splitter overflow `|x| ≥ FLT_MAX/8193 ≈ 2^115`,
subnormal operands, underflow tail `< 2⁻¹⁰²`) are inherited verbatim from
`ff_eft_test`. `qf_quick_two_sum` is tested with operands ordered `|a| ≥ |b|` (its
precondition); `qf_two_sqr` reuses the twoProd domain on `(a,a)`.

**`renorm` oracle (T3.1-original).** Two value-preservation regimes:

- **Bounded spread (exact FP64, unconditional):** input words drawn inside a common
  ≤29-bit exponent window, so the exact real sum spans ≤53 bits — it fits *exactly*
  in FP64 **and** within QF's 96-bit capacity, so `renorm` drops nothing and
  `(double)(Σ out) == (double)(Σ in)` is a **provable bit-equality**. This is the
  primary gate.
- **Wide spread (quadmath, behind the guard):** words span the full ~96-bit range so
  `renorm` genuinely truncates; the residual is checked against QF's truncation
  threshold **rel ≤ 2⁻⁸⁸** (256× margin above `u = 2⁻⁹⁶`, per `qf_math.hpp`'s
  header precision note). SKIPs
  cleanly without LIBQUADMATH — the exact FP64 bounded test still gates.

The **Priest non-overlap invariant** `|f_{i+1}| ≤ ½ ulp(f_i)` (bit form
`fl(f_i + f_{i+1}) == f_i`) is checked on every `renorm` output, oracle-independent,
with the same underflow-tail gate (`< 2⁻¹⁰⁰`) as `ff_invariant_test` (T2.2), plus a
packing check (no nonzero word after a zero word). Named cases cover inf/nan
propagation through `renorm` (its `if (isinf(c0)) return;` guard must not crash).
Device parity (Test E) re-runs `qf_two_sum` / `qf_two_prod` / `renorm_4` in a
`parallel_for`. Registered with `kokkos_ep_add_eft_test(qf_eft_test)` (contraction
OFF); the contraction-ON reporter mirror is T3.5.

## Non-overlap invariant (Layer 2)

Layer 2 (`dd_invariant_test`, **T1.2**) checks a **structural** property of every
DD op's *output* rather than its accuracy: a double-double `(hi, lo)` must be
**non-overlapping**, i.e. `lo` carries only bits below the last bit of `hi`. The
bit-exact statement of that, evaluated in **raw FP64** (a single hardware add +
compare), is

```
fl(hi + lo) == hi          (equivalently  |lo| <= 1/2 ulp(hi))
```

If `lo` held any bit at or above `hi`'s ulp, the rounded sum would land on a
different double and the equality would fail — localizing a normalization bug to
the exact op. Because this is a statement *about* FP64 rounding, the check is
deliberately **not** a `__float128` promotion (that would test the exact real
sum, a different thing). So this layer carries **no oracle and no
`KOKKOS_EP_HAVE_QUADMATH` guard**: it runs even on a quadmath-less Kokkos.
Accuracy-vs-oracle is the separate concern of T1.4.

**Coverage.** Every DD op that returns a double-double — unary, binary, ternary
(`fma`), two-output (`sincos`/`sinhcosh`, each output checked separately), and
`pow_int(dd,int)`. Each op runs two passes: 10^6 op-appropriate random inputs and
a full `corpus.hpp` pass (`include_zero=true`, `include_inf/nan=false`). Results
outside an op's domain (NaN/inf/subnormal `hi`, or out-of-domain input) are
**skipped, not failed**. Five ops (`add`, `multiply`, `sqrt`, `exp`, `sin`) also
run a device pass. Registered with the plain `kokkos_ep_add_test` helper — no
contraction flags (the invariant holds regardless of FMA contraction).

### FF non-overlap invariant (Layer 2, Phase 2)

`ff_invariant_test.cpp` (**T2.2**) is the FF analogue of `dd_invariant_test.cpp`,
mirroring its structure verbatim (50-row op inventory, two-pass random+corpus
shape, skip-not-fail domain gating, per-op reporting, 5-op device pass, Test C
PORT_NOTES §4 regressions). The one type change: the invariant is evaluated in
**raw FP32** — `(f.hi + f.lo) == f.hi` in `float` — where the DD test uses
`double`. Still **no** `__float128` promotion (that tests the exact real sum, a
different property) and **no** `KOKKOS_EP_HAVE_QUADMATH` gate: like T1.2 it runs
unconditionally. Every op in `ff_math.hpp` that returns a `FloatFloat` (or two via
out-params for `sincos`/`sinhcosh`) has a corresponding FF entry — no DD op is
missing on the FF side. Registered with the plain `kokkos_ep_add_test` helper (no
contraction flags).

**FP32-narrower domain predicates.** Every predicate was **re-derived** from the
shipped `ff_math.hpp` guards (not copied from T1.2) and empirically confirmed to
emit **zero** internal diagnostics. FP32's exponent range is ~6× narrower than
FP64, and the port has FP32-specific hazards T1.2's FP64 code never hit. The
material tightenings:

- `exp` gates at `a.hi < 88.0` (FP32 ln-range guard in `ff_math.hpp`), not DD's
  300; `exp2`/`exp10` scale accordingly (`|a|<126` / `|a|<38`).
- **trig family** (`sin`/`cos`/`tan`/`atan`, plus `atan2` and the `tgamma`
  reflection path) carries a **lower** magnitude bound (`|x| ≥ 1e-25`, or exactly
  0): FF's `sincos` Taylor loop hits its iteration limit (`FFCSSNR`) for tiny
  nonzero arguments because `r = x/2^nq` underflows at FP32. DD's FP64 `sincos`
  never saw this, so this floor has no T1.2 counterpart.
- `atan2` additionally floors **both** operands away from `|·|<1e-18` (0 allowed):
  a subnormal-tiny operand paired with a normal one drives the internal `sincos`
  degenerate — an FF-specific tightening of DD's larger-magnitude-only gate.
- `log`-family window `[1e-34, 1e34]` (keeps `|log x| < ~78`, inside `exp`'s
  88-guard); `sinh`/`cosh` `|x|<40`, `tanh` `|x|<20`, `tgamma` `x∈[1e-3, 23)`,
  `asinh`/`acosh`/`atan` upper caps `1e18` (so `x·x` stays finite in FP32).

**Test C — PORT_NOTES §4 regressions.** The `exp` §4a cases split into **two
distinct roles** (kept explicit in the test output so readers don't conflate them):

- **79.5 / 80 / 85 — bug-regression cases.** The historical §4a bug was
  NaN-from-splitter-overflow (`exp`'s internal Dekker split did `b * 8193.0f`,
  which overflowed FP32 → NaN). These three are the **load-bearing** regression
  cases: pre-fix they returned NaN; post-fix (direct scaling) they return finite,
  invariant-clean results. No diagnostic expected.
- **88.7 / 88.72 — edge-of-saturation guard cases.** These sit **past** the shipped
  `a.hi ≥ 88` guard and do *not* re-test the §4a bug; they assert the guard fires
  sensibly at the edge — saturating to **+0** (invariant trivially holds, not-NaN)
  rather than producing garbage. Each emits one `FFEXP: argument too large` print;
  those **2** diagnostics are the *only* internal `ff_math.hpp` output in the whole
  run (Test A/B are diagnostic-clean) and are **expected and normal** — documented
  safety-guard behavior is a pass, not a report-and-stop.

`round_to_nearest_int` at 19.4999993 and the k±0.5 family; `remainder(68.379,
3.5066)` gated against `std::remainderf` (the **same-precision** FP32 oracle).

> **Finding (remainder sign).** The T2.2 prompt (following PORT_NOTES §4b) expected
> a *positive* FP32 remainder here, on the premise that `a/b ≈ 19.4999993 < 19.5`
> at FP32 → `nint=19`. That premise does **not** hold for the corpus literals: at
> FP32 `68.379f/3.5066f = 19.5000858` (**>** 19.5), so the correct `nint` is 20 and
> the correct remainder is **negative** (−1.75300026). `std::remainderf` agrees,
> and the shipped FF `remainder` reproduces it exactly. The §4b "+1.7533" text
> describes a different rounding of `a/b` than these specific literals produce. The
> test gates against `std::remainderf` (not `std::remainder`, which would compare an
> FP32 op to the FP64 answer) and passes. See the Test C comment for the full
> derivation.

> **Finding (nint literal).** `19.4999993f` rounds to **exactly** `19.5f` at FP32
> (`lo=0`), so `round_to_nearest_int` of the pure-float value returns 20 — correct.
> The historical `19`-vs-`20` distinction only appears when the full-precision value
> `19.4999993` is carried in the FF pair via the Route-A double split (`hi=19.5`,
> `lo=−7e-7`, total < 19.5), where the fixed `ffnint` (rounding `hi+lo` in FP64)
> returns 19. Test C checks both constructions.

> **Finding (underflow-tail ties → `kUnderflowTail` skip).** The bulk random+corpus
> pass initially reported "failures" for `exp`/`exp2`/`exp10` (and `device:exp`) on
> **very negative** arguments — e.g. `exp(-84.32) → hi=2.40e-37, lo=-1.121e-44`.
> These are **not** normalization defects. Each failing `lo` is a **subnormal** that
> lands *exactly* on the `½ ulp(hi)` tie point (`-1.121e-44 = -½·2⁻¹⁴⁵` for that
> `hi`), where round-to-even flips `fl(hi+lo)` off `hi` by one ulp **even though the
> mathematical non-overlap `|lo| ≤ ½ ulp(hi)` still holds**. This is systematic only
> in the FP32 denormal tail: once `|hi| < 2⁻¹⁰²`, the tie value `½ ulp(hi) = 2^(e−24)`
> is itself subnormal, so the residual `lo` is quantized straight onto the tie. The
> strict bit-exact form `fl(hi+lo)==hi` is therefore **ill-posed** there — a property
> of double-word arithmetic near underflow, universal to DD/FF/QF, not an
> `ff_math.hpp` bug (DD rarely hits it because FP64 underflows ~270 decades lower).
> `result_checkable` skips this tail via `kUnderflowTail = 2⁻¹⁰⁰` (a 4× margin above
> `2⁻¹⁰²`); the guard is output-side and general (any op), and does **not** mask a
> real overlap — a normal-range `hi` with `|lo| > ½ ulp(hi)` is still checked. With
> the guard, all ~48.8M checked inputs pass with **zero** failures.
>
> **Not FF-specific — DD has the same latent hole.** This round-to-even hole in the
> `fl(hi+lo)==hi` *evaluation* is a property of double-word arithmetic, not of FF:
> DD (and a future QF) can hit it too. `dd_invariant_test` (T1.2) simply never
> tripped it across ~50.5M inputs because FP64's exponent range is ~6× wider, so the
> denormal tail is out of reach for realistic random inputs. FP32's narrower range
> brings the tail into reach at *ordinary* op inputs (`exp` of any sufficiently
> negative argument), which is why FF surfaced it first. **Follow-up (not urgent, not
> a blocker):** give `dd_invariant_test` the same `kUnderflowTail`-style guard so it
> is not surprised by the same hole if the DD op inventory grows or a future random
> seed lands in the tail. Flagged as a cross-cutting known-lurking issue in the T2.2
> DONE block.

### QF non-overlap invariant (Layer 2, Phase 3)

`qf_nonoverlap_test.cpp` (**T3.2**) is the QF (QuadFloat, 4×FP32) analogue of
`ff_invariant_test`/`dd_invariant_test`, extended from the length-2 pair to the
**length-4** Priest non-overlap chain `|f_{i+1}| ≤ ½ ulp(f_i)` for `i = 0, 1, 2`.
Two structural changes vs T2.2:

1. **Mathematical ½-ulp form via `frexp`, not the pair bit-form `fl(hi+lo)==hi`.**
   The bit-form (a) only speaks about a single pair — QuadFloat is length-4, so the
   per-word chain is needed — and (b) has round-to-even **false positives** at exact
   ties. `classify_nonoverlap` evaluates `half_ulp(f_i) = 2^(e−25)` (from `frexp`)
   directly on each of the three adjacent word pairs. Same `half_ulp`/`pair_checkable`
   machinery T3.1 proved on the renorm atoms, now applied to **every** op's output.
2. **Inputs enriched to ~96-bit ordered width via `__float128`.** `make_wide_input`
   promotes each nominal `double`, adds sub-leading-ulp tail terms (~2⁻²⁸/2⁻⁵⁶/2⁻⁸⁴
   relative), then decomposes into four ordered FP32 words by successive
   round-to-nearest (T3.1's construction), so the non-renorming passthrough ops
   (`negate`/`abs`/`mul_pwr2`/`copysign`/`fmax`/`fmin`) are exercised on genuinely
   length-4 inputs. Enrichment is `KOKKOS_EP_HAVE_QUADMATH`-gated; the invariant check
   is **not** (runs without LIBQUADMATH).

**Op inventory: 54 host ops** (29 unary + 4 joint `sincos`/`sinhcosh` components +
17 binary + 4 special-form `multiply_scalar`/`mul_pwr2`/`fma`/`pow_int`) plus a
5-op device tripwire (`add`/`multiply`/`sqrt`/`exp`/`sin`) — vs T2.2's 50 FF ops
(the +4 are QF's `ieee_add`/`sloppy_add`/`divide_accurate` arithmetic variants).
Registered with the plain `kokkos_ep_add_test` helper (no contraction flags).
`kRandomN = 2×10⁵`, tuned down from the plan's 10⁶ (~5.5 min vs ~13.5 min); the
higher-rate ops still surface deterministically under the fixed per-op seeds. The QF
trig family carries **no tiny-argument lower bound** (unlike T2.2's `|x| ≥ 1e-25`
floor): QF `sincos` has no `FFCSSNR`-style iteration stall.

> **Finding (QD weak-normalization → strict Priest gate ships RED).** qf_math's
> `renorm`/`renorm_4` (the QD 2.3.24 Hida-Li-Bailey normalization) is a
> `quick_two_sum` **cascade**, which provably delivers only the weaker **Shewchuk**
> non-overlap `|f_{i+1}| ≤ ulp(f_i)`, *not* strict Priest `≤ ½ ulp` (Joldes/Muller/
> Popescu, "Tight & rigorous error bounds for … double-word arithmetic"). So a
> handful of renorming ops legitimately land a word in the half-open band
> `(½ ulp, ulp]` — strictly Priest-violating but **not** corruption. The test
> classifies each result three ways — `NOVL_OK` (strict), `NOVL_WEAK` (QD band),
> `NOVL_FAIL` (>ulp, packing break, or NaN/inf — always fatal) — and gates on the
> single flag `kStrictPriestGate`. At `kRandomN=2×10⁵`: **zero `NOVL_FAIL` across all
> ~11.2M checked results** (no genuine corruption), **19 weak deviations across 9
> ops** (`log`, `cos`, `sincos.cos`, `subtract`, `fmod`, `remainder`, `fdim`,
> `multiply_scalar`, `pow_int`; worst ratio **1.375** = `fmod`, all `< 2.0` = all
> inside the ulp band, robustly driven by `fmod`/`remainder`). This is *systemic*
> (shared renorm path), not per-op bugs, and unfixable without replacing the library's
> normalization — **Rule 4 forbids** patching `qf_math.hpp`. The
> device tripwire (5 ops) and all 13 Test-C corner cases pass.
>
> **RESOLVED — the Shewchuk gate is the adopted posture.** T3.2 shipped this as the
> "unclear" branch of the acceptance-gate decision tree and deferred the call. It has
> since been made and recorded in **`PORT_NOTES_QF.md` §16**, which carries the
> algorithmic proof: QD's `renorm` is a `quick_two_sum` cascade, which *cannot* beat
> `|f_{i+1}| ≤ ulp(f_i)`, so a word landing in `(½ ulp, ulp]` is expected behaviour,
> not a defect. `kStrictPriestGate` is therefore **`false` by default** and the test
> is **GREEN**. Setting it `true` restores the strict Priest reading and turns those
> 9 ops RED again — retained as a diagnostic switch only. Either way the strict
> ½-ulp check still runs on every result and every deviation is tallied with its
> overlap ratio, so nothing is hidden. **Not silently loosened** — changing the gate
> means flipping that one reviewed flag, never weakening the check itself.

> **Expected internal diagnostic.** `angle`/`atan2` each emit 9 `QFCSSNR: argument
> too large` prints (18 total) — QF `sincos`'s internal large-argument guard firing
> on an intermediate value; the results stay checkable and pass. Benign guard output
> (analogous to T2.2's `FFEXP` prints), not a failure.

## Property/identity tests (Layer 3)

Layer 3 (`dd_property_test`, **T1.3**) checks **algebraic identities** the DD ops
must satisfy — a form of correctness orthogonal to Layer 1 (are the EFTs exact?)
and Layer 2 (are outputs well-formed?): *do the ops compose the way the algebra
says they should?* Identities are split by whether verifying them needs an oracle
at all.

**Group A — bit-exact, no oracle.** Identities whose two sides must produce the
*identical* `(hi, lo)` bit pattern, so the test is a raw `==` with no tolerance
and no `__float128`. This group runs unconditionally (even on a quadmath-less
Kokkos). Members: `add(a, negate(a)) == 0`, `a - a == 0`, `a·1 == a`,
`a·(-1) == negate(a)`, `abs` sign branches, `negate(negate(a)) == a`, and **add
commutativity** `add(a,b) == add(b,a)`. Add commutativity is bit-exact because
Knuth twoSum's error term is order-independent, so it stays in Group A.

**Group B — tolerance, needs the `__float128` oracle** (`#ifdef
KOKKOS_EP_HAVE_QUADMATH`; runtime-SKIP otherwise). Round-trip and trig
identities where both sides are correct but rounding makes them differ in the
last place(s): `sqrt(a)²≈a`, `exp(log(a))≈a`, `log(exp(a))≈a`,
`sin²+cos²≈1`, `sin(-a)==-sin(a)`, `cos(-a)==cos(a)`, `tan·cos≈sin`,
`2·sin·cos≈sin(2a)`, `exp(a)·exp(-a)≈1`, `hypot²≈a²+b²`, `pow(a,2)≈a·a`,
`atanh(a)≈½(log(1+a)−log(1−a))`, plus **multiply commutativity** — demoted here
from Group A because Dekker twoProduct's partial-sum ordering reorders under
operand swap (FP add is non-associative), so `multiply(a,b)` and `multiply(b,a)`
agree to ~31 digits but are **not** guaranteed bit-identical. Each identity is
scored with `digits_of_accuracy` and **fail-gates on the mean** against
`tolerance_digits = -log10(N·u²)` (`u = 2⁻⁵³`); per-identity proven bounds
(2u²/4u²/10u²) are cited in code comments (plan rule 5). Gating on the mean, not
the min, keeps conditioning-limited samples (e.g. `sin²+cos²` near `±π·k`, which
needs triple-float argument reduction) from false-failing.

**Test C — named-constant regressions.** Spot-checks that named constants /
transcendental round-trips hold to ≥30 digits: `|sin(π)|≈0` (softened via
`lookup_expected_min_drop("sin")`), `log(e)≈1`, `exp(log2)≈2`, `√2·√2≈2`,
`log(10)≈log10` constant. `euler_gamma`/digamma is **skipped** (no digamma op and
no independent DD oracle for the constant in `dd_math.hpp`).

**Device pass.** 3 Group A (`add_neg`, `mul_one`, `abs_branch`) + 2 Group B
(`sqrt_sq`, `pythag`) rerun on device (10⁵ inputs).

**Anti-tests (deliberately NOT tested, documented in the source).**
Associativity of `add` and distributivity across large-magnitude cancellations
are **false for any finite-precision format** (rounding is grouping-dependent) —
asserting them would be testing IEEE rounding, not the DD port. Registered with
the plain `kokkos_ep_add_test` helper (no contraction flags — not an EFT test).

### FF property/identity (Layer 3, Phase 2)

`ff_property_test.cpp` (**T2.3**) is the FF analogue of `dd_property_test.cpp`,
mirroring its structure verbatim. The only substantive change is the **precision
scale**: FF's unit roundoff is `u = 2⁻²⁴` (vs DD's `2⁻⁵³`), so
`u² = 2⁻⁴⁸` and the statistical floor becomes
`tolerance_digits = -log10(N·u²) ≈ 8.45` at `N = 10⁶` (vs DD's ≈ 25.91).
Everything is computed at runtime from `BackendTraits<FF>::u_squared`, not
hardcoded. Group B means land around 13 digits, clearing the ~8.45 floor with
room to spare.

**Group A (7 bit-exact).** Same identities as DD: `add(a,negate(a))==0`,
`a-a==0`, `a·1==a`, `a·(-1)==negate(a)`, `abs` sign branches,
`negate(negate(a))==a`, add commutativity. Failures dump raw FP32 bit patterns
(`0x%08x` per limb). A1–A6 use full Route-A FF operands (`FloatFloat(double)`,
generally **nonzero** `lo`); A8 (add commutativity) uses **single-float**
operands (`lo==0`) — the FF analogue of DD's single-double convention: with a
nonzero `lo`, `add()`'s trailing `+a.lo+b.lo` reorders under operand swap and
would break bit-exactness. Multiply-by-±1 (A3/A4) is Dekker-domain-gated
(`split_safe_max() = FLT_MAX/8193`), and any strict-`==` mismatch whose limbs
land in the FP32 denormal tail (`< 2⁻¹⁰⁰`) is counted **skipped**, not failed
(the T2.2 round-to-even hole).

**Group B (13 tolerance).** Same identities as DD, mean-gated at ≈ 8.45. The
exp round-trips narrow their domains to respect `ff_math.hpp`'s exp guard
(`a.hi ≥ 88.0f` returns 0): B2 `exp(log(a))` on `[1e-30,1e30]`, B11 `pow(a,2)` on
`[1e-15,1e15]` (`2·ln(1e15) ≈ 69 < 88`). B3 `log(exp(a))` and B9 `exp(a)·exp(-a)`
run on `[-79,79]`, RESTORED from a temporary `[-69,69]` narrowing once bug task
**B8** landed. History: T2.3 saw exp round-trips stall to their iteration cap
(`FFEXP: iteration limit`, return 0) on `~[70,85]` arguments and hypothesized
exp's Taylor `eps=1e-15f` (finer than FloatFloat's `~3.55e-15`) as the cause,
deferring to bug task **B4**. B4's investigation falsified the eps theory: the
real defect was a Dekker splitter overflow in `divide()` (`b.hi*split → inf`,
`inf − inf = NaN`) that poisoned `log()`'s Newton exp for `x ≳ e^79.7`. **B8**
fixed `divide()` with a scaled splitter, clearing the stall; `[-79,79]` sits one
integer under the `~79.7` empirical clean ceiling (not pushed to `[-85,85]`). B8 double-angle narrows to `|a| < 3` because it
compares two *different* reduced arguments and FF's double-float argument
reduction degrades for large `|a|` (PORT_NOTES §5). B0 is the demoted multiply
commutativity.

**Test C.** Target ≥ 12 of FF's 14 digits (DD used ≥ 30 of 31): `log(e)≈1`,
`exp(log2)≈2`, `√2·√2≈2`, `log(10)≈log10` constant. C1 `|sin(π)|≈0` is softened
to a conditioning-aware floor (arg reduction near π, PORT_NOTES §5); C6
euler_gamma/digamma is skipped (no digamma op in `ff_math.hpp`).

**Device pass.** 3 Group A (`A1`, `A3`, `A5`) + 2 Group B (`B1`, `B4`) on 10⁵
inputs, with `View<float*>` limb transfer. Registered with the plain
`kokkos_ep_add_test` helper (no contraction flags — identities are
FMA-contraction agnostic). `ff_math.hpp`/`ff_complex.hpp` are **not** modified by
this layer.

### QF property/identity (Layer 3, Phase 3)

`qf_property_test.cpp` (**T3.3**) is the QF (QuadFloat, 4×FP32) analogue of
`ff_property_test.cpp`, mirroring its Group A / Group B / Test C taxonomy verbatim.
There is **one deliberate structural divergence — the tolerance model** — plus one
QF **strengthening**.

**Tolerance model (the divergence).** DD/FF are *double-word* formats: their nominal
precision is `u²` (`2⁻¹⁰⁶` / `2⁻⁴⁸`), and T1.3/T2.3 gate Group B on the statistical
floor `−log10(N·u²)`. QF is a *quad-word* — its resolution **is** `U = 2⁻⁹⁶` itself
(four FP32 limbs ≈ 96 bits), not some `u⁴`. Re-using `−log10(N·U)` would give a
*looser* ~23.6 floor, so T3.3 instead follows the **plan's stated policy** ("1–10 ulp
at QF's ~2⁻⁹⁶ resolution") with an **absolute ulp floor**, still gating on the mean:
`digits(k ulp) = 96·log10(2) − log10(k)`, i.e. **10 ulp → 27.90** (default),
**30 ulp → 27.42** (for the two exp-tail-limited round-trips). Per-identity ulp
bounds are cited in-source (plan rule 5).

**Group A (12 bit-exact, no oracle, raw 4-word `==`).** `a+(-a)==0`, `a-a==0`,
`a+0==a`, `a-0==a`, `a·1==a`, `a·0==0`, `a·(-1)==negate(a)`, `-(-a)==a`, `abs` sign
branches, `abs(-a)==abs(a)`, add commutativity, and the `mul_pwr2` power-of-2
round-trip `mul_pwr2(mul_pwr2(a,2ᵏ),2⁻ᵏ)==a`. **Strengthening vs FF/DD:** add
commutativity is bit-exact on **full-width Route-A (double) operands** here — QF's
`sloppy_add` is a symmetric componentwise twoSum cascade whose `renorm` collapses the
accumulator identically regardless of operand order (verified bit-exact over 3×10⁶
4-word operands), so it does **not** need the single-word restriction FF/DD imposed.
Multiply-based identities are Dekker-domain-gated (`split_safe_max()=FLT_MAX/8193`);
the `mul_pwr2` round-trip additionally skips inputs whose ×2ᵏ intermediate overflows
FP32 (`|x|≈FLT_MAX`) — a range limit, not a defect. All 12 pass with **0 failures**.

**Group B (15 tolerance).** `B0` demoted multiply commutativity (Dekker cross-term
reorders under swap — same demotion as DD/FF), `B1` `sqrt(a)²≈a`, `B2` `exp(log a)≈a`,
`B3` `log(exp a)≈a`, `B4` `sin²+cos²≈1`, `B5` `cosh²−sinh²≈1` (`|a|<5`: at large `|a|`
both terms are ~`e²ᵃ/4` and their difference is catastrophic cancellation, an FP
property), `B6`/`B7` `sin/cos(a+b)` addition formulas (`|a|,|b|<3`, reduction clean),
`B8` `tanh≈sinh/cosh`, `B9`/`B10` inverse pairs `asin(sin)`/`atan(tan)` (`|a|<1.4`),
`B11` `pow(a,2)≈a·a`, `B12` `sqrt(a²)≈|a|`, `B13` `hypot²≈a²+b²`, and `B14` the
small-argument sensitivity `exp(a+eps)≈exp(a)·(1+eps)`. `B2` (`[1e-13,1e13]`) and
`B11` (`[1e-3,1e3]`) are gated at **30 ulp** with a **PORT_NOTES_QF §10** citation —
their internal `exp` maps arguments whose low QF words fall into the FP32 denormal
tail, a documented conditioning limit (marked EXEMPT, not loosened). All 15 clear
their gates; means 27.79–29.00.

**B14 = the T2.3 B4 pattern — and it does NOT recur in QF.** T2.3 surfaced a real
`ff_math.hpp` bug (B4): exp's convergence `eps=1e-15f` is *finer* than FF's ~3.55e-15
resolution, so small-arg exp stalled and returned 0. `qf_math.hpp` exp uses
`eps=1e-28f`, deliberately *coarser* than `U=2⁻⁹⁶` (PORT_NOTES_QF §7/§10 — the QF port
fixed exactly this class of bug at authoring time). B14 draws `|eps|≤1e-15` (so the
identity's own dropped `O(eps²)` term stays below `U`) and holds to **mean 28.06, min
26.40 with no stall**. B14 is GREEN; **no QF B-task is filed**. It stays in the suite
as a durable regression guard and for T2.3 parity.

**Test C.** Target ≥27 of QF's 29 digits: `C1 |sin(π)|≈0` (softened to a
conditioning-aware floor, arg reduction near π), `C2 log(e)≈1`, `C3 exp(log2)≈2`,
`C4 √2·√2≈2`, `C5 log(10)≈log10` constant — **5/5 pass**. `C6` euler_gamma/digamma is
skipped (no digamma op in `qf_math.hpp`).

**Device pass.** 3 Group A (`A1`, `A5`, `A9`) + 2 Group B (`B1`, `B4`) rerun on
device (10⁵ inputs, `View<float*>` 4-limb transfer) — all pass. `kRandomN = 2×10⁵`,
tuned down from the plan's 10⁶ for the same wall-time reason as `qf_nonoverlap_test`
(the absolute ulp floor is N-independent, so the reduction does not shift the gate).
Registered with the plain `kokkos_ep_add_test` helper. `qf_math.hpp` is **not**
modified (rule 4).

## Differential accuracy (Layer 4)

Layer 4 (`dd_accuracy_test`, **T1.4**) asks the question Layers 1-3 deliberately
did not: *does `op(x)` equal the true real answer to N digits?* For **every** op
in the T1.2 inventory (~50 rows), it widens the device result to `__float128` and
compares against a quadmath oracle evaluated on the SAME input, scoring each
element in **digits of accuracy** `digits = -log10(rel_err)` (capped at DD's 31 =
`-log10(u²)`, u = 2⁻⁵³). Two passes per op — **10⁶ random** (ranges taken
verbatim from the T1.2 domain predicates) plus a **corpus** pass (the PORT_NOTES
§3/§4 named accessor where one exists, e.g. `exp_overflow`, `trig_near_pi`,
otherwise the generic bundler) — combined into one (min, mean, n) per op.

**Fail-gates on the MEAN**, not the min, against a single uniform
`tolerance_digits = -log10(N·u²) ≈ 25.91` at N = 10⁶ (no per-op tolerance
overrides — that would defeat the point of a differential-accuracy gate). Ops in
the shared PORT_NOTES §5 conditioning registry (`lookup_expected_min_drop`)
report **EXPECTED-MIN-DROP: OK** when the mean clears tolerance and the low min is
sanctioned (near-cancellation, derivative → ∞ near |a|=1, arg-reduction near ±π,
output-denormal `exp`). Oracle subtleties handled: ties-to-even round-family →
`nearbyint` oracle (not `round`); `exp10` → `pow(10, x)` (no `__float128`
overload). The whole file is `#ifdef KOKKOS_EP_HAVE_QUADMATH` and runtime-SKIPs
(77) otherwise.

`dd_accuracy_test` ships **`(DONE, RED)`**: it flags three real `dd_math.hpp`
accuracy defects (`tgamma` mean ≈ 14.56 — FP64 Lanczos coefficients; `erfc` ≈
19.50 — `1−erf` cancellation; `erf` ≈ 24.64 — large-|z| asymptotic branch) and
fails on them. The red is the point — it is the durable regression gate for the
follow-up bug tasks B1/B2/B3. Per rule 4 the surfacing test reports; it does not
patch the library.

### FF differential accuracy (Layer 4, Phase 2)

`ff_accuracy_test.cpp` (**T2.4**) is the FF analogue of `dd_accuracy_test.cpp`,
mirroring its structure verbatim. The substantive changes are the **precision
scale** and the **FP32-narrower op domains**:

- **Scale.** FF's unit roundoff is `u = 2⁻²⁴` (vs DD's `2⁻⁵³`), so `u² = 2⁻⁴⁸`,
  digits are capped at **14** (`BackendTraits<FF>::max_digits`), and the
  statistical floor becomes `tolerance_digits = -log10(N·u²) ≈ 8.45` at N = 10⁶
  (vs DD's ≈ 25.91) — computed at runtime from `BackendTraits<FF>::u_squared`, not
  hardcoded. Expected mean per PORT_NOTES: 13.3-14.0.
- **Domains.** Random ranges and domain predicates are taken **verbatim from the
  T2.2 inventory** (`ff_invariant_test.cpp`), not re-derived: `exp` guards at
  `a.hi ≥ 88` (not DD's 300); trig carries the FP32 tiny-argument lower bound
  (`|x| ≥ 1e-25`, else 0, to dodge the sincos iteration-limit hazard);
  `sinh`/`cosh` cap at `|x| < 40`, `tanh` at `|x| < 20`; the log family window is
  `[1e-34, 1e34]`; `erf`/`erfc` use `[-6, 6]` (FF saturates to ±1 past |z|=6);
  `tgamma` uses `[1e-3, 23)`. The corpus pass uses the FP32 accessors
  (`corpus::unary<float>` / `<float>` named accessors).

Same op set as T1.4 (~50 rows: arithmetic, transcendentals, roots, comparisons,
two-output `sincos`/`sinhcosh`, ternary `fma`, integer-scalar `pow_int`), the same
oracle subtleties (`nearbyint` for the ties-to-even round-family, `pow(10,x)` for
`exp10`), the same MEAN fail-gate with EXPECTED-MIN-DROP for the **shared**
PORT_NOTES §5 registry (conditioning is a property of the algorithm, not the
width, so DD and FF read the same table). Registered with the plain
`kokkos_ep_add_test` helper (no contraction flags — not an EFT test), mirroring
`dd_accuracy_test`. Per rule 4, `ff_math.hpp` / `ff_complex.hpp` are **not**
modified: any op whose mean falls below tolerance is REPORTED (op, pass, offending
input, digit count) and fails; it is not patched or xfailed.

### QF differential accuracy (Layer 4, Phase 3)

`qf_accuracy_test.cpp` (**T3.4**) is the QF (QuadFloat, 4×FP32) analogue of
`ff_accuracy_test.cpp` / `dd_accuracy_test.cpp`, mirroring their structure end to
end. As with T3.3, the two substantive changes are the **precision scale** and the
**tolerance model**:

- **Scale.** QF's resolution is `U = 2⁻⁹⁶` (four FP32 limbs ≈ 96 bits), digits
  capped at **29** (`kMaxDig`).
- **Tolerance model — T3.3's absolute ulp-of-U floor, NOT the DD/FF statistical
  one.** DD/FF are double-word (nominal precision `u²`) and gate on
  `−log10(N·u²)`. QF is a *quad-word* — its resolution **is** `U`, not some `u⁴` —
  so, exactly as T3.3 established, this test uses the plan's per-op **absolute ulp
  floor**, gating on the **mean**:
  `digits(k ulp) = 96·log10(2) − log10(k)`, i.e. **10 ulp → 27.90** (default) and
  **30 ulp → 27.42** (the exp-family output-denormal tail, PORT_NOTES_QF §10).

**Four regimes, three passes.** The plan's broad / narrow / near-edge / corpus
regimes are delivered as three passes combined per op (min over all, mean weighted
by count): **narrow** random (Route-A `QuadFloat(double)` over the op's
well-conditioned domain, taken from `src/demo_qf_real.cpp`'s `fill_inputs` = the
§11 reference domains), **broad** random (the same domain enriched to a full
~96-bit ordered QuadFloat via `make_wide_input`, copied from `qf_nonoverlap_test`),
and the **corpus / near-edge** pass (the PORT_NOTES §3/§4 named accessor where one
exists — `exp_overflow`, `trig_near_pi`, `sinh_cosh_small`, `atanh_small`,
`remainder_regression` — else the generic bundler). Out-of-domain corpus values are
**skipped, not failed**.

**Oracle at the exact widened input (a QF-specific twist).** T1.4/T2.4 evaluate the
oracle at the nominal double `(float128)x`; that is exact for DD/FF because their
claimed precision is coarser than a double. QF claims ~29 digits — *finer* than a
double — so this test evaluates the oracle at the **exact widened value**
`qf_to_q(input)` (sum of the four input words). For the narrow regime the two
coincide exactly; for the broad regime (whose inputs carry sub-double bits by
construction) it is the only honest choice.

**exp-family domain narrowing (documented deviation from §11).** `qf_math.hpp` exp
guards at `a.f0 ≥ 88`; for sufficiently negative arguments the quad-word result's
low limbs fall into the FP32 output-denormal band (§10), which is why the demo/§11
sample of `exp` over [−80,80] reads mean **25.99** (below even the 30-ulp floor). To
gate the mean honestly the random passes narrow the negative end so the full
quad-word result stays in FP32 normal range (`exp` [−35,80], `exp2` [−50,120],
`exp10` [−15,30]); the excluded tail is the §10 conditioning limit (min-drop
exempt), exercised at the high edge by `exp_overflow` and cited via
`lookup_expected_min_drop("exp")`. exp/exp2/exp10 gate at 30 ulp; `expm1`
(result ≥ −1, no denormal tail) at 10 ulp.

**Round-family oracle / tie semantics.** `round_to_nearest_int`/`round` use
`qf_nint = floor(d+0.5)` (round-half-up). On continuous random inputs (and the
generic corpus, which has no exact half-integers) round-half-up, ties-to-even and
ties-away all agree — ties are measure-zero — so the oracle is
`Kokkos::round/ceil/floor/trunc` (matching the demo). The `nint_half_integer`
corpus is **deliberately not used** for the round-family (exact-tie behavior is a
separate, out-of-scope concern). *(This differs from T1.4/T2.4, which used a
`nearbyint` ties-to-even oracle — those backends' `nint` rounds to even; QF's
rounds half-up, and on continuous inputs the distinction never surfaces.)*

**Op surface (49 scored + 5 skipped = T3.2's 54).** Every QF op returning a
`QuadFloat` with a quadmath analogue is scored: 29 unary
(abs/negate/sqr/sqrt, round-family ×5, exp-family ×4, log-family ×4, trig ×3,
inverse-trig ×3, hyperbolic ×3, inverse-hyperbolic ×3), 4 two-output components
(`sincos.sin`/`.cos`, `sinhcosh.sinh`/`.cosh` — SIN/SINH first per §12), 13 binary
(add/subtract/multiply/divide/pow/atan2/hypot/fmod/remainder/copysign/fmax/fmin/
fdim), and 3 custom (`multiply_scalar`, `fma`, `pow_int`). Skipped with in-source
rationale (5): `sloppy_add` (`add()` is its public alias), `ieee_add` (internal, no
distinct public op), `divide_accurate` (internal, `divide()` wraps it), `mul_pwr2`
(exact power-of-2 scaling — exact by construction, covered bit-exactly by T3.3 A12),
`angle` (identical to `atan2(y,x)`). `qf_math.hpp` has **no** erf/erfc/tgamma, so
T2.4's three RED candidates have no QF counterpart.

`kNarrowN = kBroadN = 10⁵` (2×10⁵ random/op), tuned down from the plan's 10⁶ for the
same wall-time reason as T3.2/T3.3 (the absolute ulp floor is N-independent, so the
reduction does not shift the gate). The whole file is `#ifdef
KOKKOS_EP_HAVE_QUADMATH` and runtime-SKIPs (77) otherwise. Every pass runs through
`Kokkos::parallel_for` on the Serial `DefaultExecutionSpace` (so the whole test is
the device path); a representative subset (`add`/`multiply`/`sqrt`/`exp`/`sin`) is
re-run under fresh seeds as an explicit device-parity checkpoint. Registered with
the plain `kokkos_ep_add_test` helper (no contraction flags), mirroring
`dd_accuracy_test` / `ff_accuracy_test`. Per rule 4, `qf_math.hpp` is **not**
modified: any op whose mean falls below tolerance is REPORTED (op, pass, offending
input, digit count) and fails as a candidate B-task for Reet to file; it is not
patched or xfailed.

## FMA-contraction guard (Layer 5)

Layer 5 (`dd_fma_guard_test`, **T1.5**) is the *positive* counterpart to the
defensive posture above. T1.1 builds `-ffp-contract=off` to protect its **own**
results; T1.5 asks whether that posture is actually **needed** by building the
identical Dekker `twoProduct` under **both** contraction settings and cross-
checking against a contraction-immune `__float128` oracle.

**One source, two targets.** `dd_fma_guard_test.cpp` is compiled twice:

```cmake
kokkos_ep_add_eft_test(dd_fma_guard_test)              # -> dd_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(dd_fma_guard_test)  # -> dd_fma_guard_test_contract_on  (ON, reports)
```

Compiling the *same bytes* under different flags makes "identical inputs, identical
logic" a guarantee of the build system rather than a claim a reviewer must verify
across two drifting files. The only per-variant knobs are compile definitions the
helpers set: `KOKKOS_EP_CONTRACTION_MODE` (`0` = OFF/gate, `1` = ON/report) and,
for the ON variant, `KOKKOS_EP_BASELINE_PATH`.

`kokkos_ep_add_eft_test_contract_on(<name>)` mirrors `kokkos_ep_add_eft_test` but
forces contraction **on** into a distinct `<name>_contract_on` target:
`-ffp-contract=fast` (GNU/Clang), `-fp-model=fast` (Intel), and `--fmad=true`
(nvcc's default, stated explicitly). Both variants coexist because of the suffix.

**OFF variant — gates.** Asserts the Dekker error term is bit-exact (`F == 0`);
this is a stronger restatement of what T1.1 asserts, plus a `twoSum` control that
must stay exact (it has no contractible mul-then-± adjacency).

**ON variant — reports.** The compiler is *allowed* to contract; two outcomes,
both informative, neither a failure:

- `F == 0` — the compiler either did not contract, **or** contracted harmlessly.
  (On GCC 13.3.0 the latter holds: even with `-mfma` it emits 8 FMA instructions
  for the Dekker sequence, yet `F` stays 0 — Veltkamp splitting makes each partial
  product exactly representable, so fusing `partial ± accumulator` introduces no
  rounding difference. On this ISA target the `-ffp-contract=off` posture is
  belt+suspenders.)
- `F > 0` — the compiler contracted in a way that *does* change the result; the
  `-ffp-contract=off` posture in `dd_math.hpp`'s build is **required**, and `F` is
  the evidence.

The ON variant **always exits 0** — its value is the number, not a gate. A
*change* in `F` between builds is the regression signal, so the observed count is
recorded in `tests/dd_fma_guard_baseline.txt`; each ON run compares its live count
to that baseline and prints `baseline: OK` or `*** DRIFT ***` (a warning, never a
failure — investigate, then update the file if the new value is correct for the new
toolchain). **Scope:** the Dekker `twoProduct` only — the one DD primitive where
contraction is a documented hazard.

### FF FMA-contraction guard (Layer 5, Phase 2)

`ff_fma_guard_test.cpp` (**T2.5**) is the FF analogue of `dd_fma_guard_test.cpp`,
mirroring its structure verbatim: one source compiled into two targets under
opposite contraction postures, a contraction-immune oracle, a `twoSum` **control**
(no mul-then-± adjacency → must stay exact both ways), host + device passes, OFF
gates / ON reports with a committed baseline. It reuses the **same** CMake helpers
as the DD guard:

```cmake
kokkos_ep_add_eft_test(ff_fma_guard_test)              # -> ff_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(ff_fma_guard_test)  # -> ff_fma_guard_test_contract_on  (ON, reports)
```

The `_contract_on` helper derives the per-test baseline path
(`ff_fma_guard_baseline.txt`) from the target name, so the DD and FF guards share
one helper with no duplication.

The FF Dekker `twoProduct` is the **same algorithm** at FP32: splitter `8193.0f`
(2¹³+1), hi/lo split, the cross-term subtraction `a1*b1 - p`. The contraction
hazard is identical — a fused `fma(a1, b1, -p)` computes the tail with
full-precision residuals the algorithm's rounded-intermediate algebra does not
assume, silently breaking the error term.

**Two divergences from the DD shape, both reported (see the T2.5 DONE block):**

- **FP64 oracle, no quadmath gate.** Like `ff_eft_test` (T2.1), ground truth is
  plain **FP64**, not `__float128`: the exact FP32 product needs ≤48 bits, which
  fits FP64's 53-bit mantissa, so the reference `(p, e)` decomposition is
  *provably* exact — a stronger oracle than DD's quadmath, needing no external
  library. Consequently both variants run **unconditionally** (no
  `KOKKOS_EP_HAVE_QUADMATH` gate, no runtime SKIP-77), unlike the DD guard which
  SKIPs without LIBQUADMATH.
- **`twoSum` control gets an FP32-specific oracle-faithfulness guard.** The DD
  guard reuses its `twoProduct` input list for the `twoSum` control unchanged;
  that is safe at FP64 scale but **not** at FP32. A pair like `(FLT_MIN = 2⁻¹²⁶,
  2²⁴)` has an in-range *product* (2⁻¹⁰², admitted by the `twoProduct` domain) yet
  an exact *sum* spanning 174 bits — far beyond the FP64 oracle's 53. There the
  FP32 `twoSum` is **correct** (`hi = 2²⁴`, `lo = 2⁻¹²⁶`); it is the FP64
  *decomposition* that collapses the tiny tail, so the exact `lo` looks like a
  false "mismatch". The control therefore skips pairs the oracle cannot witness
  (`sum_oracle_faithful`: the FP64 `twoSum` error term is zero ⇔ the double sum is
  exact), excluding **only** wide-exponent sums, never a pair where FP32 `twoSum`
  is actually wrong. `twoProduct` needs no such guard — its ≤48-bit product is
  always oracle-faithful (confirmed: 0 mismatches).

**Observed on this toolchain (GCC 13.3.0, baseline x86-64, no `-mfma`):** the FF
Dekker `twoProduct` is bit-exact over **220,410** in-domain checks (host + device)
under **both** postures — **`F = 0`**, the *same* outcome T1.5 recorded for DD.
GCC emits plain mul+sub and contracts nothing on this ISA; `-ffp-contract=off` is
belt+suspenders here, and the ON reporter's baseline (`0`) arms drift detection for
a future toolchain where that stops being true. Because `F = 0`, the ON variant is
a **reporter** (exits 0), not a `WILL_FAIL` test — matching
`dd_fma_guard_test_contract_on` exactly (the guard's realness is proven by
sensitivity, not by contraction on this compiler; T1.5 verified a deliberately-
broken `twoProduct` flags ~95 % of checks). **Scope:** the Dekker `twoProduct`
only; no complex, no other ops; `ff_math.hpp` NOT modified.

### QF FMA-contraction guard (Layer 5, Phase 3)

`qf_fma_guard_test.cpp` (**T3.5**) is the QF analogue of `ff_fma_guard_test.cpp`,
mirroring its structure verbatim: one source compiled into two targets under
opposite contraction postures, a contraction-immune FP64 oracle, a `qf_two_sum`
**control** (no mul-then-± adjacency → must stay exact both ways), host + device
passes, OFF gates / ON reports with a committed baseline. It reuses the **same**
CMake helpers as the DD/FF guards:

```cmake
kokkos_ep_add_eft_test(qf_fma_guard_test)              # -> qf_fma_guard_test              (OFF, gates)
kokkos_ep_add_eft_test_contract_on(qf_fma_guard_test)  # -> qf_fma_guard_test_contract_on  (ON, reports)
```

The `_contract_on` helper derives the per-test baseline path
(`qf_fma_guard_baseline.txt`) from the target name, so the DD, FF, and QF guards
share one helper with no duplication.

**Two deliberate divergences from the FF shape (both reported in the T3.5 DONE block):**

- **No mirror-and-comment — calls the shipped primitives directly.** `ff_math.hpp`
  embeds its Dekker `twoProduct` inside `multiply()`, so T2.5 had to *duplicate* it
  into the test file. `qf_math.hpp` **exposes** `qf_two_prod` / `qf_two_sqr` /
  `qf_two_sum` as free functions, so T3.5 compiles the **shipped source** under the
  ON flags — strictly stronger for a contraction guard: it characterizes whether
  GCC contracts *`qf_math.hpp`'s own code*, not a copy that could drift. Same
  divergence `qf_eft_test` (T3.1) took from `ff_eft_test`.
- **`qf_two_sqr` is also guarded.** T2.5 was `twoProduct`-only (FF exposes no
  squaring EFT). QF ships `qf_two_sqr`, a **second** Dekker sequence with a
  `hi*hi - q` contraction hazard, so T3.5 guards both — matching T3.1's op surface.

**Three-way classification (the ON reporter's refinement over T2.5's binary `F`).**
T2.5 counted one number `F` = "error-term mismatches" and exited 0 regardless. QF
refines this into three mutually-exclusive buckets per in-domain input (with
`e_ref` = the unique exact residual `float(exact − hi)` and `id_ok` = the Dekker
identity `(double)hi + (double)lo == exact`):

- `e_ref == 0` → **TRIVIAL** (true error legitimately zero, e.g. an exact product;
  uninformative about contraction — reported separately, not gated).
- `e_ref ≠ 0` and `id_ok` → **`ERR_NONZERO_CORRECT`** (compiler did not contract,
  or contracted harmlessly — Veltkamp splitting keeps each partial product exactly
  representable, so a fused `partial ± accumulator` introduces no rounding
  difference; see T1.5's `-mfma` analysis).
- `e_ref ≠ 0`, `!id_ok`, `lo == 0` → **`ERR_ZERO`** (error term **collapsed** — the
  classic contraction signature).
- `e_ref ≠ 0`, `!id_ok`, `lo ≠ 0` → **`ERR_NONZERO_WRONG`** (nonzero error term that
  violates the identity — the *only* genuinely-broken outcome; a real
  miscompilation/bug, expected to never fire).

**OFF gate:** `ERR_ZERO == 0 && ERR_NONZERO_WRONG == 0` (`F := their sum`), plus the
`qf_two_sum` control exact and no named-case failure. **ON PASS:**
`ERR_NONZERO_WRONG == 0` — this is T2.5's ratified reporter policy, refined to the
three-way scheme (T2.5 could not tell "contracted-to-zero" from "contracted-to-
wrong", so it exited 0 on both; the `ERR_NONZERO_WRONG` bucket lets T3.5 fail *only*
on the genuinely-broken case). The ground truth is plain **FP64** (exact — 48 ≤ 53
bits), so both variants run **unconditionally** (no `KOKKOS_EP_HAVE_QUADMATH` gate,
no SKIP-77), like the FF guard.

**Observed on this toolchain (GCC 13.3.0, baseline x86-64, no `-mfma`):** over
**511,988** `qf_two_prod` checks + **399,286** `qf_two_sqr` checks (host + device,
each ≈256 K / 200 K per side) under **both** postures, `ERR_ZERO = 0`,
`ERR_NONZERO_WRONG = 0`, `ERR_NONZERO_CORRECT` = everything informative — **`F = 0`**,
the *same* outcome T1.5/T2.5 recorded. GCC does **not** contract `a1*b1 − p`
(`qf_two_prod`) or `hi*hi − q` (`qf_two_sqr`) on this ISA; `-ffp-contract=off` is
belt+suspenders here, and the ON reporter's baseline (`0`) arms drift detection for a
future toolchain where that stops being true. **Scope:** the two shipped Dekker
sequences only; no complex, no other ops; `qf_math.hpp` NOT modified. FMA-contraction
posture is a compiler characterization, not input conditioning, so it is **not** a
PORT_NOTES_QF §5 entry.

## End-to-end cancellation kernels (Layer 6)

Layer 6 (`dd_e2e_test`, **T1.6**) is the payoff the end user actually cares about.
Layers 1-5 validated a backend's atoms (EFT), structure (non-overlap), identities,
per-op accuracy, and FMA-contraction posture — all machinery. Layer 6 asks: on
classic **cancellation-hostile** problems that the base scalar mangles, does the
double-word backend deliver its advertised digits? Four kernels, each with a known
higher-precision or closed-form oracle:

- **K1:** `√(x²+1) − x` — catastrophic cancellation at large `x`
  (`√(x²+1) ≈ x`, the answer `~1/(2x)` lives in the surviving low bits).
- **K2:** `Σ 1/k²`, k=1..10⁶ — Basel problem, closed form `π²/6`.
- **K3:** Machin's `π = 16·atan(1/5) − 4·atan(1/239)` — transcendental composition.
- **K4:** `Σ (−1)^(k+1)/k`, k=1..10⁶ — alternating harmonic, closed form `ln 2`.

**Two-oracle strategy (K2, K4).** Each finite sum is scored twice. The
sum-vs-**quadmath-partial-sum** comparison (identical N, order, terms) carries the
**arithmetic-precision** claim — it isolates accumulation quality from truncation.
The sum-vs-**closed-form** comparison (K2 vs π²/6, K4 vs ln 2) is a
**truncation-limited sanity check**, gated at `truncation_floor − 1`; at N=10⁶ the
floor is ~6 digits (the Basel tail and the alternating-series error are both ≈ 1/N).

**K1 deviation (documented, both backends).** The literal spec named naive
`√(x²+1) − x` as the DUT expecting full precision; that expectation is numerically
false and **not** a library defect — cancellation loses ~2·log₁₀(x) digits
regardless of arithmetic (Higham §1.7). So K1 ships as a **gated** stable form
`1/(√(x²+1) + x)` (algebraically cancellation-free) plus a **reported, not gated**
naive form (backend vs base scalar, per magnitude) that demonstrates the
extra-word lift under the hostile algorithm.

Both kernels are **host-side** (inherently serial reductions/recurrences), whole
file `#ifdef KOKKOS_EP_HAVE_QUADMATH` (SKIP 77 without quadmath), and neither
modifies the backend math header (rule 4). Registered with the plain
`kokkos_ep_add_test` helper (not an EFT test).

**DD (T1.6).** Gate `mean_digits ≥ 28.0` (= DD's cap 31 − 3 headroom). K1 uses
x ∈ {1e6, 1e10, 1e15} and reports naive DD vs **FP64** (DD's base scalar). Measured
means: `K1_stable` 31.00 (capped), `K2` 29.48, `K3` 28.09, `K4` 29.56 — all PASS.

### FF end-to-end cancellation (Layer 6, Phase 2)

`ff_cancellation_test.cpp` (**T2.5**) is the FF analogue of `dd_e2e_test.cpp`,
mirroring its structure verbatim (same four kernels, same two-oracle strategy, same
K1 gated-stable + reported-naive shape, host-side, quadmath-gated, `ff_math.hpp`
untouched). The substantive changes are the **precision scale** and two
**FP32-forced K1 deviations**, both derived (not fabricated) and reported in-source:

- **Gate.** `mean_digits ≥ 11.0`, derived by the **same "cap − 3" formula** T1.6
  used: FF's harness cap is `BackendTraits<FF>::max_digits = 14` (u² = 2⁻⁴⁸ ≈ 14.45
  decimal digits), so `14 − 3 = 11.0`. Computed from `max_digits` at compile time,
  not hardcoded.
- **K1 baseline = FP32, not FP64.** T1.6 compared naive-DD against naive-FP64 (DD's
  1-word base). The faithful FF mirror compares naive-FF against naive-**FP32** (FF's
  1-word base). Comparing FF against FP64 would be dishonest — FP64 (~16 digits) is
  *wider* than FF (~14), so it would "win" the naive contest while saying nothing
  about FF. FF's advantage is over its own base scalar, exactly as DD's is over FP64.
- **K1 magnitudes {1e2, 1e4, 1e6}, not {1e6, 1e10, 1e15}.** The cancellation
  gradient lives ~3 decades lower for FF: plain FP32 loses the `+1` in `x²+1` once
  `x² > 2²⁴` (x ≳ 4100) and FF loses it once `x²` exceeds FF's ~14-digit reach
  (x ≳ 1e7). At T1.6's magnitudes both naive forms would read 0 at the upper two x —
  no gradient. At {1e2,1e4,1e6} the FP32→FF lift is visible across the whole sweep.

K2/K4 keep N = 10⁶: at that N the smallest term (1/N = 1e-6 for K4, 1e-12 for K2)
stays well above FF's running-sum resolution, so no term stalls into the precision
floor and the arithmetic-precision comparison is well-posed (the iteration-bound
concern the plan flags for FP32 does not bite here). Per-kernel measured results
are printed by the test and recorded in the T2.5 DONE block. Registered with the
plain `kokkos_ep_add_test` helper (no contraction flags — see the CMake comment on
why K1's naive mul-then-sub adjacency is not a gated-path hazard).

### QF end-to-end cancellation (Layer 6, Phase 3)

`qf_cancellation_test.cpp` (**T3.6**) is the QF analogue of `dd_e2e_test.cpp`
(T1.6) and `ff_cancellation_test.cpp` (T2.6), mirroring their structure verbatim
(same four kernels, same two-oracle strategy, same K1 gated-stable + reported-naive
shape, host-side, quadmath-gated, `qf_math.hpp` / `qf_complex.hpp` untouched). The
substantive change is the **precision scale**; the K1 base-scalar and magnitude
choices follow **T2.6's FF precedent**, because QF's base scalar is FP32, exactly
as FF's is:

- **Gate.** `mean_digits ≥ 26.0`, derived by the **same "cap − 3" formula** T1.6
  and T2.6 used: QF's harness cap is the QF-local `kMaxDig = 29` (4×FP32 ≈ 96 bits
  ≈ 28.9 decimal digits), so `29 − 3 = 26.0` (DD used `31 − 3 = 28.0`; FF used
  `14 − 3 = 11.0`). Computed from `kMaxDig` at compile time, not hardcoded.
- **K1 baseline = FP32, not FP64.** T1.6 compared naive-DD against naive-FP64 (DD's
  1-word base). The faithful QF mirror compares naive-QF against naive-**FP32** (QF's
  1-word base) — the same choice T2.6 made. QF's advertised advantage is over its
  own base scalar, and for the small-x cases FP64 can be *wider* than the
  target-scaled QF quad-word, so an FP64 baseline would misrepresent QF's lift.
- **K1 magnitudes {1e2, 1e4, 1e6}** (T2.6's set). The "+1" collapse that drives the
  FP32 baseline to 0 is determined by the FP32 **high-word** arithmetic — plain FP32
  loses the `+1` in `x²+1` once `x² > 2²⁴` (x ≳ 4100) — not by QF's composed
  quad-word precision. A pilot confirmed the cancellation gradient IS present at this
  set: measured naive-QF reads {28.23, 23.50, 23.87} at x ∈ {1e2, 1e4, 1e6} (not
  uniform 29), while FP32 naive collapses to {3.28, 0.00, 0.00} — so both the
  gradient and the FP32→QF lift are visible without extending upward to {1e2,1e6,1e10}.

K2/K4 keep N = 10⁶: at that N the smallest term (1/N = 1e-6 for K4, 1/N² = 1e-12 for
K2) stays ~15 decades above QF's `u = 2⁻⁹⁶ ≈ 1.3e-29`, so no term stalls — the
FP32-narrow term-stall concern that gated T2.6's calibration does not bite here at
all (QF's `u` is far finer). Per-kernel measured means (GCC 13.3.0, Serial):
`K1_stable` 29.00 (capped), `K2_basel` 27.67, `K3_machin` 28.73, `K4_alt_harmonic`
28.64 — all PASS by comfortable margins (K3 clears by +2.73, no §10 denormal-tail
hazard on the atan path). Registered with the plain `kokkos_ep_add_test` helper
(no contraction flags — same K1 rationale as T1.6/T2.6, see the CMake comment).

## Framework

CTest + the lightweight `test_utils.hpp` header — **no** GoogleTest/Catch2. The
sole assertion primitive is `KOKKOS_EP_ASSERT(cond, msg)`, which prints
`file:line` + message and makes `main()` return nonzero. Rationale (and the
option to revisit in Phase 1) is documented at the top of `test_utils.hpp`.

## Graceful degradation (no LIBQUADMATH)

The `__float128` oracle comes from Kokkos's quadmath overloads, available only
when Kokkos was built with `-DKokkos_ENABLE_LIBQUADMATH=ON`. When that TPL is
absent:

- CMake still configures and the tests still **build** (no configure error).
- `KOKKOS_EP_HAVE_QUADMATH` is left undefined, so the oracle code paths compile
  out.
- Oracle-dependent tests return exit code **77** at runtime, which CTest reports
  as **`Skipped`** (not failed) via `SKIP_RETURN_CODE`.

This is the same skip-don't-fail posture T0.0/T0.3 used for the demos: a
legitimately quadmath-less Kokkos config should not turn the suite red.
